from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from src.models.schemas import RetrievalCandidate, RetrievalRequest, RetrievalResult
from src.services.memory_engine import MemoryEngine


@dataclass
class ContextOptimizerStats:
    decomposition_terms: list[str]
    total_candidates: int
    reranked_candidates: int
    kept_candidates: int
    conflicts_dropped: int


class QueryDecomposer:
    """Layer 1: decompose complex queries into focused search terms."""

    def __init__(self) -> None:
        self._client = self._build_openai_client()
        self._model = os.getenv("GENMIND_QUERY_DECOMP_MODEL", "gpt-4o-mini")

    def decompose(self, query: str) -> list[str]:
        query = " ".join(query.split()).strip()
        if not query:
            return []

        if self._client is not None:
            llm_terms = self._decompose_with_llm(query)
            if llm_terms:
                return llm_terms

        return self._fallback_decompose(query)

    def _decompose_with_llm(self, query: str) -> list[str]:
        prompt = (
            "Decompose this user query into concise searchable terms and short phrases. "
            "Return JSON with key 'terms' (array of strings). Keep numbers, SKUs, versions, and proper nouns."
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                max_tokens=120,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You output compact JSON only."},
                    {"role": "user", "content": f"{prompt}\n\nQuery: {query}"},
                ],
            )
            content = response.choices[0].message.content or "{}"
            import json

            payload = json.loads(content)
            terms = payload.get("terms", []) if isinstance(payload, dict) else []
            cleaned = [self._clean_term(str(term)) for term in terms]
            return [term for term in cleaned if term]
        except Exception:
            return []

    def _fallback_decompose(self, query: str) -> list[str]:
        # Keep exact technical tokens while also deriving phrase chunks.
        exact = re.findall(r"[A-Za-z0-9_.:-]{2,}", query)
        phrases = [
            p.strip()
            for p in re.split(r"\b(?:and|or|but|then|also|with|about|for)\b", query, flags=re.IGNORECASE)
            if p.strip()
        ]
        terms = [self._clean_term(token) for token in exact]
        terms.extend(self._clean_term(phrase) for phrase in phrases)

        unique: list[str] = []
        seen: set[str] = set()
        for term in terms:
            if not term:
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(term)
        return unique[:16]

    def _clean_term(self, value: str) -> str:
        value = " ".join(value.split()).strip()
        return value[:80]

    @staticmethod
    @lru_cache(maxsize=1)
    def _build_openai_client():
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        try:
            from openai import OpenAI  # type: ignore

            return OpenAI(api_key=api_key)
        except Exception:
            return None


class LocalCrossEncoderReranker:
    """Layer 2: local reranking with optional cross-encoder model and lexical fallback."""

    def __init__(self) -> None:
        self._cross_encoder = self._build_cross_encoder()
        self._flag_reranker = self._build_flag_reranker()

    def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
        if not candidates:
            return []

        # Prefer lightweight local neural rerankers only.
        if self._flag_reranker is not None:
            try:
                pairs = [[query, row.content] for row in candidates]
                scores = self._flag_reranker.compute_score(pairs)
                if not isinstance(scores, list):
                    scores = [scores]
                enriched = []
                for row, score in zip(candidates, scores):
                    score_f = float(score)
                    final = max(0.0, min(1.0, (0.7 * row.final_score) + (0.3 * self._sigmoid(score_f))))
                    enriched.append(row.model_copy(update={"final_score": final}))
                return sorted(enriched, key=lambda r: r.final_score, reverse=True)[:top_k]
            except Exception:
                pass

        if self._cross_encoder is not None:
            pairs = [(query, row.content) for row in candidates]
            try:
                scores = self._cross_encoder.predict(pairs)
                enriched = []
                for row, score in zip(candidates, scores):
                    score_f = float(score)
                    final = max(0.0, min(1.0, (0.7 * row.final_score) + (0.3 * self._sigmoid(score_f))))
                    enriched.append(row.model_copy(update={"final_score": final}))
                return sorted(enriched, key=lambda r: r.final_score, reverse=True)[:top_k]
            except Exception:
                pass

        # If no local neural reranker is available, keep existing ordering and score.
        return sorted(candidates, key=lambda r: r.final_score, reverse=True)[:top_k]

    @staticmethod
    @lru_cache(maxsize=1)
    def _build_cross_encoder():
        model_name = os.getenv("GENMIND_RERANKER_MODEL", "BAAI/bge-reranker-base")
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            return CrossEncoder(model_name)
        except Exception:
            return None

    @staticmethod
    @lru_cache(maxsize=1)
    def _build_flag_reranker():
        model_name = os.getenv("GENMIND_FLAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
        try:
            from FlagEmbedding import FlagReranker  # type: ignore

            return FlagReranker(model_name, use_fp16=False)
        except Exception:
            return None

    def _terms(self, text: str) -> set[str]:
        return {tok for tok in re.findall(r"[a-z0-9_.:-]{2,}", text.lower())}

    def _sigmoid(self, x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0


class ContextOptimizer:
    """
    Coordinates hybrid search, local reranking, conflict guards, and sentence pruning.
    """

    MAX_CANDIDATES = 50
    DEFAULT_TOP_K = 5

    def __init__(self, memory_engine: MemoryEngine) -> None:
        self.memory_engine = memory_engine
        self.decomposer = QueryDecomposer()
        self.reranker = LocalCrossEncoderReranker()
        self.last_stats: ContextOptimizerStats | None = None

    async def optimize(self, request: RetrievalRequest) -> RetrievalResult:
        terms = self.decomposer.decompose(request.query)
        hybrid_query = " | ".join(terms) if terms else request.query

        # Layer 1: retrieve broader set for better reranking recall.
        expanded_request = request.model_copy(update={"query": hybrid_query, "max_items": self.MAX_CANDIDATES})
        base_result = await self.memory_engine.hybrid_retrieve(expanded_request)
        semantic_candidates = list(base_result.selected_items)
        keyword_candidates = await self._keyword_bm25_search(request, terms)
        all_candidates = self._merge_hybrid_candidates(semantic_candidates, keyword_candidates)

        # Layer 2: local rerank and hard truncate.
        top_k = 5
        reranked = self.reranker.rerank(request.query, all_candidates, top_k=top_k)

        # Layer 3 + context guard.
        no_conflict_rows, dropped_conflicts = self._resolve_conflicts(reranked)
        pruned_rows = [self._prune_candidate_sentences(request.query, row) for row in no_conflict_rows]
        pruned_rows = [row for row in pruned_rows if row.content.strip()]

        payload = self._render_payload(pruned_rows, request.max_tokens)
        token_estimate = self._estimate_tokens(payload)

        self.last_stats = ContextOptimizerStats(
            decomposition_terms=terms,
            total_candidates=len(all_candidates),
            reranked_candidates=len(reranked),
            kept_candidates=len(pruned_rows),
            conflicts_dropped=dropped_conflicts,
        )

        return RetrievalResult(
            tenant=request.tenant,
            payload_markdown=payload,
            consumed_tokens_estimate=token_estimate,
            selected_items=pruned_rows,
        )

    async def _keyword_bm25_search(
        self,
        request: RetrievalRequest,
        decomposition_terms: list[str],
    ) -> list[RetrievalCandidate]:
        records = await self.memory_engine.list_active_memories(
            request.tenant,
            maker_id=request.maker_id,
            agent_id=request.agent_id,
        )
        if not records:
            return []

        query_terms = self._terms_for_bm25(" ".join([request.query] + decomposition_terms))
        if not query_terms:
            return []

        docs = [self._terms_for_bm25(record.content) for record in records]
        avgdl = sum(len(doc) for doc in docs) / max(len(docs), 1)
        avgdl = max(avgdl, 1.0)

        # Compute document frequencies.
        df: dict[str, int] = {}
        for doc in docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        k1 = 1.2
        b = 0.75
        N = len(docs)

        scored: list[tuple[float, RetrievalCandidate]] = []
        for record, doc_terms in zip(records, docs):
            if not doc_terms:
                continue
            tf: dict[str, int] = {}
            for term in doc_terms:
                tf[term] = tf.get(term, 0) + 1

            doc_len = len(doc_terms)
            score = 0.0
            for term in query_terms:
                n_q = df.get(term, 0)
                if n_q == 0:
                    continue
                idf = math.log(((N - n_q + 0.5) / (n_q + 0.5)) + 1.0)
                f_qd = tf.get(term, 0)
                if f_qd == 0:
                    continue
                numer = f_qd * (k1 + 1.0)
                denom = f_qd + k1 * (1.0 - b + b * (doc_len / avgdl))
                score += idf * (numer / max(denom, 1e-9))

            if score <= 0:
                continue

            norm_score = min(score / 8.0, 1.0)
            candidate = RetrievalCandidate(
                memory_id=record.memory_id,
                content=record.content,
                semantic_score=0.0,
                graph_score=0.0,
                recency_score=0.0,
                final_score=norm_score,
                updated_at=record.updated_at,
            )
            scored.append((norm_score, candidate))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[: self.MAX_CANDIDATES]]

    def _merge_hybrid_candidates(
        self,
        semantic_candidates: list[RetrievalCandidate],
        keyword_candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        by_id: dict[str, RetrievalCandidate] = {}

        for row in semantic_candidates:
            by_id[row.memory_id] = row

        for row in keyword_candidates:
            existing = by_id.get(row.memory_id)
            if existing is None:
                by_id[row.memory_id] = row
                continue

            merged_final = min(1.0, (0.6 * existing.final_score) + (0.4 * row.final_score))
            by_id[row.memory_id] = existing.model_copy(update={"final_score": merged_final})

        merged = list(by_id.values())
        merged.sort(key=lambda item: item.final_score, reverse=True)
        return merged[: self.MAX_CANDIDATES]

    def _resolve_conflicts(self, rows: list[RetrievalCandidate]) -> tuple[list[RetrievalCandidate], int]:
        grouped: dict[str, list[RetrievalCandidate]] = {}
        passthrough: list[RetrievalCandidate] = []

        for row in rows:
            key, value = self._extract_key_value(row.content)
            if not key or not value:
                passthrough.append(row)
                continue
            grouped.setdefault(key, []).append(row)

        resolved: list[RetrievalCandidate] = list(passthrough)
        dropped = 0
        for key, items in grouped.items():
            distinct_values = {self._norm(v) for _, v in (self._extract_key_value(i.content) for i in items) if v}
            if len(distinct_values) <= 1:
                resolved.extend(items)
                continue

            # Newest verified timestamp wins.
            winner = max(items, key=lambda r: r.updated_at)
            resolved.append(winner)
            dropped += max(0, len(items) - 1)

        resolved_sorted = sorted(resolved, key=lambda r: r.final_score, reverse=True)
        return resolved_sorted, dropped

    def _prune_candidate_sentences(self, query: str, row: RetrievalCandidate) -> RetrievalCandidate:
        sentences = re.split(r"(?<=[.!?])\s+", row.content.strip())
        if len(sentences) <= 1:
            return row

        terms = {t.lower() for t in re.findall(r"[a-z0-9_.:-]{2,}", query)}
        key, value = self._extract_key_value(row.content)
        if key:
            terms.update(re.findall(r"[a-z0-9_.:-]{2,}", key.lower()))
        if value:
            terms.update(re.findall(r"[a-z0-9_.:-]{2,}", value.lower()))

        kept: list[str] = []
        for sent in sentences:
            sent_terms = set(re.findall(r"[a-z0-9_.:-]{2,}", sent.lower()))
            if terms.intersection(sent_terms):
                kept.append(sent)

        if not kept:
            # Keep first sentence as fallback to avoid empty context chunk.
            kept = [sentences[0]]

        compact = " ".join(kept).strip()
        return row.model_copy(update={"content": compact})

    def _render_payload(self, rows: list[RetrievalCandidate], max_tokens: int) -> str:
        lines = ["# GM_CTX_V2", "scope=tenant_tuple", "optimizer=context_optimizer_v1"]
        for idx, row in enumerate(rows, start=1):
            compact = " ".join(row.content.split())
            updated = row.updated_at.strftime("%Y%m%dT%H%MZ")
            mid = row.memory_id[-12:]
            line = f"i={idx}|s={row.final_score:.3f}|u={updated}|m={mid}|c={compact}"
            prospective = "\n".join(lines + [line])
            if self._estimate_tokens(prospective) > max_tokens:
                break
            lines.append(line)

        return "\n".join(lines)

    def _extract_key_value(self, content: str) -> tuple[str | None, str | None]:
        # profile.key=value
        profile_match = re.match(r"^profile\.([a-z0-9_\-]+)=(.+)$", content.strip(), flags=re.IGNORECASE)
        if profile_match:
            return profile_match.group(1).strip().lower(), profile_match.group(2).split("|", 1)[0].strip()

        # uome compact: ...|tgt=foo|val=bar|...
        tgt = re.search(r"\|tgt=([^|]+)", content)
        val = re.search(r"\|val=([^|]+)", content)
        if tgt and val:
            return tgt.group(1).strip().lower(), val.group(1).strip()

        return None, None

    def _norm(self, text: str) -> str:
        return " ".join(text.lower().split())

    def _terms_for_bm25(self, text: str) -> list[str]:
        return re.findall(r"[a-z0-9_.:-]{2,}", text.lower())

    def _estimate_tokens(self, text: str) -> int:
        words = len(text.split())
        return max(int(words * 1.33), 1)
