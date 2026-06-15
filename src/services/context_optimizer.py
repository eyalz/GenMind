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
    retrieval_mode: str
    top_k_selected: int
    score_threshold: float
    recent_questions_appended: bool
    light_memory_mode: bool
    claim_rows_reconciled: int


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
    MIN_SCORE_THRESHOLD = 0.12
    MAX_SCORE_THRESHOLD = 0.35
    RETRIEVAL_MODE_PATTERNS = {
        "profile": re.compile(
            r"\b(my|me|i\s+(prefer|like|need|usually|always)|preference|remember|profile|about\s+me)\b",
            re.IGNORECASE,
        ),
        "followup": re.compile(
            r"\b(it|they|them|those|that|ones|same|again|still|also|too)\b",
            re.IGNORECASE,
        ),
        "fact_lookup": re.compile(
            r"\b(what|who|when|where|why|how|status|latest|current|version|deadline|owner)\b",
            re.IGNORECASE,
        ),
        "task": re.compile(
            r"\b(run|build|deploy|fix|debug|investigate|configure|set\s+up|implement)\b",
            re.IGNORECASE,
        ),
    }
    EXTERNAL_INFO_PATTERNS = re.compile(
        r"\b(weather|temperature|population|news|stock|sports|score|time\s+in|currency|exchange\s+rate|distance)\b",
        re.IGNORECASE,
    )
    USER_CONTEXT_HINTS = re.compile(
        r"\b(my|me|i|profile|preference|remember|last\s+time|again|still|same)\b",
        re.IGNORECASE,
    )
    CURRENT_TIME_INTENT = re.compile(r"\b(now|current|latest|today|right\s+now|still)\b", re.IGNORECASE)

    def __init__(self, memory_engine: MemoryEngine) -> None:
        self.memory_engine = memory_engine
        self.decomposer = QueryDecomposer()
        self.reranker = LocalCrossEncoderReranker()
        self.last_stats: ContextOptimizerStats | None = None

    async def optimize(self, request: RetrievalRequest) -> RetrievalResult:
        retrieval_mode = self._classify_retrieval_mode(request.query)
        light_memory_mode = self._should_use_light_memory_mode(request.query, retrieval_mode)
        terms = self.decomposer.decompose(request.query)
        hybrid_query = " | ".join(terms) if terms else request.query

        # Layer 1: retrieve broader set for better reranking recall.
        expanded_request = request.model_copy(update={"query": hybrid_query, "max_items": self.MAX_CANDIDATES})
        base_result = await self.memory_engine.hybrid_retrieve(expanded_request)
        semantic_candidates = list(base_result.selected_items)
        keyword_candidates = await self._keyword_bm25_search(request, terms)
        all_candidates = self._merge_hybrid_candidates(semantic_candidates, keyword_candidates)

        # Layer 2: local rerank and adaptive truncate.
        top_k = self._adaptive_top_k(
            retrieval_mode,
            len(all_candidates),
            request.max_tokens,
            light_memory_mode=light_memory_mode,
        )
        reranked = self.reranker.rerank(request.query, all_candidates, top_k=top_k)
        threshold = self._score_threshold(retrieval_mode, light_memory_mode=light_memory_mode)
        filtered = self._apply_score_threshold(reranked, threshold)
        diversified = self._dedupe_semantic_duplicates(filtered, top_k=top_k)

        # Layer 3 + context guard.
        no_conflict_rows, dropped_conflicts = self._resolve_conflicts(diversified, request.query)
        pruned_rows = [self._prune_candidate_sentences(request.query, row) for row in no_conflict_rows]
        pruned_rows = [row for row in pruned_rows if row.content.strip()]
        reconciled_rows, claim_rows_reconciled = self._reconcile_claim_rows(pruned_rows, request.query)

        payload = self._render_payload(reconciled_rows, request.max_tokens)
        recent_questions_appended = False
        recent_questions = await self.memory_engine.list_recent_user_questions(
            request.tenant,
            maker_id=request.maker_id,
            agent_id=request.agent_id,
            limit=3,
        )
        if self._should_append_recent_questions(request.query, retrieval_mode, light_memory_mode):
            updated_payload = self._append_recent_questions_block(payload, recent_questions, request.max_tokens)
            recent_questions_appended = updated_payload != payload
            payload = updated_payload
        token_estimate = self._estimate_tokens(payload)

        self.last_stats = ContextOptimizerStats(
            decomposition_terms=terms,
            total_candidates=len(all_candidates),
            reranked_candidates=len(diversified),
            kept_candidates=len(reconciled_rows),
            conflicts_dropped=dropped_conflicts,
            retrieval_mode=retrieval_mode,
            top_k_selected=top_k,
            score_threshold=threshold,
            recent_questions_appended=recent_questions_appended,
            light_memory_mode=light_memory_mode,
            claim_rows_reconciled=claim_rows_reconciled,
        )

        return RetrievalResult(
            tenant=request.tenant,
            payload_markdown=payload,
            consumed_tokens_estimate=token_estimate,
            selected_items=reconciled_rows,
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

    def _classify_retrieval_mode(self, query: str) -> str:
        normalized = " ".join(query.lower().split())
        if not normalized:
            return "generic"
        if self.RETRIEVAL_MODE_PATTERNS["profile"].search(normalized):
            return "profile"
        if self.RETRIEVAL_MODE_PATTERNS["followup"].search(normalized):
            return "followup"
        if self.RETRIEVAL_MODE_PATTERNS["task"].search(normalized):
            return "task"
        if self.RETRIEVAL_MODE_PATTERNS["fact_lookup"].search(normalized):
            return "fact_lookup"
        return "generic"

    def _adaptive_top_k(
        self,
        mode: str,
        total_candidates: int,
        max_tokens: int,
        *,
        light_memory_mode: bool,
    ) -> int:
        base = self.DEFAULT_TOP_K
        if mode in {"profile", "followup"}:
            base = 6
        elif mode == "fact_lookup":
            base = 4
        elif mode == "task":
            base = 5

        if light_memory_mode:
            base = min(base, 3)

        if max_tokens <= 500:
            base -= 1
        elif max_tokens >= 1400:
            base += 1

        if total_candidates <= 3:
            base = min(base, total_candidates)

        return max(2, min(base, 8))

    def _score_threshold(self, mode: str, *, light_memory_mode: bool) -> float:
        if light_memory_mode:
            return 0.24
        if mode == "fact_lookup":
            return 0.2
        if mode in {"profile", "followup"}:
            return 0.14
        if mode == "task":
            return 0.16
        return 0.15

    def _should_use_light_memory_mode(self, query: str, mode: str) -> bool:
        if mode != "fact_lookup":
            return False
        if not self.EXTERNAL_INFO_PATTERNS.search(query):
            return False
        return not bool(self.USER_CONTEXT_HINTS.search(query))

    def _apply_score_threshold(
        self,
        rows: list[RetrievalCandidate],
        threshold: float,
    ) -> list[RetrievalCandidate]:
        if not rows:
            return []
        bounded = max(self.MIN_SCORE_THRESHOLD, min(self.MAX_SCORE_THRESHOLD, threshold))
        kept = [row for row in rows if row.final_score >= bounded]
        if kept:
            return kept
        return rows[:1]

    def _dedupe_semantic_duplicates(
        self,
        rows: list[RetrievalCandidate],
        *,
        top_k: int,
    ) -> list[RetrievalCandidate]:
        deduped: list[RetrievalCandidate] = []
        seen_keys: set[tuple[str, str]] = set()

        for row in rows:
            key, value = self._extract_key_value(row.content)
            if key and value:
                fingerprint = (self._norm(key), self._norm(value))
            else:
                tokens = re.findall(r"[a-z0-9_.:-]{3,}", row.content.lower())[:8]
                fingerprint = ("text", "|".join(tokens))

            if fingerprint in seen_keys:
                continue
            seen_keys.add(fingerprint)
            deduped.append(row)
            if len(deduped) >= top_k:
                break

        return deduped

    def _should_append_recent_questions(self, query: str, mode: str, light_memory_mode: bool) -> bool:
        if light_memory_mode:
            return False
        if mode in {"followup", "profile"}:
            return True
        query_terms = re.findall(r"[a-z0-9_.:-]{2,}", query.lower())
        short_query = len(query_terms) <= 6
        has_followup_pronoun = any(
            token in {"it", "they", "them", "that", "those", "ones", "again", "still"}
            for token in query_terms
        )
        return short_query and has_followup_pronoun

    def _resolve_conflicts(
        self,
        rows: list[RetrievalCandidate],
        query: str,
    ) -> tuple[list[RetrievalCandidate], int]:
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

            winner = self._select_claim_winner(items, query)
            resolved.append(winner)
            dropped += max(0, len(items) - 1)

        resolved_sorted = sorted(resolved, key=lambda r: r.final_score, reverse=True)
        return resolved_sorted, dropped

    def _reconcile_claim_rows(
        self,
        rows: list[RetrievalCandidate],
        query: str,
    ) -> tuple[list[RetrievalCandidate], int]:
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
        for _, items in grouped.items():
            if len(items) == 1:
                resolved.extend(items)
                continue
            winner = self._select_claim_winner(items, query)
            resolved.append(winner)
            dropped += len(items) - 1

        return sorted(resolved, key=lambda r: r.final_score, reverse=True), dropped

    def _select_claim_winner(self, items: list[RetrievalCandidate], query: str) -> RetrievalCandidate:
        now = datetime.now(timezone.utc)
        prefer_current = bool(self.CURRENT_TIME_INTENT.search(query))

        def winner_score(row: RetrievalCandidate) -> float:
            age_hours = max((now - row.updated_at).total_seconds() / 3600.0, 0.0)
            recency = math.exp(-age_hours / 168.0)
            source = self._source_priority(row.content)
            temporal = self._temporal_priority(row.content, prefer_current=prefer_current)
            return (0.65 * row.final_score) + (0.25 * recency) + (0.07 * source) + (0.03 * temporal)

        return max(items, key=winner_score)

    def _source_priority(self, content: str) -> float:
        text = content.strip().lower()
        if text.startswith("profile."):
            return 1.0
        if text.startswith("uome|v=1"):
            return 0.9
        if text.startswith("fact|v=1"):
            return 0.7
        return 0.6

    def _temporal_priority(self, content: str, *, prefer_current: bool) -> float:
        temporal_type = self._extract_temporal_type(content)
        if temporal_type == "temporary":
            return 1.0 if prefer_current else 0.55
        if temporal_type == "historical_archive":
            return 0.4
        return 0.95 if not prefer_current else 0.75

    def _extract_temporal_type(self, content: str) -> str:
        lowered = content.lower()
        if "|scope=temporary" in lowered:
            return "temporary"
        match = re.search(r"\|tt=([^|]+)", lowered)
        if not match:
            return "permanent"
        value = match.group(1).strip()
        if value in {"temporary", "historical_archive", "permanent"}:
            return value
        return "permanent"

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

    def _append_recent_questions_block(self, payload: str, questions: list[str], max_tokens: int) -> str:
        if not questions:
            return payload

        block_lines = ["", "recent_user_questions:"]
        for idx, text in enumerate(questions, start=1):
            compact = " ".join(text.split())[:220]
            block_lines.append(f"rq={idx}|q={compact}")

        candidate = payload + "\n" + "\n".join(block_lines)
        if self._estimate_tokens(candidate) <= max_tokens:
            return candidate
        return payload

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
