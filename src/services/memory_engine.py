from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Iterable

from src.db import get_pool
from src.models.schemas import (
    MemoryRecord,
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalResult,
    TenantContext,
)


class MemoryEngine:
    """
    Hybrid retrieval orchestrator.

    Production implementations should wire:
    1) pgvector semantic similarity search
    2) explicit entity graph traversal (LlamaIndex/LangChain graph adapters)
    3) recency weighting to prioritize fresh truths
    """

    HALF_LIFE_HOURS = 168.0
    MAX_RENDER_ITEMS = 8
    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in", "is",
        "it", "me", "my", "of", "on", "or", "that", "the", "this", "to", "we", "what", "where",
        "who", "why", "you", "your", "with", "about", "do", "does", "did",
    }

    def __init__(self, pg_dsn: str | None = None) -> None:
        self.pg_dsn = pg_dsn

    async def hybrid_retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Run vector + graph + recency retrieval and return a compressed markdown payload."""
        vector_rows = await self._vector_similarity_search(request)
        graph_rows = await self._resolve_entity_nodes(request)

        merged = self._merge_candidates(vector_rows, graph_rows)
        rescored = self._apply_recency_decay(merged)
        ranked = sorted(rescored, key=lambda row: row.final_score, reverse=True)

        payload_markdown, selected = self._render_truth_payload_markdown(
            ranked,
            request.max_tokens,
        )
        token_estimate = self._estimate_tokens(payload_markdown)

        return RetrievalResult(
            tenant=request.tenant,
            payload_markdown=payload_markdown,
            consumed_tokens_estimate=token_estimate,
            selected_items=selected,
        )

    async def _vector_similarity_search(
        self,
        request: RetrievalRequest,
    ) -> list[RetrievalCandidate]:
        """
        Fetch candidates from pgvector.

        This baseline implementation uses lexical overlap as a deterministic fallback.
        It still enforces the full tenant tuple in SQL.
        """
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT memory_id, content, confidence, updated_at
                FROM memory_records
                WHERE customer_id = $1
                  AND workspace_id = $2
                  AND end_user_id = $3
                  AND session_id = $4
                                    AND maker_id = $5
                                    AND agent_id = $6
                  AND is_active = TRUE
                ORDER BY updated_at DESC
                                LIMIT $7
                """,
                request.tenant.customer_id,
                request.tenant.workspace_id,
                request.tenant.end_user_id,
                request.tenant.session_id,
                                request.maker_id,
                                request.agent_id,
                request.max_items,
            )

        query_terms = self._extract_terms(request.query)
        candidates: list[RetrievalCandidate] = []
        for row in rows:
            content = str(row["content"])
            content_terms = self._extract_terms(content)
            overlap = len(query_terms.intersection(content_terms))
            denom = max(len(query_terms), 1)

            lexical_overlap = min(overlap / denom, 1.0)
            phrase_bonus = 0.15 if request.query.lower().strip() in content.lower() else 0.0
            property_bonus = 0.1 if "profile." in content.lower() and overlap > 0 else 0.0
            semantic_score = min(lexical_overlap + phrase_bonus + property_bonus, 1.0)

            confidence = float(row["confidence"])
            semantic_score = min(max((semantic_score * 0.75) + (confidence * 0.25), 0.0), 1.0)

            candidates.append(
                RetrievalCandidate(
                    memory_id=row["memory_id"],
                    content=content,
                    semantic_score=semantic_score,
                    graph_score=0.0,
                    recency_score=0.0,
                    final_score=semantic_score,
                    updated_at=row["updated_at"],
                )
            )
        return candidates

    async def _resolve_entity_nodes(
        self,
        request: RetrievalRequest,
    ) -> list[RetrievalCandidate]:
        """
        Resolve entity-linked facts from the knowledge graph tier.

        In production, this can use LlamaIndex KG index or LangChain graph interfaces.
        """
        return []

    async def list_active_memories(
        self,
        tenant: TenantContext,
        *,
        maker_id: str,
        agent_id: str,
    ) -> list[MemoryRecord]:
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    memory_id,
                    customer_id,
                    workspace_id,
                    end_user_id,
                    session_id,
                    maker_id,
                    agent_id,
                    content,
                    source,
                    confidence,
                    recency_boost,
                    embedding,
                    entity_ids,
                    created_at,
                    updated_at,
                    is_active
                FROM memory_records
                WHERE customer_id = $1
                  AND workspace_id = $2
                  AND end_user_id = $3
                  AND session_id = $4
                                    AND maker_id = $5
                                    AND agent_id = $6
                  AND is_active = TRUE
                ORDER BY updated_at DESC
                """,
                tenant.customer_id,
                tenant.workspace_id,
                tenant.end_user_id,
                tenant.session_id,
                                maker_id,
                                agent_id,
            )

        records: list[MemoryRecord] = []
        for row in rows:
            records.append(
                MemoryRecord(
                    memory_id=row["memory_id"],
                    tenant=TenantContext(
                        customer_id=row["customer_id"],
                        workspace_id=row["workspace_id"],
                        end_user_id=row["end_user_id"],
                        session_id=row["session_id"],
                    ),
                    maker_id=row["maker_id"],
                    agent_id=row["agent_id"],
                    content=row["content"],
                    source=row["source"],
                    confidence=row["confidence"],
                    recency_boost=row["recency_boost"],
                    embedding=row["embedding"],
                    entity_ids=row["entity_ids"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    is_active=row["is_active"],
                )
            )
        return records

    def _merge_candidates(
        self,
        vector_rows: list[RetrievalCandidate],
        graph_rows: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        by_id: dict[str, RetrievalCandidate] = {}

        for row in vector_rows + graph_rows:
            if row.memory_id not in by_id:
                by_id[row.memory_id] = row
                continue

            existing = by_id[row.memory_id]
            merged = RetrievalCandidate(
                memory_id=existing.memory_id,
                content=existing.content,
                semantic_score=max(existing.semantic_score, row.semantic_score),
                graph_score=max(existing.graph_score, row.graph_score),
                recency_score=max(existing.recency_score, row.recency_score),
                final_score=max(existing.final_score, row.final_score),
                updated_at=max(existing.updated_at, row.updated_at),
            )
            by_id[row.memory_id] = merged

        return list(by_id.values())

    def _apply_recency_decay(
        self,
        rows: Iterable[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        now = datetime.now(timezone.utc)
        half_life_lambda = math.log(2) / self.HALF_LIFE_HOURS

        rescored: list[RetrievalCandidate] = []
        for row in rows:
            age_hours = max((now - row.updated_at).total_seconds() / 3600, 0.0)
            recency_multiplier = math.exp(-half_life_lambda * age_hours)
            final_score = (
                (0.5 * row.semantic_score) +
                (0.3 * row.graph_score) +
                (0.2 * recency_multiplier)
            )
            rescored.append(
                row.model_copy(
                    update={
                        "recency_score": recency_multiplier,
                        "final_score": min(max(final_score, 0.0), 1.0),
                    }
                )
            )

        return rescored

    def _render_truth_payload_markdown(
        self,
        ranked_rows: list[RetrievalCandidate],
        max_tokens: int,
    ) -> tuple[str, list[RetrievalCandidate]]:
        """Build a compact machine-oriented payload that fits in the token budget."""
        body_lines: list[str] = ["# GM_CTX_V2\n", "scope=tenant_tuple\n"]
        selected: list[RetrievalCandidate] = []

        dedup_seen: set[str] = set()
        display_index = 1
        for row in ranked_rows:
            if len(selected) >= self.MAX_RENDER_ITEMS:
                break

            compact_content = self._compact_content_for_payload(row.content)
            dedup_key = compact_content.lower().strip()
            if dedup_key in dedup_seen:
                continue
            dedup_seen.add(dedup_key)

            short_updated_at = row.updated_at.strftime("%Y%m%dT%H%MZ")
            short_memory_id = row.memory_id[-12:]
            candidate_line = (
                f"i={display_index}|s={row.final_score:.3f}|u={short_updated_at}|m={short_memory_id}|c={compact_content}\n"
            )
            projected = "".join(body_lines) + candidate_line
            if self._estimate_tokens(projected) > max_tokens:
                break
            body_lines.append(candidate_line)
            selected.append(row)
            display_index += 1

        final_payload = "".join(body_lines).strip()
        return final_payload, selected

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate used as a safety boundary before model injection."""
        words = len(text.split())
        return max(int(words * 1.33), 1)

    def _extract_terms(self, text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9_.-]{2,}", text.lower())
        return {token for token in tokens if token not in self.STOP_WORDS}

    def _compact_content_for_payload(self, content: str) -> str:
        collapsed = " ".join(content.split())
        if len(collapsed) <= 220:
            return collapsed
        return f"{collapsed[:217].rstrip()}..."

    async def persist_memory_record(self, record: MemoryRecord) -> None:
        """
        Persist a memory record.

        Stub only; replace with INSERT/UPSERT into PostgreSQL + pgvector columns.
        """
        pool = get_pool()
        query = """
        INSERT INTO memory_records (
            memory_id,
            customer_id,
            workspace_id,
            end_user_id,
            session_id,
            maker_id,
            agent_id,
            content,
            source,
            confidence,
            recency_boost,
            embedding,
            entity_ids,
            is_active,
            created_at,
            updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, TRUE, $14, $15
        )
        ON CONFLICT (memory_id) DO UPDATE SET
            content = EXCLUDED.content,
            source = EXCLUDED.source,
            confidence = EXCLUDED.confidence,
            recency_boost = EXCLUDED.recency_boost,
            embedding = EXCLUDED.embedding,
            entity_ids = EXCLUDED.entity_ids,
            is_active = TRUE,
            updated_at = EXCLUDED.updated_at
        """
        async with pool.acquire() as conn:
            await conn.execute(
                query,
                record.memory_id,
                record.tenant.customer_id,
                record.tenant.workspace_id,
                record.tenant.end_user_id,
                record.tenant.session_id,
                record.maker_id,
                record.agent_id,
                record.content,
                record.source,
                record.confidence,
                record.recency_boost,
                record.embedding,
                record.entity_ids,
                record.created_at,
                record.updated_at,
            )

    async def deactivate_memory_record(self, memory_id: str) -> None:
        """
        Soft delete a stale memory item by marking it inactive.

        Stub only; replace with update SQL statement.
        """
        pool = get_pool()
        query = """
        UPDATE memory_records
        SET is_active = FALSE, updated_at = NOW()
        WHERE memory_id = $1
        """
        async with pool.acquire() as conn:
            await conn.execute(query, memory_id)
