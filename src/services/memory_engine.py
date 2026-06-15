from __future__ import annotations

import math
import re
import json
from datetime import datetime, timezone
from hashlib import sha1
from typing import Iterable

from src.db import get_pool
from src.models.schemas import (
    AUDNAction,
    AUDNDecision,
    ClaimRecord,
    MemoryRecord,
    MutationType,
    RetrievalCandidate,
    RetrievalRequest,
    RetrievalResult,
    TenantContext,
    TemporalScope,
    TemporalType,
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

    async def append_recent_user_question(
        self,
        tenant: TenantContext,
        *,
        maker_id: str,
        agent_id: str,
        question_text: str,
        keep_last: int = 3,
    ) -> None:
        text = " ".join(question_text.split()).strip()
        if not text:
            return

        pool = get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    """
                    INSERT INTO session_recent_questions (
                        customer_id,
                        workspace_id,
                        end_user_id,
                        session_id,
                        maker_id,
                        agent_id,
                        question_text,
                        created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    tenant.customer_id,
                    tenant.workspace_id,
                    tenant.end_user_id,
                    tenant.session_id,
                    maker_id,
                    agent_id,
                    text,
                    datetime.now(timezone.utc),
                )

                await conn.execute(
                    """
                    DELETE FROM session_recent_questions
                    WHERE question_id IN (
                        SELECT question_id
                        FROM session_recent_questions
                        WHERE customer_id = $1
                          AND workspace_id = $2
                          AND end_user_id = $3
                          AND session_id = $4
                          AND maker_id = $5
                          AND agent_id = $6
                        ORDER BY created_at DESC, question_id DESC
                        OFFSET $7
                    )
                    """,
                    tenant.customer_id,
                    tenant.workspace_id,
                    tenant.end_user_id,
                    tenant.session_id,
                    maker_id,
                    agent_id,
                    keep_last,
                )

    async def list_recent_user_questions(
        self,
        tenant: TenantContext,
        *,
        maker_id: str,
        agent_id: str,
        limit: int = 3,
    ) -> list[str]:
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT question_text
                FROM (
                    SELECT question_text, created_at, question_id
                    FROM session_recent_questions
                    WHERE customer_id = $1
                      AND workspace_id = $2
                      AND end_user_id = $3
                      AND session_id = $4
                      AND maker_id = $5
                      AND agent_id = $6
                    ORDER BY created_at DESC, question_id DESC
                    LIMIT $7
                ) latest
                ORDER BY created_at ASC, question_id ASC
                """,
                tenant.customer_id,
                tenant.workspace_id,
                tenant.end_user_id,
                tenant.session_id,
                maker_id,
                agent_id,
                limit,
            )

        return [str(row["question_text"]).strip() for row in rows if str(row["question_text"]).strip()]

    async def get_session_context_state(
        self,
        tenant: TenantContext,
        *,
        maker_id: str,
        agent_id: str,
    ) -> dict:
        pool = get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT context_snapshot
                FROM session_context_state
                WHERE customer_id = $1
                  AND workspace_id = $2
                  AND end_user_id = $3
                  AND session_id = $4
                  AND maker_id = $5
                  AND agent_id = $6
                """,
                tenant.customer_id,
                tenant.workspace_id,
                tenant.end_user_id,
                tenant.session_id,
                maker_id,
                agent_id,
            )

        snapshot = row["context_snapshot"] if row else None
        if isinstance(snapshot, str):
            try:
                snapshot = json.loads(snapshot)
            except json.JSONDecodeError:
                snapshot = None

        if not isinstance(snapshot, dict):
            snapshot = {
                "primary_subject_entity": None,
                "inferred_current_location": None,
                "user_constraints": [],
            }

        return {
            "session_id": tenant.session_id,
            "context_snapshot": {
                "primary_subject_entity": snapshot.get("primary_subject_entity"),
                "inferred_current_location": snapshot.get("inferred_current_location"),
                "user_constraints": list(snapshot.get("user_constraints", []))
                if isinstance(snapshot.get("user_constraints", []), list)
                else [],
            },
        }

    async def upsert_session_context_state(
        self,
        tenant: TenantContext,
        *,
        maker_id: str,
        agent_id: str,
        context_snapshot: dict,
    ) -> None:
        safe_snapshot = {
            "primary_subject_entity": context_snapshot.get("primary_subject_entity"),
            "inferred_current_location": context_snapshot.get("inferred_current_location"),
            "user_constraints": context_snapshot.get("user_constraints", []),
        }

        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO session_context_state (
                    customer_id,
                    workspace_id,
                    end_user_id,
                    session_id,
                    maker_id,
                    agent_id,
                    context_snapshot,
                    updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8)
                ON CONFLICT (customer_id, workspace_id, end_user_id, session_id, maker_id, agent_id)
                DO UPDATE SET
                    context_snapshot = EXCLUDED.context_snapshot,
                    updated_at = EXCLUDED.updated_at
                """,
                tenant.customer_id,
                tenant.workspace_id,
                tenant.end_user_id,
                tenant.session_id,
                maker_id,
                agent_id,
                json.dumps(safe_snapshot, ensure_ascii=True),
                datetime.now(timezone.utc),
            )

    async def hybrid_retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Run vector + graph + recency retrieval and return a compressed markdown payload."""
        await self._ensure_claim_projection(request)

        vector_rows = await self._vector_similarity_search(request)
        claim_rows = await self._claim_projection_search(request)
        graph_rows = await self._resolve_entity_nodes(request)

        merged = self._merge_candidates(vector_rows, claim_rows)
        merged = self._merge_candidates(merged, graph_rows)
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

    async def _ensure_claim_projection(self, request: RetrievalRequest) -> None:
        # Lazy one-time backfill per retrieval scope when claims are still empty.
        existing_claims = await self.list_active_claims(
            request.tenant,
            maker_id=request.maker_id,
            agent_id=request.agent_id,
            limit=1,
        )
        if existing_claims:
            return

        await self.backfill_claims_from_active_memories(
            request.tenant,
            maker_id=request.maker_id,
            agent_id=request.agent_id,
        )

    async def _claim_projection_search(
        self,
        request: RetrievalRequest,
    ) -> list[RetrievalCandidate]:
        claims = await self.list_active_claims(
            request.tenant,
            maker_id=request.maker_id,
            agent_id=request.agent_id,
            limit=request.max_items,
        )
        if not claims:
            return []

        query_terms = self._extract_terms(request.query)
        if not query_terms:
            return []

        prefer_current = bool(re.search(r"\b(now|current|latest|today|right\s+now|still)\b", request.query, re.IGNORECASE))
        rows: list[RetrievalCandidate] = []

        for claim in claims:
            value_text = self._stringify_claim_value(claim.value_json)
            content = (
                f"uome|v=1|mt={claim.mutation_type.value}|src={claim.source_entity}|"
                f"tgt={claim.target_property_or_entity}|val={value_text}|"
                f"tt={claim.temporal_type.value}|vf={claim.valid_from}|vu={claim.valid_until}"
            )

            claim_terms = self._extract_terms(
                f"{claim.source_entity} {claim.target_property_or_entity} {value_text}"
            )
            overlap = len(query_terms.intersection(claim_terms))
            denom = max(len(query_terms), 1)
            lexical = min(overlap / denom, 1.0)

            temporal_bonus = 0.08 if (prefer_current and claim.temporal_type == TemporalType.TEMPORARY) else 0.0
            confidence_bonus = 0.15 * max(min(claim.confidence, 1.0), 0.0)
            semantic_score = min(lexical + temporal_bonus + confidence_bonus, 1.0)

            rows.append(
                RetrievalCandidate(
                    memory_id=claim.claim_id,
                    content=content,
                    semantic_score=semantic_score,
                    graph_score=0.2,
                    recency_score=0.0,
                    final_score=min(semantic_score + 0.08, 1.0),
                    updated_at=claim.updated_at,
                )
            )

        rows.sort(key=lambda item: item.final_score, reverse=True)
        return rows[: request.max_items]

    async def backfill_claims_from_active_memories(
        self,
        tenant: TenantContext,
        *,
        maker_id: str,
        agent_id: str,
    ) -> int:
        records = await self.list_active_memories(
            tenant,
            maker_id=maker_id,
            agent_id=agent_id,
        )
        if not records:
            return 0

        projected = 0
        # Replay oldest->newest so latest state wins deterministic claim upsert keys.
        for record in reversed(records):
            decision = self._decision_from_memory_record(record)
            if decision is None:
                continue
            await self.upsert_claim_from_decision(decision, memory_id=record.memory_id)
            projected += 1

        return projected

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

    def _decision_from_memory_record(self, record: MemoryRecord) -> AUDNDecision | None:
        content = " ".join(record.content.split()).strip()
        if not content:
            return None

        profile_match = re.match(r"^profile\.([a-z0-9_\-]+)=(.+)$", content, flags=re.IGNORECASE)
        if profile_match:
            target = profile_match.group(1).strip().lower()
            value = profile_match.group(2).split("|", 1)[0].strip()
            if len(target) < 2 or not value:
                return None
            temporal = TemporalScope(
                type=TemporalType.TEMPORARY if "|scope=temporary" in content.lower() else TemporalType.PERMANENT,
                valid_from="current_interaction",
                valid_until="conditional_trigger" if "|scope=temporary" in content.lower() else "indefinite",
            )
            return AUDNDecision(
                tenant=record.tenant,
                maker_id=record.maker_id,
                agent_id=record.agent_id,
                action=AUDNAction.UPDATE,
                mutation_type=MutationType.PROPERTY_MODIFICATION,
                source_entity="User_Profile",
                target_property_or_entity=target,
                value=value,
                temporal_scope=temporal,
                reasoning_justification="historical_backfill_projection",
                reason="Project active profile memory into typed claims.",
                candidate_fact=content,
                confidence=record.confidence,
                target_memory_id=record.memory_id,
            )

        tgt = re.search(r"\|tgt=([^|]+)", content)
        val = re.search(r"\|val=([^|]+)", content)
        if not tgt or not val:
            return None

        target = tgt.group(1).strip().lower()
        value = val.group(1).strip()
        if len(target) < 2 or not value:
            return None

        src = re.search(r"\|src=([^|]+)", content)
        mt = re.search(r"\|mt=([^|]+)", content)
        tt = re.search(r"\|tt=([^|]+)", content)
        vf = re.search(r"\|vf=([^|]+)", content)
        vu = re.search(r"\|vu=([^|]+)", content)

        mutation = MutationType.PROPERTY_MODIFICATION
        if mt:
            mt_value = mt.group(1).strip().lower()
            if mt_value in {item.value for item in MutationType}:
                mutation = MutationType(mt_value)

        temporal_type = TemporalType.PERMANENT
        if tt:
            tt_value = tt.group(1).strip().lower()
            if tt_value in {item.value for item in TemporalType}:
                temporal_type = TemporalType(tt_value)

        temporal = TemporalScope(
            type=temporal_type,
            valid_from=vf.group(1).strip() if vf else "current_interaction",
            valid_until=vu.group(1).strip() if vu else "indefinite",
        )

        return AUDNDecision(
            tenant=record.tenant,
            maker_id=record.maker_id,
            agent_id=record.agent_id,
            action=AUDNAction.UPDATE,
            mutation_type=mutation,
            source_entity=src.group(1).strip() if src else "User_Profile",
            target_property_or_entity=target,
            value=value,
            temporal_scope=temporal,
            reasoning_justification="historical_backfill_projection",
            reason="Project active UOME memory into typed claims.",
            candidate_fact=content,
            confidence=record.confidence,
            target_memory_id=record.memory_id,
        )

    def _stringify_claim_value(self, value: object) -> str:
        if isinstance(value, str):
            return " ".join(value.split())
        if isinstance(value, (int, float, bool)):
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        except TypeError:
            return str(value)

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

    async def upsert_claim_from_decision(
        self,
        decision: AUDNDecision,
        *,
        memory_id: str,
    ) -> None:
        """Project AUDN decision into typed claim storage for deterministic retrieval/reconciliation."""
        claim = self._build_claim_record(decision, memory_id=memory_id)
        pool = get_pool()
        query = """
        INSERT INTO memory_claims (
            claim_id,
            memory_id,
            customer_id,
            workspace_id,
            end_user_id,
            session_id,
            maker_id,
            agent_id,
            mutation_type,
            source_entity,
            target_property_or_entity,
            value_json,
            temporal_type,
            valid_from,
            valid_until,
            confidence,
            is_active,
            created_at,
            updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::jsonb, $13, $14, $15, $16, TRUE, $17, $18
        )
        ON CONFLICT (claim_id) DO UPDATE SET
            memory_id = EXCLUDED.memory_id,
            mutation_type = EXCLUDED.mutation_type,
            source_entity = EXCLUDED.source_entity,
            target_property_or_entity = EXCLUDED.target_property_or_entity,
            value_json = EXCLUDED.value_json,
            temporal_type = EXCLUDED.temporal_type,
            valid_from = EXCLUDED.valid_from,
            valid_until = EXCLUDED.valid_until,
            confidence = EXCLUDED.confidence,
            is_active = TRUE,
            updated_at = EXCLUDED.updated_at
        """
        async with pool.acquire() as conn:
            await conn.execute(
                query,
                claim.claim_id,
                claim.memory_id,
                claim.tenant.customer_id,
                claim.tenant.workspace_id,
                claim.tenant.end_user_id,
                claim.tenant.session_id,
                claim.maker_id,
                claim.agent_id,
                claim.mutation_type.value,
                claim.source_entity,
                claim.target_property_or_entity,
                json.dumps(claim.value_json, ensure_ascii=True),
                claim.temporal_type.value,
                claim.valid_from,
                claim.valid_until,
                claim.confidence,
                claim.created_at,
                claim.updated_at,
            )

    async def deactivate_claims_for_memory(self, memory_id: str) -> None:
        pool = get_pool()
        query = """
        UPDATE memory_claims
        SET is_active = FALSE, updated_at = NOW()
        WHERE memory_id = $1
        """
        async with pool.acquire() as conn:
            await conn.execute(query, memory_id)

    async def list_active_claims(
        self,
        tenant: TenantContext,
        *,
        maker_id: str,
        agent_id: str,
        limit: int = 200,
    ) -> list[ClaimRecord]:
        pool = get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    claim_id,
                    memory_id,
                    customer_id,
                    workspace_id,
                    end_user_id,
                    session_id,
                    maker_id,
                    agent_id,
                    mutation_type,
                    source_entity,
                    target_property_or_entity,
                    value_json,
                    temporal_type,
                    valid_from,
                    valid_until,
                    confidence,
                    is_active,
                    created_at,
                    updated_at
                FROM memory_claims
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
                tenant.customer_id,
                tenant.workspace_id,
                tenant.end_user_id,
                tenant.session_id,
                maker_id,
                agent_id,
                limit,
            )

        claims: list[ClaimRecord] = []
        for row in rows:
            claims.append(
                ClaimRecord(
                    claim_id=row["claim_id"],
                    memory_id=row["memory_id"],
                    tenant=TenantContext(
                        customer_id=row["customer_id"],
                        workspace_id=row["workspace_id"],
                        end_user_id=row["end_user_id"],
                        session_id=row["session_id"],
                    ),
                    maker_id=row["maker_id"],
                    agent_id=row["agent_id"],
                    mutation_type=MutationType(str(row["mutation_type"])),
                    source_entity=row["source_entity"],
                    target_property_or_entity=row["target_property_or_entity"],
                    value_json=row["value_json"],
                    temporal_type=TemporalType(str(row["temporal_type"])),
                    valid_from=row["valid_from"],
                    valid_until=row["valid_until"],
                    confidence=float(row["confidence"]),
                    is_active=bool(row["is_active"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return claims

    def _build_claim_record(self, decision: AUDNDecision, *, memory_id: str) -> ClaimRecord:
        value_obj = decision.value
        if isinstance(value_obj, str):
            value_obj = value_obj.strip()

        claim_key = "|".join(
            [
                decision.tenant.customer_id,
                decision.tenant.workspace_id,
                decision.tenant.end_user_id,
                decision.tenant.session_id,
                decision.maker_id,
                decision.agent_id,
                decision.source_entity.strip().lower(),
                decision.target_property_or_entity.strip().lower(),
            ]
        )
        claim_hash = sha1(claim_key.encode("utf-8")).hexdigest()[:20]
        claim_id = f"clm_{claim_hash}"

        now = datetime.now(timezone.utc)
        return ClaimRecord(
            claim_id=claim_id,
            memory_id=memory_id,
            tenant=decision.tenant,
            maker_id=decision.maker_id,
            agent_id=decision.agent_id,
            mutation_type=decision.mutation_type,
            source_entity=decision.source_entity,
            target_property_or_entity=decision.target_property_or_entity,
            value_json=value_obj,
            temporal_type=decision.temporal_scope.type,
            valid_from=decision.temporal_scope.valid_from,
            valid_until=decision.temporal_scope.valid_until,
            confidence=decision.confidence,
            is_active=True,
            created_at=now,
            updated_at=now,
        )
