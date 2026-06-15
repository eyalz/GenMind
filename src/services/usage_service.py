from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from src.db import get_pool
from src.models.schemas import (
    RecentCustomerActivity,
    RetrievalQualitySnapshot,
    RetrievalSLOAlert,
    UsageDailyAggregate,
    UsageEvent,
)


class UsageService:
    async def emit(self, event: UsageEvent) -> None:
        pool = get_pool()
        query = """
        INSERT INTO usage_events (
            event_id, customer_id, workspace_id, end_user_id, session_id,
            request_id, endpoint, tokens_in, tokens_out, context_tokens,
            vector_reads, graph_reads, memory_writes,
            retrieval_mode, top_k_selected, score_threshold_milli,
            retrieval_candidates_total, retrieval_candidates_kept,
            retrieval_conflicts_dropped, retrieval_claim_rows_reconciled,
            retrieval_light_memory_mode,
            latency_ms, status_code, occurred_at
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9, $10,
            $11, $12, $13,
            $14, $15, $16,
            $17, $18,
            $19, $20,
            $21,
            $22, $23, $24
        )
        """
        async with pool.acquire() as conn:
            await conn.execute(
                query,
                event.event_id,
                event.customer_id,
                event.workspace_id,
                event.end_user_id,
                event.session_id,
                event.request_id,
                event.endpoint,
                event.tokens_in,
                event.tokens_out,
                event.context_tokens,
                event.vector_reads,
                event.graph_reads,
                event.memory_writes,
                event.retrieval_mode,
                event.top_k_selected,
                event.score_threshold_milli,
                event.retrieval_candidates_total,
                event.retrieval_candidates_kept,
                event.retrieval_conflicts_dropped,
                event.retrieval_claim_rows_reconciled,
                event.retrieval_light_memory_mode,
                event.latency_ms,
                event.status_code,
                event.occurred_at,
            )

    async def emit_simple(
        self,
        *,
        customer_id: str,
        workspace_id: str,
        end_user_id: str,
        session_id: str,
        request_id: str,
        endpoint: str,
        status_code: int,
        latency_ms: int,
        tokens_in: int = 0,
        tokens_out: int = 0,
        context_tokens: int = 0,
        vector_reads: int = 0,
        graph_reads: int = 0,
        memory_writes: int = 0,
        retrieval_mode: str = "",
        top_k_selected: int = 0,
        score_threshold_milli: int = 0,
        retrieval_candidates_total: int = 0,
        retrieval_candidates_kept: int = 0,
        retrieval_conflicts_dropped: int = 0,
        retrieval_claim_rows_reconciled: int = 0,
        retrieval_light_memory_mode: bool = False,
    ) -> None:
        event = UsageEvent(
            event_id=str(uuid4()),
            customer_id=customer_id,
            workspace_id=workspace_id,
            end_user_id=end_user_id,
            session_id=session_id,
            request_id=request_id,
            endpoint=endpoint,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            context_tokens=context_tokens,
            vector_reads=vector_reads,
            graph_reads=graph_reads,
            memory_writes=memory_writes,
            retrieval_mode=retrieval_mode,
            top_k_selected=top_k_selected,
            score_threshold_milli=score_threshold_milli,
            retrieval_candidates_total=retrieval_candidates_total,
            retrieval_candidates_kept=retrieval_candidates_kept,
            retrieval_conflicts_dropped=retrieval_conflicts_dropped,
            retrieval_claim_rows_reconciled=retrieval_claim_rows_reconciled,
            retrieval_light_memory_mode=retrieval_light_memory_mode,
            latency_ms=latency_ms,
            status_code=status_code,
            occurred_at=datetime.now(timezone.utc),
        )
        await self.emit(event)

    async def get_daily_aggregates(
        self,
        *,
        customer_id: str,
        workspace_id: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        include_admin: bool = False,
    ) -> list[UsageDailyAggregate]:
        pool = get_pool()
        query = """
        SELECT
            customer_id,
            workspace_id,
            to_char(date_trunc('day', occurred_at), 'YYYY-MM-DD') AS date,
            COUNT(*)::INT AS total_requests,
            COALESCE(SUM(tokens_in), 0)::INT AS total_tokens_in,
            COALESCE(SUM(tokens_out), 0)::INT AS total_tokens_out,
            COALESCE(SUM(context_tokens), 0)::INT AS total_context_tokens,
            COALESCE(SUM(vector_reads), 0)::INT AS total_vector_reads,
            COALESCE(SUM(memory_writes), 0)::INT AS total_memory_writes,
            COUNT(DISTINCT end_user_id)::INT AS active_end_users,
            COALESCE(AVG(latency_ms), 0)::FLOAT AS avg_latency_ms
        FROM usage_events
        WHERE customer_id = $1
          AND ($2::TEXT IS NULL OR workspace_id = $2)
          AND ($3::DATE IS NULL OR occurred_at::DATE >= $3)
          AND ($4::DATE IS NULL OR occurred_at::DATE <= $4)
          AND ($5::BOOLEAN = TRUE OR endpoint LIKE '/mcp/%')
        GROUP BY customer_id, workspace_id, date_trunc('day', occurred_at)
        ORDER BY date_trunc('day', occurred_at) DESC
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, customer_id, workspace_id, date_from, date_to, include_admin)

        return [UsageDailyAggregate(**dict(row)) for row in rows]

    async def get_recent_customer_activity(self, *, seconds: int = 10) -> list[RecentCustomerActivity]:
        pool = get_pool()
        query = """
        SELECT
            customer_id,
            COUNT(*) FILTER (
                WHERE endpoint LIKE '/mcp/%'
                  AND (tokens_in > 0 OR memory_writes > 0)
            )::INT AS inbound_calls,
            COUNT(*) FILTER (
                WHERE endpoint LIKE '/mcp/%'
                  AND (tokens_out > 0 OR context_tokens > 0)
            )::INT AS outbound_calls,
            MAX(occurred_at) FILTER (WHERE endpoint LIKE '/mcp/%') AS last_seen_at
        FROM usage_events
        WHERE occurred_at >= NOW() - ($1::INT * INTERVAL '1 second')
        GROUP BY customer_id
        HAVING COUNT(*) FILTER (
            WHERE endpoint LIKE '/mcp/%'
              AND (tokens_in > 0 OR tokens_out > 0 OR context_tokens > 0 OR memory_writes > 0)
        ) > 0
        ORDER BY last_seen_at DESC
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, seconds)

        return [RecentCustomerActivity(**dict(row)) for row in rows]

    async def get_retrieval_quality_snapshots(
        self,
        *,
        customer_id: str,
        workspace_id: str | None = None,
        window_seconds: int = 3600,
    ) -> list[RetrievalQualitySnapshot]:
        pool = get_pool()
        query = """
        WITH scoped AS (
            SELECT *
            FROM usage_events
            WHERE customer_id = $1
              AND ($2::TEXT IS NULL OR workspace_id = $2)
              AND endpoint LIKE '/mcp/%'
              AND occurred_at >= NOW() - ($3::INT * INTERVAL '1 second')
        ), ranked AS (
            SELECT
                customer_id,
                workspace_id,
                latency_ms,
                ROW_NUMBER() OVER (PARTITION BY customer_id, workspace_id ORDER BY latency_ms) AS rn,
                COUNT(*) OVER (PARTITION BY customer_id, workspace_id) AS cnt
            FROM scoped
        ), p95 AS (
            SELECT
                customer_id,
                workspace_id,
                COALESCE(MAX(latency_ms), 0)::INT AS p95_latency_ms
            FROM ranked
            WHERE rn >= CEIL(cnt * 0.95)
            GROUP BY customer_id, workspace_id
        )
        SELECT
            s.customer_id,
            s.workspace_id,
            COUNT(*)::INT AS requests_total,
            COALESCE(AVG(s.retrieval_candidates_total), 0)::FLOAT AS avg_candidates_total,
            COALESCE(AVG(s.retrieval_candidates_kept), 0)::FLOAT AS avg_candidates_kept,
            COALESCE(AVG(s.retrieval_conflicts_dropped), 0)::FLOAT AS avg_conflicts_dropped,
            COALESCE(AVG(s.retrieval_claim_rows_reconciled), 0)::FLOAT AS avg_claim_rows_reconciled,
            COALESCE(AVG(CASE WHEN s.retrieval_light_memory_mode THEN 1.0 ELSE 0.0 END), 0)::FLOAT AS light_memory_mode_ratio,
            COALESCE(AVG(CASE WHEN s.context_tokens = 0 THEN 1.0 ELSE 0.0 END), 0)::FLOAT AS empty_context_ratio,
            COALESCE(AVG(CASE WHEN s.retrieval_candidates_kept <= 1 THEN 1.0 ELSE 0.0 END), 0)::FLOAT AS low_kept_ratio,
            COALESCE(p.p95_latency_ms, 0)::INT AS p95_latency_ms
        FROM scoped s
        LEFT JOIN p95 p
          ON p.customer_id = s.customer_id
         AND p.workspace_id = s.workspace_id
        GROUP BY s.customer_id, s.workspace_id, p.p95_latency_ms
        ORDER BY requests_total DESC
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, customer_id, workspace_id, window_seconds)
        return [RetrievalQualitySnapshot(**dict(row)) for row in rows]

    async def get_retrieval_slo_alerts(
        self,
        *,
        customer_id: str,
        workspace_id: str | None = None,
        window_seconds: int = 3600,
        max_empty_context_ratio: float = 0.2,
        max_low_kept_ratio: float = 0.35,
        max_p95_latency_ms: int = 1200,
    ) -> list[RetrievalSLOAlert]:
        snapshots = await self.get_retrieval_quality_snapshots(
            customer_id=customer_id,
            workspace_id=workspace_id,
            window_seconds=window_seconds,
        )
        now = datetime.now(timezone.utc)
        alerts: list[RetrievalSLOAlert] = []
        for row in snapshots:
            if row.empty_context_ratio > max_empty_context_ratio:
                alerts.append(
                    RetrievalSLOAlert(
                        customer_id=row.customer_id,
                        workspace_id=row.workspace_id,
                        metric="empty_context_ratio",
                        observed_value=row.empty_context_ratio,
                        threshold_value=max_empty_context_ratio,
                        severity="high",
                        window_seconds=window_seconds,
                        triggered_at=now,
                    )
                )
            if row.low_kept_ratio > max_low_kept_ratio:
                alerts.append(
                    RetrievalSLOAlert(
                        customer_id=row.customer_id,
                        workspace_id=row.workspace_id,
                        metric="low_kept_ratio",
                        observed_value=row.low_kept_ratio,
                        threshold_value=max_low_kept_ratio,
                        severity="medium",
                        window_seconds=window_seconds,
                        triggered_at=now,
                    )
                )
            if row.p95_latency_ms > max_p95_latency_ms:
                alerts.append(
                    RetrievalSLOAlert(
                        customer_id=row.customer_id,
                        workspace_id=row.workspace_id,
                        metric="p95_latency_ms",
                        observed_value=float(row.p95_latency_ms),
                        threshold_value=float(max_p95_latency_ms),
                        severity="medium",
                        window_seconds=window_seconds,
                        triggered_at=now,
                    )
                )
        return alerts
