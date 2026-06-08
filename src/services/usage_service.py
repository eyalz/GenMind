from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from src.db import get_pool
from src.models.schemas import RecentCustomerActivity, UsageDailyAggregate, UsageEvent


class UsageService:
    async def emit(self, event: UsageEvent) -> None:
        pool = get_pool()
        query = """
        INSERT INTO usage_events (
            event_id, customer_id, workspace_id, end_user_id, session_id,
            request_id, endpoint, tokens_in, tokens_out, context_tokens,
            vector_reads, graph_reads, memory_writes, latency_ms, status_code, occurred_at
        ) VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16
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
