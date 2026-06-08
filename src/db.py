from __future__ import annotations

import asyncpg

from src.config import settings

_pool: asyncpg.Pool | None = None


async def init_db() -> asyncpg.Pool:
    global _pool
    _pool = await asyncpg.create_pool(settings.db_dsn, min_size=2, max_size=12)
    await _create_schema(_pool)
    return _pool


async def close_db() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool is not initialized.")
    return _pool


async def _create_schema(pool: asyncpg.Pool) -> None:
    schema_sql = """
    CREATE TABLE IF NOT EXISTS customers (
        customer_id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        status TEXT NOT NULL,
        plan TEXT NOT NULL,
        region TEXT NOT NULL,
        retention_days INT NOT NULL,
        is_demo BOOLEAN NOT NULL DEFAULT FALSE,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    );

    CREATE TABLE IF NOT EXISTS workspaces (
        workspace_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
        display_name TEXT NOT NULL,
        environment TEXT NOT NULL,
        status TEXT NOT NULL,
        monthly_request_quota BIGINT,
        created_at TIMESTAMPTZ NOT NULL
    );

    CREATE TABLE IF NOT EXISTS api_credentials (
        credential_id TEXT PRIMARY KEY,
        workspace_id TEXT NOT NULL REFERENCES workspaces(workspace_id) ON DELETE CASCADE,
        customer_id TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
        key_hash TEXT NOT NULL,
        key_prefix TEXT NOT NULL,
        scopes TEXT[] NOT NULL,
        last_used_at TIMESTAMPTZ,
        rotated_at TIMESTAMPTZ,
        expires_at TIMESTAMPTZ,
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ NOT NULL
    );

    CREATE TABLE IF NOT EXISTS memory_records (
        memory_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        end_user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        maker_id TEXT NOT NULL DEFAULT 'maker_default',
        agent_id TEXT NOT NULL DEFAULT 'agent_default',
        content TEXT NOT NULL,
        source TEXT NOT NULL,
        confidence DOUBLE PRECISION NOT NULL,
        recency_boost DOUBLE PRECISION NOT NULL,
        embedding DOUBLE PRECISION[] NOT NULL DEFAULT '{}',
        entity_ids TEXT[] NOT NULL DEFAULT '{}',
        is_active BOOLEAN NOT NULL DEFAULT TRUE,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    );

        CREATE INDEX IF NOT EXISTS idx_memory_scope
            ON memory_records(customer_id, workspace_id, end_user_id, session_id, is_active);

    CREATE TABLE IF NOT EXISTS usage_events (
        event_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        end_user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        endpoint TEXT NOT NULL,
        tokens_in INT NOT NULL,
        tokens_out INT NOT NULL,
        context_tokens INT NOT NULL,
        vector_reads INT NOT NULL,
        graph_reads INT NOT NULL,
        memory_writes INT NOT NULL,
        latency_ms INT NOT NULL,
        status_code INT NOT NULL,
        occurred_at TIMESTAMPTZ NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_usage_scope_time
      ON usage_events(customer_id, workspace_id, occurred_at DESC);

    CREATE TABLE IF NOT EXISTS audn_audit_log (
        audit_id BIGSERIAL PRIMARY KEY,
        customer_id TEXT NOT NULL,
        workspace_id TEXT NOT NULL,
        end_user_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        maker_id TEXT NOT NULL DEFAULT 'maker_default',
        agent_id TEXT NOT NULL DEFAULT 'agent_default',
        action TEXT NOT NULL,
        mutation_type TEXT NOT NULL DEFAULT 'property_modification',
        source_entity TEXT NOT NULL DEFAULT 'User_Profile',
        target_property_or_entity TEXT NOT NULL DEFAULT 'unknown',
        value_json JSONB NOT NULL DEFAULT '""'::jsonb,
        previous_value_reference JSONB,
        temporal_type TEXT NOT NULL DEFAULT 'permanent',
        valid_from TEXT NOT NULL DEFAULT 'current_interaction',
        valid_until TEXT NOT NULL DEFAULT 'indefinite',
        reasoning_justification TEXT NOT NULL DEFAULT '',
        candidate_fact TEXT NOT NULL,
        target_memory_id TEXT,
        confidence DOUBLE PRECISION NOT NULL,
        reason TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL
    );

    CREATE TABLE IF NOT EXISTS admin_audit_log (
        log_id BIGSERIAL PRIMARY KEY,
        operator_id TEXT NOT NULL,
        operator_role TEXT NOT NULL,
        action TEXT NOT NULL,
        target_resource TEXT NOT NULL,
        target_id TEXT NOT NULL,
        ip_address TEXT NOT NULL,
        occurred_at TIMESTAMPTZ NOT NULL
    );

        ALTER TABLE customers
            ADD COLUMN IF NOT EXISTS is_demo BOOLEAN NOT NULL DEFAULT FALSE;

        ALTER TABLE memory_records
            ADD COLUMN IF NOT EXISTS maker_id TEXT NOT NULL DEFAULT 'maker_default';

        ALTER TABLE memory_records
            ADD COLUMN IF NOT EXISTS agent_id TEXT NOT NULL DEFAULT 'agent_default';

        ALTER TABLE audn_audit_log
            ADD COLUMN IF NOT EXISTS maker_id TEXT NOT NULL DEFAULT 'maker_default';

        ALTER TABLE audn_audit_log
            ADD COLUMN IF NOT EXISTS agent_id TEXT NOT NULL DEFAULT 'agent_default';

        ALTER TABLE audn_audit_log
            ADD COLUMN IF NOT EXISTS mutation_type TEXT NOT NULL DEFAULT 'property_modification';

        ALTER TABLE audn_audit_log
            ADD COLUMN IF NOT EXISTS source_entity TEXT NOT NULL DEFAULT 'User_Profile';

        ALTER TABLE audn_audit_log
            ADD COLUMN IF NOT EXISTS target_property_or_entity TEXT NOT NULL DEFAULT 'unknown';

        ALTER TABLE audn_audit_log
            ADD COLUMN IF NOT EXISTS value_json JSONB NOT NULL DEFAULT '""'::jsonb;

        ALTER TABLE audn_audit_log
            ADD COLUMN IF NOT EXISTS previous_value_reference JSONB;

        ALTER TABLE audn_audit_log
            ADD COLUMN IF NOT EXISTS temporal_type TEXT NOT NULL DEFAULT 'permanent';

        ALTER TABLE audn_audit_log
            ADD COLUMN IF NOT EXISTS valid_from TEXT NOT NULL DEFAULT 'current_interaction';

        ALTER TABLE audn_audit_log
            ADD COLUMN IF NOT EXISTS valid_until TEXT NOT NULL DEFAULT 'indefinite';

        ALTER TABLE audn_audit_log
            ADD COLUMN IF NOT EXISTS reasoning_justification TEXT NOT NULL DEFAULT '';

        CREATE INDEX IF NOT EXISTS idx_memory_scope_lvl123
            ON memory_records(customer_id, workspace_id, maker_id, agent_id, end_user_id, session_id, is_active);
    """

    async with pool.acquire() as conn:
        await conn.execute(schema_sql)
