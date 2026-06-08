from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from src.auth import create_jwt, hash_api_key, issue_api_key, require_scope
from src.db import get_pool
from src.models.schemas import (
    AdminAuditEntry,
    CredentialIssuedResponse,
    CustomerPlan,
    CreateCustomerRequest,
    CreateWorkspaceRequest,
    Customer,
    CustomerStatus,
    DatabaseSummaryResponse,
    EndUserDeleteRequest,
    RecentCustomerActivity,
    UpdateCustomerRequest,
    WorkspaceEnvironment,
    Workspace,
    WorkspaceStatus,
)
from src.services.usage_service import UsageService

router = APIRouter(prefix="/admin", tags=["admin"])
usage_service = UsageService()


class DevTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    customer_id: str = Field(default="cust_dev", min_length=2, max_length=128)
    workspace_id: str = Field(default="ws_dev", min_length=2, max_length=128)
    end_user_id: str = Field(default="user_dev", min_length=2, max_length=256)
    session_id: str = Field(default="session_dev", min_length=2, max_length=256)
    scopes: list[str] = Field(default_factory=lambda: ["admin:*", "memory:read", "memory:write"])
    expires_minutes: int = Field(default=60, ge=1, le=240)


class NormalizeCustomersResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kept_live_customer_id: str
    kept_demo_customer_ids: list[str]
    total_customers: int


@router.post("/dev/token")
async def issue_dev_token(request: Request, body: DevTokenRequest) -> dict[str, str]:
    """Local-only helper to bootstrap authenticated testing quickly."""
    if request.headers.get("x-dev-bootstrap") != "allow":
        raise HTTPException(status_code=403, detail="Missing x-dev-bootstrap header")

    token = create_jwt(
        customer_id=body.customer_id,
        workspace_id=body.workspace_id,
        end_user_id=body.end_user_id,
        session_id=body.session_id,
        scopes=body.scopes,
        expires_minutes=body.expires_minutes,
    )
    return {"access_token": token, "token_type": "bearer"}


def _ip(request: Request) -> str:
    if request.client:
        return request.client.host
    return "unknown"


async def _audit(
    *,
    request: Request,
    operator_id: str,
    operator_role: str,
    action: str,
    target_resource: str,
    target_id: str,
) -> None:
    entry = AdminAuditEntry(
        operator_id=operator_id,
        operator_role=operator_role,
        action=action,
        target_resource=target_resource,
        target_id=target_id,
        ip_address=_ip(request),
        occurred_at=datetime.now(timezone.utc),
    )
    pool = get_pool()
    query = """
    INSERT INTO admin_audit_log (
      operator_id, operator_role, action, target_resource, target_id, ip_address, occurred_at
    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
    """
    async with pool.acquire() as conn:
        await conn.execute(
            query,
            entry.operator_id,
            entry.operator_role,
            entry.action,
            entry.target_resource,
            entry.target_id,
            entry.ip_address,
            entry.occurred_at,
        )


async def _emit_admin_usage(
    *,
    request: Request,
    claims: dict[str, object],
    status_code: int,
    latency_ms: int,
) -> None:
    await usage_service.emit_simple(
        customer_id=str(claims.get("customer_id", "system")),
        workspace_id=str(claims.get("workspace_id", "system")),
        end_user_id="system",
        session_id="system",
        request_id=request.headers.get("x-request-id", str(uuid4())),
        endpoint=request.url.path,
        status_code=status_code,
        latency_ms=latency_ms,
    )


@router.post("/customers", response_model=Customer, status_code=201)
async def create_customer(body: CreateCustomerRequest, request: Request) -> Customer:
    started = perf_counter()
    claims = require_scope(request, {"admin:customers:write"})
    customer = Customer(
        customer_id=f"cust_{uuid4().hex[:20]}",
        display_name=body.display_name,
        status=CustomerStatus.ACTIVE,
        plan=body.plan,
        region=body.region,
        retention_days=body.retention_days,
        is_demo=body.is_demo,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    pool = get_pool()
    query = """
    INSERT INTO customers (
            customer_id, display_name, status, plan, region, retention_days, is_demo, created_at, updated_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    """
    async with pool.acquire() as conn:
        await conn.execute(
            query,
            customer.customer_id,
            customer.display_name,
            customer.status.value,
            customer.plan.value,
            customer.region,
            customer.retention_days,
            customer.is_demo,
            customer.created_at,
            customer.updated_at,
        )

    await _audit(
        request=request,
        operator_id=claims.get("sub", "admin"),
        operator_role="admin",
        action="create_customer",
        target_resource="customer",
        target_id=customer.customer_id,
    )
    await _emit_admin_usage(
        request=request,
        claims=claims,
        status_code=201,
        latency_ms=int((perf_counter() - started) * 1000),
    )
    return customer


@router.get("/customers", response_model=list[Customer])
async def list_customers(request: Request) -> list[Customer]:
    started = perf_counter()
    claims = require_scope(request, {"admin:customers:read"})
    pool = get_pool()
    query = """
    SELECT customer_id, display_name, status, plan, region, retention_days, is_demo, created_at, updated_at
    FROM customers
    ORDER BY created_at DESC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query)

    customers = []
    for row in rows:
        data = dict(row)
        data["status"] = CustomerStatus(data["status"])
        data["plan"] = CustomerPlan(data["plan"])
        customers.append(Customer(**data))

    await _emit_admin_usage(
        request=request,
        claims=claims,
        status_code=200,
        latency_ms=int((perf_counter() - started) * 1000),
    )
    return customers


@router.post("/customers/normalize", response_model=NormalizeCustomersResponse)
async def normalize_customers(request: Request) -> NormalizeCustomersResponse:
    started = perf_counter()
    claims = require_scope(request, {"admin:customers:write"})

    demo_names = ["Delta Learning", "Cobalt Logistics", "Beacon Finance"]
    live_name = "Copilot Studio Test 1"
    now = datetime.now(timezone.utc)

    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            live_row = await conn.fetchrow(
                """
                SELECT customer_id
                FROM customers
                WHERE display_name = $1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                live_name,
            )

            if live_row:
                live_customer_id = str(live_row["customer_id"])
                await conn.execute(
                    """
                    UPDATE customers
                    SET is_demo = FALSE,
                        status = 'active',
                        updated_at = $2
                    WHERE customer_id = $1
                    """,
                    live_customer_id,
                    now,
                )
            else:
                live_customer_id = f"cust_live_{uuid4().hex[:20]}"
                await conn.execute(
                    """
                    INSERT INTO customers (
                        customer_id, display_name, status, plan, region, retention_days, is_demo, created_at, updated_at
                    ) VALUES ($1, $2, 'active', 'growth', 'us-east-1', 90, FALSE, $3, $3)
                    """,
                    live_customer_id,
                    live_name,
                    now,
                )

            kept_demo_customer_ids: list[str] = []
            for display_name in demo_names:
                row = await conn.fetchrow(
                    """
                    SELECT customer_id
                    FROM customers
                    WHERE display_name = $1
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    display_name,
                )
                if row:
                    customer_id = str(row["customer_id"])
                    await conn.execute(
                        """
                        UPDATE customers
                        SET is_demo = TRUE,
                            status = 'active',
                            updated_at = $2
                        WHERE customer_id = $1
                        """,
                        customer_id,
                        now,
                    )
                else:
                    customer_id = f"cust_demo_{uuid4().hex[:20]}"
                    await conn.execute(
                        """
                        INSERT INTO customers (
                            customer_id, display_name, status, plan, region, retention_days, is_demo, created_at, updated_at
                        ) VALUES ($1, $2, 'active', 'starter', 'us-east-1', 30, TRUE, $3, $3)
                        """,
                        customer_id,
                        display_name,
                        now,
                    )
                kept_demo_customer_ids.append(customer_id)

            allowed_customer_ids = [live_customer_id, *kept_demo_customer_ids]
            await conn.execute(
                """
                DELETE FROM customers
                WHERE customer_id <> ALL($1::TEXT[])
                """,
                allowed_customer_ids,
            )

        total_customers = await conn.fetchval("SELECT COUNT(*)::INT FROM customers")

    await _audit(
        request=request,
        operator_id=claims.get("sub", "admin"),
        operator_role="admin",
        action="normalize_customers",
        target_resource="customer",
        target_id=live_customer_id,
    )
    await _emit_admin_usage(
        request=request,
        claims=claims,
        status_code=200,
        latency_ms=int((perf_counter() - started) * 1000),
    )

    return NormalizeCustomersResponse(
        kept_live_customer_id=live_customer_id,
        kept_demo_customer_ids=kept_demo_customer_ids,
        total_customers=total_customers,
    )


@router.patch("/customers/{customer_id}", response_model=Customer)
async def update_customer(
    customer_id: str,
    body: UpdateCustomerRequest,
    request: Request,
) -> Customer:
    started = perf_counter()
    claims = require_scope(request, {"admin:customers:write"})

    updates = {
        "display_name": body.display_name,
        "status": body.status.value if body.status else None,
        "plan": body.plan.value if body.plan else None,
        "region": body.region,
        "retention_days": body.retention_days,
        "is_demo": body.is_demo,
    }
    mutable_keys = [key for key, value in updates.items() if value is not None]
    if not mutable_keys:
        raise HTTPException(status_code=400, detail="No mutable fields provided")

    pool = get_pool()
    set_clause_parts = [f"{key} = ${idx}" for idx, key in enumerate(mutable_keys, start=2)]
    set_clause_parts.append(f"updated_at = ${len(mutable_keys) + 2}")
    query = f"""
    UPDATE customers
    SET {', '.join(set_clause_parts)}
    WHERE customer_id = $1
    RETURNING customer_id, display_name, status, plan, region, retention_days, is_demo, created_at, updated_at
    """
    values = [customer_id]
    values.extend(updates[key] for key in mutable_keys)
    values.append(datetime.now(timezone.utc))

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, *values)
        if not row:
            raise HTTPException(status_code=404, detail="Customer not found")

    data = dict(row)
    data["status"] = CustomerStatus(data["status"])
    data["plan"] = CustomerPlan(data["plan"])
    customer = Customer(**data)

    await _audit(
        request=request,
        operator_id=claims.get("sub", "admin"),
        operator_role="admin",
        action="update_customer",
        target_resource="customer",
        target_id=customer.customer_id,
    )
    await _emit_admin_usage(
        request=request,
        claims=claims,
        status_code=200,
        latency_ms=int((perf_counter() - started) * 1000),
    )
    return customer


@router.post("/customers/{customer_id}/workspaces", response_model=Workspace, status_code=201)
async def create_workspace(
    customer_id: str,
    body: CreateWorkspaceRequest,
    request: Request,
) -> Workspace:
    started = perf_counter()
    claims = require_scope(request, {"admin:customers:write"})
    workspace = Workspace(
        workspace_id=f"ws_{uuid4().hex[:20]}",
        customer_id=customer_id,
        display_name=body.display_name,
        environment=body.environment,
        status=WorkspaceStatus.ACTIVE,
        monthly_request_quota=body.monthly_request_quota,
        created_at=datetime.now(timezone.utc),
    )

    pool = get_pool()
    ensure_query = "SELECT customer_id FROM customers WHERE customer_id = $1"
    insert_query = """
    INSERT INTO workspaces (
      workspace_id, customer_id, display_name, environment, status,
      monthly_request_quota, created_at
    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
    """
    async with pool.acquire() as conn:
        exists = await conn.fetchrow(ensure_query, customer_id)
        if not exists:
            raise HTTPException(status_code=404, detail="Customer not found")
        await conn.execute(
            insert_query,
            workspace.workspace_id,
            workspace.customer_id,
            workspace.display_name,
            workspace.environment.value,
            workspace.status.value,
            workspace.monthly_request_quota,
            workspace.created_at,
        )

    await _audit(
        request=request,
        operator_id=claims.get("sub", "admin"),
        operator_role="admin",
        action="create_workspace",
        target_resource="workspace",
        target_id=workspace.workspace_id,
    )
    await _emit_admin_usage(
        request=request,
        claims=claims,
        status_code=201,
        latency_ms=int((perf_counter() - started) * 1000),
    )
    return workspace


@router.get("/customers/{customer_id}/workspaces", response_model=list[Workspace])
async def list_customer_workspaces(customer_id: str, request: Request) -> list[Workspace]:
    started = perf_counter()
    claims = require_scope(request, {"admin:customers:read"})

    pool = get_pool()
    query = """
    SELECT workspace_id, customer_id, display_name, environment, status, monthly_request_quota, created_at
    FROM workspaces
    WHERE customer_id = $1
    ORDER BY created_at DESC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, customer_id)

    result = []
    for row in rows:
        data = dict(row)
        data["environment"] = WorkspaceEnvironment(data["environment"])
        data["status"] = WorkspaceStatus(data["status"])
        result.append(Workspace(**data))

    await _emit_admin_usage(
        request=request,
        claims=claims,
        status_code=200,
        latency_ms=int((perf_counter() - started) * 1000),
    )
    return result


@router.post("/workspaces/{workspace_id}/credentials", response_model=CredentialIssuedResponse, status_code=201)
async def issue_workspace_credential(
    workspace_id: str,
    request: Request,
) -> CredentialIssuedResponse:
    started = perf_counter()
    claims = require_scope(request, {"admin:credentials:write"})

    pool = get_pool()
    ws_query = "SELECT customer_id FROM workspaces WHERE workspace_id = $1"
    async with pool.acquire() as conn:
        workspace = await conn.fetchrow(ws_query, workspace_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")

    raw_key = issue_api_key()
    key_hash, key_prefix = hash_api_key(raw_key)
    now = datetime.now(timezone.utc)
    credential_id = f"cred_{uuid4().hex[:20]}"

    insert_query = """
    INSERT INTO api_credentials (
      credential_id, workspace_id, customer_id, key_hash, key_prefix,
      scopes, is_active, created_at
    ) VALUES ($1, $2, $3, $4, $5, $6, TRUE, $7)
    """
    async with pool.acquire() as conn:
        await conn.execute(
            insert_query,
            credential_id,
            workspace_id,
            workspace["customer_id"],
            key_hash,
            key_prefix,
            ["memory:read", "memory:write"],
            now,
        )

    await _audit(
        request=request,
        operator_id=claims.get("sub", "admin"),
        operator_role="admin",
        action="issue_credential",
        target_resource="credential",
        target_id=credential_id,
    )

    response = CredentialIssuedResponse(
        credential_id=credential_id,
        raw_key=raw_key,
        key_prefix=key_prefix,
        scopes=["memory:read", "memory:write"],
        created_at=now,
    )
    await _emit_admin_usage(
        request=request,
        claims=claims,
        status_code=201,
        latency_ms=int((perf_counter() - started) * 1000),
    )
    return response


@router.get("/usage/{customer_id}")
async def get_customer_usage(
    customer_id: str,
    request: Request,
    workspace_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    include_admin: bool = Query(default=False),
) -> list[dict[str, object]]:
    started = perf_counter()
    claims = require_scope(request, {"admin:usage:read"})
    rows = await usage_service.get_daily_aggregates(
        customer_id=customer_id,
        workspace_id=workspace_id,
        date_from=date_from,
        date_to=date_to,
        include_admin=include_admin,
    )
    await _emit_admin_usage(
        request=request,
        claims=claims,
        status_code=200,
        latency_ms=int((perf_counter() - started) * 1000),
    )
    return [row.model_dump() for row in rows]


@router.get("/activity/recent")
async def get_recent_activity(
    request: Request,
    seconds: int = Query(default=10, ge=1, le=300),
) -> list[dict[str, object]]:
    started = perf_counter()
    claims = require_scope(request, {"admin:usage:read"})
    rows: list[RecentCustomerActivity] = await usage_service.get_recent_customer_activity(seconds=seconds)
    await _emit_admin_usage(
        request=request,
        claims=claims,
        status_code=200,
        latency_ms=int((perf_counter() - started) * 1000),
    )
    return [row.model_dump() for row in rows]


@router.get("/database/summary", response_model=DatabaseSummaryResponse)
async def get_database_summary(
    request: Request,
    window_seconds: int = Query(default=600, ge=10, le=86400),
) -> DatabaseSummaryResponse:
    started = perf_counter()
    claims = require_scope(request, {"admin:usage:read"})
    pool = get_pool()
    query = """
    SELECT
        (SELECT COUNT(*)::INT FROM customers) AS customers_total,
        (SELECT COUNT(*)::INT FROM workspaces) AS workspaces_total,
        (SELECT COUNT(*)::INT FROM api_credentials) AS credentials_total,
        (SELECT COUNT(*)::INT FROM api_credentials WHERE is_active = TRUE) AS active_credentials_total,
        (SELECT COUNT(*)::INT FROM memory_records) AS memory_records_total,
        (SELECT COUNT(*)::INT FROM memory_records WHERE is_active = TRUE) AS active_memory_records_total,
        (SELECT COUNT(*)::INT FROM usage_events) AS usage_events_total,
        (SELECT COUNT(*)::INT FROM audn_audit_log) AS audn_decisions_total,
        (SELECT COUNT(*)::INT FROM admin_audit_log) AS admin_audit_entries_total,
        (
            SELECT COUNT(DISTINCT customer_id)::INT
            FROM usage_events
            WHERE occurred_at >= NOW() - ($1::INT * INTERVAL '1 second')
        ) AS active_customers_last_window,
        (
            SELECT COUNT(DISTINCT workspace_id)::INT
            FROM usage_events
            WHERE occurred_at >= NOW() - ($1::INT * INTERVAL '1 second')
        ) AS active_workspaces_last_window,
        (
            SELECT COUNT(DISTINCT end_user_id)::INT
            FROM usage_events
            WHERE occurred_at >= NOW() - ($1::INT * INTERVAL '1 second')
        ) AS active_end_users_last_window,
        (
            SELECT COUNT(*)::INT
            FROM usage_events
            WHERE occurred_at >= NOW() - ($1::INT * INTERVAL '1 second')
              AND endpoint LIKE '/mcp/%'
        ) AS mcp_requests_last_window,
        (
            SELECT COUNT(*)::INT
            FROM usage_events
            WHERE occurred_at >= NOW() - ($1::INT * INTERVAL '1 second')
              AND endpoint LIKE '/admin/%'
        ) AS admin_requests_last_window,
        (SELECT MAX(updated_at) FROM memory_records) AS last_memory_write_at,
        (SELECT MAX(occurred_at) FROM usage_events) AS last_usage_event_at
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, window_seconds)

    if row is None:
        raise HTTPException(status_code=500, detail="Database summary unavailable")

    summary = DatabaseSummaryResponse(
        generated_at=datetime.now(timezone.utc),
        recent_window_seconds=window_seconds,
        **dict(row),
    )
    await _emit_admin_usage(
        request=request,
        claims=claims,
        status_code=200,
        latency_ms=int((perf_counter() - started) * 1000),
    )
    return summary


@router.post("/end-users/{end_user_id}/delete", status_code=202)
async def delete_end_user_data(
    end_user_id: str,
    body: EndUserDeleteRequest,
    request: Request,
) -> dict[str, str]:
    started = perf_counter()
    claims = require_scope(request, {"admin:privacy:write"})

    if body.end_user_id != end_user_id:
        raise HTTPException(status_code=400, detail="end_user_id path/body mismatch")

    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM memory_records WHERE customer_id = $1 AND workspace_id = $2 AND end_user_id = $3",
            body.customer_id,
            body.workspace_id,
            body.end_user_id,
        )
        await conn.execute(
            "DELETE FROM usage_events WHERE customer_id = $1 AND workspace_id = $2 AND end_user_id = $3",
            body.customer_id,
            body.workspace_id,
            body.end_user_id,
        )

    await _audit(
        request=request,
        operator_id=claims.get("sub", "admin"),
        operator_role="security",
        action="delete_end_user_data",
        target_resource="end_user",
        target_id=end_user_id,
    )
    await _emit_admin_usage(
        request=request,
        claims=claims,
        status_code=202,
        latency_ms=int((perf_counter() - started) * 1000),
    )

    return {"status": "accepted", "message": "End-user data purge completed"}
