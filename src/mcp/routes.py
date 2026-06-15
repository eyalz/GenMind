from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from src.auth import require_scope
from src.models.schemas import (
    MCPInitializationRequest,
    MCPInitializationResponse,
    MCPResourceReadRequest,
    MCPStreamRequest,
    MCPStreamResponse,
    RetrievalRequest,
    SendAndReceiveToolRequest,
    SendAndReceiveToolResponse,
    SessionTurnPayload,
    TenantContext,
    UpdateMemoryToolRequest,
)
from src.mcp.mcp_stream_handler import handle_stream_request
from src.simulator.local_recommendation_engine import analyze_local_recommendation
from src.services.audn_pipeline import AUDNPipeline
from src.services.context_optimizer import ContextOptimizer
from src.services.memory_engine import MemoryEngine
from src.services.usage_service import UsageService

router = APIRouter(prefix="/mcp", tags=["mcp"])


@lru_cache(maxsize=1)
def _get_usage_service() -> UsageService:
    return UsageService()


def _extract_session_id_from_resource_uri(uri: str) -> str:
    """Parse genmind://sessions/{session_id}/context into the session_id token."""
    prefix = "genmind://sessions/"
    suffix = "/context"
    if not uri.startswith(prefix) or not uri.endswith(suffix):
        raise HTTPException(status_code=400, detail="Invalid resource URI format.")

    session_id = uri[len(prefix) : -len(suffix)].strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is missing in URI.")
    return session_id


def _enforce_claim_tenant(claims: dict[str, object], tenant: TenantContext) -> None:
    claim_customer = str(claims.get("customer_id", ""))
    claim_workspace = str(claims.get("workspace_id", ""))
    if claim_customer != tenant.customer_id or claim_workspace != tenant.workspace_id:
        raise HTTPException(
            status_code=403,
            detail="Tenant mismatch between token claims and request payload.",
        )


def _build_session_context_uri(session_id: str) -> str:
    return f"genmind://sessions/{session_id}/context"


def _format_retrieval_result(
    *,
    uri: str,
    result,
    maker_id: str,
    agent_id: str,
) -> dict[str, object]:
    return {
        "uri": uri,
        "mime_type": "text/markdown",
        "contents": result.payload_markdown,
        "token_estimate": result.consumed_tokens_estimate,
        "selected_items": [candidate.model_dump() for candidate in result.selected_items],
        "maker_id": maker_id,
        "agent_id": agent_id,
    }


@lru_cache(maxsize=1)
def _get_memory_engine() -> MemoryEngine:
    return MemoryEngine()


@lru_cache(maxsize=1)
def _get_context_optimizer() -> ContextOptimizer:
    return ContextOptimizer(memory_engine=_get_memory_engine())


@lru_cache(maxsize=1)
def _get_audn_pipeline() -> AUDNPipeline:
    return AUDNPipeline(memory_engine=_get_memory_engine())


async def _handle_initialization(
    body: MCPInitializationRequest,
    request: Request,
) -> MCPInitializationResponse:
    require_scope(request, {"memory:read", "memory:write"})
    return MCPInitializationResponse(
        protocol_version="2026-01-01",
        server_name="genmind-mcp",
        server_version="0.1.0",
        transport="streamable/http",
        stream_endpoint="/mcp/stream",
        resources_endpoint="/mcp/resources",
        tools_endpoint="/mcp/tools",
    )


async def _handle_list_resources(request: Request) -> dict[str, object]:
    require_scope(request, {"memory:read"})
    return {
        "resources": [
            {
                "name": "session-context",
                "uri_template": "genmind://sessions/{session_id}/context",
                "description": (
                    "Hybrid-retrieved, time-decayed, tenant-scoped truth payload "
                    "for one session."
                ),
                "mime_type": "text/markdown",
            }
        ]
    }


async def _handle_read_resource(body: MCPResourceReadRequest, request: Request) -> dict[str, object]:
    claims = require_scope(request, {"memory:read"})
    _enforce_claim_tenant(claims, body.tenant)
    started = perf_counter()
    session_id = _extract_session_id_from_resource_uri(body.uri)
    if session_id != body.tenant.session_id:
        raise HTTPException(
            status_code=400,
            detail="Session ID mismatch between URI and tenant context.",
        )

    retrieval_request = RetrievalRequest(
        tenant=body.tenant,
        maker_id=body.maker_id,
        agent_id=body.agent_id,
        query=body.query,
        max_tokens=min(body.max_tokens, 1000),
    )

    optimizer = _get_context_optimizer()
    result = await optimizer.optimize(retrieval_request)
    stats = optimizer.last_stats
    latency_ms = int((perf_counter() - started) * 1000)
    request_id = request.headers.get("x-request-id", str(uuid4()))
    await _get_usage_service().emit_simple(
        customer_id=body.tenant.customer_id,
        workspace_id=body.tenant.workspace_id,
        end_user_id=body.tenant.end_user_id,
        session_id=body.tenant.session_id,
        request_id=request_id,
        endpoint="/mcp/resources",
        status_code=200,
        latency_ms=latency_ms,
        context_tokens=result.consumed_tokens_estimate,
        vector_reads=len(result.selected_items),
        graph_reads=0,
        retrieval_mode=stats.retrieval_mode if stats else "",
        top_k_selected=stats.top_k_selected if stats else 0,
        score_threshold_milli=int((stats.score_threshold * 1000)) if stats else 0,
        retrieval_candidates_total=stats.total_candidates if stats else 0,
        retrieval_candidates_kept=stats.kept_candidates if stats else 0,
        retrieval_conflicts_dropped=stats.conflicts_dropped if stats else 0,
        retrieval_claim_rows_reconciled=stats.claim_rows_reconciled if stats else 0,
        retrieval_light_memory_mode=stats.light_memory_mode if stats else False,
    )

    return _format_retrieval_result(
        uri=body.uri,
        result=result,
        maker_id=body.maker_id,
        agent_id=body.agent_id,
    )


async def _handle_list_tools(request: Request) -> dict[str, object]:
    require_scope(request, {"memory:write", "memory:read"})
    return {
        "tools": [
            {
                "name": "update_memory_state",
                "description": (
                    "Submits a session turn for asynchronous AUDN processing."
                ),
                "input_schema": {
                    "type": "object",
                    "required": [
                        "customer_id",
                        "workspace_id",
                        "maker_id",
                        "agent_id",
                        "end_user_id",
                        "session_id",
                        "user_input",
                        "model_output",
                    ],
                    "properties": {
                        "customer_id": {"type": "string"},
                        "workspace_id": {"type": "string"},
                        "maker_id": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "end_user_id": {"type": "string"},
                        "session_id": {"type": "string"},
                        "user_input": {"type": "string"},
                        "model_output": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                },
            },
            {
                "name": "send_and_receive",
                "description": (
                    "Processes one turn synchronously and returns refreshed session context "
                    "in the same response."
                ),
                "input_schema": {
                    "type": "object",
                    "required": [
                        "customer_id",
                        "workspace_id",
                        "maker_id",
                        "agent_id",
                        "end_user_id",
                        "session_id",
                        "user_input",
                        "model_output",
                    ],
                    "properties": {
                        "customer_id": {"type": "string"},
                        "workspace_id": {"type": "string"},
                        "maker_id": {"type": "string"},
                        "agent_id": {"type": "string"},
                        "end_user_id": {"type": "string"},
                        "session_id": {"type": "string"},
                        "user_input": {"type": "string"},
                        "model_output": {"type": "string"},
                        "query": {"type": "string"},
                        "max_tokens": {"type": "integer", "minimum": 100, "maximum": 2000},
                        "metadata": {"type": "object"},
                    },
                },
            }
        ]
    }


@router.post("/initialization", response_model=MCPInitializationResponse)
async def initialization(_: MCPInitializationRequest, request: Request) -> MCPInitializationResponse:
    return await _handle_initialization(_, request)


@router.get("/resources")
async def list_resources(request: Request) -> dict[str, object]:
    return await _handle_list_resources(request)


@router.post("/resources")
async def read_resource(body: MCPResourceReadRequest, request: Request) -> dict[str, object]:
    return await _handle_read_resource(body, request)


@router.get("/tools")
async def list_tools(request: Request) -> dict[str, object]:
    return await _handle_list_tools(request)


async def _run_audn_background(request: Request, payload: SessionTurnPayload) -> None:
    audn_pipeline = _get_audn_pipeline()
    decisions = await audn_pipeline.process_and_commit(payload)

    if hasattr(request.app.state, "sse_hub"):
        await request.app.state.sse_hub.publish(
            {
                "event": "audn.completed",
                "customer_id": payload.customer_id,
                "workspace_id": payload.workspace_id,
                "end_user_id": payload.end_user_id,
                "session_id": payload.session_id,
                "maker_id": payload.maker_id,
                "agent_id": payload.agent_id,
                "decision_count": len(decisions),
                "actions": [decision.action.value for decision in decisions],
            }
        )


async def _handle_update_memory_state(
    body: UpdateMemoryToolRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, object]:
    claims = require_scope(request, {"memory:write"})
    started = perf_counter()
    tenant = TenantContext(
        customer_id=body.customer_id,
        workspace_id=body.workspace_id,
        end_user_id=body.end_user_id,
        session_id=body.session_id,
    )
    _enforce_claim_tenant(claims, tenant)

    payload = SessionTurnPayload(
        customer_id=tenant.customer_id,
        workspace_id=tenant.workspace_id,
        end_user_id=tenant.end_user_id,
        session_id=tenant.session_id,
        user_input=body.user_input,
        model_output=body.model_output,
        maker_id=body.maker_id,
        agent_id=body.agent_id,
        metadata=body.metadata,
    )

    prior_history = await _get_memory_engine().list_recent_user_questions(
        tenant,
        maker_id=body.maker_id,
        agent_id=body.agent_id,
        limit=3,
    )
    session_state = await _get_memory_engine().get_session_context_state(
        tenant,
        maker_id=body.maker_id,
        agent_id=body.agent_id,
    )
    _, decision_payload = analyze_local_recommendation(
        current_query=body.user_input,
        history_list=prior_history,
        session_db_state=session_state,
    )
    context_snapshot = decision_payload.get("session_db_state", {}).get("context_snapshot", {})
    await _get_memory_engine().upsert_session_context_state(
        tenant,
        maker_id=body.maker_id,
        agent_id=body.agent_id,
        context_snapshot=context_snapshot if isinstance(context_snapshot, dict) else {},
    )

    await _get_memory_engine().append_recent_user_question(
        tenant=tenant,
        maker_id=body.maker_id,
        agent_id=body.agent_id,
        question_text=body.user_input,
        keep_last=3,
    )

    if hasattr(request.app.state, "sse_hub"):
        await request.app.state.sse_hub.publish(
            {
                "event": "audn.accepted",
                "customer_id": payload.customer_id,
                "workspace_id": payload.workspace_id,
                "end_user_id": payload.end_user_id,
                "session_id": payload.session_id,
                "maker_id": payload.maker_id,
                "agent_id": payload.agent_id,
            }
        )

    background_tasks.add_task(_run_audn_background, request, payload)

    latency_ms = int((perf_counter() - started) * 1000)
    request_id = request.headers.get("x-request-id", str(uuid4()))
    await _get_usage_service().emit_simple(
        customer_id=tenant.customer_id,
        workspace_id=tenant.workspace_id,
        end_user_id=tenant.end_user_id,
        session_id=tenant.session_id,
        request_id=request_id,
        endpoint="/mcp/tools/update_memory_state",
        status_code=202,
        latency_ms=latency_ms,
        tokens_in=len(body.user_input.split()),
        tokens_out=len(body.model_output.split()),
        memory_writes=1,
    )

    return {
        "status": "accepted",
        "tool": "update_memory_state",
        "tenant": tenant.model_dump(),
        "maker_id": body.maker_id,
        "agent_id": body.agent_id,
        "message": "AUDN ingestion queued for asynchronous processing.",
    }


async def _handle_send_and_receive(
    body: SendAndReceiveToolRequest,
    request: Request,
) -> dict[str, object]:
    claims = require_scope(request, {"memory:write", "memory:read"})
    started = perf_counter()
    tenant = TenantContext(
        customer_id=body.customer_id,
        workspace_id=body.workspace_id,
        end_user_id=body.end_user_id,
        session_id=body.session_id,
    )
    _enforce_claim_tenant(claims, tenant)

    payload = SessionTurnPayload(
        customer_id=tenant.customer_id,
        workspace_id=tenant.workspace_id,
        end_user_id=tenant.end_user_id,
        session_id=tenant.session_id,
        user_input=body.user_input,
        model_output=body.model_output,
        maker_id=body.maker_id,
        agent_id=body.agent_id,
        metadata=body.metadata,
    )

    prior_history = await _get_memory_engine().list_recent_user_questions(
        tenant,
        maker_id=body.maker_id,
        agent_id=body.agent_id,
        limit=3,
    )
    session_state = await _get_memory_engine().get_session_context_state(
        tenant,
        maker_id=body.maker_id,
        agent_id=body.agent_id,
    )
    _, decision_payload = analyze_local_recommendation(
        current_query=body.user_input,
        history_list=prior_history,
        session_db_state=session_state,
    )
    context_snapshot = decision_payload.get("session_db_state", {}).get("context_snapshot", {})
    await _get_memory_engine().upsert_session_context_state(
        tenant,
        maker_id=body.maker_id,
        agent_id=body.agent_id,
        context_snapshot=context_snapshot if isinstance(context_snapshot, dict) else {},
    )

    await _get_memory_engine().append_recent_user_question(
        tenant=tenant,
        maker_id=body.maker_id,
        agent_id=body.agent_id,
        question_text=body.user_input,
        keep_last=3,
    )

    audn_pipeline = _get_audn_pipeline()
    decisions = await audn_pipeline.process_and_commit(payload)
    uome_mutations = audn_pipeline.serialize_uome_mutations(decisions)

    retrieval_request = RetrievalRequest(
        tenant=tenant,
        maker_id=body.maker_id,
        agent_id=body.agent_id,
        query=body.query,
        max_tokens=min(body.max_tokens, 1000),
    )
    optimizer = _get_context_optimizer()
    retrieval_result = await optimizer.optimize(retrieval_request)
    stats = optimizer.last_stats

    if hasattr(request.app.state, "sse_hub"):
        await request.app.state.sse_hub.publish(
            {
                "event": "audn.completed",
                "customer_id": payload.customer_id,
                "workspace_id": payload.workspace_id,
                "end_user_id": payload.end_user_id,
                "session_id": payload.session_id,
                "maker_id": payload.maker_id,
                "agent_id": payload.agent_id,
                "decision_count": len(decisions),
                "actions": [decision.action.value for decision in decisions],
                "mode": "send_and_receive",
            }
        )

    latency_ms = int((perf_counter() - started) * 1000)
    request_id = request.headers.get("x-request-id", str(uuid4()))
    await _get_usage_service().emit_simple(
        customer_id=tenant.customer_id,
        workspace_id=tenant.workspace_id,
        end_user_id=tenant.end_user_id,
        session_id=tenant.session_id,
        request_id=request_id,
        endpoint="/mcp/tools/send_and_receive",
        status_code=200,
        latency_ms=latency_ms,
        tokens_in=len(body.user_input.split()),
        tokens_out=len(body.model_output.split()),
        context_tokens=retrieval_result.consumed_tokens_estimate,
        vector_reads=len(retrieval_result.selected_items),
        graph_reads=0,
        memory_writes=sum(1 for decision in decisions if decision.action.value != "none"),
        retrieval_mode=stats.retrieval_mode if stats else "",
        top_k_selected=stats.top_k_selected if stats else 0,
        score_threshold_milli=int((stats.score_threshold * 1000)) if stats else 0,
        retrieval_candidates_total=stats.total_candidates if stats else 0,
        retrieval_candidates_kept=stats.kept_candidates if stats else 0,
        retrieval_conflicts_dropped=stats.conflicts_dropped if stats else 0,
        retrieval_claim_rows_reconciled=stats.claim_rows_reconciled if stats else 0,
        retrieval_light_memory_mode=stats.light_memory_mode if stats else False,
    )

    response = SendAndReceiveToolResponse(
        tenant=tenant,
        maker_id=body.maker_id,
        agent_id=body.agent_id,
        uri=_build_session_context_uri(tenant.session_id),
        contents=retrieval_result.payload_markdown,
        token_estimate=retrieval_result.consumed_tokens_estimate,
        selected_items=retrieval_result.selected_items,
        decision_count=len(decisions),
        decisions=decisions,
        uome_mutations=uome_mutations,
    )
    return response.model_dump()


@router.post("/tools/update_memory_state")
async def update_memory_state(
    body: UpdateMemoryToolRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, object]:
    return await _handle_update_memory_state(body, request, background_tasks)


@router.post("/tools/send_and_receive", response_model=SendAndReceiveToolResponse)
async def send_and_receive(
    body: SendAndReceiveToolRequest,
    request: Request,
) -> SendAndReceiveToolResponse:
    response = await _handle_send_and_receive(body, request)
    return SendAndReceiveToolResponse.model_validate(response)


@router.post("/stream", response_model=MCPStreamResponse)
async def streamable_mcp(
    body: MCPStreamRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> MCPStreamResponse:
    return await handle_stream_request(
        body=body,
        request=request,
        background_tasks=background_tasks,
        handle_initialization=_handle_initialization,
        handle_list_resources=_handle_list_resources,
        handle_read_resource=_handle_read_resource,
        handle_list_tools=_handle_list_tools,
        handle_update_memory_state=_handle_update_memory_state,
        handle_send_and_receive=_handle_send_and_receive,
    )
