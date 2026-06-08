from __future__ import annotations

import re
from typing import Any

from fastapi import BackgroundTasks, HTTPException, Request

from src.auth import require_auth
from src.models.schemas import MCPInitializationRequest, MCPResourceReadRequest, MCPStreamRequest, MCPStreamResponse, SendAndReceiveToolRequest, UpdateMemoryToolRequest


def _extract_meta(params: dict[str, Any]) -> dict[str, Any]:
    meta = params.get("_meta", {})
    return meta if isinstance(meta, dict) else {}


def _first_str(mapping: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _session_from_uri(uri: str) -> str | None:
    match = re.match(r"^genmind://sessions/([^/]+)/context$", uri.strip())
    if not match:
        return None
    session = match.group(1).strip()
    return session or None


def _inject_runtime_context(
    *,
    request: Request,
    params: dict[str, Any],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Stateless context resolver using Authorization claims + per-request _meta hints."""
    claims = require_auth(request)
    meta = _extract_meta(params)

    merged = dict(arguments)

    customer_id = (
        _first_str(merged, ["customer_id", "customerId"])
        or _first_str(meta, ["customer_id", "customerId"])
        or str(claims.get("customer_id", "")).strip()
    )
    workspace_id = (
        _first_str(merged, ["workspace_id", "workspaceId"])
        or _first_str(meta, ["workspace_id", "workspaceId"])
        or str(claims.get("workspace_id", "")).strip()
    )
    end_user_id = (
        _first_str(merged, ["end_user_id", "endUserId", "user_id", "userId"])
        or _first_str(meta, ["end_user_id", "endUserId", "user_id", "userId"])
        or str(claims.get("end_user_id", "system")).strip()
        or "system"
    )
    session_id = (
        _first_str(merged, ["session_id", "sessionId", "conversationId", "threadId"])
        or _first_str(meta, ["session_id", "sessionId", "conversationId", "threadId"])
        or str(claims.get("session_id", "system")).strip()
        or "system"
    )

    merged["customer_id"] = customer_id
    merged["workspace_id"] = workspace_id
    merged["end_user_id"] = end_user_id
    merged["session_id"] = session_id
    return merged


def _inject_runtime_tenant_for_resource(
    *,
    request: Request,
    params: dict[str, Any],
) -> dict[str, Any]:
    claims = require_auth(request)
    meta = _extract_meta(params)
    merged = dict(params)

    tenant = merged.get("tenant") if isinstance(merged.get("tenant"), dict) else {}
    tenant = dict(tenant)

    uri = str(merged.get("uri", "")).strip()
    uri_session = _session_from_uri(uri) if uri else None

    tenant["customer_id"] = (
        _first_str(tenant, ["customer_id", "customerId"])
        or _first_str(meta, ["customer_id", "customerId"])
        or str(claims.get("customer_id", "")).strip()
    )
    tenant["workspace_id"] = (
        _first_str(tenant, ["workspace_id", "workspaceId"])
        or _first_str(meta, ["workspace_id", "workspaceId"])
        or str(claims.get("workspace_id", "")).strip()
    )
    tenant["end_user_id"] = (
        _first_str(tenant, ["end_user_id", "endUserId", "user_id", "userId"])
        or _first_str(meta, ["end_user_id", "endUserId", "user_id", "userId"])
        or str(claims.get("end_user_id", "system")).strip()
        or "system"
    )
    tenant["session_id"] = (
        _first_str(tenant, ["session_id", "sessionId", "conversationId", "threadId"])
        or _first_str(meta, ["session_id", "sessionId", "conversationId", "threadId"])
        or uri_session
        or str(claims.get("session_id", "system")).strip()
        or "system"
    )

    merged["tenant"] = tenant
    merged.pop("_meta", None)
    return merged


async def handle_stream_request(
    *,
    body: MCPStreamRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    handle_initialization,
    handle_list_resources,
    handle_read_resource,
    handle_list_tools,
    handle_update_memory_state,
    handle_send_and_receive,
) -> MCPStreamResponse:
    """Dedicated stream handler module so /mcp/stream stays thin and testable."""
    method = body.method
    params = body.params

    try:
        if method == "initialize":
            init_params = dict(params)
            if "protocolVersion" in init_params and "protocol_version" not in init_params:
                init_params["protocol_version"] = init_params["protocolVersion"]
            init_params.pop("protocolVersion", None)
            if "client_name" not in init_params:
                init_params["client_name"] = "copilot-studio"
            if "client_version" not in init_params:
                init_params["client_version"] = "unknown"

            requested_protocol = str(init_params.get("protocol_version", "")).strip()
            req = MCPInitializationRequest.model_validate(init_params)
            result = (await handle_initialization(req, request)).model_dump()
            # Copilot Studio currently requests this legacy protocol version.
            if requested_protocol == "2024-11-05":
                result["protocol_version"] = "2024-11-05"
        elif method == "resources/list":
            result = await handle_list_resources(request)
        elif method == "resources/read":
            read_params = _inject_runtime_tenant_for_resource(request=request, params=params)
            req = MCPResourceReadRequest.model_validate(read_params)
            result = await handle_read_resource(req, request)
        elif method == "tools/list":
            result = await handle_list_tools(request)
        elif method == "tools/call":
            tool_name = str(params.get("name", "")).strip()
            if tool_name == "update_memory_state":
                enriched = _inject_runtime_context(
                    request=request,
                    params=params,
                    arguments=params.get("arguments", {}),
                )
                req = UpdateMemoryToolRequest.model_validate(enriched)
                result = await handle_update_memory_state(req, request, background_tasks)
            elif tool_name == "send_and_receive":
                enriched = _inject_runtime_context(
                    request=request,
                    params=params,
                    arguments=params.get("arguments", {}),
                )
                req = SendAndReceiveToolRequest.model_validate(enriched)
                result = await handle_send_and_receive(req, request)
            else:
                return MCPStreamResponse(
                    id=body.id,
                    error={"code": -32601, "message": f"Unknown tool: {tool_name}"},
                )
        else:
            return MCPStreamResponse(
                id=body.id,
                error={"code": -32601, "message": f"Unknown method: {method}"},
            )
    except HTTPException as exc:
        return MCPStreamResponse(
            id=body.id,
            error={"code": exc.status_code, "message": str(exc.detail)},
        )
    except Exception as exc:
        return MCPStreamResponse(
            id=body.id,
            error={"code": -32000, "message": str(exc)},
        )

    return MCPStreamResponse(id=body.id, result=result)
