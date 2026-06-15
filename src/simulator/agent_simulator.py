from __future__ import annotations

import ast
from pathlib import Path
import json
import os
import re
import threading
import time
import logging
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

# Import the new smart engine
from src.simulator.local_recommendation_engine import LocalRecommendationEngine


class SimulatorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    api_base: str = Field(default_factory=lambda: os.getenv("GENMIND_API_BASE", "http://127.0.0.1:8000"))
    customer_name: str = Field(default_factory=lambda: os.getenv("SIM_CUSTOMER_NAME", "Copilot Studio Test 1"))
    customer_id: str = Field(default_factory=lambda: os.getenv("SIM_CUSTOMER_ID", ""))
    workspace_id: str = Field(default_factory=lambda: os.getenv("SIM_WORKSPACE_ID", ""))
    bootstrap_customer_id: str = Field(default_factory=lambda: os.getenv("SIM_BOOTSTRAP_CUSTOMER_ID", "cust_dev"))
    bootstrap_workspace_id: str = Field(default_factory=lambda: os.getenv("SIM_BOOTSTRAP_WORKSPACE_ID", "ws_dev"))
    maker_id: str = Field(default_factory=lambda: os.getenv("SIM_MAKER_ID", "maker_default"))
    agent_id: str = Field(default_factory=lambda: os.getenv("SIM_AGENT_ID", "Test_Agent"))
    end_user_id: str = Field(default_factory=lambda: os.getenv("SIM_END_USER_ID", "Test_User"))
    default_session_id: str = Field(default_factory=lambda: os.getenv("SIM_DEFAULT_SESSION_ID", "session_simulated_1"))
    llm_provider: str = Field(default_factory=lambda: os.getenv("SIM_LLM_PROVIDER", "github_copilot"))
    llm_api_base: str = Field(default_factory=lambda: os.getenv("SIM_LLM_API_BASE", "https://api.githubcopilot.com"))
    llm_api_key: str = Field(
        default_factory=lambda: os.getenv("SIM_GITHUB_COPILOT_TOKEN", os.getenv("SIM_LLM_API_KEY", ""))
    )
    llm_model: str = Field(default_factory=lambda: os.getenv("SIM_LLM_MODEL", "gpt-4.1"))
    copilot_integration_id: str = Field(default_factory=lambda: os.getenv("SIM_COPILOT_INTEGRATION_ID", ""))
    copilot_editor_version: str = Field(default_factory=lambda: os.getenv("SIM_COPILOT_EDITOR_VERSION", "vscode/1.100.0"))
    copilot_editor_plugin_version: str = Field(
        default_factory=lambda: os.getenv("SIM_COPILOT_EDITOR_PLUGIN_VERSION", "copilot-chat/0.29.0")
    )
    llm_timeout_seconds: int = Field(default_factory=lambda: int(os.getenv("SIM_LLM_TIMEOUT_SECONDS", "30")), ge=5, le=120)
    token_expires_minutes: int = Field(default_factory=lambda: int(os.getenv("SIM_TOKEN_EXPIRES_MINUTES", "60")), ge=1, le=240)
    fixed_access_token: str = Field(default_factory=lambda: os.getenv("SIM_ACCESS_TOKEN", ""))
    dev_bootstrap_header: str = Field(default_factory=lambda: os.getenv("SIM_DEV_BOOTSTRAP_HEADER", "allow"))
    web_provider: str = Field(default_factory=lambda: os.getenv("SIM_WEB_PROVIDER", "auto"))
    tavily_api_key: str = Field(default_factory=lambda: os.getenv("SIM_TAVILY_API_KEY", ""))
    tavily_mcp_url: str = Field(default_factory=lambda: os.getenv("SIM_TAVILY_MCP_URL", ""))
    tavily_api_base: str = Field(default_factory=lambda: os.getenv("SIM_TAVILY_API_BASE", "https://api.tavily.com/search"))
    tavily_search_depth: str = Field(default_factory=lambda: os.getenv("SIM_TAVILY_SEARCH_DEPTH", "advanced"))
    brave_api_key: str = Field(default_factory=lambda: os.getenv("SIM_BRAVE_API_KEY", ""))
    brave_api_base: str = Field(
        default_factory=lambda: os.getenv("SIM_BRAVE_API_BASE", "https://api.search.brave.com/res/v1/web/search")
    )
    brave_country: str = Field(default_factory=lambda: os.getenv("SIM_BRAVE_COUNTRY", "IL"))
    brave_search_lang: str = Field(default_factory=lambda: os.getenv("SIM_BRAVE_SEARCH_LANG", "en"))
    brave_ui_lang: str = Field(default_factory=lambda: os.getenv("SIM_BRAVE_UI_LANG", "en-IL"))
    bing_api_key: str = Field(default_factory=lambda: os.getenv("SIM_BING_API_KEY", ""))
    bing_api_base: str = Field(default_factory=lambda: os.getenv("SIM_BING_API_BASE", "https://api.bing.microsoft.com/v7.0/search"))
    bing_market: str = Field(default_factory=lambda: os.getenv("SIM_BING_MARKET", "en-US"))


class SimulateChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    user_input: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, min_length=2, max_length=256)
    end_user_id: str | None = Field(default=None, min_length=2, max_length=256)
    query: str = Field(default="latest user preferences and context", min_length=1, max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SimulateChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    customer_id: str
    workspace_id: str
    maker_id: str
    agent_id: str
    end_user_id: str
    session_id: str
    initialized: bool
    tool_result: dict[str, Any]
    context_result: dict[str, Any]
    simulated_model_output: str


class FlowTraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    actor: str
    direction: str
    title: str
    summary: str
    endpoint: str | None = None
    status: str = "ok"
    metadata: dict[str, Any] = Field(default_factory=dict)


class WebKnowledgeItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    title: str
    snippet: str
    url: str


class ChatFlowResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    customer_id: str
    workspace_id: str
    maker_id: str
    agent_id: str
    end_user_id: str
    session_id: str
    user_input: str
    customer_prompt: str
    final_answer: str
    context_excerpt: str
    db_session_snapshot: str
    decision_count: int
    memory_writes: int
    memory_write_reason: str
    action_breakdown: dict[str, int]
    extracted_candidates: list[str]
    persisted_facts: list[str]
    selected_items: int
    web_knowledge: list[WebKnowledgeItem]
    flow: list[FlowTraceStep]


class LiveSimulationStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    interval_seconds: int = Field(default=10, ge=1, le=3600)
    session_id: str | None = Field(default=None, min_length=2, max_length=256)
    end_user_id: str | None = Field(default=None, min_length=2, max_length=256)
    query: str = Field(default="latest user preferences and context", min_length=1, max_length=2000)
    questions: list[str] = Field(
        default_factory=lambda: [
            "What should I remember from our previous chat?",
            "I prefer concise answers, keep that in memory.",
            "Remind me weekly about my top priorities.",
        ],
        min_length=1,
    )


class LiveSimulationStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    running: bool
    interval_seconds: int
    call_count: int
    session_id: str
    end_user_id: str
    last_question: str
    last_error: str
    last_result_status: str


settings = SimulatorSettings()
app = FastAPI(title="GenMind Agent Simulator", version="0.1.0")

# Logging setup
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_DEBUG_LOG = _LOG_DIR / "simulator_debug.log"

_debug_logger = logging.getLogger("simulator_debug")
_debug_logger.setLevel(logging.DEBUG)
if not _debug_logger.handlers:
    handler = logging.FileHandler(_DEBUG_LOG)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    _debug_logger.addHandler(handler)

def _log_debug(msg: str, **kw) -> None:
    extras = " ".join(f"{k}={v!r}" for k, v in kw.items())
    full_msg = f"{msg} {extras}" if extras else msg
    _debug_logger.debug(full_msg)
SIMULATOR_UI_PATH = Path(__file__).with_name("chat_ui.html")


_token_cache: dict[str, dict[str, Any]] = {}
_tenant_cache: dict[str, Any] = {"customer_id": "", "workspace_id": "", "expires_at": 0.0}
_last_numeric_answer_by_session: dict[str, float] = {}
_last_user_question_by_session: dict[str, str] = {}
_recent_user_questions_by_session: dict[str, list[str]] = {}
_EPHEMERAL_OUTPUT_MARKER = "[SIMULATOR_EPHEMERAL]"
_live_lock = threading.Lock()
_live_stop_event = threading.Event()
_live_thread: threading.Thread | None = None
_live_state: dict[str, Any] = {
    "running": False,
    "interval_seconds": 10,
    "call_count": 0,
    "session_id": settings.default_session_id,
    "end_user_id": settings.end_user_id,
    "last_question": "",
    "last_error": "",
    "last_result_status": "idle",
}

_MAX_RECENT_QUESTIONS = 3


def _http_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    url = urljoin(f"{settings.api_base.rstrip('/')}/", path.lstrip("/"))
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)

    request = Request(url=url, data=body, headers=request_headers, method=method)

    try:
        with urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=f"Upstream HTTP error: {error_body}") from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream connectivity error: {exc}") from exc


def _fetch_json_url(url: str, *, extra_headers: dict[str, str] | None = None) -> dict[str, Any]:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "GenMindSimulator/1.0 (+https://genmind.local)",
    }
    if extra_headers:
        request_headers.update(extra_headers)

    request = Request(
        url=url,
        headers=request_headers,
        method="GET",
    )
    try:
        with urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=f"External lookup HTTP error: {error_body}") from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"External lookup connectivity error: {exc}") from exc


def _fetch_json_url_with_params(
    base_url: str,
    params: dict[str, str],
    *,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    query = "&".join(f"{quote_plus(k)}={quote_plus(v)}" for k, v in params.items())
    return _fetch_json_url(f"{base_url}?{query}", extra_headers=extra_headers)


def _post_json_url(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        request_headers.update(headers)

    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.llm_timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=exc.code, detail=f"External POST HTTP error: {error_body}") from exc
    except URLError as exc:
        raise HTTPException(status_code=502, detail=f"External POST connectivity error: {exc}") from exc


def _get_access_token(customer_id: str, workspace_id: str) -> str:
    if settings.fixed_access_token:
        return settings.fixed_access_token

    now = time.time()
    cache_key = f"{customer_id}::{workspace_id}"
    entry = _token_cache.get(cache_key, {})
    cached = entry.get("token")
    expires_at = float(entry.get("expires_at", 0.0))
    if cached and now < expires_at:
        return str(cached)

    token_payload = {
        "customer_id": customer_id,
        "workspace_id": workspace_id,
        "end_user_id": "simulator_service",
        "session_id": "simulator_service",
        "scopes": ["admin:*", "memory:read", "memory:write"],
        "expires_minutes": settings.token_expires_minutes,
    }
    token_response = _http_json(
        "POST",
        "/admin/dev/token",
        token_payload,
        headers={"x-dev-bootstrap": settings.dev_bootstrap_header},
    )

    token = str(token_response.get("access_token", ""))
    if not token:
        raise HTTPException(status_code=500, detail="Token endpoint did not return access_token")

    _token_cache[cache_key] = {
        "token": token,
        "expires_at": now + (settings.token_expires_minutes * 60) - 30,
    }
    return token


def _authorized_get(path: str) -> dict[str, Any] | list[dict[str, Any]]:
    token = _get_access_token(settings.bootstrap_customer_id, settings.bootstrap_workspace_id)
    return _http_json("GET", path, headers={"Authorization": f"Bearer {token}"})


def _resolve_target_tenant() -> tuple[str, str]:
    now = time.time()
    cached_customer = str(_tenant_cache.get("customer_id", ""))
    cached_workspace = str(_tenant_cache.get("workspace_id", ""))
    expires_at = float(_tenant_cache.get("expires_at", 0.0))
    if cached_customer and cached_workspace and now < expires_at:
        return cached_customer, cached_workspace

    target_customer_id = settings.customer_id.strip()
    target_workspace_id = settings.workspace_id.strip()

    if not target_customer_id:
        rows = _authorized_get("/admin/customers")
        if not isinstance(rows, list):
            raise HTTPException(status_code=500, detail="Invalid /admin/customers response")

        match = next((row for row in rows if row.get("display_name") == settings.customer_name), None)
        if not match:
            raise HTTPException(
                status_code=404,
                detail=f"Customer '{settings.customer_name}' not found. Set SIM_CUSTOMER_ID explicitly if needed.",
            )
        target_customer_id = str(match.get("customer_id", "")).strip()

    if not target_workspace_id:
        ws_rows = _authorized_get(f"/admin/customers/{target_customer_id}/workspaces")
        if not isinstance(ws_rows, list):
            raise HTTPException(status_code=500, detail="Invalid /admin/customers/{id}/workspaces response")
        if not ws_rows:
            raise HTTPException(
                status_code=400,
                detail=f"Customer {target_customer_id} has no workspaces. Create one before running simulator.",
            )
        target_workspace_id = str(ws_rows[0].get("workspace_id", "")).strip()

    if not target_customer_id or not target_workspace_id:
        raise HTTPException(status_code=500, detail="Failed to resolve target customer/workspace.")

    _tenant_cache["customer_id"] = target_customer_id
    _tenant_cache["workspace_id"] = target_workspace_id
    _tenant_cache["expires_at"] = now + 300
    return target_customer_id, target_workspace_id


def _stream_call(method: str, params: dict[str, Any], customer_id: str, workspace_id: str) -> dict[str, Any]:
    token = _get_access_token(customer_id, workspace_id)
    stream_headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json, text/event-stream",
        "Origin": os.getenv("SIM_MCP_ORIGIN", "http://127.0.0.1:5173").strip() or "http://127.0.0.1:5173",
    }
    stream_response = _http_json(
        "POST",
        "/mcp/stream",
        {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": method,
            "params": params,
        },
        headers=stream_headers,
    )

    if stream_response.get("error"):
        raise HTTPException(status_code=502, detail=f"MCP stream error: {stream_response['error']}")

    result = stream_response.get("result")
    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="MCP stream result missing or invalid")
    return result


def _send_and_receive(
    *,
    customer_id: str,
    workspace_id: str,
    end_user_id: str,
    session_id: str,
    user_input: str,
    model_output: str,
    query: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return _stream_call(
        "tools/call",
        {
            "name": "send_and_receive",
            "arguments": {
                "customer_id": customer_id,
                "workspace_id": workspace_id,
                "maker_id": settings.maker_id,
                "agent_id": settings.agent_id,
                "end_user_id": end_user_id,
                "session_id": session_id,
                "user_input": user_input,
                "model_output": model_output,
                "query": query,
                "max_tokens": 1200,
                "metadata": metadata,
            },
        },
        customer_id,
        workspace_id,
    )


def _context_result_from_one_shot(one_shot_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "uri": one_shot_result.get("uri", ""),
        "mime_type": "text/markdown",
        "contents": one_shot_result.get("contents", ""),
        "token_estimate": one_shot_result.get("token_estimate", 0),
        "selected_items": one_shot_result.get("selected_items", []),
        "maker_id": one_shot_result.get("maker_id", settings.maker_id),
        "agent_id": one_shot_result.get("agent_id", settings.agent_id),
    }


def _normalize_action_name(raw: Any) -> str:
    """Normalize action values from MCP payloads into add/update/delete/none."""
    if isinstance(raw, dict):
        # Some serializers may wrap enum/value fields.
        for key in ("value", "action", "name"):
            if key in raw:
                return _normalize_action_name(raw.get(key))
        return "none"

    text = str(raw or "none").strip().lower()
    if "." in text:
        text = text.split(".")[-1]

    if text in {"add", "update", "delete", "none"}:
        return text

    for action_name in ("add", "update", "delete", "none"):
        if action_name in text:
            return action_name

    return "none"


def _sanitize_context_text(contents: str, *, fallback: str, max_chars: int) -> str:
    if not contents.strip():
        return fallback

    # Hide simulator-internal placeholder entries from the visible context excerpt.
    filtered_lines = [
        line
        for line in contents.splitlines()
        if _EPHEMERAL_OUTPUT_MARKER not in line and "Pending customer-agent answer generation" not in line
    ]
    sanitized = "\n".join(filtered_lines).strip()
    if not sanitized:
        return fallback
    return sanitized[:max_chars]


def _summarize_context(context_result: dict[str, Any]) -> str:
    contents = str(context_result.get("contents", ""))
    return _sanitize_context_text(contents, fallback="No memory context returned.", max_chars=500)


def _read_session_db_snapshot(
    *,
    customer_id: str,
    workspace_id: str,
    end_user_id: str,
    session_id: str,
    query: str,
) -> str:
    resource_uri = f"genmind://sessions/{session_id}/context"
    snapshot_query = (
        "full session memory snapshot for this session, include remembered facts and recent turns"
    )
    resource_result = _stream_call(
        "resources/read",
        {
            "uri": resource_uri,
            "tenant": {
                "customer_id": customer_id,
                "workspace_id": workspace_id,
                "end_user_id": end_user_id,
                "session_id": session_id,
            },
            "maker_id": settings.maker_id,
            "agent_id": settings.agent_id,
            "query": snapshot_query,
            "max_tokens": 1500,
        },
        customer_id,
        workspace_id,
    )
    contents = str(resource_result.get("contents", ""))
    return _sanitize_context_text(contents, fallback="No session snapshot returned.", max_chars=700)


def _is_local_recommendation_query(question: str) -> bool:
    """
    Determine if query is a local recommendation query.
    Uses the smart dual-signal LocalRecommendationEngine instead of static lists.
    
    Returns True if query seeks local recommendations (location + intent).
    """
    result, decision_matrix = LocalRecommendationEngine.evaluate(question)
    
    # Log the detailed decision matrix for debugging
    _log_debug(f"is_local_recommendation_query={result}", 
               question=question,
               reasoning=decision_matrix.get("reasoning", ""),
               target_subject=decision_matrix.get("metrics", {}).get("dynamic_target_subject"))
    
    return result


def _is_offtopic_for_local_recommendation(item: WebKnowledgeItem) -> bool:
    text = f"{item.title} {item.snippet}".lower()
    blocked_terms = (
        "violence",
        "war",
        "gaza",
        "riot",
        "protester",
        "hospital",
        "jail",
        "shooting",
        "attack",
        "killed",
        "injured",
        "deaths",
        "ptsd",
        "uefa",
        "ajax",
        "maccabi",
        "f.c.",
        "football",
        "match",
    )
    return any(term in text for term in blocked_terms)


def _is_venue_like_result(item: WebKnowledgeItem) -> bool:
    text = f"{item.title} {item.snippet}".lower()
    venue_terms = (
        "bar",
        "bars",
        "pub",
        "nightlife",
        "cocktail",
        "rooftop",
        "wine",
        "club",
        "restaurant",
        "cafe",
        "district",
        "neighborhood",
        "where to drink",
        "where to go",
    )
    return any(term in text for term in venue_terms)


def _relevance_score(question: str, item: WebKnowledgeItem) -> int:
    text = f"{item.title} {item.snippet}".lower()
    question_tokens = {
        token
        for token in re.findall(r"[a-z0-9]{3,}", question.lower())
        if token not in {"with", "from", "this", "that", "have", "nearby", "where", "there", "your", "about"}
    }
    item_tokens = set(re.findall(r"[a-z0-9]{3,}", text))
    overlap = len(question_tokens & item_tokens)

    venue_terms = ("bar", "bars", "pub", "nightlife", "cocktail", "district", "neighborhood", "jaffa", "florentin", "dizengoff", "rothschild")
    geopolitics_terms = ("war", "military", "relations", "conflict", "security", "diplomatic", "uefa", "match", "football", "maccabi", "ajax")

    score = overlap
    if any(term in text for term in venue_terms):
        score += 3
    if any(term in text for term in geopolitics_terms):
        score -= 4
    return score


def _is_blocked_or_low_value_snippet(item: WebKnowledgeItem) -> bool:
    text = f"{item.title} {item.snippet}".lower()
    blocked_terms = (
        "sign in",
        "log in",
        "login",
        "subscribe",
        "subscription",
        "paywall",
        "members only",
        "enable javascript",
        "cookie settings",
        "accept cookies",
        "continue with",
        "create account",
        "register to continue",
    )
    if any(term in text for term in blocked_terms):
        return True
    # Drop snippets that contain almost no lexical content.
    token_count = len(re.findall(r"[a-z0-9]{2,}", item.snippet.lower()))
    return token_count < 6


def _prioritize_relevant_items(question: str, items: list[WebKnowledgeItem]) -> list[WebKnowledgeItem]:
    if not items:
        return items

    items = [item for item in items if not _is_blocked_or_low_value_snippet(item)]
    if not items:
        return []

    if _is_local_recommendation_query(question):
        filtered = [item for item in items if not _is_offtopic_for_local_recommendation(item)]
        items = filtered

        venue_like = [item for item in items if _is_venue_like_result(item)]
        if venue_like:
            items = venue_like

        ranked = sorted(items, key=lambda item: _relevance_score(question, item), reverse=True)
        relevant = [item for item in ranked if _relevance_score(question, item) > 0]
        if relevant:
            items = relevant
        else:
            items = []

    else:
        items = sorted(items, key=lambda item: _relevance_score(question, item), reverse=True)

    return items[:3]


def _extract_preferred_location(question: str, context_excerpt: str) -> str | None:
    ctx_match = re.search(r"profile\.search_location=([^|\n]+)", context_excerpt, flags=re.IGNORECASE)
    if ctx_match:
        location = ctx_match.group(1).strip()
        if location:
            _log_debug(f"extract_location=context", location=location)
            return location

    q_match = re.search(r"\bin\s+([a-zA-Z][a-zA-Z\s\-']{1,50})\b", question, flags=re.IGNORECASE)
    if q_match:
        location = q_match.group(1).strip(" ?.,!;")
        if location:
            location = " ".join(location.split())
            _log_debug(f"extract_location=question", location=location)
            return location

    _log_debug("extract_location=none")
    return None


def _location_aliases(location: str) -> set[str]:
    normalized = " ".join(location.strip().lower().split())
    aliases: set[str] = {normalized}
    alias_map: dict[str, set[str]] = {
        "tel aviv": {"tel aviv", "tel-aviv", "tel aviv-yafo", "tlv", "jaffa", "yafo"},
        "new york": {"new york", "nyc", "new york city", "manhattan", "brooklyn", "queens"},
        "san francisco": {"san francisco", "sf", "bay area"},
    }
    aliases.update(alias_map.get(normalized, set()))
    return aliases


def _is_item_in_location(item: WebKnowledgeItem, location: str) -> bool:
    haystack = f"{item.title} {item.snippet} {item.url}".lower()
    aliases = _location_aliases(location)
    matches = any(alias in haystack for alias in aliases)
    if not matches:
        _log_debug(f"item_location_mismatch", item_title=item.title[:50], location=location, aliases=list(aliases)[:3])
    return matches


def _filter_web_knowledge_by_location(
    *,
    question: str,
    context_excerpt: str,
    items: list[WebKnowledgeItem],
) -> tuple[list[WebKnowledgeItem], str | None]:
    preferred_location = _extract_preferred_location(question, context_excerpt)
    if not preferred_location:
        _log_debug(f"no_location_filter_applied", items_count=len(items))
        return items, None

    if not _is_local_recommendation_query(question):
        _log_debug(f"not_local_recommendation_skipping_filter", items_count=len(items), location=preferred_location)
        return items, preferred_location

    scoped = [item for item in items if _is_item_in_location(item, preferred_location)]
    _log_debug(f"location_filter_applied", total_items=len(items), scoped_items=len(scoped), location=preferred_location)
    if scoped:
        for item in scoped:
            _log_debug(f"  kept_item", title=item.title[:60])
        return scoped, preferred_location

    _log_debug(f"location_filter_zero_match_fallback", location=preferred_location)
    # Hard safety fallback: better to return no web snippets than inject wrong-city recommendations.
    return [], preferred_location


def _extract_latest_search_topic(context_excerpt: str) -> str | None:
    matches = re.findall(r"profile\.search_topic=([^|\n]+)", context_excerpt, flags=re.IGNORECASE)
    if not matches:
        return None
    latest = matches[-1].strip()
    return latest or None


def _extract_followup_topic(question: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", question.strip().lower())
    cleaned = cleaned.strip(" ?.!;")
    cleaned = re.sub(r"^(and\s+for|and|what\s+about|how\s+about)\s+", "", cleaned)
    if not cleaned:
        return None
    return cleaned


def _is_brief_local_followup(question: str) -> bool:
    """Heuristic for terse continuation queries like 'and tennis court ?'."""
    lowered = re.sub(r"\s+", " ", question.strip().lower())
    if not lowered:
        return False

    followup_markers = (
        "and ",
        "and for ",
        "what about ",
        "how about ",
        "also ",
    )
    has_marker = any(lowered.startswith(marker) for marker in followup_markers)
    if not has_marker:
        return False

    topic = _extract_followup_topic(question)
    if not topic:
        return False

    # Reject clearly informational/analytical follow-ups.
    blocked = (
        "history",
        "population",
        "weather",
        "distance",
        "compare",
        "comparison",
        "language",
        "timezone",
    )
    if any(term in topic for term in blocked):
        return False

    token_count = len(re.findall(r"[a-z0-9]+", topic))
    return 1 <= token_count <= 5


def _should_treat_as_local_query(question: str, preferred_location: str | None = None) -> bool:
    """Decide whether to enforce local recommendation behavior for this turn."""
    if _is_local_recommendation_query(question):
        return True

    if preferred_location:
        followup_topic = _extract_followup_topic(question)
        if followup_topic:
            synthetic = f"best {followup_topic} in {preferred_location}"
            if _is_local_recommendation_query(synthetic):
                return True

    return _is_brief_local_followup(question)


def _build_location_aware_query(question: str, context_excerpt: str) -> str:
    preferred_location = _extract_preferred_location(question, context_excerpt)
    if not preferred_location:
        return question

    if not _should_treat_as_local_query(question, preferred_location):
        return question

    if re.search(r"\bin\s+[a-zA-Z]", question, flags=re.IGNORECASE):
        return question

    followup_topic = _extract_followup_topic(question)
    if followup_topic:
        return f"best {followup_topic} in {preferred_location}"

    latest_topic = _extract_latest_search_topic(context_excerpt)
    if latest_topic:
        return f"{latest_topic.replace('_', ' ')} in {preferred_location}"

    return question


def _fetch_bing_web_knowledge(question: str) -> list[WebKnowledgeItem]:
    if not settings.bing_api_key.strip():
        return []

    try:
        payload = _fetch_json_url_with_params(
            settings.bing_api_base,
            {
                "q": question,
                "mkt": settings.bing_market,
                "count": "5",
                "responseFilter": "Webpages",
                "safeSearch": "Moderate",
                "textFormat": "Raw",
            },
            extra_headers={"Ocp-Apim-Subscription-Key": settings.bing_api_key.strip()},
        )
    except HTTPException:
        return []

    web_pages = payload.get("webPages", {})
    values = web_pages.get("value", []) if isinstance(web_pages, dict) else []
    if not isinstance(values, list):
        return []

    items: list[WebKnowledgeItem] = []
    for row in values:
        if len(items) >= 5:
            break
        if not isinstance(row, dict):
            continue
        title = str(row.get("name", "")).strip()
        snippet = str(row.get("snippet", "")).strip()
        url = str(row.get("url", "")).strip()
        if not title or not snippet:
            continue
        items.append(
            WebKnowledgeItem(
                title=title[:120],
                snippet=snippet[:600],
                url=url or "https://www.bing.com/",
            )
        )

    return items


def _resolve_tavily_api_key() -> str:
    direct = settings.tavily_api_key.strip()
    if direct:
        return direct

    mcp_url = settings.tavily_mcp_url.strip()
    if not mcp_url:
        return ""

    try:
        parsed = urlparse(mcp_url)
        query = parse_qs(parsed.query)
    except ValueError:
        return ""

    values = query.get("tavilyApiKey", [])
    if not values:
        return ""
    return str(values[0]).strip()


def _fetch_tavily_web_knowledge(question: str) -> list[WebKnowledgeItem]:
    api_key = _resolve_tavily_api_key()
    if not api_key:
        return []

    try:
        payload = _post_json_url(
            settings.tavily_api_base,
            {
                "api_key": api_key,
                "query": question,
                "search_depth": settings.tavily_search_depth,
                "max_results": 8,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
            },
        )
    except HTTPException:
        return []

    rows = payload.get("results", [])
    if not isinstance(rows, list):
        return []

    items: list[WebKnowledgeItem] = []
    for row in rows:
        if len(items) >= 8:
            break
        if not isinstance(row, dict):
            continue
        title = str(row.get("title", "")).strip()
        snippet = str(row.get("content", "")).strip()
        url = str(row.get("url", "")).strip()
        if not title or not snippet:
            continue
        items.append(
            WebKnowledgeItem(
                title=title[:120],
                snippet=snippet[:600],
                url=url or "https://tavily.com/",
            )
        )

    return items


def _fetch_brave_web_knowledge(question: str) -> list[WebKnowledgeItem]:
    if not settings.brave_api_key.strip():
        return []

    try:
        payload = _fetch_json_url_with_params(
            settings.brave_api_base,
            {
                "q": question,
                "count": "8",
                "safesearch": "moderate",
                "result_filter": "web,locations",
                "country": settings.brave_country,
                "search_lang": settings.brave_search_lang,
                "ui_lang": settings.brave_ui_lang,
                "text_decorations": "false",
            },
            extra_headers={
                "X-Subscription-Token": settings.brave_api_key.strip(),
                "Accept-Encoding": "gzip",
                "Cache-Control": "no-cache",
            },
        )
    except HTTPException:
        return []

    items: list[WebKnowledgeItem] = []

    web = payload.get("web", {})
    web_results = web.get("results", []) if isinstance(web, dict) else []
    if isinstance(web_results, list):
        for row in web_results:
            if len(items) >= 8:
                break
            if not isinstance(row, dict):
                continue
            title = str(row.get("title", "")).strip()
            snippet = str(row.get("description", "")).strip()
            url = str(row.get("url", "")).strip()
            if not title or not snippet:
                continue
            items.append(
                WebKnowledgeItem(
                    title=title[:120],
                    snippet=snippet[:600],
                    url=url or "https://search.brave.com/",
                )
            )

    locations = payload.get("locations", {})
    location_results = locations.get("results", []) if isinstance(locations, dict) else []
    if isinstance(location_results, list):
        for row in location_results:
            if len(items) >= 8:
                break
            if not isinstance(row, dict):
                continue
            title = str(row.get("title", "")).strip()
            description = str(row.get("description", "")).strip()
            url = str(row.get("url", "")).strip()
            if not title:
                continue
            snippet = description or "Location result from Brave Search."
            items.append(
                WebKnowledgeItem(
                    title=title[:120],
                    snippet=snippet[:600],
                    url=url or "https://search.brave.com/",
                )
            )

    return items


def _fetch_web_knowledge(question: str) -> list[WebKnowledgeItem]:
    provider = settings.web_provider.strip().lower() or "auto"
    items: list[WebKnowledgeItem] = []
    _log_debug(f"fetch_web_knowledge_start", provider=provider, question=question[:60])

    if provider in {"auto", "tavily"}:
        tavily_items = _fetch_tavily_web_knowledge(question)
        _log_debug(f"  tavily_result", count=len(tavily_items))
        items.extend(tavily_items)
        if provider == "tavily" and items:
            result = _prioritize_relevant_items(question, items)
            _log_debug(f"fetch_web_knowledge_end", provider="tavily", final_count=len(result))
            return result

    if provider in {"auto", "brave"}:
        brave_items = _fetch_brave_web_knowledge(question)
        _log_debug(f"  brave_result", count=len(brave_items))
        items.extend(brave_items)
        if provider == "brave" and items:
            result = _prioritize_relevant_items(question, items)
            _log_debug(f"fetch_web_knowledge_end", provider="brave", final_count=len(result))
            return result

    if provider in {"auto", "bing"}:
        bing_items = _fetch_bing_web_knowledge(question)
        _log_debug(f"  bing_result", count=len(bing_items))
        items.extend(bing_items)
        if provider == "bing" and items:
            result = _prioritize_relevant_items(question, items)
            _log_debug(f"fetch_web_knowledge_end", provider="bing", final_count=len(result))
            return result

    lookup_url = (
        "https://api.duckduckgo.com/?"
        f"q={quote_plus(question)}&format=json&no_html=1&skip_disambig=1"
    )
    ddg_error_detail = ""
    payload: dict[str, Any] = {}
    if provider in {"auto", "duckduckgo"} and not items:
        try:
            payload = _fetch_json_url(lookup_url)
        except HTTPException as exc:
            payload = {}
            ddg_error_detail = str(exc.detail)

    if payload:
        abstract_text = str(payload.get("AbstractText", "")).strip()
        abstract_url = str(payload.get("AbstractURL", "")).strip()
        heading = str(payload.get("Heading", "")).strip() or "Web summary"
        if abstract_text:
            items.append(
                WebKnowledgeItem(
                    title=heading,
                    snippet=abstract_text[:600],
                    url=abstract_url or "https://duckduckgo.com/",
                )
            )

        related_topics = payload.get("RelatedTopics", [])
        if isinstance(related_topics, list):
            for topic in related_topics:
                if len(items) >= 3:
                    break
                if not isinstance(topic, dict):
                    continue
                if "Topics" in topic and isinstance(topic["Topics"], list):
                    nested_topics = topic["Topics"]
                else:
                    nested_topics = [topic]

                for nested in nested_topics:
                    if len(items) >= 3:
                        break
                    if not isinstance(nested, dict):
                        continue
                    text = str(nested.get("Text", "")).strip()
                    first_url = str(nested.get("FirstURL", "")).strip()
                    if not text:
                        continue
                    title = text.split(" - ", 1)[0]
                    items.append(
                        WebKnowledgeItem(
                            title=title[:120] or "Web reference",
                            snippet=text[:600],
                            url=first_url or "https://duckduckgo.com/",
                        )
                    )

    if not items:
        wikipedia_items = _fetch_wikipedia_knowledge(question)
        if wikipedia_items:
            items.extend(wikipedia_items)

    if not items:
        fallback_snippet = "No external knowledge summary was returned for this question."
        if ddg_error_detail:
            fallback_snippet = f"External lookup failed: {ddg_error_detail}"
        items.append(
            WebKnowledgeItem(
                title="Web lookup unavailable",
                snippet=fallback_snippet,
                url="https://duckduckgo.com/",
            )
        )

    return _prioritize_relevant_items(question, items)


def _fetch_wikipedia_knowledge(question: str) -> list[WebKnowledgeItem]:
    try:
        search_payload = _fetch_json_url_with_params(
            "https://en.wikipedia.org/w/api.php",
            {
                "action": "query",
                "list": "search",
                "srsearch": question,
                "srlimit": "3",
                "format": "json",
                "utf8": "1",
            },
        )
    except HTTPException:
        return []

    query_obj = search_payload.get("query", {})
    search_results = query_obj.get("search", []) if isinstance(query_obj, dict) else []
    if not isinstance(search_results, list):
        return []

    items: list[WebKnowledgeItem] = []
    for result in search_results:
        if len(items) >= 3:
            break
        if not isinstance(result, dict):
            continue

        title = str(result.get("title", "")).strip()
        if not title:
            continue

        page_url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
        summary_text = _fetch_wikipedia_extract(title)
        if not summary_text:
            snippet_html = str(result.get("snippet", "")).strip()
            summary_text = re.sub(r"<[^>]+>", "", snippet_html)
        summary_text = summary_text.strip()
        if not summary_text:
            continue

        items.append(
            WebKnowledgeItem(
                title=title[:120],
                snippet=summary_text[:600],
                url=page_url,
            )
        )

    return items


def _fetch_wikipedia_extract(title: str) -> str:
    try:
        payload = _fetch_json_url_with_params(
            "https://en.wikipedia.org/w/api.php",
            {
                "action": "query",
                "prop": "extracts",
                "explaintext": "1",
                "exintro": "1",
                "titles": title,
                "format": "json",
                "utf8": "1",
            },
        )
    except HTTPException:
        return ""

    query_obj = payload.get("query", {})
    if not isinstance(query_obj, dict):
        return ""
    pages = query_obj.get("pages", {})
    if not isinstance(pages, dict):
        return ""

    for page in pages.values():
        if not isinstance(page, dict):
            continue
        extract = str(page.get("extract", "")).strip()
        if extract:
            return extract
    return ""


def _build_customer_prompt(
    question: str,
    context_excerpt: str,
    web_knowledge: list[WebKnowledgeItem],
    *,
    preferred_location: str | None = None,
) -> str:
    web_block = "\n\n".join(
        f"- {item.title}: {item.snippet}" for item in web_knowledge[:3]
    )
    location_guard = f"Required location for recommendations: {preferred_location}\n\n" if preferred_location else ""
    return (
        "Customer agent prompt\n"
        f"User question: {question}\n\n"
        f"{location_guard}"
        f"GenMind context:\n{context_excerpt}\n\n"
        f"Web knowledge:\n{web_block}\n\n"
        "Generate a concise answer that uses current web knowledge while respecting remembered preferences from GenMind. "
        "If the user asks a follow-up without a location, carry forward the latest location from GenMind context. "
        "Reject off-location web snippets that conflict with remembered location context. "
        "Do not provide venue recommendations for any city other than the required location."
    )


def _append_recent_user_question(session_id: str, question: str) -> None:
    text = question.strip()
    if not text:
        return

    bucket = _recent_user_questions_by_session.setdefault(session_id, [])
    bucket.append(text)
    if len(bucket) > _MAX_RECENT_QUESTIONS:
        _recent_user_questions_by_session[session_id] = bucket[-_MAX_RECENT_QUESTIONS:]


def _recent_user_questions_block(session_id: str) -> str:
    items = _recent_user_questions_by_session.get(session_id, [])
    if not items:
        return ""

    lines = ["Recent user questions (full text, oldest to newest):"]
    for idx, item in enumerate(items, start=1):
        lines.append(f"{idx}. {item}")
    return "\n".join(lines)


def _compose_llm_context(context_excerpt: str, db_session_snapshot: str, *, recent_questions_block: str = "") -> str:
    parts: list[str] = []

    primary = context_excerpt.strip()
    snapshot = db_session_snapshot.strip()

    if primary and primary != "No memory context returned.":
        parts.append(f"send_and_receive context:\n{primary}")

    if snapshot and snapshot != "No session snapshot returned." and snapshot != primary:
        parts.append(f"resources/read session snapshot:\n{snapshot}")

    if recent_questions_block.strip():
        parts.append(recent_questions_block.strip())

    if not parts:
        return "No memory context returned."

    return "\n\n".join(parts)[:1400]


def _evaluate_math_expression(expression: str) -> float | None:
    cleaned = expression.strip().replace("^", "**")
    if not cleaned or len(cleaned) > 120:
        return None

    try:
        node = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return None

    def _eval(expr: ast.AST) -> float:
        if isinstance(expr, ast.Expression):
            return _eval(expr.body)
        if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
            return float(expr.value)
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
            operand = _eval(expr.operand)
            return operand if isinstance(expr.op, ast.UAdd) else -operand
        if isinstance(expr, ast.BinOp):
            left = _eval(expr.left)
            right = _eval(expr.right)
            if isinstance(expr.op, ast.Add):
                return left + right
            if isinstance(expr.op, ast.Sub):
                return left - right
            if isinstance(expr.op, ast.Mult):
                return left * right
            if isinstance(expr.op, ast.Div):
                return left / right
            if isinstance(expr.op, ast.FloorDiv):
                return left // right
            if isinstance(expr.op, ast.Mod):
                return left % right
            if isinstance(expr.op, ast.Pow):
                return left**right
        raise ValueError("Unsupported expression")

    try:
        return _eval(node)
    except (ValueError, ZeroDivisionError):
        return None


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.8g}"


def _extract_direct_math_expression(question: str) -> str | None:
    text = question.strip()
    lowered = text.lower().rstrip("?.! ")

    match = re.search(r"(?:how much is|what is|calculate|compute)\s+(.+)$", lowered)
    if match:
        candidate = match.group(1).strip()
        candidate = re.sub(r"[^0-9+\-*/().\s]", "", candidate)
        return candidate or None

    if re.fullmatch(r"[0-9+\-*/().\s]+", text):
        return text.strip()

    return None


def _infer_recent_numeric_value(context_excerpt: str) -> float | None:
    quoted_questions = re.findall(r"about '([^']+)'", context_excerpt, flags=re.IGNORECASE)
    for quoted in reversed(quoted_questions):
        expression = _extract_direct_math_expression(quoted)
        if expression:
            result = _evaluate_math_expression(expression)
            if result is not None:
                return result

    expression_matches = re.findall(r"(-?\d+(?:\.\d+)?(?:\s*[-+*/]\s*-?\d+(?:\.\d+)?)+)", context_excerpt)
    for expression in reversed(expression_matches):
        result = _evaluate_math_expression(expression)
        if result is not None:
            return result
    return None


def _build_customer_answer(
    question: str,
    context_excerpt: str,
    web_knowledge: list[WebKnowledgeItem],
    session_id: str,
) -> str:
    direct_expression = _extract_direct_math_expression(question)
    if direct_expression:
        computed = _evaluate_math_expression(direct_expression)
        if computed is not None:
            _last_numeric_answer_by_session[session_id] = computed
            return f"The answer is {_format_number(computed)}."

    multiply_match = re.search(
        r"(?:multiply|multiple)\s+(?:it\s+)?by\s+(-?\d+(?:\.\d+)?)",
        question.lower(),
    )
    if multiply_match:
        factor = float(multiply_match.group(1))
        base = _last_numeric_answer_by_session.get(session_id)
        if base is None:
            base = _infer_recent_numeric_value(context_excerpt)
        if base is not None:
            result = base * factor
            _last_numeric_answer_by_session[session_id] = result
            return (
                f"Using remembered context, {_format_number(base)} multiplied by {_format_number(factor)} "
                f"is {_format_number(result)}."
            )
        return "I can multiply by that factor, but I need the base value to multiply."

    lead = web_knowledge[0].snippet if web_knowledge else "No web knowledge available."
    if _is_local_recommendation_query(question) and _is_offtopic_for_local_recommendation(
        WebKnowledgeItem(title=web_knowledge[0].title if web_knowledge else "", snippet=lead, url="")
    ):
        lead = "I could not find a trustworthy nearby-venue web snippet, so I will answer using location context only."
    if lead.startswith("No external knowledge summary"):
        lead = "No relevant external web summary was returned for this question."

    if context_excerpt and context_excerpt != "No memory context returned.":
        return (
            f"Based on current web knowledge, {lead} "
            f"I also used the remembered context from GenMind to answer your question about '{question}'."
        )
    return f"Based on current web knowledge, {lead} This answer uses external knowledge for '{question}'."


def _generate_customer_llm_answer(
    *,
    question: str,
    context_excerpt: str,
    customer_prompt: str,
    web_knowledge: list[WebKnowledgeItem],
    session_id: str,
) -> str:
    # Deterministic fallback when no model credentials are configured.
    if not settings.llm_api_key.strip():
        return _build_customer_answer(question, context_excerpt, web_knowledge, session_id)

    endpoint = f"{settings.llm_api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.llm_model,
        "temperature": 0.2,
        "max_tokens": 260,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a customer-facing agent. Answer using the user's question and provided GenMind context. "
                    "Be concise, factual, and do not invent missing facts."
                ),
            },
            {
                "role": "user",
                "content": customer_prompt,
            },
        ],
    }

    request_headers: dict[str, str] = {"Authorization": f"Bearer {settings.llm_api_key.strip()}"}
    if settings.llm_provider.strip().lower() == "github_copilot":
        request_headers.update(
            {
                "Editor-Version": settings.copilot_editor_version,
                "Editor-Plugin-Version": settings.copilot_editor_plugin_version,
                "User-Agent": "GenMind-Simulator/0.1",
            }
        )
        integration_id = settings.copilot_integration_id.strip()
        if integration_id:
            request_headers["Copilot-Integration-Id"] = integration_id

    try:
        llm_response = _post_json_url(
            endpoint,
            payload,
            headers=request_headers,
        )
    except HTTPException:
        return _build_customer_answer(question, context_excerpt, web_knowledge, session_id)

    choices = llm_response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(content, str) and content.strip():
            return content.strip()

    return _build_customer_answer(question, context_excerpt, web_knowledge, session_id)


def _run_chat_flow_internal(
    *,
    user_input: str,
    session_id: str | None,
    end_user_id: str | None,
    query: str,
    metadata: dict[str, Any],
) -> ChatFlowResponse:
    target_customer_id, target_workspace_id = _resolve_target_tenant()
    effective_session_id = session_id or settings.default_session_id
    effective_end_user_id = end_user_id or settings.end_user_id
    flow: list[FlowTraceStep] = []

    _stream_call(
        "initialize",
        {
            "protocol_version": "2026-01-01",
            "client_name": "customer-web-simulator",
            "client_version": "1.0.0",
        },
        target_customer_id,
        target_workspace_id,
    )
    flow.append(
        FlowTraceStep(
            actor="simulator",
            direction="Simulator -> GenMind",
            title="initialize",
            summary="The simulator opens an MCP session and asks GenMind for capabilities.",
            endpoint="/mcp/stream initialize",
            status="ok",
            metadata={
                "client_name": "customer-web-simulator",
                "client_version": "1.0.0",
            },
        )
    )

    previous_question = _last_user_question_by_session.get(effective_session_id, "").strip()
    prior_questions = list(_recent_user_questions_by_session.get(effective_session_id, []))
    recent_block = ""
    if prior_questions:
        joined = "\n".join(f"- {q}" for q in prior_questions)
        recent_block = f"recent user turns (full text):\n{joined}"

    effective_query = user_input
    if recent_block:
        effective_query = f"{user_input}\n{recent_block}"
    elif previous_question:
        effective_query = (
            f"{user_input}\n"
            f"follow-up context from previous user turn: {previous_question}"
        )

    one_shot_result = _send_and_receive(
        customer_id=target_customer_id,
        workspace_id=target_workspace_id,
        end_user_id=effective_end_user_id,
        session_id=effective_session_id,
        user_input=user_input,
        model_output=f"{_EPHEMERAL_OUTPUT_MARKER} Pending customer-agent answer generation.",
        query=effective_query,
        metadata={
            **metadata,
            "source": "simulator-chat-ui",
            "stage": "question_received",
            "previous_user_input": previous_question,
            "recent_user_inputs": prior_questions,
        },
    )
    _log_debug(f"send_and_receive_response", 
               status=one_shot_result.get("status"),
               decision_count=one_shot_result.get("decision_count", 0),
               uri=one_shot_result.get("uri", "")[:60])
    context_result = _context_result_from_one_shot(one_shot_result)
    db_session_snapshot = _read_session_db_snapshot(
        customer_id=target_customer_id,
        workspace_id=target_workspace_id,
        end_user_id=effective_end_user_id,
        session_id=effective_session_id,
        query=query,
    )
    decision_count = one_shot_result.get("decision_count", 0)
    decisions_raw = one_shot_result.get("decisions", [])
    decisions_list = [decision for decision in decisions_raw if isinstance(decision, dict)] if isinstance(decisions_raw, list) else []

    action_breakdown = {"add": 0, "update": 0, "delete": 0, "none": 0}
    extracted_candidates: list[str] = []
    for decision in decisions_list:
        action = _normalize_action_name(decision.get("action", "none"))
        if action in action_breakdown:
            action_breakdown[action] += 1
        candidate_fact = str(decision.get("candidate_fact", "")).strip()
        if candidate_fact and candidate_fact not in extracted_candidates:
            extracted_candidates.append(candidate_fact)

    # If some decision actions were unrecognized by format, classify remaining as none.
    parsed_actions_total = sum(action_breakdown.values())
    if decision_count > parsed_actions_total:
        action_breakdown["none"] += decision_count - parsed_actions_total
    
    _log_debug(f"audn_decisions_parsed", 
               total_decisions=decision_count,
               parsed_decisions=len(decisions_list),
               action_breakdown=action_breakdown,
               extracted_count=len(extracted_candidates))

    memory_writes = action_breakdown["add"] + action_breakdown["update"] + action_breakdown["delete"]
    persisted_facts: list[str] = []
    uome_mutations = one_shot_result.get("uome_mutations", [])
    if isinstance(uome_mutations, list):
        for mutation in uome_mutations:
            if not isinstance(mutation, dict):
                continue
            mutation_action = _normalize_action_name(mutation.get("action", "none"))
            if mutation_action == "none":
                continue
            target = str(mutation.get("target_property_or_entity", "")).strip()
            value_raw = mutation.get("value")
            if isinstance(value_raw, (dict, list)):
                value = json.dumps(value_raw, ensure_ascii=True)
            else:
                value = str(value_raw).strip()
            summary = f"{target} = {value}" if target and value else str(mutation.get("reasoning_justification", "")).strip()
            if summary and summary not in persisted_facts:
                persisted_facts.append(summary)

    # Fallback: back-compute persisted facts from retrieved items if uome_mutations is empty
    # This ensures we show what actually made it into storage.
    if not persisted_facts and memory_writes > 0:
        selected_items_list = context_result.get("selected_items", [])
        if isinstance(selected_items_list, list):
            for item in selected_items_list:
                if isinstance(item, dict):
                    content = str(item.get("content", "")).strip()
                    if content and content not in persisted_facts:
                        persisted_facts.append(content)

    # If we found persisted facts but no explicit extracted candidates, use persisted facts as candidates
    if not extracted_candidates and persisted_facts:
        extracted_candidates = persisted_facts
        # Also update memory_writes to match: if we found facts in storage but no action breakdown, infer 1 write per fact
        if memory_writes == 0:
            memory_writes = len(persisted_facts)
            action_breakdown["add"] = memory_writes

    if decision_count == 0:
        memory_write_reason = "No extractable fact matched AUDN rules for this turn."
    elif memory_writes == 0:
        memory_write_reason = "All AUDN decisions resolved to no-op actions."
    else:
        memory_write_reason = "AUDN produced persisted memory mutations."
    selected_items = len(context_result.get("selected_items", []))
    flow.append(
        FlowTraceStep(
            actor="simulator",
            direction="Simulator -> GenMind",
            title="send_and_receive request",
            summary="The simulator sends the user message and asks GenMind to write memory and return the updated context in one shot.",
            endpoint="/mcp/stream tools/call:send_and_receive",
            status="ok",
            metadata={
                "session_id": effective_session_id,
                "end_user_id": effective_end_user_id,
                "query": query[:120],
                "user_input": user_input[:160],
            },
        )
    )
    context_excerpt = _summarize_context(context_result)
    flow.append(
        FlowTraceStep(
            actor="genmind",
            direction="GenMind -> Simulator",
            title="send_and_receive response",
            summary="GenMind returns the write outcome, decision summary, and refreshed tenant-scoped context to the simulator.",
            endpoint="/mcp/tools/send_and_receive",
            status="ok",
            metadata={
                "status": one_shot_result.get("status", "completed"),
                "decision_count": one_shot_result.get("decision_count", 0),
                "memory_writes": memory_writes,
                "memory_write_reason": memory_write_reason,
                "action_breakdown": action_breakdown,
                "extracted_candidates": extracted_candidates[:3],
                "persisted_facts": persisted_facts[:3],
                "token_estimate": context_result.get("token_estimate", 0),
                "selected_items": len(context_result.get("selected_items", [])),
                "context_excerpt": context_excerpt[:180],
                "db_session_snapshot": db_session_snapshot[:180],
            },
        )
    )

    recent_questions_block = _recent_user_questions_block(effective_session_id)
    llm_context = _compose_llm_context(
        context_excerpt,
        db_session_snapshot,
        recent_questions_block=recent_questions_block,
    )
    web_query = _build_location_aware_query(user_input, llm_context)
    _log_debug(f"web_knowledge_fetch_initiated", query=web_query[:80])
    fetched_web_knowledge = _fetch_web_knowledge(web_query)
    _log_debug(f"web_knowledge_fetched", count=len(fetched_web_knowledge))
    for i, item in enumerate(fetched_web_knowledge[:3]):
        _log_debug(f"  item_{i}", title=item.title[:60])
    
    web_knowledge, preferred_location = _filter_web_knowledge_by_location(
        question=user_input,
        context_excerpt=llm_context,
        items=fetched_web_knowledge,
    )
    _log_debug(f"web_knowledge_after_filter", count=len(web_knowledge), location=preferred_location)

    if not web_knowledge and preferred_location and _should_treat_as_local_query(user_input, preferred_location):
        fallback_topic = _extract_followup_topic(user_input) or _extract_latest_search_topic(llm_context) or "places"
        targeted_query = f"best {fallback_topic.replace('_', ' ')} in {preferred_location}"
        _log_debug(f"web_knowledge_fallback_retry", query=targeted_query)
        fetched_web_knowledge = _fetch_web_knowledge(targeted_query)
        _log_debug(f"web_knowledge_fallback_fetched", count=len(fetched_web_knowledge))
        web_knowledge, preferred_location = _filter_web_knowledge_by_location(
            question=user_input,
            context_excerpt=llm_context,
            items=fetched_web_knowledge,
        )

    customer_prompt = _build_customer_prompt(
        user_input,
        llm_context,
        web_knowledge,
        preferred_location=preferred_location,
    )
    _log_debug(f"customer_prompt_built", 
               length=len(customer_prompt),
               web_items_used=len(web_knowledge),
               preferred_location=preferred_location)
    
    final_answer = _generate_customer_llm_answer(
        question=user_input,
        context_excerpt=llm_context,
        customer_prompt=customer_prompt,
        web_knowledge=web_knowledge,
        session_id=effective_session_id,
    )
    _log_debug(f"llm_answer_generated", length=len(final_answer))
    _last_user_question_by_session[effective_session_id] = user_input
    _append_recent_user_question(effective_session_id, user_input)

    return ChatFlowResponse(
        customer_id=target_customer_id,
        workspace_id=target_workspace_id,
        maker_id=settings.maker_id,
        agent_id=settings.agent_id,
        end_user_id=effective_end_user_id,
        session_id=effective_session_id,
        user_input=user_input,
        customer_prompt=customer_prompt,
        final_answer=final_answer,
        context_excerpt=context_excerpt,
        db_session_snapshot=db_session_snapshot,
        decision_count=decision_count,
        memory_writes=memory_writes,
        memory_write_reason=memory_write_reason,
        action_breakdown=action_breakdown,
        extracted_candidates=extracted_candidates,
        persisted_facts=persisted_facts,
        selected_items=selected_items,
        web_knowledge=web_knowledge,
        flow=flow,
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/simulator/config")
def simulator_config() -> dict[str, str]:
    try:
        resolved_customer_id, resolved_workspace_id = _resolve_target_tenant()
    except HTTPException as exc:
        resolved_customer_id = ""
        resolved_workspace_id = ""
        resolution_error = str(exc.detail)
    else:
        resolution_error = ""

    provider = settings.web_provider.strip().lower() or "auto"
    tavily_missing = provider in {"auto", "tavily"} and not _resolve_tavily_api_key()
    brave_missing = provider in {"auto", "brave"} and not settings.brave_api_key.strip()
    if provider == "tavily" and tavily_missing:
        web_provider_warning = (
            "Tavily web search is selected but no Tavily key was found. "
            "Set SIM_TAVILY_API_KEY or SIM_TAVILY_MCP_URL with tavilyApiKey."
        )
    elif provider == "brave" and brave_missing:
        web_provider_warning = (
            "Brave web search is selected but SIM_BRAVE_API_KEY is missing. "
            "Set SIM_BRAVE_API_KEY in your shell profile to enable live Brave results."
        )
    elif provider == "auto" and tavily_missing and brave_missing and not settings.bing_api_key.strip():
        web_provider_warning = (
            "No web provider API key is configured for auto mode. "
            "Configure Tavily, Brave, or Bing for richer live web results."
        )
    else:
        web_provider_warning = ""

    return {
        "api_base": settings.api_base,
        "llm_provider": settings.llm_provider,
        "llm_api_base": settings.llm_api_base,
        "llm_model": settings.llm_model,
        "llm_configured": "true" if bool(settings.llm_api_key.strip()) else "false",
        "web_provider": settings.web_provider,
        "web_provider_warning": web_provider_warning,
        "tavily_configured": "true" if bool(_resolve_tavily_api_key()) else "false",
        "tavily_api_base": settings.tavily_api_base,
        "brave_configured": "true" if bool(settings.brave_api_key.strip()) else "false",
        "brave_api_base": settings.brave_api_base,
        "bing_configured": "true" if bool(settings.bing_api_key.strip()) else "false",
        "bing_api_base": settings.bing_api_base,
        "customer_name": settings.customer_name,
        "customer_id": settings.customer_id,
        "workspace_id": settings.workspace_id,
        "resolved_customer_id": resolved_customer_id,
        "resolved_workspace_id": resolved_workspace_id,
        "resolution_error": resolution_error,
        "maker_id": settings.maker_id,
        "agent_id": settings.agent_id,
        "end_user_id": settings.end_user_id,
        "default_session_id": settings.default_session_id,
    }


def _simulate_chat_internal(
    *,
    user_input: str,
    session_id: str | None,
    end_user_id: str | None,
    query: str,
    metadata: dict[str, Any],
) -> SimulateChatResponse:
    target_customer_id, target_workspace_id = _resolve_target_tenant()
    effective_session_id = session_id or settings.default_session_id
    effective_end_user_id = end_user_id or settings.end_user_id
    simulated_output = f"Acknowledged for {effective_end_user_id}: {user_input}"

    _stream_call(
        "initialize",
        {
            "protocol_version": "2026-01-01",
            "client_name": "local-agent-simulator",
            "client_version": "1.0.0",
        },
        target_customer_id,
        target_workspace_id,
    )

    tool_result = _send_and_receive(
        customer_id=target_customer_id,
        workspace_id=target_workspace_id,
        end_user_id=effective_end_user_id,
        session_id=effective_session_id,
        user_input=user_input,
        model_output=simulated_output,
        query=query,
        metadata={
            **metadata,
            "source": "local-agent-simulator",
        },
    )
    context_result = _context_result_from_one_shot(tool_result)

    return SimulateChatResponse(
        customer_id=target_customer_id,
        workspace_id=target_workspace_id,
        maker_id=settings.maker_id,
        agent_id=settings.agent_id,
        end_user_id=effective_end_user_id,
        session_id=effective_session_id,
        initialized=True,
        tool_result=tool_result,
        context_result=context_result,
        simulated_model_output=simulated_output,
    )


def _live_worker(
    *,
    interval_seconds: int,
    session_id: str | None,
    end_user_id: str | None,
    query: str,
    questions: list[str],
) -> None:
    index = 0
    while not _live_stop_event.is_set():
        question = questions[index % len(questions)]
        index += 1

        try:
            _simulate_chat_internal(
                user_input=question,
                session_id=session_id,
                end_user_id=end_user_id,
                query=query,
                metadata={"mode": "live-loop", "sequence": index},
            )
            with _live_lock:
                _live_state["call_count"] += 1
                _live_state["last_question"] = question
                _live_state["last_error"] = ""
                _live_state["last_result_status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            with _live_lock:
                _live_state["last_question"] = question
                _live_state["last_error"] = str(exc)
                _live_state["last_result_status"] = "error"

        if _live_stop_event.wait(interval_seconds):
            break

    with _live_lock:
        _live_state["running"] = False


@app.post("/simulate/chat", response_model=SimulateChatResponse)
def simulate_chat(body: SimulateChatRequest) -> SimulateChatResponse:
    return _simulate_chat_internal(
        user_input=body.user_input,
        session_id=body.session_id,
        end_user_id=body.end_user_id,
        query=body.query,
        metadata=body.metadata,
    )


@app.post("/simulate/chat_flow", response_model=ChatFlowResponse)
def simulate_chat_flow(body: SimulateChatRequest) -> ChatFlowResponse:
    return _run_chat_flow_internal(
        user_input=body.user_input,
        session_id=body.session_id,
        end_user_id=body.end_user_id,
        query=body.query,
        metadata=body.metadata,
    )


@app.post("/simulate/live/start", response_model=LiveSimulationStatus)
def start_live_simulation(body: LiveSimulationStartRequest) -> LiveSimulationStatus:
    global _live_thread

    with _live_lock:
        if _live_state["running"]:
            raise HTTPException(status_code=409, detail="Live simulation is already running.")

        _live_state["running"] = True
        _live_state["interval_seconds"] = body.interval_seconds
        _live_state["call_count"] = 0
        _live_state["session_id"] = body.session_id or settings.default_session_id
        _live_state["end_user_id"] = body.end_user_id or settings.end_user_id
        _live_state["last_question"] = ""
        _live_state["last_error"] = ""
        _live_state["last_result_status"] = "starting"

    _live_stop_event.clear()
    _live_thread = threading.Thread(
        target=_live_worker,
        kwargs={
            "interval_seconds": body.interval_seconds,
            "session_id": body.session_id,
            "end_user_id": body.end_user_id,
            "query": body.query,
            "questions": body.questions,
        },
        daemon=True,
    )
    _live_thread.start()

    with _live_lock:
        return LiveSimulationStatus(**_live_state)


@app.post("/simulate/live/stop", response_model=LiveSimulationStatus)
def stop_live_simulation() -> LiveSimulationStatus:
    _live_stop_event.set()
    if _live_thread and _live_thread.is_alive():
        _live_thread.join(timeout=2)

    with _live_lock:
        _live_state["running"] = False
        if _live_state["last_result_status"] == "starting":
            _live_state["last_result_status"] = "stopped"
        return LiveSimulationStatus(**_live_state)


@app.get("/simulate/live/status", response_model=LiveSimulationStatus)
def live_simulation_status() -> LiveSimulationStatus:
    with _live_lock:
        return LiveSimulationStatus(**_live_state)


@app.get("/chat", response_class=HTMLResponse)
def chat_ui() -> HTMLResponse:
    return HTMLResponse(SIMULATOR_UI_PATH.read_text(encoding="utf-8"))
