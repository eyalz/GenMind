from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr


class AUDNAction(str, Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NONE = "none"


class MutationType(str, Enum):
    ENTITY_CREATION = "entity_creation"
    PROPERTY_MODIFICATION = "property_modification"
    RELATIONSHIP_EDGE_CHANGE = "relationship_edge_change"


class TemporalType(str, Enum):
    PERMANENT = "permanent"
    TEMPORARY = "temporary"
    HISTORICAL_ARCHIVE = "historical_archive"


class TemporalScope(BaseModel):
    """Validity window and lifecycle type for one mutation item."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: TemporalType = TemporalType.PERMANENT
    valid_from: StrictStr = Field(default="current_interaction", min_length=2, max_length=64)
    valid_until: StrictStr = Field(default="indefinite", min_length=2, max_length=128)


class UOMEMutation(BaseModel):
    """Strict JSON mutation shape for external UOME consumers."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: StrictStr = Field(min_length=3, max_length=16)
    mutation_type: MutationType
    source_entity: StrictStr = Field(min_length=2, max_length=256)
    target_property_or_entity: StrictStr = Field(min_length=2, max_length=256)
    value: Any
    previous_value_reference: Any | None = None
    temporal_scope: TemporalScope
    reasoning_justification: StrictStr = Field(min_length=3, max_length=500)


class CustomerStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    OFFBOARDED = "offboarded"


class CustomerPlan(str, Enum):
    STARTER = "starter"
    GROWTH = "growth"
    ENTERPRISE = "enterprise"


class WorkspaceEnvironment(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class WorkspaceStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"


class CredentialScope(str, Enum):
    MEMORY_READ = "memory:read"
    MEMORY_WRITE = "memory:write"
    MEMORY_RAW_READ = "memory:raw_read"
    ADMIN_ALL = "admin:*"


class DeleteReason(str, Enum):
    GDPR_ARTICLE_17 = "gdpr_article_17"
    USER_REQUEST = "user_request"
    ACCOUNT_CLOSURE = "account_closure"
    OPERATOR_INITIATED = "operator_initiated"


class PIISensitivity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TenantContext(BaseModel):
    """
    Canonical four-ID tenant boundary context used on every request and storage operation.

    Fields:
    - customer_id: GenMind B2B customer account.
    - workspace_id: Integration instance (dev/staging/prod) under that customer.
    - end_user_id: The end user inside that workspace, for privacy-correct per-user scoping.
    - session_id: One conversation or interaction unit.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    customer_id: StrictStr = Field(min_length=2, max_length=128)
    workspace_id: StrictStr = Field(min_length=2, max_length=128)
    end_user_id: StrictStr = Field(min_length=2, max_length=256)
    session_id: StrictStr = Field(min_length=2, max_length=256)


class EntityNode(BaseModel):
    """Entity representation for the explicit knowledge graph tier."""

    model_config = ConfigDict(extra="forbid", strict=True)

    tenant: TenantContext
    entity_id: StrictStr = Field(min_length=2, max_length=256)
    entity_type: StrictStr = Field(min_length=2, max_length=128)
    canonical_name: StrictStr = Field(min_length=1, max_length=256)
    aliases: list[StrictStr] = Field(default_factory=list)
    attributes: dict[StrictStr, Any] = Field(default_factory=dict)
    confidence: StrictFloat = Field(ge=0.0, le=1.0, default=0.8)
    last_seen_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryRecord(BaseModel):
    """Stored memory fact row in a normalized retrieval-friendly shape."""

    model_config = ConfigDict(extra="forbid", strict=True)

    memory_id: StrictStr = Field(min_length=2, max_length=256)
    tenant: TenantContext
    maker_id: StrictStr = Field(default="maker_default", min_length=2, max_length=128)
    agent_id: StrictStr = Field(default="agent_default", min_length=2, max_length=128)
    content: StrictStr = Field(min_length=1, max_length=4000)
    source: StrictStr = Field(min_length=2, max_length=128, default="session")
    confidence: StrictFloat = Field(ge=0.0, le=1.0, default=0.8)
    recency_boost: StrictFloat = Field(ge=0.0, le=1.0, default=1.0)
    embedding: list[StrictFloat] = Field(default_factory=list)
    entity_ids: list[StrictStr] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True


class SessionTurnPayload(BaseModel):
    """Raw conversational fragment entering the AUDN ingestion loop."""

    model_config = ConfigDict(extra="forbid", strict=True)

    customer_id: StrictStr = Field(min_length=2, max_length=128)
    workspace_id: StrictStr = Field(min_length=2, max_length=128)
    end_user_id: StrictStr = Field(min_length=2, max_length=256)
    session_id: StrictStr = Field(min_length=2, max_length=256)
    maker_id: StrictStr = Field(default="maker_default", min_length=2, max_length=128)
    agent_id: StrictStr = Field(default="agent_default", min_length=2, max_length=128)
    user_input: StrictStr = Field(min_length=1)
    model_output: StrictStr = Field(min_length=1)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)


class RetrievalRequest(BaseModel):
    """Hybrid retrieval input used by MCP resources endpoint."""

    model_config = ConfigDict(extra="forbid", strict=True)

    tenant: TenantContext
    maker_id: StrictStr = Field(default="maker_default", min_length=2, max_length=128)
    agent_id: StrictStr = Field(default="agent_default", min_length=2, max_length=128)
    query: StrictStr = Field(min_length=1, max_length=2000)
    max_items: StrictInt = Field(default=24, ge=1, le=200)
    max_tokens: StrictInt = Field(default=1000, ge=100, le=2000)


class RetrievalCandidate(BaseModel):
    """Candidate result from vector, graph, or direct fact retrieval."""

    model_config = ConfigDict(extra="forbid", strict=True)

    memory_id: StrictStr = Field(min_length=2, max_length=256)
    content: StrictStr = Field(min_length=1, max_length=4000)
    semantic_score: StrictFloat = Field(ge=0.0, le=1.0)
    graph_score: StrictFloat = Field(ge=0.0, le=1.0)
    recency_score: StrictFloat = Field(ge=0.0, le=1.0)
    final_score: StrictFloat = Field(ge=0.0, le=1.0)
    updated_at: datetime


class RetrievalResult(BaseModel):
    """Final condensed markdown truth payload for downstream LLM prompts."""

    model_config = ConfigDict(extra="forbid", strict=True)

    tenant: TenantContext
    payload_markdown: StrictStr
    consumed_tokens_estimate: StrictInt = Field(ge=0)
    selected_items: list[RetrievalCandidate] = Field(default_factory=list)


class AUDNDecision(BaseModel):
    """Decision unit for a single candidate fact in the AUDN loop."""

    model_config = ConfigDict(extra="forbid", strict=True)

    tenant: TenantContext
    maker_id: StrictStr = Field(default="maker_default", min_length=2, max_length=128)
    agent_id: StrictStr = Field(default="agent_default", min_length=2, max_length=128)
    action: AUDNAction
    mutation_type: MutationType = MutationType.PROPERTY_MODIFICATION
    source_entity: StrictStr = Field(default="User_Profile", min_length=2, max_length=256)
    target_property_or_entity: StrictStr = Field(default="unknown", min_length=2, max_length=256)
    value: Any = ""
    previous_value_reference: Any | None = None
    temporal_scope: TemporalScope = Field(default_factory=TemporalScope)
    reasoning_justification: StrictStr = Field(default="", min_length=0, max_length=500)
    reason: StrictStr = Field(min_length=3, max_length=500)
    candidate_fact: StrictStr = Field(min_length=1, max_length=4000)
    confidence: StrictFloat = Field(ge=0.0, le=1.0)
    target_memory_id: StrictStr | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MCPInitializationRequest(BaseModel):
    """MCP initialization handshake request payload."""

    model_config = ConfigDict(extra="forbid", strict=True)

    protocol_version: StrictStr = Field(default="2026-01-01")
    client_name: StrictStr = Field(min_length=2, max_length=256)
    client_version: StrictStr = Field(min_length=1, max_length=64)


class MCPInitializationResponse(BaseModel):
    """MCP handshake response defining server capabilities."""

    model_config = ConfigDict(extra="forbid", strict=True)

    protocol_version: StrictStr
    server_name: StrictStr
    server_version: StrictStr
    transport: StrictStr
    stream_endpoint: StrictStr
    resources_endpoint: StrictStr
    tools_endpoint: StrictStr


class MCPStreamRequest(BaseModel):
    """JSON-RPC compatible request envelope for unified streamable MCP endpoint."""

    model_config = ConfigDict(extra="forbid", strict=True)

    jsonrpc: StrictStr = Field(default="2.0")
    id: StrictStr | StrictInt | None = None
    method: StrictStr = Field(min_length=1, max_length=128)
    params: dict[StrictStr, Any] = Field(default_factory=dict)


class MCPStreamResponse(BaseModel):
    """JSON-RPC compatible response envelope for unified streamable MCP endpoint."""

    model_config = ConfigDict(extra="forbid", strict=True)

    jsonrpc: StrictStr = Field(default="2.0")
    id: StrictStr | StrictInt | None = None
    result: dict[StrictStr, Any] | None = None
    error: dict[StrictStr, Any] | None = None


class MCPResourceReadRequest(BaseModel):
    """Resource read body for genmind session context URI requests."""

    model_config = ConfigDict(extra="forbid", strict=True)

    uri: StrictStr = Field(min_length=1)
    tenant: TenantContext
    maker_id: StrictStr = Field(default="maker_default", min_length=2, max_length=128)
    agent_id: StrictStr = Field(default="agent_default", min_length=2, max_length=128)
    query: StrictStr = Field(default="latest context", min_length=1, max_length=2000)
    max_tokens: StrictInt = Field(default=1000, ge=100, le=2000)


class UpdateMemoryToolRequest(BaseModel):
    """Tool invocation payload for update_memory_state."""

    model_config = ConfigDict(extra="forbid", strict=True)

    customer_id: StrictStr = Field(min_length=2, max_length=128)
    workspace_id: StrictStr = Field(min_length=2, max_length=128)
    end_user_id: StrictStr = Field(min_length=2, max_length=256)
    session_id: StrictStr = Field(min_length=2, max_length=256)
    maker_id: StrictStr = Field(default="maker_default", min_length=2, max_length=128)
    agent_id: StrictStr = Field(default="agent_default", min_length=2, max_length=128)
    user_input: StrictStr = Field(min_length=1)
    model_output: StrictStr = Field(min_length=1)
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)


class SendAndReceiveToolRequest(BaseModel):
    """One-shot MCP payload that commits AUDN decisions and returns fresh context."""

    model_config = ConfigDict(extra="forbid", strict=True)

    customer_id: StrictStr = Field(min_length=2, max_length=128)
    workspace_id: StrictStr = Field(min_length=2, max_length=128)
    end_user_id: StrictStr = Field(min_length=2, max_length=256)
    session_id: StrictStr = Field(min_length=2, max_length=256)
    maker_id: StrictStr = Field(default="maker_default", min_length=2, max_length=128)
    agent_id: StrictStr = Field(default="agent_default", min_length=2, max_length=128)
    user_input: StrictStr = Field(min_length=1)
    model_output: StrictStr = Field(min_length=1)
    query: StrictStr = Field(default="latest context", min_length=1, max_length=2000)
    max_tokens: StrictInt = Field(default=1000, ge=100, le=2000)
    metadata: dict[StrictStr, Any] = Field(default_factory=dict)


class SendAndReceiveToolResponse(BaseModel):
    """Synchronous one-shot MCP response containing updated context and AUDN decisions."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: StrictStr = Field(default="completed")
    tool: StrictStr = Field(default="send_and_receive")
    tenant: TenantContext
    maker_id: StrictStr = Field(min_length=2, max_length=128)
    agent_id: StrictStr = Field(min_length=2, max_length=128)
    uri: StrictStr = Field(min_length=1)
    contents: StrictStr
    token_estimate: StrictInt = Field(ge=0)
    selected_items: list[RetrievalCandidate] = Field(default_factory=list)
    decision_count: StrictInt = Field(ge=0)
    decisions: list[AUDNDecision] = Field(default_factory=list)
    uome_mutations: list[UOMEMutation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Customer and Workspace Management
# ---------------------------------------------------------------------------


class CreateCustomerRequest(BaseModel):
    """Admin request to create a new B2B customer account."""

    model_config = ConfigDict(extra="forbid", strict=True)

    display_name: StrictStr = Field(min_length=2, max_length=256)
    plan: CustomerPlan
    region: StrictStr = Field(min_length=4, max_length=64)
    retention_days: StrictInt = Field(default=90, ge=7, le=730)
    is_demo: bool = False


class UpdateCustomerRequest(BaseModel):
    """Admin request to update mutable customer fields."""

    model_config = ConfigDict(extra="forbid", strict=True)

    display_name: StrictStr | None = Field(default=None, min_length=2, max_length=256)
    status: CustomerStatus | None = None
    plan: CustomerPlan | None = None
    region: StrictStr | None = Field(default=None, min_length=4, max_length=64)
    retention_days: StrictInt | None = Field(default=None, ge=7, le=730)
    is_demo: bool | None = None


class Customer(BaseModel):
    """Persisted customer account record."""

    model_config = ConfigDict(extra="forbid", strict=True)

    customer_id: StrictStr = Field(min_length=2, max_length=128)
    display_name: StrictStr = Field(min_length=2, max_length=256)
    status: CustomerStatus = CustomerStatus.ACTIVE
    plan: CustomerPlan
    region: StrictStr = Field(min_length=4, max_length=64)
    retention_days: StrictInt = Field(ge=7, le=730)
    is_demo: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CreateWorkspaceRequest(BaseModel):
    """Admin request to provision a workspace under a customer."""

    model_config = ConfigDict(extra="forbid", strict=True)

    display_name: StrictStr = Field(min_length=2, max_length=256)
    environment: WorkspaceEnvironment
    monthly_request_quota: StrictInt | None = Field(default=None, ge=1)


class Workspace(BaseModel):
    """Persisted workspace record."""

    model_config = ConfigDict(extra="forbid", strict=True)

    workspace_id: StrictStr = Field(min_length=2, max_length=128)
    customer_id: StrictStr = Field(min_length=2, max_length=128)
    display_name: StrictStr = Field(min_length=2, max_length=256)
    environment: WorkspaceEnvironment
    status: WorkspaceStatus = WorkspaceStatus.ACTIVE
    monthly_request_quota: StrictInt | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# API Credentials
# ---------------------------------------------------------------------------


class IssueCredentialRequest(BaseModel):
    """Admin request to issue a new workspace API credential."""

    model_config = ConfigDict(extra="forbid", strict=True)

    scopes: list[CredentialScope] = Field(min_length=1)
    expires_at: datetime | None = None


class CredentialIssuedResponse(BaseModel):
    """
    API credential issuance response.

    The raw_key is returned exactly once and must be stored securely by the caller.
    GenMind stores only the SHA-256 hash; the raw key is unrecoverable after this response.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    credential_id: StrictStr
    raw_key: StrictStr = Field(description="Shown once only. Store securely.")
    key_prefix: StrictStr = Field(description="First 8 chars of raw key for identification.")
    scopes: list[CredentialScope]
    expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class APICredential(BaseModel):
    """Stored credential record (raw key never included)."""

    model_config = ConfigDict(extra="forbid", strict=True)

    credential_id: StrictStr
    workspace_id: StrictStr
    customer_id: StrictStr
    key_hash: StrictStr = Field(description="SHA-256 hash of raw key.")
    key_prefix: StrictStr
    scopes: list[CredentialScope]
    last_used_at: datetime | None = None
    rotated_at: datetime | None = None
    expires_at: datetime | None = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Usage Tracking
# ---------------------------------------------------------------------------


class UsageEvent(BaseModel):
    """
    Immutable append-only usage event emitted on every business-critical request.

    These rows drive billing, consumption reports, and anomaly detection.
    Rows must never be updated or deleted; retention is controlled by plan policy.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    event_id: StrictStr
    customer_id: StrictStr = Field(min_length=2, max_length=128)
    workspace_id: StrictStr = Field(min_length=2, max_length=128)
    end_user_id: StrictStr = Field(min_length=2, max_length=256)
    session_id: StrictStr = Field(min_length=2, max_length=256)
    request_id: StrictStr
    endpoint: StrictStr
    tokens_in: StrictInt = Field(ge=0)
    tokens_out: StrictInt = Field(ge=0)
    context_tokens: StrictInt = Field(ge=0)
    vector_reads: StrictInt = Field(ge=0)
    graph_reads: StrictInt = Field(ge=0)
    memory_writes: StrictInt = Field(ge=0)
    latency_ms: StrictInt = Field(ge=0)
    status_code: StrictInt = Field(ge=100, le=599)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UsageDailyAggregate(BaseModel):
    """Daily materialized rollup for billing and consumption reports."""

    model_config = ConfigDict(extra="forbid", strict=True)

    customer_id: StrictStr = Field(min_length=2, max_length=128)
    workspace_id: StrictStr = Field(min_length=2, max_length=128)
    date: StrictStr = Field(description="ISO date string, e.g. 2026-06-06")
    total_requests: StrictInt = Field(ge=0)
    total_tokens_in: StrictInt = Field(ge=0)
    total_tokens_out: StrictInt = Field(ge=0)
    total_context_tokens: StrictInt = Field(ge=0)
    total_vector_reads: StrictInt = Field(ge=0)
    total_memory_writes: StrictInt = Field(ge=0)
    active_end_users: StrictInt = Field(ge=0)
    avg_latency_ms: StrictFloat = Field(ge=0.0)


class RecentCustomerActivity(BaseModel):
    """Recent activity rollup per customer for short live windows."""

    model_config = ConfigDict(extra="forbid", strict=True)

    customer_id: StrictStr = Field(min_length=2, max_length=128)
    inbound_calls: StrictInt = Field(ge=0)
    outbound_calls: StrictInt = Field(ge=0)
    last_seen_at: datetime


class DatabaseSummaryResponse(BaseModel):
    """Compact operational snapshot of core database-backed system tables."""

    model_config = ConfigDict(extra="forbid", strict=True)

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    recent_window_seconds: StrictInt = Field(ge=1)
    customers_total: StrictInt = Field(ge=0)
    workspaces_total: StrictInt = Field(ge=0)
    credentials_total: StrictInt = Field(ge=0)
    active_credentials_total: StrictInt = Field(ge=0)
    memory_records_total: StrictInt = Field(ge=0)
    active_memory_records_total: StrictInt = Field(ge=0)
    usage_events_total: StrictInt = Field(ge=0)
    audn_decisions_total: StrictInt = Field(ge=0)
    admin_audit_entries_total: StrictInt = Field(ge=0)
    active_customers_last_window: StrictInt = Field(ge=0)
    active_workspaces_last_window: StrictInt = Field(ge=0)
    active_end_users_last_window: StrictInt = Field(ge=0)
    mcp_requests_last_window: StrictInt = Field(ge=0)
    admin_requests_last_window: StrictInt = Field(ge=0)
    last_memory_write_at: datetime | None = None
    last_usage_event_at: datetime | None = None


# ---------------------------------------------------------------------------
# Privacy and Data Rights
# ---------------------------------------------------------------------------


class PIIClassification(BaseModel):
    """Result of a PII classifier pass over an ingestion fragment."""

    model_config = ConfigDict(extra="forbid", strict=True)

    fragment_id: StrictStr
    sensitivity: PIISensitivity
    detected_entity_types: list[StrictStr] = Field(default_factory=list)
    redacted_content: StrictStr = Field(
        description="PII-scrubbed version safe for embedding and storage."
    )
    has_redactions: bool = False


class EndUserDeleteRequest(BaseModel):
    """
    GDPR Article 17 / right-to-delete request.

    Triggers an async purge of all rows associated with end_user_id
    across memory_records, entity_nodes, entity_edges, and usage_events.
    A deletion event is written to admin_audit_log for compliance evidence.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    customer_id: StrictStr = Field(min_length=2, max_length=128)
    workspace_id: StrictStr = Field(min_length=2, max_length=128)
    end_user_id: StrictStr = Field(min_length=2, max_length=256)
    reason: DeleteReason
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AdminAuditEntry(BaseModel):
    """Immutable admin audit log entry. Must never be updated or deleted."""

    model_config = ConfigDict(extra="forbid", strict=True)

    operator_id: StrictStr
    operator_role: StrictStr
    action: StrictStr = Field(min_length=2, max_length=128)
    target_resource: StrictStr = Field(min_length=2, max_length=128)
    target_id: StrictStr
    ip_address: StrictStr
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
