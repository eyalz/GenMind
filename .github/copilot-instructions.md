# Copilot Instructions for Project GenMind

These directives are mandatory for all future Copilot-generated changes in this repository.

## 1) MCP Contract Preservation

- Preserve endpoints and behavior for:
	- `POST /mcp/initialization`
	- `GET /mcp/resources`
	- `POST /mcp/resources`
	- `GET /mcp/tools`
	- `POST /mcp/tools/update_memory_state`
	- `GET /mcp/events` (SSE)
- Keep resource URI semantics stable:
	- `genmind://sessions/{session_id}/context`
- Do not rename or remove tool `update_memory_state`.

## 2) Multi-Tenant Isolation (Non-Negotiable)

- The canonical tenant boundary tuple is now **four IDs**: `customer_id`, `workspace_id`, `end_user_id`, `session_id`.
- Never execute storage operations without all four identifiers verified.
- Reject requests with mismatched session identifiers between URI and payload.
- Background jobs must carry tenant context explicitly (no implicit global tenant state).
- Row-Level Security policies must be set at the DB session level from verified JWT claims before any query executes.
- Memory scoping rules:
  - End-user memory: `customer_id + workspace_id + end_user_id`
  - Session memory: full four-ID tuple
  - Shared workspace memory: only when explicitly scope-tagged

## 3) AUDN Ingestion Integrity

- Maintain asynchronous AUDN loop semantics:
	- Add
	- Update
	- Delete
	- None
- Never degrade to append-only memory writes.
- Keep explicit stale-memory invalidation paths for Delete actions.
- Preserve or improve reasoned action classification and auditability.

## 4) Hybrid Retrieval Rules

- Keep hybrid retrieval composed of:
	- vector similarity search
	- explicit entity graph resolution
	- recency time-decay scoring
- Ensure fresh facts can supersede stale facts through scoring logic.
- Preserve markdown truth payload compaction and token guardrail (`<= 1000` tokens target).

## 5) Type Safety and Validation

- Use strict Pydantic v2 models (`extra="forbid"`, strict types where practical).
- Validate all request payloads for MCP tools/resources.
- Avoid untyped dict plumbing when a typed schema exists.

## 6) Decoupled State Patterns

- Keep orchestration layers decoupled:
	- API routes -> pipeline orchestrators -> storage/retrieval services
- Prefer dependency injection or clearly scoped singletons over hidden global mutation.
- SSE transport should publish immutable event payloads and remain independent from business logic internals.

## 7) Engineering Quality Expectations

- Default to asynchronous Python patterns for IO-facing flows.
- Favor modular service files over large monolithic route handlers.
- Document scoring assumptions and security boundaries in code/doc updates.
- Any schema or contract changes must include matching updates in:
	- `docs/api/mcp-schema.json`
	- `docs/architecture/spec.md`

## 8) Authentication and Identity

- All MCP and admin endpoints must require a valid short-lived JWT (RS256, max 1h).
- JWT must carry `customer_id`, `workspace_id`, and `scopes` as verified claims.
- Auth middleware must bind claims to request tenant context before any handler logic runs.
- API credentials (raw keys) must never be stored. Only SHA-256 hash and 8-char prefix.
- Credential issuance and rotation must produce an admin_audit_log entry.
- Rate limiting must apply per workspace credential, not globally.

## 9) Usage Tracking

- Every business-critical request (MCP read, MCP write, admin call) must emit a `usage_event` record.
- Usage events are append-only. Never update or delete usage_events rows.
- Usage events must include the full tenant tuple plus tokens_in, tokens_out, context_tokens, vector_reads, graph_reads, memory_writes, latency_ms, and status_code.
- Daily aggregation jobs must roll up raw events into `usage_daily_aggregates`.
- Consumption reports must never expose individual end-user identifiers in aggregate views.

## 10) Privacy and Data Rights

- PII classifier must run on all ingestion fragments before embedding or persistence.
- Detected PII must be redacted or pseudonymized before entering `memory_records`.
- End-user deletion requests (GDPR Article 17) must purge all rows by `end_user_id` across all tables within SLA.
- Customer offboarding must purge all `customer_id`-scoped data within 30 days.
- Deletion events must produce admin_audit_log entries for compliance evidence.
- `memory:raw_read` is a restricted scope required for any access to non-redacted memory content.
- Data must not leave the customer's assigned region. Cross-region replication is disabled by default.

## 11) Dashboard and Observability Contracts

- Operational metrics must be emitted via OpenTelemetry and scraped by Prometheus.
- Business usage reports must be derivable from `usage_daily_aggregates` without touching raw `usage_events`.
- Per-customer self-service usage reads must be scoped to that customer's own credentials only.
- No cross-customer data may appear in any dashboard or API response.
