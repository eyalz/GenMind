# Local Agent Simulator

This service simulates an external customer agent that talks to GenMind on behalf of a specific end user.

It now also includes a local browser UI so you can watch the complete end-to-end flow:
- customer asks a question
- customer agent sends MCP calls into GenMind
- GenMind updates memory and returns modified context
- customer agent enriches with simple web knowledge
- final customer answer is generated and shown with a flow diagram

## Purpose

Use this when you want to test end-to-end behavior before onboarding a real customer platform:
- agent writes memory via MCP tool
- agent reads context via MCP resource
- all calls are scoped to one customer/workspace/user/session

## Run

1. Start GenMind API server on port 8000.
2. Start simulator server:

```bash
python3 -m uvicorn src.simulator.agent_simulator:app --host 127.0.0.1 --port 8100
```

3. Open the simulator UI:

```text
http://127.0.0.1:8100/chat
```

## Optional Environment Variables

- `GENMIND_API_BASE` (default: `http://127.0.0.1:8000`)
- `SIM_CUSTOMER_NAME` (default: `Copilot Studio Test 1`)
- `SIM_CUSTOMER_ID` (optional explicit override)
- `SIM_WORKSPACE_ID` (optional explicit override)
- `SIM_BOOTSTRAP_CUSTOMER_ID` (default: `cust_dev`)
- `SIM_BOOTSTRAP_WORKSPACE_ID` (default: `ws_dev`)
- `SIM_MAKER_ID` (default: `maker_default`)
- `SIM_AGENT_ID` (default: `Test_Agent`)
- `SIM_END_USER_ID` (default: `Test_User`)
- `SIM_DEFAULT_SESSION_ID` (default: `session_simulated_1`)
- `SIM_TOKEN_EXPIRES_MINUTES` (default: `60`)
- `SIM_ACCESS_TOKEN` (optional fixed bearer token; skips dev bootstrap)
- `SIM_DEV_BOOTSTRAP_HEADER` (default: `allow`)

By default the simulator resolves customer and workspace IDs from `SIM_CUSTOMER_NAME` and picks the first workspace for that customer.

## Call Example

```bash
curl -X POST 'http://127.0.0.1:8100/simulate/chat' \
  -H 'Content-Type: application/json' \
  -d '{
    "user_input": "I prefer concise summaries and Monday reminders",
    "session_id": "sess_customer_001",
    "end_user_id": "user_42",
    "query": "what are the user preferences"
  }'
```

Expected behavior:
- simulator calls `POST /mcp/stream` with `initialize`
- simulator calls `tools/call` (`update_memory_state`)
- simulator calls `resources/read`
- response includes tool result + context payload

## Full Flow Chat Demo

Use the richer flow endpoint when you want to inspect the full customer-agent lifecycle.

```bash
curl -X POST 'http://127.0.0.1:8100/simulate/chat_flow' \
  -H 'Content-Type: application/json' \
  -d '{
    "user_input": "What is retrieval augmented generation?",
    "session_id": "sess_ui_demo",
    "end_user_id": "Demo_User",
    "query": "What is retrieval augmented generation?",
    "metadata": {"source": "simulator-ui"}
  }'
```

Response includes:
- `flow`: step-by-step exchange between customer service, GenMind, web lookup, and customer LLM
- `context_excerpt`: the memory context returned by GenMind
- `customer_prompt`: the final prompt assembled by the simulated customer agent
- `final_answer`: the end-user answer shown in the simulator UI
- `web_knowledge`: simple external references used by the customer agent

## Live Mode (Question Every 10 Seconds)

Start a continuous loop that simulates one end user asking questions every 10 seconds.

```bash
curl -X POST 'http://127.0.0.1:8100/simulate/live/start' \
  -H 'Content-Type: application/json' \
  -d '{
    "interval_seconds": 10,
    "session_id": "sess_test_user_live",
    "end_user_id": "Test_User",
    "query": "latest user preferences and context",
    "questions": [
      "What should you remember about me?",
      "Keep answers short.",
      "Remind me every Monday."
    ]
  }'
```

Check status:

```bash
curl 'http://127.0.0.1:8100/simulate/live/status'
```

Stop loop:

```bash
curl -X POST 'http://127.0.0.1:8100/simulate/live/stop'
```
