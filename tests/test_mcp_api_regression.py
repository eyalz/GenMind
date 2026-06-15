#!/usr/bin/env python3
"""API-level MCP regression checks for write/read/overwrite/delete behavior."""

from __future__ import annotations

import time
from pathlib import Path
import sys
from uuid import uuid4

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auth import create_jwt
from src.main import app


def _headers(customer_id: str, workspace_id: str, end_user_id: str, session_id: str) -> dict[str, str]:
    token = create_jwt(
        customer_id=customer_id,
        workspace_id=workspace_id,
        end_user_id=end_user_id,
        session_id=session_id,
        scopes=["memory:read", "memory:write"],
    )
    return {"Authorization": f"Bearer {token}"}


def test_mcp_roundtrip_regression() -> None:
    customer_id = "cust_reg"
    workspace_id = "ws_reg"
    end_user_id = f"user_{uuid4().hex[:8]}"
    session_id = f"sess_{uuid4().hex[:10]}"
    maker_id = "maker_default"
    agent_id = "agent_default"

    headers = _headers(customer_id, workspace_id, end_user_id, session_id)

    with TestClient(app) as client:
        # 1) update_memory_state accepted and schedules AUDN pipeline processing.
        accepted = client.post(
            "/mcp/tools/update_memory_state",
            headers=headers,
            json={
                "customer_id": customer_id,
                "workspace_id": workspace_id,
                "end_user_id": end_user_id,
                "session_id": session_id,
                "maker_id": maker_id,
                "agent_id": agent_id,
                "user_input": "profile.current_location=Berlin|scope=temporary",
                "model_output": "Acknowledged.",
            },
        )
        assert accepted.status_code == 200
        assert accepted.json().get("status") == "accepted"

        # 2) Write + immediate retrieval in one call.
        first = client.post(
            "/mcp/tools/send_and_receive",
            headers=headers,
            json={
                "customer_id": customer_id,
                "workspace_id": workspace_id,
                "end_user_id": end_user_id,
                "session_id": session_id,
                "maker_id": maker_id,
                "agent_id": agent_id,
                "user_input": "profile.current_location=Berlin|scope=temporary",
                "model_output": "Stored.",
                "query": "where am i now",
                "max_tokens": 700,
            },
        )
        assert first.status_code == 200
        body = first.json()
        assert body.get("decision_count", 0) >= 1
        assert "current_location" in body.get("contents", "")
        assert "Berlin" in body.get("contents", "")

        # 3) Overwrite lifecycle check.
        overwrite = client.post(
            "/mcp/tools/send_and_receive",
            headers=headers,
            json={
                "customer_id": customer_id,
                "workspace_id": workspace_id,
                "end_user_id": end_user_id,
                "session_id": session_id,
                "maker_id": maker_id,
                "agent_id": agent_id,
                "user_input": "profile.current_location=Paris|scope=temporary",
                "model_output": "Updated.",
                "query": "where am i now",
                "max_tokens": 700,
            },
        )
        assert overwrite.status_code == 200
        overwrite_body = overwrite.json()
        assert "Paris" in overwrite_body.get("contents", "")

        # 4) Resource read should surface reconciled state.
        resource = client.post(
            "/mcp/resources",
            headers=headers,
            json={
                "uri": f"genmind://sessions/{session_id}/context",
                "tenant": {
                    "customer_id": customer_id,
                    "workspace_id": workspace_id,
                    "end_user_id": end_user_id,
                    "session_id": session_id,
                },
                "maker_id": maker_id,
                "agent_id": agent_id,
                "query": "where am i now",
                "max_tokens": 700,
            },
        )
        assert resource.status_code == 200
        assert "Paris" in resource.json().get("contents", "")

        # 5) Delete lifecycle via explicit correction cue.
        delete = client.post(
            "/mcp/tools/send_and_receive",
            headers=headers,
            json={
                "customer_id": customer_id,
                "workspace_id": workspace_id,
                "end_user_id": end_user_id,
                "session_id": session_id,
                "maker_id": maker_id,
                "agent_id": agent_id,
                "user_input": "forget that profile.current_location=Unknown|scope=temporary",
                "model_output": "Deleted.",
                "query": "where am i now",
                "max_tokens": 700,
            },
        )
        assert delete.status_code == 200

        # Background/sync state should settle immediately for send_and_receive, but keep a tiny guard.
        time.sleep(0.1)

        post_delete = client.post(
            "/mcp/resources",
            headers=headers,
            json={
                "uri": f"genmind://sessions/{session_id}/context",
                "tenant": {
                    "customer_id": customer_id,
                    "workspace_id": workspace_id,
                    "end_user_id": end_user_id,
                    "session_id": session_id,
                },
                "maker_id": maker_id,
                "agent_id": agent_id,
                "query": "where am i now",
                "max_tokens": 700,
            },
        )
        assert post_delete.status_code == 200
        assert "Paris" not in post_delete.json().get("contents", "")


if __name__ == "__main__":
    test_mcp_roundtrip_regression()
    print("MCP API regression tests passed.")
