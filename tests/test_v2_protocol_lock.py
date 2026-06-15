#!/usr/bin/env python3
"""Test V2 protocol version lock: reject legacy clients."""

from __future__ import annotations

from pathlib import Path
import sys
from uuid import uuid4

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auth import create_jwt
from src.main import app


def _stream_headers(customer_id: str, workspace_id: str, end_user_id: str, session_id: str) -> dict[str, str]:
    token = create_jwt(
        customer_id=customer_id,
        workspace_id=workspace_id,
        end_user_id=end_user_id,
        session_id=session_id,
        scopes=["memory:read", "memory:write"],
    )
    return {
        "Authorization": f"Bearer {token}",
        "Origin": "http://localhost:5173",
        "Accept": "application/json, text/event-stream",
    }


def test_v2_protocol_lock_rejects_legacy_version() -> None:
    """V2 should reject protocol_version != '2026-01-01'."""
    customer_id = "cust_lock"
    workspace_id = "ws_lock"
    end_user_id = f"user_{uuid4().hex[:8]}"
    session_id = f"sess_{uuid4().hex[:10]}"

    headers = _stream_headers(customer_id, workspace_id, end_user_id, session_id)

    with TestClient(app) as client:
        # 1) Old protocol version should be rejected.
        legacy = client.post(
            "/mcp/stream",
            headers=headers,
            json={
                "id": "test-1",
                "method": "initialize",
                "params": {
                    "protocol_version": "2024-11-05",
                    "client_name": "copilot-studio",
                    "client_version": "1.0",
                },
            },
        )
        assert legacy.status_code == 200
        body = legacy.json()
        assert "error" in body
        assert "2026-01-01" in body["error"]["message"]
        print("✓ Legacy protocol version (2024-11-05) rejected")

        # 2) V2 protocol version accepted.
        v2 = client.post(
            "/mcp/stream",
            headers=headers,
            json={
                "id": "test-2",
                "method": "initialize",
                "params": {
                    "protocol_version": "2026-01-01",
                    "client_name": "copilot-studio",
                    "client_version": "1.0",
                },
            },
        )
        assert v2.status_code == 200
        body = v2.json()
        assert body.get("error") is None
        assert body.get("result", {}).get("protocol_version") == "2026-01-01"
        print("✓ V2 protocol version (2026-01-01) accepted")

        # 3) Missing protocol_version should be rejected.
        missing = client.post(
            "/mcp/stream",
            headers=headers,
            json={
                "id": "test-3",
                "method": "initialize",
                "params": {
                    "client_name": "copilot-studio",
                    "client_version": "1.0",
                },
            },
        )
        assert missing.status_code == 200
        body = missing.json()
        assert "error" in body
        assert "protocol_version" in body["error"]["message"]
        print("✓ Missing protocol_version rejected")

        # 4) Legacy field name (protocolVersion) should be rejected.
        legacy_field = client.post(
            "/mcp/stream",
            headers=headers,
            json={
                "id": "test-4",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2026-01-01",
                    "client_name": "copilot-studio",
                    "client_version": "1.0",
                },
            },
        )
        assert legacy_field.status_code == 200
        body = legacy_field.json()
        assert "error" in body
        assert "protocol_version" in body["error"]["message"]
        print("✓ Legacy field name (protocolVersion) rejected")


if __name__ == "__main__":
    test_v2_protocol_lock_rejects_legacy_version()
    print("\nV2 Protocol Lock tests passed ✅")
