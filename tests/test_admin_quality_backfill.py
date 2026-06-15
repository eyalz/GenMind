#!/usr/bin/env python3
"""Admin API regressions for retrieval quality views and claim backfill jobs."""

from __future__ import annotations

from pathlib import Path
import sys
from uuid import uuid4

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auth import create_jwt
from src.main import app


def _admin_headers(customer_id: str, workspace_id: str) -> dict[str, str]:
    token = create_jwt(
        customer_id=customer_id,
        workspace_id=workspace_id,
        end_user_id="system",
        session_id="system",
        scopes=["admin:*", "memory:read", "memory:write"],
    )
    return {"Authorization": f"Bearer {token}"}


def _mcp_headers(customer_id: str, workspace_id: str, end_user_id: str, session_id: str) -> dict[str, str]:
    token = create_jwt(
        customer_id=customer_id,
        workspace_id=workspace_id,
        end_user_id=end_user_id,
        session_id=session_id,
        scopes=["memory:read", "memory:write"],
    )
    return {"Authorization": f"Bearer {token}"}


def test_admin_quality_and_backfill() -> None:
    customer_id = "cust_admin"
    workspace_id = "ws_admin"
    end_user_id = f"user_{uuid4().hex[:8]}"
    session_id = f"sess_{uuid4().hex[:10]}"
    maker_id = "maker_default"
    agent_id = "agent_default"

    headers = _admin_headers(customer_id, workspace_id)
    mcp_headers = _mcp_headers(customer_id, workspace_id, end_user_id, session_id)

    with TestClient(app) as client:
        # Seed telemetry + memory state through the real MCP surface.
        seeded = client.post(
            "/mcp/tools/send_and_receive",
            headers=mcp_headers,
            json={
                "customer_id": customer_id,
                "workspace_id": workspace_id,
                "end_user_id": end_user_id,
                "session_id": session_id,
                "maker_id": maker_id,
                "agent_id": agent_id,
                "user_input": "profile.current_location=Lisbon|scope=temporary",
                "model_output": "Stored",
                "query": "where am i now",
                "max_tokens": 700,
            },
        )
        assert seeded.status_code == 200

        quality = client.get(
            f"/admin/usage/{customer_id}/retrieval-quality",
            headers=headers,
            params={"workspace_id": workspace_id, "window_seconds": 3600},
        )
        assert quality.status_code == 200
        rows = quality.json()
        assert rows
        assert rows[0]["requests_total"] >= 1

        alerts = client.get(
            f"/admin/usage/{customer_id}/retrieval-alerts",
            headers=headers,
            params={
                "workspace_id": workspace_id,
                "window_seconds": 3600,
                "max_empty_context_ratio": 0.0,
            },
        )
        assert alerts.status_code == 200
        assert isinstance(alerts.json(), list)

        dry_run = client.post(
            "/admin/memory/claims/backfill",
            headers=headers,
            params={
                "customer_id": customer_id,
                "workspace_id": workspace_id,
                "end_user_id": end_user_id,
                "session_id": session_id,
                "maker_id": maker_id,
                "agent_id": agent_id,
                "dry_run": True,
            },
        )
        assert dry_run.status_code == 200
        assert dry_run.json()["projected_claims"] >= 1
        assert dry_run.json()["checkpoint_updated"] is False

        execute = client.post(
            "/admin/memory/claims/backfill",
            headers=headers,
            params={
                "customer_id": customer_id,
                "workspace_id": workspace_id,
                "end_user_id": end_user_id,
                "session_id": session_id,
                "maker_id": maker_id,
                "agent_id": agent_id,
                "dry_run": False,
            },
        )
        assert execute.status_code == 200
        assert execute.json()["projected_claims"] >= 1
        assert execute.json()["checkpoint_updated"] is True


if __name__ == "__main__":
    test_admin_quality_and_backfill()
    print("Admin quality and backfill tests passed.")
