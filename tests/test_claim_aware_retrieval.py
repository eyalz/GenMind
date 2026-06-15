#!/usr/bin/env python3
"""Regression checks for claim-aware retrieval and claim backfill utilities."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.schemas import ClaimRecord, MemoryRecord, MutationType, RetrievalRequest, TemporalType, TenantContext
from src.services.memory_engine import MemoryEngine


def _tenant() -> TenantContext:
    return TenantContext(
        customer_id="cust_claim",
        workspace_id="ws_claim",
        end_user_id="user_claim",
        session_id="sess_claim",
    )


async def test_claim_projection_search_prefers_matching_claims() -> None:
    engine = MemoryEngine()
    tenant = _tenant()

    async def fake_claims(*args, **kwargs):  # type: ignore[no-untyped-def]
        now = datetime.now(timezone.utc)
        return [
            ClaimRecord(
                claim_id="clm_a",
                memory_id="mem_a",
                tenant=tenant,
                maker_id="maker_default",
                agent_id="agent_default",
                mutation_type=MutationType.PROPERTY_MODIFICATION,
                source_entity="User_Profile",
                target_property_or_entity="current_location",
                value_json="Paris",
                temporal_type=TemporalType.TEMPORARY,
                valid_from="current_interaction",
                valid_until="conditional_trigger",
                confidence=0.95,
                is_active=True,
                created_at=now,
                updated_at=now,
            ),
            ClaimRecord(
                claim_id="clm_b",
                memory_id="mem_b",
                tenant=tenant,
                maker_id="maker_default",
                agent_id="agent_default",
                mutation_type=MutationType.PROPERTY_MODIFICATION,
                source_entity="User_Profile",
                target_property_or_entity="favorite_color",
                value_json="Green",
                temporal_type=TemporalType.PERMANENT,
                valid_from="current_interaction",
                valid_until="indefinite",
                confidence=0.5,
                is_active=True,
                created_at=now,
                updated_at=now,
            ),
        ]

    engine.list_active_claims = fake_claims  # type: ignore[method-assign]

    req = RetrievalRequest(
        tenant=tenant,
        maker_id="maker_default",
        agent_id="agent_default",
        query="where am i now",
        max_items=20,
        max_tokens=800,
    )

    rows = await engine._claim_projection_search(req)
    assert rows, "Expected claim projection retrieval rows"
    assert rows[0].memory_id == "clm_a"
    assert "|tgt=current_location|" in rows[0].content
    assert "|val=Paris|" in rows[0].content


async def test_claim_backfill_projects_profile_memories() -> None:
    engine = MemoryEngine()
    tenant = _tenant()
    now = datetime.now(timezone.utc)

    records = [
        MemoryRecord(
            memory_id="mem_old",
            tenant=tenant,
            maker_id="maker_default",
            agent_id="agent_default",
            content="profile.current_location=Berlin|scope=temporary",
            source="session",
            confidence=0.9,
            recency_boost=1.0,
            embedding=[],
            entity_ids=[],
            created_at=now,
            updated_at=now,
            is_active=True,
        )
    ]

    async def fake_memories(*args, **kwargs):  # type: ignore[no-untyped-def]
        return records

    projected: list[tuple[str, str]] = []

    async def fake_upsert(decision, *, memory_id: str):  # type: ignore[no-untyped-def]
        projected.append((memory_id, decision.target_property_or_entity))

    engine.list_active_memories = fake_memories  # type: ignore[method-assign]
    engine.upsert_claim_from_decision = fake_upsert  # type: ignore[method-assign]

    count = await engine.backfill_claims_from_active_memories(
        tenant,
        maker_id="maker_default",
        agent_id="agent_default",
    )

    assert count == 1
    assert projected == [("mem_old", "current_location")]


if __name__ == "__main__":
    asyncio.run(test_claim_projection_search_prefers_matching_claims())
    asyncio.run(test_claim_backfill_projects_profile_memories())
    print("Claim-aware retrieval tests passed.")
