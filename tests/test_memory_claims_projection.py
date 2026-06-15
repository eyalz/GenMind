#!/usr/bin/env python3
"""Quality tests for typed claim projection behavior in MemoryEngine."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.schemas import (
    AUDNAction,
    AUDNDecision,
    MutationType,
    TemporalScope,
    TemporalType,
    TenantContext,
)
from src.services.memory_engine import MemoryEngine


def _decision(*, target: str, value: str, temporal_type: TemporalType = TemporalType.PERMANENT) -> AUDNDecision:
    return AUDNDecision(
        tenant=TenantContext(
            customer_id="cust_1",
            workspace_id="ws_1",
            end_user_id="user_1",
            session_id="sess_1",
        ),
        maker_id="maker_default",
        agent_id="agent_default",
        action=AUDNAction.UPDATE,
        mutation_type=MutationType.PROPERTY_MODIFICATION,
        source_entity="User_Profile",
        target_property_or_entity=target,
        value=value,
        temporal_scope=TemporalScope(
            type=temporal_type,
            valid_from="current_interaction",
            valid_until="indefinite" if temporal_type == TemporalType.PERMANENT else "conditional_trigger",
        ),
        reasoning_justification="test",
        reason="test decision",
        candidate_fact=f"profile.{target}={value}",
        confidence=0.9,
        target_memory_id="mem_existing",
    )


def test_claim_projection_has_stable_claim_id_per_subject_property() -> None:
    engine = MemoryEngine()

    d1 = _decision(target="current_location", value="Tel Aviv")
    d2 = _decision(target="current_location", value="Haifa")

    c1 = engine._build_claim_record(d1, memory_id="mem_1")
    c2 = engine._build_claim_record(d2, memory_id="mem_2")

    # Same tenant + maker/agent + source + target should map to the same claim_id slot.
    assert c1.claim_id == c2.claim_id
    assert c1.target_property_or_entity == "current_location"
    assert c2.value_json == "Haifa"


def test_claim_projection_changes_claim_id_for_different_target() -> None:
    engine = MemoryEngine()

    location = engine._build_claim_record(_decision(target="current_location", value="Tel Aviv"), memory_id="mem_1")
    language = engine._build_claim_record(_decision(target="language", value="English"), memory_id="mem_2")

    assert location.claim_id != language.claim_id


def test_claim_projection_preserves_temporal_semantics() -> None:
    engine = MemoryEngine()

    temp = engine._build_claim_record(
        _decision(target="current_location", value="Berlin", temporal_type=TemporalType.TEMPORARY),
        memory_id="mem_3",
    )

    assert temp.temporal_type == TemporalType.TEMPORARY
    assert temp.valid_until == "conditional_trigger"


if __name__ == "__main__":
    test_claim_projection_has_stable_claim_id_per_subject_property()
    test_claim_projection_changes_claim_id_for_different_target()
    test_claim_projection_preserves_temporal_semantics()
    print("All claim projection tests passed.")
