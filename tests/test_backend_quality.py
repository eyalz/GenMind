#!/usr/bin/env python3
"""Focused backend-quality checks for retrieval policy and memory write filtering."""

from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.schemas import RetrievalCandidate
from src.services.audn_pipeline import AUDNPipeline
from src.services.context_optimizer import ContextOptimizer
from src.services.memory_engine import MemoryEngine


def _candidate(mid: str, content: str, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(
        memory_id=mid,
        content=content,
        semantic_score=score,
        graph_score=0.0,
        recency_score=0.0,
        final_score=score,
        updated_at=datetime.now(timezone.utc),
    )


def test_context_optimizer_adaptive_policy() -> None:
    optimizer = ContextOptimizer(memory_engine=MemoryEngine())

    assert optimizer._classify_retrieval_mode("what is the current status?") == "fact_lookup"
    assert optimizer._classify_retrieval_mode("my preferred language") == "profile"
    assert optimizer._classify_retrieval_mode("any good ones there?") == "followup"

    top_k_profile = optimizer._adaptive_top_k(
        "profile",
        total_candidates=40,
        max_tokens=1000,
        light_memory_mode=False,
    )
    top_k_fact = optimizer._adaptive_top_k(
        "fact_lookup",
        total_candidates=40,
        max_tokens=1000,
        light_memory_mode=False,
    )
    assert top_k_profile >= top_k_fact

    assert optimizer._should_use_light_memory_mode("What is the weather in Tel Aviv?", "fact_lookup") is True
    assert optimizer._should_use_light_memory_mode("What is my current timezone?", "fact_lookup") is False

    rows = [
        _candidate("m1", "profile.language=english", 0.19),
        _candidate("m2", "profile.language=english", 0.18),
        _candidate("m3", "profile.timezone=utc+2", 0.11),
    ]

    filtered = optimizer._apply_score_threshold(rows, 0.15)
    assert len(filtered) == 2

    deduped = optimizer._dedupe_semantic_duplicates(filtered, top_k=5)
    assert len(deduped) == 1

    assert optimizer._should_append_recent_questions("is it still valid?", "followup", False) is True
    assert optimizer._should_append_recent_questions("Explain distributed consensus in detail", "fact_lookup", False) is False
    assert optimizer._should_append_recent_questions("what is weather", "fact_lookup", True) is False


def test_claim_reconciliation_prefers_recent_and_profile() -> None:
    optimizer = ContextOptimizer(memory_engine=MemoryEngine())
    old = datetime(2025, 1, 1, tzinfo=timezone.utc)
    new = datetime(2026, 1, 1, tzinfo=timezone.utc)

    candidates = [
        RetrievalCandidate(
            memory_id="m_old",
            content="uome|v=1|mt=property_modification|src=user_profile|tgt=current_location|val=london|tt=permanent|vf=current_interaction|vu=indefinite",
            semantic_score=0.6,
            graph_score=0.0,
            recency_score=0.0,
            final_score=0.5,
            updated_at=old,
        ),
        RetrievalCandidate(
            memory_id="m_new",
            content="profile.current_location=Tel Aviv|scope=temporary",
            semantic_score=0.7,
            graph_score=0.0,
            recency_score=0.0,
            final_score=0.7,
            updated_at=new,
        ),
    ]

    winner = optimizer._select_claim_winner(candidates, "where am i now?")
    assert winner.memory_id == "m_new"

    reconciled, dropped = optimizer._reconcile_claim_rows(candidates, "where am i now?")
    assert dropped == 1
    assert len(reconciled) == 1
    assert reconciled[0].memory_id == "m_new"


def test_audn_low_value_filtering() -> None:
    pipeline = AUDNPipeline(memory_engine=MemoryEngine())

    assert pipeline._is_low_value_candidate("Can you find me a good lawyer in Tel Aviv?") is True
    assert pipeline._is_low_value_candidate("What is the weather today?") is True
    assert pipeline._is_low_value_candidate("my favorite cuisine is thai") is False
    assert pipeline._is_low_value_candidate("profile.current_location=Tel Aviv|scope=temporary") is False
    assert pipeline._is_low_value_candidate("my manager is Dana") is False


if __name__ == "__main__":
    test_context_optimizer_adaptive_policy()
    test_claim_reconciliation_prefers_recent_and_profile()
    test_audn_low_value_filtering()
    print("All backend quality tests passed.")
