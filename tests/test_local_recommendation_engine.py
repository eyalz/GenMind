#!/usr/bin/env python3
"""
Test suite for LocalRecommendationEngine.
"""

import json
from src.simulator.local_recommendation_engine import (
    LocalRecommendationEngine,
    analyze_local_recommendation,
    is_local_recommendation_query,
)


def test_local_recommendation_engine():
    """Test the dual-signal detection engine."""
    
    test_cases = [
        # (query, expected_result, description)
        # LOCAL RECOMMENDATIONS (should be True)
        ("any good lawyers in tel aviv ?", True, "lawyer query with location"),
        ("best bars in tel aviv", True, "bar query with location"),
        ("restaurants near new york", True, "restaurant query with location"),
        ("where can I find a good doctor in london", True, "discovery + location"),
        ("top rated dentists around paris", True, "quality + location"),
        ("recommend a hotel near me", True, "recommendation intent + implicit location"),
        ("best pizza nearby", True, "quality + implicit location"),
        ("find me a gym close to here", True, "find + implicit location"),
        ("good coffee shops in manhattan", True, "quality + location"),
        ("where are the best museums in rome", True, "where + best + location"),
        
        # NON-LOCAL (should be False)
        ("latest news", False, "informational, no location"),
        ("what's the weather", False, "informational, no location"),
        ("flight to new york", False, "travel logistics"),
        ("visa requirements for france", False, "travel logistics"),
        ("what is the population of tel aviv", False, "informational despite location"),
        ("comparing tech in london vs berlin", False, "comparative info despite locations"),
        ("how many people live in paris", False, "factual query despite location"),
        ("distance from new york to boston", False, "factual query despite locations"),
        ("what language do they speak in spain", False, "factual despite location"),
        ("tell me about the history of rome", False, "informational despite location"),
    ]
    
    print("=" * 80)
    print("TESTING LocalRecommendationEngine")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for query, expected, description in test_cases:
        result, decision_matrix = LocalRecommendationEngine.evaluate(query)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        
        if result == expected:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status}: {description}")
        print(f"  Query: {query!r}")
        print(f"  Expected: {expected}, Got: {result}")
        print(f"  Reasoning: {decision_matrix['reasoning']}")
        
        if result != expected:
            # Print detailed decision matrix for failed cases
            print(f"  Full Decision Matrix:")
            for key, value in decision_matrix["metrics"].items():
                if isinstance(value, (dict, list)):
                    print(f"    {key}: {json.dumps(value, indent=6)}")
                else:
                    print(f"    {key}: {value}")
    
    print("\n" + "=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


def test_location_extraction():
    """Test location extraction logic."""
    print("\n" + "=" * 80)
    print("TESTING Location Extraction")
    print("=" * 80)
    
    queries = [
        "any good lawyers in tel aviv ?",
        "best bars near me",
        "restaurants around paris",
        "find me a doctor nearby",
        "top gyms in manhattan",
    ]
    
    for query in queries:
        locations = LocalRecommendationEngine._extract_locations(query)
        print(f"\nQuery: {query!r}")
        print(f"  Extracted: {json.dumps(locations, indent=4)}")


def test_intent_extraction():
    """Test intent pattern matching."""
    print("\n" + "=" * 80)
    print("TESTING Intent Extraction")
    print("=" * 80)
    
    queries = [
        "best lawyers",
        "where can I find a good restaurant",
        "any recommendations for bars",
        "top rated hotels",
        "suggest a gym",
    ]
    
    for query in queries:
        intent = LocalRecommendationEngine._detect_recommendation_intent(query)
        print(f"\nQuery: {query!r}")
        print(f"  Intent Signals: {json.dumps(intent, indent=4)}")


def test_target_subject_extraction():
    """Test subject/entity extraction."""
    print("\n" + "=" * 80)
    print("TESTING Target Subject Extraction")
    print("=" * 80)
    
    queries = [
        "good lawyers in tel aviv",
        "best pizza near me",
        "find a dentist around paris",
        "where are top restaurants in london",
        "any good coffee shops nearby",
    ]
    
    for query in queries:
        subject = LocalRecommendationEngine._extract_target_subject(query)
        print(f"\nQuery: {query!r}")
        print(f"  Target Subject: {subject!r}")


def test_dual_layer_flow_a_layer1_success():
    """Flow A: Layer 1 rolling history should resolve subject for pronoun follow-up."""
    history = [
        "Who is the best corporate lawyer?",
        "Do they handle contract disputes?",
    ]
    session_state = {
        "session_id": "sess_flow_a",
        "context_snapshot": {
            "primary_subject_entity": None,
            "inferred_current_location": None,
            "user_constraints": [],
        },
    }

    decision, payload = analyze_local_recommendation(
        current_query="any good ones in Tel Aviv?",
        history_list=history,
        session_db_state=session_state,
    )

    assert decision is True
    assert payload["resolved_matrix"]["resolved_subject"] is not None
    assert payload["resolved_matrix"]["resolved_location"] is not None
    assert payload["memory_metrics"]["layer1_context_inherited"] is True
    assert "primary_subject_entity" in payload["memory_metrics"]["layer1_inherited_fields"]


def test_dual_layer_flow_b_layer2_fallback_success():
    """Flow B: Layer 2 DB fallback should resolve subject when rolling history misses it."""
    history = [
        "What are their typical hourly rates?",
        "Can you explain retainer fees?",
        "Do they accept credit cards?",
    ]
    session_state = {
        "session_id": "sess_flow_b",
        "context_snapshot": {
            "primary_subject_entity": "lawyer",
            "inferred_current_location": "Tel Aviv",
            "user_constraints": [],
        },
    }

    decision, payload = analyze_local_recommendation(
        current_query="Are there any good choices in Haifa?",
        history_list=history,
        session_db_state=session_state,
    )

    assert decision is True
    assert payload["memory_metrics"]["layer2_database_fallback_used"] is True
    assert payload["resolved_matrix"]["resolved_subject"] == "lawyer"
    assert payload["resolved_matrix"]["resolved_location"] is not None


def test_session_state_mutation_and_protection_rules():
    """Verify overwrite on strong new entity and no pollution on informational turns."""
    base_state = {
        "session_id": "sess_mutation",
        "context_snapshot": {
            "primary_subject_entity": "lawyer",
            "inferred_current_location": "Tel Aviv",
            "user_constraints": [],
        },
    }

    # Strong override should mutate subject.
    decision1, payload1 = analyze_local_recommendation(
        current_query="Actually, forget lawyers, find me a good pizza spot in Tel Aviv",
        history_list=["who is the best lawyer in tel aviv"],
        session_db_state=base_state,
    )
    assert decision1 is True
    snapshot1 = payload1["session_db_state"]["context_snapshot"]
    assert snapshot1["primary_subject_entity"] is not None
    assert "pizza" in snapshot1["primary_subject_entity"].lower()
    assert payload1["memory_metrics"]["database_state_mutated"] is True

    # Informational query should not clear subject.
    decision2, payload2 = analyze_local_recommendation(
        current_query="what time does the sun set in Tel Aviv?",
        history_list=["Actually, forget lawyers, find me a good pizza spot in Tel Aviv"],
        session_db_state=payload1["session_db_state"],
    )
    assert decision2 is False
    snapshot2 = payload2["session_db_state"]["context_snapshot"]
    assert snapshot2["primary_subject_entity"] == snapshot1["primary_subject_entity"]


if __name__ == "__main__":
    # Run all tests
    success = test_local_recommendation_engine()
    test_location_extraction()
    test_intent_extraction()
    test_target_subject_extraction()
    
    if success:
        print("\n✓ All tests passed!")
        exit(0)
    else:
        print("\n✗ Some tests failed")
        exit(1)
