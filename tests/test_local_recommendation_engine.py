#!/usr/bin/env python3
"""
Test suite for LocalRecommendationEngine.
"""

import json
from src.simulator.local_recommendation_engine import LocalRecommendationEngine, is_local_recommendation_query


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
