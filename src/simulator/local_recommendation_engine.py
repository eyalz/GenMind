"""
Smart Local Recommendation Detection Engine.

Replaces static keyword lists with dynamic dual-signal detection:
1. Location Presence: geographic entity or implicit local indicator
2. Recommendation Intent: quality/suggestion seeking signals
"""

import json
import logging
import re
from typing import Any, Optional, Union

# Configure logger for structured output
_engine_logger = logging.getLogger("local_rec_engine")
_engine_logger.setLevel(logging.DEBUG)

if not _engine_logger.handlers:
    handler = logging.FileHandler("/tmp/local_rec_engine.log")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    _engine_logger.addHandler(handler)


class LocalRecommendationEngine:
    """Intelligently detect local recommendation queries."""
    
    # **Abstract Nouns (Anti-Abstract Filter)** - concepts that aren't commercial entities
    ABSTRACT_NOUNS = {
        "reasons", "ideas", "problems", "issues", "concepts", "thoughts", "things",
        "views", "opinions", "perspectives", "aspects", "elements", "factors",
        "points", "arguments", "theories", "philosophies", "beliefs", "values",
        "feelings", "emotions", "sensations", "experiences", "moments", "times",
        "events", "situations", "circumstances", "conditions", "states", "stages",
        "levels", "degrees", "amounts", "quantities", "numbers", "prices",
        "costs", "expenses", "fees", "rates", "percentages", "data", "information",
        "knowledge", "facts", "figures", "statistics", "trends", "patterns",
    }
    
    # **Location Aliases** (airport codes, abbreviations, neighborhoods)
    LOCATION_ALIASES = {
        # Airport codes & city abbreviations
        "tlv": "Tel Aviv", "jfk": "New York", "lax": "Los Angeles",
        "lhr": "London", "cdg": "Paris", "fra": "Frankfurt",
        "nyc": "New York", "sf": "San Francisco", "la": "Los Angeles",
        "dc": "Washington DC", "chi": "Chicago", "bos": "Boston",
        "la": "Los Angeles", "philly": "Philadelphia", "sd": "San Diego",
        "sfo": "San Francisco", "ord": "Chicago", "mia": "Miami",
        "atl": "Atlanta", "sea": "Seattle", "den": "Denver",
        "dfw": "Dallas Fort Worth", "iah": "Houston", "sfo": "San Francisco",
        # Common neighborhoods
        "florentin": "Tel Aviv Florentin", "soho": "Manhattan Soho",
        "williamsburg": "Brooklyn Williamsburg", "tribeca": "Manhattan Tribeca",
        "midtown": "Midtown Manhattan", "downtown": "Downtown Manhattan",
        "east village": "Manhattan East Village", "west village": "Manhattan West Village",
        "upper east": "Manhattan Upper East", "upper west": "Manhattan Upper West",
        "lower east": "Manhattan Lower East", "financial district": "Manhattan Financial District",
    }
    
    # **Location Patterns** (explicit geographic indicators)
    LOCATION_PATTERNS = {
        # Explicit location phrases
        "explicit_location": r"\b(in|at|near|around|nearby|close to|around|beside)\s+([a-zA-Z][a-zA-Z\s\-']{0,50})\b",
        # Implicit "near me" / "around here" style
        "implicit_location": r"\b(near me|around me|nearby|around here|close by|local)\b",
        # Regional qualifiers
        "regional": r"\b(north|south|east|west|central|downtown|uptown|midtown|lower|upper)\s+([a-zA-Z][a-zA-Z\s]{0,40})\b",
    }
    
    # **Recommendation Intent Patterns** (signals seeking quality, suggestions)
    INTENT_PATTERNS = {
        # Quality/superlative seeking
        "quality_seeking": r"\b(best|top|good|great|excellent|recommended|highly rated|5[\-\s]?star|finest|premium|quality|popular)\b",
        # Availability/discovery seeking
        "discovery_seeking": r"\b(where|find|where to find|where can i|how to find|any|some|a good|a decent|looking for|search|recommendation|suggest|any recommendations)\b",
        # Service/professional/place noun seeking (implicit intent) - handles plurals with optional 's'
        "service_seeking": r"\b(lawyers?|attorneys?|doctors?|dentists?|pharmacists?|gyms?|fitness|hotels?|restaurants?|cafes?|bars?|pubs?|coffees?|pizzas?|sushi|thais?|chinese|indians?|japanese|italians?|frenchs?|mexicans?|americans?|barbershops?|salons?|hairdressers?|plumbers?|electricians?|mechanics?|hospitals?|clinics?|pharmacies?|shops?|stores?|supermarkets?|markets?|boutiques?|museums?|theaters?|cinemas?|banks?|posts?|offices?|libraries?|parks?|beaches?|mountains?|trails?|yogas?|pilates|trainers?|coaches?|counselors?|therapists?|psychologists?)\b",
    }
    
    # **Rejection Patterns** (queries that mention locations but aren't seeking local recommendations)
    REJECTION_PATTERNS = {
        "informational": r"\b(what is|what are|why|why are|compare|comparison|how many|population|weather|climate|history|culture|definition|explain|tell me about)\b",
        "travel_logistics": r"\b(flight|airline|ticket|booking|visa|immigration|passport|covid|covid-19|quarantine|hotel booking|accommodation booking)\b",
        "other_factual": r"\b(distance|map|geography|capital|currency|language|timezone|code|area|capital|located in|borders|history)\b",
    }
    
    # **Negation & Comparative Patterns**
    NEGATION_PATTERNS = {
        "explicit_negation": r"\b(don't|doesn't|didn't|won't|wouldn't|shouldn't|can't|couldn't|not|never|no)\s+\w+\s+(lawyers|doctors|restaurants|bars|shops|dentists|gyms)",
        "contrastive": r"\b(instead of|rather than|unlike|vs|versus|compared to|against)\b",
    }
    
    COMPARATIVE_PATTERNS = {
        "is_better": r"\b(is|are)\s+\w+(?:\s+\w+)*\s+(better|worse|different|same|similar)\s+than\b",
        "vs_comparison": r"\b(vs|versus|vs\.|compared to|compared with)\b",
        "which_comparison": r"\b(which|what)\s+\w*\s+(is|are)\s+(better|best|worse)\b",
    }
    
    @staticmethod
    def _extract_locations(query: str):
        """
        Extract all location references (explicit, implicit, regional).
        Returns dict with extraction details for logging.
        """
        locations = {
            "explicit": [],
            "implicit": False,
            "regional": [],
        }
        
        lowered = query.lower()
        
        # Extract explicit locations ("in X", "near X")
        for match in re.finditer(LocalRecommendationEngine.LOCATION_PATTERNS["explicit_location"], lowered):
            prep = match.group(1)
            location = match.group(2).strip()
            locations["explicit"].append({"preposition": prep, "location": location})
        
        # Check for implicit local reference
        if re.search(LocalRecommendationEngine.LOCATION_PATTERNS["implicit_location"], lowered):
            locations["implicit"] = True
        
        # Extract regional modifiers
        for match in re.finditer(LocalRecommendationEngine.LOCATION_PATTERNS["regional"], lowered):
            direction = match.group(1)
            area = match.group(2).strip() if match.lastindex >= 2 else None
            if area:
                locations["regional"].append({"direction": direction, "area": area})
        
        return locations
    
    @staticmethod
    def _detect_recommendation_intent(query: str):
        """
        Detect signals indicating recommendation-seeking intent.
        Returns dict with matched patterns for logging.
        """
        intent = {
            "quality_seeking": [],
            "discovery_seeking": [],
            "service_seeking": [],
            "has_intent": False,
        }
        
        lowered = query.lower()
        
        for pattern_name, pattern in LocalRecommendationEngine.INTENT_PATTERNS.items():
            if pattern_name.endswith("_seeking"):
                key = pattern_name
            else:
                key = f"{pattern_name}_seeking"
            
            matches = re.findall(pattern, lowered)
            if matches:
                intent[key] = matches
                intent["has_intent"] = True
        
        return intent
    
    @staticmethod
    def _check_rejection_criteria(query: str):
        """
        Check if query matches rejection patterns (informational, factual, etc).
        Returns dict with rejection details for logging.
        """
        rejections = {
            "matched_patterns": [],
            "is_rejected": False,
        }
        
        lowered = query.lower()
        
        for pattern_type, pattern in LocalRecommendationEngine.REJECTION_PATTERNS.items():
            if re.search(pattern, lowered):
                rejections["matched_patterns"].append(pattern_type)
                rejections["is_rejected"] = True
        
        return rejections
    
    @staticmethod
    def _is_abstract_noun(subject: Optional[str]) -> bool:
        """
        Check if the extracted subject is an abstract concept rather than a concrete entity.
        Returns True if subject is abstract (should reject), False if concrete (allow).
        """
        if not subject:
            return False
        
        subject_lower = subject.lower().strip()
        
        # Check exact word match
        if subject_lower in LocalRecommendationEngine.ABSTRACT_NOUNS:
            return True
        
        # Check if any word in the subject is an abstract noun
        # e.g., "good problems" -> check "problems", "good reasons" -> check "reasons"
        words = subject_lower.split()
        for word in words:
            if word.rstrip('s') in LocalRecommendationEngine.ABSTRACT_NOUNS or word in LocalRecommendationEngine.ABSTRACT_NOUNS:
                return True
        
        return False
    
    @staticmethod
    def _detect_negation_and_comparatives(query: str):
        """
        Detect negation modifiers and comparative questions.
        Returns dict with flags and detected patterns.
        """
        lowered = query.lower()
        result = {
            "has_negation_constraints": False,
            "negation_patterns": [],
            "is_comparative_question": False,
            "comparative_patterns": [],
        }
        
        # Check for negation patterns
        for pattern_name, pattern in LocalRecommendationEngine.NEGATION_PATTERNS.items():
            if re.search(pattern, lowered):
                result["has_negation_constraints"] = True
                result["negation_patterns"].append(pattern_name)
        
        # Check for comparative patterns
        for pattern_name, pattern in LocalRecommendationEngine.COMPARATIVE_PATTERNS.items():
            if re.search(pattern, lowered):
                result["is_comparative_question"] = True
                result["comparative_patterns"].append(pattern_name)
        
        return result
    
    @staticmethod
    def _expand_location_aliases(query: str):
        """
        Recognize and expand airport codes, abbreviations, and neighborhood slang.
        Returns dict with detected alias types and expanded names.
        """
        lowered = query.lower()
        detected_aliases = {
            "airport_codes": [],
            "city_abbreviations": [],
            "neighborhoods": [],
            "location_type": "explicit_city",  # default
        }
        
        for alias, expanded_name in LocalRecommendationEngine.LOCATION_ALIASES.items():
            # Use word boundary to avoid partial matches
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, lowered):
                if alias in ["tlv", "jfk", "lax", "lhr", "cdg", "fra", "sfo", "ord", "mia", "atl", "sea", "den", "dfw", "iah"]:
                    detected_aliases["airport_codes"].append({"code": alias, "name": expanded_name})
                    detected_aliases["location_type"] = "explicit_city"
                elif alias in ["nyc", "sf", "la", "dc", "chi", "bos", "philly", "sd"]:
                    detected_aliases["city_abbreviations"].append({"abbreviation": alias, "name": expanded_name})
                    detected_aliases["location_type"] = "explicit_city"
                else:
                    detected_aliases["neighborhoods"].append({"neighborhood": alias, "expanded": expanded_name})
                    detected_aliases["location_type"] = "neighborhood_slang"
        
        # Check for implicit location (near me, nearby, etc.)
        if re.search(r"\b(near me|around me|nearby|around here|close by)\b", lowered):
            detected_aliases["location_type"] = "implicit_proximity"
        
        return detected_aliases
    
    @staticmethod
    def _extract_target_subject(query: str) -> Optional[str]:
        """
        Heuristically extract the primary subject being searched for.
        e.g., "lawyers", "pizza", "gym", "dentist", "reasons"
        """
        # Pattern 1: Look for nouns/entities typically appearing before "in/near"
        match = re.search(
            r"(?:good|best|any|some|the|top|find)\s+([a-zA-Z][a-zA-Z\s]{0,40}?)\s+(?:in|near|around|at)",
            query,
            re.IGNORECASE
        )
        if match:
            subject = match.group(1).strip().rstrip(" ?,.")
            return subject if len(subject) < 50 else None
        
        # Pattern 2: Handle "any good X to/for" structures
        match = re.search(
            r"(?:any|some)\s+(?:good|bad|best|top|great)\s+([a-zA-Z][a-zA-Z\s]{0,40}?)\s+(?:to|for)",
            query,
            re.IGNORECASE
        )
        if match:
            subject = match.group(1).strip().rstrip(" ?,.")
            return subject if len(subject) < 50 else None
        
        # Pattern 3: Fallback - extract word/phrase before first location preposition
        match = re.search(
            r"(?:^|\s)([a-zA-Z][a-zA-Z\s]{0,40}?)\s+(?:in|near|around|at)\s",
            query,
            re.IGNORECASE
        )
        if match:
            subject = match.group(1).strip().rstrip(" ?,.")
            return subject if len(subject) < 50 else None
        
        return None
    
    @staticmethod
    def evaluate(query: str):
        """
        Evaluate if a query is a local recommendation query.
        
        Args:
            query: The user's question
        
        Returns:
            (is_local_recommendation: bool, decision_matrix: dict)
        """
        # Step 1: Extract location signals
        locations = LocalRecommendationEngine._extract_locations(query)
        location_aliases = LocalRecommendationEngine._expand_location_aliases(query)
        has_location = bool(locations["explicit"] or locations["implicit"] or locations["regional"] or 
                          location_aliases["airport_codes"] or location_aliases["city_abbreviations"] or 
                          location_aliases["neighborhoods"])
        
        # Step 2: Detect recommendation intent
        intent = LocalRecommendationEngine._detect_recommendation_intent(query)
        
        # Step 3: Check rejection criteria
        rejections = LocalRecommendationEngine._check_rejection_criteria(query)
        
        # Step 4: Extract the target subject (what they're looking for)
        target_subject = LocalRecommendationEngine._extract_target_subject(query)
        is_abstract = LocalRecommendationEngine._is_abstract_noun(target_subject)
        
        # Step 5: Detect negation and comparative structures
        negation_info = LocalRecommendationEngine._detect_negation_and_comparatives(query)
        
        # Final decision: location + intent + no rejection + not abstract + not comparative
        # Note: negation doesn't block (still local rec) but is logged as a constraint
        is_local_rec = (
            has_location and
            intent["has_intent"] and
            not rejections["is_rejected"] and
            not is_abstract and
            not negation_info["is_comparative_question"]
        )
        
        # Build detailed decision matrix for logging
        decision_matrix = {
            "query": query,
            "metrics": {
                "location_detected": has_location,
                "location_type": location_aliases["location_type"],
                "extracted_locations": locations,
                "location_aliases_detected": {
                    "airport_codes": location_aliases["airport_codes"],
                    "city_abbreviations": location_aliases["city_abbreviations"],
                    "neighborhoods": location_aliases["neighborhoods"],
                },
                "recommendation_intent_detected": intent["has_intent"],
                "matched_intent_signals": intent,
                "rejection_patterns_matched": rejections["matched_patterns"],
                "dynamic_target_subject": target_subject,
                "is_abstract_noun": is_abstract,
                "has_negation_constraints": negation_info["has_negation_constraints"],
                "negation_patterns_detected": negation_info["negation_patterns"],
                "is_comparative_question": negation_info["is_comparative_question"],
                "comparative_patterns_detected": negation_info["comparative_patterns"],
            },
            "signals": {
                "location_present": has_location,
                "intent_present": intent["has_intent"],
                "rejection_triggered": rejections["is_rejected"],
                "abstract_noun_triggered": is_abstract,
                "negation_present": negation_info["has_negation_constraints"],
                "comparative_structure": negation_info["is_comparative_question"],
            },
            "final_decision": is_local_rec,
            "reasoning": LocalRecommendationEngine._build_reasoning(
                has_location, intent["has_intent"], rejections["is_rejected"],
                is_abstract, negation_info["is_comparative_question"]
            ),
        }
        
        # Log the decision matrix as JSON
        _engine_logger.debug(json.dumps(decision_matrix, indent=2))
        
        return is_local_rec, decision_matrix
    
    @staticmethod
    def _build_reasoning(has_location: bool, has_intent: bool, is_rejected: bool,
                        is_abstract: bool = False, is_comparative: bool = False) -> str:
        """Build a human-readable explanation of the decision."""
        reasons = []
        
        if not has_location:
            reasons.append("no location detected")
        else:
            reasons.append("✓ location present")
        
        if not has_intent:
            reasons.append("no recommendation intent")
        else:
            reasons.append("✓ intent present")
        
        if is_abstract:
            reasons.append("⚠ abstract noun (not a service/entity)")
        
        if is_comparative:
            reasons.append("⚠ comparative question (not direct recommendation)")
        
        if is_rejected:
            reasons.append("⚠ rejection pattern triggered")
        
        if has_location and has_intent and not is_rejected and not is_abstract and not is_comparative:
            return "ACCEPT: location + intent, no rejections"
        else:
            return f"REJECT: {' | '.join(reasons)}"


def is_local_recommendation_query(question: str) -> bool:
    """
    Public interface: Determine if query is a local recommendation query.
    
    Args:
        question: User's question
    
    Returns:
        True if query is seeking local recommendations, False otherwise
    """
    result, _ = LocalRecommendationEngine.evaluate(question)
    return result
