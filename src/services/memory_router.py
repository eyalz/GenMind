from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from functools import lru_cache
from typing import Any


@dataclass
class ExtractionFact:
    fact: str
    confidence: float
    category: str
    source_tier: str


@dataclass
class RoutingResult:
    tier: str
    facts: list[ExtractionFact] = field(default_factory=list)
    reasoning: str = ""


class EntityResolver:
    """Lightweight synonym and canonical-ID resolver for common entities."""

    LOCATION_SYNONYMS: dict[str, str] = {
        "napoli": "Naples",
        "naples": "Naples",
        "yafo": "Jaffa",
        "jafa": "Jaffa",
        "jaffa": "Jaffa",
        "tlv": "Tel Aviv",
        "tel aviv-yafo": "Tel Aviv",
        "tel aviv yafo": "Tel Aviv",
        "tel-aviv": "Tel Aviv",
        "tel aviv": "Tel Aviv",
        "ta": "Tel Aviv",
        "beaches in tel aviv": "Tel Aviv",
        "beach in tel aviv": "Tel Aviv",
        "nyc": "New York",
        "new york city": "New York",
    }

    def resolve_location(self, raw_value: str) -> str:
        key = self._clean(raw_value)
        if key in self.LOCATION_SYNONYMS:
            return self.LOCATION_SYNONYMS[key]

        matches = get_close_matches(key, self.LOCATION_SYNONYMS.keys(), n=1, cutoff=0.9)
        if matches:
            return self.LOCATION_SYNONYMS[matches[0]]

        return " ".join(part.capitalize() for part in key.split())

    def resolve_generic(self, raw_value: str) -> str:
        return " ".join(part.capitalize() for part in self._clean(raw_value).split())

    def _clean(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())


class DeterministicExtractor:
    """Tier 1 extractor for exact deterministic patterns (cost $0)."""

    IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    UUID_PATTERN = re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
    )
    FUSE_PATTERN = re.compile(r"\bFuse\s+(\d{1,8})\b", re.IGNORECASE)
    SYSTEM_CODE_PATTERN = re.compile(r"\b(?:SYS|ERR|ALRT|MOD)-\d{2,8}\b", re.IGNORECASE)

    def extract(self, text: str) -> list[ExtractionFact]:
        facts: list[ExtractionFact] = []

        for ip in self.IPV4_PATTERN.findall(text):
            facts.append(
                ExtractionFact(
                    fact=f"system.ip_address={ip}",
                    confidence=0.99,
                    category="SYSTEM_CODE",
                    source_tier="tier_1_deterministic",
                )
            )

        for item in self.UUID_PATTERN.findall(text):
            facts.append(
                ExtractionFact(
                    fact=f"system.uuid={item.lower()}",
                    confidence=0.99,
                    category="SYSTEM_CODE",
                    source_tier="tier_1_deterministic",
                )
            )

        for fuse in self.FUSE_PATTERN.findall(text):
            facts.append(
                ExtractionFact(
                    fact=f"system.fuse_number={fuse}",
                    confidence=0.99,
                    category="HARDWARE_COMPONENT",
                    source_tier="tier_1_deterministic",
                )
            )

        for code in self.SYSTEM_CODE_PATTERN.findall(text):
            facts.append(
                ExtractionFact(
                    fact=f"system.code={code.upper()}",
                    confidence=0.99,
                    category="SYSTEM_CODE",
                    source_tier="tier_1_deterministic",
                )
            )

        return facts


class LightweightNERExtractor:
    """Tier 2 local extractor (spaCy when available, regex fallback otherwise)."""

    SOFTWARE_FRAMEWORK_PATTERN = re.compile(
        r"\b(react|next\.js|nextjs|vue|angular|svelte|django|flask|fastapi|spring|laravel|node\.js|nodejs|express)\b",
        re.IGNORECASE,
    )
    HARDWARE_COMPONENT_PATTERN = re.compile(
        r"\b(cpu|gpu|ssd|hdd|ram|router|switch|firewall|sensor|motherboard|fuse\s+\d+)\b",
        re.IGNORECASE,
    )
    DIETARY_CONSTRAINT_PATTERN = re.compile(
        r"\b(vegan|vegetarian|kosher|halal|gluten[-\s]?free|lactose[-\s]?intolerant|nut allergy|peanut allergy)\b",
        re.IGNORECASE,
    )
    LOCATION_QUERY_PATTERN = re.compile(
        r"\b(?:in|at|near)\s+([A-Za-z][A-Za-z\s\-']{1,60})(?=\?|\.|,|$)",
        re.IGNORECASE,
    )
    QUERY_TOPIC_LOCATION_PATTERN = re.compile(
        r"\b(?:any\s+good\s+|best\s+|find\s+|looking\s+for\s+)?([a-zA-Z][a-zA-Z\s\-']{2,40}?)\s+in\s+([a-zA-Z][a-zA-Z\s\-']{1,60})(?=\?|\.|,|$)",
        re.IGNORECASE,
    )
    FOLLOWUP_TOPIC_PATTERN = re.compile(
        r"\b(?:what\s+about|how\s+about|and\s+for)\s+([a-zA-Z][a-zA-Z\s\-']{2,40})(?=\?|\.|,|$)",
        re.IGNORECASE,
    )
    SHORT_CONTINUATION_TOPIC_PATTERN = re.compile(
        r"\b(?:and)\s+(?!for\b)([a-zA-Z][a-zA-Z\s\-']{2,40})(?=\?|\.|,|$)",
        re.IGNORECASE,
    )
    HOME_BASE_PATTERN = re.compile(
        r"\b(?:home\s+base|based)\s+(?:is\s+)?(?:still\s+)?([a-zA-Z][a-zA-Z\s\-']{1,60})(?=\.|,|$|\s+but\s+)",
        re.IGNORECASE,
    )
    TEMP_TRAVEL_PATTERN = re.compile(
        r"\b(?:going\s+to|travel(?:ling)?\s+to|visiting)\s+([a-zA-Z][a-zA-Z\s\-']{1,60})\s+(?:next\s+week|this\s+week|temporarily|for\s+\d+\s+days?)\b",
        re.IGNORECASE,
    )
    BEACH_TOPIC_PATTERN = re.compile(
        r"\b(beach|beaches|shore|coastline|coast|seaside|waterfront)\b",
        re.IGNORECASE,
    )

    def __init__(self, resolver: EntityResolver) -> None:
        self.resolver = resolver
        self._spacy_nlp = self._load_spacy_model()

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_spacy_model():
        try:
            import spacy  # type: ignore

            for model_name in ("en_core_web_sm", "en_core_web_md"):
                try:
                    return spacy.load(model_name)
                except Exception:
                    continue
            return None
        except Exception:
            return None

    def extract(self, text: str) -> list[ExtractionFact]:
        facts: list[ExtractionFact] = []

        if self._spacy_nlp is not None:
            facts.extend(self._extract_with_spacy(text))

        facts.extend(self._extract_with_regex(text))

        # De-duplicate by fact string while preserving higher confidence entries.
        best_by_fact: dict[str, ExtractionFact] = {}
        for fact in facts:
            existing = best_by_fact.get(fact.fact)
            if existing is None or fact.confidence > existing.confidence:
                best_by_fact[fact.fact] = fact
        return list(best_by_fact.values())

    def _extract_with_spacy(self, text: str) -> list[ExtractionFact]:
        facts: list[ExtractionFact] = []
        doc = self._spacy_nlp(text)

        for ent in doc.ents:
            label = ent.label_.upper()
            value = ent.text.strip()
            if not value:
                continue

            if label in {"GPE", "LOC"}:
                location = self.resolver.resolve_location(value)
                facts.append(
                    ExtractionFact(
                        fact=f"profile.search_location={location}|scope=temporary",
                        confidence=0.87,
                        category="LOCATION",
                        source_tier="tier_2_local_ner",
                    )
                )
            elif label in {"PERSON"}:
                facts.append(
                    ExtractionFact(
                        fact=f"entity.person={self.resolver.resolve_generic(value)}",
                        confidence=0.84,
                        category="PERSON",
                        source_tier="tier_2_local_ner",
                    )
                )
            elif label in {"ORG"}:
                facts.append(
                    ExtractionFact(
                        fact=f"entity.organization={self.resolver.resolve_generic(value)}",
                        confidence=0.84,
                        category="ORGANIZATION",
                        source_tier="tier_2_local_ner",
                    )
                )

        return facts

    def _extract_with_regex(self, text: str) -> list[ExtractionFact]:
        facts: list[ExtractionFact] = []

        for topic, location in self.QUERY_TOPIC_LOCATION_PATTERN.findall(text):
            clean_topic = re.sub(r"\s+", "_", topic.strip().lower())
            clean_topic = re.sub(r"[^a-z0-9_\-]", "", clean_topic)
            resolved_location = self.resolver.resolve_location(location)
            if clean_topic:
                facts.append(
                    ExtractionFact(
                        fact=f"profile.search_topic={clean_topic}|scope=temporary",
                        confidence=0.88,
                        category="QUERY_TOPIC",
                        source_tier="tier_2_local_ner",
                    )
                )
            facts.append(
                ExtractionFact(
                    fact=f"profile.search_location={resolved_location}|scope=temporary",
                    confidence=0.9,
                    category="LOCATION",
                    source_tier="tier_2_local_ner",
                )
            )

        for topic in self.FOLLOWUP_TOPIC_PATTERN.findall(text):
            clean_topic = re.sub(r"\s+", "_", topic.strip().lower())
            clean_topic = re.sub(r"[^a-z0-9_\-]", "", clean_topic)
            if not clean_topic:
                continue
            facts.append(
                ExtractionFact(
                    fact=f"profile.search_topic={clean_topic}|scope=temporary",
                    confidence=0.84,
                    category="QUERY_TOPIC",
                    source_tier="tier_2_local_ner",
                )
            )

        for topic in self.SHORT_CONTINUATION_TOPIC_PATTERN.findall(text):
            clean_topic = re.sub(r"\s+", "_", topic.strip().lower())
            clean_topic = re.sub(r"[^a-z0-9_\-]", "", clean_topic)
            if not clean_topic:
                continue
            facts.append(
                ExtractionFact(
                    fact=f"profile.search_topic={clean_topic}|scope=temporary",
                    confidence=0.84,
                    category="QUERY_TOPIC",
                    source_tier="tier_2_local_ner",
                )
            )

        for framework in self.SOFTWARE_FRAMEWORK_PATTERN.findall(text):
            facts.append(
                ExtractionFact(
                    fact=f"profile.software_framework={framework.lower().replace('.', '')}",
                    confidence=0.86,
                    category="SOFTWARE_FRAMEWORK",
                    source_tier="tier_2_local_ner",
                )
            )

        for component in self.HARDWARE_COMPONENT_PATTERN.findall(text):
            facts.append(
                ExtractionFact(
                    fact=f"profile.hardware_component={component.lower()}",
                    confidence=0.86,
                    category="HARDWARE_COMPONENT",
                    source_tier="tier_2_local_ner",
                )
            )

        for dietary in self.DIETARY_CONSTRAINT_PATTERN.findall(text):
            facts.append(
                ExtractionFact(
                    fact=f"profile.dietary_constraint={dietary.lower().replace(' ', '_')}",
                    confidence=0.9,
                    category="DIETARY_CONSTRAINT",
                    source_tier="tier_2_local_ner",
                )
            )

        # Low-cost location catch when spaCy isn't available.
        for loc in self.LOCATION_QUERY_PATTERN.findall(text):
            normalized = self.resolver.resolve_location(loc)
            facts.append(
                ExtractionFact(
                    fact=f"profile.search_location={normalized}|scope=temporary",
                    confidence=0.78,
                    category="LOCATION",
                    source_tier="tier_2_local_ner",
                )
            )

        for home_loc in self.HOME_BASE_PATTERN.findall(text):
            normalized = self.resolver.resolve_location(home_loc)
            facts.append(
                ExtractionFact(
                    fact=f"profile.home_base_location={normalized}",
                    confidence=0.9,
                    category="LOCATION",
                    source_tier="tier_2_local_ner",
                )
            )

        for travel_loc in self.TEMP_TRAVEL_PATTERN.findall(text):
            normalized = self.resolver.resolve_location(travel_loc)
            facts.append(
                ExtractionFact(
                    fact=f"profile.current_location={normalized}|scope=temporary",
                    confidence=0.9,
                    category="LOCATION",
                    source_tier="tier_2_local_ner",
                )
            )

        for beach_topic in self.BEACH_TOPIC_PATTERN.findall(text):
            facts.append(
                ExtractionFact(
                    fact=f"profile.search_topic={beach_topic.lower()}|scope=temporary",
                    confidence=0.82,
                    category="QUERY_TOPIC",
                    source_tier="tier_2_local_ner",
                )
            )

        return facts


class SmallLLMFallbackExtractor:
    """Tier 3 compact fallback for complex statements when local extraction is ambiguous."""

    UPDATE_CUES = (
        "update",
        "change",
        "replace",
        "correct",
        "set",
        "keep",
        "prefer",
        "use this instead",
        "actually",
        "not anymore",
        "no longer",
    )
    DELETE_CUES = (
        "delete",
        "remove",
        "forget",
        "clear",
        "drop",
        "erase",
        "ignore previous",
        "cancel",
    )

    def __init__(self, resolver: EntityResolver) -> None:
        self.resolver = resolver
        self._openai_client = self._build_client()

    def has_modification_cue(self, text: str) -> bool:
        lowered = f" {text.lower()} "
        for cue in self.UPDATE_CUES + self.DELETE_CUES:
            cue_check = f" {cue} "
            if cue_check in lowered:
                return True
        return False

    def extract(self, text: str) -> list[ExtractionFact]:
        # If API key is missing, keep behavior deterministic and local.
        if self._openai_client is None:
            return []

        prompt = {
            "task": "Extract memory mutations into compact facts",
            "format": "json",
            "rules": [
                "Return strictly valid JSON array",
                "Each item has fact, category, confidence",
                "Use canonical profile.*/entity.*/system.* keys",
                "Prefer temporary scope for short-lived contexts",
            ],
            "text": text,
        }

        try:
            response = self._openai_client.chat.completions.create(
                model=os.getenv("GENMIND_SMALL_LLM_MODEL", "gpt-4o-mini"),
                temperature=0,
                max_tokens=220,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You output compact JSON only."},
                    {"role": "user", "content": json.dumps(prompt)},
                ],
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            raw_items = payload.get("facts", []) if isinstance(payload, dict) else []

            facts: list[ExtractionFact] = []
            for item in raw_items:
                fact = str(item.get("fact", "")).strip()
                if not fact:
                    continue
                category = str(item.get("category", "GENERIC")).upper()
                confidence = float(item.get("confidence", 0.72))
                facts.append(
                    ExtractionFact(
                        fact=fact,
                        confidence=min(max(confidence, 0.0), 1.0),
                        category=category,
                        source_tier="tier_3_small_llm",
                    )
                )
            return facts
        except Exception:
            return []

    @staticmethod
    def _build_client():
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None

        try:
            from openai import OpenAI  # type: ignore

            return OpenAI(api_key=api_key)
        except Exception:
            return None


class MemoryRouter:
    """Three-tier hybrid extraction pipeline for AUDN ingestion."""

    def __init__(self) -> None:
        self.resolver = EntityResolver()
        self.tier1 = DeterministicExtractor()
        self.tier2 = LightweightNERExtractor(self.resolver)
        self.tier3 = SmallLLMFallbackExtractor(self.resolver)

    def route(self, text: str) -> RoutingResult:
        clean = " ".join(text.split()).strip()
        if not clean:
            return RoutingResult(tier="none", facts=[], reasoning="empty_input")

        deterministic = self.tier1.extract(clean)
        if deterministic:
            return RoutingResult(
                tier="tier_1_deterministic",
                facts=deterministic,
                reasoning="Matched exact deterministic patterns.",
            )

        local_facts = self.tier2.extract(clean)
        if local_facts:
            return RoutingResult(
                tier="tier_2_local_ner",
                facts=local_facts,
                reasoning="Resolved through local NER/regex without external model.",
            )

        should_run_tier3 = not deterministic and not local_facts and self.tier3.has_modification_cue(clean)
        llm_facts = self.tier3.extract(clean) if should_run_tier3 else []
        if llm_facts:
            return RoutingResult(
                tier="tier_3_small_llm",
                facts=llm_facts,
                reasoning="Tier 3 fallback used only after Tier 1/2 misses with explicit update/delete cues.",
            )

        # Final fallback keeps deterministic behavior and avoids hard failures.
        return RoutingResult(
            tier="fallback_generic",
            facts=[
                ExtractionFact(
                    fact=f"fact.raw={clean}",
                    confidence=0.65,
                    category="GENERIC",
                    source_tier="fallback_generic",
                )
            ],
            reasoning="No structured match; preserving raw fact.",
        )
