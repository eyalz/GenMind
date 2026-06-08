from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any

from src.db import get_pool
from src.models.schemas import (
    AUDNAction,
    AUDNDecision,
    MemoryRecord,
    MutationType,
    SessionTurnPayload,
    TenantContext,
    TemporalScope,
    TemporalType,
)
from src.services.memory_engine import MemoryEngine
from src.services.memory_router import MemoryRouter, RoutingResult


class AUDNPipeline:
    """
    Add/Update/Delete/None (AUDN) decision engine.

    This service now uses a three-tier extraction router:
    - Tier 1 deterministic patterns (free)
    - Tier 2 local NER + regex extraction (free / low latency)
    - Tier 3 small-LLM fallback for complex relationships
    """

    NOISE_PATTERNS = [
        r"\bok\b",
        r"\bthanks\b",
        r"\bhello\b",
        r"\bhi\b",
        r"\bgot it\b",
        r"\[simulator_ephemeral\]",
        r"pending customer-agent answer generation",
    ]

    DELETE_CUES = [
        "forget that",
        "ignore previous",
        "that is wrong",
        "no longer",
        "remove this",
        "not my",
        "is not",
        "isn't",
    ]

    UPDATE_CUES = [
        "actually",
        "correction",
        "updated",
        "changed",
        "instead",
    ]

    PROFILE_PREFIX = "profile."

    ABSOLUTE_OVERWRITE_KEYS = {
        "age",
        "current_location",
        "search_location",
        "current_version",
        "children_count",
        "marital_status",
    }

    AGE_PATTERNS = [
        re.compile(r"\b(?:i am|i'm|im)\s+(?:actually\s+)?(\d{1,3})\b", re.IGNORECASE),
        re.compile(r"\b(\d{1,3})\s*(?:years old|yrs old|yo)\b", re.IGNORECASE),
        re.compile(r"\bage\s*(?:is|=)?\s*(\d{1,3})\b", re.IGNORECASE),
    ]

    KIDS_PATTERN = re.compile(r"\b(?:have|with)\s+(\d{1,2})\s+(?:kids|children)\b", re.IGNORECASE)
    CURRENT_LOCATION_PATTERN = re.compile(
        r"\b(?:i am|i'm|im)\s+(?:now\s+|currently\s+)?in\s+([a-zA-Z][a-zA-Z\s'-]{1,60}?)(?=\s+(?:and|but|while|because|with|where)\b|[\.!?,;]|$)",
        re.IGNORECASE,
    )
    PROJECT_PATTERN = re.compile(r"\bproject\s+([a-zA-Z][a-zA-Z0-9_\-]{1,64})\b", re.IGNORECASE)
    RELATIONSHIP_PATTERN = re.compile(
        r"\b([a-zA-Z][a-zA-Z\s'-]{1,40})\s+is\s+my\s+([a-zA-Z_\s-]{2,30})\b",
        re.IGNORECASE,
    )
    MY_RELATIONSHIP_PATTERN = re.compile(
        r"\bmy\s+([a-zA-Z_\s-]{2,30})\s+is\s+([a-zA-Z][a-zA-Z\s'-]{1,40})\b",
        re.IGNORECASE,
    )
    TEMPORARY_CUES = [
        "for now",
        "currently",
        "this week",
        "this month",
        "for the summer",
        "temporary",
        "until",
        "next week",
    ]

    def __init__(self, memory_engine: MemoryEngine) -> None:
        self.memory_engine = memory_engine
        self.memory_router = MemoryRouter()

    async def process_turn(self, payload: SessionTurnPayload) -> list[AUDNDecision]:
        """Compute AUDN actions for one conversational turn."""
        tenant = TenantContext(
            customer_id=payload.customer_id,
            workspace_id=payload.workspace_id,
            end_user_id=payload.end_user_id,
            session_id=payload.session_id,
        )

        existing = await self._load_existing_memories(
            tenant,
            maker_id=payload.maker_id,
            agent_id=payload.agent_id,
        )
        candidates = self._extract_candidate_facts(payload)

        decisions: list[AUDNDecision] = []
        for candidate in candidates:
            decisions.extend(
                self._decide_actions(
                    tenant,
                    payload.maker_id,
                    payload.agent_id,
                    candidate,
                    existing,
                )
            )

        if not decisions:
            fallback_candidate = self._fallback_candidate_fact(payload)
            if fallback_candidate == "(empty)":
                return []
            decisions.append(
                self._make_decision(
                    tenant=tenant,
                    maker_id=payload.maker_id,
                    agent_id=payload.agent_id,
                    action=AUDNAction.NONE,
                    reason="No high-signal factual candidates extracted from turn.",
                    candidate_fact=fallback_candidate,
                    confidence=0.99,
                )
            )

        return decisions

    async def process_and_commit(self, payload: SessionTurnPayload) -> list[AUDNDecision]:
        decisions = await self.process_turn(payload)
        await self.apply_decisions(decisions)
        return decisions

    async def apply_decisions(self, decisions: list[AUDNDecision]) -> None:
        """Apply AUDN decisions to persistent memory storage."""
        for decision in decisions:
            if decision.action == AUDNAction.NONE:
                if decision.candidate_fact.strip().lower() in {"", "(empty)"}:
                    continue
                await self._persist_audit_row(decision)
                continue

            if decision.action == AUDNAction.DELETE and decision.target_memory_id:
                await self.memory_engine.deactivate_memory_record(decision.target_memory_id)
                await self._persist_audit_row(decision)
                continue

            if decision.action in {AUDNAction.ADD, AUDNAction.UPDATE}:
                stored_content = self._canonicalize_storage_content(self._build_storage_payload(decision))
                record = MemoryRecord(
                    memory_id=decision.target_memory_id or self._new_memory_id(decision),
                    tenant=decision.tenant,
                    maker_id=decision.maker_id,
                    agent_id=decision.agent_id,
                    content=stored_content,
                    confidence=decision.confidence,
                )
                await self.memory_engine.persist_memory_record(record)

                parsed_property = self._parse_profile_memory(record.content)
                if parsed_property is not None:
                    property_key, _ = parsed_property
                    await self._deactivate_other_property_memories(
                        decision,
                        keep_memory_id=record.memory_id,
                        property_key=property_key,
                    )

            await self._persist_audit_row(decision)

    def serialize_uome_mutations(self, decisions: list[AUDNDecision]) -> list[dict[str, Any]]:
        """Return strict, protocol-friendly UOME mutation JSON objects."""
        serialized: list[dict[str, Any]] = []
        for decision in decisions:
            serialized.append(
                {
                    "action": decision.action.value.upper() if decision.action != AUDNAction.NONE else "NO-OP",
                    "mutation_type": decision.mutation_type,
                    "source_entity": decision.source_entity,
                    "target_property_or_entity": decision.target_property_or_entity,
                    "value": decision.value,
                    "previous_value_reference": decision.previous_value_reference,
                    "temporal_scope": {
                        "type": decision.temporal_scope.type,
                        "valid_from": decision.temporal_scope.valid_from,
                        "valid_until": decision.temporal_scope.valid_until,
                    },
                    "reasoning_justification": decision.reasoning_justification or decision.reason,
                }
            )
        return serialized

    async def _deactivate_other_property_memories(
        self,
        decision: AUDNDecision,
        *,
        keep_memory_id: str,
        property_key: str,
    ) -> None:
        existing = await self.memory_engine.list_active_memories(
            decision.tenant,
            maker_id=decision.maker_id,
            agent_id=decision.agent_id,
        )
        for row in existing:
            if row.memory_id == keep_memory_id:
                continue
            parsed = self._parse_profile_memory(row.content)
            if parsed is None:
                continue
            existing_key, _ = parsed
            if existing_key != property_key:
                continue
            await self.memory_engine.deactivate_memory_record(row.memory_id)

    async def _persist_audit_row(self, decision: AUDNDecision) -> None:
        pool = get_pool()
        query = """
        INSERT INTO audn_audit_log (
            customer_id,
            workspace_id,
            end_user_id,
            session_id,
            maker_id,
            agent_id,
            action,
            mutation_type,
            source_entity,
            target_property_or_entity,
            value_json,
            previous_value_reference,
            temporal_type,
            valid_from,
            valid_until,
            reasoning_justification,
            candidate_fact,
            target_memory_id,
            confidence,
            reason,
            created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12::jsonb, $13, $14, $15, $16, $17, $18, $19, $20, $21)
        """
        async with pool.acquire() as conn:
            await conn.execute(
                query,
                decision.tenant.customer_id,
                decision.tenant.workspace_id,
                decision.tenant.end_user_id,
                decision.tenant.session_id,
                decision.maker_id,
                decision.agent_id,
                decision.action.value,
                decision.mutation_type.value,
                decision.source_entity,
                decision.target_property_or_entity,
                json.dumps(decision.value),
                json.dumps(decision.previous_value_reference),
                decision.temporal_scope.type.value,
                decision.temporal_scope.valid_from,
                decision.temporal_scope.valid_until,
                decision.reasoning_justification,
                decision.candidate_fact,
                decision.target_memory_id,
                decision.confidence,
                decision.reason,
                decision.created_at,
            )

    async def _load_existing_memories(
        self,
        tenant: TenantContext,
        *,
        maker_id: str,
        agent_id: str,
    ) -> list[MemoryRecord]:
        return await self.memory_engine.list_active_memories(
            tenant,
            maker_id=maker_id,
            agent_id=agent_id,
        )

    def _extract_candidate_facts(self, payload: SessionTurnPayload) -> list[str]:
        user_text = payload.user_input.strip()
        model_text = payload.model_output.strip()
        combined = f"{user_text}\n{model_text}".strip()

        candidates: list[str] = []

        # Tiered extraction router over user input (most stable source for memory facts).
        routed: RoutingResult = self.memory_router.route(user_text)
        for fact in routed.facts:
            if fact.fact and not self._is_noise(fact.fact):
                candidates.append(fact.fact)

        # Preserve existing profile extraction patterns for strong backward compatibility.
        for line in [line.strip() for line in re.split(r"[\n\.]+", combined) if line.strip()]:
            if len(line) < 5 or self._is_noise(line):
                continue
            for key, value_payload in self._extract_profile_properties(line).items():
                candidates.append(self._canonical_property_fact(key, value_payload["value"], value_payload["temporal_scope"]))

        # Keep original user fact as context when no structured extraction succeeded.
        if not candidates and user_text:
            candidates.append(user_text)

        # Dedup while preserving order.
        unique: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            key = item.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)

        return unique[:16]

    def _fallback_candidate_fact(self, payload: SessionTurnPayload) -> str:
        user_text = payload.user_input.strip()
        if user_text:
            return user_text[:240]
        model_text = payload.model_output.strip()
        if model_text and not self._is_noise(model_text):
            return model_text[:240]
        return "(empty)"

    def _decide_actions(
        self,
        tenant: TenantContext,
        maker_id: str,
        agent_id: str,
        candidate_fact: str,
        existing: list[MemoryRecord],
    ) -> list[AUDNDecision]:
        normalized = candidate_fact.lower()

        if self._is_noise(candidate_fact):
            return [
                self._make_decision(
                    tenant=tenant,
                    maker_id=maker_id,
                    agent_id=agent_id,
                    action=AUDNAction.NONE,
                    reason="Candidate is low-signal conversational filler.",
                    candidate_fact=candidate_fact,
                    confidence=0.97,
                )
            ]

        parsed_profile = self._parse_profile_memory(candidate_fact)
        if parsed_profile is not None:
            return [self._decide_profile_property(tenant, maker_id, agent_id, candidate_fact, existing)]

        return [
            self._decide_general_action(
                tenant=tenant,
                maker_id=maker_id,
                agent_id=agent_id,
                candidate_fact=candidate_fact,
                existing=existing,
            )
        ]

    def _decide_profile_property(
        self,
        tenant: TenantContext,
        maker_id: str,
        agent_id: str,
        candidate_fact: str,
        existing: list[MemoryRecord],
    ) -> AUDNDecision:
        parsed = self._parse_profile_memory(candidate_fact)
        if parsed is None:
            return self._decide_general_action(tenant, maker_id, agent_id, candidate_fact, existing)

        property_key, value = parsed
        temporal_scope = self._temporal_scope_from_hint(
            "temporary" if "|scope=temporary" in candidate_fact else "",
            candidate_fact,
        )

        target = self._best_property_memory(existing, property_key)
        if target is None:
            return self._make_decision(
                tenant=tenant,
                maker_id=maker_id,
                agent_id=agent_id,
                action=AUDNAction.ADD,
                reason=f"New profile property '{property_key}' with no prior active value.",
                candidate_fact=candidate_fact,
                confidence=0.95,
                mutation_type=MutationType.PROPERTY_MODIFICATION,
                source_entity="User_Profile",
                target_property_or_entity=property_key,
                value=value,
                temporal_scope=temporal_scope,
            )

        _, existing_value = self._parse_profile_memory(target.content) or ("", "")
        if existing_value == value:
            return self._make_decision(
                tenant=tenant,
                maker_id=maker_id,
                agent_id=agent_id,
                action=AUDNAction.NONE,
                reason=f"Profile property '{property_key}' already matches stored value.",
                candidate_fact=candidate_fact,
                confidence=0.97,
                target_memory_id=target.memory_id,
                mutation_type=MutationType.PROPERTY_MODIFICATION,
                source_entity="User_Profile",
                target_property_or_entity=property_key,
                value=value,
                previous_value_reference=existing_value,
                temporal_scope=temporal_scope,
            )

        lowered = candidate_fact.lower()
        if any(cue in lowered for cue in self.DELETE_CUES):
            return self._make_decision(
                tenant=tenant,
                maker_id=maker_id,
                agent_id=agent_id,
                action=AUDNAction.DELETE,
                reason=f"Explicit correction indicates removing stale '{property_key}' property.",
                candidate_fact=candidate_fact,
                confidence=0.93,
                target_memory_id=target.memory_id,
                mutation_type=MutationType.PROPERTY_MODIFICATION,
                source_entity="User_Profile",
                target_property_or_entity=property_key,
                value=value,
                previous_value_reference=existing_value,
                temporal_scope=TemporalScope(
                    type=TemporalType.HISTORICAL_ARCHIVE,
                    valid_from="current_interaction",
                    valid_until="indefinite",
                ),
            )

        # Absolute overwrite for mutually exclusive keys; otherwise keep as timeline append (ADD).
        if property_key in self.ABSOLUTE_OVERWRITE_KEYS:
            return self._make_decision(
                tenant=tenant,
                maker_id=maker_id,
                agent_id=agent_id,
                action=AUDNAction.UPDATE,
                reason=f"Profile property '{property_key}' changed; old value archived through previous_value_reference.",
                candidate_fact=candidate_fact,
                confidence=0.95,
                target_memory_id=target.memory_id,
                mutation_type=MutationType.PROPERTY_MODIFICATION,
                source_entity="User_Profile",
                target_property_or_entity=property_key,
                value=value,
                previous_value_reference=existing_value,
                temporal_scope=temporal_scope,
            )

        return self._make_decision(
            tenant=tenant,
            maker_id=maker_id,
            agent_id=agent_id,
            action=AUDNAction.ADD,
            reason=f"Property '{property_key}' appended as non-exclusive contextual memory.",
            candidate_fact=candidate_fact,
            confidence=0.84,
            mutation_type=MutationType.PROPERTY_MODIFICATION,
            source_entity="User_Profile",
            target_property_or_entity=property_key,
            value=value,
            previous_value_reference=existing_value,
            temporal_scope=temporal_scope,
        )

    def _decide_general_action(
        self,
        tenant: TenantContext,
        maker_id: str,
        agent_id: str,
        candidate_fact: str,
        existing: list[MemoryRecord],
    ) -> AUDNDecision:
        normalized = candidate_fact.lower()

        target = self._best_match(candidate_fact, existing)
        inferred = self._infer_uome_shape(candidate_fact)
        if target is None:
            return self._make_decision(
                tenant=tenant,
                maker_id=maker_id,
                agent_id=agent_id,
                action=AUDNAction.ADD,
                reason="New high-signal fact not present in tenant memory.",
                candidate_fact=candidate_fact,
                confidence=0.88,
                mutation_type=inferred["mutation_type"],
                source_entity=inferred["source_entity"],
                target_property_or_entity=inferred["target_property_or_entity"],
                value=inferred["value"],
                temporal_scope=inferred["temporal_scope"],
            )

        if any(cue in normalized for cue in self.DELETE_CUES):
            return self._make_decision(
                tenant=tenant,
                maker_id=maker_id,
                agent_id=agent_id,
                action=AUDNAction.DELETE,
                reason="Explicit user correction indicates stale memory invalidation.",
                candidate_fact=candidate_fact,
                confidence=0.92,
                target_memory_id=target.memory_id,
                mutation_type=inferred["mutation_type"],
                source_entity=inferred["source_entity"],
                target_property_or_entity=inferred["target_property_or_entity"],
                value=inferred["value"],
                previous_value_reference=target.content,
                temporal_scope=TemporalScope(
                    type=TemporalType.HISTORICAL_ARCHIVE,
                    valid_from="current_interaction",
                    valid_until="indefinite",
                ),
            )

        if any(cue in normalized for cue in self.UPDATE_CUES):
            return self._make_decision(
                tenant=tenant,
                maker_id=maker_id,
                agent_id=agent_id,
                action=AUDNAction.UPDATE,
                reason="Correction cue found; existing memory should be amended.",
                candidate_fact=candidate_fact,
                confidence=0.84,
                target_memory_id=target.memory_id,
                mutation_type=inferred["mutation_type"],
                source_entity=inferred["source_entity"],
                target_property_or_entity=inferred["target_property_or_entity"],
                value=inferred["value"],
                previous_value_reference=target.content,
                temporal_scope=inferred["temporal_scope"],
            )

        if self._is_near_duplicate(candidate_fact, target.content):
            return self._make_decision(
                tenant=tenant,
                maker_id=maker_id,
                agent_id=agent_id,
                action=AUDNAction.NONE,
                reason="Fact is a near-duplicate of existing active memory.",
                candidate_fact=candidate_fact,
                confidence=0.9,
                target_memory_id=target.memory_id,
                mutation_type=inferred["mutation_type"],
                source_entity=inferred["source_entity"],
                target_property_or_entity=inferred["target_property_or_entity"],
                value=inferred["value"],
                previous_value_reference=target.content,
                temporal_scope=inferred["temporal_scope"],
            )

        return self._make_decision(
            tenant=tenant,
            maker_id=maker_id,
            agent_id=agent_id,
            action=AUDNAction.ADD,
            reason="New related fact is appended to preserve timeline context instead of overwriting prior memory.",
            candidate_fact=candidate_fact,
            confidence=0.78,
            mutation_type=inferred["mutation_type"],
            source_entity=inferred["source_entity"],
            target_property_or_entity=inferred["target_property_or_entity"],
            value=inferred["value"],
            temporal_scope=inferred["temporal_scope"],
        )

    def _make_decision(
        self,
        *,
        tenant: TenantContext,
        maker_id: str,
        agent_id: str,
        action: AUDNAction,
        reason: str,
        candidate_fact: str,
        confidence: float,
        target_memory_id: str | None = None,
        mutation_type: MutationType | None = None,
        source_entity: str | None = None,
        target_property_or_entity: str | None = None,
        value: Any = "",
        previous_value_reference: Any | None = None,
        temporal_scope: TemporalScope | None = None,
    ) -> AUDNDecision:
        inferred = self._infer_uome_shape(candidate_fact)
        chosen_mutation_type = mutation_type or inferred["mutation_type"]
        chosen_source = source_entity or inferred["source_entity"]
        chosen_target = target_property_or_entity or inferred["target_property_or_entity"]
        chosen_temporal = temporal_scope or inferred["temporal_scope"]
        chosen_value = inferred["value"] if value == "" else value

        return AUDNDecision(
            tenant=tenant,
            maker_id=maker_id,
            agent_id=agent_id,
            action=action,
            mutation_type=chosen_mutation_type,
            source_entity=chosen_source,
            target_property_or_entity=chosen_target,
            value=chosen_value,
            previous_value_reference=previous_value_reference,
            temporal_scope=chosen_temporal,
            reasoning_justification=reason,
            reason=reason,
            candidate_fact=candidate_fact,
            confidence=confidence,
            target_memory_id=target_memory_id,
        )

    def _infer_uome_shape(self, candidate_fact: str) -> dict[str, Any]:
        fact = candidate_fact.strip()

        parsed_profile = self._parse_profile_memory(fact)
        if parsed_profile is not None:
            key, value = parsed_profile
            temporal = TemporalScope(
                type=TemporalType.TEMPORARY if "|scope=temporary" in fact else TemporalType.PERMANENT,
                valid_from="current_interaction",
                valid_until="conditional_trigger" if "|scope=temporary" in fact else "indefinite",
            )
            return {
                "mutation_type": MutationType.PROPERTY_MODIFICATION,
                "source_entity": "User_Profile",
                "target_property_or_entity": key,
                "value": value,
                "temporal_scope": temporal,
            }

        if fact.startswith("system.") and "=" in fact:
            key, value = fact.split("=", 1)
            return {
                "mutation_type": MutationType.PROPERTY_MODIFICATION,
                "source_entity": "System_Context",
                "target_property_or_entity": key.replace("system.", "", 1),
                "value": value.strip(),
                "temporal_scope": TemporalScope(
                    type=TemporalType.PERMANENT,
                    valid_from="current_interaction",
                    valid_until="indefinite",
                ),
            }

        if fact.startswith("entity.") and "=" in fact:
            key, value = fact.split("=", 1)
            entity_type = key.replace("entity.", "", 1)
            return {
                "mutation_type": MutationType.ENTITY_CREATION,
                "source_entity": entity_type.title(),
                "target_property_or_entity": "entity",
                "value": {"name": value.strip()},
                "temporal_scope": TemporalScope(
                    type=TemporalType.PERMANENT,
                    valid_from="current_interaction",
                    valid_until="indefinite",
                ),
            }

        rel_match = self.RELATIONSHIP_PATTERN.search(fact)
        if rel_match:
            target_entity = self._normalize_symbol(rel_match.group(1))
            relation = rel_match.group(2).strip().lower().replace(" ", "_")
            return {
                "mutation_type": MutationType.RELATIONSHIP_EDGE_CHANGE,
                "source_entity": "User_Profile",
                "target_property_or_entity": target_entity,
                "value": relation,
                "temporal_scope": self._temporal_scope_from_hint("", fact),
            }

        my_rel_match = self.MY_RELATIONSHIP_PATTERN.search(fact)
        if my_rel_match:
            relation = my_rel_match.group(1).strip().lower().replace(" ", "_")
            target_entity = self._normalize_symbol(my_rel_match.group(2))
            return {
                "mutation_type": MutationType.RELATIONSHIP_EDGE_CHANGE,
                "source_entity": "User_Profile",
                "target_property_or_entity": target_entity,
                "value": relation,
                "temporal_scope": self._temporal_scope_from_hint("", fact),
            }

        project_match = self.PROJECT_PATTERN.search(fact)
        if project_match:
            entity_name = self._normalize_symbol(project_match.group(1))
            return {
                "mutation_type": MutationType.ENTITY_CREATION,
                "source_entity": entity_name,
                "target_property_or_entity": "entity",
                "value": {"name": project_match.group(1).strip()},
                "temporal_scope": TemporalScope(
                    type=TemporalType.PERMANENT,
                    valid_from="current_interaction",
                    valid_until="indefinite",
                ),
            }

        return {
            "mutation_type": MutationType.PROPERTY_MODIFICATION,
            "source_entity": "Conversation_Context",
            "target_property_or_entity": "fact",
            "value": fact,
            "temporal_scope": self._temporal_scope_from_hint("", fact),
        }

    def _temporal_scope_from_hint(self, temporal_scope: str, text: str) -> TemporalScope:
        normalized_hint = (temporal_scope or "").strip().lower()
        lowered = text.lower()
        is_temporary = normalized_hint == "temporary" or any(cue in lowered for cue in self.TEMPORARY_CUES)
        if is_temporary:
            return TemporalScope(
                type=TemporalType.TEMPORARY,
                valid_from="current_interaction",
                valid_until="conditional_trigger",
            )
        return TemporalScope(
            type=TemporalType.PERMANENT,
            valid_from="current_interaction",
            valid_until="indefinite",
        )

    def _extract_profile_properties(self, text: str) -> dict[str, dict[str, str]]:
        normalized = text.strip().lower()
        props: dict[str, dict[str, str]] = {}

        age = self._extract_age(text)
        if age is not None:
            props["age"] = {"value": str(age), "temporal_scope": "permanent"}

        if re.search(r"\bmarried\b", normalized):
            props["marital_status"] = {"value": "married", "temporal_scope": "permanent"}
        elif re.search(r"\bsingle\b", normalized):
            props["marital_status"] = {"value": "single", "temporal_scope": "permanent"}

        kids_match = self.KIDS_PATTERN.search(normalized)
        if kids_match:
            props["children_count"] = {"value": kids_match.group(1), "temporal_scope": "permanent"}

        location_match = self.CURRENT_LOCATION_PATTERN.search(text)
        if location_match:
            location = " ".join(location_match.group(1).split()).strip(" .,!?:;").title()
            if location:
                props["current_location"] = {"value": location, "temporal_scope": "temporary"}

        return props

    def _canonical_property_fact(self, property_key: str, value: str, temporal_scope: str) -> str:
        if temporal_scope == "temporary":
            return f"{self.PROFILE_PREFIX}{property_key}={value}|scope=temporary"
        return f"{self.PROFILE_PREFIX}{property_key}={value}"

    def _parse_profile_memory(self, content: str) -> tuple[str, str] | None:
        if not content.startswith(self.PROFILE_PREFIX) or "=" not in content:
            return None
        raw_key, raw_value = content[len(self.PROFILE_PREFIX) :].split("=", 1)
        key = raw_key.strip().lower()
        value = raw_value.split("|", 1)[0].strip()
        if not key or not value:
            return None
        return key, value

    def _best_property_memory(self, existing: list[MemoryRecord], property_key: str) -> MemoryRecord | None:
        for row in existing:
            parsed = self._parse_profile_memory(row.content)
            if parsed is None:
                continue
            key, _ = parsed
            if key == property_key:
                return row
        return None

    def _best_match(self, candidate_fact: str, existing: list[MemoryRecord]) -> MemoryRecord | None:
        if not existing:
            return None

        candidate_words = set(candidate_fact.lower().split())
        best_score = 0.0
        best_row: MemoryRecord | None = None

        for row in existing:
            row_words = set(row.content.lower().split())
            overlap = len(candidate_words.intersection(row_words))
            union = max(len(candidate_words.union(row_words)), 1)
            score = overlap / union
            if score > best_score:
                best_score = score
                best_row = row

        if best_score < 0.2:
            return None
        return best_row

    def _is_near_duplicate(self, a: str, b: str) -> bool:
        a_words = set(a.lower().split())
        b_words = set(b.lower().split())
        overlap = len(a_words.intersection(b_words))
        union = max(len(a_words.union(b_words)), 1)
        return (overlap / union) >= 0.8

    def _is_noise(self, text: str) -> bool:
        lowered = text.lower().strip()
        return any(re.search(pattern, lowered) for pattern in self.NOISE_PATTERNS)

    def _extract_age(self, text: str) -> int | None:
        lowered = text.lower()
        for pattern in self.AGE_PATTERNS:
            match = pattern.search(lowered)
            if not match:
                continue
            try:
                age = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if 1 <= age <= 120:
                return age
        return None

    def _normalize_symbol(self, text: str) -> str:
        normalized = "_".join(text.strip().split()).strip("_")
        normalized = re.sub(r"[^a-zA-Z0-9_]", "", normalized)
        return normalized or "unknown"

    def _new_memory_id(self, decision: AUDNDecision) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        return f"mem_{decision.tenant.session_id}_{ts}"

    def _build_storage_payload(self, decision: AUDNDecision) -> str:
        # Keep profile properties stable so update/deactivate logic remains deterministic.
        if decision.candidate_fact.startswith(self.PROFILE_PREFIX):
            return decision.candidate_fact.strip()

        value = decision.value
        if isinstance(value, dict):
            compact_value = ",".join(
                f"{self._normalize_symbol(str(k)).lower()}:{self._normalize_symbol(str(v)).lower()}"
                for k, v in value.items()
            )
        elif isinstance(value, list):
            compact_value = ",".join(self._normalize_symbol(str(item)).lower() for item in value)
        else:
            compact_value = self._normalize_symbol(str(value)).lower()

        temporal = decision.temporal_scope
        return (
            "uome|v=1"
            f"|mt={decision.mutation_type.value}"
            f"|src={self._normalize_symbol(decision.source_entity).lower()}"
            f"|tgt={self._normalize_symbol(decision.target_property_or_entity).lower()}"
            f"|val={compact_value}"
            f"|tt={temporal.type.value}"
            f"|vf={self._normalize_symbol(temporal.valid_from).lower()}"
            f"|vu={self._normalize_symbol(temporal.valid_until).lower()}"
        )

    def _canonicalize_storage_content(self, content: str) -> str:
        # Keep profile properties as canonical key-value facts for deterministic updates.
        if content.startswith(self.PROFILE_PREFIX) or content.startswith("uome|v=1"):
            return content.strip()

        normalized = " ".join(content.lower().split())
        normalized = re.sub(r"[^a-z0-9\s._=|:-]", "", normalized)
        normalized = normalized.strip()
        if len(normalized) > 220:
            normalized = normalized[:220].rstrip()

        terms = [
            term
            for term in re.findall(r"[a-z0-9]{3,}", normalized)
            if term not in {"the", "and", "for", "with", "from", "that"}
        ]
        tags = ",".join(terms[:6])
        digest = sha1(normalized.encode("utf-8")).hexdigest()[:10]
        return f"fact|v=1|id={digest}|tags={tags}|c={normalized}"
