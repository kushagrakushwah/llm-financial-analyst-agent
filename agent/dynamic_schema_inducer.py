"""
dynamic_schema_inducer.py
=========================
Tier 2 AI Dynamic Schema Induction Engine.
When unclassified or novel clauses fail standard fixed taxonomy classification,
this engine dynamically induces bespoke parent and child schemas and maps them
to standard canonical database types, closing the loop on unclassified contract text.
"""

import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from agent.schemas import ParentClause, ChildEntity


@dataclass
class InducedSchema:
    parent_label: str
    canonical_parent: str
    child_labels: List[str]
    canonical_child_mapping: Dict[str, str]


# Canonical taxonomy normalizers for database consistency
CANONICAL_PARENT_CATEGORIES = [
    "liability_clause",
    "penalty_clause",
    "termination_clause",
    "payment_terms_clause",
    "governing_law_clause",
    "indemnity_clause",
    "confidentiality_clause",
    "insurance_clause",
    "compliance_clause",
]


class DynamicSchemaInducer:
    """Induces tailored schemas for unclassified legal clauses and resolves exact spans."""

    def __init__(self, llm_pipeline=None):
        self.llm_pipeline = llm_pipeline

    def induce_schema(self, clause_text: str) -> Optional[InducedSchema]:
        """
        Inspects unclassified clause text and induces:
        1. Specific parent label (e.g. ip_infringement_indemnity)
        2. Canonical parent category (e.g. indemnity_clause)
        3. Specific child labels (e.g. [defense_counsel_obligation, liability_reimbursement])
        4. Canonical child mappings (e.g. {"defense_counsel_obligation": "remedy_obligation"})
        """
        text_clean = clause_text.strip()
        if not (25 <= len(text_clean) <= 1200):
            return None

        # 1. Attempt LLM Induction if active LLM pipeline is available
        if self.llm_pipeline is not None:
            try:
                schema = self._induce_via_llm(text_clean)
                if schema:
                    return schema
            except Exception as e:
                print(f"[DynamicSchemaInducer] LLM induction failed ({e}). Falling back to semantic induction.")

        # 2. Semantic Expert Induction (deterministic, sub-millisecond, zero-cost)
        return self._induce_via_semantic_rules(text_clean)

    def _induce_via_llm(self, clause_text: str) -> Optional[InducedSchema]:
        prompt = (
            "You are an expert legal schema engineer. Inspect this unclassified contract clause:\n"
            f"\"{clause_text}\"\n\n"
            "Return a strictly valid JSON object with:\n"
            "- \"parent_label\": concise snake_case noun phrase (e.g. maritime_demurrage_clause)\n"
            "- \"canonical_parent\": one of [liability_clause, penalty_clause, termination_clause, payment_terms_clause, governing_law_clause, indemnity_clause, confidentiality_clause, insurance_clause]\n"
            "- \"child_labels\": list of up to 4 concise snake_case entity names to extract within the clause\n"
            "- \"canonical_child_mapping\": dict mapping each child label to standard category (e.g. penalty_rate, monetary_amount, grace_period, contracting_party)\n"
            "JSON:"
        )

        res = self.llm_pipeline(prompt, max_new_tokens=180, do_sample=False)
        generated_text = res[0]["generated_text"]
        # Extract JSON substring
        json_match = re.search(r"\{.*\}", generated_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return InducedSchema(
                parent_label=data.get("parent_label", "custom_legal_clause"),
                canonical_parent=data.get("canonical_parent", "liability_clause"),
                child_labels=data.get("child_labels", []),
                canonical_child_mapping=data.get("canonical_child_mapping", {})
            )
        return None

    def _induce_via_semantic_rules(self, text: str) -> Optional[InducedSchema]:
        t_low = text.lower()

        # Patent / IP / Indemnification
        if any(w in t_low for w in ["indemnif", "hold harmless", "defend and hold", "infringement", "patent", "copyright"]):
            return InducedSchema(
                parent_label="ip_infringement_indemnity_clause",
                canonical_parent="indemnity_clause",
                child_labels=["indemnifying_party", "indemnified_scope", "defense_obligation", "monetary_cap"],
                canonical_child_mapping={
                    "indemnifying_party": "contracting_party",
                    "indemnified_scope": "penalty_condition",
                    "defense_obligation": "remedy_obligation",
                    "monetary_cap": "liability_cap",
                }
            )

        # Maritime / Shipping / Demurrage / Delay
        if any(w in t_low for w in ["demurrage", "laytime", "berth", "vessel", "charterer", "despatch"]):
            return InducedSchema(
                parent_label="demurrage_laytime_delay_clause",
                canonical_parent="penalty_clause",
                child_labels=["demurrage_rate", "laytime_allowance", "weather_condition", "charterer_party"],
                canonical_child_mapping={
                    "demurrage_rate": "penalty_rate",
                    "laytime_allowance": "grace_period",
                    "weather_condition": "penalty_condition",
                    "charterer_party": "contracting_party",
                }
            )

        # Escrow / Deposit / Security
        if any(w in t_low for w in ["escrow", "security deposit", "retainer", "collateral", "pledge"]):
            return InducedSchema(
                parent_label="collateral_escrow_deposit_clause",
                canonical_parent="payment_terms_clause",
                child_labels=["deposit_amount", "escrow_agent", "release_condition", "holding_period"],
                canonical_child_mapping={
                    "deposit_amount": "monetary_amount",
                    "escrow_agent": "contracting_party",
                    "release_condition": "penalty_condition",
                    "holding_period": "grace_period",
                }
            )

        # Insurance / Coverage
        if any(w in t_low for w in ["insurance", "commercial general liability", "worker's compensation", "umbrella policy"]):
            return InducedSchema(
                parent_label="insurance_coverage_clause",
                canonical_parent="insurance_clause",
                child_labels=["coverage_minimum", "insurance_type", "policy_term", "insured_party"],
                canonical_child_mapping={
                    "coverage_minimum": "monetary_amount",
                    "insurance_type": "penalty_condition",
                    "policy_term": "grace_period",
                    "insured_party": "contracting_party",
                }
            )

        # Confidentiality / Non-Disclosure
        if any(w in t_low for w in ["confidential", "proprietary information", "trade secret", "non-disclosure"]):
            return InducedSchema(
                parent_label="confidentiality_protection_clause",
                canonical_parent="confidentiality_clause",
                child_labels=["survival_duration", "disclosing_party", "receiving_party", "confidential_scope"],
                canonical_child_mapping={
                    "survival_duration": "grace_period",
                    "disclosing_party": "contracting_party",
                    "receiving_party": "contracting_party",
                    "confidential_scope": "penalty_condition",
                }
            )

        # Fallback generic financial / legal clause
        return InducedSchema(
            parent_label="bespoke_commercial_obligation_clause",
            canonical_parent="liability_clause",
            child_labels=["monetary_figure", "percentage_rate", "contracting_entity", "time_period"],
            canonical_child_mapping={
                "monetary_figure": "monetary_amount",
                "percentage_rate": "penalty_rate",
                "contracting_entity": "contracting_party",
                "time_period": "grace_period",
            }
        )

    def resolve_unclassified_clause(
        self,
        unclassified: ParentClause,
        gliner_model: Any,
        threshold: float = 0.35,
    ) -> ParentClause:
        """
        Dynamically extracts fine-grained child entities using the induced schema,
        attaching canonical categories and global coordinate provenance.
        """
        text = unclassified.clause_text
        schema = self.induce_schema(text)
        if not schema:
            return unclassified

        children: List[ChildEntity] = []
        if schema.child_labels and hasattr(gliner_model, "predict_entities"):
            preds = gliner_model.predict_entities(text, schema.child_labels, threshold=threshold, flat_ner=True)
            for p in preds:
                local_s = int(p["start"])
                local_e = int(p["end"])
                global_s = unclassified.global_start + local_s
                global_e = unclassified.global_start + local_e
                canon_type = schema.canonical_child_mapping.get(p["label"], "custom_entity")

                children.append(
                    ChildEntity(
                        text=p["text"],
                        label=p["label"],
                        score=float(p["score"]),
                        canonical_type=canon_type,
                        local_start=local_s,
                        local_end=local_e,
                        global_start=global_s,
                        global_end=global_e,
                    )
                )

        return ParentClause(
            parent_label=schema.parent_label,
            canonical_parent=schema.canonical_parent,
            clause_text=text,
            score=unclassified.score,
            global_start=unclassified.global_start,
            global_end=unclassified.global_end,
            is_unclassified=False,
            is_dynamically_induced=True,
            children=children,
        )
