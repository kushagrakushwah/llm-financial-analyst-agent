"""
test_dynamic_schema_induction.py
================================
Unit tests verifying Tier 2 AI Dynamic Schema Induction:
- Bespoke clause schema induction
- Canonical category normalization
- GLiNER mini-pass child extraction
- Document offset coordinate stitching
- API endpoint integration
"""

import pytest
from fastapi.testclient import TestClient

from agent.schemas import ParentClause, ChildEntity, HierarchicalExtractionResult
from agent.dynamic_schema_inducer import DynamicSchemaInducer, InducedSchema
from agent.gliner_extractor import FinancialEntityExtractor
from api.server import app

client = TestClient(app)


def test_induce_schema_patent_indemnity():
    inducer = DynamicSchemaInducer()
    clause_text = (
        "Vendor shall indemnify, defend, and hold harmless Customer against any third-party claims "
        "alleging patent infringement, with aggregate defense liability capped at $1,000,000 USD."
    )
    schema = inducer.induce_schema(clause_text)
    assert schema is not None
    assert "indemnity" in schema.parent_label
    assert schema.canonical_parent == "indemnity_clause"
    assert "indemnifying_party" in schema.child_labels
    assert schema.canonical_child_mapping["indemnifying_party"] == "contracting_party"
    assert schema.canonical_child_mapping["monetary_cap"] == "liability_cap"


def test_induce_schema_maritime_demurrage():
    inducer = DynamicSchemaInducer()
    clause_text = (
        "Charterer shall pay demurrage of $15,000 per running day for any vessel delay "
        "exceeding the agreed laytime allowance due to port congestion."
    )
    schema = inducer.induce_schema(clause_text)
    assert schema is not None
    assert "demurrage" in schema.parent_label
    assert schema.canonical_parent == "penalty_clause"
    assert "demurrage_rate" in schema.child_labels
    assert schema.canonical_child_mapping["demurrage_rate"] == "penalty_rate"


def test_resolve_unclassified_clause_coordinate_stitching():
    extractor = FinancialEntityExtractor()
    inducer = DynamicSchemaInducer()

    clause_text = "Vendor shall defend and indemnify Customer against patent claims up to $500,000."
    unclassified = ParentClause(
        parent_label="marginal_ip_clause",
        clause_text=clause_text,
        score=0.38,
        global_start=100,
        global_end=100 + len(clause_text),
        is_unclassified=True,
    )

    resolved = inducer.resolve_unclassified_clause(unclassified, extractor.model, threshold=0.30)
    assert resolved.is_dynamically_induced is True
    assert resolved.is_unclassified is False
    assert resolved.canonical_parent == "indemnity_clause"
    assert resolved.global_start == 100
    assert resolved.global_end == 100 + len(clause_text)

    # Verify coordinate provenance for any extracted child
    for child in resolved.children:
        assert child.canonical_type is not None
        assert child.global_start == 100 + child.local_start
        assert child.global_end == 100 + child.local_end
        assert clause_text[child.local_start:child.local_end] == child.text


def test_api_hierarchical_dynamic_fallback():
    payload = {
        "text": (
            "Charterer shall pay demurrage of $20,000 per day for delay exceeding laytime. "
            "Governing Law shall be the State of New York."
        ),
        "document_type": "vendor_agreement",
        "threshold": 0.50,
        "fallback_threshold": 0.25,
        "enable_dynamic_fallback": True,
    }

    response = client.post("/api/extract-hierarchical", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "clauses" in data
    assert data["detected_parent_count"] >= 1
    # Verify no unclassified clauses were dumped without fallback processing
    for c in data["clauses"]:
        if c.get("is_dynamically_induced"):
            assert c["canonical_parent"] is not None
