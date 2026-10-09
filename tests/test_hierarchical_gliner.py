"""
Unit and integration tests for the Dynamic, Context-Aware, and Hierarchical GLiNER pipeline.
Validates taxonomy resolution, two-pass extraction, global offset stitching, and fallback tagging.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from agent.label_taxonomy import (
    TAXONOMY_REGISTRY,
    detect_document_type,
    get_taxonomy_for_type,
)
from agent.schemas import (
    ChildEntity,
    ParentClause,
    HierarchicalExtractionResult,
    HierarchicalExtractRequest,
)
from agent.gliner_extractor import FinancialEntityExtractor
from api.server import app


# ---------------------------------------------------------------------------
# Test 1: Taxonomy & Dynamic Document Router
# ---------------------------------------------------------------------------

def test_taxonomy_registry_structure():
    """Ensures all configured document types have valid parent clauses and child mappings."""
    for doc_type, data in TAXONOMY_REGISTRY.items():
        assert "parent_clauses" in data
        assert "child_labels" in data
        assert len(data["parent_clauses"]) > 0
        # Every parent clause with children must map to a non-empty list of child labels
        for parent_label, children in data["child_labels"].items():
            assert isinstance(children, list)
            assert len(children) > 0


def test_document_type_router_detection():
    """Validates automatic heuristic routing on distinct contract clauses."""
    employment_sample = (
        "Executive Employment Agreement: The Employee shall receive an initial base salary of "
        "$175,000 USD per annum with a 90-day probation period."
    )
    credit_sample = (
        "Credit Facility Agreement: Horizon Capital Bank as Lender agrees to extend to the Borrower "
        "a revolving credit facility principal amount of $50,000,000 USD."
    )
    lease_sample = (
        "Commercial Lease: Landlord leases to Tenant the premises located at 450 Lexington Ave "
        "at a monthly base rent of $12,500 with a security deposit."
    )
    vendor_sample = (
        "Master Cloud Agreement: Provider commits to maintaining an uptime SLA target of 99.95% "
        "with liquidated damages and a total liability cap of $250,000 USD."
    )

    assert detect_document_type(employment_sample) == "employment_agreement"
    assert detect_document_type(credit_sample) == "credit_loan_agreement"
    assert detect_document_type(lease_sample) == "commercial_lease"
    assert detect_document_type(vendor_sample) == "vendor_agreement"


# ---------------------------------------------------------------------------
# Test 2: Pydantic Schema Validation & Offset Verification
# ---------------------------------------------------------------------------

def test_pydantic_schema_integrity():
    """Validates strict Pydantic model serialization and coordinate alignment."""
    raw_doc = (
        "Provider will maintain 99.95% uptime SLA. In case of delay, liquidated damages "
        "of a 3.5% penalty rate apply."
    )
    clause_text = "liquidated damages of a 3.5% penalty rate apply."
    clause_global_start = raw_doc.index(clause_text)
    clause_global_end = clause_global_start + len(clause_text)

    # Sub-span "3.5%" within clause_text
    local_s = clause_text.index("3.5%")
    local_e = local_s + len("3.5%")
    global_s = clause_global_start + local_s
    global_e = clause_global_start + local_e

    # Verify global coordinate accuracy
    assert raw_doc[global_s:global_e] == "3.5%"

    child = ChildEntity(
        text="3.5%",
        label="penalty_rate",
        score=0.94,
        local_start=local_s,
        local_end=local_e,
        global_start=global_s,
        global_end=global_e,
    )

    parent = ParentClause(
        parent_label="termination_clause",
        clause_text=clause_text,
        score=0.91,
        global_start=clause_global_start,
        global_end=clause_global_end,
        is_unclassified=False,
        children=[child],
    )

    result = HierarchicalExtractionResult(
        document_type="vendor_agreement",
        detected_parent_count=1,
        detected_child_count=1,
        execution_time_ms=125.4,
        clauses=[parent],
        unclassified_clauses=[],
    )

    serialized = result.to_dict()
    assert serialized["document_type"] == "vendor_agreement"
    assert serialized["detected_parent_count"] == 1
    assert serialized["clauses"][0]["children"][0]["text"] == "3.5%"
    assert serialized["clauses"][0]["children"][0]["global_start"] == global_s


# ---------------------------------------------------------------------------
# Test 3: Two-Pass Extraction Mock Execution
# ---------------------------------------------------------------------------

def test_two_pass_extraction_engine_mock():
    """Validates the two-pass algorithm logic and coordinate transformation using a mocked GLiNER."""
    extractor = FinancialEntityExtractor(model_name_or_path="mock")

    # Mock the internal GLiNER model
    mock_model = MagicMock()

    full_contract = (
        "Master Agreement dated October 15, 2026. SECTION 4: Contractor shall pay liquidated "
        "damages equal to a 2.5% penalty rate. Disputes governed by Delaware law."
    )
    parent_clause_str = "Contractor shall pay liquidated damages equal to a 2.5% penalty rate."
    p_start = full_contract.index(parent_clause_str)
    p_end = p_start + len(parent_clause_str)

    child_token = "2.5%"
    c_local_start = parent_clause_str.index(child_token)
    c_local_end = c_local_start + len(child_token)

    def mock_predict(text, labels, threshold=0.45, flat_ner=True):
        if "termination_clause" in labels:
            return [
                {
                    "text": parent_clause_str,
                    "label": "termination_clause",
                    "score": 0.93,
                    "start": p_start,
                    "end": p_end,
                }
            ]
        elif "penalty_rate" in labels:
            return [
                {
                    "text": child_token,
                    "label": "penalty_rate",
                    "score": 0.96,
                    "start": c_local_start,
                    "end": c_local_end,
                }
            ]
        return []

    mock_model.predict_entities.side_effect = mock_predict
    extractor._model = mock_model

    result = extractor.extract_hierarchical(
        text=full_contract,
        doc_type="vendor_agreement",
        threshold=0.45,
    )

    assert result.document_type == "vendor_agreement"
    assert result.detected_parent_count == 1
    assert result.detected_child_count == 1

    clause = result.clauses[0]
    assert clause.parent_label == "termination_clause"
    assert len(clause.children) == 1

    child = clause.children[0]
    assert child.label == "penalty_rate"
    assert child.text == "2.5%"
    assert child.global_start == p_start + c_local_start
    assert child.global_end == p_start + c_local_end
    assert full_contract[child.global_start:child.global_end] == "2.5%"


# ---------------------------------------------------------------------------
# Test 4: FastAPI Server Hierarchical & Taxonomy Endpoints
# ---------------------------------------------------------------------------

def test_api_taxonomy_endpoint():
    """Tests GET /api/taxonomy."""
    client = TestClient(app)
    response = client.get("/api/taxonomy")
    assert response.status_code == 200
    data = response.json()
    assert "document_types" in data
    assert "vendor_agreement" in data["document_types"]
    assert "employment_agreement" in data["document_types"]
    assert "taxonomies" in data


def test_api_extract_hierarchical_endpoint():
    """Tests POST /api/extract-hierarchical with mock extractor injection."""
    client = TestClient(app)
    payload = {
        "text": "This Cloud Services Agreement maintains a 99.95% uptime SLA.",
        "document_type": "vendor_agreement",
        "threshold": 0.45,
    }
    response = client.post("/api/extract-hierarchical", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "document_type" in data
    assert data["document_type"] == "vendor_agreement"
    assert "clauses" in data
    assert "execution_time_ms" in data
