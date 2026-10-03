"""
test_production_pipeline.py
===========================
Comprehensive unit and integration tests for enterprise production components:
1. DocumentChunker (boundary splitting & NMS span stitching)
2. FinancialRiskEngine (quantitative exposure, SLA calculations, penalty APR, scorecards)
3. Production API Endpoints (/api/batch-extract, /api/audit-document, /api/export/csv, /api/health)
"""

import os
os.environ["USE_MOCK_LLM"] = "true"

import pytest
from fastapi.testclient import TestClient
from api.server import app
from agent.gliner_extractor import EntitySpan, get_extractor
from agent.document_chunker import DocumentChunker, DocumentChunk
from agent.risk_engine import get_risk_engine, FinancialRiskEngine


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. DocumentChunker Tests
# ---------------------------------------------------------------------------

def test_document_chunker_short_text():
    chunker = DocumentChunker(max_chars=500, overlap_chars=100)
    short_text = "This is a short clause under the max limit."
    chunks = chunker.chunk(short_text)
    assert len(chunks) == 1
    assert chunks[0].text == short_text
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(short_text)


def test_document_chunker_long_text_boundaries():
    chunker = DocumentChunker(max_chars=120, overlap_chars=30)
    para1 = "Paragraph 1: Provider agrees to maintain 99.95% uptime SLA.\n\n"
    para2 = "Paragraph 2: Customer agrees to pay recurring fees of $50,000 USD.\n\n"
    para3 = "Paragraph 3: Governing law shall be the laws of the State of Delaware."
    full_doc = para1 + para2 + para3

    chunks = chunker.chunk(full_doc)
    assert len(chunks) >= 2
    # Verify all text is covered across chunks
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(full_doc)


def test_document_chunker_span_stitching_and_nms():
    chunker = DocumentChunker(max_chars=200, overlap_chars=50)
    full_text = "Parties are Alpha Corp and Beta LLC. Total cap is $100,000 USD."

    # Simulate two overlapping chunks both extracting 'Alpha Corp'
    chunk1 = DocumentChunk(chunk_id=0, text="Parties are Alpha Corp and Beta LLC.", start_char=0, end_char=36)
    chunk2 = DocumentChunk(chunk_id=1, text="Alpha Corp and Beta LLC. Total cap is $100,000 USD.", start_char=12, end_char=63)

    ent1 = {"text": "Alpha Corp", "label": "contracting_party", "score": 0.95, "start": 12, "end": 22}
    ent2 = {"text": "Alpha Corp", "label": "contracting_party", "score": 0.98, "start": 0, "end": 10}
    ent3 = {"text": "$100,000 USD", "label": "liability_cap", "score": 0.92, "start": 38, "end": 50}

    chunk_extractions = [
        (chunk1, [ent1]),
        (chunk2, [ent2, ent3])
    ]

    stitched = chunker.stitch_spans(full_text, chunk_extractions)

    # Should deduplicate 'Alpha Corp' into a single entity with the higher score (0.98)
    alpha_ents = [e for e in stitched if e["text"] == "Alpha Corp"]
    assert len(alpha_ents) == 1
    assert alpha_ents[0]["score"] == 0.98
    assert alpha_ents[0]["start"] == 12
    assert alpha_ents[0]["end"] == 22

    # Should also include the liability cap
    cap_ents = [e for e in stitched if e["label"] == "liability_cap"]
    assert len(cap_ents) == 1
    assert cap_ents[0]["start"] == 50
    assert cap_ents[0]["end"] == 62


# ---------------------------------------------------------------------------
# 2. FinancialRiskEngine Tests
# ---------------------------------------------------------------------------

def test_risk_engine_uncapped_penalty():
    engine = FinancialRiskEngine()
    text = "Contractor shall pay 2.5% per week for delay."
    entities = [
        EntitySpan(text="2.5% per week", label="penalty_rate", score=0.92, start=21, end=34),
        EntitySpan(text="Contractor", label="contracting_party", score=0.95, start=0, end=10),
    ]
    scorecard = engine.evaluate(text, entities)
    assert scorecard.overall_risk_tier in ("CRITICAL", "HIGH")
    assert not scorecard.liability_capped
    assert scorecard.governance_score < 70
    assert any(f.severity == "CRITICAL" for f in scorecard.findings)


def test_risk_engine_protected_capped_contract():
    engine = FinancialRiskEngine()
    text = "Contractor shall pay 1.0% per month for delay. Liability cap is $250,000 USD. Governing law is Delaware."
    entities = [
        EntitySpan(text="Contractor", label="contracting_party", score=0.95, start=0, end=10),
        EntitySpan(text="1.0% per month", label="penalty_rate", score=0.90, start=21, end=35),
        EntitySpan(text="$250,000 USD", label="liability_cap", score=0.96, start=64, end=76),
        EntitySpan(text="Delaware", label="governing_law", score=0.91, start=96, end=104),
    ]
    scorecard = engine.evaluate(text, entities)
    assert scorecard.liability_capped
    assert scorecard.total_liability_cap_amount == 250000.0
    assert scorecard.governing_jurisdiction == "Delaware"
    assert scorecard.governance_score >= 80


def test_risk_engine_sla_downtime_calculation():
    engine = FinancialRiskEngine()
    text = "Provider commits to 99.95% uptime SLA."
    entities = [
        EntitySpan(text="99.95%", label="sla_target", score=0.95, start=20, end=26)
    ]
    scorecard = engine.evaluate(text, entities)
    assert scorecard.sla_monthly_downtime_minutes is not None
    # 43800 * 0.0005 = 21.9 minutes
    assert abs(scorecard.sla_monthly_downtime_minutes - 21.9) < 0.5


# ---------------------------------------------------------------------------
# 3. Production API Endpoints Tests
# ---------------------------------------------------------------------------

def test_api_health_production(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] in ("ok", "healthy")
    assert data["version"] == "2.5.0"
    assert data["supported_labels_count"] >= 12


def test_api_batch_extract(client):
    payload = {
        "documents": [
            "Party A agrees to pay $10,000 USD.",
            "Governing law is New York. SLA target is 99.9%."
        ],
        "threshold": 0.40
    }
    res = client.post("/api/batch-extract", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_documents"] == 2
    assert len(data["results"]) == 2


def test_api_audit_document(client):
    payload = {
        "document": (
            "This Agreement is between Acme Corp and Beta LLC. "
            "Liquidated damages of 1.5% per month apply to overdue amounts. "
            "Liability shall not exceed $100,000 USD. Governing law is Delaware."
        ),
        "threshold": 0.40
    }
    res = client.post("/api/audit-document", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "scorecard" in data
    assert "governance_score" in data["scorecard"]
    assert "findings" in data["scorecard"]
    assert len(data["entities"]) > 0


def test_api_export_csv(client):
    payload = {
        "text": "Acme Corp agrees to liability cap of $500,000 USD.",
        "threshold": 0.40
    }
    res = client.post("/api/export/csv", json=payload)
    assert res.status_code == 200
    assert "text/csv" in res.headers["content-type"]
    csv_text = res.text
    assert "Entity Text" in csv_text
    assert "Label" in csv_text
