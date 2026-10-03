"""
Unit tests for GLiNER integration in LLM Financial Analyst Agent.
"""

import os
os.environ["USE_MOCK_LLM"] = "true"

import pytest
from fastapi.testclient import TestClient
from agent.gliner_extractor import FinancialEntityExtractor, get_extractor
from api.server import app


def test_extractor_initialization():
    extractor = get_extractor()
    assert extractor is not None
    assert len(extractor.default_labels) > 0
    assert "penalty_rate" in extractor.default_labels


def test_entity_extraction_spans():
    extractor = get_extractor()
    text = "Vendor shall pay 2% per week of delay penalty up to a cap of $100,000 USD."
    entities = extractor.extract(text, threshold=0.3)
    assert isinstance(entities, list)
    assert len(entities) > 0
    first = entities[0]
    assert hasattr(first, "text")
    assert hasattr(first, "label")
    assert hasattr(first, "score")
    assert hasattr(first, "start")
    assert hasattr(first, "end")


def test_format_for_llm_prompt():
    extractor = get_extractor()
    text = "Contractor penalty is 5% per month."
    entities = extractor.extract(text, threshold=0.3)
    prompt_str = extractor.format_for_llm_prompt(entities)
    assert "===" in prompt_str
    assert "GLiNER" in prompt_str


def test_api_health_endpoint():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "gliner_model" in data


def test_api_schema_endpoint():
    client = TestClient(app)
    response = client.get("/api/entities/schema")
    assert response.status_code == 200
    data = response.json()
    assert "supported_labels" in data
    assert "penalty_rate" in data["supported_labels"]


def test_api_extract_entities_endpoint():
    client = TestClient(app)
    payload = {
        "text": "Acme Corp shall deliver on June 15, 2026 or pay a $5,000 daily penalty.",
        "threshold": 0.35
    }
    response = client.post("/api/extract-entities", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "entities" in data
    assert "entity_count" in data
    assert data["entity_count"] >= 1


def test_api_analyze_with_entities():
    client = TestClient(app)
    payload = {
        "document": "Overdue balance of $40,000 incurs 2% interest per week.",
        "task_type": "penalty_detection",
        "extract_entities": True
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["task_type"] == "penalty_detection"
    assert "analysis" in data
    assert "entities" in data
    assert data["entities"] is not None
