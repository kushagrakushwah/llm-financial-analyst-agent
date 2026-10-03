"""
api/server.py
=============
Enterprise Production FastAPI server for Financial Contract Auditing.
Combines ultra-fast GLiNER deterministic entity extraction with
quantitative financial risk scorecards and Qwen2.5-7B audit synthesis.
"""

import os
import io
import csv
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from agent.gliner_extractor import (
    FinancialEntityExtractor,
    EntitySpan,
    get_extractor,
    DEFAULT_FINANCIAL_LABELS,
)
from agent.risk_engine import get_risk_engine, AuditScorecard

# Initialize FastAPI app with production metadata
app = FastAPI(
    title="Financial Analyst Agent Engine",
    description="Enterprise Contract Intelligence API: Fast GLiNER span extraction + Quantitative Risk Rules + LLM Governance Reasoning.",
    version="2.5.0",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Process Timing Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response


MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "auto").lower()

_pipe = None


def get_pipeline():
    global _pipe
    if _pipe is None:
        if USE_MOCK_LLM == "true":
            return None
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            _pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                do_sample=False,
            )
        except Exception as e:
            print(f"[Warning] Could not load LLM '{MODEL_PATH}' ({e}). Running in GLiNER-augmented deterministic mode.")
            _pipe = None
    return _pipe


# ---------------------------------------------------------------------------
# Pydantic Request & Response Schemas
# ---------------------------------------------------------------------------

class EntityItem(BaseModel):
    text: str
    label: str
    score: float
    start: int
    end: int


class ExtractEntitiesRequest(BaseModel):
    text: str = Field(..., description="Contract text or legal clause.")
    labels: Optional[List[str]] = Field(
        default=None,
        description="List of target labels. Defaults to standard 12 contract auditing categories."
    )
    threshold: float = Field(0.45, ge=0.0, le=1.0, description="Confidence score threshold.")
    flat_ner: bool = Field(True, description="Whether to suppress overlapping entity spans.")
    chunk_long_text: bool = Field(True, description="Auto-chunk text longer than 1200 chars.")


class ExtractEntitiesResponse(BaseModel):
    entities: List[EntityItem]
    entity_count: int
    labels_used: List[str]
    model_source: str
    document_length: int


class BatchExtractRequest(BaseModel):
    documents: List[str] = Field(..., max_length=50, description="List of contract documents or clauses to extract.")
    labels: Optional[List[str]] = None
    threshold: float = 0.45


class BatchExtractResponse(BaseModel):
    results: List[ExtractEntitiesResponse]
    total_documents: int
    total_entities_detected: int


class RiskFindingItem(BaseModel):
    severity: str
    title: str
    description: str
    clause_reference: Optional[str] = None
    action_item: Optional[str] = None


class AuditScorecardItem(BaseModel):
    overall_risk_tier: str
    governance_score: int
    liability_capped: bool
    total_liability_cap_amount: Optional[float] = None
    total_liability_cap_currency: Optional[str] = "USD"
    identified_parties: List[str]
    max_penalty_rate_annualized_pct: Optional[float] = None
    sla_monthly_downtime_minutes: Optional[float] = None
    governing_jurisdiction: Optional[str] = None
    findings: List[RiskFindingItem]
    executive_verdict: str


class FullAuditRequest(BaseModel):
    document: str = Field(..., description="Full contract or agreement text.")
    labels: Optional[List[str]] = None
    threshold: float = 0.45
    task_type: str = "contract_review"


class FullAuditResponse(BaseModel):
    task_type: str
    document_length: int
    entities: List[EntityItem]
    scorecard: AuditScorecardItem
    llm_analysis: str


class AnalyzeRequest(BaseModel):
    document: str
    task_type: str = "contract_review"
    extract_entities: bool = True
    threshold: float = 0.45


class AnalyzeResponse(BaseModel):
    task_type: str
    analysis: str
    entities: Optional[List[EntityItem]] = None


SYSTEM_PROMPT = (
    "You are a precise financial contract auditing assistant. "
    "Provide concise, structured governance analysis grounded strictly in extracted facts."
)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    """System health check and runtime telemetry."""
    extractor = get_extractor()
    return {
        "status": "ok",
        "service": "Financial Analyst Agent",
        "version": "2.5.0",
        "gliner_model": extractor.model_name_or_path,
        "device": extractor.device,
        "supported_labels_count": len(extractor.default_labels),
        "llm_model": MODEL_PATH,
        "mock_llm_mode": (USE_MOCK_LLM == "true" or _pipe is None),
    }


@app.get("/api/entities/schema")
def get_entity_schema():
    """Returns the 12 default contract categories supported by the extractor."""
    return {
        "supported_labels": DEFAULT_FINANCIAL_LABELS,
        "description": "Pre-configured labels for financial agreements, penalties, SLAs, and liability caps.",
    }


@app.post("/api/extract-entities", response_model=ExtractEntitiesResponse)
def extract_entities(req: ExtractEntitiesRequest):
    """
    Dedicated endpoint for ultra-fast, zero-hallucination entity span extraction.
    Automatically chunks large documents exceeding 1200 characters without truncation.
    """
    if len(req.text) > 500000:
        raise HTTPException(status_code=413, detail="Document exceeds maximum allowed length of 500,000 characters.")

    extractor = get_extractor()
    labels = req.labels or extractor.default_labels
    spans = extractor.extract(
        text=req.text,
        labels=labels,
        threshold=req.threshold,
        flat_ner=req.flat_ner,
        chunk_long_text=req.chunk_long_text,
    )
    return ExtractEntitiesResponse(
        entities=[EntityItem(**s.to_dict()) for s in spans],
        entity_count=len(spans),
        labels_used=labels,
        model_source=extractor.model_name_or_path,
        document_length=len(req.text),
    )


@app.post("/api/batch-extract", response_model=BatchExtractResponse)
def batch_extract(req: BatchExtractRequest):
    """
    High-throughput batch extraction across multiple clauses or agreements.
    """
    extractor = get_extractor()
    labels = req.labels or extractor.default_labels
    results = []
    total_entities = 0

    for doc in req.documents:
        spans = extractor.extract(text=doc, labels=labels, threshold=req.threshold)
        items = [EntityItem(**s.to_dict()) for s in spans]
        total_entities += len(items)
        results.append(ExtractEntitiesResponse(
            entities=items,
            entity_count=len(items),
            labels_used=labels,
            model_source=extractor.model_name_or_path,
            document_length=len(doc),
        ))

    return BatchExtractResponse(
        results=results,
        total_documents=len(req.documents),
        total_entities_detected=total_entities,
    )


@app.post("/api/audit-document", response_model=FullAuditResponse)
def audit_document(req: FullAuditRequest):
    """
    Complete Enterprise Financial Audit:
    1. Extracts entity spans via GLiNER with boundary-aware sliding chunking.
    2. Runs quantitative FinancialRiskEngine to compute exposure, statutory checks, and scorecard.
    3. Synthesizes executive audit verdict.
    """
    extractor = get_extractor()
    risk_engine = get_risk_engine()

    # Step 1: Extraction
    labels = req.labels or extractor.default_labels
    spans = extractor.extract(text=req.document, labels=labels, threshold=req.threshold)
    entity_items = [EntityItem(**s.to_dict()) for s in spans]

    # Step 2: Quantitative Risk Scoring
    scorecard = risk_engine.evaluate(req.document, spans)

    # Step 3: Executive LLM Reasoning
    entity_context_str = extractor.format_for_llm_prompt(spans)
    augmented_user_prompt = (
        f"Task: {req.task_type}\n"
        f"Quantitative Risk Tier: {scorecard.overall_risk_tier} (Score: {scorecard.governance_score}/100)\n\n"
        f"{entity_context_str}\n\n"
        f"Document Content:\n{req.document}"
    )

    pipe = get_pipeline()
    if pipe is not None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": augmented_user_prompt},
        ]
        output = pipe(messages)
        llm_text = output[0]["generated_text"][-1]["content"].strip()
    else:
        llm_text = scorecard.executive_verdict

    return FullAuditResponse(
        task_type=req.task_type,
        document_length=len(req.document),
        entities=entity_items,
        scorecard=AuditScorecardItem(**scorecard.to_dict()),
        llm_analysis=llm_text,
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Backward-compatible hybrid audit endpoint.
    """
    if req.task_type not in ("contract_review", "penalty_detection", "cost_benefit"):
        raise HTTPException(status_code=400, detail="Invalid task_type")

    extractor = get_extractor()
    risk_engine = get_risk_engine()

    spans = []
    if req.extract_entities:
        spans = extractor.extract(text=req.document, threshold=req.threshold)

    entity_items = [EntityItem(**s.to_dict()) for s in spans]
    scorecard = risk_engine.evaluate(req.document, spans)

    pipe = get_pipeline()
    if pipe is not None:
        entity_context_str = extractor.format_for_llm_prompt(spans)
        augmented_user_prompt = f"Task: {req.task_type}\n\n{entity_context_str}\n\nDocument Content:\n{req.document}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": augmented_user_prompt},
        ]
        output = pipe(messages)
        analysis_text = output[0]["generated_text"][-1]["content"].strip()
    else:
        analysis_text = _format_legacy_audit_report(req.task_type, scorecard, entity_items)

    return AnalyzeResponse(
        task_type=req.task_type,
        analysis=analysis_text,
        entities=entity_items,
    )


@app.post("/api/export/csv")
def export_csv(req: ExtractEntitiesRequest):
    """Exports extracted entity spans as a downloadable CSV spreadsheet."""
    extractor = get_extractor()
    spans = extractor.extract(text=req.text, labels=req.labels, threshold=req.threshold)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Entity Text", "Label", "Confidence", "Start Offset", "End Offset"])
    for s in spans:
        writer.writerow([s.text, s.label, f"{s.score:.4f}", s.start, s.end])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=extracted_entities.csv"}
    )


def _format_legacy_audit_report(task_type: str, scorecard: AuditScorecard, entities: List[EntityItem]) -> str:
    lines = [
        f"=== AUDIT REPORT: {task_type.upper().replace('_', ' ')} ===",
        f"• Governance Score: {scorecard.governance_score}/100 ({scorecard.overall_risk_tier} RISK)",
        f"• Parties Identified: {', '.join(scorecard.identified_parties) if scorecard.identified_parties else 'Standard Commercial Entities'}",
        f"• Total Verified Legal Entities: {len(entities)}",
        f"• Liability Capped: {'Yes' if scorecard.liability_capped else 'NO (CRITICAL UNBOUNDED EXPOSURE)'}",
    ]
    if scorecard.governing_jurisdiction:
        lines.append(f"• Governing Jurisdiction: {scorecard.governing_jurisdiction}")
    if scorecard.max_penalty_rate_annualized_pct:
        lines.append(f"• Annualized Penalty Exposure: {scorecard.max_penalty_rate_annualized_pct:.1f}%")

    lines.append("")
    lines.append("EXECUTIVE FINDINGS:")
    for f in scorecard.findings:
        lines.append(f"  [{f.severity}] {f.title}: {f.description}")
        if f.action_item:
            lines.append(f"    -> Action: {f.action_item}")

    lines.append("")
    lines.append(f"FINAL VERDICT: {scorecard.executive_verdict}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Frontend Static App Delivery
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
PLAYGROUND_DIR = os.path.join(BASE_DIR, "playground")

if os.path.exists(FRONTEND_DIR):
    @app.get("/")
    def serve_root():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"status": "ok", "message": "Financial Analyst Agent API Running"}

    @app.get("/app")
    def serve_app():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"status": "ok", "message": "Financial Analyst Agent API Running"}

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

if os.path.exists(PLAYGROUND_DIR):
    @app.get("/playground")
    def serve_playground():
        index_path = os.path.join(PLAYGROUND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"status": "ok", "message": "Playground not found"}
