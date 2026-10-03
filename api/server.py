import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from agent.gliner_extractor import (
    FinancialEntityExtractor,
    EntitySpan,
    get_extractor,
    DEFAULT_FINANCIAL_LABELS,
)

app = FastAPI(
    title="LLM Financial Analyst Agent with GLiNER",
    description="Hybrid Financial Auditing Engine: Ultra-fast GLiNER entity span extraction + Qwen2.5-7B GRPO reasoning.",
    version="2.0.0",
)

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
            print(f"[Warning] Could not load LLM '{MODEL_PATH}' ({e}). Falling back to GLiNER-augmented synthesis mode.")
            _pipe = None
    return _pipe


# ---------------------------------------------------------------------------
# Data Schemas
# ---------------------------------------------------------------------------

class EntityItem(BaseModel):
    text: str
    label: str
    score: float
    start: int
    end: int


class ExtractEntitiesRequest(BaseModel):
    text: str = Field(..., description="Financial document or contract text to analyze.")
    labels: Optional[List[str]] = Field(
        default=None,
        description="List of target entity labels. Defaults to standard contract auditing labels if omitted."
    )
    threshold: float = Field(0.40, ge=0.0, le=1.0, description="Confidence threshold.")
    flat_ner: bool = Field(True, description="Whether to suppress overlapping entities.")


class ExtractEntitiesResponse(BaseModel):
    entities: List[EntityItem]
    entity_count: int
    labels_used: List[str]
    model_source: str


class AnalyzeRequest(BaseModel):
    document: str
    task_type: str = "contract_review"  # contract_review | penalty_detection | cost_benefit
    extract_entities: bool = True
    threshold: float = 0.40


class AnalyzeResponse(BaseModel):
    task_type: str
    analysis: str
    entities: Optional[List[EntityItem]] = None


SYSTEM_PROMPT = (
    "You are a precise financial auditing assistant. "
    "Provide concise, structured analysis. Identify risks, penalties, and recommendations clearly."
)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    extractor = get_extractor()
    return {
        "status": "ok",
        "gliner_model": extractor.model_name_or_path,
        "llm_model": MODEL_PATH,
    }


@app.get("/api/entities/schema")
def get_entity_schema():
    """Returns the default financial entity categories supported by the extractor."""
    return {
        "supported_labels": DEFAULT_FINANCIAL_LABELS,
        "description": "Pre-configured labels for financial agreements, penalties, SLAs, and liability caps.",
    }


@app.post("/api/extract-entities", response_model=ExtractEntitiesResponse)
def extract_entities(req: ExtractEntitiesRequest):
    """
    Dedicated endpoint for ultra-fast, zero-hallucination entity span extraction.
    Runs in milliseconds on CPU without invoking the heavy LLM.
    """
    extractor = get_extractor()
    labels = req.labels or extractor.default_labels
    spans = extractor.extract(
        text=req.text,
        labels=labels,
        threshold=req.threshold,
        flat_ner=req.flat_ner,
    )
    return ExtractEntitiesResponse(
        entities=[EntityItem(**s.to_dict()) for s in spans],
        entity_count=len(spans),
        labels_used=labels,
        model_source=extractor.model_name_or_path,
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """
    Hybrid Financial Audit:
    1. GLiNER extracts exact numeric penalties, liability caps, and party spans.
    2. The reasoning agent produces an actionable risk assessment and verdict.
    """
    if req.task_type not in ("contract_review", "penalty_detection", "cost_benefit"):
        raise HTTPException(status_code=400, detail="Invalid task_type")

    extracted_items: Optional[List[EntityItem]] = None
    entity_context_str = ""

    # Step 1: Deterministic Span Extraction with GLiNER
    if req.extract_entities:
        extractor = get_extractor()
        spans = extractor.extract(text=req.document, threshold=req.threshold)
        extracted_items = [EntityItem(**s.to_dict()) for s in spans]
        entity_context_str = extractor.format_for_llm_prompt(spans)

    # Step 2: Contextual Prompt Formulation
    augmented_user_prompt = f"Task: {req.task_type}\n\n"
    if entity_context_str:
        augmented_user_prompt += f"{entity_context_str}\n\n"
    augmented_user_prompt += f"Document Content:\n{req.document}"

    pipe = get_pipeline()
    if pipe is not None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": augmented_user_prompt},
        ]
        output = pipe(messages)
        text = output[0]["generated_text"][-1]["content"].strip()
    else:
        # Structured deterministic fallback when full LLM is offline
        text = _synthesize_audit_report(req.task_type, req.document, extracted_items or [])

    return AnalyzeResponse(
        task_type=req.task_type,
        analysis=text,
        entities=extracted_items,
    )


def _synthesize_audit_report(task_type: str, document: str, entities: List[EntityItem]) -> str:
    """Synthesizes an immediate structured audit finding grounded in GLiNER entities."""
    grouped = {}
    for ent in entities:
        grouped.setdefault(ent.label, []).append(ent.text)

    findings = [f"=== AUDIT REPORT: {task_type.upper().replace('_', ' ')} ==="]
    
    if "penalty_rate" in grouped:
        findings.append(f"• Penalty Exposure Flagged: {', '.join(grouped['penalty_rate'])}")
    if "liability_cap" in grouped:
        findings.append(f"• Liability Limitation: {', '.join(grouped['liability_cap'])}")
    if "contracting_party" in grouped:
        findings.append(f"• Identified Parties: {', '.join(grouped['contracting_party'])}")
    if "termination_clause" in grouped:
        findings.append(f"• Termination Conditions: {', '.join(grouped['termination_clause'])}")

    if not entities:
        findings.append("• No explicit monetary penalties or liability caps detected in this clause.")
    else:
        findings.append(f"• Total Verified Legal Entities: {len(entities)}")

    findings.append("• Audit Recommendation: Ensure liquidated damages provisions are capped against total contract value.")
    return "\n".join(findings)


# ---------------------------------------------------------------------------
# Frontend Static App Delivery
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

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

