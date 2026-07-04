from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

app = FastAPI(title="LLM Financial Analyst Agent")

MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")

_pipe = None


def get_pipeline():
    global _pipe
    if _pipe is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model     = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        )
        _pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                         max_new_tokens=256, do_sample=False)
    return _pipe


class AnalyzeRequest(BaseModel):
    document: str
    task_type: str = "contract_review"  # contract_review | penalty_detection | cost_benefit


class AnalyzeResponse(BaseModel):
    task_type: str
    analysis:  str


SYSTEM_PROMPT = (
    "You are a precise financial auditing assistant. "
    "Provide concise, structured analysis. Identify risks, penalties, and recommendations clearly."
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if req.task_type not in ("contract_review", "penalty_detection", "cost_benefit"):
        raise HTTPException(status_code=400, detail="Invalid task_type")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Task: {req.task_type}\n\n{req.document}"},
    ]
    pipe   = get_pipeline()
    output = pipe(messages)
    text   = output[0]["generated_text"][-1]["content"]
    return AnalyzeResponse(task_type=req.task_type, analysis=text.strip())
