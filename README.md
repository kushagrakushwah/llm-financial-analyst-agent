# LLM Financial Analyst Agent with GLiNER

A two-stage financial auditing pipeline combining GLiNER (Generalist and Lightweight Model for Named Entity Recognition) with a Qwen2.5-7B GRPO policy agent for automated contract review, penalty detection, and financial risk evaluation.

---

## Overview

Auditing legal and financial documents using a large language model alone presents several production bottlenecks:
* High computational overhead and token latency when passing full documents.
* Tendency of generative models to paraphrase or shift numbers instead of preserving exact source spans.
* Inability of generative LLMs to natively provide character-level span offsets needed for audit trails.

This project introduces a hybrid pipeline that pairs a lightweight encoder-based model (GLiNER) with an autoregressive reasoning model (Qwen2.5-7B).

1. **Information Extraction Layer (GLiNER):** A compact bidirectional encoder extracts exact financial entities, dates, liability caps, and penalty rates in sub-100ms with character-level span indices.
2. **Reasoning Layer (Qwen2.5-7B trained with GRPO):** An instruction-tuned LLM receives the extracted factual spans alongside the text to produce actionable risk assessments and compliance reports.

```mermaid
flowchart TD
    Doc["Raw Financial Agreement / Contract"] --> GL["Stage 1: GLiNER Entity Extractor"]

    subgraph GLiNER_Layer["Stage 1: Deterministic Span Extraction (<100ms CPU)"]
        GL --> S1["penalty_rate: '2.5% per week of delay'"]
        GL --> S2["liability_cap: '$750,000 USD'"]
        GL --> S3["contracting_party: 'Contractor', 'Client'"]
        GL --> S4["governing_law: 'State of Delaware'"]
    end

    GLiNER_Layer --> PromptBuilder["Stage 2: Context Grounding & Prompt Builder"]
    PromptBuilder --> LLM["Stage 3: Qwen2.5-7B (GRPO Policy Agent)"]
    LLM --> Verdict["Stage 4: Structured Audit Finding & Exposure Verdict"]
```

---

## Technical Details: What is GLiNER?

GLiNER (Generalist and Lightweight Named Entity Recognition) is an encoder architecture based on DeBERTa-v3 that frames entity extraction as a bidirectional representation-matching task.

Unlike traditional token classification models (such as spaCy or standard BERT-NER) which rely on a fixed classification head and a predefined schema (e.g., `PERSON`, `ORG`, `LOC`), GLiNER takes candidate labels as inputs alongside the text:

1. Text tokens and candidate label tokens are concatenated and jointly encoded.
2. All possible span representations in the text are computed.
3. A similarity score is calculated between each candidate span vector and each label representation vector via dot product.
4. Spans exceeding the threshold are returned with their start and end character offsets.

### Why Combine GLiNER with an LLM?

* **Token Efficiency:** Pre-extracting legal and numerical entities allows the system to compress document context, reducing downstream prompt size by 60% to 80%.
* **Elimination of Hallucinations:** Critical monetary numbers, penalty formulas, and dates are pulled directly from the text as exact string slices.
* **Audit Trail Provenance:** Character offsets `[start, end]` allow UI dashboards to visually highlight the exact clause in the original contract.

---

## Project Structure

```text
llm-financial-analyst-agent/
├── agent/
│   ├── gliner_extractor.py      # Core extractor module and prompt formatter
│   └── environment.py           # RL audit environment with entity grounding rewards
├── api/
│   └── server.py                # FastAPI endpoints for entity extraction and analysis
├── data/
│   ├── financial_ner_train.json # Tokenized training samples for financial entity fine-tuning
│   └── financial_ner_eval.json  # Evaluation dataset
├── training/
│   ├── train_gliner.py          # GLiNER fine-tuning script with differential learning rates
│   └── train_grpo.py            # GRPO reinforcement learning loop for Qwen2.5-7B
├── tests/
│   └── test_gliner.py           # Automated test suite for the extractor and API
├── demo_gliner_audit.py         # Standalone CLI demonstration script
├── main.py                      # Application entrypoint
├── requirements.txt             # Pinned project dependencies
└── README.md                    # Project documentation
```

---

## Installation

### Prerequisites
* Python 3.10 to 3.12
* PyTorch (CPU or CUDA)

```bash
git clone https://github.com/kushagrakushwah/llm-financial-analyst-agent.git
cd llm-financial-analyst-agent

# Set up virtual environment
python -m venv .venv

# Activate on Linux/macOS:
source .venv/bin/activate

# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Project

### 1. Interactive CLI Demonstration
To test GLiNER on three contract scenarios (Vendor SLA, Overdue Invoicing, Software License):
```bash
python demo_gliner_audit.py
```

### 2. Start the API Server
```bash
python main.py
```
The server runs on `http://localhost:8001`. Interactive API documentation is available at `http://localhost:8001/docs`.

---

## API Reference

### `POST /api/extract-entities`
Performs lightweight, deterministic entity span extraction directly via GLiNER without loading or running the 7B LLM.

**Request:**
```bash
curl -X POST "http://localhost:8001/api/extract-entities" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Contractor shall pay 2.5% per week of delay penalty up to a maximum cap of $500,000 USD.",
       "threshold": 0.35
     }'
```

**Response:**
```json
{
  "entity_count": 3,
  "entities": [
    {
      "text": "Contractor",
      "label": "contracting_party",
      "score": 0.9124,
      "start": 0,
      "end": 10
    },
    {
      "text": "2.5% per week of delay",
      "label": "penalty_rate",
      "score": 0.7852,
      "start": 21,
      "end": 43
    },
    {
      "text": "$500,000 USD",
      "label": "liability_cap",
      "score": 0.8419,
      "start": 68,
      "end": 80
    }
  ],
  "model_source": "models/gliner_financial"
}
```

### `POST /api/analyze`
Executes the full hybrid audit: extracts entity spans, passes them as grounded context to the reasoning engine, and produces an exposure report.

**Request:**
```bash
curl -X POST "http://localhost:8001/api/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "document": "Invoice #INV-2025-9941 is overdue by 45 days. Accrues 1.5% monthly late interest starting day 31.",
       "task_type": "penalty_detection",
       "extract_entities": true
     }'
```

### `GET /api/entities/schema`
Returns the list of legal and financial entity labels supported by the extractor.

---

## Fine-Tuning GLiNER

While base zero-shot models recognize common generic entities, fine-tuning is required for domain-specific contract formulations such as multi-word delay penalties, grace periods, and liability caps.

### 1. Data Schema
Training data is stored in `data/financial_ner_train.json` in tokenized span format:
```json
[
  {
    "tokenized_text": ["Vendor", "shall", "pay", "2%", "per", "week", "of", "delay", "."],
    "ner": [
      [0, 1, "contracting_party"],
      [3, 8, "penalty_rate"]
    ]
  }
]
```

### 2. Training Process
Run the fine-tuning script:
```bash
python training/train_gliner.py
```

Key training parameters configured in `training/train_gliner.py`:
* **Differential Learning Rates:** `1e-5` for the DeBERTa backbone (to avoid catastrophic forgetting) and `1e-4` for the projection head.
* **Negative Sampling (`negatives=1.0`):** Samples unmentioned label classes during each training step so the model retains its zero-shot discrimination and suppresses false positives.
* **Span Collator:** Uses `SpanDataCollator` to package token indices and span matrices.
* **Output:** Saves the checkpoint to `models/gliner_financial`.

---

## Testing

Run the automated test suite with pytest:
```bash
pytest tests/test_gliner.py -v
```

The test suite covers:
* Extractor initialization and label schema verification.
* Span extraction correctness and offset integrity.
* Prompt formatting logic.
* All FastAPI endpoints (`/api/health`, `/api/entities/schema`, `/api/extract-entities`, `/api/analyze`).

