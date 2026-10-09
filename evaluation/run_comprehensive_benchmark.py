"""
run_comprehensive_benchmark.py
==============================
Rigorous multi-dataset evaluation suite benchmarking:
1. Baseline Heuristic (Regex & Rule-based)
2. Zero-Shot GLiNER Base (urchade/gliner_small-v2.1)
3. Fine-Tuned Financial GLiNER v1 (models/gliner_financial)
4. Upgraded GLiNER v2 (Trained on Internet Datasets: models/gliner_financial_v2)

Evaluated across 5 distinct benchmark datasets:
- Dataset 1: In-Domain Financial & Contract Evaluation (30 clauses, dense multi-label)
- Dataset 2: Real SEC Edgar Commercial Contracts (100 clauses, CUAD held-out 400-510)
- Dataset 3: High-Risk Liability & Governance Provisions (80 clauses, CUAD 300-399)
- Dataset 4: Adversarial Distractor & Boilerplate Suite (50 clauses, 0 entities)
- Dataset 5: Hugging Face ContractNER Internet Test Set (158 clauses)

Outputs:
- logs/comprehensive_benchmark_report.json
- Formatted tabular console results with zero emojis.
"""

import json
import os
import re
import time
import torch
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from gliner import GLiNER

torch.set_num_threads(14)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_FINANCIAL_LABELS = [
    "contracting_party",
    "penalty_rate",
    "penalty_condition",
    "liability_cap",
    "monetary_amount",
    "effective_date",
    "expiration_date",
    "termination_clause",
    "governing_law",
    "payment_terms",
    "sla_target",
    "grace_period",
]

CONTRACTNER_MAPPING = {
    "Parties": "contracting_party",
    "EffectiveDate": "effective_date",
    "TerminationDate": "expiration_date",
    "RenewalTerm": "termination_clause",
    "Price": "monetary_amount",
    "Principal": "monetary_amount",
    "Rent": "monetary_amount",
    "Salary": "monetary_amount",
    "Percentage": "penalty_rate",
    "Shares": "monetary_amount",
    "Court": "governing_law",
    "Regulation": "governing_law",
    "Act": "governing_law",
}


class HeuristicBaselineExtractor:
    """Regex and pattern matcher baseline for legal/financial clauses."""

    def extract(self, text: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        entities = []
        # Monetary amounts & caps
        for m in re.finditer(r"\$\s*[\d,]+(?:\.\d+)?(?:\s*(?:USD|million|billion|dollars))?", text, re.IGNORECASE):
            is_cap = "cap" in text[max(0, m.start()-25):m.end()+25].lower() or "exceed" in text[max(0, m.start()-25):m.end()+25].lower()
            label = "liability_cap" if is_cap else "monetary_amount"
            entities.append({
                "text": m.group(0),
                "label": label,
                "score": 0.80,
                "start": m.start(),
                "end": m.end()
            })
        # Percentages / penalty rates
        for m in re.finditer(r"\b\d+(?:\.\d+)?%\s*(?:per\s+\w+|annum|monthly)?", text, re.IGNORECASE):
            entities.append({
                "text": m.group(0),
                "label": "penalty_rate",
                "score": 0.75,
                "start": m.start(),
                "end": m.end()
            })
        # Governing law mentions
        for m in re.finditer(r"laws of (?:the State of\s+)?[A-Za-z\s]+|State of [A-Za-z]+ law", text, re.IGNORECASE):
            entities.append({
                "text": m.group(0),
                "label": "governing_law",
                "score": 0.85,
                "start": m.start(),
                "end": m.end()
            })
        # Dates
        for m in re.finditer(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", text):
            surrounding = text[max(0, m.start()-30):m.end()+30].lower()
            is_eff = any(w in surrounding for w in ["effective", "commence", "made as of", "dated as of"])
            label = "effective_date" if is_eff else "expiration_date"
            entities.append({
                "text": m.group(0),
                "label": label,
                "score": 0.70,
                "start": m.start(),
                "end": m.end()
            })
        return entities


def load_dataset_1() -> List[Dict[str, Any]]:
    path = os.path.join(BASE_DIR, "data", "financial_ner_eval.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    samples = []
    for s in raw:
        tokens = s["tokenized_text"]
        raw_ner = s.get("ner", [])
        full_text = ""
        token_char_offsets = []

        for idx, tok in enumerate(tokens):
            if idx > 0 and tok not in [",", ".", ";", ":", "'", ")", "]", "}", "USD"]:
                full_text += " "
            start_char = len(full_text)
            full_text += tok
            end_char = len(full_text)
            token_char_offsets.append((start_char, end_char))

        gold_entities = []
        for start_tok, end_tok, label in raw_ner:
            if start_tok < len(token_char_offsets) and (end_tok - 1) < len(token_char_offsets):
                c_start = token_char_offsets[start_tok][0]
                c_end = token_char_offsets[end_tok - 1][1]
                gold_entities.append({
                    "label": label,
                    "text": full_text[c_start:c_end],
                    "start": c_start,
                    "end": c_end
                })

        samples.append({
            "clause": full_text,
            "entities": gold_entities
        })
    return samples


def load_dataset_generic(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_5_contractner() -> List[Dict[str, Any]]:
    path = os.path.join(BASE_DIR, "data", "internet_datasets", "contract_ner", "test.jsonl")
    if not os.path.exists(path):
        return []

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            tokens = item.get("tokenized_text", [])
            raw_ner = item.get("ner", [])
            if not tokens or not raw_ner:
                continue

            full_text = ""
            token_char_offsets = []
            for idx, tok in enumerate(tokens):
                if idx > 0 and tok not in [",", ".", ";", ":", "'", ")", "]", "}"]:
                    full_text += " "
                start_char = len(full_text)
                full_text += tok
                end_char = len(full_text)
                token_char_offsets.append((start_char, end_char))

            gold_entities = []
            for s_tok, e_tok, raw_lbl in raw_ner:
                if raw_lbl in CONTRACTNER_MAPPING and s_tok < len(token_char_offsets) and (e_tok - 1) < len(token_char_offsets):
                    c_start = token_char_offsets[s_tok][0]
                    c_end = token_char_offsets[e_tok - 1][1]
                    gold_entities.append({
                        "label": CONTRACTNER_MAPPING[raw_lbl],
                        "text": full_text[c_start:c_end],
                        "start": c_start,
                        "end": c_end
                    })

            if gold_entities:
                samples.append({
                    "clause": full_text,
                    "entities": gold_entities
                })
    return samples


def span_matches(pred: Dict[str, Any], gold: Dict[str, Any], relaxed: bool = True) -> bool:
    if pred["label"].lower() != gold["label"].lower():
        return False
    p_start, p_end = int(pred["start"]), int(pred["end"])
    g_start, g_end = int(gold["start"]), int(gold["end"])

    if not relaxed:
        return p_start == g_start and p_end == g_end

    pred_text = pred["text"].strip().lower()
    gold_text = gold["text"].strip().lower()
    if pred_text == gold_text:
        return True
    if pred_text in gold_text or gold_text in pred_text:
        return True
    overlap = max(0, min(p_end, g_end) - max(p_start, g_start))
    return overlap > 0


def evaluate_entity_dataset(
    model_name: str,
    model_obj: Any,
    samples: List[Dict[str, Any]],
    labels: List[str],
    threshold: float = 0.40,
    relaxed_match: bool = True
) -> Dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    latencies = []
    class_stats = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in labels}

    for sample in samples:
        text = sample["clause"]
        gold_entities = sample["entities"]

        t0 = time.perf_counter()
        if hasattr(model_obj, "predict_entities"):
            preds = model_obj.predict_entities(text, labels, threshold=threshold, flat_ner=True)
        else:
            preds = model_obj.extract(text, threshold=threshold)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        matched_golds = set()
        matched_preds = set()

        for p_idx, pred in enumerate(preds):
            matched = False
            for g_idx, gold in enumerate(gold_entities):
                if g_idx not in matched_golds and span_matches(pred, gold, relaxed=relaxed_match):
                    matched = True
                    matched_golds.add(g_idx)
                    matched_preds.add(p_idx)
                    total_tp += 1
                    lbl = gold["label"]
                    if lbl in class_stats:
                        class_stats[lbl]["tp"] += 1
                    break
            if not matched:
                total_fp += 1
                lbl = pred["label"]
                if lbl in class_stats:
                    class_stats[lbl]["fp"] += 1

        for g_idx, gold in enumerate(gold_entities):
            if g_idx not in matched_golds:
                total_fn += 1
                lbl = gold["label"]
                if lbl in class_stats:
                    class_stats[lbl]["fn"] += 1

    precision = total_tp / max(1, (total_tp + total_fp))
    recall = total_tp / max(1, (total_tp + total_fn))
    micro_f1 = (2 * precision * recall) / max(1e-6, (precision + recall))

    macro_f1s = []
    for lbl, stats in class_stats.items():
        if (stats["tp"] + stats["fn"]) > 0:
            p = stats["tp"] / max(1, (stats["tp"] + stats["fp"]))
            r = stats["tp"] / max(1, (stats["tp"] + stats["fn"]))
            f1 = (2 * p * r) / max(1e-6, (p + r))
            macro_f1s.append(f1)
    macro_f1 = sum(macro_f1s) / max(1, len(macro_f1s))

    latencies_sorted = sorted(latencies)
    p50_lat = latencies_sorted[int(0.50 * len(latencies_sorted))]
    p95_lat = latencies_sorted[int(0.95 * len(latencies_sorted))]

    return {
        "model_name": model_name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "total_gold": sum(len(s["entities"]) for s in samples),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_clauses_sec": round(1000.0 / (sum(latencies) / max(1, len(latencies))), 1),
        "class_breakdown": class_stats
    }


def evaluate_adversarial_dataset(
    model_name: str,
    model_obj: Any,
    samples: List[Dict[str, Any]],
    labels: List[str],
    threshold: float = 0.40
) -> Dict[str, Any]:
    total_hallucinations = 0
    clean_clauses = 0
    scores = []
    latencies = []

    for sample in samples:
        text = sample["clause"]
        t0 = time.perf_counter()
        if hasattr(model_obj, "predict_entities"):
            preds = model_obj.predict_entities(text, labels, threshold=threshold, flat_ner=True)
        else:
            preds = model_obj.extract(text, threshold=threshold)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        if len(preds) == 0:
            clean_clauses += 1
        else:
            total_hallucinations += len(preds)
            for p in preds:
                scores.append(float(p.get("score", 0.0)))

    clean_rate = clean_clauses / max(1, len(samples))
    mean_score = sum(scores) / max(1, len(scores)) if scores else 0.0

    return {
        "model_name": model_name,
        "total_clauses": len(samples),
        "clean_clauses": clean_clauses,
        "clean_clause_rate": round(clean_rate, 4),
        "total_hallucinations": total_hallucinations,
        "mean_hallucination_confidence": round(mean_score, 4),
        "avg_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2)
    }


def print_table(title: str, headers: List[str], rows: List[List[Any]]):
    print("\n" + "=" * 90)
    print(f" {title.upper()}")
    print("=" * 90)
    col_widths = [len(h) for h in headers]
    for r in rows:
        for idx, val in enumerate(r):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for r in rows:
        row_line = " | ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(r))
        print(row_line)
    print("=" * 90)


def main():
    print("=" * 90)
    print(" MULTI-DATASET COMPREHENSIVE BENCHMARK: V1 VS V2 (INTERNET TRAINED)")
    print("=" * 90)

    # 1. Load Datasets
    print("\n[1] Loading 5 benchmark datasets...")
    d1_samples = load_dataset_1()
    d2_samples = load_dataset_generic(os.path.join(BASE_DIR, "data", "cuad_commercial_100.json"))
    d3_samples = load_dataset_generic(os.path.join(BASE_DIR, "data", "cuad_high_risk_liability_80.json"))
    d4_samples = load_dataset_generic(os.path.join(BASE_DIR, "data", "adversarial_distractors_50.json"))
    d5_samples = load_dataset_5_contractner()

    print(f"  Dataset 1 (In-Domain Auditing):         {len(d1_samples)} clauses, {sum(len(s['entities']) for s in d1_samples)} gold entities")
    print(f"  Dataset 2 (SEC Commercial Contracts):    {len(d2_samples)} clauses, {sum(len(s['entities']) for s in d2_samples)} gold entities")
    print(f"  Dataset 3 (High-Risk Liability Clauses): {len(d3_samples)} clauses, {sum(len(s['entities']) for s in d3_samples)} gold entities")
    print(f"  Dataset 4 (Adversarial Distractors):     {len(d4_samples)} clauses, 0 gold entities (Noise rejection test)")
    print(f"  Dataset 5 (ContractNER Internet Test):   {len(d5_samples)} clauses, {sum(len(s['entities']) for s in d5_samples)} gold entities")

    # 2. Instantiate Models
    print("\n[2] Initializing benchmark candidates...")
    heuristic_model = HeuristicBaselineExtractor()

    print("  Loading Zero-Shot GLiNER Base ('urchade/gliner_small-v2.1')...")
    zeroshot_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

    print("  Loading Fine-Tuned Financial GLiNER v1 ('models/gliner_financial')...")
    finetuned_v1 = GLiNER.from_pretrained(os.path.join(BASE_DIR, "models", "gliner_financial"))

    print("  Loading Upgraded GLiNER v2 (Internet Trained: 'models/gliner_financial_v2')...")
    finetuned_v2 = GLiNER.from_pretrained(os.path.join(BASE_DIR, "models", "gliner_financial_v2"))

    models = [
        ("Heuristic Baseline (Regex)", heuristic_model),
        ("Zero-Shot GLiNER (Base)", zeroshot_model),
        ("Fine-Tuned GLiNER v1", finetuned_v1),
        ("Upgraded GLiNER v2 (Internet)", finetuned_v2),
    ]

    benchmark_records = {
        "dataset_1_in_domain": [],
        "dataset_2_sec_commercial": [],
        "dataset_3_high_risk_liability": [],
        "dataset_4_adversarial": [],
        "dataset_5_contractner_test": [],
    }

    # 3. Benchmark Dataset 1
    print("\n[3] Benchmarking Dataset 1: In-Domain Financial Evaluation...")
    d1_headers = ["Model", "Precision", "Recall", "Micro F1", "Macro F1", "TP", "FP", "FN", "Avg Latency"]
    d1_rows = []
    for name, m_obj in models:
        res = evaluate_entity_dataset(name, m_obj, d1_samples, DEFAULT_FINANCIAL_LABELS, threshold=0.40)
        benchmark_records["dataset_1_in_domain"].append(res)
        d1_rows.append([
            res["model_name"],
            f"{res['precision']*100:.1f}%",
            f"{res['recall']*100:.1f}%",
            f"{res['micro_f1']*100:.1f}%",
            f"{res['macro_f1']*100:.1f}%",
            res["total_tp"],
            res["total_fp"],
            res["total_fn"],
            f"{res['avg_latency_ms']:.1f} ms"
        ])
    print_table("Dataset 1: In-Domain Financial Auditing Benchmark (30 clauses)", d1_headers, d1_rows)

    # 4. Benchmark Dataset 2
    print("\n[4] Benchmarking Dataset 2: Real SEC Commercial Contracts (CUAD Held-Out)...")
    d2_headers = ["Model", "Precision", "Recall", "Micro F1", "Macro F1", "TP", "FP", "FN", "Avg Latency"]
    d2_rows = []
    for name, m_obj in models:
        res = evaluate_entity_dataset(name, m_obj, d2_samples, DEFAULT_FINANCIAL_LABELS, threshold=0.40)
        benchmark_records["dataset_2_sec_commercial"].append(res)
        d2_rows.append([
            res["model_name"],
            f"{res['precision']*100:.1f}%",
            f"{res['recall']*100:.1f}%",
            f"{res['micro_f1']*100:.1f}%",
            f"{res['macro_f1']*100:.1f}%",
            res["total_tp"],
            res["total_fp"],
            res["total_fn"],
            f"{res['avg_latency_ms']:.1f} ms"
        ])
    print_table("Dataset 2: Real SEC Commercial Contracts Benchmark (100 clauses, 180 entities)", d2_headers, d2_rows)

    # 5. Benchmark Dataset 3
    print("\n[5] Benchmarking Dataset 3: High-Risk Liability & Governance Clauses...")
    d3_headers = ["Model", "Precision", "Recall", "Micro F1", "Macro F1", "TP", "FP", "FN", "Avg Latency"]
    d3_rows = []
    for name, m_obj in models:
        res = evaluate_entity_dataset(name, m_obj, d3_samples, DEFAULT_FINANCIAL_LABELS, threshold=0.40)
        benchmark_records["dataset_3_high_risk_liability"].append(res)
        d3_rows.append([
            res["model_name"],
            f"{res['precision']*100:.1f}%",
            f"{res['recall']*100:.1f}%",
            f"{res['micro_f1']*100:.1f}%",
            f"{res['macro_f1']*100:.1f}%",
            res["total_tp"],
            res["total_fp"],
            res["total_fn"],
            f"{res['avg_latency_ms']:.1f} ms"
        ])
    print_table("Dataset 3: High-Risk Liability & Governance Benchmark (80 clauses, 91 entities)", d3_headers, d3_rows)

    # 6. Benchmark Dataset 4
    print("\n[6] Benchmarking Dataset 4: Adversarial Distractors & Legal Boilerplate (Stress-Test)...")
    d4_headers = ["Model", "Clauses", "Clean Clauses", "Clean Rate", "False Positives", "Mean Conf", "Latency"]
    d4_rows = []
    for name, m_obj in models:
        res = evaluate_adversarial_dataset(name, m_obj, d4_samples, DEFAULT_FINANCIAL_LABELS, threshold=0.40)
        benchmark_records["dataset_4_adversarial"].append(res)
        d4_rows.append([
            res["model_name"],
            res["total_clauses"],
            res["clean_clauses"],
            f"{res['clean_clause_rate']*100:.1f}%",
            res["total_hallucinations"],
            f"{res['mean_hallucination_confidence']:.2f}",
            f"{res['avg_latency_ms']:.1f} ms"
        ])
    print_table("Dataset 4: Adversarial Distractor Rejection Test (50 clauses, 0 entities)", d4_headers, d4_rows)

    # 7. Benchmark Dataset 5 (Hugging Face ContractNER Test Set)
    print("\n[7] Benchmarking Dataset 5: Hugging Face ContractNER Test Set (158 clauses)...")
    d5_headers = ["Model", "Precision", "Recall", "Micro F1", "Macro F1", "TP", "FP", "FN", "Avg Latency"]
    d5_rows = []
    for name, m_obj in models:
        res = evaluate_entity_dataset(name, m_obj, d5_samples, DEFAULT_FINANCIAL_LABELS, threshold=0.40)
        benchmark_records["dataset_5_contractner_test"].append(res)
        d5_rows.append([
            res["model_name"],
            f"{res['precision']*100:.1f}%",
            f"{res['recall']*100:.1f}%",
            f"{res['micro_f1']*100:.1f}%",
            f"{res['macro_f1']*100:.1f}%",
            res["total_tp"],
            res["total_fp"],
            res["total_fn"],
            f"{res['avg_latency_ms']:.1f} ms"
        ])
    print_table("Dataset 5: Hugging Face ContractNER Test Set (158 clauses)", d5_headers, d5_rows)

    # 8. Save report
    out_log_path = os.path.join(BASE_DIR, "logs", "comprehensive_benchmark_report.json")
    with open(out_log_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2)
    print(f"\nSaved complete benchmark metrics to: {out_log_path}")


if __name__ == "__main__":
    main()
