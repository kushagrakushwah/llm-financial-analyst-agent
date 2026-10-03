"""
benchmark_cuad_realworld.py
============================
Evaluates out-of-distribution performance on genuine SEC commercial contract clauses
from the Atticus Project (CUAD - Contract Understanding Atticus Dataset).

Evaluates:
1. Heuristic Baseline (Regex + Rules)
2. Zero-Shot GLiNER (urchade/gliner_small-v2.1)
3. Fine-Tuned Financial GLiNER (models/gliner_financial/checkpoint-328)

Computes Precision, Recall, Micro F1, Macro F1, and Latency on real-world legal text.
Logs complete traces to logs/cuad_realworld_benchmark_results.json.
"""

import json
import os
import re
import time
import glob
from typing import List, Dict, Any
from gliner import GLiNER

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


class HeuristicBaselineExtractor:
    """Regex + pattern matcher baseline for real-world legal text."""

    def extract(self, text: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        entities = []
        # Monetary amounts / caps
        for m in re.finditer(r"\$\s*[\d,]+(?:\.\d+)?(?:\s*USD)?", text, re.IGNORECASE):
            label = "liability_cap" if "cap" in text[max(0, m.start()-20):m.end()+20].lower() else "monetary_amount"
            entities.append({
                "text": m.group(0),
                "label": label,
                "score": 0.80,
                "start": m.start(),
                "end": m.end()
            })
        # Percentage / penalty rates
        for m in re.finditer(r"\b\d+(?:\.\d+)?%\s*(?:per\s+\w+|monthly)?", text, re.IGNORECASE):
            entities.append({
                "text": m.group(0),
                "label": "penalty_rate",
                "score": 0.75,
                "start": m.start(),
                "end": m.end()
            })
        # Common law mentions
        for m in re.finditer(r"laws of the State of [A-Za-z\s]+|State of [A-Za-z]+ law", text, re.IGNORECASE):
            entities.append({
                "text": m.group(0),
                "label": "governing_law",
                "score": 0.85,
                "start": m.start(),
                "end": m.end()
            })
        # Dates
        for m in re.finditer(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b", text):
            label = "effective_date" if any(w in text[max(0, m.start()-30):m.end()+30].lower() for w in ["effective", "commence", "made as of"]) else "expiration_date"
            entities.append({
                "text": m.group(0),
                "label": label,
                "score": 0.70,
                "start": m.start(),
                "end": m.end()
            })
        return entities


def spans_overlap_or_match(pred: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    if pred["label"].lower() != gold["label"].lower():
        return False
    pred_text = pred["text"].strip().lower()
    gold_text = gold["text"].strip().lower()
    if pred_text == gold_text:
        return True
    if pred_text in gold_text or gold_text in pred_text:
        return True
    p_start, p_end = pred["start"], pred["end"]
    g_start, g_end = gold["start"], gold["end"]
    return max(0, min(p_end, g_end) - max(p_start, g_start)) > 0


def evaluate_model(name: str, model_obj, eval_samples: List[Dict[str, Any]], labels: List[str], threshold: float = 0.35) -> Dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    latencies = []
    sample_traces = []

    class_metrics = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in labels}

    for sample_idx, sample in enumerate(eval_samples):
        text = sample["clause"]
        gold_entities = sample["entities"]

        t0 = time.perf_counter()
        if hasattr(model_obj, "predict_entities"):
            predictions = model_obj.predict_entities(text, labels, threshold=threshold, flat_ner=True)
        else:
            predictions = model_obj.extract(text, threshold=threshold)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        matched_gold_indices = set()
        matched_pred_indices = set()

        for p_idx, pred in enumerate(predictions):
            matched = False
            for g_idx, gold in enumerate(gold_entities):
                if g_idx not in matched_gold_indices and spans_overlap_or_match(pred, gold):
                    matched = True
                    matched_gold_indices.add(g_idx)
                    matched_pred_indices.add(p_idx)
                    total_tp += 1
                    lbl = gold["label"]
                    if lbl in class_metrics:
                        class_metrics[lbl]["tp"] += 1
                    break
            if not matched:
                total_fp += 1
                lbl = pred["label"]
                if lbl in class_metrics:
                    class_metrics[lbl]["fp"] += 1

        for g_idx, gold in enumerate(gold_entities):
            if g_idx not in matched_gold_indices:
                total_fn += 1
                lbl = gold["label"]
                if lbl in class_metrics:
                    class_metrics[lbl]["fn"] += 1

        sample_traces.append({
            "sample_index": sample_idx,
            "contract": sample.get("contract", "N/A"),
            "clause": text,
            "gold_entities": gold_entities,
            "predicted_entities": predictions,
            "matched_count": len(matched_gold_indices),
            "latency_ms": round(latency_ms, 2)
        })

    precision = total_tp / max(1, (total_tp + total_fp))
    recall = total_tp / max(1, (total_tp + total_fn))
    f1 = 2 * precision * recall / max(1e-8, (precision + recall))

    macro_f1_list = []
    for lbl, stats in class_metrics.items():
        c_p = stats["tp"] / max(1, (stats["tp"] + stats["fp"]))
        c_r = stats["tp"] / max(1, (stats["tp"] + stats["fn"]))
        c_f1 = (2 * c_p * c_r) / max(1e-8, (c_p + c_r)) if (c_p + c_r) > 0 else 0.0
        if (stats["tp"] + stats["fn"]) > 0:
            macro_f1_list.append(c_f1)
    macro_f1 = sum(macro_f1_list) / max(1, len(macro_f1_list))

    avg_latency = sum(latencies) / max(1, len(latencies))

    return {
        "model_name": name,
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "micro_f1": round(f1, 4),
            "macro_f1": round(macro_f1, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn
        },
        "per_class_metrics": class_metrics,
        "sample_traces": sample_traces
    }


def main():
    print("=" * 80)
    print("REAL-WORLD CUAD DATASET BENCHMARK EVALUATION")
    print("Dataset: Atticus Project CUAD (SEC Edgar Out-of-Distribution Test Contracts)")
    print("=" * 80)

    dataset_path = "data/cuad_realworld_eval.json"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Real-world evaluation dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_samples = json.load(f)

    print(f"Loaded {len(eval_samples)} real-world contract clauses.")

    # 1. Baseline Heuristic
    print("\n[1/3] Running Heuristic Baseline on Real CUAD clauses...")
    baseline_model = HeuristicBaselineExtractor()
    res_baseline = evaluate_model("Heuristic Baseline", baseline_model, eval_samples, DEFAULT_FINANCIAL_LABELS, threshold=0.5)

    # 2. Zero-Shot GLiNER
    print("\n[2/3] Loading Zero-Shot GLiNER (urchade/gliner_small-v2.1)...")
    zero_shot_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
    res_zero_shot = evaluate_model("Zero-Shot GLiNER", zero_shot_model, eval_samples, DEFAULT_FINANCIAL_LABELS, threshold=0.35)

    # 3. Fine-Tuned GLiNER
    print("\n[3/3] Loading Fine-Tuned GLiNER...")
    ckpt_dirs = glob.glob("models/gliner_financial/checkpoint-*")
    if ckpt_dirs:
        latest_ckpt = sorted(ckpt_dirs, key=lambda x: int(x.split("-")[-1]))[-1]
    else:
        latest_ckpt = "models/gliner_financial"
    print(f"Using checkpoint: {latest_ckpt}")
    finetuned_model = GLiNER.from_pretrained(latest_ckpt)
    res_finetuned = evaluate_model("Fine-Tuned GLiNER (Ours)", finetuned_model, eval_samples, DEFAULT_FINANCIAL_LABELS, threshold=0.45)

    # Results Table
    print("\n" + "=" * 80)
    print("REAL-WORLD SEC/CUAD TEST RESULTS (OUT-OF-DISTRIBUTION CONTRACT TEXT)")
    print("=" * 80)
    header = f"{'Model':<28} | {'Precision':<10} | {'Recall':<10} | {'Micro F1':<10} | {'Macro F1':<10} | {'Latency (ms)':<12}"
    print(header)
    print("-" * len(header))

    all_results = [res_baseline, res_zero_shot, res_finetuned]
    for res in all_results:
        m = res["metrics"]
        print(f"{res['model_name']:<28} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['micro_f1']:<10.4f} | {m['macro_f1']:<10.4f} | {m['avg_latency_ms']:<12.2f}")
    print("=" * 80)

    # Save to logs
    os.makedirs("logs", exist_ok=True)
    out_file = "logs/cuad_realworld_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "Atticus Project CUAD (SEC Edgar)",
            "num_samples": len(eval_samples),
            "results": [
                {k: v for k, v in r.items() if k != "sample_traces"} for r in all_results
            ],
            "traces": {
                r["model_name"]: r["sample_traces"] for r in all_results
            }
        }, f, indent=2)
    print(f"\nReal-world evaluation traces saved to: {out_file}")


if __name__ == "__main__":
    main()
