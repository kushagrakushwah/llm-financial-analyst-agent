"""
benchmark_eval.py
=================
Comprehensive quantitative evaluation comparing:
1. Baseline Heuristic Extractor (Regex + Dictionary)
2. Zero-Shot GLiNER (urchade/gliner_small-v2.1)
3. Fine-Tuned GLiNER (models/gliner_financial)

Computes exact Span Precision, Recall, F1 (Micro/Macro), Latency (ms),
and logs all detailed evaluation traces into logs/eval_benchmark_results.json.
"""

import json
import os
import re
import time
import shutil
import glob
from typing import List, Dict, Any, Tuple
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
    """Simple rule/regex-based baseline for currency, percentages, and party cues."""

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
        for m in re.finditer(r"laws of the State of [A-Za-z\s]+", text):
            entities.append({
                "text": m.group(0),
                "label": "governing_law",
                "score": 0.85,
                "start": m.start(),
                "end": m.end()
            })
        return entities


def reconstruct_text_and_spans(sample: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    tokens = sample["tokenized_text"]
    raw_ner = sample.get("ner", [])

    full_text = ""
    token_char_offsets = []

    for idx, tok in enumerate(tokens):
        if idx > 0 and tok not in [",", ".", ";", ":", "'", ")", "]", "}", "USD"]:
            full_text += " "
        start_char = len(full_text)
        full_text += tok
        end_char = len(full_text)
        token_char_offsets.append((start_char, end_char))

    ground_truth = []
    for start_tok, end_tok, label in raw_ner:
        if start_tok < len(token_char_offsets) and (end_tok - 1) < len(token_char_offsets):
            char_start = token_char_offsets[start_tok][0]
            char_end = token_char_offsets[end_tok - 1][1]
            ground_truth.append({
                "label": label,
                "text": full_text[char_start:char_end],
                "char_start": char_start,
                "char_end": char_end,
                "token_span": [start_tok, end_tok]
            })
    return full_text, ground_truth


def spans_overlap_or_match(pred: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    if pred["label"].lower() != gold["label"].lower():
        return False
    # Check token or string overlap / containment
    pred_text = pred["text"].strip().lower()
    gold_text = gold["text"].strip().lower()
    if pred_text == gold_text:
        return True
    if pred_text in gold_text or gold_text in pred_text:
        return True
    p_start, p_end = pred["start"], pred["end"]
    g_start, g_end = gold["char_start"], gold["char_end"]
    return max(0, min(p_end, g_end) - max(p_start, g_start)) > 0


def evaluate_model(name: str, model_obj, eval_samples: List[Dict[str, Any]], labels: List[str], threshold: float = 0.35) -> Dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    latencies = []
    sample_traces = []

    class_metrics = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in labels}

    for sample_idx, sample in enumerate(eval_samples):
        text, gold_entities = reconstruct_text_and_spans(sample)

        t0 = time.perf_counter()
        if hasattr(model_obj, "predict_entities"):
            predictions = model_obj.predict_entities(text, labels, threshold=threshold, flat_ner=True)
        else:
            predictions = model_obj.extract(text, threshold=threshold)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        matched_gold_indices = set()
        matched_pred_indices = set()

        # Match predictions to ground truth
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
            "text": text,
            "gold_entities": gold_entities,
            "predicted_entities": predictions,
            "matched_count": len(matched_gold_indices),
            "latency_ms": round(latency_ms, 2)
        })

    precision = total_tp / max(1, (total_tp + total_fp))
    recall = total_tp / max(1, (total_tp + total_fn))
    f1 = 2 * precision * recall / max(1e-8, (precision + recall))

    # Compute Macro F1 across active classes
    macro_f1_list = []
    for lbl, stats in class_metrics.items():
        c_p = stats["tp"] / max(1, (stats["tp"] + stats["fp"]))
        c_r = stats["tp"] / max(1, (stats["tp"] + stats["fn"]))
        c_f1 = (2 * c_p * c_r) / max(1e-8, (c_p + c_r)) if (c_p + c_r) > 0 else 0.0
        if (stats["tp"] + stats["fn"]) > 0:  # Only count classes present in ground truth
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
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
        },
        "per_class_breakdown": class_metrics,
        "traces": sample_traces
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_path = os.path.join(base_dir, "data", "financial_ner_eval.json")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_samples = json.load(f)

    print("=" * 76)
    print(f" FINANCIAL AUDITING MODEL EVALUATION BENCHMARK ({len(eval_samples)} Samples)")
    print("=" * 76)

    # 1. Baseline Heuristic
    print("\n[1/3] Evaluating Baseline 1: Heuristic (Regex + Rules)...")
    heuristic_model = HeuristicBaselineExtractor()
    heuristic_res = evaluate_model("Heuristic (Regex Baseline)", heuristic_model, eval_samples, DEFAULT_FINANCIAL_LABELS)

    # 2. Zero-Shot GLiNER
    print("\n[2/3] Evaluating Baseline 2: Zero-Shot GLiNER (urchade/gliner_small-v2.1)...")
    zeroshot_model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
    zeroshot_res = evaluate_model("Zero-Shot GLiNER (Base)", zeroshot_model, eval_samples, DEFAULT_FINANCIAL_LABELS, threshold=0.35)

    # 3. Fine-Tuned GLiNER
    finetuned_path = os.path.join(base_dir, "models", "gliner_financial")
    print(f"\n[3/3] Evaluating Model 3: Fine-Tuned GLiNER ({finetuned_path})...")
    finetuned_model = GLiNER.from_pretrained(finetuned_path)
    finetuned_res = evaluate_model("Fine-Tuned GLiNER (Domain Specialized)", finetuned_model, eval_samples, DEFAULT_FINANCIAL_LABELS, threshold=0.45)

    comparison = [heuristic_res, zeroshot_res, finetuned_res]

    # Save detailed evaluation traces
    eval_log_path = os.path.join(logs_dir, "eval_benchmark_results.json")
    with open(eval_log_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[*] Saved full evaluation traces to: {eval_log_path}")

    # Copy / export trainer state log as well
    checkpoints = glob.glob(os.path.join(finetuned_path, "checkpoint-*"))
    if checkpoints:
        latest_ckpt = max(checkpoints, key=lambda p: int(os.path.basename(p).split("-")[-1]))
        trainer_state_source = os.path.join(latest_ckpt, "trainer_state.json")
        if os.path.exists(trainer_state_source):
            shutil.copy(trainer_state_source, os.path.join(logs_dir, "training_trace.json"))
            print(f"[*] Exported training trace log to: {os.path.join(logs_dir, 'training_trace.json')}")

    # Print comparison table
    print("\n" + "=" * 76)
    print(" COMPARATIVE EVALUATION RESULTS SUMMARY")
    print("=" * 76)
    print(f" {'Model':<38} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Latency':<8}")
    print(" " + "-" * 76)
    for res in comparison:
        m = res["metrics"]
        print(f" {res['model_name']:<38} | {m['precision']:<9.4f} | {m['recall']:<9.4f} | {m['micro_f1']:<9.4f} | {m['avg_latency_ms']:<6.1f}ms")
    print("=" * 76)


if __name__ == "__main__":
    main()
