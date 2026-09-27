import json

with open("logs/eval_benchmark_results.json", "r", encoding="utf-8") as f:
    eval_data = json.load(f)

print("=" * 65)
print(" DETAILED LOG & ERROR TRACE ANALYSIS")
print("=" * 65)

for model in eval_data:
    m = model["metrics"]
    print(f"\nModel: {model['model_name']}")
    print(f"  Overall: Precision={m['precision']:.3f}, Recall={m['recall']:.3f}, F1={m['micro_f1']:.3f}, Latency={m['avg_latency_ms']:.1f}ms")
    print("  Per-Class Performance:")
    for cls_name, counts in model["per_class_breakdown"].items():
        if counts["tp"] > 0 or counts["fp"] > 0 or counts["fn"] > 0:
            tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
            prec = tp / max(1, (tp + fp))
            rec = tp / max(1, (tp + fn))
            f1 = 2 * prec * rec / max(1e-8, (prec + rec))
            print(f"    - {cls_name:<20}: TP={tp}, FP={fp}, FN={fn} | P={prec:.2f}, R={rec:.2f}, F1={f1:.2f}")

    print("  Sample Error Trace Highlights:")
    for trace in model["traces"][:3]:
        print(f"    [Sample #{trace['sample_index']}] \"{trace['text'][:60]}...\"")
        print(f"      Gold: {[g['label'] + ':' + g['text'] for g in trace['gold_entities']]}")
        print(f"      Pred: {[p['label'] + ':' + p['text'] for p in trace['predicted_entities']]}")

# Training loss trace analysis
print("\n" + "=" * 65)
print(" TRAINING TRACE LOG ANALYSIS (logs/training_trace.json)")
print("=" * 65)
try:
    with open("logs/training_trace.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    epochs = {}
    eval_epochs = {}
    for entry in train_data.get("log_history", []):
        if "loss" in entry:
            ep = round(entry["epoch"], 1)
            epochs[ep] = entry
        if "eval_loss" in entry:
            ep = round(entry["epoch"], 1)
            eval_epochs[ep] = entry

    print(f"Total Steps: {train_data.get('global_step')}, Epochs: {train_data.get('epoch')}")
    print("Loss Progression across Epochs:")
    for ep in sorted(set(list(epochs.keys()) + list(eval_epochs.keys()))):
        t_loss = f"{epochs[ep]['loss']:.2f}" if ep in epochs else "N/A"
        lr = f"{epochs[ep]['learning_rate']:.2e}" if ep in epochs else "N/A"
        e_loss = f"{eval_epochs[ep]['eval_loss']:.2f}" if ep in eval_epochs else "N/A"
        print(f"  Epoch {ep:>3.1f} | Train Loss: {t_loss:>6} | Eval Loss: {e_loss:>6} | Learning Rate: {lr}")
except Exception as e:
    print(f"Error reading training trace: {e}")
