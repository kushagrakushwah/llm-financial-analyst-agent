"""
analyze_false_positives.py
==========================
Inspects what Fine-Tuned GLiNER actually predicts on:
1. Dataset 4 (Adversarial distractors & boilerplate)
2. Dataset 2 (Real CUAD contracts)
To see if errors are true hallucinations, un-annotated entities, or boundary mismatches.
"""

import json
import os
from gliner import GLiNER

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "gliner_financial")
model = GLiNER.from_pretrained(model_path)

LABELS = [
    "contracting_party",
    "penalty_rate",
    "penalty_condition",
    "liability_cap",
    "monetary_amount",
    "effective_date",
    "expiration_date",
    "termination_clause",
    "governing_law",
]

# 1. Distractor analysis
print("=" * 80)
print("1. HALLUCINATION ANALYSIS ON DATASET 4 (ZERO-ENTITY BOILERPLATE)")
print("=" * 80)
with open(os.path.join(BASE_DIR, "data", "adversarial_distractors_50.json"), "r", encoding="utf-8") as f:
    d4 = json.load(f)

for idx, s in enumerate(d4[:15]):
    preds = model.predict_entities(s["clause"], LABELS, threshold=0.45)
    if preds:
        print(f"\n[Distractor #{idx+1}] ({s['distractor_type']}):")
        print(f"  Text: \"{s['clause'][:120]}...\"")
        for p in preds:
            print(f"    -> [{p['label']}] \"{p['text']}\" (confidence: {p['score']:.2f})")

# 2. CUAD Dataset 2 false positives inspection
print("\n" + "=" * 80)
print("2. CUAD FALSE POSITIVE INSPECTION (ARE THEY REAL ENTITIES OR FALSE DISCOVERIES?)")
print("=" * 80)
with open(os.path.join(BASE_DIR, "data", "cuad_commercial_100.json"), "r", encoding="utf-8") as f:
    d2 = json.load(f)

for idx, s in enumerate(d2[:10]):
    preds = model.predict_entities(s["clause"], LABELS, threshold=0.45)
    golds = s["entities"]
    print(f"\n[CUAD Clause #{idx+1}]: \"{s['clause'][:100]}...\"")
    print(f"  Gold Entities ({len(golds)}):")
    for g in golds:
        print(f"    * [{g['label']}] \"{g['text']}\"")
    print(f"  Predicted Entities ({len(preds)}):")
    for p in preds:
        print(f"    * [{p['label']}] \"{p['text']}\" (conf: {p['score']:.2f})")
