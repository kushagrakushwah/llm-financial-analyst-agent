"""
prepare_optimized_internet_dataset.py
=====================================
Builds an optimized, high-density training corpus from the internet datasets:
- Agile Lab ContractNER (Hugging Face)
- Stanford Atticus CUAD (SEC Edgar)
- In-domain financial auditing clauses
- Boilerplate distractor clauses (empty NER)

Filters sentences to 15 <= tokens <= 128 to ensure rapid convergence without CPU memory thrashing.
"""

import json
import os
import random
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "data", "large_internet_train.json"), "r", encoding="utf-8") as f:
    raw_train = json.load(f)

# Filter samples by token length (15 to 130 tokens)
valid_length_samples = [s for s in raw_train if 15 <= len(s.get("tokenized_text", [])) <= 130]

# Separate by source / characteristics
with_entities = [s for s in valid_length_samples if len(s.get("ner", [])) > 0]
empty_negatives = [s for s in valid_length_samples if len(s.get("ner", [])) == 0]

print(f"Total length-filtered samples: {len(valid_length_samples)}")
print(f"  Positive samples (with entities): {len(with_entities)}")
print(f"  Empty distractor samples:         {len(empty_negatives)}")

# Ensure balance across key financial and contract entity types
# Prioritize samples containing rare/critical legal classes: liability_cap, penalty_condition, penalty_rate, termination_clause
priority_samples = []
standard_samples = []

for s in with_entities:
    labels = set(lbl for _, _, lbl in s["ner"])
    if labels.intersection({"liability_cap", "penalty_condition", "penalty_rate", "termination_clause", "monetary_amount"}):
        priority_samples.append(s)
    else:
        standard_samples.append(s)

print(f"  Priority financial/liability samples: {len(priority_samples)}")
print(f"  Standard contract samples:            {len(standard_samples)}")

random.seed(42)
random.shuffle(priority_samples)
random.shuffle(standard_samples)
random.shuffle(empty_negatives)

# Assemble an optimal 600-sample training dataset
selected_train = priority_samples[:350] + standard_samples[:200] + empty_negatives[:50]
random.shuffle(selected_train)

# Build a compact 80-sample evaluation dataset from the rest
remaining_pos = [s for s in with_entities if s not in selected_train]
selected_eval = remaining_pos[:70] + empty_negatives[50:60]

print(f"\nFinal Optimized Train Set: {len(selected_train)} samples")
print(f"Final Optimized Eval Set:  {len(selected_eval)} samples")

train_labels = Counter(lbl for s in selected_train for _, _, lbl in s.get("ner", []))
print("\nTrain Set Entity Distribution:")
for k, v in train_labels.most_common():
    print(f"  {k:<20}: {v}")
print(f"  [Empty Distractors] : {sum(1 for s in selected_train if len(s.get('ner', [])) == 0)}")

train_out = os.path.join(BASE_DIR, "data", "optimized_internet_train.json")
eval_out = os.path.join(BASE_DIR, "data", "optimized_internet_eval.json")

with open(train_out, "w", encoding="utf-8") as f:
    json.dump(selected_train, f, indent=2)
with open(eval_out, "w", encoding="utf-8") as f:
    json.dump(selected_eval, f, indent=2)

print(f"\nSaved to {train_out}")
print(f"Saved to {eval_out}")
