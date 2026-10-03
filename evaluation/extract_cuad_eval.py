import json
import re

with open('data/cuad/CUAD_v1/CUAD_v1.json', 'r', encoding='utf-8') as f:
    cuad = json.load(f)

# Use out-of-sample contracts (contracts 400 to 510)
test_contracts = cuad['data'][400:]

category_mapping = {
    'Parties': 'contracting_party',
    'Effective Date': 'effective_date',
    'Agreement Date': 'effective_date',
    'Expiration Date': 'expiration_date',
    'Governing Law': 'governing_law',
    'Cap On Liability': 'liability_cap',
    'Liquidated Damages': 'penalty_condition',
    'Termination For Convenience': 'termination_clause',
    'Notice Period To Terminate Renewal': 'termination_clause',
}

samples = []
for contract in test_contracts:
    title = contract['title']
    p = contract['paragraphs'][0]
    full_text = p['context']
    
    for qa in p['qas']:
        category = qa['id'].split('__')[-1]
        target_label = category_mapping.get(category)
        if not target_label:
            continue
            
        for ans in qa.get('answers', []):
            start = ans['answer_start']
            ans_text = ans['text'].strip()
            if not ans_text or len(ans_text) > 300:
                continue
                
            # Find sentence/clause boundaries around start
            clause_start = max(0, full_text.rfind('\n', 0, start))
            clause_end = full_text.find('\n', start + len(ans_text))
            if clause_end == -1:
                clause_end = len(full_text)
                
            clause_text = full_text[clause_start:clause_end].strip()
            
            # If clause is too long (> 600 chars), narrow down
            if len(clause_text) > 600:
                s_idx = full_text.rfind('. ', max(0, start - 200), start)
                if s_idx != -1:
                    clause_start = s_idx + 2
                e_idx = full_text.find('. ', start + len(ans_text), start + len(ans_text) + 200)
                if e_idx != -1:
                    clause_end = e_idx + 1
                clause_text = full_text[clause_start:clause_end].strip()
                
            # Verify ans_text is in clause_text
            pos = clause_text.find(ans_text)
            if pos != -1 and 30 < len(clause_text) < 800:
                samples.append({
                    "contract": title,
                    "clause": clause_text,
                    "entities": [
                        {
                            "label": target_label,
                            "text": ans_text,
                            "start": pos,
                            "end": pos + len(ans_text)
                        }
                    ],
                    "cuad_category": category
                })

print(f"Extracted {len(samples)} valid real-world contract clauses across test contracts.")

# Count per label
from collections import Counter
counts = Counter(s['entities'][0]['label'] for s in samples)
print("Distribution per label:")
for l, c in counts.items():
    print(f"  {l}: {c}")

# Let's save a balanced 50-clause real-world test set
balanced_samples = []
per_label_limit = 10
selected_counts = Counter()

for s in samples:
    label = s['entities'][0]['label']
    # Check if clause isn't already added
    if selected_counts[label] < per_label_limit and not any(b['clause'] == s['clause'] for b in balanced_samples):
        # Scan if clause also contains any of the other entities from the contract
        balanced_samples.append(s)
        selected_counts[label] += 1

print(f"\nFinal balanced real-world evaluation dataset size: {len(balanced_samples)}")
with open('data/cuad_realworld_eval.json', 'w', encoding='utf-8') as f:
    json.dump(balanced_samples, f, indent=2)
print("Saved to data/cuad_realworld_eval.json")
