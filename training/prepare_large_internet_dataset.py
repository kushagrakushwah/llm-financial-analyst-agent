"""
prepare_large_internet_dataset.py
=================================
Aggregates and standardizes large real-world contract & financial datasets from the internet:
1. agilelab-org/ContractNER_Dataset (Hugging Face Hub: 2,310 real commercial clauses)
2. Stanford Atticus CUAD Training Split (Contracts 0-350: ~450 dense legal clauses)
3. Specialized In-Domain Financial Auditing Dataset (162 clauses)
4. Hard Negative Boilerplate & Distractor Clauses (100 clauses with empty NER: [])

Produces:
- data/large_internet_train.json (~1,800-2,500 clauses)
- data/large_internet_eval.json (250 held-out evaluation clauses)
"""

import json
import os
import re
import random
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

CUAD_MAPPING = {
    "Parties": "contracting_party",
    "Effective Date": "effective_date",
    "Agreement Date": "effective_date",
    "Expiration Date": "expiration_date",
    "Governing Law": "governing_law",
    "Cap On Liability": "liability_cap",
    "Liquidated Damages": "penalty_condition",
    "Termination For Convenience": "termination_clause",
    "Notice Period To Terminate Renewal": "termination_clause",
    "Minimum Commitment": "monetary_amount",
    "Revenue/Profit Sharing": "penalty_rate",
}


def tokenize_with_char_offsets(text):
    tokens = []
    offsets = []
    for m in re.finditer(r"\w+|[^\w\s]", text):
        tokens.append(m.group(0))
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def char_span_to_token_span(offsets, start_char, end_char):
    tok_start = None
    tok_end = None
    for idx, (s, e) in enumerate(offsets):
        if tok_start is None and e > start_char:
            tok_start = idx
        if s < end_char:
            tok_end = idx + 1
    if tok_start is not None and tok_end is not None and tok_start < tok_end:
        return [tok_start, tok_end]
    return None


def load_contractner():
    path = os.path.join(BASE_DIR, "data", "internet_datasets", "contract_ner", "train.jsonl")
    samples = []
    if not os.path.exists(path):
        print(f"ContractNER path {path} not found.")
        return samples

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            toks = item.get("tokenized_text", [])
            raw_ner = item.get("ner", [])
            if not toks or not raw_ner:
                continue

            mapped_ner = []
            for s, e, l in raw_ner:
                if l in CONTRACTNER_MAPPING and s < len(toks) and e <= len(toks) and s < e:
                    mapped_ner.append([s, e, CONTRACTNER_MAPPING[l]])

            if mapped_ner:
                samples.append({
                    "tokenized_text": toks,
                    "ner": mapped_ner,
                    "source": "contract_ner_huggingface"
                })
    return samples


def extract_cuad_training(max_samples=500):
    cuad_path = os.path.join(BASE_DIR, "data", "cuad", "CUAD_v1", "CUAD_v1.json")
    if not os.path.exists(cuad_path):
        return []

    with open(cuad_path, "r", encoding="utf-8") as f:
        cuad = json.load(f)

    # Use strictly training contracts 0 to 350
    contracts = cuad["data"][:350]
    extracted = []
    seen = set()

    for contract in contracts:
        p = contract["paragraphs"][0]
        context = p["context"]

        annots = []
        for qa in p["qas"]:
            cat = qa["id"].split("__")[-1]
            mapped_lbl = CUAD_MAPPING.get(cat)
            if not mapped_lbl:
                continue
            for ans in qa.get("answers", []):
                t = ans["text"].strip()
                if not t or len(t) > 300:
                    continue
                annots.append({
                    "start": ans["answer_start"],
                    "end": ans["answer_start"] + len(t),
                    "text": t,
                    "label": mapped_lbl
                })

        for a in annots:
            s_char = a["start"]
            e_char = a["end"]

            c_s = max(0, context.rfind("\n", 0, s_char))
            c_e = context.find("\n", e_char)
            if c_e == -1:
                c_e = len(context)

            if (c_e - c_s) > 600:
                s_idx = context.rfind(". ", max(0, s_char - 200), s_char)
                if s_idx != -1:
                    c_s = s_idx + 2
                e_idx = context.find(". ", e_char, min(len(context), e_char + 200))
                if e_idx != -1:
                    c_e = e_idx + 1

            clause_str = context[c_s:c_e].strip()
            if not (40 <= len(clause_str) <= 650):
                continue

            norm = re.sub(r"\s+", " ", clause_str)
            if norm in seen:
                continue

            tokens, offsets = tokenize_with_char_offsets(clause_str)
            if len(tokens) > 250:
                continue

            clause_ner = []
            for other_a in annots:
                if other_a["start"] >= c_s and other_a["end"] <= c_e:
                    rel_s = other_a["start"] - c_s
                    rel_e = other_a["end"] - c_s
                    t_span = char_span_to_token_span(offsets, rel_s, rel_e)
                    if t_span:
                        clause_ner.append([t_span[0], t_span[1], other_a["label"]])

            if clause_ner:
                seen.add(norm)
                extracted.append({
                    "tokenized_text": tokens,
                    "ner": clause_ner,
                    "source": "stanford_cuad_v1"
                })

            if len(extracted) >= max_samples:
                break
        if len(extracted) >= max_samples:
            break

    return extracted


def load_in_domain_train():
    path = os.path.join(BASE_DIR, "data", "financial_ner_train.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for d in data:
        d["source"] = "in_domain_financial"
    return data


def build_hard_negatives(count=100):
    boilerplate = [
        "IN WITNESS WHEREOF the parties hereto have executed this Agreement by their duly authorized representatives.",
        "This Agreement shall be binding upon and inure to the benefit of the respective successors and permitted assigns.",
        "The headings in this Agreement are inserted for convenience of reference only and shall not affect interpretation.",
        "This Agreement may be executed in one or more counterparts each of which shall be deemed an original instrument.",
        "If any provision of this Agreement is held to be invalid or unenforceable such provision shall be severed cleanly.",
        "Neither party shall be liable for failure or delay caused by acts of God, war, riot, fire, flood, or earthquake.",
        "All notices requests and demands hereunder shall be in writing and delivered personally or by certified courier.",
        "This Agreement constitutes the entire understanding between the parties concerning subject matter hereof.",
        "Nothing contained herein shall be deemed to create an agency partnership joint venture or fiduciary relationship.",
        "The waiver by either party of any breach shall not operate or be construed as a waiver of any subsequent breach.",
        "Any amendment or modification must be in writing and signed by authorized representatives of both parties.",
        "Provisions which by their nature should survive termination shall survive any expiration of this Agreement.",
        "Each party represents and warrants that it has full corporate power to perform all obligations hereunder.",
        "No remedy made available in this Agreement is intended to be exclusive and every remedy shall be cumulative.",
        "All schedules exhibits and attachments are incorporated into and made an integral part of this Agreement.",
        "Neither party may assign or transfer rights or obligations without prior written consent of the other party.",
        "The failure of either party to enforce any provision shall not be construed as a waiver of future enforcement.",
        "This Agreement is the product of negotiation and shall not be construed against either party as the drafter.",
        "Confidential Information shall remain the sole and exclusive property of the disclosing party at all times.",
        "The receiving party shall protect all Proprietary Information using the same degree of care as its own data."
    ]

    samples = []
    for i in range(count):
        text = boilerplate[i % len(boilerplate)]
        if i >= len(boilerplate):
            text += f" Reference clause appendix {i+1} section general terms."
        tokens, _ = tokenize_with_char_offsets(text)
        samples.append({
            "tokenized_text": tokens,
            "ner": [],
            "source": "hard_negative_boilerplate"
        })
    return samples


def main():
    print("=" * 80)
    print("Preparing Large Internet Multi-Corpus Dataset for GLiNER Training")
    print("=" * 80)

    # 1. ContractNER (Hugging Face)
    print("\n[1] Loading ContractNER from Hugging Face...")
    contractner_data = load_contractner()
    print(f"Loaded {len(contractner_data)} samples from ContractNER.")

    # 2. Stanford CUAD (SEC Edgar)
    print("\n[2] Extracting dense training clauses from Stanford CUAD...")
    cuad_data = extract_cuad_training(max_samples=500)
    print(f"Extracted {len(cuad_data)} samples from CUAD.")

    # 3. In-Domain Financial
    print("\n[3] Loading specialized In-Domain Financial training data...")
    in_domain_data = load_in_domain_train()
    print(f"Loaded {len(in_domain_data)} in-domain samples.")

    # 4. Hard Negatives (Boilerplate with empty NER)
    print("\n[4] Generating hard negative boilerplate clauses (ner: [])...")
    negatives = build_hard_negatives(count=120)
    print(f"Generated {len(negatives)} negative distractor samples.")

    # Combine all
    all_samples = contractner_data + cuad_data + in_domain_data + negatives
    random.seed(42)
    random.shuffle(all_samples)
    print(f"\nTotal combined corpus size: {len(all_samples)} samples")

    # Label statistics
    lbl_counts = Counter(l for s in all_samples for _, _, l in s.get("ner", []))
    print("\nOverall Label Distribution across Combined Corpus:")
    for k, v in lbl_counts.most_common():
        print(f"  {k:<20}: {v}")
    print(f"  [Empty Negative Samples] : {len(negatives)}")

    # Split train and validation (90% / 10%)
    split_idx = int(0.90 * len(all_samples))
    train_split = all_samples[:split_idx]
    eval_split = all_samples[split_idx:]

    print(f"\nFinal Train Split: {len(train_split)} samples")
    print(f"Final Eval Split:  {len(eval_split)} samples")

    train_out = os.path.join(BASE_DIR, "data", "large_internet_train.json")
    eval_out = os.path.join(BASE_DIR, "data", "large_internet_eval.json")

    with open(train_out, "w", encoding="utf-8") as f:
        json.dump(train_split, f, indent=2)
    with open(eval_out, "w", encoding="utf-8") as f:
        json.dump(eval_split, f, indent=2)

    print(f"\nSaved train dataset: {train_out}")
    print(f"Saved eval dataset:  {eval_out}")


if __name__ == "__main__":
    main()
