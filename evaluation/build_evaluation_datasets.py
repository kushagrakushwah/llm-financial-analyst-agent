"""
build_evaluation_datasets.py
============================
Constructs standardized, high-quality evaluation datasets for rigorous GLiNER benchmarking:
1. data/cuad_commercial_100.json: 100 commercial contract clauses from held-out SEC Edgar contracts (CUAD 400-510).
2. data/cuad_high_risk_liability_80.json: 80 high-risk liability and financial clauses (CUAD 300-399).
3. data/adversarial_distractors_50.json: 50 boilerplate & distractor clauses with zero true financial entities.
"""

import json
import os
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUAD_PATH = os.path.join(BASE_DIR, "data", "cuad", "CUAD_v1", "CUAD_v1.json")

CATEGORY_MAPPING = {
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


def extract_multi_entity_clauses(contracts, target_count=100, min_len=40, max_len=750):
    extracted_clauses = []
    seen_texts = set()

    for contract in contracts:
        title = contract["title"]
        paragraph = contract["paragraphs"][0]
        context = paragraph["context"]

        # Collect all valid annotations in this contract
        contract_annotations = []
        for qa in paragraph["qas"]:
            category = qa["id"].split("__")[-1]
            mapped_label = CATEGORY_MAPPING.get(category)
            if not mapped_label:
                continue

            for ans in qa.get("answers", []):
                ans_text = ans["text"].strip()
                if not ans_text:
                    continue
                start = ans["answer_start"]
                end = start + len(ans_text)
                contract_annotations.append({
                    "category": category,
                    "label": mapped_label,
                    "text": ans_text,
                    "start": start,
                    "end": end,
                })

        # Process each annotation as an anchor to extract clause boundaries
        for annot in contract_annotations:
            start = annot["start"]
            end = annot["end"]

            # Sentence/paragraph boundary search
            c_start = max(0, context.rfind("\n", 0, start))
            c_end = context.find("\n", end)
            if c_end == -1:
                c_end = len(context)

            # Refine if clause is too wide
            if (c_end - c_start) > max_len:
                s_idx = context.rfind(". ", max(0, start - 250), start)
                if s_idx != -1:
                    c_start = s_idx + 2
                e_idx = context.find(". ", end, min(len(context), end + 250))
                if e_idx != -1:
                    c_end = e_idx + 1

            clause_str = context[c_start:c_end].strip()
            if not (min_len <= len(clause_str) <= max_len):
                continue

            # Normalized dedup
            norm_str = re.sub(r"\s+", " ", clause_str)
            if norm_str in seen_texts:
                continue

            # Capture ALL annotations located within this clause window
            clause_entities = []
            for other_annot in contract_annotations:
                o_start = other_annot["start"]
                o_end = other_annot["end"]
                if o_start >= c_start and o_end <= c_end:
                    rel_start = o_start - c_start
                    rel_end = o_end - c_start
                    # Verify text matches slice
                    sliced_text = clause_str[rel_start:rel_end]
                    if sliced_text.strip() == other_annot["text"].strip():
                        clause_entities.append({
                            "label": other_annot["label"],
                            "text": other_annot["text"],
                            "start": rel_start,
                            "end": rel_end,
                            "cuad_category": other_annot["category"]
                        })

            if not clause_entities:
                continue

            seen_texts.add(norm_str)
            extracted_clauses.append({
                "contract": title,
                "clause": clause_str,
                "entities": clause_entities,
                "primary_category": annot["category"]
            })

            if len(extracted_clauses) >= target_count:
                return extracted_clauses

    return extracted_clauses


def build_adversarial_distractors(target_count=50):
    boilerplate_texts = [
        "IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the date first written above by their duly authorized representatives.",
        "This Agreement shall be binding upon and inure to the benefit of the parties hereto and their respective successors and permitted assigns.",
        "The headings in this Agreement are inserted for convenience of reference only and shall not affect the interpretation or construction hereof.",
        "This Agreement may be executed in one or more counterparts, each of which shall be deemed an original, but all of which together shall constitute one and the same instrument.",
        "If any provision of this Agreement is held to be invalid or unenforceable, such provision shall be severed and the remaining provisions shall continue in full force and effect.",
        "Neither party shall be liable for any failure or delay in performing its obligations hereunder if such failure or delay is caused by acts of God, war, riot, fire, or earthquake.",
        "All notices, requests, demands and other communications hereunder shall be in writing and shall be deemed to have been duly given if delivered personally or mailed by certified mail.",
        "This Agreement constitutes the entire agreement between the parties concerning the subject matter hereof and supersedes all prior agreements, understandings, negotiations, and discussions.",
        "Nothing contained herein shall be deemed or construed to create an agency, joint venture, partnership, or fiduciary relationship between the parties.",
        "The waiver by either party of a breach of any provision of this Agreement shall not operate or be construed as a waiver of any subsequent breach.",
        "Any amendment or modification of this Agreement must be in writing and signed by an authorized representative of both parties.",
        "The provisions of this Agreement which by their nature should survive termination shall survive any termination or expiration of this Agreement.",
        "Each party represents and warrants that it has full corporate power and authority to enter into and perform its obligations under this Agreement.",
        "The parties agree to cooperate reasonably with each other and execute any additional documents necessary to carry out the purposes of this Agreement.",
        "No remedy made available to either party in this Agreement is intended to be exclusive of any other remedy, and each and every remedy shall be cumulative.",
        "All schedules, exhibits, and attachments referred to herein are incorporated into and made an integral part of this Agreement.",
        "The language used in this Agreement shall be deemed to be the language chosen by the parties to express their mutual intent.",
        "Neither party may assign or transfer its rights or obligations under this Agreement without the prior written consent of the other party.",
        "The failure of either party to enforce at any time any provision of this Agreement shall not be construed to be a waiver of such provision.",
        "This Agreement is the product of negotiation between sophisticated parties and shall not be construed against either party as the drafter.",
    ]

    numeric_distractor_texts = [
        "Please direct all technical inquiries to support at Extension 4022 or via facsimile to (555) 019-2831.",
        "The equipment shall operate within temperature tolerances of 15 to 35 degrees Celsius at relative humidity between 20% and 80%.",
        "Refer to Section 14.3(b)(iv) for procedural requirements regarding the submission of change orders under Schedule C.",
        "All shipments must be routed to Warehouse Dock 4 located at 1200 Industrial Parkway, Suite 300, Dallas, Texas 75201.",
        "Pursuant to Patent Application Serial No. 16/482,901 filed on docket reference 4920-A, inventor rights remain unassigned.",
        "The batch processing server configuration requires a minimum of 64 GB of RAM, 16 CPU cores, and 500 GB NVMe storage.",
        "Deliveries shall arrive no later than 08:30 AM Central Standard Time on Mondays through Thursdays.",
        "The software version 3.2.1-rc4 must pass regression test suite suites 101 through 115 before staging deployment.",
        "Notice must be dispatched with at least 5 business days prior written communication before initiating routine scheduled maintenance.",
        "The container capacity specifications shall adhere to ISO standard 14001:2015 clause 4.4 and section 7.2 requirements.",
        "The test samples shall be incubated for a period of 48 hours at a constant rotational speed of 250 RPM.",
        "Reference invoice number INV-98234-X in all email correspondence regarding purchase order PO-0048172.",
        "The API rate limit is restricted to 100 requests per minute per IP address with a burst allowance of 150 requests.",
        "Field inspections will occur at intervals of 90 days following initial equipment commissioning and handover.",
        "Serial numbers 100482 through 100599 are designated solely for internal quality assurance testing and evaluation.",
        "Please file copy 3 of Form 1099-MISC with the regional accounting department in Room 204.",
        "The protocol requires three consecutive trial runs with a minimum confidence coefficient of 0.95 across 1000 test cases.",
        "Employee ID numbers consist of an alphanumeric string containing 2 uppercase characters followed by 6 numerical digits.",
        "The vehicle fleet must maintain odometer readings below 120,000 miles throughout the operational lifecycle.",
        "Section 8.1(a) supersedes paragraphs 2 and 3 of Appendix B regarding documentation storage protocols.",
        "All customer calls will be recorded for quality monitoring purposes under compliance standard ISO-9001 section 8.",
        "A quorum of at least 5 directors present in person or via telephone conference shall be required for formal votes.",
        "The frequency spectrum allocation encompasses channels 36, 40, 44, and 48 within the 5.2 GHz operational band.",
        "Dispatch confirmation shall be logged via automated EDI transaction set 856 within 2 hours of shipment departure.",
        "The building permit reference number is BLD-2023-09812 registered in Precinct 7.",
        "Filter replacements are mandatory every 3,000 operating hours or 12 months, whichever occurs first.",
        "The standard test report consists of pages 1 through 18 with supplementary raw data tables in Appendix D.",
        "Submissions received after 5:00 PM local time shall be stamped as received on the following business day.",
        "The network latency between data center nodes must remain under 15 milliseconds round-trip time.",
        "Authorized personnel badge holders must re-authenticate credentials every 180 calendar days at Terminal 6.",
    ]

    all_distractors = []
    for i, t in enumerate(boilerplate_texts):
        all_distractors.append({
            "sample_id": f"distractor_boilerplate_{i+1}",
            "clause": t,
            "distractor_type": "legal_boilerplate",
            "entities": []
        })

    for i, t in enumerate(numeric_distractor_texts):
        all_distractors.append({
            "sample_id": f"distractor_numeric_{i+1}",
            "clause": t,
            "distractor_type": "non_financial_numeric",
            "entities": []
        })

    return all_distractors[:target_count]


def main():
    print("=" * 70)
    print("Building Multi-Dataset Benchmark Suites")
    print("=" * 70)

    with open(CUAD_PATH, "r", encoding="utf-8") as f:
        cuad = json.load(f)

    all_contracts = cuad["data"]
    print(f"Total CUAD contracts: {len(all_contracts)}")

    # Dataset 2: Commercial Contracts Test Set (Contracts 400 - 510)
    test_contracts_commercial = all_contracts[400:]
    commercial_clauses = extract_multi_entity_clauses(
        test_contracts_commercial,
        target_count=100,
        min_len=50,
        max_len=800
    )
    print(f"Extracted {len(commercial_clauses)} commercial contract clauses from held-out CUAD contracts 400-510.")

    comm_entities_count = sum(len(c["entities"]) for c in commercial_clauses)
    comm_cats = Counter(e["label"] for c in commercial_clauses for e in c["entities"])
    print(f"  Total gold entities: {comm_entities_count}")
    for k, v in comm_cats.most_common():
        print(f"    {k}: {v}")

    comm_path = os.path.join(BASE_DIR, "data", "cuad_commercial_100.json")
    with open(comm_path, "w", encoding="utf-8") as f:
        json.dump(commercial_clauses, f, indent=2)
    print(f"Saved: {comm_path}")

    # Dataset 3: High-Risk Liability & Governance (Contracts 300 - 399)
    # Target high-risk categories specifically: Cap On Liability, Liquidated Damages, Minimum Commitment, Governing Law
    high_risk_contracts = all_contracts[300:400]
    high_risk_clauses = []
    seen_high_risk = set()

    for contract in high_risk_contracts:
        title = contract["title"]
        paragraph = contract["paragraphs"][0]
        context = paragraph["context"]

        for qa in paragraph["qas"]:
            category = qa["id"].split("__")[-1]
            if category not in ["Cap On Liability", "Liquidated Damages", "Minimum Commitment", "Governing Law", "Termination For Convenience"]:
                continue
            mapped_label = CATEGORY_MAPPING.get(category)
            for ans in qa.get("answers", []):
                ans_text = ans["text"].strip()
                if not ans_text or len(ans_text) > 350:
                    continue
                start = ans["answer_start"]
                end = start + len(ans_text)

                c_start = max(0, context.rfind("\n", 0, start))
                c_end = context.find("\n", end)
                if c_end == -1:
                    c_end = len(context)

                if (c_end - c_start) > 700:
                    s_idx = context.rfind(". ", max(0, start - 200), start)
                    if s_idx != -1:
                        c_start = s_idx + 2
                    e_idx = context.find(". ", end, min(len(context), end + 200))
                    if e_idx != -1:
                        c_end = e_idx + 1

                clause_str = context[c_start:c_end].strip()
                if not (40 <= len(clause_str) <= 750):
                    continue

                norm_str = re.sub(r"\s+", " ", clause_str)
                if norm_str in seen_high_risk:
                    continue

                # Find all answers falling into this window
                ents = []
                for q2 in paragraph["qas"]:
                    c2 = q2["id"].split("__")[-1]
                    m2 = CATEGORY_MAPPING.get(c2)
                    if not m2:
                        continue
                    for a2 in q2.get("answers", []):
                        a2_start = a2["answer_start"]
                        a2_end = a2_start + len(a2["text"])
                        if a2_start >= c_start and a2_end <= c_end:
                            r_start = a2_start - c_start
                            r_end = a2_end - c_start
                            if clause_str[r_start:r_end].strip() == a2["text"].strip():
                                ents.append({
                                    "label": m2,
                                    "text": a2["text"].strip(),
                                    "start": r_start,
                                    "end": r_end,
                                    "cuad_category": c2
                                })

                if ents:
                    seen_high_risk.add(norm_str)
                    high_risk_clauses.append({
                        "contract": title,
                        "clause": clause_str,
                        "entities": ents,
                        "primary_category": category
                    })

                if len(high_risk_clauses) >= 80:
                    break
        if len(high_risk_clauses) >= 80:
            break

    print(f"\nExtracted {len(high_risk_clauses)} high-risk liability clauses from CUAD contracts 300-399.")
    risk_entities_count = sum(len(c["entities"]) for c in high_risk_clauses)
    risk_cats = Counter(e["label"] for c in high_risk_clauses for e in c["entities"])
    print(f"  Total gold entities: {risk_entities_count}")
    for k, v in risk_cats.most_common():
        print(f"    {k}: {v}")

    risk_path = os.path.join(BASE_DIR, "data", "cuad_high_risk_liability_80.json")
    with open(risk_path, "w", encoding="utf-8") as f:
        json.dump(high_risk_clauses, f, indent=2)
    print(f"Saved: {risk_path}")

    # Dataset 4: Adversarial Distractors & Boilerplate (50 samples)
    distractors = build_adversarial_distractors(50)
    print(f"\nConstructed {len(distractors)} adversarial distractor clauses (0 true entities).")
    dist_path = os.path.join(BASE_DIR, "data", "adversarial_distractors_50.json")
    with open(dist_path, "w", encoding="utf-8") as f:
        json.dump(distractors, f, indent=2)
    print(f"Saved: {dist_path}")

    print("\nDataset construction completed successfully!")


if __name__ == "__main__":
    main()
