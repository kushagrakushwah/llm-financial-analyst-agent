"""
demo_gliner_audit.py
====================
Interactive demonstration of GLiNER-augmented financial contract review.
Compares Zero-Shot vs Fine-Tuned GLiNER extraction and shows the hybrid audit pipeline.
"""

import os
import sys
from agent.gliner_extractor import FinancialEntityExtractor

CONTRACT_SAMPLES = [
    {
        "title": "Vendor SLA & Liquidated Damages Agreement",
        "text": (
            "This Master Services Agreement is entered into by Acme Cloud Technologies and Global Logistics Inc. "
            "In the event of unexcused milestone delivery delay, Contractor shall incur a liquidated damages penalty "
            "of 2.5% per week of delay against the contract fee of $1,200,000 USD. "
            "In no event shall total cumulative liability exceed the liability cap of $300,000 USD under this Agreement."
        ),
        "task_type": "contract_review"
    },
    {
        "title": "Overdue Invoicing & Statutory Late Interest",
        "text": (
            "Invoice #INV-2025-9941 for consulting services totaling $85,000 USD remains unpaid and is overdue by 45 days. "
            "Pursuant to Section 8.2, overdue balances accrue a penalty rate of 1.5% monthly late interest starting from day 31."
        ),
        "task_type": "penalty_detection"
    },
    {
        "title": "Enterprise Software Licensing & Termination Notice",
        "text": (
            "Either party may terminate this Software License Agreement with 60 days prior written notice. "
            "This Agreement is governed by the laws of the State of Delaware, and any disputes shall be settled by binding arbitration."
        ),
        "task_type": "contract_review"
    }
]


def print_banner(title: str):
    print("\n" + "=" * 76)
    print(f"  {title}")
    print("=" * 76)


def main():
    print_banner("HYBRID FINANCIAL AUDITOR: GLiNER + AGENTIC REASONING")

    # 1. Initialize Extractor (automatically uses local fine-tuned model if present)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    finetuned_path = os.path.join(base_dir, "models", "gliner_financial")

    if os.path.exists(finetuned_path):
        print(f"[*] Detected fine-tuned model checkpoint at: {finetuned_path}")
        extractor = FinancialEntityExtractor(model_name_or_path=finetuned_path)
    else:
        print("[*] Using base pre-trained checkpoint: urchade/gliner_small-v2.1")
        extractor = FinancialEntityExtractor(model_name_or_path="urchade/gliner_small-v2.1")

    # 2. Process each contract sample
    for idx, sample in enumerate(CONTRACT_SAMPLES, 1):
        print_banner(f"Sample {idx}: {sample['title']}")
        print(f"Contract Text:\n  \"{sample['text']}\"\n")

        # Extract entity spans
        entities = extractor.extract(sample["text"], threshold=0.35)

        print("Extracted Financial & Legal Spans (GLiNER):")
        print(f"  {'Entity Span':<32} | {'Label':<20} | {'Score':<6} | {'Offsets'}")
        print("  " + "-" * 72)
        for ent in entities:
            print(f"  {ent.text:<32} | {ent.label:<20} | {ent.score:.2f}   | [{ent.start}:{ent.end}]")

        # Show prompt formatted for the Qwen2.5-7B GRPO Agent
        print("\nAugmented LLM Prompt Context:")
        print("  " + extractor.format_for_llm_prompt(entities).replace("\n", "\n  "))

        print("\nAudit Summary Findings:")
        penalty_spans = [e.text for e in entities if "penalty" in e.label]
        cap_spans = [e.text for e in entities if "cap" in e.label or "liability" in e.label]
        party_spans = [e.text for e in entities if "party" in e.label]

        if penalty_spans:
            print(f"  [!] Financial Risk Alert: Liquidated damages found ({', '.join(penalty_spans)})")
        if cap_spans:
            print(f"  [✓] Risk Mitigation: Aggregate liability limit verified ({', '.join(cap_spans)})")
        if party_spans:
            print(f"  [i] Contracting Entities: {', '.join(party_spans)}")


if __name__ == "__main__":
    main()
