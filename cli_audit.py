"""
cli_audit.py
============
Production Command-Line Financial Contract Auditing Tool.
Scans single contracts or entire document directories, extracts legal entities via GLiNER,
evaluates quantitative exposure via FinancialRiskEngine, and generates audit reports.

Usage:
  python cli_audit.py --file path/to/contract.txt
  python cli_audit.py --dir path/to/contracts/ --output-dir reports/audits/
"""

import os
import sys
import json
import glob
import argparse
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.gliner_extractor import get_extractor
from agent.risk_engine import get_risk_engine, AuditScorecard


def audit_single_document(text: str, filename: str = "document", threshold: float = 0.45) -> Dict[str, Any]:
    extractor = get_extractor()
    risk_engine = get_risk_engine()

    spans = extractor.extract(text, threshold=threshold)
    scorecard = risk_engine.evaluate(text, spans)

    return {
        "filename": filename,
        "document_length_chars": len(text),
        "entities_extracted": [s.to_dict() for s in spans],
        "scorecard": scorecard.to_dict()
    }


def print_scorecard_summary(result: Dict[str, Any]):
    sc = result["scorecard"]
    print("=" * 80)
    print(f"FINANCIAL AUDIT REPORT: {result['filename']}")
    print("=" * 80)
    print(f"Risk Tier        : {sc['overall_risk_tier']}")
    print(f"Governance Score : {sc['governance_score']}/100")
    print(f"Liability Capped : {'YES' if sc['liability_capped'] else 'NO (CRITICAL UNBOUNDED EXPOSURE)'}")
    if sc['total_liability_cap_amount']:
        print(f"Liability Ceiling: ${sc['total_liability_cap_amount']:,.2f} {sc['total_liability_cap_currency']}")
    print(f"Parties          : {', '.join(sc['identified_parties']) if sc['identified_parties'] else 'None detected'}")
    if sc['governing_jurisdiction']:
        print(f"Jurisdiction     : {sc['governing_jurisdiction']}")
    if sc['max_penalty_rate_annualized_pct']:
        print(f"Annualized Penalty: {sc['max_penalty_rate_annualized_pct']:.1f}%")

    print("\nRISK FINDINGS & GOVERNANCE ALERTS:")
    if not sc['findings']:
        print("  - No critical legal exposure flags detected.")
    else:
        for f in sc['findings']:
            print(f"  [{f['severity']}] {f['title']}")
            print(f"    Details: {f['description']}")
            if f.get('action_item'):
                print(f"    Action : {f['action_item']}")

    print("\nEXECUTIVE VERDICT:")
    print(f"  {sc['executive_verdict']}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Financial Analyst Agent Contract Audit CLI")
    parser.add_argument("--file", "-f", type=str, help="Path to single contract text file")
    parser.add_argument("--dir", "-d", type=str, help="Path to directory containing contract files")
    parser.add_argument("--threshold", "-t", type=float, default=0.45, help="Entity confidence threshold (default: 0.45)")
    parser.add_argument("--output", "-o", type=str, help="Output path for JSON audit results")
    args = parser.parse_args()

    if not args.file and not args.dir:
        # Default sample run
        sample_text = (
            "This Master Services Agreement is executed as of October 15, 2026 by and between "
            "Enterprise Cloud Systems Inc. and Apex Global Logistics LLC. Provider shall maintain 99.95% uptime SLA. "
            "In case of unexcused delay, Supplier shall pay liquidated damages of 2.0% per week. "
            "Total cumulative liability of Supplier shall not exceed $500,000 USD. "
            "Governing law shall be the laws of the State of Delaware."
        )
        print("No input specified. Executing audit on sample enterprise agreement...\n")
        res = audit_single_document(sample_text, "Sample_Enterprise_MSA.txt", args.threshold)
        print_scorecard_summary(res)
        return

    results = []

    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
        res = audit_single_document(content, os.path.basename(args.file), args.threshold)
        results.append(res)
        print_scorecard_summary(res)

    elif args.dir:
        if not os.path.exists(args.dir):
            print(f"Error: Directory not found: {args.dir}")
            sys.exit(1)
        files = glob.glob(os.path.join(args.dir, "*.txt")) + glob.glob(os.path.join(args.dir, "*.json"))
        if not files:
            print(f"No .txt or .json files found in {args.dir}")
            sys.exit(0)

        print(f"Auditing {len(files)} contracts found in {args.dir}...")
        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            res = audit_single_document(content, os.path.basename(filepath), args.threshold)
            results.append(res)
            print_scorecard_summary(res)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nAudit results successfully exported to: {args.output}")


if __name__ == "__main__":
    main()
