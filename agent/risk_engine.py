"""
risk_engine.py
==============
Automated Financial Risk & Contract Governance Rules Engine.
Transforms deterministic entity spans into quantitative financial risk metrics,
liability exposure calculations, statutory penalty enforceability analysis,
and executive audit scorecards.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from agent.gliner_extractor import EntitySpan


@dataclass
class RiskFinding:
    severity: str  # CRITICAL | HIGH | MEDIUM | LOW | INFO
    title: str
    description: str
    clause_reference: Optional[str] = None
    action_item: Optional[str] = None


@dataclass
class AuditScorecard:
    overall_risk_tier: str  # CRITICAL | HIGH | MODERATE | LOW
    governance_score: int    # 0 to 100 (100 = completely protected, <50 = high risk)
    liability_capped: bool
    total_liability_cap_amount: Optional[float]
    total_liability_cap_currency: Optional[str]
    identified_parties: List[str]
    max_penalty_rate_annualized_pct: Optional[float]
    sla_monthly_downtime_minutes: Optional[float]
    governing_jurisdiction: Optional[str]
    findings: List[RiskFinding] = field(default_factory=list)
    executive_verdict: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_risk_tier": self.overall_risk_tier,
            "governance_score": self.governance_score,
            "liability_capped": self.liability_capped,
            "total_liability_cap_amount": self.total_liability_cap_amount,
            "total_liability_cap_currency": self.total_liability_cap_currency,
            "identified_parties": self.identified_parties,
            "max_penalty_rate_annualized_pct": self.max_penalty_rate_annualized_pct,
            "sla_monthly_downtime_minutes": self.sla_monthly_downtime_minutes,
            "governing_jurisdiction": self.governing_jurisdiction,
            "findings": [
                {
                    "severity": f.severity,
                    "title": f.title,
                    "description": f.description,
                    "clause_reference": f.clause_reference,
                    "action_item": f.action_item,
                }
                for f in self.findings
            ],
            "executive_verdict": self.executive_verdict,
        }


class FinancialRiskEngine:
    """Evaluates compliance, statutory enforceability, and quantitative risk from extracted entities."""

    def evaluate(self, text: str, entities: List[EntitySpan]) -> AuditScorecard:
        findings: List[RiskFinding] = []
        score = 100

        # Group entities by category
        grouped: Dict[str, List[EntitySpan]] = {}
        for ent in entities:
            grouped.setdefault(ent.label, []).append(ent)

        parties = [e.text.strip().rstrip(",;\"") for e in grouped.get("contracting_party", [])]
        # Deduplicate parties while preserving order
        unique_parties = list(dict.fromkeys(parties))

        liability_caps = grouped.get("liability_cap", [])
        penalty_rates = grouped.get("penalty_rate", [])
        penalty_conditions = grouped.get("penalty_condition", [])
        slas = grouped.get("sla_target", [])
        laws = grouped.get("governing_law", [])
        grace_periods = grouped.get("grace_period", [])
        terminations = grouped.get("termination_clause", [])

        # ------------------------------------------------------------------
        # 1. Liability Cap vs Uncapped Penalty Exposure
        # ------------------------------------------------------------------
        cap_amount = None
        cap_currency = "USD"
        has_cap = len(liability_caps) > 0
        has_penalties = len(penalty_rates) > 0 or len(penalty_conditions) > 0

        if has_cap:
            # Parse currency and numerical amount
            for cap_ent in liability_caps:
                parsed = self._parse_monetary_amount(cap_ent.text)
                if parsed:
                    cap_amount, cap_currency = parsed
                    break

        if has_penalties and not has_cap:
            score -= 40
            findings.append(RiskFinding(
                severity="CRITICAL",
                title="Uncapped Financial Liability Exposure",
                description="Liquidated damages or penalty clauses are present, but no overall liability limitation cap was identified.",
                clause_reference=penalty_rates[0].text if penalty_rates else (penalty_conditions[0].text if penalty_conditions else None),
                action_item="Negotiate an aggregate liability cap (e.g. 1x or 2x annual contract fees) prior to contract execution."
            ))
        elif has_penalties and has_cap:
            findings.append(RiskFinding(
                severity="INFO",
                title="Liability Cap Established",
                description=f"Liability is bounded by explicit limitation cap: {liability_caps[0].text}.",
                clause_reference=liability_caps[0].text,
                action_item="Ensure specific carve-outs (e.g. gross negligence, indemnification) are clearly defined."
            ))

        # ------------------------------------------------------------------
        # 2. Penalty Rate Enforceability & Usury Assessment
        # ------------------------------------------------------------------
        max_annualized_rate = None
        for p_ent in penalty_rates:
            ann_rate = self._compute_annualized_rate(p_ent.text)
            if ann_rate is not None:
                if max_annualized_rate is None or ann_rate > max_annualized_rate:
                    max_annualized_rate = ann_rate

                if ann_rate > 50.0:
                    score -= 25
                    findings.append(RiskFinding(
                        severity="HIGH",
                        title="Aggressive Penalty Rate (Usurious / Punitive Risk)",
                        description=f"Penalty rate '{p_ent.text}' annualizes to ~{ann_rate:.1f}%. In commercial courts, excessive rates may be challenged as unenforceable punitive damages rather than genuine pre-estimates of loss.",
                        clause_reference=p_ent.text,
                        action_item="Reduce late penalty rate to customary commercial standard (typically 1.0% to 1.5% per month, capped at 10% total)."
                    ))
                elif ann_rate > 20.0:
                    score -= 10
                    findings.append(RiskFinding(
                        severity="MEDIUM",
                        title="Elevated Late Fee Schedule",
                        description=f"Identified penalty rate annualizes to ~{ann_rate:.1f}%. Exceeds standard prime interest benchmark.",
                        clause_reference=p_ent.text,
                        action_item="Confirm late fees do not compound daily."
                    ))

        # ------------------------------------------------------------------
        # 3. SLA Availability & Downtime Budget
        # ------------------------------------------------------------------
        monthly_downtime = None
        for sla_ent in slas:
            pct_match = re.search(r"(\d+(?:\.\d+)?)%", sla_ent.text)
            if pct_match:
                sla_pct = float(pct_match.group(1))
                # 43,800 minutes in a 30.4-day average month
                downtime_mins = (1.0 - (sla_pct / 100.0)) * 43800.0
                monthly_downtime = round(downtime_mins, 1)

                if sla_pct < 99.0:
                    score -= 15
                    findings.append(RiskFinding(
                        severity="MEDIUM",
                        title="Sub-standard Availability Target",
                        description=f"SLA target of {sla_pct}% allows over {monthly_downtime / 60.0:.1f} hours of unpenalized downtime per month.",
                        clause_reference=sla_ent.text,
                        action_item="Request standard enterprise tier of 99.9% or higher."
                    ))
                else:
                    findings.append(RiskFinding(
                        severity="INFO",
                        title="Enterprise SLA Commitment",
                        description=f"Uptime target {sla_pct}% permits a maximum of {monthly_downtime} minutes downtime monthly before triggering service credits.",
                        clause_reference=sla_ent.text
                    ))

        # ------------------------------------------------------------------
        # 4. Governing Law & Jurisdiction
        # ------------------------------------------------------------------
        gov_law = laws[0].text if laws else None
        if not gov_law:
            score -= 10
            findings.append(RiskFinding(
                severity="MEDIUM",
                title="Unspecified Governing Jurisdiction",
                description="No explicit governing law clause detected. Increases litigation cost and forum dispute risks.",
                action_item="Specify predictable commercial jurisdiction (e.g. Delaware, New York, England & Wales)."
            ))
        else:
            findings.append(RiskFinding(
                severity="INFO",
                title="Governing Jurisdiction Identified",
                description=f"Contract disputes are designated under: {gov_law}.",
                clause_reference=gov_law
            ))

        # ------------------------------------------------------------------
        # 5. Grace Period & Notice Rights
        # ------------------------------------------------------------------
        if penalty_conditions and not grace_periods:
            score -= 5
            findings.append(RiskFinding(
                severity="LOW",
                title="Absence of Explicit Cure / Grace Period",
                description="Penalties apply upon default without an explicitly identified cure window.",
                action_item="Incorporate standard 10 to 15 business day cure notice before liquidated damages accrue."
            ))

        # ------------------------------------------------------------------
        # Determine Overall Risk Tier & Executive Verdict
        # ------------------------------------------------------------------
        score = max(10, min(100, score))

        if score >= 85:
            tier = "LOW"
            verdict = "[PASS] Contract terms are well-structured with defined boundaries and standard commercial provisions."
        elif score >= 65:
            tier = "MODERATE"
            verdict = "[ACCEPTABLE WITH MODIFICATIONS] Contract contains minor exposure areas (elevated late rates or missing cure periods). Recommended for negotiation before signing."
        elif score >= 45:
            tier = "HIGH"
            verdict = "[HIGH RISK ALERT] Significant exposure identified (punitive fee schedules or ambiguous jurisdiction). Requires legal counsel escalation."
        else:
            tier = "CRITICAL"
            verdict = "[CRITICAL RISK - DO NOT SIGN AS-IS] Uncapped financial liabilities detected with compounding penalty triggers. Hard liability ceiling must be inserted."

        return AuditScorecard(
            overall_risk_tier=tier,
            governance_score=score,
            liability_capped=has_cap,
            total_liability_cap_amount=cap_amount,
            total_liability_cap_currency=cap_currency,
            identified_parties=unique_parties,
            max_penalty_rate_annualized_pct=max_annualized_rate,
            sla_monthly_downtime_minutes=monthly_downtime,
            governing_jurisdiction=gov_law,
            findings=findings,
            executive_verdict=verdict,
        )

    def _parse_monetary_amount(self, text: str) -> Optional[tuple]:
        if "%" in text:
            return None
        clean = text.replace(",", "").strip()
        # Look for explicit currency sign or USD/EUR/GBP
        match = re.search(r"(\$|USD|EUR|GBP|CAD)\s*([\d\.]+)\s*(k|m|million|billion|usd)?", clean, re.IGNORECASE)
        if not match:
            match = re.search(r"([\d\.]+)\s*(k|m|million|billion)\s*(USD|EUR|GBP|dollars)?", clean, re.IGNORECASE)
            if not match:
                # Must have leading $ or trailing USD/dollars to avoid matching random clause numbers like 2.0
                match = re.search(r"(\$)\s*([\d\.]+)|([\d\.]+)\s*(USD|dollars|cents)", clean, re.IGNORECASE)
                if not match:
                    return None
                if match.group(2):
                    num_str = match.group(2)
                    raw_sym = "$"
                else:
                    num_str = match.group(3)
                    raw_sym = match.group(4)
                suffix = ""
            else:
                num_str, suffix, raw_sym = match.groups()
        else:
            raw_sym, num_str, suffix = match.groups()

        try:
            val = float(num_str)
            suf = (suffix or "").lower()
            if suf in ("k",):
                val *= 1000
            elif suf in ("m", "million"):
                val *= 1000000
            elif suf in ("b", "billion"):
                val *= 1000000000

            curr = "USD"
            if raw_sym and ("EUR" in raw_sym.upper() or "eur" in text.lower()):
                curr = "EUR"
            elif raw_sym and ("GBP" in raw_sym.upper() or "gbp" in text.lower()):
                curr = "GBP"
            return (val, curr)
        except (ValueError, TypeError):
            return None

    def _compute_annualized_rate(self, text: str) -> Optional[float]:
        # Extract percentage number
        match = re.search(r"(\d+(?:\.\d+)?)%", text)
        if not match:
            return None
        rate = float(match.group(1))

        t_lower = text.lower()
        if "day" in t_lower or "daily" in t_lower:
            return round(rate * 365.0, 1)
        if "week" in t_lower or "weekly" in t_lower:
            return round(rate * 52.0, 1)
        if "month" in t_lower or "monthly" in t_lower:
            return round(rate * 12.0, 1)
        if "annum" in t_lower or "annual" in t_lower or "year" in t_lower:
            return round(rate, 1)

        # Default assumption: monthly
        return round(rate * 12.0, 1)


# Singleton instance
_risk_engine_instance: Optional[FinancialRiskEngine] = None

def get_risk_engine() -> FinancialRiskEngine:
    global _risk_engine_instance
    if _risk_engine_instance is None:
        _risk_engine_instance = FinancialRiskEngine()
    return _risk_engine_instance
