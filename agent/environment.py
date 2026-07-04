"""
Financial Auditing RL Environment.
Simulates tasks: contract review, penalty detection, cost-benefit analysis.
"""
import random
from dataclasses import dataclass
from typing import Optional


TASK_TEMPLATES = {
    "contract_review": [
        "Review the following vendor contract clause and flag any penalty clauses: {clause}",
        "Identify compliance issues in this contract section: {clause}",
    ],
    "penalty_detection": [
        "Calculate the total penalty exposure from: {scenario}",
        "Determine if a penalty applies under these conditions: {scenario}",
    ],
    "cost_benefit": [
        "Perform a cost-benefit analysis for: {proposal}",
        "Evaluate the financial impact of: {proposal}",
    ],
}

SAMPLE_INPUTS = {
    "contract_review": [
        "Vendor shall pay 2% of contract value per week of delay beyond agreed delivery.",
        "All disputes shall be settled by arbitration; costs borne by the losing party.",
    ],
    "penalty_detection": [
        "Invoice overdue by 45 days; contract specifies 1.5% monthly interest.",
        "Delivery missed by 3 weeks; SLA penalty is $5,000 per day.",
    ],
    "cost_benefit": [
        "Upgrading legacy ERP system at $250K with projected savings of $80K/year.",
        "Outsourcing payroll processing at $30K/year vs current in-house cost of $75K/year.",
    ],
}


@dataclass
class AuditTask:
    task_type: str
    prompt: str
    expected_keywords: list


class FinancialAuditEnv:
    """Simple RL environment for financial auditing tasks."""

    def __init__(self):
        self.current_task: Optional[AuditTask] = None

    def reset(self) -> str:
        task_type = random.choice(list(TASK_TEMPLATES.keys()))
        template  = random.choice(TASK_TEMPLATES[task_type])
        content   = random.choice(SAMPLE_INPUTS[task_type])
        key       = list({"contract_review": "clause",
                          "penalty_detection": "scenario",
                          "cost_benefit": "proposal"}[task_type].split())

        prompt = template.replace(f"{{{key[0]}}}", content)
        self.current_task = AuditTask(
            task_type=task_type,
            prompt=prompt,
            expected_keywords=self._get_keywords(task_type),
        )
        return prompt

    def _get_keywords(self, task_type: str) -> list:
        kw_map = {
            "contract_review":    ["penalty", "clause", "compliance", "risk"],
            "penalty_detection":  ["penalty", "amount", "applicable", "calculation"],
            "cost_benefit":       ["benefit", "cost", "ROI", "recommendation"],
        }
        return kw_map.get(task_type, [])

    def compute_reward(self, response: str) -> float:
        """Reward based on keyword coverage and response conciseness."""
        if not self.current_task:
            return 0.0
        response_lower = response.lower()
        kw_hits  = sum(1 for kw in self.current_task.expected_keywords
                       if kw.lower() in response_lower)
        coverage = kw_hits / max(len(self.current_task.expected_keywords), 1)
        # Penalise verbose responses (>300 words)
        word_count    = len(response.split())
        verbosity_pen = max(0, (word_count - 300) / 1000)
        return round(coverage - verbosity_pen, 4)
