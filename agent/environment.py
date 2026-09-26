"""
Financial Auditing RL Environment with GLiNER Entity Augmentation.
Simulates tasks: contract review, penalty detection, cost-benefit analysis.
"""
import random
from dataclasses import dataclass, field
from typing import Optional, List
from agent.gliner_extractor import get_extractor, EntitySpan


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
        "Contractor liability is capped at $500,000 USD with a 30-day cure period for SLA breaches.",
    ],
    "penalty_detection": [
        "Invoice overdue by 45 days; contract specifies 1.5% monthly interest.",
        "Delivery missed by 3 weeks; SLA penalty is $5,000 per day.",
        "System downtime of 12 hours incurred; SLA demands a 10% monthly rebate.",
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
    extracted_entities: List[EntitySpan] = field(default_factory=list)


class FinancialAuditEnv:
    """RL environment for financial auditing tasks with optional GLiNER entity extraction."""

    def __init__(self, use_gliner: bool = True):
        self.use_gliner = use_gliner
        self.current_task: Optional[AuditTask] = None

    def reset(self) -> str:
        task_type = random.choice(list(TASK_TEMPLATES.keys()))
        template = random.choice(TASK_TEMPLATES[task_type])
        content = random.choice(SAMPLE_INPUTS[task_type])
        key = list(
            {
                "contract_review": "clause",
                "penalty_detection": "scenario",
                "cost_benefit": "proposal",
            }[task_type].split()
        )

        base_prompt = template.replace(f"{{{key[0]}}}", content)
        extracted: List[EntitySpan] = []

        if self.use_gliner:
            try:
                extractor = get_extractor()
                extracted = extractor.extract(content, threshold=0.35)
                if extracted:
                    entity_block = extractor.format_for_llm_prompt(extracted)
                    base_prompt = f"{entity_block}\n\nTask: {base_prompt}"
            except Exception:
                extracted = []

        self.current_task = AuditTask(
            task_type=task_type,
            prompt=base_prompt,
            expected_keywords=self._get_keywords(task_type),
            extracted_entities=extracted,
        )
        return base_prompt

    def _get_keywords(self, task_type: str) -> list:
        kw_map = {
            "contract_review": ["penalty", "clause", "compliance", "risk", "liability"],
            "penalty_detection": ["penalty", "amount", "applicable", "calculation", "rate"],
            "cost_benefit": ["benefit", "cost", "ROI", "recommendation", "savings"],
        }
        return kw_map.get(task_type, [])

    def compute_reward(self, response: str) -> float:
        """Reward based on keyword coverage, entity groundedness, and response conciseness."""
        if not self.current_task:
            return 0.0

        response_lower = response.lower()

        # 1. Domain Keyword Coverage
        kw_hits = sum(
            1
            for kw in self.current_task.expected_keywords
            if kw.lower() in response_lower
        )
        kw_coverage = kw_hits / max(len(self.current_task.expected_keywords), 1)

        # 2. Entity Grounding Bonus: Did the model cite extracted numbers/parties?
        entity_bonus = 0.0
        if self.current_task.extracted_entities:
            entity_hits = sum(
                1
                for ent in self.current_task.extracted_entities
                if ent.text.lower() in response_lower
            )
            entity_bonus = 0.2 * (entity_hits / len(self.current_task.extracted_entities))

        # 3. Conciseness Penalty for responses exceeding 300 words
        word_count = len(response.split())
        verbosity_pen = max(0, (word_count - 300) / 1000)

        total_reward = kw_coverage + entity_bonus - verbosity_pen
        return round(max(0.0, min(1.2, total_reward)), 4)
