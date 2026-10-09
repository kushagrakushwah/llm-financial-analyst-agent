"""
Dynamic Label Taxonomy Registry and Context-Aware Document Router.
Enables runtime adaptation of extraction labels based on detected document context.
"""

from typing import Dict, List, Any, Optional


TAXONOMY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "vendor_agreement": {
        "display_name": "Vendor Services & Cloud Agreement (MSA/SLA)",
        "description": "Commercial agreements covering SaaS, cloud infrastructure, and procurement.",
        "parent_clauses": [
            "service_level_clause",
            "liability_clause",
            "termination_clause",
            "payment_clause",
            "governing_law_clause",
        ],
        "child_labels": {
            "service_level_clause": [
                "sla_uptime_target",
                "measurement_window",
                "outage_threshold",
                "service_credit_rate",
            ],
            "liability_clause": [
                "liability_cap_amount",
                "exclusion_category",
                "aggregate_ceiling_type",
            ],
            "termination_clause": [
                "notice_period",
                "cure_period",
                "penalty_rate",
                "liquidated_damages_amount",
            ],
            "payment_clause": [
                "recurring_fee_amount",
                "payment_terms",
                "late_interest_rate",
                "currency",
            ],
            "governing_law_clause": [
                "governing_jurisdiction",
                "dispute_venue",
                "arbitration_body",
            ],
        },
    },
    "employment_agreement": {
        "display_name": "Executive & Employee Contract",
        "description": "Employment relationships, executive retention, and compensation schedules.",
        "parent_clauses": [
            "compensation_clause",
            "probation_and_term_clause",
            "termination_clause",
            "restrictive_covenant_clause",
        ],
        "child_labels": {
            "compensation_clause": [
                "base_salary",
                "bonus_percentage",
                "equity_grant_shares",
                "vesting_schedule",
                "currency",
            ],
            "probation_and_term_clause": [
                "job_title",
                "effective_date",
                "probation_duration",
                "work_location",
            ],
            "termination_clause": [
                "notice_days",
                "severance_multiplier",
                "cause_definition",
            ],
            "restrictive_covenant_clause": [
                "non_compete_duration",
                "non_solicitation_period",
                "geographic_scope",
            ],
        },
    },
    "credit_loan_agreement": {
        "display_name": "Credit Facility & Loan Agreement",
        "description": "Financing facilities, corporate lending, and credit lines.",
        "parent_clauses": [
            "facility_commitment_clause",
            "interest_and_margin_clause",
            "default_penalty_clause",
            "covenants_clause",
        ],
        "child_labels": {
            "facility_commitment_clause": [
                "principal_facility_amount",
                "borrower_entity",
                "administrative_agent",
                "maturity_date",
            ],
            "interest_and_margin_clause": [
                "benchmark_rate",
                "margin_spread",
                "payment_frequency",
            ],
            "default_penalty_clause": [
                "default_interest_surcharge",
                "grace_period",
                "acceleration_trigger",
            ],
            "covenants_clause": [
                "debt_service_ratio",
                "minimum_liquidity_cap",
            ],
        },
    },
    "commercial_lease": {
        "display_name": "Commercial Real Estate Lease",
        "description": "Industrial leases, office occupancy, and property agreements.",
        "parent_clauses": [
            "premises_term_clause",
            "rent_and_deposit_clause",
            "maintenance_clause",
            "default_eviction_clause",
        ],
        "child_labels": {
            "premises_term_clause": [
                "property_address",
                "commencement_date",
                "lease_term_months",
            ],
            "rent_and_deposit_clause": [
                "monthly_base_rent",
                "security_deposit",
                "escalation_percentage",
            ],
            "maintenance_clause": [
                "operating_expense_ratio",
                "repairs_deductible",
            ],
            "default_eviction_clause": [
                "cure_notice_days",
                "late_fee_percentage",
            ],
        },
    },
}


def detect_document_type(text: str) -> str:
    """
    Lightweight keyword and heuristic router to detect document context on the fly (<10ms CPU).
    Analyzes heading tokens and semantic clause anchors.
    """
    if not text or not text.strip():
        return "vendor_agreement"

    sample = text[:1500].lower()

    # Score each candidate document type
    scores = {
        "employment_agreement": 0,
        "credit_loan_agreement": 0,
        "commercial_lease": 0,
        "vendor_agreement": 0,
    }

    # Employment indicators
    employment_anchors = [
        "employee", "employment", "salary", "bonus", "severance",
        "probation", "job title", "non-compete", "vesting", "employer"
    ]
    for w in employment_anchors:
        if w in sample:
            scores["employment_agreement"] += 2

    # Credit/Loan indicators
    credit_anchors = [
        "borrower", "credit facility", "lender", "revolving credit",
        "administrative agent", "principal amount", "margin spread",
        "interest surcharge", "debt-service"
    ]
    for w in credit_anchors:
        if w in sample:
            scores["credit_loan_agreement"] += 2

    # Real Estate Lease indicators
    lease_anchors = [
        "landlord", "tenant", "premises", "base rent", "security deposit",
        "lease agreement", "occupancy", "commercial lease"
    ]
    for w in lease_anchors:
        if w in sample:
            scores["commercial_lease"] += 2

    # Vendor / SaaS / Cloud indicators
    vendor_anchors = [
        "provider", "customer", "uptime", "sla", "service level",
        "liquidated damages", "liability cap", "master cloud", "software services",
        "vendor"
    ]
    for w in vendor_anchors:
        if w in sample:
            scores["vendor_agreement"] += 2

    # Default fallback to vendor agreement if tie or low confidence
    best_match = max(scores, key=scores.get)
    if scores[best_match] > 0:
        return best_match

    return "vendor_agreement"


def get_taxonomy_for_type(doc_type: str) -> Dict[str, Any]:
    """Retrieves taxonomy configuration for a given document type with fallback."""
    return TAXONOMY_REGISTRY.get(doc_type, TAXONOMY_REGISTRY["vendor_agreement"])
