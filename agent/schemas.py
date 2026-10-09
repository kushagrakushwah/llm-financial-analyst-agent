"""
Strict Pydantic data schemas for hierarchical contract extraction.
Ensures downstream reasoning agents receive typed, validated parent-child data trees.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChildEntity(BaseModel):
    """Fine-grained sub-span extracted within a specific parent clause."""
    text: str = Field(..., description="Extracted substring.")
    label: str = Field(..., description="Semantic entity sub-label.")
    score: float = Field(..., description="Model confidence score [0.0 - 1.0].")
    canonical_type: Optional[str] = Field(None, description="Standard canonical entity category for database normalization.")
    local_start: int = Field(..., description="Start character offset relative to parent clause text.")
    local_end: int = Field(..., description="End character offset relative to parent clause text.")
    global_start: int = Field(..., description="Start character offset in full document for provenance audit.")
    global_end: int = Field(..., description="End character offset in full document for provenance audit.")

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "text": self.text,
            "label": self.label,
            "score": round(self.score, 4),
            "local_start": self.local_start,
            "local_end": self.local_end,
            "global_start": self.global_start,
            "global_end": self.global_end,
        }
        if self.canonical_type:
            d["canonical_type"] = self.canonical_type
        return d


class ParentClause(BaseModel):
    """High-level contractual clause segment enclosing fine-grained child entities."""
    parent_label: str = Field(..., description="Category of the clause (e.g. termination_clause).")
    canonical_parent: Optional[str] = Field(None, description="Standard normalized parent category (e.g. penalty_clause).")
    clause_text: str = Field(..., description="Full text span of the clause.")
    score: float = Field(..., description="Detection confidence score.")
    global_start: int = Field(..., description="Start character offset in the source document.")
    global_end: int = Field(..., description="End character offset in the source document.")
    is_unclassified: bool = Field(default=False, description="Flag indicating novel or unclassified clause fallback.")
    is_dynamically_induced: bool = Field(default=False, description="Flag indicating clause schema was induced by Tier 2 AI.")
    children: List[ChildEntity] = Field(default_factory=list, description="Nested fine-grained entities within this clause.")

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "parent_label": self.parent_label,
            "clause_text": self.clause_text,
            "score": round(self.score, 4),
            "global_start": self.global_start,
            "global_end": self.global_end,
            "is_unclassified": self.is_unclassified,
            "is_dynamically_induced": self.is_dynamically_induced,
            "children": [c.to_dict() for c in self.children],
        }
        if self.canonical_parent:
            d["canonical_parent"] = self.canonical_parent
        return d


class HierarchicalExtractionResult(BaseModel):
    """Complete hierarchical extraction response container."""
    document_type: str = Field(..., description="Identified contract classification.")
    detected_parent_count: int = Field(..., description="Count of parent clauses identified.")
    detected_child_count: int = Field(..., description="Total fine-grained entity tokens extracted.")
    execution_time_ms: float = Field(..., description="Extraction latency in milliseconds.")
    clauses: List[ParentClause] = Field(default_factory=list, description="Structured parent-child clauses.")
    unclassified_clauses: List[ParentClause] = Field(default_factory=list, description="Clauses flagged as novel or fallback.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_type": self.document_type,
            "detected_parent_count": self.detected_parent_count,
            "detected_child_count": self.detected_child_count,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "clauses": [c.to_dict() for c in self.clauses],
            "unclassified_clauses": [c.to_dict() for c in self.unclassified_clauses],
        }


class HierarchicalExtractRequest(BaseModel):
    """Input payload for hierarchical extraction API."""
    text: str = Field(..., description="Raw contract text or clause sequence.")
    document_type: Optional[str] = Field(None, description="Optional manual override for document taxonomy.")
    threshold: float = Field(0.45, ge=0.0, le=1.0, description="Confidence threshold for parent and child entities.")
    fallback_threshold: float = Field(0.35, ge=0.0, le=1.0, description="Threshold for novel clause fallback tagging.")
    enable_dynamic_fallback: bool = Field(default=True, description="Enable Tier 2 dynamic AI schema induction for unclassified clauses.")
