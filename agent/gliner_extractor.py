"""
Financial Entity Extractor using GLiNER (Generalist and Lightweight Model for NER).
Provides zero-shot and fine-tuned extraction of contract clauses, monetary figures,
liability caps, penalty rates, and legal entities.
"""

import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from gliner import GLiNER

from agent.label_taxonomy import TAXONOMY_REGISTRY, detect_document_type, get_taxonomy_for_type
from agent.schemas import ChildEntity, ParentClause, HierarchicalExtractionResult


DEFAULT_FINANCIAL_LABELS = [
    "contracting_party",
    "penalty_rate",
    "penalty_condition",
    "liability_cap",
    "monetary_amount",
    "interest_rate",
    "effective_date",
    "expiration_date",
    "termination_clause",
    "governing_law",
    "payment_terms",
    "sla_target",
    "grace_period",
]


@dataclass
class EntitySpan:
    text: str
    label: str
    score: float
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "score": round(self.score, 4),
            "start": self.start,
            "end": self.end,
        }


class FinancialEntityExtractor:
    """Extracts financial and legal entity spans from raw contract text using GLiNER."""

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        default_labels: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        # Prefer local fine-tuned model if available
        local_finetuned_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "gliner_financial"
        )
        if model_name_or_path is None:
            if os.path.exists(local_finetuned_path):
                model_name_or_path = local_finetuned_path
            else:
                model_name_or_path = os.getenv(
                    "GLINER_MODEL", "urchade/gliner_small-v2.1"
                )

        self.model_name_or_path = model_name_or_path
        self.default_labels = default_labels or DEFAULT_FINANCIAL_LABELS
        self.device = device
        self._model: Optional[GLiNER] = None

    @property
    def model(self) -> GLiNER:
        if self._model is None:
            self._model = GLiNER.from_pretrained(self.model_name_or_path)
            if self.device != "cpu":
                self._model = self._model.to(self.device)
        return self._model

    def extract(
        self,
        text: str,
        labels: Optional[List[str]] = None,
        threshold: float = 0.45,
        flat_ner: bool = True,
        chunk_long_text: bool = True,
    ) -> List[EntitySpan]:
        """
        Extracts entity spans from text with exact character offsets.
        Automatically chunks documents exceeding 1200 characters to prevent
        DeBERTa subword context truncation.
        """
        if not text or not text.strip():
            return []

        # Route large contracts through boundary-aware sliding window chunker
        if chunk_long_text and len(text) > 1200:
            return self.extract_long_document(
                text=text,
                labels=labels,
                threshold=threshold,
                flat_ner=flat_ner,
            )

        target_labels = labels or self.default_labels
        predictions = self.model.predict_entities(
            text, target_labels, threshold=threshold, flat_ner=flat_ner
        )

        return [
            EntitySpan(
                text=ent["text"],
                label=ent["label"],
                score=float(ent["score"]),
                start=int(ent["start"]),
                end=int(ent["end"]),
            )
            for ent in predictions
        ]

    def extract_long_document(
        self,
        text: str,
        labels: Optional[List[str]] = None,
        threshold: float = 0.45,
        flat_ner: bool = True,
        chunk_chars: int = 1200,
        overlap_chars: int = 200,
    ) -> List[EntitySpan]:
        """
        Processes multi-page contracts using sliding window chunking and
        reconciles spans with Non-Maximum Suppression (NMS).
        """
        from agent.document_chunker import DocumentChunker

        chunker = DocumentChunker(max_chars=chunk_chars, overlap_chars=overlap_chars)
        chunks = chunker.chunk(text)

        target_labels = labels or self.default_labels
        chunk_extractions = []

        for ch in chunks:
            preds = self.model.predict_entities(
                ch.text, target_labels, threshold=threshold, flat_ner=flat_ner
            )
            chunk_extractions.append((ch, preds))

        stitched = chunker.stitch_spans(text, chunk_extractions)

        return [
            EntitySpan(
                text=ent["text"],
                label=ent["label"],
                score=float(ent["score"]),
                start=int(ent["start"]),
                end=int(ent["end"]),
            )
            for ent in stitched
        ]

    def extract_batch(
        self,
        texts: List[str],
        labels: Optional[List[str]] = None,
        threshold: float = 0.45,
        flat_ner: bool = True,
    ) -> List[List[EntitySpan]]:
        """
        Extracts entities across multiple contract clauses in a batch.
        """
        return [
            self.extract(t, labels=labels, threshold=threshold, flat_ner=flat_ner)
            for t in texts
        ]

    def extract_hierarchical(
        self,
        text: str,
        doc_type: Optional[str] = None,
        threshold: float = 0.45,
        fallback_threshold: float = 0.35,
    ) -> HierarchicalExtractionResult:
        """
        Two-pass hierarchical extraction pipeline.
        Pass 1: Segments the contract into parent clauses based on context taxonomy.
        Pass 2: Extracts fine-grained child entities within each parent clause,
                stitching relative coordinates back to global document offsets.
        Fallback: Identifies novel/unclassified clauses that do not meet standard classification.
        """
        start_time = time.perf_counter()

        if not text or not text.strip():
            return HierarchicalExtractionResult(
                document_type=doc_type or "unknown",
                detected_parent_count=0,
                detected_child_count=0,
                execution_time_ms=0.0,
                clauses=[],
                unclassified_clauses=[],
            )

        # 1. Resolve Document Type & Taxonomy
        resolved_type = doc_type if (doc_type and doc_type in TAXONOMY_REGISTRY) else detect_document_type(text)
        taxonomy = get_taxonomy_for_type(resolved_type)
        parent_labels = taxonomy["parent_clauses"]
        child_label_map = taxonomy["child_labels"]

        # 2. Pass 1: Extract Parent Clauses
        parent_preds = self.model.predict_entities(
            text, parent_labels, threshold=threshold, flat_ner=True
        )

        clauses: List[ParentClause] = []
        unclassified_clauses: List[ParentClause] = []
        covered_ranges: List[tuple] = []
        total_child_count = 0

        parent_preds_sorted = sorted(parent_preds, key=lambda x: int(x["start"]))

        for p in parent_preds_sorted:
            p_label = p["label"]
            p_text = p["text"]
            p_score = float(p["score"])
            p_start = int(p["start"])
            p_end = int(p["end"])

            covered_ranges.append((p_start, p_end))

            target_children = child_label_map.get(p_label, [])
            children: List[ChildEntity] = []

            if target_children and len(p_text.strip()) > 0:
                child_preds = self.model.predict_entities(
                    p_text, target_children, threshold=threshold, flat_ner=True
                )
                for c in child_preds:
                    c_text = c["text"]
                    c_score = float(c["score"])
                    local_s = int(c["start"])
                    local_e = int(c["end"])
                    global_s = p_start + local_s
                    global_e = p_start + local_e

                    child_ent = ChildEntity(
                        text=c_text,
                        label=c["label"],
                        score=c_score,
                        local_start=local_s,
                        local_end=local_e,
                        global_start=global_s,
                        global_end=global_e,
                    )
                    children.append(child_ent)
                    total_child_count += 1

            clauses.append(
                ParentClause(
                    parent_label=p_label,
                    clause_text=p_text,
                    score=p_score,
                    global_start=p_start,
                    global_end=p_end,
                    is_unclassified=False,
                    children=children,
                )
            )

        # 3. Fallback / Novel Clause Detection (Suggestion 1)
        marginal_preds = self.model.predict_entities(
            text, parent_labels, threshold=fallback_threshold, flat_ner=True
        )
        for m in marginal_preds:
            m_start = int(m["start"])
            m_end = int(m["end"])
            is_covered = any(
                max(m_start, cs) < min(m_end, ce) for cs, ce in covered_ranges
            )
            if not is_covered and float(m["score"]) < threshold:
                unclassified_clauses.append(
                    ParentClause(
                        parent_label=f"marginal_{m['label']}",
                        clause_text=m["text"],
                        score=float(m["score"]),
                        global_start=m_start,
                        global_end=m_end,
                        is_unclassified=True,
                        children=[],
                    )
                )

        exec_time = (time.perf_counter() - start_time) * 1000.0

        return HierarchicalExtractionResult(
            document_type=resolved_type,
            detected_parent_count=len(clauses),
            detected_child_count=total_child_count,
            execution_time_ms=exec_time,
            clauses=clauses,
            unclassified_clauses=unclassified_clauses,
        )

    def format_for_llm_prompt(self, entities: List[EntitySpan]) -> str:
        """
        Formats extracted entities into a clean, structured context block
        ready to inject into the LLM reasoning agent prompt.
        """
        if not entities:
            return "No high-confidence financial entities detected in pre-extraction."

        lines = ["=== Pre-Extracted Financial & Legal Entities (via GLiNER) ==="]
        for ent in entities:
            lines.append(
                f"- [{ent.label.upper()}]: \"{ent.text}\" (confidence: {ent.score:.2f}, span: {ent.start}:{ent.end})"
            )
        lines.append("=============================================================")
        return "\n".join(lines)


# Global singleton instance for efficient memory reuse across API requests
_extractor_instance: Optional[FinancialEntityExtractor] = None


def get_extractor() -> FinancialEntityExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = FinancialEntityExtractor()
    return _extractor_instance
