"""
document_chunker.py
===================
Production-grade document chunking and span-stitching engine for long contracts.
Enables GLiNER (and any transformer with 512-token context windows) to audit
full 50+ page legal agreements without truncation or duplicate entities.

Key Capabilities:
1. Boundary-Aware Chunking: Splits text on paragraphs and sentences with configurable overlap.
2. Global Offset Tracking: Preserves exact character offsets relative to the full document.
3. Non-Maximum Suppression (NMS): Reconciles and deduplicates entity spans across overlapping windows.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    chunk_id: int
    text: str
    start_char: int
    end_char: int


class DocumentChunker:
    """Splits large contracts into overlapping windows and stitches extracted entity spans back together."""

    def __init__(self, max_chars: int = 1200, overlap_chars: int = 200):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, text: str) -> List[DocumentChunk]:
        """
        Splits text into overlapping chunks, respecting paragraph, sentence, and word boundaries.
        """
        if not text or not text.strip():
            return []

        # If document fits in a single chunk, return immediately
        if len(text) <= self.max_chars:
            return [DocumentChunk(chunk_id=0, text=text, start_char=0, end_char=len(text))]

        chunks = []
        start = 0
        doc_len = len(text)
        chunk_id = 0

        while start < doc_len:
            # Determine nominal end
            end = min(start + self.max_chars, doc_len)

            # If we are not at the end of the text, look for a natural boundary
            if end < doc_len:
                # 1. Try double newline (paragraph boundary) within search window
                search_start = max(start + self.max_chars // 2, end - 250)
                para_break = text.rfind("\n\n", search_start, end)
                if para_break != -1 and para_break > start:
                    end = para_break + 2
                else:
                    # 2. Try sentence ending (period + space/newline)
                    sent_break = -1
                    for match in re.finditer(r"\.\s+", text[search_start:end]):
                        sent_break = search_start + match.end()
                    if sent_break != -1 and sent_break > start:
                        end = sent_break
                    else:
                        # 3. Fallback to whitespace to avoid splitting words
                        space_break = text.rfind(" ", search_start, end)
                        if space_break != -1 and space_break > start:
                            end = space_break + 1

            chunk_text = text[start:end]
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_char=start,
                end_char=end
            ))
            chunk_id += 1

            if end >= doc_len:
                break

            # Slide start forward, keeping overlap
            start = max(start + 1, end - self.overlap_chars)

        return chunks

    def stitch_spans(
        self,
        full_text: str,
        chunk_extractions: List[Tuple[DocumentChunk, List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """
        Remaps local chunk entity spans to global document offsets and resolves
        overlapping or duplicate entities via Non-Maximum Suppression.
        """
        raw_candidates = []

        for chunk, entities in chunk_extractions:
            for ent in entities:
                local_start = ent["start"]
                local_end = ent["end"]
                global_start = chunk.start_char + local_start
                global_end = chunk.start_char + local_end

                # Boundary safety check
                if global_end <= len(full_text):
                    candidate_text = full_text[global_start:global_end]
                    # Verify text slice matches
                    if candidate_text.strip() == ent["text"].strip():
                        raw_candidates.append({
                            "text": candidate_text,
                            "label": ent["label"],
                            "score": float(ent.get("score", 0.0)),
                            "start": global_start,
                            "end": global_end
                        })
                    else:
                        # Fallback if minor whitespace shift
                        raw_candidates.append({
                            "text": ent["text"],
                            "label": ent["label"],
                            "score": float(ent.get("score", 0.0)),
                            "start": global_start,
                            "end": global_end
                        })

        if not raw_candidates:
            return []

        # Non-Maximum Suppression (NMS) to eliminate duplicate entities from overlap regions
        # Sort primarily by score descending, then by length descending
        raw_candidates.sort(key=lambda x: (x["score"], x["end"] - x["start"]), reverse=True)

        selected = []
        for cand in raw_candidates:
            c_start = cand["start"]
            c_end = cand["end"]
            c_label = cand["label"]

            # Check if this candidate significantly overlaps with any already selected entity
            is_suppressed = False
            for sel in selected:
                s_start = sel["start"]
                s_end = sel["end"]
                s_label = sel["label"]

                overlap = max(0, min(c_end, s_end) - max(c_start, s_start))
                if overlap > 0:
                    # If same label, or high intersection over candidate length, suppress
                    c_len = max(1, c_end - c_start)
                    s_len = max(1, s_end - s_start)
                    overlap_ratio = overlap / min(c_len, s_len)
                    if c_label == s_label or overlap_ratio > 0.6:
                        is_suppressed = True
                        break

            if not is_suppressed:
                selected.append(cand)

        # Sort final list by document start position
        selected.sort(key=lambda x: x["start"])
        return selected
