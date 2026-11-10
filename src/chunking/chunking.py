"""
src/chunking.py
Task 1: Corpus Preparation and Chunking Strategy Comparison
Implements:
1. Fixed-Size Chunking (with overlap)
2. Structure-Aware Chunking (respecting clause & section boundaries)
3. Semantic Chunking (sentence embeddings with similarity drop threshold)
Also computes chunk statistics and clause-splitting frequency.
"""

import os
import re
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer


class ChunkingEngine:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model for semantic chunking and evaluation."""
        self.model = SentenceTransformer(embedding_model_name)

    # -------------------------------------------------------------
    # 1. FIXED-SIZE CHUNKING
    # -------------------------------------------------------------
    def fixed_size_chunking(
        self, text: str, doc_name: str, chunk_size: int = 500, overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Splits text into fixed character windows with sliding overlap.
        Naive approach: does not respect sentence or clause boundaries.
        """
        chunks = []
        step = chunk_size - overlap
        start = 0
        chunk_idx = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_content = text[start:end].strip()

            if chunk_content:
                # Detect which clauses got touched by this chunk
                clauses_present = re.findall(r"Clause\s+\d+\.\d+", chunk_content)
                chunks.append({
                    "chunk_id": f"{doc_name}_fixed_{chunk_idx:03d}",
                    "doc_name": doc_name,
                    "strategy": "fixed_size",
                    "text": chunk_content,
                    "char_length": len(chunk_content),
                    "clauses_present": list(set(clauses_present)),
                    "clause_count": len(set(clauses_present)),
                })
                chunk_idx += 1

            start += step
            if end >= len(text):
                break

        return chunks

    # -------------------------------------------------------------
    # 2. STRUCTURE-AWARE CHUNKING
    # -------------------------------------------------------------
    def structure_aware_chunking(self, text: str, doc_name: str) -> List[Dict[str, Any]]:
        """
        Splits text along document hierarchy: Sections and Clauses.
        Preserves clause integrity and associates parent section title.
        """
        chunks = []
        chunk_idx = 0

        # Regex to capture Section headers: e.g., '## SECTION 1: DEFINITIONS'
        section_pattern = re.compile(r"(##\s+SECTION\s+[^\n]+)", re.IGNORECASE)
        # Regex to capture Clause headers: e.g., '### Clause 1.1: Accident'
        clause_pattern = re.compile(r"(###\s+Clause\s+\d+\.\d+:[^\n]+)", re.IGNORECASE)

        # Split into sections first
        sections = section_pattern.split(text)
        current_section = "Preamble / Metadata"

        for part in sections:
            if not part.strip():
                continue

            # If this part is a Section Header, update current section
            if section_pattern.match(part.strip()):
                current_section = part.strip().replace("##", "").strip()
                continue

            # Within this section, split by clauses
            clause_parts = clause_pattern.split(part)
            current_clause_header = ""

            for subpart in clause_parts:
                if not subpart.strip():
                    continue

                if clause_pattern.match(subpart.strip()):
                    current_clause_header = subpart.strip()
                    continue

                # Combine header with clause body
                clause_text = f"{current_clause_header}\n{subpart.strip()}".strip() if current_clause_header else subpart.strip()
                clauses_found = re.findall(r"Clause\s+\d+\.\d+", clause_text)
                clause_id = clauses_found[0] if clauses_found else f"Preamble_{chunk_idx}"

                chunks.append({
                    "chunk_id": f"{doc_name}_struct_{chunk_idx:03d}",
                    "doc_name": doc_name,
                    "section": current_section,
                    "clause_id": clause_id,
                    "strategy": "structure_aware",
                    "text": clause_text,
                    "char_length": len(clause_text),
                    "clauses_present": [clause_id] if clauses_found else [],
                    "clause_count": 1 if clauses_found else 0,
                })
                chunk_idx += 1
                current_clause_header = ""

        return chunks

    # -------------------------------------------------------------
    # 3. SEMANTIC CHUNKING
    # -------------------------------------------------------------
    def semantic_chunking(
        self, text: str, doc_name: str, distance_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Splits text by sentence embeddings similarity.
        Computes cosine distance between consecutive sentences and
        cuts when distance exceeds distance_threshold.
        """
        # Split text into sentences using regex
        sentence_delimiters = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9#])")
        raw_sentences = [s.strip() for s in sentence_delimiters.split(text) if s.strip()]

        if not raw_sentences:
            return []

        # Generate embeddings for each individual sentence
        embeddings = self.model.encode(raw_sentences, convert_to_numpy=True, normalize_embeddings=True)

        chunks = []
        current_chunk_sentences = [raw_sentences[0]]
        chunk_idx = 0

        for i in range(len(raw_sentences) - 1):
            # Cosine distance = 1 - cosine_similarity (embeddings are already normalized)
            similarity = float(np.dot(embeddings[i], embeddings[i + 1]))
            distance = 1.0 - similarity

            # If semantic distance exceeds threshold, cut and start new chunk
            if distance > distance_threshold:
                chunk_content = " ".join(current_chunk_sentences).strip()
                clauses_present = list(set(re.findall(r"Clause\s+\d+\.\d+", chunk_content)))
                chunks.append({
                    "chunk_id": f"{doc_name}_semantic_{chunk_idx:03d}",
                    "doc_name": doc_name,
                    "strategy": "semantic",
                    "text": chunk_content,
                    "char_length": len(chunk_content),
                    "clauses_present": clauses_present,
                    "clause_count": len(clauses_present),
                })
                chunk_idx += 1
                current_chunk_sentences = [raw_sentences[i + 1]]
            else:
                current_chunk_sentences.append(raw_sentences[i + 1])

        # Add any remaining sentences as the final chunk
        if current_chunk_sentences:
            chunk_content = " ".join(current_chunk_sentences).strip()
            clauses_present = list(set(re.findall(r"Clause\s+\d+\.\d+", chunk_content)))
            chunks.append({
                "chunk_id": f"{doc_name}_semantic_{chunk_idx:03d}",
                "doc_name": doc_name,
                "strategy": "semantic",
                "text": chunk_content,
                "char_length": len(chunk_content),
                "clauses_present": clauses_present,
                "clause_count": len(clauses_present),
            })

        return chunks

    # -------------------------------------------------------------
    # 4. CHUNKING COMPARISON METRICS
    # -------------------------------------------------------------
    @staticmethod
    def calculate_metrics(chunks: List[Dict[str, Any]], all_clause_ids: List[str]) -> Dict[str, Any]:
        """
        Calculates Task 1 required metrics:
        - Chunk count
        - Mean chunk length
        - Standard deviation of chunk length
        - Frequency of clause splitting across chunks
        """
        if not chunks:
            return {"chunk_count": 0, "mean_len": 0, "std_len": 0, "split_clause_freq": 0}

        lengths = [c["char_length"] for c in chunks]
        mean_len = float(np.mean(lengths))
        std_len = float(np.std(lengths))

        # Check how many distinct clauses got split across >= 2 chunks
        clause_to_chunks = {}
        for c in chunks:
            for clause in c["clauses_present"]:
                clause_to_chunks.setdefault(clause, set()).add(c["chunk_id"])

        split_clauses = sum(1 for clause, chunk_set in clause_to_chunks.items() if len(chunk_set) > 1)
        total_unique_clauses = len(all_clause_ids) if all_clause_ids else len(clause_to_chunks)

        split_frequency = (split_clauses / total_unique_clauses * 100.0) if total_unique_clauses > 0 else 0.0

        return {
            "chunk_count": len(chunks),
            "mean_len": round(mean_len, 2),
            "std_len": round(std_len, 2),
            "split_clauses_count": split_clauses,
            "total_clauses": total_unique_clauses,
            "split_clause_frequency_pct": round(split_frequency, 2),
        }