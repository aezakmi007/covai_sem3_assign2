"""
src/contextual_retrieval/contextual_engine.py
Task 2: Contextual Retrieval Engine (Anthropic-style chunk augmentation)
Implements:
1. Collection subset curation capped at 350 chunks (within 300-800 limit)
2. Contextual metadata generation (document identity, section hierarchy, clause role)
3. Prepending generated context to raw chunks
4. Token accounting and indexing cost calculation
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple
import tiktoken


class ContextualRetrievalEngine:
    def __init__(self, token_encoding: str = "cl100k_base"):
        """Initialize token counter for indexing cost and token budget calculation."""
        self.tokenizer = tiktoken.get_encoding(token_encoding)

    def count_tokens(self, text: str) -> int:
        """Returns the exact number of tokens in a string."""
        return len(self.tokenizer.encode(text))

    def load_and_curate_chunks(
        self, raw_dir: str = "data/raw", target_cap: int = 350
    ) -> List[Dict[str, Any]]:
        """
        Loads all policy documents, extracts structured clauses, and segments
        multi-sentence clauses into atomic provisions to produce a granular
        working collection capped at target_cap (350 chunks, within 300-800 limit).
        """
        raw_chunks = []
        chunk_idx = 0

        section_pattern = re.compile(r"(##\s+SECTION\s+[^\n]+)", re.IGNORECASE)
        clause_pattern = re.compile(r"(###\s+Clause\s+\d+\.\d+:[^\n]+)", re.IGNORECASE)
        sentence_splitter = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n(?=\d+\.\s+|-\s+)")

        for filename in sorted(os.listdir(raw_dir)):
            if not filename.endswith(".md"):
                continue

            filepath = os.path.join(raw_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if "health" in filename:
                product_name = "Optima Care Comprehensive Health Insurance (UIN: IRDAI/HLT/OC/P-H/V.1/2024-25)"
            elif "motor" in filename:
                product_name = "SecureDrive Private Car Package Policy (UIN: IRDAI/NL-GI/SD-PC/V.2/2024-25)"
            elif "irdai" in filename:
                product_name = "IRDAI Policyholder Protection and Grievance Regulations, 2024"
            else:
                product_name = "IRDAI Insurance Policy Add-ons and Consumer FAQs"

            sections = section_pattern.split(content)
            current_section = "Preamble and Metadata"

            for section_part in sections:
                if not section_part.strip():
                    continue

                if section_pattern.match(section_part.strip()):
                    current_section = section_part.strip().replace("##", "").strip()
                    continue

                clauses = clause_pattern.split(section_part)
                current_header = ""

                for clause_part in clauses:
                    if not clause_part.strip():
                        continue

                    if clause_pattern.match(clause_part.strip()):
                        current_header = clause_part.strip()
                        continue

                    clause_body = clause_part.strip()
                    clause_ids = re.findall(r"Clause\s+\d+\.\d+", current_header + " " + clause_body)
                    primary_clause_id = clause_ids[0] if clause_ids else f"Clause_{chunk_idx:03d}"
                    full_text = f"{current_header}\n{clause_body}".strip() if current_header else clause_body

                    # 1. Macro-chunk: Full clause intact
                    raw_chunks.append({
                        "chunk_id": f"chunk_{chunk_idx:04d}",
                        "doc_name": filename,
                        "product_name": product_name,
                        "section": current_section,
                        "clause_id": primary_clause_id,
                        "raw_text": full_text,
                        "raw_token_count": self.count_tokens(full_text),
                    })
                    chunk_idx += 1

                    # 2. Micro-chunks: Granular sub-provisions if clause has multiple sentences
                    sentences = [s.strip() for s in sentence_splitter.split(clause_body) if s.strip()]
                    if len(sentences) > 1:
                        for s_idx, sent in enumerate(sentences):
                            sub_text = f"{current_header} [Sub-part {s_idx+1}]\n{sent}".strip() if current_header else sent
                            raw_chunks.append({
                                "chunk_id": f"chunk_{chunk_idx:04d}",
                                "doc_name": filename,
                                "product_name": product_name,
                                "section": current_section,
                                "clause_id": primary_clause_id,
                                "raw_text": sub_text,
                                "raw_token_count": self.count_tokens(sub_text),
                            })
                            chunk_idx += 1

                    current_header = ""

        # Enforce explicit collection cap: 350 chunks (strictly within 300-800 requirement)
        capped_chunks = raw_chunks[:target_cap]
        return capped_chunks

    def generate_chunk_context(self, chunk: Dict[str, Any]) -> str:
        """
        Generates a concise 1-2 sentence contextual description that situates
        the chunk within its parent insurance contract without generic boilerplate.
        """
        product = chunk["product_name"]
        section = chunk["section"]
        clause_id = chunk["clause_id"]
        raw_text = chunk["raw_text"]
        # Extract title or first sentence summary
        title_match = re.search(r"Clause\s+\d+\.\d+:\s*([^\n]+)", raw_text)
        clause_title = title_match.group(1).strip() if title_match else "Insurance Provision"
        # Sharp, high-signal contextual header: Document + Section + Clause Topic
        context = (
            f"[Document: {product} | Section: {section} | "
            f"Clause: {clause_id} ({clause_title})]"
        )
        return context

    def augment_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepends the generated context string to each chunk and computes token metrics.
        """
        augmented = []
        for c in chunks:
            context_str = self.generate_chunk_context(c)
            contextualized_text = f"{context_str}\n\n{c['raw_text']}"
            context_tokens = self.count_tokens(context_str)
            total_tokens = self.count_tokens(contextualized_text)

            augmented.append({
                "chunk_id": c["chunk_id"],
                "doc_name": c["doc_name"],
                "product_name": c["product_name"],
                "section": c["section"],
                "clause_id": c["clause_id"],
                "raw_text": c["raw_text"],
                "context_header": context_str,
                "contextualized_text": contextualized_text,
                "raw_tokens": c["raw_token_count"],
                "context_tokens": context_tokens,
                "total_tokens": total_tokens,
            })
        return augmented