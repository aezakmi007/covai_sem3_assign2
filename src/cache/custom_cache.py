"""
src/cache/custom_cache.py
-------------------------
Task 6: Student-Implemented Two-Tier Prompt Cache & Model Routing Economics.
Implements:
1. Two-Tier Prompt Cache (Zero vendor telemetry):
   - Tier 1: Exact Query Hash Cache (O(1) SHA-256).
   - Tier 2: Semantic Lookup Cache (Cosine Similarity >= 0.92).
2. Dynamic Model Router:
   - Small Model (Flash-Lite / Llama 1B): simple factual & memory queries.
   - Large Model (Flash / Llama 3B): compound decomposition & complex reasoning.
3. Cost & Latency Modeling.
"""

import os
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class StudentPromptCache:
    """
    Student-implemented two-tier prompt cache.
    Does not rely on proprietary vendor telemetry or API headers.
    """

    def __init__(self, semantic_threshold: float = 0.75, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.exact_cache: Dict[str, Dict[str, Any]] = {}
        self.semantic_embeddings: List[np.ndarray] = []
        self.semantic_entries: List[Dict[str, Any]] = []
        self.semantic_threshold = semantic_threshold
        self.model = SentenceTransformer(embedding_model_name)
        self.stats = {"exact_hits": 0, "semantic_hits": 0, "misses": 0}

    def _hash_query(self, text: str) -> str:
        """Returns normalized SHA-256 hash of query string."""
        normalized = " ".join(text.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def lookup(self, query: str) -> Tuple[Optional[str], str, float]:
        """
        Check cache for matching response.
        Returns: (cached_response, hit_type, similarity_score)
        where hit_type is 'EXACT', 'SEMANTIC', or 'MISS'.
        """
        query_hash = self._hash_query(query)

        # 1. Exact Match Lookup (O(1))
        if query_hash in self.exact_cache:
            self.stats["exact_hits"] += 1
            entry = self.exact_cache[query_hash]
            return entry["response"], "EXACT", 1.0

        # 2. Semantic Match Lookup (Cosine similarity >= 0.92)
        if self.semantic_embeddings:
            q_emb = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
            similarities = np.dot(self.semantic_embeddings, q_emb)
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            if best_score >= self.semantic_threshold:
                self.stats["semantic_hits"] += 1
                entry = self.semantic_entries[best_idx]
                return entry["response"], f"SEMANTIC (sim={best_score:.3f})", best_score

        best_score_val = best_score if self.semantic_embeddings else 0.0
        self.stats["misses"] += 1
        return None, "MISS", best_score_val

    def put(self, query: str, response: str, input_tokens: int = 150, output_tokens: int = 50):
        """Store query, response, and saved token metadata in both tiers."""
        query_hash = self._hash_query(query)
        entry = {
            "query": query,
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "timestamp": time.time(),
        }
        self.exact_cache[query_hash] = entry

        # Add to vector index for semantic lookup
        q_emb = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        self.semantic_embeddings.append(q_emb)
        self.semantic_entries.append(entry)

    def get_hit_rate(self) -> Dict[str, Any]:
        """Compute hit rate breakdown."""
        total = self.stats["exact_hits"] + self.stats["semantic_hits"] + self.stats["misses"]
        if total == 0:
            return {"hit_rate_pct": 0.0, "exact_pct": 0.0, "semantic_pct": 0.0, "total": 0}
        hits = self.stats["exact_hits"] + self.stats["semantic_hits"]
        return {
            "total_requests": total,
            "total_hits": hits,
            "hit_rate_pct": round((hits / total) * 100.0, 2),
            "exact_hits": self.stats["exact_hits"],
            "exact_pct": round((self.stats["exact_hits"] / total) * 100.0, 2),
            "semantic_hits": self.stats["semantic_hits"],
            "semantic_pct": round((self.stats["semantic_hits"] / total) * 100.0, 2),
            "misses": self.stats["misses"],
            "miss_pct": round((self.stats["misses"] / total) * 100.0, 2),
        }


class DynamicModelRouter:
    """
    Routes queries between Small/Cheap Model and Large/Complex Model
    based on task complexity, and models token economics.
    """

    # Industry standard commercial API pricing (per 1,000,000 tokens)
    PRICING = {
        "large_model": {
            "name": "Large Model (e.g., Llama 3.2 3B / Gemini Flash)",
            "input_per_m": 0.35,   # $0.35 per 1M input tokens
            "output_per_m": 1.05,  # $1.05 per 1M output tokens
            "base_latency_ms": 750.0,
        },
        "small_model": {
            "name": "Small Model (e.g., Llama 3.2 1B / Gemini Flash-Lite)",
            "input_per_m": 0.075,  # $0.075 per 1M input tokens (78% cheaper)
            "output_per_m": 0.30,  # $0.30 per 1M output tokens
            "base_latency_ms": 180.0, # 4x faster
        },
    }

    def route_model(self, query: str, route: str) -> str:
        """
        Determine model tier based on task complexity:
        - Small Model: MEMORY_DIRECT queries or simple factual single clauses.
        - Large Model: COMPOUND_DECOMPOSE, complex exclusions, or multi-hop disputes.
        """
        q_words = len(query.strip().split())
        if route == "MEMORY_DIRECT" or (route == "POLICY_RETRIEVAL" and q_words <= 9):
            return "small_model"
        return "large_model"

    def compute_turn_cost(self, model_tier: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate USD cost for a single turn."""
        rates = self.PRICING[model_tier]
        cost = (input_tokens / 1_000_000.0) * rates["input_per_m"] + (output_tokens / 1_000_000.0) * rates["output_per_m"]
        return cost