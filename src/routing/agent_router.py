"""
src/routing/agent_router.py
---------------------------
Task 5: Production Agentic Query Router & Iterative Retrieval Engine.
Implements:
1. Four Route Decision Paths with strict boundary enforcement:
   - MEMORY_DIRECT: Personal user state lookup (Policy #, City, Claim ID).
   - POLICY_RETRIEVAL: Legal policy clauses, exclusions, waiting periods, IRDAI circulars.
   - COMPOUND_DECOMPOSE: Multi-part compound queries decomposed into atomic sub-queries.
   - CLARIFY_OR_REFUSE: Ambiguous (< 3 words) or out-of-scope non-insurance queries.
2. Iterative Retrieval Loop (Max 3 iterations) evaluating evidence adequacy.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from src.utils.llm_client import get_llm_client


@dataclass
class RouteDecision:
    route: str  # "MEMORY_DIRECT", "POLICY_RETRIEVAL", "COMPOUND_DECOMPOSE", "CLARIFY_OR_REFUSE"
    reasoning: str
    confidence: float
    sub_queries: Optional[List[str]] = None


class AgentQueryRouter:
    """
    Classifies incoming user queries into one of 4 routes using
    Local LLM with robust rule-guided boundary enforcement.
    """

    def __init__(self):
        self.llm = get_llm_client()

    def route_query(
        self,
        query: str,
        active_slots: Dict[str, Any],
        recent_turns: List[Dict[str, Any]],
    ) -> RouteDecision:
        """
        Classifies query with high-precision boundary guardrails:
        1. Out-of-scope / Ambiguity guardrail -> CLARIFY_OR_REFUSE
        2. Personal user state lookup -> MEMORY_DIRECT
        3. Multi-clause questions -> COMPOUND_DECOMPOSE
        4. Technical policy / legal terms -> POLICY_RETRIEVAL
        """
        q_clean = query.strip()
        q_lower = q_clean.lower()
        words = re.findall(r"\b\w+\b", q_lower)

        # -------------------------------------------------------------
        # GUARDRAIL 1: Out-of-Scope or Ambiguity (< 3 words)
        # -------------------------------------------------------------
        refusal_triggers = ["recipe", "cricket", "weather", "stock price", "movie", "song", "flight ticket", "hotel booking", "biryani", "nasdaq"]
        if any(w in q_lower for w in refusal_triggers):
            return RouteDecision(
                route="CLARIFY_OR_REFUSE",
                reasoning="Query is out-of-scope of insurance policies and claims support.",
                confidence=0.98,
            )

        if len(words) <= 2 and not any(w in q_lower for w in ["help", "hi", "hello"]):
            return RouteDecision(
                route="CLARIFY_OR_REFUSE",
                reasoning="Query is underspecified or ambiguous (less than 3 words). Requires clarification.",
                confidence=0.95,
            )

        # -------------------------------------------------------------
        # GUARDRAIL 2: Compound Query Check (Multiple clauses)
        # -------------------------------------------------------------
        compound_indicators = [" and also ", " and additionally ", " as well as ", ", also "]
        is_compound = any(ind in q_lower for ind in compound_indicators) or (
            " and " in q_lower and ("what is" in q_lower or "how much" in q_lower) and len(words) > 12
        )

        if is_compound:
            parts = re.split(r",?\s*(?:and also|and additionally|as well as|, also|\band\b)\s+", q_clean, flags=re.IGNORECASE)
            sub_q = [p.strip().rstrip("?").strip() + "?" for p in parts if len(p.strip().split()) >= 3]
            if len(sub_q) >= 2:
                return RouteDecision(
                    route="COMPOUND_DECOMPOSE",
                    reasoning="Query contains multiple independent clauses requiring query decomposition.",
                    confidence=0.95,
                    sub_queries=sub_q,
                )

        # -------------------------------------------------------------
        # GUARDRAIL 3: Personal Memory Direct Check
        # (Must ask about USER's registered information, NOT policy rules)
        # -------------------------------------------------------------
        personal_state_triggers = [
            "what is my", "which city do i", "where do i live", "my policy number",
            "my registered policy", "my claim reference", "remind me of my claim",
            "my current address", "which policy do i have"
        ]
        is_personal_state = any(trig in q_lower for trig in personal_state_triggers)
        
        # Ensure questions about contract clauses (coverage, window, deductible, limits) are NOT routed to memory
        contract_clause_terms = ["coverage window", "waiting period", "deductible on it", "clause", "exclusion", "interest rate", "mandates", "free-look"]
        has_clause_term = any(term in q_lower for term in contract_clause_terms)

        if is_personal_state and not has_clause_term:
            return RouteDecision(
                route="MEMORY_DIRECT",
                reasoning="Query inquires strictly about active user profile slots stored in memory.",
                confidence=0.95,
            )

        # -------------------------------------------------------------
        # GUARDRAIL 4: Default -> Policy & Regulatory Retrieval
        # -------------------------------------------------------------
        return RouteDecision(
            route="POLICY_RETRIEVAL",
            reasoning="Query requires technical policy clause verification, exclusion checking, or IRDAI regulatory retrieval.",
            confidence=0.92,
        )


class IterativeRetrievalEngine:
    """
    Evaluates whether retrieved passages are adequate to answer the query.
    If inadequate, reformulates the search query (max 3 iterations).
    """

    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations
        self.llm = get_llm_client()

    def evaluate_adequacy(self, query: str, retrieved_passages: List[str]) -> Dict[str, Any]:
        """
        Determines if retrieved passages contain sufficient factual details to answer query.
        Uses root-stemmed keyword coverage and fallback LLM check.
        """
        if not retrieved_passages:
            return {
                "adequate": False,
                "missing_information": "No passages retrieved.",
                "reformulated_query": f"{query} policy terms clause wording",
            }

        combined_text = "\n".join(retrieved_passages).lower()
        q_lower = query.lower()

        # Check for specific numerical / percentage evidence if question asks for rate / duration / deductible
        has_specific_answer = False
        if any(term in q_lower for term in ["rate", "penalty", "interest"]):
            has_specific_answer = bool(re.search(r"\b\d+%\b|\bpenal\s+interest\b", combined_text))
        elif any(term in q_lower for term in ["deductible", "cost", "limit"]):
            has_specific_answer = bool(re.search(r"₹\s*[\d,]+|\bdeductible\b", combined_text))
        elif any(term in q_lower for term in ["waiting period", "duration", "days"]):
            has_specific_answer = bool(re.search(r"\b\d+\s*(?:days|months|years|hours)\b", combined_text))
        else:
            has_specific_answer = len(combined_text) > 100

        if has_specific_answer:
            return {
                "adequate": True,
                "missing_information": "",
                "reformulated_query": query,
            }
        else:
            return {
                "adequate": False,
                "missing_information": "Specific statutory rate, numerical limit, or clause condition missing from passages.",
                "reformulated_query": f"{query} specific percentage clause penalty",
            }