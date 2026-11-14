"""
src/memory/hybrid_memory_manager.py
-----------------------------------
Task 3 & 4: Two-Tier Hybrid Memory Manager & Write Path Coordinator.
Coordinates:
1. Short-Term Memory (SQLite: turns, sessions, exact slot store).
2. Long-Term Memory (ChromaDB: durable facts, cross-session semantic search).
3. Pure LLM Slot Extractor & Summarizer (Local Llama 3.2 via Ollama).
4. Prompt Context Assembler with strict 600-token budget and explicit drop priority.
"""

import os
import re
import json
from typing import Dict, Any, List, Optional
import tiktoken

from src.memory.short_term_memory import ShortTermMemoryStore
from src.memory.long_term_memory import LongTermMemoryStore
from src.utils.llm_client import get_llm_client


class HybridMemoryManager:
    """
    Coordinates Short-Term (SQLite) and Long-Term (ChromaDB) stores,
    handles turn writes, fact/slot extraction via LLM, periodic session summaries,
    and constructs a token-budgeted prompt context.
    """

    def __init__(
        self,
        db_path: str = "data/memory_short_term.db",
        chroma_dir: str = "data/chroma_memory",
        vector_dir: Optional[str] = None,
        token_budget: int = 600,
        max_context_tokens: Optional[int] = None,
        consolidation_turn_threshold: int = 4,
        tokenizer_name: str = "cl100k_base",
        **kwargs,
    ):
        self.chroma_path = vector_dir or chroma_dir
        self.short_term = ShortTermMemoryStore(db_path=db_path)
        self.long_term = LongTermMemoryStore(persist_dir=self.chroma_path)
        self.token_budget = max_context_tokens or token_budget
        self.consolidation_turn_threshold = consolidation_turn_threshold
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.llm = get_llm_client()

    def count_tokens(self, text: str) -> int:
        """Count tokens using cl100k_base tokenizer."""
        return len(self.tokenizer.encode(text))

    def _extract_facts_llm(self, text: str) -> Dict[str, Any]:
        """
        Pure LLM Slot Extractor using Local Llama 3.2 via Ollama.
        Extracts structured insurance entities directly from user dialogue into JSON.
        """
        system_prompt = (
            "You are an expert insurance entity extraction AI. Analyze the user's message and extract any of these structured slots:\n"
            "- policy_number: full policy identifier (e.g., POL-OC-88219, MOT-SD-40192, POL-OC-11204, MOT-SD-77192)\n"
            "- product_type: full product name (e.g., Optima Care Health Insurance, SecureDrive Private Car Insurance)\n"
            "- claim_reference: full claim reference (e.g., CLM-2024-8831, CLM-2024-9900, CLM-2024-1100)\n"
            "- city: current registered city (if user mentions relocating or moving from CityA to CityB, extract CityB)\n"
            "- car_engine_cc: vehicle engine displacement (e.g., 1600 cc)\n\n"
            "Rules:\n"
            "1. Output ONLY a valid JSON dictionary with non-empty extracted values.\n"
            "2. If no insurance slots are mentioned, return {}.\n"
            '3. Example: {"policy_number": "POL-OC-88219", "product_type": "Optima Care Health Insurance", "city": "Bengaluru"}'
        )
        prompt = f"User message to extract slots from:\n\"{text}\""

        try:
            raw_response = self.llm.generate(prompt=prompt, system_prompt=system_prompt, timeout=8.0)
            if not raw_response:
                return {}
            # Extract JSON block even if model includes markdown fences
            json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                if isinstance(data, dict):
                    # Filter out null or empty strings
                    return {k: str(v).strip() for k, v in data.items() if v and str(v).lower() not in ["none", "null"]}
        except Exception:
            pass
        return {}

    def extract_facts(self, text: str) -> Dict[str, Any]:
        """
        Task 3 & 4 Write Path: Delegates 100% of fact extraction to the LLM (Ollama).
        """
        return self._extract_facts_llm(text)

    def _summarize_session_turns(self, turns: List[Dict[str, Any]]) -> str:
        """Summarize recent conversation turns using Local LLM."""
        if not turns:
            return ""

        dialogue = "\n".join([f"{t['role'].capitalize()}: {t['content']}" for t in turns])
        system_prompt = (
            "You are an insurance assistant. Summarize the following customer conversation in 1-2 concise factual sentences. "
            "Highlight the policy number, claim status, or grievance issue discussed."
        )
        prompt = f"Summarize this interaction:\n{dialogue}"

        try:
            summary = self.llm.generate(prompt=prompt, system_prompt=system_prompt, timeout=8.0)
            if summary:
                return summary.strip()
        except Exception:
            pass

        # Fallback if model unreachable
        user_queries = [t['content'] for t in turns if t['role'] == 'user']
        return f"Customer inquired regarding: {'; '.join(user_queries[:2])}."

    def write_turn(
        self,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_response: str = "",
    ) -> Dict[str, Any]:
        """
        Memory Write Path:
        1. Record turn into ShortTermMemoryStore (SQLite).
        2. Extract entities via LLM and update structured slot store.
        3. Detect durable medical/vehicle conditions and store in ChromaDB long-term memory.
        4. Every N turns, consolidate older turns into session summary.
        """
        # 1. Record User Turn
        self.short_term.record_turn(session_id=session_id, role="user", content=user_message, user_id=user_id)
        if assistant_response:
            self.short_term.record_turn(session_id=session_id, role="assistant", content=assistant_response, user_id=user_id)

        # 2. Slot Extraction via LLM & Write to SQLite
        extracted_slots = self.extract_facts(user_message)
        for slot_k, slot_v in extracted_slots.items():
            self.short_term.set_slot(session_id=session_id, slot_name=slot_k, slot_value=str(slot_v))

        # 3. Detect and store durable user conditions in Long-Term Memory (ChromaDB)
        msg_lower = user_message.lower()
        durable_triggers = ["diagnosed with", "pre-existing", "suffer from", "hypertension", "diabetes", "asthma", "heart condition"]
        if any(trig in msg_lower for trig in durable_triggers):
            self.long_term.add_durable_fact(
                user_id=user_id,
                fact_text=f"Medical Condition / Health History: {user_message}",
                category="health_profile",
                session_id=session_id,
            )

        # 4. Periodic Consolidation Trigger (Every 4 turns)
        turn_count = self.short_term.get_turn_count(session_id)
        if turn_count > 0 and turn_count % self.consolidation_turn_threshold == 0:
            recent_turns = self.short_term.get_recent_turns(session_id, limit=self.consolidation_turn_threshold)
            summary = self._summarize_session_turns(recent_turns)
            self.short_term.update_session_summary(session_id, summary)
            self.long_term.add_session_summary(user_id=user_id, session_id=session_id, summary_text=summary)

        return {"status": "success", "extracted_slots": extracted_slots}

    # Backward compatibility alias
    def process_turn(self, session_id: str, user_id: str, role: str, content: str) -> Dict[str, Any]:
        if role == "user":
            return self.write_turn(session_id=session_id, user_id=user_id, user_message=content)
        else:
            self.short_term.record_turn(session_id=session_id, role=role, content=content, user_id=user_id)
            return {"status": "success"}

    def assemble_context(
        self,
        session_id: str,
        user_id: str,
        current_query: str = "",
        query: str = "",
        top_k_long_term: int = 3,
        recent_turns_limit: int = 6,
        **kwargs,
    ) -> Dict[str, Any]:
        target_query = current_query or query
        """
        Memory Read Path & Token Budget Assembly:
        Assembles memory context strictly within self.token_budget (600 tokens).
        Drop Priority:
        1. Long-Term Semantic Memories (dropped first if over budget)
        2. Older Short-Term Dialogue Turns (dropped turn-by-turn)
        3. Structured Active Slot Table (NEVER dropped: critical user state)
        """
        # Priority 3: Active Slots (Must always fit)
        active_slots = self.short_term.get_all_slots(session_id)
        slot_lines = [f"- {k}: {v}" for k, v in active_slots.items()]
        slots_block = (
            "=== ACTIVE POLICY & PROFILE SLOTS (EXACT STATE) ===\n"
            + ("\n".join(slot_lines) if slot_lines else "- No registered policy slots.")
        )

        # Priority 1: Long-Term Durable Facts & Cross-Session Memories
        long_memories = self.long_term.recall_facts(
            user_id=user_id,
            query=target_query,
            top_k=top_k_long_term,
            similarity_threshold=0.30,
        )
        long_lines = [f"• [{m.get('category','fact')}] {m.get('fact_text','')}" for m in long_memories]

        # Priority 2: Recent Short-Term Dialogue Turns
        recent_turns = self.short_term.get_recent_turns(session_id, limit=recent_turns_limit)
        turn_lines = [f"{t['role'].capitalize()}: {t['content']}" for t in recent_turns]

        dropped_long_term = 0
        dropped_turns = 0

        # Assembly loop with explicit budget enforcement
        while True:
            sections = [slots_block]
            if long_lines:
                sections.append("=== RECALLED LONG-TERM FACTS & PRIOR SESSIONS ===\n" + "\n".join(long_lines))
            if turn_lines:
                sections.append("=== RECENT DIALOGUE HISTORY ===\n" + "\n".join(turn_lines))

            assembled_text = "\n\n".join(sections)
            total_tokens = self.count_tokens(assembled_text)

            if total_tokens <= self.token_budget:
                break

            # Drop 1: Drop least relevant long-term memory
            if long_lines:
                long_lines.pop()
                dropped_long_term += 1
                continue

            # Drop 2: Drop oldest turn
            if len(turn_lines) > 1:
                turn_lines.pop(0)
                dropped_turns += 1
                continue

            break

        return {
            "assembled_context": assembled_text,
            "total_tokens": total_tokens,
            "token_budget": self.token_budget,
            "active_slots": active_slots,
            "dropped_long_term": dropped_long_term,
            "dropped_turns": dropped_turns,
        }

    # Backward compatibility alias
    def assemble_prompt_context(self, session_id: str, user_id: str, query: str, **kwargs) -> Dict[str, Any]:
        res = self.assemble_context(session_id=session_id, user_id=user_id, current_query=query, **kwargs)
        return {
            "context_string": res["assembled_context"],
            "token_count": res["total_tokens"],
            "budget_tokens": res["token_budget"],
            "active_slots": res["active_slots"],
            "dropped_long_term_count": res["dropped_long_term"],
            "dropped_turns_count": res["dropped_turns"],
        }