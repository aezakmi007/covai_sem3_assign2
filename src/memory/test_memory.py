"""
src/memory/test_memory.py
Task 3 Verification Runner:
Validates Short-Term SQLite store, ChromaDB vector store,
slot overwrite, semantic recall across sessions, and token budgeting.
"""

import os
import shutil
from tabulate import tabulate

from src.memory.hybrid_memory_manager import HybridMemoryManager


def run_memory_tests():
    print("=" * 75)
    print(" TASK 3: HYBRID SHORT-TERM & LONG-TERM MEMORY VERIFICATION")
    print("=" * 75)

    # Clean previous test database files for a clean run
    test_db = "data/test_memory_short_term.db"
    test_chroma = "data/test_chroma_memory"
    if os.path.exists(test_db):
        os.remove(test_db)
    if os.path.exists(test_chroma):
        shutil.rmtree(test_chroma)

    manager = HybridMemoryManager(
        db_path=test_db,
        chroma_dir=test_chroma,
        token_budget=600,
        consolidation_turn_threshold=4,
    )

    test_results = []
    user_id = "policyholder_rahul"
    session_1 = "sess_2024_01"

    # -------------------------------------------------------------
    # TEST 1: Turn Recording & Slot Extraction
    # -------------------------------------------------------------
    print("\n[Test 1] Simulating Turn 1 (Initial Policy & City statement)...")
    t1_res = manager.write_turn(
        session_id=session_1,
        user_id=user_id,
        user_message="Hello, my name is Rahul. My policy number is POL-OC-88219 and I live in Mumbai.",
        assistant_response="Hello Rahul! I see your Optima Care Health Insurance policy POL-OC-88219 registered in Mumbai. How can I help you today?"
    )

    slots_t1 = manager.short_term.get_all_slots(session_1)
    pol_match = slots_t1.get("policy_number") == "POL-OC-88219"
    city_match = slots_t1.get("city") == "Mumbai"
    test_results.append(("Slot Extraction (policy_number)", "PASS" if pol_match else "FAIL", slots_t1.get("policy_number")))
    test_results.append(("Slot Extraction (city)", "PASS" if city_match else "FAIL", slots_t1.get("city")))

    # -------------------------------------------------------------
    # TEST 2: Exact Slot Lookup
    # -------------------------------------------------------------
    exact_lookup_val = manager.short_term.get_slot(session_1, "policy_number")
    exact_pass = exact_lookup_val == "POL-OC-88219"
    test_results.append(("Exact Slot Lookup (get_slot)", "PASS" if exact_pass else "FAIL", exact_lookup_val))

    # -------------------------------------------------------------
    # TEST 3: User Fact Correction (Newer value must take precedence)
    # -------------------------------------------------------------
    print("[Test 3] Simulating Turn 2 (User Correction: moved to Bengaluru)...")
    manager.write_turn(
        session_id=session_1,
        user_id=user_id,
        user_message="Actually, I recently relocated from Mumbai to Bengaluru last month.",
        assistant_response="Understood! I have updated your registered location to Bengaluru."
    )
    updated_city = manager.short_term.get_slot(session_1, "city")
    correction_pass = updated_city == "Bengaluru"
    test_results.append(("Slot Correction Precedence (City update)", "PASS" if correction_pass else "FAIL", f"Updated to: {updated_city}"))

    # -------------------------------------------------------------
    # TEST 4: Long-Term Memory (Writing Durable Health & Vehicle Facts)
    # -------------------------------------------------------------
    print("[Test 4] Simulating Turn 3 (Stating Pre-Existing Disease)...")
    manager.write_turn(
        session_id=session_1,
        user_id=user_id,
        user_message="I was diagnosed with Type 2 Diabetes in 2021, and take daily medications.",
        assistant_response="Thank you for sharing. I have recorded your pre-existing condition (Type 2 Diabetes since 2021) in your health profile."
    )
    durable_facts = manager.long_term.get_all_user_facts(user_id)
    fact_stored_pass = len(durable_facts) > 0
    test_results.append(("Long-Term Memory Write (ChromaDB)", "PASS" if fact_stored_pass else "FAIL", f"{len(durable_facts)} facts stored"))

    # -------------------------------------------------------------
    # TEST 5: Cross-Session Semantic Recall (Different Wording)
    # -------------------------------------------------------------
    session_2 = "sess_2024_02"  # New session weeks later
    print("\n[Test 5] Simulating New Session (Semantic Recall with completely different wording)...")
    query = "Will my policy cover routine blood sugar tests and insulin expenses?"
    recalled = manager.long_term.recall_facts(user_id=user_id, query=query, top_k=2, similarity_threshold=0.30)
    
    recalled_pass = len(recalled) > 0 and any("diabetes" in r["fact_text"].lower() or "health" in r["fact_text"].lower() for r in recalled)
    top_score = recalled[0]["similarity_score"] if recalled else 0.0
    test_results.append(("Cross-Session Semantic Recall", "PASS" if recalled_pass else "FAIL", f"Score: {top_score} | Recalled: {recalled[0]['fact_text'] if recalled else 'None'}"))

    # -------------------------------------------------------------
    # TEST 6: Context Assembly within Explicit Token Budget (600 tokens)
    # -------------------------------------------------------------
    print("\n[Test 6] Assembling Context for Current Turn...")
    assembled = manager.assemble_context(session_id=session_1, user_id=user_id, current_query=query)
    budget_pass = assembled["total_tokens"] <= manager.token_budget
    test_results.append(("Token Budget Compliance (<= 600 tok)", "PASS" if budget_pass else "FAIL", f"{assembled['total_tokens']} / 600 tokens"))

    # Display Results Table
    print("\n" + "=" * 75)
    print("TASK 3: MEMORY MODULE VERIFICATION SUMMARY")
    print("=" * 75)
    headers = ["Component / Feature Tested", "Status", "Observed Value / Details"]
    print(tabulate(test_results, headers=headers, tablefmt="fancy_grid"))

    print("\n--- ASSEMBLED CONTEXT PREVIEW ---")
    print(assembled["assembled_context"])
    print("=" * 75)


if __name__ == "__main__":
    run_memory_tests()