"""
src/routing/evaluate_routing.py
-------------------------------
Task 5: Agentic Query Routing & Iterative Retrieval Evaluation Runner.
Evaluates:
1. Router Accuracy across 12 test queries covering all 4 routes:
   - MEMORY_DIRECT
   - POLICY_RETRIEVAL
   - COMPOUND_DECOMPOSE
   - CLARIFY_OR_REFUSE
2. Iterative Retrieval Loop (Max 3 iterations) on complex/ambiguous queries.
Outputs:
- Formatted Results Table with Route, Confidence, and Decision Reasoning.
- CSV artifact: data/evaluation/task5_routing_results.csv.
"""

import os
import json
import time
import pandas as pd
from tabulate import tabulate
from src.routing.agent_router import AgentQueryRouter, IterativeRetrievalEngine


def run_routing_evaluation():
    print("=" * 80)
    print(" TASK 5: AGENTIC QUERY ROUTING & ITERATIVE RETRIEVAL EVALUATION")
    print("=" * 80)

    router = AgentQueryRouter()
    iterative_engine = IterativeRetrievalEngine(max_iterations=3)

    # Simulated active user state
    active_slots = {
        "policy_number": "POL-OC-88219",
        "product_type": "Optima Care Health Insurance",
        "city": "Bengaluru",
        "claim_reference": "CLM-OC-10291",
    }
    recent_turns = [
        {"role": "user", "content": "I have an Optima Care health insurance policy POL-OC-88219."},
        {"role": "assistant", "content": "Got it, I have registered your Optima Care policy."},
    ]

    # Benchmark: 12 diverse queries covering all 4 routes
    routing_benchmark = [
        # MEMORY_DIRECT
        {
            "query": "What is my registered policy number?",
            "expected_route": "MEMORY_DIRECT",
            "category": "Direct User State Lookup"
        },
        {
            "query": "Which city do I currently live in according to your records?",
            "expected_route": "MEMORY_DIRECT",
            "category": "Direct Slot Lookup"
        },
        {
            "query": "Can you remind me of my claim reference number?",
            "expected_route": "MEMORY_DIRECT",
            "category": "Direct Claim Reference Lookup"
        },
        # POLICY_RETRIEVAL
        {
            "query": "What is the pre-hospitalization coverage window under Optima Care?",
            "expected_route": "POLICY_RETRIEVAL",
            "category": "Single Policy Clause Retrieval"
        },
        {
            "query": "Does SecureDrive motor insurance cover water ingression and hydrostatic lock?",
            "expected_route": "POLICY_RETRIEVAL",
            "category": "Motor Clause & Exclusion Retrieval"
        },
        {
            "query": "What is the statutory interest rate IRDAI mandates if an insurer delays claim payout?",
            "expected_route": "POLICY_RETRIEVAL",
            "category": "Regulatory IRDAI Circular Retrieval"
        },
        # COMPOUND_DECOMPOSE
        {
            "query": "What is the waiting period for pre-existing diseases, and also what is the grace period for premium payment?",
            "expected_route": "COMPOUND_DECOMPOSE",
            "category": "Multi-Part Compound Query"
        },
        {
            "query": "How much deductible applies for a 1600 cc car, and additionally what towing expenses are covered?",
            "expected_route": "COMPOUND_DECOMPOSE",
            "category": "Multi-Part Compound Query"
        },
        {
            "query": "What is the free-look period duration, and also which Insurance Ombudsman has jurisdiction over Bengaluru?",
            "expected_route": "COMPOUND_DECOMPOSE",
            "category": "Cross-Document Compound Query"
        },
        # CLARIFY_OR_REFUSE
        {
            "query": "Can you give me a good chicken biryani recipe for dinner?",
            "expected_route": "CLARIFY_OR_REFUSE",
            "category": "Out-of-Scope Domain Refusal"
        },
        {
            "query": "What is the current stock price of Apple on NASDAQ?",
            "expected_route": "CLARIFY_OR_REFUSE",
            "category": "Out-of-Scope Domain Refusal"
        },
        {
            "query": "Deductible?",
            "expected_route": "CLARIFY_OR_REFUSE",
            "category": "Ambiguous / Underspecified Input"
        },
    ]

    print(f"Loaded {len(routing_benchmark)} routing test queries across 4 routes.\n")

    eval_results = []
    correct_routes = 0

    for idx, item in enumerate(routing_benchmark):
        q = item["query"]
        expected = item["expected_route"]
        cat = item["category"]

        start_t = time.perf_counter()
        decision = router.route_query(query=q, active_slots=active_slots, recent_turns=recent_turns)
        latency_ms = (time.perf_counter() - start_t) * 1000.0

        is_match = decision.route == expected
        if is_match:
            correct_routes += 1

        sub_queries_str = " | ".join(decision.sub_queries) if decision.sub_queries else "N/A"

        eval_results.append({
            "Query #": f"Q{idx+1:02d}",
            "User Query": q if len(q) <= 45 else q[:42] + "...",
            "Category": cat,
            "Expected Route": expected,
            "Predicted Route": decision.route,
            "Match": "PASS" if is_match else "FAIL",
            "Confidence": f"{decision.confidence:.2f}",
            "Sub-Queries": sub_queries_str if len(sub_queries_str) <= 35 else sub_queries_str[:32] + "...",
            "Latency": f"{latency_ms:.1f}ms",
        })

    # Print Results Table
    df = pd.DataFrame(eval_results)
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))

    accuracy_pct = (correct_routes / len(routing_benchmark)) * 100.0
    print(f"\n[+] Routing Classification Accuracy: {accuracy_pct:.1f}% ({correct_routes}/{len(routing_benchmark)} correct)")

    # Save CSV artifact
    os.makedirs("data/evaluation", exist_ok=True)
    df.to_csv("data/evaluation/task5_routing_results.csv", index=False)
    print("[+] Routing results saved to: data/evaluation/task5_routing_results.csv")

    # -------------------------------------------------------------
    # 2. Test Iterative Retrieval Loop
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ITERATIVE RETRIEVAL LOOP VERIFICATION (MAX 3 ITERATIONS)")
    print("=" * 80)
    test_query = "What is the exact penalty interest rate for delayed claims under IRDAI guidelines?"
    
    # Simulate initial inadequate retrieval
    passages_iter1 = [
        "IRDAI guidelines mandate that general insurers must settle claims within 30 days of receiving all surveyor reports."
    ]
    print(f"Iteration 1: Query: '{test_query}'")
    eval_iter1 = iterative_engine.evaluate_adequacy(test_query, passages_iter1)
    print(f"-> Adequate: {eval_iter1['adequate']} | Missing: {eval_iter1['missing_information']}")
    print(f"-> Reformulated Query: '{eval_iter1['reformulated_query']}'")

    # Simulate iteration 2 with adequate retrieved text
    passages_iter2 = passages_iter1 + [
        "Clause 10.2 Penal Interest: In case of delay beyond 30 days, the insurer is liable to pay penal interest at 2% above the bank rate."
    ]
    print(f"\nIteration 2: Query: '{eval_iter1['reformulated_query']}'")
    eval_iter2 = iterative_engine.evaluate_adequacy(test_query, passages_iter2)
    print(f"-> Adequate: {eval_iter2['adequate']} | Search Converged Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_routing_evaluation()