"""
src/cache/evaluate_cache.py
---------------------------
Task 6: 50-Turn Prompt Cache & Model Routing Economics Evaluation.
Compares:
- Condition A: No Cache, Large Model Only (Naive Baseline)
- Condition B: Prompt Cache Only + Large Model
- Condition C: Prompt Cache + Dynamic Model Routing (Optimal System)

Measures:
- Cache Hit Rate (%) [Exact vs Semantic vs Miss]
- Total Tokens & Token Savings (%)
- Latency per Turn (ms) & Latency Reduction (%)
- Cost per 1,000 Turns ($)
Outputs:
- Formatted Results Table
- CSV: data/evaluation/task6_economics_results.csv
- Visual Comparison Chart: data/evaluation/task6_economics_comparison.png
"""

import os
import time
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

from src.cache.custom_cache import StudentPromptCache, DynamicModelRouter
from src.routing.agent_router import AgentQueryRouter


def generate_50_turn_workload() -> list:
    """
    Constructs a realistic 50-turn insurance customer workload
    with authentic query distributions:
    - 20 unique policy / claim queries
    - 15 exact repeat queries (popular FAQs)
    - 10 semantic rephrasings of earlier questions
    - 5 memory slot lookup queries
    """
    core_questions = [
        ("What is the pre-hospitalization coverage window under Optima Care?", "POLICY_RETRIEVAL", 180, 45),
        ("Does SecureDrive motor insurance cover water ingression and hydrostatic lock?", "POLICY_RETRIEVAL", 220, 60),
        ("What is the statutory interest rate IRDAI mandates if an insurer delays claim payout?", "POLICY_RETRIEVAL", 195, 50),
        ("How much compulsory deductible applies for a 1600 cc private car?", "POLICY_RETRIEVAL", 160, 40),
        ("What is the free-look cancellation period for health policies?", "POLICY_RETRIEVAL", 150, 40),
        ("What is my registered policy number?", "MEMORY_DIRECT", 80, 20),
        ("Which city do I currently live in according to your records?", "MEMORY_DIRECT", 75, 15),
        ("Can you remind me of my claim reference number?", "MEMORY_DIRECT", 85, 20),
        ("What is the waiting period for pre-existing diseases under Optima Care?", "POLICY_RETRIEVAL", 210, 55),
        ("What is the depreciation rate on nylon and rubber plastic parts under motor insurance?", "POLICY_RETRIEVAL", 175, 45),
        ("What is the maximum claim amount that can be escalated to the Insurance Ombudsman?", "POLICY_RETRIEVAL", 190, 45),
        ("What is the waiting period for pre-existing diseases, and also what is the grace period for premium payment?", "COMPOUND_DECOMPOSE", 320, 90),
        ("How much deductible applies for a 1600 cc car, and additionally what towing expenses are covered?", "COMPOUND_DECOMPOSE", 310, 85),
        ("What is the grace period for renewing an annual health policy?", "POLICY_RETRIEVAL", 165, 40),
        ("Are ambulance charges covered under the Optima Care policy?", "POLICY_RETRIEVAL", 170, 35),
    ]

    # Rephrasings for semantic match
    semantic_variants = [
        ("What is the pre-hospitalization coverage duration under Optima Care?", 0),
        ("Does SecureDrive car insurance cover engine water damage and hydrostatic lock?", 1),
        ("What is the penalty interest rate IRDAI mandates for delayed claim settlements?", 2),
        ("What is the compulsory deductible amount for cars with 1600 cc engines?", 3),
        ("How many days do I have for free-look policy cancellation?", 4),
        ("Where do I currently reside according to my profile?", 6),
        ("What is the PED waiting duration under Optima Care health insurance?", 8),
        ("What depreciation applies to rubber and plastic parts in car claims?", 9),
        ("What is the monetary jurisdiction limit of the Insurance Ombudsman?", 10),
        ("What is the renewal grace period for health insurance policies?", 13),
    ]

    workload = []
    # Turns 1-15: First-time queries (initial population)
    for q, route, in_tok, out_tok in core_questions:
        workload.append({"query": q, "route": route, "in_tokens": in_tok, "out_tokens": out_tok, "type": "UNIQUE"})

    # Turns 16-30: Exact repeats of popular FAQ questions
    repeat_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 3, 4, 10]
    for idx in repeat_indices:
        q, route, in_tok, out_tok = core_questions[idx]
        workload.append({"query": q, "route": route, "in_tokens": in_tok, "out_tokens": out_tok, "type": "EXACT_REPEAT"})

    # Turns 31-40: Semantic rephrasings
    for variant_q, ref_idx in semantic_variants:
        _, route, in_tok, out_tok = core_questions[ref_idx]
        workload.append({"query": variant_q, "route": route, "in_tokens": in_tok, "out_tokens": out_tok, "type": "SEMANTIC_REPHRASE"})

    # Turns 41-50: Mix of repeats and slot queries
    final_indices = [5, 6, 7, 0, 1, 2, 8, 11, 12, 14]
    for idx in final_indices:
        q, route, in_tok, out_tok = core_questions[idx]
        workload.append({"query": q, "route": route, "in_tokens": in_tok, "out_tokens": out_tok, "type": "EXACT_REPEAT"})

    return workload


def evaluate_task6():
    print("=" * 80)
    print(" TASK 6: PROMPT CACHE & MODEL ROUTING ECONOMICS EVALUATION (50 TURNS)")
    print("=" * 80)

    workload = generate_50_turn_workload()
    print(f"Constructed realistic 50-turn customer workload.")
    print(f"Distribution: 15 Initial Queries | 25 Exact Repeats | 10 Semantic Paraphrases\n")

    router = DynamicModelRouter()

    # -------------------------------------------------------------
    # CONDITION A: No Cache, Large Model Only (Naive Baseline)
    # -------------------------------------------------------------
    cond_a_tokens = 0
    cond_a_cost = 0.0
    cond_a_latencies = []

    for turn in workload:
        in_tok = turn["in_tokens"]
        out_tok = turn["out_tokens"]
        cond_a_tokens += (in_tok + out_tok)
        cond_a_cost += router.compute_turn_cost("large_model", in_tok, out_tok)
        cond_a_latencies.append(router.PRICING["large_model"]["base_latency_ms"])

    # -------------------------------------------------------------
    # CONDITION B: Prompt Cache Only + Large Model
    # -------------------------------------------------------------
    cache_b = StudentPromptCache(semantic_threshold=0.75)
    cond_b_tokens = 0
    cond_b_cost = 0.0
    cond_b_latencies = []

    for turn in workload:
        q = turn["query"]
        cached_resp, hit_type, _ = cache_b.lookup(q)
        if hit_type != "MISS":
            # Cache Hit: 0 tokens, 0 cost, ~1.5ms memory retrieval
            cond_b_latencies.append(1.5)
        else:
            # Cache Miss: Call Large Model and populate cache
            in_tok = turn["in_tokens"]
            out_tok = turn["out_tokens"]
            cond_b_tokens += (in_tok + out_tok)
            cond_b_cost += router.compute_turn_cost("large_model", in_tok, out_tok)
            cond_b_latencies.append(router.PRICING["large_model"]["base_latency_ms"])
            cache_b.put(q, f"Response for {q}", input_tokens=in_tok, output_tokens=out_tok)

    # -------------------------------------------------------------
    # CONDITION C: Prompt Cache + Dynamic Model Routing (Optimal System)
    # -------------------------------------------------------------
    cache_c = StudentPromptCache(semantic_threshold=0.75)
    cond_c_tokens = 0
    cond_c_cost = 0.0
    cond_c_latencies = []
    small_model_count = 0
    large_model_count = 0

    for turn in workload:
        q = turn["query"]
        route = turn["route"]
        cached_resp, hit_type, _ = cache_c.lookup(q)
        if hit_type != "MISS":
            cond_c_latencies.append(1.5)
        else:
            model_tier = router.route_model(q, route)
            in_tok = turn["in_tokens"]
            out_tok = turn["out_tokens"]
            cond_c_tokens += (in_tok + out_tok)
            cond_c_cost += router.compute_turn_cost(model_tier, in_tok, out_tok)
            cond_c_latencies.append(router.PRICING[model_tier]["base_latency_ms"])
            if model_tier == "small_model":
                small_model_count += 1
            else:
                large_model_count += 1
            cache_c.put(q, f"Response for {q}", input_tokens=in_tok, output_tokens=out_tok)

    # Cache Performance Breakdown
    hit_stats = cache_c.get_hit_rate()

    # -------------------------------------------------------------
    # Print Results Summary Table
    # -------------------------------------------------------------
    summary_rows = [
        {
            "Condition / Architecture": "Condition A: No Cache (Large Model Only)",
            "Cache Hit Rate (%)": "0.0% (No Cache)",
            "Total Tokens Consumed": f"{cond_a_tokens:,} tok",
            "Token Savings (%)": "Baseline (0.0%)",
            "Mean Latency / Turn": f"{pd.Series(cond_a_latencies).mean():.1f} ms",
            "Latency Reduction (%)": "Baseline (0.0%)",
            "Cost / 1,000 Turns ($)": f"${(cond_a_cost / 50.0) * 1000.0:.4f}",
            "Cost Savings (%)": "Baseline (0.0%)",
        },
        {
            "Condition / Architecture": "Condition B: Prompt Cache Only (Large Model)",
            "Cache Hit Rate (%)": f"{hit_stats['hit_rate_pct']:.1f}% ({hit_stats['total_hits']}/50)",
            "Total Tokens Consumed": f"{cond_b_tokens:,} tok",
            "Token Savings (%)": f"-{((cond_a_tokens - cond_b_tokens)/cond_a_tokens)*100.0:.1f}%",
            "Mean Latency / Turn": f"{pd.Series(cond_b_latencies).mean():.1f} ms",
            "Latency Reduction (%)": f"-{((pd.Series(cond_a_latencies).mean() - pd.Series(cond_b_latencies).mean())/pd.Series(cond_a_latencies).mean())*100.0:.1f}%",
            "Cost / 1,000 Turns ($)": f"${(cond_b_cost / 50.0) * 1000.0:.4f}",
            "Cost Savings (%)": f"-{((cond_a_cost - cond_b_cost)/cond_a_cost)*100.0:.1f}%",
        },
        {
            "Condition / Architecture": "Condition C: Prompt Cache + Model Routing",
            "Cache Hit Rate (%)": f"{hit_stats['hit_rate_pct']:.1f}% ({hit_stats['total_hits']}/50)",
            "Total Tokens Consumed": f"{cond_c_tokens:,} tok",
            "Token Savings (%)": f"-{((cond_a_tokens - cond_c_tokens)/cond_a_tokens)*100.0:.1f}%",
            "Mean Latency / Turn": f"{pd.Series(cond_c_latencies).mean():.1f} ms",
            "Latency Reduction (%)": f"-{((pd.Series(cond_a_latencies).mean() - pd.Series(cond_c_latencies).mean())/pd.Series(cond_a_latencies).mean())*100.0:.1f}%",
            "Cost / 1,000 Turns ($)": f"${(cond_c_cost / 50.0) * 1000.0:.4f}",
            "Cost Savings (%)": f"-{((cond_a_cost - cond_c_cost)/cond_a_cost)*100.0:.1f}%",
        },
    ]

    df_summary = pd.DataFrame(summary_rows)
    print("\n" + tabulate(df_summary, headers="keys", tablefmt="fancy_grid", showindex=False))

    # Save CSV artifact
    os.makedirs("data/evaluation", exist_ok=True)
    df_summary.to_csv("data/evaluation/task6_economics_results.csv", index=False)
    print("\n[+] Economics results saved to: data/evaluation/task6_economics_results.csv")

    # -------------------------------------------------------------
    # Generate Visual Plot
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    conditions = ["Condition A\n(No Cache)", "Condition B\n(Cache Only)", "Condition C\n(Cache + Routing)"]
    costs = [(cond_a_cost / 50.0) * 1000.0, (cond_b_cost / 50.0) * 1000.0, (cond_c_cost / 50.0) * 1000.0]
    latencies = [pd.Series(cond_a_latencies).mean(), pd.Series(cond_b_latencies).mean(), pd.Series(cond_c_latencies).mean()]

    # Subplot 1: Cost per 1,000 Turns
    bars1 = ax1.bar(conditions, costs, color=["#d95f02", "#7570b3", "#1b9e77"], width=0.55, alpha=0.9)
    ax1.set_ylabel("Cost per 1,000 Turns ($ USD)", fontsize=11, fontweight="bold")
    ax1.set_title("API Economics: Cost per 1,000 Turns", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"${yval:.3f}", ha="center", va="bottom", fontweight="bold")

    # Subplot 2: Latency per Turn
    bars2 = ax2.bar(conditions, latencies, color=["#d95f02", "#7570b3", "#1b9e77"], width=0.55, alpha=0.9)
    ax2.set_ylabel("Mean Response Latency (ms)", fontsize=11, fontweight="bold")
    ax2.set_title("User Experience: Mean Latency per Turn", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f"{yval:.1f} ms", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    chart_path = "data/evaluation/task6_economics_comparison.png"
    plt.savefig(chart_path, dpi=200)
    print(f"[+] Economics chart saved to: {chart_path}")
    print("=" * 80)


if __name__ == "__main__":
    evaluate_task6()