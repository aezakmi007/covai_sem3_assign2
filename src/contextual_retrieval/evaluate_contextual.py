"""
src/contextual_retrieval/evaluate_contextual.py
Task 2: Empirical Evaluation of Raw vs. Contextualized Retrieval
Compares:
1. Raw Chunk Retrieval
2. Contextualized Chunk Retrieval
Metrics:
- Recall@5
- Mean Reciprocal Rank (MRR)
- Indexing Token Overhead and Generation Costs
"""

import os
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from src.contextual_retrieval.contextual_engine import ContextualRetrievalEngine


def evaluate_retrieval_metrics(
    chunks: List[Dict[str, Any]],
    text_key: str,
    questions: List[Dict[str, Any]],
    model: SentenceTransformer,
    top_k: int = 5,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Computes Recall@k and Mean Reciprocal Rank (MRR) for a given chunk text key.
    - Recall@k: % of queries with target clause in top-k
    - MRR: average of 1 / rank of first relevant chunk across all queries
    """
    chunk_texts = [c[text_key] for c in chunks]
    chunk_embeddings = model.encode(
        chunk_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )

    hits = 0
    reciprocal_ranks = []
    per_query_details = []

    for q in questions:
        query_text = q["question"]
        target_clause = q["target_clause_id"]  # e.g., 'Clause 3.3'
        target_doc = q["target_doc"]

        query_emb = model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
        similarities = np.dot(chunk_embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        matched_rank = None
        for rank_idx, idx in enumerate(top_indices):
            candidate_chunk = chunks[idx]
            # Match document and clause ID
            doc_matched = (candidate_chunk["doc_name"] == target_doc) or (target_doc in candidate_chunk["doc_name"])
            clause_matched = (target_clause in candidate_chunk["clause_id"]) or (target_clause in candidate_chunk["raw_text"])

            if doc_matched and clause_matched:
                matched_rank = rank_idx + 1  # 1-indexed rank
                break

        if matched_rank is not None:
            hits += 1
            rr = 1.0 / matched_rank
            reciprocal_ranks.append(rr)
        else:
            reciprocal_ranks.append(0.0)

        per_query_details.append({
            "query_id": q["id"],
            "target": target_clause,
            "rank": matched_rank if matched_rank else "Not in Top 5",
            "reciprocal_rank": round(reciprocal_ranks[-1], 3),
        })

    recall_at_k = (hits / len(questions)) * 100.0
    mrr = float(np.mean(reciprocal_ranks))
    return round(recall_at_k, 2), round(mrr, 4), per_query_details


def main():
    print("=" * 75)
    print(" TASK 2: CONTEXTUAL RETRIEVAL IMPLEMENTATION & EVALUATION")
    print("=" * 75)

    # 1. Initialize Engine & Token Counter
    engine = ContextualRetrievalEngine()
    print("1. Curating working collection (Cap: 300–800 chunks)...")
    raw_chunks = engine.load_and_curate_chunks(raw_dir="data/raw", target_cap=350)
    print(f"   [+] Curated Working Collection Size: {len(raw_chunks)} chunks (Cap = 350)\n")

    # 2. Augment Chunks with Context
    print("2. Augmenting chunks with 1-2 sentence contextual metadata...")
    augmented_chunks = engine.augment_chunks(raw_chunks)

    # Save contextualized chunks to data/processed
    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/contextualized_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(augmented_chunks, f, indent=2)
    print(f"   [+] Saved {len(augmented_chunks)} augmented chunks to: {out_path}\n")

    # 3. Calculate Indexing Cost & Token Statistics
    total_raw_tokens = sum(c["raw_tokens"] for c in augmented_chunks)
    total_context_tokens = sum(c["context_tokens"] for c in augmented_chunks)
    total_augmented_tokens = sum(c["total_tokens"] for c in augmented_chunks)
    token_overhead_pct = (total_context_tokens / total_raw_tokens) * 100.0

    print("3. Indexing & Token Cost Analysis:")
    print(f"   - Total Chunks Indexed:         {len(augmented_chunks)}")
    print(f"   - Total Raw Chunk Tokens:       {total_raw_tokens:,} tokens (Avg: {total_raw_tokens/len(augmented_chunks):.1f} tok/chunk)")
    print(f"   - Total Context Metadata Tokens:{total_context_tokens:,} tokens (Avg: {total_context_tokens/len(augmented_chunks):.1f} tok/chunk)")
    print(f"   - Total Augmented Tokens:       {total_augmented_tokens:,} tokens")
    print(f"   - Context Token Overhead:       +{token_overhead_pct:.2f}%\n")

    # 4. Load Ground-Truth QA benchmark
    with open("data/evaluation/ground_truth_qa.json", "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"4. Loaded {len(questions)} test questions for retrieval benchmarking.\n")

    # 5. Initialize Embedding Model
    print("5. Generating vector embeddings with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 6. Evaluate Raw vs. Contextualized Retrieval
    print("   -> Evaluating Raw Chunks...")
    raw_recall5, raw_mrr, raw_details = evaluate_retrieval_metrics(
        augmented_chunks, text_key="raw_text", questions=questions, model=model, top_k=5
    )

    print("   -> Evaluating Contextualized Chunks...")
    ctx_recall5, ctx_mrr, ctx_details = evaluate_retrieval_metrics(
        augmented_chunks, text_key="contextualized_text", questions=questions, model=model, top_k=5
    )

    # 7. Print Comparative Results Table
    results = [
        {
            "Retrieval Method": "Raw Chunks (Baseline)",
            "Recall@5 (%)": f"{raw_recall5}%",
            "Mean Reciprocal Rank (MRR)": raw_mrr,
            "Total Indexed Tokens": f"{total_raw_tokens:,}",
            "Avg Tokens / Chunk": f"{total_raw_tokens/len(augmented_chunks):.1f}",
        },
        {
            "Retrieval Method": "Contextualized Chunks (Anthropic-style)",
            "Recall@5 (%)": f"{ctx_recall5}%",
            "Mean Reciprocal Rank (MRR)": ctx_mrr,
            "Total Indexed Tokens": f"{total_augmented_tokens:,}",
            "Avg Tokens / Chunk": f"{total_augmented_tokens/len(augmented_chunks):.1f}",
        },
    ]

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 75)
    print("TASK 2: RAW VS. CONTEXTUALIZED RETRIEVAL RESULTS:")
    print("=" * 75)
    print(tabulate(df_results, headers="keys", tablefmt="fancy_grid", showindex=False))

    # Save results to CSV
    df_results.to_csv("data/evaluation/task2_results.csv", index=False)
    print("\n[+] Results saved to: data/evaluation/task2_results.csv")

    # 8. Generate Visual Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart 1: Recall@5 & MRR
    categories = ["Raw Chunks", "Contextualized Chunks"]
    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(x - width/2, [raw_recall5, ctx_recall5], width, label="Recall@5 (%)", color="#3470a3")
    ax1.bar(x + width/2, [raw_mrr * 100, ctx_mrr * 100], width, label="MRR (scaled x100)", color="#2ca02c")
    ax1.set_ylabel("Score (%)", fontsize=11)
    ax1.set_title("Retrieval Quality: Recall@5 vs. MRR", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotations
    ax1.annotate(f"{raw_recall5}%", (-width/2, raw_recall5 + 2), ha="center", fontweight="bold")
    ax1.annotate(f"{ctx_recall5}%", (1 - width/2, ctx_recall5 + 2), ha="center", fontweight="bold")
    ax1.annotate(f"{raw_mrr:.3f}", (width/2, raw_mrr * 100 + 2), ha="center", fontweight="bold")
    ax1.annotate(f"{ctx_mrr:.3f}", (1 + width/2, ctx_mrr * 100 + 2), ha="center", fontweight="bold")

    # Bar chart 2: Token Indexing Cost
    tokens = [total_raw_tokens, total_augmented_tokens]
    colors = ["#7f7f7f", "#d95f02"]
    bars = ax2.bar(categories, tokens, color=colors, width=0.5)
    ax2.set_ylabel("Total Tokens Indexed", fontsize=11)
    ax2.set_title("Indexing Token Overhead (+{:.1f}%)".format(token_overhead_pct), fontsize=12, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:,} tok", (bar.get_x() + bar.get_width()/2, h + 500), ha="center", fontweight="bold")

    plt.tight_layout()
    chart_path = "data/evaluation/task2_contextual_comparison.png"
    plt.savefig(chart_path, dpi=200)
    print(f"[+] Comparison Chart saved to: {chart_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()