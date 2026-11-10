"""
src/evaluate_chunking.py
Task 1: Chunking Strategy Evaluation Runner
Compares:
- Fixed-Size Chunking
- Structure-Aware Chunking
- Semantic Chunking
Reports Chunk Count, Mean Length, Std Dev, Clause Split %, and Recall@5.
"""

import os
import re
import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from src.chunking.chunking import ChunkingEngine


def load_raw_documents(raw_dir: str = "data/raw") -> Dict[str, str]:
    """Loads all markdown policy documents from data/raw directory."""
    documents = {}
    for filename in sorted(os.listdir(raw_dir)):
        if filename.endswith(".md"):
            filepath = os.path.join(raw_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                documents[filename] = f.read()
    return documents


def evaluate_retrieval(
    chunks: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
    model: SentenceTransformer,
    top_k: int = 5,
) -> float:
    """
    Computes Recall@k on the test questions against the given chunks.
    A question is a 'hit' if the target clause is found within the top-k chunks.
    """
    if not chunks or not questions:
        return 0.0

    # Extract chunk texts and encode them into normalized vectors
    chunk_texts = [c["text"] for c in chunks]
    chunk_embeddings = model.encode(
        chunk_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )

    hits = 0
    for q in questions:
        query_text = q["question"]
        target_clause = q["target_clause_id"]  # e.g., 'Clause 3.3'
        target_doc = q["target_doc"]

        # Encode query into normalized vector
        query_emb = model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)

        # Compute cosine similarity (dot product of unit vectors)
        similarities = np.dot(chunk_embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Check if any of the top_k chunks contains the target clause from the target doc
        matched = False
        for idx in top_indices:
            candidate_chunk = chunks[idx]
            if candidate_chunk["doc_name"] == target_doc:
                # Check if target clause ID appears in the chunk's text or metadata
                if target_clause in candidate_chunk.get("text", "") or target_clause in candidate_chunk.get("clauses_present", []):
                    matched = True
                    break

        if matched:
            hits += 1

    recall_at_k = (hits / len(questions)) * 100.0
    return round(recall_at_k, 2)


def main():
    print("=" * 70)
    print(" TASK 1: CHUNKING STRATEGY COMPARISON & RETRIEVAL EVALUATION")
    print("=" * 70)

    # 1. Load raw documents
    docs = load_raw_documents("data/raw")
    print(f"Loaded {len(docs)} documents: {list(docs.keys())}\n")

    # 2. Extract all distinct clause IDs across all documents for baseline count
    all_clause_ids = set()
    for doc_name, text in docs.items():
        found = re.findall(r"Clause\s+\d+\.\d+", text)
        all_clause_ids.update(found)
    print(f"Total Unique Clauses in Corpus: {len(all_clause_ids)}")

    # 3. Load ground-truth QA pairs
    with open("data/evaluation/ground_truth_qa.json", "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    print(f"Loaded {len(qa_data)} Ground Truth Questions for Recall@5 Evaluation\n")

    # 4. Initialize chunking engine
    engine = ChunkingEngine(embedding_model_name="all-MiniLM-L6-v2")

    strategies = {
        "Fixed-Size (500 chars, 100 ov)": lambda text, doc: engine.fixed_size_chunking(
            text, doc, chunk_size=500, overlap=100
        ),
        "Structure-Aware (Clauses)": lambda text, doc: engine.structure_aware_chunking(
            text, doc
        ),
        "Semantic (Distance Thresh 0.45)": lambda text, doc: engine.semantic_chunking(
            text, doc, distance_threshold=0.45
        ),
    }

    results = []
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    for strat_name, strat_fn in strategies.items():
        print(f"--> Processing strategy: {strat_name}...")
        all_chunks = []
        for doc_name, text in docs.items():
            chunks = strat_fn(text, doc_name)
            all_chunks.extend(chunks)

        # Compute chunk length & splitting metrics
        metrics = ChunkingEngine.calculate_metrics(all_chunks, list(all_clause_ids))

        # Evaluate Recall@5 on test questions
        recall_5 = evaluate_retrieval(all_chunks, qa_data, engine.model, top_k=5)

        # Save chunks for later tasks
        clean_strat_name = strat_name.split()[0].lower().replace("-", "_")
        out_filepath = os.path.join(processed_dir, f"chunks_{clean_strat_name}.json")
        with open(out_filepath, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2)

        results.append({
            "Strategy": strat_name,
            "Chunk Count": metrics["chunk_count"],
            "Mean Length (chars)": metrics["mean_len"],
            "Std Dev (chars)": metrics["std_len"],
            "Split Clauses Count": metrics["split_clauses_count"],
            "Split Clause Freq (%)": f"{metrics['split_clause_frequency_pct']}%",
            "Recall@5 (%)": f"{recall_5}%",
            "_raw_recall": recall_5,
            "_raw_split_freq": metrics["split_clause_frequency_pct"],
        })

    # 5. Display comparison table
    df_results = pd.DataFrame(results)
    display_cols = [
        "Strategy",
        "Chunk Count",
        "Mean Length (chars)",
        "Std Dev (chars)",
        "Split Clause Freq (%)",
        "Recall@5 (%)",
    ]
    print("\n" + "=" * 70)
    print("TASK 1 EXPERIMENTAL RESULTS TABLE:")
    print("=" * 70)
    print(tabulate(df_results[display_cols], headers="keys", tablefmt="fancy_grid", showindex=False))

    # 6. Save results to CSV for notebook & report
    df_results[display_cols].to_csv("data/evaluation/task1_results.csv", index=False)
    print("\n[+] Results saved to: data/evaluation/task1_results.csv")

    # 7. Generate comparison visualization plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x_labels = [r["Strategy"].split()[0] for r in results]
    x = np.arange(len(x_labels))
    width = 0.35

    recall_vals = [r["_raw_recall"] for r in results]
    split_vals = [r["_raw_split_freq"] for r in results]

    rects1 = ax1.bar(x - width/2, recall_vals, width, label="Recall@5 (%)", color="#2b5c8f")
    rects2 = ax1.bar(x + width/2, split_vals, width, label="Clause Split Freq (%)", color="#d95f02")

    ax1.set_ylabel("Percentage (%)", fontsize=12)
    ax1.set_title("Task 1: Chunking Strategy Performance & Clause Integrity", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=11)
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Attach numbers on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f"{height}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    for rect in rects2:
        height = rect.get_height()
        ax1.annotate(f"{height}%", xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    chart_path = "data/evaluation/task1_chunking_comparison.png"
    plt.savefig(chart_path, dpi=200)
    print(f"[+] Comparison Chart saved to: {chart_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()