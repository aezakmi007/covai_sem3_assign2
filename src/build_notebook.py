"""
src/build_notebook.py
---------------------
Generates the comprehensive, end-to-end Jupyter Notebook
at notebooks/main_assignment.ipynb covering all 7 tasks.
"""

import os
import json


def create_assignment_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    cell_counter = 1

    def add_md(source):
        nonlocal cell_counter
        notebook["cells"].append({
            "cell_type": "markdown",
            "id": f"cell_md_{cell_counter}",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })
        cell_counter += 1

    def add_code(source):
        nonlocal cell_counter
        notebook["cells"].append({
            "cell_type": "code",
            "id": f"cell_code_{cell_counter}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })
        cell_counter += 1

    # =========================================================================
    # Header & Title
    # =========================================================================
    add_md("""# Conversational AI (AIMLCZG521) — EC-1 Assessment
## Memory-Augmented Agentic RAG for Insurance Claims and Policy Support
**Course:** Conversational AI (AIMLCZG521)  
**Problem Statement:** Problem Statement 1  
**Deliverable:** End-to-End Executable Demonstration Notebook
""")

    # =========================================================================
    # Section 0: Setup & Imports
    # =========================================================================
    add_md("""---
### Section 0: Environment Setup, Dependencies & Imports
Ensures project root is on Python's path, loads configuration from `.env`, and suppresses harmless background warnings.""")

    add_code("""import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from tabulate import tabulate
from IPython.display import Image, display

# Ensure project root is in sys.path whether run from repo root or notebooks/
repo_root = Path.cwd().resolve()
if repo_root.name == "notebooks":
    repo_root = repo_root.parent
os.chdir(repo_root)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

print("Python Version:", sys.version.split()[0])
print("Repository Root:", repo_root)
print("Environment Loaded Successfully!")""")

    # =========================================================================
    # Section 1: Task 1 (Chunking Strategy Comparison)
    # =========================================================================
    add_md("""---
### Section 1: Task 1 — Corpus Preparation & Chunking Comparison (1 Mark)
Evaluates **Fixed-Size**, **Structure-Aware**, and **Semantic** chunking on 81 legal insurance clauses across 12 benchmark questions.""")

    add_code("""from src.chunking.evaluate_chunking import main as evaluate_chunking

# Run Task 1 Benchmark
evaluate_chunking()

# Display Visual Plot
if os.path.exists("data/evaluation/task1_chunking_comparison.png"):
    display(Image(filename="data/evaluation/task1_chunking_comparison.png"))""")

    # =========================================================================
    # Section 2: Task 2 (Contextual Retrieval Implementation)
    # =========================================================================
    add_md("""---
### Section 2: Task 2 — Contextual Retrieval Implementation (1 Mark)
Evaluates **Raw Chunks** vs. **Contextualized Chunks** across the 307-chunk working collection (Anthropic-style document provenance).""")

    add_code("""from src.contextual_retrieval.evaluate_contextual import main as evaluate_contextual

# Run Task 2 Benchmark
evaluate_contextual()

# Display Visual Plot
if os.path.exists("data/evaluation/task2_contextual_comparison.png"):
    display(Image(filename="data/evaluation/task2_contextual_comparison.png"))""")

    # =========================================================================
    # Section 3: Task 3 (Hybrid Short-Term and Long-Term Memory Store)
    # =========================================================================
    add_md("""---
### Section 3: Task 3 — Hybrid Short-Term & Long-Term Memory Store (2 Marks)
Verifies:
1. SQLite relational turn sequence and structured slot storage.
2. $O(1)$ exact lookup and atomic slot overwrite on user correction.
3. ChromaDB persistent vector store for durable facts and cross-session recall.
4. Strict 600-token budget context assembly.""")

    add_code("""from src.memory.test_memory import run_memory_tests

# Run Task 3 Verification Suite
run_memory_tests()""")

    # =========================================================================
    # Section 4: Task 4 (Multi-Turn Memory Evaluation)
    # =========================================================================
    add_md("""---
### Section 4: Task 4 — Multi-Turn Memory Evaluation (1 Mark)
Evaluates **Memory Enabled** vs. **Memory Disabled (Naive Baseline)** across 15 multi-turn conversations (60 total turns).""")

    add_code("""from src.memory.evaluate_memory import evaluate_multi_turn_memory

# Run Task 4 Benchmark
evaluate_multi_turn_memory()

# Display Correctness Comparison Chart
if os.path.exists("data/evaluation/task4_memory_correctness.png"):
    display(Image(filename="data/evaluation/task4_memory_correctness.png"))

# Display Token Growth Chart
if os.path.exists("data/evaluation/task4_memory_token_growth.png"):
    display(Image(filename="data/evaluation/task4_memory_token_growth.png"))""")

    # =========================================================================
    # Section 5: Task 5 (Agentic Query Routing & Iterative Retrieval)
    # =========================================================================
    add_md("""---
### Section 5: Task 5 — Agentic Query Routing & Iterative Retrieval (2 Marks)
Evaluates:
1. Router classification across 4 routes (`MEMORY_DIRECT`, `POLICY_RETRIEVAL`, `COMPOUND_DECOMPOSE`, `CLARIFY_OR_REFUSE`).
2. Iterative Retrieval loop evaluating retrieved evidence adequacy (max 3 iterations).""")

    add_code("""from src.routing.evaluate_routing import run_routing_evaluation

# Run Task 5 Benchmark
run_routing_evaluation()""")

    # =========================================================================
    # Section 6: Task 6 (Prompt Cache & Model Routing Economics)
    # =========================================================================
    add_md("""---
### Section 6: Task 6 — Prompt Cache & Model Routing Economics (1 Mark)
Replays 50 customer turns comparing:
- Condition A: No Cache (Large Model Baseline)
- Condition B: Prompt Cache Only
- Condition C: Prompt Cache + Dynamic Model Routing""")

    add_code("""from src.cache.evaluate_cache import evaluate_task6

# Run Task 6 50-turn Simulation
evaluate_task6()

# Display Economics & Latency Comparison Chart
if os.path.exists("data/evaluation/task6_economics_comparison.png"):
    display(Image(filename="data/evaluation/task6_economics_comparison.png"))""")

    # =========================================================================
    # Section 7: Task 7 (Failure Analysis & Error Diagnosis)
    # =========================================================================
    add_md("""---
### Section 7: Task 7 — Memory & Retrieval Failure Analysis (1 Mark)
Summary table of the 5 concrete failure cases diagnosed and resolved during development.""")

    add_code("""failure_cases = [
    {
        "ID": "Case 1",
        "Category": "Retrieval Failure",
        "Failure Mode": "Clause Severing in Fixed Chunking",
        "Root Cause": "500-char window cut Clause 5.10 between exclusion rule and exception.",
        "Engineering Fix": "Structure-Aware Chunking preserving clause boundaries (100% Recall@5)."
    },
    {
        "ID": "Case 2",
        "Category": "Retrieval Failure",
        "Failure Mode": "Semantic Dilution in Contextual Chunks",
        "Root Cause": "62-token prepended header pulled embedding towards document centroid.",
        "Engineering Fix": "Compact header templates bounding overhead while keeping provenance."
    },
    {
        "ID": "Case 3",
        "Category": "Memory State Failure",
        "Failure Mode": "Coreference Collapse in Naive RAG",
        "Root Cause": "Stateless retrieval failed on pronoun 'it' on Turn 4 (28.33% accuracy).",
        "Engineering Fix": "SQLite slot store perpetually injecting active policy state (100% accuracy)."
    },
    {
        "ID": "Case 4",
        "Category": "Memory State Failure",
        "Failure Mode": "City Relocation Collision",
        "Root Cause": "Append-only logs contained both Mumbai & Bengaluru, confusing Ombudsman.",
        "Engineering Fix": "Relational ON CONFLICT DO UPDATE with directional regex overwrite."
    },
    {
        "ID": "Case 5",
        "Category": "Agentic State Failure",
        "Failure Mode": "Router Memory vs. Corpus Confusion",
        "Root Cause": "Zero-shot router sent policy clause questions to memory (66.7% accuracy).",
        "Engineering Fix": "Strict guardrails: personal queries to memory, contract terms to retrieval (100%)."
    }
]

df_failures = pd.DataFrame(failure_cases)
print(tabulate(df_failures, headers="keys", tablefmt="fancy_grid", showindex=False))
print("\\n[+] Pipeline verification complete. All 7 modules successfully executed!")""")

    # Save notebook
    os.makedirs("notebooks", exist_ok=True)
    out_path = "notebooks/main_assignment.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"[+] Successfully generated Jupyter Notebook: {out_path}")


if __name__ == "__main__":
    create_assignment_notebook()