# Memory-Augmented Agentic RAG for Insurance Claims and Policy Support
**Course:** Conversational AI (AIMLCZG521) — EC-1 Assessment (Assignment 2)  
**Problem Statement:** Problem Statement 1  
**Author:** Abdullah Farhan  
**Date:** November 2026  

---

## 1. Project Overview

This repository implements a production-grade **Memory-Augmented Agentic Retrieval-Augmented Generation (RAG)** system specialized for insurance claims and policy support under Insurance Regulatory and Development Authority of India (IRDAI) regulations.

### Key Capabilities & Architectural Components
* **Structure-Aware & Contextual Retrieval:** Preserves contractual clause integrity and injects Anthropic-style document provenance metadata into dense embeddings (`all-MiniLM-L6-v2`), eliminating ambiguous cross-product matches across Health, Motor, and Regulatory guidelines.
* **Two-Tier Hybrid Memory Store:** Separates active conversational state and structured slots (`policy_number`, `claim_reference`, `city`, `engine_cc`) in short-term relational storage (SQLite) from durable cross-session facts and summaries in long-term vector storage (ChromaDB), bounded by a strict 600-token prompt context budget.
* **Pure LLM Slot & Fact Extraction:** High-accuracy entity and relational extraction powered locally via Ollama (`llama3.2`), achieving **98.65% slot recall**.
* **Agentic Query Routing & Iterative Retrieval:** Classifies queries into 4 distinct execution paths (`MEMORY_DIRECT`, `POLICY_RETRIEVAL`, `COMPOUND_DECOMPOSE`, `CLARIFY_OR_REFUSE`) with **100.0% classification accuracy**, backed by an iterative adequacy feedback loop.
* **Student-Implemented Prompt Cache & Model Router:** Combines $O(1)$ SHA-256 cryptographic exact caching with dense semantic cosine caching ($\ge 0.75$) and dynamic model routing, achieving **66.0% cache hit rate**, **-67.3% token savings**, **-76.5% latency reduction**, and **-74.2% cost savings**.

---

## 2. Repository Structure

```
assignment_2/
├── REPORT.md                         # Comprehensive 8-module technical report
├── README.md                         # Project documentation and reproduction guide
├── requirements.txt                  # Python dependencies
├── notebooks/
│   └── main_assignment.ipynb        # Fully executed end-to-end Jupyter Notebook
├── src/
│   ├── chunking/
│   │   └── evaluate_chunking.py      # Task 1: Fixed vs. Structure-Aware vs. Semantic chunking
│   ├── contextual_retrieval/
│   │   ├── contextual_retriever.py   # Task 2: Anthropic-style metadata augmentation
│   │   └── evaluate_contextual.py    # Task 2: Retrieval benchmark (Recall@5, MRR)
│   ├── memory/
│   │   ├── short_term_store.py       # Task 3: SQLite relational slots and turn history
│   │   ├── long_term_store.py        # Task 3: ChromaDB persistent vector store
│   │   ├── hybrid_memory_manager.py  # Task 3: LLM extraction, consolidation, token budgeting
│   │   ├── test_memory.py            # Task 3: 6-part memory verification suite
│   │   └── evaluate_memory.py        # Task 4: 15-conversation multi-turn memory benchmark
│   ├── routing/
│   │   ├── agent_router.py           # Task 5: 4-path agentic classifier and query decomposer
│   │   ├── iterative_retrieval.py    # Task 5: Evidence adequacy feedback loop
│   │   └── evaluate_routing.py       # Task 5: Routing & convergence evaluation
│   ├── cache/
│   │   ├── student_cache.py          # Task 6: Two-tier SHA-256 + semantic prompt cache
│   │   ├── model_router.py           # Task 6: Small vs. Large dynamic model router
│   │   └── evaluate_cache.py         # Task 6: 50-turn workload economic simulation
│   ├── utils/
│   │   └── llm_client.py             # Resilient multi-provider client (Local Ollama / Fallback API)
│   └── build_notebook.py             # Automated notebook generator
└── data/
    ├── raw/                          # Legal insurance contracts and IRDAI guidelines
    │   ├── health_policy_optima_care.md
    │   ├── motor_policy_secure_drive.md
    │   ├── irdai_guidelines_grievance.md
    │   ├── irdai_master_circular_claims.md
    │   ├── insurance_qa_and_grievance_faq.md
    │   └── policy_annexures_and_schedules.md
    ├── processed/                    # Segmented chunks and contextualized collections
    │   ├── chunks_structure_aware.json
    │   └── contextualized_chunks.json
    └── evaluation/                   # Evaluation datasets, CSV results, and generated figures
        ├── ground_truth_qa.json
        ├── multi_turn_conversations.json
        ├── task1_results.csv
        ├── task1_chunking_comparison.png
        ├── task2_results.csv
        ├── task2_contextual_comparison.png
        ├── task4_results.csv
        ├── task4_memory_correctness.png
        ├── task4_memory_token_growth.png
        ├── task5_routing_results.csv
        ├── task6_economics_results.csv
        └── task6_economics_comparison.png
```

---

## 3. Setup and Installation

### Prerequisites
* Python 3.12+
* [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
* [Ollama](https://ollama.com/) with `llama3.2` pulled (recommended for local execution):
  ```bash
  ollama pull llama3.2
  ollama serve
  ```

### Virtual Environment Setup
```bash
# Clone or navigate into the directory
cd assignment_2

# Install dependencies using uv
uv sync
# OR using standard pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Configuration (`.env`)
Create a `.env` file in the root directory:
```bash
# Provider selection: "auto", "ollama", or "cloud"
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

---

## 4. Reproducing Results

Each task can be executed independently from the command line:

### Task 1: Chunking Strategy Comparison
```bash
python -m src.chunking.evaluate_chunking
```
*Outputs: `data/evaluation/task1_results.csv`, `data/evaluation/task1_chunking_comparison.png`*

### Task 2: Contextual Retrieval Implementation
```bash
python -m src.contextual_retrieval.evaluate_contextual
```
*Outputs: `data/evaluation/task2_results.csv`, `data/evaluation/task2_contextual_comparison.png`*

### Task 3: Hybrid Memory Verification Suite
```bash
python -m src.memory.test_memory
```
*Executes all 6 unit verification tests for SQLite slots, ChromaDB semantic recall, and 600-token cap enforcement.*

### Task 4: Multi-Turn Memory Evaluation
```bash
python -m src.memory.evaluate_memory
```
*Evaluates 15 conversations (60 turns). Outputs: `data/evaluation/task4_results.csv`, `data/evaluation/task4_memory_correctness.png`, `data/evaluation/task4_memory_token_growth.png`*

### Task 5: Agentic Query Routing & Iterative Retrieval
```bash
python -m src.routing.evaluate_routing
```
*Evaluates 12 diverse test queries and executes the iterative retrieval adequacy feedback loop.*

### Task 6: Prompt Cache & Model Routing Economics
```bash
python -m src.cache.evaluate_cache
```
*Simulates 50 customer turns across Baseline, Cache Only, and Cache + Model Routing.*

### Rebuilding & Executing the Complete Notebook
```bash
# Rebuild notebook structure
python -m src.build_notebook

# Execute in-place
jupyter nbconvert --to notebook --execute --inplace notebooks/main_assignment.ipynb
```

---

## 5. Summary of Benchmark Results

| Module / Task | Key Metric | Observed Performance | Benchmark Baseline / Contrast |
| :--- | :--- | :---: | :---: |
| **Task 1: Chunking** | Clause Split Frequency (%) | **47.46% (Structure-Aware)** | 55.93% (Fixed-Size) / 50.85% (Semantic) |
| **Task 1: Chunking** | Recall@5 (%) | **100.0%** | 100.0% (Fixed-Size) / 91.67% (Semantic) |
| **Task 2: Contextual Retrieval** | Recall@5 / MRR | **100.0% / 0.8333** | 100.0% / 0.8750 (Raw Chunks) |
| **Task 3: Memory Store** | 600-Token Budget Headroom | **63.5% Headroom (219 tok)** | Strictly $\le 600$ tokens |
| **Task 4: Multi-Turn Memory** | Slot Recall Accuracy (%) | **98.65% (Ollama LLM)** | 28.33% (Memory Disabled Baseline) |
| **Task 4: Multi-Turn Memory** | Answer Correctness (%) | **100.00%** | 28.33% (Memory Disabled Baseline, +71.67% $\Delta$) |
| **Task 4: Memory Context** | Mean Turn 4 Context Tokens | **234.8 tokens** | 39.1% of 600-token cap |
| **Task 5: Agentic Routing** | Classification Accuracy (%) | **100.0% (12/12 correct)** | 66.7% (Unconstrained Zero-Shot) |
| **Task 5: Iterative Retrieval** | Convergence Iterations | **2 iterations** | Guaranteed $\le 3$ iterations |
| **Task 6: Prompt Caching** | Cache Hit Rate (%) | **66.0% (33/50 turns)** | 0.0% (No Cache) |
| **Task 6: Cost Savings** | Cost per 1,000 Turns ($) | **\$0.0276 (-74.2%)** | \$0.1073 (Baseline) |
| **Task 6: Latency Reduction** | Mean Latency per Turn | **176.2 ms (-76.5%)** | 750.0 ms (Baseline) |

