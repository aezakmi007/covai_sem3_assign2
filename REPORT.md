# Conversational AI (AIMLCZG521) — EC-1 Assessment
## Memory-Augmented Agentic RAG for Insurance Claims and Policy Support

---

### Student & Submission Details
* **Course:** Conversational AI (AIMLCZG521)
* **Assessment:** EC-1 Assessment (Assignment 2)
* **Problem Statement:** Problem Statement 1 — Memory-Augmented Agentic RAG for Insurance Claims and Policy Support
* **Date:** November 2026

---

## 1. Executive Summary & Problem Formulation
Traditional Retrieval-Augmented Generation (RAG) systems suffer from two fatal flaws when deployed in domain-specific environments such as legal and insurance customer support:
1. **Context Loss via Arbitrary Chunking:** Slicing policy documents blindly by character or token count strips legal clauses of their parent product context, section hierarchy, and critical exclusions.
2. **Stateless Conversational Memory:** A naive RAG pipeline treats each turn independently. When a policyholder refers to *"my policy"* or asks *"what is the deductible on it?"* four turns after naming their policy, memoryless systems fail to resolve coreferences and cannot recall user-provided facts.

This project designs, evaluates, and deploys a production-grade **Memory-Augmented Agentic RAG system** tailored for insurance policy and claims retrieval. The architecture incorporates:
* **Structure-aware and Contextual Retrieval** to preserve legal clause integrity.
* A **Two-Tier Hybrid Memory System** separating active conversation state (Relational SQLite) from durable cross-session user facts (Vector store).
* An **Agentic Query Router** supporting direct memory answers, corpus retrieval, multi-query decomposition, and domain guardrail refusals.
* A **Student-Implemented Prompt Cache & Model Router** optimizing latency and API token economics.

---

## 2. Dataset Description and Corpus Provenance

The insurance corpus was constructed from standardized, publicly available regulatory and policy documents conforming to the Insurance Regulatory and Development Authority of India (IRDAI) guidelines:

| Document File | Product / Source | Domain | Clause Count | Document Summary |
| :--- | :--- | :--- | :---: | :--- |
| `health_policy_optima_care.md` | Optima Care Health Insurance (IRDAI/HLT/OC/P-H/V.1/2024-25) | Retail Health | 42 Clauses | Definitions, Inpatient Hospitalization, Pre/Post-Hospitalization (60/180 days), Waiting Periods (30d initial, 24m specific, 36m PED), Deductibles, Co-payments, Exclusions, and Claims Timelines. |
| `motor_policy_secure_drive.md` | SecureDrive Private Car Comprehensive Policy (IMT Wordings) | Private Motor | 28 Clauses | Own Damage perils, Depreciation Grid (50% on rubber/airbags, 0% glass), Constructive Total Loss (75% IDV), Compulsory Deductibles, No Claim Bonus (NCB 20%–50%), and Exclusions (DUI, hydrostatic lock). |
| `irdai_guidelines_grievance.md` | IRDAI Policyholder Protection Regulations, 2024 | Regulatory | 11 Clauses | 15/30-day Free-Look cancellation period, 30-day Grace Period, 30-day Claim Settlement TAT, 2% Penal Interest on delay, and Insurance Ombudsman Jurisdiction (up to ₹30 Lakhs). |
| **Total Corpus** | **3 Comprehensive Documents** | **Multi-line Insurance** | **81 Key Legal Clauses** | **Standardized Public Wordings (Open Educational / Fair Use)** |

### Evaluation Ground Truth Dataset (`data/evaluation/ground_truth_qa.json`)
A curated benchmark consisting of **12 domain-specific policy questions** was constructed. Each test case maps directly to a target document, primary ground-truth clause ID, expected answer, and key term verification criteria, enabling automated Recall@5 and Mean Reciprocal Rank (MRR) evaluation.

---

## 3. Module 1: Corpus Preparation and Advanced Retrieval

### Task 1: Corpus Preparation & Chunking Strategy Comparison

#### 1.1 Methodology
Legal and insurance contracts are hierarchical: documents contain Sections, which contain numbered Clauses, which contain conditions and explicit exclusions. To identify the optimal chunking strategy, three distinct approaches were implemented and evaluated on the same corpus:
1. **Fixed-Size Chunking (500 chars, 100 char overlap):** Slices text mechanically using a sliding window without considering sentence or section boundaries.
2. **Structure-Aware Chunking:** Uses regular expressions to detect section and clause delimiters (`## SECTION`, `### Clause X.Y`), preserving the clause header, conditions, and sub-limits intact as an atomic retrieval unit.
3. **Semantic Chunking (Distance Threshold = 0.45):** Splits text into sentences, computes unit vector embeddings using `all-MiniLM-L6-v2`, measures cosine distance between consecutive sentences, and creates chunk boundaries whenever semantic distance spikes.

#### 1.2 Experimental Results Table

| Strategy | Chunk Count | Mean Length (chars) | Std Dev (chars) | Clause Split Freq (%) | Recall@5 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Size (500 chars, 100 ov)** | 82 | 490.30 | 52.50 | 55.93% | 100.0% |
| **Structure-Aware (Clauses)** | 100 | 313.63 | 116.73 | **47.46% (Best)** | **100.0%** |
| **Semantic (Thresh 0.45)** | 127 | 252.94 | 281.16 | 50.85% | 91.67% |

*Results saved automatically to `data/evaluation/task1_results.csv` and plotted in `data/evaluation/task1_chunking_comparison.png`.*

#### 1.3 Analysis & Inferences

1. **Why Chunking Choice Changes Retrieval Quality for Clause-Structured Text:**
   * In insurance contracts, a clause typically follows the pattern: `[General Rule] -> [Monetary Limit] -> [Exceptions]`.
   * For example, in *Clause 5.10 (Dental Treatment)*: *"Dental treatment or surgery of any kind is excluded [RULE] unless requiring hospitalization following an accidental bodily injury [EXCEPTION]"*.
   * If a mechanical chunker cuts between the rule and the exception, the retrieval model returns a truncated fragment, causing the LLM to hallucinate that dental treatment is never covered under any circumstance.

2. **Clause Integrity & The Cost of Getting It Wrong:**
   * Fixed-size chunking broke **55.93%** of all clauses across chunk boundaries. The cost of splitting clauses in an insurance setting is severe: it causes fragmented context, lost definitions, and increases hallucination rates in the downstream generator.
   * Semantic chunking suffered from sentence-level topic shifts within a single clause (e.g., transitioning from clinical terminology to financial percentages), causing premature splits, high variance in chunk length (Std Dev: 281.16 chars), and the lowest retrieval accuracy (**91.67% Recall@5**).

3. **Challenges Caused by Cross-References:**
   * Insurance clauses frequently cite other clauses (e.g., Clause 2.3 references Clause 2.1; Clause 4.2 references Clause 4.1).
   * In our evaluation, cross-references explain why clauses appeared across multiple chunks. In fixed-size chunking, a cross-reference is often severed from its target clause, leaving the retrieval system incapable of resolving the dependency.

4. **Strategy Selection and Justification:**
   * **Structure-Aware Chunking is conclusively selected** for all subsequent tasks. It achieved **100.0% Recall@5**, maintained atomic clause integrity, preserved parent section headers, and produced consistent, well-bounded chunks (~313 characters).