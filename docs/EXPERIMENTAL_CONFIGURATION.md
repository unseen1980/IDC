# Experimental Configuration: Complete Parameter Settings

This document provides all configuration parameters used in the IDC evaluation for reproducibility and thesis documentation.

---

## 1. Language Models and Embeddings

### 1.1 Intent Generation Model
- **Model**: `gemini-2.5-flash` (Google Gemini)
- **Purpose**: Generate diverse intents (questions) from documents
- **Temperature**: Default (creative but coherent)
- **Max output tokens**: 8,192 (for chunked mode)

### 1.2 Embedding Model
- **Model**: `gemini-embedding-001` (Google Gemini)
- **Dimension**: 1,536
- **Task type**: 
  - `RETRIEVAL_DOCUMENT` for sentences and chunks
  - `RETRIEVAL_QUERY` for intents and evaluation queries
- **Normalization**: L2 normalized for cosine similarity

---

## 2. IDC Algorithm Parameters

### 2.1 Default Parameters (Before Auto-tuning)

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| **Lambda** | λ | 0.1 | Length penalty weight (lower = longer chunks) |
| **Max length** | L_max | 20 | Maximum sentences per chunk |
| **Min length** | L_min | 3 | Minimum sentences per chunk |
| **Boundary penalty** | β | 1.2 | Cost per chunk boundary |
| **Coherence weight** | α | 0.3 | Weight for intra-chunk coherence bonus |
| **Merge adjacent** | — | 1 | Merge adjacent chunks with same intent |

### 2.2 Auto-Tuning Grid Search Space

**Used for SQuAD 2-docs evaluation** (n=293 spans enables reliable optimization)

| Parameter | Grid Values | Count |
|-----------|-------------|-------|
| **Lambda (λ)** | 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1 | 8 |
| **Boundary penalty (β)** | 0.20, 0.25, 0.30, 0.40, 0.60, 0.80, 1.00, 1.20 | 8 |
| **Max length (L_max)** | 12, 16, 20 | 3 |
| **Coherence weight (α)** | 0.0, 0.1, 0.2, 0.3 | 4 |

**Total configurations**: 8 × 8 × 3 × 4 = **768 combinations**

**Optimal values found for SQuAD**:
- λ = 0.0005 (200× smaller than default!)
- β = 1.2
- L_max = 20
- α = 0.2

---

## 3. Intent Generation Parameters

### 3.1 Default Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Number of questions** | 15 | Intents generated per document |
| **Generation multiplier** | 1.5 | Raw intents = questions × multiplier |
| **Diversity threshold** | 0.4 | Cosine similarity threshold for MMR clustering |
| **Questions per chunk** | 3 | For chunked mode (long documents) |
| **Chunk size** | 12,000 chars | For chunked intent generation |

### 3.2 Auto-Adaptive Parameters (arXiv, Qasper)

**Enabled with**: `AUTO_ADAPT_INTENTS=1`

**Adaptation formula**:
```
num_questions = BASE_NUM_QUESTIONS × (doc_length / BASE_LENGTH)^0.7

Where:
- BASE_NUM_QUESTIONS = 15
- BASE_LENGTH = 200 sentences
- doc_length = actual document sentences
```

**Example (arXiv, 495 sentences)**:
```
num_questions = 15 × (495 / 200)^0.7 ≈ 37 intents
generation_multiplier = 1.2 (adjusted for narrative text)
diversity_threshold = 0.4 (unchanged)
```

---

## 4. Baseline Methods Configuration

### 4.1 Fixed-Length Chunking
- **Chunk size**: 6 sentences
- **Overlap**: None
- **Symbol**: Fixed-6

### 4.2 Sliding Window
- **Window size**: 6 sentences
- **Stride**: 3 sentences (50% overlap)
- **Symbol**: Sliding-6r3

### 4.3 Coherence-Based (TextTiling-like)
- **Window size (W)**: 1 sentence
- **Min chunk length**: 1 sentence
- **Max chunk length**: 10 sentences
- **Approximate target**: 6 sentences
- **Symbol**: Coh-w1

### 4.4 Paragraph-Based
- **Split on**: Blank lines (natural paragraphs)
- **No parameters**: Structure-driven
- **Symbol**: Paragraphs

---

## 5. Retrieval Configuration

### 5.1 Multi-View Indexing

**Views enabled**: 4 views per chunk
1. **Text view**: Raw chunk text
2. **Intent view**: Best-matching predicted intent
3. **Summary view**: LLM-generated chunk summary
4. **Keywords view**: Extracted key phrases

### 5.2 Hybrid Retrieval

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Hybrid retrieval** | True | Combine dense + sparse |
| **Dense weight** | 0.6 (60%) | Semantic embedding similarity |
| **Sparse weight** | 0.4 (40%) | BM25 lexical matching |

### 5.3 Reranking

**Reranker**: `lexical` (applied to ALL methods)

**Reranking formula**:
```
final_score = semantic_score + 0.3 × bm25_score + 0.05 × lexical_overlap + length_penalty

Where:
- semantic_score: Initial dense retrieval score
- bm25_score: BM25 lexical matching
- lexical_overlap: Token overlap between query and chunk intent
- length_penalty: -0.05 if chunk < min_chunk_sent (2 sentences)
```

**Alternative reranker** (not used): `cross-encoder/ms-marco-MiniLM-L-6-v2`

### 5.4 Retrieval Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Top-K** | 5 | Retrieve top-5 chunks for R@5 |
| **Rerank candidates** | 50 | Initial candidates before reranking |
| **Min chunk sentences** | 2 | Penalize very short chunks |
| **Coherence weight** | 0.1 | Bonus for coherent chunks |

---

## 6. Evaluation Span Configuration

### 6.1 Pseudo-Span Generation

**Used for**: Datasets without pre-existing questions (arXiv, Fiori)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Minimum spans** | 15 | Target number of evaluation spans |
| **Similarity threshold** | 0.45 (initial) | Intent-sentence similarity for span creation |
| **Minimum threshold** | 0.10 | Floor to prevent negative thresholds |
| **Max iterations** | 10 | Threshold adjustment attempts |

### 6.2 Span Statistics by Dataset

| Dataset | Spans | Source | Avg Span Length |
|---------|-------|--------|-----------------|
| **SQuAD 2-docs** | 293 | Human Q&A | 1.2 sentences |
| **SQuAD 1-doc** | 12 | Human Q&A | 1.1 sentences |
| **NewsQA** | 15 | LLM-generated | 2.3 sentences |
| **arXiv** | 15 | LLM-generated | 1.8 sentences |
| **Qasper** | 10 | Human Q&A | 1.5 sentences |
| **Fiori** | 15 | LLM-generated | 1.5 sentences |

---

## 7. Contextual Embeddings and Density-Aware Features

### 7.1 Contextual Embeddings

**Enabled**: True (default)

**Configuration**:
```
chunk_embedding = (1 - 2×context_weight) × chunk_emb 
                  + context_weight × prev_chunk_emb
                  + context_weight × next_chunk_emb

Where:
- context_weight = 0.15 (15% per adjacent chunk)
- chunk_emb = mean of sentence embeddings in chunk
```

### 7.2 Density-Aware Segmentation

**Enabled**: True (default)

**Configuration**:
- **Discount factor**: 0.3
- **Purpose**: Reduce segmentation in dense information regions
- **Metric**: Information density based on vocabulary richness

---

## 8. Dataset-Specific Configurations

### 8.1 NewsQA
```bash
FORCE_INTENTS=1
FORCE_IDC_SEGMENTS=1  
FORCE_SPANS=1
DOC_NAME=newsqa_corpus
# Uses default parameters (no auto-tune, no auto-adapt)
```

### 8.2 SQuAD 1-doc (Normans)
```bash
AUTO_TUNE=1
AUTO_TUNE_BASELINES=1
DOC_NAME=Normans
# Enables auto-tuning (768 configurations tested)
```

### 8.3 SQuAD 2-docs
```bash
LIMIT=2
AUTO_TUNE=1
AUTO_TUNE_BASELINES=1
# Optimal λ=0.0005 found through grid search
```

### 8.4 arXiv (Finance NLP Paper)
```bash
AUTO_ADAPT_INTENTS=1
FORCE_INTENTS=1
FORCE_IDC_SEGMENTS=1
FORCE_SPANS=1
INPUT_FILE=data/arxiv_long/arxiv_bert_finance.txt
DOC_NAME=arxiv_bert_finance
# Auto-adaptation: 37 intents (vs default 15)
```

### 8.5 Qasper (10 Papers)
```bash
LIMIT=10
AUTO_ADAPT_INTENTS=1
FORCE_INTENTS=1
FORCE_IDC_SEGMENTS=1
FORCE_SPANS=1
# Auto-adaptation enabled for varying document lengths
```

---

## 9. Computational Environment

### 9.1 Software Versions
- **Python**: 3.12.9
- **NumPy**: Latest stable
- **Google Generative AI**: Latest API version
- **BM25**: Rank-BM25 implementation

### 9.2 Hardware (not critical for reproducibility)
- **Platform**: macOS 14.6.0 (Darwin 24.6.0)
- **CPU**: Apple Silicon (details vary)
- **API-based**: Computation delegated to Google Cloud

---

## 10. Thesis-Ready Configuration Summary

### For Methodology Section

> "IDC uses the Gemini 2.5 Flash model for intent generation and Gemini Embedding-001 (1,536 dimensions) for vector representations. The algorithm employs four hyperparameters: λ (length penalty), β (boundary penalty), L_max (maximum chunk length), and α (coherence weight). For SQuAD (n=293 spans), grid search over 768 configurations (8×8×3×4) found optimal values λ=0.0005, β=1.2, L_max=20, α=0.2, improving R@1 from 0.604 (default) to 0.689 (tuned). For long documents (arXiv, Qasper), auto-adaptive intent generation scales the number of intents: num_questions = 15 × (doc_length/200)^0.7, generating 37 intents for a 495-sentence paper versus 15 for shorter documents."

> "Retrieval employs hybrid search combining dense semantic similarity (60% weight) with BM25 lexical matching (40% weight), followed by lexical reranking. All chunking methods (IDC and baselines) use identical retrieval and reranking configurations to ensure fair comparison. Chunks are indexed using four views: raw text, predicted intent, LLM-generated summary, and extracted keywords."

### For Reproducibility

All experiments can be reproduced using the commands in Section 8 with the specified parameters. The complete codebase and configuration files are available in this repository.

---

**Status**: ✅ Complete experimental configuration  
**Date**: October 2025  
**Purpose**: Thesis documentation and reproducibility
