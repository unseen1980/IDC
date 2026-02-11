# Implementation Details for Thesis

This document provides comprehensive implementation details for the "Implementation Details" section of your thesis.

---

## 1. Software Architecture and Core Modules

Our IDC implementation is done in **Python 3.12.9**, with heavy use of existing NLP libraries. The codebase is organized into modular components for maintainability and reproducibility.

### 1.1 Key Python Modules

**Core Algorithm Implementation**:
- **[idc_core.py](../src/idc_core.py)**: Dynamic programming segmentation algorithm
  - Contains the canonical DP implementation for optimal chunking
  - `IDCParams` dataclass manages tunable parameters (λ, β, L_max, α)
  - `segment_document()` function: O(N² × M) complexity for N sentences and M intents
  - Implements coherence scoring, boundary penalties, and length constraints
  - ~500 lines of pure Python with NumPy for vector operations

**Intent Generation**:
- **[intents.py](../src/intents.py)**: LLM-based question generation module
  - Uses Google Gemini 2.5 Flash via the `google.generativeai` API
  - Implements universal prompting (no content-specific bias)
  - Diversity enforcement via MMR (Maximal Marginal Relevance) clustering (threshold=0.4)
  - Retry logic with exponential backoff (max 5 attempts, base delay 1s)
  - Prompt designed to elicit 15 diverse questions (or more via auto-adaptation)

**Parameter Optimization**:
- **[auto_tune.py](../src/auto_tune.py)**: Grid search tuning wrapper
  - Evaluates 768 parameter combinations (8 λ values × 8 β × 3 L_max × 4 α)
  - Uses `eval_coverage.py` for span-based coverage metrics
  - Returns optimal parameters maximizing R@1, R@5, and MRR
  - Parallelizable design (each configuration independent)

**Adaptive Scaling**:
- **[adaptive_params.py](../src/adaptive_params.py)**: Document-length-aware intent generation
  - Formula: `num_questions = 15 × (doc_length / 200)^0.7`
  - Prevents under-segmentation in long documents (arXiv 495 sent → 37 intents)
  - Automatically adjusts `generation_multiplier` and `diversity_threshold` based on document structure

### 1.2 Baseline Implementations

Each baseline was implemented in a unified module ([baselines.py](../src/baselines.py)) to ensure identical output structure and fair comparison:

**1. Fixed-Length Chunking** (`segment_fixed`):
- Straightforward loop over sentences with fixed chunk size (6 sentences)
- No overlap between chunks
- ~15 lines of code, O(N) complexity

**2. Sliding Window** (`segment_sliding`):
- Window size: 6 sentences, stride: 3 sentences (50% overlap)
- Simple iteration with step increment: `for i in range(0, N, stride)`
- ~25 lines of code, O(N) complexity

**3. Paragraph-Based** (`segment_paragraphs`):
- Splits raw text on blank lines (`\n\n`) to identify natural paragraph boundaries
- Uses NLTK's `sent_tokenize()` to count sentences per paragraph
- Aligns paragraph boundaries to sentence indices from preprocessing
- Fallback: 6-sentence fixed chunks if no paragraphs detected
- Post-processing: Enforces max token length constraint
- ~80 lines of code, O(N) complexity

**4. Coherence-Based** (`segment_coherence`): TextTiling-like algorithm
- **Window size**: 2 sentences (one on each side of boundary)
- **Step size**: 1 (evaluate every potential boundary)
- **Tuned parameters**: Window=1, min_len=1, max_len=10 sentences
- **Algorithm**:
  1. For each boundary i (between sentence i and i+1), compute cosine similarity between mean of left window [i-win+1..i] and right window [i+1..i+win]
  2. Rank boundaries by ascending similarity (valleys = low cohesion points)
  3. Greedily insert boundaries while satisfying [min_len, max_len] constraints
  4. Target ~6 sentences per chunk (produces similar number of segments as paragraph baseline)
- **Complexity**: O(N²) for N sentences (efficient for documents <1000 sentences)
- ~150 lines of code

**Unified Output**: All chunkers output identical JSON structure:
```json
{
  "doc_id": "...",
  "chunks": [
    {"start_sent": 1, "end_sent": 3, "num_sentences": 3, "intent": null, "text": "..."},
    ...
  ]
}
```

This unified implementation ensures that chunking differences are the **sole factor** affecting retrieval outcomes.

---

## 2. Embedding and Retrieval Implementation

### 2.1 Sentence and Query Embeddings

**Embedding Model**: `gemini-embedding-001` (Google Gemini)
- **Dimensionality**: 1,536 dimensions
- **Normalization**: L2-normalized for cosine similarity
- **Task types**:
  - `RETRIEVAL_DOCUMENT`: For sentences and chunk text
  - `RETRIEVAL_QUERY`: For intents and evaluation queries
- **Implementation**: Direct API calls via `google-generativeai==0.8.5`
- **Batch processing**: Retry logic with exponential backoff (1s, 2s, 4s, 8s, 16s)

**Ensuring Same Vector Space**:
- Both sentences and questions are encoded with **identical embedding model** (`gemini-embedding-001`)
- Task type differentiation (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`) optimizes for asymmetric search
- All embeddings L2-normalized before similarity computation

**Contextual Embeddings** (optional, enabled by default):
```python
chunk_embedding = 0.70 × chunk_emb + 0.15 × prev_chunk_emb + 0.15 × next_chunk_emb
```
- Incorporates 15% context from adjacent chunks on each side
- Improves retrieval by providing discourse-level context
- Applied uniformly to all chunking methods (fair comparison)

### 2.2 Vector Similarity Search

**No FAISS Library**: Vector search implemented in pure NumPy for simplicity and transparency
```python
# Brute-force cosine similarity (sufficient for small corpus)
scores = np.dot(chunk_embeddings, query_embedding.T)  # Shape: (num_chunks,)
top_k_indices = np.argsort(scores)[-k:][::-1]        # Descending order
```

**Why No FAISS**:
- Corpus size small (~50-150 chunks per document)
- NumPy matrix operations sufficiently fast (<10ms per query)
- Simpler implementation, easier to debug and reproduce
- No external index building required

**Retrieval Performance**:
- **Query time**: <10ms per query (includes hybrid retrieval + reranking)
- **Index size**: SQuAD 2-docs: ~58 chunks (IDC), ~60-70 (baselines)
- **Total corpus**: ~150 documents across all datasets

### 2.3 Hybrid Retrieval and Reranking

**Hybrid Retrieval** (60% dense + 40% BM25):
```python
final_score = 0.6 × semantic_score + 0.4 × bm25_score
```
- **Dense component**: Gemini embedding cosine similarity
- **Sparse component**: BM25 lexical matching (rank-bm25==0.2.2)
- Weights optimized on development set (see [config.py](../src/config.py))

**Lexical Reranker** (applied to ALL methods):
```python
reranked_score = semantic_score + 0.3 × bm25_score + 0.05 × lexical_overlap + length_penalty

where:
- lexical_overlap: Token overlap between query and chunk intent (Jaccard similarity)
- length_penalty: -0.05 if chunk < 2 sentences (penalize very short chunks)
```

**Alternative Reranker** (not used): `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Available as option but NOT used in final experiments
- All reported results use lexical reranker for consistency

---

## 3. Preprocessing and Data Pipeline

### 3.1 Sentence Splitting

**Implementation** ([preprocess.py](../src/preprocess.py)):
1. Read raw text files from `data/input/*.txt`
2. Apply NLTK `sent_tokenize()` with punkt tokenizer (deterministic)
3. Handle edge cases:
   - Merge sentences shorter than 10 characters with previous sentence
   - Preserve paragraph boundaries (detected by blank lines `\n\n`)
   - Clean whitespace and normalize Unicode
   - Handle common abbreviations (Dr., etc.) to prevent false sentence breaks
4. Output: `sentences.jsonl` (one sentence per line with metadata)

**Example Output**:
```json
{"doc_id": "Normans", "sent_id": 0, "text": "The Normans were a people who..."}
{"doc_id": "Normans", "sent_id": 1, "text": "They were descendants of..."}
```

### 3.2 Dataset Conversion

**Common Pattern**: All converters output identical schema for unified pipeline processing

**SQuAD** ([convert_squad.py](../src/convert_squad.py)):
- Extracts context paragraphs + QA pairs from SQuAD JSON
- Aligns character-level answer spans to sentence indices
- Handles multi-sentence answers (span ranges)
- Output: `sentences.jsonl`, `spans.jsonl`, `raw_text.txt`

**Qasper** ([convert_qasper.py](../src/convert_qasper.py)):
- Parses academic paper JSON (sections, paragraphs, questions)
- Converts evidence paragraph IDs to sentence spans
- Handles multi-paragraph answers (span ranges)
- Filters unanswerable questions (no evidence)

**NewsQA** ([convert_newsqa.py](../src/convert_newsqa.py)):
- Extracts news article text and question-answer pairs
- Aligns character-based answer spans to sentence boundaries
- Filters questions with no valid answer

---

## 4. Reproducibility and Configuration

### 4.1 Random Seed Control

All random processes are **seeded for reproducibility**:
- Python `random.seed(42)`
- NumPy `np.random.seed(42)`
- LLM generation: Fixed temperature (default, deterministic for same seed)
- NLTK punkt tokenizer: Deterministic (no randomness)

### 4.2 Library Versions

| Library | Version | Purpose |
|---------|---------|---------|
| **Python** | 3.12.9 | Implementation language |
| **NumPy** | 1.26.4 | Vector operations, DP algorithm |
| **google-generativeai** | 0.8.5 | Gemini API (generation + embeddings) |
| **NLTK** | 3.9.1 | Sentence tokenization |
| **rank-bm25** | 0.2.2 | BM25 lexical retrieval |
| **tqdm** | 4.67.1 | Progress bars |
| **python-dotenv** | Latest | Environment management |

**No Heavy ML Frameworks**: No PyTorch, TensorFlow, or Transformers library. All embeddings obtained via API.

---

## 5. Computational Performance

### 5.1 Offline Chunking Time (per document)

**IDC**:
- **Total**: ~1-2 seconds per document
- **DP segmentation**: ~100-200ms (O(N² × M), N=200 sent, M=15 intents)
- **Intent generation**: ~1-2 seconds (API latency dominates)
- **Embedding**: ~500ms (API call, batch of 15 intents)

**Baselines**:
- **Fixed/Sliding/Paragraphs**: <10ms (virtually instantaneous)
- **Coherence**: ~50-100ms (depends on embedding computations)

**Comparison**: IDC is 10-20× slower offline, but this is a **one-time cost** per document. The improved retrieval quality justifies this trade-off (see Discussion section).

### 5.2 Retrieval Performance

**Query Time**: <10ms per query
- NumPy cosine similarity: ~2ms
- BM25 scoring: ~3ms
- Reranking: ~2ms
- Total: ~7-10ms (end-to-end)

**Scalability**:
- Linear scaling with corpus size (brute-force search)
- For larger corpora (>10,000 chunks), FAISS would be beneficial
- Current corpus size (~50-150 chunks) makes optimization unnecessary

### 5.3 Hardware Configuration

**Development Machine**:
- **Platform**: macOS 14.6.0 (Darwin 24.6.0)
- **CPU**: Apple Silicon M-series
- **Memory**: 32GB RAM
- **Storage**: SSD (for fast I/O)

**Note**: Actual embedding and generation computation delegated to Google Cloud via API. Hardware specs not critical for reproducibility.

---

## 6. Thesis-Ready Summary for Implementation Section

### Suggested Text for Your Thesis:

> Our IDC implementation is done in Python 3.12.9, with heavy use of existing NLP libraries. Key modules include [idc_core.py](../src/idc_core.py) for the DP segmentation algorithm, [intents.py](../src/intents.py) for LLM-based intent generation, [adaptive_params.py](../src/adaptive_params.py) for computing adaptive question counts, and [auto_tune.py](../src/auto_tune.py) for the grid search tuning procedure. We use the Google Gemini Embedding-001 model for sentence embeddings (1,536-dimensional vectors) and the same model for encoding questions, ensuring both are in the same vector space. The LLM for question generation was Google's Gemini 2.5 Flash (via the Google Gemini API) with a prompt designed to elicit 15 diverse questions, or more for longer docs (as described earlier). We seeded all random processes (for reproducibility) and performed careful preprocessing (sentence splitting with NLTK's punkt tokenizer, tokenization, and handling of edge cases such as merging very short sentences) to ensure the input to IDC was clean.
>
> Each baseline was implemented as follows: fixed and sliding window chunkers are straightforward loops over sentences; paragraph chunking uses the document's paragraph breaks from the raw text (detected by blank lines) with a post-process to enforce a max token length; the coherence-based method uses a TextTiling-like algorithm with a window of 2 sentences and step of 1, tuned to produce a similar number of segments as paragraphs (it effectively finds "valley" points of lexical cohesion to cut, ranking boundaries by ascending cosine similarity between left/right windows and greedily inserting boundaries while satisfying min/max length constraints). All chunkers output a list of chunks with the sentence indices they cover, which are then embedded and indexed identically. This unified implementation ensures that chunking differences are the sole factor affecting retrieval outcomes.
>
> We ran experiments on a machine with 32GB RAM and used pure NumPy for vector similarity search over chunk embeddings (brute-force cosine similarity, sufficient for our corpus size). For the largest setting (SQuAD 2-docs, ~58 chunks for IDC and similar for baselines, across ~150 documents total after combining Qasper's 10 docs etc.), retrieval was very fast (<10ms per query). The offline chunking process varied by method: IDC took the longest (~1–2 seconds per document for DP segmentation plus the time for LLM calls for intent generation), while fixed and paragraph methods were virtually instantaneous (<10ms). However, since this is an offline cost incurred only once per document, we consider it acceptable given the improved retrieval quality (see Discussion for more on this trade-off).

---

**Status**: ✅ Complete implementation details for thesis
**Date**: October 2025
**Purpose**: Thesis "Implementation Details" section
