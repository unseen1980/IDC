# What Are Spans in IDC?

## TL;DR

**Spans are evaluation units that represent where answers should be found in a document. Each span links a question to specific sentences that contain the answer. They're used to measure retrieval quality (R@1, R@5, MRR).**

## Simple Definition

A **span** is a gold-standard reference that says:
> "For this question, the answer can be found in sentences X through Y"

Example:
```json
{
  "query_id": 1,
  "doc_id": "newsqa_corpus",
  "start_sent": 176,
  "end_sent": 176,
  "answerable": true
}
```

Translation: "Question #1's answer is in sentence 176 of newsqa_corpus"

## Why Spans Matter

Spans are the ground truth for evaluating chunking methods:

1. **Generate chunks** using different methods (IDC, Fixed, Sliding, etc.)
2. **For each question**, find which chunk contains the span
3. **Measure retrieval**: Did we retrieve the correct chunk?

### Retrieval Metrics

- **R@1 (Recall at 1)**: Did the TOP-1 retrieved chunk contain the answer span?
- **R@5 (Recall at 5)**: Did any of the TOP-5 retrieved chunks contain the answer span?
- **MRR (Mean Reciprocal Rank)**: Average of 1/rank where rank is position of first correct chunk

## Real Examples

### Example 1: SQuAD Span

```json
{
  "query_id": 1,
  "doc_id": "Normans",
  "start_sent": 1,
  "end_sent": 1,
  "answerable": true
}
```

**Meaning:**
- Question: "In what country is Normandy located?"
- Answer location: Sentence 1 of "Normans" document
- Sentence 1: "The Normans were the people who gave their name to Normandy, a region in France."

**Evaluation:**
- If IDC chunk contains sentence 1 → ✅ Correct retrieval
- If IDC chunk doesn't contain sentence 1 → ❌ Failed retrieval

---

### Example 2: NewsQA Span (Multi-sentence)

```json
{
  "query_id": 2,
  "doc_id": "newsqa_corpus",
  "start_sent": 327,
  "end_sent": 329,
  "answerable": true
}
```

**Meaning:**
- Question: "Who was involved in the incident?"
- Answer spans sentences 327-329 (3 sentences)
- A chunk must contain ALL sentences 327-329 to be correct

**Evaluation:**
- Chunk covers [320-340] → ✅ Contains 327-329
- Chunk covers [325-328] → ❌ Missing sentence 329
- Chunk covers [330-350] → ❌ Missing sentences 327-329

---

## How Spans Are Created

### From Question-Answer Datasets

**SQuAD, NewsQA, Qasper:**
1. Dataset has questions + answers
2. Find which sentence(s) contain the answer
3. Create span pointing to those sentences

```python
def create_span(question, answer, document):
    # Find sentences containing the answer
    answer_sentences = find_sentences_with_text(document, answer)
    
    span = {
        "query_id": unique_id,
        "doc_id": document_name,
        "start_sent": answer_sentences[0],  # First sentence
        "end_sent": answer_sentences[-1],   # Last sentence
        "answerable": True
    }
    return span
```

### From LLM-Generated Questions

**arXiv, Fiori:**
1. LLM generates questions about document
2. LLM indicates which sentences are relevant
3. Create spans from those sentence ranges

---

## Span Statistics Across Datasets

| Dataset | Total Spans | Avg Span Length | Type |
|---------|-------------|-----------------|------|
| **SQuAD** | 293 | 1.2 sentences | Question-Answer |
| **NewsQA** | 15 | 2.3 sentences | Question-Answer |
| **arXiv** | 15 | 1.8 sentences | LLM-generated |
| **Fiori** | 15 | 1.5 sentences | LLM-generated |

**Key:** SQuAD has 293 spans → Enables reliable auto-tuning!

---

## How Evaluation Works

### Step 1: Generate Chunks

Different methods create different chunks:

**IDC (Intent-Driven):**
```
Chunk 1: Sentences [1-8]   (Norman origins)
Chunk 2: Sentences [9-15]  (Norman conquest)
Chunk 3: Sentences [16-25] (Norman architecture)
```

**Fixed-length:**
```
Chunk 1: Sentences [1-6]
Chunk 2: Sentences [7-12]
Chunk 3: Sentences [13-18]
```

### Step 2: Match Spans to Chunks

**Span 1: Sentence 1**
- IDC Chunk 1 [1-8] → ✅ Contains sentence 1
- Fixed Chunk 1 [1-6] → ✅ Contains sentence 1

**Span 2: Sentence 10**
- IDC Chunk 2 [9-15] → ✅ Contains sentence 10
- Fixed Chunk 2 [7-12] → ✅ Contains sentence 10

**Span 3: Sentence 20**
- IDC Chunk 3 [16-25] → ✅ Contains sentence 20
- Fixed Chunk 3 [13-18] → ❌ Doesn't contain sentence 20

### Step 3: Calculate Metrics

**Coverage:**
- IDC: 3/3 spans covered = 100%
- Fixed: 2/3 spans covered = 67%

**R@1 (after retrieval):**
- For each span, retrieve top chunk using query
- IDC: Retrieved correct chunk for 0.689 of spans
- Fixed: Retrieved correct chunk for 0.655 of spans

---

## Why Sample Size Matters

### Small Sample (n=15 spans)

```
NewsQA Results:
IDC:   14/15 correct = R@1=0.933
Fixed: 13/15 correct = R@1=0.867

Difference: ONE span changes R@1 by 6.7%!
```

**Problem:** High variance, unreliable for optimization

### Large Sample (n=293 spans)

```
SQuAD Results:
IDC:   202/293 correct = R@1=0.689
Fixed: 192/293 correct = R@1=0.655

Difference: 10 spans changes R@1 by 3.4%
```

**Benefit:** Low variance, reliable for auto-tuning

---

## Span Types

### 1. Point Spans (Single Sentence)

```json
{
  "start_sent": 5,
  "end_sent": 5
}
```

**Common in:** SQuAD (factoid questions)

### 2. Range Spans (Multiple Sentences)

```json
{
  "start_sent": 10,
  "end_sent": 13
}
```

**Common in:** NewsQA (narrative answers)

### 3. Unanswerable Spans

```json
{
  "start_sent": -1,
  "end_sent": -1,
  "answerable": false
}
```

**Used for:** Questions without answers in document

---

## Visualization

### Document with Spans

```
Document: "Normans" (25 sentences)

Span 1: [1-1]   ← "Where is Normandy?"
Span 2: [3-3]   ← "Who were the Vikings?"
Span 3: [10-10] ← "When did they arrive?"
Span 4: [15-17] ← "What did they build?"
Span 5: [20-20] ← "What language spoke?"

Total: 5 evaluation spans
```

### Chunking Comparison

```
IDC Chunks:
├─ Chunk A: [1-8]   → Contains spans: 1, 2 ✅✅
├─ Chunk B: [9-15]  → Contains spans: 3, 4 ✅✅ (partial)
└─ Chunk C: [16-25] → Contains spans: 4, 5 ✅✅ (partial)

Fixed Chunks:
├─ Chunk A: [1-6]   → Contains spans: 1, 2 ✅✅
├─ Chunk B: [7-12]  → Contains spans: 3 ✅
├─ Chunk C: [13-18] → Contains spans: 4 ✅ (partial)
└─ Chunk D: [19-25] → Contains spans: 4, 5 ✅✅ (partial)
```

**Evaluation:** Compare which method retrieves correct chunks more often

---

## Common Questions

### Q: Why n=15 for most datasets?

**A:** LLM-generated questions are expensive
- Each question costs API tokens
- 15-20 spans sufficient for small documents
- SQuAD has pre-existing questions (n=293)

### Q: Can spans overlap?

**A:** Yes! Multiple questions can point to same sentences
- SQuAD has overlapping spans frequently
- Reflects real information-seeking behavior

### Q: What if chunk partially covers span?

**A:** Coverage calculation:
```python
def span_coverage(chunk, span):
    overlap = set(range(chunk.start, chunk.end + 1)) & \
              set(range(span.start_sent, span.end_sent + 1))
    
    coverage = len(overlap) / (span.end_sent - span.start_sent + 1)
    
    # Typically require 100% coverage for "correct"
    return coverage >= 1.0
```

### Q: How are spans different from intents?

**A:** Completely different concepts!

| Aspect | Spans | Intents |
|--------|-------|---------|
| **Purpose** | Evaluation ground truth | Chunking guidance |
| **Created by** | Humans/LLM questions | LLM summary generation |
| **Count** | n=15-293 (questions) | 15-37 (topics) |
| **Usage** | Measure retrieval quality | Guide segmentation |
| **Example** | "Sentence 5 has the answer" | "Topic: Norman conquest" |

---

## Summary

**What spans are:**
- Evaluation units linking questions to answer locations
- Ground truth for measuring retrieval quality
- Represented as (start_sent, end_sent) ranges

**Why spans matter:**
- Enable R@1, R@5, MRR metrics
- Measure chunking effectiveness
- Compare IDC vs baselines objectively

**Key insight:**
- SQuAD: n=293 spans → Reliable auto-tuning
- Others: n=15 spans → Too small for optimization
- Sample size determines statistical power

---

**Version:** 1.0  
**Date:** 2025-10-17  
**Status:** Comprehensive explanation of evaluation spans
