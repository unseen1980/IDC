#!/usr/bin/env python3
"""IDC segmentation module and CLI.

The module exposes programmatic helpers for running the Intent-Driven Dynamic
Chunking (IDC) algorithm, including both the canonical dynamic-programming
implementation and the multi-pass splitter heuristics. The implementation keeps
the original objective while adding structure so that other components
(auto-tuning, batch scripts, UI) can reuse the logic without shelling out to
stand-alone scripts.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_jsonl(path: str | Path) -> List[Dict]:
    """Read a JSONL file into a list of dictionaries."""
    path = Path(path)
    items: List[Dict] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    """Write dictionaries as JSON lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Data grouping helpers
# ---------------------------------------------------------------------------

def group_sentences_by_doc(sent_rows: List[Dict]) -> Dict[str, List[int]]:
    """Return mapping doc_id -> list of global row indices in sentence order."""
    by_doc: Dict[str, List[Tuple[int, int]]] = {}
    for idx, row in enumerate(sent_rows):
        by_doc.setdefault(row["doc_id"], []).append((int(row["sent_id"]), idx))

    ordered: Dict[str, List[int]] = {}
    for doc_id, pairs in by_doc.items():
        pairs.sort(key=lambda item: item[0])
        ordered[doc_id] = [idx for _, idx in pairs]
    return ordered


def group_intents_by_doc(flat_intents: List[Dict]) -> Dict[str, List[int]]:
    """Return mapping doc_id -> list of global intent indices in query order."""
    by_doc: Dict[str, List[Tuple[int, int]]] = {}
    for idx, row in enumerate(flat_intents):
        by_doc.setdefault(row["doc_id"], []).append((int(row["query_id"]), idx))

    ordered: Dict[str, List[int]] = {}
    for doc_id, pairs in by_doc.items():
        pairs.sort(key=lambda item: item[0])
        ordered[doc_id] = [idx for _, idx in pairs]
    return ordered


# ---------------------------------------------------------------------------
# Parameter/data containers
# ---------------------------------------------------------------------------

@dataclass
class IDCParams:
    """Tunable parameters for IDC segmentation."""

    lam: float
    max_len: int
    min_len: int = 2
    boundary_penalty: float = 0.25
    coherence_weight: float = 0.10
    merge_adjacent: bool = False
    structural_priors: bool = True
    para_discount: float = 0.5
    diversity_weight: float = 0.0
    length_penalty_mode: str = "linear"
    min_chunk_sent: int = 2
    max_chunk_sent: int = 10
    respect_paragraphs: bool = True
    postprocess: bool = True

    def __post_init__(self) -> None:
        """Validate and harmonise parameter relationships."""
        mode = self.length_penalty_mode.lower()
        if mode not in {"linear", "quadratic"}:
            raise ValueError(f"Unsupported length_penalty_mode: {self.length_penalty_mode!r}")
        self.length_penalty_mode = mode
        self.min_len = max(self.min_len, self.min_chunk_sent)
        self.max_len = max(self.max_len, self.min_len)
        if self.max_chunk_sent < self.max_len:
            self.max_chunk_sent = self.max_len


@dataclass
class SegmentDocumentResult:
    """Result for a single document segmentation."""

    doc_id: str
    chunks: List[Dict]
    merged_from: int
    merged_to: int


# ---------------------------------------------------------------------------
# Multi-pass helpers
# ---------------------------------------------------------------------------

def initial_idc_pass(
    S: np.ndarray,
    Q: np.ndarray,
    max_len_initial: int = 15,
    min_len: int = 2,
    lam: float = 0.0005,
    boundary_pen: float = 0.1,
    coherence_weight: float = 0.03,
) -> List[Tuple[int, int, int, float]]:
    """Stage 1 IDC pass with generous length limits to find strong boundaries."""
    if S.size == 0:
        return []

    if Q is None or len(Q) == 0:
        chunks: List[Tuple[int, int, int, float]] = []
        i = 0
        N = S.shape[0]
        while i < N:
            j = min(N - 1, i + max_len_initial - 1)
            chunks.append((i, j, -1, 0.0))
            i = j + 1
        return chunks

    N, D = S.shape
    Q_norms = np.linalg.norm(Q, axis=1, keepdims=True)
    Q_norms = np.maximum(Q_norms, 1e-8)
    Qn = Q / Q_norms

    S_norms = np.linalg.norm(S, axis=1, keepdims=True)
    S_norms = np.maximum(S_norms, 1e-8)
    S_normalized = S / S_norms
    similarity_matrix = S_normalized @ S_normalized.T

    prefix = np.zeros((N + 1, D), dtype=np.float32)
    for i in range(1, N + 1):
        prefix[i] = prefix[i - 1] + S[i - 1]

    def chunk_mean(l: int, r: int) -> np.ndarray:
        vec = prefix[r + 1] - prefix[l]
        return vec / max(r - l + 1, 1)

    def compute_coherence(l: int, r: int) -> float:
        if r <= l:
            return 1.0
        chunk_len = r - l + 1
        if chunk_len < 2:
            return 1.0
        chunk_sim = similarity_matrix[l : r + 1, l : r + 1]
        mask = np.triu(np.ones((chunk_len, chunk_len), dtype=bool), k=1)
        if np.sum(mask) == 0:
            return 1.0
        return float(np.mean(chunk_sim[mask]))

    def compute_intent_score(l: int, r: int) -> Tuple[float, int]:
        v = chunk_mean(l, r)
        vn = v / (np.linalg.norm(v) + 1e-8)
        sims = Qn @ vn
        best_idx = int(np.argmax(sims))
        max_sim = float(sims[best_idx])
        if max_sim > 0.5:
            max_sim *= 1.1
        return max_sim, best_idx

    dp = np.full(N + 1, -1e9, dtype=np.float32)
    prv = np.full(N + 1, -1, dtype=np.int32)
    best_intent = np.full(N + 1, -1, dtype=np.int32)
    dp[0] = 0.0

    for i in range(1, N + 1):
        best_score = -1e9
        best_j = -1
        best_k = -1
        j_min = max(0, i - max_len_initial)
        for j in range(i - 1, j_min - 1, -1):
            length = i - j
            if length < min_len:
                continue
            intent_score, intent_idx = compute_intent_score(j, i - 1)
            coherence = compute_coherence(j, i - 1)
            length_penalty = lam * length
            score = (
                dp[j]
                + intent_score
                + coherence_weight * coherence
                - length_penalty
                - boundary_pen
            )
            if score > best_score:
                best_score = score
                best_j = j
                best_k = intent_idx
        dp[i] = best_score
        prv[i] = best_j
        best_intent[i] = best_k

    spans: List[Tuple[int, int, int, float]] = []
    i = N
    while i > 0:
        j = int(prv[i])
        k = int(best_intent[i])
        sim, _ = compute_intent_score(j, i - 1)
        spans.append((j, i - 1, k, sim))
        i = j
    spans.reverse()
    return spans


def find_split_points(
    S: np.ndarray,
    sentence_texts: List[str],
    chunk_start: int,
    target_splits: int = 2,
    min_split_size: int = 2,
    sentence_rows: Optional[List[Dict]] = None,
    respect_paragraphs: bool = True,
) -> List[int]:
    """Return local split indices weighted by coherence and structure."""
    chunk_len = len(sentence_texts)
    if chunk_len < min_split_size * target_splits or chunk_len < 2:
        return []

    S_norms = np.linalg.norm(S, axis=1, keepdims=True)
    S_norms = np.maximum(S_norms, 1e-8)
    S_normalized = S / S_norms

    coherence_scores = []
    for i in range(1, chunk_len):
        coherence_scores.append((i, float(S_normalized[i - 1] @ S_normalized[i])))

    boundary_scores: List[Tuple[int, float]] = []
    for i in range(1, chunk_len):
        boundary_score = 0.0
        coherence = coherence_scores[i - 1][1]
        boundary_score += (1.0 - coherence) * 0.5

        curr_text = sentence_texts[i - 1] if i - 1 < len(sentence_texts) else ""
        next_text = sentence_texts[i] if i < len(sentence_texts) else ""
        lowered_next = next_text.lower().strip()
        discourse_markers = (
            "however",
            "moreover",
            "furthermore",
            "therefore",
            "meanwhile",
            "first",
            "second",
            "third",
            "finally",
            "next",
            "then",
            "in conclusion",
            "to summarize",
        )
        if any(lowered_next.startswith(marker) for marker in discourse_markers):
            boundary_score += 0.3

        if curr_text.rstrip().endswith(('.', '!', '?')):
            boundary_score += 0.1
        if curr_text.rstrip().endswith(":"):
            boundary_score -= 0.1
        if lowered_next.startswith(("- ", "* ")) or lowered_next[:2].isdigit():
            boundary_score += 0.1

        if sentence_rows and chunk_start + i < len(sentence_rows):
            prev_row = sentence_rows[chunk_start + i - 1]
            next_row = sentence_rows[chunk_start + i]
            para_keys = ("paragraph_id", "para_id", "paragraph")
            prev_para = next_para = None
            for key in para_keys:
                if prev_para is None:
                    prev_para = prev_row.get(key)
                if next_para is None:
                    next_para = next_row.get(key)
            if prev_para is not None and next_para is not None:
                if prev_para != next_para:
                    boundary_score += 0.4
                elif respect_paragraphs:
                    boundary_score -= 0.2
            prev_section = prev_row.get("section_path")
            next_section = next_row.get("section_path")
            if prev_section and next_section and prev_section != next_section:
                boundary_score += 0.3
            block_prev = str(prev_row.get("block_type", "")).lower()
            block_next = str(next_row.get("block_type", "")).lower()
            if respect_paragraphs and (block_prev in {"code", "table"} or block_next in {"code", "table"}):
                boundary_score -= 0.4

        boundary_scores.append((i, boundary_score))

    boundary_scores.sort(key=lambda x: x[1], reverse=True)
    selected_splits: List[int] = []
    for pos, _ in boundary_scores:
        if len(selected_splits) >= target_splits - 1:
            break
        if pos < min_split_size or pos > chunk_len - min_split_size:
            continue
        if any(abs(pos - existing) < min_split_size for existing in selected_splits):
            continue
        selected_splits.append(pos)
    return sorted(selected_splits)


def split_long_chunk(
    chunk: Tuple[int, int, int, float],
    S: np.ndarray,
    Q: np.ndarray,
    sentence_texts: List[str],
    target_length: int = 7,
    min_length: int = 2,
    sentence_rows: Optional[List[Dict]] = None,
    respect_paragraphs: bool = True,
) -> List[Tuple[int, int, int, float]]:
    """Split an oversized chunk into intent-aligned sub-chunks."""
    start, end, original_intent_idx, original_sim = chunk
    chunk_length = end - start + 1
    upper_cap = max(min_length + 1, 8)
    effective_target = min(target_length, upper_cap)

    if chunk_length <= effective_target:
        return [chunk]

    desired_chunks = max(2, int(math.ceil(chunk_length / float(effective_target))))
    num_splits = desired_chunks - 1

    chunk_sentences = S[start : end + 1]
    chunk_texts = sentence_texts[start : end + 1]

    split_points = find_split_points(
        chunk_sentences,
        chunk_texts,
        start,
        target_splits=desired_chunks,
        min_split_size=min_length,
        sentence_rows=sentence_rows,
        respect_paragraphs=respect_paragraphs,
    )

    if len(split_points) < num_splits:
        step = chunk_length / float(desired_chunks)
        for k in range(1, desired_chunks):
            candidate = int(round(k * step))
            candidate = max(min_length, min(chunk_length - min_length, candidate))
            if candidate not in split_points:
                split_points.append(candidate)
            if len(split_points) == num_splits:
                break

    split_points = sorted({p for p in split_points if 0 < p < chunk_length})
    if not split_points:
        return [chunk]

    if Q is not None and Q.size > 0:
        Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)
    else:
        Qn = None

    sub_chunks: List[Tuple[int, int, int, float]] = []
    prev_pos = 0

    for split_pos in split_points + [chunk_length]:
        if split_pos <= prev_pos:
            continue
        sub_start = start + prev_pos
        sub_end = start + split_pos - 1
        sub_length = sub_end - sub_start + 1

        if sub_length < min_length:
            if sub_chunks:
                last_start, last_end, last_intent, last_sim = sub_chunks[-1]
                sub_chunks[-1] = (last_start, sub_end, last_intent, last_sim)
            prev_pos = split_pos
            continue

        if Qn is not None:
            sub_embeddings = S[sub_start : sub_end + 1]
            sub_mean = np.mean(sub_embeddings, axis=0)
            vn = sub_mean / (np.linalg.norm(sub_mean) + 1e-8)
            sims = Qn @ vn
            best_intent_idx = int(np.argmax(sims))
            best_sim = float(sims[best_intent_idx])
            if 0 <= original_intent_idx < len(sims):
                original_sim_sub = float(sims[original_intent_idx])
                if original_sim_sub >= best_sim - 0.1:
                    best_intent_idx = original_intent_idx
                    best_sim = original_sim_sub
        else:
            best_intent_idx = -1
            best_sim = 0.0

        sub_chunks.append((sub_start, sub_end, best_intent_idx, best_sim))
        prev_pos = split_pos

    return sub_chunks


def multi_pass_segmentation(
    S: np.ndarray,
    Q: np.ndarray,
    sentence_texts: List[str],
    max_len_final: int = 8,
    min_len: int = 2,
    initial_max_len: int = 15,
    lam: float = 0.0005,
    boundary_pen: float = 0.1,
    coherence_weight: float = 0.03,
) -> List[Tuple[int, int, int, float]]:
    """Run two-stage multi-pass IDC segmentation."""
    print(f"Stage 1: Initial IDC pass (max_len={initial_max_len})")
    initial_chunks = initial_idc_pass(
        S,
        Q,
        max_len_initial=initial_max_len,
        min_len=min_len,
        lam=lam,
        boundary_pen=boundary_pen,
        coherence_weight=coherence_weight,
    )
    print(f"  Initial chunks: {len(initial_chunks)}")
    if initial_chunks:
        initial_lengths = [end - start + 1 for start, end, _, _ in initial_chunks]
        print(f"  Average length: {np.mean(initial_lengths):.1f}")
        print(
            f"  Long chunks (>{max_len_final}): "
            f"{sum(1 for l in initial_lengths if l > max_len_final)}"
        )
    else:
        print("  No chunks produced during initial pass")

    print(f"Stage 2: Splitting long chunks (max_len={max_len_final})")
    final_chunks: List[Tuple[int, int, int, float]] = []
    splits_made = 0

    for chunk in initial_chunks:
        start, end, intent_idx, sim = chunk
        chunk_length = end - start + 1
        if chunk_length <= max_len_final:
            final_chunks.append(chunk)
        else:
            sub_chunks = split_long_chunk(
                chunk,
                S,
                Q,
                sentence_texts,
                target_length=max_len_final,
                min_length=min_len,
            )
            final_chunks.extend(sub_chunks)
            splits_made += max(len(sub_chunks) - 1, 0)

    print(f"  Final chunks: {len(final_chunks)}")
    print(f"  Splits made: {splits_made}")
    if final_chunks:
        final_lengths = [end - start + 1 for start, end, _, _ in final_chunks]
        print(f"  Final average length: {np.mean(final_lengths):.1f}")
        print(f"  Max length: {max(final_lengths)}")
    else:
        print("  No final chunks produced")

    return final_chunks


def segment_document_multi_pass(
    sentences_path: str,
    sentence_embs_path: str,
    intents_flat_path: str,
    intent_embs_path: str,
    doc_id: str,
    max_len_final: int = 8,
    min_len: int = 2,
    initial_max_len: int = 15,
    lam: float = 0.0005,
    boundary_pen: float = 0.1,
    coherence_weight: float = 0.03,
) -> List[Dict]:
    """Convenience wrapper that loads data and runs multi-pass IDC for one doc."""
    sent_rows = read_jsonl(sentences_path)
    S_all = np.load(sentence_embs_path)
    intent_rows = read_jsonl(intents_flat_path)
    Q_all = np.load(intent_embs_path)

    doc_sent_indices: List[int] = []
    sentence_texts: List[str] = []
    for i, row in enumerate(sent_rows):
        if row.get("doc_id") == doc_id:
            doc_sent_indices.append(i)
            sentence_texts.append(row.get("text", ""))

    doc_intent_indices: List[int] = []
    for i, row in enumerate(intent_rows):
        if row.get("doc_id") == doc_id:
            doc_intent_indices.append(i)

    if not doc_sent_indices:
        return []

    S = S_all[np.array(doc_sent_indices)]
    Q = (
        Q_all[np.array(doc_intent_indices)]
        if doc_intent_indices
        else np.empty((0, S.shape[1]))
    )

    segments = multi_pass_segmentation(
        S,
        Q,
        sentence_texts,
        max_len_final=max_len_final,
        min_len=min_len,
        initial_max_len=initial_max_len,
        lam=lam,
        boundary_pen=boundary_pen,
        coherence_weight=coherence_weight,
    )

    chunks: List[Dict] = []
    for start_local, end_local, intent_idx, sim in segments:
        start_global = doc_sent_indices[start_local]
        end_global = doc_sent_indices[end_local]

        start_sent = sent_rows[start_global]["sent_id"]
        end_sent = sent_rows[end_global]["sent_id"]
        text = " ".join(
            sent_rows[doc_sent_indices[i]]["text"] for i in range(start_local, end_local + 1)
        )
        intent_text = None
        if doc_intent_indices and 0 <= intent_idx < len(doc_intent_indices):
            global_intent_idx = doc_intent_indices[intent_idx]
            intent_text = intent_rows[global_intent_idx].get("text")

        chunks.append(
            {
                "start_sent": start_sent,
                "end_sent": end_sent,
                "num_sentences": end_local - start_local + 1,
                "intent": intent_text,
                "similarity": round(float(sim), 4),
                "text": text,
            }
        )

    return chunks

def chunk_coherence(S: np.ndarray, l: int, r: int) -> float:
    """Return average pairwise cosine for sentences ``S[l:r]`` inclusive."""
    n = r - l + 1
    if n <= 1:
        return 1.0
    window = S[l : r + 1]
    norms = np.linalg.norm(window, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalised = window / norms
    sim = normalised @ normalised.T
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    denom = mask.sum()
    if denom == 0:
        return 1.0
    return float(sim[mask].mean())


def boundary_cost(
    sentence_rows: Optional[List[Dict]],
    j: int,
    adjacent_cos: Optional[np.ndarray],
    respect_paragraphs: bool = True,
) -> float:
    """Heuristic boundary attractiveness cost (positive=discourage)."""
    if not sentence_rows or j <= 0 or j >= len(sentence_rows):
        return 0.0
    prev_row = sentence_rows[j - 1]
    next_row = sentence_rows[j]
    prev_text = (prev_row.get("text") or "").strip()
    next_text = (next_row.get("text") or "").strip()
    cost = 0.0

    prev_section = prev_row.get("section_path")
    next_section = next_row.get("section_path")
    if prev_section and prev_section != next_section:
        cost -= 0.06

    para_keys = ("paragraph_id", "para_id", "paragraph")
    prev_para = next_para = None
    for key in para_keys:
        if prev_para is None:
            prev_para = prev_row.get(key)
        if next_para is None:
            next_para = next_row.get(key)
    if prev_para is not None and next_para is not None:
        if prev_para != next_para:
            cost -= 0.05
        elif respect_paragraphs:
            cost += 0.04

    weak_starts = ("it", "this", "these", "they", "such", "therefore", "however")
    lowered = next_text.lower()
    for marker in weak_starts:
        if lowered.startswith(marker + " ") or lowered == marker:
            cost += 0.05
            break

    if prev_text.endswith(":"):
        if next_text.startswith(('-', '*')) or next_text[:2].isdigit():
            cost += 0.10  # Strengthened: discourage splitting lists from their headers

    block_prev = str(prev_row.get("block_type", "")).lower()
    block_next = str(next_row.get("block_type", "")).lower()
    if respect_paragraphs and (block_prev in {"code", "table"} or block_next in {"code", "table"}):
        cost += 0.20  # Strengthened: preserve code/table block integrity

    if adjacent_cos is not None and 0 <= j - 1 < adjacent_cos.shape[0]:
        cosine = float(adjacent_cos[j - 1])
        valley = max(0.0, 0.7 - cosine)
        cost -= 0.05 * valley

    if prev_text and prev_text[-1] in ".!?":
        cost -= 0.03  # Strengthened: reward natural sentence boundaries

    return cost




# ---------------------------------------------------------------------------
# Information Density Scoring
# ---------------------------------------------------------------------------

def compute_information_density(
    sentence_texts: List[str],
    sentences_data: Optional[List[Dict]] = None,
) -> np.ndarray:
    """Compute content-agnostic information density score for each sentence.

    Information density captures how much important information a sentence contains
    using universal, domain-independent signals:
    - TF-IDF of content words (high-value terms)
    - Sentence complexity (normalized length)
    - Structural signals (e.g., list items, headings)

    Args:
        sentence_texts: List of sentence text strings
        sentences_data: Optional list of sentence metadata dicts

    Returns:
        Array of density scores (shape: N,) normalized to [0, 1]
    """
    import re
    from collections import Counter

    N = len(sentence_texts)
    if N == 0:
        return np.array([], dtype=np.float32)

    density_scores = np.zeros(N, dtype=np.float32)

    # Tokenize and compute document frequency
    all_tokens = []
    sentence_tokens = []
    for text in sentence_texts:
        # Extract tokens (alphanumeric sequences)
        tokens = re.findall(r'\b[A-Za-z0-9]+\b', text.lower())
        sentence_tokens.append(tokens)
        all_tokens.extend(tokens)

    # Document frequency (how many sentences contain each term)
    doc_freq = Counter()
    for tokens in sentence_tokens:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            doc_freq[token] += 1

    # Stop words (common words with low information value)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'it', 'its', 'which', 'who', 'what', 'where',
        'when', 'why', 'how'
    }

    for i, (text, tokens) in enumerate(zip(sentence_texts, sentence_tokens)):
        # Signal 1: TF-IDF score (emphasizes distinctive terms)
        tfidf_score = 0.0
        content_tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        if content_tokens:
            tf = Counter(content_tokens)
            for term, count in tf.items():
                tf_val = count / len(content_tokens)
                idf_val = math.log(N / max(1, doc_freq[term]))
                tfidf_score += tf_val * idf_val
            tfidf_score /= len(content_tokens)  # Normalize by content length

        # Signal 2: Sentence complexity (longer sentences may contain more info)
        # Normalized length score (sigmoid to cap at reasonable values)
        length_score = len(tokens) / (len(tokens) + 10)  # Sigmoid-like, caps around 20 tokens

        # Signal 3: Structural importance
        # Check for list markers, headings, emphasis
        structural_score = 0.0
        if sentences_data and i < len(sentences_data):
            row = sentences_data[i]
            # List items often contain important information
            if row.get('is_list_item', False):
                structural_score += 0.2
            # Headings are very important
            if row.get('is_heading', False):
                structural_score += 0.3

        # Check for textual markers
        if re.match(r'^\s*[-*•]\s+', text):  # Bullet point
            structural_score += 0.15
        if re.match(r'^\s*\d+\.\s+', text):  # Numbered list
            structural_score += 0.15

        # Combine signals with content-agnostic weights
        # Only TF-IDF (term importance), length (complexity), and structural markers
        density = (
            0.60 * tfidf_score +
            0.20 * length_score +
            0.20 * structural_score
        )

        density_scores[i] = density

    # Normalize to [0, 1] range
    if density_scores.max() > 0:
        density_scores = density_scores / density_scores.max()

    return density_scores


def apply_density_discount_to_penalty(
    base_penalty: float,
    density_score: float,
    discount_factor: float = 0.3,
) -> float:
    """Reduce length penalty for information-dense regions.

    High-density sentences should be allowed to form longer chunks without penalty,
    as they contain more valuable information per sentence.

    Args:
        base_penalty: The base length penalty
        density_score: Density score for the sentence/region (0-1)
        discount_factor: Maximum discount (0-1); default 0.3 means up to 30% reduction

    Returns:
        Adjusted penalty value
    """
    discount = density_score * discount_factor
    return base_penalty * (1.0 - discount)


# ---------------------------------------------------------------------------
# Core dynamic-programming segmentation
# ---------------------------------------------------------------------------

def dp_segment(
    S: np.ndarray,
    Q: np.ndarray,
    params: IDCParams,
    structural_priors: Optional[List[float]] = None,
    sentences_data: Optional[List[Dict]] = None,
    sentence_texts: Optional[List[str]] = None,
    use_density_awareness: bool = False,
    density_discount_factor: float = 0.3,
) -> List[Tuple[int, int, int, float]]:
    """Dynamic-programming segmentation returning chunk tuples.

    Args:
        S: Sentence embedding matrix (N, D)
        Q: Intent embedding matrix (M, D)
        params: IDC parameters
        structural_priors: Optional structural prior costs per sentence
        sentences_data: Optional sentence metadata
        sentence_texts: Optional sentence text strings
        use_density_awareness: Enable information density-aware penalties (NEW)
        density_discount_factor: Discount factor for dense regions (NEW)

    Returns:
        List of chunk tuples (start, end, intent_idx, similarity)
    """
    N, D = S.shape
    if N == 0:
        return []

    if sentence_texts is None and sentences_data:
        sentence_texts = [row.get("text", "") for row in sentences_data]
    if sentence_texts is None:
        sentence_texts = ["" for _ in range(N)]

    # NEW: Compute information density scores for all sentences
    density_scores = None
    if use_density_awareness:
        density_scores = compute_information_density(sentence_texts, sentences_data)
        print(f"  💡 Density-aware segmentation enabled: discount_factor={density_discount_factor:.2f}")
        print(f"     Density scores: min={density_scores.min():.3f}, max={density_scores.max():.3f}, mean={density_scores.mean():.3f}")


    if Q is None or len(Q) == 0:
        chunks: List[Tuple[int, int, int, float]] = []
        idx = 0
        while idx < N:
            end = min(N - 1, idx + params.max_len - 1)
            length = end - idx + 1
            if length < params.min_len and idx > 0:
                end = min(N - 1, idx + params.min_len - 1)
            chunks.append((idx, end, -1, 0.0))
            idx = end + 1
        return chunks

    min_len = params.min_len
    max_len = params.max_len
    target_len = max(5.0, max_len / 2.0)

    norms = np.linalg.norm(S, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    Sn = S / norms
    adjacent_cos = None
    if N > 1:
        adjacent_cos = np.sum(Sn[:-1] * Sn[1:], axis=1)

    Q_norms = np.linalg.norm(Q, axis=1, keepdims=True)
    Qn = Q / np.maximum(Q_norms, 1e-8)

    prefix = np.zeros((N + 1, D), dtype=np.float32)
    for i in range(1, N + 1):
        prefix[i] = prefix[i - 1] + S[i - 1]

    def chunk_mean(l: int, r: int) -> np.ndarray:
        vec = prefix[r + 1] - prefix[l]
        return vec / max(r - l + 1, 1)

    def length_penalty(length: int, chunk_start: int, chunk_end: int) -> float:
        """Compute length penalty with optional density-aware discount.

        Args:
            length: Chunk length in sentences
            chunk_start: Start sentence index
            chunk_end: End sentence index

        Returns:
            Penalty value (lower for information-dense chunks)
        """
        base_penalty = 0.0
        if params.length_penalty_mode == "quadratic":
            overflow = max(0.0, length - max_len)
            underflow = max(0.0, target_len - length)
            base_penalty = params.lam * (overflow ** 2 + 0.5 * underflow ** 2)
        else:
            overflow = max(0.0, length - target_len)
            underflow = max(0.0, target_len - length)
            base_penalty = params.lam * (overflow + 0.5 * underflow)

        # Apply density-aware discount if enabled
        if use_density_awareness and density_scores is not None:
            # Compute average density for the chunk
            chunk_density = float(density_scores[chunk_start:chunk_end+1].mean())
            base_penalty = apply_density_discount_to_penalty(
                base_penalty, chunk_density, density_discount_factor
            )

        return base_penalty

    dp = np.full(N + 1, -1e9, dtype=np.float32)
    prv = np.full(N + 1, -1, dtype=np.int32)
    best_intent = np.full(N + 1, -1, dtype=np.int32)
    last_chunk_len = np.full(N + 1, min_len, dtype=np.int32)
    dp[0] = 0.0

    for i in range(1, N + 1):
        best_score = -1e9
        best_j = -1
        best_k = -1
        best_len = min_len
        j_min = max(0, i - max_len)
        for j in range(i - 1, j_min - 1, -1):
            length = i - j
            if length < min_len:
                continue
            if j > 0 and last_chunk_len[j] < min_len:
                continue

            mean_vec = chunk_mean(j, i - 1)
            vn = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)
            sims = Qn @ vn
            relevance_idx = int(np.argmax(sims))
            relevance = float(sims[relevance_idx])

            coherence = (
                chunk_coherence(S, j, i - 1) if params.coherence_weight > 0 else 0.0
            )
            pen = length_penalty(length, j, i - 1)  # Pass chunk boundaries for density awareness
            struct_cost = 0.0
            if params.structural_priors and structural_priors and j < len(structural_priors):
                struct_cost = float(structural_priors[j])
            b_cost = boundary_cost(
                sentences_data,
                j,
                adjacent_cos,
                respect_paragraphs=params.respect_paragraphs,
            )

            score = (
                dp[j]
                + relevance
                + params.coherence_weight * coherence
                - pen
                - params.boundary_penalty
                - struct_cost
                - b_cost
            )
            if score > best_score:
                best_score = score
                best_j = j
                best_k = relevance_idx
                best_len = length
        dp[i] = best_score
        prv[i] = best_j
        best_intent[i] = best_k
        last_chunk_len[i] = best_len

    spans: List[Tuple[int, int, int]] = []
    i = N
    while i > 0:
        j = int(prv[i])
        if j < 0 or j >= i:
            j = max(0, i - min_len)
        spans.append((j, i - 1, int(best_intent[i])))
        i = j
    spans.reverse()

    chunks: List[Tuple[int, int, int, float]] = []
    for left, right, best_idx in spans:
        mean_vec = chunk_mean(left, right)
        vn = mean_vec / (np.linalg.norm(mean_vec) + 1e-8)
        sims = Qn @ vn
        best_intent_idx = int(np.argmax(sims)) if best_idx < 0 else best_idx
        similarity = float(sims[best_intent_idx])
        chunks.append((left, right, best_intent_idx, similarity))

    return chunks



# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def refine_chunks(
    chunks: List[Tuple[int, int, int, float]],
    S: np.ndarray,
    Q: np.ndarray,
    params: IDCParams,
    sentence_rows: Optional[List[Dict]] = None,
    sentence_texts: Optional[List[str]] = None,
) -> List[Tuple[int, int, int, float]]:
    """Merge tiny fragments and split incoherent or long chunks."""
    if not chunks:
        return []

    sentence_texts = sentence_texts or [row.get("text", "") for row in (sentence_rows or [])]
    coherences = [chunk_coherence(S, start, end) for start, end, _, _ in chunks]
    if coherences:
        coherence_threshold = float(np.percentile(coherences, 25))
    else:
        coherence_threshold = 0.0

    states = [
        {
            "start": start,
            "end": end,
            "intent": intent_idx,
            "similarity": sim,
            "coherence": chunk_coherence(S, start, end),
        }
        for start, end, intent_idx, sim in chunks
    ]

    max_len = params.max_len
    min_chunk = params.min_chunk_sent

    idx = 0
    while idx < len(states):
        state = states[idx]
        length = state["end"] - state["start"] + 1
        if length >= min_chunk or len(states) == 1:
            idx += 1
            continue
        merged = False
        if idx > 0:
            prev = states[idx - 1]
            if prev["end"] + 1 == state["start"]:
                combined_len = prev["end"] - prev["start"] + 1 + length
                if combined_len <= max_len:
                    combined_coh = chunk_coherence(S, prev["start"], state["end"])
                    if combined_coh >= min(prev["coherence"], state["coherence"]) * 0.95:
                        prev["end"] = state["end"]
                        prev["coherence"] = combined_coh
                        states.pop(idx)
                        merged = True
                        continue
        if not merged and idx + 1 < len(states):
            nxt = states[idx + 1]
            if state["end"] + 1 == nxt["start"]:
                combined_len = nxt["end"] - nxt["start"] + 1 + length
                if combined_len <= max_len:
                    combined_coh = chunk_coherence(S, state["start"], nxt["end"])
                    if combined_coh >= min(nxt["coherence"], state["coherence"]) * 0.95:
                        nxt["start"] = state["start"]
                        nxt["coherence"] = combined_coh
                        states.pop(idx)
                        merged = True
                        continue
        if not merged:
            idx += 1

    Qn = None
    if Q is not None and Q.size > 0:
        Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-8)

    def recompute(start: int, end: int) -> Tuple[int, float]:
        if Qn is None:
            return -1, 0.0
        vec = np.mean(S[start : end + 1], axis=0)
        vn = vec / (np.linalg.norm(vec) + 1e-8)
        sims = Qn @ vn
        best = int(np.argmax(sims))
        return best, float(sims[best])

    idx = 0
    while idx < len(states):
        state = states[idx]
        length = state["end"] - state["start"] + 1
        coherence = state["coherence"]
        if length <= params.max_chunk_sent and coherence >= coherence_threshold:
            idx += 1
            continue
        sub_chunks = split_long_chunk(
            (state["start"], state["end"], state["intent"], state["similarity"]),
            S,
            Q,
            sentence_texts,
            sentence_rows=sentence_rows,
            respect_paragraphs=params.respect_paragraphs,
            min_length=min_chunk,
        )
        if len(sub_chunks) <= 1:
            idx += 1
            continue
        states.pop(idx)
        for offset, (s, e, intent_idx, sim) in enumerate(sub_chunks):
            coh = chunk_coherence(S, s, e)
            states.insert(idx + offset, {
                "start": s,
                "end": e,
                "intent": intent_idx,
                "similarity": sim,
                "coherence": coh,
            })

    refined: List[Tuple[int, int, int, float]] = []
    for state in states:
        intent_idx, sim = recompute(state["start"], state["end"])
        refined.append((state["start"], state["end"], intent_idx, sim))
    return refined


def merge_similar_adjacent_chunks(
    chunks: List[Tuple[int, int, int, float]],
    max_len: int,
    merge_same_intent: bool = True,
) -> List[Tuple[int, int, int, float]]:
    if len(chunks) <= 1 or not merge_same_intent:
        return chunks

    merged: List[Tuple[int, int, int, float]] = []
    current_start, current_end, current_intent, current_sim = chunks[0]
    for next_start, next_end, next_intent, next_sim in chunks[1:]:
        if (
            current_intent == next_intent
            and current_end + 1 == next_start
            and (next_end - current_start + 1) <= max_len
        ):
            current_end = next_end
            current_sim = max(current_sim, next_sim)
        else:
            merged.append((current_start, current_end, current_intent, current_sim))
            current_start, current_end, current_intent, current_sim = (
                next_start,
                next_end,
                next_intent,
                next_sim,
            )
    merged.append((current_start, current_end, current_intent, current_sim))
    return merged


def apply_diversity_bonus(
    chunks: List[Tuple[int, int, int, float]],
    intent_texts: List[str],
    diversity_weight: float = 0.1,
) -> List[Tuple[int, int, int, float]]:
    if diversity_weight <= 0 or len(chunks) <= 1:
        return chunks

    intent_counts: Dict[str, int] = {}
    for _, _, intent_idx, _ in chunks:
        if 0 <= intent_idx < len(intent_texts):
            key = intent_texts[intent_idx]
            intent_counts[key] = intent_counts.get(key, 0) + 1

    adjusted: List[Tuple[int, int, int, float]] = []
    for start, end, intent_idx, sim in chunks:
        if 0 <= intent_idx < len(intent_texts):
            key = intent_texts[intent_idx]
            count = intent_counts.get(key, 1)
            bonus = diversity_weight * (1.0 / count if count > 0 else 1.0)
            adjusted.append((start, end, intent_idx, sim + bonus))
        else:
            adjusted.append((start, end, intent_idx, sim))
    return adjusted


def compute_structural_priors(
    sent_rows: List[Dict],
    para_boundary_discount: float = 0.5,
    respect_paragraphs: bool = True,
) -> List[float]:
    """Return structural boundary costs (negative favours a split)."""
    N = len(sent_rows)
    priors = [0.0] * N
    if N <= 1:
        return priors
    para_keys = ("paragraph_id", "para_id", "paragraph")
    for i in range(N - 1):
        prev_row = sent_rows[i]
        next_row = sent_rows[i + 1]
        prev_text = (prev_row.get("text") or "").strip()
        next_text = (next_row.get("text") or "").strip()

        cost = 0.0
        prev_para = next_para = None
        for key in para_keys:
            if prev_para is None:
                prev_para = prev_row.get(key)
            if next_para is None:
                next_para = next_row.get(key)
        if prev_para is not None and next_para is not None:
            if prev_para != next_para:
                cost -= para_boundary_discount
            elif respect_paragraphs:
                cost += para_boundary_discount * 0.5

        prev_section = prev_row.get("section_path")
        next_section = next_row.get("section_path")
        if prev_section and next_section and prev_section != next_section:
            cost -= para_boundary_discount * 0.6

        if respect_paragraphs:
            block_prev = str(prev_row.get("block_type", "")).lower()
            block_next = str(next_row.get("block_type", "")).lower()
            if block_prev in {"code", "table"} or block_next in {"code", "table"}:
                cost += para_boundary_discount

        if prev_text.endswith((".", "!", "?")) and next_text and next_text[0].isupper():
            cost -= para_boundary_discount * 0.3

        priors[i] = cost
    return priors





# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def segment_document(
    doc_id: str,
    sentence_rows: List[Dict],
    sentence_vectors: np.ndarray,
    intent_rows: List[Dict],
    intent_vectors: np.ndarray,
    params: IDCParams,
    use_density_awareness: bool = False,
    density_discount_factor: float = 0.3,
) -> SegmentDocumentResult:
    """Segment a document into intent-aligned chunks.

    Args:
        doc_id: Document identifier
        sentence_rows: Sentence metadata
        sentence_vectors: Sentence embeddings
        intent_rows: Intent metadata
        intent_vectors: Intent embeddings
        params: IDC parameters
        use_density_awareness: Enable density-aware penalties (NEW)
        density_discount_factor: Discount factor for dense regions (NEW)

    Returns:
        Segmentation result with chunks
    """
    sentence_texts = [row.get("text", "") for row in sentence_rows]
    structural_priors = (
        compute_structural_priors(sentence_rows, params.para_discount, params.respect_paragraphs)
        if params.structural_priors
        else None
    )

    chunks_raw = dp_segment(
        sentence_vectors,
        intent_vectors,
        params=params,
        structural_priors=structural_priors,
        sentences_data=sentence_rows,
        sentence_texts=sentence_texts,
        use_density_awareness=use_density_awareness,
        density_discount_factor=density_discount_factor,
    )

    chunks_working = chunks_raw
    if params.postprocess:
        chunks_working = refine_chunks(
            chunks_working,
            sentence_vectors,
            intent_vectors,
            params,
            sentence_rows=sentence_rows,
            sentence_texts=sentence_texts,
        )

    if params.merge_adjacent:
        chunks_working = merge_similar_adjacent_chunks(
            chunks_working, params.max_len, merge_same_intent=True
        )
    if params.diversity_weight > 0 and intent_rows:
        intent_texts = [row.get("text", "") for row in intent_rows]
        chunks_working = apply_diversity_bonus(
            chunks_working, intent_texts, params.diversity_weight
        )

    chunk_dicts: List[Dict] = []
    for start, end, intent_idx, sim in chunks_working:
        start_row = sentence_rows[start]
        end_row = sentence_rows[end]
        text = " ".join(row.get("text", "") for row in sentence_rows[start : end + 1])
        intent_text = None
        if 0 <= intent_idx < len(intent_rows):
            intent_text = intent_rows[intent_idx].get("text")
        chunk_dicts.append(
            {
                "start_sent": int(start_row["sent_id"]),
                "end_sent": int(end_row["sent_id"]),
                "num_sentences": end - start + 1,
                "intent": intent_text,
                "similarity": round(float(sim), 4),
                "text": text,
            }
        )

    return SegmentDocumentResult(
        doc_id=doc_id,
        chunks=chunk_dicts,
        merged_from=len(chunks_raw),
        merged_to=len(chunks_working),
    )


def segment_corpus(
    sentence_rows: List[Dict],
    sentence_vectors: np.ndarray,
    intent_rows: List[Dict],
    intent_vectors: np.ndarray,
    params: IDCParams,
    doc_ids: Optional[Iterable[str]] = None,
    use_density_awareness: bool = False,
    density_discount_factor: float = 0.3,
) -> List[SegmentDocumentResult]:
    """Segment all documents in a corpus.

    Args:
        sentence_rows: List of sentence metadata dicts
        sentence_vectors: Sentence embeddings matrix
        intent_rows: List of intent metadata dicts
        intent_vectors: Intent embeddings matrix
        params: IDC parameters
        doc_ids: Optional list of document IDs to process
        use_density_awareness: Enable density-aware segmentation (NEW)
        density_discount_factor: Discount factor for dense regions (NEW)

    Returns:
        List of segmentation results per document
    """
    by_doc_sent = group_sentences_by_doc(sentence_rows)
    by_doc_int = group_intents_by_doc(intent_rows)

    results: List[SegmentDocumentResult] = []
    doc_iter = list(doc_ids) if doc_ids is not None else list(by_doc_sent.keys())
    for doc_id in doc_iter:
        sent_idx_list = by_doc_sent.get(doc_id)
        if not sent_idx_list:
            raise ValueError(f"No sentences found for document '{doc_id}'")
        S = sentence_vectors[np.array(sent_idx_list)]
        doc_sentence_rows = [sentence_rows[i] for i in sent_idx_list]

        intent_idx_list = by_doc_int.get(doc_id, [])
        if intent_idx_list:
            Q = intent_vectors[np.array(intent_idx_list)]
            doc_intent_rows = [intent_rows[i] for i in intent_idx_list]
        else:
            Q = np.zeros((0, sentence_vectors.shape[1]), dtype=np.float32)
            doc_intent_rows = []

        result = segment_document(
            doc_id, doc_sentence_rows, S, doc_intent_rows, Q, params,
            use_density_awareness=use_density_awareness,
            density_discount_factor=density_discount_factor,
        )
        results.append(result)
    return results


def segment_corpus_from_files(
    sentences_path: str | Path,
    sentence_embs_path: str | Path,
    sentences_meta_path: Optional[str | Path],
    intents_flat_path: str | Path,
    intent_embs_path: str | Path,
    params: IDCParams,
    doc_ids: Optional[Iterable[str]] = None,
    use_density_awareness: bool = False,
    density_discount_factor: float = 0.3,
) -> List[SegmentDocumentResult]:
    """Segment corpus from file paths.

    Args:
        sentences_path: Path to sentences.jsonl
        sentence_embs_path: Path to sentence embeddings
        sentences_meta_path: Optional path to sentence metadata
        intents_flat_path: Path to flattened intents
        intent_embs_path: Path to intent embeddings
        params: IDC parameters
        doc_ids: Optional list of doc IDs to process
        use_density_awareness: Enable density-aware segmentation (NEW)
        density_discount_factor: Discount factor for dense regions (NEW)

    Returns:
        List of segmentation results
    """
    sentences_path = Path(sentences_path)
    sentence_embs_path = Path(sentence_embs_path)
    intents_flat_path = Path(intents_flat_path)
    intent_embs_path = Path(intent_embs_path)

    sent_rows = read_jsonl(sentences_path)
    sent_meta = read_jsonl(sentences_meta_path) if sentences_meta_path else None
    sentence_vectors = np.load(sentence_embs_path)
    if len(sent_rows) != sentence_vectors.shape[0]:
        raise ValueError("Sentence embeddings do not match sentences.jsonl length")
    if sent_meta is not None and len(sent_meta) != len(sent_rows):
        raise ValueError("sentences.meta.jsonl does not align with sentences.jsonl")

    intent_rows = read_jsonl(intents_flat_path)
    if intent_rows:
        intent_vectors = np.load(intent_embs_path)
        if intent_vectors.shape[0] != len(intent_rows):
            raise ValueError("Intent embeddings do not match intents.flat.jsonl length")
    else:
        intent_vectors = np.zeros((0, sentence_vectors.shape[1]), dtype=np.float32)

    return segment_corpus(
        sentence_rows=sent_rows,
        sentence_vectors=sentence_vectors,
        intent_rows=intent_rows,
        intent_vectors=intent_vectors,
        params=params,
        doc_ids=doc_ids,
        use_density_awareness=use_density_awareness,
        density_discount_factor=density_discount_factor,
    )


__all__ = [
    "IDCParams",
    "SegmentDocumentResult",
    "initial_idc_pass",
    "split_long_chunk",
    "multi_pass_segmentation",
    "segment_document_multi_pass",
    "segment_document",
    "segment_corpus",
    "segment_corpus_from_files",
    "read_jsonl",
    "write_jsonl",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IDC segmentation (DP or multi-pass)")
    parser.add_argument(
        "--algorithm",
        choices=["dp", "multi-pass"],
        default="dp",
        help="Choose between standard DP IDC and the multi-pass variant.",
    )
    parser.add_argument("--sentences", type=str, default="out/sentences.jsonl")
    parser.add_argument("--sentence-embs", type=str, default="out/sentence_embs.npy")
    parser.add_argument("--sentences-meta", type=str, default="out/sentences.meta.jsonl")
    parser.add_argument("--intents-flat", type=str, default="out/intents.flat.jsonl")
    parser.add_argument("--intent-embs", type=str, default="out/intent_embs.npy")

    parser.add_argument("--lambda", dest="lam", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--min-len", type=int, default=3)
    parser.add_argument("--boundary-penalty", type=float, default=1.2)
    parser.add_argument("--coherence-weight", type=float, default=0.3)
    parser.add_argument(
        "--length-penalty",
        choices=["linear", "quadratic"],
        default="linear",
        help="Length penalty mode for DP objective.",
    )
    parser.add_argument("--min-chunk-sent", type=int, default=2)
    parser.add_argument(
        "--max-chunk-sent",
        type=int,
        default=None,
        help="Maximum sentences per chunk after post-processing (defaults to --max-len).",
    )
    parser.add_argument(
        "--respect-paragraphs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to discourage boundaries inside paragraphs or code blocks.",
    )
    parser.add_argument(
        "--postprocess",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable post-processing refine/merge stage.",
    )
    parser.add_argument("--no-structural-priors", action="store_true", help="Disable structural priors")
    parser.add_argument("--merge-adjacent", action="store_true")
    parser.add_argument("--para-discount", type=float, default=0.5)
    parser.add_argument("--diversity-weight", type=float, default=0.0)
    parser.add_argument(
        "--max-len-final",
        type=int,
        default=8,
        help="Final max sentences per chunk when using multi-pass.",
    )
    parser.add_argument(
        "--initial-max-len",
        type=int,
        default=15,
        help="Initial generous max sentences per chunk for multi-pass.",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        dest="doc_ids",
        help="Restrict to specific doc ids; required for multi-pass.",
    )

    # NEW: Information density awareness arguments
    parser.add_argument(
        "--density-aware",
        action="store_true",
        default=False,
        help="Enable information density-aware segmentation (reduces penalties for dense regions)",
    )
    parser.add_argument(
        "--density-discount-factor",
        type=float,
        default=0.3,
        help="Discount factor for information-dense regions (0.0-1.0, default: 0.3)",
    )

    parser.add_argument("--out", type=str, default="out/segments.jsonl")
    return parser.parse_args(argv)



def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    if args.algorithm == "multi-pass":
        if not args.doc_ids or len(args.doc_ids) != 1:
            raise SystemExit("Multi-pass mode requires exactly one --doc-id argument.")
        doc_id = args.doc_ids[0]
        chunks = segment_document_multi_pass(
            sentences_path=args.sentences,
            sentence_embs_path=args.sentence_embs,
            intents_flat_path=args.intents_flat,
            intent_embs_path=args.intent_embs,
            doc_id=doc_id,
            max_len_final=args.max_len_final,
            min_len=args.min_len,
            initial_max_len=args.initial_max_len,
            lam=args.lam,
            boundary_pen=args.boundary_penalty,
            coherence_weight=args.coherence_weight,
        )
        write_jsonl(args.out, [{"doc_id": doc_id, "chunks": chunks}])
        print(f"Saved multi-pass segments → {args.out}")
        return

    max_chunk_sent = args.max_chunk_sent if args.max_chunk_sent else args.max_len
    params = IDCParams(
        lam=args.lam,
        max_len=args.max_len,
        min_len=args.min_len,
        boundary_penalty=args.boundary_penalty,
        coherence_weight=args.coherence_weight,
        merge_adjacent=args.merge_adjacent,
        structural_priors=not args.no_structural_priors,
        para_discount=args.para_discount,
        diversity_weight=args.diversity_weight,
        length_penalty_mode=args.length_penalty,
        min_chunk_sent=max(args.min_chunk_sent, args.min_len),
        max_chunk_sent=max_chunk_sent,
        respect_paragraphs=args.respect_paragraphs,
        postprocess=args.postprocess,
    )

    results = segment_corpus_from_files(
        sentences_path=args.sentences,
        sentence_embs_path=args.sentence_embs,
        sentences_meta_path=args.sentences_meta,
        intents_flat_path=args.intents_flat,
        intent_embs_path=args.intent_embs,
        params=params,
        doc_ids=args.doc_ids,
        use_density_awareness=args.density_aware,
        density_discount_factor=args.density_discount_factor,
    )

    for res in results:
        if res.merged_from != res.merged_to:
            print(
                f"  Merged {res.merged_from} -> {res.merged_to} chunks "
                f"(same intent adjacency)"
            )

    write_jsonl(
        args.out,
        ({"doc_id": res.doc_id, "chunks": res.chunks} for res in results),
    )
    print(f"Saved segments → {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
