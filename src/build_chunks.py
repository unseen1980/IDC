#!/usr/bin/env python3
from __future__ import annotations

"""Build chunk metadata and multi-view embeddings from IDC segments."""

import argparse
from argparse import BooleanOptionalAction
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from config import DEFAULT_VIEWS, parse_views_arg

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "has", "have", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "there", "this", "to", "was", "were",
    "which", "with",
}


def ensure_dir(path: str | Path) -> None:
    """Ensure the parent directory for ``path`` exists."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> List[Dict]:
    """Read a JSONL file into memory."""
    out: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    """Write dictionaries to ``path`` as JSON lines."""
    path = Path(path)
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def group_sentences_by_doc(sent_rows: List[Dict]) -> Dict[str, List[Tuple[int, int]]]:
    """Group sentence rows by document, preserving order."""
    buckets: Dict[str, List[Tuple[int, int]]] = {}
    for idx, row in enumerate(sent_rows):
        buckets.setdefault(row["doc_id"], []).append((int(row["sent_id"]), idx))
    for key in buckets:
        buckets[key].sort(key=lambda item: item[0])
    return buckets


def l2_normalize(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalise rows of ``mat`` in-place safe manner."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


def compute_contextual_embedding(
    S: np.ndarray,
    sentence_indices: List[int],
    prev_chunk_indices: Optional[List[int]] = None,
    next_chunk_indices: Optional[List[int]] = None,
    context_weight: float = 0.15,
) -> np.ndarray:
    """Compute chunk embedding with contextual information from adjacent chunks.

    Args:
        S: Sentence embedding matrix (n_sentences, dim)
        sentence_indices: Indices of sentences in current chunk
        prev_chunk_indices: Indices of sentences in previous chunk (optional)
        next_chunk_indices: Indices of sentences in next chunk (optional)
        context_weight: Weight for context from adjacent chunks (default: 0.15 each)

    Returns:
        Contextual chunk embedding incorporating adjacent context
    """
    if not sentence_indices:
        return np.zeros((S.shape[1],), dtype=np.float32)

    # Main chunk embedding
    chunk_sents = S[sentence_indices]
    main_emb = chunk_sents.mean(axis=0)

    # Weight for main chunk (leave room for prev + next context)
    main_weight = 1.0 - (2 * context_weight)
    contextual_emb = main_emb * main_weight

    # Add context from previous chunk (last 2 sentences)
    if prev_chunk_indices and len(prev_chunk_indices) > 0:
        context_size = min(2, len(prev_chunk_indices))
        prev_context_indices = prev_chunk_indices[-context_size:]
        prev_context = S[prev_context_indices].mean(axis=0)
        contextual_emb += prev_context * context_weight

    # Add context from next chunk (first 2 sentences)
    if next_chunk_indices and len(next_chunk_indices) > 0:
        context_size = min(2, len(next_chunk_indices))
        next_context_indices = next_chunk_indices[:context_size]
        next_context = S[next_context_indices].mean(axis=0)
        contextual_emb += next_context * context_weight

    return contextual_emb.astype(np.float32)


def compute_intent_weighted_embedding(
    S: np.ndarray,
    sentence_indices: List[int],
    assigned_intent: Optional[str],
    intent_rows: Optional[List[Dict]],
    intent_embs: Optional[np.ndarray],
    prev_chunk_indices: Optional[List[int]] = None,
    next_chunk_indices: Optional[List[int]] = None,
    use_contextual: bool = False,
    context_weight: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Return (weighted chunk vector, anchor vector, intent embedding).

    Args:
        S: Sentence embedding matrix
        sentence_indices: Indices of sentences in current chunk
        assigned_intent: Intent assigned to this chunk
        intent_rows: List of intent metadata
        intent_embs: Intent embedding matrix
        prev_chunk_indices: Indices from previous chunk for context (NEW)
        next_chunk_indices: Indices from next chunk for context (NEW)
        use_contextual: Whether to add contextual information (NEW)
        context_weight: Weight for contextual information (NEW)

    Returns:
        Tuple of (chunk embedding, anchor embedding, intent embedding)
    """
    if not sentence_indices:
        return (
            np.zeros((S.shape[1],), dtype=np.float32),
            np.zeros((S.shape[1],), dtype=np.float32),
            None,
        )

    chunk_sents = S[sentence_indices]
    mean_vec = chunk_sents.mean(axis=0)
    anchor_vec = chunk_sents[len(sentence_indices) // 2]

    if not assigned_intent or not intent_rows or intent_embs is None or intent_embs.size == 0:
        # No intent weighting, but still apply contextual embedding if enabled
        if use_contextual:
            mean_vec = compute_contextual_embedding(
                S, sentence_indices, prev_chunk_indices, next_chunk_indices, context_weight
            )
        return mean_vec, anchor_vec, None

    match_idx = None
    for idx, row in enumerate(intent_rows):
        if row.get("text") == assigned_intent:
            match_idx = idx
            break
    if match_idx is None:
        if use_contextual:
            mean_vec = compute_contextual_embedding(
                S, sentence_indices, prev_chunk_indices, next_chunk_indices, context_weight
            )
        return mean_vec, anchor_vec, None

    intent_vec = intent_embs[match_idx : match_idx + 1]
    chunk_norm = l2_normalize(chunk_sents.copy())
    intent_norm = l2_normalize(intent_vec.copy())
    sims = chunk_norm @ intent_norm.T
    weights = np.exp(sims.flatten() - np.max(sims))
    weights = weights / max(weights.sum(), 1e-8)
    weighted_emb = np.sum(chunk_sents * weights.reshape(-1, 1), axis=0)
    anchor_idx = int(np.argmax(weights))
    anchor_vec = chunk_sents[anchor_idx]

    # Apply contextual embedding if enabled
    if use_contextual:
        # Blend intent-weighted embedding with contextual information
        contextual_part = compute_contextual_embedding(
            S, sentence_indices, prev_chunk_indices, next_chunk_indices, context_weight
        )
        # 70% intent-weighted, 30% contextual (adjust as needed)
        weighted_emb = 0.7 * weighted_emb + 0.3 * contextual_part

    return weighted_emb, anchor_vec, intent_elem(intent_embs, match_idx)


def intent_elem(intent_embs: Optional[np.ndarray], idx: int) -> Optional[np.ndarray]:
    """Safe helper returning a 1D intent embedding copy."""
    if intent_embs is None or intent_embs.size == 0:
        return None
    if idx < 0 or idx >= intent_embs.shape[0]:
        return None
    return intent_embs[idx].astype(np.float32)


def tokenize(text: str) -> List[str]:
    """Tokenise ``text`` using a simple regex and stopword filter."""
    tokens: List[str] = []
    for match in TOKEN_RE.findall(text.lower()):
        tok = match.strip("'")
        if tok and tok not in STOPWORDS:
            tokens.append(tok)
    return tokens


def extract_keywords(
    tokens: List[str],
    doc_df: Counter,
    total_sentences: int,
    top_k: int = 8,
) -> List[str]:
    """Compute lightweight TF-IDF keywords from tokens."""
    if not tokens:
        return []
    counts = Counter(tokens)
    scores: List[Tuple[str, float]] = []
    for term, tf in counts.items():
        df = doc_df.get(term, 0)
        idf = math.log(1.0 + total_sentences / (1.0 + df))
        scores.append((term, (tf / len(tokens)) * idf))
    scores.sort(key=lambda item: item[1], reverse=True)
    return [term for term, _ in scores[:top_k]]


def summarise_chunk(sentences: List[str], max_sentences: int = 2) -> str:
    """Return a naive summary using the first ``max_sentences`` sentences."""
    if not sentences:
        return ""
    return " ".join(sentences[:max_sentences]).strip()


def weighted_sentence_vector(
    S: np.ndarray,
    indices: List[int],
    weights: List[float],
) -> np.ndarray:
    """Weighted average of sentence vectors given indices and weights."""
    if not indices:
        return np.zeros((S.shape[1],), dtype=np.float32)
    vecs = S[indices]
    weights_arr = np.array(weights, dtype=np.float32)
    total = float(weights_arr.sum())
    if total <= 0:
        return vecs.mean(axis=0)
    weights_arr = weights_arr / total
    return np.sum(vecs * weights_arr.reshape(-1, 1), axis=0)


def compute_chunk_coherence(S: np.ndarray, indices: List[int]) -> float:
    """Compute average pairwise cosine similarity for the given sentences."""
    if len(indices) <= 1:
        return 1.0
    window = S[indices]
    norms = np.linalg.norm(window, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalised = window / norms
    sim = normalised @ normalised.T
    mask = np.triu(np.ones((len(indices), len(indices)), dtype=bool), k=1)
    denom = mask.sum()
    if denom == 0:
        return 1.0
    return float(sim[mask].mean())


def add_view_vector(
    view_name: str,
    vector: Optional[np.ndarray],
    view_text: str,
    view_vectors: List[np.ndarray],
    view_records: List[Dict],
    chunk_uid: str,
) -> None:
    """Append a view vector and record metadata if valid."""
    if vector is None or vector.size == 0:
        return
    if view_name != "text" and not view_text.strip():
        return
    view_index = len(view_vectors)
    view_vectors.append(vector.astype(np.float32))
    view_records.append(
        {
            "chunk_uid": chunk_uid,
            "view": view_name,
            "vector_index": view_index,
            "text": view_text,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="IDC: Build chunk embeddings and multi-view index with contextual information")
    parser.add_argument("--sentences", type=str, default="out/sentences.jsonl")
    parser.add_argument("--sentence-embs", type=str, default="out/sentence_embs.npy")
    parser.add_argument("--segments", type=str, default="out/segments.jsonl")
    parser.add_argument("--out-embs", type=str, default="out/chunk_embs.npy")
    parser.add_argument("--out-chunks", type=str, default="out/chunks.jsonl")
    parser.add_argument("--out-anchor-embs", type=str, default=None)
    parser.add_argument("--intent-weighted", action="store_true")
    parser.add_argument("--intent-embs", type=str)
    parser.add_argument("--intents-flat", type=str)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--multi-view-index", action=BooleanOptionalAction, default=True)
    parser.add_argument("--views", type=str, default=",".join(DEFAULT_VIEWS))
    parser.add_argument("--out-view-embs", type=str, default="out/chunk_view_embs.npy")
    parser.add_argument("--out-index", type=str, default="out/chunks.index.jsonl")
    parser.add_argument("--keyword-topk", type=int, default=8)

    # NEW: Contextual embedding arguments
    parser.add_argument("--contextual-embeddings", action="store_true", default=False,
                        help="Enable contextual embeddings using adjacent chunk context")
    parser.add_argument("--context-weight", type=float, default=0.15,
                        help="Weight for context from adjacent chunks (default: 0.15 per side)")

    args = parser.parse_args()

    # Display configuration
    if args.contextual_embeddings:
        print(f"✨ Contextual embeddings enabled (context_weight={args.context_weight})")
        print(f"   Each chunk will incorporate context from adjacent chunks")

    sent_rows = read_jsonl(args.sentences)
    S = np.load(args.sentence_embs)
    seg_docs = read_jsonl(args.segments)
    if len(sent_rows) != S.shape[0]:
        raise ValueError("Mismatch: sentences.jsonl vs sentence_embs.npy")

    by_doc = group_sentences_by_doc(sent_rows)

    intent_rows: Optional[List[Dict]] = None
    intent_embs: Optional[np.ndarray] = None
    if args.intent_weighted:
        if not args.intent_embs or not args.intents_flat:
            raise ValueError("--intent-embs and --intents-flat required for --intent-weighted")
        intent_rows = read_jsonl(args.intents_flat)
        intent_embs = np.load(args.intent_embs)
        if intent_rows and intent_embs.shape[0] != len(intent_rows):
            raise ValueError("Intent embeddings shape mismatch")

    active_views = parse_views_arg(args.views)
    if "text" not in active_views:
        active_views = ("text",) + tuple(v for v in active_views if v != "text")
    if not args.multi_view_index:
        active_views = ("text",)

    chunk_vecs: List[np.ndarray] = []
    anchor_vecs: List[np.ndarray] = []
    chunk_meta: List[Dict] = []
    view_vectors: List[np.ndarray] = []
    index_rows: List[Dict] = []

    D = S.shape[1]
    global_chunk_id = 0

    for doc in seg_docs:
        doc_id = doc["doc_id"]
        mapping = by_doc.get(doc_id, [])
        if not mapping:
            print(f"[WARN] No sentences for doc_id={doc_id}; skipping")
            continue

        ordered_rows = [sent_rows[idx] for _, idx in mapping]
        doc_texts = [row.get("text", "") for row in ordered_rows]
        doc_tokens = [tokenize(text) for text in doc_texts]
        doc_df = Counter()
        for tokens in doc_tokens:
            for term in set(tokens):
                doc_df[term] += 1
        total_sentences_doc = len(doc_tokens)

        sid_to_global = {sid: gidx for sid, gidx in mapping}
        global_to_local = {gidx: pos for pos, (_, gidx) in enumerate(mapping)}

        # Precompute all chunk indices for contextual embedding
        chunks_list = doc.get("chunks", [])
        all_chunk_indices = []
        for chunk in chunks_list:
            chunk_start = int(chunk.get("start_sent", 0))
            chunk_end = int(chunk.get("end_sent", chunk_start))
            global_indices = [sid_to_global.get(i) for i in range(chunk_start, chunk_end + 1)]
            global_indices = [idx for idx in global_indices if idx is not None]
            all_chunk_indices.append(global_indices)

        for local_cid, chunk in enumerate(chunks_list):
            # Get chunk sentence boundaries
            chunk_start_sent = int(chunk.get("start_sent", 0))
            chunk_end_sent = int(chunk.get("end_sent", chunk_start_sent))

            global_indices = all_chunk_indices[local_cid]

            # Get previous and next chunk indices for contextual embedding
            prev_chunk_indices = all_chunk_indices[local_cid - 1] if local_cid > 0 else None
            next_chunk_indices = all_chunk_indices[local_cid + 1] if local_cid < len(chunks_list) - 1 else None

            intent_vec: Optional[np.ndarray] = None
            coherence = 1.0
            if not global_indices:
                vec = np.zeros((D,), dtype=np.float32)
                anchor = vec
            else:
                idx_array = np.array(global_indices)
                coherence = compute_chunk_coherence(S, idx_array.tolist())
                if args.intent_weighted and intent_rows is not None and intent_embs is not None and chunk.get("intent"):
                    vec, anchor, intent_vec = compute_intent_weighted_embedding(
                        S,
                        idx_array.tolist(),
                        chunk.get("intent"),
                        intent_rows,
                        intent_embs,
                        prev_chunk_indices=prev_chunk_indices,
                        next_chunk_indices=next_chunk_indices,
                        use_contextual=args.contextual_embeddings,
                        context_weight=args.context_weight,
                    )
                else:
                    # For non-intent-weighted, still apply contextual if enabled
                    if args.contextual_embeddings:
                        vec = compute_contextual_embedding(
                            S,
                            idx_array.tolist(),
                            prev_chunk_indices=prev_chunk_indices,
                            next_chunk_indices=next_chunk_indices,
                            context_weight=args.context_weight,
                        )
                    else:
                        vec = S[idx_array].mean(axis=0)
                    anchor = S[idx_array[len(idx_array) // 2]]
                    intent_vec = None
            chunk_uid = f"{doc_id}::c{local_cid + 1}"
            text = chunk.get("text", "")

            local_positions = [global_to_local[idx] for idx in global_indices if idx in global_to_local]
            sentences_subset = [doc_texts[pos] for pos in local_positions]
            summary_text = summarise_chunk(sentences_subset)
            token_stream: List[str] = []
            for pos in local_positions:
                token_stream.extend(doc_tokens[pos])
            keywords = extract_keywords(token_stream, doc_df, total_sentences_doc, top_k=args.keyword_topk)
            keywords_text = " ".join(keywords)

            chunk_payload = {
                "start_sent": chunk_start_sent,
                "end_sent": chunk_end_sent,
                "num_sentences": int(chunk_end_sent - chunk_start_sent + 1),
                "sentences": sentences_subset,
                "intent": chunk.get("intent"),
                "similarity": chunk.get("similarity"),
                "text": text,
                "summary": summary_text,
                "keywords": keywords,
                "coherence": coherence,
            }
            chunk_meta.append({"chunk_uid": chunk_uid, "doc_id": doc_id, **chunk_payload})

            chunk_vecs.append(vec.astype(np.float32))
            anchor_vecs.append(anchor.astype(np.float32))

            index_entry = {
                "chunk_uid": chunk_uid,
                "doc_id": doc_id,
                **chunk_payload,
                "views": [],
            }

            if args.multi_view_index:
                view_records: List[Dict] = index_entry["views"]
                add_view_vector("text", vec, text, view_vectors, view_records, chunk_uid)

                if "intent" in active_views:
                    intent_text = chunk.get("intent") or ""
                    intent_vector = intent_vec if intent_vec is not None else None
                    if intent_vector is None and intent_rows is not None and intent_embs is not None and intent_text:
                        try:
                            match_idx = next(
                                idx for idx, row in enumerate(intent_rows) if row.get("text") == intent_text and row.get("doc_id") == doc_id
                            )
                            intent_vector = intent_elem(intent_embs, match_idx)
                        except StopIteration:
                            intent_vector = None
                    add_view_vector("intent", intent_vector, intent_text, view_vectors, view_records, chunk_uid)

                if "summary" in active_views:
                    summary_indices = global_indices[: min(2, len(global_indices))]
                    summary_vector = S[summary_indices].mean(axis=0) if summary_indices else None
                    add_view_vector("summary", summary_vector, summary_text, view_vectors, view_records, chunk_uid)

                if "keywords" in active_views:
                    keyword_set = set(keywords)
                    if keyword_set and global_indices:
                        weights = []
                        for idx in global_indices:
                            local_idx = global_to_local.get(idx, -1)
                            if local_idx < 0:
                                weights.append(1.0)
                                continue
                            sent_tokens = doc_tokens[local_idx]
                            weight = 1.0 + sum(1 for tok in sent_tokens if tok in keyword_set)
                            weights.append(weight)
                        keyword_vector = weighted_sentence_vector(S, global_indices, weights)
                    else:
                        keyword_vector = None
                    add_view_vector("keywords", keyword_vector, keywords_text, view_vectors, view_records, chunk_uid)

            index_rows.append(index_entry)
            global_chunk_id += 1

    if chunk_vecs:
        chunk_mat = np.vstack(chunk_vecs)
    else:
        chunk_mat = np.zeros((0, D), dtype=np.float32)
    if args.normalize and chunk_mat.size:
        chunk_mat = l2_normalize(chunk_mat)

    ensure_dir(args.out_embs)
    np.save(args.out_embs, chunk_mat)
    if args.out_anchor_embs:
        ensure_dir(args.out_anchor_embs)
        anchor_mat = np.vstack(anchor_vecs) if anchor_vecs else np.zeros((0, D), dtype=np.float32)
        if args.normalize and anchor_mat.size:
            anchor_mat = l2_normalize(anchor_mat)
        np.save(args.out_anchor_embs, anchor_mat)
        print(f"Saved anchor embeddings → {args.out_anchor_embs}")

    write_jsonl(args.out_chunks, chunk_meta)

    if args.multi_view_index:
        if view_vectors:
            view_mat = np.vstack(view_vectors)
            if args.normalize and view_mat.size:
                view_mat = l2_normalize(view_mat)
        else:
            view_mat = np.zeros((0, D), dtype=np.float32)
        ensure_dir(args.out_view_embs)
        np.save(args.out_view_embs, view_mat)
        write_jsonl(args.out_index, index_rows)
        print(
            f"Built {len(chunk_meta)} chunks; view embeddings shape = {view_mat.shape}"
        )
        print(f"Saved view index → {args.out_index}")
    else:
        print(f"Built {len(chunk_meta)} chunks; embeddings shape = {chunk_mat.shape}")
    print(f"Saved embeddings → {args.out_embs}")
    print(f"Saved chunk manifest → {args.out_chunks}")


if __name__ == "__main__":
    main()
