#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate retrieval quality with multi-view embeddings and hybrid scoring."""

import argparse
from argparse import BooleanOptionalAction
import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from dotenv import load_dotenv

from config import parse_views_arg, get_optimal_hybrid_weights

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_FAISS = False
    faiss = None

import google.generativeai as genai

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "has", "have", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "there", "this", "to", "was", "were",
    "which", "with",
}


@dataclass
class ChunkRecord:
    """Lightweight representation of a chunk used for scoring/metrics."""

    chunk_uid: str
    doc_id: str
    start_sent: int
    end_sent: int
    text: str
    intent: Optional[str]
    summary: Optional[str]
    keywords: List[str]
    coherence: float
    num_sentences: int


@dataclass
class ViewRecord:
    """Metadata describing a chunk view vector."""

    chunk_uid: str
    view: str
    vector_index: int


@dataclass
class RetrievalCandidate:
    """Dense or hybrid retrieval candidate for reranking."""

    chunk_uid: str
    score: float
    view: Optional[str] = None


def ensure_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> List[Dict]:
    rows: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    path = Path(path)
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for match in TOKEN_RE.findall(text.lower()):
        token = match.strip("'")
        if token and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def configure_genai() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)


def normalize_model(model_name: str) -> str:
    return model_name if model_name.startswith("models/") else f"models/{model_name}"


def extract_vec(resp) -> List[float]:
    if hasattr(resp, "embedding"):
        emb = resp.embedding
        if hasattr(emb, "values"):
            return list(emb.values)
        return list(emb)
    if isinstance(resp, dict) and "embedding" in resp:
        e = resp["embedding"]
        if isinstance(e, dict) and "values" in e:
            return list(e["values"])
        return list(e)
    raise ValueError("Cannot extract embedding")


def embed_queries(queries: Sequence[str], model: str, dim: int) -> np.ndarray:
    import time
    model = normalize_model(model)
    vecs = []
    for idx, q in enumerate(queries):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                resp = genai.embed_content(
                    model=model,
                    content=q,
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=dim,
                )
                v = extract_vec(resp)
                if len(v) != dim:
                    if len(v) > dim:
                        v = v[:dim]
                    else:
                        v = v + [0.0] * (dim - len(v))
                vecs.append(v)
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s, 8s
                    print(f"  Embedding query {idx+1}/{len(queries)} failed (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s... ({type(e).__name__})")
                    time.sleep(wait_time)
                else:
                    print(f"  ERROR: Query {idx+1}/{len(queries)} failed after {max_retries} attempts: {e}")
                    raise
    Q = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return Q / norms


def build_index(X: np.ndarray):
    if HAVE_FAISS:
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        return index
    return None


def dense_search(index, X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    if HAVE_FAISS and index is not None:
        sims, idxs = index.search(Q, X.shape[0])
        full = np.zeros((Q.shape[0], X.shape[0]), dtype=np.float32)
        for qi in range(Q.shape[0]):
            full[qi, idxs[qi]] = sims[qi]
        return full
    return Q @ X.T


class BM25Index:
    """Lightweight BM25 scorer for sparse hybrid retrieval."""

    def __init__(self, documents: Dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens: Dict[str, List[str]] = {}
        self.df: Counter = Counter()
        for doc_id, text in documents.items():
            tokens = tokenize(text)
            self.doc_tokens[doc_id] = tokens
            for term in set(tokens):
                self.df[term] += 1
        self.num_docs = max(len(self.doc_tokens), 1)
        lengths = [len(tokens) for tokens in self.doc_tokens.values()]
        self.avg_len = sum(lengths) / max(len(lengths), 1)

    def score(self, query: str) -> Dict[str, float]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return {}
        q_counts = Counter(q_tokens)
        scores: Dict[str, float] = defaultdict(float)
        for term, qf in q_counts.items():
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))
            for doc_id, tokens in self.doc_tokens.items():
                tf = tokens.count(term)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * len(tokens) / max(self.avg_len, 1e-8))
                score = idf * tf * (self.k1 + 1.0) / max(denom, 1e-8)
                scores[doc_id] += score
        return scores


def parse_hybrid_weights(arg: str) -> Tuple[float, float]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--hybrid-weights must be two comma-separated floats")
    try:
        dense, sparse = float(parts[0]), float(parts[1])
    except ValueError as exc:  # pragma: no cover - CLI validation
        raise ValueError("--hybrid-weights must be numeric") from exc
    return dense, sparse


def z_normalise(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=np.float32)
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-8:
        return {k: 0.0 for k in scores}
    return {k: (v - mean) / std for k, v in scores.items()}


def combine_dense_sparse(
    dense_scores: Dict[str, float],
    sparse_scores: Dict[str, float],
    dense_weight: float,
    sparse_weight: float,
) -> Dict[str, float]:
    zd = z_normalise(dense_scores)
    zs = z_normalise(sparse_scores)
    keys = set(zd) | set(zs)
    return {
        key: dense_weight * zd.get(key, 0.0) + sparse_weight * zs.get(key, 0.0)
        for key in keys
    }


def lexical_overlap(query: str, text: str) -> float:
    q_tokens = set(tokenize(query))
    t_tokens = set(tokenize(text))
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)


def rerank_candidates(
    query_text: str,
    candidates: List[Dict],
    method: str,
    bm25: Optional[BM25Index],
    min_chunk_sent: int,
) -> List[Dict]:
    if method == "mini-crossencoder":  # pragma: no cover - optional dependency
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query_text, cand["text"]) for cand in candidates]
            ce_scores = model.predict(pairs).tolist()
            for cand, ce_score in zip(candidates, ce_scores):
                cand["score"] = 0.7 * cand["score"] + 0.3 * ce_score
        except Exception:
            method = "lexical"

    if method == "lexical" and bm25 is not None:
        bm25_scores = bm25.score(query_text)
        intent_tokens = tokenize(query_text)
        for cand in candidates:
            bm25_boost = bm25_scores.get(cand["chunk_uid"], 0.0)
            overlap = lexical_overlap(query_text, cand.get("intent", "") or "")
            length_penalty = -0.05 if cand.get("num_sentences", 0) < min_chunk_sent else 0.0
            cand["score"] = cand["score"] + 0.3 * bm25_boost + 0.05 * overlap + length_penalty

    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def load_chunk_index(
    index_path: str | Path,
    selected_views: Tuple[str, ...],
) -> Tuple[Dict[str, ChunkRecord], List[ViewRecord]]:
    rows = read_jsonl(index_path)
    allowed = {v.lower() for v in selected_views}
    chunks: Dict[str, ChunkRecord] = {}
    view_records: List[ViewRecord] = []
    for row in rows:
        chunk_uid = row.get("chunk_uid")
        if not chunk_uid:
            continue
        coherence = float(row.get("coherence", 1.0))
        record = ChunkRecord(
            chunk_uid=chunk_uid,
            doc_id=row.get("doc_id", ""),
            start_sent=int(row.get("start_sent", 0)),
            end_sent=int(row.get("end_sent", row.get("start_sent", 0))),
            text=row.get("text", ""),
            intent=row.get("intent"),
            summary=row.get("summary"),
            keywords=row.get("keywords", []),
            coherence=coherence,
            num_sentences=int(row.get("num_sentences", 0)),
        )
        chunks[chunk_uid] = record
        for view_info in row.get("views", []):
            name = (view_info.get("view") or view_info.get("name") or "").lower()
            if allowed and name not in allowed:
                continue
            idx = view_info.get("vector_index")
            if idx is None:
                continue
            view_records.append(ViewRecord(chunk_uid=chunk_uid, view=name, vector_index=int(idx)))
    return chunks, view_records


def load_view_matrix(
    view_path: str | Path,
    view_records: List[ViewRecord],
) -> np.ndarray:
    view_embs = np.load(view_path)
    if not view_records:
        return np.zeros((0, view_embs.shape[1] if view_embs.ndim == 2 else 0), dtype=np.float32)
    indices = [rec.vector_index for rec in view_records]
    if max(indices, default=-1) >= view_embs.shape[0]:
        raise ValueError("View metadata references out-of-range vector index")
    return view_embs[indices]


def dense_chunk_scores(
    query_vecs: np.ndarray,
    view_matrix: np.ndarray,
    view_records: List[ViewRecord],
) -> List[Dict[str, Tuple[float, str]]]:
    if view_matrix.size == 0:
        return [defaultdict(lambda: (0.0, "")) for _ in range(query_vecs.shape[0])]
    index = build_index(view_matrix)
    sims = dense_search(index, view_matrix, query_vecs)
    per_query: List[Dict[str, Tuple[float, str]]] = []
    for q_idx in range(sims.shape[0]):
        scores: Dict[str, Tuple[float, str]] = {}
        for v_idx, rec in enumerate(view_records):
            score = float(sims[q_idx, v_idx])
            current = scores.get(rec.chunk_uid)
            if current is None or score > current[0]:
                scores[rec.chunk_uid] = (score, rec.view)
        per_query.append(scores)
    return per_query


def top_candidates(
    scores: Dict[str, Tuple[float, str]],
    topn: int,
) -> List[RetrievalCandidate]:
    sorted_items = sorted(scores.items(), key=lambda item: item[1][0], reverse=True)
    return [RetrievalCandidate(chunk_uid=k, score=v[0], view=v[1]) for k, v in sorted_items[:topn]]


def evaluate_doc_mode(
    rankings: List[List[str]],
    query_docs: Sequence[str],
    chunk_records: Dict[str, ChunkRecord],
    topk: int,
) -> Tuple[float, float, float]:
    hits1 = 0
    hitsk = 0
    rr = 0.0
    total = len(query_docs)
    for q_idx, ranked in enumerate(rankings):
        target_doc = query_docs[q_idx]
        found_rank = None
        for rank, chunk_uid in enumerate(ranked[:topk], start=1):
            if chunk_records.get(chunk_uid, ChunkRecord("", "", 0, 0, "", None, None, [], 1.0, 0)).doc_id == target_doc:
                found_rank = rank
                break
        if found_rank is None:
            continue
        if found_rank == 1:
            hits1 += 1
        if found_rank <= topk:
            hitsk += 1
            rr += 1.0 / found_rank
    denom = max(total, 1)
    return hits1 / denom, hitsk / denom, rr / denom


def evaluate_span_mode(
    rankings: List[List[str]],
    query_meta: List[Dict],
    chunk_records: Dict[str, ChunkRecord],
    spans: Dict[Tuple[str, int], Dict],
    topk: int,
    compute_extended_metrics: bool = True,
) -> Tuple[float, float, float, int, Optional[Dict[str, float]]]:
    """Evaluate span-mode retrieval with optional extended metrics.

    Args:
        rankings: List of ranked chunk UIDs per query
        query_meta: Query metadata
        chunk_records: Chunk metadata mapping
        spans: Ground truth spans
        topk: Number of top results to consider
        compute_extended_metrics: Whether to compute completeness, redundancy, diversity

    Returns:
        Tuple of (R@1, R@k, MRR, num_evaluated, extended_metrics_dict)
    """
    hits1 = 0
    hitsk = 0
    rr = 0.0
    evaluated = 0

    # NEW: Extended metrics tracking
    completeness_scores = []
    redundancy_scores = []
    diversity_scores = []
    efficiency_scores = []

    for qi, row in enumerate(query_meta):
        key = (row.get("doc_id"), int(row.get("query_id", qi)))
        span = spans.get(key)
        if span is None or not span.get("answerable", True):
            continue
        evaluated += 1
        candidates = rankings[qi][:topk]
        true_chunk = None
        for chunk_uid, record in chunk_records.items():
            if record.doc_id != row.get("doc_id"):
                continue
            if int(record.start_sent) <= int(span.get("start_sent", 0)) and int(record.end_sent) >= int(span.get("end_sent", 0)):
                true_chunk = chunk_uid
                break
        if true_chunk is None:
            continue

        # Standard metrics
        if candidates and candidates[0] == true_chunk:
            hits1 += 1
            hitsk += 1
            rr += 1.0
        elif true_chunk in candidates:
            hitsk += 1
            rank = candidates.index(true_chunk) + 1
            rr += 1.0 / rank

        # NEW: Extended metrics computation
        if compute_extended_metrics and candidates:
            # Completeness: how much of the answer is in top-k
            completeness = compute_answer_completeness(candidates, span, chunk_records, topk=topk)
            completeness_scores.append(completeness)

            # Redundancy: information overlap in top-k
            redundancy = compute_redundancy(candidates, chunk_records, topk=min(topk, 5))
            redundancy_scores.append(redundancy)

            # Diversity: coverage of different aspects
            diversity = compute_diversity(candidates, chunk_records, topk=min(topk, 5))
            diversity_scores.append(diversity)

            # Efficiency: relevant tokens / total tokens
            efficiency = compute_context_efficiency(candidates, span, chunk_records, topk=topk)
            efficiency_scores.append(efficiency)

    denom = max(evaluated, 1)

    # Compute average extended metrics
    extended_metrics = None
    if compute_extended_metrics and completeness_scores:
        extended_metrics = {
            "completeness": sum(completeness_scores) / len(completeness_scores),
            "redundancy": sum(redundancy_scores) / len(redundancy_scores),
            "diversity": sum(diversity_scores) / len(diversity_scores),
            "efficiency": sum(efficiency_scores) / len(efficiency_scores),
        }

    return hits1 / denom, hitsk / denom, rr / denom, evaluated, extended_metrics


def compute_answer_coverage(
    spans: Dict[Tuple[str, int], Dict],
    chunk_records: Dict[str, ChunkRecord],
) -> float:
    """Compute percentage of answer spans fully contained within chunks."""
    if not spans:
        return 0.0
    covered = 0
    total = 0
    chunks_by_doc: Dict[str, List[ChunkRecord]] = defaultdict(list)
    for record in chunk_records.values():
        chunks_by_doc[record.doc_id].append(record)
    for records in chunks_by_doc.values():
        records.sort(key=lambda r: r.start_sent)
    for (doc_id, _), span in spans.items():
        if not span.get("answerable", True):
            continue
        total += 1
        doc_chunks = chunks_by_doc.get(doc_id, [])
        for record in doc_chunks:
            if record.start_sent <= int(span.get("start_sent", 0)) and record.end_sent >= int(span.get("end_sent", 0)):
                covered += 1
                break
    return covered / max(total, 1)


def compute_answer_completeness(
    retrieved_chunks: List[str],
    query_span: Dict,
    chunk_records: Dict[str, ChunkRecord],
    topk: int = 5,
) -> float:
    """Compute completeness: does retrieved chunk contain full answer context?

    Measures whether the top-k retrieved chunks provide sufficient context
    to answer the query, considering partial overlap and context windows.

    Args:
        retrieved_chunks: List of chunk UIDs in ranked order
        query_span: Ground truth span with start_sent, end_sent
        chunk_records: Mapping of chunk UID to metadata
        topk: Number of top chunks to consider

    Returns:
        Completeness score [0, 1]: 1.0 = full answer contained, 0.0 = no overlap
    """
    if not query_span or "start_sent" not in query_span:
        return 0.0

    answer_start = int(query_span["start_sent"])
    answer_end = int(query_span["end_sent"])
    answer_length = answer_end - answer_start + 1

    # Check top-k retrieved chunks
    best_overlap = 0.0
    for chunk_uid in retrieved_chunks[:topk]:
        if chunk_uid not in chunk_records:
            continue

        chunk = chunk_records[chunk_uid]

        # Compute overlap with answer span
        overlap_start = max(chunk.start_sent, answer_start)
        overlap_end = min(chunk.end_sent, answer_end)
        overlap_length = max(0, overlap_end - overlap_start + 1)

        # Completeness = overlap / answer_length
        completeness = overlap_length / max(answer_length, 1)
        best_overlap = max(best_overlap, completeness)

        if best_overlap >= 1.0:  # Full answer found
            break

    return best_overlap


def compute_redundancy(
    retrieved_chunks: List[str],
    chunk_records: Dict[str, ChunkRecord],
    topk: int = 5,
) -> float:
    """Compute redundancy: information overlap between top-k chunks.

    Lower redundancy is better - indicates diverse, complementary results.

    Args:
        retrieved_chunks: List of chunk UIDs in ranked order
        chunk_records: Mapping of chunk UID to metadata
        topk: Number of top chunks to analyze

    Returns:
        Redundancy score [0, 1]: 0.0 = no overlap, 1.0 = completely redundant
    """
    if topk < 2:
        return 0.0

    # Extract text for top-k chunks
    chunk_texts = []
    for chunk_uid in retrieved_chunks[:topk]:
        if chunk_uid in chunk_records:
            chunk_texts.append(chunk_records[chunk_uid].text)

    if len(chunk_texts) < 2:
        return 0.0

    # Tokenize texts (simple whitespace + lowercase)
    chunk_tokens = []
    for text in chunk_texts:
        tokens = set(TOKEN_RE.findall(text.lower())) - STOPWORDS
        chunk_tokens.append(tokens)

    # Compute pairwise Jaccard similarity
    similarities = []
    for i in range(len(chunk_tokens)):
        for j in range(i + 1, len(chunk_tokens)):
            if not chunk_tokens[i] or not chunk_tokens[j]:
                continue

            intersection = chunk_tokens[i] & chunk_tokens[j]
            union = chunk_tokens[i] | chunk_tokens[j]

            if union:
                sim = len(intersection) / len(union)
                similarities.append(sim)

    # Average pairwise similarity = redundancy
    if not similarities:
        return 0.0

    return sum(similarities) / len(similarities)


def compute_diversity(
    retrieved_chunks: List[str],
    chunk_records: Dict[str, ChunkRecord],
    topk: int = 5,
) -> float:
    """Compute diversity: coverage of different document aspects in top-k.

    Higher diversity is better - indicates comprehensive coverage.

    Args:
        retrieved_chunks: List of chunk UIDs in ranked order
        chunk_records: Mapping of chunk UID to metadata
        topk: Number of top chunks to analyze

    Returns:
        Diversity score [0, 1]: 0.0 = redundant, 1.0 = highly diverse
    """
    # Diversity is inverse of redundancy
    redundancy = compute_redundancy(retrieved_chunks, chunk_records, topk)
    return 1.0 - redundancy


def compute_context_efficiency(
    retrieved_chunks: List[str],
    query_span: Dict,
    chunk_records: Dict[str, ChunkRecord],
    topk: int = 5,
) -> float:
    """Compute context efficiency: ratio of relevant tokens to total tokens.

    Measures how much of the retrieved context is actually needed.

    Args:
        retrieved_chunks: List of chunk UIDs in ranked order
        query_span: Ground truth span
        chunk_records: Mapping of chunk UID to metadata
        topk: Number of top chunks to consider

    Returns:
        Efficiency score [0, 1]: 1.0 = perfect efficiency, 0.0 = all irrelevant
    """
    if not query_span or "start_sent" not in query_span:
        return 0.0

    answer_start = int(query_span["start_sent"])
    answer_end = int(query_span["end_sent"])

    # Count tokens in answer span and retrieved chunks
    answer_tokens = 0
    retrieved_tokens = 0

    for chunk_uid in retrieved_chunks[:topk]:
        if chunk_uid not in chunk_records:
            continue

        chunk = chunk_records[chunk_uid]
        chunk_length = chunk.end_sent - chunk.start_sent + 1
        retrieved_tokens += chunk_length

        # Count overlap with answer
        overlap_start = max(chunk.start_sent, answer_start)
        overlap_end = min(chunk.end_sent, answer_end)
        overlap_length = max(0, overlap_end - overlap_start + 1)
        answer_tokens += overlap_length

    if retrieved_tokens == 0:
        return 0.0

    return answer_tokens / retrieved_tokens


def aggregate_context(chunks: Dict[str, ChunkRecord], chunk_ids: List[str], topk: int) -> str:
    selected = [chunks[cid].text for cid in chunk_ids[:topk] if cid in chunks]
    return "\n\n".join(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IDC retrieval performance")
    parser.add_argument("--chunk-embs", type=str, default="out/chunk_embs.npy")
    parser.add_argument("--chunks", type=str, default="out/chunks.jsonl")
    parser.add_argument("--view-embs", type=str, default="out/chunk_view_embs.npy")
    parser.add_argument("--index", type=str, default="out/chunks.index.jsonl")
    parser.add_argument("--queries", type=str, required=True)
    parser.add_argument("--embedder", type=str, default="gemini-embedding-001")
    parser.add_argument("--dim", type=int, default=1536)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--context-topk", type=int, default=3)
    parser.add_argument("--contexts-out", type=str)
    parser.add_argument("--mode", choices=["doc", "span"], default="doc")
    parser.add_argument("--spans", type=str)
    parser.add_argument("--multi-view-index", action=BooleanOptionalAction, default=True)
    parser.add_argument("--views", type=str, default="text,intent,summary,keywords")
    parser.add_argument("--hybrid-retrieval", action=BooleanOptionalAction, default=True)
    parser.add_argument("--hybrid-weights", type=str, default="0.6,0.4")
    parser.add_argument("--auto-hybrid-weights", action="store_true",
                        help="Automatically tune hybrid weights based on document length (overrides --hybrid-weights)")
    parser.add_argument("--reranker", choices=["none", "lexical", "mini-crossencoder"], default="lexical")
    parser.add_argument("--eval-answer-coverage", action=BooleanOptionalAction, default=True)
    parser.add_argument("--eval-coherence", action=BooleanOptionalAction, default=True)
    parser.add_argument("--csv-out", type=str)
    parser.add_argument("--coherence-weight", type=float, default=0.10)
    parser.add_argument("--min-chunk-sent", type=int, default=2)
    args = parser.parse_args()

    selected_views = parse_views_arg(args.views)

    # Determine hybrid weights (auto or manual)
    if args.auto_hybrid_weights:
        # Load chunks to compute document length for auto weights
        if args.multi_view_index:
            # Load from index to get chunk count
            chunk_index_path = Path(args.index)
            if chunk_index_path.exists():
                with chunk_index_path.open("r", encoding="utf-8") as f:
                    temp_chunks = [json.loads(line) for line in f if line.strip()]
                doc_length = sum(row.get("num_sentences", 1) for row in temp_chunks)
            else:
                doc_length = 500  # Default to long doc weights if index not found
        else:
            chunk_rows_temp = read_jsonl(args.chunks)
            doc_length = sum(row.get("num_sentences", 1) for row in chunk_rows_temp)

        dense_weight, sparse_weight = get_optimal_hybrid_weights(doc_length)
        print(f"Auto-tuned hybrid weights based on doc_length={doc_length} sentences")
    else:
        dense_weight, sparse_weight = parse_hybrid_weights(args.hybrid_weights)

    print(
        "Config:",
        f"Views={','.join(selected_views)}",
        f"Hybrid={args.hybrid_retrieval} (w={dense_weight:.2f}/{sparse_weight:.2f})",
        f"Reranker={args.reranker}",
        f"CoherenceWeight={args.coherence_weight}",
        f"MinChunkSent={args.min_chunk_sent}",
    )

    queries = read_jsonl(args.queries)
    query_texts = [row["text"] for row in queries]
    query_docs = [row.get("doc_id", "") for row in queries]

    if args.multi_view_index:
        chunk_records, view_records = load_chunk_index(args.index, selected_views)
        view_matrix = load_view_matrix(args.view_embs, view_records)
    else:
        chunk_rows = read_jsonl(args.chunks)
        chunk_records = {
            row["chunk_uid"]: ChunkRecord(
                chunk_uid=row["chunk_uid"],
                doc_id=row.get("doc_id", ""),
                start_sent=int(row.get("start_sent", 0)),
                end_sent=int(row.get("end_sent", row.get("start_sent", 0))),
                text=row.get("text", ""),
                intent=row.get("intent"),
                summary=row.get("summary"),
                keywords=row.get("keywords", []),
                coherence=float(row.get("coherence", 1.0)),
                num_sentences=int(row.get("num_sentences", 0)),
            )
            for row in chunk_rows
        }
        array = np.load(args.chunk_embs)
        view_records = [ViewRecord(chunk_uid=cid, view="text", vector_index=i) for i, cid in enumerate(chunk_records)]
        view_matrix = array

    configure_genai()
    query_vectors = embed_queries(query_texts, args.embedder, args.dim)

    dense_per_query = dense_chunk_scores(query_vectors, view_matrix, view_records)

    bm25_index = BM25Index({uid: record.text for uid, record in chunk_records.items()}) if args.hybrid_retrieval else None

    hybrid_rankings: List[List[str]] = []
    rerank_inputs: List[List[Dict]] = []
    for q_idx, dense_scores_dict in enumerate(dense_per_query):
        dense_scores = {cid: score for cid, (score, _) in dense_scores_dict.items()}
        if args.hybrid_retrieval and bm25_index is not None:
            sparse_scores = bm25_index.score(query_texts[q_idx])
            combined = combine_dense_sparse(dense_scores, sparse_scores, dense_weight, sparse_weight)
        else:
            combined = dense_scores
        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        hybrid_rankings.append([cid for cid, _ in ranked])
        top_for_rerank = ranked[: max(args.topk * 3, 50)]
        candidate_payloads = []
        for cid, score in top_for_rerank:
            record = chunk_records[cid]
            candidate_payloads.append(
                {
                    "chunk_uid": cid,
                    "score": score,
                    "text": record.text,
                    "intent": record.intent,
                    "num_sentences": record.num_sentences,
                }
            )
        rerank_inputs.append(candidate_payloads)

    final_rankings: List[List[str]] = []
    for q_idx, candidates in enumerate(rerank_inputs):
        if args.reranker == "none" or not candidates:
            final = sorted(candidates, key=lambda x: x["score"], reverse=True)
        else:
            final = rerank_candidates(query_texts[q_idx], candidates, args.reranker, bm25_index, args.min_chunk_sent)
        final_rankings.append([cand["chunk_uid"] for cand in final])

    if args.mode == "doc":
        hit1, hitk, mrr = evaluate_doc_mode(final_rankings, query_docs, chunk_records, args.topk)
        print(f"Doc-hit@1: {hit1:.3f}")
        print(f"Doc-hit@{args.topk}: {hitk:.3f}")
        print(f"MRR(doc): {mrr:.3f}")
        evaluated_spans = 0
    else:
        if not args.spans:
            raise SystemExit("--spans required in span mode")
        span_rows = read_jsonl(args.spans)
        span_by_key = {}
        for span in span_rows:
            key = (span.get("doc_id"), int(span.get("query_id", 0)))
            span_by_key[key] = span
        hit1, hitk, mrr, evaluated_spans, extended_metrics = evaluate_span_mode(
            final_rankings,
            queries,
            chunk_records,
            span_by_key,
            args.topk,
            compute_extended_metrics=True,
        )
        print(f"Span-hit@1: {hit1:.3f}")
        print(f"Span-hit@{args.topk}: {hitk:.3f}")
        print(f"MRR(span): {mrr:.3f}")
        print(f"Evaluated spans: {evaluated_spans}")

        # NEW: Display extended metrics
        if extended_metrics:
            print("\n📊 Extended Metrics:")
            print(f"  Completeness: {extended_metrics['completeness']:.3f}  (answer coverage in top-{args.topk})")
            print(f"  Redundancy:   {extended_metrics['redundancy']:.3f}  (lower is better - less overlap)")
            print(f"  Diversity:    {extended_metrics['diversity']:.3f}  (higher is better - more aspects)")
            print(f"  Efficiency:   {extended_metrics['efficiency']:.3f}  (relevant/total tokens ratio)")

    if args.eval_answer_coverage and args.spans:
        span_rows = read_jsonl(args.spans)
        span_by_key = {(row.get("doc_id"), int(row.get("query_id", 0))): row for row in span_rows}
        coverage = compute_answer_coverage(span_by_key, chunk_records)
        print(f"Answer coverage: {coverage:.3f}")
    else:
        coverage = 0.0

    coherence_values = [record.coherence for record in chunk_records.values() if not math.isnan(record.coherence)]
    if args.eval_coherence and coherence_values:
        mean_coh = statistics.mean(coherence_values)
        std_coh = statistics.pstdev(coherence_values)
        print(f"Avg coherence: {mean_coh:.3f} ± {std_coh:.3f}")
    else:
        mean_coh = std_coh = 0.0

    if args.contexts_out:
        contexts = []
        for q_idx, ranked in enumerate(final_rankings):
            context_text = aggregate_context(chunk_records, ranked, args.context_topk)
            contexts.append({"query_id": int(queries[q_idx].get("query_id", q_idx)), "context": context_text})
        write_jsonl(args.contexts_out, contexts)
        print(f"Saved aggregated contexts → {args.contexts_out}")

    if args.csv_out:
        ensure_dir(args.csv_out)
        with Path(args.csv_out).open("w", encoding="utf-8", newline="") as handle:
            # NEW: Extended fieldnames for new metrics
            fieldnames = [
                "Mode",
                "TopK",
                "Hit@1",
                "Hit@K",
                "MRR",
                "AnswerCoverage",
                "AvgCoherence",
                "Hybrid",
                "Views",
                "Reranker",
            ]

            # Add extended metrics if available
            if extended_metrics:
                fieldnames.extend(["Completeness", "Redundancy", "Diversity", "Efficiency"])

            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            row_data = {
                "Mode": args.mode,
                "TopK": args.topk,
                "Hit@1": f"{hit1:.4f}",
                "Hit@K": f"{hitk:.4f}",
                "MRR": f"{mrr:.4f}",
                "AnswerCoverage": f"{coverage:.4f}",
                "AvgCoherence": f"{mean_coh:.4f}",
                "Hybrid": args.hybrid_retrieval,
                "Views": ",".join(selected_views),
                "Reranker": args.reranker,
            }

            # Add extended metrics to CSV if available
            if extended_metrics:
                row_data.update({
                    "Completeness": f"{extended_metrics['completeness']:.4f}",
                    "Redundancy": f"{extended_metrics['redundancy']:.4f}",
                    "Diversity": f"{extended_metrics['diversity']:.4f}",
                    "Efficiency": f"{extended_metrics['efficiency']:.4f}",
                })

            writer.writerow(row_data)
        print(f"Saved metrics CSV → {args.csv_out}")


if __name__ == "__main__":
    main()
