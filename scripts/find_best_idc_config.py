#!/usr/bin/env python3
"""
Hybrid-aware IDC parameter search.

Finds the best IDC parameters for a given dataset by evaluating each
parameter combination with the SAME hybrid retrieval (dense + BM25)
that the final pipeline uses — unlike auto_tune.py which uses dense-only.

All computation is local: reuses existing sentence/intent embeddings.
No API calls are made.

Usage:
    # Single-document mode:
    python scripts/find_best_idc_config.py --doc-dir out/Normans --target-r1 0.917

    # Multi-document mode (e.g., SQuAD 2-doc, Qasper):
    python scripts/find_best_idc_config.py --doc-ids-file data/squad/doc_ids.txt \
        --base-out-dir out --report-dir out/squad --target-r1 0.689
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import existing code (unchanged)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from idc_core import IDCParams, segment_corpus, read_jsonl
from build_chunks import (
    compute_intent_weighted_embedding,
    group_sentences_by_doc,
)
from eval_retrieval import BM25Index, combine_dense_sparse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_file(directory: Path, prefix: str, suffix: str) -> Path:
    """Find a file matching prefix*.suffix in directory."""
    candidates = sorted(directory.glob(f"{prefix}*{suffix}"))
    if not candidates:
        raise FileNotFoundError(
            f"No file matching '{prefix}*{suffix}' in {directory}"
        )
    return candidates[0]


def load_doc_data(
    doc_dir: Path,
) -> Tuple[List[Dict], np.ndarray, List[Dict], np.ndarray, List[Dict]]:
    """Load all preprocessed data for a single document directory.

    Returns (sent_rows, sent_embs, intent_rows, intent_embs, spans).
    """
    sentences_path = doc_dir / "sentences.jsonl"
    sent_embs_path = find_file(doc_dir, "sentence_embs", ".npy")
    intents_flat_path = doc_dir / "intents.flat.jsonl"
    intent_embs_path = find_file(doc_dir, "intent_embs", ".npy")

    spans_path = doc_dir / "gt_spans.jsonl"
    if not spans_path.exists():
        spans_path = doc_dir / "tune_spans.jsonl"
    if not spans_path.exists():
        raise FileNotFoundError(f"No gt_spans.jsonl or tune_spans.jsonl in {doc_dir}")

    sent_rows = read_jsonl(sentences_path)
    sent_embs = np.load(sent_embs_path)
    intent_rows = read_jsonl(intents_flat_path)
    intent_embs = np.load(intent_embs_path)
    spans = read_jsonl(spans_path)

    return sent_rows, sent_embs, intent_rows, intent_embs, spans


def build_chunk_embeddings(
    segments: List[Dict],
    sent_rows: List[Dict],
    S: np.ndarray,
    intent_rows: List[Dict],
    intent_embs: np.ndarray,
) -> Tuple[np.ndarray, List[Dict]]:
    """Build intent-weighted chunk embeddings from segments.

    Returns (chunk_embedding_matrix, chunk_metadata_list).
    """
    by_doc = group_sentences_by_doc(sent_rows)
    chunk_vecs: List[np.ndarray] = []
    chunk_meta: List[Dict] = []
    D = S.shape[1]

    for doc in segments:
        doc_id = doc["doc_id"]
        mapping = by_doc.get(doc_id, [])
        if not mapping:
            continue

        sid_to_global = {sid: gidx for sid, gidx in mapping}
        ordered_rows = [sent_rows[idx] for _, idx in mapping]
        doc_texts = [row.get("text", "") for row in ordered_rows]
        global_to_local = {gidx: pos for pos, (_, gidx) in enumerate(mapping)}

        chunks_list = doc.get("chunks", [])
        all_chunk_indices = []
        for chunk in chunks_list:
            start = int(chunk.get("start_sent", 0))
            end = int(chunk.get("end_sent", start))
            indices = [sid_to_global.get(i) for i in range(start, end + 1)]
            all_chunk_indices.append([i for i in indices if i is not None])

        for local_cid, chunk in enumerate(chunks_list):
            global_indices = all_chunk_indices[local_cid]
            chunk_start = int(chunk.get("start_sent", 0))
            chunk_end = int(chunk.get("end_sent", chunk_start))

            if not global_indices:
                vec = np.zeros((D,), dtype=np.float32)
            else:
                idx_array = np.array(global_indices)
                intent_text = chunk.get("intent")
                if intent_text and intent_rows and intent_embs is not None:
                    vec, _, _ = compute_intent_weighted_embedding(
                        S, idx_array.tolist(), intent_text, intent_rows, intent_embs,
                    )
                else:
                    vec = S[idx_array].mean(axis=0)

            chunk_uid = f"{doc_id}::c{local_cid + 1}"
            local_positions = [
                global_to_local[idx]
                for idx in global_indices
                if idx in global_to_local
            ]
            text = " ".join(doc_texts[pos] for pos in local_positions)

            chunk_meta.append({
                "chunk_uid": chunk_uid,
                "doc_id": doc_id,
                "start_sent": chunk_start,
                "end_sent": chunk_end,
                "num_sentences": chunk_end - chunk_start + 1,
                "text": text,
                "intent": chunk.get("intent"),
            })
            chunk_vecs.append(vec.astype(np.float32))

    if not chunk_vecs:
        return np.zeros((0, D), dtype=np.float32), chunk_meta
    return np.vstack(chunk_vecs), chunk_meta


def evaluate_hybrid(
    chunk_embs: np.ndarray,
    chunk_meta: List[Dict],
    intent_embs: np.ndarray,
    intent_rows: List[Dict],
    spans: List[Dict],
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    topk: int = 5,
) -> Dict[str, float]:
    """Evaluate R@1, R@K, MRR using hybrid retrieval (dense + BM25).

    Mirrors the evaluation in eval_retrieval.py's span-mode pipeline.
    """
    if chunk_embs.shape[0] == 0 or intent_embs.shape[0] == 0:
        return {"R1": 0.0, "RK": 0.0, "MRR": 0.0, "evaluated": 0}

    # Normalize for dense similarity
    def l2n(a: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(a, axis=1, keepdims=True)
        return a / np.maximum(n, 1e-8)

    Xn = l2n(chunk_embs)
    Qn = l2n(intent_embs)

    # Dense scores: (n_queries, n_chunks)
    dense_sims = Qn @ Xn.T

    # Build BM25 index from chunk text
    bm25_docs = {
        meta["chunk_uid"]: meta["text"] for meta in chunk_meta
    }
    bm25 = BM25Index(bm25_docs)

    # Build span lookup
    spans_by_key = {}
    for s in spans:
        try:
            spans_by_key[(s["doc_id"], int(s["query_id"]))] = s
        except (KeyError, ValueError):
            continue

    # Chunk lookup
    chunks_by_doc: Dict[str, List[int]] = {}
    for idx, meta in enumerate(chunk_meta):
        chunks_by_doc.setdefault(meta["doc_id"], []).append(idx)

    hits1 = hits_k = 0
    rr = 0.0
    n_eval = 0

    for qi, q_row in enumerate(intent_rows):
        key = (q_row.get("doc_id"), int(q_row.get("query_id", 0)))
        span = spans_by_key.get(key)
        if span is None or not span.get("answerable", True):
            continue

        doc_id = span["doc_id"]
        doc_chunk_indices = chunks_by_doc.get(doc_id, [])
        if not doc_chunk_indices:
            continue

        s0 = int(span["start_sent"])
        s1 = int(span["end_sent"])

        # Find the true chunk that fully contains this span
        true_idx = None
        for local_idx, global_idx in enumerate(doc_chunk_indices):
            meta = chunk_meta[global_idx]
            if int(meta["start_sent"]) <= s0 and int(meta["end_sent"]) >= s1:
                true_idx = global_idx
                break

        if true_idx is None:
            n_eval += 1
            continue

        # Dense scores for this query against doc chunks only
        doc_chunk_uids = {chunk_meta[j]["chunk_uid"] for j in doc_chunk_indices}
        dense_scores = {
            chunk_meta[j]["chunk_uid"]: float(dense_sims[qi, j])
            for j in doc_chunk_indices
        }

        # BM25 scores filtered to same document's chunks
        query_text = q_row.get("text", "")
        all_sparse = bm25.score(query_text)
        sparse_scores = {k: v for k, v in all_sparse.items() if k in doc_chunk_uids}

        # Hybrid combine with z-normalization (both dicts scoped to same doc)
        combined = combine_dense_sparse(
            dense_scores, sparse_scores, dense_weight, sparse_weight
        )

        # Rank by combined score
        ranked_uids = sorted(combined.keys(), key=lambda k: combined[k], reverse=True)
        top_uids = ranked_uids[:topk]

        true_uid = chunk_meta[true_idx]["chunk_uid"]

        n_eval += 1
        if top_uids and top_uids[0] == true_uid:
            hits1 += 1
        if true_uid in top_uids:
            hits_k += 1
            rank = top_uids.index(true_uid) + 1
            rr += 1.0 / rank

    n = max(n_eval, 1)
    return {
        "R1": hits1 / n,
        "RK": hits_k / n,
        "MRR": rr / n,
        "evaluated": n_eval,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find best IDC parameters using hybrid retrieval evaluation"
    )
    # Single-doc mode
    parser.add_argument(
        "--doc-dir",
        help="Path to single-doc output directory (e.g., out/Normans)"
    )
    # Multi-doc mode
    parser.add_argument(
        "--doc-ids-file",
        help="Path to file with doc IDs, one per line (e.g., data/squad/doc_ids.txt)"
    )
    parser.add_argument(
        "--base-out-dir", default="out",
        help="Base output directory containing per-doc subdirs (default: out)"
    )
    parser.add_argument(
        "--report-dir",
        help="Directory to save the report (defaults to --doc-dir or --base-out-dir)"
    )

    parser.add_argument("--target-r1", type=float, default=0.0,
                        help="Target R@1 to beat (for reporting)")
    parser.add_argument("--dense-weight", type=float, default=0.6)
    parser.add_argument("--sparse-weight", type=float, default=0.4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--min-len", type=int, default=3)

    # Grid overrides
    parser.add_argument("--lambda-values", nargs="+", type=float,
                        default=[0.0001, 0.0005, 0.001, 0.002, 0.005,
                                 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--bp-values", nargs="+", type=float,
                        default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0])
    parser.add_argument("--max-len-values", nargs="+", type=int,
                        default=[8, 10, 12, 16, 20, 25, 30])
    parser.add_argument("--cw-values", nargs="+", type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    args = parser.parse_args()

    if not args.doc_dir and not args.doc_ids_file:
        parser.error("Either --doc-dir or --doc-ids-file is required")

    # ---- Load preprocessed data ----
    if args.doc_ids_file:
        # Multi-document mode
        doc_ids_path = Path(args.doc_ids_file)
        base_out = Path(args.base_out_dir)
        doc_ids = [
            line.strip()
            for line in doc_ids_path.read_text().splitlines()
            if line.strip()
        ]
        print(f"Multi-doc mode: {len(doc_ids)} documents from {doc_ids_path}")

        all_sent_rows: List[Dict] = []
        all_S: List[np.ndarray] = []
        all_intent_rows: List[Dict] = []
        all_intent_embs: List[np.ndarray] = []
        all_spans: List[Dict] = []

        for doc_id in doc_ids:
            doc_path = base_out / doc_id
            print(f"  Loading {doc_id} from {doc_path}...")
            sr, se, ir, ie, sp = load_doc_data(doc_path)
            all_sent_rows.extend(sr)
            all_S.append(se)
            all_intent_rows.extend(ir)
            all_intent_embs.append(ie)
            all_spans.extend(sp)

        sent_rows = all_sent_rows
        S = np.vstack(all_S)
        intent_rows = all_intent_rows
        intent_embs = np.vstack(all_intent_embs)
        spans = all_spans
        report_dir = Path(args.report_dir) if args.report_dir else base_out
    else:
        # Single-document mode (existing behavior)
        doc_dir = Path(args.doc_dir)
        print(f"Loading data from {doc_dir}...")
        sr, se, ir, ie, sp = load_doc_data(doc_dir)
        sent_rows, S, intent_rows, intent_embs, spans = sr, se, ir, ie, sp
        report_dir = Path(args.report_dir) if args.report_dir else doc_dir

    print(f"  Sentences: {len(sent_rows)}")
    print(f"  Intents: {len(intent_rows)}")
    print(f"  Gold spans: {len(spans)}")
    print(f"  Embedding dim: {S.shape[1]}")
    print(f"  Hybrid weights: dense={args.dense_weight}, sparse={args.sparse_weight}")

    # ---- Grid search ----
    total = (
        len(args.lambda_values)
        * len(args.bp_values)
        * len(args.max_len_values)
        * len(args.cw_values)
    )
    print(f"\nGrid: {len(args.lambda_values)} lambda x {len(args.bp_values)} bp "
          f"x {len(args.max_len_values)} max_len x {len(args.cw_values)} cw "
          f"= {total} combinations")

    best_r1 = -1.0
    best_mrr = -1.0
    best_params = None
    results: List[Dict] = []
    t0 = time.time()
    count = 0

    for lam in args.lambda_values:
        for bp in args.bp_values:
            for max_len in args.max_len_values:
                for cw in args.cw_values:
                    count += 1
                    params = IDCParams(
                        lam=lam,
                        max_len=max_len,
                        min_len=args.min_len,
                        boundary_penalty=bp,
                        coherence_weight=cw,
                    )

                    # Step 1: Segment (in-memory, no redundant I/O)
                    try:
                        seg_results = segment_corpus(
                            sentence_rows=sent_rows,
                            sentence_vectors=S,
                            intent_rows=intent_rows,
                            intent_vectors=intent_embs,
                            params=params,
                        )
                    except Exception as exc:
                        results.append({
                            "lambda": lam, "bp": bp, "max_len": max_len,
                            "cw": cw, "status": f"SEGMENT_FAIL: {exc}",
                        })
                        continue

                    segments = [
                        {"doc_id": r.doc_id, "chunks": r.chunks}
                        for r in seg_results
                    ]

                    # Step 2: Build chunk embeddings
                    try:
                        chunk_embs, chunk_meta = build_chunk_embeddings(
                            segments, sent_rows, S, intent_rows, intent_embs,
                        )
                    except Exception as exc:
                        results.append({
                            "lambda": lam, "bp": bp, "max_len": max_len,
                            "cw": cw, "status": f"EMBED_FAIL: {exc}",
                        })
                        continue

                    n_chunks = chunk_embs.shape[0]
                    avg_len = (
                        np.mean([m["num_sentences"] for m in chunk_meta])
                        if chunk_meta else 0.0
                    )

                    # Step 3: Hybrid evaluation
                    metrics = evaluate_hybrid(
                        chunk_embs, chunk_meta,
                        intent_embs, intent_rows, spans,
                        dense_weight=args.dense_weight,
                        sparse_weight=args.sparse_weight,
                        topk=args.topk,
                    )

                    r1 = metrics["R1"]
                    rk = metrics["RK"]
                    mrr = metrics["MRR"]

                    entry = {
                        "lambda": lam, "bp": bp, "max_len": max_len, "cw": cw,
                        "R1": r1, "RK": rk, "MRR": mrr,
                        "n_chunks": n_chunks, "avg_len": float(avg_len),
                        "evaluated": metrics["evaluated"],
                        "status": "OK",
                    }
                    results.append(entry)

                    # Track best by combined (R1 primary, MRR secondary)
                    if r1 > best_r1 or (r1 == best_r1 and mrr > best_mrr):
                        best_r1 = r1
                        best_mrr = mrr
                        best_params = entry.copy()
                        print(
                            f"  [{count}/{total}] NEW BEST: "
                            f"R@1={r1:.3f} R@5={rk:.3f} MRR={mrr:.3f} "
                            f"| λ={lam} bp={bp} L={max_len} cw={cw} "
                            f"| {n_chunks} chunks, avg_len={avg_len:.1f}"
                        )

                    if count % 100 == 0:
                        elapsed = time.time() - t0
                        rate = count / max(elapsed, 1e-6)
                        remaining = (total - count) / rate
                        print(
                            f"  [{count}/{total}] "
                            f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining "
                            f"| best R@1={best_r1:.3f}"
                        )

    elapsed = time.time() - t0

    # ---- Report ----
    print(f"\n{'=' * 60}")
    print(f"SEARCH COMPLETE: {total} combinations in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    if best_params:
        print(f"\nBEST CONFIG:")
        print(f"  lambda:           {best_params['lambda']}")
        print(f"  boundary_penalty: {best_params['bp']}")
        print(f"  max_len:          {best_params['max_len']}")
        print(f"  coherence_weight: {best_params['cw']}")
        print(f"  R@1:              {best_params['R1']:.4f}")
        print(f"  R@5:              {best_params['RK']:.4f}")
        print(f"  MRR:              {best_params['MRR']:.4f}")
        print(f"  Chunks:           {best_params['n_chunks']}")
        print(f"  Avg chunk len:    {best_params['avg_len']:.1f} sentences")

        if args.target_r1 > 0:
            if best_params["R1"] >= args.target_r1:
                print(f"\n  TARGET MET: R@1 {best_params['R1']:.3f} >= {args.target_r1:.3f}")
            else:
                print(f"\n  TARGET MISSED: R@1 {best_params['R1']:.3f} < {args.target_r1:.3f}")

    # Top 10
    ok_results = [r for r in results if r.get("status") == "OK"]
    ok_results.sort(key=lambda x: (-x["R1"], -x["MRR"], -x["RK"]))
    print(f"\nTOP 10 CONFIGS:")
    print(f"{'Rank':>4} {'R@1':>6} {'R@5':>6} {'MRR':>6} {'Chunks':>6} {'AvgLen':>6} "
          f"{'Lambda':>8} {'BP':>5} {'MaxL':>4} {'CW':>4}")
    for i, r in enumerate(ok_results[:10], 1):
        print(
            f"{i:>4} {r['R1']:>6.3f} {r['RK']:>6.3f} {r['MRR']:>6.3f} "
            f"{r['n_chunks']:>6} {r['avg_len']:>6.1f} "
            f"{r['lambda']:>8.4f} {r['bp']:>5.1f} {r['max_len']:>4} {r['cw']:>4.1f}"
        )

    # Count configs meeting target
    if args.target_r1 > 0:
        meeting = [r for r in ok_results if r["R1"] >= args.target_r1]
        print(f"\n{len(meeting)} configs meet target R@1 >= {args.target_r1:.3f}")

    # Save report
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "hybrid_tune_report.json"
    report = {
        "best": best_params,
        "target_r1": args.target_r1,
        "target_met": best_params["R1"] >= args.target_r1 if best_params else False,
        "total_combos": total,
        "elapsed_seconds": elapsed,
        "grid": {
            "lambda": args.lambda_values,
            "bp": args.bp_values,
            "max_len": args.max_len_values,
            "cw": args.cw_values,
        },
        "hybrid_weights": {
            "dense": args.dense_weight,
            "sparse": args.sparse_weight,
        },
        "top_20": ok_results[:20],
        "all_results": results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
