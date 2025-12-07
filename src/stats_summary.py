#!/usr/bin/env python3
"""
Compute retrieval metrics (doc-mode and span-mode), coverage, and bootstrap CIs
for all methods from files under an output directory.

Assumes the pipeline produced:
  - intents: out/<doc>/intents.flat.jsonl
  - pseudo or human spans: out/<doc>/gt_spans.jsonl
  - chunk manifests + embeddings per method: out/<doc>/chunks.*.jsonl, out/<doc>/chunk_embs.*.npy

Example:
  python src/stats_summary.py --out-dir out/idc --embedder gemini-embedding-001 --dim 1536 \
    --variants idc fixed sliding coh paragraphs --bootstrap 2000 --seed 13

Outputs a JSON summary to stdout; optionally write to --json-out.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from dotenv import load_dotenv

import google.generativeai as genai


def load_env_api():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)


def norm_model(name: str) -> str:
    return name if name.startswith("models/") else f"models/{name}"


def read_jsonl(path: str | Path) -> List[Dict]:
    out: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def l2_normalize(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return mat / n


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


def embed_queries(texts: List[str], model: str, dim: int) -> np.ndarray:
    model = norm_model(model)
    vecs: List[List[float]] = []
    for t in texts:
        resp = genai.embed_content(
            model=model,
            content=t,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=dim,
        )
        v = extract_vec(resp)
        if len(v) != dim:
            if len(v) > dim: v = v[:dim]
            else: v = v + [0.0] * (dim - len(v))
        vecs.append(v)
    return l2_normalize(np.asarray(vecs, dtype=np.float32))


def find_variant_files(out_dir: Path, variant: str, dim: int) -> Tuple[Optional[Path], Optional[Path]]:
    # Returns (chunk_embs.npy, chunks.jsonl) for a variant, or (None, None) if not found
    if variant == "idc":
        emb = out_dir / f"chunk_embs.idc.d{dim}.npy"
        ch = out_dir / "chunks.idc.jsonl"
        return (emb if emb.exists() else None, ch if ch.exists() else None)
    pat_map = {
        "fixed": (f"chunk_embs.fixed*.d{dim}.npy", "chunks.fixed*.jsonl"),
        "sliding": (f"chunk_embs.sliding*.d{dim}.npy", "chunks.sliding*.jsonl"),
        "coh": (f"chunk_embs.coh*.d{dim}.npy", "chunks.coh*.jsonl"),
        "paragraphs": (f"chunk_embs.paragraphs*.d{dim}.npy", "chunks.paragraphs*.jsonl"),
    }
    if variant not in pat_map:
        return (None, None)
    emb_glob, chunk_glob = pat_map[variant]
    embs = list(out_dir.glob(emb_glob))
    if not embs:
        return (None, None)
    # pick newest embs and derive chunks path by name
    emb = max(embs, key=lambda p: p.stat().st_mtime)
    suffix = emb.name.replace("chunk_embs.", "").replace(f".d{dim}.npy", "")
    # try to find matching chunks by suffix
    ch_path = out_dir / f"chunks.{suffix}.jsonl"
    if not ch_path.exists():
        # fallback: pick newest matching pattern
        chunks = list(out_dir.glob(chunk_glob))
        ch_path = max(chunks, key=lambda p: p.stat().st_mtime) if chunks else None
    return (emb, ch_path)


def per_query_doc_metrics(I: np.ndarray, chunk_docs: List[str], q_docs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    # Returns (hit@1 flags, reciprocal ranks)
    n = len(q_docs)
    hit1 = np.zeros((n,), dtype=np.float32)
    rr = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        ranks = None
        for r, idx in enumerate(I[i]):
            if chunk_docs[idx] == q_docs[i]:
                ranks = r + 1
                break
        if ranks is not None:
            rr[i] = 1.0 / ranks
            if ranks == 1:
                hit1[i] = 1.0
    return hit1, rr


def per_query_span_metrics(
    I: np.ndarray,
    topk: int,
    chunks: List[Dict],
    spans: List[Dict],
    queries: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Returns arrays restricted to answerable queries: hit1, hitK, rr, mask_used (indices of queries used)
    # Pre-index chunk ids by doc
    chunks_by_doc: Dict[str, List[int]] = {}
    for idx, c in enumerate(chunks):
        chunks_by_doc.setdefault(c["doc_id"], []).append(idx)

    # Map spans by (doc_id, query_id)
    span_by_key: Dict[Tuple[str, int], Dict] = {}
    for s in spans:
        try:
            key = (s["doc_id"], int(s["query_id"]))
            span_by_key[key] = s
        except Exception:
            continue

    hit1_list: List[float] = []
    hitk_list: List[float] = []
    rr_list: List[float] = []
    used_qidx: List[int] = []

    for qi, q in enumerate(queries):
        key = (q["doc_id"], int(q["query_id"]))
        sp = span_by_key.get(key)
        if sp is None or sp.get("answerable") is False:
            continue
        doc_id = sp["doc_id"]
        s0 = int(sp["start_sent"]); s1 = int(sp["end_sent"])
        doc_chunk_indices = chunks_by_doc.get(doc_id, [])
        chunks_for_doc = [chunks[j] for j in doc_chunk_indices]
        # find local index of true chunk
        true_local = None
        for local_idx, ch in enumerate(chunks_for_doc):
            if int(ch["start_sent"]) <= s0 and int(ch["end_sent"]) >= s1:
                true_local = local_idx
                break
        if true_local is None:
            continue
        true_global = doc_chunk_indices[true_local]
        cand = list(I[qi, :topk])
        h1 = 1.0 if (len(cand) > 0 and cand[0] == true_global) else 0.0
        if true_global in cand:
            rk = cand.index(true_global) + 1
            hk = 1.0
            rr = 1.0 / rk
        else:
            hk = 0.0
            rr = 0.0
        hit1_list.append(h1); hitk_list.append(hk); rr_list.append(rr); used_qidx.append(qi)

    if not used_qidx:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    return (
        np.asarray(hit1_list, dtype=np.float32),
        np.asarray(hitk_list, dtype=np.float32),
        np.asarray(rr_list, dtype=np.float32),
        np.asarray(used_qidx, dtype=np.int32),
    )


def coverage_per_query(chunks: List[Dict], spans: List[Dict]) -> np.ndarray:
    # Returns vector over answerable spans: 1 if covered else 0
    chunks_by_doc: Dict[str, List[Dict]] = {}
    for ch in chunks:
        chunks_by_doc.setdefault(ch["doc_id"], []).append(ch)
    vals: List[float] = []
    for q in spans:
        if q.get("answerable") is False:
            continue
        s0 = int(q["start_sent"]); s1 = int(q["end_sent"])
        covered = False
        for ch in chunks_by_doc.get(q["doc_id"], []):
            if int(ch["start_sent"]) <= s0 and int(ch["end_sent"]) >= s1:
                covered = True
                break
        vals.append(1.0 if covered else 0.0)
    return np.asarray(vals, dtype=np.float32)


def bootstrap_mean(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, Tuple[float, float]]:
    if values.size == 0:
        return (0.0, (0.0, 0.0))
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    means = np.empty((n_boot,), dtype=np.float32)
    for b in range(n_boot):
        idxs = rng.integers(0, n, size=(n,))
        means[b] = float(values[idxs].mean())
    lo, hi = np.percentile(means, [2.5, 97.5]).tolist()
    return float(values.mean()), (float(lo), float(hi))


def avg_sentences_per_chunk(chunks: List[Dict]) -> float:
    lens: List[int] = [int(ch.get("num_sentences", 0) or (int(ch["end_sent"]) - int(ch["start_sent"]) + 1)) for ch in chunks]
    return float(np.mean(lens)) if lens else 0.0


def unique_docs(chunks: List[Dict]) -> List[str]:
    return sorted(set([ch["doc_id"] for ch in chunks]))


def compute_for_variant(
    out_dir: Path,
    variant: str,
    dim: int,
    queries: List[Dict],
    spans: List[Dict],
    Q: np.ndarray,
    topk: int,
    n_boot: int,
    seed: int,
) -> Dict:
    emb_path, chunks_path = find_variant_files(out_dir, variant, dim)
    if not emb_path or not chunks_path or (not emb_path.exists()) or (not chunks_path.exists()):
        return {"variant": variant, "available": False}
    X = np.load(emb_path)
    chunks = read_jsonl(chunks_path)
    chunk_docs = [c["doc_id"] for c in chunks]

    # Similarities and ranks
    sims = Q @ X.T if (Q.shape[1] == X.shape[1]) else Q[:, :X.shape[1]] @ X.T
    order = np.argsort(-sims, axis=1)
    I = order[:, :topk]

    # Doc-mode
    doc_list = unique_docs(chunks)
    doc_metrics = None
    if len(doc_list) >= 2:
        hit1_doc, rr_doc = per_query_doc_metrics(I, chunk_docs, [q["doc_id"] for q in queries])
        # Note: hit@K doc is indicator if any of topK has correct doc; approximate via RR>0 across topk
        # For CI, we report hit@1 and MRR; hit@K computed directly from I and chunk_docs at topk
        hitk_doc = np.zeros_like(hit1_doc)
        for i in range(len(queries)):
            retrieved_docs = [chunk_docs[idx] for idx in I[i]]
            hitk_doc[i] = 1.0 if queries[i]["doc_id"] in retrieved_docs else 0.0
        m_doc, ci_doc = bootstrap_mean(rr_doc, n_boot, seed)
        h1_doc_m, h1_doc_ci = bootstrap_mean(hit1_doc, n_boot, seed + 1)
        hk_doc_m, hk_doc_ci = bootstrap_mean(hitk_doc, n_boot, seed + 2)
        doc_metrics = {
            "doc_count": len(doc_list),
            "hit1": {"mean": float(h1_doc_m), "ci95": h1_doc_ci},
            "hitK": {"mean": float(hk_doc_m), "ci95": hk_doc_ci},
            "mrr": {"mean": float(m_doc), "ci95": ci_doc},
        }

    # Span-mode
    h1, hk, rr, used_qidx = per_query_span_metrics(I, topk, chunks, spans, queries)
    span_metrics = {
        "evaluated": int(h1.size),
        "total_queries": int(len(queries)),
        "R1": None,
        "RK": None,
        "MRR": None,
    }
    if h1.size > 0:
        r1_m, r1_ci = bootstrap_mean(h1, n_boot, seed + 3)
        rk_m, rk_ci = bootstrap_mean(hk, n_boot, seed + 4)
        mrr_m, mrr_ci = bootstrap_mean(rr, n_boot, seed + 5)
        span_metrics.update({
            "R1": {"mean": float(r1_m), "ci95": r1_ci},
            "RK": {"mean": float(rk_m), "ci95": rk_ci},
            "MRR": {"mean": float(mrr_m), "ci95": mrr_ci},
        })

    # Coverage
    cov_vec = coverage_per_query(chunks, spans)
    cov_m, cov_ci = bootstrap_mean(cov_vec, n_boot, seed + 6)

    return {
        "variant": variant,
        "available": True,
        "avg_sentences_per_chunk": float(avg_sentences_per_chunk(chunks)),
        "num_chunks": int(len(chunks)),
        "doc_mode": doc_metrics,
        "span_mode": span_metrics,
        "coverage": {"mean": float(cov_m), "ci95": cov_ci, "n": int(cov_vec.size)},
    }


def main():
    ap = argparse.ArgumentParser(description="Summarize metrics with bootstrap CIs for all methods")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--embedder", default="gemini-embedding-001")
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--variants", nargs="+", default=["idc", "fixed", "sliding", "coh", "paragraphs"],
                    help="Which methods to include: idc, fixed, sliding, coh, paragraphs")
    ap.add_argument("--spans", default=None, help="Path to spans JSONL; defaults to <out-dir>/gt_spans.jsonl")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    intents_path = out_dir / "intents.flat.jsonl"
    spans_path = Path(args.spans) if args.spans else (out_dir / "gt_spans.jsonl")
    assert intents_path.exists(), f"Missing intents: {intents_path}"
    assert spans_path.exists(), f"Missing spans: {spans_path}"

    # Load queries and spans
    queries = read_jsonl(intents_path)
    spans = read_jsonl(spans_path)

    # Embed queries
    load_env_api()
    Q = embed_queries([q["text"] for q in queries], args.embedder, args.dim)

    # Compute per variant
    results: List[Dict] = []
    for v in args.variants:
        res = compute_for_variant(out_dir, v, args.dim, queries, spans, Q, args.topk, args.bootstrap, args.seed)
        results.append(res)

    # Emit JSON summary
    summary = {
        "out_dir": str(out_dir),
        "embedder": args.embedder,
        "dim": args.dim,
        "topk": args.topk,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

