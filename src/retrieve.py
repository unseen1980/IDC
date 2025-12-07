#!/usr/bin/env python3
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from dotenv import load_dotenv

# Try FAISS; fallback to NumPy search if unavailable
try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False
    faiss = None

import google.generativeai as genai

def read_jsonl(path: str | Path) -> List[Dict]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def l2_normalize(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return mat / n

def configure_genai():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)

def normalize_model(model_name: str) -> str:
    return model_name if model_name.startswith("models/") else f"models/{model_name}"

def extract_vec(resp) -> List[float]:
    # Your embed.py change returns resp.embedding as a flat list already
    if hasattr(resp, "embedding"):
        emb = resp.embedding
        # some versions use .values; handle both
        if hasattr(emb, "values"):
            return list(emb.values)
        return list(emb)
    if isinstance(resp, dict) and "embedding" in resp:
        e = resp["embedding"]
        if isinstance(e, dict) and "values" in e:
            return list(e["values"])
        return list(e)
    raise ValueError("Cannot extract embedding")

def embed_queries(queries: List[str], model: str, dim: int) -> np.ndarray:
    model = normalize_model(model)
    vecs = []
    for q in queries:
        resp = genai.embed_content(
            model=model,
            content=q,
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=dim,
        )
        v = extract_vec(resp)
        if len(v) != dim:
            if len(v) > dim: v = v[:dim]
            else: v = v + [0.0]*(dim - len(v))
        vecs.append(v)
    Q = np.asarray(vecs, dtype=np.float32)
    return l2_normalize(Q)

def build_index(X: np.ndarray):
    if HAVE_FAISS:
        index = faiss.IndexFlatIP(X.shape[1])  # cosine if vectors are L2-normalized
        index.add(X)
        return index
    return None  # we'll use numpy fallback

def search(index, X: np.ndarray, Q: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    if HAVE_FAISS and index is not None:
        D, I = index.search(Q, topk)  # (nq, k)
        return D, I
    # NumPy fallback (cosine via dot product on normalized vectors)
    sims = Q @ X.T  # (nq, n)
    I = np.argsort(-sims, axis=1)[:, :topk]
    D = np.take_along_axis(sims, I, axis=1)
    return D, I

def compute_r_at_k_and_mrr(I: np.ndarray, truth: List[int], k_vals=(1,5)) -> Dict[str, float]:
    n = len(truth)
    metrics = {}
    # R@K
    for k in k_vals:
        hit = 0
        for i in range(n):
            if truth[i] in I[i, :k]:
                hit += 1
        metrics[f"R@{k}"] = hit / max(n, 1)
    # MRR
    rr = 0.0
    for i in range(n):
        ranks = np.where(I[i] == truth[i])[0]
        if len(ranks) > 0:
            rr += 1.0 / (ranks[0] + 1)
    metrics["MRR"] = rr / max(n, 1)
    return metrics

def main():
    ap = argparse.ArgumentParser(description="IDC: Retrieval smoke test using chunk intents as queries")
    ap.add_argument("--chunk-embs", type=str, default="out/chunk_embs.npy")
    ap.add_argument("--anchor-embs", type=str, help="Optional anchor embeddings")
    ap.add_argument("--chunks", type=str, default="out/chunks.jsonl")
    ap.add_argument("--embedder", type=str, default="gemini-embedding-001")
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    # Load chunk vectors & manifest
    X = np.load(args.chunk_embs)            # (n_chunks, dim) ideally L2-normalized
    chunks = read_jsonl(args.chunks)        # [{chunk_uid, intent, ...}]

    anchor_vecs = None
    if args.anchor_embs:
        anchor_path = Path(args.anchor_embs)
        if anchor_path.exists():
            anchor_vecs = np.load(anchor_path)
            if anchor_vecs.shape[0] != X.shape[0]:
                raise ValueError("Anchor embeddings/metadata mismatch")
        else:
            print(f"[WARN] Anchor embeddings not found: {anchor_path}")

    # Derive a query set from chunks' own 'intent' labels (intrinsic sanity check)
    query_texts = []
    truth_idx = []   # index of the matching chunk
    uid_to_idx = {c["chunk_uid"]: i for i, c in enumerate(chunks)}

    for i, c in enumerate(chunks):
        q = (c.get("intent") or "").strip()
        if not q:
            continue
        query_texts.append(q)
        truth_idx.append(i)

    if not query_texts:
        print("No intents found in chunks.jsonl to use as queries.")
        return

    # Embed queries
    configure_genai()
    Q = embed_queries(query_texts, model=args.embedder, dim=args.dim)

    # Prepare chunk vectors (blend anchors if available)
    X = X.astype(np.float32)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)
    if anchor_vecs is not None:
        anchor_vecs = anchor_vecs.astype(np.float32)
        anchor_vecs = anchor_vecs / np.maximum(np.linalg.norm(anchor_vecs, axis=1, keepdims=True), 1e-8)
        X = anchor_vecs
    # Build index & search
    index = build_index(X)  # FAISS if available
    D, I = search(index, X, Q, args.topk)

    # Evaluate: success if the correct chunk is retrieved
    metrics = compute_r_at_k_and_mrr(I, truth_idx, k_vals=(1, args.topk))
    print(f"Chunks: {len(chunks)} | Queries: {len(query_texts)} | FAISS: {HAVE_FAISS}")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # Show a couple of qualitative examples
    for qi in range(min(3, len(query_texts))):
        print("\nQuery:", query_texts[qi])
        print("Top hits:")
        for rank in range(min(args.topk, I.shape[1])):
            idx = int(I[qi, rank])
            print(f"  {rank+1:>2}. {chunks[idx]['chunk_uid']}  ·  score={D[qi, rank]:.3f}")

if __name__ == "__main__":
    main()
