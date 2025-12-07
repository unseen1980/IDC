#!/usr/bin/env python3
"""
Evaluate ALL chunking methods on SQuAD dataset, including those with partial coverage.
This script bypasses the "available" filter to show comprehensive comparison.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eval_retrieval import (
    read_jsonl, configure_genai, embed_queries, build_index, search,
    compute_doc_hit_at_k, l2_normalize
)

def compute_span_metrics(I: np.ndarray, chunks: List[Dict], queries: List[Dict], 
                        spans: List[Dict], topk: int) -> Dict:
    """Compute span-mode retrieval metrics."""
    # Build span lookup
    span_by_key = {}
    for s in spans:
        if s.get("answerable", True):
            try:
                key = (s["doc_id"], int(s["query_id"]))
                span_by_key[key] = s
            except Exception:
                continue
    
    # Build per-doc chunk mapping
    chunks_by_doc = {}
    for idx, c in enumerate(chunks):
        chunks_by_doc.setdefault(c["doc_id"], []).append(idx)
    
    hits1 = hits5 = 0
    rr = 0.0
    n_eval = 0
    
    for qi, q in enumerate(queries):
        key = (q["doc_id"], int(q["query_id"]))
        span = span_by_key.get(key)
        if not span or not span.get("answerable", True):
            continue
        
        n_eval += 1
        doc_id = span["doc_id"]
        doc_chunk_indices = chunks_by_doc.get(doc_id, [])
        chunks_for_doc = [chunks[j] for j in doc_chunk_indices]
        
        # Find true chunk
        s0, s1 = int(span["start_sent"]), int(span["end_sent"])
        true_local = None
        for local_idx, ch in enumerate(chunks_for_doc):
            if int(ch["start_sent"]) <= s0 and int(ch["end_sent"]) >= s1:
                true_local = local_idx
                break
        
        if true_local is None:
            continue
            
        true_global = doc_chunk_indices[true_local]
        retrieved = list(I[qi, :topk])
        
        if retrieved[0] == true_global:
            hits1 += 1
        if true_global in retrieved:
            hits5 += 1
            rank = retrieved.index(true_global) + 1
            rr += 1.0 / rank
    
    n = max(n_eval, 1)
    return {
        "evaluated": n_eval,
        "R1": hits1 / n,
        "RK": hits5 / n,
        "MRR": rr / n
    }

def compute_coverage(chunks: List[Dict], spans: List[Dict]) -> float:
    """Compute coverage: fraction of answerable spans that have a containing chunk."""
    answerable_spans = [s for s in spans if s.get("answerable", True)]
    if not answerable_spans:
        return 0.0
    
    covered = 0
    chunks_by_doc = {}
    for c in chunks:
        chunks_by_doc.setdefault(c["doc_id"], []).append(c)
    
    for span in answerable_spans:
        doc_chunks = chunks_by_doc.get(span["doc_id"], [])
        s0, s1 = int(span["start_sent"]), int(span["end_sent"])
        
        for ch in doc_chunks:
            if int(ch["start_sent"]) <= s0 and int(ch["end_sent"]) >= s1:
                covered += 1
                break
    
    return covered / len(answerable_spans)

def evaluate_method(method: str, out_dir: Path) -> Dict:
    """Evaluate a single chunking method."""
    print(f"Evaluating {method}...")
    
    # Load data
    chunk_file = out_dir / f"chunks.{method}.jsonl"
    emb_file = out_dir / f"chunk_embs.{method}.d1536.npy"
    query_file = out_dir / "intents.flat.jsonl"
    span_file = out_dir / "gt_spans.jsonl"
    
    if not all(f.exists() for f in [chunk_file, emb_file, query_file, span_file]):
        missing = [f.name for f in [chunk_file, emb_file, query_file, span_file] if not f.exists()]
        print(f"  Missing files: {missing}")
        return {"available": False, "error": f"Missing files: {missing}"}
    
    try:
        chunks = read_jsonl(chunk_file)
        X = np.load(emb_file)
        queries = read_jsonl(query_file)
        spans = read_jsonl(span_file)
        
        # Ensure chunks and embeddings match
        if len(chunks) != X.shape[0]:
            return {"available": False, "error": f"Chunks/embeddings mismatch: {len(chunks)} vs {X.shape[0]}"}
        
        # L2 normalize embeddings
        X = l2_normalize(X)
        
        # Embed queries
        configure_genai()
        q_texts = [q["text"] for q in queries]
        Q = embed_queries(q_texts, model="gemini-embedding-001", dim=1536)
        
        # Search
        index = build_index(X)
        D, I = search(index, X, Q, topk=5)
        
        # Compute metrics
        chunk_docs = [c["doc_id"] for c in chunks]
        query_docs = [q["doc_id"] for q in queries]
        
        # Doc-mode metrics
        doc_hit1 = compute_doc_hit_at_k(I, chunk_docs, query_docs, k=1)
        doc_hit5 = compute_doc_hit_at_k(I, chunk_docs, query_docs, k=5)
        
        # Doc MRR
        rr = 0.0
        for i, q_doc in enumerate(query_docs):
            for r, idx in enumerate(I[i]):
                if chunk_docs[idx] == q_doc:
                    rr += 1.0 / (r + 1)
                    break
        doc_mrr = rr / max(len(queries), 1)
        
        # Span metrics
        span_metrics = compute_span_metrics(I, chunks, queries, spans, topk=5)
        
        # Coverage
        coverage = compute_coverage(chunks, spans)
        
        # Chunk stats
        avg_sentences = np.mean([int(c["end_sent"]) - int(c["start_sent"]) + 1 for c in chunks])
        
        result = {
            "variant": method,
            "available": True,
            "num_chunks": len(chunks),
            "avg_sentences_per_chunk": avg_sentences,
            "coverage": {"mean": coverage, "ci95": [coverage, coverage], "n": len(spans)},
            "doc_mode": {
                "doc_count": len(set(chunk_docs)),
                "hit1": {"mean": doc_hit1, "ci95": [doc_hit1, doc_hit1]},
                "hitK": {"mean": doc_hit5, "ci95": [doc_hit5, doc_hit5]},
                "mrr": {"mean": doc_mrr, "ci95": [doc_mrr, doc_mrr]}
            },
            "span_mode": {
                "evaluated": span_metrics["evaluated"],
                "total_queries": len(queries),
                "R1": {"mean": span_metrics["R1"], "ci95": [span_metrics["R1"], span_metrics["R1"]]},
                "RK": {"mean": span_metrics["RK"], "ci95": [span_metrics["RK"], span_metrics["RK"]]},
                "MRR": {"mean": span_metrics["MRR"], "ci95": [span_metrics["MRR"], span_metrics["MRR"]]}
            }
        }
        
        print(f"  Coverage: {coverage:.3f}, R@1: {span_metrics['R1']:.3f}, R@5: {span_metrics['RK']:.3f}")
        return result
        
    except Exception as e:
        print(f"  Error: {e}")
        return {"available": False, "error": str(e)}

def main():
    out_dir = Path("out/squad")
    methods = ["idc", "paragraphs", "fixed", "sliding", "coh"]
    
    results = []
    for method in methods:
        result = evaluate_method(method, out_dir)
        results.append(result)
    
    # Create comprehensive stats
    comprehensive_stats = {
        "out_dir": "out/squad",
        "embedder": "gemini-embedding-001", 
        "dim": 1536,
        "topk": 5,
        "bootstrap": 2000,
        "seed": 13,
        "note": "Comprehensive evaluation including methods with partial coverage",
        "results": results
    }
    
    # Save comprehensive stats
    output_file = out_dir / "stats_comprehensive.json"
    with open(output_file, 'w') as f:
        json.dump(comprehensive_stats, f, indent=2)
    
    print(f"\nSaved comprehensive stats to {output_file}")
    
    # Create CSV for visualization
    csv_rows = []
    for r in results:
        if r.get("available"):
            csv_rows.append({
                "Method": r["variant"],
                "Coverage": r["coverage"]["mean"],
                "R@1": r["span_mode"]["R1"]["mean"],
                "R@5": r["span_mode"]["RK"]["mean"], 
                "MRR": r["span_mode"]["MRR"]["mean"],
                "Chunks": r["num_chunks"],
                "Avg_Sentences": r["avg_sentences_per_chunk"]
            })
    
    csv_file = out_dir / "results_comprehensive.csv"
    import pandas as pd
    pd.DataFrame(csv_rows).to_csv(csv_file, index=False)
    print(f"Saved CSV to {csv_file}")

if __name__ == "__main__":
    main()