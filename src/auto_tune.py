#!/usr/bin/env python3
"""
Auto-tuning wrapper for IDC parameters based on coverage on pseudo-gold spans.
"""
import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import tempfile
import os

from idc_core import IDCParams, segment_corpus_from_files, write_jsonl

def read_jsonl(path: str | Path) -> List[Dict]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def run_segment_with_params(
    sentences: str,
    sentence_embs: str,
    sentences_meta: str,
    intents_flat: str,
    intent_embs: str,
    lam: float,
    max_len: int,
    min_len: int,
    boundary_pen: float,
    coherence_weight: float,
    merge_adjacent: bool,
    output_file: str,
    algorithm: str = "standard",
) -> bool:
    """Run IDC segmentation with the canonical algorithm."""
    if algorithm != "standard":
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Only 'standard' is available after cleanup."
        )

    params = IDCParams(
        lam=lam,
        max_len=max_len,
        min_len=min_len,
        boundary_penalty=boundary_pen,
        coherence_weight=coherence_weight,
        merge_adjacent=merge_adjacent,
    )

    results = segment_corpus_from_files(
        sentences_path=sentences,
        sentence_embs_path=sentence_embs,
        sentences_meta_path=sentences_meta,
        intents_flat_path=intents_flat,
        intent_embs_path=intent_embs,
        params=params,
    )

    write_jsonl(
        output_file,
        ({"doc_id": res.doc_id, "chunks": res.chunks} for res in results),
    )
    return True

def compute_coverage(segments_file: str, spans_file: str) -> float:
    """Compute coverage score using eval_coverage.py"""
    try:
        cmd = [
            "python", "src/eval_coverage.py",
            "--segments", segments_file,
            "--spans", spans_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return 0.0
        
        # Parse coverage from output like "Answer coverage: 34/51 = 0.667"
        for line in result.stdout.split('\n'):
            if 'Answer coverage:' in line and '=' in line:
                try:
                    coverage_str = line.split('=')[1].strip()
                    return float(coverage_str)
                except (IndexError, ValueError):
                    continue
        return 0.0
    except Exception:
        return 0.0

def compute_span_metrics_fast(chunk_embs_path: str, chunks_path: str, intents_flat_path: str, intent_embs_path: str,
                              spans_path: str, anchor_embs_path: str | None = None, topk: int = 5) -> Dict[str, float]:
    """Compute span-mode retrieval metrics quickly using precomputed intent embeddings.
    Avoids re-embedding queries and external subprocess calls.
    """
    try:
        import numpy as np
        from pathlib import Path
        import json

        # Load data
        X = np.load(chunk_embs_path)  # (n_chunks, d)
        with Path(chunks_path).open("r", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]
        Q = np.load(intent_embs_path)  # (n_queries, d)
        with Path(intents_flat_path).open("r", encoding="utf-8") as f:
            q_rows = [json.loads(line) for line in f if line.strip()]
        anchors = None
        if anchor_embs_path:
            anchor_file = Path(anchor_embs_path)
            if anchor_file.exists():
                anchors = np.load(anchor_file)
        with Path(spans_path).open("r", encoding="utf-8") as f:
            span_rows = [json.loads(line) for line in f if line.strip()]

        # Normalize
        def l2n(a):
            n = np.linalg.norm(a, axis=1, keepdims=True)
            n = np.maximum(n, 1e-8)
            return a / n
        # Use anchor embeddings if available, otherwise use chunk embeddings
        if anchors is not None and anchors.shape == X.shape:
            Xn = l2n(anchors)
        else:
            Xn = l2n(X)
        Qn = l2n(Q) if Q.shape[0] > 0 else Q

        # Build lookup structures
        chunk_docs = [c.get("doc_id") for c in chunks]
        chunks_by_doc: Dict[str, List[int]] = {}
        for idx, c in enumerate(chunks):
            chunks_by_doc.setdefault(c["doc_id"], []).append(idx)
        spans_by_key = {}
        for s in span_rows:
            try:
                spans_by_key[(s["doc_id"], int(s["query_id"]))] = s
            except Exception:
                continue

        # Compute similarities and topk indices
        sims = Qn @ Xn.T  # (n_queries, n_chunks)
        I = np.argsort(-sims, axis=1)[:, :topk]

        # Evaluate
        hits1 = hitsK = 0
        rr = 0.0
        n_eval = 0
        total_queries = 0
        for qi, q in enumerate(q_rows):
            total_queries += 1
            key = (q.get("doc_id"), int(q.get("query_id", 0)))
            span = spans_by_key.get(key)
            if span is None or not span.get("answerable", True):
                continue
            n_eval += 1
            doc_id = span["doc_id"]
            doc_chunk_indices = chunks_by_doc.get(doc_id, [])
            chunks_for_doc = [chunks[j] for j in doc_chunk_indices]
            s0 = int(span["start_sent"])
            s1 = int(span["end_sent"])
            # Find local true chunk
            true_local = None
            for local_idx, ch in enumerate(chunks_for_doc):
                if int(ch["start_sent"]) <= s0 and int(ch["end_sent"]) >= s1:
                    true_local = local_idx
                    break
            if true_local is None:
                continue
            true_global = doc_chunk_indices[true_local]
            cand = list(I[qi])
            if not cand:
                continue
            if cand[0] == true_global:
                hits1 += 1
            if true_global in cand:
                hitsK += 1
                rank = cand.index(true_global) + 1
                rr += 1.0 / rank
        n = max(n_eval, 1)
        return {
            "R1": hits1 / n,
            "RK": hitsK / n,
            "MRR": rr / n,
        }
    except Exception:
        return {"R1": 0.0, "RK": 0.0, "MRR": 0.0}

def avg_sentences_per_chunk(segments_file: str) -> float:
    try:
        rows = read_jsonl(segments_file)
        lens = []
        for doc in rows:
            for ch in doc.get("chunks", []):
                n = ch.get("num_sentences")
                if n is None:
                    try:
                        n = int(ch["end_sent"]) - int(ch["start_sent"]) + 1
                    except Exception:
                        n = 0
                lens.append(int(n))
        import numpy as np
        return float(np.mean(lens)) if lens else 0.0
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser(description="Auto-tune IDC parameters for maximum coverage")
    
    # Input files
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--sentence-embs", required=True) 
    ap.add_argument("--sentences-meta", required=True)
    ap.add_argument("--intents-flat", required=True)
    ap.add_argument("--intent-embs", required=True)
    ap.add_argument("--spans", required=True, help="Pseudo-gold spans file")
    
    # Output
    ap.add_argument("--out", required=True, help="Best segmentation output file")
    ap.add_argument("--report", help="Tuning report output file")
    
    # Algorithm selection
    ap.add_argument("--algorithm", choices=["standard"],
                    default="standard", help="IDC algorithm variant to use")
    
    # Fixed parameters (can be overridden by grids below)
    ap.add_argument("--max-len", type=int, default=10)
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--coherence-weight", type=float, default=0.05)
    ap.add_argument("--merge-adjacent", action="store_true")
    # Retrieval-aware optimization
    ap.add_argument("--optimize", choices=["coverage", "r1", "mrr", "combined", "length_aware"], default="coverage",
                    help="Optimization objective across grid")
    ap.add_argument("--eval-embedder", type=str, default="gemini-embedding-001")
    ap.add_argument("--dim", type=int, default=1536)
    
    # Tuning grids
    ap.add_argument("--lambda-values", nargs='+', type=float,
                    default=[0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004],
                    help="Lambda values to try (lower → longer chunks)")
    ap.add_argument("--boundary-penalty-values", nargs='+', type=float,
                    default=[0.05, 0.15, 0.25, 0.35, 0.50],
                    help="Boundary penalty values to try (lower → fewer chunks)")
    ap.add_argument("--max-len-grid", nargs='+', type=int,
                    default=None,
                    help="Optional grid for max_len (overrides --max-len if provided)")
    ap.add_argument("--coherence-weight-grid", nargs='+', type=float,
                    default=None,
                    help="Optional grid for coherence weight (overrides --coherence-weight if provided)")

    # Optional length matching objective
    ap.add_argument("--target-avg", type=float, default=None,
                    help="Optional target average sentences/chunk (prefer configs within tolerance)")
    ap.add_argument("--tolerance", type=float, default=None,
                    help="Tolerance in sentences for target-avg; if present, prefer configs within ±tolerance")
    
    args = ap.parse_args()
    
    print(f"Auto-tuning IDC parameters...")
    print(f"Lambda grid: {args.lambda_values}")
    print(f"Boundary penalty grid: {args.boundary_penalty_values}")
    M_grid = args.max_len_grid if args.max_len_grid else [args.max_len]
    C_grid = args.coherence_weight_grid if args.coherence_weight_grid else [args.coherence_weight]
    total = len(args.lambda_values) * len(args.boundary_penalty_values) * len(M_grid) * len(C_grid)
    print(f"Max-len grid: {M_grid}")
    print(f"Coherence-weight grid: {C_grid}")
    print(f"Total combinations: {total}")
    
    best_coverage = -1.0
    best_params = None
    best_segments_file = None
    best_avg_len = None  # track best average length for tie-breaking
    best_r1 = 0.0  # track best R@1 for comparison
    best_mrr = 0.0  # track best MRR for comparison
    results = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        for lam in args.lambda_values:
            for boundary_pen in args.boundary_penalty_values:
                for max_len in M_grid:
                    for coh_w in C_grid:
                        print(f"  Trying λ={lam}, boundary_pen={boundary_pen}, max_len={max_len}, C={coh_w}...")

                        temp_segments = os.path.join(temp_dir, f"seg_l{lam}_b{boundary_pen}_L{max_len}_c{coh_w}.jsonl")

                        success = run_segment_with_params(
                            args.sentences, args.sentence_embs, args.sentences_meta,
                            args.intents_flat, args.intent_embs,
                            lam, max_len, args.min_len,
                            boundary_pen, coh_w,
                            args.merge_adjacent, temp_segments,
                            algorithm=args.algorithm
                        )

                        if not success:
                            print(f"    FAILED to run segmentation")
                            results.append((lam, boundary_pen, max_len, coh_w, 0.0, 0.0, "FAILED"))
                            continue

                        coverage = compute_coverage(temp_segments, args.spans)
                        avg_len = avg_sentences_per_chunk(temp_segments)
                        print(f"    Coverage: {coverage:.3f} | Avg len: {avg_len:.2f}")

                        # If optimizing for retrieval, build temp chunk embeddings and evaluate span metrics
                        r1 = mrr = rk = 0.0
                        if args.optimize in {"r1", "mrr", "combined"}:
                            temp_chunks = os.path.join(temp_dir, f"chunks_l{lam}_b{boundary_pen}_L{max_len}_c{coh_w}.jsonl")
                            temp_embs = os.path.join(temp_dir, f"embs_l{lam}_b{boundary_pen}_L{max_len}_c{coh_w}.npy")
                            temp_anchor = os.path.join(temp_dir, f"anchor_l{lam}_b{boundary_pen}_L{max_len}_c{coh_w}.npy")
                            # Build chunk embeddings with intent-weighting to mirror pipeline
                            bc_cmd = [
                                "python", "src/build_chunks.py",
                                "--sentences", args.sentences,
                                "--sentence-embs", args.sentence_embs,
                                "--segments", temp_segments,
                                "--out-embs", temp_embs,
                                "--out-chunks", temp_chunks,
                                "--out-anchor-embs", temp_anchor,
                                "--normalize",
                                "--intent-weighted",
                                "--intent-embs", args.intent_embs,
                                "--intents-flat", args.intents_flat,
                            ]
                            _ = subprocess.run(bc_cmd, capture_output=True, text=True, timeout=120)
                            # Fast metrics using precomputed intent embeddings
                            metrics = compute_span_metrics_fast(temp_embs, temp_chunks, args.intents_flat, args.intent_embs, args.spans, temp_anchor, topk=5)
                            r1 = metrics.get("R1", 0.0)
                            rk = metrics.get("RK", 0.0)
                            mrr = metrics.get("MRR", 0.0)
                            print(f"    R@1: {r1:.3f} | R@5: {rk:.3f} | MRR: {mrr:.3f}")

                        results.append((lam, boundary_pen, max_len, coh_w, coverage, avg_len, "OK", r1, mrr))

                        # Selection logic: add gentle length regularization around target band
                        def is_preferred(curr_cov, curr_avg, best_cov, best_avg, curr_r1=0.0, curr_mrr=0.0, best_r1=0.0, best_mrr=0.0):
                            """Return True if current beats best according to objective.
                            Notes:
                              - For coverage objective: prefer configs inside target band when provided.
                              - For combined objective: apply a soft penalty when avg length is outside target±tol.
                            """
                            tgt = args.target_avg
                            tol = args.tolerance
                            # helper: soft length penalty when outside band
                            def length_penalty(avg: float) -> float:
                                if tgt is None:
                                    # default target ~6 sentences
                                    t = 6.0
                                    return max(0.0, abs(avg - t) - 1.0) * 0.02
                                if tol is None:
                                    # penalize distance from target
                                    return max(0.0, abs(avg - tgt)) * 0.02
                                # penalize only the excess beyond tolerance band
                                excess = max(0.0, abs(avg - tgt) - tol)
                                return excess * 0.05

                            if args.optimize == "coverage":
                                # Prefer inside band when provided, then maximize coverage, then closest to target
                                if tgt is not None and tol is not None:
                                    curr_in = abs(curr_avg - tgt) <= tol
                                    best_in = (best_avg is not None) and (abs(best_avg - tgt) <= tol)
                                    if curr_in and not best_in:
                                        return True
                                    if best_in and not curr_in:
                                        return False
                                # compare coverage next
                                if curr_cov != best_cov:
                                    return curr_cov > best_cov
                                # tie-breaker: closer to target (or 6.0 if not provided)
                                target = tgt if tgt is not None else 6.0
                                curr_dist = abs(curr_avg - target)
                                best_dist = abs((best_avg if best_avg is not None else target) - target)
                                return curr_dist < best_dist
                            elif args.optimize == "length_aware":
                                # Length-aware optimization: balance coverage with length constraints
                                target = tgt if tgt is not None else 6.0
                                
                                # Length penalty: exponential penalty for exceeding target
                                def length_score(avg_len):
                                    if avg_len <= target:
                                        return 1.0 - (target - avg_len) * 0.05  # Small penalty for being too short
                                    else:
                                        excess = avg_len - target
                                        return 1.0 - (excess ** 1.5) * 0.1  # Exponential penalty for being too long
                                
                                curr_length_score = length_score(curr_avg)
                                best_length_score = length_score(best_avg if best_avg is not None else curr_avg)
                                
                                # Combined score: 70% coverage, 30% length adherence
                                curr_combined = 0.7 * curr_cov + 0.3 * curr_length_score
                                best_combined = 0.7 * best_cov + 0.3 * best_length_score
                                
                                if abs(curr_combined - best_combined) > 1e-6:
                                    return curr_combined > best_combined
                                    
                                # Tie-break by better length adherence
                                return curr_length_score > best_length_score
                            else:
                                # Retrieval-aware: combined score with soft length regularization
                                curr_score = curr_cov + 0.6 * curr_r1 + 0.3 * curr_mrr - length_penalty(curr_avg)
                                best_score = best_cov + 0.6 * best_r1 + 0.3 * best_mrr - length_penalty(best_avg if best_avg is not None else curr_avg)
                                if abs(curr_score - best_score) > 1e-6:
                                    return curr_score > best_score
                                # tie-break by closeness to target length
                                target = tgt if tgt is not None else 6.0
                                curr_dist = abs(curr_avg - target)
                                best_dist = abs((best_avg if best_avg is not None else target) - target)
                                return curr_dist < best_dist

                        # BUG FIX: Pass current best_r1 and best_mrr instead of results[-2]
                        if (best_params is None) or is_preferred(coverage, avg_len, best_coverage, best_avg_len, r1, mrr, best_r1, best_mrr):
                            best_coverage = coverage
                            best_params = (lam, boundary_pen, max_len, coh_w)
                            best_segments_file = temp_segments
                            best_avg_len = avg_len
                            best_r1 = r1  # Update best R@1
                            best_mrr = mrr  # Update best MRR
                            import shutil
                            shutil.copy2(temp_segments, args.out)
        
        print(f"\n=== Auto-tuning Results ===")
        print(f"Best coverage: {best_coverage:.3f}")
        if args.optimize in {"r1", "mrr", "combined"}:
            print(f"Best R@1: {best_r1:.3f}")
            print(f"Best MRR: {best_mrr:.3f}")
        if best_params:
            print(f"Best λ: {best_params[0]}")
            print(f"Best boundary penalty: {best_params[1]}")
            print(f"Best max_len: {best_params[2]}")
            print(f"Best coherence_weight: {best_params[3]}")
        else:
            print("No successful runs!")
        
        # Save detailed report
        if args.report:
            report = {
                "best_coverage": best_coverage,
                "best_r1": best_r1,
                "best_mrr": best_mrr,
                "best_lambda": best_params[0] if best_params else None,
                "best_boundary_penalty": best_params[1] if best_params else None,
                "best_max_len": best_params[2] if best_params else None,
                "best_coherence_weight": best_params[3] if best_params else None,
                "optimize_objective": args.optimize,
                "all_results": [
                    {
                        "lambda": lam,
                        "boundary_penalty": bp,
                        "max_len": ml,
                        "coherence_weight": cw,
                        "coverage": cov,
                        "avg_len": al,
                        "R1": r1,
                        "MRR": mrr,
                        "status": status
                    }
                    for lam, bp, ml, cw, cov, al, status, r1, mrr in results
                ]
            }
            
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            with open(args.report, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Detailed report saved to: {args.report}")
        
        return best_coverage > 0

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
