#!/usr/bin/env python3
"""
Generate stats.json file for QT app visualization.
This script collects evaluation results and formats them for the visualization interface.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import tiktoken


TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")

def _l2n(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n = np.maximum(n, 1e-8)
    return a / n

def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def _pick_latest(glob_pat: str, root: Path) -> Optional[Path]:
    files = list(root.glob(glob_pat))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

def _compute_span_metrics_fast(chunk_embs: Path, chunks: Path, intents_flat: Path, intent_embs: Path,
                               spans: Path, anchor_embs: Path | None = None, topk: int = 5) -> Dict[str, float]:
    try:
        X = np.load(chunk_embs)
        C = _read_jsonl(chunks)
        Q = np.load(intent_embs)
        Q_rows = _read_jsonl(intents_flat)
        S_rows = _read_jsonl(spans)
        anchors = None
        if anchor_embs is not None and anchor_embs.exists():
            anchors = np.load(anchor_embs)

        # Use anchor embeddings if available, otherwise use chunk embeddings
        if anchors is not None and anchors.shape == X.shape:
            Xn = _l2n(anchors)
        else:
            Xn = _l2n(X)
        Qn = _l2n(Q) if Q.shape[0] > 0 else Q

        # Indices by doc
        chunk_docs = [c.get("doc_id") for c in C]
        chunks_by_doc: Dict[str, List[int]] = {}
        for idx, c in enumerate(C):
            chunks_by_doc.setdefault(c["doc_id"], []).append(idx)
        spans_by_key = {}
        for s in S_rows:
            try:
                spans_by_key[(s["doc_id"], int(s["query_id"]))] = s
            except Exception:
                continue

        sims = Qn @ Xn.T
        I = np.argsort(-sims, axis=1)[:, :topk]

        hits1 = hitsK = 0
        rr = 0.0
        n_eval = 0
        total_queries = 0
        for qi, q in enumerate(Q_rows):
            total_queries += 1
            key = (q.get("doc_id"), int(q.get("query_id", 0)))
            span = spans_by_key.get(key)
            if span is None or not span.get("answerable", True):
                continue
            n_eval += 1
            doc_id = span["doc_id"]
            doc_chunk_indices = chunks_by_doc.get(doc_id, [])
            chunks_for_doc = [C[j] for j in doc_chunk_indices]
            s0 = int(span["start_sent"])
            s1 = int(span["end_sent"])
            true_local = None
            for local_idx, ch in enumerate(chunks_for_doc):
                if int(ch.get("start_sent", 0)) <= s0 and int(ch.get("end_sent", -1)) >= s1:
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
        return {"R1": hits1 / n, "RK": hitsK / n, "MRR": rr / n, "evaluated": n, "total_queries": total_queries}
    except Exception:
        return {"R1": 0.0, "RK": 0.0, "MRR": 0.0, "evaluated": 0, "total_queries": 0}

def _answer_coverage(segments_file: Path, spans_file: Path) -> float:
    try:
        seg_docs = _read_jsonl(segments_file)
        spans = _read_jsonl(spans_file)
        chunks_by_doc: Dict[str, List[Dict]] = {d["doc_id"]: d.get("chunks", []) for d in seg_docs}
        ok = 0; total = 0
        for q in spans:
            if q.get("answerable") is False:
                continue
            total += 1
            doc_id = q.get("doc_id")
            s0 = int(q.get("start_sent", -1)); s1 = int(q.get("end_sent", -1))
            if s0 < 0 or s1 < 0:
                continue
            covered = False
            for ch in chunks_by_doc.get(doc_id, []):
                if int(ch.get("start_sent", 0)) <= s0 and int(ch.get("end_sent", -1)) >= s1:
                    covered = True; break
            if covered:
                ok += 1
        return ok / max(total, 1)
    except Exception:
        return 0.0


def _compute_extended_metrics(chunk_embs: Path, chunks: Path, intents_flat: Path, intent_embs: Path,
                               spans: Path, anchor_embs: Path | None = None, topk: int = 5) -> Dict[str, float]:
    """Compute extended retrieval metrics: completeness, redundancy, diversity, efficiency."""
    try:
        X = np.load(chunk_embs)
        C = _read_jsonl(chunks)
        Q = np.load(intent_embs)
        Q_rows = _read_jsonl(intents_flat)
        S_rows = _read_jsonl(spans)
        anchors = None
        if anchor_embs is not None and anchor_embs.exists():
            anchors = np.load(anchor_embs)

        # Use anchor embeddings if available, otherwise use chunk embeddings
        if anchors is not None and anchors.shape == X.shape:
            Xn = _l2n(anchors)
        else:
            Xn = _l2n(X)
        Qn = _l2n(Q) if Q.shape[0] > 0 else Q

        # Build chunks lookup
        chunk_docs = [c.get("doc_id") for c in C]
        chunks_by_doc: Dict[str, List[int]] = {}
        for idx, c in enumerate(C):
            chunks_by_doc.setdefault(c["doc_id"], []).append(idx)

        spans_by_key = {}
        for s in S_rows:
            try:
                spans_by_key[(s["doc_id"], int(s["query_id"]))] = s
            except Exception:
                continue

        sims = Qn @ Xn.T
        I = np.argsort(-sims, axis=1)[:, :topk]

        completeness_scores = []
        redundancy_scores = []
        diversity_scores = []
        efficiency_scores = []

        for qi, q in enumerate(Q_rows):
            key = (q.get("doc_id"), int(q.get("query_id", 0)))
            span = spans_by_key.get(key)
            if span is None or not span.get("answerable", True):
                continue

            doc_id = span["doc_id"]
            doc_chunk_indices = chunks_by_doc.get(doc_id, [])

            s0 = int(span["start_sent"])
            s1 = int(span["end_sent"])
            answer_length = s1 - s0 + 1

            # Get top-k retrieved chunks
            cand = list(I[qi])
            if not cand:
                continue

            # Completeness: best overlap with answer
            best_overlap = 0.0
            for chunk_idx in cand[:topk]:
                if chunk_idx >= len(C):
                    continue
                chunk = C[chunk_idx]
                chunk_start = int(chunk.get("start_sent", 0))
                chunk_end = int(chunk.get("end_sent", 0))

                overlap_start = max(chunk_start, s0)
                overlap_end = min(chunk_end, s1)
                overlap_length = max(0, overlap_end - overlap_start + 1)

                completeness = overlap_length / max(answer_length, 1)
                best_overlap = max(best_overlap, completeness)

            completeness_scores.append(best_overlap)

            # Redundancy & Diversity: token overlap in top-k
            chunk_texts = []
            for chunk_idx in cand[:min(topk, 5)]:
                if chunk_idx < len(C):
                    chunk_texts.append(C[chunk_idx].get("text", ""))

            if len(chunk_texts) >= 2:
                # Simple word-based overlap
                import re
                chunk_tokens = []
                for text in chunk_texts:
                    tokens = set(re.findall(r'\b\w+\b', text.lower()))
                    chunk_tokens.append(tokens)

                similarities = []
                for i in range(len(chunk_tokens)):
                    for j in range(i + 1, len(chunk_tokens)):
                        if chunk_tokens[i] and chunk_tokens[j]:
                            intersection = chunk_tokens[i] & chunk_tokens[j]
                            union = chunk_tokens[i] | chunk_tokens[j]
                            if union:
                                sim = len(intersection) / len(union)
                                similarities.append(sim)

                if similarities:
                    redundancy = sum(similarities) / len(similarities)
                    redundancy_scores.append(redundancy)
                    diversity_scores.append(1.0 - redundancy)
                else:
                    redundancy_scores.append(0.0)
                    diversity_scores.append(1.0)
            else:
                redundancy_scores.append(0.0)
                diversity_scores.append(1.0)

            # Efficiency: relevant sentences / total sentences
            total_sents = 0
            relevant_sents = 0
            for chunk_idx in cand[:topk]:
                if chunk_idx < len(C):
                    chunk = C[chunk_idx]
                    chunk_start = int(chunk.get("start_sent", 0))
                    chunk_end = int(chunk.get("end_sent", 0))
                    chunk_length = chunk_end - chunk_start + 1
                    total_sents += chunk_length

                    overlap_start = max(chunk_start, s0)
                    overlap_end = min(chunk_end, s1)
                    overlap_length = max(0, overlap_end - overlap_start + 1)
                    relevant_sents += overlap_length

            efficiency = relevant_sents / max(total_sents, 1)
            efficiency_scores.append(efficiency)

        # Return averages
        n = len(completeness_scores)
        if n == 0:
            return {
                "completeness": 0.0,
                "redundancy": 0.0,
                "diversity": 0.0,
                "efficiency": 0.0
            }

        return {
            "completeness": sum(completeness_scores) / n,
            "redundancy": sum(redundancy_scores) / len(redundancy_scores) if redundancy_scores else 0.0,
            "diversity": sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0,
            "efficiency": sum(efficiency_scores) / n
        }
    except Exception as e:
        return {
            "completeness": 0.0,
            "redundancy": 0.0,
            "diversity": 0.0,
            "efficiency": 0.0
        }

def parse_pipeline_logs(output_dir: Path) -> Dict[str, Dict]:
    """Parse evaluation results from pipeline output files and logs."""
    results = {}
    
    # Method name mappings
    method_mappings = {
        'idc': 'idc',
        'fixed': 'fixed', 
        'sliding': 'sliding',
        'coh': 'coh',
        'paragraphs': 'paragraphs'
    }
    
    # Initialize results structure
    for method in method_mappings.values():
        results[method] = {
            'variant': method,
            'available': False,
            'num_chunks': 0,
            'avg_sentences_per_chunk': 0.0,
            'avg_tokens_per_chunk': 0.0,
            'coverage': {'mean': 0.0, 'ci95': [0.0, 0.0], 'n': 0},
            'doc_mode': {
                'doc_count': 1,
                'hit1': {'mean': 0.0, 'ci95': [0.0, 0.0]},
                'hitK': {'mean': 0.0, 'ci95': [0.0, 0.0]},
                'mrr': {'mean': 0.0, 'ci95': [0.0, 0.0]}
            },
            'span_mode': {
                'evaluated': 0,
                'total_queries': 0,
                'R1': {'mean': 0.0, 'ci95': [0.0, 0.0]},
                'RK': {'mean': 0.0, 'ci95': [0.0, 0.0]},
                'MRR': {'mean': 0.0, 'ci95': [0.0, 0.0]}
            },
            # NEW: Extended metrics
            'extended_metrics': {
                'completeness': {'mean': 0.0, 'ci95': [0.0, 0.0]},
                'redundancy': {'mean': 0.0, 'ci95': [0.0, 0.0]},
                'diversity': {'mean': 0.0, 'ci95': [0.0, 0.0]},
                'efficiency': {'mean': 0.0, 'ci95': [0.0, 0.0]}
            }
        }
    
    # Compute span-mode metrics directly from files (preferred over logs/defaults)
    intents_flat = output_dir / "intents.flat.jsonl"
    intent_embs = _pick_latest("intent_embs*.npy", output_dir)
    spans_file = output_dir / "gt_spans.jsonl"
    
    # Count chunks and compute coverage + metrics from data
    chunk_patterns = {
        'idc': 'chunks.idc.jsonl',
        'fixed': 'chunks.fixed*.jsonl',
        'sliding': 'chunks.sliding*.jsonl', 
        'coh': 'chunks.coh*.jsonl',
        'paragraphs': 'chunks.paragraphs.jsonl'
    }
    
    for method, pattern in chunk_patterns.items():
        chunk_files = list(output_dir.glob(pattern))
        if chunk_files:
            try:
                # Use latest file (by modification time) to match embedding file selection logic
                chunk_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                chunk_file = chunk_files[0]
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunks = [json.loads(line) for line in f if line.strip()]
                
                results[method]['num_chunks'] = len(chunks)
                results[method]['available'] = True
                
                # Calculate average chunk length if not already set
                if results[method]['avg_sentences_per_chunk'] == 0.0 and chunks:
                    lengths = []
                    token_lengths = []
                    for chunk in chunks:
                        if 'num_sentences' in chunk:
                            lengths.append(chunk['num_sentences'])
                        elif 'start_sent' in chunk and 'end_sent' in chunk:
                            lengths.append(int(chunk['end_sent']) - int(chunk['start_sent']) + 1)
                        if 'text' in chunk and isinstance(chunk['text'], str):
                            token_lengths.append(len(TOKEN_ENCODER.encode(chunk['text'])) )

                    if lengths:
                        results[method]['avg_sentences_per_chunk'] = sum(lengths) / len(lengths)
                    if token_lengths:
                        results[method]['avg_tokens_per_chunk'] = sum(token_lengths) / len(token_lengths)
                    else:
                        results[method]['avg_tokens_per_chunk'] = 0.0

                # Coverage from segments if available
                seg_globs = {
                    'idc': 'segments.idc*.jsonl',
                    'fixed': 'segments.fixed*.jsonl',
                    'sliding': 'segments.sliding*.jsonl',
                    'coh': 'segments.coh*.jsonl',
                    'paragraphs': 'segments.paragraphs*.jsonl',
                }
                seg_file = _pick_latest(seg_globs.get(method, ''), output_dir)
                if seg_file and spans_file.exists():
                    cov = _answer_coverage(seg_file, spans_file)
                    results[method]['coverage']['mean'] = cov
                    results[method]['coverage']['ci95'] = [cov, cov]

                # Span-mode retrieval metrics (fast) if we have intent embeddings
                if intent_embs and intents_flat.exists() and spans_file.exists():
                    emb_file = _pick_latest(f"chunk_embs.{method}*.npy", output_dir)
                    if emb_file is not None:
                        anchor_file = _pick_latest(f"chunk_anchor_embs.{method}*.npy", output_dir)
                        m = _compute_span_metrics_fast(emb_file, chunk_file, intents_flat, intent_embs, spans_file, anchor_file, topk=5)
                        results[method]['span_mode']['R1']['mean'] = m['R1']
                        results[method]['span_mode']['RK']['mean'] = m['RK']
                        results[method]['span_mode']['MRR']['mean'] = m['MRR']
                        results[method]['span_mode']['evaluated'] = m['evaluated']
                        results[method]['span_mode']['total_queries'] = m['total_queries']

                        # NEW: Compute extended metrics
                        try:
                            ext = _compute_extended_metrics(emb_file, chunk_file, intents_flat, intent_embs, spans_file, anchor_file, topk=5)
                            results[method]['extended_metrics']['completeness']['mean'] = ext['completeness']
                            results[method]['extended_metrics']['redundancy']['mean'] = ext['redundancy']
                            results[method]['extended_metrics']['diversity']['mean'] = ext['diversity']
                            results[method]['extended_metrics']['efficiency']['mean'] = ext['efficiency']
                        except Exception as e:
                            print(f"Warning: Could not compute extended metrics for {method}: {e}")

            except Exception as e:
                print(f"Warning: Could not parse chunk file for {method}: {e}")
    print(f"Looking for evaluation results in {output_dir}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Generate stats.json for QT app")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory containing evaluation results")
    parser.add_argument("--methods", type=str, default="idc fixed sliding coh paragraphs", help="Space-separated list of methods")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    methods = args.methods.split()
    
    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        sys.exit(1)
    
    print(f"Generating stats.json for methods: {methods}")
    print(f"Output directory: {output_dir}")
    
    # Parse results
    results = parse_pipeline_logs(output_dir)
    
    # Filter to requested methods and convert to list format expected by QT app
    stats_list = []
    for method in methods:
        if method in results and results[method]['available']:
            stats_list.append(results[method])
            print(f"✓ {method}: {results[method]['num_chunks']} chunks, "
                  f"coverage={results[method]['coverage']['mean']:.3f}, "
                  f"R@1={results[method]['span_mode']['R1']['mean']:.3f}, "
                  f"AvgLen={results[method]['avg_sentences_per_chunk']:.2f}, "
                  f"AvgTok={results[method]['avg_tokens_per_chunk']:.0f}")
        else:
            print(f"✗ {method}: No data found")
    
    if not stats_list:
        print("Warning: No valid results found. Creating minimal stats file.")
        # Create minimal entry so QT app doesn't crash
        stats_list = [{
            "variant": "idc",
            "available": False,
            "num_chunks": 0,
            "avg_sentences_per_chunk": 0.0,
            "avg_tokens_per_chunk": 0.0,
            "coverage": {"mean": 0.0, "ci95": [0.0, 0.0], "n": 0},
            "doc_mode": {
                "doc_count": 1,
                "hit1": {"mean": 0.0, "ci95": [0.0, 0.0]},
                "hitK": {"mean": 0.0, "ci95": [0.0, 0.0]},
                "mrr": {"mean": 0.0, "ci95": [0.0, 0.0]}
            },
            "span_mode": {
                "evaluated": 0,
                "total_queries": 0,
                "R1": {"mean": 0.0, "ci95": [0.0, 0.0]},
                "RK": {"mean": 0.0, "ci95": [0.0, 0.0]},
                "MRR": {"mean": 0.0, "ci95": [0.0, 0.0]}
            }
        }]
    
    # Write stats.json in format expected by QT app
    stats_file = output_dir / "stats.json"
    stats_data = {"results": stats_list}
    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)
    
    print(f"✅ Generated {stats_file}")
    print(f"📊 Ready for QT app visualization!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
