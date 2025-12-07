#!/usr/bin/env python3
"""
Auto-tune baseline segmenters (fixed, sliding, coherence, paragraphs) on a dev spans file.

Objective: maximize answer coverage; optionally prefer configs that match a
target average sentences-per-chunk within a tolerance.

Outputs tuned segments to the provided output paths and writes a short report.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Import baseline helpers
from baselines import (
    read_jsonl as read_jsonl_baseline,
    write_jsonl as write_jsonl_baseline,
    group_sentences,
    segment_fixed,
    segment_sliding,
    segment_paragraphs,
    segment_coherence,
)


def read_jsonl(path: str | Path) -> List[Dict]:
    out: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def answer_coverage(segments: List[Dict], spans_path: str | Path) -> float:
    """Compute answer coverage (fraction of answerable spans fully contained by one chunk)."""
    spans = read_jsonl(spans_path)
    chunks_by_doc: Dict[str, List[Dict]] = {d["doc_id"]: d["chunks"] for d in segments}
    ok = 0
    total = 0
    for q in spans:
        if q.get("answerable") is False:
            continue
        total += 1
        doc_id = q["doc_id"]
        s0 = int(q["start_sent"]) if "start_sent" in q else None
        s1 = int(q["end_sent"]) if "end_sent" in q else None
        if s0 is None or s1 is None:
            continue
        chunks = chunks_by_doc.get(doc_id, [])
        covered = False
        for ch in chunks:
            if int(ch["start_sent"]) <= s0 and int(ch["end_sent"]) >= s1:
                covered = True
                break
        if covered:
            ok += 1
    return ok / max(total, 1)


def avg_sentences_per_chunk(segments: List[Dict]) -> float:
    lens: List[int] = []
    for doc in segments:
        for ch in doc.get("chunks", []):
            lens.append(int(ch.get("num_sentences", 0)))
    return float(np.mean(lens)) if lens else 0.0


def build_idx_map(sent_rows: List[Dict]) -> Dict[Tuple[str, int], int]:
    idx_map: Dict[Tuple[str, int], int] = {}
    for i, r in enumerate(sent_rows):
        idx_map[(r["doc_id"], int(r["sent_id"]))] = i
    return idx_map


def load_raw_map(input_dir: Optional[str], glob_pat: str) -> Dict[str, str]:
    raw_map: Dict[str, str] = {}
    if input_dir:
        root = Path(input_dir)
        for p in sorted(root.rglob(glob_pat)):
            if p.is_file():
                raw_map[p.stem] = p.read_text(encoding="utf-8", errors="ignore")
    return raw_map


def segment_all_docs_fixed(by_doc: Dict[str, List[Dict]], target_len: int) -> List[Dict]:
    out: List[Dict] = []
    for doc_id, sents in by_doc.items():
        chunks = segment_fixed(sents, target_len=target_len)
        out.append({"doc_id": doc_id, "chunks": chunks})
    return out


def segment_all_docs_sliding(by_doc: Dict[str, List[Dict]], size: int, stride: int) -> List[Dict]:
    out: List[Dict] = []
    for doc_id, sents in by_doc.items():
        chunks = segment_sliding(sents, size=size, stride=stride)
        out.append({"doc_id": doc_id, "chunks": chunks})
    return out


def segment_all_docs_paragraphs(by_doc: Dict[str, List[Dict]], raw_map: Dict[str, str]) -> List[Dict]:
    out: List[Dict] = []
    for doc_id, sents in by_doc.items():
        raw_text = raw_map.get(doc_id)
        chunks = segment_paragraphs(sents, raw_text=raw_text)
        out.append({"doc_id": doc_id, "chunks": chunks})
    return out


def segment_all_docs_coherence(
    by_doc: Dict[str, List[Dict]],
    S_all: np.ndarray,
    idx_map: Dict[Tuple[str, int], int],
    win: int,
    min_len: int,
    max_len: int,
    approx_chunk_len: int,
) -> List[Dict]:
    out: List[Dict] = []
    for doc_id, sents in by_doc.items():
        idxs = [idx_map[(doc_id, int(s["sent_id"]))] for s in sents]
        S_doc = S_all[np.array(idxs)]
        chunks = segment_coherence(
            sents, S_doc, win=win, min_len=min_len, max_len=max_len, approx_chunk_len=approx_chunk_len
        )
        out.append({"doc_id": doc_id, "chunks": chunks})
    return out


@dataclass
class TunedResult:
    name: str
    params: Dict[str, int]
    coverage: float
    avg_len: float
    segments: List[Dict]


def pick_best(results: List[TunedResult], target_avg: Optional[float], tolerance: Optional[float]) -> TunedResult:
    if not results:
        raise ValueError("No results to choose from")
    if target_avg is None:
        # simply choose highest coverage; tie-breaker: closest to 6
        results = sorted(results, key=lambda r: (r.coverage, -abs(r.avg_len - 6.0)), reverse=True)
        return results[0]
    # Filter by tolerance if provided
    if tolerance is not None:
        eligible = [r for r in results if abs(r.avg_len - target_avg) <= tolerance]
        if eligible:
            return sorted(eligible, key=lambda r: (r.coverage, -abs(r.avg_len - target_avg)), reverse=True)[0]
    # Otherwise, pick by smallest distance to target, then coverage
    return sorted(results, key=lambda r: (-min(0.0, tolerance or 0.0), abs(r.avg_len - target_avg), -r.coverage))[0]


def main():
    ap = argparse.ArgumentParser(description="Auto-tune baseline segmenters against coverage on dev spans")
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--sentence-embs", help="Required for coherence baseline")
    ap.add_argument("--input-dir", help="Raw docs dir for paragraphs baseline")
    ap.add_argument("--glob", default="*.txt")
    ap.add_argument("--spans", required=True, help="Dev spans JSONL")

    # Target outputs
    ap.add_argument("--out-fixed", required=True)
    ap.add_argument("--out-sliding", required=True)
    ap.add_argument("--out-coherence", required=True)
    ap.add_argument("--out-paragraphs", required=True)
    ap.add_argument("--report", default=None)

    # Optional target average sentences per chunk
    ap.add_argument("--target-avg", type=float, help="Target average sentences/chunk to match (optional)")
    ap.add_argument("--tolerance", type=float, default=None, help="Allowable deviation from target-avg in sentences")

    args = ap.parse_args()

    sent_rows = read_jsonl_baseline(args.sentences)
    by_doc = group_sentences(sent_rows)

    # Prepare optional resources
    idx_map: Optional[Dict[Tuple[str, int], int]] = None
    S_all: Optional[np.ndarray] = None
    if args.sentence_embs:
        S_all = np.load(args.sentence_embs)
        idx_map = build_idx_map(sent_rows)
    raw_map = load_raw_map(args.input_dir, args.glob) if args.input_dir else {}

    # 1) Fixed grid
    fixed_grid = [4, 6, 8, 10]
    fixed_results: List[TunedResult] = []
    for L in fixed_grid:
        segs = segment_all_docs_fixed(by_doc, target_len=L)
        cov = answer_coverage(segs, args.spans)
        av = avg_sentences_per_chunk(segs)
        fixed_results.append(TunedResult("fixed", {"target_len": L}, cov, av, segs))
    best_fixed = pick_best(fixed_results, args.target_avg, args.tolerance)

    # 2) Sliding grid
    sliding_sizes = [4, 6, 8, 10]
    sliding_results: List[TunedResult] = []
    for size in sliding_sizes:
        strides = sorted(set([max(1, size // 2), max(1, size // 3)]))
        for stride in strides:
            segs = segment_all_docs_sliding(by_doc, size=size, stride=stride)
            cov = answer_coverage(segs, args.spans)
            av = avg_sentences_per_chunk(segs)
            sliding_results.append(TunedResult("sliding", {"size": size, "stride": stride}, cov, av, segs))
    best_sliding = pick_best(sliding_results, args.target_avg, args.tolerance)

    # 3) Coherence grid (requires embeddings)
    coh_results: List[TunedResult] = []
    if S_all is not None and idx_map is not None:
        for win in [1, 2, 3]:
            for approx in [5, 6, 8]:
                segs = segment_all_docs_coherence(
                    by_doc, S_all=S_all, idx_map=idx_map,
                    win=win, min_len=1, max_len=12, approx_chunk_len=approx
                )
                cov = answer_coverage(segs, args.spans)
                av = avg_sentences_per_chunk(segs)
                coh_results.append(TunedResult("coherence", {"win": win, "approx_chunk_len": approx}, cov, av, segs))
        best_coh = pick_best(coh_results, args.target_avg, args.tolerance)
    else:
        # Degenerate: no embeddings → skip coherence
        best_coh = TunedResult("coherence", {}, 0.0, 0.0, [])

    # 4) Paragraphs (no tuning)
    para_segs = segment_all_docs_paragraphs(by_doc, raw_map=raw_map)
    para_cov = answer_coverage(para_segs, args.spans)
    para_av = avg_sentences_per_chunk(para_segs)
    best_para = TunedResult("paragraphs", {}, para_cov, para_av, para_segs)

    # Write outputs
    write_jsonl_baseline(args.out_fixed, best_fixed.segments)
    write_jsonl_baseline(args.out_sliding, best_sliding.segments)
    if best_coh.segments:
        write_jsonl_baseline(args.out_coherence, best_coh.segments)
    else:
        # Ensure an empty file exists for coherence if it was skipped
        Path(args.out_coherence).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_coherence).write_text("", encoding="utf-8")
    write_jsonl_baseline(args.out_paragraphs, best_para.segments)

    report = {
        "fixed": {"params": best_fixed.params, "coverage": best_fixed.coverage, "avg_len": best_fixed.avg_len},
        "sliding": {"params": best_sliding.params, "coverage": best_sliding.coverage, "avg_len": best_sliding.avg_len},
        "coherence": {"params": best_coh.params, "coverage": best_coh.coverage, "avg_len": best_coh.avg_len},
        "paragraphs": {"params": best_para.params, "coverage": best_para.coverage, "avg_len": best_para.avg_len},
    }
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

