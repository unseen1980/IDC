#!/usr/bin/env python3
"""
Merge multiple per-document outputs into a single multi-doc corpus for evaluation.

Inputs: a list of doc_ids (matching out/<doc_id>/ ... files) and a method variant.
Outputs:
  - Combined chunks JSONL
  - Combined chunk embeddings NPY
  - Combined intents.flat.jsonl
  - Combined gt_spans.jsonl

Usage:
  python src/merge_corpus.py \
    --docs-file data/squad/doc_ids.txt \
    --variant paragraphs --dim 1536 \
    --out-root out/squad

Then evaluate across docs:
  python src/eval_retrieval.py --chunk-embs out/squad/chunk_embs.paragraphs.d1536.npy \
    --chunks out/squad/chunks.paragraphs.jsonl --queries out/squad/intents.flat.jsonl \
    --embedder gemini-embedding-001 --dim 1536 --mode doc --topk 5

  python src/eval_retrieval.py --chunk-embs out/squad/chunk_embs.paragraphs.d1536.npy \
    --chunks out/squad/chunks.paragraphs.jsonl --queries out/squad/intents.flat.jsonl \
    --spans out/squad/gt_spans.jsonl --embedder gemini-embedding-001 --dim 1536 --mode span --topk 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np


def read_jsonl(path: str | Path) -> List[Dict]:
    out: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: List[Dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_docs_list(docfile: Path) -> List[str]:
    ids = [ln.strip() for ln in docfile.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return ids


def main():
    ap = argparse.ArgumentParser(description="Merge multiple doc outputs into a single corpus")
    ap.add_argument("--docs-file", required=True, help="Text file with one doc_id per line")
    ap.add_argument("--variant", required=True, choices=["idc", "fixed", "sliding", "coh", "paragraphs"])
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--out-root", required=True, help="Output directory for merged artifacts (e.g., out/squad)")
    args = ap.parse_args()

    doc_ids = load_docs_list(Path(args.docs_file))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Detect per-variant file patterns
    if args.variant == "idc":
        chunk_name = "chunks.idc.jsonl"
        emb_name = f"chunk_embs.idc.d{args.dim}.npy"
        seg_name = None
    elif args.variant == "fixed":
        chunk_glob = "chunks.fixed.*.jsonl"
        emb_glob = f"chunk_embs.fixed.*.d{args.dim}.npy"
    elif args.variant == "sliding":
        chunk_glob = "chunks.sliding.*.jsonl"
        emb_glob = f"chunk_embs.sliding.*.d{args.dim}.npy"
    elif args.variant == "coh":
        chunk_glob = "chunks.coh.*.jsonl"
        emb_glob = f"chunk_embs.coh.*.d{args.dim}.npy"
    else:  # paragraphs
        chunk_name = "chunks.paragraphs.jsonl"
        emb_name = f"chunk_embs.paragraphs.d{args.dim}.npy"

    all_chunks: List[Dict] = []
    all_Q: List[Dict] = []
    all_spans: List[Dict] = []
    arrays: List[np.ndarray] = []

    for d in doc_ids:
        doc_dir = Path("out") / d
        # chunks/embs
        if args.variant in {"idc", "paragraphs"}:
            chunks_path = doc_dir / (chunk_name)
            embs_path = doc_dir / (emb_name)
        else:
            # pick newest matching
            cands_c = sorted(doc_dir.glob(chunk_glob), key=lambda p: p.stat().st_mtime)
            cands_e = sorted(doc_dir.glob(emb_glob), key=lambda p: p.stat().st_mtime)
            if not cands_c or not cands_e:
                continue
            chunks_path = cands_c[-1]
            embs_path = cands_e[-1]

        if chunks_path.exists() and embs_path.exists():
            all_chunks += read_jsonl(chunks_path)
            arrays.append(np.load(embs_path))

        # queries and spans
        intents_flat = doc_dir / "intents.flat.jsonl"
        if intents_flat.exists():
            all_Q += read_jsonl(intents_flat)
        spans = doc_dir / "gt_spans.jsonl"
        if spans.exists():
            all_spans += read_jsonl(spans)

    if not arrays:
        raise SystemExit("No arrays found to merge. Check docs list and variant.")

    E = np.vstack(arrays)
    out_embs = out_root / f"chunk_embs.{args.variant}.d{args.dim}.npy"
    np.save(out_embs, E)
    out_chunks = out_root / f"chunks.{args.variant}.jsonl"
    write_jsonl(out_chunks, all_chunks)

    out_Q = out_root / "intents.flat.jsonl"
    write_jsonl(out_Q, all_Q)
    out_S = out_root / "gt_spans.jsonl"
    write_jsonl(out_S, all_spans)
    print(f"Merged {len(doc_ids)} docs → {out_embs} | {out_chunks} | {out_Q} | {out_S}")


if __name__ == "__main__":
    main()

