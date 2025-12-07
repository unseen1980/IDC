#!/usr/bin/env python3
"""
Answer coverage: proportion of gold spans fully contained within a single chunk.

Inputs:
  --segments  segments.jsonl      (IDC or baseline)
  --spans     gt_spans.jsonl      rows: {query_id, doc_id, start_sent, end_sent}
Outputs:
  Prints coverage %. Optionally writes per-query diagnostics.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

def read_jsonl(path: str | Path) -> List[Dict]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def main():
    ap = argparse.ArgumentParser(description="Answer coverage")
    ap.add_argument("--segments", type=str, required=True)
    ap.add_argument("--spans", type=str, required=True)
    ap.add_argument("--diagnostics", type=str, help="Optional: write mismatches to JSONL")
    args = ap.parse_args()

    seg_docs = read_jsonl(args.segments)   # [{doc_id, chunks:[...]}]
    spans = read_jsonl(args.spans)         # [{query_id, doc_id, start_sent, end_sent}]

    chunks_by_doc = {}
    for doc in seg_docs:
        chunks_by_doc[doc["doc_id"]] = doc["chunks"]

    ok = 0
    total = 0
    bad_rows = []
    for q in spans:
        if q.get("answerable") is False:
            continue
        total += 1
        doc_id = q["doc_id"]
        s0 = int(q["start_sent"])
        s1 = int(q["end_sent"])
        chunks = chunks_by_doc.get(doc_id, [])
        covered = False
        for ch in chunks:
            if int(ch["start_sent"]) <= s0 and int(ch["end_sent"]) >= s1:
                covered = True
                break
        if covered:
            ok += 1
        else:
            bad_rows.append(q)

    cov = ok / max(total, 1)
    print(f"Answer coverage: {ok}/{total} = {cov:.3f}")

if __name__ == "__main__":
    main()
