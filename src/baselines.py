#!/usr/bin/env python3
"""
Baseline segmenters for comparison with IDC.

Outputs the same structure as IDC's segments file:
[
  {"doc_id": "...", "chunks": [
      {"start_sent": 1, "end_sent": 3, "num_sentences": 3, "intent": null, "similarity": null, "text": "..."},
      ...
  ]},
  ...
]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    _HAVE_NLTK = True
except Exception:
    _HAVE_NLTK = False

def read_jsonl(path: str | Path) -> List[Dict]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def write_jsonl(path: str | Path, rows: List[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def group_sentences(sent_rows: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Returns: doc_id -> list of rows sorted by sent_id (each row has text, sent_id)
    """
    buckets: Dict[str, List[Tuple[int, Dict]]] = {}
    for r in sent_rows:
        buckets.setdefault(r["doc_id"], []).append((int(r["sent_id"]), r))
    out: Dict[str, List[Dict]] = {}
    for k, pairs in buckets.items():
        pairs.sort(key=lambda x: x[0])
        out[k] = [row for _, row in pairs]
    return out

# ---------------- Fixed-length ----------------
def segment_fixed(sents: List[Dict], target_len: int) -> List[Dict]:
    chunks = []
    n = len(sents)
    i = 0
    while i < n:
        j = min(n, i + target_len)
        start = sents[i]["sent_id"]
        end = sents[j-1]["sent_id"]
        text = " ".join([s["text"] for s in sents[i:j]])
        chunks.append({
            "start_sent": start,
            "end_sent": end,
            "num_sentences": int(end - start + 1),
            "intent": None,
            "similarity": None,
            "text": text
        })
        i = j
    return chunks

# ---------------- Sliding window ----------------
def segment_sliding(sents: List[Dict], size: int, stride: int) -> List[Dict]:
    chunks = []
    n = len(sents)
    i = 0
    while i < n:
        j = min(n, i + size)
        if i >= j:
            break
        start = sents[i]["sent_id"]
        end = sents[j-1]["sent_id"]
        text = " ".join([s["text"] for s in sents[i:j]])
        chunks.append({
            "start_sent": start,
            "end_sent": end,
            "num_sentences": int(end - start + 1),
            "intent": None,
            "similarity": None,
            "text": text
        })
        if j == n:
            break
        i += stride
    return chunks

# ---------------- Paragraph-based ----------------
def _split_paragraphs_blanklines(raw_text: str) -> List[str]:
    # split on blank lines as paragraphs
    paras = []
    cur = []
    for line in raw_text.splitlines():
        if line.strip() == "":
            if cur:
                paras.append(" ".join(cur).strip())
                cur = []
        else:
            cur.append(line.strip())
    if cur:
        paras.append(" ".join(cur).strip())
    return [p for p in paras if p]

def segment_paragraphs(
    sents: List[Dict],
    raw_text: str | None = None,
) -> List[Dict]:
    """
    If raw_text is provided and nltk is available, we use blank line paragraphs
    and sent_tokenize each paragraph to count sentences -> align to sents order.
    Otherwise, fallback: treat each sentence as a paragraph (not ideal).
    """
    if raw_text and _HAVE_NLTK:
        # Ensure punkt
        try:
            _ = sent_tokenize("Test.")
        except LookupError:
            try:
                nltk.download("punkt")
            except Exception:
                nltk.download("punkt_tab")
        paras = _split_paragraphs_blanklines(raw_text)
        # Count sentences per paragraph using the same tokenizer
        para_sizes = [len(sent_tokenize(p)) for p in paras] if paras else []
        if len(para_sizes) > 0:
            chunks = []
            cursor = 0
            for m in para_sizes:
                i = cursor
                j = min(len(sents), cursor + max(1, m))
                if i >= j:
                    break
                start = sents[i]["sent_id"]
                end = sents[j-1]["sent_id"]
                text = " ".join([s["text"] for s in sents[i:j]])
                chunks.append({
                    "start_sent": start,
                    "end_sent": end,
                    "num_sentences": int(end - start + 1),
                    "intent": None,
                    "similarity": None,
                    "text": text
                })
                cursor = j
            # absorb any leftover sentences into a final chunk
            if cursor < len(sents):
                i, j = cursor, len(sents)
                start = sents[i]["sent_id"]
                end = sents[j-1]["sent_id"]
                text = " ".join([s["text"] for s in sents[i:j]])
                chunks.append({
                    "start_sent": start,
                    "end_sent": end,
                    "num_sentences": int(end - start + 1),
                    "intent": None,
                    "similarity": None,
                    "text": text
                })
            return chunks
    # Fallback only if we truly couldn't detect paragraphs
    return segment_fixed(sents, target_len=6)

# ---------------- Coherence/TextTiling-like ----------------
def segment_coherence(
    sents: List[Dict],
    S_doc: np.ndarray,      # sentence embeddings in document order (N, D)
    win: int = 2,           # window size on each side
    min_len: int = 1,       # min sentences per chunk
    max_len: int = 12,      # max sentences per chunk
    approx_chunk_len: int = 6,  # to estimate number of boundaries
) -> List[Dict]:
    """
    Simple TextTiling-like segmentation using embedding valleys.

    1) For each boundary i (between sent i and i+1), compute cosine similarity
       between mean of left window [i-win+1..i] and right window [i+1..i+win].
    2) Rank boundaries by ascending similarity (valleys first).
    3) Greedily insert boundaries while satisfying [min_len, max_len].
    """
    N = S_doc.shape[0]
    if N == 0:
        return []

    def mean_vec(a: int, b: int) -> np.ndarray:
        # clamp to valid indices
        a = max(a, 0)
        b = min(b, N-1)
        if a > b:
            a, b = b, a
        return S_doc[a:b+1].mean(axis=0)

    sims = []
    for i in range(N - 1):
        left = mean_vec(i - win + 1, i)
        right = mean_vec(i + 1, i + win)
        # cosine
        ln = left / (np.linalg.norm(left) + 1e-8)
        rn = right / (np.linalg.norm(right) + 1e-8)
        sim = float(np.dot(ln, rn))
        sims.append((i, sim))  # boundary after i

    # how many boundaries do we want? ~ N / approx_chunk_len
    nb = max(0, int(round(N / max(approx_chunk_len, 1))) - 1)
    # sort candidates by lowest similarity
    sims.sort(key=lambda x: x[1])

    # Greedy insert boundaries that satisfy size constraints
    boundaries = []
    def ok_with(bnds: List[int], new_b: int) -> bool:
        """
        Check only the completed segments up to new_b are within [min_len, max_len].
        For the tail (new_b+1..N-1), just ensure there's enough room (>= min_len)
        to continue placing more boundaries later.
        """
        tmp = sorted(bnds + [new_b])
        prev = 0
        for b in tmp:
            L = (b - prev + 1)
            if L < min_len or L > max_len:
                return False
            prev = b + 1
        tail_len = (N - prev)
        return tail_len >= min_len  # can be further split later

    for b, _ in sims:
        if len(boundaries) >= nb:
            break
        if ok_with(boundaries, b):
            boundaries.append(b)

    boundaries = sorted(boundaries)

    # Enforce max_len post-hoc by splitting any overlong segment uniformly.
    fixed = []
    prev = 0
    for b in boundaries + [N - 1]:
        seg_len = (b - prev + 1)
        if seg_len <= max_len:
            fixed.append(b)
            prev = b + 1
            continue
        # Split this long region into subsegments of length <= max_len
        cur = prev
        while (b - cur + 1) > max_len:
            split = cur + max_len - 1
            if split < N - 1:  # only add valid boundaries
                fixed.append(split)
            cur = split + 1
        if b < N:  # only add valid boundaries
            fixed.append(b)
        prev = b + 1
    # Remove duplicates and ensure boundaries are in valid range
    boundaries = sorted(list(set([b for b in fixed if 0 <= b < N - 1])))

    # Build chunks from (possibly adjusted) boundaries
    chunks = []
    starts = [0] + [b + 1 for b in boundaries]
    ends = boundaries + [N - 1]
    for i, j in zip(starts, ends):
        start = sents[i]["sent_id"]
        end = sents[j]["sent_id"]
        text = " ".join([s["text"] for s in sents[i:j+1]])
        chunks.append({
            "start_sent": start,
            "end_sent": end,
            "num_sentences": int(end - start + 1),
            "intent": None,
            "similarity": None,
            "text": text
        })
    return chunks

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Baseline segmenters")
    ap.add_argument("--sentences", type=str, default="out/sentences.jsonl", help="Sentences JSONL")
    ap.add_argument("--sentence-embs", type=str, help="Required for 'coherence'")
    ap.add_argument("--input-dir", type=str, help="Optional: raw docs dir (for paragraphs)")
    ap.add_argument("--glob", type=str, default="*.txt", help="Glob for input-dir")
    ap.add_argument("--out", type=str, required=True, help="Output segments JSONL")

    sub = ap.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("fixed")
    p1.add_argument("--target-len", type=int, default=6, help="sentences per chunk")

    p2 = sub.add_parser("sliding")
    p2.add_argument("--size", type=int, default=6, help="window size in sentences")
    p2.add_argument("--stride", type=int, default=3, help="stride in sentences")

    p3 = sub.add_parser("paragraphs")

    p4 = sub.add_parser("coherence")
    p4.add_argument("--win", type=int, default=2)
    p4.add_argument("--min-len", type=int, default=1)
    p4.add_argument("--max-len", type=int, default=12)
    p4.add_argument("--approx-chunk-len", type=int, default=6)

    args = ap.parse_args()

    # Load sentences (required)
    sent_rows = read_jsonl(args.sentences)
    by_doc = group_sentences(sent_rows)

    # Optionally load embeddings (for coherence)
    S_all = None
    if args.mode == "coherence":
        assert args.sentence_embs, "--sentence-embs is required for coherence mode"
        S_all = np.load(args.sentence_embs)

        # Map global row index for easy slicing
        # Build global order index (in the same order as sentences.jsonl)
        # We'll reconstruct per-doc S_doc slices via indices:
        global_indices_by_doc: Dict[str, List[int]] = {}
        cursor = 0
        # Build a stable index map by re-walking sent_rows
        idx_map: Dict[Tuple[str,int], int] = {}
        for i, r in enumerate(sent_rows):
            idx_map[(r["doc_id"], int(r["sent_id"]))] = i

    # Optionally load raw texts for paragraphs
    raw_map: Dict[str, str] = {}
    if args.mode == "paragraphs" and args.input_dir:
        root = Path(args.input_dir)
        for p in sorted(root.rglob(args.glob)):
            if p.is_file():
                raw_map[p.stem] = p.read_text(encoding="utf-8", errors="ignore")

    outputs: List[Dict] = []

    for doc_id, sents in by_doc.items():
        if args.mode == "fixed":
            chunks = segment_fixed(sents, target_len=args.target_len)

        elif args.mode == "sliding":
            chunks = segment_sliding(sents, size=args.size, stride=args.stride)

        elif args.mode == "paragraphs":
            raw_text = raw_map.get(doc_id)
            chunks = segment_paragraphs(sents, raw_text=raw_text)

        elif args.mode == "coherence":
            # Slice this doc's sentence vectors in order
            idxs = [idx_map[(doc_id, int(s["sent_id"]))] for s in sents]
            S_doc = S_all[np.array(idxs)]
            chunks = segment_coherence(
                sents, S_doc,
                win=args.win, min_len=args.min_len, max_len=args.max_len,
                approx_chunk_len=args.approx_chunk_len
            )
        else:
            raise ValueError("Unknown mode")

        outputs.append({"doc_id": doc_id, "chunks": chunks})

    write_jsonl(args.out, outputs)
    print(f"Saved baseline segments → {args.out}")

if __name__ == "__main__":
    main()
