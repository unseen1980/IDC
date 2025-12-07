#!/usr/bin/env python3
"""Convert NewsQA summarization JSONL into IDC inputs + gold spans."""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import nltk
from nltk.tokenize import sent_tokenize


def ensure_nltk_resources() -> None:
    try:
        _ = sent_tokenize("Test.")
    except LookupError:
        try:
            nltk.download("punkt")
        except Exception:  # pragma: no cover - fallback for stripped punkt
            nltk.download("punkt_tab")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def sentence_offsets(raw_text: str, sentences: Sequence[str]) -> List[int]:
    offsets: List[int] = []
    cursor = 0
    lower_text = raw_text.lower()
    for sent in sentences:
        norm_sent = sent.lower()
        pos = lower_text.find(norm_sent, cursor)
        if pos == -1:
            pos = lower_text.find(norm_sent)
        if pos == -1:
            pos = cursor
        offsets.append(pos)
        cursor = pos + len(norm_sent)
    return offsets


def answer_span_indices(
    raw_text: str,
    sentences: Sequence[str],
    offsets: Sequence[int],
    answer_variants: Iterable[str],
) -> Set[int]:
    indices: Set[int] = set()
    if not answer_variants:
        return indices

    lower_text = raw_text.lower()
    sent_count = len(sentences)
    sentence_lengths = [len(s.lower()) for s in sentences]

    for variant in answer_variants:
        variant = (variant or "").strip()
        if not variant:
            continue
        lower_variant = variant.lower()
        start = lower_text.find(lower_variant)
        if start == -1:
            continue
        end = start + len(lower_variant) - 1

        start_idx = max(bisect_right(offsets, start) - 1, 0)
        end_idx = max(bisect_right(offsets, end) - 1, 0)

        # Make sure indices are within bounds and adjust if answer spills over
        start_idx = min(max(start_idx, 0), sent_count - 1)
        end_idx = min(max(end_idx, start_idx), sent_count - 1)

        indices.update(range(start_idx, end_idx + 1))

    return indices


def convert_newsqa(jsonl_path: Path, limit: Optional[int]) -> List[str]:
    ensure_nltk_resources()
    doc_ids: List[str] = []

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and len(doc_ids) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            story = (record.get("story") or "").strip()
            questions = record.get("questions") or []
            answers = record.get("answers") or []

            if not story:
                continue

            doc_id = f"newsqa_{idx+1:05d}"

            sentences = [s.strip() for s in sent_tokenize(story) if s.strip()]
            if not sentences:
                sentences = [story]

            offsets = sentence_offsets(story, sentences)

            write_text(Path("data/input") / f"{doc_id}.txt", "\n\n".join(sentences))

            intents = {
                "doc_id": doc_id,
                "predicted_queries": [str(q).strip() for q in questions],
            }

            out_dir = Path("out") / doc_id
            out_dir.mkdir(parents=True, exist_ok=True)
            write_text(out_dir / "predicted_intents.jsonl", json.dumps(intents, ensure_ascii=False) + "\n")

            span_lines: List[str] = []
            for qi, q in enumerate(questions, start=1):
                ans_variants = answers[qi - 1] if qi - 1 < len(answers) else []
                indices = answer_span_indices(story, sentences, offsets, ans_variants)
                if indices:
                    span_lines.append(
                        json.dumps(
                            {
                                "query_id": qi,
                                "doc_id": doc_id,
                                "start_sent": min(indices) + 1,
                                "end_sent": max(indices) + 1,
                                "answerable": True,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                else:
                    span_lines.append(
                        json.dumps(
                            {
                                "query_id": qi,
                                "doc_id": doc_id,
                                "answerable": False,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            write_text(out_dir / "gt_spans.jsonl", "".join(span_lines))

            summary = record.get("summary")
            if summary:
                write_text(out_dir / "summary.ref.txt", summary.strip())

            doc_ids.append(doc_id)

    return doc_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert NewsQA summarization to IDC inputs + spans")
    ap.add_argument("--data", required=True, help="Path to news-qa-summarization data.jsonl")
    ap.add_argument("--doc-ids-out", type=str, default=None, help="Optional path to write doc ids")
    ap.add_argument("--limit", type=int, default=0, help="Maximum docs to convert (0 = all)")
    args = ap.parse_args()

    src = Path(args.data)
    assert src.exists(), f"Not found: {src}"

    limit = args.limit if args.limit and args.limit > 0 else None
    doc_ids = convert_newsqa(src, limit)

    if args.doc_ids_out:
        out_path = Path(args.doc_ids_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(doc_ids) + "\n", encoding="utf-8")

    print(f"Converted NewsQA: {len(doc_ids)} docs")


if __name__ == "__main__":
    main()

