#!/usr/bin/env python3
import argparse
from typing import List, Dict
from pathlib import Path

import nltk
from nltk.tokenize import sent_tokenize

from utils import read_text_files, write_jsonl, ensure_dir


def ensure_nltk_resources() -> None:
    try:
        _ = sent_tokenize("Test.")
    except LookupError:
        try:
            nltk.download("punkt")
        except Exception:
            nltk.download("punkt_tab")


def preprocess_document(doc_text: str, doc_id: str) -> List[Dict]:
    sentences = sent_tokenize(doc_text)
    rows = []
    for i, s in enumerate(sentences, start=1):
        s = s.strip()
        if not s:
            continue
        rows.append({"doc_id": doc_id, "sent_id": i, "text": s})
    return rows


def main():
    ap = argparse.ArgumentParser(description="IDC: Preprocess documents into sentences.jsonl")
    ap.add_argument("--input", type=str, default="data/input", help="Input directory of .txt files")
    ap.add_argument("--out", type=str, default="out/sentences.jsonl", help="Output JSONL path")
    ap.add_argument("--glob", type=str, default="*.txt", help="Glob for input files")
    args = ap.parse_args()

    ensure_dir("out")
    ensure_nltk_resources()

    docs = read_text_files(args.input, args.glob)
    total_sentences = 0
    all_rows: List[Dict] = []

    for d in docs:
        rows = preprocess_document(d["text"], d["doc_id"])
        all_rows.extend(rows)
        total_sentences += len(rows)
        print(f"{d['doc_id']}: {len(rows)} sentences")

    write_jsonl(args.out, all_rows)
    print(f"\nSaved {total_sentences} sentences → {args.out}")


if __name__ == "__main__":
    main()
