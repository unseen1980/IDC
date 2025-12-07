#!/usr/bin/env python3
"""
Convert SQuAD v1.1 JSON into IDC inputs:

- Writes one .txt per article (concatenated paragraphs) to data/input/<doc_id>.txt
- Writes predicted intents per article to out/<doc_id>/predicted_intents.jsonl
- Writes gold spans per article to out/<doc_id>/gt_spans.jsonl (sentence indices)

Assumptions:
- Sentence splitting uses NLTK sent_tokenize, matching preprocess.py to keep indices consistent.
- We map paragraph-local char offsets to global doc offsets by joining paragraphs with two newlines. This preserves all
  original paragraph text and ensures deterministic offsets.

Usage:
  python src/convert_squad.py --squad data/squad/train-v1.1.json --limit 50
  python src/convert_squad.py --squad data/squad/dev-v1.1.json --limit 50

Then run the pipeline per doc_id via src/cli.py menu or:
  DOC_NAME=<doc_id> AUTO_TUNE=1 AUTO_TUNE_BASELINES=1 FORCE_SPANS=0 ./scripts/run_idc_pipeline.sh

Set FORCE_SPANS=0 to keep gold spans.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import nltk
from nltk.tokenize import sent_tokenize


def ensure_nltk():
    try:
        _ = sent_tokenize("Test.")
    except LookupError:
        try:
            nltk.download("punkt")
        except Exception:
            nltk.download("punkt_tab")


def slugify(title: str) -> str:
    s = title.strip().replace(" ", "_")
    valid = [c for c in s if c.isalnum() or c in {"_", "-"}]
    return "".join(valid)[:80] or "doc"


@dataclass
class ArticleOut:
    doc_id: str
    text_path: Path
    intents_path: Path
    spans_path: Path


def write_doc_text(doc_id: str, paragraphs: List[str], root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(paragraphs)
    p = root / f"{doc_id}.txt"
    p.write_text(text, encoding="utf-8")
    return p


def sentence_offsets(doc_text: str) -> List[Tuple[int, int]]:
    """Return list of (start_char, end_char) inclusive for each sentence according to NLTK."""
    sents = sent_tokenize(doc_text)
    offsets: List[Tuple[int, int]] = []
    pos = 0
    for s in sents:
        start = doc_text.find(s, pos)
        if start < 0:
            # Fallback: search from 0 (rare); still attempt to find
            start = doc_text.find(s)
        if start < 0:
            # Give up; approximate by consuming length
            start = pos
        end = start + len(s) - 1
        offsets.append((start, end))
        pos = end + 1
    return offsets


def map_answer_to_sentences(
    abs_start: int, abs_end: int, sent_offs: List[Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    # Find sentence index covering start and end
    s_idx = None
    e_idx = None
    for i, (s0, s1) in enumerate(sent_offs, start=1):
        if s_idx is None and s0 <= abs_start <= s1:
            s_idx = i
        if e_idx is None and s0 <= abs_end <= s1:
            e_idx = i
        if s_idx is not None and e_idx is not None:
            break
    if s_idx is None or e_idx is None:
        return None
    if e_idx < s_idx:
        s_idx, e_idx = e_idx, s_idx
    return (s_idx, e_idx)


def convert_squad_file(squad_json: Path, limit: Optional[int]) -> List[ArticleOut]:
    data = json.loads(squad_json.read_text(encoding="utf-8"))
    articles = data["data"]

    out_dir_input = Path("data/input")
    out_dir_out = Path("out")

    ensure_nltk()

    outputs: List[ArticleOut] = []
    count = 0
    for art in articles:
        title = art.get("title", f"doc{count+1}")
        doc_id = slugify(title)
        paras = [p["context"] for p in art.get("paragraphs", [])]
        if not paras:
            continue

        text_path = write_doc_text(doc_id, paras, out_dir_input)
        doc_text = text_path.read_text(encoding="utf-8")
        sent_offs = sentence_offsets(doc_text)

        # Compute base offsets for each paragraph within doc
        bases: List[int] = []
        offset = 0
        for i, ctx in enumerate(paras):
            bases.append(offset)
            offset += len(ctx)
            if i < len(paras) - 1:
                offset += 2  # for two newlines we inserted

        # Gather questions and spans
        predicted_queries: List[str] = []
        gold_rows: List[Dict] = []
        qid = 0
        for pi, par in enumerate(art.get("paragraphs", [])):
            base = bases[pi]
            ctx = par["context"]
            for qa in par.get("qas", []):
                qid += 1
                q_text = qa.get("question", "").strip()
                predicted_queries.append(q_text)
                ans_list = qa.get("answers", [])
                if not ans_list:
                    gold_rows.append({"query_id": qid, "doc_id": doc_id, "answerable": False})
                    continue
                ans = ans_list[0]
                a_start_local = int(ans["answer_start"])
                a_text = ans.get("text", "")
                a_end_local = a_start_local + len(a_text) - 1
                abs_start = base + a_start_local
                abs_end = base + a_end_local
                sent_span = map_answer_to_sentences(abs_start, abs_end, sent_offs)
                if sent_span is None:
                    gold_rows.append({"query_id": qid, "doc_id": doc_id, "answerable": False})
                else:
                    s0, s1 = sent_span
                    gold_rows.append({
                        "query_id": qid,
                        "doc_id": doc_id,
                        "start_sent": s0,
                        "end_sent": s1,
                        "answerable": True,
                    })

        # Write per-article outputs
        out_dir_doc = out_dir_out / doc_id
        out_dir_doc.mkdir(parents=True, exist_ok=True)
        intents_path = out_dir_doc / "predicted_intents.jsonl"
        spans_path = out_dir_doc / "gt_spans.jsonl"

        # predicted_intents.jsonl expects one row per doc: {doc_id, predicted_queries: [...]}
        with intents_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"doc_id": doc_id, "predicted_queries": predicted_queries}, ensure_ascii=False) + "\n")

        with spans_path.open("w", encoding="utf-8") as f:
            for r in gold_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        outputs.append(ArticleOut(doc_id, text_path, intents_path, spans_path))
        count += 1
        if limit and count >= limit:
            break

    return outputs


def main():
    ap = argparse.ArgumentParser(description="Convert SQuAD v1.1/v2.0 JSON to IDC inputs + gold spans")
    ap.add_argument("--squad", required=True, help="Path to SQuAD v1.1 JSON")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of articles to convert")
    ap.add_argument("--doc-ids-out", type=str, default=None, help="Optional path to write converted doc_ids (one per line)")
    args = ap.parse_args()

    squad_json = Path(args.squad)
    assert squad_json.exists(), f"Not found: {squad_json}"

    outs = convert_squad_file(squad_json, limit=args.limit)
    print(f"Converted {len(outs)} articles.")
    for o in outs[:5]:
        print(f"  {o.doc_id}: {o.text_path} | {o.intents_path} | {o.spans_path}")
    if args.doc_ids_out:
        p = Path(args.doc_ids_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for o in outs:
                f.write(o.doc_id + "\n")
        print(f"Wrote doc id list → {p}")


if __name__ == "__main__":
    main()
