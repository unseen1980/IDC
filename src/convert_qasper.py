#!/usr/bin/env python3
"""Convert official QASPER JSON into IDC inputs + gold spans."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


@dataclass
class ParagraphInfo:
    norm_text: str
    start_sent: int
    end_sent: int


def flatten_full_text(full_text) -> List[str]:
    if not full_text:
        return []

    flat: List[str] = []

    # Official release stores full_text as a list of section dicts.
    if isinstance(full_text, list):
        for section in full_text:
            if not isinstance(section, dict):
                continue
            section_name = (section.get("section_name") or "").strip()
            if section_name:
                flat.append(section_name)
            for para in section.get("paragraphs", []) or []:
                para = (para or "").strip()
                if para:
                    flat.append(para)
        return flat

    # HuggingFace loader keeps a dict of lists.
    paragraphs = full_text.get("paragraphs") or []
    section_names = full_text.get("section_name") or []
    for idx, para_list in enumerate(paragraphs):
        section = section_names[idx] if idx < len(section_names) else ""
        if section:
            sec = section.strip()
            if sec:
                flat.append(sec)
        for para in para_list:
            para = (para or "").strip()
            if para:
                flat.append(para)
    return flat


def build_sentence_index(paragraphs: Sequence[str]) -> tuple[List[str], List[ParagraphInfo]]:
    sentences: List[str] = []
    paragraph_infos: List[ParagraphInfo] = []
    current = 0
    for para in paragraphs:
        para_sentences = [s.strip() for s in sent_tokenize(para) if s.strip()]
        if not para_sentences:
            continue
        start = current
        sentences.extend(para_sentences)
        current += len(para_sentences)
        paragraph_infos.append(
            ParagraphInfo(
                norm_text=normalize(para),
                start_sent=start,
                end_sent=current - 1,
            )
        )
    return sentences, paragraph_infos


def match_text_to_sentences(
    text: str,
    sentence_map: Dict[str, List[int]],
    normalized_sentences: List[str],
) -> Set[int]:
    matches: Set[int] = set()
    if not text:
        return matches

    candidates = [text]
    # highlighted evidence can contain multiple sentences – tokenize further
    tokenized = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if tokenized:
        candidates = tokenized

    for snippet in candidates:
        norm = normalize(snippet)
        if not norm:
            continue
        if norm in sentence_map:
            matches.update(sentence_map[norm])
            continue
        for idx, sent_norm in enumerate(normalized_sentences):
            if norm in sent_norm or sent_norm in norm:
                matches.add(idx)
                break
    return matches


def match_paragraphs_to_sentences(texts: Iterable[str], para_infos: Sequence[ParagraphInfo]) -> Set[int]:
    indices: Set[int] = set()
    for text in texts:
        norm = normalize(text)
        if not norm:
            continue
        for info in para_infos:
            if info.start_sent > info.end_sent:
                continue
            if norm == info.norm_text or norm in info.norm_text or info.norm_text in norm:
                indices.update(range(info.start_sent, info.end_sent + 1))
    return indices


def cluster_indices(indices: Set[int], max_gap: int = 1) -> List[Tuple[int, int, int]]:
    """Return contiguous clusters as (start_idx, end_idx, size).

    max_gap controls how far apart indices can be while still belonging to the
    same cluster. A value of 1 keeps strictly consecutive sentences together,
    while larger values would tolerate small gaps (e.g., sentence splitting
    mismatches)."""

    if not indices:
        return []

    sorted_idx = sorted(indices)
    clusters: List[List[int]] = [[sorted_idx[0]]]
    for idx in sorted_idx[1:]:
        if idx - clusters[-1][-1] <= max_gap:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])

    summarized: List[Tuple[int, int, int]] = []
    for cluster in clusters:
        start = cluster[0]
        end = cluster[-1]
        size = len(cluster)
        summarized.append((start, end, size))
    return summarized


def aggregate_question_spans(
    question: Dict,
    sentence_map: Dict[str, List[int]],
    normalized_sentences: List[str],
    para_infos: Sequence[ParagraphInfo],
) -> tuple[bool, Optional[int], Optional[int]]:
    evidence_indices: Set[int] = set()
    has_positive = False
    annotations = question.get("answers", []) or []
    for ann in annotations:
        answer = ann.get("answer") or {}
        if answer.get("unanswerable"):
            continue
        has_positive = True
        highlights = answer.get("highlighted_evidence") or []
        for highlight in highlights:
            evidence_indices.update(match_text_to_sentences(highlight, sentence_map, normalized_sentences))
        if not highlights:
            extractive = answer.get("extractive_spans") or []
            for span in extractive:
                evidence_indices.update(match_text_to_sentences(span, sentence_map, normalized_sentences))
        if not highlights and not answer.get("extractive_spans"):
            evidence_indices.update(match_paragraphs_to_sentences(answer.get("evidence", []), para_infos))

    if evidence_indices:
        clusters = cluster_indices(evidence_indices, max_gap=1)
        if clusters:
            # Prefer the largest cluster; break ties by choosing the tighter span
            clusters.sort(key=lambda c: (c[2], -(c[1] - c[0])), reverse=True)
            best_start, best_end, _ = clusters[0]
            return True, best_start + 1, best_end + 1

    # If we saw any positive annotations but failed to align, mark as answerable
    if has_positive:
        return True, None, None
    return False, None, None


def convert_official(records: Dict[str, Dict], limit: Optional[int]) -> List[str]:
    ensure_nltk_resources()
    doc_ids: List[str] = []

    items = list(records.items())
    for doc_key, payload in items:
        if limit is not None and len(doc_ids) >= limit:
            break

        doc_id = payload.get("id") or doc_key
        paragraphs = flatten_full_text(payload.get("full_text") or {})
        if not paragraphs:
            abstract = (payload.get("abstract") or "").strip()
            if abstract:
                paragraphs = [abstract]
            else:
                title = (payload.get("title") or "").strip()
                paragraphs = [title] if title else []

        doc_text = "\n\n".join(paragraphs)
        sentences, para_infos = build_sentence_index(paragraphs)
        sentence_map: Dict[str, List[int]] = {}
        normalized_sentences = []
        for idx, sentence in enumerate(sentences):
            norm = normalize(sentence)
            normalized_sentences.append(norm)
            sentence_map.setdefault(norm, []).append(idx)

        write_text(Path("data/input") / f"{doc_id}.txt", doc_text)

        questions = payload.get("qas", []) or []
        intents = {"doc_id": doc_id, "predicted_queries": []}
        spans: List[str] = []
        for qi, q in enumerate(questions, start=1):
            text = (q.get("question") or "").strip()
            intents["predicted_queries"].append(text)
            answerable, start_sent, end_sent = aggregate_question_spans(q, sentence_map, normalized_sentences, para_infos)
            if answerable and start_sent is not None and end_sent is not None:
                spans.append(
                    json.dumps(
                        {
                            "query_id": qi,
                            "doc_id": doc_id,
                            "start_sent": start_sent,
                            "end_sent": end_sent,
                            "answerable": True,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            else:
                spans.append(
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

        out_dir = Path("out") / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        write_text(out_dir / "predicted_intents.jsonl", json.dumps(intents, ensure_ascii=False) + "\n")
        write_text(out_dir / "gt_spans.jsonl", "".join(spans))
        doc_ids.append(doc_id)

    return doc_ids


def convert_simplified(records: List[Dict], limit: Optional[int]) -> List[str]:
    doc_ids: List[str] = []
    for rec in records:
        if limit is not None and len(doc_ids) >= limit:
            break
        doc_id = rec["doc_id"]
        sents = rec.get("sentences", [])
        qas = rec.get("questions", [])
        write_text(Path("data/input") / f"{doc_id}.txt", "\n\n".join(sents))
        intents = {"doc_id": doc_id, "predicted_queries": [q.get("text", "").strip() for q in qas]}
        out_dir = Path("out") / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)
        write_text(out_dir / "predicted_intents.jsonl", json.dumps(intents, ensure_ascii=False) + "\n")
        lines: List[str] = []
        for i, q in enumerate(qas, start=1):
            ev = q.get("evidence_sentences", [])
            if not ev:
                lines.append(json.dumps({"query_id": i, "doc_id": doc_id, "answerable": False}, ensure_ascii=False) + "\n")
                continue
            ev = [int(x) for x in ev]
            lines.append(
                json.dumps(
                    {
                        "query_id": i,
                        "doc_id": doc_id,
                        "start_sent": min(ev),
                        "end_sent": max(ev),
                        "answerable": True,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        write_text(out_dir / "gt_spans.jsonl", "".join(lines))
        doc_ids.append(doc_id)
    return doc_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Qasper JSON to IDC inputs + gold spans")
    ap.add_argument("--qasper", required=True, help="Path to qasper-dev-v0.3.json or similar")
    ap.add_argument("--doc-ids-out", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0, help="Maximum documents to convert (0 = all)")
    args = ap.parse_args()

    src = Path(args.qasper)
    assert src.exists(), f"Not found: {src}"

    data = json.loads(src.read_text(encoding="utf-8"))
    limit = args.limit if args.limit and args.limit > 0 else None

    if isinstance(data, dict):
        doc_ids = convert_official(data, limit)
    elif isinstance(data, list):
        doc_ids = convert_simplified(data, limit)
    else:
        raise ValueError("Unsupported Qasper JSON structure")

    if args.doc_ids_out:
        p = Path(args.doc_ids_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(doc_ids) + "\n", encoding="utf-8")
    print(f"Converted Qasper: {len(doc_ids)} docs")


if __name__ == "__main__":
    main()
