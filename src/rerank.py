#!/usr/bin/env python3
"""Flexible reranking utilities for IDC retrieval."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "has", "have", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "there", "this", "to", "was", "were",
    "which", "with",
}


@dataclass
class Candidate:
    """Represents a chunk candidate for reranking."""

    chunk_id: str
    text: str
    intent: Optional[str]
    score: float
    num_sentences: int = 0


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for match in TOKEN_RE.findall(text.lower()):
        token = match.strip("'")
        if token and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def lexical_bm25(query: str, document: str) -> float:
    q_tokens = tokenize(query)
    d_tokens = tokenize(document)
    if not q_tokens or not d_tokens:
        return 0.0
    q_counts = {t: q_tokens.count(t) for t in set(q_tokens)}
    d_counts = {t: d_tokens.count(t) for t in set(d_tokens)}
    d_len = len(d_tokens)
    avg_len = d_len
    k1 = 1.5
    b = 0.75
    score = 0.0
    for term, qf in q_counts.items():
        tf = d_counts.get(term, 0)
        if tf == 0:
            continue
        idf = math.log(1.0 + (1.0 / (1.0 + d_counts.get(term, 0))))
        denom = tf + k1 * (1.0 - b + b * d_len / max(avg_len, 1e-8))
        score += idf * tf * (k1 + 1.0) / max(denom, 1e-8)
    return score


def token_overlap(query: str, intent: str) -> float:
    if not intent:
        return 0.0
    q_tokens = set(tokenize(query))
    i_tokens = set(tokenize(intent))
    if not q_tokens or not i_tokens:
        return 0.0
    return len(q_tokens & i_tokens) / len(q_tokens)


def rerank(
    query: str,
    candidates: Iterable[Candidate],
    method: str = "lexical",
    min_chunk_sent: int = 2,
) -> List[Candidate]:
    """Rerank candidates in-place and return a sorted list."""
    items = list(candidates)
    if method == "none" or not items:
        return sorted(items, key=lambda c: c.score, reverse=True)

    if method == "mini-crossencoder":  # pragma: no cover - optional dependency
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, cand.text) for cand in items]
            ce_scores = model.predict(pairs).tolist()
            for cand, ce_score in zip(items, ce_scores):
                cand.score = 0.7 * cand.score + 0.3 * ce_score
        except Exception:
            method = "lexical"

    if method == "lexical":
        bm25 = {cand.chunk_id: lexical_bm25(query, cand.text) for cand in items}
        for cand in items:
            overlap = token_overlap(query, cand.intent or "")
            length_pen = -0.05 if cand.num_sentences and cand.num_sentences < min_chunk_sent else 0.0
            cand.score = cand.score + 0.3 * bm25.get(cand.chunk_id, 0.0) + 0.05 * overlap + length_pen

    return sorted(items, key=lambda c: c.score, reverse=True)


def load_candidates(path: str | Path) -> List[Dict]:
    rows: List[Dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_candidates(path: str | Path, rows: Iterable[Dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Rerank chunk candidates using lexical or cross-encoder heuristics")
    parser.add_argument("--input", type=str, required=True, help="JSONL with entries: {query, candidates:[{chunk_id,text,intent,score,num_sentences}]} ")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--method", choices=["none", "lexical", "mini-crossencoder"], default="lexical")
    parser.add_argument("--min-chunk-sent", type=int, default=2)
    args = parser.parse_args()

    rows = load_candidates(args.input)
    reranked_rows = []
    for row in rows:
        candidates = [
            Candidate(
                chunk_id=entry["chunk_id"],
                text=entry.get("text", ""),
                intent=entry.get("intent"),
                score=float(entry.get("score", 0.0)),
                num_sentences=int(entry.get("num_sentences", 0)),
            )
            for entry in row.get("candidates", [])
        ]
        reranked = rerank(row.get("query", ""), candidates, method=args.method, min_chunk_sent=args.min_chunk_sent)
        row["candidates"] = [cand.__dict__ for cand in reranked]
        reranked_rows.append(row)
    save_candidates(args.output, reranked_rows)
    print(f"Saved reranked candidates → {args.output}")


if __name__ == "__main__":  # pragma: no cover
    cli()
