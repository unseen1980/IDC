#!/usr/bin/env python3
"""
Convert NewsQA dataset into a single concatenated corpus document.

Instead of treating each news story as a separate document, this script
concatenates multiple stories into ONE large corpus document. This allows
IDC to segment the entire corpus as a cohesive unit, demonstrating its
ability to handle large, multi-topic documents.

Usage:
    python src/convert_newsqa_corpus.py \\
        --data data/news-qa-summarization/data.jsonl \\
        --output data/input/newsqa_corpus.txt \\
        --doc-name newsqa_corpus \\
        --limit 10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def load_newsqa_entries(jsonl_path: Path, limit: int = 10) -> List[Dict[str, Any]]:
    """Load NewsQA entries from JSONL file."""
    entries = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            if line.strip():
                entries.append(json.loads(line))
    return entries


def create_corpus_document(entries: List[Dict[str, Any]], separator: str = "\n\n=== DOCUMENT SEPARATOR ===\n\n") -> str:
    """
    Concatenate multiple NewsQA stories into a single corpus document.

    Args:
        entries: List of NewsQA entries (each with 'story', 'questions', 'answers')
        separator: String to insert between stories for clarity

    Returns:
        Single concatenated text corpus
    """
    corpus_parts = []

    for i, entry in enumerate(entries, 1):
        story = entry.get("story", "").strip()
        if story:
            # Add document marker for tracking
            corpus_parts.append(f"[Story {i:03d}]")
            corpus_parts.append(story)
            corpus_parts.append(separator)

    # Remove trailing separator
    if corpus_parts and corpus_parts[-1] == separator:
        corpus_parts.pop()

    return "\n".join(corpus_parts)


def create_gold_spans(entries: List[Dict[str, Any]], doc_name: str, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Create pseudo-gold evaluation spans from NewsQA questions.

    For each question, we'll create a span that marks where the answer
    should be found in the concatenated corpus.

    Args:
        entries: List of NewsQA entries
        doc_name: Document name for span tracking
        output_dir: Directory to write spans

    Returns:
        List of span dictionaries
    """
    spans = []

    for doc_idx, entry in enumerate(entries):
        questions = entry.get("questions", [])
        answers = entry.get("answers", [])

        for q_idx, question in enumerate(questions):
            if q_idx < len(answers) and answers[q_idx]:
                # Create a span entry for this question
                span = {
                    "doc_id": doc_name,
                    "query_id": len(spans),  # Global query ID across all documents
                    "story_id": doc_idx,  # Which story this came from
                    "question": question,
                    "answers": answers[q_idx] if q_idx < len(answers) else [],
                    "answerable": True
                }
                spans.append(span)

    return spans


def main():
    parser = argparse.ArgumentParser(description="Convert NewsQA to single corpus document")
    parser.add_argument("--data", type=Path, required=True, help="Path to NewsQA JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output corpus text file")
    parser.add_argument("--doc-name", type=str, default="newsqa_corpus", help="Document name")
    parser.add_argument("--limit", type=int, default=10, help="Number of stories to concatenate (0=all)")
    parser.add_argument("--separator", type=str, default="\n\n=== DOCUMENT SEPARATOR ===\n\n",
                        help="Separator between stories")
    parser.add_argument("--spans-output", type=Path, help="Output file for gold spans (optional)")

    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: NewsQA file not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    # Load NewsQA entries
    print(f"Loading NewsQA entries from {args.data}...")
    entries = load_newsqa_entries(args.data, limit=args.limit)
    print(f"Loaded {len(entries)} stories")

    # Count total questions
    total_questions = sum(len(e.get("questions", [])) for e in entries)
    print(f"Total questions across all stories: {total_questions}")

    # Create concatenated corpus
    print("Creating concatenated corpus document...")
    corpus_text = create_corpus_document(entries, separator=args.separator)

    # Count sentences (approximate)
    sentence_count = corpus_text.count('. ') + corpus_text.count('! ') + corpus_text.count('? ')
    print(f"Approximate sentence count: {sentence_count}")
    print(f"Character count: {len(corpus_text):,}")

    # Write corpus document
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write(corpus_text)
    print(f"✓ Wrote corpus to: {args.output}")

    # Optionally create gold spans
    if args.spans_output:
        print("Creating gold evaluation spans...")
        spans = create_gold_spans(entries, args.doc_name, args.spans_output.parent)
        print(f"Created {len(spans)} evaluation spans")

        args.spans_output.parent.mkdir(parents=True, exist_ok=True)
        with args.spans_output.open("w", encoding="utf-8") as f:
            for span in spans:
                f.write(json.dumps(span) + "\n")
        print(f"✓ Wrote spans to: {args.spans_output}")

    print(f"\n✓ Success! Created corpus with {len(entries)} stories, ~{sentence_count} sentences, {total_questions} questions")


if __name__ == "__main__":
    main()
