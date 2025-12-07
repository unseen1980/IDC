#!/usr/bin/env python3
"""
Compute adaptive intent generation parameters based on document characteristics.

This script analyzes document length and complexity to recommend optimal parameters
for intent generation WITHOUT any benchmark-specific cheating.

The adaptation is based on general document properties:
- Length (number of sentences)
- Structural complexity (paragraphs, sections)
- Content density (avg sentence length)
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def read_jsonl(path: str):
    """Read JSONL file."""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_adaptive_params(
    num_sentences: int,
    avg_sentence_length: float,
    num_paragraphs: int = None
) -> Dict[str, float]:
    """
    Compute adaptive parameters based on document characteristics.

    Rationale:
    - Longer documents need more intents to cover diverse topics
    - Complex documents (shorter sentences, more paragraphs) need higher diversity
    - Technical papers need more question candidates for better filtering

    Args:
        num_sentences: Total sentences in document
        avg_sentence_length: Average sentence length in words
        num_paragraphs: Number of paragraphs (optional)

    Returns:
        Dict with recommended parameters
    """
    # Base parameters for ~200 sentence documents
    BASE_SENTENCES = 200
    BASE_NUM_QUESTIONS = 15
    BASE_MULTIPLIER = 1.5
    BASE_DIVERSITY = 0.4

    # 1. Scale number of questions with document length
    # Use sub-linear scaling (sqrt) to avoid over-generation on very long docs
    length_ratio = num_sentences / BASE_SENTENCES
    if length_ratio <= 0.5:
        # Very short docs (<100 sentences): slightly fewer questions
        num_questions = max(8, int(BASE_NUM_QUESTIONS * 0.7))
    elif length_ratio <= 1.0:
        # Short docs (100-200 sentences): use base
        num_questions = BASE_NUM_QUESTIONS
    elif length_ratio <= 2.5:
        # Medium docs (200-500 sentences): scale linearly
        num_questions = int(BASE_NUM_QUESTIONS * length_ratio)
    else:
        # Long docs (>500 sentences): sub-linear scaling
        num_questions = int(BASE_NUM_QUESTIONS * (length_ratio ** 0.7))

    # Cap at reasonable maximum
    num_questions = min(num_questions, 50)

    # 2. Adjust generation multiplier based on document complexity
    # Technical docs (shorter sentences) need more candidates for filtering
    if avg_sentence_length < 15:
        # Short sentences suggest lists, technical content → generate more
        multiplier = BASE_MULTIPLIER * 1.5
    elif avg_sentence_length < 20:
        # Normal sentences
        multiplier = BASE_MULTIPLIER
    else:
        # Long sentences suggest narrative content → generate less
        multiplier = BASE_MULTIPLIER * 0.8

    # For longer docs, increase multiplier to ensure quality after filtering
    if num_sentences > 500:
        multiplier *= 1.2

    # Cap multiplier
    multiplier = min(multiplier, 3.0)

    # 3. Adjust diversity threshold based on document length
    # Longer docs need more diverse intents to cover all topics
    if num_sentences < 200:
        # Short docs: moderate diversity (similar topics)
        diversity_threshold = 0.35
    elif num_sentences < 500:
        # Medium docs: base diversity
        diversity_threshold = BASE_DIVERSITY
    else:
        # Long docs: higher diversity (cover more ground)
        diversity_threshold = 0.30  # Lower threshold = more diverse

    # 4. Adjust based on structural complexity if available
    if num_paragraphs:
        para_per_sentence = num_paragraphs / num_sentences
        if para_per_sentence > 0.3:
            # Many short paragraphs → technical/structured content
            diversity_threshold -= 0.05  # More diversity
            multiplier *= 1.1

    return {
        "num_questions": int(num_questions),
        "generation_multiplier": round(multiplier, 2),
        "diversity_threshold": round(diversity_threshold, 2),
        "rationale": {
            "doc_length_category": (
                "very_short" if num_sentences < 100 else
                "short" if num_sentences < 200 else
                "medium" if num_sentences < 500 else
                "long"
            ),
            "sentence_complexity": (
                "technical" if avg_sentence_length < 15 else
                "normal" if avg_sentence_length < 20 else
                "narrative"
            ),
            "num_sentences": num_sentences,
            "avg_sentence_length": round(avg_sentence_length, 1)
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute adaptive intent generation parameters"
    )
    parser.add_argument(
        "--sentences",
        required=True,
        help="Path to sentences.jsonl file"
    )
    parser.add_argument(
        "--output-env",
        action="store_true",
        help="Output as shell environment variables"
    )
    parser.add_argument(
        "--output-json",
        help="Optional: save parameters to JSON file"
    )

    args = parser.parse_args()

    # Read sentences
    sentences = read_jsonl(args.sentences)
    num_sentences = len(sentences)

    if num_sentences == 0:
        print("ERROR: No sentences found", flush=True)
        return 1

    # Compute average sentence length
    total_words = 0
    num_paragraphs = 0
    prev_para_id = None

    for sent in sentences:
        text = sent.get("text", "")
        words = len(text.split())
        total_words += words

        # Count paragraph transitions
        para_id = sent.get("para_id")
        if para_id is not None and para_id != prev_para_id:
            num_paragraphs += 1
            prev_para_id = para_id

    avg_sentence_length = total_words / num_sentences

    # Compute adaptive parameters
    params = compute_adaptive_params(
        num_sentences=num_sentences,
        avg_sentence_length=avg_sentence_length,
        num_paragraphs=num_paragraphs if num_paragraphs > 0 else None
    )

    # Output results
    if args.output_env:
        # Shell environment variable format
        print(f"export NUM_QUESTIONS_ONEPASS={params['num_questions']}")
        print(f"export GENERATION_MULTIPLIER={params['generation_multiplier']}")
        print(f"export DIVERSITY_THRESHOLD={params['diversity_threshold']}")
    else:
        # Human-readable format
        print(f"📊 Document Analysis:")
        print(f"  Sentences: {num_sentences}")
        print(f"  Avg sentence length: {avg_sentence_length:.1f} words")
        if num_paragraphs > 0:
            print(f"  Paragraphs: {num_paragraphs}")
        print(f"  Category: {params['rationale']['doc_length_category']}")
        print(f"  Complexity: {params['rationale']['sentence_complexity']}")
        print()
        print(f"🎯 Recommended Parameters:")
        print(f"  NUM_QUESTIONS_ONEPASS={params['num_questions']}")
        print(f"  GENERATION_MULTIPLIER={params['generation_multiplier']}")
        print(f"  DIVERSITY_THRESHOLD={params['diversity_threshold']}")

    # Save to JSON if requested
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        if not args.output_env:
            print(f"\n💾 Saved parameters to: {args.output_json}")

    return 0


if __name__ == "__main__":
    exit(main())
