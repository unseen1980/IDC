#!/usr/bin/env python3
import os
import time
import argparse
import re
from typing import List, Dict
from pathlib import Path

from dotenv import load_dotenv
import google.generativeai as genai

from utils import read_text_files, write_jsonl, ensure_dir


def configure_genai() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (.env or environment)")
    genai.configure(api_key=api_key)


def clean_and_split_lines(text: str) -> List[str]:
    qs = []
    for line in text.splitlines():
        q = re.sub(r'^[\s\-\*\d\)\.]+', '', line.strip())  # strip numbering/bullets
        if not q:
            continue
        if q[-1] not in {"?", "？", "؟"}:
            q += "?"
        qs.append(q)
    # de-duplicate, preserve order (case-insensitive)
    seen, out = set(), []
    for q in qs:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out


def truncate_for_prompt(s: str, max_chars: int = 12000) -> str:
    # Basic safeguard; adjust if needed for longer docs
    return s if len(s) <= max_chars else s[:max_chars]


def _analyze_document_structure(doc_text: str) -> Dict:
    """Analyze document structure universally without content bias."""
    sentences = [s.strip() for s in doc_text.split('.') if len(s.strip()) > 10]

    # Content-agnostic structural analysis only
    analysis = {
        'has_lists': bool(re.search(r'(?:\n\s*[-*•]\s+|\n\s*\d+\.\s+)', doc_text)),  # Bullet/numbered lists
        'sentence_count': len(sentences),
        'avg_sentence_length': sum(len(s.split()) for s in sentences) / max(len(sentences), 1),
        'word_count': len(doc_text.split()),
    }

    return analysis

def _get_universal_prompt(num_questions: int, existing_intents: List[str] = None) -> str:
    """Generate truly universal intent prompts without content bias.

    Args:
        num_questions: Number of questions to generate
        existing_intents: Previously generated intents for diversity enforcement

    Returns:
        Prompt string for intent generation
    """
    base_instruction = f"Generate {num_questions} diverse, high-quality questions that cover different aspects of the document."

    diversity_instruction = ""
    if existing_intents:
        diversity_instruction = f"\n\n**CRITICAL: Avoid duplicate topics!**\nThese questions already exist:\n"
        for idx, intent in enumerate(existing_intents[:5], 1):  # Show first 5
            diversity_instruction += f"{idx}. {intent}\n"
        diversity_instruction += "\nYour new questions MUST cover DIFFERENT topics and aspects not addressed above."

    prompt = f"""{base_instruction}

**Question Style Requirements:**
- **PRIORITIZE simple, atomic questions** - each question should ask about ONE thing only
- Most questions should be SHORT (4-8 words)
- Each question must be directly answerable from a specific part of the document
- Use natural, conversational phrasing that actual users would ask

**Generate a diverse MIX including:**
1. **Basic factual questions (50% of questions):**
   - "Who was X?" "What is Y?" "When did Z occur?" "Where is A located?"
   - "What religion were X?" "Who ruled Y?" "What country is Z in?"

2. **Relational questions (30% of questions):**
   - "What did X do?" "Who did Y marry?" "What role did Z play?"
   - "How did A influence B?" "What impact did X have?"

3. **Complex explanatory questions (20% of questions):**
   - "How did X develop?" "Why did Y occur?" "Describe the relationship between A and B"

**Example of good diversity:**
✓ "Who was the Norse leader?" (short, factual)
✓ "What century did the Normans first gain their separate identity?" (medium, factual)
✓ "How did the Norman language develop?" (explanatory)
{diversity_instruction}

Return exactly {num_questions} questions, one per line, no numbering or bullets."""

    return prompt


def _compute_intent_similarity(intent1: str, intent2: str) -> float:
    """Compute semantic similarity between two intents using simple token overlap.

    Args:
        intent1: First intent string
        intent2: Second intent string

    Returns:
        Similarity score between 0 and 1
    """
    # Simple token-based similarity (case-insensitive)
    tokens1 = set(intent1.lower().split())
    tokens2 = set(intent2.lower().split())

    # Remove common stop words
    stop_words = {'what', 'how', 'when', 'where', 'why', 'who', 'which', 'is', 'are', 'the', 'a', 'an', 'do', 'does', 'can', 'should'}
    tokens1 = tokens1 - stop_words
    tokens2 = tokens2 - stop_words

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union) if union else 0.0


def _filter_diverse_intents(intents: List[str], min_similarity_threshold: float = 0.4) -> List[str]:
    """Filter out similar intents to ensure diversity.

    Args:
        intents: List of intent strings
        min_similarity_threshold: Maximum allowed similarity between intents

    Returns:
        Filtered list of diverse intents
    """
    if not intents:
        return []

    diverse_intents = [intents[0]]  # Always keep the first one

    for candidate in intents[1:]:
        # Check if candidate is too similar to any existing intent
        is_diverse = True
        for existing in diverse_intents:
            similarity = _compute_intent_similarity(candidate, existing)
            if similarity > min_similarity_threshold:
                is_diverse = False
                break

        if is_diverse:
            diverse_intents.append(candidate)

    return diverse_intents


def generate_intents_for_doc(
    doc_text: str,
    model_name: str = "gemini-2.5-flash",
    num_questions: int = 5,
    temperature: float = 0.3,
    max_output_tokens: int = 256,
    retries: int = 3,
    backoff: float = 1.5,
    use_multi_stage: bool = True,
    diversity_threshold: float = 0.4,
) -> List[str]:
    """Generate diverse intents for a document using universal prompts.

    Args:
        doc_text: The document text
        model_name: Model name for generation
        num_questions: Target number of questions
        temperature: Generation temperature (higher = more diverse)
        max_output_tokens: Maximum tokens in output
        retries: Number of retries on failure
        backoff: Backoff multiplier for retries
        use_multi_stage: Whether to use multi-stage generation with diversity enforcement
        diversity_threshold: Maximum similarity threshold for filtering (lower = more diverse)

    Returns:
        List of diverse intent questions
    """
    if not use_multi_stage:
        # Single-stage generation
        prompt = f"{_get_universal_prompt(num_questions)}\n\nDocument:\n\"\"\"\n{truncate_for_prompt(doc_text)}\n\"\"\"\n"
        return _generate_with_prompt(prompt, model_name, num_questions, temperature, max_output_tokens, retries, backoff)

    # Multi-stage generation with diversity enforcement
    all_intents = []

    # Stage 1: Generate initial batch
    first_batch_size = min(num_questions, 3)
    prompt1 = f"{_get_universal_prompt(first_batch_size)}\n\nDocument:\n\"\"\"\n{truncate_for_prompt(doc_text)}\n\"\"\"\n"
    batch1 = _generate_with_prompt(prompt1, model_name, first_batch_size, temperature, max_output_tokens, retries, backoff)
    all_intents.extend(batch1)

    # Stage 2: Generate additional intents with diversity enforcement
    remaining = num_questions - len(all_intents)
    if remaining > 0 and len(all_intents) > 0:
        prompt2 = f"{_get_universal_prompt(remaining, existing_intents=all_intents)}\n\nDocument:\n\"\"\"\n{truncate_for_prompt(doc_text)}\n\"\"\"\n"
        batch2 = _generate_with_prompt(prompt2, model_name, remaining, temperature * 1.2, max_output_tokens, retries, backoff)
        all_intents.extend(batch2)

    # Stage 3: Filter for diversity
    diverse_intents = _filter_diverse_intents(all_intents, min_similarity_threshold=diversity_threshold)

    # If we don't have enough diverse intents, try one more time with higher temperature
    if len(diverse_intents) < num_questions and len(all_intents) >= num_questions:
        remaining = num_questions - len(diverse_intents)
        prompt3 = f"{_get_universal_prompt(remaining, existing_intents=diverse_intents)}\n\nDocument:\n\"\"\"\n{truncate_for_prompt(doc_text)}\n\"\"\"\n"
        batch3 = _generate_with_prompt(prompt3, model_name, remaining, temperature * 1.5, max_output_tokens, retries, backoff)
        all_intents.extend(batch3)
        diverse_intents = _filter_diverse_intents(all_intents, min_similarity_threshold=diversity_threshold)

    return diverse_intents[:num_questions]


def _generate_with_prompt(
    prompt: str,
    model_name: str,
    num_questions: int,
    temperature: float,
    max_output_tokens: int,
    retries: int,
    backoff: float,
) -> List[str]:
    """Internal helper to generate intents with a given prompt.

    Args:
        prompt: The complete prompt
        model_name: Model name
        num_questions: Number of questions
        temperature: Generation temperature
        max_output_tokens: Max output tokens
        retries: Number of retries
        backoff: Backoff multiplier

    Returns:
        List of generated intent questions
    """

    for attempt in range(1, retries + 1):
        try:
            model = genai.GenerativeModel(model_name)
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            
            # Set safety settings to be less restrictive
            safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
            
            resp = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            
            # Check if response was blocked or truncated
            if resp.candidates and len(resp.candidates) > 0:
                candidate = resp.candidates[0]
                
                if candidate.finish_reason.name == "SAFETY":
                    print(f"  [WARN] Content blocked by safety filter")
                    return []
                elif candidate.finish_reason.name == "RECITATION":
                    print(f"  [WARN] Content blocked due to recitation")
                    return []
                elif candidate.finish_reason.name == "MAX_TOKENS":
                    print(f"  [WARN] Hit max token limit - increase --max-output-tokens")
                    # Still try to get partial content if available
            
            # Try to get text from response
            try:
                text = (resp.text or "").strip()
            except ValueError as ve:
                # Handle case where response doesn't have valid text
                if resp.candidates:
                    # Try to extract text from parts if available
                    parts_text = []
                    for candidate in resp.candidates:
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    parts_text.append(part.text)
                    text = " ".join(parts_text).strip()
                    if not text:
                        print(f"  [WARN] No text generated - try increasing --max-output-tokens")
                        return []
                else:
                    print(f"  [WARN] No candidates in response")
                    return []
            
            if not text:
                return []
            qs = clean_and_split_lines(text)
            return qs[:num_questions]
        except Exception as e:
            if attempt == retries:
                print(f"[ERROR] Generation failed after {retries} attempts: {e}")
                return []
            sleep_s = backoff ** attempt
            print(f"[WARN] Generation error ({e}); retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)

    return []


def main():
    ap = argparse.ArgumentParser(description="IDC: Generate predicted intents per document with diversity enforcement")
    ap.add_argument("--input", type=str, default="data/input", help="Input directory of .txt files")
    ap.add_argument("--out", type=str, default="out/predicted_intents.jsonl", help="Output JSONL path")
    ap.add_argument("--model", type=str, default="gemini-2.5-flash",
                    help="Model name (e.g., gemini-2.5-flash)")
    ap.add_argument("--num-questions", type=int, default=5, help="Questions per doc")
    ap.add_argument("--temperature", type=float, default=0.7, help="Generation temperature (higher = more diverse)")
    ap.add_argument("--max-output-tokens", type=int, default=8192, help="Max output tokens")
    ap.add_argument("--multi-stage", action="store_true", default=True,
                    help="Use multi-stage generation with diversity enforcement (default: True)")
    ap.add_argument("--no-multi-stage", dest="multi_stage", action="store_false",
                    help="Disable multi-stage generation")
    ap.add_argument("--diversity-threshold", type=float, default=0.4,
                    help="Maximum similarity threshold for intent diversity (lower = more diverse, default: 0.4)")
    args = ap.parse_args()

    ensure_dir("out")
    configure_genai()

    docs = read_text_files(args.input, "*.txt")
    results: List[Dict] = []

    print(f"Intent generation mode: {'Multi-stage with diversity enforcement' if args.multi_stage else 'Single-stage'}")
    print(f"Temperature: {args.temperature}")
    print(f"Target questions per document: {args.num_questions}\n")

    for d in docs:
        print(f"Generating intents for {d['doc_id']} using {args.model}...")
        qs = generate_intents_for_doc(
            doc_text=d["text"],
            model_name=args.model,
            num_questions=args.num_questions,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            use_multi_stage=args.multi_stage,
            diversity_threshold=args.diversity_threshold,
        )
        print(f"  Generated {len(qs)} diverse intents:")
        for idx, q in enumerate(qs, 1):
            print(f"  {idx}. {q}")

        # Compute and display diversity metrics
        if len(qs) > 1:
            similarities = []
            for i in range(len(qs)):
                for j in range(i + 1, len(qs)):
                    sim = _compute_intent_similarity(qs[i], qs[j])
                    similarities.append(sim)
            avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
            print(f"  📊 Avg pairwise similarity: {avg_sim:.3f} (lower = more diverse)")

        results.append({"doc_id": d["doc_id"], "predicted_queries": qs})

    write_jsonl(args.out, results)
    print(f"\n✅ Saved predicted intents → {args.out}")


if __name__ == "__main__":
    main()
