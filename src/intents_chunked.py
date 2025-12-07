#!/usr/bin/env python3
import os, re, time, argparse, json
from pathlib import Path
from typing import List, Dict, Set

import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

# Import embedding function
try:
    from embed import embed_texts, configure_genai as embed_configure_genai
except ImportError:
    # Fallback if embed module not importable
    embed_texts = None
    embed_configure_genai = None

def configure_genai():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY","").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)

def clean_lines(text: str) -> List[str]:
    qs = []
    for line in text.splitlines():
        q = re.sub(r'^[\s\-\*\d\)\.]+', '', line.strip())
        if not q: continue
        if q[-1] not in {"?","？","؟"}:
            q += "?"
        qs.append(q)
    # de-dup (case-insensitive) preserving order
    seen: Set[str] = set(); out=[]
    for q in qs:
        k = q.lower()
        if k not in seen:
            seen.add(k); out.append(q)
    return out

def ask(model_name: str, prompt: str, temperature: float, max_output_tokens: int) -> List[str]:
    model = genai.GenerativeModel(model_name)
    cfg = genai.GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
    
    # Set safety settings to be less restrictive
    safety_settings = {
        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
    }
    
    resp = model.generate_content(prompt, generation_config=cfg, safety_settings=safety_settings)
    
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
    except ValueError:
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
    
    return clean_lines(text)

def chunk_text(s: str, max_chars: int) -> List[str]:
    if len(s) <= max_chars: return [s]
    # try to split on blank lines near boundaries
    chunks=[]; i=0; n=len(s)
    while i < n:
        j = min(i+max_chars, n)
        if j < n:
            k = s.rfind("\n\n", i+int(0.8*max_chars), j)
            if k != -1: j = k
        chunks.append(s[i:j]); i = j
    return chunks

def l2_normalize(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2 normalize rows of matrix."""
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return mat / n

def greedy_mmr_selection(texts: List[str], embeddings: np.ndarray, 
                        lambda_mmr: float = 0.5, max_selected: int = 20) -> List[int]:
    """
    Greedy MMR (Maximal Marginal Relevance) selection.
    
    Args:
        texts: List of text strings
        embeddings: (N, D) normalized embeddings 
        lambda_mmr: Balance between relevance and diversity (0.5 = balanced)
        max_selected: Maximum number of items to select
        
    Returns:
        List of selected indices
    """
    if len(texts) <= max_selected:
        return list(range(len(texts)))
    
    # For simplicity, use mean embedding as "query" for relevance
    query = np.mean(embeddings, axis=0, keepdims=True)
    query = l2_normalize(query)
    
    # Compute initial relevance scores (cosine similarity to mean)
    relevance = embeddings @ query.T  # (N, 1)
    relevance = relevance.flatten()
    
    selected = []
    remaining = set(range(len(texts)))
    
    # Select first item with highest relevance
    first_idx = int(np.argmax(relevance))
    selected.append(first_idx)
    remaining.remove(first_idx)
    
    # Greedily select remaining items
    while len(selected) < max_selected and remaining:
        best_score = -np.inf
        best_idx = None
        
        for idx in remaining:
            # Relevance component
            rel_score = relevance[idx]
            
            # Diversity component (max similarity to already selected)
            if selected:
                selected_embs = embeddings[selected]  # (len(selected), D)
                current_emb = embeddings[idx:idx+1]   # (1, D)
                sims = current_emb @ selected_embs.T  # (1, len(selected))
                max_sim = np.max(sims)
            else:
                max_sim = 0.0
            
            # MMR score: balance relevance and diversity
            mmr_score = lambda_mmr * rel_score - (1 - lambda_mmr) * max_sim
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            break
    
    return selected

def simple_clustering_selection(texts: List[str], embeddings: np.ndarray, 
                               max_clusters: int = 15) -> List[int]:
    """
    Simple k-means-style clustering to ensure diverse coverage of topic areas.
    Returns indices of representative intents (one per cluster).
    """
    if len(texts) <= max_clusters:
        return list(range(len(texts)))
    
    # Use k-means clustering
    try:
        from sklearn.cluster import KMeans
        
        # Perform clustering
        kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Find the point closest to each cluster center
        selected_indices = []
        for i in range(max_clusters):
            cluster_mask = cluster_labels == i
            if not np.any(cluster_mask):
                continue
            
            cluster_points = embeddings[cluster_mask]
            cluster_center = kmeans.cluster_centers_[i:i+1]
            
            # Find closest point to center within this cluster
            cluster_indices = np.where(cluster_mask)[0]
            distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
            closest_idx_in_cluster = np.argmin(distances)
            selected_indices.append(cluster_indices[closest_idx_in_cluster])
        
        return selected_indices
    except ImportError:
        # Fallback to MMR if sklearn not available
        return greedy_mmr_selection(texts, embeddings, lambda_mmr=0.3, max_selected=max_clusters)

def main():
    ap = argparse.ArgumentParser(description="Chunked intent generation for large docs")
    ap.add_argument("--input", default="data/input/idc.txt")
    ap.add_argument("--out", default="out/idc/predicted_intents.jsonl")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--chars-per-chunk", type=int, default=12000)
    ap.add_argument("--questions-per-chunk", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-output-tokens", type=int, default=8192)
    ap.add_argument("--sleep", type=float, default=0.0, help="optional delay between calls")
    
    # Enhanced intent simulation parameters
    ap.add_argument("--max-intents", type=int, default=15, help="Max intents to keep after diversity selection (set 0 for unlimited)")
    ap.add_argument("--lambda-mmr", type=float, default=0.5, help="MMR lambda (relevance vs diversity)")
    ap.add_argument("--embedder", type=str, default="gemini-embedding-001", help="Embedding model for diversity")
    ap.add_argument("--embed-dim", type=int, default=1536, help="Embedding dimension")
    ap.add_argument("--disable-mmr", action="store_true", help="Skip diversity-based selection")
    ap.add_argument("--generation-multiplier", type=float, default=2.5, help="Generate N*multiplier intents before clustering")
    
    args = ap.parse_args()

    configure_genai()
    input_path = Path(args.input)
    doc_id = input_path.stem or "idc"
    text = input_path.read_text(encoding="utf-8")
    windows = chunk_text(text, args.chars_per_chunk)
    all_qs: List[str] = []

    # Generate fewer, higher-quality intents to reduce noise - OPTIMIZED
    expanded_questions_per_chunk = int(args.questions_per_chunk * min(1.5, args.generation_multiplier))  # Cap at 1.5x
    
    for idx, w in enumerate(windows, start=1):
        prompt = (
            f"Document chunk {idx}/{len(windows)}:\n\"\"\"\n{w}\n\"\"\"\n\n"
            f"Generate {expanded_questions_per_chunk} high-quality questions that users would likely ask when searching for information in this text.\n"
            f"Focus on questions that this specific chunk directly answers or addresses.\n"
            f"Make questions specific and answerable from the content provided.\n"
            f"Do NOT number or bullet them. One question per line."
        )
        qs = ask(args.model, prompt, args.temperature, args.max_output_tokens)
        all_qs.extend(qs)
        if args.sleep > 0: time.sleep(args.sleep)
    
    print(f"Generated {len(all_qs)} raw intents using {args.generation_multiplier}x multiplier")

    # global de-dup & light pruning
    seen=set(); deduped=[]
    for q in all_qs:
        k=q.lower().strip()
        if k and k not in seen:
            seen.add(k); deduped.append(q)
    
    print(f"Generated {len(all_qs)} raw intents, {len(deduped)} after string dedup")
    
    # Enhanced semantic diversity selection or unlimited
    if args.max_intents is not None and args.max_intents <= 0:
        # Unlimited / auto: keep all deduplicated intents
        final = deduped
        print(f"Auto intents count enabled: keeping all {len(final)} deduped intents")
    elif not args.disable_mmr and embed_texts and len(deduped) > args.max_intents:
        print(f"Applying diversity selection to choose {args.max_intents} intents from {len(deduped)}...")
        
        # Embed all questions
        embeddings = embed_texts(
            texts=deduped,
            model_name=args.embedder,
            output_dim=args.embed_dim,
            task_type="RETRIEVAL_QUERY",
            throttle_s=0.1  # Small delay for embeddings
        )
        
        # Normalize embeddings for cosine similarity
        embeddings = l2_normalize(embeddings)
        
        # Try clustering first for better topic coverage, fallback to MMR
        try:
            selected_indices = simple_clustering_selection(
                texts=deduped,
                embeddings=embeddings,
                max_clusters=args.max_intents
            )
            method = "clustering"
        except:
            # Fallback to MMR
            selected_indices = greedy_mmr_selection(
                texts=deduped,
                embeddings=embeddings,
                lambda_mmr=args.lambda_mmr,
                max_selected=args.max_intents
            )
            method = "MMR"
        
        final = [deduped[i] for i in selected_indices]
        print(f"{method} selected {len(final)} diverse intents")
    else:
        final = deduped
        if args.disable_mmr:
            print("Diversity selection disabled, keeping all deduped intents")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w", encoding="utf-8") as f:
        payload = {"doc_id": doc_id, "predicted_queries": final}
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"Wrote {len(final)} intents → {args.out}")

if __name__ == "__main__":
    main()
