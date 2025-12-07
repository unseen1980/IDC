#!/usr/bin/env python3
import os
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai
import json

from config import parse_views_arg

# ---------- IO helpers ----------
def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def read_jsonl(path: str | Path) -> List[Dict]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



@dataclass
class ChunkViewEmbedding:
    """Payload describing a chunk view requiring embedding."""

    chunk_id: str
    view: str
    text: str


def load_chunk_view_payloads(index_path: str | Path, views: Tuple[str, ...]) -> List[ChunkViewEmbedding]:
    """Load chunk views from ``chunks.index.jsonl`` filtered by ``views``."""
    rows = read_jsonl(index_path)
    allowed = {v.lower() for v in views}
    payloads: List[ChunkViewEmbedding] = []
    for row in rows:
        chunk_id = row.get("chunk_uid") or row.get("chunk_id")
        if not chunk_id:
            continue
        for view_info in row.get("views", []):
            name = (view_info.get("view") or view_info.get("name") or "").lower()
            if name not in allowed:
                continue
            text = view_info.get("text") or ""
            if not text.strip():
                continue
            payloads.append(ChunkViewEmbedding(chunk_id=chunk_id, view=name, text=text))
    return payloads


def embed_chunk_views(
    index_path: str,
    out_npy: str,
    out_meta: str,
    model_name: str,
    output_dim: int,
    views: Tuple[str, ...],
) -> None:
    """Embed chunk views using the configured embedding model."""
    items = load_chunk_view_payloads(index_path, views)
    if not items:
        ensure_dir(Path(out_npy).parent)
        np.save(out_npy, np.zeros((0, output_dim), dtype=np.float32))
        write_jsonl(out_meta, [])
        print("No chunk views to embed; wrote empty artefacts.")
        return
    texts = [item.text for item in items]
    vectors = embed_texts_batch(
        texts=texts,
        model_name=model_name,
        output_dim=output_dim,
        task_type="RETRIEVAL_DOCUMENT",
    )
    ensure_dir(Path(out_npy).parent)
    np.save(out_npy, vectors)
    meta_rows = [
        {
            "chunk_uid": item.chunk_id,
            "view": item.view,
            "vector_index": idx,
            "text": item.text,
        }
        for idx, item in enumerate(items)
    ]
    write_jsonl(out_meta, meta_rows)
    print(f"Saved chunk-view embeddings: {vectors.shape} → {out_npy}")
    print(f"Saved chunk-view manifest → {out_meta}")


# ---------- Gemini config ----------
def configure_genai() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set (.env or environment)")
    genai.configure(api_key=api_key)

def _normalize_model_name(model_name: str) -> str:
    """
    google.generativeai historically expects 'models/...'.
    We'll accept either 'gemini-embedding-001' or 'models/gemini-embedding-001'.
    """
    if model_name.startswith("models/"):
        return model_name
    return f"models/{model_name}"

def _extract_vector(resp) -> List[float]:
    """
    Extract the embedding vector from google.generativeai response.
    The response has an 'embedding' attribute that contains the vector.
    """
    if hasattr(resp, 'embedding'):
        return list(resp.embedding)
    
    # Fallback for dict-like responses
    if isinstance(resp, dict) and 'embedding' in resp:
        return list(resp['embedding'])
    
    raise ValueError("Could not extract embedding vector from response")

# ---------- Embedding core ----------
def embed_texts_query_aware(
    texts: List[str],
    model_name: str,
    output_dim: int,
    task_type: str,
    sample_queries: List[str] = None,
    query_aware_weight: float = 0.3,
    batch_size: int = 100,
    max_retries: int = 5,
    base_delay: float = 1.0
) -> np.ndarray:
    """
    Query-aware embedding combining content and retrieval-optimized representations.
    """
    if task_type == "RETRIEVAL_QUERY" or not sample_queries:
        # For queries or when no sample queries available, use standard embedding
        return embed_texts_batch(
            texts, model_name, output_dim, task_type, batch_size, max_retries, base_delay
        )
    
    # For documents/chunks: create query-aware embeddings
    content_embeddings = embed_texts_batch(
        texts, model_name, output_dim, "RETRIEVAL_DOCUMENT", batch_size, max_retries, base_delay
    )
    
    # Create query-contextualized versions
    query_context = " ".join(sample_queries[:3]) if sample_queries else ""
    query_aware_texts = [f"Answer questions about: {text}. Relevant queries: {query_context}" for text in texts]
    
    query_aware_embeddings = embed_texts_batch(
        query_aware_texts, model_name, output_dim, "RETRIEVAL_DOCUMENT", batch_size, max_retries, base_delay
    )
    
    # Combine embeddings
    combined = (1 - query_aware_weight) * content_embeddings + query_aware_weight * query_aware_embeddings
    return combined

def embed_texts_batch(
    texts: List[str],
    model_name: str,
    output_dim: int,
    task_type: str,
    batch_size: int = 100,
    max_retries: int = 5,
    base_delay: float = 1.0
) -> np.ndarray:
    """
    Standard batch embedding with exponential backoff for better throughput and reliability.
    """
    model = _normalize_model_name(model_name)
    all_vecs: List[List[float]] = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding batches ({task_type})", unit="batch"):
        batch_texts = texts[i:i + batch_size]
        batch_vecs = []
        
        for text_idx, t in enumerate(batch_texts):
            t = t.strip()
            if not t:
                batch_vecs.append([0.0] * output_dim)
                continue
            
            # Retry logic with exponential backoff
            for attempt in range(max_retries):
                try:
                    resp = genai.embed_content(
                        model=model,
                        content=t,
                        task_type=task_type,
                        output_dimensionality=output_dim,
                    )
                    vec = _extract_vector(resp)
                    
                    # Normalize vector length
                    if len(vec) != output_dim:
                        if len(vec) > output_dim:
                            vec = vec[:output_dim]
                        else:
                            vec = vec + [0.0] * (output_dim - len(vec))
                    
                    batch_vecs.append(vec)
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed to embed text after {max_retries} attempts: {str(e)[:100]}")
                        batch_vecs.append([0.0] * output_dim)  # Fallback
                    else:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        print(f"Embedding failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
                        time.sleep(delay)
        
        all_vecs.extend(batch_vecs)
        
        # Small delay between batches to avoid rate limiting
        if i + batch_size < len(texts):
            time.sleep(0.1)
    
    return np.asarray(all_vecs, dtype=np.float32)

def embed_texts(
    texts: List[str],
    model_name: str,
    output_dim: int,
    task_type: str,
    throttle_s: float = 0.0
) -> np.ndarray:
    """
    Legacy interface - now uses batch embedding internally for better performance.
    """
    return embed_texts_batch(texts, model_name, output_dim, task_type)

# ---------- Pipelines ----------
def embed_sentences(
    sentences_jsonl: str,
    out_npy: str,
    out_meta: str,
    model_name: str,
    output_dim: int,
    sample_queries: List[str] = None,
    query_aware: bool = True
) -> None:
    rows = read_jsonl(sentences_jsonl)
    texts = [r["text"] for r in rows]
    
    if query_aware and sample_queries:
        arr = embed_texts_query_aware(
            texts=texts,
            model_name=model_name,
            output_dim=output_dim,
            task_type="RETRIEVAL_DOCUMENT",
            sample_queries=sample_queries
        )
    else:
        arr = embed_texts_batch(
            texts=texts,
            model_name=model_name,
            output_dim=output_dim,
            task_type="RETRIEVAL_DOCUMENT"
        )
    ensure_dir(Path(out_npy).parent)
    np.save(out_npy, arr)
    # Save meta to preserve ordering (maps each row to original sentence)
    meta = [{"i": i, "doc_id": r["doc_id"], "sent_id": r["sent_id"]} for i, r in enumerate(rows)]
    write_jsonl(out_meta, meta)
    print(f"Saved sentence embeddings: {arr.shape} → {out_npy}")
    print(f"Saved sentence meta → {out_meta}")

def embed_intents(
    predicted_intents_jsonl: str,
    out_npy: str,
    out_flat_jsonl: str,
    model_name: str,
    output_dim: int
) -> None:
    """
    Flattens {doc_id, predicted_queries: [q1,...]} into
    one row per query: {doc_id, query_id, text}
    """
    docs = read_jsonl(predicted_intents_jsonl)
    flat: List[Dict] = []
    for d in docs:
        doc_id = d["doc_id"]
        for j, q in enumerate(d.get("predicted_queries", []), start=1):
            flat.append({"doc_id": doc_id, "query_id": j, "text": q})

    texts = [r["text"] for r in flat]
    arr = embed_texts_batch(
        texts=texts,
        model_name=model_name,
        output_dim=output_dim,
        task_type="RETRIEVAL_QUERY"
    )
    ensure_dir(Path(out_npy).parent)
    np.save(out_npy, arr)
    write_jsonl(out_flat_jsonl, flat)
    print(f"Saved intent embeddings: {arr.shape} → {out_npy}")
    print(f"Saved flattened intents → {out_flat_jsonl}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="IDC: Embed sentences and predicted intents")
    ap.add_argument("--embedder", type=str, default="gemini-embedding-001",
                    help="Embedding model (google.generativeai uses 'models/<name>')")
    ap.add_argument("--dim", type=int, default=1536, help="Embedding dimensionality (128..3072). Default 1536.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_s = sub.add_parser("sentences")
    p_s.add_argument("--sentences", type=str, default="out/sentences.jsonl")
    p_s.add_argument("--out-npy", type=str, default="out/sentence_embs.npy")
    p_s.add_argument("--out-meta", type=str, default="out/sentences.meta.jsonl")
    p_s.add_argument("--query-aware", action="store_true", help="Use query-aware embedding")
    p_s.add_argument("--sample-queries", type=str, nargs="*", help="Sample queries for query-aware embedding")

    p_i = sub.add_parser("intents")
    p_i.add_argument("--intents", type=str, default="out/predicted_intents.jsonl")
    p_i.add_argument("--out-npy", type=str, default="out/intent_embs.npy")
    p_i.add_argument("--out-flat", type=str, default="out/intents.flat.jsonl")

    p_v = sub.add_parser("views")
    p_v.add_argument("--index", type=str, default="out/chunks.index.jsonl")
    p_v.add_argument("--out-npy", type=str, default="out/chunk_view_embs.npy")
    p_v.add_argument("--out-meta", type=str, default="out/chunk_view_embs.meta.jsonl")
    p_v.add_argument("--views", type=str, default="text,intent,summary,keywords")

    args = ap.parse_args()
    configure_genai()

    if args.cmd == "sentences":
        sample_queries = getattr(args, 'sample_queries', None) or []
        query_aware = getattr(args, 'query_aware', False)
        embed_sentences(args.sentences, args.out_npy, args.out_meta, args.embedder, args.dim, sample_queries, query_aware)
    elif args.cmd == "intents":
        embed_intents(args.intents, args.out_npy, args.out_flat, args.embedder, args.dim)
    elif args.cmd == "views":
        selected_views = parse_views_arg(getattr(args, 'views', None))
        embed_chunk_views(args.index, args.out_npy, args.out_meta, args.embedder, args.dim, selected_views)

if __name__ == "__main__":
    main()
