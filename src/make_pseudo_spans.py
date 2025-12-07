#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

def read_jsonl(p): return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with Path(p).open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+"\n")

def configure_genai():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY","").strip()
    if not api_key: raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)

def normalize(M: np.ndarray, eps=1e-8) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return M / n

def embed_texts(texts: List[str], model_name: str, dim: int) -> np.ndarray:
    model = model_name if model_name.startswith("models/") else f"models/{model_name}"
    vecs=[]
    for t in texts:
        t = (t or "").strip()
        if not t:
            vecs.append([0.0]*dim); continue
        resp = genai.embed_content(
            model=model, content=t, task_type="RETRIEVAL_QUERY", output_dimensionality=dim
        )
        v = resp.embedding if hasattr(resp, "embedding") else resp["embedding"]
        v = list(v)
        if len(v) != dim:
            v = v[:dim] if len(v) > dim else v + [0.0]*(dim-len(v))
        vecs.append(v)
    return normalize(np.asarray(vecs, dtype=np.float32))

def main():
    ap = argparse.ArgumentParser(description="Create pseudo-gold spans by nearest sentence + neighbor expansion")
    ap.add_argument("--intents", required=True)               # out/<doc>/intents.flat.jsonl
    ap.add_argument("--sentences", required=True)             # out/<doc>/sentences.jsonl
    ap.add_argument("--sentence-embs", required=True)         # out/<doc>/sentence_embs.d*.npy
    ap.add_argument("--out", required=True)                   # out/<doc>/gt_spans.jsonl
    ap.add_argument("--embedder", default="gemini-embedding-001")
    ap.add_argument("--eval-embedder", default=None,
                    help="Optional: alternate embedder used ONLY for query embeddings to decouple span creation from sentence/embedder space")
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--threshold", type=float, default=0.40)  # lower = more answerable
    ap.add_argument("--neighbor-frac", type=float, default=0.80)
    ap.add_argument("--max-span-len", type=int, default=6)
    ap.add_argument("--min-spans", type=int, default=100, 
                    help="Minimum number of answerable spans to generate (lower threshold if needed)")
    args = ap.parse_args()

    configure_genai()
    intents = read_jsonl(args.intents)        # [{doc_id, query_id, text}]
    sents = read_jsonl(args.sentences)        # [{doc_id, sent_id, text}]
    S_all = np.load(args.sentence_embs)       # (N, D)

    # group sentences & normalized embeddings by doc, preserving order
    by_doc_sents: Dict[str, List[Dict]] = {}
    by_doc_idxs:  Dict[str, List[int]]  = {}
    for i, r in enumerate(sents):
        d = r["doc_id"]; by_doc_sents.setdefault(d, []).append(r); by_doc_idxs.setdefault(d, []).append(i)
    by_doc_Sn: Dict[str, np.ndarray] = {d: normalize(S_all[np.array(idxs)]) for d, idxs in by_doc_idxs.items()}

    q_texts = [r.get("text", "") for r in intents]
    if not q_texts:
        print("No queries found in intents; writing empty spans file.")
        write_jsonl(args.out, [])
        return
    # Use eval-embedder for queries if provided to reduce model coupling bias
    q_embedder = args.eval_embedder if args.eval_embedder else args.embedder
    Qn = embed_texts(q_texts, q_embedder, args.dim)
    if Qn.size == 0:
        print("Failed to embed queries (empty). Writing empty spans file.")
        write_jsonl(args.out, [])
        return

    # Adapt min_spans to be realistic based on number of intents
    # Can't have more spans than intents (1 span per intent max)
    num_intents = len(intents)
    effective_min_spans = min(args.min_spans, num_intents)
    if effective_min_spans < args.min_spans:
        print(f"⚠️  Adjusted min_spans from {args.min_spans} to {effective_min_spans} (can't exceed {num_intents} intents)")

    # Try initial threshold, then lower it if we don't get enough spans
    current_threshold = args.threshold
    min_threshold = 0.10  # Never accept spans below 0.10 similarity (too low quality)
    
    while current_threshold >= min_threshold:
        rows=[]; answerable=0
        for i, q in enumerate(intents):
            d = q["doc_id"]; qid = int(q["query_id"])
            Sdoc = by_doc_Sn.get(d); sdoc = by_doc_sents.get(d)
            if Sdoc is None or Sdoc.shape[0]==0:
                rows.append({"query_id": qid, "doc_id": d, "answerable": False})
                continue
            sims = Sdoc @ Qn[i]                  # cosine (both normalized)
            best_idx = int(np.argmax(sims)); best = float(sims[best_idx])
            if best < current_threshold:
                rows.append({"query_id": qid, "doc_id": d, "answerable": False})
                continue
            # expand to neighbors
            L = R = best_idx
            cutoff = best * args.neighbor_frac
            # expand left
            while L-1 >= 0 and (sims[L-1] >= cutoff) and (R-L+1) < args.max_span_len:
                L -= 1
            # expand right
            while R+1 < len(sims) and (sims[R+1] >= cutoff) and (R-L+1) < args.max_span_len:
                R += 1
            rows.append({"query_id": qid, "doc_id": d, "start_sent": int(sdoc[L]["sent_id"]),
                         "end_sent": int(sdoc[R]["sent_id"]), "answerable": True})
            answerable += 1
        
        # Check if we have enough spans
        if answerable >= effective_min_spans:
            break
        elif current_threshold > min_threshold:
            print(f"Only {answerable} spans with threshold {current_threshold:.3f}, lowering to {current_threshold-0.05:.3f}")
            current_threshold -= 0.05
        else:
            print(f"Reached minimum threshold {min_threshold:.3f}, got {answerable}/{effective_min_spans} spans")
            break

    # If still none answerable, force fallback: pick best match for each query
    if answerable == 0 and intents:
        print("No answerable spans at min threshold; forcing best-match fallback for each query.")
        rows=[]; answerable=0
        for i, q in enumerate(intents):
            d = q["doc_id"]; qid = int(q.get("query_id", i+1))
            Sdoc = by_doc_Sn.get(d); sdoc = by_doc_sents.get(d)
            if Sdoc is None or Sdoc.shape[0]==0:
                rows.append({"query_id": qid, "doc_id": d, "answerable": False}); continue
            sims = Sdoc @ Qn[i]
            best_idx = int(np.argmax(sims)); best = float(sims[best_idx])
            L = R = best_idx
            cutoff = max(0.0, best * args.neighbor_frac)
            while L-1 >= 0 and (sims[L-1] >= cutoff) and (R-L+1) < args.max_span_len:
                L -= 1
            while R+1 < len(sims) and (sims[R+1] >= cutoff) and (R-L+1) < args.max_span_len:
                R += 1
            rows.append({"query_id": qid, "doc_id": d, "start_sent": int(sdoc[L]["sent_id"]),
                         "end_sent": int(sdoc[R]["sent_id"]), "answerable": True})
            answerable += 1

    write_jsonl(args.out, rows)
    print(f"Created {answerable}/{len(rows)} answerable spans (threshold={current_threshold:.3f}) → {args.out}")

if __name__ == "__main__":
    main()
