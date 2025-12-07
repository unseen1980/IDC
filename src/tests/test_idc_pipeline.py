import unittest
from typing import Dict, List

import numpy as np

from idc_core import IDCParams, chunk_coherence, refine_chunks, segment_document
from eval_retrieval import (
    BM25Index,
    ChunkRecord,
    ViewRecord,
    combine_dense_sparse,
    compute_answer_coverage,
    dense_chunk_scores,
    parse_hybrid_weights,
    rerank_candidates,
)


class SegmentationTests(unittest.TestCase):
    def _make_sentence_rows(self) -> List[Dict]:
        rows: List[Dict] = []
        texts = [
            "Intro sentence one.",
            "Intro sentence two.",
            "Intro sentence three.",
            "Intro sentence four.",
            "Details sentence five.",
            "Details sentence six.",
            "Details sentence seven.",
            "Details sentence eight.",
        ]
        for idx, text in enumerate(texts, start=1):
            rows.append(
                {
                    "doc_id": "doc",
                    "sent_id": idx,
                    "text": text,
                    "section_path": "intro" if idx <= 4 else "details",
                    "paragraph_id": 0 if idx <= 4 else 1,
                }
            )
        return rows

    def test_respect_paragraph_boundary(self) -> None:
        sentence_rows = self._make_sentence_rows()
        dim = 4
        S = np.zeros((len(sentence_rows), dim), dtype=np.float32)
        S[:4, 0] = 1.0
        S[4:, 1] = 1.0
        params = IDCParams(
            lam=0.0,
            max_len=8,
            min_len=2,
            boundary_penalty=0.0,
            coherence_weight=0.20,
            merge_adjacent=False,
            structural_priors=True,
            para_discount=0.5,
            diversity_weight=0.0,
            length_penalty_mode="linear",
            min_chunk_sent=2,
            max_chunk_sent=8,
            respect_paragraphs=True,
        )
        intent_rows = [
            {"doc_id": "doc", "text": "intro"},
            {"doc_id": "doc", "text": "details"},
        ]
        intent_vectors = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        result = segment_document(
            "doc",
            sentence_rows,
            S,
            intent_rows,
            intent_vectors,
            params,
        )
        self.assertGreaterEqual(len(result.chunks), 2)
        boundaries = {chunk["end_sent"] for chunk in result.chunks}
        self.assertIn(4, boundaries)
        self.assertTrue(all(chunk["num_sentences"] >= 2 for chunk in result.chunks))

    def test_refine_merges_short_chunk(self) -> None:
        sentence_rows = []
        for idx in range(1, 6):
            sentence_rows.append({"doc_id": "doc", "sent_id": idx, "text": f"Sentence {idx}."})
        S = np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0.0, 1, 0],
                [0.0, 1, 0],
                [0.0, 1, 0],
            ],
            dtype=np.float32,
        )
        chunks = [(0, 1, -1, 0.5), (2, 2, -1, 0.2), (3, 4, -1, 0.4)]
        params = IDCParams(
            lam=0.05,
            max_len=5,
            min_len=2,
            boundary_penalty=0.1,
            coherence_weight=0.05,
            merge_adjacent=False,
            structural_priors=False,
            para_discount=0.5,
            diversity_weight=0.0,
            length_penalty_mode="linear",
            min_chunk_sent=2,
            max_chunk_sent=5,
            respect_paragraphs=False,
        )
        refined = refine_chunks(
            chunks,
            S,
            np.zeros((0, S.shape[1]), dtype=np.float32),
            params,
            sentence_rows=sentence_rows,
            sentence_texts=[row["text"] for row in sentence_rows],
        )
        self.assertTrue(all(end - start + 1 >= 2 for start, end, *_ in refined))


class MultiViewTests(unittest.TestCase):
    def test_dense_scores_collapse_to_one_chunk(self) -> None:
        query_vec = np.array([[1.0, 0.0]], dtype=np.float32)
        view_matrix = np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.0],
                [0.8, 0.0],
            ],
            dtype=np.float32,
        )
        view_records = [
            ViewRecord(chunk_uid="chunk-1", view="text", vector_index=0),
            ViewRecord(chunk_uid="chunk-1", view="intent", vector_index=1),
            ViewRecord(chunk_uid="chunk-1", view="summary", vector_index=2),
            ViewRecord(chunk_uid="chunk-1", view="keywords", vector_index=3),
        ]
        scores = dense_chunk_scores(query_vec, view_matrix, view_records)[0]
        self.assertEqual(set(scores.keys()), {"chunk-1"})
        self.assertAlmostEqual(scores["chunk-1"][0], 1.0, places=5)


class HybridRetrievalTests(unittest.TestCase):
    def test_hybrid_and_rerank(self) -> None:
        dense_scores = {"A": 0.6, "B": 0.5}
        bm25_index = BM25Index({"A": "alpha beta", "B": "beta gamma gamma"})
        sparse_scores = bm25_index.score("beta gamma")
        dense_weight, sparse_weight = parse_hybrid_weights("0.4,0.6")
        combined = combine_dense_sparse(dense_scores, sparse_scores, dense_weight, sparse_weight)
        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        self.assertEqual(ranked[0][0], "B")
        candidates = [
            {"chunk_uid": key, "score": score, "text": text, "intent": None, "num_sentences": 3}
            for key, score in ranked
            for text in ["alpha beta" if key == "A" else "beta gamma gamma"]
        ]
        reranked = rerank_candidates("beta gamma", candidates, method="lexical", bm25=bm25_index, min_chunk_sent=2)
        self.assertEqual(reranked[0]["chunk_uid"], "B")


class MetricsTests(unittest.TestCase):
    def test_answer_coverage(self) -> None:
        chunk_records = {
            "c1": ChunkRecord("c1", "doc", 1, 3, "text1", None, None, [], 1.0, 3),
            "c2": ChunkRecord("c2", "doc", 4, 6, "text2", None, None, [], 1.0, 3),
        }
        spans = {("doc", 1): {"start_sent": 4, "end_sent": 6, "answerable": True}}
        coverage = compute_answer_coverage(spans, chunk_records)
        self.assertEqual(coverage, 1.0)

    def test_coherence_increases_after_merge(self) -> None:
        sentence_rows = []
        for idx in range(1, 5):
            sentence_rows.append({"doc_id": "doc", "sent_id": idx, "text": f"Sentence {idx}."})
        S = np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0.2, 1, 0],
                [0.2, 1, 0],
            ],
            dtype=np.float32,
        )
        chunks = [(0, 1, -1, 0.5), (2, 2, -1, 0.2), (3, 3, -1, 0.2)]
        params = IDCParams(
            lam=0.01,
            max_len=4,
            min_len=2,
            boundary_penalty=0.1,
            coherence_weight=0.05,
            merge_adjacent=False,
            structural_priors=False,
            para_discount=0.5,
            diversity_weight=0.0,
            length_penalty_mode="linear",
            min_chunk_sent=2,
            max_chunk_sent=4,
            respect_paragraphs=False,
        )
        before = [chunk_coherence(S, start, end) for start, end, *_ in chunks]
        refined = refine_chunks(
            chunks,
            S,
            np.zeros((0, S.shape[1]), dtype=np.float32),
            params,
            sentence_rows=sentence_rows,
            sentence_texts=[row["text"] for row in sentence_rows],
        )
        after = [chunk_coherence(S, start, end) for start, end, *_ in refined]
        self.assertGreaterEqual(sum(after) / len(after), sum(before) / len(before))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
