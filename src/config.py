"""Central configuration objects and helpers for the IDC pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Tuple

DEFAULT_VIEWS: Tuple[str, ...] = ("text", "intent", "summary", "keywords")
DEFAULT_HYBRID_WEIGHTS: Tuple[float, float] = (0.6, 0.4)
DEFAULT_HYBRID_WEIGHTS_SHORT: Tuple[float, float] = (0.5, 0.5)  # Balanced for short docs
DEFAULT_HYBRID_WEIGHTS_LONG: Tuple[float, float] = (0.7, 0.3)   # Dense-favored for long docs


def _normalise_views(views: Iterable[str]) -> Tuple[str, ...]:
    """Normalise, deduplicate, and order view names while preserving intent."""
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in views:
        name = raw.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


@dataclass(slots=True)
class IDCConfig:
    """Top-level configuration for IDC chunking, indexing, and evaluation."""

    coherence_weight: float = 0.10
    length_penalty: str = "linear"
    min_chunk_sent: int = 2
    max_chunk_sent: int = 10
    respect_paragraphs: bool = True
    structural_priors: bool = True
    postprocess: bool = True
    hybrid_retrieval: bool = True
    hybrid_weights: Tuple[float, float] = field(default_factory=lambda: DEFAULT_HYBRID_WEIGHTS)
    multi_view_index: bool = True
    views: Tuple[str, ...] = field(default_factory=lambda: DEFAULT_VIEWS)
    reranker: str = "lexical"
    topk: int = 10
    eval_answer_coverage: bool = True
    eval_coherence: bool = True

    def with_views(self, views: Iterable[str]) -> "IDCConfig":
        """Return a copy of the config with the provided view list."""
        return replace(self, views=_normalise_views(views) or DEFAULT_VIEWS)

    def hybrid_weight_tuple(self) -> Tuple[float, float]:
        """Return dense/sparse weights ensuring they sum to one when possible."""
        dense, sparse = self.hybrid_weights
        total = dense + sparse
        if total <= 0:
            return DEFAULT_HYBRID_WEIGHTS
        return (dense / total, sparse / total)


def get_optimal_hybrid_weights(doc_length: int) -> Tuple[float, float]:
    """Return optimal hybrid weights based on document characteristics.

    Args:
        doc_length: Number of sentences in the document

    Returns:
        Tuple of (dense_weight, sparse_weight) optimized for doc length
    """
    if doc_length < 500:  # Short documents (e.g., SQuAD articles)
        return DEFAULT_HYBRID_WEIGHTS_SHORT
    return DEFAULT_HYBRID_WEIGHTS_LONG


def parse_views_arg(arg: str | None) -> Tuple[str, ...]:
    """Parse a comma-separated view argument into a normalised tuple."""
    if arg is None or not arg.strip():
        return DEFAULT_VIEWS
    parts = [part for token in arg.split(',') if (part := token.strip())]
    views = _normalise_views(parts)
    return views or DEFAULT_VIEWS


__all__ = [
    "IDCConfig",
    "DEFAULT_VIEWS",
    "DEFAULT_HYBRID_WEIGHTS",
    "DEFAULT_HYBRID_WEIGHTS_SHORT",
    "DEFAULT_HYBRID_WEIGHTS_LONG",
    "get_optimal_hybrid_weights",
    "parse_views_arg",
]
