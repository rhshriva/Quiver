"""Okapi BM25 tokenizer and sparse vector generator.

Produces sparse vectors compatible with Quiver's SparseIndex, enabling
full-text keyword search via the existing ``upsert_hybrid()`` /
``search_hybrid()`` infrastructure.

Zero external dependencies — uses only the Python standard library.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

# BM25 defaults (Okapi BM25 standard values)
_DEFAULT_K1 = 1.5
_DEFAULT_B = 0.75

# Simple tokenizer: lowercase, split on non-alphanumeric
_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "how", "if",
    "in", "into", "is", "it", "its", "no", "not", "of", "on", "or",
    "our", "she", "so", "such", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "was", "we",
    "were", "what", "when", "which", "who", "will", "with", "you",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords, min length 2."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]


class BM25:
    """Okapi BM25 tokenizer and sparse vector generator.

    Tracks vocabulary, document frequencies, and average document length.
    Produces sparse vectors (``dict[int, float]``) compatible with
    ``Collection.upsert_hybrid(sparse_vector=...)``.

    Example::

        bm25 = BM25()
        sparse = bm25.index_document(1, "the quick brown fox")
        query  = bm25.encode_query("brown fox")
        # sparse and query are {dim_index: weight} dicts
    """

    def __init__(self, k1: float = _DEFAULT_K1, b: float = _DEFAULT_B) -> None:
        self.k1 = k1
        self.b = b
        self._vocab: Dict[str, int] = {}
        self._next_id: int = 0
        self._df: Dict[int, int] = {}           # term_id -> document frequency
        self._doc_count: int = 0
        self._total_dl: int = 0                  # sum of all doc lengths
        self._doc_lengths: Dict[int, int] = {}   # doc_id -> token count

    @property
    def doc_count(self) -> int:
        """Total number of indexed documents."""
        return self._doc_count

    @property
    def vocab_size(self) -> int:
        """Number of unique terms in the vocabulary."""
        return len(self._vocab)

    @property
    def avg_dl(self) -> float:
        """Average document length (in tokens)."""
        return self._total_dl / self._doc_count if self._doc_count > 0 else 0.0

    def _get_or_create_term_id(self, token: str) -> int:
        tid = self._vocab.get(token)
        if tid is None:
            tid = self._next_id
            self._vocab[token] = tid
            self._next_id += 1
        return tid

    def index_document(self, doc_id: int, text: str) -> Dict[int, float]:
        """Tokenize text, update IDF stats, return BM25 sparse vector.

        Args:
            doc_id: The vector ID (same as used with ``upsert_hybrid``).
            text: Raw document text to index.

        Returns:
            Sparse vector as ``{dimension_index: bm25_weight}`` suitable for
            ``Collection.upsert_hybrid(sparse_vector=...)``.
            Empty dict if text produces no tokens.
        """
        tokens = _tokenize(text)
        if not tokens:
            return {}

        # If re-indexing, remove old stats first
        if doc_id in self._doc_lengths:
            old_dl = self._doc_lengths[doc_id]
            self._doc_count -= 1
            self._total_dl -= old_dl

        tf_counts = Counter(tokens)
        dl = len(tokens)

        # Update document frequency for each unique term
        for token in tf_counts:
            tid = self._get_or_create_term_id(token)
            self._df[tid] = self._df.get(tid, 0) + 1

        # Update global stats
        self._doc_count += 1
        self._total_dl += dl
        self._doc_lengths[doc_id] = dl

        # Compute BM25 weights
        avgdl = self.avg_dl
        sparse: Dict[int, float] = {}
        for token, tf in tf_counts.items():
            tid = self._vocab[token]
            df = self._df.get(tid, 0)
            idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
            weight = idf * numerator / denominator
            if weight > 0:
                sparse[tid] = weight

        return sparse

    def encode_query(self, text: str) -> Dict[int, float]:
        """Encode a query into an IDF-weighted sparse vector.

        Only uses terms already in the vocabulary — unseen terms are ignored.

        Args:
            text: Query text.

        Returns:
            Sparse vector as ``{dimension_index: idf_weight}``.
        """
        tokens = _tokenize(text)
        tf_counts = Counter(tokens)
        sparse: Dict[int, float] = {}
        for token, tf in tf_counts.items():
            tid = self._vocab.get(token)
            if tid is None:
                continue
            df = self._df.get(tid, 0)
            if df == 0:
                continue
            idf = math.log((self._doc_count - df + 0.5) / (df + 0.5) + 1.0)
            if idf > 0:
                sparse[tid] = idf * tf
        return sparse

    def remove_document(self, doc_id: int) -> None:
        """Remove a document's contribution to global stats.

        Note: per-term document frequencies are not decremented (would
        require storing per-doc term sets). They self-correct as the
        corpus evolves. This is the standard BM25 trade-off used by
        Elasticsearch and similar systems.
        """
        if doc_id not in self._doc_lengths:
            return
        dl = self._doc_lengths.pop(doc_id)
        self._doc_count -= 1
        self._total_dl -= dl

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialize BM25 state to a JSON file."""
        state = {
            "k1": self.k1,
            "b": self.b,
            "vocab": self._vocab,
            "next_id": self._next_id,
            "df": {str(k): v for k, v in self._df.items()},
            "doc_count": self._doc_count,
            "total_dl": self._total_dl,
            "doc_lengths": {str(k): v for k, v in self._doc_lengths.items()},
        }
        Path(path).write_text(json.dumps(state), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BM25":
        """Load BM25 state from a JSON file."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(k1=raw["k1"], b=raw["b"])
        obj._vocab = raw["vocab"]
        obj._next_id = raw["next_id"]
        obj._df = {int(k): v for k, v in raw["df"].items()}
        obj._doc_count = raw["doc_count"]
        obj._total_dl = raw["total_dl"]
        obj._doc_lengths = {int(k): v for k, v in raw["doc_lengths"].items()}
        return obj
