"""Document-oriented collection with automatic embedding and BM25.

``TextCollection`` wraps a Rust ``Collection`` and adds:

- Automatic dense embedding generation from text
- BM25 full-text keyword indexing via the existing sparse vector infrastructure
- Hybrid semantic + keyword search

Example::

    import quiver_vector_db as quiver
    from quiver_vector_db import TextCollection, SentenceTransformerEmbedding

    db = quiver.Client(path="./data")
    col = db.create_collection("docs", dimensions=384, metric="cosine")

    text_col = TextCollection(col, SentenceTransformerEmbedding("all-MiniLM-L6-v2"))

    text_col.add(ids=[1, 2], documents=["Hello world", "Vector search is fast"])
    hits = text_col.query("greeting", k=5)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .bm25 import BM25


class TextCollection:
    """A document-oriented collection with automatic embedding + BM25.

    Wraps a Rust ``Collection`` and provides text-in / text-out methods.
    Dense embeddings are generated via the provided ``embedding_function``.
    BM25 sparse vectors are automatically generated and stored via the
    existing ``upsert_hybrid()`` infrastructure.
    """

    def __init__(
        self,
        collection: Any,
        embedding_function: Any,
        enable_bm25: bool = True,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        """
        Args:
            collection: A Rust ``Collection`` from ``Client.create_collection()``
                or ``Client.get_collection()``.
            embedding_function: Any callable satisfying the ``EmbeddingFunction``
                protocol (``__call__(texts) -> list[list[float]]``).
            enable_bm25: Enable BM25 full-text indexing (default ``True``).
            bm25_k1: BM25 k1 parameter (term frequency saturation).
            bm25_b: BM25 b parameter (document length normalization).
        """
        self._collection = collection
        self._ef = embedding_function
        self._enable_bm25 = enable_bm25
        self._bm25: Optional[BM25] = BM25(k1=bm25_k1, b=bm25_b) if enable_bm25 else None

    def add(
        self,
        ids: Sequence[int],
        documents: Sequence[str],
        payloads: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        """Add documents by text. Embedding and BM25 indexing happen automatically.

        Args:
            ids: Unique integer IDs (one per document).
            documents: Text strings to embed and index.
            payloads: Optional metadata dicts (one per document).

        Raises:
            ValueError: If ``ids``, ``documents``, and ``payloads`` have
                mismatched lengths.
        """
        if len(ids) != len(documents):
            raise ValueError(
                f"ids and documents must have the same length "
                f"(got {len(ids)} and {len(documents)})"
            )
        if payloads is not None and len(payloads) != len(ids):
            raise ValueError(
                f"payloads must have the same length as ids "
                f"(got {len(payloads)} and {len(ids)})"
            )

        # Batch embed all texts at once (efficient for API-based providers)
        vectors = self._ef(list(documents))

        for i, (doc_id, doc_text, vector) in enumerate(zip(ids, documents, vectors)):
            payload = dict(payloads[i]) if payloads and payloads[i] else {}
            # Store document text in payload for retrieval
            payload["_document"] = doc_text

            if self._bm25 is not None:
                sparse = self._bm25.index_document(doc_id, doc_text)
                self._collection.upsert_hybrid(
                    id=doc_id,
                    vector=vector,
                    sparse_vector=sparse if sparse else None,
                    payload=payload,
                )
            else:
                self._collection.upsert(
                    id=doc_id,
                    vector=vector,
                    payload=payload,
                )

    def query(
        self,
        query_text: str,
        k: int = 10,
        mode: str = "hybrid",
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search by natural language query.

        Args:
            query_text: Natural language query string.
            k: Number of results to return.
            mode: Search mode:

                - ``"hybrid"`` (default) — weighted fusion of dense + BM25
                - ``"semantic"`` — dense embedding search only
                - ``"keyword"`` — BM25 keyword search only

            dense_weight: Weight for dense similarity (hybrid mode).
            sparse_weight: Weight for sparse/keyword similarity (hybrid mode).
            filter: Optional payload filter dict.

        Returns:
            List of result dicts. In semantic mode: ``{id, distance, document, payload}``.
            In hybrid/keyword mode: ``{id, score, dense_distance, sparse_score, document, payload}``.
        """
        if mode == "semantic" or (mode == "hybrid" and self._bm25 is None):
            query_vector = self._ef([query_text])[0]
            raw = self._collection.search(query=query_vector, k=k, filter=filter)
            return self._format_dense_results(raw)

        if mode == "keyword":
            if self._bm25 is None:
                raise ValueError("BM25 is not enabled on this TextCollection")
            query_vector = self._ef([query_text])[0]
            sparse_query = self._bm25.encode_query(query_text)
            if not sparse_query:
                return []
            raw = self._collection.search_hybrid(
                dense_query=query_vector,
                sparse_query=sparse_query,
                k=k,
                dense_weight=0.0,
                sparse_weight=1.0,
                filter=filter,
            )
            return self._format_hybrid_results(raw)

        if mode == "hybrid":
            query_vector = self._ef([query_text])[0]
            sparse_query = self._bm25.encode_query(query_text)
            if not sparse_query:
                # No known terms — fall back to pure semantic
                raw = self._collection.search(query=query_vector, k=k, filter=filter)
                return self._format_dense_results(raw)
            raw = self._collection.search_hybrid(
                dense_query=query_vector,
                sparse_query=sparse_query,
                k=k,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                filter=filter,
            )
            return self._format_hybrid_results(raw)

        raise ValueError(
            f"Unknown mode {mode!r}; expected 'hybrid', 'semantic', or 'keyword'"
        )

    def delete(self, ids: Sequence[int]) -> None:
        """Delete documents by ID.

        Updates BM25 statistics if BM25 is enabled.
        """
        for doc_id in ids:
            self._collection.delete(id=doc_id)
            if self._bm25 is not None:
                self._bm25.remove_document(doc_id)

    @property
    def count(self) -> int:
        """Number of documents in the collection."""
        return self._collection.count

    @property
    def name(self) -> str:
        """Collection name."""
        return self._collection.name

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _format_dense_results(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for r in raw:
            payload = dict(r.get("payload") or {})
            document = payload.pop("_document", None)
            results.append({
                "id": r["id"],
                "distance": r.get("distance", 0.0),
                "document": document,
                "payload": payload if payload else None,
            })
        return results

    @staticmethod
    def _format_hybrid_results(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for r in raw:
            payload = dict(r.get("payload") or {})
            document = payload.pop("_document", None)
            results.append({
                "id": r["id"],
                "score": r.get("score", 0.0),
                "dense_distance": r.get("dense_distance", 0.0),
                "sparse_score": r.get("sparse_score", 0.0),
                "document": document,
                "payload": payload if payload else None,
            })
        return results
