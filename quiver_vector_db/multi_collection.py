"""Multi-vector / multi-modal collection.

``MultiVectorCollection`` lets you store multiple named embedding spaces per
document — for example "text" and "image" vectors — and search across one or
more spaces with weighted fusion.

Example::

    import quiver_vector_db as quiver
    from quiver_vector_db import MultiVectorCollection

    db = quiver.Client(path="./data")

    multi = MultiVectorCollection(
        client=db,
        name="products",
        vector_spaces={
            "text":  {"dimensions": 384, "metric": "cosine"},
            "image": {"dimensions": 512, "metric": "cosine"},
        },
    )

    # Upsert with one or more vector spaces
    multi.upsert(id=1, vectors={"text": [...], "image": [...]}, payload={"title": "Shirt"})
    multi.upsert(id=2, vectors={"text": [...]}, payload={"title": "Pants"})  # image optional

    # Search within a single space
    hits = multi.search("text", query=[...], k=5)

    # Multi-space search with weighted fusion
    hits = multi.search_multi(
        queries={"text": [...], "image": [...]},
        k=5,
        weights={"text": 0.6, "image": 0.4},
    )

    # Batch upsert
    multi.upsert_batch([
        (1, {"text": [...], "image": [...]}, {"title": "A"}),
        (2, {"text": [...]}, {"title": "B"}),
    ])
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class MultiVectorCollection:
    """A collection supporting multiple named vector spaces per document.

    Each vector space is backed by a separate Quiver ``Collection`` under the
    hood. Documents share the same integer IDs across all spaces, and payloads
    are stored in the *first* vector space alphabetically (to avoid
    duplication).
    """

    def __init__(
        self,
        client: Any,
        name: str,
        vector_spaces: Dict[str, Dict[str, Any]],
        index_type: str = "hnsw",
    ) -> None:
        """
        Args:
            client: A ``quiver_vector_db.Client`` instance.
            name: Base name for the collection group.
            vector_spaces: Mapping of ``space_name`` to config dict. Each
                config must include ``dimensions`` (int) and optionally
                ``metric`` (str, default ``"cosine"``).
            index_type: Index algorithm for all sub-collections (default ``"hnsw"``).
        """
        if not vector_spaces:
            raise ValueError("vector_spaces must have at least one entry")

        self._client = client
        self._name = name
        self._space_names = sorted(vector_spaces.keys())
        self._spaces = vector_spaces
        self._index_type = index_type

        # The "primary" space stores payloads (first alphabetically).
        self._primary_space = self._space_names[0]

        # Create or get sub-collections: {name}__{space}
        self._collections: Dict[str, Any] = {}
        for space in self._space_names:
            cfg = vector_spaces[space]
            dims = cfg["dimensions"]
            metric = cfg.get("metric", "cosine")
            col_name = f"{name}__{space}"
            try:
                col = client.get_collection(col_name)
            except KeyError:
                col = client.create_collection(
                    col_name,
                    dimensions=dims,
                    metric=metric,
                    index_type=index_type,
                )
            self._collections[space] = col

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        """Base name of the multi-vector collection."""
        return self._name

    @property
    def vector_spaces(self) -> List[str]:
        """Sorted list of vector space names."""
        return list(self._space_names)

    @property
    def count(self) -> int:
        """Number of documents (max across all vector spaces)."""
        return max(col.count for col in self._collections.values())

    # ── Upsert ──────────────────────────────────────────────────────────

    def upsert(
        self,
        id: int,
        vectors: Dict[str, List[float]],
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Insert or update a document with vectors in one or more spaces.

        Args:
            id: Unique integer ID.
            vectors: Mapping of ``space_name`` to vector. You can omit spaces
                — only provided spaces are written.
            payload: Optional metadata dict (stored once in the primary space).

        Raises:
            KeyError: If a vector space name is not recognised.
        """
        for space, vec in vectors.items():
            if space not in self._collections:
                raise KeyError(f"unknown vector space: {space!r}")
            p = payload if space == self._primary_space else None
            self._collections[space].upsert(id=id, vector=vec, payload=p)

    def upsert_batch(
        self,
        entries: Sequence[
            Union[
                Tuple[int, Dict[str, List[float]]],
                Tuple[int, Dict[str, List[float]], Optional[Dict[str, Any]]],
            ]
        ],
    ) -> None:
        """Batch upsert multiple documents at once.

        Args:
            entries: List of tuples ``(id, vectors)`` or
                ``(id, vectors, payload)``. ``vectors`` is a dict mapping
                space names to vectors.
        """
        # Bucket entries by space for efficient batch upsert
        per_space: Dict[str, List[Tuple[int, List[float], Optional[Dict[str, Any]]]]] = {
            s: [] for s in self._space_names
        }

        for entry in entries:
            if len(entry) == 2:
                doc_id, vectors = entry[0], entry[1]
                payload = None
            else:
                doc_id, vectors, payload = entry[0], entry[1], entry[2]

            for space, vec in vectors.items():
                if space not in self._collections:
                    raise KeyError(f"unknown vector space: {space!r}")
                p = payload if space == self._primary_space else None
                per_space[space].append((doc_id, vec, p))

        # Batch upsert per space
        for space, batch in per_space.items():
            if batch:
                self._collections[space].upsert_batch(
                    [(doc_id, vec, p) for doc_id, vec, p in batch]
                )

    # ── Search ──────────────────────────────────────────────────────────

    def search(
        self,
        vector_space: str,
        query: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search within a single vector space.

        Args:
            vector_space: Which vector space to search.
            query: Query vector.
            k: Number of results.
            filter: Optional payload filter (only works if this is the primary
                space that stores payloads).

        Returns:
            List of dicts with ``id``, ``distance``, and optionally ``payload``.
        """
        if vector_space not in self._collections:
            raise KeyError(f"unknown vector space: {vector_space!r}")
        return self._collections[vector_space].search(
            query=query, k=k, filter=filter,
        )

    def search_multi(
        self,
        queries: Dict[str, List[float]],
        k: int = 10,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """Search across multiple vector spaces with weighted fusion.

        Results are ranked by a weighted combination of normalised distances
        from each space (lower distance = better).

        Args:
            queries: Mapping of ``space_name`` to query vector.
            k: Number of results.
            weights: Optional mapping of ``space_name`` to weight. Defaults to
                equal weight across all queried spaces.

        Returns:
            List of dicts with ``id``, ``score``, ``distances`` (per-space),
            and optionally ``payload``.
        """
        if not queries:
            return []

        for space in queries:
            if space not in self._collections:
                raise KeyError(f"unknown vector space: {space!r}")

        # Default: equal weights
        if weights is None:
            w = 1.0 / len(queries)
            weights = {s: w for s in queries}

        # Normalise weights
        total = sum(weights.get(s, 0.0) for s in queries)
        if total <= 0:
            total = 1.0
        norm_weights = {s: weights.get(s, 0.0) / total for s in queries}

        # Overscan: fetch more results per space to get enough candidates
        overscan = max(k * 5, 50)

        # Collect raw results per space
        per_space_results: Dict[str, List[Dict[str, Any]]] = {}
        for space, query in queries.items():
            per_space_results[space] = self._collections[space].search(
                query=query, k=overscan,
            )

        # Build a map: id -> {space: distance}
        candidates: Dict[int, Dict[str, float]] = {}
        for space, results in per_space_results.items():
            for r in results:
                doc_id = r["id"]
                if doc_id not in candidates:
                    candidates[doc_id] = {}
                candidates[doc_id][space] = r["distance"]

        # Normalise distances per space (min-max to [0, 1])
        space_mins: Dict[str, float] = {}
        space_maxs: Dict[str, float] = {}
        for space in queries:
            dists = [
                candidates[doc_id].get(space)
                for doc_id in candidates
                if space in candidates[doc_id]
            ]
            if dists:
                space_mins[space] = min(dists)
                space_maxs[space] = max(dists)
            else:
                space_mins[space] = 0.0
                space_maxs[space] = 1.0

        # Score each candidate
        scored: List[Tuple[int, float, Dict[str, float]]] = []
        for doc_id, distances in candidates.items():
            score = 0.0
            for space in queries:
                if space in distances:
                    d = distances[space]
                    range_ = space_maxs[space] - space_mins[space]
                    norm_d = (d - space_mins[space]) / range_ if range_ > 0 else 0.0
                    # Lower distance = higher similarity
                    score += norm_weights[space] * (1.0 - norm_d)
                # If missing: contribute 0 for this space
            scored.append((doc_id, score, distances))

        # Sort by score descending, take top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = scored[:k]

        # Fetch payloads from primary space
        results: List[Dict[str, Any]] = []
        for doc_id, score, distances in top_k:
            results.append({
                "id": doc_id,
                "score": score,
                "distances": distances,
            })

        return results

    # ── Delete ──────────────────────────────────────────────────────────

    def delete(self, id: int) -> None:
        """Delete a document across all vector spaces.

        Args:
            id: Document ID to delete.
        """
        for col in self._collections.values():
            col.delete(id=id)

    def delete_batch(self, ids: Sequence[int]) -> None:
        """Delete multiple documents across all vector spaces.

        Args:
            ids: List of document IDs to delete.
        """
        for doc_id in ids:
            self.delete(doc_id)
