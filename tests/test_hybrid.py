"""Functional tests for hybrid dense+sparse search."""

import pytest
import quiver_vector_db as quiver


@pytest.fixture()
def hybrid_col(tmp_path):
    """Collection with both dense and sparse vectors."""
    db = quiver.Client(path=str(tmp_path))
    col = db.create_collection("hybrid", dimensions=4, metric="l2")

    col.upsert_hybrid(
        id=1,
        vector=[1.0, 0.0, 0.0, 0.0],
        sparse_vector={0: 0.9, 10: 0.5},
        payload={"cat": "tech"},
    )
    col.upsert_hybrid(
        id=2,
        vector=[0.0, 1.0, 0.0, 0.0],
        sparse_vector={10: 0.8, 20: 0.6},
        payload={"cat": "science"},
    )
    col.upsert_hybrid(
        id=3,
        vector=[0.0, 0.0, 1.0, 0.0],
        sparse_vector={0: 0.7, 20: 0.3},
        payload={"cat": "tech"},
    )
    return col


class TestUpsertHybrid:
    def test_counts(self, hybrid_col):
        assert hybrid_col.count == 3
        assert hybrid_col.sparse_count == 3

    def test_dense_only_upsert_no_sparse_count(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        col = db.create_collection("test", dimensions=4)
        col.upsert(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        assert col.count == 1
        assert col.sparse_count == 0


class TestSearchHybrid:
    def test_returns_expected_keys(self, hybrid_col):
        hits = hybrid_col.search_hybrid(
            dense_query=[1.0, 0.0, 0.0, 0.0],
            sparse_query={0: 1.0},
            k=3,
        )
        assert len(hits) >= 1
        hit = hits[0]
        assert "id" in hit
        assert "score" in hit
        assert "dense_distance" in hit
        assert "sparse_score" in hit

    def test_dense_only_weight(self, hybrid_col):
        """With sparse_weight=0, top result should match pure dense search."""
        dense_hits = hybrid_col.search(query=[1.0, 0.0, 0.0, 0.0], k=3)
        hybrid_hits = hybrid_col.search_hybrid(
            dense_query=[1.0, 0.0, 0.0, 0.0],
            sparse_query={0: 1.0},
            k=3,
            dense_weight=1.0,
            sparse_weight=0.0,
        )
        # Top-1 must match; remaining order may vary for equidistant vectors
        assert dense_hits[0]["id"] == hybrid_hits[0]["id"]
        assert {h["id"] for h in dense_hits} == {h["id"] for h in hybrid_hits}

    def test_hybrid_with_filter(self, hybrid_col):
        hits = hybrid_col.search_hybrid(
            dense_query=[1.0, 0.0, 0.0, 0.0],
            sparse_query={0: 1.0},
            k=10,
            filter={"cat": {"$eq": "tech"}},
        )
        ids = {h["id"] for h in hits}
        assert ids <= {1, 3}  # only tech vectors


class TestMixedUpsert:
    def test_mixed_dense_and_hybrid(self, tmp_path):
        """Can mix upsert() and upsert_hybrid() in the same collection."""
        db = quiver.Client(path=str(tmp_path))
        col = db.create_collection("mixed", dimensions=4, metric="l2")

        col.upsert(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        col.upsert_hybrid(
            id=2,
            vector=[0.0, 1.0, 0.0, 0.0],
            sparse_vector={5: 0.9},
        )
        assert col.count == 2
        assert col.sparse_count == 1

        # Dense search should find both
        hits = col.search(query=[0.5, 0.5, 0.0, 0.0], k=5)
        ids = {h["id"] for h in hits}
        assert ids == {1, 2}
