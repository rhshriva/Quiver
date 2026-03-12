"""Functional tests for Client and Collection from a user's perspective."""

import os
import pytest
import quiver_vector_db as quiver
from conftest import SAMPLE_DIM, random_vectors, random_vector, ALL_INDEX_TYPES


# ---------------------------------------------------------------------------
# Client: collection management
# ---------------------------------------------------------------------------


class TestClientCollections:
    def test_create_and_list(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        db.create_collection("a", dimensions=SAMPLE_DIM, metric="cosine")
        db.create_collection("b", dimensions=SAMPLE_DIM, metric="l2")
        names = db.list_collections()
        assert sorted(names) == ["a", "b"]

    def test_get_collection(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        db.create_collection("docs", dimensions=SAMPLE_DIM)
        col = db.get_collection("docs")
        assert col.name == "docs"

    def test_get_collection_not_found(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        with pytest.raises(KeyError):
            db.get_collection("nonexistent")

    def test_get_or_create(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        col1 = db.get_or_create_collection("items", dimensions=SAMPLE_DIM)
        col1.upsert(id=1, vector=random_vector())
        col2 = db.get_or_create_collection("items", dimensions=SAMPLE_DIM)
        assert col2.count == 1  # same collection, data persisted

    def test_delete_collection(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        db.create_collection("temp", dimensions=SAMPLE_DIM)
        assert "temp" in db.list_collections()
        db.delete_collection("temp")
        assert "temp" not in db.list_collections()

    def test_duplicate_collection_raises(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        db.create_collection("dup", dimensions=SAMPLE_DIM)
        with pytest.raises(KeyError):
            db.create_collection("dup", dimensions=SAMPLE_DIM)


# ---------------------------------------------------------------------------
# Collection: upsert, search, delete
# ---------------------------------------------------------------------------


class TestCollectionBasics:
    def test_upsert_and_search(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        col = db.create_collection("test", dimensions=4, metric="l2")
        col.upsert(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        col.upsert(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        col.upsert(id=3, vector=[0.0, 0.0, 1.0, 0.0])

        hits = col.search(query=[1.0, 0.0, 0.0, 0.0], k=1)
        assert len(hits) == 1
        assert hits[0]["id"] == 1
        assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-5)

    def test_upsert_with_payload(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        col = db.create_collection("test", dimensions=4, metric="l2")
        col.upsert(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={"tag": "hello"})

        hits = col.search(query=[1.0, 0.0, 0.0, 0.0], k=1)
        assert hits[0]["payload"]["tag"] == "hello"

    def test_count_property(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        col = db.create_collection("test", dimensions=4)
        assert col.count == 0
        col.upsert(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        col.upsert(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        assert col.count == 2

    def test_delete_vector(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        col = db.create_collection("test", dimensions=4, metric="l2")
        col.upsert(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        col.upsert(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        assert col.count == 2
        col.delete(id=1)
        assert col.count == 1

        hits = col.search(query=[1.0, 0.0, 0.0, 0.0], k=5)
        ids = [h["id"] for h in hits]
        assert 1 not in ids

    def test_dimension_mismatch(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        col = db.create_collection("test", dimensions=4)
        with pytest.raises(ValueError, match="dimension mismatch"):
            col.upsert(id=1, vector=[1.0, 2.0])  # expects 4, got 2


# ---------------------------------------------------------------------------
# Filtered search
# ---------------------------------------------------------------------------


class TestFilteredSearch:
    @pytest.fixture()
    def col(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        col = db.create_collection("filt", dimensions=4, metric="l2")
        col.upsert(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={"cat": "tech", "score": 4.8})
        col.upsert(id=2, vector=[0.0, 1.0, 0.0, 0.0], payload={"cat": "science", "score": 3.2})
        col.upsert(id=3, vector=[0.0, 0.0, 1.0, 0.0], payload={"cat": "tech", "score": 2.5})
        col.upsert(id=4, vector=[0.0, 0.0, 0.0, 1.0], payload={"cat": "art", "score": 4.1})
        return col

    def test_eq_filter(self, col):
        hits = col.search(query=[0.5, 0.5, 0.5, 0.5], k=10, filter={"cat": {"$eq": "tech"}})
        ids = {h["id"] for h in hits}
        assert ids == {1, 3}

    def test_gte_filter(self, col):
        hits = col.search(query=[0.5, 0.5, 0.5, 0.5], k=10, filter={"score": {"$gte": 4.0}})
        ids = {h["id"] for h in hits}
        assert ids == {1, 4}

    def test_in_filter(self, col):
        hits = col.search(query=[0.5, 0.5, 0.5, 0.5], k=10, filter={"cat": {"$in": ["tech", "art"]}})
        ids = {h["id"] for h in hits}
        assert ids == {1, 3, 4}

    def test_and_filter(self, col):
        hits = col.search(
            query=[0.5, 0.5, 0.5, 0.5],
            k=10,
            filter={
                "$and": [
                    {"cat": {"$eq": "tech"}},
                    {"score": {"$gte": 4.0}},
                ]
            },
        )
        ids = {h["id"] for h in hits}
        assert ids == {1}

    def test_or_filter(self, col):
        hits = col.search(
            query=[0.5, 0.5, 0.5, 0.5],
            k=10,
            filter={
                "$or": [
                    {"cat": {"$eq": "art"}},
                    {"score": {"$gte": 4.5}},
                ]
            },
        )
        ids = {h["id"] for h in hits}
        assert ids == {1, 4}


# ---------------------------------------------------------------------------
# All index types via Client
# ---------------------------------------------------------------------------


class TestAllIndexTypesViaClient:
    @pytest.mark.parametrize("index_type", ALL_INDEX_TYPES)
    def test_upsert_and_search(self, tmp_path, index_type):
        db = quiver.Client(path=str(tmp_path / index_type))
        col = db.create_collection(
            "test", dimensions=4, metric="cosine", index_type=index_type
        )
        col.upsert(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        col.upsert(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        col.upsert(id=3, vector=[0.0, 0.0, 1.0, 0.0])

        hits = col.search(query=[1.0, 0.0, 0.0, 0.0], k=3)
        assert len(hits) >= 1
        # For all index types, the exact match should be in results
        ids = [h["id"] for h in hits]
        assert 1 in ids
