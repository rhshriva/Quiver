"""Tests for the Collection.upsert_batch() API."""

import pytest
import tempfile
import os

import quiver_vector_db as quiver


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield quiver.Client(path=tmpdir)


class TestUpsertBatch:
    """Tests for Collection.upsert_batch()."""

    def test_basic_batch(self, db):
        """Batch upsert inserts all vectors."""
        col = db.create_collection("test", dimensions=3, metric="l2")
        col.upsert_batch([
            (1, [1.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0]),
            (3, [0.0, 0.0, 1.0]),
        ])
        assert col.count == 3

    def test_batch_with_payloads(self, db):
        """Batch upsert with payloads stores metadata correctly."""
        col = db.create_collection("test", dimensions=3, metric="l2")
        col.upsert_batch([
            (1, [1.0, 0.0, 0.0], {"tag": "a"}),
            (2, [0.0, 1.0, 0.0], {"tag": "b"}),
            (3, [0.0, 0.0, 1.0], None),
        ])
        results = col.search(query=[1.0, 0.0, 0.0], k=1)
        assert results[0]["id"] == 1
        assert results[0]["payload"]["tag"] == "a"

    def test_batch_search_correctness(self, db):
        """Vectors from batch upsert are searchable with correct results."""
        col = db.create_collection("test", dimensions=3, metric="cosine")
        col.upsert_batch([
            (1, [1.0, 0.0, 0.0]),
            (2, [0.0, 1.0, 0.0]),
            (3, [0.0, 0.0, 1.0]),
        ])
        results = col.search(query=[1.0, 0.1, 0.0], k=1)
        assert results[0]["id"] == 1

    def test_batch_upsert_overwrites(self, db):
        """Batch upsert overwrites existing vectors."""
        col = db.create_collection("test", dimensions=3, metric="l2")
        col.upsert(id=1, vector=[1.0, 0.0, 0.0])
        # Overwrite id=1 via batch
        col.upsert_batch([
            (1, [0.0, 1.0, 0.0], {"updated": True}),
        ])
        assert col.count == 1
        results = col.search(query=[0.0, 1.0, 0.0], k=1)
        assert results[0]["id"] == 1
        assert results[0]["distance"] < 1e-6
        assert results[0]["payload"]["updated"] is True

    def test_empty_batch(self, db):
        """Empty batch is a no-op."""
        col = db.create_collection("test", dimensions=3, metric="l2")
        col.upsert_batch([])
        assert col.count == 0

    def test_large_batch(self, db):
        """Batch upsert handles large batches."""
        import random
        random.seed(42)
        col = db.create_collection("test", dimensions=128, metric="cosine")
        n = 1000
        entries = [(i, [random.gauss(0, 1) for _ in range(128)]) for i in range(n)]
        col.upsert_batch(entries)
        assert col.count == n

    def test_batch_mixed_payloads(self, db):
        """Mix of tuples with and without payloads."""
        col = db.create_collection("test", dimensions=2, metric="l2")
        col.upsert_batch([
            (1, [1.0, 0.0], {"x": 1}),
            (2, [0.0, 1.0]),  # no payload (2-tuple)
        ])
        assert col.count == 2
        r1 = col.search(query=[1.0, 0.0], k=1)
        assert r1[0]["payload"]["x"] == 1
        r2 = col.search(query=[0.0, 1.0], k=1)
        assert r2[0]["id"] == 2

    def test_batch_persists(self, db):
        """Batch-upserted data survives reopen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = quiver.Client(path=tmpdir)
            col = db1.create_collection("persist", dimensions=3, metric="l2")
            col.upsert_batch([
                (1, [1.0, 0.0, 0.0], {"key": "val"}),
                (2, [0.0, 1.0, 0.0]),
            ])
            del db1

            db2 = quiver.Client(path=tmpdir)
            col2 = db2.get_collection("persist")
            assert col2.count == 2
            results = col2.search(query=[1.0, 0.0, 0.0], k=1)
            assert results[0]["id"] == 1

    def test_batch_all_index_types(self, db):
        """Batch upsert works across all index types."""
        for it in ["flat", "hnsw", "quantized_flat", "fp16_flat", "binary_flat"]:
            col = db.create_collection(f"test_{it}", dimensions=8, metric="l2", index_type=it)
            entries = [(i, [float(i == j) for j in range(8)]) for i in range(5)]
            col.upsert_batch(entries)
            assert col.count == 5

    def test_batch_invalid_entry(self, db):
        """Invalid entries raise ValueError."""
        col = db.create_collection("test", dimensions=3, metric="l2")
        with pytest.raises((ValueError, TypeError)):
            col.upsert_batch([(1,)])  # too short
