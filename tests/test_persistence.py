"""Tests for save/load and WAL-backed persistence."""

import os
import pytest
import quiver_vector_db as quiver


# ---------------------------------------------------------------------------
# Standalone index save/load
# ---------------------------------------------------------------------------


class TestFlatSaveLoad:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "flat.bin")
        idx = quiver.FlatIndex(dimensions=4, metric="l2")
        idx.add(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        idx.add(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        idx.save(path)

        loaded = quiver.FlatIndex.load(path)
        assert len(loaded) == 2
        results = loaded.search(query=[1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0]["id"] == 1

    def test_save_bad_path(self):
        idx = quiver.FlatIndex(dimensions=4, metric="l2")
        with pytest.raises(OSError):
            idx.save("/nonexistent/dir/flat.bin")


class TestHnswSaveLoad:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "hnsw.bin")
        idx = quiver.HnswIndex(dimensions=4, metric="l2")
        for i in range(20):
            idx.add(id=i, vector=[float(i), 0.0, 0.0, 0.0])
        idx.flush()
        idx.save(path)

        loaded = quiver.HnswIndex.load(path)
        assert len(loaded) == 20
        results = loaded.search(query=[5.0, 0.0, 0.0, 0.0], k=1)
        assert results[0]["id"] == 5


class TestQuantizedFlatSaveLoad:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "quant.bin")
        idx = quiver.QuantizedFlatIndex(dimensions=4, metric="l2")
        idx.add(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        idx.add(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        idx.save(path)

        loaded = quiver.QuantizedFlatIndex.load(path)
        assert len(loaded) == 2
        results = loaded.search(query=[1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0]["id"] == 1


class TestFp16FlatSaveLoad:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "fp16.bin")
        idx = quiver.Fp16FlatIndex(dimensions=4, metric="l2")
        idx.add(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        idx.add(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        idx.save(path)

        loaded = quiver.Fp16FlatIndex.load(path)
        assert len(loaded) == 2
        results = loaded.search(query=[1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0]["id"] == 1


class TestIvfSaveLoad:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "ivf.bin")
        idx = quiver.IvfIndex(dimensions=4, metric="l2", n_lists=4, nprobe=4, train_size=16)
        for i in range(20):
            idx.add(id=i, vector=[float(i), 0.0, 0.0, 0.0])
        idx.save(path)

        loaded = quiver.IvfIndex.load(path)
        assert len(loaded) == 20
        results = loaded.search(query=[5.0, 0.0, 0.0, 0.0], k=3)
        ids = [r["id"] for r in results]
        assert 5 in ids


class TestIvfPqSaveLoad:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "ivfpq.bin")
        idx = quiver.IvfPqIndex(dimensions=4, metric="l2", n_lists=4, nprobe=4, train_size=16, pq_m=2, pq_k_sub=4)
        for i in range(20):
            idx.add(id=i, vector=[float(i), 0.0, 0.0, 0.0])
        idx.save(path)

        loaded = quiver.IvfPqIndex.load(path)
        assert len(loaded) == 20
        results = loaded.search(query=[5.0, 0.0, 0.0, 0.0], k=10)
        ids = [r["id"] for r in results]
        assert 5 in ids


# ---------------------------------------------------------------------------
# WAL recovery via Client
# ---------------------------------------------------------------------------


class TestWalRecovery:
    def test_data_survives_reopen(self, tmp_path):
        db_path = str(tmp_path / "waltest")

        # First session: create and insert
        db = quiver.Client(path=db_path)
        col = db.create_collection("docs", dimensions=4, metric="l2")
        col.upsert(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={"k": "v"})
        col.upsert(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        del col
        del db

        # Second session: reopen and verify
        db2 = quiver.Client(path=db_path)
        col2 = db2.get_collection("docs")
        assert col2.count == 2

        hits = col2.search(query=[1.0, 0.0, 0.0, 0.0], k=1)
        assert hits[0]["id"] == 1
        assert hits[0]["payload"]["k"] == "v"

    def test_collections_survive_reopen(self, tmp_path):
        db_path = str(tmp_path / "waltest2")

        db = quiver.Client(path=db_path)
        db.create_collection("a", dimensions=4)
        db.create_collection("b", dimensions=8)
        del db

        db2 = quiver.Client(path=db_path)
        names = sorted(db2.list_collections())
        assert names == ["a", "b"]
