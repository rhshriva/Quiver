"""Functional tests for standalone index classes."""

import os
import pytest
import quiver_vector_db as quiver
from conftest import SAMPLE_DIM, random_vectors, random_vector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_index(cls, dim=4, metric="l2", **kwargs):
    """Create an index instance with sensible defaults."""
    return cls(dimensions=dim, metric=metric, **kwargs)


# ---------------------------------------------------------------------------
# FlatIndex
# ---------------------------------------------------------------------------


class TestFlatIndex:
    def test_construct(self):
        idx = quiver.FlatIndex(dimensions=128, metric="cosine")
        assert idx.dimensions == 128
        assert idx.metric == "cosine"
        assert len(idx) == 0

    def test_add_and_search(self):
        idx = quiver.FlatIndex(dimensions=4, metric="l2")
        idx.add(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        idx.add(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        assert len(idx) == 2

        results = idx.search(query=[1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0]["id"] == 1
        assert results[0]["distance"] == pytest.approx(0.0, abs=1e-5)

    def test_add_batch(self):
        idx = quiver.FlatIndex(dimensions=4, metric="l2")
        entries = [(i, [float(i), 0.0, 0.0, 0.0]) for i in range(10)]
        idx.add_batch(entries)
        assert len(idx) == 10

    def test_delete(self):
        idx = quiver.FlatIndex(dimensions=4, metric="l2")
        idx.add(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        idx.add(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        assert idx.delete(1) is True
        assert len(idx) == 1
        results = idx.search(query=[1.0, 0.0, 0.0, 0.0], k=5)
        assert all(r["id"] != 1 for r in results)

    def test_dimension_mismatch(self):
        idx = quiver.FlatIndex(dimensions=4, metric="l2")
        with pytest.raises(ValueError, match="dimension mismatch"):
            idx.add(id=1, vector=[1.0, 2.0])


# ---------------------------------------------------------------------------
# HnswIndex
# ---------------------------------------------------------------------------


class TestHnswIndex:
    def test_construct(self):
        idx = quiver.HnswIndex(dimensions=64, metric="cosine", ef_construction=100, ef_search=30, m=8)
        assert idx.dimensions == 64
        assert idx.metric == "cosine"
        assert len(idx) == 0

    def test_add_search_flush(self):
        idx = quiver.HnswIndex(dimensions=4, metric="l2")
        for i in range(20):
            idx.add(id=i, vector=[float(i), 0.0, 0.0, 0.0])
        idx.flush()
        assert len(idx) == 20

        results = idx.search(query=[5.0, 0.0, 0.0, 0.0], k=3)
        assert len(results) == 3
        assert results[0]["id"] == 5

    def test_add_batch_and_flush(self):
        idx = quiver.HnswIndex(dimensions=4, metric="l2")
        entries = [(i, [float(i), 0.0, 0.0, 0.0]) for i in range(20)]
        idx.add_batch(entries)
        idx.flush()
        results = idx.search(query=[10.0, 0.0, 0.0, 0.0], k=1)
        assert results[0]["id"] == 10

    def test_delete(self):
        idx = quiver.HnswIndex(dimensions=4, metric="l2")
        for i in range(10):
            idx.add(id=i, vector=[float(i), 0.0, 0.0, 0.0])
        idx.flush()
        assert idx.delete(5) is True
        assert len(idx) == 9


# ---------------------------------------------------------------------------
# QuantizedFlatIndex
# ---------------------------------------------------------------------------


class TestQuantizedFlatIndex:
    def test_construct(self):
        idx = quiver.QuantizedFlatIndex(dimensions=128, metric="cosine")
        assert idx.dimensions == 128
        assert len(idx) == 0

    def test_add_and_search(self):
        idx = quiver.QuantizedFlatIndex(dimensions=4, metric="l2")
        idx.add(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        idx.add(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        results = idx.search(query=[1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0]["id"] == 1

    def test_add_batch(self):
        idx = quiver.QuantizedFlatIndex(dimensions=4, metric="l2")
        entries = [(i, [float(i), 0.0, 0.0, 0.0]) for i in range(10)]
        idx.add_batch(entries)
        assert len(idx) == 10

    def test_delete(self):
        idx = quiver.QuantizedFlatIndex(dimensions=4, metric="l2")
        idx.add(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        assert idx.delete(1) is True
        assert len(idx) == 0


# ---------------------------------------------------------------------------
# Fp16FlatIndex
# ---------------------------------------------------------------------------


class TestFp16FlatIndex:
    def test_construct(self):
        idx = quiver.Fp16FlatIndex(dimensions=128, metric="cosine")
        assert idx.dimensions == 128
        assert len(idx) == 0

    def test_add_and_search(self):
        idx = quiver.Fp16FlatIndex(dimensions=4, metric="l2")
        idx.add(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        idx.add(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        results = idx.search(query=[1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0]["id"] == 1

    def test_add_batch(self):
        idx = quiver.Fp16FlatIndex(dimensions=4, metric="l2")
        entries = [(i, [float(i), 0.0, 0.0, 0.0]) for i in range(10)]
        idx.add_batch(entries)
        assert len(idx) == 10

    def test_delete(self):
        idx = quiver.Fp16FlatIndex(dimensions=4, metric="l2")
        idx.add(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        assert idx.delete(1) is True
        assert len(idx) == 0


# ---------------------------------------------------------------------------
# IvfIndex
# ---------------------------------------------------------------------------


class TestIvfIndex:
    def test_construct(self):
        idx = quiver.IvfIndex(dimensions=64, metric="l2", n_lists=4, nprobe=2, train_size=16)
        assert idx.dimensions == 64
        assert len(idx) == 0

    def test_auto_train_and_search(self):
        """IVF auto-trains after train_size inserts."""
        idx = quiver.IvfIndex(dimensions=4, metric="l2", n_lists=4, nprobe=4, train_size=16)
        for i in range(20):
            idx.add(id=i, vector=[float(i), 0.0, 0.0, 0.0])

        results = idx.search(query=[5.0, 0.0, 0.0, 0.0], k=3)
        assert len(results) >= 1
        ids = [r["id"] for r in results]
        assert 5 in ids

    def test_delete(self):
        idx = quiver.IvfIndex(dimensions=4, metric="l2", n_lists=4, nprobe=4, train_size=16)
        for i in range(20):
            idx.add(id=i, vector=[float(i), 0.0, 0.0, 0.0])
        assert idx.delete(5) is True
        assert len(idx) == 19


# ---------------------------------------------------------------------------
# IvfPqIndex
# ---------------------------------------------------------------------------


class TestIvfPqIndex:
    def test_construct(self):
        idx = quiver.IvfPqIndex(dimensions=8, metric="l2", n_lists=4, nprobe=4, train_size=16, pq_m=2, pq_k_sub=4)
        assert idx.dimensions == 8
        assert len(idx) == 0

    def test_auto_train_and_search(self):
        idx = quiver.IvfPqIndex(dimensions=4, metric="l2", n_lists=4, nprobe=4, train_size=16, pq_m=2, pq_k_sub=4)
        for i in range(20):
            idx.add(id=i, vector=[float(i), 0.0, 0.0, 0.0])

        results = idx.search(query=[5.0, 0.0, 0.0, 0.0], k=10)
        assert len(results) >= 1
        ids = [r["id"] for r in results]
        assert 5 in ids

    def test_delete(self):
        idx = quiver.IvfPqIndex(dimensions=4, metric="l2", n_lists=4, nprobe=4, train_size=16, pq_m=2, pq_k_sub=4)
        for i in range(20):
            idx.add(id=i, vector=[float(i), 0.0, 0.0, 0.0])
        assert idx.delete(5) is True
        assert len(idx) == 19


# ---------------------------------------------------------------------------
# MmapFlatIndex
# ---------------------------------------------------------------------------


class TestMmapFlatIndex:
    def test_construct(self, tmp_path):
        path = str(tmp_path / "vectors.qvec")
        idx = quiver.MmapFlatIndex(dimensions=128, metric="cosine", path=path)
        assert idx.dimensions == 128
        assert len(idx) == 0

    def test_add_and_search(self, tmp_path):
        path = str(tmp_path / "vectors.qvec")
        idx = quiver.MmapFlatIndex(dimensions=4, metric="l2", path=path)
        idx.add(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        idx.add(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        idx.flush()

        results = idx.search(query=[1.0, 0.0, 0.0, 0.0], k=1)
        assert results[0]["id"] == 1

    def test_file_created(self, tmp_path):
        path = str(tmp_path / "test.qvec")
        idx = quiver.MmapFlatIndex(dimensions=4, metric="l2", path=path)
        idx.add(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        idx.flush()
        assert os.path.exists(path)

    def test_delete(self, tmp_path):
        path = str(tmp_path / "vectors.qvec")
        idx = quiver.MmapFlatIndex(dimensions=4, metric="l2", path=path)
        idx.add(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        idx.add(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        idx.flush()
        assert idx.delete(1) is True
        assert len(idx) == 1


# ---------------------------------------------------------------------------
# Cross-index: all metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    @pytest.mark.parametrize("metric", ["cosine", "l2", "dot_product"])
    def test_flat_with_all_metrics(self, metric):
        idx = quiver.FlatIndex(dimensions=4, metric=metric)
        idx.add(id=1, vector=[1.0, 0.0, 0.0, 0.0])
        idx.add(id=2, vector=[0.0, 1.0, 0.0, 0.0])
        results = idx.search(query=[1.0, 0.0, 0.0, 0.0], k=2)
        assert len(results) == 2
        assert results[0]["id"] == 1  # self-match should be closest

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="unknown metric"):
            quiver.FlatIndex(dimensions=4, metric="hamming")
