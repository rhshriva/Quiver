"""Performance benchmarks for Quiver.

Run with: pytest tests/test_perf.py -v -s
"""

import os
import time
import random
import statistics
import pytest
import quiver_vector_db as quiver


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DIM = 128
N_SMALL = 10_000
N_LARGE = 100_000
N_SEARCH = 1_000
K = 10
SEED = 42


def gen_vectors(n, dim=DIM, seed=SEED):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]


def fmt_table(headers, rows):
    """Print a formatted ASCII table."""
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        print("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |")
    print(sep)


# ---------------------------------------------------------------------------
# Insert throughput
# ---------------------------------------------------------------------------


class TestInsertThroughput:
    """Benchmark insert throughput for each index type."""

    INDEXES = {
        "FlatIndex": lambda: quiver.FlatIndex(dimensions=DIM, metric="l2"),
        "HnswIndex": lambda: quiver.HnswIndex(dimensions=DIM, metric="l2"),
        "QuantizedFlat": lambda: quiver.QuantizedFlatIndex(dimensions=DIM, metric="l2"),
        "Fp16Flat": lambda: quiver.Fp16FlatIndex(dimensions=DIM, metric="l2"),
    }

    def test_insert_10k(self):
        vectors = gen_vectors(N_SMALL)
        rows = []
        for name, factory in self.INDEXES.items():
            idx = factory()
            t0 = time.perf_counter()
            for i, vec in enumerate(vectors):
                idx.add(id=i, vector=vec)
            elapsed = time.perf_counter() - t0
            rate = N_SMALL / elapsed
            rows.append((name, N_SMALL, f"{elapsed:.3f}s", f"{rate:,.0f} vec/s"))

        print("\n\n=== Insert Throughput (10K vectors) ===")
        fmt_table(["Index", "Vectors", "Time", "Throughput"], rows)

    def test_batch_vs_single(self):
        vectors = gen_vectors(N_SMALL)
        entries = [(i, v) for i, v in enumerate(vectors)]

        # Single insert
        idx1 = quiver.FlatIndex(dimensions=DIM, metric="l2")
        t0 = time.perf_counter()
        for i, vec in enumerate(vectors):
            idx1.add(id=i, vector=vec)
        single_time = time.perf_counter() - t0

        # Batch insert
        idx2 = quiver.FlatIndex(dimensions=DIM, metric="l2")
        t0 = time.perf_counter()
        idx2.add_batch(entries)
        batch_time = time.perf_counter() - t0

        print(f"\n\n=== Batch vs Single Insert ({N_SMALL} vectors, FlatIndex) ===")
        fmt_table(
            ["Method", "Time", "Throughput", "Speedup"],
            [
                ("Single add()", f"{single_time:.3f}s", f"{N_SMALL/single_time:,.0f} vec/s", "1.0x"),
                ("add_batch()", f"{batch_time:.3f}s", f"{N_SMALL/batch_time:,.0f} vec/s", f"{single_time/batch_time:.1f}x"),
            ],
        )


# ---------------------------------------------------------------------------
# Search latency
# ---------------------------------------------------------------------------


class TestSearchLatency:
    """Benchmark search latency across index types."""

    def _build_index(self, name, vectors):
        entries = [(i, v) for i, v in enumerate(vectors)]
        if name == "FlatIndex":
            idx = quiver.FlatIndex(dimensions=DIM, metric="l2")
            idx.add_batch(entries)
        elif name == "HnswIndex":
            idx = quiver.HnswIndex(dimensions=DIM, metric="l2")
            idx.add_batch(entries)
            idx.flush()
        elif name == "QuantizedFlat":
            idx = quiver.QuantizedFlatIndex(dimensions=DIM, metric="l2")
            idx.add_batch(entries)
        elif name == "Fp16Flat":
            idx = quiver.Fp16FlatIndex(dimensions=DIM, metric="l2")
            idx.add_batch(entries)
        elif name == "IvfIndex":
            idx = quiver.IvfIndex(dimensions=DIM, metric="l2", n_lists=32, nprobe=8, train_size=len(vectors))
            idx.add_batch(entries)
        elif name == "IvfPqIndex":
            idx = quiver.IvfPqIndex(
                dimensions=DIM, metric="l2", n_lists=32, nprobe=8,
                train_size=len(vectors), pq_m=8, pq_k_sub=16,
            )
            idx.add_batch(entries)
        else:
            raise ValueError(f"Unknown index: {name}")
        return idx

    def test_search_latency_comparison(self):
        vectors = gen_vectors(N_SMALL)
        queries = gen_vectors(N_SEARCH, seed=999)

        index_names = ["FlatIndex", "HnswIndex", "QuantizedFlat", "Fp16Flat", "IvfIndex", "IvfPqIndex"]
        rows = []

        for name in index_names:
            idx = self._build_index(name, vectors)
            latencies = []
            for q in queries:
                t0 = time.perf_counter()
                idx.search(query=q, k=K)
                latencies.append((time.perf_counter() - t0) * 1000)  # ms

            avg = statistics.mean(latencies)
            p50 = statistics.median(latencies)
            p99 = sorted(latencies)[int(0.99 * len(latencies))]
            rows.append((name, f"{avg:.3f}", f"{p50:.3f}", f"{p99:.3f}"))

        print(f"\n\n=== Search Latency ({N_SMALL} vectors, {N_SEARCH} queries, k={K}) ===")
        fmt_table(["Index", "Avg (ms)", "p50 (ms)", "p99 (ms)"], rows)


# ---------------------------------------------------------------------------
# Recall measurement
# ---------------------------------------------------------------------------


class TestRecall:
    """Compare ANN recall vs exact FlatIndex ground truth."""

    def test_recall_at_10(self):
        vectors = gen_vectors(N_SMALL)
        queries = gen_vectors(100, seed=999)
        entries = [(i, v) for i, v in enumerate(vectors)]

        # Ground truth from FlatIndex
        flat = quiver.FlatIndex(dimensions=DIM, metric="l2")
        flat.add_batch(entries)
        ground_truth = {}
        for qi, q in enumerate(queries):
            results = flat.search(query=q, k=K)
            ground_truth[qi] = {r["id"] for r in results}

        # Test each ANN index
        ann_configs = {
            "HnswIndex": lambda: self._build("hnsw", vectors),
            "QuantizedFlat": lambda: self._build("quantized", vectors),
            "Fp16Flat": lambda: self._build("fp16", vectors),
            "IvfIndex": lambda: self._build("ivf", vectors),
            "IvfPqIndex": lambda: self._build("ivfpq", vectors),
        }

        rows = []
        for name, builder in ann_configs.items():
            idx = builder()
            recalls = []
            for qi, q in enumerate(queries):
                results = idx.search(query=q, k=K)
                found = {r["id"] for r in results}
                recall = len(found & ground_truth[qi]) / K
                recalls.append(recall)
            avg_recall = statistics.mean(recalls)
            min_recall = min(recalls)
            rows.append((name, f"{avg_recall:.4f}", f"{min_recall:.4f}"))

        print(f"\n\n=== Recall@{K} vs FlatIndex ({N_SMALL} vectors, 100 queries) ===")
        fmt_table(["Index", "Avg Recall", "Min Recall"], rows)

    def _build(self, kind, vectors):
        entries = [(i, v) for i, v in enumerate(vectors)]
        if kind == "hnsw":
            idx = quiver.HnswIndex(dimensions=DIM, metric="l2")
            idx.add_batch(entries)
            idx.flush()
        elif kind == "quantized":
            idx = quiver.QuantizedFlatIndex(dimensions=DIM, metric="l2")
            idx.add_batch(entries)
        elif kind == "fp16":
            idx = quiver.Fp16FlatIndex(dimensions=DIM, metric="l2")
            idx.add_batch(entries)
        elif kind == "ivf":
            idx = quiver.IvfIndex(dimensions=DIM, metric="l2", n_lists=32, nprobe=8, train_size=len(vectors))
            idx.add_batch(entries)
        elif kind == "ivfpq":
            idx = quiver.IvfPqIndex(
                dimensions=DIM, metric="l2", n_lists=32, nprobe=8,
                train_size=len(vectors), pq_m=8, pq_k_sub=16,
            )
            idx.add_batch(entries)
        return idx


# ---------------------------------------------------------------------------
# Persistence overhead
# ---------------------------------------------------------------------------


class TestPersistenceOverhead:
    """Benchmark save/load time for each index type."""

    def test_save_load_timing(self, tmp_path):
        vectors = gen_vectors(N_SMALL)
        entries = [(i, v) for i, v in enumerate(vectors)]

        configs = {
            "FlatIndex": lambda: self._mk(quiver.FlatIndex(dimensions=DIM, metric="l2"), entries),
            "HnswIndex": lambda: self._mk_hnsw(entries),
            "QuantizedFlat": lambda: self._mk(quiver.QuantizedFlatIndex(dimensions=DIM, metric="l2"), entries),
            "Fp16Flat": lambda: self._mk(quiver.Fp16FlatIndex(dimensions=DIM, metric="l2"), entries),
            "IvfIndex": lambda: self._mk(quiver.IvfIndex(dimensions=DIM, metric="l2", n_lists=32, nprobe=8, train_size=N_SMALL), entries),
            "IvfPqIndex": lambda: self._mk(quiver.IvfPqIndex(dimensions=DIM, metric="l2", n_lists=32, nprobe=8, train_size=N_SMALL, pq_m=8, pq_k_sub=16), entries),
        }

        loaders = {
            "FlatIndex": quiver.FlatIndex.load,
            "HnswIndex": quiver.HnswIndex.load,
            "QuantizedFlat": quiver.QuantizedFlatIndex.load,
            "Fp16Flat": quiver.Fp16FlatIndex.load,
            "IvfIndex": quiver.IvfIndex.load,
            "IvfPqIndex": quiver.IvfPqIndex.load,
        }

        rows = []
        for name in configs:
            idx = configs[name]()
            path = str(tmp_path / f"{name}.bin")

            t0 = time.perf_counter()
            idx.save(path)
            save_time = time.perf_counter() - t0

            size_mb = os.path.getsize(path) / (1024 * 1024)

            t0 = time.perf_counter()
            loaders[name](path)
            load_time = time.perf_counter() - t0

            rows.append((name, f"{save_time:.3f}s", f"{load_time:.3f}s", f"{size_mb:.1f} MB"))

        print(f"\n\n=== Save/Load Overhead ({N_SMALL} vectors, {DIM}d) ===")
        fmt_table(["Index", "Save Time", "Load Time", "File Size"], rows)

    def _mk(self, idx, entries):
        idx.add_batch(entries)
        return idx

    def _mk_hnsw(self, entries):
        idx = quiver.HnswIndex(dimensions=DIM, metric="l2")
        idx.add_batch(entries)
        idx.flush()
        return idx
