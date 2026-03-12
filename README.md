# Quiver

An embedded vector database written in Rust with a Python SDK.
No server, no network — runs fully in-process.

## Installation

```bash
pip install quiver-vector-db
```

Or build from source:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release -m crates/quiver-python/Cargo.toml
```

## Quick start

```python
import quiver_vector_db as quiver

db  = quiver.Client(path="./my_data")
col = db.create_collection("docs", dimensions=384, metric="cosine")

col.upsert(id=1, vector=[0.12, 0.45, ...], payload={"title": "Hello world"})
col.upsert(id=2, vector=[0.98, 0.01, ...], payload={"title": "Vector search"})

hits = col.search(query=[0.13, 0.44, ...], k=5)
for hit in hits:
    print(hit["id"], hit["distance"], hit["payload"])
```

Collections are persisted via WAL — reopen the same path and everything is restored.

## Index types

Seven index types, all usable from Python:

```python
db = quiver.Client(path="./data")

col = db.create_collection("name", dimensions=768, metric="cosine", index_type="hnsw")          # default
col = db.create_collection("name", dimensions=768, metric="cosine", index_type="flat")           # exact
col = db.create_collection("name", dimensions=768, metric="cosine", index_type="quantized_flat") # int8, ~4x less RAM
col = db.create_collection("name", dimensions=768, metric="cosine", index_type="fp16_flat")      # float16, 2x less RAM
col = db.create_collection("name", dimensions=768, metric="l2",     index_type="ivf")            # cluster-based ANN
col = db.create_collection("name", dimensions=768, metric="l2",     index_type="ivf_pq")         # PQ compressed, ~96x less RAM
col = db.create_collection("name", dimensions=768, metric="cosine", index_type="mmap_flat")      # disk-mapped, near-zero RAM
```

| Index | Recall | RAM | Best for |
|-------|--------|-----|----------|
| `hnsw` | 95-99% | Vectors + graph | General purpose (default) |
| `flat` | 100% | All vectors (f32) | Small datasets, exact required |
| `quantized_flat` | ~99% | ~4x less (int8) | Memory-constrained exact search |
| `fp16_flat` | >99.5% | ~2x less (float16) | Balanced memory vs accuracy |
| `ivf` | Tunable | Vectors + centroids | Large datasets |
| `ivf_pq` | ~90%+ | ~96x less (PQ codes) | Million-scale, extreme compression |
| `mmap_flat` | 100% | Near-zero RSS | Dataset larger than RAM |

## In-memory indexes

Low-level index objects live in RAM only. Nothing hits disk unless you call `.save()`.

```python
import quiver_vector_db as quiver

# Exact brute-force
idx = quiver.FlatIndex(dimensions=384, metric="cosine")
idx.add(id=1, vector=[...])
idx.add_batch([(2, [...]), (3, [...])])
results = idx.search(query=[...], k=10)
idx.save("index.bin")
loaded = quiver.FlatIndex.load("index.bin")

# HNSW approximate
hnsw = quiver.HnswIndex(dimensions=384, metric="cosine", ef_construction=200, ef_search=50, m=12)
hnsw.add(id=1, vector=[...])
hnsw.flush()  # build graph after bulk inserts
results = hnsw.search(query=[...], k=10)

# Int8 quantized          — same API as FlatIndex
idx = quiver.QuantizedFlatIndex(dimensions=384, metric="cosine")

# Float16 quantized       — same API as FlatIndex
idx = quiver.Fp16FlatIndex(dimensions=384, metric="cosine")

# IVF cluster-based
idx = quiver.IvfIndex(dimensions=384, metric="l2", n_lists=256, nprobe=16, train_size=4096)

# IVF + Product Quantization
idx = quiver.IvfPqIndex(dimensions=384, metric="l2", n_lists=256, nprobe=16, train_size=4096, pq_m=8, pq_k_sub=256)

# Memory-mapped flat
idx = quiver.MmapFlatIndex(dimensions=384, metric="cosine", path="./vectors.qvec")
```

## Payload & filtered search

Attach metadata to vectors and filter at query time:

```python
col.upsert(id=1, vector=[...], payload={"category": "tech", "score": 4.8})

# Filter operators: $eq, $ne, $in, $gt, $gte, $lt, $lte, $and, $or
hits = col.search(query=[...], k=5, filter={"category": {"$eq": "tech"}})
hits = col.search(query=[...], k=5, filter={"score": {"$gte": 4.0}})
hits = col.search(query=[...], k=5, filter={
    "$and": [
        {"category": {"$in": ["tech", "science"]}},
        {"score": {"$gte": 4.0}},
    ]
})
```

## Hybrid dense+sparse search

Combine dense vector similarity with sparse keyword signals (e.g. BM25/SPLADE weights):

```python
# Upsert with both dense and sparse vectors
col.upsert_hybrid(
    id=1, vector=[...],
    sparse_vector={42: 0.8, 100: 0.5, 3001: 0.3},
    payload={"title": "Rust guide"},
)

# Hybrid search — weighted fusion of dense and sparse scores
hits = col.search_hybrid(
    dense_query=[...],
    sparse_query={42: 0.7, 100: 0.6},
    k=10,
    dense_weight=0.7,
    sparse_weight=0.3,
    filter={"category": {"$eq": "tech"}},  # optional
)

for hit in hits:
    print(hit["id"], hit["score"], hit["dense_distance"], hit["sparse_score"])
```

Regular `upsert()` and `upsert_hybrid()` can be mixed freely in the same collection.

## Collection management

```python
db = quiver.Client(path="./data")

col = db.get_or_create_collection("docs", dimensions=768, metric="cosine")
col = db.get_collection("docs")

db.list_collections()       # ['docs', ...]
col.count                   # number of dense vectors
col.sparse_count            # number of sparse vectors

col.delete(id=42)
db.delete_collection("docs")
```

## Distance metrics

| Metric | String | Use when |
|--------|--------|----------|
| Cosine | `"cosine"` | Text/image embeddings (most common) |
| L2 | `"l2"` | Geometry, sensor data |
| Dot product | `"dot_product"` | Pre-normalised vectors |

## Parameter tuning

**HNSW:** `ef_construction` (build quality, default 200), `ef_search` (query quality, default 50), `m` (graph connectivity, default 12).

**IVF / IVF-PQ:** `n_lists` (clusters, default 256, rule of thumb: `sqrt(N)`), `nprobe` (clusters scanned, default 16), `train_size` (auto-train threshold, default 4096).

**PQ-specific:** `pq_m` (sub-quantizers, must divide dimensions), `pq_k_sub` (centroids per sub-quantizer, default 256). Memory per vector = `pq_m` bytes.

## Build

```bash
./dev_build.sh               # build + test Rust core
./dev_build.sh --python       # also build Python wheel
./dev_build.sh --faiss --python  # with FAISS support
```

## License

MIT
