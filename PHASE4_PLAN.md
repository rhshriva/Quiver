# Phase 4 & 5 Plan: Query Power + Developer Experience

Scope: **embedded only**. All features are delivered via the Rust library
(`quiver-core`) and the Python PyO3 bindings (`quiver-python`). No HTTP
server, no REST API, no CLI changes.

---

## Phase 4 — Query Power (weeks 7–8)

---

### 4.0 — Python Embedded Client Gap Closure

Close the 5 gaps between Quiver's current Python SDK and the Qdrant/FAISS
standard. Changes are confined to `crates/quiver-python/` with one small
addition to `crates/quiver-core/src/collection.rs`.

| Gap | Current state | Target state |
|-----|--------------|--------------|
| Index types | only `"flat"` / `"hnsw"` | all 5 types + per-type config kwargs |
| NumPy support | `list[float]` only | `list` **or** `numpy.ndarray` |
| Typed filter API | raw `dict` only | `FieldCondition` + `Filter` classes |
| Batch upsert | not exposed | `upsert_batch` on `PyCollection` |
| Scroll / metadata | missing | `iter_vectors()`, `meta` property, `__iter__` |

#### 4.0.1 — Expose All Index Types in `create_collection`

`PyClient::create_collection` currently hard-codes only `"flat"` and `"hnsw"`.
New signature (all kwargs optional, backwards-compatible):

```python
client.create_collection(
    name, dimensions, metric="cosine", index_type="hnsw",
    ef_construction=200, ef_search=50, m=12,   # HNSW
    n_lists=256, nprobe=16, train_size=4096,    # IVF
    faiss_factory="Flat",                       # FAISS (feature-gated)
)
```

Mapping:

```
"flat"           → IndexType::Flat
"hnsw"           → IndexType::Hnsw,          HnswConfig { ef_construction, ef_search, m }
"quantized_flat" → IndexType::QuantizedFlat
"ivf"            → IndexType::Ivf,            IvfConfig { n_lists, nprobe, train_size, max_iter: 25 }
"mmap_flat"      → IndexType::MmapFlat
"faiss"          → IndexType::Faiss,          faiss_factory string (feature-gated)
```

Files: `crates/quiver-python/src/lib.rs`

#### 4.0.2 — NumPy Array Support

Add `numpy = "0.22"` to `crates/quiver-python/Cargo.toml`.
Replace `Vec<f32>` parameters with `PyArrayLike1<f32>` from the `numpy` crate.
`PyArrayLike1<f32>` accepts both `list[float]` and `numpy.ndarray` with zero
copy when the array is already a contiguous float32 buffer.

Affected: `PyFlatIndex::add/search`, `PyHnswIndex::add/search`,
`PyCollection::upsert/search`.

Files: `crates/quiver-python/Cargo.toml`, `crates/quiver-python/src/lib.rs`

#### 4.0.3 — Typed Filter Classes

Add two Python classes that wrap the existing `FilterCondition` Rust enum:

```python
from quiver import FieldCondition, Filter

col.search(query, k=5, filter=FieldCondition("tag", eq="news"))
col.search(query, k=5, filter=Filter(
    must=[FieldCondition("tag", eq="news"), FieldCondition("score", gte=0.5)]
))
col.search(query, k=5, filter=Filter(
    should=[FieldCondition("tag", eq="news"), FieldCondition("tag", eq="sports")]
))

# Old dict API still works
col.search(query, k=5, filter={"tag": {"$eq": "news"}})
```

**`PyFieldCondition`** — wraps `FilterCondition::Field`. Positional `field`
arg plus exactly one of `eq`, `ne`, `in_`, `gt`, `gte`, `lt`, `lte`.

**`PyFilter`** — wraps `FilterCondition::And` / `::Or`. Accepts `must` (→ And)
and/or `should` (→ Or), each a list of `PyFieldCondition` or nested `PyFilter`.

Update `py_to_filter` to detect both dict and the new class types.

Files: `crates/quiver-python/src/lib.rs`

#### 4.0.4 — Batch Upsert

```python
col.upsert_batch([
    (1, vec_1, {"text": "hello"}),
    (2, vec_2, {"text": "world"}),
    (3, vec_3, None),
])
```

Add `upsert_batch` to `Collection` in `quiver-core` (iterates `upsert`, single
method to reduce FFI round-trips and allow future WAL-sync optimisation).
Add `PyCollection::upsert_batch` that converts entries (NumPy-aware) and
delegates to `col.upsert_batch`.

Files: `crates/quiver-core/src/collection.rs`,
`crates/quiver-python/src/lib.rs`

#### 4.0.5 — Scroll / Iterator + Collection Metadata

```python
for id, vector in col.iter_vectors():   # materialises as list[(int, list[float])]
    ...

for id, vector in col:                  # __iter__ shorthand
    ...

col.meta  # → {"name":..., "dimensions":..., "metric":..., "index_type":..., "count":...}
```

Files: `crates/quiver-python/src/lib.rs`

---

### 4.1 — Filter-Aware ANN (Pre-filter HNSW)

**Problem:** Post-filter degrades when <10% of vectors match — requires 10–100×
overscan to return k valid results.

**Target:** Correct recall at any selectivity level with no API change.

#### Design

Add `search_filtered(query, k, allow: &HashSet<u64>)` to `HnswIndex`. During
graph traversal, candidate nodes are only enqueued when their ID is in `allow`
(pre-filtering — the graph explores only the matching subspace).

Selectivity logic in `Collection::search`:

1. Count matching payload IDs: `matches = payloads.iter().filter(matches_filter).count()`
2. Selectivity = `matches / total`
3. If selectivity ≥ 10%: existing post-filter (cheap)
4. If selectivity < 10%: build `allow` set from matching IDs, call
   `index.search_filtered(query, k, &allow)`

Add `search_filtered` to the `VectorIndex` trait with a default impl
(post-filter) so `FlatIndex`, `IvfIndex`, etc. inherit it without changes.

#### Files touched

`crates/quiver-core/src/index/mod.rs`,
`crates/quiver-core/src/index/hnsw.rs`,
`crates/quiver-core/src/collection.rs`

---

### 4.2 — Sparse Vectors + Hybrid Search

**Target:** BM25 sparse retrieval combined with dense ANN via RRF, enabling
full RAG pipelines in a single embedded call.

#### New types (`crates/quiver-core/src/sparse.rs` — new file)

```rust
pub struct SparseVector {
    pub indices: Vec<u32>,  // sorted, unique token IDs
    pub values:  Vec<f32>,  // TF-IDF weights, parallel to indices
}
```

#### BM25 posting-list index (`crates/quiver-core/src/index/sparse_bm25.rs` — new file)

```rust
pub struct SparseBm25Index {
    posting_lists: HashMap<u32, Vec<(u64, f32)>>,  // term → [(doc_id, weight)]
    doc_lengths:   HashMap<u64, u32>,
    avg_doc_len:   f32,
    k1: f32,  // 1.2
    b:  f32,  // 0.75
}
```

`add(id, sparse_vec)`, `search(query_sparse, k) → Vec<SearchResult>`.
Only posting lists for query terms are touched — sub-linear.

#### Hybrid search (`crates/quiver-core/src/collection.rs`)

```rust
pub fn hybrid_search(
    &self,
    dense_query:  &[f32],
    sparse_query: &SparseVector,
    k:            usize,
    alpha:        f32,   // 0.0 = sparse only, 1.0 = dense only
) -> Result<Vec<CollectionSearchResult>, VectorDbError>
```

Score fusion via **Reciprocal Rank Fusion (RRF)**:

```
score(id) = alpha * 1/(rank_dense(id) + 60) + (1-alpha) * 1/(rank_sparse(id) + 60)
```

Add `sparse_index: bool` to `CollectionMeta` (default `false`). When `true`,
`Collection` maintains a `SparseBm25Index` alongside the dense index.

**Python binding additions** in `PyCollection`:
- `upsert_sparse(id, sparse_indices, sparse_values, payload=None)`
- `hybrid_search(dense_query, sparse_indices, sparse_values, k, alpha=0.5)`

#### Files touched

`crates/quiver-core/src/sparse.rs` (new),
`crates/quiver-core/src/index/sparse_bm25.rs` (new),
`crates/quiver-core/src/collection.rs`,
`crates/quiver-python/src/lib.rs`

---

### 4.3 — Multi-Vector Per Document

**Target:** ColBERT-style passage-level embeddings — one logical `doc_id`
stores multiple vectors; `search()` deduplicates by `doc_id` and returns the
best-matching vector per document.

#### Design

Add a surrogate ID layer inside `Collection`:

```rust
// New fields in Collection
multi_vecs:       HashMap<u64, Vec<u64>>,  // doc_id → [surrogate_ids]
surrogate_to_doc: HashMap<u64, u64>,       // surrogate_id → doc_id
next_surrogate:   u64,
```

New Rust API:

```rust
pub fn upsert_multi(
    &mut self,
    doc_id:  u64,
    vectors: Vec<Vec<f32>>,
    payload: Option<serde_json::Value>,
) -> Result<(), VectorDbError>
```

Each vector is inserted under an auto-assigned surrogate ID. After ANN search,
a dedup pass keeps the closest surrogate per `doc_id`:

```rust
let mut best: HashMap<u64, CollectionSearchResult> = HashMap::new();
for r in raw_results {
    let doc_id = surrogate_to_doc.get(&r.id).copied().unwrap_or(r.id);
    best.entry(doc_id)
        .and_modify(|e| { if r.distance < e.distance { e.distance = r.distance; } })
        .or_insert(CollectionSearchResult { id: doc_id, distance: r.distance, payload: ... });
}
```

`meta.json` gains `multi_vec: bool`. WAL adds a `UpsertMulti` entry variant.

**Python binding** on `PyCollection`:
- `upsert_multi(doc_id, vectors, payload=None)` — vectors is `list[list[float]]`
  or a 2D `numpy.ndarray`

#### Files touched

`crates/quiver-core/src/collection.rs`,
`crates/quiver-core/src/wal.rs`,
`crates/quiver-python/src/lib.rs`

---

## Phase 5 — Developer Experience (week 9)

---

### 5.1 — Idiomatic Python API

**Target:** `pip install quiver-db` → fully embedded, no server, matches
Chroma's simplicity.

#### Design

A pure-Python ergonomics layer (`crates/quiver-python/python/quiver/__init__.py`)
that wraps the PyO3 types:

```python
import quiver

client = quiver.Client("./data", embedder="openai/text-embedding-3-small")

col = client.collection("docs")     # get_or_create, hnsw cosine by default

id = col.add(vector=[...], payload={"text": "hello"})    # auto-ID (uuid4 int)
id = col.add(id=42, vector=[...])
id = col.add_text("Hello world")    # embed + auto-ID, requires embedder

results = col.search(vector=[...], k=5)
results = col.search_text("query", k=5)
results = col.search(vector=[...], k=5, filter=FieldCondition("tag", eq="news"))
```

Auto-ID generates the top-64-bit integer of a `uuid4()`. `add` / `add_text`
return the assigned ID. The embedder can be any callable `(str) -> list[float]`
or a string key like `"openai/text-embedding-3-small"` that maps to
`quiver-embeddings` providers.

#### Files

`crates/quiver-python/python/quiver/__init__.py` (new pure-Python wrapper),
`crates/quiver-python/pyproject.toml` (maturin packaging for `quiver-db` on PyPI)

---

### 5.2 — Rust Embedded API Polish

**Target:** The native Rust API should be as ergonomic as the Python one for
Rust services using Quiver as a library.

#### Design

Extend `crates/quiver-core/src/db.rs` (the `Quiver` struct already exists):

```rust
let mut db = Quiver::open("./data")?;

// get_or_create shorthand
let col = db.collection("docs")?;

// auto-ID insert
let id: u64 = col.add_auto(&vector, Some(json!({"text": "hello"})))?;

// bulk load
col.upsert_batch(entries)?;

// flush WAL for all collections at once
db.compact_all()?;
```

Specifically:
- `Quiver::collection(name)` — get-or-create with default HNSW cosine config
- `CollectionHandle::add_auto(vector, payload) -> u64` — inserts with a
  generated surrogate ID, returns it
- `Quiver::compact_all()` — calls `Collection::compact()` on every loaded
  collection

#### Files

`crates/quiver-core/src/db.rs`

---

### 5.3 — Benchmarks + Recall Tests

**Target:** Published Recall@10/QPS results in README; CI regression guard.

#### Benchmark harness (`crates/quiver-bench/` — new crate, Criterion-based)

- **Datasets:** SIFT-1M (128-dim L2), GIST-1M (960-dim L2), Deep-1B 10M subset
  (96-dim cosine). Downloaded by script, cached in `~/.cache/quiver-bench/`.
- **Metrics per index type:** Recall@10, QPS (single-threaded), peak RSS.
- Results written to `benchmarks/results.json`; table auto-generated in README.

#### Recall regression tests (`crates/quiver-core/tests/recall.rs` — new file)

```rust
// HNSW on SIFT-100K:  recall@10 >= 0.95
// IVF nprobe=32:       recall@10 >= 0.90
// Pre-filter HNSW at 5% selectivity: recall@10 >= 0.90
```

Ground truth built from `FlatIndex` (brute force). Runs in CI on every PR.

#### Files

`crates/quiver-bench/` (new crate),
`crates/quiver-core/tests/recall.rs` (new integration test),
`benchmarks/results.json` (generated artifact),
`.github/workflows/bench.yml` (new CI job)

---

## Delivery Order

```
Phase 4:
  4.0.1  expose all index types in Python  ← pure PyO3 wiring, no core changes
  4.0.2  NumPy support                     ← add numpy dep + signature changes
  4.0.3  typed filter classes              ← new PyO3 classes, backwards-compatible
  4.0.4  batch upsert                      ← quiver-core first, then Python binding
  4.0.5  scroll / metadata                 ← additive only, no breaking changes
  4.1    pre-filter HNSW                   ← core change; 4.0 complete first
  4.2    sparse + hybrid search            ← new index + new core API + Python binding
  4.3    multi-vector per document         ← new WAL variant + collection internals

Phase 5 (can begin in parallel with 4.2/4.3):
  5.1    idiomatic Python API              ← pure Python wrapper, no Rust changes
  5.2    Rust embedded API polish          ← db.rs additions only
  5.3    benchmarks + recall CI            ← independent of all above
```

## Tests per Step

| Step | Test |
|------|------|
| 4.0.1 | create each index type, insert 10 vectors, search, verify top-1 |
| 4.0.2 | `np.array(...)` input to `add`/`search`/`upsert` gives identical results to list |
| 4.0.3 | `FieldCondition`/`Filter` objects produce same results as equivalent dict filter |
| 4.0.4 | `upsert_batch(N)` matches N sequential `upsert` calls |
| 4.0.5 | `iter_vectors()` yields exactly inserted IDs; `meta` dict fields correct |
| 4.1 | 5%-selectivity filter on HNSW: recall@10 ≥ 0.90 (vs brute force ground truth) |
| 4.2 | `hybrid_search(α=0.5)` differs from dense-only and sparse-only results |
| 4.3 | `upsert_multi(doc_id=1, [v1,v2,v3])` → search returns exactly one result with `id=1` |
| 5.1 | `col.add_text("hello")` returns ID; `search_text("hello", k=1)` returns that ID |
| 5.2 | `Quiver::collection("docs")` round-trip; `compact_all()` reduces WAL entry count |
| 5.3 | CI: HNSW recall@10 ≥ 0.95 on SIFT-100K; IVF recall@10 ≥ 0.90 |
