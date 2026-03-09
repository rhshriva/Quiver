# vectordb

A high-performance vector database written in Rust — available as an **embedded Python library** or a **standalone HTTP server**.

---

## Table of Contents

- [Overview](#overview)
- [Two Ways to Use vectordb](#two-ways-to-use-vectordb)
- [Offering 1: Embedded Python Library](#offering-1-embedded-python-library)
- [Offering 2: HTTP Server](#offering-2-http-server)
- [Payload Metadata & Filtered Search](#payload-metadata--filtered-search)
- [Persistence & Durability](#persistence--durability)
- [REST API Reference](#rest-api-reference)
- [CLI](#cli)
- [Index Types](#index-types)
- [Distance Metrics](#distance-metrics)
- [Low-Level Python Bindings](#low-level-python-bindings)
- [Architecture](#architecture)
- [Development Setup](#development-setup)
- [Building on macOS](#building-on-macos)

---

## Overview

**vectordb** stores, indexes, and searches high-dimensional vectors at scale. It supports:

- **Exact search** via `FlatIndex` (brute-force, 100% recall)
- **Approximate search** via `HnswIndex` (graph-based ANN, ~95–99% recall, sub-linear time)
- **Payload metadata** attached to every vector — filter search results by arbitrary JSON fields
- **WAL-based persistence** — all writes survive process restarts automatically
- **Two deployment modes** — embedded Python library or standalone HTTP server

---

## Two Ways to Use vectordb

### Which should I choose?

| | Embedded Python | HTTP Server |
|---|---|---|
| **Setup** | `pip install` + one line of code | Start a binary |
| **Process** | Runs in your Python process | Separate service |
| **Use case** | Scripts, notebooks, ML pipelines | Multi-client, microservices |
| **Networking** | None (in-process) | HTTP/JSON |
| **Language** | Python | Any language |
| **Persistence** | Yes (WAL on disk) | Yes (WAL on disk) |

---

## Offering 1: Embedded Python Library

The `vectordb.Client` API runs the database entirely inside your Python process — no server to start.

### Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release   # build and install into the active venv
```

### Quickstart

```python
import vectordb

# Open (or create) a database at the given path.
# All data is persisted to disk automatically.
db = vectordb.Client(path="./mydata")

# Create a collection (or retrieve it next time)
col = db.create_collection("articles", dimensions=3, metric="cosine")

# Insert vectors with optional metadata payload
col.upsert(1, [1.0, 0.0, 0.0], payload={"category": "tech",  "year": 2024})
col.upsert(2, [0.0, 1.0, 0.0], payload={"category": "sport", "year": 2023})
col.upsert(3, [0.9, 0.1, 0.0], payload={"category": "tech",  "year": 2023})
col.upsert(4, [0.1, 0.9, 0.0], payload={"category": "sport", "year": 2024})

# Plain search — returns top-k with distance and payload
results = col.search([1.0, 0.0, 0.0], k=2)
# [{"id": 1, "distance": 0.0,   "payload": {"category": "tech", "year": 2024}},
#  {"id": 3, "distance": 0.02,  "payload": {"category": "tech", "year": 2023}}]

# Filtered search — only return vectors whose payload matches
results = col.search(
    [1.0, 0.0, 0.0],
    k=5,
    filter={"category": {"$eq": "tech"}},
)
# Only "tech" articles, still ranked by vector distance

# Combine filters with $and / $or
results = col.search(
    [1.0, 0.0, 0.0],
    k=5,
    filter={"$and": [
        {"category": {"$eq": "tech"}},
        {"year":     {"$gte": 2024}},
    ]},
)
```

### Persistence across restarts

```python
import vectordb

# First run — insert data
db = vectordb.Client(path="./mydata")
col = db.create_collection("docs", dimensions=768, metric="cosine")
col.upsert(1, embedding_of("hello world"), payload={"text": "hello world"})

# Second run — data is automatically reloaded from disk
db = vectordb.Client(path="./mydata")
col = db.get_collection("docs")
print(col.count)  # 1 — survived the restart
```

### Managing collections

```python
db = vectordb.Client(path="./mydata")

# List all collections
names = db.list_collections()           # ["docs", "images"]

# Get an existing collection
col = db.get_collection("docs")

# Get if exists, create if not (idempotent)
col = db.get_or_create_collection("docs", dimensions=768, metric="cosine")

# Delete a collection and all its data
db.delete_collection("docs")           # returns True if found
```

### Client & Collection API

**`vectordb.Client(path="./data")`**

| Method | Returns | Description |
|--------|---------|-------------|
| `create_collection(name, dimensions, metric="cosine", index_type="hnsw")` | `Collection` | Create a new collection |
| `get_collection(name)` | `Collection` | Get existing collection (raises `KeyError` if not found) |
| `get_or_create_collection(name, dimensions, metric="cosine")` | `Collection` | Create if absent |
| `delete_collection(name)` | `bool` | Delete collection and all data |
| `list_collections()` | `list[str]` | Names of all collections |

**`vectordb.Collection`**

| Method / property | Description |
|-------------------|-------------|
| `upsert(id, vector, payload=None)` | Add or replace a vector |
| `search(query, k, filter=None)` → `list[dict]` | kNN search with optional filter |
| `delete(id)` → `bool` | Remove a vector |
| `count` | Number of stored vectors |
| `name` | Collection name |

---

## Offering 2: HTTP Server

Run `vectordb-server` as a standalone service and call it from any language over HTTP.

### Start the server

```bash
# Build
cargo build --release

# Run (data persisted to ./data by default)
./target/release/vectordb-server
# INFO vectordb-server listening on 0.0.0.0:8080

# Custom data directory
VECTORDB_DATA_DIR=/var/lib/vectordb ./target/release/vectordb-server

# With API key authentication
VECTORDB_API_KEY=mysecretkey ./target/release/vectordb-server
```

### End-to-end example with curl

```bash
# Create a collection
curl -s -X POST http://localhost:8080/collections/articles \
  -H "Content-Type: application/json" \
  -d '{"dimensions": 3, "metric": "cosine", "index_type": "flat"}'

# Insert vectors with payload metadata
curl -s -X POST http://localhost:8080/collections/articles/vectors \
  -H "Content-Type: application/json" \
  -d '{"id": 1, "vector": [1.0, 0.0, 0.0], "payload": {"category": "tech", "year": 2024}}'

curl -s -X POST http://localhost:8080/collections/articles/vectors \
  -H "Content-Type: application/json" \
  -d '{"id": 2, "vector": [0.0, 1.0, 0.0], "payload": {"category": "sport", "year": 2023}}'

# Plain search
curl -s -X POST http://localhost:8080/collections/articles/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [1.0, 0.0, 0.0], "k": 2}'
# {"results":[{"id":1,"distance":0.0,"payload":{"category":"tech","year":2024}}, ...]}

# Filtered search — only tech articles
curl -s -X POST http://localhost:8080/collections/articles/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [1.0, 0.0, 0.0],
    "k": 5,
    "filter": {"category": {"$eq": "tech"}}
  }'

# Compound filter
curl -s -X POST http://localhost:8080/collections/articles/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [1.0, 0.0, 0.0],
    "k": 5,
    "filter": {"$and": [
      {"category": {"$eq": "tech"}},
      {"year":     {"$gte": 2024}}
    ]}
  }'

# Delete a vector
curl -s -X DELETE http://localhost:8080/collections/articles/vectors/1

# Delete the collection
curl -s -X DELETE http://localhost:8080/collections/articles
```

### API key authentication

When `VECTORDB_API_KEY` is set, every request must include the key as a Bearer token:

```bash
VECTORDB_API_KEY=mysecretkey ./target/release/vectordb-server

# Authenticated request
curl -s http://localhost:8080/collections \
  -H "Authorization: Bearer mysecretkey"

# Missing / wrong key → 401 Unauthorized
curl -s http://localhost:8080/collections
# {"error":"invalid or missing API key"}
```

When `VECTORDB_API_KEY` is **not** set, the server runs in dev mode — all requests are allowed without authentication.

---

## Payload Metadata & Filtered Search

Every vector can carry an arbitrary JSON payload. Payloads are stored alongside the vector and returned in search results.

### Upsert with payload

```json
POST /collections/docs/vectors
{
  "id": 42,
  "vector": [0.1, 0.2, 0.3],
  "payload": {
    "title": "Introduction to Rust",
    "author": "Alice",
    "tags": ["rust", "systems"],
    "score": 0.95
  }
}
```

The `payload` field is optional — existing clients that omit it continue to work unchanged.

### Filter syntax

Filters are applied after the ANN search (post-filtering with 10× overscan). The filter format uses MongoDB-style operators:

| Operator | Meaning | Example |
|----------|---------|---------|
| `$eq` | Equal | `{"field": {"$eq": "value"}}` |
| `$ne` | Not equal | `{"field": {"$ne": "value"}}` |
| `$in` | One of | `{"field": {"$in": ["a", "b"]}}` |
| `$gt` | Greater than | `{"score": {"$gt": 0.5}}` |
| `$gte` | Greater than or equal | `{"year": {"$gte": 2020}}` |
| `$lt` | Less than | `{"score": {"$lt": 0.9}}` |
| `$lte` | Less than or equal | `{"year": {"$lte": 2024}}` |
| `$and` | All conditions true | `{"$and": [cond1, cond2]}` |
| `$or` | Any condition true | `{"$or": [cond1, cond2]}` |

Dot-notation field paths are supported: `"meta.author"` matches `{"meta": {"author": "alice"}}`.

Missing fields and type mismatches return `false` (the vector is excluded from results) — they do not raise errors.

### Filter examples

```json
// Single field equality
{"category": {"$eq": "tech"}}

// Numeric range
{"score": {"$gte": 0.8}}

// Member of a set
{"status": {"$in": ["published", "featured"]}}

// Compound: tech articles from 2024 onwards
{"$and": [
  {"category": {"$eq": "tech"}},
  {"year":     {"$gte": 2024}}
]}

// Either category
{"$or": [
  {"category": {"$eq": "tech"}},
  {"category": {"$eq": "science"}}
]}

// Nested field via dot-notation
{"meta.author": {"$eq": "alice"}}
```

---

## Persistence & Durability

vectordb uses a **write-ahead log (WAL)** for durability. Every upsert and delete is appended to an NDJSON log file before the in-memory index is updated.

### What this means for you

- **Crash-safe** — if the process dies mid-write, the partial entry is silently skipped on restart; all prior entries are replayed intact.
- **Automatic** — no `save()` or `flush()` calls needed. Just write vectors and restart freely.
- **Transparent** — data is stored in `{VECTORDB_DATA_DIR}/{collection_name}/`:
  ```
  ./data/
    articles/
      meta.json     ← collection config (dimensions, metric, index type)
      wal.log       ← append-only NDJSON journal
  ```

### WAL compaction

The WAL is automatically compacted (rewritten to contain only live entries) when it exceeds 50 000 entries. Compaction uses an atomic rename — if the process crashes during compaction, the original log is untouched.

---

## REST API Reference

All endpoints consume and produce `application/json`.

### Collections

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/collections` | List all collection names |
| `POST` | `/collections/:name` | Create a collection |
| `GET` | `/collections/:name` | Get collection info |
| `DELETE` | `/collections/:name` | Delete a collection and all its vectors |

**Create collection:**
```json
{
  "dimensions": 1536,
  "metric": "cosine",
  "index_type": "hnsw",
  "hnsw": {
    "ef_construction": 200,
    "ef_search": 50,
    "m": 12
  }
}
```

`metric`: `"l2"` | `"cosine"` | `"dot_product"`
`index_type`: `"flat"` | `"hnsw"` (default: `"hnsw"`)

### Vectors

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/collections/:name/vectors` | Upsert a vector |
| `POST` | `/collections/:name/search` | kNN search |
| `DELETE` | `/collections/:name/vectors/:id` | Delete a vector |

**Upsert:**
```json
{
  "id": 42,
  "vector": [0.1, 0.2, 0.3],
  "payload": {"title": "optional metadata"}
}
```

**Search:**
```json
{
  "vector": [0.1, 0.2, 0.3],
  "k": 10,
  "filter": {"category": {"$eq": "tech"}}
}
```

**Search response:**
```json
{
  "results": [
    {"id": 1, "distance": 0.012, "payload": {"category": "tech"}},
    {"id": 7, "distance": 0.034, "payload": {"category": "tech"}}
  ]
}
```

Vectors without a payload omit the `payload` key in the response. The `filter` field in search requests is optional — omitting it returns all candidates.

---

## CLI

`vdb` is the command-line client for a running `vectordb-server`.

### Build and install

```bash
cargo build --release
# binary: target/release/vdb
```

### Server URL

Default: `http://localhost:8080`. Override with `--host` or `VDB_HOST`:

```bash
vdb --host http://prod-server:8080 list
VDB_HOST=http://prod-server:8080 vdb list
```

### Commands

#### `vdb list`
```bash
vdb list
```

#### `vdb create <name>`
| Flag | Default | Description |
|------|---------|-------------|
| `--dimensions <N>` | *(required)* | Vector dimensions |
| `--metric <m>` | `cosine` | `l2` \| `cosine` \| `dot_product` |
| `--index <type>` | `hnsw` | `flat` \| `hnsw` |

```bash
vdb create articles --dimensions 768 --metric cosine --index hnsw
```

#### `vdb insert <collection>`
```bash
vdb insert articles --id 1 --vector "0.1,0.2,0.3"
```

#### `vdb search <collection>`
```bash
vdb search articles --vector "0.1,0.2,0.3" --k 10
```

#### `vdb delete <collection> --id <id>`
```bash
vdb delete articles --id 42
```

#### `vdb drop <collection>`
```bash
vdb drop articles
```

### End-to-end CLI example

```bash
# Start the server (persists to ./data)
./target/release/vectordb-server &

# Create a 3-dimensional collection
vdb create colours --dimensions 3 --metric l2 --index flat

# Insert some vectors
vdb insert colours --id 1 --vector "1.0,0.0,0.0"   # red
vdb insert colours --id 2 --vector "0.0,1.0,0.0"   # green
vdb insert colours --id 3 --vector "0.0,0.0,1.0"   # blue

# Search for the 2 closest to "mostly red"
vdb search colours --vector "0.9,0.1,0.0" --k 2

# Clean up
vdb delete colours --id 3
vdb drop colours
```

---

## Index Types

| Index | Recall | Query complexity | Use when |
|-------|--------|------------------|----------|
| `flat` | 100% | O(N · D) | < 100 K vectors, ground-truth eval |
| `hnsw` | ~95–99% (tunable) | O(log N · ef) | > 100 K vectors, latency-sensitive |

### HNSW tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ef_construction` | 200 | Graph quality during build. Higher → better recall, slower build. |
| `ef_search` | 50 | Beam width at query time. Higher → better recall, slower query. |
| `m` | 12 | Edges per node. Higher → better recall, more memory. |

---

## Distance Metrics

| Metric | Formula | Best for |
|--------|---------|----------|
| `l2` | `‖a − b‖₂` | Absolute position matters (coordinates, pixel embeddings) |
| `cosine` | `1 − (a·b) / (‖a‖ ‖b‖)` | Direction matters, magnitude doesn't (NLP embeddings) |
| `dot_product` | `−(a · b)` | Pre-normalised vectors, max-inner-product search |

---

## Low-Level Python Bindings

For direct, low-level index access without collections or persistence, the original `FlatIndex` and `HnswIndex` classes are still available:

```python
import vectordb

# --- FlatIndex (exact search) ---
idx = vectordb.FlatIndex(dimensions=3, metric="cosine")
idx.add(1, [1.0, 0.0, 0.0])
idx.add(2, [0.0, 1.0, 0.0])
idx.add_batch([(3, [0.0, 0.0, 1.0]), (4, [1.0, 1.0, 0.0])])

results = idx.search([1.0, 0.0, 0.0], k=2)
# [{"id": 1, "distance": 0.0}, {"id": 4, "distance": ...}]

idx.delete(4)
print(len(idx))       # 3
print(idx.dimensions) # 3
print(idx.metric)     # "cosine"

# Persist to disk (JSON snapshot)
idx.save("my_index.json")
idx2 = vectordb.FlatIndex.load("my_index.json")

# --- HnswIndex (approximate search) ---
hnsw = vectordb.HnswIndex(dimensions=128, metric="l2",
                           ef_construction=200, ef_search=50, m=12)
hnsw.add_batch([(i, [float(i)] * 128) for i in range(10_000)])
hnsw.flush()   # build the HNSW graph

results = hnsw.search([0.0] * 128, k=5)
hnsw.save("hnsw.json")
hnsw2 = vectordb.HnswIndex.load("hnsw.json")  # graph rebuilt automatically
```

> **Note:** `FlatIndex` and `HnswIndex` are in-memory only. Use `vectordb.Client` for persistence.

### Low-level API reference

| Class | Method / property | Description |
|-------|-------------------|-------------|
| `FlatIndex(dimensions, metric="l2")` | constructor | Create exact index |
| `HnswIndex(dimensions, metric="l2", ef_construction=200, ef_search=50, m=12)` | constructor | Create ANN index |
| both | `.add(id, vector)` | Insert one vector |
| both | `.add_batch([(id, vector), ...])` | Insert many vectors |
| both | `.search(query, k)` → `list[dict]` | Return k nearest neighbours |
| both | `.delete(id)` → `bool` | Remove a vector |
| both | `.save(path)` | Persist index to JSON |
| both | `cls.load(path)` | Restore index from JSON |
| both | `len(idx)` | Number of stored vectors |
| both | `.dimensions`, `.metric` | Read-only properties |
| `HnswIndex` | `.flush()` | Rebuild HNSW graph immediately |

---

## Architecture

```
                    ┌────────────────────────────────────────┐
                    │          vectordb-core (shared)        │
                    │                                        │
                    │  Collection  — index + WAL + payloads  │
                    │  CollectionManager — multi-collection  │
                    │  WAL         — NDJSON append-only log  │
                    │  FilterCondition — payload predicates  │
                    │  FlatIndex / HnswIndex                 │
                    └──────────────┬─────────────────────────┘
                                   │
               ┌───────────────────┴───────────────────────┐
               │                                           │
  ┌────────────▼────────────┐            ┌────────────────▼──────────┐
  │  vectordb-python        │            │  vectordb-server           │
  │                         │            │                            │
  │  Client(path=...)       │            │  REST API (Axum + Tokio)   │
  │  Collection             │            │  VECTORDB_DATA_DIR         │
  │  FlatIndex (low-level)  │            │  VECTORDB_API_KEY auth     │
  │  HnswIndex (low-level)  │            │                            │
  └─────────────────────────┘            └────────────────────────────┘
      Embedded Python                        Client-Server
```

| Crate | Role |
|-------|------|
| `vectordb-core` | Index trait, FlatIndex, HnswIndex, WAL, Collection, CollectionManager, payload filtering |
| `vectordb-server` | REST API server with persistence and optional API key auth |
| `vectordb-cli` | `vdb` command-line client |
| `vectordb-python` | PyO3 bindings — `Client`, `Collection`, `FlatIndex`, `HnswIndex` |

---

## Development Setup

### Prerequisites

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Python 3.8+ and maturin (only needed for Python bindings)
pip install maturin
```

### Build

```bash
# Rust (server + CLI)
cargo build

# Python bindings (venv must be active)
python3 -m venv .venv && source .venv/bin/activate
maturin develop
python3 -c "import vectordb; print('OK')"
```

### Test

```bash
cargo test
```

### Run the dev server

```bash
RUST_LOG=debug cargo run -p vectordb-server
```

### Project structure

```
crates/
├── vectordb-core/     # index trait, WAL, Collection, CollectionManager, filters
├── vectordb-server/   # Axum HTTP server
├── vectordb-cli/      # vdb CLI binary
└── vectordb-python/   # PyO3 Python bindings
```

---

## Building on macOS

### Apple Silicon — universal wheel

```bash
rustup target add x86_64-apple-darwin aarch64-apple-darwin
maturin build --release --target universal2-apple-darwin
```

### Known issues

#### `cargo build` fails with "Undefined symbols for architecture arm64"

PyO3 extension modules leave Python symbols unresolved at link time by design. Use `maturin develop` instead of plain `cargo build` for the Python bindings crate. The repository ships `.cargo/config.toml` with `-undefined dynamic_lookup` for macOS targets.

#### Python not found / wrong Python

```bash
export PYO3_PYTHON=$(which python3)
maturin develop
```

#### pyenv Python missing shared library

```bash
PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.11.9
pyenv global 3.11.9
```

---

## License

Apache 2.0
