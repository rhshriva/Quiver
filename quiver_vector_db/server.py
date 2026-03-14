"""Lightweight REST API server for Quiver Vector DB.

Wraps the Quiver Python client with a zero-dependency HTTP server (stdlib
only). Ideal for language-agnostic access or quick prototyping.

Usage::

    # Start server on default port 8080
    python -m quiver_vector_db.server

    # Custom port and data directory
    python -m quiver_vector_db.server --port 9090 --data ./my_data

Endpoints::

    GET  /healthz                              → {"status": "ok"}
    GET  /collections                          → {"collections": [...]}
    POST /collections                          → create collection
    DELETE /collections/{name}                 → delete collection

    POST /collections/{name}/upsert            → upsert a vector
    POST /collections/{name}/upsert_batch      → batch upsert
    POST /collections/{name}/search            → search vectors
    POST /collections/{name}/delete            → delete a vector
    GET  /collections/{name}/count             → {"count": N}

    POST /collections/{name}/snapshots         → create snapshot
    GET  /collections/{name}/snapshots         → list snapshots
    POST /collections/{name}/snapshots/restore → restore snapshot
    DELETE /collections/{name}/snapshots/{snap} → delete snapshot
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse, parse_qs


def _json_response(handler: BaseHTTPRequestHandler, data: Any, status: int = 200) -> None:
    """Write a JSON response."""
    body = json.dumps(data, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    """Read and parse the JSON request body."""
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw)


def _error(handler: BaseHTTPRequestHandler, message: str, status: int = 400) -> None:
    """Write an error JSON response."""
    _json_response(handler, {"error": message}, status)


def _parse_path(path: str) -> Tuple[str, ...]:
    """Split URL path into segments, stripping leading/trailing slashes."""
    return tuple(s for s in urlparse(path).path.strip("/").split("/") if s)


class QuiverHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Quiver REST API."""

    # Set by the server factory
    db: Any = None

    def log_message(self, format: str, *args: Any) -> None:
        """Override to use compact log format."""
        sys.stderr.write(f"[quiver] {self.address_string()} {format % args}\n")

    # ── CORS preflight ──────────────────────────────────────────────────

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── GET ──────────────────────────────────────────────────────────────

    def do_GET(self) -> None:
        parts = _parse_path(self.path)

        # GET /healthz
        if parts == ("healthz",):
            return _json_response(self, {"status": "ok"})

        # GET /collections
        if parts == ("collections",):
            names = self.db.list_collections()
            return _json_response(self, {"collections": names})

        # GET /collections/{name}/count
        if len(parts) == 3 and parts[0] == "collections" and parts[2] == "count":
            name = parts[1]
            try:
                col = self.db.get_collection(name)
                return _json_response(self, {"count": col.count})
            except KeyError:
                return _error(self, f"collection '{name}' not found", 404)

        # GET /collections/{name}/snapshots
        if len(parts) == 3 and parts[0] == "collections" and parts[2] == "snapshots":
            name = parts[1]
            try:
                col = self.db.get_collection(name)
                snaps = col.list_snapshots()
                return _json_response(self, {"snapshots": snaps})
            except KeyError:
                return _error(self, f"collection '{name}' not found", 404)

        return _error(self, "not found", 404)

    # ── POST ─────────────────────────────────────────────────────────────

    def do_POST(self) -> None:
        parts = _parse_path(self.path)

        try:
            body = _read_body(self)
        except (json.JSONDecodeError, ValueError) as e:
            return _error(self, f"invalid JSON: {e}")

        # POST /collections → create collection
        if parts == ("collections",):
            return self._create_collection(body)

        if len(parts) >= 3 and parts[0] == "collections":
            name = parts[1]
            action = parts[2]

            if action == "upsert":
                return self._upsert(name, body)
            if action == "upsert_batch":
                return self._upsert_batch(name, body)
            if action == "search":
                return self._search(name, body)
            if action == "delete":
                return self._delete_vector(name, body)
            if action == "snapshots":
                if len(parts) == 3:
                    return self._create_snapshot(name, body)
                if len(parts) == 4 and parts[3] == "restore":
                    return self._restore_snapshot(name, body)

        return _error(self, "not found", 404)

    # ── DELETE ───────────────────────────────────────────────────────────

    def do_DELETE(self) -> None:
        parts = _parse_path(self.path)

        # DELETE /collections/{name}
        if len(parts) == 2 and parts[0] == "collections":
            name = parts[1]
            try:
                deleted = self.db.delete_collection(name)
                return _json_response(self, {"deleted": deleted})
            except KeyError:
                return _error(self, f"collection '{name}' not found", 404)

        # DELETE /collections/{name}/snapshots/{snap_name}
        if len(parts) == 4 and parts[0] == "collections" and parts[2] == "snapshots":
            col_name = parts[1]
            snap_name = parts[3]
            try:
                col = self.db.get_collection(col_name)
                col.delete_snapshot(snap_name)
                return _json_response(self, {"deleted": True})
            except KeyError as e:
                return _error(self, str(e), 404)

        return _error(self, "not found", 404)

    # ── Handlers ─────────────────────────────────────────────────────────

    def _create_collection(self, body: Dict[str, Any]) -> None:
        name = body.get("name")
        dims = body.get("dimensions")
        metric = body.get("metric", "cosine")
        index_type = body.get("index_type", "hnsw")

        if not name or not dims:
            return _error(self, "'name' and 'dimensions' are required")

        try:
            self.db.create_collection(name, dimensions=dims, metric=metric, index_type=index_type)
            return _json_response(self, {"created": name}, 201)
        except KeyError:
            return _error(self, f"collection '{name}' already exists", 409)

    def _upsert(self, name: str, body: Dict[str, Any]) -> None:
        try:
            col = self.db.get_collection(name)
        except KeyError:
            return _error(self, f"collection '{name}' not found", 404)

        doc_id = body.get("id")
        vector = body.get("vector")
        payload = body.get("payload")

        if doc_id is None or vector is None:
            return _error(self, "'id' and 'vector' are required")

        col.upsert(id=doc_id, vector=vector, payload=payload)
        return _json_response(self, {"upserted": doc_id})

    def _upsert_batch(self, name: str, body: Dict[str, Any]) -> None:
        try:
            col = self.db.get_collection(name)
        except KeyError:
            return _error(self, f"collection '{name}' not found", 404)

        entries = body.get("entries")
        if not entries or not isinstance(entries, list):
            return _error(self, "'entries' array is required")

        batch = []
        for e in entries:
            doc_id = e.get("id")
            vector = e.get("vector")
            payload = e.get("payload")
            if doc_id is None or vector is None:
                return _error(self, "each entry must have 'id' and 'vector'")
            batch.append((doc_id, vector, payload))

        col.upsert_batch(batch)
        return _json_response(self, {"upserted": len(batch)})

    def _search(self, name: str, body: Dict[str, Any]) -> None:
        try:
            col = self.db.get_collection(name)
        except KeyError:
            return _error(self, f"collection '{name}' not found", 404)

        query = body.get("query")
        k = body.get("k", 10)
        filter_ = body.get("filter")

        if query is None:
            return _error(self, "'query' vector is required")

        results = col.search(query=query, k=k, filter=filter_)
        return _json_response(self, {"results": results})

    def _delete_vector(self, name: str, body: Dict[str, Any]) -> None:
        try:
            col = self.db.get_collection(name)
        except KeyError:
            return _error(self, f"collection '{name}' not found", 404)

        doc_id = body.get("id")
        if doc_id is None:
            return _error(self, "'id' is required")

        deleted = col.delete(id=doc_id)
        return _json_response(self, {"deleted": deleted})

    def _create_snapshot(self, name: str, body: Dict[str, Any]) -> None:
        try:
            col = self.db.get_collection(name)
        except KeyError:
            return _error(self, f"collection '{name}' not found", 404)

        snap_name = body.get("name")
        if not snap_name:
            return _error(self, "'name' is required")

        try:
            meta = col.create_snapshot(snap_name)
            return _json_response(self, meta, 201)
        except KeyError:
            return _error(self, f"snapshot '{snap_name}' already exists", 409)

    def _restore_snapshot(self, name: str, body: Dict[str, Any]) -> None:
        try:
            col = self.db.get_collection(name)
        except KeyError:
            return _error(self, f"collection '{name}' not found", 404)

        snap_name = body.get("name")
        if not snap_name:
            return _error(self, "'name' is required")

        try:
            col.restore_snapshot(snap_name)
            return _json_response(self, {"restored": snap_name})
        except KeyError:
            return _error(self, f"snapshot '{snap_name}' not found", 404)


def create_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    data_path: str = "./data",
) -> HTTPServer:
    """Create and return a Quiver REST API server.

    Args:
        host: Bind address (default ``"0.0.0.0"``).
        port: Port number (default ``8080``).
        data_path: Path to data directory (default ``"./data"``).

    Returns:
        An ``HTTPServer`` instance ready to ``serve_forever()``.
    """
    import quiver_vector_db as quiver

    db = quiver.Client(path=data_path)

    handler = type("Handler", (QuiverHandler,), {"db": db})
    server = HTTPServer((host, port), handler)
    return server


def main() -> None:
    """CLI entry point for the Quiver REST server."""
    parser = argparse.ArgumentParser(
        description="Quiver Vector DB — REST API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Endpoints:\n"
            "  GET  /healthz                              Health check\n"
            "  GET  /collections                          List collections\n"
            "  POST /collections                          Create collection\n"
            "  DELETE /collections/{name}                 Delete collection\n"
            "  POST /collections/{name}/upsert            Upsert vector\n"
            "  POST /collections/{name}/upsert_batch      Batch upsert\n"
            "  POST /collections/{name}/search            Search vectors\n"
            "  POST /collections/{name}/delete            Delete vector\n"
            "  GET  /collections/{name}/count             Count vectors\n"
            "  POST /collections/{name}/snapshots         Create snapshot\n"
            "  GET  /collections/{name}/snapshots         List snapshots\n"
            "  POST /collections/{name}/snapshots/restore Restore snapshot\n"
            "  DELETE /collections/{name}/snapshots/{s}   Delete snapshot\n"
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--data", default="./data", help="Data directory (default: ./data)")
    args = parser.parse_args()

    server = create_server(host=args.host, port=args.port, data_path=args.data)
    print(f"Quiver REST server listening on http://{args.host}:{args.port}")
    print(f"Data directory: {args.data}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
