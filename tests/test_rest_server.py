"""Tests for the Quiver REST API server."""

import json
import pytest
import tempfile
import threading
import time
import urllib.request
import urllib.error

from quiver_vector_db.server import create_server


@pytest.fixture(scope="module")
def server_url():
    """Start a test server on a random port and return its base URL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        server = create_server(host="127.0.0.1", port=0, data_path=tmpdir)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}"
        # Wait for server to start
        for _ in range(20):
            try:
                req = urllib.request.Request(f"{base_url}/healthz")
                urllib.request.urlopen(req, timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        yield base_url
        server.shutdown()


def _request(url, method="GET", data=None):
    """Helper: make HTTP request and return (status, parsed_json)."""
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestRestServer:
    """Tests for the REST API server endpoints."""

    def test_healthz(self, server_url):
        """Health check returns ok."""
        status, data = _request(f"{server_url}/healthz")
        assert status == 200
        assert data["status"] == "ok"

    def test_create_collection(self, server_url):
        """Create a collection via REST."""
        status, data = _request(f"{server_url}/collections", method="POST", data={
            "name": "test_col",
            "dimensions": 3,
            "metric": "l2",
        })
        assert status == 201
        assert data["created"] == "test_col"

    def test_list_collections(self, server_url):
        """List collections includes the created one."""
        status, data = _request(f"{server_url}/collections")
        assert status == 200
        assert "test_col" in data["collections"]

    def test_upsert_and_search(self, server_url):
        """Upsert a vector and search for it."""
        # Upsert
        status, data = _request(f"{server_url}/collections/test_col/upsert", method="POST", data={
            "id": 1,
            "vector": [1.0, 0.0, 0.0],
            "payload": {"tag": "first"},
        })
        assert status == 200
        assert data["upserted"] == 1

        # Upsert another
        _request(f"{server_url}/collections/test_col/upsert", method="POST", data={
            "id": 2,
            "vector": [0.0, 1.0, 0.0],
        })

        # Search
        status, data = _request(f"{server_url}/collections/test_col/search", method="POST", data={
            "query": [1.0, 0.0, 0.0],
            "k": 1,
        })
        assert status == 200
        assert len(data["results"]) == 1
        assert data["results"][0]["id"] == 1

    def test_count(self, server_url):
        """Count endpoint returns correct count."""
        status, data = _request(f"{server_url}/collections/test_col/count")
        assert status == 200
        assert data["count"] >= 2

    def test_upsert_batch(self, server_url):
        """Batch upsert via REST."""
        # Create fresh collection for batch test
        _request(f"{server_url}/collections", method="POST", data={
            "name": "batch_col",
            "dimensions": 3,
        })
        status, data = _request(f"{server_url}/collections/batch_col/upsert_batch", method="POST", data={
            "entries": [
                {"id": 10, "vector": [1.0, 0.0, 0.0], "payload": {"x": 1}},
                {"id": 11, "vector": [0.0, 1.0, 0.0]},
                {"id": 12, "vector": [0.0, 0.0, 1.0], "payload": {"x": 3}},
            ],
        })
        assert status == 200
        assert data["upserted"] == 3

    def test_delete_vector(self, server_url):
        """Delete a vector via REST."""
        status, data = _request(f"{server_url}/collections/test_col/delete", method="POST", data={
            "id": 2,
        })
        assert status == 200
        assert data["deleted"] is True

    def test_delete_collection(self, server_url):
        """Delete a collection via REST."""
        # Create then delete
        _request(f"{server_url}/collections", method="POST", data={
            "name": "to_delete",
            "dimensions": 3,
        })
        status, data = _request(f"{server_url}/collections/to_delete", method="DELETE")
        assert status == 200
        assert data["deleted"] is True

    def test_not_found_collection(self, server_url):
        """Operations on missing collection return 404."""
        status, data = _request(f"{server_url}/collections/nonexistent/count")
        assert status == 404

    def test_snapshot_lifecycle(self, server_url):
        """Create, list, and delete snapshot via REST."""
        # Create collection with data
        _request(f"{server_url}/collections", method="POST", data={
            "name": "snap_col",
            "dimensions": 3,
        })
        _request(f"{server_url}/collections/snap_col/upsert", method="POST", data={
            "id": 1, "vector": [1.0, 0.0, 0.0],
        })

        # Create snapshot
        status, data = _request(f"{server_url}/collections/snap_col/snapshots", method="POST", data={
            "name": "v1",
        })
        assert status == 201
        assert data["name"] == "v1"

        # List snapshots
        status, data = _request(f"{server_url}/collections/snap_col/snapshots")
        assert status == 200
        assert len(data["snapshots"]) == 1

        # Delete snapshot
        status, data = _request(f"{server_url}/collections/snap_col/snapshots/v1", method="DELETE")
        assert status == 200
        assert data["deleted"] is True

    def test_invalid_json(self, server_url):
        """Invalid JSON body returns 400."""
        req = urllib.request.Request(
            f"{server_url}/collections",
            data=b"not json",
            method="POST",
        )
        req.add_header("Content-Type", "application/json")
        try:
            resp = urllib.request.urlopen(req, timeout=5)
            assert False, "expected error"
        except urllib.error.HTTPError as e:
            assert e.code == 400
