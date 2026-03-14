"""Tests for the MultiVectorCollection (multi-modal) feature."""

import pytest
import tempfile
import random

import quiver_vector_db as quiver
from quiver_vector_db import MultiVectorCollection


random.seed(42)


def rand_vec(dim):
    return [random.gauss(0, 1) for _ in range(dim)]


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield quiver.Client(path=tmpdir)


class TestMultiVectorCollection:
    """Tests for multi-vector / multi-modal collections."""

    def test_create(self, db):
        """Create a multi-vector collection with two spaces."""
        multi = MultiVectorCollection(
            client=db,
            name="products",
            vector_spaces={
                "text":  {"dimensions": 32, "metric": "cosine"},
                "image": {"dimensions": 64, "metric": "cosine"},
            },
        )
        assert multi.name == "products"
        assert sorted(multi.vector_spaces) == ["image", "text"]

    def test_upsert_and_search_single_space(self, db):
        """Upsert with multiple spaces, search in one."""
        multi = MultiVectorCollection(
            client=db,
            name="docs",
            vector_spaces={
                "text":  {"dimensions": 8, "metric": "l2"},
                "image": {"dimensions": 8, "metric": "l2"},
            },
        )

        text_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        image_vec = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        multi.upsert(id=1, vectors={"text": text_vec, "image": image_vec}, payload={"title": "Doc1"})
        multi.upsert(id=2, vectors={"text": [0.0]*8, "image": [0.0]*8})

        # Search in text space
        results = multi.search("text", query=text_vec, k=1)
        assert results[0]["id"] == 1

        # Search in image space
        results = multi.search("image", query=image_vec, k=1)
        assert results[0]["id"] == 1

    def test_partial_upsert(self, db):
        """Upsert with only some vector spaces."""
        multi = MultiVectorCollection(
            client=db,
            name="partial",
            vector_spaces={
                "text":  {"dimensions": 4, "metric": "l2"},
                "image": {"dimensions": 4, "metric": "l2"},
            },
        )

        multi.upsert(id=1, vectors={"text": [1.0, 0.0, 0.0, 0.0]})
        multi.upsert(id=2, vectors={"image": [0.0, 1.0, 0.0, 0.0]})

        text_results = multi.search("text", query=[1.0, 0.0, 0.0, 0.0], k=5)
        assert any(r["id"] == 1 for r in text_results)

    def test_search_multi(self, db):
        """Cross-space search with weighted fusion."""
        multi = MultiVectorCollection(
            client=db,
            name="fusion",
            vector_spaces={
                "text":  {"dimensions": 4, "metric": "l2"},
                "image": {"dimensions": 4, "metric": "l2"},
            },
        )

        # Doc 1: good text match, bad image match
        multi.upsert(id=1, vectors={
            "text":  [1.0, 0.0, 0.0, 0.0],
            "image": [0.0, 0.0, 0.0, 1.0],
        })
        # Doc 2: bad text match, good image match
        multi.upsert(id=2, vectors={
            "text":  [0.0, 0.0, 0.0, 1.0],
            "image": [1.0, 0.0, 0.0, 0.0],
        })
        # Doc 3: decent at both
        multi.upsert(id=3, vectors={
            "text":  [0.7, 0.3, 0.0, 0.0],
            "image": [0.7, 0.3, 0.0, 0.0],
        })

        # Search with equal weights
        results = multi.search_multi(
            queries={
                "text":  [1.0, 0.0, 0.0, 0.0],
                "image": [1.0, 0.0, 0.0, 0.0],
            },
            k=3,
            weights={"text": 0.5, "image": 0.5},
        )

        assert len(results) == 3
        # Doc 3 should score best (decent at both)
        assert results[0]["id"] == 3
        # All results have scores
        assert all("score" in r for r in results)
        assert all("distances" in r for r in results)

    def test_count(self, db):
        """Count returns document count from primary space."""
        multi = MultiVectorCollection(
            client=db,
            name="cnt",
            vector_spaces={"a": {"dimensions": 2}, "b": {"dimensions": 2}},
        )
        multi.upsert(id=1, vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]})
        multi.upsert(id=2, vectors={"a": [0.0, 1.0], "b": [1.0, 0.0]})
        assert multi.count == 2

    def test_delete(self, db):
        """Delete removes document from all spaces."""
        multi = MultiVectorCollection(
            client=db,
            name="del",
            vector_spaces={"a": {"dimensions": 2}, "b": {"dimensions": 2}},
        )
        multi.upsert(id=1, vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]})
        multi.upsert(id=2, vectors={"a": [0.0, 1.0], "b": [1.0, 0.0]})
        multi.delete(1)
        assert multi.count == 1

    def test_delete_batch(self, db):
        """Batch delete removes multiple documents."""
        multi = MultiVectorCollection(
            client=db,
            name="delbatch",
            vector_spaces={"a": {"dimensions": 2}},
        )
        for i in range(5):
            multi.upsert(id=i, vectors={"a": [float(i), 0.0]})
        multi.delete_batch([0, 1, 2])
        assert multi.count == 2

    def test_upsert_batch(self, db):
        """Batch upsert multiple documents."""
        multi = MultiVectorCollection(
            client=db,
            name="batch",
            vector_spaces={"text": {"dimensions": 4}, "image": {"dimensions": 4}},
        )
        multi.upsert_batch([
            (1, {"text": [1.0, 0.0, 0.0, 0.0], "image": [0.0, 1.0, 0.0, 0.0]}, {"title": "A"}),
            (2, {"text": [0.0, 1.0, 0.0, 0.0]}, {"title": "B"}),
            (3, {"text": [0.0, 0.0, 1.0, 0.0], "image": [0.0, 0.0, 0.0, 1.0]}),
        ])
        assert multi.count == 3

    def test_unknown_space_raises(self, db):
        """Using an unknown vector space raises KeyError."""
        multi = MultiVectorCollection(
            client=db,
            name="ukn",
            vector_spaces={"text": {"dimensions": 2}},
        )
        with pytest.raises(KeyError, match="image"):
            multi.upsert(id=1, vectors={"image": [1.0, 0.0]})

    def test_empty_vector_spaces_raises(self, db):
        """Empty vector_spaces raises ValueError."""
        with pytest.raises(ValueError):
            MultiVectorCollection(client=db, name="empty", vector_spaces={})

    def test_reopens(self, db):
        """Multi-vector collection data persists across client reopens."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db1 = quiver.Client(path=tmpdir)
            multi = MultiVectorCollection(
                client=db1,
                name="persist",
                vector_spaces={"text": {"dimensions": 4, "metric": "cosine"}},
            )
            multi.upsert(id=1, vectors={"text": [1.0, 0.0, 0.0, 0.0]})
            del db1

            db2 = quiver.Client(path=tmpdir)
            multi2 = MultiVectorCollection(
                client=db2,
                name="persist",
                vector_spaces={"text": {"dimensions": 4, "metric": "cosine"}},
            )
            assert multi2.count == 1
            results = multi2.search("text", query=[1.0, 0.0, 0.0, 0.0], k=1)
            assert results[0]["id"] == 1
