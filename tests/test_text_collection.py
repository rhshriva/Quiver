"""Integration tests for TextCollection (embedding + BM25 + Rust Collection)."""

import hashlib
import pytest
import quiver_vector_db as quiver
from quiver_vector_db.text_collection import TextCollection


class MockEmbedding:
    """Deterministic mock embedder for testing without real models."""

    def __init__(self, dimensions: int = 4):
        self._dimensions = dimensions

    def __call__(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            h = int(hashlib.md5(text.encode()).hexdigest(), 16)
            vec = [(h >> (i * 8) & 0xFF) / 255.0 for i in range(self._dimensions)]
            results.append(vec)
        return results

    @property
    def dimensions(self) -> int:
        return self._dimensions


@pytest.fixture
def text_col(tmp_path):
    db = quiver.Client(path=str(tmp_path))
    raw_col = db.create_collection("docs", dimensions=4, metric="cosine")
    return TextCollection(
        collection=raw_col,
        embedding_function=MockEmbedding(dimensions=4),
    )


class TestAdd:
    def test_add_single_document(self, text_col):
        text_col.add(ids=[1], documents=["hello world"])
        assert text_col.count == 1

    def test_add_multiple_documents(self, text_col):
        text_col.add(
            ids=[1, 2, 3],
            documents=["first doc", "second doc", "third doc"],
        )
        assert text_col.count == 3

    def test_add_with_payloads(self, text_col):
        text_col.add(
            ids=[1, 2],
            documents=["hello", "world"],
            payloads=[{"tag": "a"}, {"tag": "b"}],
        )
        assert text_col.count == 2

    def test_add_mismatched_ids_docs_raises(self, text_col):
        with pytest.raises(ValueError, match="same length"):
            text_col.add(ids=[1, 2], documents=["only one"])

    def test_add_mismatched_payloads_raises(self, text_col):
        with pytest.raises(ValueError, match="same length"):
            text_col.add(ids=[1], documents=["doc"], payloads=[{}, {}])

    def test_add_with_none_payload(self, text_col):
        text_col.add(
            ids=[1, 2],
            documents=["hello", "world"],
            payloads=[{"tag": "a"}, None],
        )
        assert text_col.count == 2


class TestQuery:
    def test_semantic_search(self, text_col):
        text_col.add(
            ids=[1, 2, 3],
            documents=[
                "machine learning algorithms",
                "deep neural networks",
                "cooking recipes for dinner",
            ],
        )
        results = text_col.query("machine learning", k=3, mode="semantic")
        assert len(results) >= 1
        assert "id" in results[0]
        assert "distance" in results[0]
        assert "document" in results[0]

    def test_hybrid_search(self, text_col):
        text_col.add(
            ids=[1, 2, 3],
            documents=[
                "machine learning algorithms",
                "deep neural networks",
                "cooking recipes for dinner",
            ],
        )
        results = text_col.query("machine learning", k=3, mode="hybrid")
        assert len(results) >= 1
        assert "id" in results[0]
        assert "score" in results[0]
        assert "dense_distance" in results[0]
        assert "sparse_score" in results[0]

    def test_keyword_search(self, text_col):
        text_col.add(
            ids=[1, 2, 3],
            documents=[
                "machine learning algorithms python",
                "deep neural networks tensorflow",
                "cooking recipes dinner kitchen",
            ],
        )
        results = text_col.query("machine learning python", k=3, mode="keyword")
        assert len(results) >= 1
        # Doc 1 should rank highest for keyword match
        assert results[0]["id"] == 1

    def test_document_text_returned(self, text_col):
        text_col.add(ids=[1], documents=["hello world programming"])
        results = text_col.query("hello programming", k=1, mode="semantic")
        assert results[0]["document"] == "hello world programming"

    def test_payload_returned_without_internal_key(self, text_col):
        text_col.add(
            ids=[1],
            documents=["test doc"],
            payloads=[{"category": "tech"}],
        )
        results = text_col.query("test", k=1, mode="semantic")
        payload = results[0]["payload"]
        assert payload is not None
        assert payload["category"] == "tech"
        assert "_document" not in payload

    def test_query_with_filter(self, text_col):
        text_col.add(
            ids=[1, 2],
            documents=["tech article about python", "cooking guide for pasta"],
            payloads=[{"cat": "tech"}, {"cat": "food"}],
        )
        results = text_col.query(
            "article",
            k=10,
            mode="semantic",
            filter={"cat": {"$eq": "tech"}},
        )
        ids = {r["id"] for r in results}
        assert ids <= {1}

    def test_unknown_mode_raises(self, text_col):
        text_col.add(ids=[1], documents=["test"])
        with pytest.raises(ValueError, match="Unknown mode"):
            text_col.query("test", mode="invalid")

    def test_hybrid_default_mode(self, text_col):
        """Default mode should be hybrid."""
        text_col.add(ids=[1, 2], documents=["hello world", "foo bar"])
        results = text_col.query("hello", k=2)
        assert len(results) >= 1
        assert "score" in results[0]

    def test_keyword_with_unknown_terms_returns_empty(self, text_col):
        text_col.add(ids=[1], documents=["hello world"])
        results = text_col.query("xyznonexistent", k=5, mode="keyword")
        assert results == []


class TestDelete:
    def test_delete_documents(self, text_col):
        text_col.add(ids=[1, 2, 3], documents=["aa bb", "cc dd", "ee ff"])
        assert text_col.count == 3
        text_col.delete(ids=[1, 2])
        assert text_col.count == 1

    def test_delete_updates_bm25(self, text_col):
        text_col.add(ids=[1, 2], documents=["hello world", "foo bar"])
        assert text_col._bm25.doc_count == 2
        text_col.delete(ids=[1])
        assert text_col._bm25.doc_count == 1


class TestProperties:
    def test_name_property(self, text_col):
        assert text_col.name == "docs"

    def test_count_property(self, text_col):
        assert text_col.count == 0
        text_col.add(ids=[1], documents=["test"])
        assert text_col.count == 1


class TestBM25Disabled:
    def test_semantic_only_collection(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        raw_col = db.create_collection("no_bm25", dimensions=4, metric="cosine")
        col = TextCollection(
            collection=raw_col,
            embedding_function=MockEmbedding(dimensions=4),
            enable_bm25=False,
        )
        col.add(ids=[1, 2], documents=["hello world", "foo bar"])
        assert col.count == 2
        results = col.query("hello", k=1, mode="semantic")
        assert len(results) >= 1

    def test_keyword_mode_raises_when_bm25_disabled(self, tmp_path):
        db = quiver.Client(path=str(tmp_path))
        raw_col = db.create_collection("no_bm25", dimensions=4, metric="cosine")
        col = TextCollection(
            collection=raw_col,
            embedding_function=MockEmbedding(dimensions=4),
            enable_bm25=False,
        )
        col.add(ids=[1], documents=["test"])
        with pytest.raises(ValueError, match="BM25 is not enabled"):
            col.query("test", mode="keyword")

    def test_hybrid_falls_back_to_semantic(self, tmp_path):
        """When BM25 is disabled, hybrid mode should fall back to semantic."""
        db = quiver.Client(path=str(tmp_path))
        raw_col = db.create_collection("no_bm25", dimensions=4, metric="cosine")
        col = TextCollection(
            collection=raw_col,
            embedding_function=MockEmbedding(dimensions=4),
            enable_bm25=False,
        )
        col.add(ids=[1], documents=["test doc"])
        results = col.query("test", k=1, mode="hybrid")
        assert len(results) >= 1
        assert "distance" in results[0]  # semantic result format
