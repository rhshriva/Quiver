"""Tests for embedding function protocol and mock embedder."""

import pytest
from quiver_vector_db.embedding import EmbeddingFunction


class MockEmbedding:
    """A deterministic mock embedder for testing."""

    def __init__(self, dimensions: int = 4):
        self._dimensions = dimensions

    def __call__(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            h = hash(text) & 0xFFFFFFFF
            vec = [(h >> (i * 4) & 0xF) / 15.0 for i in range(self._dimensions)]
            results.append(vec)
        return results

    @property
    def dimensions(self) -> int:
        return self._dimensions


class TestEmbeddingProtocol:
    def test_mock_satisfies_protocol(self):
        ef = MockEmbedding(dimensions=4)
        assert isinstance(ef, EmbeddingFunction)

    def test_returns_correct_dimensions(self):
        ef = MockEmbedding(dimensions=8)
        result = ef(["hello"])
        assert len(result) == 1
        assert len(result[0]) == 8

    def test_batch_embedding(self):
        ef = MockEmbedding(dimensions=4)
        result = ef(["hello", "world", "test"])
        assert len(result) == 3
        assert all(len(v) == 4 for v in result)

    def test_deterministic(self):
        ef = MockEmbedding(dimensions=4)
        r1 = ef(["hello"])
        r2 = ef(["hello"])
        assert r1 == r2

    def test_different_texts_different_vectors(self):
        ef = MockEmbedding(dimensions=4)
        r1 = ef(["hello"])[0]
        r2 = ef(["world"])[0]
        assert r1 != r2

    def test_dimensions_property(self):
        ef = MockEmbedding(dimensions=16)
        assert ef.dimensions == 16

    def test_empty_batch(self):
        ef = MockEmbedding(dimensions=4)
        result = ef([])
        assert result == []


class TestSentenceTransformerImportError:
    def test_import_error_message(self):
        """SentenceTransformerEmbedding should give clear error if not installed."""
        try:
            from quiver_vector_db.embedding import SentenceTransformerEmbedding
            SentenceTransformerEmbedding("all-MiniLM-L6-v2")
        except ImportError as e:
            assert "sentence-transformers" in str(e)
        except Exception:
            pass  # sentence-transformers might be installed


class TestOpenAIImportError:
    def test_import_error_message(self):
        """OpenAIEmbedding should give clear error if not installed."""
        try:
            from quiver_vector_db.embedding import OpenAIEmbedding
            OpenAIEmbedding(api_key="fake")
        except ImportError as e:
            assert "openai" in str(e)
        except Exception:
            pass  # openai might be installed
