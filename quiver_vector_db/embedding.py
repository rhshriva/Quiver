"""Embedding function protocol and built-in providers.

Defines the ``EmbeddingFunction`` protocol that any embedding provider must
satisfy, plus two convenience classes for the most common providers:

- ``SentenceTransformerEmbedding`` — local models via sentence-transformers
- ``OpenAIEmbedding`` — OpenAI embedding API

Both use lazy imports so their heavy dependencies are only required when
actually instantiated.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingFunction(Protocol):
    """Protocol that any embedding provider must satisfy.

    Implement ``__call__`` to embed a batch of texts, and optionally
    ``dimensions`` to report the output dimensionality.

    Example::

        class MyEmbedder:
            def __call__(self, texts: list[str]) -> list[list[float]]:
                return [my_model.encode(t) for t in texts]

            @property
            def dimensions(self) -> int:
                return 384
    """

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts into dense float vectors."""
        ...

    @property
    def dimensions(self) -> Optional[int]:
        """Number of dimensions per vector, or None if unknown."""
        ...


class SentenceTransformerEmbedding:
    """Local embedding via sentence-transformers.

    Requires: ``pip install quiver-vector-db[sentence-transformers]``

    Example::

        ef = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
        vectors = ef(["hello world", "vector search"])
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedding. "
                "Install it with: pip install quiver-vector-db[sentence-transformers]"
            )
        self._model = SentenceTransformer(model_name, device=device)
        self._dimensions: int = self._model.get_sentence_embedding_dimension()

    def __call__(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimensions(self) -> int:
        return self._dimensions


class OpenAIEmbedding:
    """OpenAI embedding API wrapper.

    Requires: ``pip install quiver-vector-db[openai]``

    Example::

        ef = OpenAIEmbedding(api_key="sk-...")
        vectors = ef(["hello world"])
    """

    _KNOWN_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> None:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai is required for OpenAIEmbedding. "
                "Install it with: pip install quiver-vector-db[openai]"
            )
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions or self._KNOWN_DIMS.get(model)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    @property
    def dimensions(self) -> Optional[int]:
        return self._dimensions
