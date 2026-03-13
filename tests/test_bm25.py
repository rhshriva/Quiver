"""Unit tests for BM25 tokenizer and sparse vector generation."""

import pytest
from quiver_vector_db.bm25 import BM25, _tokenize


class TestTokenizer:
    def test_basic_tokenization(self):
        tokens = _tokenize("Hello World! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_stopword_removal(self):
        tokens = _tokenize("this is a the and or")
        assert tokens == []

    def test_min_length_filter(self):
        tokens = _tokenize("I a go ok am do")
        # "am", "ok", "go", "do" are length 2 and not stopwords
        assert "am" in tokens
        assert "ok" in tokens

    def test_numeric_tokens(self):
        tokens = _tokenize("Python 3 version 42")
        assert "python" in tokens
        assert "42" in tokens
        assert "version" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_punctuation_splitting(self):
        tokens = _tokenize("hello-world foo_bar baz.qux")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens
        assert "bar" in tokens


class TestBM25:
    def test_index_single_document(self):
        bm25 = BM25()
        sparse = bm25.index_document(1, "the quick brown fox jumps over the lazy dog")
        assert bm25.doc_count == 1
        assert bm25.vocab_size > 0
        assert len(sparse) > 0
        assert all(isinstance(k, int) for k in sparse)
        assert all(isinstance(v, float) for v in sparse.values())

    def test_index_multiple_documents(self):
        bm25 = BM25()
        bm25.index_document(1, "the quick brown fox")
        bm25.index_document(2, "the lazy brown dog")
        assert bm25.doc_count == 2

    def test_query_returns_known_terms(self):
        bm25 = BM25()
        bm25.index_document(1, "quick brown fox")
        bm25.index_document(2, "lazy brown dog")
        query = bm25.encode_query("quick brown fox")
        assert len(query) > 0

    def test_unknown_query_terms_ignored(self):
        bm25 = BM25()
        bm25.index_document(1, "hello world")
        query = bm25.encode_query("unknown xyz abc")
        assert query == {}

    def test_empty_document_returns_empty(self):
        bm25 = BM25()
        sparse = bm25.index_document(1, "")
        assert sparse == {}
        assert bm25.doc_count == 0

    def test_stopword_only_document(self):
        bm25 = BM25()
        sparse = bm25.index_document(1, "the a is are")
        assert sparse == {}

    def test_remove_document(self):
        bm25 = BM25()
        bm25.index_document(1, "hello world foo bar")
        bm25.index_document(2, "hello baz qux")
        assert bm25.doc_count == 2
        bm25.remove_document(1)
        assert bm25.doc_count == 1

    def test_remove_nonexistent_is_noop(self):
        bm25 = BM25()
        bm25.index_document(1, "hello world")
        bm25.remove_document(999)  # should not raise
        assert bm25.doc_count == 1

    def test_reindex_document(self):
        bm25 = BM25()
        bm25.index_document(1, "hello world")
        assert bm25.doc_count == 1
        bm25.index_document(1, "goodbye world updated")
        assert bm25.doc_count == 1  # same doc_id, count should not increase

    def test_save_and_load(self, tmp_path):
        bm25 = BM25(k1=1.2, b=0.8)
        bm25.index_document(1, "hello world python programming")
        bm25.index_document(2, "rust systems programming language")
        path = str(tmp_path / "bm25.json")
        bm25.save(path)

        loaded = BM25.load(path)
        assert loaded.doc_count == 2
        assert loaded.k1 == 1.2
        assert loaded.b == 0.8
        assert loaded.vocab_size == bm25.vocab_size

        # Queries should produce identical results
        q1 = bm25.encode_query("programming")
        q2 = loaded.encode_query("programming")
        assert q1 == q2

    def test_weights_are_positive(self):
        bm25 = BM25()
        sparse = bm25.index_document(1, "machine learning deep learning neural networks")
        for weight in sparse.values():
            assert weight > 0

    def test_idf_rewards_rarity(self):
        bm25 = BM25()
        bm25.index_document(1, "common rare unique")
        bm25.index_document(2, "common other stuff")
        bm25.index_document(3, "common another thing")
        # "rare" appears in 1/3 docs, "common" in 3/3
        query = bm25.encode_query("common rare")
        common_id = bm25._vocab.get("common")
        rare_id = bm25._vocab.get("rare")
        assert common_id is not None and rare_id is not None
        # rare term should have higher IDF weight
        assert query[rare_id] > query[common_id]

    def test_avg_dl_property(self):
        bm25 = BM25()
        bm25.index_document(1, "one two three")       # 3 tokens
        bm25.index_document(2, "four five six seven")  # 4 tokens
        assert bm25.avg_dl == pytest.approx(3.5)

    def test_sparse_keys_are_consistent(self):
        """Same term should always map to the same dimension index."""
        bm25 = BM25()
        s1 = bm25.index_document(1, "hello world")
        s2 = bm25.index_document(2, "hello python")
        # "hello" should map to the same key in both
        hello_id = bm25._vocab["hello"]
        assert hello_id in s1
        assert hello_id in s2
