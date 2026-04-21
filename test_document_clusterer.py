import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from document_clusterer.document_clusterer import DocumentClusterer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_clusterer(**overrides):
    """Return a DocumentClusterer with sensible test defaults."""
    defaults = dict(
        ngram=(1, 2),
        n_components=5,
        n_iter=2,
        dist_threshold=1.5,
        linkage="ward",
        input_type="content",
        random_state=42,
        llm_model="llama3.2:3b",
        n_llm_samples=3,
    )
    defaults.update(overrides)
    return DocumentClusterer(**defaults)


SAMPLE_DOCS = {
    "Software License A": "This agreement governs the use of software provided by the licensor.",
    "Software License B": "The licensee agrees to the terms of this software license agreement.",
    "Distribution Agreement A": "This contract outlines the terms for distributing goods to retailers.",
    "Distribution Agreement B": "The distributor agrees to sell products within the agreed territory.",
    "Employment Contract A": "This employment agreement sets out the terms of the working relationship.",
    "Employment Contract B": "The employee agrees to perform duties as directed by the employer.",
}


# ---------------------------------------------------------------------------
# Unit Tests — tfidf_vectorizer
# ---------------------------------------------------------------------------

class TestTfidfVectorizer:
    def test_returns_correct_number_of_rows(self):
        c = make_clusterer()
        tdm = c.tfidf_vectorizer(SAMPLE_DOCS)
        assert tdm.shape[0] == len(SAMPLE_DOCS)

    def test_sets_tfidf_attribute(self):
        c = make_clusterer()
        c.tfidf_vectorizer(SAMPLE_DOCS)
        assert c.tfidf_ is not None

    def test_returns_sparse_matrix(self):
        from scipy.sparse import issparse
        c = make_clusterer()
        tdm = c.tfidf_vectorizer(SAMPLE_DOCS)
        assert issparse(tdm)


# ---------------------------------------------------------------------------
# Unit Tests — dim_reduction
# ---------------------------------------------------------------------------

class TestDimReduction:
    def test_output_shape_matches_n_components(self):
        c = make_clusterer(n_components=5)
        tdm = c.tfidf_vectorizer(SAMPLE_DOCS)
        fdm = c.dim_reduction(tdm)
        assert fdm.shape[0] == len(SAMPLE_DOCS)
        assert fdm.shape[1] <= 5

    def test_n_components_clamped_below_n_features(self):
        # Request more components than there are features — should not crash
        c = make_clusterer(n_components=100_000)
        tdm = c.tfidf_vectorizer(SAMPLE_DOCS)
        fdm = c.dim_reduction(tdm)
        assert fdm.shape[1] < tdm.shape[1]

    def test_sets_svd_attribute(self):
        c = make_clusterer()
        tdm = c.tfidf_vectorizer(SAMPLE_DOCS)
        c.dim_reduction(tdm)
        assert c.svd_ is not None

    def test_returns_dense_array(self):
        c = make_clusterer()
        tdm = c.tfidf_vectorizer(SAMPLE_DOCS)
        fdm = c.dim_reduction(tdm)
        assert isinstance(fdm, np.ndarray)


# ---------------------------------------------------------------------------
# Unit Tests — clusterer
# ---------------------------------------------------------------------------

class TestClusterer:
    def test_label_count_matches_document_count(self):
        c = make_clusterer()
        tdm = c.tfidf_vectorizer(SAMPLE_DOCS)
        fdm = c.dim_reduction(tdm)
        labels = c.clusterer(fdm)
        assert len(labels) == len(SAMPLE_DOCS)

    def test_labels_are_integers(self):
        c = make_clusterer()
        tdm = c.tfidf_vectorizer(SAMPLE_DOCS)
        fdm = c.dim_reduction(tdm)
        labels = c.clusterer(fdm)
        assert all(isinstance(int(l), int) for l in labels)

    def test_higher_threshold_produces_fewer_clusters(self):
        c_tight = make_clusterer(dist_threshold=0.5)
        c_loose = make_clusterer(dist_threshold=5.0)

        tdm_tight = c_tight.tfidf_vectorizer(SAMPLE_DOCS)
        fdm_tight = c_tight.dim_reduction(tdm_tight)
        labels_tight = c_tight.clusterer(fdm_tight)

        tdm_loose = c_loose.tfidf_vectorizer(SAMPLE_DOCS)
        fdm_loose = c_loose.dim_reduction(tdm_loose)
        labels_loose = c_loose.clusterer(fdm_loose)

        assert len(set(labels_loose)) <= len(set(labels_tight))


# ---------------------------------------------------------------------------
# Unit Tests — fit
# ---------------------------------------------------------------------------

class TestFit:
    def test_returns_dict_with_same_keys(self):
        c = make_clusterer()
        results = c.fit(SAMPLE_DOCS)
        assert set(results.keys()) == set(SAMPLE_DOCS.keys())

    def test_sets_labels_attribute(self):
        c = make_clusterer()
        c.fit(SAMPLE_DOCS)
        assert c.labels_ is not None

    def test_sets_doc_ids_attribute(self):
        c = make_clusterer()
        c.fit(SAMPLE_DOCS)
        assert c.doc_ids_ == list(SAMPLE_DOCS.keys())

    def test_stable_across_runs_with_same_random_state(self):
        c1 = make_clusterer(random_state=42)
        c2 = make_clusterer(random_state=42)
        r1 = c1.fit(SAMPLE_DOCS)
        r2 = c2.fit(SAMPLE_DOCS)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Unit Tests — llm_cluster_label
# ---------------------------------------------------------------------------

class TestLlmClusterLabel:
    def test_raises_if_fit_not_called(self):
        c = make_clusterer()
        with pytest.raises(ValueError, match="fit\\(\\)"):
            c.llm_cluster_label()

    @patch("document_clusterer.document_clusterer.ollama.chat")
    def test_returns_dict_with_cluster_ids_as_keys(self, mock_chat):
        mock_chat.return_value = MagicMock(message=MagicMock(content="Software Agreements"))
        c = make_clusterer()
        c.fit(SAMPLE_DOCS)
        labels = c.llm_cluster_label()
        assert all(isinstance(k, int) for k in labels.keys())

    @patch("document_clusterer.document_clusterer.ollama.chat")
    def test_calls_ollama_once_per_cluster(self, mock_chat):
        mock_chat.return_value = MagicMock(message=MagicMock(content="Some Label"))
        c = make_clusterer()
        c.fit(SAMPLE_DOCS)
        n_clusters = len(set(c.labels_))
        c.llm_cluster_label()
        assert mock_chat.call_count == n_clusters

    @patch("document_clusterer.document_clusterer.ollama.chat")
    def test_warns_when_cluster_count_exceeds_30(self, mock_chat):
        mock_chat.return_value = MagicMock(message=MagicMock(content="Label"))
        c = make_clusterer(dist_threshold=0.01)  # very tight → many clusters
        # Build a larger synthetic corpus to force 30+ clusters
        big_docs = {f"Doc {i}": f"Unique content number {i} with distinct terms xyz{i}" for i in range(60)}
        c.fit(big_docs)
        if len(set(c.labels_)) > 30:
            with pytest.warns(UserWarning, match="API calls"):
                c.llm_cluster_label()


# ---------------------------------------------------------------------------
# Unit Tests — error_detection
# ---------------------------------------------------------------------------

class TestErrorDetection:
    @patch("document_clusterer.document_clusterer.ollama.chat")
    def test_returns_expected_keys(self, mock_chat):
        mock_chat.return_value = MagicMock(message=MagicMock(content="YES, all titles match."))
        c = make_clusterer()
        c.fit(SAMPLE_DOCS)
        generated_labels = {int(l): "Test Label" for l in set(c.labels_)}
        cluster_id = list(generated_labels.keys())[0]
        result = c.error_detection(cluster_id, generated_labels)
        assert set(result.keys()) == {"cluster_id", "label", "verdict"}

    @patch("document_clusterer.document_clusterer.ollama.chat")
    def test_cluster_id_in_result_matches_input(self, mock_chat):
        mock_chat.return_value = MagicMock(message=MagicMock(content="NO, mixed types."))
        c = make_clusterer()
        c.fit(SAMPLE_DOCS)
        generated_labels = {int(l): "Some Label" for l in set(c.labels_)}
        cluster_id = list(generated_labels.keys())[0]
        result = c.error_detection(cluster_id, generated_labels)
        assert result["cluster_id"] == cluster_id

    @patch("document_clusterer.document_clusterer.ollama.chat")
    def test_raises_for_invalid_cluster_id(self, mock_chat):
        c = make_clusterer()
        c.fit(SAMPLE_DOCS)
        with pytest.raises((KeyError, ValueError)):
            c.error_detection(cluster_id=9999, generated_labels={0: "Label"})

    @patch("document_clusterer.document_clusterer.ollama.chat")
    def test_only_calls_ollama_once(self, mock_chat):
        mock_chat.return_value = MagicMock(message=MagicMock(content="YES"))
        c = make_clusterer()
        c.fit(SAMPLE_DOCS)
        generated_labels = {int(l): "Label" for l in set(c.labels_)}
        cluster_id = list(generated_labels.keys())[0]
        c.error_detection(cluster_id, generated_labels)
        assert mock_chat.call_count == 1


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_document(self):
        c = make_clusterer()
        single = {"Only Doc": "This is the only document in the collection."}
        result = c.fit(single)
        assert len(result) == 1

    def test_identical_documents_form_one_cluster(self):
        c = make_clusterer(dist_threshold=5.0)
        identical = {f"Doc {i}": "Exactly the same content in every document." for i in range(5)}
        result = c.fit(identical)
        assert len(set(result.values())) == 1

    def test_n_llm_samples_larger_than_cluster_size(self):
        # n_llm_samples=100 but cluster may have fewer docs — should not crash
        with patch("document_clusterer.document_clusterer.ollama.chat") as mock_chat:
            mock_chat.return_value = MagicMock(message=MagicMock(content="Label"))
            c = make_clusterer(n_llm_samples=100)
            c.fit(SAMPLE_DOCS)
            labels = c.llm_cluster_label()
            assert isinstance(labels, dict)

    def test_empty_string_documents(self):
        c = make_clusterer()
        empty_docs = {"Doc A": "", "Doc B": "", "Doc C": "Some real content here"}
        # Should not raise — may produce degenerate clusters but must not crash
        try:
            c.fit(empty_docs)
        except Exception as e:
            pytest.fail(f"fit() raised unexpectedly with empty docs: {e}")
