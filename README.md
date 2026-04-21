# TF-IDF Document Clustering and LLM Analysis

This project blah blah blah

# Installation

The dependecnies are listed in requirements.txt. The can be installed with pip install -r requirements.txt

# Example Usage

from document_clusterer import DocumentClusterer

clusterer = DocumentClusterer(
    ngram=(1, 3),
    n_components=100,
    n_iter=5,
    dist_threshold=1.5,
    linkage="ward",
    input_type="content",
    random_state=42,
    llm_model='llama3.2:3b',
    n_llm_samples=5
)

results = clusterer.fit(documents)
labels = clusterer.llm_cluster_label()