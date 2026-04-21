# TF-IDF Document Clustering and LLM Analysis

This project uses a TF-IDF vectoriser and an alogmorative clusterer to cluster legal text. A call is then made to the Ollama LLM to label the cluster based on the text inside it.

# Installation

The dependecnies are listed in ```py requirements.txt```. The can be installed with ```py pip install -r requirements.txt```

# Example Usage
```py
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
```