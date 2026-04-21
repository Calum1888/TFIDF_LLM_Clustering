from setuptools import setup, find_packages

setup(
    name="document_clusterer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn==1.8.0",
        "scipy==1.17.1",
        "numpy==2.4.4",
        "ollama==0.6.1",
        "tqdm==4.67.3",
    ],
)