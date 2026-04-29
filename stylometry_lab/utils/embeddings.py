# utils/embeddings.py

import numpy as np
from typing import List
from functools import lru_cache

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# -----------------------------
# MODEL LOADING (cached)
# -----------------------------
@lru_cache(maxsize=1)
def load_model(model_name: str = "all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )
    return SentenceTransformer(model_name)


# -----------------------------
# EMBEDDING
# -----------------------------
def embed_chunks(chunks: List[List[str]], model=None) -> np.ndarray:
    """
    chunks: list of token lists
    returns: np.ndarray [n_chunks, dim]
    """
    if model is None:
        model = load_model()

    texts = [" ".join(chunk) for chunk in chunks]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True  # IMPORTANT for cosine similarity
    )
    return embeddings


# -----------------------------
# SIMILARITY
# -----------------------------
def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    A: [n, d]
    B: [m, d]
    returns: [n, m]
    """
    return np.dot(A, B.T)


def mean_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """
    Mean cosine similarity between two embedding sets
    """
    sim_matrix = cosine_similarity_matrix(A, B)
    return float(np.mean(sim_matrix))
