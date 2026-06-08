"""
utils/embeddings.py — Text Embedding Utilities

Handles loading pre-trained models and generating semantic embeddings
for text chunks using sentence-transformers.
"""

import numpy as np
import logging
from typing import List, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# ============================================================================
# MODEL LOADING
# ============================================================================

@lru_cache(maxsize=1)
def load_model(model_name: str = "all-MiniLM-L6-v2") -> "SentenceTransformer":
    """
    Load a pre-trained sentence transformer model with caching.

    Uses LRU cache to avoid reloading the model on repeated calls.
    First call downloads and caches the model (~300MB for default).
    Subsequent calls return the cached model instantly from memory.

    Args:
        model_name: HuggingFace model identifier string.
                   Default: "all-MiniLM-L6-v2" (384-dim, fast)

    Returns:
        Loaded SentenceTransformer model object ready for inference

    Raises:
        ImportError: If sentence-transformers package not installed
        ValueError: If model_name is invalid

    Example:
        >>> model = load_model()  # First: ~5s, Second: instant
        >>> embeddings = model.encode(["Hello world"])
        >>> embeddings.shape
        (1, 384)
    """
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )

    try:
        logger.info(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info(f"Model loaded successfully: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise ValueError(f"Could not load model '{model_name}': {e}") from e


# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

def embed_chunks(
    chunks: List[List[str]],
    model: Optional["SentenceTransformer"] = None
) -> np.ndarray:
    """
    Encode text chunks into dense semantic embeddings.

    Converts each chunk (list of tokens) into a vector representation
    using a pre-trained sentence transformer model.

    Args:
        chunks: List of token lists to encode.
                Example: [['hello', 'world'], ['foo', 'bar']]

        model: Pre-loaded SentenceTransformer model. If None, loads default.
               Reusing a model improves performance.

    Returns:
        np.ndarray of shape (n_chunks, embedding_dim) with embeddings.
        For default model: shape = (n_chunks, 384)
        All vectors are L2-normalized (unit norm).

    Raises:
        ValueError: If chunks is empty
        RuntimeError: If model inference fails

    Example:
        >>> chunks = [['quick', 'brown', 'fox'], ['slow', 'red', 'dog']]
        >>> embeddings = embed_chunks(chunks)
        >>> embeddings.shape
        (2, 384)

    Notes:
        - Embeddings are L2-normalized for cosine similarity
        - Automatic batch processing
        - Uses GPU if available
    """
    if not chunks or all(len(c) == 0 for c in chunks):
        raise ValueError("chunks cannot be empty")

    if model is None:
        model = load_model()

    texts = [" ".join(chunk) for chunk in chunks]

    try:
        logger.debug(f"Embedding {len(texts)} chunks...")
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        logger.debug(f"Generated embeddings: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise RuntimeError(f"Failed to generate embeddings: {e}") from e


def embed_single_text(
    text: str,
    model: Optional["SentenceTransformer"] = None
) -> np.ndarray:
    """
    Encode a single text string into an embedding vector.

    Convenience wrapper for single texts.

    Args:
        text: Text string to encode
        model: Pre-loaded model or None to load default

    Returns:
        np.ndarray of shape (embedding_dim,)

    Example:
        >>> embedding = embed_single_text("The quick brown fox")
        >>> embedding.shape
        (384,)
    """
    chunk = text.split()
    embeddings = embed_chunks([chunk], model)
    return embeddings[0]


# ============================================================================
# SIMILARITY COMPUTATION
# ============================================================================

def cosine_similarity_matrix(
    A: np.ndarray,
    B: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity matrix between two embedding sets.

    Args:
        A: First embedding array of shape (n, embedding_dim)
        B: Second embedding array of shape (m, embedding_dim)

    Returns:
        Similarity matrix of shape (n, m)

    Example:
        >>> A = np.random.randn(3, 384)
        >>> B = np.random.randn(2, 384)
        >>> sim = cosine_similarity_matrix(A, B)
        >>> sim.shape
        (3, 2)
    """
    return np.dot(A, B.T)


def mean_similarity(
    A: np.ndarray,
    B: np.ndarray
) -> float:
    """
    Compute mean cosine similarity between two embedding sets.

    Useful for comparing overall similarity between two documents
    or author pairs. Takes the average of all pairwise similarities.

    Args:
        A: First embedding array of shape (n_chunks, embedding_dim)
        B: Second embedding array of shape (m_chunks, embedding_dim)

    Returns:
        Mean cosine similarity as float between -1 and 1
        (typically 0 to 1 for normalized embeddings)

    Example:
        >>> emb_a = embed_chunks(chunks_a)  # shape (5, 384)
        >>> emb_b = embed_chunks(chunks_b)  # shape (3, 384)
        >>> avg_sim = mean_similarity(emb_a, emb_b)
        >>> print(f"Average similarity: {avg_sim:.3f}")
        Average similarity: 0.654

    Notes:
        - Computes full similarity matrix: shape (n, m)
        - Returns mean of all values in matrix
        - Works with L2-normalized embeddings (cosine similarity)

    See Also:
        - cosine_similarity_matrix(): For pairwise comparisons
    """
    sim_matrix = cosine_similarity_matrix(A, B)
    return float(np.mean(sim_matrix))


def compute_similarity_matrix(
    embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise cosine similarities between all embeddings.

    Args:
        embeddings: Array of shape (n_embeddings, embedding_dim)

    Returns:
        Array of shape (n_embeddings, n_embeddings) with similarities

    Example:
        >>> embeddings = embed_chunks(chunks)
        >>> similarities = compute_similarity_matrix(embeddings)
        >>> similarities[0, 0]
        1.0
    """
    return np.dot(embeddings, embeddings.T)    texts = [" ".join(chunk) for chunk in chunks]
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
