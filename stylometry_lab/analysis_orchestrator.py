"""
analysis_orchestrator.py - Analysis Orchestration and Execution

Handles the main analysis pipeline, caching, and orchestration of all
analysis components (CRAIG, Delta, Embeddings, Features, Topic Modeling).
"""

from typing import Dict, Any, Optional, Tuple, List, Set
import hashlib
import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter

from utils.processing import (
    tokenize_with_choice, clean_and_tokenize, chunk_words
)
from utils.craig import (
    compute_segment_presence, compute_craig_coefficients,
    top_k_markers, chunk_marker_proportions, chi2_for_word,
    marker_stability_filter, FUNCTION_WORDS
)
from utils.pca_utils import pca_on_chunk_vectors
from utils.plots import plot_scatter_positions, plot_pca_chunk_scatter
import features
from utils.chunk_selection import select_chunks, selection_summary
from utils.topic_model import (
    train_topic_model, apply_topic_model, get_topic_keywords,
    filter_chunks_by_topic, summarize_topics_per_text
)
from utils.embeddings import embed_chunks, load_model, mean_similarity
from utils.embeddings_pipeline import (
    run_embedding_pipeline, compute_centroids, compute_distances
)
from utils.delta import burrows_delta
from analysis.pipeline import run_analysis
from config import log_info, log_error, log_debug


def prepare_tokens_and_chunks(
    text: str,
    tokenizer_choice: str,
    chunk_size: int
) -> Tuple[List[str], List[List[str]]]:
    """
    Tokenize and chunk text.
    
    Attempts tokenization with specified method, falls back to simple method
    if specified tokenizer fails.
    
    Args:
        text: Raw text to process
        tokenizer_choice: Tokenizer name ('nltk', 'spacy', 'regex', 'whitespace')
        chunk_size: Number of tokens per chunk
    
    Returns:
        Tuple of (tokens_list, chunks_list)
        - tokens_list: Complete token list
        - chunks_list: List of chunks (each chunk is list of tokens)
    
    Raises:
        ValueError: If text is too short for chunking
    
    Example:
        >>> tokens, chunks = prepare_tokens_and_chunks(text, "nltk", 2000)
    """
    log_info(f"Preparing tokens with {tokenizer_choice} tokenizer")
    
    try:
        tokens = tokenize_with_choice(text, tokenizer_choice)
    except Exception as e:
        log_error(f"Tokenizer {tokenizer_choice} failed: {e}, using fallback")
        tokens = clean_and_tokenize(text)
    
    if not tokens:
        raise ValueError("Text produced no tokens after tokenization")
    
    chunks = chunk_words(tokens, chunk_size)
    
    if len(chunks) < 2:
        raise ValueError(f"Text too short for chunking (only {len(chunks)} chunk)")
    
    log_info(f"Created {len(chunks)} chunks from {len(tokens)} tokens")
    return tokens, chunks


def run_complete_analysis(
    textA: str,
    textB: str,
    textU: str,
    tokenizer_choice: str,
    chunk_size: int,
    top_k: int,
    min_coef: float,
    use_min_coef: bool,
    use_topics: bool,
    n_topics: int,
    train_A: bool,
    train_B: bool,
    train_U: bool,
    apply_A: bool,
    apply_B: bool,
    apply_U: bool,
    custom_stopwords_text: str,
    exclusion_words: Tuple[str, ...],
    chunk_mode: str,
    selected_topics: List[int],
    min_conf: float,
    topic_usage: str,
    use_stability_filter: bool,
    variance_threshold: float,
    min_chunk_presence: int,
    clump_ratio_threshold: float,
    pca_marker_limit: int,
    analysis_flags: Tuple[Tuple[str, bool], ...]
) -> Dict[str, Any]:
    """
    Execute complete comparative analysis pipeline.
    
    Orchestrates all analysis components including CRAIG, Delta, Embeddings,
    Features, and Topic Modeling. Returns comprehensive results dictionary.
    
    Args:
        textA: Author A reference text
        textB: Author B reference text
        textU: Unknown text to classify
        tokenizer_choice: Tokenizer method
        chunk_size: Tokens per chunk
        top_k: Number of top markers to keep
        min_coef: Minimum coefficient threshold
        use_min_coef: Whether to apply min_coef filter
        use_topics: Whether to use topic modeling
        n_topics: Number of topics to extract
        train_A: Whether to train topic model on A
        train_B: Whether to train topic model on B
        train_U: Whether to train topic model on U
        apply_A: Whether to apply topic model to A
        apply_B: Whether to apply topic model to B
        apply_U: Whether to apply topic model to U
        custom_stopwords_text: Custom stopwords string
        exclusion_words: Tuple of words to exclude from analysis
        chunk_mode: Chunk selection mode
        selected_topics: Topics to filter by
        min_conf: Minimum topic confidence
        topic_usage: How to use topics
        use_stability_filter: Apply stability filtering
        variance_threshold: Variance threshold for stability
        min_chunk_presence: Minimum chunk presence for stability
        clump_ratio_threshold: Clump ratio threshold for stability
        pca_marker_limit: Number of markers for PCA
        analysis_flags: Tuple of (flag_name, enabled) pairs
    
    Returns:
        Dictionary with complete analysis results:
        - tokens_* and chunks_* for each text
        - Analysis outputs from pipeline
        - Topic modeling results
        - Feature comparison results
    
    Example:
        >>> result = run_complete_analysis(...)
        >>> deltaA = result['deltaA']
        >>> verdict = result['feature_compare']['verdict']
    """
    log_info("Starting complete analysis pipeline")
    
    # Convert analysis_flags tuple back to dict
    analysis_flags_dict = dict(analysis_flags)
    
    # Call the main pipeline analysis
    result = run_analysis(
        textA, textB, textU,
        tokenizer_choice, chunk_size,
        top_k, min_coef, use_min_coef,
        use_topics, n_topics if use_topics else 0,
        train_A, train_B, train_U,
        apply_A, apply_B, apply_U,
        custom_stopwords_text,
        exclusion_words,
        chunk_mode, selected_topics, min_conf,
        topic_usage,
        use_stability_filter,
        variance_threshold, min_chunk_presence, clump_ratio_threshold,
        pca_marker_limit,
        analysis_flags
    )
    
    log_info("Analysis pipeline completed")
    return result


@st.cache_data(show_spinner=False)
def cached_run_analysis(*args) -> Dict[str, Any]:
    """
    Cached wrapper for complete analysis with Streamlit caching.
    
    Caches results based on all input parameters, avoiding redundant computation.
    Parameters must be hashable (use tuples for sequences).
    
    Args:
        *args: All arguments passed to run_complete_analysis
    
    Returns:
        Analysis results dictionary (cached)
    
    Note:
        This function signature must exactly match run_complete_analysis
        for proper caching. Do not add type hints to function definition.
    
    Example:
        >>> result = cached_run_analysis(
        ...     textA, textB, textU,
        ...     "nltk", 2000,
        ...     50, 0.0, False,
        ...     ...
        ... )
    """
    return run_complete_analysis(*args)


def prepare_analysis_arguments(
    textA: str,
    textB: str,
    textU: str,
    tokenizer_choice: str,
    chunk_size: int,
    top_k: int,
    min_coef: float,
    use_min_coef: bool,
    use_topics: bool,
    n_topics: int,
    train_A: bool,
    train_B: bool,
    train_U: bool,
    apply_A: bool,
    apply_B: bool,
    apply_U: bool,
    custom_stopwords_text: str,
    exclusion_words_tuple: Tuple[str, ...],
    chunk_mode: str,
    selected_topics: List[int],
    min_conf: float,
    topic_usage: str,
    use_stability_filter: bool,
    variance_threshold: float,
    min_chunk_presence: int,
    clump_ratio_threshold: float,
    pca_marker_limit: int,
    analysis_flags_dict: Dict[str, bool]
) -> Tuple:
    """
    Prepare and convert all analysis arguments to proper format for caching.
    
    Converts dictionaries to tuples (for hashability) and validates parameters.
    
    Args:
        All analysis parameters (see run_complete_analysis)
    
    Returns:
        Tuple of arguments ready for cached_run_analysis
    
    Example:
        >>> args = prepare_analysis_arguments(...)
        >>> result = cached_run_analysis(*args)
    """
    # Convert analysis_flags dict to hashable tuple of tuples
    analysis_flags_tuple = tuple(sorted(analysis_flags_dict.items()))
    
    # Create argument tuple
    args = (
        textA, textB, textU,
        tokenizer_choice, chunk_size,
        top_k, min_coef, use_min_coef,
        use_topics, n_topics,
        train_A, train_B, train_U,
        apply_A, apply_B, apply_U,
        custom_stopwords_text,
        exclusion_words_tuple,
        chunk_mode, tuple(selected_topics), min_conf,
        topic_usage,
        use_stability_filter,
        variance_threshold, min_chunk_presence, clump_ratio_threshold,
        pca_marker_limit,
        analysis_flags_tuple
    )
    
    return args


def perform_internal_delta_analysis(chunks: List[List[str]]) -> np.ndarray:
    """
    Compute internal Burrows Delta for single-text analysis.
    
    Measures stylistic deviation of each chunk from the text's average style.
    Useful for identifying internally anomalous or stylistically distinct sections.
    
    Args:
        chunks: List of token chunks
    
    Returns:
        NumPy array of delta values, one per chunk
    
    Example:
        >>> deltas = perform_internal_delta_analysis(chunks)
        >>> for i, delta in enumerate(deltas):
        ...     st.write(f"Chunk {i}: delta = {delta:.4f}")
    """
    log_debug(f"Computing internal delta for {len(chunks)} chunks")
    
    # Create vocabulary from top 300 words
    freq = Counter(w for c in chunks for w in c)
    vocab = [w for w, _ in freq.most_common(300)]
    
    def chunk_vector(c: List[str]) -> np.ndarray:
        """Convert chunk to frequency vector."""
        cnt = Counter(c)
        tot = max(1, len(c))
        return np.array([cnt[w] / tot for w in vocab])
    
    # Create document-term matrix
    X = np.vstack([chunk_vector(c) for c in chunks])
    
    # Compute mean and std
    mean = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1e-8  # Avoid division by zero
    
    # Compute delta for each chunk
    deltas = np.mean(np.abs((X - mean) / sd), axis=1)
    
    log_debug(f"Internal delta computed: mean={deltas.mean():.4f}")
    return deltas


def extract_analysis_results_to_locals(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract all analysis result variables for display.
    
    Unpacks the comprehensive result dictionary into individual variables
    for use in display functions.
    
    Args:
        result: Complete analysis result dictionary from run_analysis
    
    Returns:
        Dictionary with all extracted variables for locals() update
    
    Example:
        >>> result = cached_run_analysis(...)
        >>> locals().update(extract_analysis_results_to_locals(result))
        >>> st.write(f"Delta A: {deltaA}")
    """
    # Safely extract all possible result keys
    extracted = {}
    
    # Direct extractions with defaults
    for key in ['deltaA', 'deltaB', 'topicsA', 'topicsB', 'topicsU',
                'topic_keywords', 'pca_result', 'feature_compare',
                'chunksA_full', 'chunksB_full', 'chunksU_full',
                'chunksA_sel', 'chunksB_sel', 'chunksU_sel',
                'tokensA', 'tokensB', 'tokensU',
                'propsA_A', 'propsA_B', 'propsB_A', 'propsB_B',
                'propsU_A', 'propsU_B', 'coef_A_vs_B', 'coef_B_vs_A']:
        extracted[key] = result.get(key)
    
    return extracted


def validate_analysis_inputs(
    textA: str,
    textB: str,
    textU: str,
    analysis_mode: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate analysis inputs before running analysis.
    
    Args:
        textA: Author A text
        textB: Author B text
        textU: Unknown text
        analysis_mode: Either "Compare" or "Single"
    
    Returns:
        Tuple of (is_valid, error_message)
    
    Example:
        >>> is_valid, error = validate_analysis_inputs(
        ...     textA, textB, textU, "Compare"
        ... )
        >>> if not is_valid:
        ...     st.error(error)
    """
    if analysis_mode.startswith("Compare"):
        # Comparative analysis requires all three texts
        if not textA.strip():
            return False, "Author A text is required"
        if not textB.strip():
            return False, "Author B text is required"
        if not textU.strip():
            return False, "Unknown text is required"
    else:
        # Single-text analysis requires only Unknown
        if not textU.strip():
            return False, "Text for analysis is required"
    
    return True, None
