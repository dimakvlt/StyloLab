from utils.processing import (
    clean_and_tokenize,
    tokenize_with_choice,
    chunk_words,
)
import numpy as np
from collections import Counter
from utils.craig import (
    compute_craig_coefficients,
    top_k_markers,
    chunk_marker_proportions,
    marker_stability_filter,
)
from utils.craig import (
    compute_craig_coefficients,
    top_k_markers,
    chunk_marker_proportions,
    marker_stability_filter,
)

from utils.pca_utils import pca_on_chunk_vectors
from utils.topic_model import (
    train_topic_model,
    apply_topic_model,
    get_topic_keywords,
)

import features
from utils.delta import burrows_delta

def run_analysis(
    textA, textB, textU,
    tokenizer_choice, chunk_size,
    top_k, min_coef, use_min_coef,
    use_topics, n_topics,
    train_A, train_B, train_U,
    apply_A, apply_B, apply_U,
    custom_stopwords_text,
    exclusion_words_frozen,
    chunk_mode, selected_topics, min_conf,
    topic_usage,
    use_stability_filter,
    variance_threshold, min_chunk_presence, clump_ratio_threshold,
    pca_marker_limit,
    analysis_flags_frozen,
):
    exclusion_words = set(exclusion_words_frozen)
    analysis_flags = dict(analysis_flags_frozen)

    # ------------------------------------------------------------
    # TOKENIZE + CHUNK
    # ------------------------------------------------------------
    try:
        tokensA = tokenize_with_choice(textA, tokenizer_choice)
        tokensB = tokenize_with_choice(textB, tokenizer_choice)
        tokensU = tokenize_with_choice(textU, tokenizer_choice)
    except Exception:
        tokensA = clean_and_tokenize(textA)
        tokensB = clean_and_tokenize(textB)
        tokensU = clean_and_tokenize(textU)
    def apply_exclusion_tokens(tokens, exclude):
        if not exclude:
            return tokens
        return [w for w in tokens if w not in exclude]

    tokensA = apply_exclusion_tokens(tokensA, exclusion_words)
    tokensB = apply_exclusion_tokens(tokensB, exclusion_words)
    tokensU = apply_exclusion_tokens(tokensU, exclusion_words)

    chunksA_full = chunk_words(tokensA, chunk_size)
    chunksB_full = chunk_words(tokensB, chunk_size)
    chunksU_full = chunk_words(tokensU, chunk_size)

    chunksA_sel = chunksA_full
    chunksB_sel = chunksB_full
    chunksU_sel = chunksU_full
    def apply_exclusion(chunks, exclude):
        if not exclude:
            return chunks
        return [
            [w for w in chunk if w not in exclude]
            for chunk in chunks
        ]

    chunksA_full = apply_exclusion(chunksA_full, exclusion_words)
    chunksB_full = apply_exclusion(chunksB_full, exclusion_words)
    chunksU_full = apply_exclusion(chunksU_full, exclusion_words)

    # ------------------------------------------------------------
    # TOPIC MODELLING
    # ------------------------------------------------------------
    topicsA = topicsB = topicsU = None
    topic_keywords = {}
    custom_stopwords = set()

    if analysis_flags.get("topic_modeling", False):
        training_chunks = []
        if train_A: training_chunks += chunksA_full
        if train_B: training_chunks += chunksB_full
        if train_U: training_chunks += chunksU_full

        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        custom_stopwords = {
            lemmatizer.lemmatize(w.strip().lower())
            for w in custom_stopwords_text.replace(",", "\n").split()
            if w.strip()
        }

        if len(training_chunks) >= 2:
            effective_n_topics = min(n_topics, len(training_chunks))

            topic_model = train_topic_model(
                training_chunks,
                n_topics=effective_n_topics,
                stop_words=custom_stopwords if custom_stopwords else None
            )

            topic_keywords = get_topic_keywords(topic_model)
            topicsA = apply_topic_model(topic_model, chunksA_full) if apply_A else None
            topicsB = apply_topic_model(topic_model, chunksB_full) if apply_B else None
            topicsU = apply_topic_model(topic_model, chunksU_full) if apply_U else None
        else:
            # explicitly keep everything None / empty
            topic_keywords = {}
            topicsA = topicsB = topicsU = None
    # ------------------------------------------------------------
    # TOPIC-BASED CHUNK FILTERING
    # ------------------------------------------------------------
    if use_topics and chunk_mode == "Topic-filtered chunks":

        if not selected_topics:
            selected_topics = sorted(topic_keywords.keys())

        if topicsA:
            chunksA_sel = [
                c for c, ann in zip(chunksA_full, topicsA)
                if ann["topic"] in selected_topics and ann["confidence"] >= min_conf
            ]
        if topicsB:
            chunksB_sel = [
                c for c, ann in zip(chunksB_full, topicsB)
                if ann["topic"] in selected_topics and ann["confidence"] >= min_conf
            ]
        if topicsU:
            chunksU_sel = [
                c for c, ann in zip(chunksU_full, topicsU)
                if ann["topic"] in selected_topics and ann["confidence"] >= min_conf
            ]

    # ------------------------------------------------------------
    # CRAIG MARKERS
    # ------------------------------------------------------------
    coef_A_vs_B = coef_B_vs_A = {}
    markersA = markersB = set()

    if analysis_flags.get("craig_markers", False):

        coef_A_vs_B = compute_craig_coefficients(chunksA_sel, chunksB_sel)
        coef_B_vs_A = compute_craig_coefficients(chunksB_sel, chunksA_sel)

        if use_min_coef:
            markersA = {w for w, v in top_k_markers(coef_A_vs_B, top_k) if v >= min_coef}
            markersB = {w for w, v in top_k_markers(coef_B_vs_A, top_k) if v >= min_coef}
        else:
            markersA = {w for w, _ in top_k_markers(coef_A_vs_B, top_k)}
            markersB = {w for w, _ in top_k_markers(coef_B_vs_A, top_k)}
    filtered_markers = []
    filtered_markersA = []
    filtered_markersB = []
    dropped = []

    # ------------------------------------------------------------
    # STABILITY FILTER
    # ------------------------------------------------------------
    propsA_A = propsA_B = propsB_A = propsB_B = propsU_A = propsU_B = None

    filtered_markers = []
    filtered_markersA = []
    filtered_markersB = []
    dropped = []

    if analysis_flags.get("craig_markers", False) and markersA and markersB:

        all_markers = sorted(markersA | markersB)
        all_chunks = chunksA_sel + chunksB_sel + chunksU_sel

        marker_matrix = np.zeros((len(all_chunks), len(all_markers)))
        for i, chunk in enumerate(all_chunks):
            cnt = Counter(chunk)
            total = max(1, len(chunk))
            for j, m in enumerate(all_markers):
                marker_matrix[i, j] = cnt[m] / total

        filtered_markers = all_markers

        if use_stability_filter:
            marker_matrix, filtered_markers, dropped = marker_stability_filter(
                marker_matrix,
                all_markers,
                variance_threshold,
                min_chunk_presence,
                clump_ratio_threshold,
            )

        filtered_markersA = [m for m in filtered_markers if m in markersA]
        filtered_markersB = [m for m in filtered_markers if m in markersB]



    propsA_A, _ = chunk_marker_proportions(chunksA_sel, filtered_markersA)
    propsA_B, _ = chunk_marker_proportions(chunksA_sel, filtered_markersB)
    propsB_A, _ = chunk_marker_proportions(chunksB_sel, filtered_markersA)
    propsB_B, _ = chunk_marker_proportions(chunksB_sel, filtered_markersB)
    propsU_A, _ = chunk_marker_proportions(chunksU_sel, filtered_markersA)
    propsU_B, _ = chunk_marker_proportions(chunksU_sel, filtered_markersB)

    # ------------------------------------------------------------
    # FEATURES + DELTA + PCA
    # ------------------------------------------------------------
    feature_compare = {}

    if analysis_flags.get("feature_analysis", False):
        feature_compare = features.compare_texts(textA, textB, textU)

    deltaA = deltaB = None

    if analysis_flags.get("burrows_delta", False):
        deltaA, deltaB, _ = burrows_delta(tokensA, tokensB, tokensU)

    pca_result = None
    if (
            analysis_flags.get("pca", False)
            and analysis_flags.get("craig_markers", False)
            and (filtered_markersA or filtered_markersB)

    ):
        pca_result = pca_on_chunk_vectors(
            chunksA_sel, chunksB_sel, chunksU_sel,
            filtered_markersA, filtered_markersB,
            pca_marker_limit
        )

    # ------------------------------------------------------------
    # RETURN EVERYTHING
    # ------------------------------------------------------------
    return {
        "tokensA": tokensA,
        "tokensB": tokensB,
        "tokensU": tokensU,
        "chunksA_full": chunksA_full,
        "chunksB_full": chunksB_full,
        "chunksU_full": chunksU_full,
        "chunksA_sel": chunksA_sel,
        "chunksB_sel": chunksB_sel,
        "chunksU_sel": chunksU_sel,
        "topicsA": topicsA,
        "topicsB": topicsB,
        "topicsU": topicsU,
        "topic_keywords": topic_keywords,
        "custom_stopwords": custom_stopwords,
        "propsA_A": propsA_A,
        "propsA_B": propsA_B,
        "propsB_A": propsB_A,
        "propsB_B": propsB_B,
        "propsU_A": propsU_A,
        "propsU_B": propsU_B,
        "deltaA": deltaA,
        "deltaB": deltaB,
        "pca_result": pca_result,
        "feature_compare": feature_compare,
        "coef_A_vs_B": coef_A_vs_B,
        "coef_B_vs_A": coef_B_vs_A,
        "dropped_markers": dropped,
    }

