import streamlit as st
from utils.craig import FUNCTION_WORDS

def render_sidebar(analysis_mode):
    st.sidebar.header("Parameters")

    # === Analysis modules ===
    st.sidebar.header("Analysis modules")

    sidebar_analysis_flags = {
        "burrows_delta": st.sidebar.checkbox(
            "Burrows' Delta", True,
            help="Measures overall stylistic distance using frequent word distributions."
        ),
        "craig_markers": st.sidebar.checkbox(
            "Craig marker analysis", True,
            help="Identifies words with asymmetric distribution across chunks (Craig method)."
        ),
        "feature_analysis": st.sidebar.checkbox(
            "Global stylistic features", True,
            help="Compares texts using aggregated stylistic features."
        ),
        "pca": st.sidebar.checkbox(
            "PCA visualization", True,
            help="Visualizes chunk-level stylistic structure."
        ),
        "topic_modeling": st.sidebar.checkbox(
            "Topic modelling", False,
            help="Exploratory semantic analysis."
        ),
    }

    # === Text preprocessing ===
    st.sidebar.header("Text preprocessing")
    tokenizer_choice = st.sidebar.selectbox(
        "Tokenizer",
        ["nltk", "simple", "regex", "unicode", "char_ngrams"],
        index=0,
    )

    chunk_size = st.sidebar.number_input(
        "Chunk size (words)", 200, 5000, 2000, 100
    )

    # === Craig parameters ===
    st.sidebar.header("Craig marker parameters")
    top_k = st.sidebar.slider("Top-K marker words", 20, 1000, 100, 10)

    use_min_coef = st.sidebar.checkbox("Enforce minimum coefficient threshold", False)
    min_coef = st.sidebar.slider(
        "Minimum coefficient", 0.0, 2.0, 0.0, 0.01,
        disabled=not use_min_coef
    )

    pca_marker_limit = st.sidebar.slider("PCA marker limit", 50, 2000, 500, 50)

    # === Lexical filtering ===
    st.sidebar.header("Lexical filtering")
    use_exclusion_words = st.sidebar.checkbox(
        "Exclude structural / genre-specific words", False
    )

    exclusion_words_text = st.sidebar.text_area(
        "Words to exclude",
        height=100,
        disabled=not use_exclusion_words
    )

    # === Stability filter ===
    st.sidebar.subheader("Marker stability filter")
    use_stability_filter = st.sidebar.checkbox("Enable stability filtering", True)

    variance_threshold = st.sidebar.slider("Variance threshold", 0.00001, 0.01, 0.0005)
    min_chunk_presence = st.sidebar.slider("Minimum chunk presence", 0.05, 1.0, 0.30)
    clump_ratio_threshold = st.sidebar.slider("Clump ratio threshold", 0.0, 1.0, 0.65)

    # === Topic modelling ===
    st.sidebar.header("Topic modelling (semantic)")

    topic_usage = st.sidebar.radio(
        "Use topics to:",
        [
            "Explore topics only",
            "Filter chunks for analysis",
            "Control for topic (authorship within topic)",
        ],
        key = "topic_usage"
    )

    topic_for_authorship = None
    # === Topic modelling parameters ===
    n_topics = st.sidebar.slider(
        "Number of topics",
        min_value=2,
        max_value=20,
        value=8,
        help="Number of latent topics to extract (limited by chunk count)."
    )
    if (
            sidebar_analysis_flags.get("topic_modeling", False)
            and topic_usage == "Control for topic (authorship within topic)"
    ):
        topic_for_authorship = st.sidebar.selectbox(
            "Select topic for authorship test",
            options=list(range(n_topics)),
            help=(
                "Authorship attribution will be re-run "
                "using only chunks belonging to this topic."
            ),
        )



    st.sidebar.markdown("**Train topic model on:**")
    train_A = st.sidebar.checkbox("Author A", True)
    train_B = st.sidebar.checkbox("Author B", True)
    train_U = st.sidebar.checkbox("Unknown", False)

    st.sidebar.markdown("**Apply topics to:**")
    apply_A = st.sidebar.checkbox("Author A ", True, key="apply_A")
    apply_B = st.sidebar.checkbox("Author B ", True, key="apply_B")
    apply_U = st.sidebar.checkbox("Unknown ", True, key="apply_U")

    use_custom_stopwords = st.sidebar.checkbox("Apply additional stopwords", False)
    custom_stopwords_text = st.sidebar.text_area(
        "Additional stopwords",
        height=100,
        disabled=not use_custom_stopwords
    )
    chunk_mode = st.sidebar.radio(
        "Chunk selection mode",
        [
            "all",
            "topic-filtered",
            "stable-only",
        ],
        index=0,
        help=(
            "Controls which chunks are used for analysis.\n\n"
            "• all — use all chunks\n"
            "• topic-filtered — keep only chunks matching selected topics\n"
            "• stable-only — keep chunks with stable stylistic profiles"
        )
    )
    # === Reference helpers ===
    st.sidebar.subheader("Craig marker analysis (reference)")
    with st.sidebar.expander("📘 Craig function-word list", expanded=False):
        st.code(", ".join(sorted(FUNCTION_WORDS)))
    st.sidebar.header("Chunk selection")



    return {
        "analysis_flags": sidebar_analysis_flags,
        "tokenizer": tokenizer_choice,
        "chunk_size": chunk_size,
        "top_k": top_k,
        "use_min_coef": use_min_coef,
        "min_coef": min_coef,
        "pca_marker_limit": pca_marker_limit,
        "use_exclusion_words": use_exclusion_words,
        "exclusion_words_text": exclusion_words_text,
        "use_stability_filter": use_stability_filter,
        "variance_threshold": variance_threshold,
        "chunk_mode": chunk_mode,
        "min_chunk_presence": min_chunk_presence,
        "clump_ratio_threshold": clump_ratio_threshold,
        "topic_usage": topic_usage,
        "topic_for_authorship": topic_for_authorship,
        "custom_stopwords_text": custom_stopwords_text,
        "n_topics": n_topics,
        "train_A": train_A,
        "train_B": train_B,
        "train_U": train_U,
        "apply_A": apply_A,
        "apply_B": apply_B,
        "apply_U": apply_U,
    }
