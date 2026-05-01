# app.py — Stylometry Lab (Modular, version A)
import streamlit as st
import logging
import pandas as pd
import numpy as np
from collections import Counter

from utils.processing import extract_text_file, clean_and_tokenize, chunk_words, tokenize_with_choice
from utils.craig import (
    compute_segment_presence, compute_craig_coefficients,
    top_k_markers, chunk_marker_proportions, chi2_for_word,
    marker_stability_filter, FUNCTION_WORDS
)
from utils.pca_utils import pca_on_chunk_vectors
from utils.plots import plot_scatter_positions, plot_pca_chunk_scatter
import features
from utils.report import (
    build_single_text_report,
    build_comparative_report
)
import tempfile
import os
import matplotlib.pyplot as plt
from utils.chunk_selection import select_chunks, selection_summary

from utils.topic_model import (
    train_topic_model,
    apply_topic_model,
    get_topic_keywords,
    filter_chunks_by_topic,
    summarize_topics_per_text,
)
from utils.embeddings import embed_chunks, load_model, mean_similarity
from utils.embedding_store import save_embeddings, load_embeddings
from utils.embeddings_pipeline import (
    run_embedding_pipeline,
    compute_centroids,
    compute_distances
)
import hashlib
from utils.delta import burrows_delta

def text_fingerprint(*texts):
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def ensure_session_key(key, default):
    if key not in st.session_state:
        st.session_state[key] = default


# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------
LOGFILE = "stylometry.log"
logger = logging.getLogger("stylometry_app")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    fh = logging.FileHandler(LOGFILE)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def log_warning(msg): logger.warning(msg)

# ------------------------------------------------------------

# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------
st.set_page_config(page_title="Stylometry Lab", layout="wide")
st.title("StyloLab")
st.markdown("""
<style>
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    font-weight: 600;
    padding: 0.6em 1.2em;
    border-radius: 8px;
}
div.stButton > button:hover {
    background-color: #45a049;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# SESSION STATE
# --------------------
if "params" not in st.session_state:
    st.session_state.params = {}

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

#  DEFAULTS
num_rows = None
rowsA = None
rowsB = None
rowsU = None
df_feat = None
fig_craig = None

# -------------------------
# ANALYSIS MODE
# --------------------
analysis_mode = st.sidebar.radio(
    "Analysis mode",
    [
        "Compare authors (A vs B vs Unknown)",
        "Single text (internal analysis)"
    ]
)


from ui.inputs import render_inputs, get_text

fileA, pastedA, fileB, pastedB, fileU, pastedU = render_inputs(analysis_mode)

textA = get_text(fileA, pastedA)
textB = get_text(fileB, pastedB)
textU = get_text(fileU, pastedU)


from ui.sidebar import render_sidebar

sidebar_params = render_sidebar(analysis_mode)
analysis_type = sidebar_params.get("analysis_type", "Classical")
use_embeddings = sidebar_params.get("use_embeddings", False)

save_embedding_name = sidebar_params.get("save_embedding_name")
load_embedding_name = sidebar_params.get("load_embedding_name")
save_embeddings_flag = sidebar_params.get("save_embeddings_flag", False)
sidebar_analysis_flags = sidebar_params["analysis_flags"]
analysis_flags = sidebar_analysis_flags
use_topics = analysis_flags["topic_modeling"]

tokenizer_choice = sidebar_params["tokenizer"]
chunk_size = sidebar_params["chunk_size"]
top_k = sidebar_params["top_k"]
use_min_coef = sidebar_params["use_min_coef"]
min_coef = sidebar_params["min_coef"]
pca_marker_limit = sidebar_params["pca_marker_limit"]
use_exclusion_words = sidebar_params["use_exclusion_words"]
exclusion_words_text = sidebar_params["exclusion_words_text"]
use_stability_filter = sidebar_params["use_stability_filter"]
variance_threshold = sidebar_params["variance_threshold"]
min_chunk_presence = sidebar_params["min_chunk_presence"]
clump_ratio_threshold = sidebar_params["clump_ratio_threshold"]
topic_usage = sidebar_params["topic_usage"]
custom_stopwords_text = sidebar_params["custom_stopwords_text"]
chunk_mode = sidebar_params["chunk_mode"]
n_topics = sidebar_params["n_topics"]

train_A = sidebar_params["train_A"]
train_B = sidebar_params["train_B"]
train_U = sidebar_params["train_U"]

apply_A = sidebar_params["apply_A"]
apply_B = sidebar_params["apply_B"]
apply_U = sidebar_params["apply_U"]

topic_for_authorship = sidebar_params.get("topic_for_authorship")



# ---------------------
# TOPIC FILTER CONTROLS
# ----------------------
selected_topics = []
min_conf = 0.0

if use_topics and topic_usage == "Filter chunks for analysis" and not st.session_state.analysis_done:
    st.sidebar.warning(
        "Topic-based filtering becomes available **after** you run the analysis once.\n\n"
        "1️⃣ Click **Run analysis** to train topics.\n"
        "2️⃣ Select topics and confidence.\n"
        "3️⃣ Re-run to apply filtering."
    )

st.sidebar.markdown("---")
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "params" not in st.session_state:
    st.session_state.params = {}

if st.button("Run analysis"):
    st.session_state.analysis_done = True
    st.session_state.text_fingerprint = text_fingerprint(textA, textB, textU)
    st.session_state.params = {
        "tokenizer": tokenizer_choice,
        "chunk_size": chunk_size,
        "top_k": top_k,
        "min_coef": min_coef,
        "use_topics": use_topics,
        "selected_topics": selected_topics,
        "min_conf": min_conf,
        "topic_usage": topic_usage,
        "chunk_mode": chunk_mode,
        "use_exclusion_words": use_exclusion_words,
        "exclusion_words_text": exclusion_words_text,
        "n_topics": n_topics,
        "train_A": train_A,
        "train_B": train_B,
        "train_U": train_U,
        "apply_A": apply_A,
        "apply_B": apply_B,
        "apply_U": apply_U,
        "analysis_flags": sidebar_analysis_flags.copy(),
        "topic_for_authorship": topic_for_authorship,
    }
    st.session_state.analysis_result = None
current_fp = text_fingerprint(textA, textB, textU)

if (
    st.session_state.analysis_done
    and st.session_state.get("text_fingerprint") != current_fp
):
    st.session_state.analysis_done = False

if not st.session_state.analysis_done:
    st.stop()
    
selected_topics = st.session_state.params.get("selected_topics", [])
min_conf = st.session_state.params.get("min_conf", 0.0)
topic_usage = st.session_state.params.get("topic_usage", topic_usage)
chunk_mode = st.session_state.params.get("chunk_mode", chunk_mode)
n_topics = st.session_state.params.get("n_topics", n_topics)
chunk_mode = st.session_state.params.get("chunk_mode", chunk_mode)

if not analysis_flags.get("topic_modeling", False):
    use_topics = False

train_A = st.session_state.params.get("train_A", train_A)
train_B = st.session_state.params.get("train_B", train_B)
train_U = st.session_state.params.get("train_U", train_U)

apply_A = st.session_state.params.get("apply_A", apply_A)
apply_B = st.session_state.params.get("apply_B", apply_B)
apply_U = st.session_state.params.get("apply_U", apply_U)


# -----------------
# FREEZE ANALYSIS PARAMETERS
# ----------------------
tokenizer_choice = st.session_state.params.get("tokenizer", tokenizer_choice)
chunk_size = st.session_state.params.get("chunk_size", chunk_size)
top_k = st.session_state.params.get("top_k", top_k)
min_coef = st.session_state.params.get("min_coef", min_coef)
topic_for_authorship = st.session_state.params.get("topic_for_authorship", topic_for_authorship)

exclusion_words = set()
if use_exclusion_words:
    exclusion_words = {
        w.strip().lower()
        for w in exclusion_words_text.replace(",", "\n").split()
        if w.strip()
    }
use_exclusion_words = st.session_state.params.get(
    "use_exclusion_words", use_exclusion_words
)

exclusion_words_text = st.session_state.params.get(
    "exclusion_words_text", exclusion_words_text
)
analysis_flags = st.session_state.params["analysis_flags"]
use_topics = analysis_flags["topic_modeling"]

exclusion_words_frozen = tuple(sorted(exclusion_words))
analysis_flags_frozen = tuple(sorted(analysis_flags.items()))

if analysis_flags.get("pca", False) and not analysis_flags.get("craig_markers", False):
    st.warning(
        "PCA is based on Craig marker-word frequencies and requires "
        "Craig marker analysis to be enabled. "
        "PCA has been disabled for this run."
    )
    analysis_flags["pca"] = False

if analysis_mode.startswith("Compare"):
    if not textA.strip() or not textB.strip() or not textU.strip():
        st.error("Please provide Author A, Author B, and Unknown texts.")
        st.stop()
else:
    if not textU.strip():
        st.error("Please provide a text for single-text analysis.")
        st.stop()


from analysis.pipeline import run_analysis

@st.cache_data(show_spinner=False)
def cached_run_analysis(*args):
    return run_analysis(*args)

analysis_result = st.session_state.analysis_result or {}
topicsA = analysis_result.get("topicsA")
topicsB = analysis_result.get("topicsB")
topicsU = analysis_result.get("topicsU")
topic_keywords = analysis_result.get("topic_keywords", {})


pca_result = None
fig_craig = None
custom_stopwords = set()


# ============================================================
# SINGLE-TEXT INTERNAL ANALYSIS
# ============================================================
if analysis_mode.startswith("Single"):

    with st.spinner("Running single-text analysis..."):

        try:
            tokens = tokenize_with_choice(textU, tokenizer_choice)
        except Exception:
            tokens = clean_and_tokenize(textU)

        chunks = chunk_words(tokens, chunk_size)

        if len(chunks) < 2:
            st.error("Text too short for chunk-based analysis.")
            st.stop()


        
        # --- Topic modelling
        if analysis_flags.get("topic_modeling", False) and len(chunks) >= 2:
            effective_n_topics = min(n_topics, len(chunks))

            custom_stopwords_single = set()

            if use_custom_stopwords and custom_stopwords_text:
                from nltk.stem import WordNetLemmatizer
                lemmatizer = WordNetLemmatizer()
                custom_stopwords_single = {
                    lemmatizer.lemmatize(w.strip().lower())
                    for w in custom_stopwords_text.replace(",", "\n").split()
                    if w.strip()
                }

            topic_model = train_topic_model(
                chunks,
                n_topics=effective_n_topics,
                stop_words=custom_stopwords_single if custom_stopwords_single else None
            )

            topicsU = apply_topic_model(topic_model, chunks)
            topic_keywords = get_topic_keywords(topic_model)
        else:
            topicsU = None
            topic_keywords = {}


        unique_topics = {
            tuple(words[:5])
            for words in topic_keywords.values()
        }

        if len(unique_topics) < len(topic_keywords):
            st.info(
               "Some topics share very similar keyword sets. "
                "This suggests the text supports fewer distinct themes "
                "than the selected number of topics."
            )

        st.subheader("Topic definitions (keywords)")
        st.caption(
            "Topic keywords are shown in normalized (lemmatized, lowercased) form. "
            "This improves semantic coherence and does not reflect exact surface spelling."
        )

        for t, words in topic_keywords.items():
            st.markdown(f"**Topic {t}**: {', '.join(words[:10])}")

        if topicsU:
            st.subheader("Topic distribution across chunks")

            topic_summary = summarize_topics_per_text(topicsU)

            df_topics = (
                pd.DataFrame.from_dict(topic_summary, orient="index", columns=["proportion"])
                .rename_axis("topic")
                .reset_index()
            )

            df_topics["keywords"] = df_topics["topic"].map(
                lambda t: ", ".join(topic_keywords.get(t, [])[:6])
            )

            st.dataframe(df_topics.sort_values("proportion", ascending=False))

        if topicsU:
            st.subheader("Topic assignment per chunk")

            df_chunks = pd.DataFrame([
                {
                    "chunk": i,
                    "topic": ann["topic"],
                    "confidence": ann["confidence"],
                    "keywords": ", ".join(topic_keywords.get(ann["topic"], [])[:6])
                }
                for i, ann in enumerate(topicsU)
            ])

            st.dataframe(df_chunks)
        else:
            st.info("Topic assignment not available (topic modelling not run or insufficient data).")


        if topicsU:
            with st.expander("Inspect Unknown chunks by topic", expanded=False):

                topic_choice = st.selectbox(
                    "Select topic",
                    sorted(topic_keywords.keys()),
                    format_func=lambda t: (
                            f"Topic {t}: " + ", ".join(topic_keywords[t][:4])
                    )
                )

                # collect chunks
                rows = []
                for i, ann in enumerate(topicsU):
                    if ann["topic"] == topic_choice:
                        rows.append({
                            "chunk": i,
                            "confidence": ann["confidence"],
                            "text": " ".join(chunksU_full[i][:200])
                        })

                if not rows:
                    st.info("No Unknown chunks assigned to this topic.")
                else:
                    df = (
                        pd.DataFrame(rows)
                        .sort_values("confidence", ascending=False)
                        .reset_index(drop=True)
                    )

                    st.caption(
                        f"{len(df)} chunks assigned to Topic {topic_choice}, "
                        "sorted by confidence."
                    )

                    for _, r in df.iterrows():
                        st.markdown(
                            f"**Chunk {r['chunk']} "
                            f"(conf={r['confidence']:.2f})**"
                        )
                        st.text(r["text"] + " …")

        if analysis_flags.get("topic_modeling", False) and not topicsU:
            st.warning(
                "Topic modelling was skipped because the text produced too few chunks "
                "for the selected chunk size / number of topics."
            )




       


        # ---  Burrows-style delta
        def internal_delta(chunks):
            freq = Counter(w for c in chunks for w in c)
            vocab = [w for w, _ in freq.most_common(300)]

            def vec(c):
                cnt = Counter(c)
                tot = max(1, len(c))
                return np.array([cnt[w] / tot for w in vocab])

            X = np.vstack([vec(c) for c in chunks])
            mean = X.mean(axis=0)
            sd = X.std(axis=0)
            sd[sd == 0] = 1e-8
            return np.mean(np.abs((X - mean) / sd), axis=1)

        deltas = internal_delta(chunks)



        # --- Features
        feat = features.extract_features(textU)
        stab = features.chunk_feature_stability(textU, chunk_size)

    # =======================
    # OUTPUT
    # =======================
    st.header("Single-text analysis results")
    
    

    st.subheader("Internal Burrows-style delta per chunk")
    with st.expander("ℹ️ What does this measure?", expanded=False):
        st.markdown("""
    ### Internal Burrows-style delta (single-text analysis)

    This measure adapts **Burrows' Delta** to a *single text* by comparing
    each chunk **against the text’s own average style**, rather than
    against another author.

    #### How it is computed
    1. The text is split into chunks.
    2. For each chunk, relative word frequencies are computed.
    3. An average frequency profile is computed across all chunks.
    4. Each chunk’s deviation from this average is measured using
       normalized absolute differences (z-scores).

    #### How to interpret the values
    - **Low delta** → stylistically typical chunk  
    - **High delta** → stylistically atypical chunk  

    High values often indicate:
    - scene or act boundaries  
    - shifts between dialogue and narration  
    - emotionally marked passages  
    - inserted genres (letters, songs, proclamations)

    This measure is **exploratory**, not classificatory.
    It reveals **internal stylistic structure and instability**.
    """)
    st.dataframe(
        pd.DataFrame({
            "chunk": [f"C{i}" for i in range(len(deltas))],
            "delta": deltas
        })
    )

    st.subheader("PCA of chunk vectors (contrastive)")
    st.info(
        "PCA on marker-word vectors requires contrastive reference texts. "
        "In single-text analysis, this view is not applicable."
    )

    fig_pca = None
    if pca_result:
        fig_pca = plot_pca_chunk_scatter(
            pca_result,
            title="PCA — internal stylistic structure"
        )

        if fig_pca is not None:
            st.pyplot(fig_pca)


    st.subheader("Global stylistic features")
    st.dataframe(
        pd.DataFrame(feat.items(), columns=["feature", "value"])
    )

    st.subheader("Feature stability across chunks")
    if stab:
        st.dataframe(
            pd.DataFrame(stab.items(), columns=["feature", "stability"])
            .sort_values("stability", ascending=False)
        )
    else:
        st.info("No stable features detected.")

   

    if "single_pdf" not in st.session_state:
        st.session_state.single_pdf = None

    if st.button("📄 Generate PDF report"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            
            build_single_text_report(
                filepath=tmp.name,
                deltas=deltas,
                global_features=feat,
                feature_stability=stab,
                params={
                    "chunk_size": chunk_size,
                    "pca_marker_limit": pca_marker_limit,
                },
                figs={
                    "pca": fig_pca
                }
            )



            st.session_state.single_pdf = tmp.name

    if st.session_state.single_pdf:
        with open(st.session_state.single_pdf, "rb") as f:
            st.download_button(
                label="⬇️ Download PDF report",
                data=f,
                file_name="stylometry_single_text_report.pdf",
                mime="application/pdf",
            )


    st.stop()





if not st.session_state.analysis_result:
    with st.spinner("Running analysis..."):
        st.session_state.analysis_result = cached_run_analysis(
            textA, textB, textU,
            tokenizer_choice, chunk_size,
            top_k, min_coef, use_min_coef,
            use_topics, n_topics if use_topics else 0,
            train_A, train_B, train_U,
            apply_A, apply_B, apply_U,
            custom_stopwords_text,
            exclusion_words_frozen,
            chunk_mode, selected_topics, min_conf,
            topic_usage,
            use_stability_filter,
            variance_threshold, min_chunk_presence, clump_ratio_threshold,
            pca_marker_limit,
            analysis_flags_frozen
        )

# unpack ONCE, reused on every rerun
locals().update(st.session_state.analysis_result)
st.write("Chunks A full:", len(chunksA_full))
st.write("Chunks A selected:", len(chunksA_sel))

st.write("Chunks B full:", len(chunksB_full))
st.write("Chunks B selected:", len(chunksB_sel))

st.write("Chunks U full:", len(chunksU_full))
st.write("Chunks U selected:", len(chunksU_sel))
# ============================================================
# EMBEDDINGS (AI LAYER)
# ============================================================
embedding_results = {}

if use_embeddings:
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import numpy as np
    with st.spinner("Computing embeddings..."):

        @st.cache_data
        def cached_embeddings(chunksA, chunksB, chunksU):
            return run_embedding_pipeline(chunksA, chunksB, chunksU)


        pipeline_out = cached_embeddings(
            chunksA_sel,
            chunksB_sel,
            chunksU_sel
        )

        embA = pipeline_out["embA"]
        embB = pipeline_out["embB"]
        embU = pipeline_out["embU"]
        sim_A = pipeline_out["sim_A"]
        sim_B = pipeline_out["sim_B"]
        embedding_results = {
            "sim_A": sim_A,
            "sim_B": sim_B
        }
        from utils.embedding_store import list_embeddings, load_embeddings

        # ============================================================
        # GLOBAL SEARCH OVER SAVED EMBEDDINGS
        # ============================================================

        if use_embeddings and embU is not None:

            st.header("📚 Similarity search across saved embeddings")

            results = []

            for name in list_embeddings():
                try:
                    stored = load_embeddings(name)
                    emb_saved = stored["embeddings"]

                    score = mean_similarity(embU, emb_saved)

                    results.append((name, score))
                except Exception:
                    continue

            # sort best first
            results = sorted(results, key=lambda x: x[1], reverse=True)

            # show top results
            for name, score in results[:10]:
                st.write(f"**{name}** → similarity: {score:.4f}")
        # ============================================================
        # INTERPRETABLE EMBEDDING VISUALIZATION
        # ============================================================
        if embA is not None and embB is not None and embU is not None:
            X = np.vstack([embA, embB, embU])
            labels = (
                    ["A"] * len(embA) +
                    ["B"] * len(embB) +
                    ["U"] * len(embU)
            )

            # PCA
            pca = PCA(n_components=2)
            X2 = pca.fit_transform(X)

            # Split back
            XA = X2[:len(embA)]
            XB = X2[len(embA):len(embA) + len(embB)]
            XU = X2[len(embA) + len(embB):]

            # Centroids
            centroid_A, centroid_B, centroid_U = compute_centroids(
                embA, embB, embU
            )

            centroid_A_2d = pca.transform([centroid_A])[0]
            centroid_B_2d = pca.transform([centroid_B])[0]
            centroid_U_2d = pca.transform([centroid_U])[0]

            fig, ax = plt.subplots()

            # Scatter
            ax.scatter(XA[:, 0], XA[:, 1], label="Author A", alpha=0.5)
            ax.scatter(XB[:, 0], XB[:, 1], label="Author B", alpha=0.5)
            ax.scatter(XU[:, 0], XU[:, 1], label="Unknown", alpha=0.8)

            # Centroids (BIG markers)
            ax.scatter(*centroid_A_2d, marker="X", s=200, label="A center")
            ax.scatter(*centroid_B_2d, marker="X", s=200, label="B center")
            ax.scatter(*centroid_U_2d, marker="X", s=200, label="U center")

            # Lines from U → A/B
            ax.plot(
                [centroid_U_2d[0], centroid_A_2d[0]],
                [centroid_U_2d[1], centroid_A_2d[1]]
            )

            ax.plot(
                [centroid_U_2d[0], centroid_B_2d[0]],
                [centroid_U_2d[1], centroid_B_2d[1]]
            )

            ax.legend()
            ax.set_title("Semantic similarity space (Embeddings)")
            st.pyplot(fig)
        dist_A, dist_B = compute_distances(
            centroid_A,
            centroid_B,
            centroid_U
        )

        st.markdown(f"""
        ### 🧠 How to read this plot

        - Each dot = a chunk of text
        - Clusters = consistent writing style/meaning
        - Big X = average "style center"

        **Distances:**
        - U → A: {dist_A:.3f}
        - U → B: {dist_B:.3f}

        👉 The closer the Unknown center is to an author, the more similar the writing.
        """)
        # ============================================================
        # NEAREST CHUNKS (EXPLAINABILITY)
        # ============================================================

        from utils.embeddings import cosine_similarity_matrix

        if embU is not None and embA is not None:
            sim_matrix = cosine_similarity_matrix(embU, embA)

            # find best match
            u_idx, a_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)

            st.subheader("🔍 Closest match (Unknown ↔ Author A)")

            st.markdown("**Unknown chunk:**")
            st.write(" ".join(chunksU_sel[u_idx][:50]))

            st.markdown("**Most similar Author A chunk:**")
            st.write(" ".join(chunksA_sel[a_idx][:50]))

            st.markdown(f"Similarity: **{sim_matrix[u_idx, a_idx]:.4f}**")

        # ============================================================
        # NEAREST CHUNKS (EXPLAINABILITY)
        # ============================================================

        from utils.embeddings import cosine_similarity_matrix

        if embU is not None and embB is not None:
            sim_matrix = cosine_similarity_matrix(embU, embB)

            # find best match
            u_idx, b_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)

            st.subheader("🔍 Closest match (Unknown ↔ Author B)")

            st.markdown("**Unknown chunk:**")
            st.write(" ".join(chunksU_sel[u_idx][:50]))

            st.markdown("**Most similar Author B chunk:**")
            st.write(" ".join(chunksB_sel[b_idx][:50]))

            st.markdown(f"Similarity: **{sim_matrix[u_idx, b_idx]:.4f}**")
        # ====================================================
        # SAVE EMBEDDINGS (USER FLOW)
        # ====================================================
        if save_embeddings_flag and save_embedding_name:

            save_embeddings(
                save_embedding_name + "_A",
                chunksA_sel,
                embA,
                {"label": "Author A"}
            )

            save_embeddings(
                save_embedding_name + "_B",
                chunksB_sel,
                embB,
                {"label": "Author B"}
            )

            if embU is not None:
                save_embeddings(
                    save_embedding_name + "_U",
                    chunksU_sel,
                    embU,
                    {"label": "Unknown"}
                )

            st.success(f"Embeddings saved under '{save_embedding_name}'")
# ========================
# AUTHorship WITHIN TOPIC
# ======
if (
    analysis_flags.get("topic_modeling", False)
    and topic_usage == "Control for topic (authorship within topic)"
    and topic_for_authorship is not None
    and topicsA
    and topicsB
):

    st.header("🟣 Authorship within topic")

    # --- Filter chunks
    chunksA_T = filter_chunks_by_topic(
        chunksA_full, topicsA, topic_for_authorship, min_conf
    )
    chunksB_T = filter_chunks_by_topic(
        chunksB_full, topicsB, topic_for_authorship, min_conf
    )
    chunksU_T = (
        filter_chunks_by_topic(chunksU_full, topicsU, topic_for_authorship, min_conf)
        if topicsU else []
    )

    st.caption(
        f"Topic {topic_for_authorship} — "
        f"A: {len(chunksA_T)} chunks, "
        f"B: {len(chunksB_T)} chunks, "
        f"U: {len(chunksU_T)} chunks"
    )

    # --- Safety check
    if len(chunksA_T) < 2 or len(chunksB_T) < 2:
        st.warning(
            "Too few chunks in this topic to perform authorship attribution."
        )
    else:
        # --- Craig within topic
        coef_T = compute_craig_coefficients(chunksA_T, chunksB_T)

        top_markers_T = top_k_markers(
            coef_T, k=top_k, min_coef=min_coef if use_min_coef else None
        )

        marker_words_T = [w for w, _ in top_markers_T]

        propsA_T_A, _ = chunk_marker_proportions(chunksA_T, marker_words_T)
        propsB_T_B, _ = chunk_marker_proportions(chunksB_T, marker_words_T)
        propsU_T, _ = (
            chunk_marker_proportions(chunksU_T, marker_words_T)
            if chunksU_T else ([], [])
        )

        # --- Delta within topic
        deltaA_T = burrows_delta(chunksU_T, chunksA_T) if chunksU_T else None
        deltaB_T = burrows_delta(chunksU_T, chunksB_T) if chunksU_T else None

        # --- Display verdict
        if deltaA_T is not None and deltaB_T is not None:
            st.markdown(
                f"**Burrows' Delta (within topic)** — "
                f"U→A: **{deltaA_T:.4f}**, "
                f"U→B: **{deltaB_T:.4f}**"
            )

            if deltaA_T < deltaB_T:
                st.success("Within this topic, Unknown is closer to Author A")
            elif deltaB_T < deltaA_T:
                st.success("Within this topic, Unknown is closer to Author B")
            else:
                st.info("Within-topic result is inconclusive")

        # --- Optional
        if chunksU_T:
            fig_T = plot_scatter_positions(
                propsA_T_A, propsA_T_A,
                propsB_T_B, propsB_T_B,
                propsU_T, propsU_T,
            )
            st.pyplot(fig_T)

        st.markdown(
            """
            **Interpretation**

            This analysis tests whether authorship attribution
            survives **after controlling for topic**.

            - If the verdict matches the global result → style is topic-independent.
            - If the verdict flips or collapses → attribution was topic-driven.
            """
        )


# ------------------------------------------------------------
# PCA FIGURE (COMPARE MODE)
# ------------------------------------------------------------
fig_pca = None
if (
    analysis_flags.get("pca", False)
    and pca_result is not None
):
    fig_pca = plot_pca_chunk_scatter(
        pca_result,
        title="PCA on chunk marker-frequency vectors"
    )

# ------------------------------------------------------------
# DEBUG
# ------------------------------------------------------------
with st.expander("🧪 Debug: frozen analysis parameters", expanded=False):
    if "params" not in st.session_state:
        st.info("No analysis has been run yet.")
    else:
        debug_rows = []
        for k, v in st.session_state.params.items():
            debug_rows.append({
                "parameter": k,
                "value": str(v),
            })

        st.dataframe(
            pd.DataFrame(debug_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These values were frozen at the moment you clicked "
            "**Run analysis** and are guaranteed to match the results above."
        )


# ============================================================
# TOPIC MODELLING RESULTS (COMPARE MODE)
# ============================================================
if analysis_flags.get("topic_modeling", False):

    st.header("🧠 Topic modelling")

    if not topic_keywords:
        st.warning(
            "Topic modelling was enabled, but no stable topics could be learned. "
            "This usually means there were too few chunks for the selected number of topics."
        )
    else:
        st.subheader("Topic definitions (keywords)")
        st.caption(
            "Keywords are shown in lemmatized, lowercased form. "
            "They represent semantic themes, not surface word forms."
        )

        for t, words in topic_keywords.items():
            st.markdown(f"**Topic {t}**: {', '.join(words[:10])}")

    # --- Topic distribution per text
    if topicsA or topicsB or topicsU:
        st.subheader("Topic distribution per text")

        rows = []
        if topicsA:
            for t, p in summarize_topics_per_text(topicsA).items():
                rows.append({"text": "Author A", "topic": t, "proportion": p})
        if topicsB:
            for t, p in summarize_topics_per_text(topicsB).items():
                rows.append({"text": "Author B", "topic": t, "proportion": p})
        if topicsU:
            for t, p in summarize_topics_per_text(topicsU).items():
                rows.append({"text": "Unknown", "topic": t, "proportion": p})

        df_topics = pd.DataFrame(rows)
        st.dataframe(
            df_topics
            .sort_values(["text", "proportion"], ascending=[True, False]),
            use_container_width=True
        )

    # ---  chunk-level inspection
    if topicsU:
        with st.expander("Inspect Unknown chunks by topic", expanded=False):
            topic_choice = st.selectbox(
                "Select topic",
                sorted(topic_keywords.keys()),
                format_func=lambda t: f"Topic {t}: {', '.join(topic_keywords[t][:4])}"
            )

            for i, ann in enumerate(topicsU):
                if ann["topic"] == topic_choice:
                    st.markdown(f"**Chunk {i} (conf={ann['confidence']:.2f})**")
                    st.text(" ".join(chunksU_full[i][:200]) + " …")

# ------------
# TOPIC FILTER CONTROLS
# -----
if (
    use_topics
    and analysis_flags.get("topic_modeling", False)
    and topic_usage == "Filter chunks for analysis"
    and topic_keywords
):


    st.sidebar.subheader("Topic-based filtering")

    selected_topics = st.sidebar.multiselect(
        "Select topic(s)",
        list(topic_keywords.keys()),
        default=selected_topics if selected_topics else list(topic_keywords.keys()),
        format_func=lambda t: (
            f"Topic {t}: " + ", ".join(topic_keywords[t][:4])
        ),
    )

    min_conf = st.sidebar.slider(
        "Minimum topic confidence",
        0.0, 1.0,
        min_conf,
        0.05,
    )

    st.sidebar.caption(
        "Changes take effect only after re-running the analysis."
    )
    st.session_state.params["selected_topics"] = selected_topics
    st.session_state.params["min_conf"] = min_conf
    st.session_state.analysis_done = False
    st.rerun()

    if st.sidebar.button("🔁 Re-run with selected topics"):
        st.session_state.analysis_done = False
        st.rerun()





# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------
# ============================================================
# CHUNK INSPECTION
# ============================================================
st.header("🔎 Chunk inspection")

with st.expander("Inspect individual chunks", expanded=False):

    source = st.radio(
        "Select text",
        ["Author A", "Author B", "Unknown"],
        horizontal=True
    )

    if source == "Author A":
        chunks = chunksA_sel
        props_A = propsA_A
        props_B = propsA_B
        label = "A"
    elif source == "Author B":
        chunks = chunksB_sel
        props_A = propsB_A
        props_B = propsB_B
        label = "B"
    else:
        chunks = chunksU_sel
        props_A = propsU_A
        props_B = propsU_B
        label = "U"

    idx = st.selectbox(
        "Chunk",
        range(len(chunks)),
        format_func=lambda i: f"{label}_{i}"
    )

    st.text_area(
        "Chunk text",
        " ".join(chunks[idx]),
        height=260
    )

    st.markdown(
        f"""
        **Author-A markers:** {props_A[idx]:.4f}  
        **Author-B markers:** {props_B[idx]:.4f}
        """
    )

st.header("Results summary")
with st.expander("🧭 Active analyses in this run", expanded=False):
    for k, v in analysis_flags.items():
        if v:
            st.markdown(f"• **{k.replace('_', ' ').title()}**")

st.header("Attribution result")
if deltaA is not None and deltaB is not None:
    st.markdown(f"**Burrows' Delta** — Unknown→A: **{deltaA:.4f}**, Unknown→B: **{deltaB:.4f}**")
else:
    st.info("Burrows' Delta was disabled.")

if deltaA is not None and deltaB is not None:
    if deltaA < deltaB:
        st.success("Unknown closer to Author A")
    elif deltaB < deltaA:
        st.success("Unknown closer to Author B")
    else:
        st.info("Tie / inconclusive")
else:
    st.info("Burrows’ Delta verdict not available.")

st.divider()
st.header("Chunk-level evidence")
if analysis_flags.get("craig_markers", False):
    st.subheader("Craig (marker-word scatter)")
    st.markdown("""
    **How to read this plot**

    Each point represents a **text chunk**, not a word.

    - The **x-axis** shows the proportion of **Author-A marker words** in the chunk.
    - The **y-axis** shows the proportion of **Author-B marker words** in the chunk.
    - Chunks closer to the **A axis** exhibit stronger stylistic affinity with Author A.
    - Chunks closer to the **B axis** exhibit stronger stylistic affinity with Author B.
    - Chunks near the diagonal or center are stylistically mixed or ambiguous.

    This visualization operationalizes Craig’s idea at the **chunk level**,
    making stylistic dominance and hybridity directly observable.
    """)


    fig_craig = plot_scatter_positions(
        propsA_A, propsA_B,
        propsB_A, propsB_B,
        propsU_A, propsU_B
    )
    st.pyplot(fig_craig)


    st.markdown("### Numeric values for Craig scatter (Unknown chunks only)")
    st.markdown("""
    **Column interpretation**

    - **U_A_markers** — proportion of Author-A marker words in the chunk  
    - **U_B_markers** — proportion of Author-B marker words in the chunk  

    Values are normalized by chunk length.  
    Higher values indicate stronger stylistic alignment.
    """)

    num_rows = []
    for i, (uA, uB) in enumerate(zip(propsU_A, propsU_B)):
        num_rows.append({"chunk": f"U_{i}", "U_A_markers": uA, "U_B_markers": uB})
    st.dataframe(pd.DataFrame(num_rows))

    st.subheader("Top marker words — Author A (A vs B), Author B (B vs A), and Unknown (U vs A/B)")
    st.markdown("""
    **Column descriptions:**
    - **word** – the marker word
    - **k** – Craig coefficient (distinctiveness measure)
    - **Av / Bv** – number of chunks where the word appears for A / B
    - **An** – number of opposing-author chunks lacking the word
    - **freqA / freqB** – absolute frequencies
    - **relA / relB** – relative frequencies per total tokens
    - **chi2** – chi-square contrast A vs B
    """)
    st.markdown("""
    **How to interpret each row**

    Each row represents a **single lexical marker** whose distribution differs
    systematically between the compared texts.

    - A **high k value** means the word is disproportionately associated
    with one author.
    - **High Av but low Bv** indicates a word that is widespread in A
    but rare or absent in B.
    - **High chi²** confirms that the observed difference is statistically
    unlikely to be random.
    - Relative frequencies (**relA / relB**) control for text length.

    Marker words reflect **unconscious lexical preferences**, not topic alone,
    especially after stability filtering.
    """)

    st.markdown("""
    **How marker words are computed**

    Marker words are identified using a Craig-style contrastive analysis:

    1. Word frequencies are computed per chunk and normalized.
    2. For each word, a contrast score is calculated based on its relative overuse
    between the two compared corpora.
    3. Words are ranked by the absolute magnitude of this contrast.
    4. Optional stability filters remove words that:
    - appear in too few chunks,
    - show excessive variance,
    - cluster in isolated text regions.

    For the Unknown text, marker words indicate stylistic proximity to Author A or B
    based on which contrast direction they align with.
    """)

    T1 = max(1, len(chunksA_sel))
    T2 = max(1, len(chunksB_sel))
    presA = compute_segment_presence(chunksA_sel)
    presB = compute_segment_presence(chunksB_sel)
    totalA_tokens = max(1, len(tokensA))
    totalB_tokens = max(1, len(tokensB))

    items_A = sorted(coef_A_vs_B.items(), key=lambda x: x[1], reverse=True)
    if use_min_coef:
        items_A = [it for it in items_A if it[1] >= min_coef]
    topA = items_A[:top_k]
    rowsA = []
    for word, coefval in topA:
        Av = presA.get(word, 0)
        Bv = presB.get(word, 0)
        An = T2 - Bv
        freqA = tokensA.count(word)
        freqB = tokensB.count(word)
        relA = freqA / totalA_tokens
        relB = freqB / totalB_tokens
        chi2 = chi2_for_word(freqA, totalA_tokens, freqB, totalB_tokens)
        rowsA.append({"word": word, "k": coefval, "Av": Av, "Bv": Bv, "An": An,
                      "freqA": freqA, "freqB": freqB, "relA": relA, "relB": relB, "chi2": chi2})
    dfA = pd.DataFrame(rowsA).head(500)

    items_B = sorted(coef_B_vs_A.items(), key=lambda x: x[1], reverse=True)
    if use_min_coef:
        items_B = [it for it in items_B if it[1] >= min_coef]
    topB = items_B[:top_k]
    rowsB = []
    for word, coefval in topB:
        Av = presA.get(word, 0)
        Bv = presB.get(word, 0)
        An = T2 - Bv
        freqA = tokensA.count(word)
        freqB = tokensB.count(word)
        relA = freqA / totalA_tokens
        relB = freqB / totalB_tokens
        chi2 = chi2_for_word(freqA, totalA_tokens, freqB, totalB_tokens)
        rowsB.append({"word": word, "k": coefval, "Av": Av, "Bv": Bv, "An": An,
                      "freqA": freqA, "freqB": freqB, "relA": relA, "relB": relB, "chi2": chi2})
    dfB = pd.DataFrame(rowsB).head(500)

    left, right = st.columns(2)
    with left:
        st.markdown("**Top markers — Author A (A vs B)**")
        st.dataframe(dfA)
    with right:
        st.markdown("**Top markers — Author B (B vs A)**")
        st.dataframe(dfB)


    st.header("Word-level evidence")
    st.subheader("Top marker words — Unknown (U vs A/B)")
    st.markdown("""
    **How to interpret this table**
    
    Each row represents a word that is stylistically distinctive in the Unknown text.
    
    - **k_vs_A** — how strongly the word distinguishes Unknown from Author A
    - **k_vs_B** — how strongly the word distinguishes Unknown from Author B
    - Larger values indicate stronger stylistic deviation.
    
    Words with **low k_vs_A and low k_vs_B** are stylistically neutral.
    Words with **high k_vs_A but low k_vs_B** suggest alignment with Author B,
    and vice versa.
    """)

    coef_U_vs_A = compute_craig_coefficients(chunksU_sel, chunksA_sel)
    coef_U_vs_B = compute_craig_coefficients(chunksU_sel, chunksB_sel)

    items_UA = sorted(coef_U_vs_A.items(), key=lambda x: x[1], reverse=True)[:top_k]
    items_UB = sorted(coef_U_vs_B.items(), key=lambda x: x[1], reverse=True)[:top_k]
    rowsU = []
    for word, _ in items_UA + items_UB:
        freqU = tokensU.count(word)
        relU = freqU / max(1, len(tokensU))
        freqA = tokensA.count(word)
        freqB = tokensB.count(word)
        rowsU.append({
            "word": word,
            "k_vs_A": coef_U_vs_A.get(word, 0),
            "k_vs_B": coef_U_vs_B.get(word, 0),
            "freqU": freqU,
            "relU": relU,
            "freqA": freqA,
            "freqB": freqB,
        })
    dfU = pd.DataFrame(rowsU).drop_duplicates(subset=["word"]).head(500)
    st.dataframe(dfU)
else:
    st.info("Craig marker analysis is disabled.")


st.header("Global stylistic features")

if analysis_flags.get("pca", False):

    st.subheader("PCA on chunk marker-frequency vectors")

    if pca_result is None:
        st.info(
            "PCA could not be computed.\n\n"
            "This usually means that too few stable Craig markers remained "
            "after filtering (Top-K / coefficient threshold / stability filter)."
        )
    else:
        st.markdown(""" 
        **What this PCA shows**

        Each point represents a text chunk embedded in a space defined by marker-word frequencies.

        - Principal Component 1 (PC1) captures the strongest stylistic contrast in the data.
        - Principal Component 2 (PC2) captures the second strongest independent contrast.
        - Clusters indicate stylistic consistency.
        - Overlap suggests stylistic similarity or mixture.

        PCA is used here for **exploration and visualization**, not for classification. 
        """)
        st.pyplot(fig_pca)


        ev = pca_result["explained_variance_ratio"]
        st.markdown(
            f"Explained variance — PC1: **{ev[0]:.3f}**, PC2: **{ev[1]:.3f}**"
        )

        loadings = pca_result["loadings"]
        vocab = pca_result["vocab"]

        pc1_load = sorted(
            [(abs(loadings[i, 0]), vocab[i], loadings[i, 0])
             for i in range(len(vocab))],
            reverse=True
        )[:30]

        st.table(pd.DataFrame(
            pc1_load,
            columns=["abs_loading", "word", "signed_loading"]
        ))

st.header("Exploratory structure")
st.subheader("Feature-based similarity")

sim = feature_compare.get("similarity", {})

if sim:
    df_sim = pd.DataFrame([
        {
            "Compared to": "Author A",
            "Distance": sim.get("feature_distance_A"),
            "TF–IDF cosine": sim.get("tfidf_cosine_A"),
            "Craig distance": sim.get("craig_distance_A"),
            "Composite score": sim.get("score_A"),
        },
        {
            "Compared to": "Author B",
            "Distance": sim.get("feature_distance_B"),
            "TF–IDF cosine": sim.get("tfidf_cosine_B"),
            "Craig distance": sim.get("craig_distance_B"),
            "Composite score": sim.get("score_B"),
        },
    ])

    st.dataframe(
        df_sim.round(4),
        use_container_width=True,
        hide_index=True,
    )

    verdict = feature_compare.get("verdict", "N/A")
    st.success(f"**Feature-based verdict:** {verdict}")
    st.subheader("Feature-level comparison (A / B / U)")

    raw = feature_compare.get("raw_features", {})

    if raw:
        rows = []
        stab = feature_compare.get("feature_stability", {})

        for feat, vals in raw.items():
            rows.append({
                "feature": feat,
                "Author A": vals.get("A"),
                "Author B": vals.get("B"),
                "Unknown": vals.get("U"),
                "stability": stab.get(feat),
                "U−A": (
                    abs(vals.get("U") - vals.get("A"))
                    if None not in (vals.get("U"), vals.get("A")) else None
                ),
                "U−B": (
                    abs(vals.get("U") - vals.get("B"))
                    if None not in (vals.get("U"), vals.get("B")) else None
                ),
            })

        df_feat = pd.DataFrame(rows)

        st.dataframe(
            df_feat
            .sort_values("stability", ascending=False)
            .round(4),
            use_container_width=True,
        )

        st.caption(
            "This table shows **raw stylistic feature values** for each text. "
            "Distances (U−A, U−B) indicate how far the Unknown deviates from each author. "
            "Stability reflects how reliable each feature is across reference chunks."
        )
    else:
        st.info("Raw feature values not available.")

else:
    st.info("Feature-based similarity not available.")
with st.expander("ℹ️ How feature-based similarity works", expanded=False):
    st.markdown("""
This comparison uses **aggregated stylistic features** (see `features.py`)
rather than individual words.

Each feature vector summarizes a text’s stylistic profile.
Distances are computed in feature space to estimate stylistic proximity.

- **Lower distance** → more similar style  
- **Higher TF–IDF cosine** → more shared lexical profile  
- **Composite score** combines multiple signals

This method complements marker-word analysis by capturing **global style**
rather than localized lexical choices.
""")

st.subheader("Feature reliability (reference-author stability)")

with st.expander("ℹ️ What does this mean?", expanded=False):
    st.markdown("""
### Concept
Not all stylistic features are equally reliable for authorship attribution.

Some features reflect **stable writing habits** (e.g. sentence rhythm),
while others fluctuate due to **topic, genre, or local emphasis**.

This table estimates **how reliable each feature is**, based on how
**consistently it appears across chunks of the known authors (A and B)**.

The Unknown text is **not used** to compute these weights.
""")
    st.markdown("""
### How the reliability is computed

1. Each reference text (Author A and Author B) is split into chunks.
2. For every chunk, stylistic features are extracted.
3. For each feature, we measure how much it **varies across chunks**.
4. Features with **low variability** are considered more reliable.

Formally, for a feature *f* with chunk values \\(x_1, x_2, ..., x_n\\):

\\[
\\text{CV}_f = \\frac{\\sigma_f}{|\\mu_f| + \\varepsilon}
\\]

where:
- \\(\\mu_f\\) is the mean feature value across chunks  
- \\(\\sigma_f\\) is the standard deviation  
- \\(\\varepsilon\\) is a small constant for numerical stability  

The final reliability weight is:

\\[
w_f = e^{-\\text{CV}_f}
\\]

This maps stable features close to **1.0** and unstable features close to **0.0**.
""")
    st.markdown("""
### How to interpret the values

| Reliability | Interpretation |
|------------|----------------|
| > 0.8 | Very stable stylistic habit |
| 0.5 – 0.8 | Moderately stable |
| < 0.5 | Unreliable / topic-driven |
| ≈ 0 | Ignored in attribution |

Only **reliable features** significantly influence the final verdict.
Unreliable features are automatically down-weighted.
""")

stab = feature_compare.get("feature_stability", {})

if stab:
    df_stab = (
        pd.DataFrame(stab.items(), columns=["feature", "stability"])
        .sort_values("stability", ascending=False)
    )
    st.dataframe(df_stab)
else:
    st.info("No stability information available.")

# -----------------------------
explanations = feature_compare.get("explanation", [])

st.subheader("Why this decision? (Feature explanation)")

if explanations:
    for line in explanations:
        st.write("•", line)
else:
    st.info("No strong feature-level preferences detected (differences too small).")



st.subheader("Chunk selection summary")

st.info(selection_summary(len(chunksA_full), len(chunksA_sel), chunk_mode))
st.info(selection_summary(len(chunksB_full), len(chunksB_sel), chunk_mode))
st.info(selection_summary(len(chunksU_full), len(chunksU_sel), chunk_mode))


if "compare_pdf" not in st.session_state:
    st.session_state.compare_pdf = None

if st.button("📄 Generate PDF report"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        build_comparative_report(
            filepath=tmp.name,
            deltaA=deltaA,
            deltaB=deltaB,
            verdict=feature_compare.get("verdict", "N/A"),
            feature_similarity=feature_compare.get("similarity", {}),
            feature_stability=feature_compare.get("feature_stability", {}),
            topic_modeling={
                "enabled": use_topics,
                "n_topics": n_topics if use_topics else None,
                "topics": topic_keywords,
                "topic_usage": topic_usage,
                "selected_topics": selected_topics,
                "chunk_counts": {
                    "A": (len(chunksA_full), len(chunksA_sel)),
                    "B": (len(chunksB_full), len(chunksB_sel)),
                    "U": (len(chunksU_full), len(chunksU_sel)),
                },
                "custom_stopwords": sorted(custom_stopwords) if custom_stopwords else [],
            },

            params={
                "chunk_size": chunk_size,
                "top_k": top_k,
                "pca_marker_limit": pca_marker_limit,
            },
            figs={
                "pca": fig_pca,
                "craig": fig_craig,
            },
            explanation = feature_compare.get("explanation", []),
            craig_numeric_U=(
                num_rows if analysis_flags.get("craig_markers", False) else None
            ),
            markers_A=(
                rowsA if analysis_flags.get("craig_markers", False) else None
            ),
            markers_B=(
                rowsB if analysis_flags.get("craig_markers", False) else None
            ),
            markers_U=(
                rowsU if analysis_flags.get("craig_markers", False) else None
            ),
            feature_table=(
                df_feat.to_dict("records") if df_feat is not None else None
            ),
        )



        st.session_state.compare_pdf = tmp.name

if st.session_state.compare_pdf:
    with open(st.session_state.compare_pdf, "rb") as f:
        st.download_button(
            label="⬇️ Download PDF report",
            data=f,
            file_name="stylometry_authorship_report.pdf",
            mime="application/pdf",
        )


with st.expander("📘 Methodology & Interpretation", expanded=False):

    st.markdown("""
### Burrows' Delta
**Formula**  
Δ = mean(|z(U) − z(A)|)

**Interpretation**  
Lower values indicate stylistic proximity.

**Typical thresholds**
- Δ < 1.0 → strong similarity
- 1.0–1.5 → possible similarity
- >1.5 → weak evidence
""")

    st.markdown("""
### Craig’s Marker Method
**Idea**  
Identifies words whose **distribution across chunks** is asymmetric
between authors.

**Coefficient meaning**
The Craig coefficient increases when a word:
- appears in many chunks of one author,
- appears in few or no chunks of the other author.

**Interpretation**
Craig markers capture **habitual lexical preferences** rather than
raw frequency or topic vocabulary.

**Stability filter**
Removes markers that are rare, clumped, or unstable across chunks.
""")

    st.markdown("""
### Feature-Based Stylometry
**What it captures**
Sentence rhythm, lexical richness, punctuation, syntax.

**Distance**
Euclidean distance in weighted feature space.

**Important**
Features with low stability are down-weighted or ignored.
""")

    st.markdown("""
### POS Profiles
**noun vs verb**
Narrative vs action orientation.

**modifier density**
Descriptive pressure.

**function pressure**
Use of grammatical scaffolding.
""")

