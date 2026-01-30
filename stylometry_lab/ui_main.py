# ui_main.py
import streamlit as st
import pandas as pd
import tempfile

from utils.processing import extract_text_file
from utils.plots import plot_scatter_positions, plot_pca_chunk_scatter
from utils.craig import FUNCTION_WORDS
from utils.chunk_selection import selection_summary
from utils.report import (
    build_single_text_report,
    build_comparative_report
)

# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------
def setup_page():
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


# ------------------------------------------------------------
# INPUT UI
# ------------------------------------------------------------
def render_input_ui(analysis_mode):
    if analysis_mode.startswith("Compare"):
        colA, colB, colU = st.columns(3)

        with colA:
            fileA = st.file_uploader("Author A (.txt/.pdf/.docx)",
                                     type=["txt", "pdf", "docx"], key="fileA")
            pastedA = st.text_area("Or paste Author A text", height=220, key="pasteA")

        with colB:
            fileB = st.file_uploader("Author B (.txt/.pdf/.docx)",
                                     type=["txt", "pdf", "docx"], key="fileB")
            pastedB = st.text_area("Or paste Author B text", height=220, key="pasteB")

        with colU:
            fileU = st.file_uploader("Unknown (.txt/.pdf/.docx)",
                                     type=["txt", "pdf", "docx"], key="fileU")
            pastedU = st.text_area("Or paste Unknown text", height=220, key="pasteU")
    else:
        st.subheader("Single text input")
        fileU = st.file_uploader(
            "Upload text (.txt/.pdf/.docx)",
            type=["txt", "pdf", "docx"],
            key="fileU_single"
        )
        pastedU = st.text_area("Or paste text here", height=300, key="pasteU_single")
        fileA = fileB = pastedA = pastedB = None

    return fileA, pastedA, fileB, pastedB, fileU, pastedU


def get_text(fileobj, pasted, logger):
    txt = ""
    if fileobj:
        try:
            txt = extract_text_file(fileobj, fileobj.name)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            logger.warning(f"extract_text_file error: {e}")
            return ""
    if not txt and pasted and pasted.strip():
        txt = pasted
    return txt or ""


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
def render_sidebar():
    st.sidebar.header("Analysis modules")

    analysis_flags = {
        "burrows_delta": st.sidebar.checkbox("Burrows' Delta", True),
        "craig_markers": st.sidebar.checkbox("Craig marker analysis", True),
        "feature_analysis": st.sidebar.checkbox("Global stylistic features", True),
        "pca": st.sidebar.checkbox("PCA visualization", True),
        "topic_modeling": st.sidebar.checkbox("Topic modelling", False),
    }

    st.sidebar.header("Text preprocessing")

    tokenizer_choice = st.sidebar.selectbox(
        "Tokenizer",
        ["nltk", "simple", "regex", "unicode", "char_ngrams"],
        index=0,
    )

    chunk_size = st.sidebar.number_input(
        "Chunk size (words)", 200, 5000, 2000, 100
    )

    st.sidebar.header("Craig marker parameters")

    top_k = st.sidebar.slider("Top-K marker words", 20, 1000, 100, 10)
    use_min_coef = st.sidebar.checkbox("Enforce minimum coefficient threshold", False)
    min_coef = st.sidebar.slider(
        "Minimum coefficient", 0.0, 2.0, 0.0, 0.01, disabled=not use_min_coef
    )
    pca_marker_limit = st.sidebar.slider(
        "PCA marker limit", 50, 2000, 500, 50
    )

    st.sidebar.subheader("Craig marker analysis (reference)")
    with st.sidebar.expander("📘 Craig function-word list (used internally)", False):
        st.code(", ".join(sorted(FUNCTION_WORDS)))

    return {
        "analysis_flags": analysis_flags,
        "tokenizer": tokenizer_choice,
        "chunk_size": chunk_size,
        "top_k": top_k,
        "use_min_coef": use_min_coef,
        "min_coef": min_coef,
        "pca_marker_limit": pca_marker_limit,
    }


# ------------------------------------------------------------
# RESULTS (COMPARE MODE)
# ------------------------------------------------------------
def render_compare_results(state, figs):
    st.header("Attribution result")

    deltaA = state.get("deltaA")
    deltaB = state.get("deltaB")

    if deltaA is not None and deltaB is not None:
        st.markdown(
            f"**Burrows' Delta** — Unknown→A: **{deltaA:.4f}**, Unknown→B: **{deltaB:.4f}**"
        )
        if deltaA < deltaB:
            st.success("Unknown closer to Author A")
        elif deltaB < deltaA:
            st.success("Unknown closer to Author B")
        else:
            st.info("Tie / inconclusive")
    else:
        st.info("Burrows' Delta not available.")

    if figs.get("pca"):
        st.header("Global stylistic features")
        st.pyplot(figs["pca"])


# ------------------------------------------------------------
# PDF BUTTONS
# ------------------------------------------------------------
def render_compare_pdf(state, figs):
    if st.button("📄 Generate PDF report"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            build_comparative_report(
                filepath=tmp.name,
                deltaA=state.get("deltaA"),
                deltaB=state.get("deltaB"),
                verdict=state.get("feature_compare", {}).get("verdict", "N/A"),
                feature_similarity=state.get("feature_compare", {}).get("similarity", {}),
                feature_stability=state.get("feature_compare", {}).get("feature_stability", {}),
                params={},
                figs=figs,
            )
            st.session_state.compare_pdf = tmp.name

    if st.session_state.get("compare_pdf"):
        with open(st.session_state.compare_pdf, "rb") as f:
            st.download_button(
                "⬇️ Download PDF report",
                data=f,
                file_name="stylometry_authorship_report.pdf",
                mime="application/pdf",
            )
