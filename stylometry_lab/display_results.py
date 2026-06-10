"""
display_results.py - Result Visualization and Display

Handles all result presentation, visualizations, and display logic
for both single-text and comparative analyses.
"""

from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from ui_setup import (
    render_subheader, render_info_box, render_success_box,
    render_warning_box, create_columns, create_expander,
    render_markdown, render_caption, render_divider,
    create_tabs, show_spinner, stop_execution
)
from utils.plots import plot_scatter_positions, plot_pca_chunk_scatter
from utils.report import build_single_text_report, build_comparative_report
from utils.chunk_selection import selection_summary
from utils.embeddings import cosine_similarity_matrix
from utils.embeddings_pipeline import compute_centroids, compute_distances
from analysis_orchestrator import perform_internal_delta_analysis
from config import log_info, log_error


def display_single_text_results(
    chunks: List[List[str]],
    deltas: np.ndarray,
    global_features: Dict[str, float],
    feature_stability: Dict[str, float],
    topic_keywords: Dict[int, List[str]],
    topicsU: Optional[List[Dict[str, Any]]],
    chunksU_full: List[List[str]],
    chunk_size: int,
    pca_marker_limit: int,
    fig_pca: Optional[plt.Figure] = None
) -> None:
    """
    Display all results for single-text analysis.
    
    Args:
        chunks: Token chunks from text
        deltas: Internal delta values per chunk
        global_features: Global stylistic features
        feature_stability: Feature stability scores
        topic_keywords: Topic model keywords
        topicsU: Topic assignments per chunk
        chunksU_full: Full token chunks
        chunk_size: Chunk size parameter
        pca_marker_limit: PCA marker limit
        fig_pca: Optional pre-computed PCA figure
    
    Example:
        >>> display_single_text_results(chunks, deltas, features, ...)
    """
    st.header("Single-text analysis results")
    
    # Display internal delta
    render_subheader("Internal Burrows-style delta per chunk")
    with create_expander("ℹ️ What does this measure?", expanded=False):
        render_markdown("""
### Internal Burrows-style delta (single-text analysis)

This measure adapts **Burrows' Delta** to a *single text* by comparing
each chunk **against the text's own average style**, rather than
against another author.

#### How it is computed
1. The text is split into chunks.
2. For each chunk, relative word frequencies are computed.
3. An average frequency profile is computed across all chunks.
4. Each chunk's deviation from this average is measured using
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
    
    # Display features
    render_subheader("Global stylistic features")
    st.dataframe(
        pd.DataFrame(global_features.items(), columns=["feature", "value"])
    )
    
    # Display feature stability
    render_subheader("Feature stability across chunks")
    if feature_stability:
        st.dataframe(
            pd.DataFrame(feature_stability.items(), columns=["feature", "stability"])
            .sort_values("stability", ascending=False)
        )
    else:
        render_info_box("No stable features detected.")
    
    # Display topics if available
    if topicsU:
        display_topic_results_single(topic_keywords, topicsU, chunksU_full)
    
    # PDF report generation
    display_pdf_generation_single(deltas, global_features, feature_stability, chunk_size, fig_pca)


def display_topic_results_single(
    topic_keywords: Dict[int, List[str]],
    topicsU: List[Dict[str, Any]],
    chunksU_full: List[List[str]]
) -> None:
    """
    Display topic modeling results for single-text analysis.
    
    Args:
        topic_keywords: Topic keywords dictionary
        topicsU: Topic assignments
        chunksU_full: Full text chunks
    """
    render_subheader("Topic definitions (keywords)")
    render_caption(
        "Topic keywords are shown in normalized (lemmatized, lowercased) form. "
        "This improves semantic coherence and does not reflect exact surface spelling."
    )
    
    for t, words in topic_keywords.items():
        st.markdown(f"**Topic {t}**: {', '.join(words[:10])}")
    
    # Topic distribution
    render_subheader("Topic distribution across chunks")
    from utils.topic_model import summarize_topics_per_text
    
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


def display_pdf_generation_single(
    deltas: np.ndarray,
    global_features: Dict[str, float],
    feature_stability: Dict[str, float],
    chunk_size: int,
    fig_pca: Optional[plt.Figure] = None
) -> None:
    """
    Handle PDF report generation for single-text analysis.
    
    Args:
        deltas: Delta values
        global_features: Global features
        feature_stability: Feature stability
        chunk_size: Chunk size parameter
        fig_pca: PCA figure if available
    """
    if "single_pdf" not in st.session_state:
        st.session_state.single_pdf = None
    
    if st.button("📄 Generate PDF report"):
        with show_spinner("Generating PDF report..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                try:
                    build_single_text_report(
                        filepath=tmp.name,
                        deltas=deltas,
                        global_features=global_features,
                        feature_stability=feature_stability,
                        params={
                            "chunk_size": chunk_size,
                        },
                        figs={
                            "pca": fig_pca
                        }
                    )
                    st.session_state.single_pdf = tmp.name
                    render_success_box("PDF report generated successfully")
                except Exception as e:
                    log_error(f"PDF generation failed: {e}")
                    render_warning_box(f"PDF generation failed: {str(e)}")
    
    if st.session_state.single_pdf:
        with open(st.session_state.single_pdf, "rb") as f:
            st.download_button(
                label="⬇️ Download PDF report",
                data=f,
                file_name="stylometry_single_text_report.pdf",
                mime="application/pdf",
            )


def display_comparative_results(
    result: Dict[str, Any],
    analysis_flags: Dict[str, bool],
    pca_marker_limit: int
) -> None:
    """
    Display all results for comparative analysis.
    
    Args:
        result: Complete analysis result dictionary
        analysis_flags: Dictionary of enabled analysis features
        pca_marker_limit: PCA marker limit parameter
    """
    st.header("Attribution result")
    
    # Extract key results
    deltaA = result.get("deltaA")
    deltaB = result.get("deltaB")
    feature_compare = result.get("feature_compare", {})
    pca_result = result.get("pca_result")
    fig_craig = None
    
    # Display verdict
    if deltaA is not None and deltaB is not None:
        st.markdown(
            f"**Burrows' Delta** — Unknown→A: **{deltaA:.4f}**, "
            f"Unknown→B: **{deltaB:.4f}**"
        )
        
        if deltaA < deltaB:
            render_success_box("Unknown closer to Author A")
        elif deltaB < deltaA:
            render_success_box("Unknown closer to Author B")
        else:
            render_info_box("Tie / inconclusive")
    else:
        render_info_box("Burrows' Delta verdict not available.")
    
    render_divider()
    
    # Display evidence sections based on analysis flags
    if analysis_flags.get("craig_markers", False):
        display_craig_results(result)
    
    if analysis_flags.get("pca", False):
        display_pca_results(pca_result)
    
    display_feature_results(feature_compare)
    
    # Display chunk inspection
    display_chunk_inspection(result)
    
    # Generate PDF report
    display_pdf_generation_comparative(result, analysis_flags)


def display_craig_results(result: Dict[str, Any]) -> None:
    """
    Display CRAIG marker analysis results.
    
    Args:
        result: Analysis result dictionary
    """
    st.header("Chunk-level evidence")
    render_subheader("Craig (marker-word scatter)")
    
    render_markdown("""
**How to read this plot**

Each point represents a **text chunk**, not a word.

- The **x-axis** shows the proportion of **Author-A marker words** in the chunk.
- The **y-axis** shows the proportion of **Author-B marker words** in the chunk.
- Chunks closer to the **A axis** exhibit stronger stylistic affinity with Author A.
- Chunks closer to the **B axis** exhibit stronger stylistic affinity with Author B.
- Chunks near the diagonal or center are stylistically mixed or ambiguous.
    """)
    
    # Extract marker data and plot
    propsA_A = result.get("propsA_A", [])
    propsA_B = result.get("propsA_B", [])
    propsB_A = result.get("propsB_A", [])
    propsB_B = result.get("propsB_B", [])
    propsU_A = result.get("propsU_A", [])
    propsU_B = result.get("propsU_B", [])
    
    if all([propsA_A, propsA_B, propsB_A, propsB_B, propsU_A, propsU_B]):
        fig_craig = plot_scatter_positions(
            propsA_A, propsA_B,
            propsB_A, propsB_B,
            propsU_A, propsU_B
        )
        st.pyplot(fig_craig)


def display_pca_results(pca_result: Optional[Dict[str, Any]]) -> None:
    """
    Display PCA visualization results.
    
    Args:
        pca_result: PCA result dictionary or None
    """
    st.header("Global stylistic features")
    render_subheader("PCA on chunk marker-frequency vectors")
    
    if pca_result is None:
        render_info_box(
            "PCA could not be computed. This usually means that too few stable "
            "Craig markers remained after filtering."
        )
    else:
        render_markdown("""
**What this PCA shows**

Each point represents a text chunk embedded in a space defined by marker-word frequencies.

- Principal Component 1 (PC1) captures the strongest stylistic contrast in the data.
- Principal Component 2 (PC2) captures the second strongest independent contrast.
- Clusters indicate stylistic consistency.
- Overlap suggests stylistic similarity or mixture.

PCA is used here for **exploration and visualization**, not for classification.
        """)
        
        fig_pca = plot_pca_chunk_scatter(
            pca_result,
            title="PCA on chunk marker-frequency vectors"
        )
        st.pyplot(fig_pca)


def display_feature_results(feature_compare: Dict[str, Any]) -> None:
    """
    Display feature-based analysis results.
    
    Args:
        feature_compare: Feature comparison dictionary
    """
    st.header("Exploratory structure")
    render_subheader("Feature-based similarity")
    
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
        render_success_box(f"**Feature-based verdict:** {verdict}")


def display_chunk_inspection(result: Dict[str, Any]) -> None:
    """
    Display chunk inspection interface.
    
    Args:
        result: Analysis result dictionary
    """
    st.header("🔎 Chunk inspection")
    
    chunksA_sel = result.get("chunksA_sel", [])
    chunksB_sel = result.get("chunksB_sel", [])
    chunksU_sel = result.get("chunksU_sel", [])
    propsA_A = result.get("propsA_A", [])
    propsA_B = result.get("propsA_B", [])
    propsB_A = result.get("propsB_A", [])
    propsB_B = result.get("propsB_B", [])
    propsU_A = result.get("propsU_A", [])
    propsU_B = result.get("propsU_B", [])
    
    with create_expander("Inspect individual chunks", expanded=False):
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
        
        if chunks:
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
            
            if idx < len(props_A) and idx < len(props_B):
                st.markdown(
                    f"**Author-A markers:** {props_A[idx]:.4f}  \n"
                    f"**Author-B markers:** {props_B[idx]:.4f}"
                )


def display_pdf_generation_comparative(
    result: Dict[str, Any],
    analysis_flags: Dict[str, bool]
) -> None:
    """
    Handle PDF report generation for comparative analysis.
    
    Args:
        result: Analysis result dictionary
        analysis_flags: Analysis feature flags
    """
    if "compare_pdf" not in st.session_state:
        st.session_state.compare_pdf = None
    
    if st.button("📄 Generate PDF report"):
        with show_spinner("Generating PDF report..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    build_comparative_report(
                        filepath=tmp.name,
                        deltaA=result.get("deltaA"),
                        deltaB=result.get("deltaB"),
                        verdict=result.get("feature_compare", {}).get("verdict", "N/A"),
                        feature_similarity=result.get("feature_compare", {}).get("similarity", {}),
                        feature_stability=result.get("feature_compare", {}).get("feature_stability", {}),
                        topic_modeling={
                            "enabled": analysis_flags.get("topic_modeling", False),
                            "topics": result.get("topic_keywords", {}),
                        },
                        params={},
                        figs={},
                        explanation=result.get("feature_compare", {}).get("explanation", []),
                        craig_numeric_U=None,
                        markers_A=None,
                        markers_B=None,
                        markers_U=None,
                        feature_table=None,
                    )
                    st.session_state.compare_pdf = tmp.name
                    render_success_box("PDF report generated successfully")
            except Exception as e:
                log_error(f"PDF generation failed: {e}")
                render_warning_box(f"PDF generation failed: {str(e)}")
    
    if st.session_state.compare_pdf:
        with open(st.session_state.compare_pdf, "rb") as f:
            st.download_button(
                label="⬇️ Download PDF report",
                data=f,
                file_name="stylometry_authorship_report.pdf",
                mime="application/pdf",
            )


def display_methodology() -> None:
    """
    Display methodology and interpretation guide.
    """
    with create_expander("📘 Methodology & Interpretation", expanded=False):
        render_markdown("""
### Burrows' Delta
**Formula**  
Δ = mean(|z(U) − z(A)|)

**Interpretation**  
Lower values indicate stylistic proximity.

**Typical thresholds**
- Δ < 1.0 → strong similarity
- 1.0–1.5 → possible similarity
- >1.5 → weak evidence

### Craig's Marker Method
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
        """)
