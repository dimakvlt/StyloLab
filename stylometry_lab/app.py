"""
app.py - StyloLab Main Application Entry Point

Main Streamlit application that orchestrates all modular components.
Handles page setup, state management, input collection, and result display.
"""

from typing import Tuple, Optional
import streamlit as st
from datetime import datetime

# Import all modular components
import config
from session_manager import (
    initialize_session_state, get_session_value, set_session_value,
    is_analysis_done, set_analysis_done, get_analysis_result,
    set_analysis_result, get_frozen_params, set_frozen_params,
    get_text_fingerprint, set_text_fingerprint, invalidate_analysis,
    get_analysis_flag
)
from ui_setup import (
    initialize_ui, render_section_header, render_info_box,
    render_error_box, stop_execution, show_spinner
)
from text_input import (
    text_fingerprint, extract_text_from_input, validate_text_input
)
from analysis_orchestrator import (
    prepare_analysis_arguments, cached_run_analysis,
    validate_analysis_inputs, extract_analysis_results_to_locals
)
from display_results import (
    display_single_text_results, display_comparative_results,
    display_methodology
)

# Import UI modules
from ui.inputs import render_inputs, get_text
from ui.sidebar import render_sidebar


# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

def main() -> None:
    """
    Main application entry point.
    
    Orchestrates all application components:
    1. Initialize Streamlit and session state
    2. Setup UI and styling
    3. Collect user inputs
    4. Perform analysis
    5. Display results
    """
    # Setup logger
    logger = config.setup_logging()
    logger.info("=" * 80)
    logger.info(f"Application started at {datetime.now()}")
    logger.info("=" * 80)
    
    # Initialize Streamlit UI
    initialize_ui()
    
    # Initialize session state
    initialize_session_state()
    
    # ========================================================================
    # USER INPUT SECTION
    # ========================================================================
    
    # Get analysis mode
    analysis_mode = st.sidebar.radio(
        "Analysis mode",
        config.ANALYSIS_MODES
    )
    
    logger.info(f"Analysis mode: {analysis_mode}")
    
    # Render input interface (from ui.inputs module)
    fileA, pastedA, fileB, pastedB, fileU, pastedU = render_inputs(analysis_mode)
    
    # Extract text from inputs
    try:
        textA = get_text(fileA, pastedA)
        textB = get_text(fileB, pastedB)
        textU = get_text(fileU, pastedU)
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        render_error_box(f"Error processing input: {str(e)}")
        stop_execution()
    
    # ========================================================================
    # SIDEBAR PARAMETERS
    # ========================================================================
    
    # Render sidebar to collect analysis parameters
    sidebar_params = render_sidebar(analysis_mode)
    logger.info(f"Sidebar parameters collected")
    
    # Extract sidebar parameters
    analysis_type = sidebar_params.get("analysis_type", "Classical")
    use_embeddings = sidebar_params.get("use_embeddings", False)
    tokenizer_choice = sidebar_params.get("tokenizer")
    chunk_size = sidebar_params.get("chunk_size")
    top_k = sidebar_params.get("top_k")
    use_min_coef = sidebar_params.get("use_min_coef", False)
    min_coef = sidebar_params.get("min_coef", 0.0)
    pca_marker_limit = sidebar_params.get("pca_marker_limit")
    use_exclusion_words = sidebar_params.get("use_exclusion_words", False)
    exclusion_words_text = sidebar_params.get("exclusion_words_text", "")
    use_stability_filter = sidebar_params.get("use_stability_filter", False)
    variance_threshold = sidebar_params.get("variance_threshold", 0.1)
    min_chunk_presence = sidebar_params.get("min_chunk_presence", 2)
    clump_ratio_threshold = sidebar_params.get("clump_ratio_threshold", 0.7)
    topic_usage = sidebar_params.get("topic_usage", "Exploratory (no filtering)")
    custom_stopwords_text = sidebar_params.get("custom_stopwords_text", "")
    chunk_mode = sidebar_params.get("chunk_mode", "All chunks")
    n_topics = sidebar_params.get("n_topics", 5)
    
    train_A = sidebar_params.get("train_A", False)
    train_B = sidebar_params.get("train_B", False)
    train_U = sidebar_params.get("train_U", False)
    
    apply_A = sidebar_params.get("apply_A", False)
    apply_B = sidebar_params.get("apply_B", False)
    apply_U = sidebar_params.get("apply_U", False)
    
    sidebar_analysis_flags = sidebar_params.get("analysis_flags", {})
    topic_for_authorship = sidebar_params.get("topic_for_authorship")
    
    use_topics = sidebar_analysis_flags.get("topic_modeling", False)
    analysis_flags = sidebar_analysis_flags
    
    # Parse exclusion words
    exclusion_words = set()
    if use_exclusion_words and exclusion_words_text:
        exclusion_words = {
            w.strip().lower()
            for w in exclusion_words_text.replace(",", "\n").split()
            if w.strip()
        }
    
    # ========================================================================
    # ANALYSIS BUTTON & STATE MANAGEMENT
    # ========================================================================
    
    st.sidebar.markdown("---")
    
    if st.button("Run analysis"):
        set_analysis_done(True)
        set_text_fingerprint(text_fingerprint(textA, textB, textU))
        
        # Freeze parameters
        params = {
            "tokenizer": tokenizer_choice,
            "chunk_size": chunk_size,
            "top_k": top_k,
            "min_coef": min_coef,
            "use_topics": use_topics,
            "n_topics": n_topics,
            "topic_usage": topic_usage,
            "chunk_mode": chunk_mode,
            "use_exclusion_words": use_exclusion_words,
            "exclusion_words_text": exclusion_words_text,
            "train_A": train_A,
            "train_B": train_B,
            "train_U": train_U,
            "apply_A": apply_A,
            "apply_B": apply_B,
            "apply_U": apply_U,
            "analysis_flags": sidebar_analysis_flags.copy(),
            "topic_for_authorship": topic_for_authorship,
        }
        set_frozen_params(params)
        set_analysis_result(None)
        logger.info("Analysis initiated by user")
    
    # Check if inputs changed
    current_fp = text_fingerprint(textA, textB, textU)
    if (
        is_analysis_done()
        and get_text_fingerprint() != current_fp
    ):
        logger.info("Input texts changed - invalidating analysis")
        set_analysis_done(False)
    
    # Stop if analysis not done
    if not is_analysis_done():
        render_info_box("👈 Configure parameters and click **Run analysis** to start")
        stop_execution()
    
    # Retrieve frozen parameters
    frozen_params = get_frozen_params()
    
    # ========================================================================
    # VALIDATE INPUTS
    # ========================================================================
    
    is_valid, error_msg = validate_analysis_inputs(
        textA, textB, textU, analysis_mode
    )
    
    if not is_valid:
        render_error_box(error_msg)
        stop_execution()
    
    logger.info(f"Input validation passed")
    
    # ========================================================================
    # PERFORM ANALYSIS
    # ========================================================================
    
    # Get cached result or compute new one
    if not get_analysis_result():
        logger.info("Running analysis (no cached result)")
        
        with show_spinner("Running analysis..."):
            try:
                # Prepare arguments for caching
                args = prepare_analysis_arguments(
                    textA, textB, textU,
                    frozen_params.get("tokenizer", "nltk"),
                    frozen_params.get("chunk_size", 2000),
                    frozen_params.get("top_k", 50),
                    frozen_params.get("min_coef", 0.0),
                    use_min_coef,
                    frozen_params.get("use_topics", False),
                    frozen_params.get("n_topics", 5),
                    frozen_params.get("train_A", False),
                    frozen_params.get("train_B", False),
                    frozen_params.get("train_U", False),
                    frozen_params.get("apply_A", False),
                    frozen_params.get("apply_B", False),
                    frozen_params.get("apply_U", False),
                    frozen_params.get("custom_stopwords_text", ""),
                    tuple(sorted(exclusion_words)),
                    frozen_params.get("chunk_mode", "All chunks"),
                    frozen_params.get("selected_topics", []),
                    frozen_params.get("min_conf", 0.0),
                    frozen_params.get("topic_usage", "Exploratory (no filtering)"),
                    use_stability_filter,
                    variance_threshold,
                    min_chunk_presence,
                    clump_ratio_threshold,
                    pca_marker_limit,
                    frozen_params.get("analysis_flags", {})
                )
                
                # Run cached analysis
                result = cached_run_analysis(*args)
                set_analysis_result(result)
                logger.info("Analysis completed successfully")
            
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                render_error_box(f"Analysis failed: {str(e)}")
                stop_execution()
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    result = get_analysis_result()
    
    if result is None:
        render_error_box("No analysis results available")
        stop_execution()
    
    logger.info("Displaying analysis results")
    
    # Extract results to locals for easier access
    analysis_vars = extract_analysis_results_to_locals(result)
    
    # Display based on analysis mode
    if analysis_mode.startswith("Single"):
        # Single-text analysis
        deltas = analysis_vars.get("deltas")
        global_features = result.get("global_features", {})
        feature_stability = result.get("feature_stability", {})
        topic_keywords = analysis_vars.get("topic_keywords", {})
        topicsU = analysis_vars.get("topicsU")
        chunksU_full = analysis_vars.get("chunksU_full", [])
        
        display_single_text_results(
            chunks=chunksU_full,
            deltas=deltas if deltas is not None else [],
            global_features=global_features,
            feature_stability=feature_stability,
            topic_keywords=topic_keywords,
            topicsU=topicsU,
            chunksU_full=chunksU_full,
            chunk_size=frozen_params.get("chunk_size", 2000),
            pca_marker_limit=pca_marker_limit,
        )
    else:
        # Comparative analysis
        display_comparative_results(
            result=result,
            analysis_flags=frozen_params.get("analysis_flags", {}),
            pca_marker_limit=pca_marker_limit
        )
    
    # Display methodology guide
    display_methodology()
    
    logger.info("Results display completed")


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger = config.get_logger()
        logger.error(f"Uncaught exception in main: {e}", exc_info=True)
        render_error_box(
            "An unexpected error occurred. Please check the logs and try again."
        )
