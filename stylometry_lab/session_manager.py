"""
session_manager.py - Session State Management for StyloLab

Handles initialization, management, and retrieval of Streamlit session state.
Provides a clean interface for managing application state across reruns.
"""

from typing import Any, Optional, Dict, List
import streamlit as st
from config import SESSION_STATE_DEFAULTS, DEFAULT_ANALYSIS_FLAGS


def initialize_session_state() -> None:
    """
    Initialize all session state variables with default values.
    
    Runs on application startup to ensure all expected session variables exist.
    Uses lazy initialization - only creates keys that don't already exist.
    
    Session State Keys Created:
        - params: Dict containing frozen analysis parameters
        - analysis_done: Boolean flag indicating if analysis has been run
        - analysis_result: Dict with complete analysis results or None
        - text_fingerprint: Hash of input texts for change detection
        - single_pdf: Path to generated single-text PDF report
        - compare_pdf: Path to generated comparative PDF report
    
    Example:
        >>> initialize_session_state()
        >>> assert "params" in st.session_state
    """
    for key, default_value in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def ensure_session_key(key: str, default: Any) -> None:
    """
    Ensure a session state key exists, creating it if necessary.
    
    Lazy initialization for optional or dynamic session keys.
    Does nothing if key already exists.
    
    Args:
        key: Session state key name
        default: Default value if key doesn't exist
    
    Example:
        >>> ensure_session_key("custom_setting", "default_value")
        >>> assert st.session_state.custom_setting == "default_value"
    """
    if key not in st.session_state:
        st.session_state[key] = default


def get_session_value(key: str, default: Any = None) -> Any:
    """
    Get a value from session state with optional default.
    
    Args:
        key: Session state key name
        default: Default value if key doesn't exist
    
    Returns:
        Value from session state or default
    
    Example:
        >>> value = get_session_value("analysis_done", False)
    """
    return st.session_state.get(key, default)


def set_session_value(key: str, value: Any) -> None:
    """
    Set a value in session state.
    
    Args:
        key: Session state key name
        value: Value to set
    
    Example:
        >>> set_session_value("analysis_done", True)
    """
    st.session_state[key] = value


def is_analysis_done() -> bool:
    """
    Check if analysis has been completed.
    
    Returns:
        True if analysis has been run and results are cached
    
    Example:
        >>> if is_analysis_done():
        ...     display_results()
    """
    return st.session_state.get("analysis_done", False)


def set_analysis_done(done: bool) -> None:
    """
    Set the analysis completion flag.
    
    Args:
        done: Boolean indicating if analysis is complete
    
    Example:
        >>> set_analysis_done(True)
    """
    st.session_state.analysis_done = done


def get_analysis_result() -> Optional[Dict[str, Any]]:
    """
    Get the cached analysis result.
    
    Returns:
        Analysis result dictionary or None if not available
    
    Example:
        >>> result = get_analysis_result()
        >>> if result:
        ...     deltaA = result.get("deltaA")
    """
    return st.session_state.get("analysis_result")


def set_analysis_result(result: Dict[str, Any]) -> None:
    """
    Cache the analysis result in session state.
    
    Args:
        result: Complete analysis result dictionary
    
    Example:
        >>> set_analysis_result(analysis_result)
    """
    st.session_state.analysis_result = result


def get_frozen_params() -> Dict[str, Any]:
    """
    Get the frozen analysis parameters from the last run.
    
    These parameters are frozen when "Run analysis" is clicked
    and don't change until analysis is re-run.
    
    Returns:
        Dictionary of frozen parameters with values
    
    Example:
        >>> params = get_frozen_params()
        >>> chunk_size = params.get("chunk_size")
    """
    return st.session_state.get("params", {})


def set_frozen_params(params: Dict[str, Any]) -> None:
    """
    Set the frozen analysis parameters.
    
    Args:
        params: Dictionary of analysis parameters to freeze
    
    Example:
        >>> set_frozen_params({
        ...     "chunk_size": 2000,
        ...     "tokenizer": "nltk",
        ... })
    """
    st.session_state.params = params


def get_text_fingerprint() -> Optional[str]:
    """
    Get the fingerprint (hash) of the current input texts.
    
    Used to detect if input texts have changed since last analysis.
    
    Returns:
        SHA256 hash of concatenated input texts or None
    
    Example:
        >>> fp = get_text_fingerprint()
        >>> if fp != current_fp:
        ...     mark_analysis_invalid()
    """
    return st.session_state.get("text_fingerprint")


def set_text_fingerprint(fingerprint: str) -> None:
    """
    Set the fingerprint of current input texts.
    
    Args:
        fingerprint: SHA256 hash of concatenated input texts
    
    Example:
        >>> set_text_fingerprint(text_hash)
    """
    st.session_state.text_fingerprint = fingerprint


def invalidate_analysis() -> None:
    """
    Mark the current analysis as invalid.
    
    Called when inputs change to force a re-run of analysis.
    Clears cached results but preserves parameter history.
    
    Example:
        >>> invalidate_analysis()
        >>> st.stop()  # Stop execution and force rerun
    """
    st.session_state.analysis_done = False
    st.session_state.analysis_result = None


def get_pdf_path(pdf_type: str) -> Optional[str]:
    """
    Get the path to a generated PDF report.
    
    Args:
        pdf_type: Either "single" or "compare"
    
    Returns:
        File path to PDF or None if not generated
    
    Example:
        >>> path = get_pdf_path("compare")
    """
    key = "compare_pdf" if pdf_type == "compare" else "single_pdf"
    return st.session_state.get(key)


def set_pdf_path(pdf_type: str, path: str) -> None:
    """
    Set the path to a generated PDF report.
    
    Args:
        pdf_type: Either "single" or "compare"
        path: File path to the PDF
    
    Example:
        >>> set_pdf_path("compare", "/tmp/report.pdf")
    """
    key = "compare_pdf" if pdf_type == "compare" else "single_pdf"
    st.session_state[key] = path


def get_param(key: str, default: Any = None) -> Any:
    """
    Get a specific frozen parameter.
    
    Args:
        key: Parameter key name
        default: Default value if parameter not found
    
    Returns:
        Parameter value or default
    
    Example:
        >>> chunk_size = get_param("chunk_size", 2000)
    """
    params = get_frozen_params()
    return params.get(key, default)


def update_param(key: str, value: Any) -> None:
    """
    Update a specific frozen parameter.
    
    Args:
        key: Parameter key name
        value: New value for parameter
    
    Example:
        >>> update_param("chunk_size", 3000)
    """
    params = get_frozen_params()
    params[key] = value
    set_frozen_params(params)


def get_all_analysis_flags() -> Dict[str, bool]:
    """
    Get all analysis feature flags from frozen parameters.
    
    Returns:
        Dictionary of analysis flags and their boolean values
    
    Example:
        >>> flags = get_all_analysis_flags()
        >>> if flags.get("topic_modeling"):
        ...     process_topics()
    """
    params = get_frozen_params()
    return params.get("analysis_flags", {})


def get_analysis_flag(flag_name: str, default: bool = False) -> bool:
    """
    Get a specific analysis feature flag.
    
    Args:
        flag_name: Name of the flag (e.g., "craig_markers", "pca")
        default: Default value if flag not found
    
    Returns:
        Boolean value of the flag
    
    Example:
        >>> if get_analysis_flag("pca"):
        ...     perform_pca()
    """
    flags = get_all_analysis_flags()
    return flags.get(flag_name, default)


def reset_session_state() -> None:
    """
    Reset all session state to defaults.
    
    Useful for debugging or providing a "reset" button to users.
    
    Example:
        >>> if st.button("Reset Analysis"):
        ...     reset_session_state()
        ...     st.rerun()
    """
    for key, default_value in SESSION_STATE_DEFAULTS.items():
        st.session_state[key] = default_value
