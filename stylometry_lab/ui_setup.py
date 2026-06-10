"""
ui_setup.py - UI Configuration and Styling for StyloLab

Handles Streamlit page setup, custom CSS styling, and UI initialization.
Centralizes all UI configuration to ensure consistent appearance.
"""

from typing import Tuple
import streamlit as st
from config import PAGE_TITLE, PAGE_LAYOUT, CUSTOM_CSS


def setup_page_config() -> None:
    """
    Configure Streamlit page settings.
    
    Sets page title, layout, and other page-level configuration.
    Should be called early in the application (before any st.write calls).
    
    Example:
        >>> setup_page_config()
        >>> st.write("Welcome to StyloLab")
    """
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout=PAGE_LAYOUT
    )


def apply_custom_styling() -> None:
    """
    Apply custom CSS styling to the application.
    
    Enhances button appearance and overall UI aesthetics.
    Must be called after setup_page_config().
    
    Example:
        >>> apply_custom_styling()
    """
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header() -> None:
    """
    Render the application header and title.
    
    Displays the main title and initializes the page layout.
    
    Example:
        >>> render_header()
    """
    st.title("StyloLab")


def initialize_ui() -> None:
    """
    Initialize all UI components and styling.
    
    Coordinates page setup, styling, and header rendering.
    Should be called once at the start of the application.
    
    Example:
        >>> initialize_ui()
    """
    setup_page_config()
    apply_custom_styling()
    render_header()


def render_section_header(title: str, emoji: str = "📊") -> None:
    """
    Render a section header with emoji.
    
    Args:
        title: Title text for the section
        emoji: Emoji to prefix the title (default: 📊)
    
    Example:
        >>> render_section_header("Analysis Results", "📈")
    """
    st.header(f"{emoji} {title}")


def render_subheader(title: str, emoji: str = "") -> None:
    """
    Render a subheader with optional emoji.
    
    Args:
        title: Title text for the subheader
        emoji: Optional emoji to prefix the title
    
    Example:
        >>> render_subheader("Burrows Delta", "📊")
    """
    if emoji:
        st.subheader(f"{emoji} {title}")
    else:
        st.subheader(title)


def render_divider() -> None:
    """
    Render a visual divider line.
    
    Example:
        >>> st.write("Section 1")
        >>> render_divider()
        >>> st.write("Section 2")
    """
    st.divider()


def render_info_box(message: str) -> None:
    """
    Render an informational message box.
    
    Args:
        message: Information message to display
    
    Example:
        >>> render_info_box("Analysis is complete!")
    """
    st.info(message)


def render_success_box(message: str) -> None:
    """
    Render a success message box.
    
    Args:
        message: Success message to display
    
    Example:
        >>> render_success_box("Analysis completed successfully!")
    """
    st.success(message)


def render_warning_box(message: str) -> None:
    """
    Render a warning message box.
    
    Args:
        message: Warning message to display
    
    Example:
        >>> render_warning_box("Please provide all required inputs")
    """
    st.warning(message)


def render_error_box(message: str) -> None:
    """
    Render an error message box.
    
    Args:
        message: Error message to display
    
    Example:
        >>> render_error_box("Analysis failed: insufficient data")
    """
    st.error(message)


def render_markdown(text: str, unsafe_html: bool = False) -> None:
    """
    Render markdown-formatted text.
    
    Args:
        text: Markdown-formatted text
        unsafe_html: Whether to allow raw HTML (default: False)
    
    Example:
        >>> render_markdown("**Bold text** and *italic text*")
    """
    st.markdown(text, unsafe_allow_html=unsafe_html)


def render_caption(text: str) -> None:
    """
    Render a caption (small, muted text).
    
    Args:
        text: Caption text to display
    
    Example:
        >>> render_caption("This is a small caption")
    """
    st.caption(text)


def create_columns(num_columns: int) -> Tuple:
    """
    Create a layout with multiple columns.
    
    Args:
        num_columns: Number of columns to create
    
    Returns:
        Tuple of column objects
    
    Example:
        >>> left, right = create_columns(2)
        >>> with left:
        ...     st.write("Left column")
        >>> with right:
        ...     st.write("Right column")
    """
    return st.columns(num_columns)


def create_tabs(tab_names: list) -> Tuple:
    """
    Create tabs for organizing content.
    
    Args:
        tab_names: List of tab names
    
    Returns:
        Tuple of tab objects
    
    Example:
        >>> tab1, tab2 = create_tabs(["Results", "Debug"])
        >>> with tab1:
        ...     st.write("Results")
        >>> with tab2:
        ...     st.write("Debug info")
    """
    return st.tabs(tab_names)


def create_expander(label: str, expanded: bool = False):
    """
    Create an expandable section.
    
    Args:
        label: Label for the expander
        expanded: Whether to start expanded (default: False)
    
    Returns:
        Expander context manager
    
    Example:
        >>> with create_expander("Advanced Options"):
        ...     st.write("Hidden by default")
    """
    return st.expander(label, expanded=expanded)


def show_spinner(message: str = "Loading..."):
    """
    Show a loading spinner while performing operations.
    
    Args:
        message: Message to display while loading
    
    Returns:
        Context manager for spinner
    
    Example:
        >>> with show_spinner("Computing analysis..."):
        ...     result = run_heavy_computation()
    """
    return st.spinner(message)


def stop_execution() -> None:
    """
    Stop execution and prevent further code from running.
    
    Useful for validation checks that should halt processing.
    
    Example:
        >>> if not has_valid_input():
        ...     st.error("Invalid input")
        ...     stop_execution()
    """
    st.stop()


def trigger_rerun() -> None:
    """
    Trigger a full application rerun.
    
    Forces Streamlit to rerun from the beginning.
    
    Example:
        >>> if st.button("Refresh"):
        ...     trigger_rerun()
    """
    st.rerun()
