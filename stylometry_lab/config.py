"""
config.py - Configuration and Constants for StyloLab

This module contains all configurable constants and settings used throughout
the application, including logging configuration, UI settings, and defaults.
"""

import logging
import os
from typing import Dict, Any

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGFILE: str = "stylometry.log"

def setup_logging() -> logging.Logger:
    """
    Configure and return the application logger.
    
    Sets up both file and console logging with appropriate formatters
    and log levels. File logging captures DEBUG level, console shows INFO.
    
    Returns:
        logging.Logger: Configured logger instance for the application
    
    Example:
        >>> logger = setup_logging()
        >>> logger.info("Application started")
    """
    logger = logging.getLogger("stylometry_app")
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(LOGFILE)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


# ============================================================================
# PAGE & UI CONFIGURATION
# ============================================================================

PAGE_TITLE: str = "Stylometry Lab"
PAGE_LAYOUT: str = "wide"

CUSTOM_CSS: str = """
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
"""

# ============================================================================
# ANALYSIS MODE OPTIONS
# ============================================================================

ANALYSIS_MODES: list = [
    "Compare authors (A vs B vs Unknown)",
    "Single text (internal analysis)"
]

# ============================================================================
# DEFAULT PARAMETERS
# ============================================================================

DEFAULT_CHUNK_SIZE: int = 2000
DEFAULT_TOP_K: int = 50
DEFAULT_N_TOPICS: int = 5
DEFAULT_MIN_COEF: float = 0.0
DEFAULT_VARIANCE_THRESHOLD: float = 0.1
DEFAULT_MIN_CHUNK_PRESENCE: int = 2
DEFAULT_CLUMP_RATIO_THRESHOLD: float = 0.7
DEFAULT_PCA_MARKER_LIMIT: int = 100
DEFAULT_TOKENIZER: str = "nltk"
DEFAULT_CHUNK_MODE: str = "All chunks"
DEFAULT_TOPIC_USAGE: str = "Exploratory (no filtering)"

# ============================================================================
# SIDEBAR PARAMETER RANGES
# ============================================================================

CHUNK_SIZE_RANGE: tuple = (10, 5000)
TOP_K_RANGE: tuple = (5, 200)
N_TOPICS_RANGE: tuple = (2, 50)
PCA_MARKER_LIMIT_RANGE: tuple = (10, 500)
VARIANCE_THRESHOLD_RANGE: tuple = (0.0, 1.0)
MIN_CHUNK_PRESENCE_RANGE: tuple = (1, 20)
CLUMP_RATIO_THRESHOLD_RANGE: tuple = (0.0, 1.0)

# ============================================================================
# SESSION STATE DEFAULTS
# ============================================================================

SESSION_STATE_DEFAULTS: Dict[str, Any] = {
    "params": {},
    "analysis_done": False,
    "analysis_result": None,
    "text_fingerprint": None,
    "single_pdf": None,
    "compare_pdf": None,
}

# ============================================================================
# VARIABLE DEFAULTS (Used during analysis)
# ============================================================================

DEFAULT_VARIABLES: Dict[str, Any] = {
    "num_rows": None,
    "rowsA": None,
    "rowsB": None,
    "rowsU": None,
    "df_feat": None,
    "fig_craig": None,
    "pca_result": None,
    "selected_topics": [],
    "min_conf": 0.0,
    "custom_stopwords": set(),
}

# ============================================================================
# ANALYSIS FLAGS (Features to enable/disable)
# ============================================================================

DEFAULT_ANALYSIS_FLAGS: Dict[str, bool] = {
    "craig_markers": True,
    "pca": True,
    "topic_modeling": False,
    "embeddings": False,
}

# ============================================================================
# TOKENIZER OPTIONS
# ============================================================================

TOKENIZER_OPTIONS: list = ["whitespace", "regex", "nltk", "spacy"]

# ============================================================================
# CHUNK MODE OPTIONS
# ============================================================================

CHUNK_MODES: list = [
    "All chunks",
    "First N chunks",
    "Last N chunks",
    "Random N chunks"
]

# ============================================================================
# TOPIC USAGE OPTIONS
# ============================================================================

TOPIC_USAGE_OPTIONS: list = [
    "Exploratory (no filtering)",
    "Filter chunks for analysis",
    "Control for topic (authorship within topic)"
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_logger() -> logging.Logger:
    """
    Get or create the application logger.
    
    Returns:
        logging.Logger: The configured logger instance
    """
    return logging.getLogger("stylometry_app")


def log_warning(msg: str) -> None:
    """
    Log a warning message to the application logger.
    
    Args:
        msg: Warning message to log
    
    Example:
        >>> log_warning("Unexpected behavior detected")
    """
    logger = get_logger()
    logger.warning(msg)


def log_info(msg: str) -> None:
    """
    Log an info message to the application logger.
    
    Args:
        msg: Info message to log
    """
    logger = get_logger()
    logger.info(msg)


def log_debug(msg: str) -> None:
    """
    Log a debug message to the application logger.
    
    Args:
        msg: Debug message to log
    """
    logger = get_logger()
    logger.debug(msg)


def log_error(msg: str) -> None:
    """
    Log an error message to the application logger.
    
    Args:
        msg: Error message to log
    """
    logger = get_logger()
    logger.error(msg)
