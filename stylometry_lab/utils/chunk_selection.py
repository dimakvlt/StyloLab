# utils/chunk_selection.py
"""
Chunk selection utilities for Stylometry Lab.

This module defines a generic, extensible layer that decides
which chunks are passed to downstream analyses.

Design principles:
- Analysis-agnostic
- Topic-agnostic (topics are just one selector)
- Safe defaults (no selection = all chunks)
"""

from typing import List, Optional


# ------------------------------------------------------------
# Core selector
# ------------------------------------------------------------

def select_chunks(
    chunks: List[list],
    *,
    mode: str = "all",
    topic_annotations: Optional[List[dict]] = None,
    topic_id: Optional[int] = None,
    min_topic_confidence: float = 0.0,
):
    """
    Select chunks according to user-defined rules.

    Parameters
    ----------
    chunks : list[list[str]]
        Tokenized chunks
    mode : str
        Selection mode:
        - "all"
        - "topic"
    topic_annotations : list[dict], optional
        Topic annotations aligned with chunks
    topic_id : int, optional
        Topic to select
    min_topic_confidence : float
        Minimum topic probability

    Returns
    -------
    list[list[str]]
        Selected chunks
    """

    if mode == "all" or topic_annotations is None:
        return chunks

    if mode == "topic":
        if topic_id is None:
            return chunks

        selected = []
        for chunk, ann in zip(chunks, topic_annotations):
            if (
                ann["dominant_topic"] == topic_id
                and ann["confidence"] >= min_topic_confidence
            ):
                selected.append(chunk)
        return selected

    # Fallback safety
    return chunks


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

def selection_summary(
    total_chunks: int,
    selected_chunks: int,
    mode: str,
):
    """
    Produce a human-readable summary of selection.

    Returns
    -------
    str
    """
    if mode == "all":
        return f"All chunks used ({selected_chunks}/{total_chunks})."

    return (
        f"Selected {selected_chunks} of {total_chunks} chunks "
        f"using mode '{mode}'."
    )
