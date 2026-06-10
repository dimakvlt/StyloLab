"""
text_input.py - Text Input Handling for StyloLab

Handles file uploads, text paste input, and input validation.
Provides unified interface for text collection from various sources.
"""

from typing import Optional, Tuple
import hashlib
from config import log_error, log_info


def text_fingerprint(*texts: str) -> str:
    """
    Generate a SHA256 hash fingerprint of concatenated texts.
    
    Used to detect when input texts have changed, triggering
    invalidation of cached analysis results.
    
    Args:
        *texts: Variable number of text strings to hash
    
    Returns:
        SHA256 hexdigest of concatenated texts
    
    Example:
        >>> fp1 = text_fingerprint(textA, textB, textU)
        >>> fp2 = text_fingerprint(textA, textB, textU)
        >>> assert fp1 == fp2
    """
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def validate_text_input(text: str, min_length: int = 1) -> Tuple[bool, Optional[str]]:
    """
    Validate that input text meets minimum requirements.
    
    Args:
        text: Text to validate
        min_length: Minimum required text length (default: 1)
    
    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if text meets requirements
        - error_message: String describing error or None if valid
    
    Example:
        >>> is_valid, error = validate_text_input(text, min_length=50)
        >>> if not is_valid:
        ...     st.error(error)
    """
    if text is None:
        return False, "Text input is None"
    
    if not isinstance(text, str):
        return False, "Text input must be a string"
    
    stripped = text.strip()
    
    if len(stripped) < min_length:
        return False, f"Text must be at least {min_length} characters"
    
    return True, None


def validate_file_input(uploaded_file) -> Tuple[bool, Optional[str]]:
    """
    Validate that uploaded file is suitable for processing.
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        Tuple of (is_valid, error_message)
    
    Example:
        >>> is_valid, error = validate_file_input(file_obj)
        >>> if not is_valid:
        ...     st.error(error)
    """
    if uploaded_file is None:
        return True, None  # File is optional
    
    if not hasattr(uploaded_file, 'name'):
        return False, "Invalid file object"
    
    # Check file size (max 100MB)
    max_size = 100 * 1024 * 1024
    if hasattr(uploaded_file, 'size') and uploaded_file.size > max_size:
        return False, "File is too large (max 100MB)"
    
    # Check file extension
    name_lower = uploaded_file.name.lower()
    allowed_extensions = {'.txt', '.pdf', '.docx'}
    has_valid_ext = any(name_lower.endswith(ext) for ext in allowed_extensions)
    
    if not has_valid_ext:
        return False, "File type not supported (use .txt, .pdf, or .docx)"
    
    return True, None


def extract_text_from_input(
    uploaded_file,
    pasted_text: str
) -> str:
    """
    Extract text from either uploaded file or pasted text.
    
    Prioritizes pasted text if both are provided.
    Uses the improved extract_text_file from utils.processing.
    
    Args:
        uploaded_file: Streamlit uploaded file object or None
        pasted_text: Text pasted directly into text area
    
    Returns:
        Extracted text string
    
    Raises:
        ValueError: If no valid input is provided
    
    Example:
        >>> text = extract_text_from_input(file_obj, pasted_text)
        >>> st.write(f"Extracted {len(text)} characters")
    """
    from utils.processing import extract_text_file
    
    # Use pasted text if provided
    if pasted_text and pasted_text.strip():
        log_info("Using pasted text input")
        return pasted_text.strip()
    
    # Try to extract from uploaded file
    if uploaded_file is not None:
        try:
            log_info(f"Extracting text from uploaded file: {uploaded_file.name}")
            text = extract_text_file(uploaded_file, uploaded_file.name)
            
            if text.strip():
                log_info(f"Successfully extracted {len(text)} characters")
                return text.strip()
            else:
                log_error("Uploaded file appears to be empty")
                return ""
        
        except Exception as e:
            log_error(f"Failed to extract text from file: {str(e)}")
            return ""
    
    # No valid input found
    return ""


def get_text_summary(text: str, max_length: int = 100) -> str:
    """
    Generate a summary of text input (first N characters).
    
    Used for preview/summary display.
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary (default: 100)
    
    Returns:
        Summary string with ellipsis if truncated
    
    Example:
        >>> summary = get_text_summary(long_text)
        >>> st.write(f"Input preview: {summary}")
    """
    if not text:
        return ""
    
    text_stripped = text.strip()
    
    if len(text_stripped) <= max_length:
        return text_stripped
    
    return text_stripped[:max_length] + "..."


def count_tokens_estimate(text: str) -> int:
    """
    Estimate the number of tokens in text.
    
    Uses simple word-count approximation (1 token ≈ 4 characters on average).
    This is a rough estimate; actual tokenization may vary.
    
    Args:
        text: Text to estimate token count for
    
    Returns:
        Estimated number of tokens
    
    Example:
        >>> tokens = count_tokens_estimate(text)
        >>> st.write(f"Estimated {tokens} tokens")
    """
    if not text:
        return 0
    
    # Simple approximation: average word is 5 characters + 1 space
    words = text.split()
    return len(words)


def count_lines(text: str) -> int:
    """
    Count the number of lines in text.
    
    Args:
        text: Text to count lines in
    
    Returns:
        Number of lines (newline-separated)
    
    Example:
        >>> lines = count_lines(text)
        >>> st.write(f"Text has {lines} lines")
    """
    if not text:
        return 0
    
    return len(text.split('\n'))


def preprocess_text(text: str) -> str:
    """
    Preprocess text by normalizing whitespace and line endings.
    
    Args:
        text: Raw text input
    
    Returns:
        Preprocessed text
    
    Example:
        >>> cleaned = preprocess_text(raw_text)
    """
    if not text:
        return ""
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Normalize multiple spaces to single space within lines
    lines = text.split('\n')
    normalized_lines = [' '.join(line.split()) for line in lines]
    
    # Remove leading/trailing empty lines
    while normalized_lines and not normalized_lines[0].strip():
        normalized_lines.pop(0)
    while normalized_lines and not normalized_lines[-1].strip():
        normalized_lines.pop()
    
    return '\n'.join(normalized_lines)


def truncate_text(text: str, max_tokens: int = 50000) -> str:
    """
    Truncate text to maximum token count if necessary.
    
    Args:
        text: Text to potentially truncate
        max_tokens: Maximum allowed tokens (default: 50000)
    
    Returns:
        Truncated text or original if under limit
    
    Example:
        >>> text = truncate_text(very_long_text, max_tokens=10000)
    """
    if not text:
        return ""
    
    tokens = count_tokens_estimate(text)
    
    if tokens <= max_tokens:
        return text
    
    # Approximate truncation: tokens ≈ words
    words = text.split()
    truncated_words = words[:max_tokens]
    
    return ' '.join(truncated_words) + " ... [TEXT TRUNCATED]"


def text_to_words(text: str) -> list:
    """
    Convert text to list of words (simple split).
    
    Args:
        text: Text to split into words
    
    Returns:
        List of words
    
    Example:
        >>> words = text_to_words(text)
        >>> st.write(f"Text has {len(words)} words")
    """
    if not text:
        return []
    
    return text.split()


def sanitize_file_name(filename: str) -> str:
    """
    Sanitize a filename for safe use.
    
    Removes special characters and replaces with underscores.
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    
    Example:
        >>> safe_name = sanitize_file_name("my report (final).pdf")
        >>> # Returns: "my_report_final_.pdf"
    """
    import re
    
    # Keep only alphanumeric, dots, hyphens, underscores
    sanitized = re.sub(r'[^\w\.\-]', '_', filename)
    
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    return sanitized
