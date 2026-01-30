import numpy as np
from collections import Counter

# -----------------------------------------------------
# Craig-style preprocessing (from Craig's methodology)
# -----------------------------------------------------
PUNCTUATION = set([',', '.', '/', '?', ';', ':', '!', '-', '--', ')', '[', ']', '(', '"', '""', "'", '``'])

FUNCTION_WORDS = set([
    'a','about','above','after','again','against','all','almost','along','although','am','among','amongst','an','and','another',
    'any','anything','are','art','as','at','back','be','because','been','before','being','besides','beyond','both','but','by',
    'can','cannot','canst','could','dare','did','didst','do','does','done','dost','doth','down','durst','each','either','enough',
    'ere','even','ever','every','few','for','from','had','hadst','hath','have','he','hence','her','here','him','himself','his',
    'how','i','if','in','is','it','itself','least','like','many','may','me','might','mine','more','most','much','must','my',
    'myself','neither','never','no','none','nor','not','nothing','now','o','of','off','oft','often','on','one','only','or','other',
    'our','ourselves','out','over','own','past','perhaps','quite','rather','round','same','shall','shalt','she','should','since',
    'so','some','something','somewhat','still','such','than','that','the','thee','their','them','themselves','then','there','these',
    'they','thine','this','those','thou','thought','through','thus','thy','thyself','till','to','too','under','unto','up','upon','us',
    'very','was','we','well','were','wert','what','when','where','which','while','whilst','who','whom','whose','why','will','with',
    'within','without','ye','yet','you','your','yours','yourself','yourselves'
])

# -----------------------------------------------------
# Token cleaning (Craig-correct preprocessing)
# -----------------------------------------------------
def clean_tokens_for_craig(tokens):
    cleaned = []
    for t in tokens:
        t = t.lower().strip()
        if not t:
            continue
        if t in FUNCTION_WORDS:
            continue
        if t in PUNCTUATION:
            continue
        # Remove multi-character punctuation tokens
        if all(c in PUNCTUATION for c in t):
            continue
        cleaned.append(t)
    return cleaned

# -----------------------------------------------------
# Normalisation to ensure all chunks are strings
# -----------------------------------------------------
def normalize_chunks_to_strings(chunks):
    normalized = []
    for ch in chunks:
        if isinstance(ch, list):
            normalized.append(" ".join(ch))
        else:
            normalized.append(str(ch))
    return normalized

# -----------------------------------------------------
# Segment presence after cleaning
# -----------------------------------------------------
def compute_segment_presence(chunks):
    """
    Returns Counter: word -> number of segments (chunks) containing the word.
    Applies Craig-style cleaning to tokens.
    """
    chunks_str = normalize_chunks_to_strings(chunks)
    pres = Counter()
    for ch in chunks_str:
        raw_tokens = ch.split()
        tokens = clean_tokens_for_craig(raw_tokens)
        pres.update(set(tokens))
    return pres

# -----------------------------------------------------
# Craig coefficients
# -----------------------------------------------------
def compute_craig_coefficients(chunksA, chunksB):
    """
    Compute Craig coefficients after Craig-style cleaning.
    k = (Av / T1) + (An / T2)
    where Av = presence in A, Bv = presence in B, An = T2 - Bv.
    """
    chunksA_str = normalize_chunks_to_strings(chunksA)
    chunksB_str = normalize_chunks_to_strings(chunksB)

    T1 = max(1, len(chunksA_str))
    T2 = max(1, len(chunksB_str))

    presA = compute_segment_presence(chunksA_str)
    presB = compute_segment_presence(chunksB_str)

    vocab = set(presA.keys()) | set(presB.keys())
    coef = {}

    for w in vocab:
        Av = presA.get(w, 0)
        Bv = presB.get(w, 0)
        An = T2 - Bv
        k = (Av / T1) + (An / T2)
        coef[w] = k
    return coef

# -----------------------------------------------------
# Top markers
# -----------------------------------------------------
def top_k_markers(coef_dict, k=500, min_coef=None):
    items = sorted(coef_dict.items(), key=lambda x: x[1], reverse=True)
    if min_coef is not None:
        items = [it for it in items if it[1] >= min_coef]
    return items[:k]

# -----------------------------------------------------
# Chunk proportions of marker words
# -----------------------------------------------------
def chunk_marker_proportions(chunks, marker_set):
    """
    Returns (proportions, absolute_counts)
    Applies Craig-style cleaning so only meaningful markers remain.
    """
    chunks_str = normalize_chunks_to_strings(chunks)
    mset = set(marker_set)

    props = []
    abs_counts = []

    for ch in chunks_str:
        tokens_raw = ch.split()
        tokens = clean_tokens_for_craig(tokens_raw)
        L = len(tokens) if tokens else 1
        cnt = sum(1 for t in tokens if t in mset)
        props.append(cnt / L)
        abs_counts.append(cnt)

    return props, abs_counts

# -----------------------------------------------------
# Chi-square (used for marker tables)
# -----------------------------------------------------
def chi2_for_word(freqA, totalA, freqB, totalB):
    obs = np.array([[freqA, totalA - freqA], [freqB, totalB - freqB]], dtype=float)
    row_sums = obs.sum(axis=1)
    col_sums = obs.sum(axis=0)
    N = obs.sum()

    expected = np.outer(row_sums, col_sums) / N
    mask = expected > 0
    chi2 = np.sum(((obs - expected) ** 2 / (expected + 1e-12))[mask])
    return float(chi2)

def marker_stability_filter(
    marker_matrix,
    markers,
    variance_threshold=0.0005,
    min_chunk_presence=0.3,
    clump_ratio_threshold=0.65,
):
    """
    Filters markers based on statistical stability across text chunks.

    Parameters
    ----------
    marker_matrix : np.ndarray
        2D array (chunks x markers) containing normalized marker frequencies.
    markers : list
        List of marker labels corresponding to columns in marker_matrix.
    variance_threshold : float
        Maximum allowed variance across chunks. Lower keeps only very stable markers.
    min_chunk_presence : float
        Minimum fraction of chunks in which the marker must appear (nonzero).
        Eg. 0.3 → must appear in 30% of chunks.
    clump_ratio_threshold : float
        Maximum allowed clumping ratio: (#chunk transitions with marker) / (#total appearances).
        High clumping means the marker appears in bursts → unstable.

    Returns
    -------
    filtered_matrix : np.ndarray
        Matrix with unstable markers removed (same rows, fewer columns).
    filtered_markers : list
        Updated list of kept markers.
    dropped_markers : list
        Markers removed for instability diagnostics.
    """

    n_chunks, n_markers = marker_matrix.shape

    kept_indices = []
    dropped = []

    for i in range(n_markers):
        col = marker_matrix[:, i]

        # 1. Variance check
        var = np.var(col)
        if var > variance_threshold:
            dropped.append((markers[i], "variance"))
            continue

        # 2. Presence check
        presence = np.count_nonzero(col) / n_chunks
        if presence < min_chunk_presence:
            dropped.append((markers[i], "presence"))
            continue

        # 3. Clumping check
        nonzero = np.where(col > 0)[0]
        if len(nonzero) > 1:
            jumps = np.diff(nonzero)
            clump_ratio = np.mean(jumps > 1)  # high => clumpy usage
        else:
            clump_ratio = 1.0  # not enough points → unstable

        if clump_ratio > clump_ratio_threshold:
            dropped.append((markers[i], "clumping"))
            continue

        # keep if passed all checks
        kept_indices.append(i)

    filtered_matrix = marker_matrix[:, kept_indices]
    filtered_markers = [markers[i] for i in kept_indices]
    dropped_markers = dropped

    return filtered_matrix, filtered_markers, dropped_markers

