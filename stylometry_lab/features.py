# features.py — Feature extraction & comparison 
import re
import math
from collections import Counter
import numpy as np


try:
    import nltk
    from nltk import word_tokenize, sent_tokenize, pos_tag
except Exception:
    nltk = None

if nltk:
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    except Exception as e:
        print("NLTK download warning:", e)




try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAVE_SK = True
except Exception:
    HAVE_SK = False


# ----------------------------
# Tokenization 
# ----------------------------

def safe_sent_tokenize(text):
    if nltk:
        try:
            return sent_tokenize(text)
        except Exception:
            pass
    return [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]


def safe_word_tokenize(text):
    if nltk:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"\w+['-]?\w*|\w+", text.lower())

FEATURE_WEIGHTS = {
    # Sentence structure
    "avg_sentence_length": 1.5,
    "sentence_length_std": 0.8,

    # Word structure
    "avg_word_length": 1.2,
    "short_word_ratio": 0.6,
    "long_word_ratio": 0.6,

    # Lexical richness
    "yules_k": 1.3,
    "honore": 0.7,
    "ttr": 0.3,
    "hapax_ratio": 0.3,

    # Punctuation
    "commas_per_sentence": 1.4,
    "semicolons_per_sentence": 1.0,
    "colons_per_sentence": 0.8,

    # POS
    "noun_ratio": 1.2,
    "verb_ratio": 1.3,
    "adj_ratio": 0.7,
    "adv_ratio": 0.7,
    "pron_ratio": 1.2,
}

# ----------------------------
# Feature extraction
# ----------------------------

def extract_features(text):
    text = text.strip()
    if not text:
        return {}

    sents = safe_sent_tokenize(text)
    tokens = safe_word_tokenize(text)
    N = len(tokens)
    if N == 0:
        return {}

    # ---------- Sentence-level ----------
    sent_lengths = [len(s.split()) for s in sents if s.strip()]
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    sent_len_std = float(np.std(sent_lengths)) if sent_lengths else 0.0
    median_sent_len = float(np.median(sent_lengths)) if sent_lengths else 0.0

    # ---------- Word-level ----------
    word_lengths = [len(t) for t in tokens]
    avg_word_len = float(np.mean(word_lengths))
    word_len_std = float(np.std(word_lengths))

    short_word_ratio = sum(1 for t in tokens if len(t) <= 3) / N
    long_word_ratio = sum(1 for t in tokens if len(t) >= 8) / N

    # ---------- Lexical richness ----------
    counts = Counter(tokens)
    types = len(counts)
    ttr = types / N

    hapax = sum(1 for v in counts.values() if v == 1)
    dislegomena = sum(1 for v in counts.values() if v == 2)

    hapax_ratio = hapax / N
    dislegomena_ratio = dislegomena / N

    M1 = N
    M2 = sum(freq * (freq ** 2) for freq in counts.values())
    yules_k = 10000 * (M2 - M1) / (M1 * M1) if M1 > 0 else 0.0

    try:
        R = hapax / max(1, types)
        honore = 100 * math.log(N) / max(1e-8, (1 - R))
    except Exception:
        honore = 0.0

    # ---------- Punctuation ----------
    comma_count = text.count(",")
    semicolon_count = text.count(";")
    colon_count = text.count(":")
    exclam_count = text.count("!")
    question_count = text.count("?")
    dash_count = text.count("—") + text.count("--")

    n_sents = max(1, len(sents))

    commas_per_sentence = comma_count / n_sents
    semicolons_per_sentence = semicolon_count / n_sents
    colons_per_sentence = colon_count / n_sents

    exclamation_ratio = exclam_count / N
    question_ratio = question_count / N
    dash_ratio = dash_count / N

    # ---------- POS (aggregated) ----------
    pos_ratios = {
        "noun_ratio": 0.0,
        "verb_ratio": 0.0,
        "adj_ratio": 0.0,
        "adv_ratio": 0.0,
        "pron_ratio": 0.0,
    }

    if nltk:
        try:
            tags = pos_tag(tokens)
            tag_counts = Counter(tag for (_, tag) in tags)
            total = sum(tag_counts.values()) or 1

            noun_tags = {"NN","NNS","NNP","NNPS"}
            verb_tags = {"VB","VBD","VBG","VBN","VBP","VBZ"}
            adj_tags  = {"JJ","JJR","JJS"}
            adv_tags  = {"RB","RBR","RBS"}
            pron_tags = {"PRP","PRP$","WP","WP$"}

            pos_ratios["noun_ratio"] = sum(tag_counts[t] for t in noun_tags) / total
            pos_ratios["verb_ratio"] = sum(tag_counts[t] for t in verb_tags) / total
            pos_ratios["adj_ratio"]  = sum(tag_counts[t] for t in adj_tags) / total
            pos_ratios["adv_ratio"]  = sum(tag_counts[t] for t in adv_tags) / total
            pos_ratios["pron_ratio"] = sum(tag_counts[t] for t in pron_tags) / total

        except Exception as e:
            print("POS tagging failed:", e)



    return {
        # Size
        "n_tokens": N,
        "n_types": types,

        # Sentence structure
        "avg_sentence_length": avg_sent_len,
        "sentence_length_std": sent_len_std,
        "median_sentence_length": median_sent_len,

        # Word structure
        "avg_word_length": avg_word_len,
        "word_length_std": word_len_std,
        "short_word_ratio": short_word_ratio,
        "long_word_ratio": long_word_ratio,

        # Lexical richness
        "ttr": ttr,
        "hapax_ratio": hapax_ratio,
        "dislegomena_ratio": dislegomena_ratio,
        "yules_k": yules_k,
        "honore": honore,

        # Punctuation style
        "commas_per_sentence": commas_per_sentence,
        "semicolons_per_sentence": semicolons_per_sentence,
        "colons_per_sentence": colons_per_sentence,
        "exclamation_ratio": exclamation_ratio,
        "question_ratio": question_ratio,
        "dash_ratio": dash_ratio,

        # POS summary
        "noun_ratio": pos_ratios["noun_ratio"],
        "verb_ratio": pos_ratios["verb_ratio"],
        "adj_ratio": pos_ratios["adj_ratio"],
        "adv_ratio": pos_ratios["adv_ratio"],
        "pron_ratio": pos_ratios["pron_ratio"],
    }


# ----------------------------
# Feature vector comparison
# ----------------------------

def profile_author_chunks(chunks):
    rows = []
    for c in chunks:
        f = extract_features(" ".join(c))
        rows.append(f)
    df = pd.DataFrame(rows)
    return df


def extract_features_by_chunk(chunks):
    """
    chunks: list[list[str]]
    returns: dict[feature -> list of values per chunk]
    """
    feature_series = {}

    for chunk in chunks:
        text = " ".join(chunk)
        f = extract_features(text)
        for k, v in f.items():
            if isinstance(v, (int, float)):
                feature_series.setdefault(k, []).append(v)

    return feature_series

def compute_feature_stability(feature_series):
    """
    returns: dict[feature -> stability weight in [0,1]]
    """
    stability = {}

    for k, values in feature_series.items():
        if len(values) < 2:
            stability[k] = 0.0
            continue

        mean = np.mean(values)
        std = np.std(values)

        if mean == 0:
            stability[k] = 0.0
        else:
            stability[k] = max(0.0, 1.0 - min(1.0, std / abs(mean)))

    return stability

def compute_stability(df):
    stats = df.describe().T
    stats["stability"] = np.where(
        stats["mean"] > 1e-6,
        1 - (stats["std"] / stats["mean"]),
        0.0
    )
    return stats[["mean", "std", "stability"]]



def compare_feature_vectors_stable(fA, fB, fU, stability):
    keys = [
        k for k in FEATURE_WEIGHTS
        if k in fA and k in stability
    ]

    def vec(f):
        return np.array([
            f[k] * FEATURE_WEIGHTS[k] * stability[k]
            for k in keys
        ], dtype=float)

    vA, vB, vU = vec(fA), vec(fB), vec(fU)

    return (
        float(np.linalg.norm(vU - vA)),
        float(np.linalg.norm(vU - vB)),
        keys
    )


def craig_pos_profile(features):
    return {
        "noun_vs_verb": features["noun_ratio"] - features["verb_ratio"],
        "modifier_density": features["adj_ratio"] + features["adv_ratio"],
        "function_pressure": features["pron_ratio"] + features["verb_ratio"],
    }
def craig_distance(fA, fB, fU):
    A = craig_pos_profile(fA)
    B = craig_pos_profile(fB)
    U = craig_pos_profile(fU)

    def dist(X, Y):
        return sum(abs(X[k] - Y[k]) for k in X)

    return dist(U, A), dist(U, B)

def explain_decision(fA, fB, fU):
    explanation = []

    for k, w in FEATURE_WEIGHTS.items():
        if k not in fU:
            continue

        dA = abs(fU[k] - fA[k])
        dB = abs(fU[k] - fB[k])

        if abs(dA - dB) > 0.02:
            closer = "Author A" if dA < dB else "Author B"
            explanation.append(
                f"{k}: closer to {closer} "
                f"(U={fU[k]:.3f}, A={fA[k]:.3f}, B={fB[k]:.3f})"
            )

    return explanation[:8]  # keep it readable

def extract_features_by_chunk(chunks):
    """
    chunks: list[list[str]]
    returns: dict[feature -> list of values per chunk]
    """
    feature_series = {}

    for chunk in chunks:
        text = " ".join(chunk)
        f = extract_features(text)
        for k, v in f.items():
            if isinstance(v, (int, float)):
                feature_series.setdefault(k, []).append(v)

    return feature_series


# ----------------------------
# TF–IDF similarity
# ----------------------------

def tfidf_similarity(textA, textB, textU):
    corpus = [textA, textB, textU]

    if HAVE_SK:
        try:
            vec = TfidfVectorizer(min_df=2, stop_words="english")
            X = vec.fit_transform(corpus).toarray()
            A, B, U = X

            def cos(a, b):
                den = np.linalg.norm(a) * np.linalg.norm(b)
                return float(np.dot(a, b) / den) if den else 0.0

            return cos(U, A), cos(U, B)
        except Exception:
            pass

    return 0.0, 0.0


# ----------------------------
# Final comparison
# ----------------------------

def compare_texts(textA, textB, textU):
    fA = extract_features(textA)
    fB = extract_features(textB)
    fU = extract_features(textU)

    if not fA or not fB or not fU:
        return {"similarity": {}, "verdict": "inconclusive (missing features)"}

    # compute chunk-level stability from UNKNOWN text (most honest)
    tokensU = safe_word_tokenize(textU)
    chunksU = [
        tokensU[i:i+1000]
        for i in range(0, len(tokensU), 1000)
    ]

    feature_series = extract_features_by_chunk(chunksU)
    stability = compute_feature_stability(feature_series)

    dA, dB, used_features = compare_feature_vectors_stable(
        fA, fB, fU, stability
    )

    simA, simB = tfidf_similarity(textA, textB, textU)
    cA, cB = craig_distance(fA, fB, fU)

    scoreA = (-dA * 0.5) + (simA * 0.2) - (cA * 0.3)
    scoreB = (-dB * 0.5) + (simB * 0.2) - (cB * 0.3)
    explanation = explain_decision(fA, fB, fU)


    if abs(scoreA - scoreB) < 0.01:
        verdict = "inconclusive"
    elif scoreA > scoreB:
        verdict = "Author A"
    else:
        verdict = "Author B"

    return {
        "similarity": {
            "feature_distance_A": dA,
            "feature_distance_B": dB,
            "tfidf_cosine_A": simA,
            "tfidf_cosine_B": simB,
            "score_A": scoreA,
            "score_B": scoreB,
            "craig_distance_A": cA,
            "craig_distance_B": cB,
        },
        "verdict": verdict,
        "explanation": explanation,
        "features": {
            "Author A": fA,
            "Author B": fB,
            "Unknown": fU,
        },
        "feature_stability": {
            k: stability.get(k, 0.0)
            for k in used_features
        },
        "raw_features": {
            k: {"A": fA.get(k), "B": fB.get(k), "U": fU.get(k)}
            for k in used_features
        },


    }
    print("Craig distances:", cA, cB)
    print("Weighted distances:", dA, dB)
    print("Explanation:", explanation[:3])




# ------------------------------------------------------------
# Chunk-level feature stability (single-text analysis)
# ------------------------------------------------------------

import numpy as np

def chunk_feature_stability(text, chunk_size=2000):
    """
    Robust stability estimation for stylistic features
    across chunks of a single text.

    Returns:
        dict: {feature_name: stability_score in [0, 1]}
    """

    from utils.processing import clean_and_tokenize, chunk_words

    tokens = clean_and_tokenize(text)
    chunks = chunk_words(tokens, chunk_size)

    if len(chunks) < 2:
        return {}

    # Extract features per chunk
    per_chunk = []
    for chunk in chunks:
        chunk_text = " ".join(chunk)
        feats = extract_features(chunk_text)
        per_chunk.append(feats)

    feature_names = [
        k for k, v in per_chunk[0].items()
        if isinstance(v, (int, float))
    ]

    stability = {}

    for fname in feature_names:
        values = np.array([f.get(fname, 0.0) for f in per_chunk])

        nonzero = values[values != 0]

        # Feature never appears → ignore
        if len(nonzero) == 0:
            continue

        mean = nonzero.mean()
        std = nonzero.std()

        # Robust coefficient of variation
        cv = std / (abs(mean) + 1e-8)

        # Presence ratio (important!)
        presence = len(nonzero) / len(values)

        # Combine stability components
        stability_score = np.exp(-cv) * presence

        stability[fname] = float(stability_score)

    return stability




