
# Topic annotation schema (stable API)
# {
#   "topic": int,              # dominant topic id
#   "confidence": float,       # max topic probability
#   "distribution": list[float]
# }


import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer




def _chunks_to_strings(chunks):
    """
    Convert tokenized chunks into whitespace-joined strings.

    Parameters
    ----------
    chunks : list[list[str]]
        List of token lists (chunks)

    Returns
    -------
    list[str]
        Each chunk as a single string
    """
    return [" ".join(chunk) for chunk in chunks]



# ------------------------------------------------------------

def preprocess_for_topics(
    texts,
    language="english",
    min_word_len=3,
    extra_stopwords=None,
):

    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    stop_words = set(stopwords.words(language))
    if extra_stopwords:
        stop_words |= {w.lower() for w in extra_stopwords}

    lemmatizer = WordNetLemmatizer()

    cleaned = []
    for text in texts:
        tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        tokens = [
            lemmatizer.lemmatize(t)
            for t in tokens
            if t not in stop_words and len(t) >= min_word_len
        ]
        cleaned.append(" ".join(tokens))

    return cleaned




def train_topic_model(chunks, n_topics=10, stop_words=None):
    texts = preprocess_for_topics(
        [" ".join(ch) for ch in chunks],
        extra_stopwords=stop_words
    )

    n_docs = len(texts)

    min_df = max(1, min(int(0.02 * n_docs), 5))
    max_df = 1.0 if n_docs < 5 else 0.9

    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        token_pattern=r"(?u)\b\w+\b"
    )

    X = vectorizer.fit_transform(texts)
    if X.shape[1] < 10:
        return None

    lda = LatentDirichletAllocation(
        n_components=min(n_topics, n_docs),
        random_state=42,
        learning_method="batch"
    )

    lda.fit(X)



    print("TOPIC DEBUG")
    print("n_docs:", X.shape[0])
    print("vocab_size:", X.shape[1])
    print("nonzeros:", X.nnz)

    return {"model": lda, "vectorizer": vectorizer}





    

def apply_topic_model(topic_model, chunks):
    """
    Apply a trained topic model to a new set of chunks
    (or the same chunks).

    Parameters
    ----------
    topic_model : dict
        Object returned by train_topic_model
    chunks : list[list[str]]
        Tokenized chunks to annotate

    Returns
    -------
    list[dict]
        One dict per chunk with:
        - topic_dist       : np.ndarray of shape (n_topics,)
        - dominant_topic   : int
        - confidence       : float (max topic probability)
    """

    raw_texts = _chunks_to_strings(chunks)
    texts = preprocess_for_topics(raw_texts)

    vectorizer = topic_model["vectorizer"]
    lda = topic_model["model"]        


    X = vectorizer.transform(texts)
    topic_dists = lda.transform(X)

    results = []
    for topic_dist in topic_dists:
        topic_id = int(np.argmax(topic_dist))
        confidence = float(np.max(topic_dist))

        results.append({
            "topic": topic_id,
            "confidence": confidence,
            "distribution": topic_dist.tolist(),
        })


    return results


def get_topic_keywords(topic_model, top_n=10):
    """
    Extract top keywords for each topic.

    Parameters
    ----------
    topic_model : dict
        Trained topic model
    top_n : int
        Number of keywords per topic

    Returns
    -------
    dict
        { topic_id : [keyword1, keyword2, ...] }
    """

    lda = topic_model["model"]        
    vectorizer = topic_model["vectorizer"]

    vocab = np.array(vectorizer.get_feature_names_out())

    topic_words = {}
    for topic_id, weights in enumerate(lda.components_):
        top_idx = np.argsort(weights)[::-1][:top_n]
        topic_words[topic_id] = vocab[top_idx].tolist()

    return topic_words


def summarize_topics_per_text(topic_annotations):
    """
    Compute topic frequency distribution for a text.

    Parameters
    ----------
    topic_annotations : list[dict]
        Output of apply_topic_model for one text

    Returns
    -------
    dict
        { topic_id : proportion_of_chunks }
    """

    if not topic_annotations:
        return {}

    counts = {}
    for ann in topic_annotations:
        t = ann["topic"]
        counts[t] = counts.get(t, 0) + 1


    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def filter_chunks_by_topic(
    chunks,
    topic_annotations,
    topic_id,
    min_confidence=0.0,
):
    """
    Select chunks belonging to a given topic.

    Parameters
    ----------
    chunks : list[list[str]]
        Original chunks
    topic_annotations : list[dict]
        Topic annotations aligned with chunks
    topic_id : int
        Topic to select
    min_confidence : float
        Minimum topic probability

    Returns
    -------
    list[list[str]]
        Filtered chunks
    """

    selected = []
    for chunk, ann in zip(chunks, topic_annotations):
        if (
            ann["topic"] == topic_id
            and ann["confidence"] >= min_confidence
        ):

            selected.append(chunk)

    return selected
