# utils/pca_utils.py
import numpy as np
from sklearn.decomposition import PCA

def build_marker_vocab(markersA, markersB, pca_limit):
    vocab = list(set(list(markersA) + list(markersB)))
    if len(vocab) > pca_limit:
        vocab = vocab[:pca_limit]
    return vocab

def chunk_vectors_for_vocab(chunks_list, vocab):
    rows = []
    labels = []
    indices = []
    for label, chunks in chunks_list:
        for idx, chunk in enumerate(chunks):
            if isinstance(chunk, list):
                total = len(chunk) if chunk else 1
                row = [chunk.count(w) / total for w in vocab]
            else:
                toks = chunk.lower().split()
                total = len(toks) if toks else 1
                row = [toks.count(w) / total for w in vocab]
            rows.append(row)
            labels.append(label)
            indices.append(idx)
    return np.array(rows, dtype=float), labels, indices

def pca_on_chunk_vectors(chunksA, chunksB, chunksU, markersA, markersB, pca_marker_limit=500):
    vocab = build_marker_vocab(markersA, markersB, pca_marker_limit)
    rows, labels, indices = chunk_vectors_for_vocab([("A", chunksA), ("B", chunksB), ("U", chunksU)], vocab)
    if rows.shape[0] < 3 or rows.shape[1] < 1:
        return None
    pca = PCA(n_components=2)
    coords = pca.fit_transform(rows)
    explained = pca.explained_variance_ratio_
    loadings = pca.components_.T
    return {"coords": coords, "labels": labels, "indices": indices, "vocab": vocab, "explained_variance_ratio": explained, "loadings": loadings}
