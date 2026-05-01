# utils/embeddings_pipeline.py

import numpy as np
from utils.embeddings import embed_chunks, load_model, mean_similarity


def run_embedding_pipeline(chunksA, chunksB, chunksU):
    """
    Full embedding workflow:
    - loads model
    - embeds chunks
    - computes similarities
    """

    model = load_model()

    embA = embed_chunks(chunksA, model) if chunksA else None
    embB = embed_chunks(chunksB, model) if chunksB else None
    embU = embed_chunks(chunksU, model) if chunksU else None

    sim_A = mean_similarity(embU, embA) if embA is not None else None
    sim_B = mean_similarity(embU, embB) if embB is not None else None

    return {
        "embA": embA,
        "embB": embB,
        "embU": embU,
        "sim_A": sim_A,
        "sim_B": sim_B,
    }


def compute_centroids(embA, embB, embU):
    """
    Compute embedding centroids for visualization
    """
    centroid_A = embA.mean(axis=0) if embA is not None else None
    centroid_B = embB.mean(axis=0) if embB is not None else None
    centroid_U = embU.mean(axis=0) if embU is not None else None

    return centroid_A, centroid_B, centroid_U


def compute_distances(centroid_A, centroid_B, centroid_U):
    """
    Euclidean distances in embedding space
    """
    dist_A = np.linalg.norm(centroid_U - centroid_A)
    dist_B = np.linalg.norm(centroid_U - centroid_B)

    return dist_A, dist_B