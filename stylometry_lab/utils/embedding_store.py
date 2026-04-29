# utils/embedding_store.py

import os
import pickle
from datetime import datetime

BASE_DIR = "app_data/embeddings"


def ensure_dir():
    os.makedirs(BASE_DIR, exist_ok=True)


# -----------------------------
# SAVE
# -----------------------------
def save_embeddings(name, chunks, embeddings, metadata=None):
    ensure_dir()

    path = os.path.join(BASE_DIR, f"{name}.pkl")

    data = {
        "name": name,
        "chunks": chunks,
        "embeddings": embeddings,
        "metadata": metadata or {},
        "created_at": datetime.utcnow().isoformat(),
    }

    with open(path, "wb") as f:
        pickle.dump(data, f)

    return path


# -----------------------------
# LOAD
# -----------------------------
def load_embeddings(name):
    path = os.path.join(BASE_DIR, f"{name}.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(f"No embedding found: {name}")

    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------
# LIST
# -----------------------------
def list_embeddings():
    ensure_dir()
    files = [f.replace(".pkl", "") for f in os.listdir(BASE_DIR) if f.endswith(".pkl")]
    return sorted(files)


# -----------------------------
# DELETE (optional)
# -----------------------------
def delete_embeddings(name):
    path = os.path.join(BASE_DIR, f"{name}.pkl")
    if os.path.exists(path):
        os.remove(path)
