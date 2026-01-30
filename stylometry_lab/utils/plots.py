
import matplotlib.pyplot as plt
import numpy as np

def plot_scatter_positions(posA_x, posA_y, posB_x, posB_y, posU_x, posU_y):
    fig, ax = plt.subplots(figsize=(7,6))

    if posA_x:
        ax.scatter(posA_x, posA_y, marker='^', label='Author A', alpha=0.8)
    if posB_x:
        ax.scatter(posB_x, posB_y, marker='^', label='Author B', alpha=0.8)
    if posU_x:
        ax.scatter(posU_x, posU_y, marker='o', label='Unknown',
                   edgecolors='w', s=70)

    ax.set_xlabel("Prop A markers")
    ax.set_ylabel("Prop B markers")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    return fig


def plot_top_freq_diffs(pairs, log_scale=False):
    if not pairs:
        return None

    labels = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(8,4))

    if log_scale:
        signed_log = [np.sign(v) * np.log1p(abs(v)) for v in vals]
        ax.bar(labels, signed_log)
    else:
        ax.bar(labels, vals)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)

    return fig


def plot_pca_chunk_scatter(pca_result, title="PCA on chunk marker vectors"):
    if pca_result is None:
        return None

    coords = pca_result["coords"]
    labels = pca_result["labels"]

    fig, ax = plt.subplots(figsize=(8, 6))

    xs_A = coords[[i for i, l in enumerate(labels) if l == "A"], 0]
    ys_A = coords[[i for i, l in enumerate(labels) if l == "A"], 1]
    xs_B = coords[[i for i, l in enumerate(labels) if l == "B"], 0]
    ys_B = coords[[i for i, l in enumerate(labels) if l == "B"], 1]
    xs_U = coords[[i for i, l in enumerate(labels) if l == "U"], 0]
    ys_U = coords[[i for i, l in enumerate(labels) if l == "U"], 1]

    if xs_A.size:
        ax.scatter(xs_A, ys_A, marker="^", label="Author A")
    if xs_B.size:
        ax.scatter(xs_B, ys_B, marker="^", label="Author B")
    if xs_U.size:
        ax.scatter(xs_U, ys_U, marker="o", label="Unknown", edgecolors="w")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig


