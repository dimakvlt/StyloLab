from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from datetime import datetime
from reportlab.platypus import Image
from reportlab.lib.units import inch
import os
import tempfile
import matplotlib.pyplot as plt





def _header(text, styles):
    return Paragraph(f"<b>{text}</b>", styles["Heading2"])


def _para(text, styles):
    return Paragraph(text, styles["BodyText"])


def _spacer(h=0.4):
    return Spacer(1, h * cm)


def _table(data):
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl

def save_fig_temp(fig, filename):
    if fig is None:
        return None
    path = os.path.join(tempfile.gettempdir(), filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path

def explain(title, text, styles):
    blocks = []
    blocks.append(Spacer(1, 12))
    blocks.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
    blocks.append(Spacer(1, 6))
    blocks.append(Paragraph(text, styles["BodyText"]))
    blocks.append(Spacer(1, 12))
    return blocks


# ============================================================
# SINGLE-TEXT REPORT
# ============================================================
def build_single_text_report(
    filepath,
    deltas,
    global_features,
    feature_stability,
    params,
    figs=None,
):

    feature_stability = feature_stability or {}

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    story = []

    story.append(Paragraph("<b>Stylometry Lab — Single Text Report</b>", styles["Title"]))
    story.append(_spacer())

    story.append(_para(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles
    ))
    story.append(_spacer())
    story.append(_header("Analyses included in this report", styles))
    story.append(_para(
        "This report combines multiple independent stylometric methods. "
        "Not all analyses are always active; only enabled analyses are shown below.",
        styles
    ))

    story.append(_header("Analysis parameters", styles))
    story.append(_para(
        f"Chunk size: {params['chunk_size']} words", styles
    ))

    story.append(_spacer())

    

    import numpy as np

    delta_arr = np.array(deltas)

    summary_table = [
        ["Metric", "Value"],
        ["Number of chunks", str(len(deltas))],
        ["Mean delta", f"{delta_arr.mean():.4f}"],
        ["Median delta", f"{np.median(delta_arr):.4f}"],
        ["Standard deviation", f"{delta_arr.std():.4f}"],
        ["Minimum delta", f"{delta_arr.min():.4f}"],
        ["Maximum delta", f"{delta_arr.max():.4f}"],
    ]

    story.append(_header("Internal stylistic consistency", styles))
    story.append(_para(
        "This section summarizes how consistently the text adheres to a single "
        "stylistic profile across its length. Delta values measure how much individual "
        "chunks deviate from the overall style.\n\n"
        "Rather than listing all chunks, the report presents aggregate statistics and "
        "highlights the most stylistically atypical sections.",
        styles
    ))

    story.append(_table(summary_table))
    story.append(_spacer())

    TOP_N = min(10, len(deltas))

    top_chunks = sorted(
        enumerate(deltas),
        key=lambda x: x[1],
        reverse=True
    )[:TOP_N]

    story.append(_header("Most stylistically deviant chunks", styles))
    story.append(_para(
        f"The following table lists the {TOP_N} chunks with the highest internal "
        "delta values. These sections deviate most strongly from the dominant "
        "stylistic profile and may warrant closer inspection.",
        styles
    ))

    top_table = [["Chunk", "Delta"]] + [
        [f"C{i}", f"{d:.4f}"] for i, d in top_chunks
    ]

    story.append(_table(top_table))
    story.append(_para(
        "Full chunk-level diagnostics are available in the interactive application "
        "interface and are omitted here for clarity.",
        styles
    ))



    story.append(PageBreak())

    story.append(_header("Global stylistic features", styles))
    story.append(_para(
        "The following table reports global stylistic features computed over the "
        "entire text. These features capture structural and stylistic properties such "
        "as lexical richness, sentence structure, punctuation usage, and syntactic balance.\n\n"
        "They are designed to reflect *how* the text is written rather than *what* it "
        "is about, and are relatively robust to topic variation.",
        styles
    ))

    feat_table = [["Feature", "Value"]] + [
        [k, f"{v:.4f}"] for k, v in global_features.items()
        if isinstance(v, (int, float))
    ]
    story.append(_table(feat_table))

    story.append(_spacer())

    story.append(_header("Feature reliability across chunks", styles))
    story.append(_para(
        "Feature reliability measures how consistently each stylistic feature appears "
        "across different chunks of the same text. It is derived from the variability "
        "of feature values across chunks and normalized to a 0–1 scale.\n\n"
        "Higher reliability values indicate stable, habitual stylistic behavior, while "
        "lower values suggest features that fluctuate due to local context, topic, or "
        "structural variation.",
        styles
    ))

    stab_table = [["Feature", "Reliability"]] + [
        [k, f"{v:.3f}"] for k, v in sorted(
            feature_stability.items(), key=lambda x: x[1], reverse=True
        )
    ]
    story.append(_table(stab_table))
    if figs and "pca" in figs:
        pca_img = None
        if figs and figs.get("pca") is not None:
            pca_img = save_fig_temp(figs["pca"], "single_pca.png")

        if pca_img:
            img = Image(pca_img, width=6*inch, height=4*inch)
            story.append(img)



    doc.build(story)


# ============================================================
# COMPARATIVE REPORT
# ============================================================
def build_comparative_report(
    filepath,
    deltaA,
    deltaB,
    verdict,
    feature_similarity,
    feature_stability,
    params,
    figs=None,
    topic_modeling=None,
    explanation=None,
    craig_numeric_U=None,
    markers_A=None,
    markers_B=None,
    markers_U=None,
    feature_table=None,
):


    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    story = []

    story.append(Paragraph("<b>Stylometry Lab — Authorship Comparison Report</b>", styles["Title"]))
    story.append(_spacer())

    story.append(_para(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles
    ))

    story.append(_spacer())
    story.append(_header("Analyses included in this report", styles))
    story.append(_para(
        "This report combines multiple independent stylometric methods. "
        "Not all analyses are always active; only enabled analyses are shown below.",
        styles
    ))

    story.append(_header("Analysis parameters", styles))
    story.append(_para(
        f"Chunk size: {params['chunk_size']} words<br/>"
        f"Top-K marker words: {params['top_k']}",
        styles
    ))

    story.append(_spacer())

    story.append(_header("Burrows' Delta results", styles))
    story.append(_para(
        "Burrows’ Delta is a distance measure based on standardized word frequency "
        "differences between texts. Lower values indicate stronger stylistic similarity.\n\n"
        "The table below reports the stylistic distance between the Unknown text and "
        "each reference author. Differences should be interpreted probabilistically, "
        "not as definitive authorship attribution.",
        styles
    ))


    delta_table = [
        ["Comparison", "Delta"],
        ["Unknown → Author A", f"{deltaA:.4f}"],
        ["Unknown → Author B", f"{deltaB:.4f}"],
    ]
    story.append(_table(delta_table))

    story.append(_spacer())

    story.append(_header("Feature-based verdict", styles))
    story.append(_para(
        "The following verdict summarizes the comparative feature-based analysis. "
        "It is derived from weighted distances in a multi-dimensional stylistic feature "
        "space and reflects overall stylistic proximity rather than isolated indicators.",
        styles
    ))

    story.append(_para(verdict, styles))
        # ------------------------------------------------------------
    # Feature-based explanation 
    # ------------------------------------------------------------
    if explanation:
        story.append(_spacer())
        story.append(_header("Why this decision? (Feature explanation)", styles))

        story.append(_para(
            "The following stylistic features were both sufficiently stable "
            "across the reference authors and sufficiently discriminative to "
            "influence the final attribution decision.",
            styles
        ))

        for line in explanation:
            story.append(Paragraph("• " + line, styles["BodyText"]))

    story.append(PageBreak())

    story.append(_header("Feature reliability (reference authors)", styles))
    story.append(_para(
        "This table reports the reliability of stylistic features across chunks of the "
        "reference texts. Features with high reliability show consistent usage patterns "
        "within an author's writing and are therefore weighted more strongly in "
        "comparative analysis.",
        styles
    ))

    stab_table = [["Feature", "Reliability"]] + [
        [k, f"{v:.3f}"] for k, v in sorted(
            feature_stability.items(), key=lambda x: x[1], reverse=True
        )
    ]
    story.append(_table(stab_table))

    story.append(_spacer())

    story.append(_header("Feature-based similarity scores", styles))
    story.append(_para(
        "The similarity scores below represent distances between the Unknown text and "
        "each reference author in a weighted stylistic feature space. Lower values "
        "indicate greater stylistic similarity.",
        styles
    ))

    sim_table = [["Target", "Distance"]] + [
        [k, f"{v:.4f}"] for k, v in feature_similarity.items()
    ]
    story.append(_table(sim_table))
    if figs:
        if "craig" in figs:
            story.append(_para(
                "The Craig marker-word scatter plot visualizes the proportion of marker words "
                "associated with each author across text chunks. Marker words are lexical items "
                "that occur disproportionately often in one author relative to another.\n\n"
                "Chunks closer to an author's marker region are stylistically more aligned with "
                "that author.",
                styles
            ))

            craig_img = save_fig_temp(figs.get("craig"), "craig.png")
            if craig_img:
                story.append(Image(craig_img, width=6 * inch, height=4 * inch))
                story.append(Spacer(1, 12))

    if craig_numeric_U:
            story.append(PageBreak())
            story.append(_header("Craig marker proportions — Unknown chunks", styles))

            story.append(_para(
                "This table reports numeric proportions of Author-A and Author-B "
                "marker words in each Unknown chunk. Values are normalized by chunk length.",
                styles
            ))

            table = [["Chunk", "Author-A marker proportion", "Author-B marker proportion"]]
            for row in craig_numeric_U:
                table.append([
                    row["chunk"],
                    f"{row['U_A_markers']:.4f}",
                    f"{row['U_B_markers']:.4f}",
                ])

            story.append(_table(table))
    if markers_A:
        story.append(PageBreak())
        story.append(_header("Top marker words — Author A (A vs B)", styles))

        table = [[
            "Marker word",
            "Craig coefficient (k)",
            "Chunks (Author A)",
            "Chunks (Author B)",
            "Frequency (A)",
            "Frequency (B)",
            "Relative freq. (A)",
            "Relative freq. (B)",
            "χ² contrast"
        ]
        ]
        for r in markers_A:
            table.append([
                r["word"],
                f"{r['k']:.3f}",
                r["Av"],
                r["Bv"],
                r["freqA"],
                r["freqB"],
                f"{r['relA']:.4f}",
                f"{r['relB']:.4f}",
                f"{r['chi2']:.2f}",
            ])

        story.append(_table(table))

    if markers_B:
        story.append(PageBreak())
        story.append(_header("Top marker words — Author B (B vs A)", styles))

        table = [[
            "Marker word",
            "Craig coefficient (k)",
            "Chunks (Author A)",
            "Chunks (Author B)",
            "Frequency (A)",
            "Frequency (B)",
            "Relative freq. (A)",
            "Relative freq. (B)",
            "χ² contrast"
        ]
        ]
        for r in markers_B:
            table.append([
                r["word"],
                f"{r['k']:.3f}",
                r["Av"],
                r["Bv"],
                r["freqA"],
                r["freqB"],
                f"{r['relA']:.4f}",
                f"{r['relB']:.4f}",
                f"{r['chi2']:.2f}",
            ])

        story.append(_table(table))

    if markers_U:
        story.append(PageBreak())
        story.append(_header("Marker words — Unknown vs A / B", styles))

        table = [[
            "Word",
            "Distinctiveness vs Author A",
            "Distinctiveness vs Author B",
            "Frequency (Unknown)",
            "Relative freq. (Unknown)",
            "Frequency (A)",
            "Frequency (B)"
        ]
        ]
        for r in markers_U:
            table.append([
                r["word"],
                f"{r['k_vs_A']:.3f}",
                f"{r['k_vs_B']:.3f}",
                r["freqU"],
                f"{r['relU']:.4f}",
                r["freqA"],
                r["freqB"],
            ])

        story.append(_table(table))

        if feature_table:
            story.append(PageBreak())
            story.append(_header("Feature-level comparison (A / B / U)", styles))

            table = [["Feature", "Author A", "Author B", "Unknown", "Feature stability", "Distance U–A", "Distance U–B"]]

            for r in feature_table:
                table.append([
                    r["feature"],
                    f"{r['Author A']:.4f}",
                    f"{r['Author B']:.4f}",
                    f"{r['Unknown']:.4f}",
                    f"{r['stability']:.3f}" if r["stability"] is not None else "—",
                    f"{r['U−A']:.4f}" if r["U−A"] is not None else "—",
                    f"{r['U−B']:.4f}" if r["U−B"] is not None else "—",
                ])

            story.append(_table(table))


        if "pca" in figs:
            story.append(_para(
                "This PCA plot visualizes stylistic relationships between chunks of the reference "
                "authors and the Unknown text. Each point represents a chunk projected into a "
                "space capturing dominant stylistic contrasts.\n\n"
                "Clustering patterns provide insight into stylistic overlap and separation "
                "between authors.",
                styles
            ))

            pca_img = save_fig_temp(figs.get("pca"), "pca.png")
            if pca_img:
                story.append(Image(pca_img, width=6 * inch, height=4 * inch))
                story.append(Spacer(1, 12))

    if topic_modeling and topic_modeling.get("enabled"):
        story.append(Paragraph(
            "<b>Topic Modelling</b>",
            styles["Heading2"]
        ))

        story.append(Paragraph(
            f"Number of topics: {topic_modeling.get('n_topics')}",
            styles["Normal"]
        ))

        story.append(Paragraph(
            f"Topic usage mode: {topic_modeling.get('topic_usage')}",
            styles["Normal"]
        ))

        if topic_modeling.get("selected_topics"):
            story.append(Paragraph(
                "Selected topics: "
                + ", ".join(str(t) for t in topic_modeling["selected_topics"]),
                styles["Normal"]
            ))

    # Chunk usage summary
        cc = topic_modeling.get("chunk_counts", {})
        for label, (total, used) in cc.items():
            story.append(Paragraph(
                f"{label}: {used} / {total} chunks used",
                styles["Normal"]
            ))

    topics = topic_modeling.get("topics", {})
    if topics:
        story.append(Paragraph(
            "<b>Discovered Topics</b>",
            styles["Heading3"]
        ))
        for tid, words in topics.items():
            story.append(Paragraph(
                f"Topic {tid}: {', '.join(words[:10])}",
                styles["Normal"]
            ))





    doc.build(story)
