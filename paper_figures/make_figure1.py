"""
Generates Figure 1 (overall system architecture) for the research paper,
as a clean, precise vector-style diagram rendered with matplotlib (not an
AI image generator), so every label is exact and matches the actual
implemented architecture described elsewhere in this project.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_W, FIG_H = 13.5, 8.2

COLOR_USER = "#dbe9ff"
COLOR_CLIENT = "#cfe8d8"
COLOR_BACKEND = "#fdf1c7"
COLOR_NLP = "#e7d9f7"
COLOR_DB = "#f7d9d9"
COLOR_MODULES = "#eef0f3"
EDGE = "#2b2b2b"


def box(ax, xy, w, h, text, facecolor, fontsize=10.5, weight="bold", ha="center",
        style="round,pad=0.02,rounding_size=0.08"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=style,
        linewidth=1.4,
        edgecolor=EDGE,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(patch)
    tx = x + 0.25 if ha == "left" else x + w / 2
    ax.text(tx, y + h / 2, text, ha=ha, va="center",
             fontsize=fontsize, fontweight=weight, color="#1a1a1a", zorder=3, linespacing=1.5)
    return (x, y, w, h)


def arrow(ax, start, end, label=None, rad=0.0, color=EDGE, label_dy=0.18, fontsize=8.7, style="-|>"):
    patch = FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=style,
        mutation_scale=16,
        linewidth=1.3,
        color=color,
        zorder=1,
    )
    ax.add_patch(patch)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + label_dy, label, ha="center", va="bottom", fontsize=fontsize,
                color="#333333", style="italic", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def elbow_loop(ax, start, end, top_y, label, color="#7a3fbf", fontsize=9.2):
    """Draws an up -> across -> down dashed 'feedback loop' connector, the
    same style used for feedback/update loops in typical architecture
    diagrams (an elbow connector, not a smooth arc, so it stays clearly
    above the main row instead of overlapping box edges)."""
    x1, y1 = start
    x2, y2 = end
    ax.plot([x1, x1], [y1, top_y], color=color, linewidth=1.3, linestyle="--", zorder=1)
    ax.plot([x1, x2], [top_y, top_y], color=color, linewidth=1.3, linestyle="--", zorder=1)
    arrow(ax, (x2, top_y), (x2, y2), color=color, style="-|>")
    ax.text((x1 + x2) / 2, top_y + 0.15, label, ha="center", va="bottom", fontsize=fontsize,
            color=color, style="italic", fontweight="bold", zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    # --- Main horizontal pipeline row ---
    y_main = 5.3
    h_main = 1.0

    user = box(ax, (0.3, y_main), 1.7, h_main, "User", COLOR_USER)
    client = box(ax, (2.5, y_main), 2.7, h_main, "React + Socket.io\nClient (Chat UI)", COLOR_CLIENT)
    backend = box(ax, (5.7, y_main), 3.1, h_main, "Node.js + Express\n+ Socket.io Server", COLOR_BACKEND)
    nlp = box(ax, (9.3, y_main), 3.5, h_main, "Python FastAPI\nNLP Service", COLOR_NLP)

    def right_mid(b):
        x, y, w, h = b
        return (x + w, y + h / 2)

    def left_mid(b):
        x, y, w, h = b
        return (x, y + h / 2)

    def top_mid(b):
        x, y, w, h = b
        return (x + w / 2, y + h)

    def bottom_mid(b):
        x, y, w, h = b
        return (x + w / 2, y)

    arrow(ax, right_mid(user), left_mid(client), "login / signup\n(REST)")
    arrow(ax, right_mid(client), left_mid(backend), "sendMessage()\nREST + Socket.io")
    arrow(ax, right_mid(backend), left_mid(nlp), "POST /analyze/message\n(async, non-blocking)")

    # --- Feedback loop: NLP result pushed live back to the client ---
    elbow_loop(
        ax,
        start=top_mid(nlp),
        end=top_mid(client),
        top_y=7.55,
        label="Live Socket.io push: sentiment \u00b7 emotion \u00b7 toxicity \u00b7 ACA context level",
    )

    # --- MongoDB persistence, hanging below the backend ---
    y_db = 3.0
    mongo = box(ax, (5.6, y_db), 3.3, 1.0, "MongoDB\nUsers \u00b7 Messages \u00b7 MessageAnalysis", COLOR_DB, fontsize=9.8)
    arrow(ax, bottom_mid(backend), top_mid(mongo), "persist / read\nconversation history", rad=0.0, label_dy=0.12)

    # --- Internal NLP modules, hanging below the NLP service box ---
    y_mod = 0.85
    modules = box(
        ax, (8.6, y_mod), 4.5, 1.75,
        "Internal Modules\n\u2022 Adaptive Context Activation (ACA)\n"
        "\u2022 Sentiment \u2014 DistilBERT\n\u2022 Emotion \u2014 DistilRoBERTa\n"
        "\u2022 Toxicity \u2014 BERT\n\u2022 Summarization \u2014 BART\n\u2022 Topic Extraction \u2014 TF-IDF",
        COLOR_MODULES, fontsize=9.6, weight="normal", ha="left",
        style="round,pad=0.03,rounding_size=0.10",
    )
    arrow(ax, bottom_mid(nlp), top_mid(modules), None)

    ax.text(6.75, 0.25, "Fig. 1. Real-Time Conversational NLP System Architecture.",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color="#111111")

    fig.tight_layout()
    fig.savefig("figure1_system_architecture.png", dpi=220, facecolor="white")
    print("Saved figure1_system_architecture.png")


if __name__ == "__main__":
    main()
