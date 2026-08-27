"""
Generates Figure 2 (Adaptive Context Activation decision pipeline) for the
research paper. This diagram mirrors the actual implementation in
nlp-service/app/context/ (signals.py, scoring.py, selector.py,
composer.py) - every box/label corresponds to a real function or
constant in that code, not an idealized or invented pipeline.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

FIG_W, FIG_H = 11.5, 14.5

COLOR_INPUT = "#dbe9ff"
COLOR_SIGNAL = "#fdf1c7"
COLOR_SCORE = "#e7d9f7"
COLOR_LOW = "#d9f2df"
COLOR_MED = "#fdf1c7"
COLOR_HIGH = "#f9d9d9"
COLOR_SELECT = "#cfe8d8"
COLOR_COMPOSE = "#dbe9ff"
COLOR_MODEL = "#e7d9f7"
EDGE = "#2b2b2b"


def box(ax, xy, w, h, text, facecolor, fontsize=10, weight="bold", ha="center",
        style="round,pad=0.02,rounding_size=0.08"):
    x, y = xy
    patch = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=1.4,
                            edgecolor=EDGE, facecolor=facecolor, zorder=2)
    ax.add_patch(patch)
    tx = x + 0.2 if ha == "left" else x + w / 2
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fontsize,
             fontweight=weight, color="#1a1a1a", zorder=3, linespacing=1.4)
    return (x, y, w, h)


def diamond(ax, center, w, h, text, facecolor, fontsize=9.6):
    cx, cy = center
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    patch = Polygon(pts, closed=True, linewidth=1.4, edgecolor=EDGE, facecolor=facecolor, zorder=2)
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, fontweight="bold",
             color="#1a1a1a", zorder=3, linespacing=1.3)
    return (cx - w / 2, cy - h / 2, w, h)


def arrow(ax, start, end, color=EDGE, style="-|>", lw=1.3):
    patch = FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=15,
                             linewidth=lw, color=color, zorder=1)
    ax.add_patch(patch)


def top_mid(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def bottom_mid(b):
    x, y, w, h = b
    return (x + w / 2, y)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 14.5)
    ax.axis("off")

    # Row A: input
    a = box(ax, (2.25, 13.0), 7.0, 1.0,
            "New Message + Recent Conversation History\n(up to 8 prior turns considered)",
            COLOR_INPUT, fontsize=10.5)

    # Row B: four parallel signals
    y_b = 11.0
    w_b, h_b, gap = 2.55, 1.15, 0.15
    x0 = 0.3
    signals = [
        "Reference Detection\nregex: pronouns / corrections\n(\"that\", \"it\", \"I meant\")",
        "Brevity\nword count ramp\n(< 6 words = higher score)",
        "Semantic Similarity\ncosine similarity via\nall-MiniLM-L6-v2 embeddings",
        "Classifier Uncertainty\n1 \u2212 2|sentiment_conf \u2212 0.5|",
    ]
    b_boxes = []
    for i, text in enumerate(signals):
        bx = x0 + i * (w_b + gap)
        b_boxes.append(box(ax, (bx, y_b), w_b, h_b, text, COLOR_SIGNAL, fontsize=8.8, weight="normal"))

    for bb in b_boxes:
        arrow(ax, (bb[0] + bb[2] / 2, a[1]), (bb[0] + bb[2] / 2, bb[1] + bb[3]))

    # Row C: weighted score
    c = box(ax, (1.75, 9.15), 8.0, 1.15,
            "Weighted Context-Requirement Score\nscore = 0.35\u00b7reference + 0.15\u00b7brevity + 0.30\u00b7similarity + 0.20\u00b7uncertainty",
            COLOR_SCORE, fontsize=10)
    for bb in b_boxes:
        arrow(ax, bottom_mid(bb), (bb[0] + bb[2] / 2, c[1] + c[3]))
    # merge the 4 arrow endpoints visually into the single score box top edge
    ax.plot([b_boxes[0][0] + b_boxes[0][2] / 2, b_boxes[-1][0] + b_boxes[-1][2] / 2],
            [c[1] + c[3] + 0.35, c[1] + c[3] + 0.35], color=EDGE, linewidth=1.0, alpha=0.0)

    # Decision diamond
    dpos = diamond(ax, (5.75, 7.7), 3.0, 1.0, "Level?", "#ffffff", fontsize=10)
    arrow(ax, bottom_mid(c), (5.75, 7.7 + 0.5))

    # Row D: three branch outcomes
    y_d = 5.3
    w_d, h_d = 3.3, 1.25
    d_low = box(ax, (0.3, y_d), w_d, h_d,
                "LOW\nscore < 0.35\n\u2192 current message only\n(0 messages selected)",
                COLOR_LOW, fontsize=9.4)
    d_med = box(ax, (4.1, y_d), w_d, h_d,
                "MEDIUM\n0.35 \u2264 score < 0.65\n\u2192 up to 2 relevant\nmessages selected",
                COLOR_MED, fontsize=9.4)
    d_high = box(ax, (7.9, y_d), w_d, h_d,
                 "HIGH\nscore \u2265 0.65\n\u2192 up to 5 relevant\nmessages selected",
                 COLOR_HIGH, fontsize=9.4)

    diamond_left = (5.75 - 1.5, 7.7)
    diamond_bottom = (5.75, 7.7 - 0.5)
    diamond_right = (5.75 + 1.5, 7.7)
    arrow(ax, diamond_left, top_mid(d_low))
    arrow(ax, diamond_bottom, top_mid(d_med))
    arrow(ax, diamond_right, top_mid(d_high))

    # Row E: context selector
    e = box(ax, (1.75, 4.15), 8.0, 1.05,
            "Context Selector\nrank candidate messages by 0.7\u00d7similarity + 0.3\u00d7recency; restore chronological order",
            COLOR_SELECT, fontsize=9.8)
    for bb in (d_low, d_med, d_high):
        arrow(ax, bottom_mid(bb), (bb[0] + bb[2] / 2, e[1] + e[3]))

    # Row F: composer
    f = box(ax, (2.25, 2.75), 7.0, 1.0,
            "Effective Text Composer\nselected context messages + current message (chronological order)",
            COLOR_COMPOSE, fontsize=9.8)
    arrow(ax, bottom_mid(e), top_mid(f))

    # Row G: models
    g = box(ax, (1.4, 1.05), 8.7, 1.15,
            "Sentiment (DistilBERT)   \u2022   Emotion (DistilRoBERTa)   \u2022   Toxicity (BERT)",
            COLOR_MODEL, fontsize=10.5)
    arrow(ax, bottom_mid(f), top_mid(g))

    ax.text(5.75, 0.35, "Fig. 2. Adaptive Context Activation (ACA) Decision Pipeline.",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color="#111111")

    fig.tight_layout()
    fig.savefig("figure2_aca_pipeline.png", dpi=220, facecolor="white")
    print("Saved figure2_aca_pipeline.png")


if __name__ == "__main__":
    main()
