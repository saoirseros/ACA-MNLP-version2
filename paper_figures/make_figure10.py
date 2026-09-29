"""
Generates Figure 10 - a design mockup of the PROPOSED "Algorithm Showcase"
extension to the left sidebar (client/src/components/Sidebar.jsx).

This is a design/wireframe mockup for a planned enhancement, not a
screenshot of an implemented feature. It re-uses the exact box/arrow
vocabulary of Fig. 2 (the real, implemented ACA pipeline) and the exact
inline result-tag style already rendered per message in ChatContainer.jsx
(sentiment / emotion / toxicity / "ctx:" pills), so the proposed panel is
shown as a guided, presentation-oriented walk-through of a real message
through the real pipeline, with one concrete worked example annotated at
every stage, rather than an invented new algorithm.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

FIG_W, FIG_H = 7.6, 9.6

COLOR_FRAME = "#f4f5f7"
COLOR_MSG = "#dbe9ff"
COLOR_SIGNAL = "#fde68a"
COLOR_SCORE = "#e7d9f7"
COLOR_SELECTED = "#cfe8d8"
COLOR_SKIPPED = "#e5e5e5"
COLOR_MODEL = "#e7d9f7"
COLOR_OUTPUT = "#282142"
EDGE = "#2b2b2b"

MARGIN_X = 0.55
MARGIN_W = 1.05
INNER_X0 = 1.75
INNER_X1 = FIG_W - 0.35


def box(ax, xy, w, h, text, facecolor, fontsize=8.8, weight="bold", ha="center", textcolor="#1a1a1a", zorder=2):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.07", linewidth=1.3,
                        edgecolor=EDGE, facecolor=facecolor, zorder=zorder)
    ax.add_patch(p)
    tx = x + 0.18 if ha == "left" else x + w / 2
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fontsize, fontweight=weight,
             color=textcolor, zorder=zorder + 1, linespacing=1.3)
    return (x, y, w, h)


def margin_note(ax, y_center, text, fontsize=7.2):
    ax.text(MARGIN_X, y_center, text, ha="left", va="center", fontsize=fontsize, color="#555555",
             style="italic", zorder=5, linespacing=1.25)


def section_title(ax, y, text, fontsize=9.6):
    ax.text((INNER_X0 + INNER_X1) / 2, y, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", zorder=5)


def arrow(ax, start, end, color=EDGE, lw=1.3):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=lw, color=color, zorder=1))


def top_mid(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def bottom_mid(b):
    x, y, w, h = b
    return (x + w / 2, y)


def pill(ax, x, y, text, facecolor, textcolor="#1a1a1a", fontsize=6.8):
    w = 0.072 * len(text) + 0.2
    p = FancyBboxPatch((x, y), w, 0.28, boxstyle="round,pad=0.01,rounding_size=0.14", linewidth=0.7,
                        edgecolor="none", facecolor=facecolor, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + 0.14, text, ha="center", va="center", fontsize=fontsize, fontweight="bold",
             color=textcolor, zorder=4)
    return w


def pill_row(ax, items, y, x0=INNER_X0):
    px = x0
    for text, fc, tc in items:
        w = pill(ax, px, y, text, fc, textcolor=tc)
        px += w + 0.14
    return px


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    cursor = FIG_H - 0.95  # below title + frame top padding
    content_w = INNER_X1 - INNER_X0

    # --- Step 1: live example message ---
    msg_h = 0.5
    m = box(ax, (INNER_X0 + content_w * 0.08, cursor - msg_h), content_w * 0.84, msg_h,
            "\u201cNo, I meant the other one.\u201d", COLOR_MSG, fontsize=9.0)
    margin_note(ax, cursor - msg_h / 2, "1. Live example\nmessage arrives")
    cursor -= msg_h + 0.3
    arrow(ax, (FIG_W / 2, cursor + 0.3), (FIG_W / 2, cursor))

    # --- Step 2: ACA signal scoring (four pills, real signals from Fig. 2) ---
    section_title(ax, cursor, "Adaptive Context Activation \u2014 signal scoring")
    cursor -= 0.34
    pill_items = [("reference: yes", COLOR_SIGNAL, "#1a1a1a"), ("brevity: 6w", COLOR_SIGNAL, "#1a1a1a"),
                  ("similarity: 0.82", COLOR_SIGNAL, "#1a1a1a"), ("uncertainty: 0.41", COLOR_SIGNAL, "#1a1a1a")]
    pill_row(ax, pill_items, cursor - 0.28)
    margin_note(ax, cursor - 0.14, "2. Four\ninterpretable\nsignals (Fig. 2)")
    cursor -= 0.28 + 0.35
    arrow(ax, (FIG_W / 2, cursor + 0.35), (FIG_W / 2, cursor))

    # --- Step 3: weighted score -> context level ---
    score_h = 0.5
    s = box(ax, (INNER_X0, cursor - score_h), content_w, score_h,
            "score = 0.71  \u2192  HIGH context level", COLOR_SCORE, fontsize=9.0)
    margin_note(ax, cursor - score_h / 2, "3. Weighted score\nis thresholded")
    cursor -= score_h + 0.3
    arrow(ax, (FIG_W / 2, cursor + 0.3), (FIG_W / 2, cursor))

    # --- Step 4: context selector (ranked history bubbles) ---
    section_title(ax, cursor, "Context Selector \u2014 ranks by similarity + recency")
    cursor -= 0.36
    hist = [("\u201cany updates?\u201d", True), ("\u201clunch?\u201d", False),
            ("\u201cthe results are in\u201d", True), ("\u201cok cool\u201d", True)]
    hx = INNER_X0
    bh = 0.3
    for txt, selected in hist:
        w = 0.082 * len(txt) + 0.16
        fc = COLOR_SELECTED if selected else COLOR_SKIPPED
        ec = "#1a7a3f" if selected else "#aaaaaa"
        p = FancyBboxPatch((hx, cursor - bh), w, bh, boxstyle="round,pad=0.01,rounding_size=0.06",
                            linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2)
        ax.add_patch(p)
        ax.text(hx + w / 2, cursor - bh / 2, txt, ha="center", va="center", fontsize=6.6,
                 color="#1a1a1a" if selected else "#888888", zorder=3)
        hx += w + 0.1
    margin_note(ax, cursor - bh / 2, "4. Selected\n(green) vs.\nskipped (grey)")
    cursor -= bh + 0.3
    arrow(ax, (FIG_W / 2, cursor + 0.3), (FIG_W / 2, cursor))

    # --- Step 5: parallel classifiers ---
    section_title(ax, cursor, "Effective text \u2192 classifiers")
    cursor -= 0.32
    clf_h = 0.55
    c_w = (content_w - 0.3) / 3
    c1 = box(ax, (INNER_X0, cursor - clf_h), c_w, clf_h, "Sentiment\n(DistilBERT)", COLOR_MODEL, fontsize=7.6)
    c2 = box(ax, (INNER_X0 + c_w + 0.15, cursor - clf_h), c_w, clf_h, "Emotion\n(DistilRoBERTa)", COLOR_MODEL, fontsize=7.6)
    c3 = box(ax, (INNER_X0 + 2 * (c_w + 0.15), cursor - clf_h), c_w, clf_h, "Toxicity\n(BERT)", COLOR_MODEL, fontsize=7.6)
    margin_note(ax, cursor - clf_h / 2, "5. Same classifiers\nalready in production;\nonly context size\nchanged")
    cursor -= clf_h + 0.35
    for cb in (c1, c2, c3):
        arrow(ax, bottom_mid(cb), (FIG_W / 2, cursor + 0.05))
        arrow(ax, (FIG_W / 2, cursor + 0.35), top_mid(cb))

    # --- Step 6: annotated output bubble (matches existing ChatContainer.jsx pill UI) ---
    out_h = 0.5
    o = box(ax, (INNER_X0 + content_w * 0.04, cursor - out_h), content_w * 0.92, out_h,
            "\u201cNo, I meant the other one.\u201d", COLOR_OUTPUT, fontsize=8.8, weight="normal", textcolor="#ffffff")
    cursor -= out_h + 0.06
    pill_row(ax, [("negative (81%)", "#7a1f1f", "#fecaca"), ("confusion", "#3730a3", "#e0e7ff"),
                  ("ctx: high (3)", "#374151", "#e5e7eb")], cursor - 0.28, x0=INNER_X0 + content_w * 0.06)
    margin_note(ax, cursor - 0.14, "6. Result pills\n(already the real\nUI pattern)")
    cursor -= 0.28 + 0.25

    frame_bottom = cursor - 0.15
    frame_top = FIG_H - 0.55

    ax.add_patch(Rectangle((0.36, frame_bottom + 0.01), FIG_W - 0.72, frame_top - frame_bottom - 0.02,
                            facecolor=COLOR_FRAME, edgecolor="none", zorder=-1))
    ax.add_patch(FancyBboxPatch((0.35, frame_bottom), FIG_W - 0.7, frame_top - frame_bottom,
                                 boxstyle="round,pad=0.02,rounding_size=0.10", linewidth=1.6,
                                 edgecolor=EDGE, facecolor="none", zorder=0))
    ax.text(FIG_W / 2, frame_top - 0.25, "Left Sidebar \u2014 \u201cAlgorithm Showcase\u201d Panel (proposed, new)",
            ha="center", va="center", fontsize=10.6, fontweight="bold")

    ax.text(FIG_W / 2, frame_bottom - 0.5,
            "Fig. 10. Proposed Algorithm Showcase \u2014 Left-Sidebar Extension\n"
            "(design mockup; walks one live example through the real,\n"
            "implemented ACA + classification pipeline of Fig. 2)",
            ha="center", va="center", fontsize=9.4, fontweight="bold")

    ax.set_ylim(frame_bottom - 1.0, FIG_H)

    fig.tight_layout()
    fig.savefig("figure10_proposed_algorithm_showcase.png", dpi=220, facecolor="white")
    print("Saved figure10_proposed_algorithm_showcase.png")


if __name__ == "__main__":
    main()
