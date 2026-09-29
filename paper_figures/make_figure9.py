"""
Generates Figure 9 - a design mockup of the PROPOSED Analytics Dashboard
extension to the right sidebar (client/src/components/RightSidebar.jsx).

IMPORTANT: this is a design/wireframe mockup for a planned enhancement, not
a screenshot of an implemented feature. The existing profile photo / bio /
media block and the "Conversation Intelligence" text panel it extends are
real and already implemented (see RightSidebar.jsx); the tabbed
"Dashboard" view with per-module trend charts drawn below is the proposed
addition described in the paper as future/planned work.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge

FIG_W, FIG_H = 7.4, 12.6

COLOR_FRAME = "#f4f5f7"
COLOR_EXISTING = "#e7e9ee"
COLOR_TAB_ACTIVE = "#7a5cf0"
COLOR_TAB_INACTIVE = "#dfe1e8"
COLOR_SENTIMENT = "#4c8bf5"
COLOR_EMOTION = "#8a5cf6"
COLOR_TOXICITY = "#f0554c"
COLOR_TOPIC = "#e0a72e"
COLOR_SUMMARY = "#2fa66b"
EDGE = "#2b2b2b"


def rbox(ax, xy, w, h, facecolor, edgecolor=EDGE, lw=1.2, style="round,pad=0.015,rounding_size=0.06", zorder=2):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                        edgecolor=edgecolor, facecolor=facecolor, zorder=zorder)
    ax.add_patch(p)
    return (x, y, w, h)


def label(ax, x, y, text, fontsize=8.6, weight="normal", color="#1a1a1a", ha="left", va="center", style="normal"):
    ax.text(x, y, text, fontsize=fontsize, fontweight=weight, color=color,
             ha=ha, va=va, zorder=4, fontstyle=style)


def mini_line_chart(ax, x, y, w, h, color):
    """Small sentiment-trend sketch: a zig-zag polyline with markers."""
    rbox(ax, (x, y), w, h, "#ffffff", lw=0.8, zorder=2)
    xs = np.linspace(x + 0.14, x + w - 0.14, 7)
    ys = y + h * 0.5 + np.array([-0.05, 0.15, 0.05, -0.12, 0.10, 0.18, 0.02]) * (h / 0.5)
    ax.plot(xs, ys, color=color, linewidth=1.8, zorder=3, marker="o", markersize=3.2)


def mini_bar_chart(ax, x, y, w, h, color, heights):
    rbox(ax, (x, y), w, h, "#ffffff", lw=0.8, zorder=2)
    n = len(heights)
    gap = (w - 0.2) / n
    bw = gap * 0.62
    for i, hh in enumerate(heights):
        bx = x + 0.1 + i * gap
        bar_h = max(hh * (h - 0.2), 0.03)
        c = color if hh < 0.75 else "#f0554c"
        ax.add_patch(Rectangle((bx, y + 0.1), bw, bar_h, facecolor=c, edgecolor="none", zorder=3))


def mini_donut(ax, x, y, w, h, fractions, colors):
    rbox(ax, (x, y), w, h, "#ffffff", lw=0.8, zorder=2)
    cx, cy = x + w * 0.32, y + h * 0.5
    r = min(w * 0.28, h * 0.42)
    start = 90.0
    for frac, c in zip(fractions, colors):
        theta2 = start - frac * 360
        ax.add_patch(Wedge((cx, cy), r, theta2, start, width=r * 0.5, facecolor=c, edgecolor="white",
                            linewidth=0.8, zorder=3))
        start = theta2
    legend_items = ["joy", "anger", "sadness", "neutral"]
    ly = y + h - 0.16
    for txt, c in zip(legend_items, colors):
        ax.add_patch(Rectangle((x + w * 0.58, ly - 0.05), 0.09, 0.09, facecolor=c, edgecolor="none", zorder=3))
        label(ax, x + w * 0.58 + 0.14, ly - 0.005, txt, fontsize=6.6, color="#333333")
        ly -= 0.19


def mini_chips(ax, x, y, w, h, words, color):
    rbox(ax, (x, y), w, h, "#ffffff", lw=0.8, zorder=2)
    cx = x + 0.14
    cy = y + h / 2 - 0.13
    for wtext in words:
        wchip = 0.085 * len(wtext) + 0.16
        rbox(ax, (cx, cy), wchip, 0.26, color, lw=0.6, style="round,pad=0.01,rounding_size=0.13", zorder=3)
        label(ax, cx + wchip / 2, cy + 0.13, wtext, fontsize=6.8, weight="bold", ha="center", color="#ffffff")
        cx += wchip + 0.1


def card(ax, x, y, w, h, title, color, chart_fn, note_text):
    rbox(ax, (x, y), w, h, "#fbfbfc", lw=1.0, zorder=2)
    ax.add_patch(Rectangle((x, y + h - 0.07), w, 0.07, facecolor=color, edgecolor="none", zorder=3))
    label(ax, x + 0.16, y + h - 0.32, title, fontsize=9.2, weight="bold")
    chart_h = h - 0.85
    chart_fn(ax, x + 0.16, y + 0.42, w - 0.32, chart_h)
    label(ax, x + 0.16, y + 0.18, note_text, fontsize=7.2, color="#555555")


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    inner_x0, inner_x1 = 0.7, FIG_W - 0.7
    card_w = inner_x1 - inner_x0
    card_h = 1.65
    gap = 0.2

    cursor = FIG_H - 1.1  # top of first content block, below title+frame padding

    # --- Existing block: profile photo, name, bio, media (already implemented) ---
    y_exist = cursor - 1.55
    rbox(ax, (inner_x0, y_exist), card_w, 1.55, COLOR_EXISTING, lw=1.0, zorder=2)
    ax.add_patch(Circle((inner_x0 + 0.55, y_exist + 1.12), 0.34, facecolor="#c9cfe0", edgecolor=EDGE, linewidth=1.0, zorder=3))
    ax.add_patch(Circle((inner_x0 + 1.05, y_exist + 1.26), 0.05, facecolor="#22c55e", edgecolor="none", zorder=4))
    label(ax, inner_x0 + 1.2, y_exist + 1.22, "Full Name", fontsize=10.0, weight="bold")
    label(ax, inner_x0 + 1.2, y_exist + 0.9, "\u201cAvailable after 6pm\u201d  (bio)", fontsize=8.0, color="#444444")
    label(ax, inner_x0 + 0.25, y_exist + 0.42, "Media", fontsize=8.0, weight="bold", color="#444444")
    for i in range(4):
        rbox(ax, (inner_x0 + 0.25 + i * 0.62, y_exist + 0.1), 0.52, 0.28, "#c9cfe0", lw=0.6, zorder=3)
    label(ax, inner_x1 - 0.2, y_exist + 1.4, "existing", fontsize=7.4, style="italic", color="#666666", ha="right")
    cursor = y_exist - 0.3

    # --- Tabs: Media | Dashboard (Dashboard = proposed) ---
    y_tabs = cursor - 0.42
    rbox(ax, (inner_x0, y_tabs), 1.7, 0.42, COLOR_TAB_INACTIVE, lw=0.8, zorder=2)
    label(ax, inner_x0 + 0.85, y_tabs + 0.21, "Media", fontsize=8.8, ha="center", weight="bold", color="#444444")
    rbox(ax, (inner_x0 + 1.9, y_tabs), 2.1, 0.42, COLOR_TAB_ACTIVE, lw=0.8, zorder=2)
    label(ax, inner_x0 + 2.95, y_tabs + 0.21, "\u25b6 Dashboard", fontsize=9.0, ha="center", weight="bold", color="#ffffff")
    label(ax, inner_x1 - 0.2, y_tabs + 0.21, "proposed (new)", fontsize=7.6, style="italic", color="#7a5cf0",
          ha="right", weight="bold")
    cursor = y_tabs - 0.28

    # --- Dashboard body: five module analytics cards ---
    titles = [
        ("Sentiment Trend", COLOR_SENTIMENT,
         lambda a, x, y, w, h: mini_line_chart(a, x, y, w, h, COLOR_SENTIMENT),
         "positive/negative polarity over the conversation timeline"),
        ("Emotion Distribution", COLOR_EMOTION,
         lambda a, x, y, w, h: mini_donut(a, x, y, w, h, [0.42, 0.28, 0.18, 0.12],
                                           [COLOR_EMOTION, "#c4b5fd", "#f0abfc", "#e5e7eb"]),
         "share of joy / anger / sadness / neutral turns"),
        ("Toxicity Incidents", COLOR_TOXICITY,
         lambda a, x, y, w, h: mini_bar_chart(a, x, y, w, h, "#f6b8b3", [0.2, 0.15, 0.8, 0.1, 0.3, 0.15]),
         "flagged-message count per session, over time"),
        ("Topic Keywords", COLOR_TOPIC,
         lambda a, x, y, w, h: mini_chips(a, x, y, w, h, ["project", "deadline", "budget"], COLOR_TOPIC),
         "top recurring keywords extracted via TF-IDF"),
        ("Summarization Timeline", COLOR_SUMMARY,
         lambda a, x, y, w, h: mini_chips(a, x, y, w, h, ["\u2022 recap 1", "\u2022 recap 2"], COLOR_SUMMARY),
         "rolling abstractive recap of the conversation so far"),
    ]
    for title, color, chart_fn, note_text in titles:
        cursor -= card_h
        card(ax, inner_x0, cursor, card_w, card_h, title, color, chart_fn, note_text)
        cursor -= gap

    frame_bottom = cursor - 0.1
    frame_top = FIG_H - 0.6

    # Outer sidebar frame, drawn last so it wraps exactly around the content
    ax.add_patch(Rectangle((0.36, frame_bottom + 0.01), FIG_W - 0.72, frame_top - frame_bottom - 0.02,
                            facecolor=COLOR_FRAME, edgecolor="none", zorder=-1))
    frame = FancyBboxPatch((0.35, frame_bottom), FIG_W - 0.7, frame_top - frame_bottom,
                            boxstyle="round,pad=0.02,rounding_size=0.10", linewidth=1.6,
                            edgecolor=EDGE, facecolor="none", zorder=0)
    ax.add_patch(frame)
    label(ax, FIG_W / 2, frame_top - 0.28, "Right Sidebar \u2014 Selected Conversation Partner", fontsize=11.0,
          weight="bold", ha="center")

    label(ax, FIG_W / 2, frame_bottom - 0.42,
          "Fig. 9. Proposed Analytics Dashboard \u2014 Right-Sidebar Extension\n"
          "(design mockup; extends the existing profile/bio/media and\nConversation-Intelligence panel)",
          fontsize=9.6, weight="bold", ha="center")

    ax.set_ylim(frame_bottom - 0.95, FIG_H)

    fig.tight_layout()
    fig.savefig("figure9_proposed_dashboard.png", dpi=220, facecolor="white")
    print("Saved figure9_proposed_dashboard.png")


if __name__ == "__main__":
    main()
