"""Draw the segment rules of ``segments.py`` as a decision tree.

Each box is one yes/no question, asked in the same order as ``segments.assign``;
each leaf is a segment. Every node shows how many invitations reach it and their
12-month response; leaves are coloured by that response and show the suggested
intervention.

Usage (writes ``segment_tree.png`` next to this file):
    uv run python tree_diagram.py
"""

from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize
from segments import COLUMNS, DUE_MONTHS, INTERVENTIONS, PREPARED, SEGMENTS, assign


LABEL = "SCREENING_12M_FLAG"
OUTPUT = Path(__file__).resolve().parent / "segment_tree.png"
INK, INK2, EDGE = "#0b0b0b", "#52514e", "#a3a29d"
NL = "\n"


def node_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Rows reaching every node; mirrors the rule order of ``segments.assign``."""
    months = df["MONTHS_SINCE_LAST_SCREEN"]
    screened = months.notna()
    recent = screened & (months < DUE_MONTHS)
    first = df["FIRST_INVITATION"] == 1
    gyn = df["GYN_PREVENTION_24M_CNT"] > 0
    other = df["OTHER_SCREENING_36M_CNT"] > 0
    contact = df["HEALTHCARE_CONTACT_12M_CNT"] > 0
    q2 = ~recent
    q3 = q2 & ~first
    q4a, q4b = q3 & screened, q3 & ~screened
    q5 = q4b & ~gyn
    q6 = q5 & ~other
    names = list(SEGMENTS)
    return {
        "q1": pd.Series(True, index=df.index),
        "q2": q2,
        "q3": q3,
        "q4a": q4a,
        "q4b": q4b,
        "q5": q5,
        "q6": q6,
        names[0]: recent,
        names[1]: q2 & first,
        names[2]: q4a & gyn,
        names[3]: q4a & ~gyn,
        names[4]: q4b & gyn,
        names[5]: q5 & other,
        names[6]: q6 & contact,
        names[7]: q6 & ~contact,
    }


QUESTIONS = {
    "q1": f"Screened less than{NL}{DUE_MONTHS} months ago?",
    "q2": "First invitation?",
    "q3": f"Screened in the{NL}last 5 years?",
    "q4a": f"Gynaecologist{NL}prevention visit (2 y)?",
    "q4b": f"Gynaecologist{NL}prevention visit (2 y)?",
    "q5": f"Other screening{NL}programme (3 y)?",
    "q6": f"Any healthcare{NL}contact (12 m)?",
}
S = list(SEGMENTS)
# node -> (x, y), one level every 3 units; edges as (parent, child, "yes" / "no")
POSITIONS = {
    "q1": (7.0, 0.0),
    S[0]: (2.5, -3.0), "q2": (10.0, -3.0),
    S[1]: (6.5, -6.0), "q3": (13.0, -6.0),
    "q4a": (8.5, -9.0), "q4b": (17.0, -9.0),
    S[2]: (5.5, -12.0), S[3]: (10.5, -12.0), S[4]: (14.5, -12.0), "q5": (20.0, -12.0),
    S[5]: (17.0, -15.0), "q6": (23.0, -15.0),
    S[6]: (20.0, -18.0), S[7]: (25.5, -18.0),
}  # fmt: skip
# Approximate half-heights of the boxes (data units), to place the yes / no labels
# in the visible gap between a question and the box below it.
HALF_HEIGHT = {"question": 0.8, "leaf": 1.15}
EDGES = [
    ("q1", S[0], "yes"), ("q1", "q2", "no"),
    ("q2", S[1], "yes"), ("q2", "q3", "no"),
    ("q3", "q4a", "yes"), ("q3", "q4b", "no"),
    ("q4a", S[2], "yes"), ("q4a", S[3], "no"),
    ("q4b", S[4], "yes"), ("q4b", "q5", "no"),
    ("q5", S[5], "yes"), ("q5", "q6", "no"),
    ("q6", S[6], "yes"), ("q6", S[7], "no"),
]  # fmt: skip


# Every text and colour of the drawing; pass another dict (e.g. a translation) to
# draw_segment_tree to restyle it without touching the layout.
TEXTS_EN = {
    "questions": QUESTIONS,
    "names": {s: s.split(" ", 1)[1] for s in SEGMENTS},
    "interventions": INTERVENTIONS,
    "yes": "yes",
    "no": "no",
    "stats": lambda n, share, rate: (
        f"{n:,} invitations ({share:.0%}){NL}{rate:.1%} screened within 12 m"
    ),
    "title": lambda rate: (
        "Segments of invitations as a decision tree: questions asked in order, "
        f"leaves coloured by 12-month response (all invitations: {rate:.1%})"
    ),
    "cmap": "Blues",
    "font": None,  # matplotlib default
    "background": "white",
}


def node_text(node: str, n: int, total: int, rate: float, texts: dict) -> str:
    stats = texts["stats"](n, n / total, rate)
    if node in QUESTIONS:
        return f"{texts['questions'][node]}{NL}{NL}{stats}"
    letter = node.split(" ", 1)[0]
    title = NL.join(wrap(f"{letter}  {texts['names'][node]}", 24))
    action = NL.join(wrap(f"→ {texts['interventions'][node]}", 26))
    return f"{title}{NL}{NL}{stats}{NL}{NL}{action}"


def draw_segment_tree(
    df: pd.DataFrame, path: Path | None = None, texts: dict | None = None
) -> plt.Figure:
    """Draw the rule tree; ``df`` needs ``segments.COLUMNS`` and the 12M label.

    ``texts``: labels and colours (keys as in ``TEXTS_EN``, the default).
    """
    texts = TEXTS_EN if texts is None else texts
    masks = node_masks(df)
    total = len(df)
    y = df[LABEL]
    cmap, norm = colormaps[texts["cmap"]], Normalize(vmin=-0.15, vmax=0.6)
    font = {} if texts["font"] is None else {"fontfamily": texts["font"]}

    fig, ax = plt.subplots(figsize=(24, 17), facecolor=texts["background"])
    ax.set_xlim(0.3, 27.7)
    ax.set_ylim(-19.4, 1.1)
    ax.axis("off")

    # Plain lines behind the boxes; yes / no labels on a white background.
    for parent, child, answer in EDGES:
        (x0, y0), (x1, y1) = POSITIONS[parent], POSITIONS[child]
        ax.plot([x0, x1], [y0, y1], color=EDGE, lw=1.4, zorder=1)
        top = y0 - HALF_HEIGHT["question"]
        bottom = y1 + HALF_HEIGHT["question" if child in QUESTIONS else "leaf"]
        frac = (y0 - (top + bottom) / 2) / (y0 - y1)  # middle of the visible gap
        ax.text(
            x0 + frac * (x1 - x0),
            y0 + frac * (y1 - y0),
            texts[answer],
            ha="center",
            va="center",
            fontsize=10,
            color=INK2,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.2", "fc": texts["background"], "ec": "none"},
            zorder=4,
            **font,
        )

    for node, (x, y_pos) in POSITIONS.items():
        mask = masks[node]
        n, rate = int(mask.sum()), y[mask].mean()
        if node in QUESTIONS:
            box = {"boxstyle": "round,pad=0.55", "fc": "#f4f3f0", "ec": EDGE}
            color = INK
        else:
            fill = cmap(norm(rate))
            box = {"boxstyle": "round,pad=0.6", "fc": fill, "ec": "white", "lw": 2}
            color = "white" if rate > 0.3 else INK
        ax.text(
            x,
            y_pos,
            node_text(node, n, total, rate, texts),
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            bbox=box,
            linespacing=1.3,
            zorder=3,
            **font,
        )

    title = texts["title"](y.mean())
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", loc="left", **font)
    if path is not None:
        fig.savefig(path, dpi=100, bbox_inches="tight", facecolor=texts["background"])
    return fig


def main() -> None:
    df = pd.read_csv(PREPARED, usecols=[*COLUMNS, LABEL])
    # The drawing re-implements the rules node by node; it must match assign() exactly.
    masks, counts = node_masks(df), assign(df).value_counts()
    for name in SEGMENTS:
        assert masks[name].sum() == counts[name], f"tree and assign() differ: {name}"
    draw_segment_tree(df, OUTPUT)
    print(f"Written to {OUTPUT}")


if __name__ == "__main__":
    main()
