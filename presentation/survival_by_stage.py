"""Chart: 5-year relative survival of breast cancer by stage at diagnosis (SEER, diagnoses 2015-2021).

Source: SEER Cancer Stat Facts, female breast (https://seer.cancer.gov/statfacts/html/breast.html);
American Cancer Society survival rates page. SEER reports summary stage (spread), not
Roman-numeral stage, so the mapping to I-IV in the labels is approximate.

Usage: uv run python survival_by_stage.py   (writes assets/survival_by_stage.png)
"""

from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path(__file__).parent / "assets" / "survival_by_stage.png"

# SEER 5-year relative survival, %
STAGES = [
    ("Lokalizovaný\n(přibližně stadium I)", 100.0),
    ("Regionální\n(přibližně II–III)", 87.2),
    ("Vzdálené metastázy\n(stadium IV)", 32.6),
]
INK, INK2, ACCENT, SURFACE = "#1D1D1F", "#6E6E73", "#D91E4E", "#FBFBFD"

fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
labels, values = zip(*STAGES, strict=True)
bars = ax.bar(range(len(values)), values, width=0.55, color=ACCENT)
for b, v in zip(bars, values, strict=True):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v + 2,
        f"{v:.0f} %",
        ha="center",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
ax.set_xticks(range(len(values)), labels, fontsize=11, color=INK)
ax.set_ylim(0, 112)
ax.set_yticks(
    [0, 25, 50, 75, 100], ["0 %", "25 %", "50 %", "75 %", "100 %"], color=INK2
)
ax.yaxis.grid(True, color="#E5E5EA", linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
for s in ("top", "right", "left", "bottom"):
    ax.spines[s].set_visible(False)
ax.set_title(
    "Čím dříve se rakovina prsu odhalí, tím vyšší je šance na přežití",
    loc="left",
    fontsize=14,
    fontweight="bold",
    color=INK,
)
ax.text(
    0,
    -0.2,
    "Relativní 5leté přežití podle rozsahu onemocnění při diagnóze, USA, diagnózy 2015–2021.\n"
    "Zdroj: SEER Cancer Stat Facts. Statistika pro skupiny žen, ne předpověď pro jednotlivkyni.",
    transform=ax.transAxes,
    fontsize=8.5,
    color=INK2,
    va="top",
)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor=SURFACE)
print(f"wrote {OUT}")
