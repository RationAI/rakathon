"""Build the presentation assets: Czech charts as high-resolution PNGs.

Reads the prepared data, the splits and the results of ``training/classifiers``
(``explain/`` and ``ablation/``), fits the gradient-boosting model once for the risk
deciles, and writes to ``assets/``:

- ``*.png``: one chart per file, Czech labels and number format, in the deck's style,
- ``cisla.json``: every number the slides quote, so no figure is typed by hand.

Usage (from presentation/, ~2 min):
    uv run python make_assets.py
"""

import json
import sys
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CLASSIFIERS = ROOT / "training" / "classifiers"
OUT = Path(__file__).resolve().parent / "assets"
sys.path[:0] = [str(CLASSIFIERS), str(ROOT / "segmentation")]

from segments import COLUMNS as SEGMENT_COLUMNS
from segments import SEGMENTS, assign
from tree_diagram import NL, QUESTIONS, TEXTS_EN, draw_segment_tree

from data import prepare
from gradient_boosting.model import make_model


LABEL = "SCREENING_12M_FLAG"

# Deck palette: toned white background, warm ink, one red accent, warm greys.
BG, INK, INK2, GRID = "#FBFAF8", "#26221F", "#5E5852", "#E6E1DA"
RED, ORANGE, GREY, GREY_DARK = "#C62D1F", "#E08A3C", "#B8B2AA", "#7A7069"
REDS = LinearSegmentedColormap.from_list("reds", ["#F7DCD6", "#E58B7C", RED, "#7E170D"])
FONT = "Segoe UI"

plt.rcParams.update({
    "font.family": [FONT, "DejaVu Sans"],
    "font.size": 15,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
    "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.frameon": False,
})  # fmt: skip


# ---------------------------------------------------------------- Czech formatting


def cz_int(n: float) -> str:
    return f"{round(n):,}".replace(",", " ")


def cz_pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f}".replace(".", ",") + " %"


def cz_num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


PCT_AXIS = FuncFormatter(lambda v, _: cz_pct(v))

FEATURES_CZ = {
    "PRIOR_INVITATION_COUNT": "Počet předchozích pozvánek",
    "INVITATION_SEQUENCE": "Pořadí pozvánky v řadě",
    "MONTHS_SINCE_LAST_SCREEN": "Měsíce od posledního screeningu",
    "GYN_PREVENTION_24M_CNT": "Prevence u gynekologa (2 roky)",
    "OTHER_SCREENING_36M_CNT": "Jiné screeningy (3 roky)",
    "HEALTHCARE_CONTACT_12M_CNT": "Kontakty se zdravotnictvím (12 měs.)",
    "MONTHS_LAST_INVITATION": "Měsíce od poslední pozvánky",
    "DENTAL_PREVENTION_24M_CNT": "Prevence u zubaře (2 roky)",
    "AGE_BAND": "Věková skupina",
    "GYN_REG_DURATION_BAND": "Délka registrace u gynekologa",
    "VZP_TERRITORIAL_AREA": "Územní pracoviště",
    "PRIOR_SCREEN_COUNT_5Y": "Počet screeningů (5 let)",
    "NEVER_SCREENED_BEFORE_FLAG": "Bez screeningu (5 let)",
    "FIRST_INVITATION": "První pozvánka",
    "INSURANCE_DURATION_YEARS": "Délka pojištění",
}
MODELS_CZ = {
    "gradient boosting": "Gradient boosting",
    "random forest": "Náhodný les",
    "MLP": "Neuronová síť (MLP)",
    "decision tree": "Rozhodovací strom",
    "spline logistic regression": "Logistická regrese se splajny",
    "logistic regression": "Logistická regrese",
    "kNN": "k nejbližších sousedů",
    "naive Bayes": "Naivní Bayes",
}
S = list(SEGMENTS)
SEGMENTS_CZ = {
    S[0]: "Zatím není na řadě",
    S[1]: "První pozvánka",
    S[2]: "Pravidelná v prodlení, s gynekologem",
    S[3]: "Pravidelná v prodlení, bez gynekologa",
    S[4]: "Bez screeningu, chodí ke gynekologovi",
    S[5]: "Bez screeningu, jiné screeningy",
    S[6]: "Bez screeningu, jen běžná péče",
    S[7]: "Mimo zdravotní systém",
}
INTERVENTIONS_CZ = {
    S[0]: "neposílat; pozvat ~24 měsíců po screeningu",
    S[1]: "standardní dopis + snadné online objednání",
    S[2]: "připomenutí v termínu + gynekolog",
    S[3]: "připomenutí v termínu + praktický lékař",
    S[4]: "doporučení od vlastního gynekologa",
    S[5]: "spojit s jiným screeningem",
    S[6]: "osobní doporučení praktika / lékárníka",
    S[7]: "místo dopisů telefon, navigátor, mobilní jednotka",
}
QUESTIONS_CZ = {
    "q1": f"Screening před méně{NL}než 24 měsíci?",
    "q2": "První pozvánka?",
    "q3": f"Screening v posledních{NL}5 letech?",
    "q4a": f"Prevence u gynekologa{NL}(2 roky)?",
    "q4b": f"Prevence u gynekologa{NL}(2 roky)?",
    "q5": f"Jiný screeningový{NL}program (3 roky)?",
    "q6": f"Kontakt se{NL}zdravotnictvím (12 měs.)?",
}
assert QUESTIONS_CZ.keys() == QUESTIONS.keys()


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("  wrote", name, flush=True)


def label_bars(ax: plt.Axes, ys, xs, texts, inside_color=None) -> None:
    for y, x, t in zip(ys, xs, texts, strict=True):
        ax.text(
            x,
            y,
            f"  {t}",
            va="center",
            ha="left",
            fontsize=14,
            color=inside_color or INK,
        )


# ---------------------------------------------------------------- charts


def risk_deciles(data: dict, x: dict, numbers: dict) -> None:
    """Actual 12M participation by decile of the model's predicted probability."""
    model = make_model(x["train"]).fit(x["train"], data["train"][LABEL])
    p = model.predict_proba(x["test"])[:, 1]
    y = data["test"][LABEL].to_numpy()
    auc = roc_auc_score(y, p)
    decile = pd.qcut(p, 10, labels=False) + 1  # 1 = lowest predicted participation
    rate = pd.Series(y).groupby(decile).mean()
    numbers["model"] = {"test_roc_auc": auc, "decile_rates": rate.round(4).tolist()}

    fig, ax = plt.subplots(figsize=(13.3, 6.2))
    colors = [
        REDS(0.2 + 0.75 * (r - rate.min()) / (rate.max() - rate.min())) for r in rate
    ]
    ax.bar(rate.index, rate, color=colors, width=0.72)
    for d, r in rate.items():
        ax.text(
            d,
            r + 0.012,
            cz_pct(r),
            ha="center",
            fontsize=15,
            color=INK,
            fontweight="bold",
        )
    ax.axhline(y.mean(), color=INK2, lw=1.2, ls="--")
    ax.text(  # left, where the bars are still low
        0.55,
        y.mean() + 0.012,
        f"průměr {cz_pct(y.mean(), 1)}",
        ha="left",
        fontsize=13,
        color=INK2,
    )
    ax.set_xticks(range(1, 11), [f"{d}." for d in range(1, 11)])
    ax.set_xlabel(
        "decil skóre modelu  (1. = nejvyšší riziko neúčasti  →  10. = nejnižší)"
    )
    ax.set_ylabel("skutečná účast do 12 měsíců")
    ax.yaxis.set_major_formatter(PCT_AXIS)
    ax.set_ylim(0, rate.max() * 1.15)
    ax.grid(axis="x", visible=False)
    save(fig, "riziko_decily.png")


def model_comparison(segment_auc: float, numbers: dict) -> None:
    models = pd.read_csv(CLASSIFIERS / "explain" / "results" / "models.csv")
    models = models[models.model.isin(MODELS_CZ)].set_index("model").test_roc_auc
    rows = [(MODELS_CZ[m], a) for m, a in models.items()]
    rows.append(("8 pravidel (segmenty)", segment_auc))
    rows.sort(key=lambda r: r[1])
    numbers["models"] = {name: round(a, 4) for name, a in rows}

    fig, ax = plt.subplots(figsize=(13.3, 6.4))
    ys = np.arange(len(rows))
    best = max(a for _, a in rows)
    for yi, (name, auc) in zip(ys, rows, strict=True):
        color = (
            RED
            if auc == best
            else (ORANGE if name.startswith("8 pravidel") else GREY_DARK)
        )
        ax.hlines(yi, 0.75, auc, color=GRID, lw=3)
        ax.scatter(auc, yi, s=170, color=color, zorder=3)
        ax.text(
            auc + 0.003,
            yi,
            cz_num(auc),
            va="center",
            fontsize=14,
            color=INK,
            fontweight="bold" if auc == best else "normal",
        )
    ax.axvspan(best - 0.01, best, color="#F3E1DC", lw=0, zorder=0)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(0.75, 0.865)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: cz_num(v, 2)))
    ax.set_xlabel("ROC AUC na testovacích datech (0,5 = náhoda, 1 = dokonalé pořadí)")
    ax.grid(axis="y", visible=False)
    save(fig, "modely.png")


def learning_curves(numbers: dict) -> None:
    res = pd.read_csv(CLASSIFIERS / "ablation" / "results" / "ablation.csv")
    lc = res[res.study == "learning curve"]
    full = res[res.study == "capacity"].merge(
        lc[["family", "setting"]].drop_duplicates()
    )
    lc = pd.concat([lc, full])
    styles = {
        "gradient boosting": ("Gradient boosting", RED),
        "MLP": ("Neuronová síť", ORANGE),
        "decision tree": ("Rozhodovací strom", GREY_DARK),
        "logistic regression": ("Logistická regrese", GREY),
    }
    fig, ax = plt.subplots(figsize=(13.3, 6.2))
    ends = {}
    for fam, (name, color) in styles.items():
        s = lc[lc.family == fam].sort_values("train_rows")
        ax.plot(s.train_rows, s.test_roc_auc, color=color, lw=3, marker="o", ms=7)
        ends[name] = (s.train_rows.iloc[-1], s.test_roc_auc.iloc[-1], color)
    # direct labels at the right end, nudged apart
    order = sorted(ends.items(), key=lambda kv: kv[1][1])
    last = -1.0
    for name, (xr, yr, color) in order:
        ytxt = max(yr, last + 0.0022)
        ax.text(
            xr * 1.18,
            ytxt,
            name,
            va="center",
            fontsize=14,
            color=color,
            fontweight="bold",
        )
        last = ytxt
    ax.set_xscale("log")
    ax.set_xlim(right=lc.train_rows.max() * 7)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: cz_int(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: cz_num(v, 3)))
    ax.set_xlabel("počet trénovacích pozvánek (logaritmická osa)")
    ax.set_ylabel("ROC AUC na testu")
    ax.grid(axis="x", visible=False)
    gb = lc[lc.family == "gradient boosting"].sort_values("train_rows")
    numbers["learning_curve_gb"] = dict(
        zip(
            gb.train_fraction.round(2).astype(str),
            gb.test_roc_auc.round(4),
            strict=True,
        )
    )
    save(fig, "uceni.png")


def drivers(numbers: dict) -> None:
    perm = pd.read_csv(
        CLASSIFIERS / "explain" / "results" / "permutation_importance.csv"
    )
    gb = (
        perm[perm.model == "gradient boosting"]
        .set_index("feature")
        .importance.clip(lower=0)
    )
    share = (gb / gb.sum()).sort_values(ascending=False).head(8)[::-1]
    numbers["drivers_share"] = share[::-1].round(3).to_dict()

    fig, ax = plt.subplots(figsize=(13.3, 6.2))
    ys = np.arange(len(share))
    colors = [RED if i >= len(share) - 3 else GREY for i in range(len(share))]
    ax.barh(ys, share, color=colors, height=0.62)
    label_bars(ax, ys, share, [cz_pct(v) for v in share])
    ax.set_yticks(ys, [FEATURES_CZ.get(f, f) for f in share.index], fontsize=15)
    ax.xaxis.set_major_formatter(PCT_AXIS)
    ax.set_xlim(0, share.max() * 1.18)
    ax.set_xlabel("podíl na důležitosti v modelu (permutační důležitost)")
    ax.grid(axis="y", visible=False)
    save(fig, "faktory.png")


def effects(numbers: dict) -> None:
    pdp = pd.read_csv(CLASSIFIERS / "explain" / "results" / "partial_dependence.csv")
    pdp = pdp[pdp.model == "gradient boosting"]
    panels = [
        ("PRIOR_INVITATION_COUNT", "počet předchozích pozvánek"),
        ("MONTHS_SINCE_LAST_SCREEN", "měsíce od posledního screeningu"),
        ("GYN_PREVENTION_24M_CNT", "preventivní prohlídky u gynekologa (2 roky)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    for ax, (feat, xlabel) in zip(axes, panels, strict=True):
        s = pdp[pdp.feature == feat]
        xs, ys = s.value.astype(float).to_numpy(), s.mean_prediction.to_numpy()
        gaps = np.diff(xs)  # break the line where (almost) no women have values
        cut = np.flatnonzero(gaps > 5 * np.median(gaps)) + 1
        ax.plot(
            np.insert(xs, cut, np.nan),
            np.insert(ys, cut, np.nan),
            color=RED,
            lw=3.2,
            marker="o",
            ms=6,
        )
        ax.set_title(
            FEATURES_CZ[feat], fontsize=16, color=INK, fontweight="bold", loc="left"
        )
        ax.set_xlabel(xlabel)
        ax.yaxis.set_major_formatter(PCT_AXIS)
        numbers.setdefault("effects", {})[feat] = dict(
            zip(s.value, s.mean_prediction.round(3), strict=True)
        )
    axes[0].set_ylabel("předpovězená účast do 12 měsíců")
    axes[1].annotate(
        "téměř nikdo\nnení zván\n2–35 měs. po\nscreeningu",
        xy=(18, 0.2),
        ha="center",
        fontsize=12,
        color=INK2,
    )
    fig.tight_layout()
    save(fig, "efekty.png")


def segment_charts(df: pd.DataFrame, numbers: dict) -> None:
    seg = assign(df)
    stats = df.groupby(seg, observed=False)[LABEL].agg(["mean", "size"])
    stats["share"] = stats["size"] / len(df)
    numbers["segments"] = {
        SEGMENTS_CZ[s]: {
            "letter": s[0],
            "rate": round(r["mean"], 4),
            "share": round(r["share"], 4),
            "invitations": int(r["size"]),
        }
        for s, r in stats.iterrows()
    }

    # response by segment
    fig, ax = plt.subplots(figsize=(13.3, 6.4))
    ys = np.arange(len(stats))[::-1]
    norm = (stats["mean"] - stats["mean"].min()) / (
        stats["mean"].max() - stats["mean"].min()
    )
    ax.barh(ys, stats["mean"], color=[REDS(0.15 + 0.8 * v) for v in norm], height=0.66)
    for yi, (r, sh) in zip(
        ys, zip(stats["mean"], stats["share"], strict=True), strict=True
    ):
        ax.text(
            r + 0.008,
            yi,
            f"{cz_pct(r)}",
            va="center",
            fontsize=15,
            color=INK,
            fontweight="bold",
        )
        # share of invitations as its own right-hand column, clear of the bars
        ax.text(0.80, yi, cz_pct(sh), va="center", ha="right", fontsize=14, color=INK2)
    ax.text(0.80, ys[0] + 0.75, "podíl pozvánek", ha="right", fontsize=13, color=INK2)
    ax.axvline(df[LABEL].mean(), color=INK2, lw=1.2, ls="--")
    ax.set_yticks(ys, [f"{s[0]}  {SEGMENTS_CZ[s]}" for s in stats.index], fontsize=15)
    ax.xaxis.set_major_formatter(PCT_AXIS)
    ax.set_xlim(0, 0.82)
    ax.set_xticks(np.arange(0, 0.61, 0.1))  # no ticks under the share column
    ax.set_xlabel(
        f"účast do 12 měsíců (čárkovaně: všechny pozvánky, {cz_pct(df[LABEL].mean(), 1)})"
    )
    ax.grid(axis="y", visible=False)
    save(fig, "segmenty_ucast.png")

    # decision tree of the rules, in Czech
    texts = {
        "questions": QUESTIONS_CZ,
        "names": SEGMENTS_CZ,
        "interventions": INTERVENTIONS_CZ,
        "yes": "ano",
        "no": "ne",
        "stats": lambda n, share, rate: (
            f"{cz_int(n)} pozvánek ({cz_pct(share)}){NL}účast {cz_pct(rate, 1)} do 12 měs."
        ),
        "title": lambda rate: "",
        "cmap": "Reds",
        "font": FONT,
        "background": BG,
    }
    assert texts.keys() == TEXTS_EN.keys()
    fig = draw_segment_tree(df, texts=texts)
    save(fig, "strom_segmentu.png")

    # repeated letters. Starts at the 2nd letter: for these repeat invitations a "1."
    # means a sequence restarted after a long break, a different group.
    seq = pd.cut(
        df.INVITATION_SEQUENCE,
        [1, 2, 3, 4, 6, 8, 11],
        labels=["2.", "3.", "4.", "5.–6.", "7.–8.", "9.–11."],
    )
    decay = (
        df.groupby([seg, seq], observed=False)[LABEL]
        .agg(["mean", "size"])
        .reset_index()
    )
    decay.columns = ["segment", "seq", "rate", "n"]
    decay = decay[decay.n >= 500]
    show = {S[4]: RED, S[5]: ORANGE, S[6]: GREY_DARK, S[7]: GREY}
    fig, ax = plt.subplots(figsize=(13.3, 6.2))
    ends = []
    for s, color in show.items():
        t = decay[decay.segment == s]
        xs = [list(seq.cat.categories).index(v) for v in t.seq]
        ax.plot(xs, t.rate, color=color, lw=3.2, marker="o", ms=7)
        ends.append((t.rate.iloc[-1], xs[-1], s, color))
    last = -1.0
    for rate_end, x_end, s, color in sorted(ends):  # bottom up, nudged apart
        y_txt = max(rate_end, last + 0.022)
        ax.text(
            x_end + 0.12,
            y_txt,
            f"{s[0]}  {SEGMENTS_CZ[s]}",
            va="center",
            fontsize=13,
            color=color,
            fontweight="bold",
        )
        last = y_txt
    ax.set_xticks(range(len(seq.cat.categories)), seq.cat.categories)
    ax.set_xlim(-0.3, len(seq.cat.categories) + 2.6)
    ax.yaxis.set_major_formatter(PCT_AXIS)
    ax.set_xlabel("kolikátá pozvánka v řadě")
    ax.set_ylabel("účast do 12 měsíců")
    ax.grid(axis="x", visible=False)
    e = decay[decay.segment == S[4]].set_index("seq").rate
    numbers["decay_E"] = {str(k): round(v, 3) for k, v in e.items()}
    save(fig, "opakovane_dopisy.png")


def district_map(df: pd.DataFrame, numbers: dict) -> None:
    """12M participation per district (okres); matching as in exploration/kraj_map.py."""

    def key(name: str) -> str:
        return (
            unicodedata.normalize("NFKD", name)
            .encode("ascii", "ignore")
            .decode()
            .upper()
            .replace(" ", "")
        )

    rate = df.groupby("VZP_TERRITORIAL_AREA")[LABEL].mean().rename(index=key)
    rate = rate.rename(index={"PRAHA-HL.MESTO": "PRAHA", "OSTRAVA": "OSTRAVA-MESTO"})
    feats = []
    for name in ["okresy", "praha"]:
        with open(DATA / f"{name}.geojson", encoding="utf8") as f:
            feats += json.load(f)["features"]
    vals = [
        rate.get(key(f["properties"]["nazev"].replace("Hlavní město ", "")))
        for f in feats
    ]
    assert sum(v is not None for v in vals) == len(rate)
    numbers["districts"] = {"min": round(rate.min(), 4), "max": round(rate.max(), 4)}

    lo, hi = rate.min(), rate.max()
    fig, ax = plt.subplots(figsize=(13.3, 7.2))
    for f, v in zip(feats, vals, strict=True):
        geom = f["geometry"]
        polys = (
            [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        )
        fill = "#E3DED8" if v is None else REDS(0.05 + 0.9 * (v - lo) / (hi - lo))
        for poly in polys:
            ax.add_patch(Polygon(poly[0], closed=True, fc=fill, ec=BG, lw=0.8))
        if v is not None:
            pts = np.array(max((p[0] for p in polys), key=len))
            cx, cy = pts.mean(axis=0)
            ax.text(
                cx,
                cy,
                f"{v * 100:.0f}",
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if (v - lo) / (hi - lo) > 0.55 else INK,
            )
    ax.set_xlim(12.0, 19.0)
    ax.set_ylim(48.5, 51.1)
    ax.set_aspect(1 / np.cos(np.radians(49.8)))
    ax.axis("off")
    # colour scale
    cax = fig.add_axes([0.32, 0.06, 0.36, 0.025])
    grad = np.linspace(0, 1, 256)[None, :]
    cax.imshow(grad, aspect="auto", cmap=REDS, extent=[lo, hi, 0, 1])
    cax.set_yticks([])
    cax.xaxis.set_major_formatter(PCT_AXIS)
    cax.tick_params(labelsize=12)
    cax.set_title(
        "účast do 12 měsíců podle okresu (šedě: Bruntál a Jeseník nejsou v datech)",
        fontsize=12,
        color=INK2,
    )
    for spine in cax.spines.values():
        spine.set_visible(False)
    save(fig, "mapa_okresy.png")


def experiment(df: pd.DataFrame, numbers: dict) -> None:
    """Women per arm needed to detect +2 p.b. (alpha 0.05, power 80 %) vs 2024 volume."""
    seg = assign(df)
    z = 1.959964 + 0.841621
    delta = 0.02
    rows = []
    for s in S[2:]:
        p0 = df.loc[seg == s, LABEL].mean()
        p1 = p0 + delta
        need = z**2 * (p0 * (1 - p0) + p1 * (1 - p1)) / delta**2
        year = int(((seg == s) & (df.INVITATION_YEAR == 2024)).sum())
        rows.append((s, p0, need, year))
    numbers["experiment"] = {
        SEGMENTS_CZ[s]: {
            "baseline": round(p0, 4),
            "per_arm": int(np.ceil(n)),
            "invitations_2024": y,
        }
        for s, p0, n, y in rows
    }

    fig, ax = plt.subplots(figsize=(13.3, 6.2))
    ys = np.arange(len(rows))[::-1]
    ax.barh(
        ys + 0.18,
        [r[3] / 2 for r in rows],
        height=0.34,
        color=GREY,
        label="dostupné na skupinu (½ pozvánek 2024)",
    )
    ax.barh(
        ys - 0.18,
        [r[2] for r in rows],
        height=0.34,
        color=RED,
        label="potřeba na skupinu (+2 p. b.)",
    )
    for yi, (_, _, need, year) in zip(ys, rows, strict=True):
        ax.text(
            year / 2 * 1.08,
            yi + 0.18,
            cz_int(year / 2),
            va="center",
            fontsize=13,
            color=INK2,
        )
        ax.text(
            need * 1.08,
            yi - 0.18,
            cz_int(np.ceil(need)),
            va="center",
            fontsize=13,
            color=RED,
            fontweight="bold",
        )
    ax.set_xscale("log")
    ax.set_xlim(500, 6e5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: cz_int(v)))
    ax.set_yticks(ys, [f"{s[0]}  {SEGMENTS_CZ[s]}" for s, *_ in rows], fontsize=14)
    ax.set_xlabel("počet žen (logaritmická osa)")
    ax.legend(loc="lower right", fontsize=13)
    ax.grid(axis="y", visible=False)
    save(fig, "experiment_velikost.png")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    numbers: dict = {}
    data, x = prepare()
    full = pd.read_csv(
        DATA / "Rakhaton2026_prepared.csv",
        usecols=[
            *SEGMENT_COLUMNS,
            LABEL,
            "INVITATION_SEQUENCE",
            "INVITATION_YEAR",
            "VZP_TERRITORIAL_AREA",
            "PERSON_ID",
        ],
        dtype={"PERSON_ID": str},
    )
    numbers["data"] = {
        "invitations": len(full),
        "women": int(full.PERSON_ID.nunique()),
        "rate_12m": round(full[LABEL].mean(), 4),
    }

    seg_train, seg_test = assign(data["train"]), assign(data["test"])
    rates = data["train"][LABEL].groupby(seg_train, observed=False).mean()
    segment_auc = roc_auc_score(data["test"][LABEL], seg_test.map(rates).astype(float))

    print("charts:", flush=True)
    risk_deciles(data, x, numbers)
    model_comparison(segment_auc, numbers)
    learning_curves(numbers)
    drivers(numbers)
    effects(numbers)
    segment_charts(full, numbers)
    district_map(full, numbers)
    experiment(full, numbers)
    (OUT / "cisla.json").write_text(
        json.dumps(numbers, indent=2, ensure_ascii=False), encoding="utf8"
    )
    print("  wrote cisla.json")


if __name__ == "__main__":
    main()
