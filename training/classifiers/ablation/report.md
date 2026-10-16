# Ablation: every model family converges to the same ceiling

**Question.** Can any model or setting beat the gradient-boosting baseline, or is performance limited by the data?

**Setup** (`run.py`, results in `results/`)
- Label `SCREENING_12M_FLAG` (19.3% positive). Official patient-level split: fit on **train**, choose settings on
  **val**, report on **test**.
- **Capacity study:** 9 model families from tiny to huge (and deliberately overfitting) settings, 38 fits, all on
  the full train split and the 19 raw features.
- **Learning curve:** logistic regression, decision tree, gradient boosting and an MLP trained on 1%–100% of the
  train women.
- **Feature study:** the same models on the 96 engineered features of the prepared data instead of the 19 raw ones.
- **Train ROC AUC** (on up to 200k of the rows the model was fitted on) shows how much a model memorises.
  A large train−test gap means overfitting.
- **Best per family:** the setting with the best *val* ROC AUC. Its *test* ROC AUC gap to the overall best has a
  95% bootstrap interval that resamples women (with all their invitations).

The question is whether the families meet at the same test ROC AUC, and whether more capacity or more data
still moves it.


```python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path("results")
res = pd.read_csv(RESULTS / "ablation.csv")
best = pd.read_csv(RESULTS / "best_per_family.csv")
cap = res[res.study == "capacity"]

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK2, GRID = "#a3a29d", "#52514e", "#e4e3df"
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titleweight": "bold",
        "axes.titlesize": 10,
        "axes.titlelocation": "left",
        "legend.frameon": False,
    }
)
TOP = best.test_roc_auc.max()
print(
    f"{len(res)} fits ({res.study.value_counts().to_dict()}), {res.fit_seconds.sum() / 60:.0f} min of fitting"
)
print(f"Best test ROC AUC: {TOP:.4f} ({best.iloc[0].family}, {best.iloc[0].setting})")
```

    57 fits ({'capacity': 38, 'learning curve': 16, 'features': 3}), 15 min of fitting
    Best test ROC AUC: 0.8413 (gradient boosting, 63 leaves, lr .05, 1000 iter)
    

## 1. Best setting of each family


```python
table = best.set_index("family")[
    [
        "setting",
        "val_roc_auc",
        "test_roc_auc",
        "test_pr_auc",
        "train_roc_auc",
        "gap_to_best",
        "gap_low",
        "gap_high",
        "fit_seconds",
    ]
]
table.style.format(
    {
        "val_roc_auc": "{:.4f}",
        "test_roc_auc": "{:.4f}",
        "test_pr_auc": "{:.4f}",
        "train_roc_auc": "{:.4f}",
        "gap_to_best": "{:+.4f}",
        "gap_low": "{:+.4f}",
        "gap_high": "{:+.4f}",
        "fit_seconds": "{:,}",
    }
)
```




<style type="text/css">
</style>
<table id="T_b6ab9">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_b6ab9_level0_col0" class="col_heading level0 col0" >setting</th>
      <th id="T_b6ab9_level0_col1" class="col_heading level0 col1" >val_roc_auc</th>
      <th id="T_b6ab9_level0_col2" class="col_heading level0 col2" >test_roc_auc</th>
      <th id="T_b6ab9_level0_col3" class="col_heading level0 col3" >test_pr_auc</th>
      <th id="T_b6ab9_level0_col4" class="col_heading level0 col4" >train_roc_auc</th>
      <th id="T_b6ab9_level0_col5" class="col_heading level0 col5" >gap_to_best</th>
      <th id="T_b6ab9_level0_col6" class="col_heading level0 col6" >gap_low</th>
      <th id="T_b6ab9_level0_col7" class="col_heading level0 col7" >gap_high</th>
      <th id="T_b6ab9_level0_col8" class="col_heading level0 col8" >fit_seconds</th>
    </tr>
    <tr>
      <th class="index_name level0" >family</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
      <th class="blank col7" >&nbsp;</th>
      <th class="blank col8" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b6ab9_level0_row0" class="row_heading level0 row0" >gradient boosting</th>
      <td id="T_b6ab9_row0_col0" class="data row0 col0" >63 leaves, lr .05, 1000 iter</td>
      <td id="T_b6ab9_row0_col1" class="data row0 col1" >0.8389</td>
      <td id="T_b6ab9_row0_col2" class="data row0 col2" >0.8413</td>
      <td id="T_b6ab9_row0_col3" class="data row0 col3" >0.5810</td>
      <td id="T_b6ab9_row0_col4" class="data row0 col4" >0.8483</td>
      <td id="T_b6ab9_row0_col5" class="data row0 col5" >+0.0000</td>
      <td id="T_b6ab9_row0_col6" class="data row0 col6" >+0.0000</td>
      <td id="T_b6ab9_row0_col7" class="data row0 col7" >+0.0000</td>
      <td id="T_b6ab9_row0_col8" class="data row0 col8" >11</td>
    </tr>
    <tr>
      <th id="T_b6ab9_level0_row1" class="row_heading level0 row1" >MLP</th>
      <td id="T_b6ab9_row1_col0" class="data row1 col0" >3 layers, strong L2 (alpha 1e-2)</td>
      <td id="T_b6ab9_row1_col1" class="data row1 col1" >0.8380</td>
      <td id="T_b6ab9_row1_col2" class="data row1 col2" >0.8404</td>
      <td id="T_b6ab9_row1_col3" class="data row1 col3" >0.5792</td>
      <td id="T_b6ab9_row1_col4" class="data row1 col4" >0.8431</td>
      <td id="T_b6ab9_row1_col5" class="data row1 col5" >-0.0009</td>
      <td id="T_b6ab9_row1_col6" class="data row1 col6" >-0.0013</td>
      <td id="T_b6ab9_row1_col7" class="data row1 col7" >-0.0005</td>
      <td id="T_b6ab9_row1_col8" class="data row1 col8" >115</td>
    </tr>
    <tr>
      <th id="T_b6ab9_level0_row2" class="row_heading level0 row2" >random forest</th>
      <td id="T_b6ab9_row2_col0" class="data row2 col0" >depth 16, leaf 20</td>
      <td id="T_b6ab9_row2_col1" class="data row2 col1" >0.8377</td>
      <td id="T_b6ab9_row2_col2" class="data row2 col2" >0.8400</td>
      <td id="T_b6ab9_row2_col3" class="data row2 col3" >0.5778</td>
      <td id="T_b6ab9_row2_col4" class="data row2 col4" >0.8479</td>
      <td id="T_b6ab9_row2_col5" class="data row2 col5" >-0.0013</td>
      <td id="T_b6ab9_row2_col6" class="data row2 col6" >-0.0018</td>
      <td id="T_b6ab9_row2_col7" class="data row2 col7" >-0.0009</td>
      <td id="T_b6ab9_row2_col8" class="data row2 col8" >13</td>
    </tr>
    <tr>
      <th id="T_b6ab9_level0_row3" class="row_heading level0 row3" >decision tree</th>
      <td id="T_b6ab9_row3_col0" class="data row3 col0" >depth 12</td>
      <td id="T_b6ab9_row3_col1" class="data row3 col1" >0.8344</td>
      <td id="T_b6ab9_row3_col2" class="data row3 col2" >0.8373</td>
      <td id="T_b6ab9_row3_col3" class="data row3 col3" >0.5710</td>
      <td id="T_b6ab9_row3_col4" class="data row3 col4" >0.8394</td>
      <td id="T_b6ab9_row3_col5" class="data row3 col5" >-0.0040</td>
      <td id="T_b6ab9_row3_col6" class="data row3 col6" >-0.0046</td>
      <td id="T_b6ab9_row3_col7" class="data row3 col7" >-0.0035</td>
      <td id="T_b6ab9_row3_col8" class="data row3 col8" >6</td>
    </tr>
    <tr>
      <th id="T_b6ab9_level0_row4" class="row_heading level0 row4" >logistic regression</th>
      <td id="T_b6ab9_row4_col0" class="data row4 col0" >splines (additive)</td>
      <td id="T_b6ab9_row4_col1" class="data row4 col1" >0.8331</td>
      <td id="T_b6ab9_row4_col2" class="data row4 col2" >0.8346</td>
      <td id="T_b6ab9_row4_col3" class="data row4 col3" >0.5619</td>
      <td id="T_b6ab9_row4_col4" class="data row4 col4" >0.8336</td>
      <td id="T_b6ab9_row4_col5" class="data row4 col5" >-0.0067</td>
      <td id="T_b6ab9_row4_col6" class="data row4 col6" >-0.0073</td>
      <td id="T_b6ab9_row4_col7" class="data row4 col7" >-0.0062</td>
      <td id="T_b6ab9_row4_col8" class="data row4 col8" >11</td>
    </tr>
    <tr>
      <th id="T_b6ab9_level0_row5" class="row_heading level0 row5" >kNN (50k rows)</th>
      <td id="T_b6ab9_row5_col0" class="data row5 col0" >k=250</td>
      <td id="T_b6ab9_row5_col1" class="data row5 col1" >0.8295</td>
      <td id="T_b6ab9_row5_col2" class="data row5 col2" >0.8314</td>
      <td id="T_b6ab9_row5_col3" class="data row5 col3" >0.5635</td>
      <td id="T_b6ab9_row5_col4" class="data row5 col4" >0.8342</td>
      <td id="T_b6ab9_row5_col5" class="data row5 col5" >-0.0100</td>
      <td id="T_b6ab9_row5_col6" class="data row5 col6" >-0.0107</td>
      <td id="T_b6ab9_row5_col7" class="data row5 col7" >-0.0092</td>
      <td id="T_b6ab9_row5_col8" class="data row5 col8" >0</td>
    </tr>
    <tr>
      <th id="T_b6ab9_level0_row6" class="row_heading level0 row6" >few-feature LR</th>
      <td id="T_b6ab9_row6_col0" class="data row6 col0" >5 features</td>
      <td id="T_b6ab9_row6_col1" class="data row6 col1" >0.8145</td>
      <td id="T_b6ab9_row6_col2" class="data row6 col2" >0.8181</td>
      <td id="T_b6ab9_row6_col3" class="data row6 col3" >0.5273</td>
      <td id="T_b6ab9_row6_col4" class="data row6 col4" >0.8161</td>
      <td id="T_b6ab9_row6_col5" class="data row6 col5" >-0.0232</td>
      <td id="T_b6ab9_row6_col6" class="data row6 col6" >-0.0243</td>
      <td id="T_b6ab9_row6_col7" class="data row6 col7" >-0.0221</td>
      <td id="T_b6ab9_row6_col8" class="data row6 col8" >1</td>
    </tr>
    <tr>
      <th id="T_b6ab9_level0_row7" class="row_heading level0 row7" >naive Bayes</th>
      <td id="T_b6ab9_row7_col0" class="data row7 col0" >smoothing 1e-2</td>
      <td id="T_b6ab9_row7_col1" class="data row7 col1" >0.7973</td>
      <td id="T_b6ab9_row7_col2" class="data row7 col2" >0.7998</td>
      <td id="T_b6ab9_row7_col3" class="data row7 col3" >0.4710</td>
      <td id="T_b6ab9_row7_col4" class="data row7 col4" >0.7979</td>
      <td id="T_b6ab9_row7_col5" class="data row7 col5" >-0.0415</td>
      <td id="T_b6ab9_row7_col6" class="data row7 col6" >-0.0428</td>
      <td id="T_b6ab9_row7_col7" class="data row7 col7" >-0.0403</td>
      <td id="T_b6ab9_row7_col8" class="data row7 col8" >3</td>
    </tr>
    <tr>
      <th id="T_b6ab9_level0_row8" class="row_heading level0 row8" >constant</th>
      <td id="T_b6ab9_row8_col0" class="data row8 col0" >base rate</td>
      <td id="T_b6ab9_row8_col1" class="data row8 col1" >0.5000</td>
      <td id="T_b6ab9_row8_col2" class="data row8 col2" >0.5000</td>
      <td id="T_b6ab9_row8_col3" class="data row8 col3" >0.1929</td>
      <td id="T_b6ab9_row8_col4" class="data row8 col4" >0.5000</td>
      <td id="T_b6ab9_row8_col5" class="data row8 col5" >-0.3413</td>
      <td id="T_b6ab9_row8_col6" class="data row8 col6" >-0.3435</td>
      <td id="T_b6ab9_row8_col7" class="data row8 col7" >-0.3388</td>
      <td id="T_b6ab9_row8_col8" class="data row8 col8" >0</td>
    </tr>
  </tbody>
</table>




## 2. All 38 settings

Each dot is one setting, at its test ROC AUC. Blue = the setting chosen on val; gray = the others.
The band marks the best test ROC AUC − 0.005, a difference too small to matter in practice.


```python
fams = best.family.tolist()[::-1]  # weakest at the bottom
fig, ax = plt.subplots(figsize=(9, 0.45 * len(fams) + 1))
ax.axvspan(TOP - 0.005, TOP, color=GRID, alpha=0.7, lw=0)
for i, fam in enumerate(fams):
    s = cap[cap.family == fam]
    chosen = best.set_index("family").loc[fam, "setting"]
    other = s[s.setting != chosen]
    ax.scatter(other.test_roc_auc, np.full(len(other), i), color=GRAY, s=28, zorder=3)
    pick = s[s.setting == chosen]
    ax.scatter(pick.test_roc_auc, [i], color=BLUE, s=55, zorder=4)
    ax.annotate(
        chosen,
        (pick.test_roc_auc.iloc[0], i),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7,
        color=INK2,
    )
ax.set_yticks(range(len(fams)), fams)
ax.grid(axis="y", visible=False)
ax.set_xlim(0.48, TOP + 0.15)  # room for the labels
ax.set_xlabel("test ROC AUC (12M)")
ax.set_title("Every setting of every family (blue: chosen on val)")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 0.45 * len(fams) + 1))  # zoom on the top
ax.axvspan(TOP - 0.005, TOP, color=GRID, alpha=0.7, lw=0)
for i, fam in enumerate(fams):
    s = cap[cap.family == fam]
    chosen = best.set_index("family").loc[fam, "setting"]
    ax.scatter(
        s[s.setting != chosen].test_roc_auc,
        np.full((s.setting != chosen).sum(), i),
        color=GRAY,
        s=28,
        zorder=3,
    )
    ax.scatter(s[s.setting == chosen].test_roc_auc, [i], color=BLUE, s=55, zorder=4)
ax.set_yticks(range(len(fams)), fams)
ax.grid(axis="y", visible=False)
ax.set_xlim(0.795, TOP + 0.005)
ax.set_xlabel("test ROC AUC (12M), zoomed to 0.795+")
ax.set_title("Zoom: the strong families meet at the same ceiling")
plt.tight_layout()
plt.show()
```


    
![png](report_files/report_5_0.png)
    



    
![png](report_files/report_5_1.png)
    


## 3. More capacity: test ROC AUC levels off while train ROC AUC keeps rising

Settings are ordered from small to large within each family. Solid = test, dashed = train (the rows the model
was fitted on); the dotted line is the overall best test ROC AUC. When the dashed line goes up and the solid line
does not, the extra capacity only memorises the training data.


```python
families = [f for f in best.family if f != "constant"]
ncol = 4
nrow = int(np.ceil(len(families) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(15, 3.4 * nrow), sharey=True)
for ax, fam in zip(axes.flat, families, strict=False):
    s = cap[cap.family == fam]
    xs = np.arange(len(s))
    ax.plot(xs, s.train_roc_auc, color=GRAY, ls="--", marker="o", ms=4, label="train")
    ax.plot(xs, s.test_roc_auc, color=BLUE, marker="o", ms=4, label="test")
    ax.axhline(TOP, color=INK2, lw=0.8, ls=":")
    ax.set_xticks(xs, s.setting, rotation=40, ha="right", fontsize=7)
    ax.set_title(fam)
    ax.set_ylim(0.60, 1.0)
for ax in axes.flat[len(families) :]:
    ax.set_visible(False)
axes.flat[0].legend(loc="lower right", fontsize=8)
axes.flat[0].set_ylabel("ROC AUC (12M)")
plt.tight_layout()
plt.show()
```


    
![png](report_files/report_7_0.png)
    


## 4. More data: learning curves flatten at the same level

Test ROC AUC (solid) and train ROC AUC (dashed) by number of training rows. Training subsets are nested samples of
women (1%, 3%, 10%, 30%, 100%).


```python
lc = pd.concat(
    [
        res[res.study == "learning curve"],
        cap.merge(
            res[res.study == "learning curve"][["family", "setting"]].drop_duplicates()
        ),
    ]
)
colors = {
    "logistic regression": BLUE,
    "decision tree": ORANGE,
    "gradient boosting": AQUA,
    "MLP": YELLOW,
}
fig, ax = plt.subplots(figsize=(9, 4.5))
for fam, color in colors.items():
    s = lc[lc.family == fam].sort_values("train_rows")
    ax.plot(
        s.train_rows,
        s.test_roc_auc,
        color=color,
        marker="o",
        ms=5,
        lw=2,
        label=f"{fam} ({s.setting.iloc[0]})",
    )
    ax.plot(s.train_rows, s.train_roc_auc, color=color, ls="--", lw=1.2, alpha=0.8)
ax.axhline(TOP, color=INK2, lw=0.8, ls=":")
ax.set_xscale("log")
ax.set_xlabel("training rows (log scale)")
ax.set_ylabel("ROC AUC (12M)")
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Learning curves: solid = test, dashed = train")
plt.tight_layout()
plt.show()

pivot = lc.pivot_table(index="family", columns="train_fraction", values="test_roc_auc")
pivot.columns = [f"{c:.0%}" for c in pivot.columns]
pivot.style.format("{:.4f}")
```


    
![png](report_files/report_9_0.png)
    





<style type="text/css">
</style>
<table id="T_09bab">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_09bab_level0_col0" class="col_heading level0 col0" >1%</th>
      <th id="T_09bab_level0_col1" class="col_heading level0 col1" >3%</th>
      <th id="T_09bab_level0_col2" class="col_heading level0 col2" >10%</th>
      <th id="T_09bab_level0_col3" class="col_heading level0 col3" >30%</th>
      <th id="T_09bab_level0_col4" class="col_heading level0 col4" >100%</th>
    </tr>
    <tr>
      <th class="index_name level0" >family</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_09bab_level0_row0" class="row_heading level0 row0" >MLP</th>
      <td id="T_09bab_row0_col0" class="data row0 col0" >0.8183</td>
      <td id="T_09bab_row0_col1" class="data row0 col1" >0.8280</td>
      <td id="T_09bab_row0_col2" class="data row0 col2" >0.8358</td>
      <td id="T_09bab_row0_col3" class="data row0 col3" >0.8379</td>
      <td id="T_09bab_row0_col4" class="data row0 col4" >0.8398</td>
    </tr>
    <tr>
      <th id="T_09bab_level0_row1" class="row_heading level0 row1" >decision tree</th>
      <td id="T_09bab_row1_col0" class="data row1 col0" >0.8071</td>
      <td id="T_09bab_row1_col1" class="data row1 col1" >0.8265</td>
      <td id="T_09bab_row1_col2" class="data row1 col2" >0.8329</td>
      <td id="T_09bab_row1_col3" class="data row1 col3" >0.8350</td>
      <td id="T_09bab_row1_col4" class="data row1 col4" >0.8364</td>
    </tr>
    <tr>
      <th id="T_09bab_level0_row2" class="row_heading level0 row2" >gradient boosting</th>
      <td id="T_09bab_row2_col0" class="data row2 col0" >0.8151</td>
      <td id="T_09bab_row2_col1" class="data row2 col1" >0.8279</td>
      <td id="T_09bab_row2_col2" class="data row2 col2" >0.8356</td>
      <td id="T_09bab_row2_col3" class="data row2 col3" >0.8391</td>
      <td id="T_09bab_row2_col4" class="data row2 col4" >0.8412</td>
    </tr>
    <tr>
      <th id="T_09bab_level0_row3" class="row_heading level0 row3" >logistic regression</th>
      <td id="T_09bab_row3_col0" class="data row3 col0" >0.8265</td>
      <td id="T_09bab_row3_col1" class="data row3 col1" >0.8302</td>
      <td id="T_09bab_row3_col2" class="data row3 col2" >0.8316</td>
      <td id="T_09bab_row3_col3" class="data row3 col3" >0.8318</td>
      <td id="T_09bab_row3_col4" class="data row3 col4" >0.8319</td>
    </tr>
  </tbody>
</table>




## 5. More features: 96 engineered vs 19 raw


```python
feat = res[res.study == "features"].merge(
    cap, on=["family", "setting"], suffixes=("_engineered", "_raw")
)
feat = feat.assign(gain=feat.test_roc_auc_engineered - feat.test_roc_auc_raw)
feat.set_index(["family", "setting"])[
    ["test_roc_auc_raw", "test_roc_auc_engineered", "gain"]
].style.format(
    {
        "test_roc_auc_raw": "{:.4f}",
        "test_roc_auc_engineered": "{:.4f}",
        "gain": "{:+.4f}",
    }
)
```




<style type="text/css">
</style>
<table id="T_fe5c5">
  <thead>
    <tr>
      <th class="blank" >&nbsp;</th>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_fe5c5_level0_col0" class="col_heading level0 col0" >test_roc_auc_raw</th>
      <th id="T_fe5c5_level0_col1" class="col_heading level0 col1" >test_roc_auc_engineered</th>
      <th id="T_fe5c5_level0_col2" class="col_heading level0 col2" >gain</th>
    </tr>
    <tr>
      <th class="index_name level0" >family</th>
      <th class="index_name level1" >setting</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_fe5c5_level0_row0" class="row_heading level0 row0" >logistic regression</th>
      <th id="T_fe5c5_level1_row0" class="row_heading level1 row0" >C=1</th>
      <td id="T_fe5c5_row0_col0" class="data row0 col0" >0.8319</td>
      <td id="T_fe5c5_row0_col1" class="data row0 col1" >0.8351</td>
      <td id="T_fe5c5_row0_col2" class="data row0 col2" >+0.0032</td>
    </tr>
    <tr>
      <th id="T_fe5c5_level0_row1" class="row_heading level0 row1" >gradient boosting</th>
      <th id="T_fe5c5_level1_row1" class="row_heading level1 row1" >31 leaves (default)</th>
      <td id="T_fe5c5_row1_col0" class="data row1 col0" >0.8412</td>
      <td id="T_fe5c5_row1_col1" class="data row1 col1" >0.8423</td>
      <td id="T_fe5c5_row1_col2" class="data row1 col2" >+0.0011</td>
    </tr>
    <tr>
      <th id="T_fe5c5_level0_row2" class="row_heading level0 row2" >MLP</th>
      <th id="T_fe5c5_level1_row2" class="row_heading level1 row2" >2 layers (128-64)</th>
      <td id="T_fe5c5_row2_col0" class="data row2 col0" >0.8398</td>
      <td id="T_fe5c5_row2_col1" class="data row2 col1" >0.8416</td>
      <td id="T_fe5c5_row2_col2" class="data row2 col2" >+0.0019</td>
    </tr>
  </tbody>
</table>




## 6. Conclusions

**1. All capable model families meet at the same ceiling, test ROC AUC ≈ 0.84.** Best setting per family (chosen on val):

| family | test ROC AUC | gap to best |
|---|---|---|
| gradient boosting | 0.8413 | — |
| MLP | 0.8404 | −0.0009 |
| random forest | 0.8400 | −0.0013 |
| single decision tree | 0.8373 | −0.0040 |
| additive (spline) logistic regression | 0.8346 | −0.0067 |
| kNN (on only 50k rows) | 0.8314 | −0.0100 |

These are very different model classes: boosted and bagged trees, a neural network, one tree, an additive
model, nearest neighbours. They all end within 0.01 of each other, and the three flexible ones within 0.0013.
The bootstrap intervals show the gaps are real but tiny. No model class has an advantage that would matter in
practice.

**2. More capacity does not raise the ceiling; it only memorises the training data.**
- Gradient boosting from 7 to 63 leaves: test 0.8393 → 0.8413 (+0.002), then down to 0.8404 at 255 leaves.
  Without early stopping: train 0.913, test 0.834.
- MLP from 1 to 4 hidden layers: test 0.8390 → 0.8399 (+0.001), at 6× the fit time.
  Without early stopping: train 0.867, test 0.824.
- Decision tree: test peaks at depth 12 (0.8373). Fully grown: train **1.000**, test **0.635**.
- Random forest: depth 16 best (0.8400). Fully grown: train 0.964, test 0.830.
- Logistic regression is insensitive to regularisation (C = 0.01 … 100: test 0.8318–0.8319). Letting each
  feature bend (splines) adds only +0.0027.

**3. More data does not raise the ceiling either.** Going from 30% to 100% of the train women (282k → 942k rows):
logistic regression +0.0001, decision tree +0.0014, MLP +0.0019, gradient boosting +0.0021. The curves are flat,
and train ROC AUC falls to meet test ROC AUC (gradient boosting: train 0.956 on 1% of the data → 0.845 on 100%).
When the training score drops to the test score, the model can no longer even fit its own training data much
better: the remaining errors are noise *with respect to these features*.

**4. More features do not raise it either.** The 96 engineered features (history, spatial, dates, flags) add
+0.001 to +0.003 (best overall: gradient boosting on engineered features, 0.8423).

**5. A handful of features already gets most of the way.** Logistic regression on 3 features
(`PRIOR_INVITATION_COUNT`, `GYN_PREVENTION_24M_CNT`, `MONTHS_SINCE_LAST_SCREEN`) reaches 0.814. With 5 features it
reaches 0.818, about 93% of the way from chance (0.5) to the best model (0.841).

**Conclusion.** Model class, capacity, regularisation, training-set size and feature engineering all lead to the
same test ROC AUC of about 0.84. The limit is the information in the data, not the model. To predict
better, VZP would need *new* information about each woman. Examples: how and when the invitation was delivered
and whether it was read, reminders, the GP's own screening practice, socio-economic context, individual travel
time, past missed appointments.

**Caveats.** 12M label only (the 3M and 6M models behave the same in `compare.py`). kNN is fitted on a 50k-row
sample because of prediction cost. Random forests use 25% of rows per tree. The settings are broad but not an
exhaustive search. The convergence across families is what makes the case, not any single setting.
