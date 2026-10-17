---
name: agent-adam-claude-feature-set-comparison
description: Adam Kukucka's agent - GB on the official split with raw vs engineered/history/spatial/date feature groups (2026-10-16): no group adds more than ~0.001 AUC
metadata:
  type: project
  author: Claude agent in Adam Kukucka's session
  date: 2026-10-16
---

`features/compare_feature_sets.py` (no CLI args, reads `data/splits/`, scores val only, writes `features/feature_set_comparison.csv`): one untuned `HistGradientBoostingClassifier` per label and feature set, on the official patient-level split.

Val ROC AUC (12M / 3M / 6M); PR AUC moves the same way, at most +0.001:
- raw 19 columns: 0.8388 / 0.8361 / 0.8378
- + engineered (12 cols: count diff, age mean, ordinals, reset flag, recency flags, contact log/zero, prevention total/types): 0.8388 / 0.8360 / 0.8375 (no change)
- + date (year, month, quarter, days since 2020): 0.8391 / 0.8369 / 0.8381
- + spatial (10 `SPATIAL_*`): 0.8388 / 0.8360 / 0.8376 (no change; the area column already carries it)
- + history (`H_*`): 0.8395 / 0.8378 / 0.8387
- all without date: 0.8395 / 0.8375 / 0.8386
- all with date: 0.8397 / 0.8384 / 0.8390 (best, +0.001)

**Conclusion:** tree models are saturated on this data at ROC AUC ~0.84. Engineered columns are mostly transformations that trees learn on their own. Date features help a little only because all years appear in every patient-level split; they would extrapolate badly to a future year, so do not rely on them.

**Gotcha:** `data/splits/` was stale (made before the refactor: 25 raw columns, A/N flags), so scripts crashed with "could not convert string to float: 'N'". Regenerate with `cd preprocessing && uv run python split.py` after any change to `prepare_data.py`.

**Not yet measured:** whether the `LIN_*` / engineered columns close the gap for logistic regression (LR 0.832 vs GB 0.841 on raw). Related: [[feature-engineering]], [[baseline-results]], [[split-decision]].

## Separate models per population group (`features/compare_split_models.py`, `features/split_model_comparison.csv`)
One global GB vs one GB per group, same official split, val only, pooled predictions, raw-19 and all-no-date feature sets.
- Groupings: first vs repeat; never vs ever screened; three-way first / repeat-ever / repeat-never.
- **Result: no gain.** Pooled ROC AUC of the split models is equal or 0.0001-0.0008 lower than the global model for every grouping, label and feature set (e.g. 12M raw: global 0.8388, first/repeat 0.8384, three-way 0.8380). Brier is the same, so no calibration problem either. Trees already split on these flags.
- Per-group ROC AUC (global model): first 0.71 (3M) - 0.76 (12M); repeat 0.79-0.81; never-screened 0.74-0.75; ever-screened 0.85; repeat-never 0.77-0.80; repeat-ever 0.72-0.73. The weak spots are first-time invitees and repeat women who were screened before, not a modelling-structure problem.
- Do not pursue more group splits of this kind; the missing signal must come from new information.

## Confident groups (`features/confident_groups.py`, official val split, GB on raw + engineered)
- Certainty exists only on the "will not attend" side. 12M: p<=0.05 covers 31% of women and only 1-3% of them screen; 'will not attend' at >=95% precision covers 53%, at >=98% covers 18% (base rate of non-attenders is 81%, so 90% is nearly trivial). Nobody reaches 90% 'will attend': the top bin (p 0.7-0.9, 3.8% of women) has 76% response. For 3M, 92% of women can be called non-attenders at 95% precision.
- Biggest clean rule (val, verified): PRIOR_INVITATION_COUNT > 2 and INVITATION_SEQUENCE >= 5 and no gynaecologist prevention visit in 24 months = 38.9% of invitations, 4.5% screen. Adding zero healthcare contacts: 4.6% of invitations, 1.8% screen. Sequence >= 7 and no gyn prevention: 30% of invitations, 3.4% screen.
- Caveat for interventions: near-certain non-attendance does not mean persuadable (uplift is unknown without the experiment). The uncertain middle (p 0.2-0.5, ~30% of women) is where a reminder has most room to change an outcome.

## 99%-certain non-attenders (`features/certain_non_attenders.py`, threshold chosen on val, applied to test)
- Model score (GB, raw + engineered), largest lowest-score set with <=1% screening on val: 12M 3.9% of women (test screened 1.11%, i.e. just misses 99%), 6M 16.4% (test 0.97%), 3M 45% (test 0.95%). Use a slightly stricter threshold if 99% must hold.
- Rule mining (deeper tree, leaves >=1000 rows): only one tiny leaf (0.1% of women) reaches <=1%, so no useful simple rule for 12M at 99%.
- Who the 3.9% are (test): ~72% sequence >=7 (36% overall), 56% zero healthcare contacts (7% overall), 56% not registered with a gynaecologist (21% overall), and 28% screened 0-3 months before the invitation (2.7% overall). Two sub-populations: chronic disconnected non-responders, and women who just screened (already covered).

## External data: UZIS district x age screening coverage (`features/test_uzis_coverage.py`)
- Source: UZIS/NZIP open data PPS-01-01 (CSV, CC BY 4.0; downloaded to `data/uzis_mamo_coverage.csv`, 2011-2024, 77 districts, 5-year age groups 45+). VZP area -> district by accent-insensitive name; 73/75 match directly, OSTRAVA -> Ostrava-mesto and PRAHA-HL.MESTO -> Praha by hand (Jesenik and Bruntal districts have no VZP area). 96% of rows get coverage (age 40-44 has none).
- Features for an invitation in year Y use only year Y-1 (and Y-3 for the trend): district x age coverage, district total coverage, trend, difference from national, district-age population.
- **Result: no gain.** Val ROC AUC 12M 0.8390 vs 0.8388 without; 3M 0.8361 vs 0.8360; 6M 0.8377 vs 0.8375.
- Counter-intuitive raw pattern: the HIGHER the district x age past coverage, the LOWER the response of invited women (train 12M: 23.4% in the lowest coverage quartile, 12.8% in the highest). Likely selection (invited women are the residual non-attenders, so high-coverage districts leave harder cases) plus age confounding (older groups have higher coverage and lower response). Do not read it as "coverage causes non-attendance".

## Hypothesis checks on external trial data (decision: no new features; test intervention hypotheses elsewhere)
Hypotheses (evidence from literature, see chat summary): H1 timed/pre-booked appointment vs open invitation (English RCT of women who missed an appointment: 22.3% vs 12.3%); H2 reminder (letter/SMS/phone); H3 personal contact / navigation for chronic non-responders; H4 GP/gynaecologist letter (weak, old, non-significant trials); H5 tailored letters (Flanders trial, results pending). Our own data speaks against convenience barriers: wait time at nearest centre OR 1.02, distance OR 0.93; invitation timing (month, holidays, weekday) has no effect (12 batches per year, one per month).
- **Downloaded and tested: Indonesian SMS-reminder RCT** (Marcus et al. 2024, J Econ Behav Organ; DOI 10.25625/SE4IDP, CC BY-NC-SA 4.0; files in `data/external/indonesia_sms/`, script `features/test_sms_trial.py`). Hypertension/diabetes screening in Aceh, not mammography; randomised by household; 1,389 endline respondents aged 40-71. Screened 33.1% control vs 38.8% SMS: +5.7 pp (se 2.6, p=0.028; +17% relative); adjusted +6.9 pp (p=0.006). No significant subgroup interactions (ever-screened +5.5 vs +6.0 never; age 50+ +2.9 vs <50 +8.3, interaction p=0.30; female/urban/phone similar). Wealth quintiles 4-5 gain more (+15, +11.5 pp) than 1-3 (~0), not formally tested. Only ~150 people gave "no time" as reason (+18 pp, p=0.022); other baseline reasons too rare to analyse.
- **Not obtainable:** the Catalan CORA datasets (M-TICS SMS reminder RCT, text vs letter breast invitation) sit behind a bot check ("Making sure you're not a bot"); needs a manual browser download. No open data for H1 (timed appointment), only published results.
- Illustration (my arithmetic): a 17% relative lift on a 4.5% baseline is +0.8 pp, needing roughly 12,000 women per arm for 80% power at alpha 0.05, so relative effects from other settings give small absolute effects in low-response segments.
