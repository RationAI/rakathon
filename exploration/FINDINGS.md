# Rakathon 2026: EDA findings

Dataset: `data/Rakhaton2026.csv` (mammography screening invitations, 2020–2024). Full analysis with charts: [`eda.ipynb`](eda.ipynb).

**Data quality**
- 1,344,556 invitations for 508,527 women. There are no duplicates, and every NaN is *structural*: each one is fully explained by a flag (`FIRST_INVITATION`, `NEVER_SCREENED_BEFORE_FLAG`, `GP_REGISTERED`, `GYN_REGISTERED_FLAG`). Impute with a sentinel or a "missing" category, not with a mean.
- Small inconsistencies:
  - 3,354 rows have `NEVER_SCREENED = N` but `PRIOR_SCREEN_COUNT_5Y = 0`. This is a 60-month window-edge effect.
  - 193 rows have `MONTHS_SINCE_LAST_SCREEN > 60`, although the documented window is 60 months.
  - 52 rows have `FIRST_INVITATION = A` but `INVITATION_SEQUENCE = 2`.
- `PRIOR_SCREEN_COUNT_5Y` is mostly **2** (196k rows) rather than 1 (8k). It probably counts claim codes, not visits (2 codes per mammography visit?). Worth asking the organisers.

**Targets**
- 3M / 6M / 12M positive rates are 7.1% / 12.5% / 19.3%. The labels are strictly nested, and about 35% of 12-month responders come after month 6.
- Rates are stable at about 19–20% for 2021–2024. 2020 is lower (17.8%) and has COVID gaps in invitation sending (spring and autumn 2020).

**Population structure (important for modelling)**
- **Invitations target non-attenders.** Previously screened women are only invited once they have lapsed (≥30 months since the last screen). A separate cluster of about 36k rows was screened 0–3 months before T0, and these rarely respond (~2%).
- Repeat invitations come about every 12 months. **Response falls sharply with `INVITATION_SEQUENCE`**, from 46% on the 1st invitation to about 3% on the 11th: chronic non-attenders get re-invited again and again.
- **The 40–44 band (100% first invitations) responds at 57%**, and 45–49 at 31%. These are first-time invitees, a very different group from the 50+ re-invited population. Consider a separate model or interaction terms.
- 86% of rows belong to women with more than one invitation, and consecutive invitations are about 12 months apart, so **12M label windows overlap with the next invitation**. Use out-of-time splits with a gap (or at least splits grouped by `PERSON_ID`).

**Predictors**
- The strongest signals are invitation history (`PRIOR_INVITATION_COUNT`, `INVITATION_SEQUENCE`), `MONTHS_SINCE_LAST_SCREEN`, and preventive behaviour: `GYN_PREVENTION_24M_CNT` (10% response at 0 visits vs. 42% at 2+), `OTHER_SCREENING_36M_CNT`, `HEALTHCARE_CONTACT_12M_CNT` (4% response with zero contacts).
- Not being registered with a gynaecologist (7%) or a GP (9%) strongly predicts non-response.
- Regional spread across the 75 VZP areas is 12.8%–23.8%: modest but real.
- Insurance duration and GP prevention counts are weak.

**Baseline**
- An untuned HistGradientBoosting model, trained on 2020–2021 and tested on 2023–2024, reaches **ROC AUC ≈ 0.83** for all three horizons. PR AUC is 0.28 / 0.43 / 0.58 against base rates of 0.07 / 0.13 / 0.20.

**Next steps**
- Sequence features built from earlier rows of the same woman: previous response, number of past non-responses.
- Handle the 40–44 cohort and first invitations explicitly.
- Calibrate predictions and evaluate them per invitation batch.
- Clarify the claim-code semantics of `PRIOR_SCREEN_COUNT_5Y` with the organisers.
