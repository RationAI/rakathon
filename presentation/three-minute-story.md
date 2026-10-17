# Three-minute jury pitch

## The central story

**VZP can use existing healthcare contacts to offer better support for mammography, then measure whether that support increases attendance.**

Tell a story about a decision VZP can take. The models establish useful targeting. The exploration reveals an opportunity within existing care. The literature helps choose an intervention. A randomized pilot establishes its local effect.

The brief explicitly asks for all four: who, associated factors and segments, support, and evaluation. Its page 9 says prediction alone is insufficient. The story below covers these without spending the three minutes on model comparisons.

## Five slides and a ready-to-speak script

Approximately 320 spoken words. Rehearse with pauses and aim to finish around 2:50. Use one speaker if possible. The timings are budgets, not a reason to rush.

### 1. An opportunity inside existing care — 0:00–0:25

**Visual:** One large number: **80%**. Caption: “Recorded preventive care among mammography non-attenders in our analysis cohort.” Avoid a dashboard screenshot.

**Say:**

“Four out of five women who missed mammography in our analysis had already used some other preventive care. That surprised us. Many women are already in contact with healthcare, yet mammography still does not happen. We set out to help VZP identify where additional support could make a difference.”

**Small source note:** First observed invitation per woman, age 45+, excluding recorded mammography less than 30 months earlier. 306,189 non-attenders within 365 days; 80.2% had at least one measured preventive claim. Analytical restriction, not a clinical eligibility rule.

### 2. Useful prediction, diminishing returns — 0:25–1:00

**Visual:** A simplified model comparison or learning curve. Show **57 experiments** and a compact label **AUC ~0.84 on held-out women**. Keep model names and split details in backup.

**Say:**

“We analyzed 1.34 million invitations and tested 57 model configurations. The strongest approaches converged around 0.84 AUC on women they had never seen. More complexity and engineered features brought only small gains. We also built readable segments from screening history, previous invitations and preventive care. This gives VZP a useful way to prioritize support. But predicting non-attendance still leaves the important question: what would help?”

**Small source note:** The 57-fit ablation uses the random patient-separated split. A separate, stricter test on later invitations from unseen women obtains AUC 0.832 in the full cohort. These are different evaluations.

### 3. The records leave personal barriers unobserved — 1:00–1:30

**Visual:** Recorded information on the left: invitation history, screening, preventive care. Candidate barriers on the right: low perceived need, fear, booking difficulty. Clearly label these “hypotheses from research.”

**Say:**

“The records show previous care. They do not tell us whether a woman feels screening is unnecessary, fears the examination, or struggles to arrange it. So we reviewed Czech and international research. We separated reported barriers from interventions that had actually been tested, and translated that evidence into seven proposed Czech experiments.”

### 4. A concrete next step for the woman — 1:30–2:10

**Visual:** A simple comparison: “Please arrange screening” and “Here is an appointment you can accept or change.” Add a small UK trial comparison, **12% / 22%**, with its context.

**Say:**

“Our leading proposal is to make booking easier. In an English study of previous non-attenders, about 22 percent attended after receiving a specific second appointment, compared with 12 percent when asked to book themselves. That result motivates a Czech pilot. We propose testing a real appointment offer, with an easy way to decline or reschedule, among eligible overdue women served by centres with available capacity.”

### 5. A pilot VZP can evaluate — 2:10–3:00

**Visual:** Two equal groups: current invitation and invitation plus appointment offer. Outcome: **completed mammography**. Below: cost per additional completion. Keep a screenshot of the evidence report small or in backup.

**Say:**

“Randomly assign women to the current invitation or the appointment offer. Measure completed mammography within 90 days, and the cost of each additional completed screening. Record delivery and booking so we can understand where the pathway breaks down. If a pilot achieved a three-percentage-point gain, that would mean 3,000 additional screenings per 100,000 women reached. That is an illustration, not a result we claim today. We have working models, understandable segments and evidence-backed test plans. Our next step with VZP is to run one focused pilot and measure the difference.”

## What makes the argument persuasive

- Lead with the finding that changes the judge's understanding: most observed non-attenders already have some preventive-care record.
- Spend about 35 seconds on models. Their purpose is to establish useful targeting and show that the team tested alternatives.
- Treat the literature as a way to choose the next action, rather than a separate academic project or a list of countries.
- Show one intervention in enough detail that the judge can picture the woman's experience.
- Finish with a measurable next step and conditional scale. Additional completed screenings are the outcome; the project has not estimated cancers detected or lives saved.

## Claims to replace

| Avoid | Use |
|---|---|
| “We cannot get better.” | “Across the approaches we tested, additional gains became small.” |
| “The dataset was too limited.” | “The records support targeting, while personal barriers require additional evidence.” |
| “We identified the causes.” | “We identified associations and used research to develop barrier hypotheses.” |
| “The model has 84% accuracy.” | “The model achieved about 0.84 ROC AUC on held-out women.” |
| “Women ignore repeated letters.” | “Women with repeated invitation histories have lower observed attendance.” |
| “They have never been screened.” | “They have no recorded screening in the available lookback.” |
| “A gynaecologist will solve this.” | “Existing care offers a potential channel for testing support.” |
| “The appointment offer will double Czech attendance.” | “An English study supports testing this approach locally.” |

## Backup answers

**Why stop optimizing?**

We tested capacity, training-set size and feature engineering. Strong models clustered near the same performance, and more training data brought diminishing gains. That supports shifting effort toward actionable information and intervention testing. It does not establish an absolute prediction ceiling.

**How did you validate?**

The 57-fit ablation keeps all invitations from a woman in one partition and selects settings on validation data. Test set: 201,851 invitations. The separate temporal analysis trains on 2020–2021, validates in January–June 2023 and tests in July–December 2024, with disjoint women and outcome-window gaps. Its full-cohort AUC is 0.832 on 17,989 invitations. The restricted analysis-cohort AUC is 0.819 on 16,718 invitations. Do not combine these populations or call the random split a future-period evaluation.

**What do the segments tell us?**

They describe observable differences and potential channels for support. They do not label individual motivations. For a clear backup, show the three mutually exclusive preventive-care profiles in the first-observed analysis cohort: no measured prevention (7.1% 12-month attendance), other prevention without gyn prevention (16.8%), and gyn prevention (49.2%). These are descriptive rates, not intervention effects.

**Why booking support rather than just clinician endorsement?**

Appointment offers have supportive trial evidence. A large US trial found no significant benefit from clinician endorsement alone. Clinicians may help deliver support, but a signature is not an established remedy. The proposed pilot tests the concrete booking step.

**Should the highest-risk women always receive the intervention?**

Predicted non-attendance and benefit from support are different quantities. Risk scores help define and stratify pilot populations. Randomization establishes which groups benefit. Preserve standard care and informed choice for everyone.

**How large is the pilot?**

Determine this using the selected segment's 90-day control rate and the smallest worthwhile absolute gain. The report's example of 20% versus 23% requires about 2,943 women per arm at two-sided alpha 0.05 and 80% power with individual randomization. This is a planning illustration, not an estimate of our proposed segment's 90-day baseline. Practice or centre randomization needs a separate cluster calculation. Do not reuse a sample size based on 12-month attendance for a 90-day endpoint.

**What new information would you collect?**

Delivery status, referral where relevant, booking attempts, offered dates and completed mammography, plus a short optional patient-reported barrier question. Obtain reasons from women across care profiles rather than inferring them from claims. A positive trial identifies an effective intervention; it does not by itself prove each woman's psychological reason.

**What exists today?**

Trained prediction models and inference code, descriptive analyses, readable segmentation and decision trees, and an offline literature report with 28 annotated sources, seven proposed A/B tests and a planning calculator. The repo does not establish a prospectively implemented intervention or a measured Czech attendance gain.

## Evidence and sources

Paths below are relative to the repository root.

| Pitch claim | Repository evidence | Interpretation boundary |
|---|---|---|
| Four-part brief | `Rakathon.pdf`, pp. 6–9 | Includes intervention and experiment, beyond prediction. |
| 1,344,556 invitations, 508,527 women | `exploration/NONATTENDANCE.md` | Invitations 2020–2024; selected VZP invitation cohort, not national screening coverage. |
| 80.2% of non-attenders have preventive care | `exploration/NONATTENDANCE.md`; `exploration/nonattendance_results/primary_prevention.csv` | 306,189 non-attenders in a restricted 436,612-woman first-observed cohort. Claims do not prove access or receipt of a recommendation. |
| 57 fits and diminishing gains | `training/classifiers/ablation/report.md`; `training/classifiers/ablation/results/ablation.csv` | 38 capacity, 16 learning-curve, 3 engineered-feature fits. Best raw-feature boosting 0.8413; engineered-feature boosting 0.8423. |
| Later unseen-woman validation | `exploration/MODEL_FINDINGS.md`; `exploration/barrier_model_results/model_summary.json` | Full-cohort AUC 0.832, restricted-cohort 0.819. Different experiment from ablation. |
| Readable fitted tree | `exploration/decision_tree_results/README.md` | Depth-three restricted-cohort tree: AUC 0.799. Selected deeper tree: 0.815. Predictive branches, not identified causes. |
| 28 sources and 7 proposed tests | `report/README.md`; `report/report.js`; `report/index.html` | Focused narrative evidence review, not a registered systematic review. |
| Fixed appointment ~22% versus ~12% | Allgood et al., 2017: https://pubmed.ncbi.nlm.nih.gov/28522311/ | England, 26,054 previous appointment non-attenders. Different pathway from Czech insurer invitations; no local effect forecast. |
| SMS +5.3 percentage points, optional backup | Kerrison et al., 2015: https://pubmed.ncbi.nlm.nih.gov/25668008/ | First screening appointments; 59.1% versus 64.4%. Different population and booking stage. |
| Clinician endorsement alone not significant | Mehta et al., 2024: https://pmc.ncbi.nlm.nih.gov/articles/PMC11074930/ | US trial, 12.5% versus 11.4%; motivates caution about signature-only interventions. |

Existing assets for backup: `presentation/assets/modely.png`, `presentation/assets/uceni.png`, `exploration/barrier_model_results/model_evidence.png`, `exploration/decision_tree_results/main_depth3.png`. Recheck every chart's population before combining it with a spoken number. The eight hand-built segments and eight leaves of the fitted tree are different artifacts.
