# Rakathon 2026 report

Four self-contained pages sharing one top navigation, plus a standalone process page (`workflow.html`). Open `index.html` in a browser; every page is a single file
(fonts load from Google Fonts and fall back to system fonts offline). Light by default; the theme button switches to dark.

| page | content | built by (from `report/`) |
|---|---|---|
| `exploration.html` | labels, time and COVID, invitation history, response by attribute, correlations | `build_exploration.py` |
| `response_report.html` | regional report, geospatial only: district and region maps, centres, access vs response, population | `build_report.py` |
| `classifiers.html` | predictive models: risk score, the models and their ceiling, drivers, segments | `build_classifiers.py` (~1 min) |
| `index.html` | interventions and tests: segment → barrier → intervention → A/B test, shared protocol, sample-size planner, system changes, 28 sources | `build_interventions.py` |
| `workflow.html` | proposed outreach workflow (Czech, standalone, own styles, no builder): model score → SMS/push/e-mail → AI call → booking and confirmation, with rules and test plan | edited by hand |

Each builder fills `<page>.template.html` with data (`__DATA__`) and inlines the shared stylesheet `site.css`
(except the regional report, which keeps its own CSS with the same tokens). Run with `PYTHONUTF8=1 uv run python build_*.py`.

`evidence.json` holds the evidence review: the 28 annotated sources, the seven mechanism-specific test designs and the six
system recommendations. It is a focused narrative review, not a registered systematic review; study findings, observational
barriers and our proposed experiments are kept apart. The primary outcome of every test is completed mammography within
12 months (our label), read early at 3 and 6 months, except the record-correction test (verified correction within 90 days).

## Public Czech coverage calculation

Source: ÚZIS/NZIP, https://www.nzip.cz/data/2068-mamografie-screening-pokryti-populace-otevrena-data (reviewed 17 October 2026).

The report's 59.9% is our aggregation of the public CSV, not a published invitation-response statistic. Select `rok = 2024` and the five age categories 45–49, 50–54, 55–59, 60–64 and 65–69. Sum district rows: `pocet_vysetrenych = 1,111,008`; `populace = 1,856,010`. Divide summed counts, rather than averaging district percentages: 59.86%, rounded to 59.9%. Use the dataset's two-year coverage definition. Eligibility for the Czech programme begins at 45 and extends beyond the 45–69 subgroup used here.

## Calculator

Two independent equal-sized arms, binary outcome, two-sided alpha 0.05, power 80% or 90%. Optional equal-weight Bonferroni adjustment for a prespecified family. Normal approximation; no continuity correction, cluster inflation or missing-data inflation. The displayed sample is for one pair of arms, not the sum of all planned studies. At 20% versus 23%, 80% power and one comparison, the estimate is 2,943 per arm. Calculator assumptions and references are also visible in the page.

The calculator does not apply to centre-randomized studies or automatically to the record-correction experiment, which needs its own control event rate. Trial designs need programme-specific statistical, clinical, ethical and data-governance review before implementation.
