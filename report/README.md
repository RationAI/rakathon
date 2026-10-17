# From invitation to attendance

Open `index.html` in a browser. It works offline with the adjacent `styles.css` and `report.js`; no installation, account, server or external assets are required. A single-file copy is supplied as `mammography-report.html` for sharing.

The report includes seven proposed Czech A/B tests, six system recommendations, national comparisons, 28 annotated sources, downloadable test briefs, a sample-size planner and print styles. In Print / save PDF, all seven plans and all references are included even when a different tab or bibliography filter is active.

This is a focused narrative evidence review, not a registered systematic review. Study findings, observational barriers and our proposed experiments are explicitly distinguished. No private patient records or model predictions are used. References link to original publications or official datasets; some publishers may restrict access.

## Public Czech coverage calculation

Source: ÚZIS/NZIP, https://www.nzip.cz/data/2068-mamografie-screening-pokryti-populace-otevrena-data (reviewed 17 October 2026).

The report's 59.9% is our aggregation of the public CSV, not a published invitation-response statistic. Select `rok = 2024` and the five age categories 45–49, 50–54, 55–59, 60–64 and 65–69. Sum district rows: `pocet_vysetrenych = 1,111,008`; `populace = 1,856,010`. Divide summed counts, rather than averaging district percentages: 59.86%, rounded to 59.9%. Use the dataset's two-year coverage definition. Eligibility for the Czech programme begins at 45 and extends beyond the 45–69 subgroup used here.

## Calculator

Two independent equal-sized arms, binary outcome, two-sided alpha 0.05, power 80% or 90%. Optional equal-weight Bonferroni adjustment for a prespecified family. Normal approximation; no continuity correction, cluster inflation or missing-data inflation. The displayed sample is for one pair of arms, not the sum of all planned studies. At 20% versus 23%, 80% power and one comparison, the estimate is 2,943 per arm. Calculator assumptions and references are also visible in the page.

The calculator does not apply to centre-randomized studies or automatically to the record-correction experiment, which needs its own control event rate. Trial designs need programme-specific statistical, clinical, ethical and data-governance review before implementation.
