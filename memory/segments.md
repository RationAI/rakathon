---
name: segments
description: 8 explainable rule-based segments (A-H) of invitations with response rates and suggested interventions (task 2)
metadata:
  type: project
---

`segmentation/segments.py` `assign(df)` (prepared data) gives each invitation one segment by rules on T0 information only (first match wins):
A not due yet (screened <24 months ago) 2.7% of invitations, 12M 2%; B first invitation 16%, 53%; C lapsed screener + gynaecologist prevention 5%, 41%; D lapsed screener, no gynaecologist 7%, 23%; E never screened (5y) + gynaecologist prevention 18%, 21%; F never screened, other screening only 13%, 9%; G never screened, healthcare contact only 33%, 6%; H disconnected (no prevention, no contact) 5%, 2%.

- Stable across 2021-2024 (2020 slightly lower everywhere). Ranking by segment rate: test ROC AUC 0.79 (best model 0.84).
- Response falls with repeated letters in every segment (E: 41% at 2nd letter -> 9% at 7th).
- Region mix similar; Praha most H (8%), Moravskoslezský most G (36%), fewest B (13%).
- Suggested interventions (hypotheses, need randomized tests): A don't send/postpone; B easy booking; C timely reminder + gynaecologist; D due-date reminder + GP; E gynaecologist recommendation (largest reachable group, best first pilot); F bundle with other screening; G GP/pharmacist nudge; H stop letters, try phone/navigator/mobile unit.

Related: [[model-explanations]], [[open-items]].
