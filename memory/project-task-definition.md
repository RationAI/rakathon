---
name: project-task-definition
description: Rakathon 2026 goal - "invitation predictor" for VZP mammography screening; one row = one invitation, predict 3 labels; team and agents
metadata:
  type: project
---

Rakathon 2026 hackathon, team of 4 from MUNI: Matěj Pekár, Jakub Pekár, Adam Kukučka, Adam Džadoň (git user `ejdam87`). Several Claude agents work on this repo in parallel; this memory merges them:
- Adam Džadoň's agent: EDA, data split, baselines (GB vs LR).
- Adam Kukučka's agent: feature engineering, spatial features, history features, feature importance, confusion matrices.

Task 1: build an **invitation predictor**. Given one invitation (one row of `data/Rakhaton2026.csv`), predict three labels: `SCREENING_3M_FLAG`, `SCREENING_6M_FLAG`, `SCREENING_12M_FLAG` = was the woman screened (mammography) within 90 / 180 / 365 days after the invitation date T0.
Tasks 2-4 (segmentation, interventions, experiment design) are not started yet.

**Why:** VZP (Czech health insurer) sends addressed invitations to mammography screening and wants to know who will respond.
**How to apply:** treat each row as a prediction unit; features may only use information available at T0. See [[dataset-facts]], [[split-decision]], [[baseline-results]], [[open-items]].
