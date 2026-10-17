---
name: presentation
description: Czech jury presentation (17-slide deck artifact + PNG assets) answering the VZP brief's tasks 01 KDO, 02 PROČ, 03 CO, 04 JAK OVĚŘIT
metadata:
  type: project
---

Brief: `data/Rakathon.pdf` (VZP, Czech): 01 KDO predict risk, 02 PROČ factors + segments, 03 CO interventions per segment, 04 JAK OVĚŘIT experiment / A-B test.

Deck: claude.ai artifact https://claude.ai/artifact/3ezqTDcMvVZ8javSX7PXS3 (private until shared), source copied to `presentation/deck/project/`. Czech, Public Sans, red accent #C62D1F on #FBFAF8, no VZP logo/branding. Placeholder `[název týmu]` (cover + closing slide) still to fill.

Assets: `presentation/make_assets.py` (run from `presentation/`) -> 10 PNGs in `assets/` + `cisla.json` with every quoted number. Uses `segmentation/tree_diagram.draw_segment_tree(texts=...)` for the Czech tree (English default unchanged).

Experiment (04) proposal: randomized test within a segment (start with E, gynaecologist recommendation), randomize by woman or by practice for doctor interventions, outcome = screening within 6/12 months. Sample size for +2 p.p. (alpha 5 %, power 80 %) per arm: C 9 570 (only ~6 400/yr available), D 7 225, E 6 748 (25 524/yr), F 3 478, G 2 493, H 1 208.

Slide `kdo-sekvence` (7, added 2026-10-17): GRU over each woman's invitation sequence vs GB by position (`assets/sekvence.png`, from `training/classifiers/rnn/results.csv`); later footers renumbered 8-16. The live artifact is the source of truth; `presentation/deck/project/` is a copy.

Related: [[segments]], [[model-ceiling]], [[model-explanations]].
