---
name: dataset-facts
description: Key facts and quirks of data/Rakhaton2026.csv (size, encoding, missingness, label rates, sequence resets, odd columns)
metadata:
  type: project
---

Source: `data/Rakhaton2026.csv` (183 MB, UTF-8, gitignored) + spec `data/Rakathon_Datove_rozhrani.docx` (Czech). 1,344,556 invitations, 508,527 women, 25 columns, invitations 2020-01-14 .. 2024-12-06, 60 distinct invitation dates (batches).

**Loading**
- Flags use `A` (ano = yes) / `N` (ne = no). Dates are `dd.mm.yyyy`.
- Load with `dtype={"PERSON_ID": str}`: IDs have leading zeros (15 digits). Use `keep_default_na=False, na_values=[""]`.

**Labels**
- Strictly nested (3M ⇒ 6M ⇒ 12M); positive rates 7.1% / 12.5% / 19.3%. So the OR of all three equals the 12M label.
- 2020 has COVID gaps in sending and lower response (17.8% vs ~19.5-20% in 2021-2024).

**Missing values** are all structural: `MONTHS_LAST_INVITATION` NaN ⇔ first invitation; `MONTHS_SINCE_LAST_SCREEN` NaN ⇔ `NEVER_SCREENED_BEFORE_FLAG`=A; registration duration bands NaN ⇔ not registered.

**Who gets invited / responds**
- Women appear up to 5 times, re-invited about every 12 months; 86% of rows belong to repeat women.
- Invitations target lapsed women: previously screened women are only re-invited after ≥30 months. Women screened 0-3 months before T0 respond ~2% (do not target them).
- First invitation (`FIRST_INVITATION`=A) responds 50.6%, repeat 12.9%. By `INVITATION_SEQUENCE`: 46% at 1, ~3% at 11.
- Age band 40-44 = 100% first invitations, 57% response (very different cohort); 45-49 = 31%.

**Invitation counters**
- `PRIOR_INVITATION_COUNT` ≥ `INVITATION_SEQUENCE` always (equal in 88% of rows, r=0.96). The count is the lifetime total; the sequence is the "current run" of reminders and restarts at 1 after a long break.
- 6,626 sequence resets (sequence ≤ the woman's previous sequence), 6,210 of them in 2024, after ~3.7-year gaps (avg 1,354 days vs 364). Reset rows respond 29% vs 19%. Not the same as `FIRST_INVITATION`. Seen another way: 1,839 women have two rows with the same (PERSON_ID, INVITATION_SEQUENCE), 3-4 years apart.
- 52 women have their first row with `FIRST_INVITATION`=A but `INVITATION_SEQUENCE`=2 (51 in 2021, 1 in 2024); an earlier invitation is probably missing from the data.

**Screening history**
- `PRIOR_SCREEN_COUNT_5Y` is mostly 2. This is CORRECT: women screen every 2 years, so 2 screenings in 5 years is normal (confirmed by Adam Kukučka). An earlier claim-code hypothesis in `exploration/FINDINGS.md` is wrong.
- 3,354 rows have NEVER_SCREENED=N but count 0 (60-month window edge); 193 rows have `MONTHS_SINCE_LAST_SCREEN` > 60.

Full analysis: `exploration/eda.ipynb`, summary `exploration/FINDINGS.md`. Related: [[project-task-definition]], [[feature-engineering]], [[split-decision]].
