# Agent Brief

This file is the stable operating prompt for Codex work on the ROGII
Wellbore Geology Prediction Kaggle competition.

Use this brief when starting a new context window or asking Codex to run a
leaderboard-improvement experiment. Keep this file stable. Put changing scores,
current defaults, pending submissions, and tactical priorities in
`CURRENT_STRATEGY.md` if present, and append experiment history to
`PATCH_NOTES.md`.

## Objective

Maximize real Kaggle leaderboard performance for ROGII Wellbore Geology
Prediction.

Optimize for hidden-test generalization and valid Kaggle notebook execution,
not merely local OOF improvement. Be skeptical of small validation gains,
especially when they come from narrow samples, suspicious NaN behavior, or
features that may encode well identity.

## Startup Checklist

At the start of a leaderboard-improvement session, read:

- `README.md`
- `CURRENT_STRATEGY.md`, if it exists
- the latest dated section of `PATCH_NOTES.md`
- the relevant code paths in `src/rogii_baseline/`
- the relevant runner scripts in `scripts/`

Then summarize the current state before changing code:

- best confirmed Kaggle LB score
- current recommended/default submission model
- strongest local OOF result
- pending Kaggle submissions
- known failed experiments that should not be repeated
- the highest-risk source of validation mismatch

## Experiment Loop

Run one leaderboard-plausible experiment per context window unless the user
explicitly asks for a broader sweep.

Before coding, write the hypothesis:

- what signal is being added, removed, or corrected
- why it should generalize to hidden test wells
- what failure mode it targets
- what would count as success
- what would count as rejection

Prefer experiments that add independent signal, reduce catastrophic well
failures, improve blend robustness, or reproduce missing strong public-kernel
components. Avoid broad refactors and avoid tuning noise.

Implement the smallest code change needed to test the hypothesis.

## Validation Policy

Always compare against meaningful controls, not just the previous edited model.
At minimum, compare against:

- `selector_pf`
- the current best formation/PF model
- any model being replaced or blended

Validation should be grouped by well. Inspect more than mean RMSE:

- weighted RMSE
- per-well RMSE
- p90/p99 error when available
- worst wells
- NaN and missing prediction counts
- whether improvements are concentrated in a suspicious subset

Treat OOF as evidence, not truth. `PATCH_NOTES.md` already shows that OOF and
public LB can disagree materially.

## Submission Readiness

Only recommend a Kaggle submission when the candidate has a clear reason to
generalize and beats relevant controls by more than likely validation noise, or
when the experiment is strategically worth a limited LB probe.

Before submission, check:

- submission has exactly the expected rows and columns
- `id` order matches `sample_submission.csv`
- no missing or non-finite `tvt` predictions
- code can run in Kaggle notebook mode with internet disabled
- runtime fits the competition limit
- the model default used by the notebook is intentional

Record exact commands and kernel/submission versions in `PATCH_NOTES.md`.

## Known Context To Preserve

- Metric: RMSE on predicted `tvt`.
- Submission columns: `id,tvt`.
- Visible test wells are copied from training for authoring and format checks;
  visible-test target copying is not a valid modeling strategy.
- Kaggle submissions run through notebooks with internet disabled.
- `selector_pf` is the safe control baseline.
- Formation-imputed physical prediction has been the strongest discovered
  signal so far.
- OOF can be misleading; previous local improvements have worsened LB.
- Naive per-row meta learners can overfit well identity and absolute
  coordinates.
- Loose NCC blends have failed; conservative NCC has added little so far.

## High-EV Directions

Prefer these directions unless newer strategy notes say otherwise:

- add independent public-kernel-style signals that are currently missing
- improve robust blending between PF and formation physical predictions
- diagnose and reduce catastrophic worst-well failures
- build better grouped OOF artifacts for signal-level analysis
- use constrained stacking only when input signals are independently useful

Be careful with:

- absolute `X`, `Y`, `MD`, or row-position features that may memorize wells
- tiny OOF improvements from small or alphabetically sampled validation sets
- NaN-handling artifacts
- blend gates that overfit one validation slice
- compute-heavy ideas that cannot fit the Kaggle runtime budget

## Memory Updates

After each experiment:

- append `PATCH_NOTES.md` with commands, metrics, changed files, outcome, and
  lessons
- update `CURRENT_STRATEGY.md` if it exists, especially current best model,
  next experiments, pending submissions, and do-not-repeat items
- leave `AGENT_BRIEF.md` unchanged unless the operating process itself needs to
  change

## Final Report

End each session with:

- experiment tried
- hypothesis result: accepted, rejected, or inconclusive
- files changed
- validation results
- whether to submit
- next best experiment

Do not optimize for looking busy. Optimize for one leaderboard-plausible
experiment with a clear rejection criterion.
