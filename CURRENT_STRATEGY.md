# Current Strategy

Last updated: 2026-06-04.

## Current State

- Best confirmed Kaggle LB: `formation_pf` v11 = **9.817**.
- Safe control baseline: `selector_pf` v10 = **11.262**.
- Current recommended local submission model: `formation_pf` with the v13
  confidence gate plus the nan-aware per-row formation combiner added on
  2026-06-04.
- Strongest saved 100-well OOF: `formation_pf_v12_100w` = **9.520**, but
  that version scored worse on LB (**10.375**) and should not be trusted by
  OOF alone.
- Latest validated current-code 100-well OOF:
  `reports/oof/formation_pf_nanaware_100w` = **9.597** weighted RMSE.
- Pending Kaggle submissions as of `kaggle competitions submissions`
  checked on 2026-06-04:
  - ref `53344867`: `formation_pf v13`, pending.
  - ref `53343945`: `formation_pf v13`, pending.

## Submission Guidance

- Do not submit the nan-aware patch immediately while v13 is still pending.
- If v13 beats or roughly matches v11 on LB, package this patch as a limited
  `formation_pf` v14 probe.
- If v13 is worse than v11, keep v11 as the best confirmed model and treat
  further formation gate tweaks as suspect until validated on a broader or more
  representative split.
- Kaggle kernel builder default is `formation_pf`; the checked-in notebook
  fallback is also `formation_pf`.

## Do Not Repeat

- Do not restore the dense-std confidence factor without a new reason; it
  improved OOF and hurt LB.
- Do not repeat the HGB per-row meta learner as previously configured; grouped
  OOF was worse than simple blends.
- Do not use loose NCC blending or switch the PF base to `selector_pf_ncc`;
  it added no meaningful signal.
- Do not extrapolate `b_well` over MD as a default; it helped drift cases but
  amplified stable-well errors.
- Be suspicious of NaN-driven OOF improvements. Count missing predictions and
  compute metrics without dropping model failures silently.

## Next Best Experiments

- Wait for v13 LB, then decide whether the nan-aware v14 probe is worth one
  submission.
- Add an independent public-kernel-style signal rather than another formation
  gate tweak: beam-search variants, GR rolling features, or constrained
  positive Ridge over existing independent signals.
- Build a stricter signal-analysis artifact that includes beam/NCC/formation
  components with per-well grouped OOF metrics and tail-error diagnostics.

