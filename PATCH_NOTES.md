# ROGII Patch Notes

## 2026-06-04 - Nan-Aware Formation Row Combiner

### Hypothesis
- The formation physical predictor was falling back to PF on rows where the
  globally selected formation set contained any per-row NaN, even when other
  formation surfaces were still valid for that row.
- A row-wise NaN-aware weighted average should recover real formation signal
  on those rows without changing the confidence gate or adding a new overfit
  feature.
- Success: improve current `formation_pf` against `selector_pf` and the
  current v13-style formation/PF control, with no missing predictions and no
  tail-error regression large enough to justify rejection.
- Rejection: only NaN-accounting artifacts, worse p90/p99/worst wells, or an
  improvement too tiny/concentrated to be leaderboard-plausible.

### What Changed
- `src/rogii_baseline/formations.py`
  - Changed the final per-formation combination in `predict_physical_tvt` to
    ignore non-finite per-row formation predictions and renormalize the
    remaining formation weights per row.
  - Rows with no finite formation predictions still produce NaN physical TVT,
    so `_formation_pf_predict` falls back to PF as before.
- `notebooks/rogii_baseline_submission.py`
  - Changed the notebook fallback model from `selector_pf` to `formation_pf`.
    The self-contained Kaggle kernel builder already defaulted to
    `formation_pf`.

### Validation
Commands:
```bash
python -m compileall src scripts

python scripts/run_oof_validation.py \
  --model-kind formation_pf \
  --mask-strategy actual \
  --max-wells 100 \
  --folds 5 \
  --pf-seeds 16 \
  --pf-particles 384 \
  --output-dir reports/oof/formation_pf_nanaware_100w

python scripts/make_submission.py \
  --model-kind formation_pf \
  --pf-seeds 16 \
  --pf-particles 384 \
  --output submissions/formation_pf_nanaware_submission.csv

python scripts/build_kaggle_kernel.py \
  --output-dir /private/tmp/rogii_kaggle_kernel_nanaware \
  --model-kind formation_pf

kaggle competitions submissions rogii-wellbore-geology-prediction | head -30
```

Exact 200-well artifact recompute from `reports/meta/train_200w.parquet`:
- `selector_pf`: weighted RMSE **12.26481**, p90 **19.90886**,
  p99 **36.28219**, worst well `1b1eba53` **44.13024**.
- pre-patch v13-style blend: weighted RMSE **10.21020**, p90 **15.83987**,
  p99 **26.20931**, worst well `1b1eba53` **38.96916**.
- nan-aware v13-style blend: weighted RMSE **10.08897**, p90 **15.83987**,
  p99 **26.20931**, worst well `1b1eba53` **38.96916**.
- Per-well deltas on the 200-well artifact: 6 wells improved, 1 worsened,
  193 unchanged. Biggest improvements were wells whose physical formation rows
  were mostly or entirely NaN before the patch (`2fe023be`, `261785ee`,
  `3e011332`, `1590af81`).

Exact first-100 artifact recompute:
- `selector_pf`: weighted RMSE **11.729875**.
- pre-patch v13-style blend: weighted RMSE **9.603457**.
- nan-aware v13-style blend: weighted RMSE **9.596567**.
- Changed rows were concentrated in one well (`1590af81` improved from
  **6.76645** to **5.34737** RMSE); the other 99 wells were effectively
  unchanged.

Full 100-well production OOF after patch:
- `selector_pf_100w`: weighted RMSE **11.72987**, p90 **18.62720**,
  p99 **28.80370**, worst **44.13024**.
- `formation_pf_100w` v11-style saved run: weighted RMSE **10.18103**,
  p90 **15.19976**, p99 **22.53967**, worst **43.32837**.
- `formation_pf_v12_100w` saved run: weighted RMSE **9.52026**,
  p90 **13.20736**, p99 **22.58881**, worst **37.19161**; this version later
  scored **10.375** LB, so its OOF advantage is suspect.
- `formation_pf_nanaware_100w`: weighted RMSE **9.59657**, mean well RMSE
  **7.92363**, median well RMSE **6.76989**, p90 **14.33595**,
  p99 **21.95054**, worst well `1b1eba53` **38.96916**.
- OOF artifact sanity: 493637 rows, no NaN predictions, no NaN actuals, no
  duplicate `(well_id, row_index)` predictions.

### Submission Readiness
- Local submission file:
  `submissions/formation_pf_nanaware_submission.csv`.
- Rows/columns: `14151` rows, columns exactly `id,tvt`.
- `id` order matches `sample_submission.csv`.
- Missing/non-finite `tvt`: **0**.
- Local visible-test generation with `pf_seeds=16`, `pf_particles=384` took
  about 12 seconds; the patch itself adds negligible runtime.
- Self-contained Kaggle kernel build succeeded at
  `/private/tmp/rogii_kaggle_kernel_nanaware`.
- Kernel metadata has `"enable_internet": false` and default
  `ROGII_MODEL_KIND='formation_pf'`.
- Recent Kaggle submissions checked on 2026-06-04:
  - v13 refs `53344867` and `53343945` are still pending.
  - v12 completed at **10.375**.
  - v11 remains best confirmed at **9.817**.

### Outcome
- Accepted as a small, low-risk production fix for the current `formation_pf`
  path.
- Not recommended for immediate Kaggle submission while v13 is still pending.
  The OOF gain over current v13-style behavior is real but concentrated in a
  small number of wells, and prior formation OOF gains have not reliably
  translated to LB.
- If v13 comes back strong, submit this as a limited v14 probe. If v13 is bad,
  prioritize independent public-kernel-style signals over more formation gate
  tuning.

## 2026-06-04 - LB Results & Failed Improvement Attempts

### Confirmed LB Scores
- `selector_pf` v10 (baseline): **11.262**
- `formation_pf` v11 (loose blend, no dens_std): **9.817** ← previous best
- `formation_pf` v12 (100w-tuned, tight blend, dens_std factor): **10.375**
- `formation_pf` v13 (200w-tuned, looser cap=0.95): **pending**

### Key Lesson: OOF Is Not LB
- v11 OOF on 100w = 10.18, LB = 9.82 (OOF pessimistic by 0.36)
- v12 OOF on 100w = 9.52, LB = 10.37 (OOF optimistic by 0.85)
- v11 OOF on the same 200w set (after rebuild) = 10.42
- v13 OOF on 200w = 9.98 → expected LB ~9.4 if delta tracks
- The 100-well alphabetical sample was much easier than LB hidden test.
  200 wells is closer but still has scoring variance vs LB.

### Best Heuristic Found (v13, deployed)
```python
quality_factor = exp(-max(quality_known - 3.0, 0) / 32.0)
plane_factor   = exp(-plane_distance / 0.05)
confidence     = clip(quality_factor * plane_factor, 0, 0.95)
prediction     = confidence * formation + (1 - confidence) * selector_pf
```
- 200-well OOF weighted_rmse = 9.98 (best in grid-search sweep)
- Other variants tried:
  - `dense_std` factor: 0.03 ft improvement, removed because it
    over-suppressed formation on LB (v12 had it, scored worse than v11
    which didn't)
  - Higher cap (1.0): identical OOF to cap=0.95 — the cap rarely binds
  - Sigmoid gate: 9.85 OOF (slightly better) — saved for v14 if needed
  - Piecewise gate: 10.20 OOF (worse)

### Meta-Learner Attempt — Failed
- Built training set: `scripts/build_meta_training_set.py` runs the PF
  and formation pipelines with `exclude_well_id` on each train well, dumps
  per-row signals + features to parquet. 200 wells, 950k rows, 38 cols.
- Trained HistGradientBoostingRegressor (`scripts/train_meta_learner.py`)
  with GroupKFold CV. Targets tested:
  - `actual_tvt - last_tvt`: OOF rmse 17.80
  - `actual_tvt - pf_pred`: OOF rmse 13.31
  - `actual_tvt - form_pred`: OOF rmse 27.04
- All WORSE than the simple heuristic (9.98). HGB overfit per-well
  patterns (X, Y, MD) that don't generalize across wells.
- Removed absolute coordinate features; tried lighter HGB
  (max_iter=150, min_samples_leaf=800, l2=2.0); tried predicting per-row
  optimal blend weight: nothing closed the gap.
- Per-row oracle blend (w_opt = (actual - pf) / (form - pf), clipped) on
  the 200w training set = **6.01** — so there is 4 ft of headroom for a
  meta-learner that actually works.

### Other Investigations
- **PF self-bias correction** (held out last 25% of known prefix, ran PF
  on first 75%, measured bias, subtracted from hidden predictions):
  marginal **+0.08 ft** improvement on 30-well sample, **doubles PF
  compute** — not worth it.
- **NCC blend** (use `selector_pf_ncc` instead of `selector_pf` as the
  blend base): no change. NCC contribution is already at most 12% inside
  `_selector_pf_ncc_predict` and adds no signal beyond PF.
- **Sharper formation weighting** (1/qk^3 vs 1/(qk+0.1)): appeared to
  improve OOF from 9.98 to 9.55 — was actually a NaN-handling artifact
  (the sharper weights produced NaN sums on rows with missing per-formation
  data, which were silently dropped from the RMSE calculation).
  NaN-aware computation gives 9.98 — identical to baseline.
- **Iterative refinement** (use v13 prediction as pseudo-known TVT to
  refit b_well, predict again): inconsistent — helps some wells (-3.78
  ft), hurts others (+8.66 ft).
- **Trend extrapolation of b_well over MD** (fit linear b_well = a*MD + b
  on known prefix, extrapolate into hidden): same — amplified errors
  where b_well was stable, only modestly helped drift cases.

### Per-Row Signal Strength (correlation with abs_err on 100w)
- `plane_distance`: **+0.671** (best per-row signal)
- `dense_std`:      +0.550
- `dense_distance`: +0.528
- `qk` (per-well):  +0.345

### What's Likely Needed for Sub-8
Top public kernels (`romantamrazov/rogii-super-solution`, public LB ~9.25)
stack ~15 signals:
- 2 PF variants (one anchored to ANCC formation, one to Z) using
  Numba-compiled inner loops
- 7 beam search configs with different (beam_size, move_cost,
  error_scale, smooth_radius)
- 3 NCC scales (half-windows 8, 15, 25)
- 6 per-formation TVT predictions + WLS b_well + known-zone RMSE per
  formation
- Dense ANCC imputer (we have this) plus FormationPlaneKNN (we have this)
- GR rolling features (mean/std at windows 5, 21, 51, 101; lags ±1, ±5,
  ±15, ±30; envelope; energy)
- LightGBM + CatBoost stacked through a positive-coefficient Ridge

The LB top (6.534) is private — no public technique is known.

## 2026-06-04 - Formation-Imputed Physical Prediction (formation_pf)

### Insight
Top public kernels (`romantamrazov/rogii-super-solution-lb-top-3`,
`mitchgansemer/drift-targeting-ncc-tree-based-rogii-wellbore`) all use the
six formation marker columns (`ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA`)
that exist only in train. The geological identity is exact:

    TVT[i] + Z[i] - formation_depth[i] = b_well  (constant per well/formation)

Verified empirically: on multiple train wells the per-row `b_well` std is
~0.007 (rounding noise). The signal disappears at test time because the
formation columns are stripped, but they can be **spatially imputed** from
neighbouring train wells via KDTree over (X, Y).

### What Changed
- New module `src/rogii_baseline/formations.py` with `FormationImputer`
  combining two spatial estimators:
  1. **`FormationPlaneKNN`** — one centroid per train well; weighted
     least-squares plane fit through the K=10 nearest neighbours.
  2. **`DenseANCCImputer`** — downsampled per-row dense point cloud (~60
     points per well, ~46k total); inverse-distance weighting over the
     K=20 nearest. Per-row dense_std flags imputation noise.
- `predict_physical_tvt` computes per-formation TVT = `-Z + form + b_well`
  with WLS b_well calibration (decay=0.02 → recent rows upweighted), and
  combines all six formations weighted by their known-zone RMSE.
- New model kind `formation_pf` in `baseline.py` blends formation
  prediction with `selector_pf` using a three-factor confidence:
  - per-well `exp(-(quality_known - 5) / 32)`
  - per-row `exp(-plane_distance / 0.08)`
  - per-row `exp(-dense_std / 100)`
  - capped at 0.70 to bound catastrophic formation failures.

### Works
- 100-well OOF (folds=5, pf_seeds=16, pf_particles=384):
  - `selector_pf` baseline: weighted_rmse=11.73, p90=18.63, p99=28.80
  - `formation_pf` v11 (first blend): weighted_rmse=10.18, p90=15.20
  - `formation_pf` v12 (re-tuned blend): weighted_rmse=9.52, p90=13.21
- Local make_submission with `--model-kind formation_pf` produces a valid
  14151-row submission CSV.
- Pushed Kaggle kernel v12 with `formation_pf` as default.

### Does Not Work Yet
- Worst-well RMSE still 37 ft (`1b1eba53`): both PF and formation fail.
- Trend extrapolation of `b_well` over MD was attempted and reverted —
  it amplified errors on wells with stable b_well while only modestly
  helping drift cases.
- Oracle per-row blend of (formation, PF) caps at RMSE 5.69; remaining
  gap of 3+ ft requires additional model signals (NCC, beam, residual
  GBT) or a meta-learner.



## 2026-06-03 - Selector PF Control

### Works
- Public-kernel-style `selector_pf` is now the recommended control model.
- Kaggle kernel `josephmontana/rogii-baseline-submission` version `10` was pushed and completed successfully.
- The completed Kaggle run produced `submission.csv`.
- Local submission sanity passed: `14151` rows, sample id order matched, no missing `tvt`.
- Tiny train-time hidden-path smoke validation passed:
  - Command: `python scripts/validate_baseline.py --model-kind selector_pf --mask-strategy actual --max-val-wells 3 --pf-seeds 4 --pf-particles 64`
  - Overall RMSE: `7.190723`

### Does Not Work Yet
- First competition submission attempt was blocked because the team had already used the daily allowance of `5` submissions.
- Retry succeeded after the UTC reset.
- Public score for `selector_pf public-style kernel v10` was `11.262`, which improved on `physical_pf_beam` but did not reproduce the public `8.863` selector baseline.
- Previous submitted scores are poor relative to the public control:
  - `physical_pf_beam self-contained`: `12.158`
  - HGB/residual submissions: `16+` or worse
- Newer formation-PF work visible in Kaggle submissions has improved further:
  - `formation_pf v11`: `9.817`
  - `formation_pf v12`: `10.375`
  - `formation_pf v13`: pending as of the latest check

### Submitted

```bash
kaggle competitions submit rogii-wellbore-geology-prediction \
  -k josephmontana/rogii-baseline-submission \
  -v 10 \
  -f submission.csv \
  -m "selector_pf public-style kernel v10"
```

### Next Patch Direction
- Add target-free GR alignment candidates, starting with multi-scale normalized cross-correlation (NCC).
- Keep `selector_pf` as the safe default until new candidates prove out locally and on Kaggle.

## 2026-06-03 - Experimental NCC Blend

### What Changed
- Added optional model kind `selector_pf_ncc`.
- It blends public-style `selector_pf` with target-free multi-scale GR normalized cross-correlation.
- NCC compares unknown-section GR windows against known-prefix GR windows and maps the best match back to known-prefix TVT.
- The blend is now heavily gated:
  - high NCC score required,
  - NCC must agree with selector-PF within about `20 ft`,
  - maximum NCC contribution is `12%`.

### Works
- Code compiles.
- Local experimental submission is valid:
  - `submissions/selector_pf_ncc_submission.csv`
  - sample id order matches,
  - no missing predictions.
- Conservative gate avoids the catastrophic first NCC attempt.
- 10-well smoke validation with small PF settings was slightly better than selector-only:
  - `selector_pf_ncc`: `8.109707`
  - `selector_pf`: `8.128296`
  - Command settings: `--mask-strategy actual --max-val-wells 10 --pf-seeds 4 --pf-particles 64`

### Does Not Work / Caution
- First direct NCC blend was bad:
  - `selector_pf_ncc` with loose gate: `49.057688` RMSE on 3 wells.
- Conservative NCC was still worse on the first 3-well smoke:
  - `selector_pf_ncc`: `7.280507`
  - `selector_pf`: `7.190723`
- Do not make `selector_pf_ncc` the Kaggle default yet.
- Treat it as an experimental candidate for future blending/selector work, not as the next submission.
