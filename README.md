# ROGII Wellbore Geology Prediction

Private working repo for the Kaggle ROGII Wellbore Geology Prediction competition.

The GitHub repo intentionally does not track Kaggle data. Keep the downloaded
competition files in `rogii-wellbore-geology-prediction/`, or set
`ROGII_DATA_DIR` to another directory containing `train/`, `test/`, and
`sample_submission.csv`.

Official competition page:
<https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction>

## Competition Notes

- Metric: RMSE on predicted `tvt`.
- Submission file: `submission.csv` with columns `id,tvt`.
- Submission mode: Kaggle Notebooks only, internet disabled, <= 9 hours runtime.
- Final deadline: August 5, 2026 at 11:59 PM UTC.
- Entry/team merger deadline: July 29, 2026 at 11:59 PM UTC.

The visible `test/` examples are copied from training data for authoring and
format checks. Hidden scoring replaces them with the real test wells, so do not
treat visible-test target copying as a real modeling strategy.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
export ROGII_DATA_DIR=./rogii-wellbore-geology-prediction
```

If the local data is missing and Kaggle credentials are configured:

```bash
kaggle competitions download -c rogii-wellbore-geology-prediction
unzip rogii-wellbore-geology-prediction.zip -d rogii-wellbore-geology-prediction
```

## Commands

Audit local data:

```bash
python scripts/audit_data.py
```

Create a geometry-only submission smoke test:

```bash
python scripts/make_submission.py --model-kind geometry --output submissions/geometry_submission.csv
```

Create the current recommended submission using the public-kernel-style
selector PF: likelihood-weighted PF variants, a small per-well beam/hold
selector, and physical/contact targets only for visible train-overlap smoke
wells.

```bash
python scripts/make_submission.py --model-kind selector_pf --output submission.csv
```

Experimental NCC blend candidate:

```bash
python scripts/make_submission.py --model-kind selector_pf_ncc --output submissions/selector_pf_ncc_submission.csv
```

Create a submission using the earlier refactored public-Code idea: a
likelihood-weighted particle-filter GR tracker, plus the visible-overlap
physical/contact blend when the public test wells are also present in `train/`.

```bash
python scripts/make_submission.py --model-kind physical_pf --output submission.csv
```

The previous sklearn residual model is still available for comparison. The
current residual setup trains on artificial prediction-start masks around the
observed competition range, which has generalized better than training only on
the visible train `TVT_input` masks.

```bash
python scripts/make_submission.py --model-kind residual --output submissions/residual_submission.csv
```

Run a masked train-time validation:

```bash
python scripts/validate_baseline.py --model-kind residual --max-train-rows 800000
python scripts/validate_baseline.py --model-kind pf --max-val-wells 20 --pf-seeds 8 --pf-particles 256 --beam-blend-weight 1.0
```

Create row-level out-of-fold validation artifacts for tuning and failure
analysis:

```bash
python scripts/run_oof_validation.py \
  --model-kind pf \
  --mask-strategy actual \
  --folds 5 \
  --pf-seeds 8 \
  --pf-particles 256 \
  --beam-blend-weight 1.0 \
  --output-dir reports/oof/pf_actual_smoke
```

The OOF runner writes `oof_predictions.csv`, `well_metrics.csv`, `folds.csv`,
`well_profiles.csv`, and `summary.json`. Plot the worst wells from a run:

```bash
python scripts/plot_oof_worst_wells.py \
  --oof-dir reports/oof/pf_actual_smoke \
  --top-n 10
```

For Kaggle, use `notebooks/rogii_baseline_submission.py` as a notebook script
and make sure the repo code is available to the notebook. It writes
`/kaggle/working/submission.csv`.
