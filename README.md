# Kaggle ROGII Wellbore 2026

This repo contains my working pipeline for the Kaggle ROGII Wellbore Geology Prediction competition.

The task is to predict `tvt` values for hidden well sections. The repository focuses on reproducible local checks, validation tooling, and notebook-ready submission code. Kaggle data is not tracked in Git.

Official competition page:
<https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction>

## What is included

- Data discovery and audit helpers
- Baseline and particle-filter style prediction code
- Out-of-fold validation and diagnostics
- Submission builders for local and Kaggle notebook use
- Empty `data/`, `models/`, `reports/`, and `submissions/` folders with `.gitkeep`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src
export ROGII_DATA_DIR=./rogii-wellbore-geology-prediction
```

Place the downloaded Kaggle files under `ROGII_DATA_DIR`. The folder should contain `train/`, `test/`, and `sample_submission.csv`.

## Common commands

Audit the local data:

```bash
python scripts/audit_data.py
```

Create a geometry smoke-test submission:

```bash
python scripts/make_submission.py --model-kind geometry --output submissions/geometry_submission.csv
```

Create the current selector particle-filter submission:

```bash
python scripts/make_submission.py --model-kind selector_pf --output submission.csv
```

Run masked train-time validation:

```bash
python scripts/run_oof_validation.py --model-kind selector_pf
```

Build a Kaggle notebook export:

```bash
python scripts/build_kaggle_kernel.py
```

## Validation approach

The validation code masks known training wells to imitate the prediction problem, then reports RMSE by fold and scenario. Diagnostics can plot worst wells so failures can be inspected by well geometry and gamma-ray alignment.

## Notes

The public repo intentionally excludes competition data, generated submissions, and local model artifacts. Recreate them with the scripts after downloading the Kaggle dataset.
