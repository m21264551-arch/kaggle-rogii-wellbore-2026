#!/usr/bin/env python3
"""Train a HistGradientBoostingRegressor meta-learner on OOF training data.

The training set is produced by ``scripts/build_meta_training_set.py``.
Each row has per-row signals (pf_pred, form_pred, per-formation preds,
geom_tvt) and per-well features (qk, plane_distance, dens_std).

We try multiple targets and pick the best:

* ``actual_tvt``: predict absolute TVT directly
* ``actual - last_tvt``: predict the delta from the last known TVT
* ``actual - form_pred``: predict the formation residual
* ``actual - pf_pred``: predict the PF residual

Then we save the best model + its target type as a pickle for the kernel
to consume. The kernel embeds the pickle base64-encoded and applies it at
inference time.
"""

from __future__ import annotations

import argparse
import base64
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold


META_FEATURES = [
    "pf_pred", "form_pred", "geom_tvt",
    "plane_distance", "dense_distance", "dens_std", "qk",
    "last_tvt", "delta_md", "delta_x", "delta_y", "delta_z",
    "dist_xy", "row_fraction", "ps_fraction",
    "gr", "gr_missing", "typewell_gr_diff",
    "geom_slope", "geom_quad_delta", "z", "md",
    # per-formation columns
    "form_ANCC_pred", "form_ANCC_qk",
    "form_ASTNU_pred", "form_ASTNU_qk",
    "form_ASTNL_pred", "form_ASTNL_qk",
    "form_EGFDU_pred", "form_EGFDU_qk",
    "form_EGFDL_pred", "form_EGFDL_qk",
    "form_BUDA_pred", "form_BUDA_qk",
]


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    if not mask.any():
        return float("inf")
    return float(np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2)))


def cv_score(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    hgb_kwargs: dict,
) -> tuple[np.ndarray, list[HistGradientBoostingRegressor]]:
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(X), np.nan)
    models: list[HistGradientBoostingRegressor] = []
    for fold, (tr, vl) in enumerate(gkf.split(X, y, groups)):
        m = HistGradientBoostingRegressor(**hgb_kwargs)
        m.fit(X.iloc[tr], y[tr])
        oof[vl] = m.predict(X.iloc[vl])
        models.append(m)
    return oof, models


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="reports/meta/train_200w.parquet")
    parser.add_argument("--model-output", default="reports/meta/meta_model.pkl")
    parser.add_argument("--metrics-output", default="reports/meta/meta_metrics.json")
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {df['well_id'].nunique()} wells")
    # Drop rows with NaN actuals or signals
    df = df.dropna(subset=["actual_tvt", "pf_pred", "form_pred", "last_tvt"]).reset_index(drop=True)
    print(f"After dropna: {len(df)} rows from {df['well_id'].nunique()} wells")

    available = [c for c in META_FEATURES if c in df.columns]
    print(f"Using {len(available)} features: {available}")

    X = df[available].astype("float64").fillna(0.0)
    actual = df["actual_tvt"].values
    last_tvt = df["last_tvt"].values
    pf = df["pf_pred"].values
    form = df["form_pred"].values
    groups = df["well_id"].values

    hgb_kwargs = dict(
        max_iter=600,
        learning_rate=0.04,
        max_leaf_nodes=63,
        min_samples_leaf=80,
        l2_regularization=0.05,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=42,
    )

    results = {}
    print(f"\nBaselines (no meta-learner):")
    print(f"  pf_pred:   rmse={rmse(actual, pf):.3f}")
    print(f"  form_pred: rmse={rmse(actual, form):.3f}")
    print(f"  0.5*pf + 0.5*form: rmse={rmse(actual, 0.5*pf + 0.5*form):.3f}")

    targets = {
        "actual_minus_last": (actual - last_tvt, last_tvt),
        "actual_minus_pf":   (actual - pf,       pf),
        "actual_minus_form": (actual - form,     form),
    }
    best = None
    for name, (y_target, baseline) in targets.items():
        print(f"\nTraining HGB target={name} ...")
        oof, models = cv_score(X, y_target, groups, hgb_kwargs=hgb_kwargs)
        predicted_actual = baseline + oof
        r = rmse(actual, predicted_actual)
        print(f"  rmse(predicted_actual) = {r:.3f}")
        results[name] = float(r)
        if best is None or r < best[0]:
            best = (r, name, models)

    print(f"\nBest target: {best[1]} (RMSE {best[0]:.3f})")
    # Refit on FULL data for the chosen target
    best_target = best[1]
    y_target, _ = targets[best_target]
    final_model = HistGradientBoostingRegressor(**hgb_kwargs)
    final_model.fit(X, y_target)

    # Save pickle and metadata
    out = Path(args.model_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({
            "model": final_model,
            "features": available,
            "target": best_target,
            "hgb_kwargs": hgb_kwargs,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")

    metrics_path = Path(args.metrics_output)
    metrics_path.write_text(json.dumps(
        {"target_rmse": results, "best": best_target, "features": available},
        indent=2,
    ) + "\n", encoding="utf-8")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
