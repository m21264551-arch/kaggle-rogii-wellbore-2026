#!/usr/bin/env python3
"""Generate per-row features + signals for meta-learner training.

For each train well: holds out that well from the FormationImputer
(``exclude_well_id``), runs the selector-PF and the spatial-formation
predictor on the actual mask, and writes a parquet row per hidden-region
position with:

* signals: ``pf_pred``, ``form_pred``, per-formation predictions, geom_tvt
* per-row imputation features: ``plane_distance``, ``dense_distance``,
  ``dens_std``
* per-well features: ``qk`` (formation known-zone RMSE), per-formation
  known-zone RMSE
* trajectory features from ``build_features``: ``delta_md``, ``delta_z``,
  ``dist_xy``, ``row_fraction``, ``gr``, ``last_tvt``, …
* the ground-truth ``actual_tvt`` for supervision.

The OOF nature comes from ``exclude_well_id`` — the imputer doesn't see
the validation well, mirroring how test-time imputation behaves.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_baseline.baseline import build_features, _selector_pf_predict
from rogii_baseline.data import list_well_ids, read_horizontal, read_typewell, require_data_dir
from rogii_baseline.formations import FORMATIONS, FormationImputer, predict_physical_tvt


META_FEATURES = [
    "pf_pred",
    "form_pred",
    "geom_tvt",
    "plane_distance",
    "dense_distance",
    "dens_std",
    "qk",
    "last_tvt",
    "last_md",
    "delta_md",
    "delta_z",
    "delta_x",
    "delta_y",
    "dist_xy",
    "row_fraction",
    "ps_fraction",
    "gr",
    "gr_missing",
    "typewell_gr_diff",
    "geom_slope",
    "geom_quad_delta",
    "z",
    "md",
]


def collect_one(
    data_dir: Path,
    well_id: str,
    imputer: FormationImputer,
    pf_seeds: int,
    pf_particles: int,
) -> pd.DataFrame | None:
    horizontal = read_horizontal(data_dir, "train", well_id)
    if "TVT" not in horizontal.columns:
        return None
    typewell = read_typewell(data_dir, "train", well_id)
    tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce").to_numpy(dtype="float64")
    tvt_true = pd.to_numeric(horizontal["TVT"], errors="coerce").to_numpy(dtype="float64")
    hidden_idx = np.flatnonzero(~np.isfinite(tvt_input) & np.isfinite(tvt_true))
    if len(hidden_idx) == 0:
        return None

    fres = predict_physical_tvt(horizontal, tvt_input, imputer, exclude_well_id=well_id)
    pf_pred = _selector_pf_predict(
        horizontal, typewell, hidden_idx,
        tvt_known=tvt_input,
        pf_n_particles=pf_particles,
        pf_n_seeds=pf_seeds,
    )
    feats = build_features(horizontal, typewell, hidden_idx, tvt_known=tvt_input)

    row = pd.DataFrame({
        "well_id": well_id,
        "row_index": hidden_idx,
        "actual_tvt": tvt_true[hidden_idx],
        "pf_pred": pf_pred,
        "form_pred": fres["physical_prediction"][hidden_idx],
        "plane_distance": fres["plane_distance"][hidden_idx],
        "dense_distance": fres["dense_distance"][hidden_idx],
        "dens_std": fres["stddev_imputed"][hidden_idx],
        "qk": fres["overall_quality_rmse"],
    })
    # Per-formation predictions and known-zone RMSEs
    pfp = fres["per_formation_predictions"][hidden_idx]
    pfr = fres["per_formation_rmse"]
    for i, fname in enumerate(FORMATIONS):
        col_pred = f"form_{fname}_pred"
        col_rmse = f"form_{fname}_qk"
        row[col_pred] = pfp[:, i] if pfp.shape[1] > i else np.nan
        row[col_rmse] = float(pfr[i]) if i < len(pfr) and np.isfinite(pfr[i]) else 99.0
    # build_features columns
    for col in [
        "row_fraction", "ps_fraction", "delta_md", "delta_x", "delta_y", "delta_z",
        "dist_xy", "gr", "gr_missing", "geom_tvt", "geom_slope", "geom_quad_delta",
        "last_tvt", "last_md", "typewell_gr_diff",
    ]:
        if col in feats.columns:
            row[col] = feats[col].to_numpy()
    # z, md raw values
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(dtype="float64")
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(dtype="float64")
    row["z"] = z[hidden_idx]
    row["md"] = md[hidden_idx]
    return row


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default="reports/meta/train_200w.parquet")
    parser.add_argument("--max-wells", type=int, default=200)
    parser.add_argument("--pf-seeds", type=int, default=16)
    parser.add_argument("--pf-particles", type=int, default=384)
    parser.add_argument("--points-per-well", type=int, default=60)
    args = parser.parse_args(argv)

    data_dir = require_data_dir(args.data_dir)
    all_wells = list_well_ids(data_dir, "train")
    wells = all_wells[: args.max_wells] if args.max_wells > 0 else all_wells
    print(f"Total train wells: {len(all_wells)}, using {len(wells)} for meta-training")

    print("Building FormationImputer...")
    t0 = time.time()
    imputer = FormationImputer.build(data_dir, all_wells, points_per_well=args.points_per_well)
    print(f"  centroids={imputer.n_centroids}, dense={imputer.n_dense_points}, built in {time.time()-t0:.1f}s")

    parts = []
    t_start = time.time()
    for i, wid in enumerate(wells):
        row = collect_one(data_dir, wid, imputer, args.pf_seeds, args.pf_particles)
        if row is not None:
            parts.append(row)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            eta = elapsed * (len(wells) - i - 1) / (i + 1)
            print(f"  [{i+1}/{len(wells)}] elapsed={elapsed:.0f}s eta={eta:.0f}s")

    df = pd.concat(parts, ignore_index=True)
    print(f"\nTotal rows: {len(df)} from {df['well_id'].nunique()} wells")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
