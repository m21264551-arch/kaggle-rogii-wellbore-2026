"""Out-of-fold validation and reporting utilities."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from .baseline import (
    FEATURE_COLUMNS,
    MODEL_KINDS,
    _as_float_array,
    _formation_pf_predict,
    _gr_correlation_adjustment,
    _pf_stack_predict,
    _postprocess_predictions,
    _scenario_indices,
    _selector_pf_ncc_predict,
    _selector_pf_predict,
    _target_indices_from_actual_mask,
    build_features,
    train_residual_model,
)
from .formations import FormationImputer
from .data import (
    list_well_ids,
    numeric_column,
    read_horizontal,
    read_typewell,
    require_data_dir,
)


MASK_STRATEGIES = ("actual", "artificial", "both")
PREDICTION_COLUMNS = [
    "model_kind",
    "fold",
    "mask_strategy",
    "scenario",
    "well_id",
    "row_index",
    "id",
    "actual_tvt",
    "predicted_tvt",
    "residual",
    "abs_error",
]


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def _safe_nan_range(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return float("nan")
    return float(numeric.max() - numeric.min())


def build_well_profiles(data_dir: Path, wells: Iterable[str]) -> pd.DataFrame:
    """Build well-level metadata used for fold balancing and diagnostics."""

    rows: list[dict[str, object]] = []
    for well_id in wells:
        horizontal = read_horizontal(data_dir, "train", well_id)
        typewell = read_typewell(data_dir, "train", well_id)
        tvt_input = numeric_column(horizontal, "TVT_input")
        true_tvt = numeric_column(horizontal, "TVT")
        gr = numeric_column(horizontal, "GR")
        known_mask = tvt_input.notna()
        hidden_mask = tvt_input.isna() & true_tvt.notna()

        first_hidden = int(np.flatnonzero(hidden_mask.to_numpy())[0]) if hidden_mask.any() else None
        rows.append(
            {
                "well_id": well_id,
                "rows": int(len(horizontal)),
                "known_rows": int(known_mask.sum()),
                "hidden_rows": int(hidden_mask.sum()),
                "known_fraction": float(known_mask.sum() / max(len(horizontal), 1)),
                "first_hidden_index": first_hidden,
                "gr_missing_fraction": float(gr.isna().sum() / max(len(horizontal), 1)),
                "tvt_range": _safe_nan_range(true_tvt),
                "typewell_rows": int(len(typewell)),
                "typewell_tvt_range": _safe_nan_range(typewell.get("TVT", pd.Series(dtype="float64"))),
                "typewell_gr_missing_fraction": float(
                    numeric_column(typewell, "GR").isna().sum() / max(len(typewell), 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def _quantile_bins(values: pd.Series, n_bins: int) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    if values.nunique(dropna=True) <= 1:
        return pd.Series(0, index=values.index, dtype="int64")
    bins = pd.qcut(values.rank(method="first"), q=min(n_bins, values.notna().sum()), labels=False, duplicates="drop")
    return bins.fillna(0).astype("int64")


def assign_validation_folds(profiles: pd.DataFrame, n_folds: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Assign wells to approximately balanced folds using well-level bins."""

    if profiles.empty:
        raise ValueError("No well profiles were provided.")
    n_folds = max(1, min(int(n_folds), len(profiles)))
    folded = profiles.copy()
    folded["known_bin"] = _quantile_bins(folded["known_fraction"], 5)
    folded["gr_missing_bin"] = _quantile_bins(folded["gr_missing_fraction"], 5)
    folded["row_count_bin"] = _quantile_bins(folded["rows"], 5)
    folded["bucket"] = (
        folded["known_bin"].astype(str)
        + "_"
        + folded["gr_missing_bin"].astype(str)
        + "_"
        + folded["row_count_bin"].astype(str)
    )
    folded["fold"] = -1

    rng = np.random.default_rng(random_state)
    fold_row_load = np.zeros(n_folds, dtype="float64")
    for _, group in folded.groupby("bucket", sort=True):
        positions = group.index.to_numpy()
        positions = positions[rng.permutation(len(positions))]
        for position in positions:
            fold = int(np.argmin(fold_row_load))
            folded.loc[position, "fold"] = fold
            fold_row_load[fold] += float(folded.loc[position, "rows"])

    return folded.drop(columns=["bucket"])


def _iter_mask_scenarios(
    horizontal: pd.DataFrame,
    true_tvt: np.ndarray,
    mask_strategy: str,
    ps_fractions: tuple[float, ...],
    target_row_limit: int | None,
) -> Iterable[tuple[str, str, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(0)
    if mask_strategy in {"actual", "both"}:
        known, target_indices = _target_indices_from_actual_mask(horizontal, true_tvt)
        target_indices = _limit_indices(target_indices, target_row_limit, rng)
        if len(target_indices):
            yield "actual", "actual", known, target_indices

    if mask_strategy in {"artificial", "both"}:
        for fraction in ps_fractions:
            ps_index, target_indices = _scenario_indices(
                len(horizontal),
                fraction,
                per_scenario_limit=len(horizontal),
            )
            known = true_tvt.copy()
            known[ps_index:] = np.nan
            target_indices = target_indices[np.isfinite(true_tvt[target_indices])]
            target_indices = _limit_indices(target_indices, target_row_limit, rng)
            if len(target_indices):
                yield "artificial", f"artificial_{fraction:.3f}", known, target_indices


def _limit_indices(indices: np.ndarray, target_row_limit: int | None, rng: np.random.Generator) -> np.ndarray:
    if target_row_limit is None or target_row_limit <= 0 or len(indices) <= target_row_limit:
        return indices.astype("int64")
    keep = np.sort(rng.choice(indices, size=target_row_limit, replace=False))
    return keep.astype("int64")


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    finite = np.isfinite(actual) & np.isfinite(predicted)
    if not finite.any():
        return float("nan")
    return float(np.sqrt(mean_squared_error(actual[finite], predicted[finite])))


def _predict_scenario(
    data_dir: Path,
    model_kind: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    target_indices: np.ndarray,
    known: np.ndarray,
    residual_models: dict[str, object],
    train_mask_strategy: str,
    gr_correction: bool,
    pf_n_particles: int,
    pf_n_seeds: int,
    pf_likelihood_scale: float,
    pf_blend_weight: float,
    beam_blend_weight: float,
    formation_imputer: FormationImputer | None = None,
    well_id: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    features = build_features(horizontal, typewell, target_indices, tvt_known=known)
    if model_kind == "formation_pf":
        if formation_imputer is None:
            raise ValueError("formation_pf requires a formation_imputer")
        prediction = _formation_pf_predict(
            horizontal,
            typewell,
            target_indices,
            imputer=formation_imputer,
            tvt_known=known,
            pf_n_particles=pf_n_particles,
            pf_n_seeds=pf_n_seeds,
            exclude_well_id=well_id,
        )
    elif model_kind in {"selector_pf", "selector_pf_ncc"}:
        selector_function = _selector_pf_ncc_predict if model_kind == "selector_pf_ncc" else _selector_pf_predict
        prediction = selector_function(
            horizontal,
            typewell,
            target_indices,
            tvt_known=known,
            pf_n_particles=pf_n_particles,
            pf_n_seeds=pf_n_seeds,
        )
    elif model_kind in {"pf", "physical_pf"}:
        prediction = _pf_stack_predict(
            horizontal,
            typewell,
            target_indices,
            tvt_known=known,
            physical_tvt=None,
            pf_n_particles=pf_n_particles,
            pf_n_seeds=pf_n_seeds,
            pf_likelihood_scale=pf_likelihood_scale,
            physical_blend_weight=0.0,
            pf_blend_weight=pf_blend_weight,
            beam_blend_weight=beam_blend_weight,
        )
    else:
        prediction = features["geom_tvt"].to_numpy(dtype="float64")
        if model_kind == "residual":
            model = residual_models[train_mask_strategy]
            prediction = prediction + model.predict(features[FEATURE_COLUMNS])
        if gr_correction:
            prediction = _gr_correlation_adjustment(horizontal, typewell, target_indices, prediction)

    return _postprocess_predictions(prediction, features), features


def _write_frame(path: Path, frame: pd.DataFrame, write_header: bool) -> None:
    frame.to_csv(path, mode="w" if write_header else "a", index=False, header=write_header)


def _summary_from_metrics(metrics: pd.DataFrame) -> dict[str, object]:
    if metrics.empty:
        return {}
    summaries = []
    group_cols = ["model_kind", "mask_strategy", "scenario"]
    for keys, group in metrics.groupby(group_cols, sort=True):
        rows = group["rows"].to_numpy(dtype="float64")
        rmse_values = group["rmse"].to_numpy(dtype="float64")
        finite = np.isfinite(rows) & np.isfinite(rmse_values) & (rows > 0)
        if not finite.any():
            continue
        weighted_rmse = float(np.sqrt(np.average(rmse_values[finite] ** 2, weights=rows[finite])))
        summaries.append(
            {
                "model_kind": keys[0],
                "mask_strategy": keys[1],
                "scenario": keys[2],
                "wells": int(finite.sum()),
                "rows": int(rows[finite].sum()),
                "weighted_rmse": weighted_rmse,
                "mean_well_rmse": float(np.nanmean(rmse_values[finite])),
                "median_well_rmse": float(np.nanmedian(rmse_values[finite])),
                "p90_well_rmse": float(np.nanquantile(rmse_values[finite], 0.90)),
                "p99_well_rmse": float(np.nanquantile(rmse_values[finite], 0.99)),
                "worst_well": str(group.loc[group["rmse"].idxmax(), "well_id"]),
                "worst_well_rmse": float(np.nanmax(rmse_values[finite])),
            }
        )
    return {"groups": summaries}


def run_oof_validation(
    data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    model_kind: str = "pf",
    n_folds: int = 5,
    mask_strategy: str = "actual",
    ps_fractions: tuple[float, ...] = (0.20, 0.24, 0.28, 0.32, 0.36),
    max_wells: int | None = None,
    target_row_limit: int | None = None,
    max_train_rows: int = 800_000,
    random_state: int = 42,
    gr_correction: bool = False,
    pf_n_particles: int = 256,
    pf_n_seeds: int = 8,
    pf_likelihood_scale: float = 5.0,
    pf_blend_weight: float = 1.0,
    beam_blend_weight: float = 1.0,
) -> dict[str, Path]:
    """Run OOF validation and write predictions, metrics, folds, and summary files."""

    if model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind must be one of {MODEL_KINDS}")
    if mask_strategy not in MASK_STRATEGIES:
        raise ValueError(f"mask_strategy must be one of {MASK_STRATEGIES}")

    start_time = perf_counter()
    resolved = require_data_dir(data_dir)
    wells = list_well_ids(resolved, "train")
    if max_wells is not None and max_wells > 0:
        wells = wells[:max_wells]
    if not wells:
        raise ValueError("No train wells found.")

    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path("reports") / "oof" / f"{stamp}_{model_kind}_{mask_strategy}"
    else:
        output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    profiles = build_well_profiles(resolved, wells)
    folds = assign_validation_folds(profiles, n_folds=n_folds, random_state=random_state)
    folds_path = output / "folds.csv"
    profiles_path = output / "well_profiles.csv"
    predictions_path = output / "oof_predictions.csv"
    metrics_path = output / "well_metrics.csv"
    summary_path = output / "summary.json"
    config_path = output / "config.json"
    folds.to_csv(folds_path, index=False)
    profiles.to_csv(profiles_path, index=False)

    config = {
        "data_dir": str(resolved),
        "model_kind": model_kind,
        "n_folds": int(min(n_folds, len(wells))),
        "mask_strategy": mask_strategy,
        "ps_fractions": list(ps_fractions),
        "max_wells": max_wells,
        "target_row_limit": target_row_limit,
        "max_train_rows": max_train_rows,
        "random_state": random_state,
        "gr_correction": gr_correction,
        "pf_n_particles": pf_n_particles,
        "pf_n_seeds": pf_n_seeds,
        "pf_likelihood_scale": pf_likelihood_scale,
        "pf_blend_weight": pf_blend_weight,
        "beam_blend_weight": beam_blend_weight,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    metrics: list[dict[str, object]] = []
    write_header = True
    actual_folds = sorted(folds["fold"].unique())

    # Build a single FormationImputer over ALL train wells.  In a strict OOF
    # setup we would rebuild per-fold, but the imputer dropping the validation
    # well itself (via ``exclude_well_id``) is already a row-level leave-one-out
    # that mirrors how test-time imputation works.
    formation_imputer: FormationImputer | None = None
    if model_kind == "formation_pf":
        all_train_well_ids = list_well_ids(resolved, "train")
        formation_imputer = FormationImputer.build(resolved, all_train_well_ids, points_per_well=60)

    for fold in actual_folds:
        fold_start = perf_counter()
        train_wells = folds.loc[folds["fold"] != fold, "well_id"].tolist()
        val_wells = folds.loc[folds["fold"] == fold, "well_id"].tolist()
        residual_models: dict[str, object] = {}
        if model_kind == "residual":
            train_strategies = ["actual", "artificial"] if mask_strategy == "both" else [mask_strategy]
            for train_mask_strategy in train_strategies:
                print(f"fold {fold}: training residual model with {train_mask_strategy} masks on {len(train_wells)} wells")
                residual_models[train_mask_strategy] = train_residual_model(
                    resolved,
                    train_wells=train_wells,
                    max_train_rows=max_train_rows,
                    ps_fractions=ps_fractions,
                    mask_strategy=train_mask_strategy,
                    random_state=random_state + int(fold),
                )

        print(f"fold {fold}: scoring {len(val_wells)} validation wells")
        for well_id in val_wells:
            horizontal = read_horizontal(resolved, "train", well_id)
            if "TVT" not in horizontal.columns:
                continue
            typewell = read_typewell(resolved, "train", well_id)
            true_tvt = _as_float_array(horizontal, "TVT")
            profile = folds.loc[folds["well_id"] == well_id].iloc[0].to_dict()
            for scenario_mask, scenario_name, known, target_indices in _iter_mask_scenarios(
                horizontal,
                true_tvt,
                mask_strategy=mask_strategy,
                ps_fractions=ps_fractions,
                target_row_limit=target_row_limit,
            ):
                train_mask_strategy = "actual" if scenario_mask == "actual" else "artificial"
                prediction, _ = _predict_scenario(
                    resolved,
                    model_kind,
                    horizontal,
                    typewell,
                    target_indices,
                    known,
                    residual_models=residual_models,
                    train_mask_strategy=train_mask_strategy,
                    gr_correction=gr_correction,
                    pf_n_particles=pf_n_particles,
                    pf_n_seeds=pf_n_seeds,
                    pf_likelihood_scale=pf_likelihood_scale,
                    pf_blend_weight=pf_blend_weight,
                    beam_blend_weight=beam_blend_weight,
                    formation_imputer=formation_imputer,
                    well_id=well_id,
                )
                actual = true_tvt[target_indices]
                finite = np.isfinite(actual) & np.isfinite(prediction)
                if not finite.any():
                    continue

                pred_frame = pd.DataFrame(
                    {
                        "model_kind": model_kind,
                        "fold": int(fold),
                        "mask_strategy": scenario_mask,
                        "scenario": scenario_name,
                        "well_id": well_id,
                        "row_index": target_indices[finite].astype("int64"),
                        "id": [f"{well_id}_{row_index}" for row_index in target_indices[finite]],
                        "actual_tvt": actual[finite],
                        "predicted_tvt": prediction[finite],
                    }
                )
                pred_frame["residual"] = pred_frame["predicted_tvt"] - pred_frame["actual_tvt"]
                pred_frame["abs_error"] = pred_frame["residual"].abs()
                _write_frame(predictions_path, pred_frame[PREDICTION_COLUMNS], write_header)
                write_header = False

                abs_error = pred_frame["abs_error"].to_numpy(dtype="float64")
                metrics.append(
                    {
                        "model_kind": model_kind,
                        "fold": int(fold),
                        "mask_strategy": scenario_mask,
                        "scenario": scenario_name,
                        "well_id": well_id,
                        "rows": int(len(pred_frame)),
                        "rmse": _rmse(actual[finite], prediction[finite]),
                        "mae": float(np.mean(abs_error)),
                        "bias": float(pred_frame["residual"].mean()),
                        "p95_abs_error": float(np.quantile(abs_error, 0.95)),
                        "max_abs_error": float(np.max(abs_error)),
                        "known_fraction": float(profile["known_fraction"]),
                        "gr_missing_fraction": float(profile["gr_missing_fraction"]),
                        "typewell_rows": int(profile["typewell_rows"]),
                    }
                )
        print(f"fold {fold}: finished in {perf_counter() - fold_start:.1f}s")

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(metrics_path, index=False)
    summary = _summary_from_metrics(metrics_frame)
    summary["runtime_seconds"] = round(perf_counter() - start_time, 3)
    summary["output_dir"] = str(output)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

    return {
        "output_dir": output,
        "folds": folds_path,
        "profiles": profiles_path,
        "predictions": predictions_path,
        "metrics": metrics_path,
        "summary": summary_path,
        "config": config_path,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run out-of-fold validation and save row-level predictions.")
    parser.add_argument("--data-dir", default=None, help="Competition data directory.")
    parser.add_argument("--output-dir", default=None, help="Directory for OOF artifacts.")
    parser.add_argument("--model-kind", choices=MODEL_KINDS, default="pf")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--mask-strategy", choices=MASK_STRATEGIES, default="actual")
    parser.add_argument("--ps-fractions", type=_parse_float_list, default=(0.20, 0.24, 0.28, 0.32, 0.36))
    parser.add_argument("--max-wells", type=int, default=None, help="Limit wells for a quick smoke run.")
    parser.add_argument("--target-row-limit", type=int, default=None, help="Limit scored target rows per well/scenario.")
    parser.add_argument("--max-train-rows", type=int, default=800_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--gr-correction", action="store_true", help="Apply GR correlation correction to geometry/residual.")
    parser.add_argument("--pf-particles", type=int, default=256)
    parser.add_argument("--pf-seeds", type=int, default=8)
    parser.add_argument("--pf-likelihood-scale", type=float, default=5.0)
    parser.add_argument("--pf-blend-weight", type=float, default=1.0)
    parser.add_argument("--beam-blend-weight", type=float, default=1.0)
    args = parser.parse_args(argv)
    run_oof_validation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_kind=args.model_kind,
        n_folds=args.folds,
        mask_strategy=args.mask_strategy,
        ps_fractions=args.ps_fractions,
        max_wells=args.max_wells,
        target_row_limit=args.target_row_limit,
        max_train_rows=args.max_train_rows,
        random_state=args.random_state,
        gr_correction=args.gr_correction,
        pf_n_particles=args.pf_particles,
        pf_n_seeds=args.pf_seeds,
        pf_likelihood_scale=args.pf_likelihood_scale,
        pf_blend_weight=args.pf_blend_weight,
        beam_blend_weight=args.beam_blend_weight,
    )


if __name__ == "__main__":
    main()
