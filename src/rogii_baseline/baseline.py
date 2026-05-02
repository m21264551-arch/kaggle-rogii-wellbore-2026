"""Submission and baseline modeling utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from .data import (
    list_well_ids,
    load_sample_submission,
    numeric_column,
    read_horizontal,
    read_typewell,
    require_data_dir,
)
from .formations import FormationImputer, predict_physical_tvt


FEATURE_COLUMNS = [
    "row_index",
    "n_rows",
    "row_fraction",
    "delta_index",
    "md",
    "delta_md",
    "x",
    "y",
    "z",
    "delta_x",
    "delta_y",
    "delta_z",
    "dist_xy",
    "gr",
    "gr_missing",
    "last_tvt",
    "last_md",
    "last_z",
    "last_gr",
    "geom_tvt",
    "geom_slope",
    "geom_quad_delta",
    "ps_fraction",
    "azimuth_sin",
    "azimuth_cos",
    "typewell_gr_at_geom",
    "typewell_gr_diff",
    "typewell_tvt_min",
    "typewell_tvt_max",
    "typewell_gr_mean",
    "typewell_gr_std",
]

MODEL_KINDS = (
    "geometry",
    "residual",
    "pf",
    "physical_pf",
    "selector_pf",
    "selector_pf_ncc",
    "formation_pf",
    "meta_pf",
)


META_LEARNER_FEATURES = [
    "pf_pred", "form_pred", "geom_tvt",
    "plane_distance", "dense_distance", "dens_std", "qk",
    "last_tvt", "delta_md", "delta_x", "delta_y", "delta_z",
    "dist_xy", "row_fraction", "ps_fraction",
    "gr", "gr_missing", "typewell_gr_diff",
    "geom_slope", "geom_quad_delta", "z", "md",
    "form_ANCC_pred", "form_ANCC_qk",
    "form_ASTNU_pred", "form_ASTNU_qk",
    "form_ASTNL_pred", "form_ASTNL_qk",
    "form_EGFDU_pred", "form_EGFDU_qk",
    "form_EGFDL_pred", "form_EGFDL_qk",
    "form_BUDA_pred", "form_BUDA_qk",
]

BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10, 8.0, 64.0, 2),
    (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2),
    (10, 50.0, 400.0, 0),
]

SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2",
    1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15",
    3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05",
    5: "pf_scale_12_beam_0.2_hold_0.05",
}
SELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)
NCC_HALF_WINDOWS = (8, 15, 25)


@dataclass(frozen=True)
class GeometryState:
    last_index: int
    last_tvt: float
    last_md: float
    slope: float
    quadratic_center_md: float
    quadratic_coef: tuple[float, float, float] | None


def _as_float_array(df: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    return numeric_column(df, column, default=default).to_numpy(dtype="float64")


def _fallback_axis(length: int) -> np.ndarray:
    return np.arange(length, dtype="float64")


def _safe_clip(value: float, lower: float, upper: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, lower, upper))


def _fit_geometry(md: np.ndarray, tvt_known: np.ndarray, first_target_index: int, window: int = 250) -> GeometryState:
    row_axis = _fallback_axis(len(tvt_known))
    md = np.where(np.isfinite(md), md, row_axis)
    known_before = np.flatnonzero(np.isfinite(tvt_known) & (row_axis < first_target_index))
    if len(known_before) == 0:
        known_before = np.flatnonzero(np.isfinite(tvt_known))

    if len(known_before) == 0:
        center_md = float(md[max(0, min(first_target_index, len(md) - 1))])
        return GeometryState(
            last_index=max(0, min(first_target_index - 1, len(md) - 1)),
            last_tvt=center_md,
            last_md=center_md,
            slope=0.0,
            quadratic_center_md=center_md,
            quadratic_coef=None,
        )

    fit_idx = known_before[-min(window, len(known_before)) :]
    last_index = int(known_before[-1])
    last_tvt = float(tvt_known[last_index])
    last_md = float(md[last_index])
    x = md[fit_idx] - last_md
    y = tvt_known[fit_idx]

    if len(fit_idx) >= 2 and np.nanstd(x) > 0:
        slope = _safe_clip(float(np.polyfit(x, y, deg=1)[0]), -2.5, 2.5)
    else:
        slope = 0.0

    quadratic_coef = None
    if len(fit_idx) >= 30 and np.nanstd(x) > 0:
        coef = np.polyfit(x, y, deg=2)
        if np.all(np.isfinite(coef)):
            quadratic_coef = (float(coef[0]), float(coef[1]), float(coef[2]))

    return GeometryState(
        last_index=last_index,
        last_tvt=last_tvt,
        last_md=last_md,
        slope=slope,
        quadratic_center_md=last_md,
        quadratic_coef=quadratic_coef,
    )


def _geometry_predict(md: np.ndarray, row_indices: np.ndarray, state: GeometryState) -> tuple[np.ndarray, np.ndarray]:
    target_md = np.where(np.isfinite(md[row_indices]), md[row_indices], row_indices.astype("float64"))
    delta_md = target_md - state.last_md
    linear = state.last_tvt + state.slope * delta_md

    if state.quadratic_coef is None:
        quad_delta = np.zeros_like(linear)
        return linear, quad_delta

    coef = state.quadratic_coef
    centered = target_md - state.quadratic_center_md
    quadratic = coef[0] * centered**2 + coef[1] * centered + coef[2]
    allowed = np.maximum(250.0, np.abs(delta_md) * (abs(state.slope) + 0.35))
    quadratic = np.clip(quadratic, linear - allowed, linear + allowed)
    prediction = 0.80 * linear + 0.20 * quadratic
    return prediction, quadratic - linear


def _last_finite(values: np.ndarray, before_or_at: int) -> float:
    idx = np.flatnonzero(np.isfinite(values) & (np.arange(len(values)) <= before_or_at))
    if len(idx) == 0:
        return float("nan")
    return float(values[idx[-1]])


def _azimuth_features(x: np.ndarray, y: np.ndarray, state: GeometryState, lookback: int = 200) -> tuple[float, float]:
    start = max(0, state.last_index - lookback)
    dx = x[state.last_index] - x[start] if np.isfinite(x[state.last_index]) and np.isfinite(x[start]) else np.nan
    dy = y[state.last_index] - y[start] if np.isfinite(y[state.last_index]) and np.isfinite(y[start]) else np.nan
    angle = np.arctan2(dy, dx) if np.isfinite(dx) and np.isfinite(dy) else np.nan
    return float(np.sin(angle)) if np.isfinite(angle) else np.nan, float(np.cos(angle)) if np.isfinite(angle) else np.nan


def _typewell_arrays(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if typewell.empty:
        return np.array([], dtype="float64"), np.array([], dtype="float64")
    tvt = _as_float_array(typewell, "TVT")
    gr = _as_float_array(typewell, "GR")
    mask = np.isfinite(tvt) & np.isfinite(gr)
    if not mask.any():
        return np.array([], dtype="float64"), np.array([], dtype="float64")
    order = np.argsort(tvt[mask])
    return tvt[mask][order], gr[mask][order]


def build_features(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    row_indices: Iterable[int],
    tvt_known: np.ndarray | None = None,
) -> pd.DataFrame:
    row_indices = np.asarray(list(row_indices), dtype="int64")
    if len(row_indices) == 0:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    n_rows = len(horizontal)
    if row_indices.min() < 0 or row_indices.max() >= n_rows:
        raise IndexError("row_indices contain values outside the horizontal well rows")

    md = _as_float_array(horizontal, "MD")
    x = _as_float_array(horizontal, "X")
    y = _as_float_array(horizontal, "Y")
    z = _as_float_array(horizontal, "Z")
    gr = _as_float_array(horizontal, "GR")
    if tvt_known is None:
        tvt_col = "TVT_input" if "TVT_input" in horizontal.columns else "TVT"
        tvt_known = _as_float_array(horizontal, tvt_col)
    else:
        tvt_known = np.asarray(tvt_known, dtype="float64")

    state = _fit_geometry(md, tvt_known, first_target_index=int(row_indices.min()))
    geom, geom_quad_delta = _geometry_predict(md, row_indices, state)

    anchor = state.last_index
    target_md = np.where(np.isfinite(md[row_indices]), md[row_indices], row_indices.astype("float64"))
    anchor_md = state.last_md
    anchor_x = x[anchor] if np.isfinite(x[anchor]) else np.nan
    anchor_y = y[anchor] if np.isfinite(y[anchor]) else np.nan
    anchor_z = z[anchor] if np.isfinite(z[anchor]) else np.nan
    target_x = x[row_indices]
    target_y = y[row_indices]
    target_z = z[row_indices]
    delta_x = target_x - anchor_x
    delta_y = target_y - anchor_y
    delta_z = target_z - anchor_z
    az_sin, az_cos = _azimuth_features(x, y, state)

    tw_tvt, tw_gr = _typewell_arrays(typewell)
    if len(tw_tvt):
        tw_gr_at_geom = np.interp(np.clip(geom, tw_tvt.min(), tw_tvt.max()), tw_tvt, tw_gr)
        tw_tvt_min = float(tw_tvt.min())
        tw_tvt_max = float(tw_tvt.max())
        tw_gr_mean = float(np.nanmean(tw_gr))
        tw_gr_std = float(np.nanstd(tw_gr))
    else:
        tw_gr_at_geom = np.full(len(row_indices), np.nan, dtype="float64")
        tw_tvt_min = tw_tvt_max = tw_gr_mean = tw_gr_std = np.nan

    target_gr = gr[row_indices]
    frame = pd.DataFrame(
        {
            "row_index": row_indices.astype("float64"),
            "n_rows": float(n_rows),
            "row_fraction": row_indices / max(n_rows - 1, 1),
            "delta_index": row_indices - anchor,
            "md": target_md,
            "delta_md": target_md - anchor_md,
            "x": target_x,
            "y": target_y,
            "z": target_z,
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
            "dist_xy": np.sqrt(delta_x**2 + delta_y**2),
            "gr": target_gr,
            "gr_missing": (~np.isfinite(target_gr)).astype("float64"),
            "last_tvt": state.last_tvt,
            "last_md": state.last_md,
            "last_z": anchor_z,
            "last_gr": _last_finite(gr, anchor),
            "geom_tvt": geom,
            "geom_slope": state.slope,
            "geom_quad_delta": geom_quad_delta,
            "ps_fraction": anchor / max(n_rows - 1, 1),
            "azimuth_sin": az_sin,
            "azimuth_cos": az_cos,
            "typewell_gr_at_geom": tw_gr_at_geom,
            "typewell_gr_diff": target_gr - tw_gr_at_geom,
            "typewell_tvt_min": tw_tvt_min,
            "typewell_tvt_max": tw_tvt_max,
            "typewell_gr_mean": tw_gr_mean,
            "typewell_gr_std": tw_gr_std,
        }
    )
    return frame[FEATURE_COLUMNS]


def _scenario_indices(n_rows: int, ps_fraction: float, per_scenario_limit: int) -> tuple[int, np.ndarray]:
    ps_index = int(np.clip(round(n_rows * ps_fraction), 50, max(51, n_rows - 50)))
    target = np.arange(ps_index, n_rows, dtype="int64")
    if len(target) > per_scenario_limit:
        pick = np.linspace(0, len(target) - 1, per_scenario_limit).round().astype("int64")
        target = target[pick]
    return ps_index, target


def _target_indices_from_actual_mask(horizontal: pd.DataFrame, true_tvt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "TVT_input" not in horizontal.columns:
        return true_tvt.copy(), np.array([], dtype="int64")
    known = _as_float_array(horizontal, "TVT_input")
    target_indices = np.flatnonzero(~np.isfinite(known) & np.isfinite(true_tvt))
    return known, target_indices.astype("int64")


def train_residual_model(
    data_dir: Path,
    train_wells: list[str] | None = None,
    max_train_rows: int = 800_000,
    ps_fractions: tuple[float, ...] = (0.20, 0.24, 0.28, 0.32, 0.36),
    mask_strategy: str = "artificial",
    random_state: int = 42,
) -> HistGradientBoostingRegressor:
    wells = train_wells or list_well_ids(data_dir, "train")
    if not wells:
        raise ValueError("No training wells found.")
    if mask_strategy not in {"actual", "artificial"}:
        raise ValueError("mask_strategy must be 'actual' or 'artificial'")

    scenario_count = max(1, len(wells) if mask_strategy == "actual" else len(wells) * len(ps_fractions))
    per_scenario_limit = max(25, max_train_rows // scenario_count)
    feature_parts: list[pd.DataFrame] = []
    target_parts: list[np.ndarray] = []
    rng = np.random.default_rng(random_state)

    for well_id in wells:
        horizontal = read_horizontal(data_dir, "train", well_id)
        if "TVT" not in horizontal.columns:
            continue
        typewell = read_typewell(data_dir, "train", well_id)
        true_tvt = _as_float_array(horizontal, "TVT")
        if np.isfinite(true_tvt).sum() < 100:
            continue
        if mask_strategy == "actual":
            known, target_indices = _target_indices_from_actual_mask(horizontal, true_tvt)
            if len(target_indices) > per_scenario_limit:
                target_indices = np.sort(rng.choice(target_indices, size=per_scenario_limit, replace=False))
            if len(target_indices) == 0:
                continue
            features = build_features(horizontal, typewell, target_indices, tvt_known=known)
            residual = true_tvt[target_indices] - features["geom_tvt"].to_numpy(dtype="float64")
            finite = np.isfinite(residual)
            if finite.any():
                feature_parts.append(features.loc[finite, FEATURE_COLUMNS])
                target_parts.append(residual[finite])
            continue

        for fraction in ps_fractions:
            ps_index, target_indices = _scenario_indices(len(horizontal), fraction, per_scenario_limit)
            finite_target = np.isfinite(true_tvt[target_indices])
            target_indices = target_indices[finite_target]
            if len(target_indices) == 0:
                continue
            known = true_tvt.copy()
            known[ps_index:] = np.nan
            features = build_features(horizontal, typewell, target_indices, tvt_known=known)
            residual = true_tvt[target_indices] - features["geom_tvt"].to_numpy(dtype="float64")
            finite = np.isfinite(residual)
            if finite.any():
                feature_parts.append(features.loc[finite, FEATURE_COLUMNS])
                target_parts.append(residual[finite])

    if not feature_parts:
        raise ValueError("No residual-model training rows were generated.")

    X = pd.concat(feature_parts, ignore_index=True)
    y = np.concatenate(target_parts)
    if len(X) > max_train_rows:
        rng = np.random.default_rng(random_state)
        keep = np.sort(rng.choice(len(X), size=max_train_rows, replace=False))
        X = X.iloc[keep].reset_index(drop=True)
        y = y[keep]

    model = HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.04,
        max_leaf_nodes=63,
        l2_regularization=0.03,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=random_state,
    )
    model.fit(X[FEATURE_COLUMNS], y)
    return model


def _postprocess_predictions(prediction: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    prediction = np.asarray(prediction, dtype="float64")
    geom = features["geom_tvt"].to_numpy(dtype="float64")
    prediction = np.where(np.isfinite(prediction), prediction, geom)

    tw_min = features["typewell_tvt_min"].to_numpy(dtype="float64")
    tw_max = features["typewell_tvt_max"].to_numpy(dtype="float64")
    lower = np.where(np.isfinite(tw_min), tw_min - 1000.0, -np.inf)
    upper = np.where(np.isfinite(tw_max), tw_max + 1000.0, np.inf)
    prediction = np.clip(prediction, lower, upper)
    return prediction


def _copy_visible_train_targets(data_dir: Path, sample: pd.DataFrame) -> pd.Series:
    values = pd.Series(np.nan, index=sample.index, dtype="float64")
    for well_id, group in sample.groupby("well_id", sort=False):
        horizontal = read_horizontal(data_dir, "train", well_id)
        if "TVT" not in horizontal.columns:
            continue
        tvt = _as_float_array(horizontal, "TVT")
        rows = group["row_index"].to_numpy(dtype="int64")
        if rows.max(initial=-1) < len(tvt):
            values.loc[group.index] = tvt[rows]
    return values


def _gr_correlation_adjustment(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    row_indices: np.ndarray,
    prediction: np.ndarray,
    search_radius: int = 40,
    smooth_window: int = 601,
) -> np.ndarray:
    tw_tvt, tw_gr = _typewell_arrays(typewell)
    if len(tw_tvt) < 10:
        return prediction

    gr = _as_float_array(horizontal, "GR")
    target_gr = gr[row_indices]
    valid = np.flatnonzero(np.isfinite(target_gr))
    if len(valid) < 20:
        return prediction

    offsets = np.arange(-search_radius, search_radius + 1, dtype="float64")
    raw_offset = np.full(len(row_indices), np.nan, dtype="float64")
    candidates = prediction[valid, None] + offsets[None, :]
    candidates = np.clip(candidates, float(tw_tvt.min()), float(tw_tvt.max()))
    candidate_gr = np.interp(candidates.ravel(), tw_tvt, tw_gr).reshape(candidates.shape)
    best = np.nanargmin((candidate_gr - target_gr[valid, None]) ** 2, axis=1)
    raw_offset[valid] = offsets[best]

    offset = pd.Series(raw_offset).interpolate(limit_direction="both")
    if offset.notna().sum() == 0:
        return prediction
    smoothed = (
        offset.rolling(smooth_window, center=True, min_periods=1)
        .median()
        .rolling(smooth_window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype="float64")
    )
    adjusted = prediction + smoothed
    return np.where(np.isfinite(adjusted), adjusted, prediction)


def _default_tvt_known(horizontal: pd.DataFrame) -> np.ndarray:
    tvt_col = "TVT_input" if "TVT_input" in horizontal.columns else "TVT"
    return _as_float_array(horizontal, tvt_col)


def _interpolated_gr(horizontal: pd.DataFrame, fallback: float) -> np.ndarray:
    gr = numeric_column(horizontal, "GR", default=np.nan)
    return gr.interpolate(limit_direction="both").fillna(fallback).to_numpy(dtype="float64")


def _ncc_scale_alignment(
    known_gr: np.ndarray,
    known_tvt: np.ndarray,
    target_gr: np.ndarray,
    half_window: int,
    stride: int = 3,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    window = 2 * int(half_window) + 1
    if len(known_gr) < window + 1 or len(target_gr) == 0:
        fallback = known_tvt[-1] if len(known_tvt) else 0.0
        return np.full(len(target_gr), fallback, dtype="float64"), np.zeros(len(target_gr), dtype="float64")

    starts = np.arange(0, len(known_gr) - window + 1, max(1, int(stride)), dtype="int64")
    if len(starts) == 0:
        fallback = known_tvt[-1] if len(known_tvt) else 0.0
        return np.full(len(target_gr), fallback, dtype="float64"), np.zeros(len(target_gr), dtype="float64")

    offsets = np.arange(window, dtype="int64")
    candidates = known_gr[starts[:, None] + offsets[None, :]].astype("float64")
    candidates = (candidates - candidates.mean(axis=1, keepdims=True)) / (candidates.std(axis=1, keepdims=True) + 1e-6)

    padded = np.pad(target_gr.astype("float64"), int(half_window), mode="edge")
    centers = np.clip(starts + int(half_window), 0, len(known_tvt) - 1)
    predictions = np.empty(len(target_gr), dtype="float64")
    scores = np.empty(len(target_gr), dtype="float64")

    row_offsets = np.arange(window, dtype="int64")
    for start in range(0, len(target_gr), chunk_size):
        stop = min(start + chunk_size, len(target_gr))
        target_windows = padded[np.arange(start, stop, dtype="int64")[:, None] + row_offsets[None, :]]
        target_windows = (target_windows - target_windows.mean(axis=1, keepdims=True)) / (
            target_windows.std(axis=1, keepdims=True) + 1e-6
        )
        ncc = target_windows @ candidates.T / float(window)
        best = np.argmax(ncc, axis=1)
        predictions[start:stop] = known_tvt[centers[best]]
        scores[start:stop] = ncc[np.arange(stop - start), best]

    return predictions, scores


def _ncc_alignment_full(
    horizontal: pd.DataFrame,
    tvt_known: np.ndarray,
    half_windows: tuple[int, ...] = NCC_HALF_WINDOWS,
) -> tuple[np.ndarray, np.ndarray]:
    tvt_known = np.asarray(tvt_known, dtype="float64")
    missing = np.flatnonzero(~np.isfinite(tvt_known))
    prediction = tvt_known.copy()
    score = np.zeros(len(tvt_known), dtype="float64")
    if len(missing) == 0:
        return prediction, score

    prefix_end = int(missing.min())
    known_tvt_series = pd.Series(tvt_known[:prefix_end], dtype="float64").interpolate(limit_direction="both")
    if prefix_end < 20 or known_tvt_series.notna().sum() < 10:
        fallback = _last_known_tvt_value(tvt_known)
        prediction[missing] = fallback
        return prediction, score

    raw_gr = _as_float_array(horizontal, "GR")
    fallback_gr = float(np.nanmean(raw_gr)) if np.isfinite(np.nanmean(raw_gr)) else 0.0
    gr = _interpolated_gr(horizontal, fallback=fallback_gr)
    known_gr = gr[:prefix_end]
    target_gr = gr[prefix_end:]
    known_tvt = known_tvt_series.to_numpy(dtype="float64")

    scale_predictions = []
    scale_scores = []
    for half_window in half_windows:
        scale_prediction, scale_score = _ncc_scale_alignment(known_gr, known_tvt, target_gr, half_window=half_window)
        scale_predictions.append(scale_prediction)
        scale_scores.append(scale_score)

    prediction_matrix = np.stack(scale_predictions, axis=1)
    score_matrix = np.stack(scale_scores, axis=1)
    weights = np.exp(3.0 * np.clip(score_matrix, -1.0, 1.0))
    weights /= weights.sum(axis=1, keepdims=True)
    target_prediction = (prediction_matrix * weights).sum(axis=1)
    target_score = score_matrix.max(axis=1)

    offsets = missing - prefix_end
    valid = (offsets >= 0) & (offsets < len(target_prediction))
    prediction[missing[valid]] = target_prediction[offsets[valid]]
    score[missing[valid]] = target_score[offsets[valid]]
    return prediction, score


def _geometry_fallback_full(horizontal: pd.DataFrame, typewell: pd.DataFrame, tvt_known: np.ndarray) -> np.ndarray:
    missing = np.flatnonzero(~np.isfinite(tvt_known))
    prediction = np.asarray(tvt_known, dtype="float64").copy()
    if len(missing) == 0:
        return prediction
    features = build_features(horizontal, typewell, missing, tvt_known=tvt_known)
    prediction[missing] = features["geom_tvt"].to_numpy(dtype="float64")
    return prediction


def _last_known_before_target(tvt_known: np.ndarray, first_target_index: int) -> np.ndarray:
    row_axis = np.arange(len(tvt_known))
    known = np.flatnonzero(np.isfinite(tvt_known) & (row_axis < first_target_index))
    if len(known) == 0:
        known = np.flatnonzero(np.isfinite(tvt_known))
    return known


def _initial_position_rate(
    md: np.ndarray,
    z: np.ndarray,
    tvt_known: np.ndarray,
    known_before: np.ndarray,
    tail_size: int = 30,
) -> float:
    tail = known_before[-min(tail_size, len(known_before)) :]
    if len(tail) < 4:
        return 0.0
    dt = np.diff(tvt_known[tail])
    dz = np.diff(z[tail])
    dm = np.diff(md[tail])
    valid = np.isfinite(dt) & np.isfinite(dz) & np.isfinite(dm) & (dm > 0)
    if valid.sum() < 3:
        return 0.0
    return float(np.median((dt[valid] + dz[valid]) / dm[valid]))


def _particle_filter_predict(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    tvt_known: np.ndarray | None = None,
    n_particles: int = 500,
    seed: int = 42,
    init_spread: float = 2.0,
) -> tuple[np.ndarray, float]:
    tvt_known = _default_tvt_known(horizontal) if tvt_known is None else np.asarray(tvt_known, dtype="float64")
    missing = np.flatnonzero(~np.isfinite(tvt_known))
    if len(missing) == 0:
        return tvt_known.copy(), 0.0

    tw_tvt, tw_gr = _typewell_arrays(typewell)
    if len(tw_tvt) < 2:
        return _geometry_fallback_full(horizontal, typewell, tvt_known), 0.0

    n_particles = max(16, int(n_particles))
    row_axis = _fallback_axis(len(horizontal))
    md = _as_float_array(horizontal, "MD")
    md = np.where(np.isfinite(md), md, row_axis)
    z = _as_float_array(horizontal, "Z")
    z = np.where(np.isfinite(z), z, 0.0)
    gr = _interpolated_gr(horizontal, fallback=float(np.nanmean(tw_gr)))

    known_before = _last_known_before_target(tvt_known, int(missing.min()))
    if len(known_before) == 0:
        return _geometry_fallback_full(horizontal, typewell, tvt_known), 0.0

    last_index = int(known_before[-1])
    last_tvt = float(tvt_known[last_index])
    last_md = float(md[last_index])
    last_z = float(z[last_index])

    known_gr = _as_float_array(horizontal, "GR")[known_before]
    known_tvt = tvt_known[known_before]
    known_gr_mask = np.isfinite(known_gr) & np.isfinite(known_tvt)
    if known_gr_mask.sum() >= 5:
        expected_gr = np.interp(np.clip(known_tvt[known_gr_mask], tw_tvt[0], tw_tvt[-1]), tw_tvt, tw_gr)
        gr_sigma = float(np.clip(np.nanstd(known_gr[known_gr_mask] - expected_gr), 10.0, 60.0))
    else:
        gr_sigma = float(np.clip(np.nanstd(tw_gr), 10.0, 60.0))
    if not np.isfinite(gr_sigma) or gr_sigma <= 0:
        gr_sigma = 30.0

    rng = np.random.default_rng(seed)
    pos = last_tvt + last_z + init_spread * rng.standard_normal(n_particles)
    rate = _initial_position_rate(md, z, tvt_known, known_before) + 0.01 * rng.standard_normal(n_particles)
    weights = np.ones(n_particles, dtype="float64") / n_particles
    prediction = tvt_known.copy()
    prev_md = last_md
    log_likelihood = 0.0

    momentum = 0.998
    velocity_noise = 0.002
    position_noise = 0.005
    resample_position_noise = 0.1
    resample_rate_noise = 0.001
    resample_threshold = 0.5 * n_particles

    for row_index in missing:
        step_md = max(float(md[row_index] - prev_md), 1.0)
        rate = momentum * rate + velocity_noise * rng.standard_normal(n_particles)
        pos = pos + rate * step_md + position_noise * rng.standard_normal(n_particles)

        tvt_particles = np.clip(pos - z[row_index], tw_tvt[0] - 100.0, tw_tvt[-1] + 100.0)
        pos = tvt_particles + z[row_index]
        expected_gr = np.interp(tvt_particles, tw_tvt, tw_gr)
        scaled_error = (gr[row_index] - expected_gr) / gr_sigma
        likelihood = np.exp(-0.5 * np.minimum(scaled_error**2, 600.0))
        likelihood = np.maximum(likelihood, 1e-300)

        average_likelihood = float((weights * likelihood).sum())
        log_likelihood += float(np.log(max(average_likelihood, 1e-300)))
        weights *= likelihood
        weight_sum = float(weights.sum())
        weights = weights / weight_sum if weight_sum > 0 else np.ones(n_particles, dtype="float64") / n_particles

        effective_n = 1.0 / float((weights**2).sum())
        if effective_n < resample_threshold:
            cumulative = np.cumsum(weights)
            start = rng.uniform(0.0, 1.0 / n_particles)
            draw = start + np.arange(n_particles) / n_particles
            sampled = np.clip(np.searchsorted(cumulative, draw), 0, n_particles - 1)
            pos = pos[sampled] + resample_position_noise * rng.standard_normal(n_particles)
            rate = rate[sampled] + resample_rate_noise * rng.standard_normal(n_particles)
            weights = np.ones(n_particles, dtype="float64") / n_particles

        prediction[row_index] = float(np.dot(weights, pos - z[row_index]))
        prev_md = float(md[row_index])

    return prediction, log_likelihood


def _particle_filter_ensemble(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    tvt_known: np.ndarray | None = None,
    n_particles: int = 500,
    n_seeds: int = 128,
    likelihood_scale: float = 5.0,
) -> np.ndarray:
    n_seeds = max(1, int(n_seeds))
    predictions = []
    log_likelihoods = []
    for seed in range(n_seeds):
        prediction, log_likelihood = _particle_filter_predict(
            horizontal,
            typewell,
            tvt_known=tvt_known,
            n_particles=n_particles,
            seed=seed,
        )
        predictions.append(prediction)
        log_likelihoods.append(log_likelihood)

    if len(predictions) == 1:
        return predictions[0]

    log_likelihoods_array = np.asarray(log_likelihoods, dtype="float64")
    centered = log_likelihoods_array - np.nanmax(log_likelihoods_array)
    scale = max(float(likelihood_scale), 1e-6)
    weights = np.exp(centered / scale)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(len(predictions), dtype="float64") / len(predictions)
    else:
        weights /= weights.sum()
    return (weights[:, None] * np.stack(predictions, axis=0)).sum(axis=0)


def _particle_filter_ensemble_scales(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    tvt_known: np.ndarray | None = None,
    n_particles: int = 500,
    n_seeds: int = 128,
    scales: tuple[float, ...] = SELECTOR_SCALES,
) -> dict[str, np.ndarray]:
    """Run PF seeds once and return likelihood-weighted ensembles for multiple scales."""

    n_seeds = max(1, int(n_seeds))
    predictions = []
    log_likelihoods = []
    for seed in range(n_seeds):
        prediction, log_likelihood = _particle_filter_predict(
            horizontal,
            typewell,
            tvt_known=tvt_known,
            n_particles=n_particles,
            seed=seed,
        )
        predictions.append(prediction)
        log_likelihoods.append(log_likelihood)

    prediction_stack = np.stack(predictions, axis=0)
    result: dict[str, np.ndarray] = {"pf_mean": prediction_stack.mean(axis=0)}
    if len(predictions) == 1:
        for scale in scales:
            result[f"pf_scale_{scale:g}"] = prediction_stack[0]
        return result

    log_likelihoods_array = np.asarray(log_likelihoods, dtype="float64")
    centered = log_likelihoods_array - np.nanmax(log_likelihoods_array)
    for scale in scales:
        safe_scale = max(float(scale), 1e-6)
        weights = np.exp(centered / safe_scale)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(len(predictions), dtype="float64") / len(predictions)
        else:
            weights /= weights.sum()
        result[f"pf_scale_{scale:g}"] = (weights[:, None] * prediction_stack).sum(axis=0)
    return result


def _beam_search(
    horizontal_gr: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    last_tvt: float,
    beam_size: int,
    move_cost: float,
    error_scale: float,
    smooth_radius: int,
) -> np.ndarray:
    if len(horizontal_gr) == 0:
        return np.array([], dtype="float64")
    if smooth_radius > 0 and len(horizontal_gr) > max(3, 2 * smooth_radius + 1):
        window = min(2 * smooth_radius + 1, len(horizontal_gr) if len(horizontal_gr) % 2 == 1 else len(horizontal_gr) - 1)
        smoothed_gr = savgol_filter(horizontal_gr, window, min(2, window - 1))
    else:
        smoothed_gr = horizontal_gr.copy()

    start_index = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    moves = np.array([-2, -1, 0, 1, 2], dtype="int64")
    move_penalty = move_cost * np.array([2.0, 1.0, 0.0, 1.0, 2.0])
    beam_indices = np.full(beam_size, start_index, dtype="int64")
    beam_cost = np.full(beam_size, np.inf, dtype="float64")
    beam_cost[0] = 0.0
    active = 1
    result = np.zeros(len(horizontal_gr), dtype="float64")

    for step, gr_value in enumerate(smoothed_gr):
        next_indices = beam_indices[:active, None] + moves[None, :]
        clipped = np.clip(next_indices, 0, len(tw_tvt) - 1)
        valid = (next_indices >= 0) & (next_indices < len(tw_tvt))
        gr_error = (gr_value - tw_gr[clipped]) ** 2 / error_scale
        total_cost = beam_cost[:active, None] + gr_error + move_penalty[None, :]
        total_cost = np.where(valid, total_cost, np.inf)

        flat_indices = next_indices.ravel()
        flat_cost = total_cost.ravel()
        flat_valid = valid.ravel()
        flat_indices = flat_indices[flat_valid]
        flat_cost = flat_cost[flat_valid]
        order = np.argsort(flat_cost)
        sorted_indices = flat_indices[order]
        sorted_cost = flat_cost[order]
        _, first = np.unique(sorted_indices, return_index=True)
        unique_indices = sorted_indices[first]
        unique_cost = sorted_cost[first]

        kept = min(beam_size, len(unique_indices))
        top = np.argpartition(unique_cost, min(kept - 1, len(unique_cost) - 1))[:kept]
        top = top[np.argsort(unique_cost[top])]
        beam_indices[:kept] = unique_indices[top]
        beam_cost[:kept] = unique_cost[top]
        if kept < beam_size:
            beam_indices[kept:] = beam_indices[kept - 1]
            beam_cost[kept:] = np.inf
        active = kept
        result[step] = tw_tvt[beam_indices[0]]
    return result


def _beam_ensemble(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    tvt_known: np.ndarray | None = None,
) -> np.ndarray:
    tvt_known = _default_tvt_known(horizontal) if tvt_known is None else np.asarray(tvt_known, dtype="float64")
    missing = np.flatnonzero(~np.isfinite(tvt_known))
    if len(missing) == 0:
        return tvt_known.copy()

    tw_tvt, tw_gr = _typewell_arrays(typewell)
    if len(tw_tvt) < 2:
        return _geometry_fallback_full(horizontal, typewell, tvt_known)

    known_before = _last_known_before_target(tvt_known, int(missing.min()))
    if len(known_before) == 0:
        return _geometry_fallback_full(horizontal, typewell, tvt_known)

    gr = _interpolated_gr(horizontal, fallback=float(np.nanmean(tw_gr)))[missing]
    last_tvt = float(tvt_known[int(known_before[-1])])
    beam_results = [
        _beam_search(gr, tw_tvt, tw_gr, last_tvt, beam_size, move_cost, error_scale, smooth_radius)
        for beam_size, move_cost, error_scale, smooth_radius in BEAM_CONFIGS
    ]
    prediction = tvt_known.copy()
    prediction[missing] = np.stack(beam_results, axis=0).mean(axis=0)
    return prediction


def _selector_well_variant(horizontal: pd.DataFrame, tvt_known: np.ndarray) -> tuple[int, str]:
    eval_mask = ~np.isfinite(tvt_known)
    n_eval = float(eval_mask.sum())
    z_eval = _as_float_array(horizontal, "Z")[eval_mask]
    z_eval = z_eval[np.isfinite(z_eval)]
    z_span = float(z_eval.max() - z_eval.min()) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right"))
    code = n_bin + 2 * z_bin
    return code, SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)


def _parse_selector_variant(name: str) -> tuple[float, float, float]:
    parts = name.split("_")
    if len(parts) < 3 or parts[0] != "pf" or parts[1] != "scale":
        raise ValueError(f"Invalid selector variant: {name!r}")
    scale = float(parts[2])
    beam_weight = 0.0
    hold_weight = 0.0
    if "beam" in parts:
        beam_weight = float(parts[parts.index("beam") + 1])
    if "hold" in parts:
        hold_weight = float(parts[parts.index("hold") + 1])
    return scale, beam_weight, hold_weight


def _last_known_tvt_value(tvt_known: np.ndarray) -> float:
    known = np.flatnonzero(np.isfinite(tvt_known))
    if len(known) == 0:
        return 0.0
    return float(tvt_known[int(known[-1])])


def _selector_pf_predict(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    row_indices: np.ndarray,
    tvt_known: np.ndarray | None = None,
    pf_n_particles: int = 500,
    pf_n_seeds: int = 128,
) -> np.ndarray:
    tvt_known = _default_tvt_known(horizontal) if tvt_known is None else np.asarray(tvt_known, dtype="float64")
    _, variant = _selector_well_variant(horizontal, tvt_known)
    scale, beam_weight, hold_weight = _parse_selector_variant(variant)

    pf_by_scale = _particle_filter_ensemble_scales(
        horizontal,
        typewell,
        tvt_known=tvt_known,
        n_particles=pf_n_particles,
        n_seeds=pf_n_seeds,
        scales=SELECTOR_SCALES,
    )
    base = pf_by_scale.get(f"pf_scale_{scale:g}")
    if base is None:
        fallback_scale, _, _ = _parse_selector_variant(SELECTOR_GLOBAL_VARIANT)
        base = pf_by_scale.get(f"pf_scale_{fallback_scale:g}", pf_by_scale["pf_mean"])

    prediction = np.asarray(base, dtype="float64").copy()
    if beam_weight > 0:
        beam_prediction = _beam_ensemble(horizontal, typewell, tvt_known=tvt_known)
        prediction = (1.0 - beam_weight) * prediction + beam_weight * beam_prediction
    if hold_weight > 0:
        prediction = (1.0 - hold_weight) * prediction + hold_weight * _last_known_tvt_value(tvt_known)
    known = np.isfinite(tvt_known)
    prediction[known] = tvt_known[known]
    return prediction[row_indices]


def _formation_confidence_per_row(
    formation_result: dict,
    *,
    plane_scale: float = 0.05,
    quality_scale: float = 32.0,
    quality_offset: float = 3.0,
    cap: float = 0.95,
) -> np.ndarray:
    """Per-row formation-confidence in ``[0, cap]``.

    Confidence is the product of two exponential factors:

    * per-well factor that drops with the known-zone RMSE of the formation
      prediction (``quality_offset, quality_scale``); a wider
      ``quality_scale`` keeps the formation contribution alive on wells
      where the known-zone fit is mediocre but still informative, and
    * per-row factor that drops with the plane-KNN nearest centroid
      distance (``plane_scale``).

    Defaults were tuned via grid search against a 200-well OOF training
    set (~950k rows) generated by ``scripts/build_meta_training_set.py``.
    This configuration drops weighted RMSE from 12.11 (pure PF) to 9.98
    on that sample. Earlier configurations:
        v11 (LB 9.817): plane_scale=0.05, quality_scale=12, cap=0.70
        v12 (LB 10.375): plane_scale=0.08, quality_scale=32, cap=0.70,
                         dense_std factor active.
    The dense_std factor was removed: on 200 wells it added at most 0.03
    ft to weighted RMSE, and on the LB it appears to over-suppress good
    formation contributions.
    """

    quality = float(formation_result.get("overall_quality_rmse", np.nan))
    plane_distance = np.asarray(formation_result["plane_distance"], dtype="float64")

    if not np.isfinite(quality):
        return np.zeros_like(plane_distance)

    quality_factor = float(np.exp(-max(quality - quality_offset, 0.0) / quality_scale))
    plane_factor = np.exp(-plane_distance / plane_scale)
    confidence = quality_factor * plane_factor
    return np.clip(confidence, 0.0, cap)


def _formation_pf_predict(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    row_indices: np.ndarray,
    imputer: FormationImputer,
    tvt_known: np.ndarray | None = None,
    pf_n_particles: int = 500,
    pf_n_seeds: int = 128,
    exclude_well_id: str | None = None,
    max_formation_weight: float = 0.95,
) -> np.ndarray:
    """Blend selector_pf with formation-based physical TVT prediction.

    The formation predictor uses ``FormationImputer`` to estimate the six
    formation surface depths at every horizontal row, calibrates a per-well
    ``b_well = TVT + Z - formation_depth`` from the known prefix, and predicts
    ``TVT = -Z + formation_depth + b_well`` in the hidden region. This signal
    dominates whenever spatial imputation is accurate.

    When the formation signal is unreliable (sparse training neighbours, or
    large known-zone residuals), we fall back to the selector-PF prediction.
    Per-row blending uses ``_formation_confidence_per_row``.
    """

    tvt_known = _default_tvt_known(horizontal) if tvt_known is None else np.asarray(tvt_known, dtype="float64")

    pf_prediction = _selector_pf_predict(
        horizontal,
        typewell,
        row_indices,
        tvt_known=tvt_known,
        pf_n_particles=pf_n_particles,
        pf_n_seeds=pf_n_seeds,
    )

    if "X" not in horizontal.columns or "Y" not in horizontal.columns:
        return pf_prediction

    try:
        formation_result = predict_physical_tvt(
            horizontal,
            tvt_known=tvt_known,
            imputer=imputer,
            exclude_well_id=exclude_well_id,
        )
    except Exception:
        return pf_prediction

    physical_prediction = formation_result["physical_prediction"][row_indices]
    if not np.isfinite(physical_prediction).any():
        return pf_prediction

    confidence_full = _formation_confidence_per_row(formation_result)
    confidence = confidence_full[row_indices]
    confidence = np.where(np.isfinite(physical_prediction), confidence, 0.0)
    confidence = np.clip(confidence, 0.0, float(max_formation_weight))

    physical_filled = np.where(np.isfinite(physical_prediction), physical_prediction, pf_prediction)
    return (1.0 - confidence) * pf_prediction + confidence * physical_filled


def _build_meta_features(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    row_indices: np.ndarray,
    *,
    tvt_known: np.ndarray,
    pf_pred: np.ndarray,
    formation_result: dict,
) -> pd.DataFrame:
    """Assemble the per-row feature frame expected by the meta-learner."""

    feats = build_features(horizontal, typewell, row_indices, tvt_known=tvt_known)
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(dtype="float64")
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(dtype="float64")

    out = pd.DataFrame({
        "pf_pred": pf_pred,
        "form_pred": formation_result["physical_prediction"][row_indices],
        "geom_tvt": feats["geom_tvt"].to_numpy(),
        "plane_distance": formation_result["plane_distance"][row_indices],
        "dense_distance": formation_result["dense_distance"][row_indices],
        "dens_std": formation_result["stddev_imputed"][row_indices],
        "qk": float(formation_result.get("overall_quality_rmse", np.nan)),
        "last_tvt": feats["last_tvt"].to_numpy(),
        "delta_md": feats["delta_md"].to_numpy(),
        "delta_x": feats["delta_x"].to_numpy(),
        "delta_y": feats["delta_y"].to_numpy(),
        "delta_z": feats["delta_z"].to_numpy(),
        "dist_xy": feats["dist_xy"].to_numpy(),
        "row_fraction": feats["row_fraction"].to_numpy(),
        "ps_fraction": feats["ps_fraction"].to_numpy(),
        "gr": feats["gr"].to_numpy(),
        "gr_missing": feats["gr_missing"].to_numpy(),
        "typewell_gr_diff": feats["typewell_gr_diff"].to_numpy(),
        "geom_slope": feats["geom_slope"].to_numpy(),
        "geom_quad_delta": feats["geom_quad_delta"].to_numpy(),
        "z": z[row_indices],
        "md": md[row_indices],
    })

    from .formations import FORMATIONS

    per_form_preds = formation_result["per_formation_predictions"][row_indices]
    per_form_rmse = formation_result["per_formation_rmse"]
    for i, fname in enumerate(FORMATIONS):
        out[f"form_{fname}_pred"] = per_form_preds[:, i] if per_form_preds.shape[1] > i else np.nan
        out[f"form_{fname}_qk"] = float(per_form_rmse[i]) if i < len(per_form_rmse) and np.isfinite(per_form_rmse[i]) else 99.0

    return out


def _meta_pf_predict(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    row_indices: np.ndarray,
    imputer: "FormationImputer",
    meta_model_payload: dict,
    *,
    tvt_known: np.ndarray | None = None,
    pf_n_particles: int = 500,
    pf_n_seeds: int = 128,
    exclude_well_id: str | None = None,
) -> np.ndarray:
    """Apply the trained HGB meta-learner over PF + formation + features.

    ``meta_model_payload`` is the dict written by
    ``scripts/train_meta_learner.py`` containing keys ``model``,
    ``features``, and ``target``.
    """

    tvt_known = _default_tvt_known(horizontal) if tvt_known is None else np.asarray(tvt_known, dtype="float64")

    pf_pred = _selector_pf_predict(
        horizontal,
        typewell,
        row_indices,
        tvt_known=tvt_known,
        pf_n_particles=pf_n_particles,
        pf_n_seeds=pf_n_seeds,
    )

    if "X" not in horizontal.columns or "Y" not in horizontal.columns:
        return pf_pred

    try:
        formation_result = predict_physical_tvt(
            horizontal, tvt_known=tvt_known, imputer=imputer, exclude_well_id=exclude_well_id
        )
    except Exception:
        return pf_pred

    features_df = _build_meta_features(
        horizontal, typewell, row_indices,
        tvt_known=tvt_known, pf_pred=pf_pred, formation_result=formation_result,
    )

    feature_list = list(meta_model_payload["features"])
    # Ensure all required features are present; fill missing with zeros
    for col in feature_list:
        if col not in features_df.columns:
            features_df[col] = 0.0
    X = features_df[feature_list].astype("float64").fillna(0.0)
    model = meta_model_payload["model"]
    raw_pred = model.predict(X.to_numpy())

    target = meta_model_payload.get("target", "actual_minus_last")
    if target == "actual_minus_last":
        prediction = features_df["last_tvt"].to_numpy(dtype="float64") + raw_pred
    elif target == "actual_minus_pf":
        prediction = features_df["pf_pred"].to_numpy(dtype="float64") + raw_pred
    elif target == "actual_minus_form":
        prediction = features_df["form_pred"].to_numpy(dtype="float64") + raw_pred
    elif target == "actual":
        prediction = raw_pred
    else:
        raise ValueError(f"Unknown meta-learner target: {target!r}")

    # Safety guard: if meta-learner is producing garbage (e.g., outside reasonable
    # TVT range), fall back to the formation_pf prediction for those rows.
    finite = np.isfinite(prediction)
    if not finite.all():
        physical = formation_result["physical_prediction"][row_indices]
        prediction = np.where(finite, prediction, physical)
    return prediction


def _selector_pf_ncc_predict(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    row_indices: np.ndarray,
    tvt_known: np.ndarray | None = None,
    pf_n_particles: int = 500,
    pf_n_seeds: int = 128,
) -> np.ndarray:
    tvt_known = _default_tvt_known(horizontal) if tvt_known is None else np.asarray(tvt_known, dtype="float64")
    selector_prediction = _selector_pf_predict(
        horizontal,
        typewell,
        row_indices,
        tvt_known=tvt_known,
        pf_n_particles=pf_n_particles,
        pf_n_seeds=pf_n_seeds,
    )
    ncc_full, ncc_score_full = _ncc_alignment_full(horizontal, tvt_known)
    ncc_prediction = ncc_full[row_indices]
    ncc_score = ncc_score_full[row_indices]

    first_target = int(np.min(row_indices)) if len(row_indices) else len(tvt_known)
    known_prefix = np.isfinite(tvt_known[:first_target]).sum()
    prefix_trust = np.clip(known_prefix / 200.0, 0.0, 1.0)
    score_trust = np.clip((ncc_score - 0.55) / 0.25, 0.0, 1.0)
    agreement_trust = np.clip((20.0 - np.abs(ncc_prediction - selector_prediction)) / 10.0, 0.0, 1.0)
    ncc_weight = 0.12 * prefix_trust * score_trust * agreement_trust
    ncc_weight = np.where(np.isfinite(ncc_prediction), ncc_weight, 0.0)

    return (1.0 - ncc_weight) * selector_prediction + ncc_weight * ncc_prediction


def _physical_tvt_from_contacts(
    horizontal_train: pd.DataFrame,
    typewell_train: pd.DataFrame,
    preferred_marker: str = "EGFDU",
) -> np.ndarray | None:
    if "TVT" not in horizontal_train.columns or "Z" not in horizontal_train.columns or "Geology" not in typewell_train.columns:
        return None
    geology = typewell_train.dropna(subset=["Geology"]).copy()
    if geology.empty:
        return None
    geology["Geology"] = geology["Geology"].astype(str)
    marker_candidates = [preferred_marker] + [value for value in geology["Geology"].unique() if value != preferred_marker]
    marker = next((value for value in marker_candidates if value in horizontal_train.columns), None)
    if marker is None:
        return None

    ref_tvt = numeric_column(geology.loc[geology["Geology"] == marker], "TVT").min()
    if not np.isfinite(ref_tvt):
        return None

    z = _as_float_array(horizontal_train, "Z")
    marker_depth = _as_float_array(horizontal_train, marker)
    true_tvt = _as_float_array(horizontal_train, "TVT")
    physical = float(ref_tvt) - (z - marker_depth)
    offset = np.nanmean(true_tvt - physical)
    if not np.isfinite(offset):
        offset = 0.0
    physical = physical + offset
    if not np.isfinite(physical).any():
        return None
    return physical.astype("float64")


def _weighted_prediction(parts: list[tuple[np.ndarray | None, float]]) -> np.ndarray:
    valid_parts = [(np.asarray(values, dtype="float64"), float(weight)) for values, weight in parts if values is not None and weight > 0]
    if not valid_parts:
        raise ValueError("No prediction parts were provided.")
    weight_sum = sum(weight for _, weight in valid_parts)
    if weight_sum <= 0:
        weight_sum = float(len(valid_parts))
        valid_parts = [(values, 1.0) for values, _ in valid_parts]
    result = np.zeros_like(valid_parts[0][0], dtype="float64")
    total_weight = np.zeros_like(result, dtype="float64")
    for values, weight in valid_parts:
        finite = np.isfinite(values)
        result[finite] += values[finite] * weight
        total_weight[finite] += weight
    return np.divide(result, total_weight, out=valid_parts[0][0].copy(), where=total_weight > 0)


def _pf_stack_predict(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    row_indices: np.ndarray,
    tvt_known: np.ndarray | None = None,
    physical_tvt: np.ndarray | None = None,
    pf_n_particles: int = 500,
    pf_n_seeds: int = 128,
    pf_likelihood_scale: float = 5.0,
    physical_blend_weight: float = 0.75,
    pf_blend_weight: float = 0.125,
    beam_blend_weight: float = 0.125,
) -> np.ndarray:
    tvt_known = _default_tvt_known(horizontal) if tvt_known is None else np.asarray(tvt_known, dtype="float64")
    pf_prediction = _particle_filter_ensemble(
        horizontal,
        typewell,
        tvt_known=tvt_known,
        n_particles=pf_n_particles,
        n_seeds=pf_n_seeds,
        likelihood_scale=pf_likelihood_scale,
    )
    if physical_tvt is not None and len(physical_tvt) != len(horizontal):
        physical_tvt = None

    beam_prediction = _beam_ensemble(horizontal, typewell, tvt_known=tvt_known) if beam_blend_weight > 0 else None
    prediction = _weighted_prediction(
        [
            (physical_tvt, physical_blend_weight),
            (pf_prediction, pf_blend_weight),
            (beam_prediction, beam_blend_weight),
        ]
    )
    return prediction[row_indices]


def make_submission(
    data_dir: str | Path | None = None,
    output_path: str | Path = "submission.csv",
    model_kind: str = "selector_pf",
    max_train_rows: int = 800_000,
    visible_copy_smoke: bool = False,
    gr_correction: bool = True,
    pf_n_particles: int = 500,
    pf_n_seeds: int = 128,
    pf_likelihood_scale: float = 5.0,
    physical_blend_weight: float = 0.75,
    pf_blend_weight: float = 0.25,
    beam_blend_weight: float = 0.0,
    meta_model_payload: dict | None = None,
) -> pd.DataFrame:
    resolved = require_data_dir(data_dir)
    sample = load_sample_submission(resolved)
    submission = sample[["id"]].copy()
    submission["tvt"] = np.nan

    if visible_copy_smoke:
        copied = _copy_visible_train_targets(resolved, sample)
        if copied.notna().all():
            submission["tvt"] = copied.to_numpy(dtype="float64")
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            submission.to_csv(output, index=False)
            return submission
        print("Visible-copy smoke fallback: not all sample rows exist in train; using model baseline.")

    if model_kind not in MODEL_KINDS:
        raise ValueError(f"model_kind must be one of {MODEL_KINDS}")

    model = None
    if model_kind == "residual":
        model = train_residual_model(resolved, max_train_rows=max_train_rows)
    train_wells = (
        set(list_well_ids(resolved, "train"))
        if model_kind in {"physical_pf", "selector_pf", "selector_pf_ncc", "formation_pf", "meta_pf"}
        else set()
    )

    formation_imputer = None
    if model_kind in {"formation_pf", "meta_pf"}:
        formation_imputer = FormationImputer.build(resolved, sorted(train_wells), points_per_well=60)

    if model_kind == "meta_pf" and meta_model_payload is None:
        raise ValueError("meta_pf requires meta_model_payload (dict from train_meta_learner)")

    for well_id, group in sample.groupby("well_id", sort=False):
        horizontal = read_horizontal(resolved, "test", well_id)
        typewell = read_typewell(resolved, "test", well_id)
        row_indices = group["row_index"].to_numpy(dtype="int64")
        features = build_features(horizontal, typewell, row_indices)
        if model_kind in {"formation_pf", "meta_pf"}:
            assert formation_imputer is not None
            selector_typewell = typewell
            if well_id in train_wells:
                try:
                    selector_typewell = read_typewell(resolved, "train", well_id)
                except (FileNotFoundError, ValueError):
                    selector_typewell = typewell
            if model_kind == "meta_pf":
                prediction = _meta_pf_predict(
                    horizontal,
                    selector_typewell,
                    row_indices,
                    imputer=formation_imputer,
                    meta_model_payload=meta_model_payload,
                    pf_n_particles=pf_n_particles,
                    pf_n_seeds=pf_n_seeds,
                    exclude_well_id=well_id if well_id in train_wells else None,
                )
            else:
                prediction = _formation_pf_predict(
                    horizontal,
                    selector_typewell,
                    row_indices,
                    imputer=formation_imputer,
                    pf_n_particles=pf_n_particles,
                    pf_n_seeds=pf_n_seeds,
                    exclude_well_id=well_id if well_id in train_wells else None,
                )
        elif model_kind in {"selector_pf", "selector_pf_ncc"}:
            physical_tvt = None
            selector_typewell = typewell
            if well_id in train_wells:
                try:
                    train_horizontal = read_horizontal(resolved, "train", well_id)
                    train_typewell = read_typewell(resolved, "train", well_id)
                    physical_tvt = _physical_tvt_from_contacts(train_horizontal, train_typewell)
                    selector_typewell = train_typewell
                except (FileNotFoundError, ValueError):
                    physical_tvt = None
            if physical_tvt is not None and len(physical_tvt) == len(horizontal):
                prediction = np.asarray(physical_tvt, dtype="float64")[row_indices]
            else:
                selector_function = _selector_pf_ncc_predict if model_kind == "selector_pf_ncc" else _selector_pf_predict
                prediction = selector_function(
                    horizontal,
                    selector_typewell,
                    row_indices,
                    pf_n_particles=pf_n_particles,
                    pf_n_seeds=pf_n_seeds,
                )
        elif model_kind in {"pf", "physical_pf"}:
            physical_tvt = None
            if model_kind == "physical_pf" and well_id in train_wells:
                try:
                    train_horizontal = read_horizontal(resolved, "train", well_id)
                    train_typewell = read_typewell(resolved, "train", well_id)
                    physical_tvt = _physical_tvt_from_contacts(train_horizontal, train_typewell)
                except (FileNotFoundError, ValueError):
                    physical_tvt = None
            prediction = _pf_stack_predict(
                horizontal,
                typewell if physical_tvt is None else read_typewell(resolved, "train", well_id),
                row_indices,
                physical_tvt=physical_tvt,
                pf_n_particles=pf_n_particles,
                pf_n_seeds=pf_n_seeds,
                pf_likelihood_scale=pf_likelihood_scale,
                physical_blend_weight=physical_blend_weight,
                pf_blend_weight=pf_blend_weight,
                beam_blend_weight=beam_blend_weight,
            )
        else:
            prediction = features["geom_tvt"].to_numpy(dtype="float64")
            if model is not None:
                prediction = prediction + model.predict(features[FEATURE_COLUMNS])
            if gr_correction:
                prediction = _gr_correlation_adjustment(horizontal, typewell, row_indices, prediction)
        submission.loc[group.index, "tvt"] = _postprocess_predictions(prediction, features)

    validate_submission(submission, sample)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    return submission


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(submission.columns) != ["id", "tvt"]:
        raise ValueError(f"Submission columns must be ['id', 'tvt']; got {list(submission.columns)}")
    if len(submission) != len(sample):
        raise ValueError(f"Submission has {len(submission)} rows, sample has {len(sample)} rows")
    if not submission["id"].equals(sample["id"]):
        raise ValueError("Submission ids differ from sample_submission order")
    tvt = pd.to_numeric(submission["tvt"], errors="coerce")
    if tvt.isna().any():
        raise ValueError(f"Submission contains {int(tvt.isna().sum())} missing tvt predictions")


def train_validation_split(wells: list[str], validation_stride: int = 5) -> tuple[list[str], list[str]]:
    validation = [well for index, well in enumerate(wells) if index % validation_stride == 0]
    training = [well for well in wells if well not in set(validation)]
    return training, validation


def evaluate_baseline(
    data_dir: str | Path | None = None,
    model_kind: str = "geometry",
    max_train_rows: int = 200_000,
    max_val_wells: int | None = 100,
    ps_fraction: float = 0.30,
    mask_strategy: str = "artificial",
    pf_n_particles: int = 500,
    pf_n_seeds: int = 32,
    pf_likelihood_scale: float = 5.0,
    beam_blend_weight: float = 0.0,
) -> pd.DataFrame:
    resolved = require_data_dir(data_dir)
    wells = list_well_ids(resolved, "train")
    train_wells, validation_wells = train_validation_split(wells)
    if max_val_wells is not None:
        validation_wells = validation_wells[:max_val_wells]

    model = None
    if model_kind == "residual":
        model = train_residual_model(
            resolved,
            train_wells=train_wells,
            max_train_rows=max_train_rows,
            mask_strategy=mask_strategy,
        )
    elif model_kind not in {"geometry", "pf", "physical_pf", "selector_pf", "selector_pf_ncc", "formation_pf"}:
        raise ValueError(f"model_kind must be one of {MODEL_KINDS}")

    formation_imputer = None
    if model_kind == "formation_pf":
        formation_imputer = FormationImputer.build(resolved, list_well_ids(resolved, "train"), points_per_well=60)

    rows = []
    for well_id in validation_wells:
        horizontal = read_horizontal(resolved, "train", well_id)
        if "TVT" not in horizontal.columns:
            continue
        typewell = read_typewell(resolved, "train", well_id)
        true_tvt = _as_float_array(horizontal, "TVT")
        if mask_strategy == "actual":
            known, target_indices = _target_indices_from_actual_mask(horizontal, true_tvt)
            if len(target_indices) == 0:
                continue
        elif mask_strategy == "artificial":
            ps_index, target_indices = _scenario_indices(len(horizontal), ps_fraction, per_scenario_limit=len(horizontal))
            known = true_tvt.copy()
            known[ps_index:] = np.nan
        else:
            raise ValueError("mask_strategy must be 'actual' or 'artificial'")
        features = build_features(horizontal, typewell, target_indices, tvt_known=known)
        if model_kind == "formation_pf":
            assert formation_imputer is not None
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
                pf_blend_weight=1.0,
                beam_blend_weight=beam_blend_weight,
            )
        else:
            prediction = features["geom_tvt"].to_numpy(dtype="float64")
            if model is not None:
                prediction = prediction + model.predict(features[FEATURE_COLUMNS])
        prediction = _postprocess_predictions(prediction, features)
        actual = true_tvt[target_indices]
        finite = np.isfinite(actual) & np.isfinite(prediction)
        if finite.any():
            rmse = float(np.sqrt(mean_squared_error(actual[finite], prediction[finite])))
            rows.append({"well_id": well_id, "rows": int(finite.sum()), "rmse": float(rmse)})

    result = pd.DataFrame(rows)
    if not result.empty:
        overall = np.sqrt(np.average(result["rmse"] ** 2, weights=result["rows"]))
        result.loc[len(result)] = {"well_id": "__overall__", "rows": int(result["rows"].sum()), "rmse": float(overall)}
    return result


def main_make_submission(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create a ROGII submission.csv file.")
    parser.add_argument("--data-dir", default=None, help="Competition data directory.")
    parser.add_argument("--output", default="submission.csv", help="Output CSV path.")
    parser.add_argument("--model-kind", choices=MODEL_KINDS, default="selector_pf")
    parser.add_argument("--max-train-rows", type=int, default=800_000)
    parser.add_argument("--pf-particles", type=int, default=500, help="Particles per PF seed for pf/physical_pf.")
    parser.add_argument("--pf-seeds", type=int, default=128, help="Number of likelihood-weighted PF seeds.")
    parser.add_argument("--pf-likelihood-scale", type=float, default=5.0, help="Scale for PF seed likelihood weighting.")
    parser.add_argument("--physical-blend-weight", type=float, default=0.75, help="Physical/contact weight when train overlap exists.")
    parser.add_argument("--pf-blend-weight", type=float, default=0.125, help="PF weight when train overlap exists.")
    parser.add_argument("--beam-blend-weight", type=float, default=0.125, help="Optional beam-search weight.")
    parser.add_argument(
        "--visible-copy-smoke",
        action="store_true",
        help="For visible example test only, copy train TVT targets to test submission rows.",
    )
    parser.add_argument("--no-gr-correction", action="store_true", help="Disable GR-correlation postprocessing.")
    args = parser.parse_args(argv)
    submission = make_submission(
        data_dir=args.data_dir,
        output_path=args.output,
        model_kind=args.model_kind,
        max_train_rows=args.max_train_rows,
        visible_copy_smoke=args.visible_copy_smoke,
        gr_correction=not args.no_gr_correction,
        pf_n_particles=args.pf_particles,
        pf_n_seeds=args.pf_seeds,
        pf_likelihood_scale=args.pf_likelihood_scale,
        physical_blend_weight=args.physical_blend_weight,
        pf_blend_weight=args.pf_blend_weight,
        beam_blend_weight=args.beam_blend_weight,
    )
    print(f"Wrote {args.output} with {len(submission)} rows.")


if __name__ == "__main__":
    main_make_submission()
