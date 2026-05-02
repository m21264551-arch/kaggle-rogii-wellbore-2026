"""Spatial formation-depth imputation and physical TVT prediction.

The horizontal-well CSV has six formation surface depth columns
(ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA) in the train set. Per-row,
each column gives the elevation of that geological top at the bit
location. The identity holds:

    TVT[i] + Z[i] - formation_depth[i] = b_well   (constant per well/formation)

So once we know the formation depth at row i, TVT is exactly
``-Z[i] + formation_depth[i] + b_well`` where b_well is calibrated
from the known prefix.

For test wells the formation columns are absent, so we impute them
spatially from train-well rows via a KDTree on (X, Y). For each query
point we use the K nearest training points and fit a local linear
plane (X, Y) -> formation_depth, which captures regional dip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .data import read_horizontal


FORMATIONS: tuple[str, ...] = ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")


@dataclass
class FormationImputer:
    """Spatial KNN imputer for the six formation depth columns.

    Two parallel structures:

    * ``centroids`` — one (X, Y, formation_depths) row per train well
      (median of its rows). KDTree over centroids is used to fit a
      local plane.
    * ``dense`` — a downsampled set of per-row (X, Y, formation_depths)
      points across all train wells. KDTree over dense points is used
      for inverse-distance-weighted nearest-neighbour interpolation.

    The plane fit gives a smooth regional estimate; the dense fit
    captures local detail (where neighbours are dense). The final
    estimate per query is the inverse-distance-weighted average of
    both, with the dense estimator favoured when its nearest neighbours
    are close.
    """

    centroids_xy: np.ndarray
    centroids_form: np.ndarray
    centroid_well_ids: np.ndarray
    centroids_tree: cKDTree
    centroids_scale: np.ndarray

    dense_xy: np.ndarray
    dense_form: np.ndarray
    dense_well_ids: np.ndarray
    dense_tree: cKDTree
    dense_scale: np.ndarray

    @classmethod
    def build(
        cls,
        data_dir: Path,
        well_ids: Iterable[str],
        points_per_well: int = 60,
    ) -> "FormationImputer":
        centroid_rows: list[dict] = []
        dense_xs: list[np.ndarray] = []
        dense_ys: list[np.ndarray] = []
        dense_forms: list[np.ndarray] = []
        dense_wids: list[str] = []

        for well_id in well_ids:
            try:
                df = pd.read_csv(
                    data_dir / "train" / f"{well_id}__horizontal_well.csv",
                    usecols=["X", "Y", *FORMATIONS],
                )
            except (FileNotFoundError, ValueError):
                continue
            df = df.dropna(subset=["X", "Y"])
            if df.empty:
                continue

            row: dict[str, object] = {
                "well_id": well_id,
                "x": float(df["X"].median()),
                "y": float(df["Y"].median()),
            }
            for column in FORMATIONS:
                row[column] = float(df[column].median()) if column in df.columns else np.nan
            centroid_rows.append(row)

            # Dense downsample: evenly spaced rows that have all formations.
            keep = df.dropna(subset=list(FORMATIONS))
            if keep.empty:
                continue
            step = max(1, len(keep) // points_per_well)
            picks = keep.iloc[::step]
            dense_xs.append(picks["X"].to_numpy(dtype="float64"))
            dense_ys.append(picks["Y"].to_numpy(dtype="float64"))
            dense_forms.append(picks[list(FORMATIONS)].to_numpy(dtype="float64"))
            dense_wids.extend([well_id] * len(picks))

        if not centroid_rows:
            raise ValueError("No training wells produced formation data.")

        centroid_df = pd.DataFrame(centroid_rows)
        centroids_xy = centroid_df[["x", "y"]].to_numpy(dtype="float64")
        centroids_form = centroid_df[list(FORMATIONS)].to_numpy(dtype="float64")
        centroid_well_ids = centroid_df["well_id"].to_numpy()
        centroids_scale = np.where(centroids_xy.std(axis=0) < 1e-3, 1.0, centroids_xy.std(axis=0))
        centroids_tree = cKDTree(centroids_xy / centroids_scale)

        if not dense_xs:
            raise ValueError("No training wells produced dense formation rows.")

        dense_xy = np.column_stack([np.concatenate(dense_xs), np.concatenate(dense_ys)])
        dense_form = np.concatenate(dense_forms, axis=0)
        dense_well_ids = np.array(dense_wids)
        dense_scale = np.where(dense_xy.std(axis=0) < 1e-3, 1.0, dense_xy.std(axis=0))
        dense_tree = cKDTree(dense_xy / dense_scale)

        return cls(
            centroids_xy=centroids_xy,
            centroids_form=centroids_form,
            centroid_well_ids=centroid_well_ids,
            centroids_tree=centroids_tree,
            centroids_scale=centroids_scale,
            dense_xy=dense_xy,
            dense_form=dense_form,
            dense_well_ids=dense_well_ids,
            dense_tree=dense_tree,
            dense_scale=dense_scale,
        )

    @property
    def n_centroids(self) -> int:
        return self.centroids_xy.shape[0]

    @property
    def n_dense_points(self) -> int:
        return self.dense_xy.shape[0]

    def _centroid_plane_estimate(
        self,
        xy: np.ndarray,
        exclude_well_id: str | None,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = xy / self.centroids_scale
        fetch = min(k + 5, self.n_centroids)
        distances, indices = self.centroids_tree.query(query, k=fetch, workers=-1)
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        if exclude_well_id is not None:
            mask_excl = self.centroid_well_ids[indices] == exclude_well_id
            distances = np.where(mask_excl, np.inf, distances)

        kept = min(k, fetch)
        order = np.argpartition(distances, min(kept - 1, fetch - 1), axis=1)[:, :kept]
        dk = np.take_along_axis(distances, order, axis=1)
        ik = np.take_along_axis(indices, order, axis=1)
        valid = np.isfinite(dk)
        weights = np.where(valid, 1.0 / (dk + 1e-3), 0.0)
        xn = self.centroids_xy[ik, 0]
        yn = self.centroids_xy[ik, 1]
        fn = self.centroids_form[ik]

        # weighted least squares plane fit: f ≈ a*X + b*Y + c
        wx = weights * xn
        wy = weights * yn
        A = np.zeros((len(query), 3, 3))
        A[:, 0, 0] = (wx * xn).sum(axis=1)
        A[:, 0, 1] = (wx * yn).sum(axis=1)
        A[:, 0, 2] = wx.sum(axis=1)
        A[:, 1, 0] = A[:, 0, 1]
        A[:, 1, 1] = (wy * yn).sum(axis=1)
        A[:, 1, 2] = wy.sum(axis=1)
        A[:, 2, 0] = A[:, 0, 2]
        A[:, 2, 1] = A[:, 1, 2]
        A[:, 2, 2] = weights.sum(axis=1)
        A[:, 0, 0] += 1e-9
        A[:, 1, 1] += 1e-9
        A[:, 2, 2] += 1e-9
        rhs = np.stack(
            [
                (wx[:, :, None] * fn).sum(axis=1),
                (wy[:, :, None] * fn).sum(axis=1),
                (weights[:, :, None] * fn).sum(axis=1),
            ],
            axis=1,
        )
        try:
            coefficients = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            coefficients = np.zeros((len(query), 3, len(FORMATIONS)))
            for r in range(len(query)):
                try:
                    coefficients[r] = np.linalg.pinv(A[r]) @ rhs[r]
                except np.linalg.LinAlgError:
                    pass

        prediction = (
            xy[:, 0, None] * coefficients[:, 0, :]
            + xy[:, 1, None] * coefficients[:, 1, :]
            + coefficients[:, 2, :]
        )
        any_valid = valid.any(axis=1)
        global_mean = self.centroids_form.mean(axis=0)
        prediction = np.where(any_valid[:, None], prediction, global_mean[None, :])
        nearest = np.where(valid, dk, np.inf).min(axis=1)
        return prediction.astype("float64"), nearest.astype("float64")

    def _dense_idw_estimate(
        self,
        xy: np.ndarray,
        exclude_well_id: str | None,
        k: int = 20,
        fetch: int = 600,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        query = xy / self.dense_scale
        fetch = min(fetch, self.n_dense_points)
        distances, indices = self.dense_tree.query(query, k=fetch, workers=-1)
        if distances.ndim == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        if exclude_well_id is not None:
            mask_excl = self.dense_well_ids[indices] == exclude_well_id
            distances = np.where(mask_excl, np.inf, distances)
        kept = min(k, fetch)
        order = np.argpartition(distances, min(kept - 1, fetch - 1), axis=1)[:, :kept]
        dk = np.take_along_axis(distances, order, axis=1)
        ik = np.take_along_axis(indices, order, axis=1)
        valid = np.isfinite(dk)
        weights = np.where(valid, 1.0 / (dk + 1e-3), 0.0)
        weight_sum = weights.sum(axis=1)
        safe = np.where(weight_sum < 1e-9, 1.0, weight_sum)
        fn = self.dense_form[ik]
        prediction = (fn * weights[:, :, None]).sum(axis=1) / safe[:, None]
        # variance (per-formation) -> use mean stddev across formations
        variance = ((fn - prediction[:, None, :]) ** 2 * weights[:, :, None]).sum(axis=1) / safe[:, None]
        stddev = np.sqrt(np.clip(variance, 0.0, None))
        no_valid = weight_sum < 1e-9
        global_mean = self.dense_form.mean(axis=0)
        prediction = np.where(no_valid[:, None], global_mean[None, :], prediction)
        nearest = np.where(valid, dk, np.inf).min(axis=1)
        return prediction.astype("float64"), stddev.astype("float64"), nearest.astype("float64")

    def impute(
        self,
        xy: np.ndarray,
        exclude_well_id: str | None = None,
        plane_k: int = 10,
        dense_k: int = 20,
    ) -> dict[str, np.ndarray]:
        """Return predicted formation depths for each (x, y) query.

        Output dict contains keys:
            ``plane`` (n, 6): plane-KNN estimate (smooth regional).
            ``dense`` (n, 6): dense IDW estimate (local detail).
            ``plane_distance`` (n,): nearest centroid distance, scaled.
            ``dense_distance`` (n,): nearest dense-point distance, scaled.
            ``dense_std`` (n, 6): dense neighbourhood stddev per formation.
        """

        plane, plane_dist = self._centroid_plane_estimate(xy, exclude_well_id, k=plane_k)
        dense, dense_std, dense_dist = self._dense_idw_estimate(xy, exclude_well_id, k=dense_k)
        return {
            "plane": plane,
            "dense": dense,
            "plane_distance": plane_dist,
            "dense_distance": dense_dist,
            "dense_std": dense_std,
        }


def _wls_b_well(
    tvt_known: np.ndarray,
    z_known: np.ndarray,
    form_known: np.ndarray,
    decay_per_row: float = 0.02,
) -> float:
    """Recent-row-weighted estimate of ``b_well = TVT + Z - formation``.

    The known prefix is ordered MD-ascending. We weight the last row
    most heavily and decay exponentially backwards so the b_well estimate
    reflects the geology at the end of the known prefix (closest to the
    hidden section).
    """

    diffs = tvt_known + z_known - form_known
    valid = np.isfinite(diffs)
    if valid.sum() == 0:
        return float("nan")
    diffs = diffs[valid]
    n = len(diffs)
    age = np.arange(n - 1, -1, -1, dtype="float64")
    weights = np.exp(-decay_per_row * age)
    return float(np.sum(weights * diffs) / np.sum(weights))


def _fit_b_well_trend(
    md_known: np.ndarray,
    tvt_known: np.ndarray,
    z_known: np.ndarray,
    form_known: np.ndarray,
    decay_per_row: float = 0.02,
) -> tuple[float, float, float]:
    """Weighted linear fit ``b_well ≈ slope * MD + intercept``.

    Returns ``(slope, intercept, residual_std)``. If b_well is well behaved
    (constant per well per formation when the formation imputation is
    accurate), the slope is near zero; when the imputation introduces a
    spatially varying bias, the slope captures it so the trend can be
    extrapolated into the hidden region.

    Residual standard deviation is reported so callers can downweight wells
    where the linear fit is poor (i.e., the b_well is changing nonlinearly
    or the imputation noise is high).
    """

    diffs = tvt_known + z_known - form_known
    valid = np.isfinite(diffs) & np.isfinite(md_known)
    if valid.sum() < 5:
        return 0.0, float("nan"), float("inf")
    x = md_known[valid].astype("float64")
    y = diffs[valid].astype("float64")
    n = len(y)
    age = np.arange(n - 1, -1, -1, dtype="float64")
    weights = np.exp(-decay_per_row * age)
    # Weighted least squares for slope, intercept
    w_sum = weights.sum()
    x_mean = float((weights * x).sum() / w_sum)
    y_mean = float((weights * y).sum() / w_sum)
    dx = x - x_mean
    dy = y - y_mean
    sxx = float((weights * dx * dx).sum())
    sxy = float((weights * dx * dy).sum())
    if sxx <= 1e-12:
        return 0.0, y_mean, float(np.sqrt((weights * dy * dy).sum() / w_sum))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    residual = y - (slope * x + intercept)
    residual_std = float(np.sqrt((weights * residual * residual).sum() / w_sum))
    return float(slope), float(intercept), residual_std


def predict_physical_tvt(
    horizontal: pd.DataFrame,
    tvt_known: np.ndarray,
    imputer: FormationImputer,
    exclude_well_id: str | None = None,
    extrapolate_b_well: bool = False,
) -> dict[str, np.ndarray]:
    """Predict TVT from spatially imputed formation depths.

    When ``extrapolate_b_well`` is true (default), the per-formation
    ``b_well`` term is fit as a linear function of MD on the known prefix
    and extrapolated into the hidden region. This handles wells where the
    spatial imputer has a slowly varying regional bias — without trend
    extrapolation the prediction systematically drifts away from the
    truth in the hidden section.

    Returns a dict with keys:
        ``prediction``: blended TVT prediction over the full horizontal
            (uses tvt_known where it is finite, otherwise the physical
            estimate).
        ``quality``: per-formation known-zone RMSE; lower is better.
        ``b_well``: per-formation WLS-calibrated b_well values.
        ``b_well_residual_std``: per-formation residual std after fitting
            a linear MD trend to b_well; large values flag unstable
            imputation calibration.
        ``stddev_imputed``: per-row uncertainty estimate (dense_std mean).
    """

    n_rows = len(horizontal)
    xy = horizontal[["X", "Y"]].to_numpy(dtype="float64")
    z = pd.to_numeric(horizontal["Z"], errors="coerce").to_numpy(dtype="float64")
    md = pd.to_numeric(horizontal["MD"], errors="coerce").to_numpy(dtype="float64")
    tvt_known = np.asarray(tvt_known, dtype="float64")
    known_mask = np.isfinite(tvt_known)
    hidden_mask = ~known_mask

    impute_result = imputer.impute(xy, exclude_well_id=exclude_well_id)
    plane = impute_result["plane"]
    dense = impute_result["dense"]
    dense_dist = impute_result["dense_distance"]
    plane_dist = impute_result["plane_distance"]

    # Blend plane (regional) and dense (local) using their nearest-distance.
    # Closer dense neighbours -> trust dense more.
    dense_weight = 1.0 / (dense_dist + 0.5)
    plane_weight = 1.0 / (plane_dist + 1.5)
    total = dense_weight + plane_weight
    dense_share = (dense_weight / total)[:, None]
    plane_share = (plane_weight / total)[:, None]
    blended_form = dense_share * dense + plane_share * plane

    per_formation_preds = np.full((n_rows, len(FORMATIONS)), np.nan, dtype="float64")
    per_formation_rmse = np.full(len(FORMATIONS), np.nan, dtype="float64")
    per_formation_b = np.full(len(FORMATIONS), np.nan, dtype="float64")
    per_formation_b_resid = np.full(len(FORMATIONS), np.inf, dtype="float64")
    z_known = z[known_mask]
    tvt_known_vals = tvt_known[known_mask]
    md_known = md[known_mask]

    for fi, _name in enumerate(FORMATIONS):
        form_known = blended_form[known_mask, fi]
        finite = np.isfinite(form_known) & np.isfinite(z_known) & np.isfinite(tvt_known_vals)
        if finite.sum() < 5:
            continue
        if extrapolate_b_well:
            slope, intercept, residual_std = _fit_b_well_trend(
                md_known[finite],
                tvt_known_vals[finite],
                z_known[finite],
                form_known[finite],
            )
            if not np.isfinite(intercept):
                continue
            # b_well as a function of MD, evaluated at every row
            b_per_row = slope * md + intercept
            prediction = -z + blended_form[:, fi] + b_per_row
            # known-zone RMSE
            residual = tvt_known_vals[finite] - prediction[known_mask][finite]
            per_formation_rmse[fi] = float(np.sqrt(np.mean(residual ** 2)))
            # Representative b_well = value at last known row (closest to hidden)
            last_md = float(md_known[finite][-1])
            per_formation_b[fi] = float(slope * last_md + intercept)
            per_formation_b_resid[fi] = float(residual_std)
            per_formation_preds[:, fi] = prediction
        else:
            b_well = _wls_b_well(tvt_known_vals[finite], z_known[finite], form_known[finite])
            if not np.isfinite(b_well):
                continue
            prediction = -z + blended_form[:, fi] + b_well
            residual = tvt_known_vals[finite] - prediction[known_mask][finite]
            per_formation_rmse[fi] = float(np.sqrt(np.mean(residual ** 2)))
            per_formation_b[fi] = b_well
            per_formation_preds[:, fi] = prediction

    # Combine the per-formation predictions: weight inversely by known-zone RMSE.
    # A formation that fits the known prefix well will fit the hidden region well.
    valid_form = np.isfinite(per_formation_rmse)
    if not valid_form.any():
        physical_prediction = np.full(n_rows, np.nan, dtype="float64")
        quality_overall = np.nan
    else:
        rmse = per_formation_rmse[valid_form]
        weights = 1.0 / (rmse + 0.1)
        weights = weights / weights.sum()
        preds_subset = per_formation_preds[:, valid_form]
        finite_preds = np.isfinite(preds_subset)
        row_weights = np.where(finite_preds, weights[None, :], 0.0)
        row_weight_sum = row_weights.sum(axis=1)
        physical_prediction = np.divide(
            (np.where(finite_preds, preds_subset, 0.0) * row_weights).sum(axis=1),
            row_weight_sum,
            out=np.full(n_rows, np.nan, dtype="float64"),
            where=row_weight_sum > 0,
        )
        quality_overall = float(np.average(rmse, weights=weights))

    full_prediction = np.where(known_mask, tvt_known, physical_prediction)

    stddev_imputed = impute_result["dense_std"].mean(axis=1)

    return {
        "prediction": full_prediction,
        "physical_prediction": physical_prediction,
        "per_formation_predictions": per_formation_preds,
        "per_formation_rmse": per_formation_rmse,
        "per_formation_b": per_formation_b,
        "per_formation_b_residual_std": per_formation_b_resid,
        "overall_quality_rmse": quality_overall,
        "stddev_imputed": stddev_imputed,
        "dense_distance": dense_dist,
        "plane_distance": plane_dist,
    }
