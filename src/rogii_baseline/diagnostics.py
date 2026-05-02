"""Diagnostics for validation artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .baseline import _as_float_array, _typewell_arrays
from .data import read_horizontal, read_typewell, require_data_dir


def _load_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _filter_frame(
    frame: pd.DataFrame,
    model_kind: str | None,
    scenario: str | None,
    mask_strategy: str | None,
) -> pd.DataFrame:
    filtered = frame
    if model_kind:
        filtered = filtered[filtered["model_kind"] == model_kind]
    if scenario:
        filtered = filtered[filtered["scenario"] == scenario]
    if mask_strategy:
        filtered = filtered[filtered["mask_strategy"] == mask_strategy]
    return filtered


def _safe_interp(tvt: np.ndarray, typewell_tvt: np.ndarray, typewell_gr: np.ndarray) -> np.ndarray:
    if len(typewell_tvt) < 2:
        return np.full(len(tvt), np.nan, dtype="float64")
    return np.interp(np.clip(tvt, typewell_tvt[0], typewell_tvt[-1]), typewell_tvt, typewell_gr)


def plot_worst_wells(
    data_dir: str | Path | None = None,
    oof_dir: str | Path | None = None,
    predictions_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    top_n: int = 10,
    model_kind: str | None = None,
    scenario: str | None = None,
    mask_strategy: str | None = None,
) -> list[Path]:
    """Plot TVT, residual, and GR alignment for the worst validation wells."""

    if oof_dir is None and (predictions_path is None or metrics_path is None):
        raise ValueError("Provide --oof-dir or both --predictions and --metrics.")
    oof = Path(oof_dir) if oof_dir is not None else None
    predictions = Path(predictions_path) if predictions_path is not None else oof / "oof_predictions.csv"
    metrics = Path(metrics_path) if metrics_path is not None else oof / "well_metrics.csv"
    output = Path(output_dir) if output_dir is not None else (oof / "diagnostics" if oof is not None else Path("diagnostics"))
    output.mkdir(parents=True, exist_ok=True)

    resolved = require_data_dir(data_dir)
    metric_frame = _filter_frame(pd.read_csv(metrics), model_kind, scenario, mask_strategy)
    if metric_frame.empty:
        raise ValueError("No metric rows matched the requested filters.")
    selected = metric_frame.sort_values("rmse", ascending=False).head(top_n).reset_index(drop=True)
    selected_keys = set(zip(selected["model_kind"], selected["scenario"], selected["well_id"]))

    prediction_frame = pd.read_csv(predictions)
    prediction_frame = prediction_frame[
        [
            (model, scenario_name, well_id) in selected_keys
            for model, scenario_name, well_id in zip(
                prediction_frame["model_kind"],
                prediction_frame["scenario"],
                prediction_frame["well_id"],
            )
        ]
    ]
    prediction_frame = _filter_frame(prediction_frame, model_kind, scenario, mask_strategy)
    if prediction_frame.empty:
        raise ValueError("No prediction rows matched the selected worst wells.")

    plt = _load_matplotlib()
    written: list[Path] = []
    for rank, metric in selected.iterrows():
        well_id = str(metric["well_id"])
        model = str(metric["model_kind"])
        scenario_name = str(metric["scenario"])
        mask_name = str(metric["mask_strategy"])
        preds = prediction_frame[
            (prediction_frame["model_kind"] == model)
            & (prediction_frame["scenario"] == scenario_name)
            & (prediction_frame["well_id"] == well_id)
        ].sort_values("row_index")
        if preds.empty:
            continue

        horizontal = read_horizontal(resolved, "train", well_id)
        typewell = read_typewell(resolved, "train", well_id)
        full_rows = np.arange(len(horizontal))
        tvt_input = _as_float_array(horizontal, "TVT_input")
        gr = _as_float_array(horizontal, "GR")
        tw_tvt, tw_gr = _typewell_arrays(typewell)

        row_index = preds["row_index"].to_numpy(dtype="int64")
        actual = preds["actual_tvt"].to_numpy(dtype="float64")
        predicted = preds["predicted_tvt"].to_numpy(dtype="float64")
        residual = preds["residual"].to_numpy(dtype="float64")
        target_gr = gr[row_index]
        tw_gr_actual = _safe_interp(actual, tw_tvt, tw_gr)
        tw_gr_predicted = _safe_interp(predicted, tw_tvt, tw_gr)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        title = (
            f"{rank + 1}. {well_id} | {model} | {scenario_name} | "
            f"RMSE {float(metric['rmse']):.3f} | rows {int(metric['rows'])}"
        )
        fig.suptitle(title, fontsize=13)

        known = np.isfinite(tvt_input)
        axes[0].plot(full_rows[known], tvt_input[known], color="#2563eb", linewidth=1.0, label="known TVT_input")
        axes[0].plot(row_index, actual, color="#111827", linewidth=1.0, label="actual TVT")
        axes[0].plot(row_index, predicted, color="#dc2626", linewidth=1.0, alpha=0.9, label="predicted TVT")
        axes[0].set_ylabel("TVT")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(row_index, residual, color="#7c2d12", linewidth=0.9)
        axes[1].axhline(0.0, color="#111827", linewidth=0.8, alpha=0.6)
        axes[1].set_ylabel("Pred - Actual")
        axes[1].grid(True, alpha=0.25)

        axes[2].plot(row_index, target_gr, color="#0f766e", linewidth=0.8, alpha=0.85, label="horizontal GR")
        axes[2].plot(row_index, tw_gr_actual, color="#111827", linewidth=0.8, alpha=0.7, label="typewell GR @ actual")
        axes[2].plot(
            row_index,
            tw_gr_predicted,
            color="#dc2626",
            linewidth=0.8,
            alpha=0.75,
            label="typewell GR @ predicted",
        )
        axes[2].set_ylabel("GR")
        axes[2].set_xlabel("row index")
        axes[2].legend(loc="best")
        axes[2].grid(True, alpha=0.25)

        first_target = int(row_index.min())
        for axis in axes:
            axis.axvline(first_target, color="#6b7280", linewidth=0.8, linestyle="--", alpha=0.75)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        safe_name = f"{rank + 1:02d}_{well_id}_{model}_{mask_name}_{scenario_name}.png".replace("/", "_")
        path = output / safe_name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot worst wells from OOF validation artifacts.")
    parser.add_argument("--data-dir", default=None, help="Competition data directory.")
    parser.add_argument("--oof-dir", default=None, help="Directory containing OOF artifacts.")
    parser.add_argument("--predictions", default=None, help="Path to oof_predictions.csv.")
    parser.add_argument("--metrics", default=None, help="Path to well_metrics.csv.")
    parser.add_argument("--output-dir", default=None, help="Directory for diagnostic PNG files.")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--model-kind", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--mask-strategy", default=None)
    args = parser.parse_args(argv)
    paths = plot_worst_wells(
        data_dir=args.data_dir,
        oof_dir=args.oof_dir,
        predictions_path=args.predictions,
        metrics_path=args.metrics,
        output_dir=args.output_dir,
        top_n=args.top_n,
        model_kind=args.model_kind,
        scenario=args.scenario,
        mask_strategy=args.mask_strategy,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
