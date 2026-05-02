"""Data audit for local ROGII competition files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    list_well_ids,
    load_sample_submission,
    read_horizontal,
    require_data_dir,
    well_files,
)


def _horizontal_summary(data_dir: Path, split: str) -> dict[str, object]:
    wells = list_well_ids(data_dir, split)
    rows = []
    missing_gr = 0
    missing_target_signal = 0
    for well_id in wells:
        path = well_files(data_dir, split, well_id).horizontal
        header = pd.read_csv(path, nrows=0).columns
        usecols = [col for col in ["GR", "TVT", "TVT_input"] if col in header]
        df = pd.read_csv(path, usecols=usecols)
        rows.append(len(df))
        if "GR" in df.columns:
            missing_gr += int(pd.to_numeric(df["GR"], errors="coerce").isna().sum())
        if split == "train":
            target_col = "TVT" if "TVT" in df.columns else "TVT_input"
        else:
            target_col = "TVT_input" if "TVT_input" in df.columns else "TVT"
        if target_col in df.columns:
            missing_target_signal += int(pd.to_numeric(df[target_col], errors="coerce").isna().sum())

    if rows:
        row_stats = {
            "total": int(sum(rows)),
            "min": int(min(rows)),
            "median": int(np.median(rows)),
            "max": int(max(rows)),
        }
    else:
        row_stats = {"total": 0, "min": 0, "median": 0, "max": 0}

    return {
        "split": split,
        "wells": len(wells),
        "rows": row_stats,
        "missing_gr": missing_gr,
        "missing_tvt_or_tvt_input": missing_target_signal,
    }


def _test_windows(data_dir: Path) -> list[dict[str, object]]:
    windows = []
    for well_id in list_well_ids(data_dir, "test"):
        horizontal = read_horizontal(data_dir, "test", well_id)
        tvt_input = pd.to_numeric(horizontal.get("TVT_input"), errors="coerce")
        missing = np.flatnonzero(tvt_input.isna().to_numpy())
        known = np.flatnonzero(tvt_input.notna().to_numpy())
        md = pd.to_numeric(horizontal.get("MD"), errors="coerce")
        first_hidden = int(missing[0]) if len(missing) else None
        last_known = int(known[-1]) if len(known) else None
        windows.append(
            {
                "well_id": well_id,
                "rows": int(len(horizontal)),
                "last_known_index": last_known,
                "first_hidden_index": first_hidden,
                "last_known_md": float(md.iloc[last_known]) if last_known is not None else None,
                "first_hidden_md": float(md.iloc[first_hidden]) if first_hidden is not None else None,
            }
        )
    return windows


def _visible_test_overlap(data_dir: Path) -> list[dict[str, object]]:
    overlaps = []
    common = sorted(set(list_well_ids(data_dir, "train")) & set(list_well_ids(data_dir, "test")))
    for well_id in common:
        train = read_horizontal(data_dir, "train", well_id)
        test = read_horizontal(data_dir, "test", well_id)
        same_rows = len(train) == len(test)
        path_same = False
        if same_rows and {"MD", "X", "Y", "Z"}.issubset(train.columns) and {"MD", "X", "Y", "Z"}.issubset(test.columns):
            path_same = bool(
                np.allclose(
                    train[["MD", "X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce"),
                    test[["MD", "X", "Y", "Z"]].apply(pd.to_numeric, errors="coerce"),
                    equal_nan=True,
                )
            )

        prefix_matches = False
        known_count = 0
        if same_rows and "TVT" in train.columns and "TVT_input" in test.columns:
            train_tvt = pd.to_numeric(train["TVT"], errors="coerce")
            test_tvt = pd.to_numeric(test["TVT_input"], errors="coerce")
            known = test_tvt.notna()
            known_count = int(known.sum())
            if known_count:
                prefix_matches = bool(np.allclose(train_tvt[known], test_tvt[known], equal_nan=True))

        overlaps.append(
            {
                "well_id": well_id,
                "same_rows": same_rows,
                "same_path": path_same,
                "known_prefix_rows": known_count,
                "known_prefix_matches_train_tvt": prefix_matches,
            }
        )
    return overlaps


def build_audit(data_dir: str | Path | None = None) -> dict[str, object]:
    resolved = require_data_dir(data_dir)
    sample = load_sample_submission(resolved)
    return {
        "data_dir": str(resolved),
        "train": _horizontal_summary(resolved, "train"),
        "test": _horizontal_summary(resolved, "test"),
        "sample_submission_rows": int(len(sample)),
        "sample_submission_wells": sorted(sample["well_id"].unique().tolist()),
        "test_prediction_windows": _test_windows(resolved),
        "visible_test_overlap": _visible_test_overlap(resolved),
    }


def print_audit(audit: dict[str, object]) -> None:
    print(f"Data directory: {audit['data_dir']}")
    for split in ["train", "test"]:
        summary = audit[split]
        rows = summary["rows"]
        print(
            f"{split}: {summary['wells']} wells, {rows['total']} rows "
            f"(min/median/max {rows['min']}/{rows['median']}/{rows['max']}), "
            f"missing GR {summary['missing_gr']}, "
            f"missing TVT/TVT_input {summary['missing_tvt_or_tvt_input']}"
        )
    print(
        f"sample_submission: {audit['sample_submission_rows']} rows across "
        f"{len(audit['sample_submission_wells'])} wells"
    )
    print("test prediction windows:")
    for window in audit["test_prediction_windows"]:
        print(
            "  {well_id}: rows={rows}, last_known_index={last_known_index}, "
            "first_hidden_index={first_hidden_index}, last_known_md={last_known_md}, "
            "first_hidden_md={first_hidden_md}".format(**window)
        )
    if audit["visible_test_overlap"]:
        print("visible test overlap with train:")
        for overlap in audit["visible_test_overlap"]:
            print(
                "  {well_id}: same_rows={same_rows}, same_path={same_path}, "
                "known_prefix_rows={known_prefix_rows}, "
                "known_prefix_matches_train_tvt={known_prefix_matches_train_tvt}".format(**overlap)
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="Competition data directory.")
    args = parser.parse_args(argv)
    print_audit(build_audit(args.data_dir))


if __name__ == "__main__":
    main()
