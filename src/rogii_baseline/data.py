"""Data paths and lightweight CSV helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

import pandas as pd


DEFAULT_LOCAL_DATA_DIR = Path("rogii-wellbore-geology-prediction")
DEFAULT_KAGGLE_DATA_DIR = Path("/kaggle/input/rogii-wellbore-geology-prediction")
SAMPLE_ID_RE = re.compile(r"^(?P<well_id>[0-9a-fA-F]{8})_(?P<row_index>\d+)$")


@dataclass(frozen=True)
class WellFiles:
    well_id: str
    horizontal: Path
    typewell: Path | None


def get_data_dir(data_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the best available competition data directory."""

    candidates: list[Path] = []
    if data_dir:
        candidates.append(Path(data_dir))
    env_data_dir = os.environ.get("ROGII_DATA_DIR")
    if env_data_dir:
        candidates.append(Path(env_data_dir))
    candidates.extend([DEFAULT_LOCAL_DATA_DIR, DEFAULT_KAGGLE_DATA_DIR])

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for sample_path in kaggle_input.rglob("sample_submission.csv"):
            candidates.append(sample_path.parent)

    for candidate in candidates:
        if (
            candidate.exists()
            and (candidate / "train").exists()
            and (candidate / "test").exists()
            and (candidate / "sample_submission.csv").exists()
        ):
            return candidate.resolve()
    return candidates[0].resolve()


def require_data_dir(data_dir: str | os.PathLike[str] | None = None) -> Path:
    resolved = get_data_dir(data_dir)
    required = [resolved / "train", resolved / "test", resolved / "sample_submission.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Competition data directory is incomplete. Missing: "
            + ", ".join(missing)
            + ". Set ROGII_DATA_DIR or download the Kaggle data."
        )
    return resolved


def split_dir(data_dir: Path, split: str) -> Path:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    return data_dir / split


def list_well_ids(data_dir: Path, split: str) -> list[str]:
    ids = []
    for path in split_dir(data_dir, split).glob("*__horizontal_well.csv"):
        ids.append(path.name.split("__", 1)[0])
    return sorted(set(ids))


def find_typewell_path(data_dir: Path, split: str, well_id: str) -> Path | None:
    candidates = sorted(split_dir(data_dir, split).glob(f"{well_id}__typewell*.csv"))
    return candidates[0] if candidates else None


def well_files(data_dir: Path, split: str, well_id: str) -> WellFiles:
    horizontal = split_dir(data_dir, split) / f"{well_id}__horizontal_well.csv"
    return WellFiles(
        well_id=well_id,
        horizontal=horizontal,
        typewell=find_typewell_path(data_dir, split, well_id),
    )


def read_horizontal(data_dir: Path, split: str, well_id: str) -> pd.DataFrame:
    path = well_files(data_dir, split, well_id).horizontal
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_typewell(data_dir: Path, split: str, well_id: str) -> pd.DataFrame:
    path = well_files(data_dir, split, well_id).typewell
    if path is None or not path.exists():
        return pd.DataFrame(columns=["TVT", "GR"])
    return pd.read_csv(path)


def load_sample_submission(data_dir: Path) -> pd.DataFrame:
    sample_path = data_dir / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"Unexpected sample_submission columns: {list(sample.columns)}")

    parsed = sample["id"].map(parse_submission_id)
    sample = sample.copy()
    sample["well_id"] = [well_id for well_id, _ in parsed]
    sample["row_index"] = [row_index for _, row_index in parsed]
    return sample


def parse_submission_id(value: str) -> tuple[str, int]:
    match = SAMPLE_ID_RE.match(value)
    if not match:
        raise ValueError(f"Invalid sample id: {value!r}")
    return match.group("well_id").lower(), int(match.group("row_index"))


def numeric_column(df: pd.DataFrame, column: str, default: float = float("nan")) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")
