#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rogii_baseline.baseline import main_make_submission


if __name__ == "__main__":
    main_make_submission()

