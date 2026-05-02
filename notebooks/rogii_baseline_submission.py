# %% [markdown]
# # ROGII Wellbore Geology Prediction Baseline
#
# Kaggle notebook script. Add this repo's source files to the notebook/session,
# keep internet disabled, and write `/kaggle/working/submission.csv`.

# %%
from pathlib import Path
import os
import sys

REPO_ROOT = Path.cwd()
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else REPO_ROOT
for candidate in [SCRIPT_DIR / "src", REPO_ROOT / "src", Path("/kaggle/src/src"), Path("/kaggle/working/src")]:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

os.environ.setdefault("ROGII_DATA_DIR", "/kaggle/input/rogii-wellbore-geology-prediction")

from rogii_baseline.baseline import make_submission


# %%
submission = make_submission(
    data_dir=os.environ["ROGII_DATA_DIR"],
    output_path="/kaggle/working/submission.csv",
    model_kind=os.environ.get("ROGII_MODEL_KIND", "formation_pf"),
    max_train_rows=int(os.environ.get("ROGII_MAX_TRAIN_ROWS", "800000")),
    pf_n_particles=int(os.environ.get("ROGII_PF_N_PARTICLES", "500")),
    pf_n_seeds=int(os.environ.get("ROGII_PF_N_SEEDS", "128")),
    pf_likelihood_scale=float(os.environ.get("ROGII_PF_LIKELIHOOD_SCALE", "5.0")),
    physical_blend_weight=float(os.environ.get("ROGII_PHYSICAL_BLEND_WEIGHT", "0.75")),
    pf_blend_weight=float(os.environ.get("ROGII_PF_BLEND_WEIGHT", "0.125")),
    beam_blend_weight=float(os.environ.get("ROGII_BEAM_BLEND_WEIGHT", "0.125")),
)
submission.head()
