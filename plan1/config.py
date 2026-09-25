"""Plan 1 settings: paths, split fractions, sample sizes and fixed model parameters.

Set AMC_DATA_DIR to point at the dataset folder (the one holding train/ and test/) if it is not ./dataset.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("AMC_DATA_DIR", ROOT / "dataset"))
# p1-baseline-v1: the frozen Plan 1 baseline. p1-v2: + quick fixes. p1-v3: + name-only fallback search, longer
# training, better dictionary, fresh Audit panel, crowded-record variant (see PLAN1_LESSONS.md).
# Set AMC_RUN_ID to an older run only for scripts that just read that run's saved outputs (e.g. step5b).
RUN_ID = os.environ.get("AMC_RUN_ID", "p1-v3")

SHARED_DIR = ROOT / "work" / "shared"  # raw caches and the split manifest, shared by all plans
RAW_DIR = SHARED_DIR / "raw"
SPLIT_DIR = SHARED_DIR / "splits"
WORK_DIR = ROOT / "work" / "plan1" / RUN_ID  # everything Plan 1 builds
OUT_DIR = ROOT / "output" / "plan1" / RUN_ID  # the two submission files

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
LABEL_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
SOURCES = ("S1", "S2", "S3")
TARGET_SOURCES = ("S2", "S3")

# ---- split manifest (shared by all plans; independent salts for each use) ----
ROLE_SALT = "amc26-role-v1"
CALIB_SALT = "amc26-calibrate-subdivision-v1"
SAMPLE_SALT = "amc26-sample-v1"
ROLE_BOUNDS = [("fit", 0.60), ("tune", 0.70), ("calibrate", 0.80), ("audit", 1.00)]  # cumulative upper bounds
SAMPLE_SIZES = {  # sample name -> (pool it is drawn from, size)
    "fit_sample": ("fit", 100_000),
    "tune_sample": ("tune", 20_000),
    "c_select_sample": ("c_select", 20_000),
    "audit_panel": ("audit", 20_000),
}
MAX_BAND = 6  # true-match-count bands 0,1,...,5,6+ used to stratify samples
# p1-v2 trains on a larger draw from the Fit role (same stratification and sample salt; contains the 100k sample)
FIT_SAMPLE_SIZE = 300_000
# p1-v3: a fresh 20k Audit panel from Audit-role S1 never used before (the first panel is now development data)
AUDIT_PANEL_2_SALT = "amc26-audit-panel-2-v1"
AUDIT_PANEL_2_SIZE = 20_000

# ---- candidate search / decisions ----
FALLBACK_TOP_K = 10  # name-only fallback search over records without an address, per source
CROWD_LIMIT = 10     # variant submission: drop records accepted for this many S1 or more

# ---- ablations (plan1/ablate.py): smaller, identical setup for every variant ----
ABLATION_FIT_SIZE = 100_000
ABLATION_MAX_ROUNDS = 3000

SEED = 42

# ---- matcher (Plan 1 section 5): fixed, no sweeps ----
LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 10,
    "min_data_in_leaf": 100,
    "lambda_l2": 5.0,
    "seed": SEED,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 20,  # pinned (physical cores of the w7-3445) so reruns are identical
    "verbosity": -1,
}
LGB_MAX_ROUNDS = 8000  # p1-v2 hit the old 2000 cap while still improving; early stopping decides
LGB_EARLY_STOPPING = 100

# Model backend for new trainings (second stage, ablations): "lightgbm" (CPU, the reference), "xgboost" or
# "catboost" (both on the GPU). Choose with AMC_BACKEND=...; compare first with: python -m plan1.ablate backends
MODEL_BACKEND = os.environ.get("AMC_BACKEND", "lightgbm")
XGB_PARAMS = {  # shaped like the LightGBM settings: leaf-wise trees of up to 63 leaves, same rate and L2
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda",
    "learning_rate": 0.05,
    "grow_policy": "lossguide",
    "max_leaves": 63,
    "max_depth": 10,
    "min_child_weight": 1.0,
    "lambda": 5.0,
    "max_bin": 256,
    "seed": SEED,
    "nthread": 20,
}
CAT_PARAMS = {  # symmetric (oblivious) trees: a different model family, useful as a blend partner
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "task_type": "GPU",
    "devices": "0",
    "learning_rate": 0.08,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "border_count": 254,
    "random_seed": SEED,
    "thread_count": 20,
}

# ---- second stage (plan1/stage2.py) ----
STAGE2_FOLDS = 3  # out-of-fold first-stage scores for the Fit sample
STAGE2_FOLD_SALT = "amc26-stage2-fold-v1"
STAGE2_ANCHOR_P = 0.5  # candidates scoring at least this are the "likely records" others are compared with

# ---- leaderboard variants (plan1/variants.py) ----
VARIANT_MARGIN = 0.05  # drop a contested record when the runner-up is this close to the winner
VARIANT_CAP = 10       # keep at most this many matches per S1 (training max is 11)
