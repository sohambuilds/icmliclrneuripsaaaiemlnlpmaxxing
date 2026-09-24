"""Plan 1 settings: paths, split fractions, sample sizes and fixed model parameters.

Set AMC_DATA_DIR to point at the dataset folder (the one holding train/ and test/) if it is not ./dataset.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("AMC_DATA_DIR", ROOT / "dataset"))
RUN_ID = "p1-baseline-v1"

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

SEED = 42
