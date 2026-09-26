"""Search candidates for 600k more Fit-role S1 (disjoint from the 300k already used), in the background on CPU.

Half ("ce_train") will train the text cross-encoder; the other half ("gbm_extra") adds training data for the tree
models. Both halves are Fit-role training data, so nothing touches Tune / C-select / the Audit panels.

  python -m plan1.retrieve_extra      -> work/plan1/<run>/candidates_extra.parquet  (~75 min)
"""
import time

import polars as pl

from . import config as C
from .retrieve import retrieve
from .splits import fit_sample_ids, stable_unit
from .step2_retrieve import normalized

EXTRA = 600_000
SPLIT_SALT = "amc26-extra-split-v1"


def extra_ids(manifest: pl.DataFrame) -> pl.DataFrame:
    """s1_id, part ('ce_train' | 'gbm_extra'): the next 600k of the Fit-role draw, split in half by a salted hash."""
    used = fit_sample_ids(manifest, C.FIT_SAMPLE_SIZE)
    bigger = fit_sample_ids(manifest, C.FIT_SAMPLE_SIZE + EXTRA)
    extra = pl.DataFrame({"s1_id": bigger}).join(pl.DataFrame({"s1_id": used}), on="s1_id", how="anti")
    u = stable_unit(extra["s1_id"].to_list(), SPLIT_SALT)
    return extra.with_columns(part=pl.Series(["ce_train" if x < 0.5 else "gbm_extra" for x in u]))


def main() -> None:
    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    parts = extra_ids(manifest)
    print(parts.group_by("part").len())
    t0 = time.time()
    cands, _ = retrieve(normalized("train"), parts["s1_id"])
    cands = cands.join(parts, on="s1_id", how="left")
    cands.write_parquet(C.WORK_DIR / "candidates_extra.parquet")
    print(f"{cands.height:,} candidates for {parts.height:,} S1 in {(time.time() - t0) / 60:.0f} min -> {C.WORK_DIR / 'candidates_extra.parquet'}")


if __name__ == "__main__":
    main()
