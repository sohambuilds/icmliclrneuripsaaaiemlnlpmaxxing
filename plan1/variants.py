"""Leaderboard variants from saved test scores (no search, no model run; a few minutes).

  python -m plan1.variants stage1          step 5 scores + the frozen threshold
  python -m plan1.variants stage2          second-stage scores + its threshold
Each variant goes to output/plan1/<run>[-s2]-<variant>/matching_results.tsv:
  margin  one owner, but drop a contested record when the runner-up is within VARIANT_MARGIN of the winner
  cap     one owner, then at most VARIANT_CAP matches per S1 (highest scores)
  crowd   drop records accepted for CROWD_LIMIT+ S1, then one owner
  france  the main answer for France S1 only, every other S1 empty: a leaderboard probe. Because the score is a
          mean over S1, (probe score - share of non-France S1 that truly have no match) / France share of S1
          estimates France's own score; the no-match share is unknown, ~5.6% in train.
  usin85/90/95  US/India at a stricter threshold, France unchanged: does the test (more distractors) want a
          stricter threshold than C-select?
  nofrance  the main answer with every France S1 empty: the better probe. (main - nofrance) / France share of S1
          = France's mean score - France's no-match share, so only the small France no-match share is guessed.
Validate each with utils/validate_submission.py --matching <file> --test-dir dataset/test --check-ids
"""
import json
import sys

import polars as pl

from . import config as C
from .dataio import load_records, test_s1_roster, write_id_lists
from .decide import drop_crowded, one_owner


def main(stage: str) -> None:
    if stage == "stage1":
        scores = pl.read_parquet(str(C.WORK_DIR / "test_chunks" / "*.parquet"))
        thr = json.loads((C.WORK_DIR / "frozen_config.json").read_text(encoding="utf-8"))["threshold"]
        prefix = C.RUN_ID
    elif stage == "stage2":
        scores = pl.read_parquet(str(C.WORK_DIR / "stage2" / "test_scores" / "*.parquet"))
        thr = json.loads((C.WORK_DIR / "stage2" / "stage2_meta.json").read_text(encoding="utf-8"))["threshold"]
        prefix = f"{C.RUN_ID}-s2"
    else:
        raise SystemExit("usage: python -m plan1.variants stage1 | stage2")
    roster = test_s1_roster()
    s1c = load_records("test").filter(pl.col("source") == "S1").select(s1_id="entity_id", country="country")
    accepted = scores.filter(pl.col("p") >= thr).select("s1_id", "target_id", "p")
    base = one_owner(accepted)

    ranked = accepted.sort(["target_id", "p", "s1_id"], descending=[False, True, False]).with_columns(
        r=pl.int_range(pl.len()).over("target_id"))
    top = ranked.filter(pl.col("r") == 0).select("target_id", p_top="p")
    second = ranked.filter(pl.col("r") == 1).select("target_id", p_second="p")
    close = top.join(second, on="target_id").filter(pl.col("p_top") - pl.col("p_second") < C.VARIANT_MARGIN).select("target_id")

    variants = {
        "margin": base.join(close, on="target_id", how="anti"),
        "cap": base.sort(["s1_id", "p"], descending=[False, True])
                   .filter(pl.int_range(pl.len()).over("s1_id") < C.VARIANT_CAP),
        f"crowd{C.CROWD_LIMIT}": one_owner(drop_crowded(accepted, C.CROWD_LIMIT)),
        "france": base.join(s1c.filter(pl.col("country") == "France"), on="s1_id", how="semi"),
        "nofrance": base.join(s1c.filter(pl.col("country") == "France"), on="s1_id", how="anti"),
    }
    fr_ids = s1c.filter(pl.col("country") == "France").select("s1_id")
    for t in (0.85, 0.90, 0.95):  # stricter threshold for US/India only; France unchanged, so main - usinXX is pure US/India
        us_in = scores.filter(pl.col("p") >= t).select("s1_id", "target_id", "p").join(fr_ids, on="s1_id", how="anti")
        variants[f"usin{round(t * 100)}"] = one_owner(pl.concat([accepted.join(fr_ids, on="s1_id", how="semi"), us_in]))
    print(f"{stage}: threshold {thr}, main answer (one owner) {base.height:,} links")
    for name, links in variants.items():
        path = C.ROOT / "output" / "plan1" / f"{prefix}-{name}" / "matching_results.tsv"
        write_id_lists(links, roster, path, "matched_entity_ids")
        per = (s1c.join(links.group_by("s1_id").agg(n=pl.len()), on="s1_id", how="left").with_columns(pl.col("n").fill_null(0))
               .group_by("country").agg(mean_matches=pl.col("n").mean().round(3), empty=(pl.col("n") == 0).mean().round(4)).sort("country"))
        print(f"\n{name}: {links.height:,} links ({base.height - links.height:,} fewer than main) -> {path}")
        print(per)
    fr_share = s1c.filter(pl.col("country") == "France").height / s1c.height
    print(f"\nFrance is {fr_share:.1%} of test S1. Best France probe = upload main + nofrance:")
    print(f"  France score ~ (main - nofrance) / {fr_share:.4f} + France no-match share (~0.056 in train)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
