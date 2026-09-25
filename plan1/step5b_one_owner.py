"""Step 5b: one owner per S2/S3 record, applied to the saved test scores (no retrieval, no model run).

In the training labels every S2/S3 record belongs to at most one S1. When a record is accepted for several S1s,
keep it only for the S1 with the highest model score (ties -> smallest S1 id). Candidates are unchanged.

Run from the repository root:  python -m plan1.step5b_one_owner
Outputs: output/plan1/p1-baseline-v1-owner/{candidate_pairs.tsv, matching_results.tsv}
"""
import json

import polars as pl

from . import config as C
from .dataio import write_id_lists
from .normalize import NORMALIZE_VERSION
from .step5_infer import CHUNK_DIR

OUT = C.ROOT / "output" / "plan1" / f"{C.RUN_ID}-owner"


def main() -> None:
    thr = json.loads((C.WORK_DIR / "frozen_config.json").read_text(encoding="utf-8"))["threshold"]
    scored = pl.read_parquet(str(CHUNK_DIR / "*.parquet"))
    s1 = (pl.read_parquet(C.WORK_DIR / f"normalized_test_v{NORMALIZE_VERSION}.parquet", columns=["entity_id", "source", "country"])
          .filter(pl.col("source") == "S1").select(s1_id="entity_id", country="country"))
    roster = s1["s1_id"].sort()

    accepted = scored.filter(pl.col("p") >= thr).join(s1, on="s1_id")
    ranked = accepted.sort(["target_id", "p", "s1_id"], descending=[False, True, False]).with_columns(
        claim_rank=pl.int_range(pl.len()).over("target_id"),
        claimants=pl.len().over("target_id"),
        p_top=pl.col("p").first().over("target_id"),
    )
    kept = ranked.filter(pl.col("claim_rank") == 0)
    dropped = ranked.filter(pl.col("claim_rank") > 0)

    contested = ranked.filter(pl.col("claimants") > 1)
    print(f"accepted links {accepted.height:,} -> kept {kept.height:,}, dropped {dropped.height:,}")
    print("\nby S1 country:")
    print(ranked.group_by("country").agg(
        accepted=pl.len(),
        contested_links=(pl.col("claimants") > 1).sum(),
        dropped=(pl.col("claim_rank") > 0).sum(),
        contested_targets=pl.col("target_id").filter(pl.col("claimants") > 1).n_unique(),
    ).with_columns(dropped_share=pl.col("dropped") / pl.col("accepted")).sort("country"))
    print("\nclaimants per contested record:")
    print(contested.group_by("target_id").agg(n=pl.first("claimants")).group_by("n").len().sort("n"))
    margin = dropped.filter(pl.col("claim_rank") == 1).select(gap=pl.col("p_top") - pl.col("p"))
    print("\nscore gap between the winner and the runner-up:")
    print(margin.select(
        median=pl.col("gap").median(),
        under_0_05=(pl.col("gap") < 0.05).mean(),
        under_0_10=(pl.col("gap") < 0.10).mean(),
    ))

    per = (pl.DataFrame({"s1_id": roster}).join(s1, on="s1_id")
           .join(accepted.group_by("s1_id").agg(before=pl.len()), on="s1_id", how="left")
           .join(kept.group_by("s1_id").agg(after=pl.len()), on="s1_id", how="left")
           .with_columns(pl.col("before", "after").fill_null(0)))
    print("\nper country after the rule:")
    print(per.group_by("country").agg(
        mean_matches_before=pl.col("before").mean(), mean_matches_after=pl.col("after").mean(),
        s1_losing_links=(pl.col("after") < pl.col("before")).mean(),
        empty_before=(pl.col("before") == 0).mean(), empty_after=(pl.col("after") == 0).mean(),
    ).sort("country"))

    write_id_lists(scored, roster, OUT / "candidate_pairs.tsv", "candidate_entity_ids")
    write_id_lists(kept, roster, OUT / "matching_results.tsv", "matched_entity_ids")
    assert kept.join(scored, on=["s1_id", "target_id"], how="anti").height == 0
    assert kept["target_id"].n_unique() == kept.height
    print("\nwritten:", OUT / "matching_results.tsv", OUT / "candidate_pairs.tsv", sep="\n  ")


if __name__ == "__main__":
    main()
