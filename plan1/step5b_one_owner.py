"""Step 5b: one owner per S2/S3 record, applied to a finished run's saved test scores (no retrieval, no model).

For the baseline run:  AMC_RUN_ID=p1-baseline-v1 python -m plan1.step5b_one_owner
(Runs from p1-v2 on already apply the rule inside step 5.)
Outputs: output/plan1/<run>-owner/{candidate_pairs.tsv, matching_results.tsv}
"""
import json

import polars as pl

from . import config as C
from .dataio import load_records, write_id_lists
from .decide import one_owner

CHUNK_DIR = C.WORK_DIR / "test_chunks"
OUT = C.ROOT / "output" / "plan1" / f"{C.RUN_ID}-owner"


def main() -> None:
    print("run:", C.RUN_ID)
    thr = json.loads((C.WORK_DIR / "frozen_config.json").read_text(encoding="utf-8"))["threshold"]
    scored = pl.read_parquet(str(CHUNK_DIR / "*.parquet"))
    s1 = load_records("test").filter(pl.col("source") == "S1").select(s1_id="entity_id", country="country")
    roster = s1["s1_id"].sort()

    accepted = scored.filter(pl.col("p") >= thr).join(s1, on="s1_id")
    ranked = accepted.sort(["target_id", "p", "s1_id"], descending=[False, True, False]).with_columns(
        claim_rank=pl.int_range(pl.len()).over("target_id"),
        claimants=pl.len().over("target_id"),
        p_top=pl.col("p").first().over("target_id"),
    )
    kept = one_owner(accepted)
    dropped = ranked.filter(pl.col("claim_rank") > 0)
    assert kept.height == ranked.filter(pl.col("claim_rank") == 0).height

    print(f"accepted links {accepted.height:,} -> kept {kept.height:,}, dropped {dropped.height:,}")
    print("\nby S1 country:")
    print(ranked.group_by("country").agg(
        accepted=pl.len(),
        contested_links=(pl.col("claimants") > 1).sum(),
        dropped=(pl.col("claim_rank") > 0).sum(),
    ).with_columns(dropped_share=pl.col("dropped") / pl.col("accepted")).sort("country"))
    print("\nclaimants per contested record:")
    print(ranked.filter(pl.col("claimants") > 1).group_by("target_id").agg(n=pl.first("claimants")).group_by("n").len().sort("n"))
    print("\nscore gap between the winner and the runner-up:")
    print(dropped.filter(pl.col("claim_rank") == 1).select(gap=pl.col("p_top") - pl.col("p")).select(
        median=pl.col("gap").median(), under_0_05=(pl.col("gap") < 0.05).mean(), under_0_10=(pl.col("gap") < 0.10).mean()))

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
