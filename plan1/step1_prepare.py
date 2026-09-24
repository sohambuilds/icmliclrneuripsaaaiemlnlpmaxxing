"""Step 1: read and check all inputs, cache them, build the shared split manifest, check the metric and TSV writer.

Run from the repository root:  python -m plan1.step1_prepare
"""
import sys
import tempfile
from pathlib import Path

import polars as pl

from . import config as C
from .dataio import label_roster, load_label_pairs, load_records, read_id_lists, write_id_lists
from .metric import score, self_check
from .splits import load_or_build_manifest


def main(force_manifest: bool = False) -> None:
    pl.Config.set_tbl_rows(30)
    print("data dir:", C.DATA_DIR)

    self_check()
    print("metric edge cases: ok")

    train = load_records("train")
    test = load_records("test")
    pairs = load_label_pairs()
    for name, df in (("train", train), ("test", test)):
        print(f"\n{name}: {df.height:,} records")
        print(df.group_by("source", "country").len().sort("source", "country"))

    # ---- label checks ----
    s1 = train.filter(pl.col("source") == "S1")
    targets = train.filter(pl.col("source") != "S1").select(target_id="entity_id", target_country="country")
    roster = label_roster()["s1_id"]
    if set(roster.to_list()) != set(s1["entity_id"].to_list()):
        raise ValueError("ground-truth S1 ids differ from train_source1 ids")
    joined = pairs.join(targets, on="target_id", how="left").join(
        s1.select(s1_id="entity_id", country="country"), on="s1_id", how="left"
    )
    missing = joined["target_country"].is_null().sum()
    if missing:
        raise ValueError(f"{missing} labelled targets are not in train S2/S3")
    owners = pairs.group_by("target_id").agg(n=pl.len())
    print(f"\ntrue links: {pairs.height:,} | S1 with no match: {roster.len() - pairs['s1_id'].n_unique():,}")
    print("targets owned by 2+ S1:", owners.filter(pl.col("n") > 1).height)
    print("links whose countries differ:", (joined["country"] != joined["target_country"]).sum())

    # ---- split manifest ----
    manifest = load_or_build_manifest(s1, pairs, force=force_manifest)
    print("\nroles:")
    print(manifest.group_by("role", "pool").agg(n=pl.len(), share=pl.len() / manifest.height).sort("role", "pool"))
    print("samples:")
    print(manifest.drop_nulls("sample").group_by("sample").agg(n=pl.len(), mean_true=pl.col("n_true").mean(),
                                                              zero_share=(pl.col("n_true") == 0).mean()).sort("sample"))
    print("population: mean_true", round(manifest["n_true"].mean(), 4), "zero_share", round((manifest["n_true"] == 0).mean(), 4))
    print(manifest.drop_nulls("sample").group_by("sample", "country").len().sort("sample", "country"))

    # ---- TSV writer round trip on a tiny fixture (commas inside lists, empty lists, string ids) ----
    fixture = pl.DataFrame({"s1_id": ["S1-001", "S1-001", "S1-002"], "target_id": ["S2-0007", "S3-12", "S2-5"]})
    fixture_roster = pl.Series(["S1-001", "S1-002", "S1-003"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fixture.tsv"
        write_id_lists(fixture, fixture_roster, path, "matched_entity_ids")
        text = path.read_text(encoding="utf-8")
        assert text.splitlines()[0] == "source1_entity_id\tmatched_entity_ids", text
        assert "S1-003\t" in text and '"' not in text, text
        back = read_id_lists(path, "matched_entity_ids")
        assert back.sort("s1_id", "target_id").equals(fixture.sort("s1_id", "target_id")), back
        assert score(back, fixture, fixture_roster)["macro_f05"] == 1.0
    print("\nTSV writer round trip: ok (header, empty rows, no quoting, leading zeros kept)")
    print("\nstep 1 done. manifest:", C.SPLIT_DIR / "manifest.parquet")


if __name__ == "__main__":
    main(force_manifest="--force-manifest" in sys.argv)
