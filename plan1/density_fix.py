"""Copy-density correction. The test has many more planted near-copies per S1 than train. Examples from
plan1.test_vs_dev: US candidates with the same name and street but a house number shifted by <= 10 appear 3.5x as
often per S1 on test as on dev; "extra word + number shift" appears 16x as often. The models are calibrated on train's
density, so on test they accept more copies.

Fix: a prior correction per KIND of pair (the kinds of test_vs_dev). Assume the test has as many TRUE pairs of each
kind per S1 as dev, so the extra test pairs of that kind are copies:
    rho_k = (test pairs of kind k - dev true pairs of kind k) / dev false pairs of kind k    (per 100 S1, 1..20)
    p_test = p / (p + (1 - p) * rho_k)      (odds divided by rho_k);  accept if p_test >= the usual threshold
Only US/India are corrected; France stays as it was (no dev reference; the legal-form bug distorts its kinds).

  python -m plan1.density_fix stage2     p1-v3-s2 scores   -> output/plan1/<run>-s2-dens/matching_results.tsv
  python -m plan1.density_fix s2ce       p1-v3-s2ce scores -> output/plan1/<run>-s2ce-dens/matching_results.tsv
"""
import json
import sys

import polars as pl

from . import config as C
from .dataio import load_records, test_s1_roster, write_id_lists
from .decide import one_owner
from .test_vs_dev import attrs, kinds

MIN_P = 0.02    # pairs below this are never accepted; kinds are counted over pairs >= MIN_P (as in test_vs_dev)
SMOOTH = 0.2    # per 100 S1, added to both counts so rare kinds stay near rho = 1
RHO_CAP = 20.0


def rho_table(dev: pl.DataFrame, test: pl.DataFrame, n_dev: dict, n_te: dict) -> pl.DataFrame:
    """dev: country, kind, label; test: country, kind (US/India pairs >= MIN_P). -> per country and kind: rho."""
    d = dev.group_by("country", "kind").agg(dev_n=pl.len(), dev_true=pl.col("label").sum())
    t = test.group_by("country", "kind").agg(test_n=pl.len())
    tab = t.join(d, on=["country", "kind"], how="left").fill_null(0)
    nd = pl.col("country").replace_strict(n_dev, return_dtype=pl.Float64) / 100
    nt = pl.col("country").replace_strict(n_te, return_dtype=pl.Float64) / 100
    tab = tab.with_columns(test_per100=pl.col("test_n") / nt, dev_per100=pl.col("dev_n") / nd, dev_true_per100=pl.col("dev_true") / nd)
    return tab.with_columns(rho=(((pl.col("test_per100") - pl.col("dev_true_per100")).clip(0) + SMOOTH)
                                 / ((pl.col("dev_per100") - pl.col("dev_true_per100")).clip(0) + SMOOTH)).clip(1.0, RHO_CAP))


def main(stage: str) -> None:
    if stage == "stage2":  # p1-v3-s2
        s2 = C.WORK_DIR / "stage2"
        thr = json.loads((s2 / "stage2_meta.json").read_text(encoding="utf-8"))["threshold"]
        dev_file, name = s2 / "decide_ef" / "p2_dev.parquet", f"{C.RUN_ID}-s2-dens"
    elif stage == "s2ce":  # p1-v3-s2ce (cross-encoder model)
        s2 = C.WORK_DIR / "stage2_ce"
        thr = json.loads((s2 / "stage2ce_meta.json").read_text(encoding="utf-8"))["ce"]["threshold"]
        dev_file, name = s2 / "pred_ce_dev.parquet", f"{C.RUN_ID}-s2ce-dens"
    else:
        raise SystemExit("usage: python -m plan1.density_fix stage2 | s2ce")
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(200)
    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    n_dev = dict(manifest.filter(pl.col("sample") == "audit_panel").group_by("country").len().iter_rows())
    s1c = load_records("test").filter(pl.col("source") == "S1").select(s1_id="entity_id", country="country")
    n_te = dict(s1c.group_by("country").len().iter_rows())

    dev = kinds(pl.read_parquet(dev_file).filter(pl.col("p") >= MIN_P), attrs("train"))
    scores = pl.concat([pl.read_parquet(f, columns=["s1_id", "target_id", "p"]).filter(pl.col("p") >= MIN_P)
                        for f in sorted((s2 / "test_scores").glob("*.parquet"))])
    test = kinds(scores, attrs("test"))
    usin, fr = test.filter(pl.col("country") != "France"), test.filter(pl.col("country") == "France")
    rho = rho_table(dev, usin, n_dev, n_te)
    adj = (usin.join(rho.select("country", "kind", "rho"), on=["country", "kind"], how="left").with_columns(pl.col("rho").fill_null(1.0))
           .with_columns(p_adj=pl.col("p") / (pl.col("p") + (1 - pl.col("p")) * pl.col("rho"))))

    dropped = adj.filter((pl.col("p") >= thr) & (pl.col("p_adj") < thr))
    print(f"threshold {thr}; US/India links dropped by the correction: {dropped.height:,} "
          f"(of {adj.filter(pl.col('p') >= thr).height:,} accepted)")
    print(rho.join(dropped.group_by("country", "kind").agg(dropped=pl.len()), on=["country", "kind"], how="left")
          .fill_null(0).filter(pl.col("rho") > 1.05).sort("dropped", descending=True).head(25)
          .select("country", "kind", "test_per100", "dev_per100", "dev_true_per100", "rho", "dropped")
          .with_columns(pl.col(pl.Float64).round(3)))

    accepted = pl.concat([adj.filter(pl.col("p_adj") >= thr).select("s1_id", "target_id", "p"),
                          fr.filter(pl.col("p") >= thr).select("s1_id", "target_id", "p")])
    matches = one_owner(accepted)
    out = C.ROOT / "output" / "plan1" / name / "matching_results.tsv"
    write_id_lists(matches, test_s1_roster(), out, "matched_entity_ids")
    per = (s1c.join(matches.group_by("s1_id").agg(n=pl.len()), on="s1_id", how="left").with_columns(pl.col("n").fill_null(0))
           .group_by("country").agg(mean_matches=pl.col("n").mean().round(3), empty=(pl.col("n") == 0).mean().round(4)).sort("country"))
    print(f"\n{matches.height:,} links after one owner -> {out}")
    print(per)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
