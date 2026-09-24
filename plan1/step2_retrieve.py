"""Step 2: normalize train and test, retrieve candidates for the four training samples, report candidate recall.

Run from the repository root:  python -m plan1.step2_retrieve
Outputs (under work/plan1/p1-baseline-v1/):
  normalized_{train,test}_v<N>.parquet   (N = NORMALIZE_VERSION)
  candidates_train.parquet   (s1_id, target_id, source, country, score, rank, sample)
  retrieval_report.json, retrieval_timings.tsv, retrieval_meta.json
"""
import json
import time

import polars as pl

from . import config as C
from .dataio import load_label_pairs, load_records
from .normalize import NORMALIZE_VERSION, normalize_records
from .retrieve import candidate_report, retrieve


def normalized(split: str) -> pl.DataFrame:
    path = C.WORK_DIR / f"normalized_{split}_v{NORMALIZE_VERSION}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    t0 = time.time()
    out = normalize_records(load_records(split))
    C.WORK_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    print(f"normalized {split}: {out.height:,} records in {time.time() - t0:.0f}s")
    return out


def show_examples(raw: pl.DataFrame, norm: pl.DataFrame) -> None:
    """A few before/after rows to eyeball Unicode, legal-form and number handling."""
    both = raw.join(norm, on=["entity_id", "source", "country"])
    pl.Config.set_fmt_str_lengths(60)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_tbl_rows(12)
    print("\nIndian-script names:")
    print(both.filter(pl.col("name_nonlatin")).sample(6, seed=C.SEED).select("business_name", "name_cons", "name_red"))
    print("\nLatin names with legal forms / accents:")
    print(both.filter(pl.col("business_name").str.contains(r"(?i)l\.l\.c|pvt|ltd|inc|[À-ÿ]")).sample(6, seed=C.SEED)
          .select("business_name", "name_cons", "name_red"))
    print("\nAddresses and number tokens:")
    print(both.filter(pl.col("business_address").str.contains(r"\d")).sample(8, seed=C.SEED)
          .select("business_address", "addr_norm", "addr_nums"))


def main() -> None:
    train_raw = load_records("train")
    norm_train = normalized("train")
    norm_test = normalized("test")
    show_examples(train_raw, norm_train)

    print("\nFrance (test) examples:")
    test_raw = load_records("test")
    fr = test_raw.filter(pl.col("country") == "France").join(norm_test, on=["entity_id", "source", "country"])
    print(fr.sample(6, seed=C.SEED).select("business_name", "name_red", "business_address", "addr_norm", "addr_nums"))
    print(norm_test.group_by("country", "source").agg(n=pl.len(), addr_missing=pl.col("addr_missing").mean(),
                                                     has_numbers=(pl.col("addr_nums").list.len() > 0).mean())
          .sort("country", "source"))

    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    samples = manifest.drop_nulls("sample").select("s1_id", "sample")
    print(f"\nretrieving for {samples.height:,} training queries (4 samples)")
    t0 = time.time()
    cands, timings = retrieve(norm_train, samples["s1_id"])
    cands = cands.join(samples, on="s1_id", how="left")
    print(f"retrieval: {cands.height:,} candidates in {time.time() - t0:.0f}s")
    cands.write_parquet(C.WORK_DIR / "candidates_train.parquet")
    timings.write_csv(C.WORK_DIR / "retrieval_timings.tsv", separator="\t")
    with open(C.WORK_DIR / "retrieval_meta.json", "w", encoding="utf-8") as f:
        json.dump({"normalize_version": NORMALIZE_VERSION, "candidates": cands.height}, f)

    # same-country sanity check (by construction, but verify)
    country_of = norm_train.select(target_id="entity_id", t_country="country")
    s1_country = norm_train.select(s1_id="entity_id", q_country="country")
    bad = cands.join(country_of, on="target_id").join(s1_country, on="s1_id").filter(pl.col("t_country") != pl.col("q_country")).height
    print("candidates with a different country:", bad)

    warm = timings.filter(pl.col("index_build_s") == 0)
    if warm.height:
        print(f"warm-batch throughput: {warm['queries'].sum() / warm['query_s'].sum():,.0f} queries/s per (country, source) index")

    pairs = load_label_pairs()
    report = {}
    for sample in ("tune_sample", "fit_sample", "c_select_sample", "audit_panel"):
        ids = samples.filter(pl.col("sample") == sample)["s1_id"]
        report[sample] = candidate_report(cands, pairs, ids, norm_train)
    with open(C.WORK_DIR / "retrieval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print("\nretrieval report (Tune sample):")
    print(json.dumps(report["tune_sample"], indent=2, default=float))
    for sample, r in report.items():
        print(f"{sample}: recall {r['candidate_recall']:.4f}, oracle F0.5 {r['candidate_oracle']['macro_f05']:.4f}, "
              f"{r['mean_candidates_per_s1']:.1f} candidates/S1")


if __name__ == "__main__":
    main()
