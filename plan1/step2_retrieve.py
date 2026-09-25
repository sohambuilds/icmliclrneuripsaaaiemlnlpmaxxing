"""Step 2: build the Indian-script dictionary (Fit role only), normalize train and test, retrieve candidates for
the training samples, report candidate recall.

Run from the repository root:  python -m plan1.step2_retrieve
Outputs (under work/plan1/<run>/):
  translit.json               learned word dictionary + stats
  normalized_{train,test}_v<N>.parquet   (N = NORMALIZE_VERSION)
  candidates_train.parquet    (s1_id, target_id, source, country, score, rank, sample)
  retrieval_report.json, retrieval_timings.tsv, retrieval_meta.json
"""
import json
import time

import polars as pl

from . import config as C
from .dataio import load_label_pairs, load_records
from .normalize import NORMALIZE_VERSION, normalize_records
from .retrieve import candidate_report, retrieve
from .splits import fit_sample_ids, fresh_audit_panel
from .translit import build_dictionary, coverage, make_converter


def translit_map() -> dict:
    path = C.WORK_DIR / "translit.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["mapping"]
    t0 = time.time()
    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    fit_role = manifest.filter(pl.col("role") == "fit")["s1_id"]
    train = load_records("train")
    mapping, stats = build_dictionary(train, load_label_pairs(), fit_role)
    stats["coverage_train_s2s3"] = coverage(train, mapping)
    stats["coverage_test_s2s3"] = coverage(load_records("test"), mapping)
    C.WORK_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stats": stats, "mapping": mapping}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Indian-script dictionary built in {time.time() - t0:.0f}s:")
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, list)}))
    for key in ("dropped_most_frequent", "kept_by_sound_only"):
        print(f"\n{key} (word, mapped to, times, appearances, share, mean sim, mean spelling sim):")
        print(pl.DataFrame(stats[key]) if stats[key] else "  none")
    return mapping


def normalized(split: str) -> pl.DataFrame:
    path = C.WORK_DIR / f"normalized_{split}_v{NORMALIZE_VERSION}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    t0 = time.time()
    out = normalize_records(load_records(split), to_latin=make_converter(translit_map()))
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
    print("\nIndian-script names -> English letters:")
    print(both.filter(pl.col("name_nonlatin")).sample(10, seed=C.SEED).select("business_name", "name_cons", "name_red"))
    print("\nLatin names with legal forms / accents / websites:")
    print(both.filter(pl.col("business_name").str.contains(r"(?i)l\.l\.c|pvt|ltd|inc|www|\.com|[À-ÿ]")).sample(8, seed=C.SEED)
          .select("business_name", "name_cons", "name_red"))
    print("\nAddresses and number tokens:")
    print(both.filter(pl.col("business_address").str.contains(r"(?i)\d|n°|no\.")).sample(10, seed=C.SEED)
          .select("business_address", "addr_norm", "addr_nums"))


def main() -> None:
    train_raw = load_records("train")
    norm_train = normalized("train")
    norm_test = normalized("test")
    show_examples(train_raw, norm_train)

    print("\nFrance (test) examples:")
    test_raw = load_records("test")
    fr = test_raw.filter(pl.col("country") == "France").join(norm_test, on=["entity_id", "source", "country"])
    print(fr.sample(8, seed=C.SEED).select("business_name", "name_red", "business_address", "addr_norm", "addr_nums"))

    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    fit_ids = fit_sample_ids(manifest, C.FIT_SAMPLE_SIZE)
    samples = pl.concat([
        pl.DataFrame({"s1_id": fit_ids}).with_columns(sample=pl.lit("fit_sample")),
        manifest.filter(pl.col("sample").is_in(["tune_sample", "c_select_sample", "audit_panel"])).select("s1_id", "sample"),
        pl.DataFrame({"s1_id": fresh_audit_panel(manifest)}).with_columns(sample=pl.lit("audit_panel_2")),
    ])
    assert samples["s1_id"].n_unique() == samples.height
    print(f"\nretrieving for {samples.height:,} training queries ({fit_ids.len():,} fit + 4 x 20,000)")
    t0 = time.time()
    cands, timings = retrieve(norm_train, samples["s1_id"])
    cands = cands.join(samples, on="s1_id", how="left")
    print(f"retrieval: {cands.height:,} candidates in {time.time() - t0:.0f}s")
    cands.write_parquet(C.WORK_DIR / "candidates_train.parquet")
    timings.write_csv(C.WORK_DIR / "retrieval_timings.tsv", separator="\t")
    with open(C.WORK_DIR / "retrieval_meta.json", "w", encoding="utf-8") as f:
        json.dump({"normalize_version": NORMALIZE_VERSION, "candidates": cands.height, "fit_sample_size": fit_ids.len()}, f)

    country_of = norm_train.select(target_id="entity_id", t_country="country")
    s1_country = norm_train.select(s1_id="entity_id", q_country="country")
    bad = cands.join(country_of, on="target_id").join(s1_country, on="s1_id").filter(pl.col("t_country") != pl.col("q_country")).height
    print("candidates with a different country:", bad)

    pairs = load_label_pairs()
    report = {}
    for sample in ("tune_sample", "fit_sample", "c_select_sample", "audit_panel", "audit_panel_2"):
        ids = samples.filter(pl.col("sample") == sample)["s1_id"]
        report[sample] = candidate_report(cands, pairs, ids, norm_train)
    with open(C.WORK_DIR / "retrieval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print("\nretrieval report (Tune sample):")
    print(json.dumps(report["tune_sample"], indent=2, default=float))
    for sample, r in report.items():
        missing = r["recall_by_addr_missing"].get("True", {}).get("recall", float("nan"))
        print(f"{sample}: recall {r['candidate_recall']:.4f} (missing address {missing:.4f}, found only by fallback "
              f"{r['true_links_found_only_by_fallback']:.4f}), oracle F0.5 {r['candidate_oracle']['macro_f05']:.4f}, "
              f"{r['mean_candidates_per_s1']:.1f} candidates/S1")


if __name__ == "__main__":
    main()
