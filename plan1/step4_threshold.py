"""Step 4: choose one global threshold on C-select, freeze the configuration, score the Audit panel once.

Run from the repository root:  python -m plan1.step4_threshold
Outputs (under work/plan1/p1-baseline-v1/):
  c_select_sweep.tsv, frozen_config.json, audit_report.json, audit_predictions.parquet
"""
import hashlib
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config as C
from .dataio import load_label_pairs
from .features import FEATURES, matrix
from .metric import per_s1, score, summarize
from .normalize import NORMALIZE_VERSION
from .retrieve import TFIDF_PARAMS, TOP_K, candidate_report
from .step3_train import labelled_features


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def predict(booster: lgb.Booster, cands: pl.DataFrame, pairs: pl.DataFrame, norm: pl.DataFrame, name: str) -> pl.DataFrame:
    feats = labelled_features(cands, pairs, norm, name)
    return feats.select("s1_id", "target_id", "source", "label", *FEATURES).with_columns(
        p=pl.Series(booster.predict(matrix(feats)), dtype=pl.Float64)
    )


def sweep(pred: pl.DataFrame, truth: pl.DataFrame, ids: pl.Series, thresholds) -> pl.DataFrame:
    rows = [{"threshold": float("inf"), **score(pred.head(0), truth, ids)}]  # the empty-all option
    for thr in thresholds:
        rows.append({"threshold": float(thr), **score(pred.filter(pl.col("p") >= thr), truth, ids)})
    return pl.DataFrame(rows)


def pick(table: pl.DataFrame) -> dict:
    """Highest macro F0.5; on an exact tie the more conservative (higher) threshold."""
    return table.sort(["macro_f05", "threshold"], descending=[True, True]).row(0, named=True)


def main() -> None:
    meta = json.loads((C.WORK_DIR / "retrieval_meta.json").read_text(encoding="utf-8"))
    if meta["normalize_version"] != NORMALIZE_VERSION:
        raise RuntimeError("candidates are from an older normalization: rerun steps 2 and 3")
    model_meta = json.loads((C.WORK_DIR / "model_meta.json").read_text(encoding="utf-8"))
    assert model_meta["features"] == FEATURES and model_meta["normalize_version"] == NORMALIZE_VERSION

    norm = pl.read_parquet(C.WORK_DIR / f"normalized_train_v{NORMALIZE_VERSION}.parquet")
    cands = pl.read_parquet(C.WORK_DIR / "candidates_train.parquet")
    pairs = load_label_pairs()
    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    booster = lgb.Booster(model_file=str(C.WORK_DIR / "model.txt"))

    # ---- threshold on C-select ----
    cs_ids = manifest.filter(pl.col("sample") == "c_select_sample")["s1_id"]
    cs_truth = pairs.join(pl.DataFrame({"s1_id": cs_ids}), on="s1_id", how="semi")
    cs = predict(booster, cands.join(pl.DataFrame({"s1_id": cs_ids}), on="s1_id", how="semi"), pairs, norm, "c_select")
    coarse = sweep(cs, cs_truth, cs_ids, np.round(np.arange(0.01, 1.0, 0.01), 2))
    best = pick(coarse)
    if np.isfinite(best["threshold"]):
        lo, hi = max(best["threshold"] - 0.01, 0.0), min(best["threshold"] + 0.01, 1.0)
        fine = sweep(cs, cs_truth, cs_ids, np.round(np.arange(lo, hi + 1e-9, 0.0005), 4))
        table = pl.concat([coarse, fine]).unique("threshold", keep="first").sort("threshold")
    else:
        table = coarse
    best = pick(table)
    thr = best["threshold"]
    table.write_csv(C.WORK_DIR / "c_select_sweep.tsv", separator="\t")
    print("\nC-select sweep (every 0.05):")
    print(coarse.filter((pl.col("threshold") * 100).round(0) % 5 == 0).select(
        "threshold", "macro_f05", "link_precision", "link_recall", "singleton_false_positive_rate", "mean_predicted_per_s1"))
    print(f"chosen threshold {thr} -> C-select macro F0.5 {best['macro_f05']:.4f}")

    # ---- freeze ----
    frozen = {
        "run_id": C.RUN_ID,
        "threshold": thr,
        "c_select_macro_f05": best["macro_f05"],
        "model_sha256": sha256(C.WORK_DIR / "model.txt"),
        "manifest_sha256": sha256(C.SPLIT_DIR / "manifest.parquet"),
        "normalize_version": NORMALIZE_VERSION,
        "features": FEATURES,
        "lgb_params": C.LGB_PARAMS,
        "best_iteration": model_meta["best_iteration"],
        "tfidf_params": {k: (v.__name__ if callable(v) else str(v) if k == "dtype" else v) for k, v in TFIDF_PARAMS.items()},
        "top_k_per_source": TOP_K,
        "lightgbm": lgb.__version__,
        "polars": pl.__version__,
    }
    path = C.WORK_DIR / "frozen_config.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["threshold"] != thr or old["model_sha256"] != frozen["model_sha256"]:
            raise RuntimeError(f"a different configuration is already frozen in {path}; move it away deliberately to refreeze")
    path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print("frozen:", path)

    # ---- score the Audit panel once with the frozen configuration ----
    au_ids = manifest.filter(pl.col("sample") == "audit_panel")["s1_id"]
    au_truth = pairs.join(pl.DataFrame({"s1_id": au_ids}), on="s1_id", how="semi")
    au_cands = cands.join(pl.DataFrame({"s1_id": au_ids}), on="s1_id", how="semi")
    t0 = time.time()
    au = predict(booster, au_cands, pairs, norm, "audit")
    score_s = time.time() - t0
    accepted = au.filter(pl.col("p") >= thr)
    headline = score(accepted, au_truth, au_ids)

    countries = manifest.select("s1_id", "country")
    per = per_s1(accepted, au_truth, au_ids).join(countries, on="s1_id")
    by_country = {c: summarize(g.drop("country")) for (c,), g in per.group_by("country")}
    retrieval = candidate_report(au_cands, pairs, au_ids, norm)
    collisions = accepted.group_by("target_id").agg(n=pl.col("s1_id").n_unique())
    rule = au.filter((pl.col("name_red_ratio") == 1.0) & (pl.col("num_jaccard") > 0))  # simple name + number rule, same candidates
    timings = pl.read_csv(C.WORK_DIR / "retrieval_timings.tsv", separator="\t")

    report = {
        "frozen_threshold": thr,
        "audit": headline,
        "audit_by_country": by_country,
        "candidate_recall": retrieval["candidate_recall"],
        "candidate_oracle_macro_f05": retrieval["candidate_oracle"]["macro_f05"],
        "retrieval_recall_by_country": retrieval["recall_by_country"],
        "retrieval_recall_indic_names": retrieval["recall_by_name_nonlatin"],
        "retrieval_recall_missing_address": retrieval["recall_by_addr_missing"],
        "mean_candidates_per_s1": retrieval["mean_candidates_per_s1"],
        "targets_accepted_by_multiple_s1": int((collisions["n"] > 1).sum()),
        "accepted_targets": collisions.height,
        "baseline_empty_all_macro_f05": score(au.head(0), au_truth, au_ids)["macro_f05"],
        "baseline_exact_name_plus_number_rule_macro_f05": score(rule, au_truth, au_ids)["macro_f05"],
        "costs_seconds": {
            "retrieval_index_build_total": float(timings["index_build_s"].sum()),
            "retrieval_query_total_160k_queries": float(timings["query_s"].sum()),
            "model_training": model_meta["train_seconds"],
            "audit_features_and_scoring_20k_s1": score_s,
        },
    }
    (C.WORK_DIR / "audit_report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    au.select("s1_id", "target_id", "source", "label", "p").write_parquet(C.WORK_DIR / "audit_predictions.parquet")

    print("\n==== Audit panel (frozen, scored once) ====")
    print(json.dumps(report, indent=2, default=float))
    if headline["macro_f05"] <= report["baseline_empty_all_macro_f05"]:
        raise RuntimeError("the learned pipeline does not beat predicting empty for every S1: something is broken")


if __name__ == "__main__":
    main()
