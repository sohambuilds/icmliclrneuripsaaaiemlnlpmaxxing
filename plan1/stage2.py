"""Second stage: decide per business, using the first-stage scores in context.

Why: search finds ~98.9% of true links but the first-stage model, which judges one pair at a time, accepts only
~90%. The second stage sees each candidate together with the rest of its business's list:
  - first-stage score context: its score, rank (overall / in its source), share of the business's best, second-best
    score, how many candidates score >= 0.5 / 0.8, sum of scores, best score in the other source;
  - duplicates: similarity to the business's likely records (first-stage score >= STAGE2_ANCHOR_P), the best score
    among near-identical likely records, and how many there are;
  - frequency, counted within each split: how common the S1's name is among S1s, and the record's name and address
    among S2/S3 of the same country (generic names need address evidence).
First-stage scores for the Fit sample come from STAGE2_FOLDS models that never saw those businesses (out of fold);
Tune / C-select / panels / test use the full first-stage model. Candidates are reused: no new search.

  python -m plan1.stage2 train   needs steps 2-4 of the run. Threshold on C-select; reports dev + fresh panels.
  python -m plan1.stage2 test    needs step 5's test_chunks. Writes output/plan1/<run>-s2/ (restartable).
Backend: AMC_BACKEND=lightgbm (default; reuses the run's first-stage model) or xgboost (GPU, retrains stage 1 too).
"""
import json
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from . import config as C
from .dataio import load_label_pairs, load_records, read_id_lists, test_s1_roster, write_id_lists
from .decide import one_owner
from .features import FEATURES, build_features, matrix, text_lookup
from .metric import per_s1, score, summarize
from .model import best_iteration, importance, load_model, predict, save_model, train_model
from .normalize import NORMALIZE_VERSION
from .splits import fit_sample_ids, fresh_audit_panel, stable_unit
from .step2_retrieve import normalized
from .step3_train import labelled_features
from .step4_threshold import pick, sweep

S2_DIR = C.WORK_DIR / "stage2"
TEST_DIR = S2_DIR / "test_scores"
OUT = C.ROOT / "output" / "plan1" / f"{C.RUN_ID}-s2"
F32 = pl.Float32
SIB_BATCH = 20_000  # S1 per batch for the duplicate features
STAGE2_FEATURES = [
    "p1", "p1_rank", "p1_rank_src", "p1_max", "p1_second", "p1_rel", "p1_gap", "n_p50", "n_p80", "p1_sum",
    "p1_max_other_src",
    "sib_name", "sib_addr", "sib_both", "sib_p", "n_sib",
    "q_name_s1_rate", "t_name_rate", "t_addr_rate",
]
ALL_FEATURES = FEATURES + STAGE2_FEATURES


# ---------------------------------------------------------------- features
def frequency_rates(norm: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per 100k records of the same split and country: S1 name among S1s; record name / address among S2+S3."""
    s1 = norm.filter(pl.col("source") == "S1")
    tg = norm.filter(pl.col("source") != "S1")
    s1_tot = s1.group_by("country").agg(n=pl.len())
    tg_tot = tg.group_by("country").agg(n=pl.len())
    rate = (pl.col("c") / pl.col("n") * 1e5).cast(F32)
    q = s1.group_by("country", "name_red").agg(c=pl.len()).join(s1_tot, on="country").select("country", "name_red", q_name_s1_rate=rate)
    tn = tg.group_by("country", "name_red").agg(c=pl.len()).join(tg_tot, on="country").select("country", "name_red", t_name_rate=rate)
    ta = (tg.filter(pl.col("addr_norm") != "").group_by("country", "addr_norm").agg(c=pl.len()).join(tg_tot, on="country")
          .select("country", "addr_norm", t_addr_rate=rate))
    s1_rates = s1.select(s1_id="entity_id", country="country", name_red="name_red").join(q, on=["country", "name_red"], how="left").select("s1_id", "q_name_s1_rate")
    t_rates = (tg.select(target_id="entity_id", country="country", name_red="name_red", addr_norm="addr_norm")
               .join(tn, on=["country", "name_red"], how="left").join(ta, on=["country", "addr_norm"], how="left")
               .select("target_id", "t_name_rate", "t_addr_rate"))
    return s1_rates, t_rates


def _sibling_features(df: pl.DataFrame, txt: pl.DataFrame) -> pl.DataFrame:
    """df: s1_id, target_id, p1 for whole S1s. Similarity of each candidate to the S1's likely records."""
    anchors = df.filter(pl.col("p1") >= C.STAGE2_ANCHOR_P).select("s1_id", anchor_id="target_id", anchor_p="p1")
    pr = df.select("s1_id", "target_id").join(anchors, on="s1_id").filter(pl.col("target_id") != pl.col("anchor_id"))
    if pr.height == 0:
        return pl.DataFrame(schema={"s1_id": pl.String, "target_id": pl.String, "sib_name": F32, "sib_addr": F32,
                                    "sib_both": F32, "sib_p": F32, "n_sib": F32})
    pr = pr.join(txt, on="target_id").join(txt.rename({"target_id": "anchor_id", "n": "an", "a": "aa"}), on="anchor_id")
    ns = process.cpdist(pr["n"].to_list(), pr["an"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
    ad = process.cpdist(pr["a"].to_list(), pr["aa"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
    ad[((pr["a"] == "") | (pr["aa"] == "")).to_numpy()] = np.nan
    pr = (pr.select("s1_id", "target_id", "anchor_p")
          .with_columns(ns=pl.Series(ns), ad=pl.Series(ad).fill_nan(None))
          .with_columns(both=pl.min_horizontal("ns", "ad")))  # address missing on either side -> name only
    return pr.group_by("s1_id", "target_id").agg(
        sib_name=pl.col("ns").max().cast(F32), sib_addr=pl.col("ad").max().cast(F32), sib_both=pl.col("both").max().cast(F32),
        sib_p=pl.col("anchor_p").filter(pl.col("both") >= 0.9).max().cast(F32),
        n_sib=(pl.col("both") >= 0.9).sum().cast(F32),
    )


def add_stage2_features(feats: pl.DataFrame, txt: pl.DataFrame, s1_rates: pl.DataFrame, t_rates: pl.DataFrame) -> pl.DataFrame:
    """feats: s1_id, target_id, source, FEATURES..., p1 (all candidates of each S1 present). txt: target_id, n, a."""
    p = pl.col("p1")
    out = feats.with_columns(p1=p.cast(F32)).with_columns(
        p1_rank=p.rank("ordinal", descending=True).over("s1_id").cast(F32),
        p1_rank_src=p.rank("ordinal", descending=True).over("s1_id", "source").cast(F32),
        p1_max=p.max().over("s1_id").cast(F32),
        p1_second=pl.when(pl.len().over("s1_id") > 1).then(p.top_k(2).min().over("s1_id")).cast(F32),
        n_p50=(p >= 0.5).sum().over("s1_id").cast(F32),
        n_p80=(p >= 0.8).sum().over("s1_id").cast(F32),
        p1_sum=p.sum().over("s1_id").cast(F32),
    ).with_columns(p1_rel=(p / pl.col("p1_max")).cast(F32), p1_gap=(pl.col("p1_max") - p).cast(F32))
    other = (out.group_by("s1_id", "source").agg(p1_max_other_src=p.max().cast(F32))
             .with_columns(source=pl.when(pl.col("source") == "S2").then(pl.lit("S3")).otherwise(pl.lit("S2"))))
    out = out.join(other, on=["s1_id", "source"], how="left", maintain_order="left")

    s1_list = out["s1_id"].unique(maintain_order=True)
    sib = pl.concat([
        _sibling_features(out.join(pl.DataFrame({"s1_id": s1_list.slice(i, SIB_BATCH)}), on="s1_id", how="semi")
                          .select("s1_id", "target_id", "p1"), txt)
        for i in range(0, s1_list.len(), SIB_BATCH)
    ])
    out = (out.join(sib, on=["s1_id", "target_id"], how="left", maintain_order="left")
           .with_columns(pl.col("n_sib").fill_null(0))
           .join(s1_rates, on="s1_id", how="left", maintain_order="left")
           .join(t_rates, on="target_id", how="left", maintain_order="left"))
    return out


def text_table(norm: pl.DataFrame, target_ids: pl.Series) -> pl.DataFrame:
    return (norm.join(pl.DataFrame({"entity_id": target_ids.unique()}), on="entity_id", how="semi")
            .select(target_id="entity_id", n="name_red", a="addr_norm"))


# ---------------------------------------------------------------- helpers
def choose_threshold(pred: pl.DataFrame, truth: pl.DataFrame, ids: pl.Series) -> dict:
    coarse = sweep(pred, truth, ids, np.round(np.arange(0.01, 1.0, 0.01), 2))
    best = pick(coarse)
    if np.isfinite(best["threshold"]):
        lo, hi = max(best["threshold"] - 0.01, 0.0), min(best["threshold"] + 0.01, 1.0)
        best = pick(pl.concat([coarse, sweep(pred, truth, ids, np.round(np.arange(lo, hi + 1e-9, 0.0005), 4))]))
    return best


def panel_scores(pred: pl.DataFrame, truth: pl.DataFrame, ids: pl.Series, countries: pl.DataFrame) -> dict:
    s = score(pred, truth, ids)
    per = per_s1(pred, truth, ids).join(countries, on="s1_id")
    by_c = {c: summarize(g.drop("country"))["macro_f05"] for (c,), g in per.group_by("country")}
    return {"macro_f05": s["macro_f05"], "US": by_c.get("US"), "India": by_c.get("India"), "precision": s["link_precision"],
            "recall": s["link_recall"], "singleton_fp": s["singleton_false_positive_rate"], "links": s["predicted_links"]}


# ---------------------------------------------------------------- train
def main_train() -> None:
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(220)
    meta = json.loads((C.WORK_DIR / "retrieval_meta.json").read_text(encoding="utf-8"))
    assert meta["normalize_version"] == NORMALIZE_VERSION, "candidates are from an older normalization"
    frozen = json.loads((C.WORK_DIR / "frozen_config.json").read_text(encoding="utf-8"))
    backend = C.MODEL_BACKEND
    S2_DIR.mkdir(parents=True, exist_ok=True)
    norm = pl.read_parquet(C.WORK_DIR / f"normalized_train_v{NORMALIZE_VERSION}.parquet")
    cands = pl.read_parquet(C.WORK_DIR / "candidates_train.parquet")
    pairs = load_label_pairs()
    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    countries = manifest.select("s1_id", "country")
    ids = {
        "fit": fit_sample_ids(manifest, C.FIT_SAMPLE_SIZE),
        "tune": manifest.filter(pl.col("sample") == "tune_sample")["s1_id"],
        "c_select": manifest.filter(pl.col("sample") == "c_select_sample")["s1_id"],
        "dev": manifest.filter(pl.col("sample") == "audit_panel")["s1_id"],
        "fresh": fresh_audit_panel(manifest),
    }
    truth = {k: pairs.join(pl.DataFrame({"s1_id": v}), on="s1_id", how="semi") for k, v in ids.items()}

    feats = {}
    for k, v in ids.items():
        path = S2_DIR / f"stage1_features_{k}.parquet"
        if path.exists():
            feats[k] = pl.read_parquet(path)
        else:
            feats[k] = labelled_features(cands.join(pl.DataFrame({"s1_id": v}), on="s1_id", how="semi"), pairs, norm, k)
            feats[k].write_parquet(path)
    y = {k: v["label"].to_numpy() for k, v in feats.items()}
    x_fit, x_tune = matrix(feats["fit"]), matrix(feats["tune"])

    # ---- first stage: full model + out-of-fold scores for Fit ----
    if backend == "lightgbm" and (C.WORK_DIR / "model.txt").exists():
        stage1_path = C.WORK_DIR / "model.txt"
        stage1 = load_model(stage1_path)
        print(f"first stage: reusing {stage1_path} (LightGBM)")
    else:
        stage1, secs = train_model(x_fit, y["fit"], x_tune, y["tune"], FEATURES, backend)
        stage1_path = save_model(stage1, S2_DIR / "stage1_full")
        print(f"first stage ({backend}) trained in {secs:.0f}s, best iteration {best_iteration(stage1)}")
    fit_s1 = feats["fit"]["s1_id"]
    uniq = fit_s1.unique()
    fold_of = pl.DataFrame({"s1_id": uniq, "fold": (stable_unit(uniq.to_list(), C.STAGE2_FOLD_SALT) * C.STAGE2_FOLDS).astype(np.int32)})
    fold = pl.DataFrame({"s1_id": fit_s1}).join(fold_of, on="s1_id", how="left", maintain_order="left")["fold"].to_numpy()
    p_oof = np.full(len(fold), np.nan)
    for k in range(C.STAGE2_FOLDS):
        tr = fold != k
        m, secs = train_model(x_fit[tr], y["fit"][tr], x_tune, y["tune"], FEATURES, backend)
        p_oof[~tr] = predict(m, x_fit[~tr])
        print(f"out-of-fold model {k + 1}/{C.STAGE2_FOLDS}: {secs:.0f}s, best iteration {best_iteration(m)}")
    assert not np.isnan(p_oof).any()
    p1 = {"fit": p_oof, **{k: predict(stage1, matrix(feats[k])) for k in ("tune", "c_select", "dev", "fresh")}}
    del x_fit

    # ---- second-stage features ----
    t0 = time.time()
    s1_rates, t_rates = frequency_rates(norm)
    s2 = {}
    for k in feats:
        f = feats[k].with_columns(p1=pl.Series(p1[k]))
        s2[k] = add_stage2_features(f, text_table(norm, f["target_id"]), s1_rates, t_rates)
    print(f"second-stage features in {time.time() - t0:.0f}s")

    # ---- second-stage model ----
    model2, secs2 = train_model(matrix(s2["fit"], ALL_FEATURES), y["fit"], matrix(s2["tune"], ALL_FEATURES), y["tune"],
                                ALL_FEATURES, backend)
    print(f"second stage ({backend}) trained in {secs2:.0f}s, best iteration {best_iteration(model2)}")
    p2 = {k: predict(model2, matrix(s2[k], ALL_FEATURES)) for k in ("c_select", "dev", "fresh")}
    pred = {k: s2[k].select("s1_id", "target_id").with_columns(p=pl.Series(p2[k])) for k in p2}
    best = choose_threshold(pred["c_select"], truth["c_select"], ids["c_select"])
    thr2 = best["threshold"]
    print(f"second-stage threshold {thr2} -> C-select F0.5 {best['macro_f05']:.4f}")

    rows = []
    for k in ("dev", "fresh"):
        one = s2[k].select("s1_id", "target_id", p="p1")
        rows.append({"panel": k, "stage": "1 (frozen p1-v3)", **panel_scores(one.filter(pl.col("p") >= frozen["threshold"]), truth[k], ids[k], countries)})
        rows.append({"panel": k, "stage": "2", **panel_scores(pred[k].filter(pl.col("p") >= thr2), truth[k], ids[k], countries)})
    report = pl.DataFrame(rows)
    print("\nstage 1 vs stage 2 (dev = first audit panel, fresh = audit_panel_2):")
    print(report)
    imp = importance(model2, ALL_FEATURES)
    print(pl.DataFrame({"feature": list(imp), "gain": list(imp.values())})
          .with_columns(share=pl.col("gain") / pl.col("gain").sum()).sort("gain", descending=True).head(25))

    model2_path = save_model(model2, S2_DIR / "stage2")
    (S2_DIR / "stage2_meta.json").write_text(json.dumps({
        "backend": backend, "stage1_model": str(stage1_path), "stage1_is_run_model": stage1_path == C.WORK_DIR / "model.txt",
        "stage2_model": str(model2_path), "threshold": thr2, "c_select_f05": best["macro_f05"],
        "stage1_features": FEATURES, "stage2_features": STAGE2_FEATURES, "folds": C.STAGE2_FOLDS,
        "fold_salt": C.STAGE2_FOLD_SALT, "anchor_p": C.STAGE2_ANCHOR_P, "report": report.to_dicts(),
        "normalize_version": NORMALIZE_VERSION,
    }, indent=2, default=float), encoding="utf-8")
    report.write_csv(S2_DIR / "stage2_report.tsv", separator="\t")
    print("saved:", model2_path)


# ---------------------------------------------------------------- test
def main_test() -> None:
    meta = json.loads((S2_DIR / "stage2_meta.json").read_text(encoding="utf-8"))
    assert meta["normalize_version"] == NORMALIZE_VERSION
    stage1 = load_model(meta["stage1_model"])
    model2 = load_model(meta["stage2_model"])
    thr2 = meta["threshold"]
    chunk_dir = C.WORK_DIR / "test_chunks"
    files = sorted(chunk_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"no test chunks in {chunk_dir}: run step 5 first")
    norm = normalized("test")
    s1_rates, t_rates = frequency_rates(norm)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    for i, f in enumerate(files, 1):
        out_path = TEST_DIR / f.name
        if out_path.exists():
            continue
        t1 = time.time()
        c = pl.read_parquet(f)
        if c.height == 0:
            pl.DataFrame(schema={"s1_id": pl.String, "target_id": pl.String, "source": pl.String, "p1": F32, "p": F32}).write_parquet(out_path)
            continue
        q, t = text_lookup(norm, c["s1_id"], c["target_id"])
        feats = build_features(c.select("s1_id", "target_id", "source", "score", "rank", "fb_score"), q, t)
        if meta["stage1_is_run_model"]:
            p1 = c["p"].to_numpy()  # step 5 already scored these with the same model
            if i == 1:  # check that the rebuilt features reproduce step 5's scores
                diff = float(np.abs(predict(stage1, matrix(feats)) - p1).max())
                print(f"check: rebuilt features reproduce step-5 scores (max difference {diff:.2e})")
                if diff > 1e-3:
                    raise RuntimeError("rebuilt features do not match step 5: code or data changed since step 5")
        else:
            p1 = predict(stage1, matrix(feats))
        feats = feats.with_columns(p1=pl.Series(p1, dtype=F32))
        s2f = add_stage2_features(feats, text_table(norm, feats["target_id"]), s1_rates, t_rates)
        p2 = predict(model2, matrix(s2f, ALL_FEATURES))
        tmp = out_path.with_suffix(".tmp")
        s2f.select("s1_id", "target_id", "source", "p1").with_columns(p=pl.Series(p2, dtype=F32)).write_parquet(tmp)
        tmp.replace(out_path)
        print(f"  {i}/{len(files)} {f.name}: {c.height:,} candidates, {int((p2 >= thr2).sum()):,} accepted, "
              f"{time.time() - t1:.0f}s (elapsed {(time.time() - t_all) / 60:.1f} min)")

    scores = pl.read_parquet(str(TEST_DIR / "*.parquet"))
    roster = test_s1_roster()
    assert scores.join(pl.DataFrame({"s1_id": roster}), on="s1_id", how="anti").height == 0
    accepted = scores.filter(pl.col("p") >= thr2)
    matches = one_owner(accepted)
    print(f"\naccepted {accepted.height:,}; one owner per record keeps {matches.height:,}")
    write_id_lists(scores, roster, OUT / "candidate_pairs.tsv", "candidate_entity_ids")
    write_id_lists(matches, roster, OUT / "matching_results.tsv", "matched_entity_ids")
    back_m = read_id_lists(OUT / "matching_results.tsv", "matched_entity_ids")
    back_c = read_id_lists(OUT / "candidate_pairs.tsv", "candidate_entity_ids")
    assert back_m.join(back_c, on=["s1_id", "target_id"], how="anti").height == 0, "a match is not in candidates"
    s1c = load_records("test").filter(pl.col("source") == "S1").select(s1_id="entity_id", country="country")
    per = (s1c.join(accepted.group_by("s1_id").agg(n_acc=pl.len()), on="s1_id", how="left")
           .join(matches.group_by("s1_id").agg(n=pl.len()), on="s1_id", how="left").with_columns(pl.col("n_acc", "n").fill_null(0)))
    print(per.group_by("country").agg(s1=pl.len(), mean_accepted=pl.col("n_acc").mean(), mean_matches=pl.col("n").mean(),
                                      empty_answer_rate=(pl.col("n") == 0).mean()).sort("country"))
    print("written:", OUT / "matching_results.tsv", OUT / "candidate_pairs.tsv", sep="\n  ")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "test"):
        raise SystemExit("usage: python -m plan1.stage2 train | test")
    main_train() if sys.argv[1] == "train" else main_test()
