"""Second stage v2 ("s2ce"): features v3 (French legal forms kept apart, cleaner addresses), a cross-encoder score
as a feature, and twice the second-stage training data. Every model trains and predicts on the GPU (XGBoost and the
transformer).

  python -m plan1.stage2ce stage1   features v3 for every set; first-stage model on Fit (+ 3 out-of-fold models);
                                    first-stage context features; keeps candidates with p1 >= S2_MIN_P1 (top S2_TOP)
  python -m plan1.cross_encoder train      (GPU; can start as soon as stage1 prints "ce_train: ...")
  python -m plan1.cross_encoder score      (GPU)
  python -m plan1.stage2ce train    second stage with and without the cross-encoder; thresholds on C-select;
                                    dev + fresh panels
  python -m plan1.stage2ce test     step 5's test chunks -> output/plan1/<run>-s2ce/ and <run>-s2ce-noce/

Sets:
  fit        300k; out-of-fold p1
  gbm_extra  300k more Fit-role S1; p1 from the full first-stage model, exactly as on test
  ce_train   300k; only trains the cross-encoder
  tune       early stopping
  c_select   threshold
  dev, fresh report
The second stage trains on fit + gbm_extra. It only sees candidates with p1 >= S2_MIN_P1; the rest are rejected
(their true links are counted as missed in every score).
Run everything with AMC_BACKEND=xgboost.
"""
import json
import sys
import time

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from . import config as C
from .cross_encoder import CE_DIR, S2_DIR, load as load_ce, pair_texts, record_text, score as ce_score
from .dataio import load_label_pairs, load_records, read_id_lists, test_s1_roster, write_id_lists
from .decide import one_owner
from .enrich import FEATURES_VERSION, enriched
from .features import FEATURES, build_features, matrix, text_lookup
from .model import best_iteration, importance, load_model, predict, save_model, train_model
from .normalize import NORMALIZE_VERSION
from .retrieve_extra import extra_ids
from .splits import fit_sample_ids, fresh_audit_panel, stable_unit
from .stage2 import STAGE2_FEATURES, add_stage2_features, choose_threshold, panel_scores, text_table
from .step3_train import labelled_features

TEST_DIR = S2_DIR / "test_scores"
OUT = {"ce": C.ROOT / "output" / "plan1" / f"{C.RUN_ID}-s2ce", "noce": C.ROOT / "output" / "plan1" / f"{C.RUN_ID}-s2ce-noce"}
F32 = pl.Float32
CE_FEATURES = ["ce", "ce_rank", "ce_rank_src", "ce_max", "ce_second", "ce_gap", "ce_n_pos", "ce_max_other_src"]
TWIN_FEATURES = ["twin_n", "twin_num_better", "twin_leaner", "twin_same_num", "twin_p_max"]
FEATS = {"noce": FEATURES + STAGE2_FEATURES + TWIN_FEATURES, "ce": FEATURES + STAGE2_FEATURES + TWIN_FEATURES + CE_FEATURES}
SETS = ("fit", "gbm_extra", "ce_train", "tune", "c_select", "dev", "fresh")


# ---------------------------------------------------------------- helpers
def set_ids() -> dict[str, pl.Series]:
    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    extra = extra_ids(manifest)
    return {
        "fit": fit_sample_ids(manifest, C.FIT_SAMPLE_SIZE),
        "gbm_extra": extra.filter(pl.col("part") == "gbm_extra")["s1_id"],
        "ce_train": extra.filter(pl.col("part") == "ce_train")["s1_id"],
        "tune": manifest.filter(pl.col("sample") == "tune_sample")["s1_id"],
        "c_select": manifest.filter(pl.col("sample") == "c_select_sample")["s1_id"],
        "dev": manifest.filter(pl.col("sample") == "audit_panel")["s1_id"],
        "fresh": fresh_audit_panel(manifest),
    }


def stage1_features(name: str, ids: pl.Series, pairs: pl.DataFrame, norm: pl.DataFrame) -> pl.DataFrame:
    path = S2_DIR / f"stage1_features_{name}_f{FEATURES_VERSION}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    src = C.WORK_DIR / ("candidates_extra.parquet" if name in ("gbm_extra", "ce_train") else "candidates_train.parquet")
    waited = False
    while True:  # the extra candidates may still be written by plan1.retrieve_extra (complete once its footer reads)
        try:
            pl.read_parquet_schema(src)
            break
        except Exception:
            if not waited:
                print(f"waiting for {src.name} (is plan1.retrieve_extra still running?) ...", flush=True)
                waited = True
            time.sleep(60)
    cands = (pl.scan_parquet(src).join(pl.LazyFrame({"s1_id": ids}), on="s1_id", how="semi")
             .select("s1_id", "target_id", "source", "score", "rank", "fb_score").collect())
    f = labelled_features(cands, pairs, norm, name)
    f.write_parquet(path)
    return f


def keep(df: pl.DataFrame) -> pl.DataFrame:
    """Candidates the second stage and the cross-encoder look at (df has p1 and p1_rank over whole S1s)."""
    return df.filter((pl.col("p1") >= C.S2_MIN_P1) & (pl.col("p1_rank") <= C.S2_TOP))


def add_ce_features(df: pl.DataFrame) -> pl.DataFrame:
    """df: the kept candidates with the cross-encoder logit `ce` (all kept candidates of each S1 present)."""
    c = pl.col("ce")
    out = df.with_columns(
        ce_rank=c.rank("ordinal", descending=True).over("s1_id").cast(F32),
        ce_rank_src=c.rank("ordinal", descending=True).over("s1_id", "source").cast(F32),
        ce_max=c.max().over("s1_id").cast(F32),
        ce_second=pl.when(pl.len().over("s1_id") > 1).then(c.top_k(2).min().over("s1_id")).cast(F32),
        ce_n_pos=(c > 0).sum().over("s1_id").cast(F32),
    ).with_columns(ce_gap=(pl.col("ce_max") - c).cast(F32))
    other = (out.group_by("s1_id", "source").agg(ce_max_other_src=c.max().cast(F32))
             .with_columns(source=pl.when(pl.col("source") == "S2").then(pl.lit("S3")).otherwise(pl.lit("S2"))))
    return out.join(other, on=["s1_id", "source"], how="left", maintain_order="left")


def add_twin_features(df: pl.DataFrame, norm: pl.DataFrame) -> pl.DataFrame:
    """Twins of a candidate: the S1's likely records (p1 >= STAGE2_ANCHOR_P) whose name (token set >= 0.9) and street
    (address without numbers, token set >= 0.9, or neither has an address) are near-identical to it. A planted copy
    sits next to the real record and differs in the number or adds a word. Whether a better twin exists does not
    depend on how many copies there are, and the test has several times more copies than train.
      twin_num_better  a twin has the S1's house number and this candidate does not
      twin_leaner      a twin has this candidate's name words minus some (this candidate adds words)
      twin_same_num    twins with this candidate's number (duplicates within a source)
      twin_n, twin_p_max
    """
    street = pl.col("addr_norm").fill_null("").str.replace_all(r"\S*\d\S*", " ").str.replace_all(r"\s+", " ").str.strip_chars()
    txt = (norm.join(pl.DataFrame({"entity_id": pl.concat([df["s1_id"], df["target_id"]]).unique()}), on="entity_id", how="semi")
           .select("entity_id", nm="name_red", st=street, hn="hn", nt="name_ntok"))
    anchors = df.filter(pl.col("p1") >= C.STAGE2_ANCHOR_P).select("s1_id", tw_id="target_id", tw_p="p1")
    pr = (df.select("s1_id", "target_id").join(anchors, on="s1_id").filter(pl.col("target_id") != pl.col("tw_id"))
          .join(txt.select(target_id="entity_id", t_nm="nm", t_st="st", t_hn="hn", t_nt="nt"), on="target_id")
          .join(txt.select(tw_id="entity_id", w_nm="nm", w_st="st", w_hn="hn", w_nt="nt"), on="tw_id")
          .join(txt.select(s1_id="entity_id", q_hn="hn"), on="s1_id", how="left"))
    out = pl.DataFrame(schema={"s1_id": pl.String, "target_id": pl.String, **{f: F32 for f in TWIN_FEATURES}})
    if pr.height:
        ns = process.cpdist(pr["t_nm"].to_list(), pr["w_nm"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
        ss = process.cpdist(pr["t_st"].to_list(), pr["w_st"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
        street_ok = (((pl.col("ss") >= 0.9) & (pl.col("t_st") != "") & (pl.col("w_st") != ""))
                     | ((pl.col("t_st") == "") & (pl.col("w_st") == "")))
        pr = pr.with_columns(ns=pl.Series(ns), ss=pl.Series(ss)).filter((pl.col("ns") >= 0.9) & street_ok)
        out = pr.group_by("s1_id", "target_id").agg(
            twin_n=pl.len().cast(F32),
            twin_num_better=((pl.col("w_hn") == pl.col("q_hn")) & (pl.col("t_hn") != pl.col("q_hn"))).fill_null(False).any().cast(F32),
            twin_leaner=((pl.col("ns") >= 0.999) & (pl.col("w_nt") < pl.col("t_nt"))).fill_null(False).any().cast(F32),
            twin_same_num=(pl.col("w_hn") == pl.col("t_hn")).fill_null(False).sum().cast(F32),
            twin_p_max=pl.col("tw_p").max().cast(F32),
        )
    return (df.join(out, on=["s1_id", "target_id"], how="left", maintain_order="left")
            .with_columns(pl.col("twin_n", "twin_num_better", "twin_leaner", "twin_same_num").fill_null(0)))


# ---------------------------------------------------------------- stage 1
def main_stage1() -> None:
    meta = json.loads((C.WORK_DIR / "retrieval_meta.json").read_text(encoding="utf-8"))
    assert meta["normalize_version"] == NORMALIZE_VERSION, "candidates are from an older normalization"
    assert C.MODEL_BACKEND == "xgboost", "run with AMC_BACKEND=xgboost (everything on the GPU)"
    S2_DIR.mkdir(parents=True, exist_ok=True)
    norm = enriched("train")
    pairs = load_label_pairs()
    ids = set_ids()
    print(", ".join(f"{k} {v.len():,}" for k, v in ids.items()), "S1")

    fit = stage1_features("fit", ids["fit"], pairs, norm)
    tune = stage1_features("tune", ids["tune"], pairs, norm)
    model_path, oof_path = S2_DIR / "stage1_full.xgb.json", S2_DIR / "p1_fit_oof.parquet"
    if model_path.exists() and oof_path.exists():
        stage1 = load_model(model_path)
        print("first stage: reusing the saved model and out-of-fold scores")
    else:
        x_fit, y_fit = matrix(fit), fit["label"].to_numpy()
        x_tune, y_tune = matrix(tune), tune["label"].to_numpy()
        stage1, secs = train_model(x_fit, y_fit, x_tune, y_tune, FEATURES, "xgboost")
        print(f"first stage trained in {secs:.0f}s, best iteration {best_iteration(stage1)}")
        uniq = fit["s1_id"].unique()
        fold_of = pl.DataFrame({"s1_id": uniq, "fold": (stable_unit(uniq.to_list(), C.STAGE2_FOLD_SALT) * C.STAGE2_FOLDS).astype(np.int32)})
        fold = fit.select("s1_id").join(fold_of, on="s1_id", how="left", maintain_order="left")["fold"].to_numpy()
        p_oof = np.full(len(fold), np.nan, dtype=np.float32)
        for k in range(C.STAGE2_FOLDS):
            tr = fold != k
            m, secs = train_model(x_fit[tr], y_fit[tr], x_tune, y_tune, FEATURES, "xgboost")
            p_oof[~tr] = predict(m, x_fit[~tr])
            print(f"out-of-fold model {k + 1}/{C.STAGE2_FOLDS}: {secs:.0f}s, best iteration {best_iteration(m)}")
        assert not np.isnan(p_oof).any()
        del x_fit, x_tune
        save_model(stage1, S2_DIR / "stage1_full")
        fit.select("s1_id", "target_id").with_columns(p1=pl.Series(p_oof)).write_parquet(oof_path)
    fit = fit.join(pl.read_parquet(oof_path), on=["s1_id", "target_id"], how="left", maintain_order="left")
    assert fit["p1"].null_count() == 0

    # ce_train first, so the cross-encoder can start training while the other sets are prepared
    for k in ("ce_train", "fit", "gbm_extra", "tune", "c_select", "dev", "fresh"):
        out = S2_DIR / (f"ce_pairs_{k}.parquet" if k == "ce_train" else f"base_{k}.parquet")
        if out.exists():
            continue
        t0 = time.time()
        if k == "fit":
            f, fit = fit, None  # free it once used
        else:
            f = tune if k == "tune" else stage1_features(k, ids[k], pairs, norm)
            f = f.with_columns(p1=pl.Series(predict(stage1, matrix(f)), dtype=F32))
        if k == "ce_train":
            kept = keep(f.with_columns(p1_rank=pl.col("p1").rank("ordinal", descending=True).over("s1_id")))
            kept = kept.select("s1_id", "target_id", "source", "label", "p1")
        else:
            kept = keep(add_stage2_features(f.with_columns(pl.col("p1").cast(F32)), text_table(norm, f["target_id"])))
        kept.write_parquet(out)
        found = int(f["label"].sum())
        print(f"{k}: {f.height:,} candidates -> kept {kept.height:,} ({kept.height / max(f['s1_id'].n_unique(), 1):.1f} per S1); "
              f"true links kept {int(kept['label'].sum()):,} of {found:,} found by search; {time.time() - t0:.0f}s", flush=True)
        del f, kept
    print("stage 1 done:", S2_DIR)


# ---------------------------------------------------------------- stage 2
def load_set(k: str, norm: pl.DataFrame) -> pl.DataFrame:
    d = pl.read_parquet(S2_DIR / f"base_{k}.parquet").join(pl.read_parquet(S2_DIR / f"ce_{k}.parquet"),
                                                           on=["s1_id", "target_id"], how="left", maintain_order="left")
    assert d["ce"].null_count() == 0, f"{k}: cross-encoder scores missing (run: python -m plan1.cross_encoder score)"
    return add_twin_features(add_ce_features(d), norm)


def main_train() -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(220)
    assert C.MODEL_BACKEND == "xgboost", "run with AMC_BACKEND=xgboost (everything on the GPU)"
    pairs = load_label_pairs()
    countries = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet").select("s1_id", "country")
    ids = set_ids()
    truth = {k: pairs.join(pl.DataFrame({"s1_id": ids[k]}), on="s1_id", how="semi") for k in ("c_select", "dev", "fresh")}
    norm = enriched("train")
    d = {k: load_set(k, norm) for k in ("fit", "gbm_extra", "tune", "c_select", "dev", "fresh")}
    del norm
    train = pl.concat([d.pop("fit"), d.pop("gbm_extra")], how="vertical_relaxed")
    y_tr, y_tune = train["label"].to_numpy(), d["tune"]["label"].to_numpy()
    print(f"second-stage training rows {train.height:,} ({y_tr.mean():.3f} positive), tune rows {d['tune'].height:,}")

    rows, saved = [], {}
    for tag in ("noce", "ce"):
        feats = FEATS[tag]
        m, secs = train_model(matrix(train, feats), y_tr, matrix(d["tune"], feats), y_tune, feats, "xgboost")
        pred = {k: d[k].select("s1_id", "target_id", "label").with_columns(p=pl.Series(predict(m, matrix(d[k], feats))))
                for k in ("c_select", "dev", "fresh")}
        for k, v in pred.items():  # kept for plan1.density_fix
            v.write_parquet(S2_DIR / f"pred_{tag}_{k}.parquet")
        best = choose_threshold(pred["c_select"], truth["c_select"], ids["c_select"])
        thr = best["threshold"]
        print(f"{tag}: trained in {secs:.0f}s, best iteration {best_iteration(m)}, threshold {thr} -> C-select F0.5 {best['macro_f05']:.4f}")
        for k in ("dev", "fresh"):
            rows.append({"panel": k, "model": tag, **panel_scores(pred[k].filter(pl.col("p") >= thr), truth[k], ids[k], countries)})
        path = save_model(m, S2_DIR / f"stage2_{tag}")
        saved[tag] = {"model": str(path), "features": feats, "threshold": thr, "c_select_f05": best["macro_f05"]}
        if tag == "ce":
            imp = importance(m, feats)
            print(pl.DataFrame({"feature": list(imp), "gain": list(imp.values())})
                  .with_columns(share=pl.col("gain") / pl.col("gain").sum()).sort("gain", descending=True).head(25))
    report = pl.DataFrame(rows)
    print("\nno cross-encoder vs cross-encoder (p1-v3-s2 was dev 0.9793 / fresh 0.9794):")
    print(report)
    (S2_DIR / "stage2ce_meta.json").write_text(json.dumps({
        "stage1_model": str(S2_DIR / "stage1_full.xgb.json"), "ce_dir": str(CE_DIR), **saved,
        "min_p1": C.S2_MIN_P1, "top": C.S2_TOP, "features_version": FEATURES_VERSION,
        "normalize_version": NORMALIZE_VERSION, "report": report.to_dicts(),
    }, indent=2, default=float), encoding="utf-8")
    report.write_csv(S2_DIR / "stage2ce_report.tsv", separator="\t")


# ---------------------------------------------------------------- test
def main_test() -> None:
    meta = json.loads((S2_DIR / "stage2ce_meta.json").read_text(encoding="utf-8"))
    assert meta["normalize_version"] == NORMALIZE_VERSION and meta["features_version"] == FEATURES_VERSION
    stage1 = load_model(meta["stage1_model"])
    models = {tag: load_model(meta[tag]["model"]) for tag in ("ce", "noce")}
    tok, ce_model = load_ce(CE_DIR)
    files = sorted((C.WORK_DIR / "test_chunks").glob("*.parquet"))
    if not files:
        raise RuntimeError("no test chunks: run step 5 first")
    norm = enriched("test")
    texts = record_text(norm)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    for i, f in enumerate(files, 1):
        out_path = TEST_DIR / f.name
        if out_path.exists():
            continue
        t1 = time.time()
        c = pl.read_parquet(f)
        res = pl.DataFrame(schema={"s1_id": pl.String, "target_id": pl.String, "source": pl.String, "p1": F32, "ce": F32, "p": F32, "p_noce": F32})
        if c.height:
            q, t = text_lookup(norm, c["s1_id"], c["target_id"])
            feats = build_features(c.select("s1_id", "target_id", "source", "score", "rank", "fb_score"), q, t)
            feats = feats.with_columns(p1=pl.Series(predict(stage1, matrix(feats)), dtype=F32))
            kept = keep(add_stage2_features(feats, text_table(norm, feats["target_id"])))
            if kept.height:
                kept = add_twin_features(kept, norm)
                a, b = pair_texts(kept, texts)
                kept = add_ce_features(kept.with_columns(ce=pl.Series(ce_score(tok, ce_model, a, b), dtype=F32)))
                res = kept.select("s1_id", "target_id", "source", "p1", "ce").with_columns(
                    p=pl.Series(predict(models["ce"], matrix(kept, meta["ce"]["features"])), dtype=F32),
                    p_noce=pl.Series(predict(models["noce"], matrix(kept, meta["noce"]["features"])), dtype=F32))
        tmp = out_path.with_suffix(".tmp")
        res.write_parquet(tmp)
        tmp.replace(out_path)
        print(f"  {i}/{len(files)} {f.name}: {c.height:,} candidates, {res.height:,} kept, "
              f"{int((res['p'] >= meta['ce']['threshold']).sum()):,} accepted, {time.time() - t1:.0f}s "
              f"(elapsed {(time.time() - t_all) / 60:.1f} min)", flush=True)

    scores = pl.read_parquet(str(TEST_DIR / "*.parquet"))
    cands = pl.read_parquet(str(C.WORK_DIR / "test_chunks" / "*.parquet"), columns=["s1_id", "target_id"])
    roster = test_s1_roster()
    assert cands.join(pl.DataFrame({"s1_id": roster}), on="s1_id", how="anti").height == 0
    s1c = load_records("test").filter(pl.col("source") == "S1").select(s1_id="entity_id", country="country")
    for tag, col in (("ce", "p"), ("noce", "p_noce")):
        accepted = scores.filter(pl.col(col) >= meta[tag]["threshold"]).select("s1_id", "target_id", p=col)
        matches = one_owner(accepted)
        out = OUT[tag]
        write_id_lists(cands, roster, out / "candidate_pairs.tsv", "candidate_entity_ids")
        write_id_lists(matches, roster, out / "matching_results.tsv", "matched_entity_ids")
        back = read_id_lists(out / "matching_results.tsv", "matched_entity_ids")
        assert back.join(cands, on=["s1_id", "target_id"], how="anti").height == 0, "a match is not in candidates"
        per = (s1c.join(accepted.group_by("s1_id").agg(n_acc=pl.len()), on="s1_id", how="left")
               .join(matches.group_by("s1_id").agg(n=pl.len()), on="s1_id", how="left").with_columns(pl.col("n_acc", "n").fill_null(0)))
        print(f"\n{tag}: threshold {meta[tag]['threshold']}, accepted {accepted.height:,}, after one owner {matches.height:,} -> {out}")
        print(per.group_by("country").agg(s1=pl.len(), mean_accepted=pl.col("n_acc").mean(), mean_matches=pl.col("n").mean(),
                                          empty_answer_rate=(pl.col("n") == 0).mean()).sort("country"))


if __name__ == "__main__":
    cmds = {"stage1": main_stage1, "train": main_train, "test": main_test}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        raise SystemExit("usage: python -m plan1.stage2ce stage1 | train | test")
    cmds[sys.argv[1]]()
