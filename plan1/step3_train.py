"""Step 3: build the features for the Fit and Tune candidates, label them, train one LightGBM model.

Run from the repository root:  python -m plan1.step3_train
Needs step 2's outputs from the current normalization version.
Outputs (under work/plan1/<run>/): model.txt, model_meta.json, tune_predictions.parquet
"""
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

from . import config as C
from .dataio import load_label_pairs
from .features import FEATURES, build_features, matrix, text_lookup
from .metric import score
from .model import train_lgb
from .normalize import NORMALIZE_VERSION
from .splits import fit_sample_ids


def labelled_features(cands: pl.DataFrame, pairs: pl.DataFrame, norm: pl.DataFrame, name: str) -> pl.DataFrame:
    t0 = time.time()
    q, t = text_lookup(norm, cands["s1_id"], cands["target_id"])
    feats = build_features(cands, q, t)
    feats = feats.join(pairs.with_columns(label=pl.lit(1, pl.Int8)), on=["s1_id", "target_id"], how="left",
                       maintain_order="left").with_columns(pl.col("label").fill_null(0))
    print(f"{name}: {feats['s1_id'].n_unique():,} S1, {feats.height:,} pairs, "
          f"{int(feats['label'].sum()):,} positives ({feats['label'].mean():.4f}), features in {time.time() - t0:.0f}s")
    return feats


def main() -> None:
    meta = json.loads((C.WORK_DIR / "retrieval_meta.json").read_text(encoding="utf-8"))
    if meta["normalize_version"] != NORMALIZE_VERSION:
        raise RuntimeError(f"candidates were built with normalize v{meta['normalize_version']}, code is v{NORMALIZE_VERSION}: rerun step 2")

    norm = pl.read_parquet(C.WORK_DIR / f"normalized_train_v{NORMALIZE_VERSION}.parquet")
    cands = pl.read_parquet(C.WORK_DIR / "candidates_train.parquet")
    pairs = load_label_pairs()
    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    fit_ids = fit_sample_ids(manifest, C.FIT_SAMPLE_SIZE)
    tune_ids = manifest.filter(pl.col("sample") == "tune_sample")["s1_id"]
    assert fit_ids.len() == C.FIT_SAMPLE_SIZE == meta["fit_sample_size"], "fit sample size differs from step 2: rerun step 2"
    assert not set(fit_ids.to_list()) & set(tune_ids.to_list()), "Fit and Tune share S1s"

    fit = labelled_features(cands.join(pl.DataFrame({"s1_id": fit_ids}), on="s1_id", how="semi"), pairs, norm, "fit")
    tune = labelled_features(cands.join(pl.DataFrame({"s1_id": tune_ids}), on="s1_id", how="semi"), pairs, norm, "tune")

    # ---- checks: labels, missing values, per-label feature means ----
    for name, df, ids in (("fit", fit, fit_ids), ("tune", tune, tune_ids)):
        truth = pairs.join(pl.DataFrame({"s1_id": ids}), on="s1_id", how="semi")
        found = truth.join(df.select("s1_id", "target_id"), on=["s1_id", "target_id"], how="semi").height
        assert found == int(df["label"].sum()), f"{name}: label join mismatch"
        print(f"{name}: positives = retrieved true links = {found:,} of {truth.height:,} true links")
    print("\nmissing-value rate per feature (fit):")
    print(fit.select([pl.col(f).is_null().mean().alias(f) for f in FEATURES]).transpose(include_header=True, header_name="feature", column_names=["missing"]))
    print("feature means by label (fit):")
    print(fit.group_by("label").agg([pl.col(f).mean() for f in FEATURES]).sort("label").transpose(include_header=True, header_name="feature"))

    # ---- one LightGBM model, early stopping on Tune ----
    x_fit, y_fit = matrix(fit), fit["label"].to_numpy()
    x_tune, y_tune = matrix(tune), tune["label"].to_numpy()
    booster, train_s = train_lgb(x_fit, y_fit, x_tune, y_tune, FEATURES)
    best = booster.best_iteration
    p_tune = booster.predict(x_tune, num_iteration=best)
    print(f"\ntrained in {train_s:.0f}s, best iteration {best}")
    print(f"tune: logloss {log_loss(y_tune, p_tune):.5f}, AUC {roc_auc_score(y_tune, p_tune):.5f}")
    imp = pl.DataFrame({"feature": FEATURES, "gain": booster.feature_importance("gain", iteration=best),
                        "splits": booster.feature_importance("split", iteration=best)})
    print(imp.with_columns(gain_share=pl.col("gain") / pl.col("gain").sum()).sort("gain", descending=True))

    # ---- development look only (the threshold is chosen on C-select in step 4) ----
    tune_pred = tune.select("s1_id", "target_id", "source", "label").with_columns(p=pl.Series(p_tune))
    tune_truth = pairs.join(pl.DataFrame({"s1_id": tune_ids}), on="s1_id", how="semi")
    print("\nTune macro F0.5 by threshold (development only):")
    print("  empty-all:", round(score(tune_pred.head(0), tune_truth, tune_ids)["macro_f05"], 4))
    for thr in np.arange(0.1, 0.96, 0.1):
        r = score(tune_pred.filter(pl.col("p") >= thr), tune_truth, tune_ids)
        print(f"  p >= {thr:.1f}: F0.5 {r['macro_f05']:.4f}  precision {r['link_precision']:.4f}  recall {r['link_recall']:.4f}  "
              f"singleton FP {r['singleton_false_positive_rate']:.4f}")

    booster.save_model(str(C.WORK_DIR / "model.txt"), num_iteration=best)
    tune_pred.write_parquet(C.WORK_DIR / "tune_predictions.parquet")
    with open(C.WORK_DIR / "model_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "features": FEATURES, "params": C.LGB_PARAMS, "best_iteration": best, "max_rounds": C.LGB_MAX_ROUNDS,
            "early_stopping": C.LGB_EARLY_STOPPING, "lightgbm": lgb.__version__, "normalize_version": NORMALIZE_VERSION,
            "fit_pairs": fit.height, "fit_positives": int(y_fit.sum()), "tune_pairs": tune.height,
            "tune_logloss": float(log_loss(y_tune, p_tune)), "tune_auc": float(roc_auc_score(y_tune, p_tune)),
            "train_seconds": train_s,
        }, f, indent=2)
    print("\nsaved:", C.WORK_DIR / "model.txt")


if __name__ == "__main__":
    main()
