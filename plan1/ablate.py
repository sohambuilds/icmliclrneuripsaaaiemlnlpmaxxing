"""Ablations for the methodology write-up: how much each fix contributes.

Every variant uses the same smaller setup: 100k Fit businesses (a subset of the 300k), early stopping on Tune,
at most ABLATION_MAX_ROUNDS rounds, threshold chosen on C-select, scored on the FIRST audit_panel (development
data). The fresh audit_panel_2 is never used here. Needs steps 2 and 3 of the current run.

  python -m plan1.ablate features            feature groups + the fallback search (reuses saved candidates)
  python -m plan1.ablate cleanup no_dictionary      one cleanup fix off (re-cleans train, re-searches 160k S1)
      (also: no_states, no_number_markers, no_fillers)

Results accumulate in work/plan1/<run>/ablation/results.tsv. One owner per record and the crowded-record rule
act across all businesses, so they can only be judged on full test runs (leaderboard), not here.
"""
import sys
import time

import numpy as np
import polars as pl

from . import config as C
from .dataio import load_label_pairs, load_records
from .features import BASE_FEATURES, FEATURE_GROUPS, FEATURES, matrix
from .metric import per_s1, score, summarize
from .model import train_lgb
from .normalize import NORMALIZE_VERSION, normalize_records
from .retrieve import TOP_K, candidate_report, retrieve
from .splits import fit_sample_ids
from .step2_retrieve import translit_map
from .step3_train import labelled_features
from .step4_threshold import pick, sweep
from .translit import make_converter

ABL_DIR = C.WORK_DIR / "ablation"
SETS = ("fit", "tune", "c_select", "dev")
CLEANUP = {
    "no_dictionary": {"to_latin": None},
    "no_states": {"states": False},
    "no_number_markers": {"number_markers": False},
    "no_fillers": {"fillers": False},
}


def sample_ids() -> dict[str, pl.Series]:
    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    return {
        "fit": fit_sample_ids(manifest, C.ABLATION_FIT_SIZE),
        "tune": manifest.filter(pl.col("sample") == "tune_sample")["s1_id"],
        "c_select": manifest.filter(pl.col("sample") == "c_select_sample")["s1_id"],
        "dev": manifest.filter(pl.col("sample") == "audit_panel")["s1_id"],
    }


def feature_sets(cands: pl.DataFrame, norm: pl.DataFrame, pairs: pl.DataFrame, ids: dict, cache: str) -> dict:
    out = {}
    for s in SETS:
        path = ABL_DIR / cache / f"features_{s}.parquet"
        if path.exists():
            out[s] = pl.read_parquet(path)
            continue
        f = labelled_features(cands.join(pl.DataFrame({"s1_id": ids[s]}), on="s1_id", how="semi"), pairs, norm, f"{cache}/{s}")
        path.parent.mkdir(parents=True, exist_ok=True)
        f.write_parquet(path)
        out[s] = f
    return out


def evaluate(name: str, feats: dict, features: list[str], pairs: pl.DataFrame, ids: dict, countries: pl.DataFrame,
             cand_recall: float) -> dict:
    t0 = time.time()
    booster, _ = train_lgb(matrix(feats["fit"], features), feats["fit"]["label"].to_numpy(),
                           matrix(feats["tune"], features), feats["tune"]["label"].to_numpy(), features,
                           max_rounds=C.ABLATION_MAX_ROUNDS, log_every=1000)
    truth = {s: pairs.join(pl.DataFrame({"s1_id": ids[s]}), on="s1_id", how="semi") for s in ("c_select", "dev")}
    pred = {s: feats[s].select("s1_id", "target_id").with_columns(p=pl.Series(booster.predict(matrix(feats[s], features))))
            for s in ("c_select", "dev")}
    best = pick(sweep(pred["c_select"], truth["c_select"], ids["c_select"], np.round(np.arange(0.05, 1.0, 0.01), 2)))
    accepted = pred["dev"].filter(pl.col("p") >= best["threshold"])
    dev = score(accepted, truth["dev"], ids["dev"])
    per = per_s1(accepted, truth["dev"], ids["dev"]).join(countries, on="s1_id")
    by_country = {c: summarize(g.drop("country"))["macro_f05"] for (c,), g in per.group_by("country")}
    row = {
        "variant": name, "n_features": len(features), "rounds": booster.best_iteration, "threshold": best["threshold"],
        "c_select_f05": best["macro_f05"], "dev_f05": dev["macro_f05"], "dev_us": by_country.get("US"),
        "dev_india": by_country.get("India"), "dev_precision": dev["link_precision"], "dev_recall": dev["link_recall"],
        "dev_singleton_fp": dev["singleton_false_positive_rate"], "dev_candidate_recall": cand_recall,
        "seconds": time.time() - t0,
    }
    print(f"==> {name}: dev F0.5 {row['dev_f05']:.4f} (US {row['dev_us']:.4f}, India {row['dev_india']:.4f}), "
          f"precision {row['dev_precision']:.4f}, recall {row['dev_recall']:.4f}, {row['rounds']} rounds")
    return row


def save(rows: list[dict]) -> None:
    path = ABL_DIR / "results.tsv"
    new = pl.DataFrame(rows)
    if path.exists():
        old = pl.read_csv(path, separator="\t")
        new = pl.concat([old.filter(~pl.col("variant").is_in(new["variant"].implode())), new], how="diagonal_relaxed")
    new.write_csv(path, separator="\t")
    ref = new.filter(pl.col("variant") == "full")
    if ref.height:
        new = new.with_columns(dev_f05_change_vs_full=pl.col("dev_f05") - ref["dev_f05"][0])
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(250)
    print(new.select("variant", "n_features", "dev_f05", *(["dev_f05_change_vs_full"] if ref.height else []),
                     "dev_us", "dev_india", "dev_precision", "dev_recall", "dev_candidate_recall", "rounds"))
    print("saved:", path)


def main(mode: str, variant: str | None = None) -> None:
    ids = sample_ids()
    pairs = load_label_pairs()
    countries = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet").select("s1_id", "country")
    rows = []

    if mode == "features":
        norm = pl.read_parquet(C.WORK_DIR / f"normalized_train_v{NORMALIZE_VERSION}.parquet")
        cands = pl.read_parquet(C.WORK_DIR / "candidates_train.parquet")
        recall = candidate_report(cands, pairs, ids["dev"], norm)["candidate_recall"]
        feats = feature_sets(cands, norm, pairs, ids, "full")
        variants = [("full", FEATURES)]
        variants += [(f"minus_{g}", [f for f in FEATURES if f not in cols]) for g, cols in FEATURE_GROUPS.items()]
        variants += [("plan1_12_features_only", BASE_FEATURES)]
        for name, features in variants:
            rows.append(evaluate(name, feats, features, pairs, ids, countries, recall))
        # no fallback search at all: drop the candidates only the fallback found, recompute features
        main_only = cands.filter(pl.col("rank") <= TOP_K).with_columns(fb_score=pl.lit(None, dtype=pl.Float32))
        recall = candidate_report(main_only, pairs, ids["dev"], norm)["candidate_recall"]
        feats = feature_sets(main_only, norm, pairs, ids, "no_fallback_search")
        rows.append(evaluate("no_fallback_search", feats, [f for f in FEATURES if f not in FEATURE_GROUPS["fallback"]],
                             pairs, ids, countries, recall))

    elif mode == "cleanup":
        if variant not in CLEANUP:
            raise SystemExit(f"cleanup variant must be one of {list(CLEANUP)}")
        kwargs = {"to_latin": make_converter(translit_map()), **CLEANUP[variant]}
        norm_path = ABL_DIR / variant / "normalized_train.parquet"
        cand_path = ABL_DIR / variant / "candidates.parquet"
        if norm_path.exists():
            norm = pl.read_parquet(norm_path)
        else:
            t0 = time.time()
            norm = normalize_records(load_records("train"), **kwargs)
            norm_path.parent.mkdir(parents=True, exist_ok=True)
            norm.write_parquet(norm_path)
            print(f"{variant}: cleaned train in {time.time() - t0:.0f}s")
        if cand_path.exists():
            cands = pl.read_parquet(cand_path)
        else:
            query_ids = pl.concat([ids[s] for s in SETS])
            print(f"{variant}: searching for {query_ids.len():,} S1")
            cands, _ = retrieve(norm, query_ids)
            cands.write_parquet(cand_path)
        recall = candidate_report(cands, pairs, ids["dev"], norm)["candidate_recall"]
        feats = feature_sets(cands, norm, pairs, ids, variant)
        rows.append(evaluate(variant, feats, FEATURES, pairs, ids, countries, recall))
    else:
        raise SystemExit("usage: python -m plan1.ablate features | cleanup <variant>")
    save(rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "", sys.argv[2] if len(sys.argv) > 2 else None)
