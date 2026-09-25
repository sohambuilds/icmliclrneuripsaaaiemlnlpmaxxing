"""Error breakdown on the DEVELOPMENT Audit panel (the first audit_panel). The fresh audit_panel_2 is not touched.

Run from the repository root (needs steps 2-4 of the current run):  python -m plan1.error_analysis
Outputs under work/plan1/<run>/error_analysis/: loss_split.tsv, rejected_true.tsv, accepted_wrong.tsv,
rescue_rules.tsv, examples_*.tsv

1. Exact split of the lost score: wrong links on businesses with matches, true links the model rejects,
   true links search never found, no-match businesses given a link (the four parts add up to 1 - F0.5).
2. Rejected true links by type vs accepted true links (what the model can't see).
3. Accepted wrong links by type.
4. What a business-level second stage could gain: accept a rejected candidate when it closely resembles a record
   already accepted for the same business (duplicates of one business look alike).
"""
import json

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from . import config as C
from .dataio import load_label_pairs, load_records
from .features import matrix
from .metric import per_s1, score
from .normalize import NORMALIZE_VERSION
from .retrieve import TOP_K
from .step3_train import labelled_features

OUT = C.WORK_DIR / "error_analysis"


def macro(pred: pl.DataFrame, truth: pl.DataFrame, ids: pl.Series, countries: pl.DataFrame) -> dict:
    per = per_s1(pred, truth, ids).join(countries, on="s1_id")
    out = {"all": per["f"].mean()}
    out.update({c: g["f"].mean() for (c,), g in per.group_by("country")})
    return out


def flag_table(df: pl.DataFrame, flags: dict, reference: pl.DataFrame | None = None) -> pl.DataFrame:
    rows = []
    for name, expr in flags.items():
        row = {"type": name, "count": int(df.select(expr.fill_null(False).sum()).item()),
               "share": float(df.select(expr.fill_null(False).mean()).item())}
        if reference is not None:
            row["share_among_accepted_true"] = float(reference.select(expr.fill_null(False).mean()).item())
        row["mean_p"] = float(df.filter(expr.fill_null(False))["p"].mean() or float("nan"))
        rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(250)
    pl.Config.set_fmt_str_lengths(55)
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((C.WORK_DIR / "frozen_config.json").read_text(encoding="utf-8"))
    thr = frozen["threshold"]
    booster = lgb.Booster(model_file=str(C.WORK_DIR / "model.txt"))
    norm = pl.read_parquet(C.WORK_DIR / f"normalized_train_v{NORMALIZE_VERSION}.parquet")
    cands = pl.read_parquet(C.WORK_DIR / "candidates_train.parquet")
    pairs = load_label_pairs()
    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    countries = manifest.select("s1_id", "country")
    ids = manifest.filter(pl.col("sample") == "audit_panel")["s1_id"]
    truth = pairs.join(pl.DataFrame({"s1_id": ids}), on="s1_id", how="semi")

    feats = labelled_features(cands.join(pl.DataFrame({"s1_id": ids}), on="s1_id", how="semi"), pairs, norm, "dev panel")
    feats = feats.with_columns(p=pl.Series(booster.predict(matrix(feats))))
    feats = feats.join(cands.select("s1_id", "target_id", "rank"), on=["s1_id", "target_id"], how="left")
    acc = feats.filter(pl.col("p") >= thr)
    n_true = truth.group_by("s1_id").agg(g=pl.len())
    singles = pl.DataFrame({"s1_id": ids}).join(n_true, on="s1_id", how="anti")

    # ---------- 1. exact split of the lost score ----------
    steps = {"now": acc}
    steps["+ drop wrong links (businesses with matches)"] = acc.filter(
        (pl.col("label") == 1) | pl.col("s1_id").is_in(singles["s1_id"].implode()))
    steps["+ accept every true link search found"] = pl.concat(
        [steps["+ drop wrong links (businesses with matches)"].select("s1_id", "target_id"),
         feats.filter(pl.col("label") == 1).select("s1_id", "target_id")]).unique()
    steps["+ every true link (perfect search)"] = pl.concat(
        [steps["+ accept every true link search found"], truth.select("s1_id", "target_id")]).unique()
    steps["+ nothing for no-match businesses"] = steps["+ every true link (perfect search)"].join(singles, on="s1_id", how="anti")
    split, prev = [], None
    for name, pred in steps.items():
        m = macro(pred.select("s1_id", "target_id"), truth, ids, countries)
        split.append({"step": name, **{k: round(v, 5) for k, v in m.items()},
                      "gain_points": None if prev is None else round((m["all"] - prev) * 100, 2)})
        prev = m["all"]
    split = pl.DataFrame(split)
    split.write_csv(OUT / "loss_split.tsv", separator="\t")
    print(f"\n1) where the lost score goes (dev panel, threshold {thr}; gain_points = F0.5 points recovered by that step):")
    print(split)

    # ---------- extra context for the categories ----------
    raw = load_records("train").select("entity_id", "business_name", "business_address")
    owner = pairs.select("target_id", owner_s1="s1_id")
    s1_name = norm.select(s1_id="entity_id", s1_name_red="name_red")
    feats = (feats
             .join(raw.rename({"entity_id": "target_id", "business_name": "t_name", "business_address": "t_addr"}), on="target_id", how="left")
             .join(owner, on="target_id", how="left")
             .join(s1_name, on="s1_id", how="left")
             .join(s1_name.rename({"s1_id": "owner_s1", "s1_name_red": "owner_name_red"}), on="owner_s1", how="left")
             .with_columns(s1_accepted=(pl.col("p") >= thr).sum().over("s1_id"), s1_max_p=pl.col("p").max().over("s1_id")))

    # similarity of every candidate to the records already accepted for the same S1 (excluding itself)
    txt = norm.select(target_id="entity_id", n="name_red", a="addr_norm")
    sib = (feats.select("s1_id", "target_id")
           .join(feats.filter(pl.col("p") >= thr).select("s1_id", sib_id="target_id"), on="s1_id")
           .filter(pl.col("target_id") != pl.col("sib_id"))
           .join(txt, on="target_id").join(txt.rename({"target_id": "sib_id", "n": "sn", "a": "sa"}), on="sib_id"))
    name_sim = process.cpdist(sib["n"].to_list(), sib["sn"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
    addr_sim = process.cpdist(sib["a"].to_list(), sib["sa"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100
    addr_sim[((sib["a"] == "") | (sib["sa"] == "")).to_numpy()] = np.nan
    sib = (sib.select("s1_id", "target_id").with_columns(sn_sim=pl.Series(name_sim), sa_sim=pl.Series(addr_sim).fill_nan(None))
           .group_by("s1_id", "target_id").agg(sib_name=pl.col("sn_sim").max(), sib_addr=pl.col("sa_sim").max(),
                                               sib_both=pl.min_horizontal("sn_sim", "sa_sim").max()))
    feats = feats.join(sib, on=["s1_id", "target_id"], how="left")

    website = pl.col("t_name").fill_null("").str.contains(r"(?i)www\.|\.(com|net|org|in|co|us|fr)\b")
    common = {
        "record has no address": pl.col("cand_addr_missing") == 1,
        "record name in Indian script": pl.col("cand_name_nonlatin") == 1,
        "record name is a website": website,
        "weak name match (token set < 0.6)": pl.col("name_red_token_set") < 0.6,
        "numbers conflict": pl.col("num_conflict") == 1,
        "search rank > 10": pl.col("retrieval_rank") > 10,
        "found only by fallback search": pl.col("retrieval_rank") == TOP_K + 1,
        "5+ candidates with a near-identical name": pl.col("n_name_close") >= 5,
        "looks like an accepted record of this S1 (name & addr >= 0.9)": pl.col("sib_both") >= 0.9,
        "business got nothing accepted": pl.col("s1_accepted") == 0,
    }

    # ---------- 2. rejected true links ----------
    rej = feats.filter((pl.col("label") == 1) & (pl.col("p") < thr))
    acc_true = feats.filter((pl.col("label") == 1) & (pl.col("p") >= thr))
    t2 = flag_table(rej, {**common, "close call (p >= 0.3)": pl.col("p") >= 0.3, "confident reject (p < 0.05)": pl.col("p") < 0.05},
                    reference=acc_true)
    t2.write_csv(OUT / "rejected_true.tsv", separator="\t")
    print(f"\n2) true links search found but the model rejected: {rej.height:,} "
          f"({rej.height / max(feats.filter(pl.col('label') == 1).height, 1):.1%} of found true links). Types overlap:")
    print(t2)

    # ---------- 3. accepted wrong links ----------
    wrong = feats.filter((pl.col("label") == 0) & (pl.col("p") >= thr))
    t3 = flag_table(wrong, {
        "business has no true match (singleton)": pl.col("s1_id").is_in(singles["s1_id"].implode()),
        "record belongs to no business (distractor)": pl.col("owner_s1").is_null(),
        "record belongs to another business": pl.col("owner_s1").is_not_null(),
        "...whose cleaned name equals this S1's": pl.col("owner_name_red") == pl.col("s1_name_red"),
        "names identical after cleaning": pl.col("name_red_ratio") == 1.0,
        "addresses identical (token set = 1)": pl.col("addr_token_set") == 1.0,
        **common,
        "confident accept (p >= 0.9)": pl.col("p") >= 0.9,
    })
    t3.write_csv(OUT / "accepted_wrong.tsv", separator="\t")
    print(f"\n3) wrong links accepted: {wrong.height:,}. Types overlap:")
    print(t3)

    # ---------- 4. business-level rescue rules (what a second stage could exploit) ----------
    base = score(acc, truth, ids)
    rows = [{"rule": "now", "macro_f05": base["macro_f05"], "change_points": 0.0, "added_true": 0, "added_wrong": 0}]
    for lo in (0.05, 0.1, 0.2, 0.3):
        for sim in (0.9, 0.95):
            extra = feats.filter((pl.col("p") < thr) & (pl.col("p") >= lo) & (pl.col("sib_both") >= sim))
            r = score(pl.concat([acc.select("s1_id", "target_id"), extra.select("s1_id", "target_id")]), truth, ids)
            rows.append({"rule": f"also accept p >= {lo} if name & addr >= {sim} vs an accepted record",
                         "macro_f05": r["macro_f05"], "change_points": (r["macro_f05"] - base["macro_f05"]) * 100,
                         "added_true": int(extra["label"].sum()), "added_wrong": int((extra["label"] == 0).sum())})
    rules = pl.DataFrame(rows)
    rules.write_csv(OUT / "rescue_rules.tsv", separator="\t")
    print("\n4) simple business-level rescue rules (development panel; indicates second-stage headroom):")
    print(rules)

    print("\nrejected true links by model score:")
    print(rej.select(pl.col("p").cut([0.05, 0.2, 0.4, thr], left_closed=True).alias("p_bin")).group_by("p_bin").len().sort("p_bin"))

    # ---------- examples ----------
    cols = ["s1_id", "p", "s1_name_red", "t_name", "t_addr", "name_red_token_set", "addr_token_set", "retrieval_rank"]
    ex_rej = rej.sort("p", descending=True).head(15).vstack(rej.sort("p").head(15)).select(cols)
    ex_wrong = wrong.sort("p", descending=True).head(25).select(cols + ["owner_name_red"])
    s1_raw = raw.rename({"entity_id": "s1_id", "business_name": "s1_name", "business_address": "s1_addr"})
    ex_rej = ex_rej.join(s1_raw, on="s1_id", how="left")
    ex_wrong = ex_wrong.join(s1_raw, on="s1_id", how="left")
    ex_rej.write_csv(OUT / "examples_rejected_true.tsv", separator="\t")
    ex_wrong.write_csv(OUT / "examples_accepted_wrong.tsv", separator="\t")
    print("\nexamples: rejected true links (top: nearest misses, bottom: most confident rejections)")
    print(ex_rej.select("p", "s1_name", "t_name", "s1_addr", "t_addr"))
    print("\nexamples: most confident wrong links")
    print(ex_wrong.select("p", "s1_name", "t_name", "s1_addr", "t_addr", "owner_name_red"))
    print("\nsaved to", OUT)


if __name__ == "__main__":
    main()
