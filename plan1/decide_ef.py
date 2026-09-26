"""Per-business decision: for each S1, pick the candidate set that maximises EXPECTED F0.5, instead of one global
threshold. Uses the second-stage scores (plan1.stage2), calibrated with isotonic regression on C-select.

For an S1 with calibrated scores q sorted high to low (independence approximation):
  taking the top k:   E[F] ~ 1.25 * sum(q_1..q_k) / (0.25 * (sum of all q) * (1 + MISS) + k)
  taking nothing:     E[F] ~ alpha * prod(1 - q_j)          (right only if none is a true match)
MISS ~ share of true links search never finds. alpha and a sharpening power gamma (q**gamma) are tuned on C-select.

  python -m plan1.decide_ef panels   calibrate + tune on C-select; compare with the global threshold on dev + fresh
  python -m plan1.decide_ef test     apply to stage-2 test scores -> output/plan1/<run>-s2-ef/matching_results.tsv
"""
import json
import sys

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from . import config as C
from .dataio import load_label_pairs, load_records, test_s1_roster, write_id_lists
from .decide import one_owner
from .enrich import FEATURES_VERSION, enriched
from .features import matrix
from .metric import per_s1, score, summarize
from .model import load_model, predict
from .splits import fresh_audit_panel
from .stage2 import ALL_FEATURES, S2_DIR, TEST_DIR, add_stage2_features, text_table

MISS = 0.011
EF_DIR = S2_DIR / "decide_ef"
OUT = C.ROOT / "output" / "plan1" / f"{C.RUN_ID}-s2-ef"


def ef_select(df: pl.DataFrame, alpha: float = 1.0, gamma: float = 1.0) -> pl.DataFrame:
    """df: s1_id, target_id, q (calibrated). Returns the selected rows."""
    d = (df.with_columns(q=pl.col("q").clip(0.0, 1 - 1e-9) ** gamma)
         .sort(["s1_id", "q"], descending=[False, True])
         .with_columns(k=pl.int_range(1, pl.len() + 1).over("s1_id"),
                       cum=pl.col("q").cum_sum().over("s1_id"),
                       tot=pl.col("q").sum().over("s1_id"),
                       none=(1 - pl.col("q")).log().sum().over("s1_id").exp() * alpha)
         .with_columns(ef=1.25 * pl.col("cum") / (0.25 * pl.col("tot") * (1 + MISS) + pl.col("k"))))
    best = (d.sort(["s1_id", "ef", "k"], descending=[False, True, False]).unique("s1_id", keep="first", maintain_order=True)
            .select("s1_id", k_sel=pl.when(pl.col("none") > pl.col("ef")).then(0).otherwise(pl.col("k"))))
    return d.join(best, on="s1_id").filter(pl.col("k") <= pl.col("k_sel")).drop("k_sel")


def stage2_scores(name: str, ids: pl.Series, meta: dict, norm: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    path = EF_DIR / f"p2_{name}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    feats = pl.read_parquet(S2_DIR / f"stage1_features_{name}_f{FEATURES_VERSION}.parquet")
    p1 = predict(load_model(meta["stage1_model"]), matrix(feats))
    s2 = add_stage2_features(feats.with_columns(p1=pl.Series(p1)), text_table(norm, feats["target_id"]))
    out = s2.select("s1_id", "target_id", "label").with_columns(p=pl.Series(predict(load_model(meta["stage2_model"]), matrix(s2, ALL_FEATURES))))
    out.write_parquet(path)
    return out


def calibrator(xs: np.ndarray, ys: np.ndarray):
    return lambda p: np.interp(p, xs, ys)


def panels() -> None:
    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(200)
    EF_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((S2_DIR / "stage2_meta.json").read_text(encoding="utf-8"))
    norm, pairs = enriched("train"), load_label_pairs()
    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    countries = manifest.select("s1_id", "country")
    ids = {"c_select": manifest.filter(pl.col("sample") == "c_select_sample")["s1_id"],
           "dev": manifest.filter(pl.col("sample") == "audit_panel")["s1_id"], "fresh": fresh_audit_panel(manifest)}
    truth = {k: pairs.join(pl.DataFrame({"s1_id": v}), on="s1_id", how="semi") for k, v in ids.items()}
    sc = {k: stage2_scores(k, v, meta, norm, pairs) for k, v in ids.items()}

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(sc["c_select"]["p"].to_numpy(), sc["c_select"]["label"].to_numpy())
    np.save(EF_DIR / "calibration.npy", np.vstack([iso.X_thresholds_, iso.y_thresholds_]))
    cal = calibrator(iso.X_thresholds_, iso.y_thresholds_)
    sc = {k: v.with_columns(q=pl.Series(cal(v["p"].to_numpy()))) for k, v in sc.items()}

    grid = [(a, g) for a in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0) for g in (0.8, 1.0, 1.25, 1.5)]
    tune = pl.DataFrame([{"alpha": a, "gamma": g, "c_select_f05": score(ef_select(sc["c_select"], a, g), truth["c_select"], ids["c_select"])["macro_f05"]}
                         for a, g in grid]).sort("c_select_f05", descending=True)
    a, g = tune.row(0)[0], tune.row(0)[1]
    print("C-select tuning (top 8):")
    print(tune.head(8))

    def row(panel: str, rule: str, pred: pl.DataFrame) -> dict:
        s = score(pred, truth[panel], ids[panel])
        per = per_s1(pred, truth[panel], ids[panel]).join(countries, on="s1_id")
        byc = {c: summarize(x.drop("country"))["macro_f05"] for (c,), x in per.group_by("country")}
        return {"panel": panel, "rule": rule, "macro_f05": s["macro_f05"], "US": byc.get("US"), "India": byc.get("India"),
                "precision": s["link_precision"], "recall": s["link_recall"], "singleton_fp": s["singleton_false_positive_rate"]}

    rows = []
    for panel in ("dev", "fresh"):
        rows.append(row(panel, f"global threshold {meta['threshold']}", sc[panel].filter(pl.col("p") >= meta["threshold"])))
        rows.append(row(panel, "expected F0.5 (alpha 1, gamma 1)", ef_select(sc[panel])))
        rows.append(row(panel, f"expected F0.5 (alpha {a}, gamma {g})", ef_select(sc[panel], a, g)))
    report = pl.DataFrame(rows)
    print("\nglobal threshold vs expected-F0.5 decision:")
    print(report)
    (EF_DIR / "decide_ef.json").write_text(json.dumps({"alpha": a, "gamma": g, "miss": MISS, "report": report.to_dicts()}, indent=2, default=float), encoding="utf-8")


def test() -> None:
    cfg = json.loads((EF_DIR / "decide_ef.json").read_text(encoding="utf-8"))
    xs, ys = np.load(EF_DIR / "calibration.npy")
    cal = calibrator(xs, ys)
    parts = []
    for f in sorted(TEST_DIR.glob("*.parquet")):
        c = pl.read_parquet(f, columns=["s1_id", "target_id", "p"])
        if c.height:
            parts.append(ef_select(c.with_columns(q=pl.Series(cal(c["p"].to_numpy()))), cfg["alpha"], cfg["gamma"]).select("s1_id", "target_id", "p", "q"))
    chosen = pl.concat(parts)
    matches = one_owner(chosen.with_columns(p=pl.col("q")))
    roster = test_s1_roster()
    write_id_lists(matches, roster, OUT / "matching_results.tsv", "matched_entity_ids")
    s1c = load_records("test").filter(pl.col("source") == "S1").select(s1_id="entity_id", country="country")
    per = s1c.join(matches.group_by("s1_id").agg(n=pl.len()), on="s1_id", how="left").with_columns(pl.col("n").fill_null(0))
    print(f"alpha {cfg['alpha']}, gamma {cfg['gamma']}: chosen {chosen.height:,}, after one owner {matches.height:,}")
    print(per.group_by("country").agg(mean_matches=pl.col("n").mean(), empty=(pl.col("n") == 0).mean()).sort("country"))
    print("written:", OUT / "matching_results.tsv")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("panels", "test"):
        raise SystemExit("usage: python -m plan1.decide_ef panels | test")
    panels() if sys.argv[1] == "panels" else test()
