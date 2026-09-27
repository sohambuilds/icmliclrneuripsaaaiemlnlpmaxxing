"""France clean-up rules for any matching_results.tsv. France changes; US/India stays exactly as it was.

Planted copies sit next to the real record and differ from it in one designed way. A French link is dropped when:
  R1 legal conflict   both names carry French legal forms and share none (SARL vs SAS, SAS vs EURL, ...). On
                      US/India a legal conflict is a planted copy 99.8% of the time. The most common French copy in
                      the test is a legal swap plus a house number shifted by a few units.
  R2 number twin      another link of the same S1 has the same name and street AND the S1's house number, while this
                      link has a different number within 15 of it
  R3 extra-word twin  another link of the same S1 has the same street and number, and this link's name is that name
                      plus extra words

  python -m plan1.france_rules dev                      R1-R3 on OUR US/India dev answer (labels): what each rule costs or gains
  python -m plan1.france_rules <in.tsv> <out_dir>       France cleaned -> <out_dir>/matching_results.tsv (R1+R2+R3)
                                                        and <out_dir>-r1/matching_results.tsv (R1 only)
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from . import config as C
from .dataio import load_label_pairs, load_records, read_id_lists, test_s1_roster, write_id_lists
from .enrich import FR_TAGS, TAG_BIT, enriched
from .metric import score

FR_MASK = sum(TAG_BIT[t] for t in FR_TAGS)
MAX_SHIFT = 15


def attrs(split: str) -> pl.DataFrame:
    street = pl.col("addr_norm").fill_null("").str.replace_all(r"\S*\d\S*", " ").str.replace_all(r"\s+", " ").str.strip_chars()
    return enriched(split).select("entity_id", "country", "legal_bits", "hn", nm="name_red", st=street,
                                  words=pl.col("name_red").fill_null("").str.split(" "))


def flag_links(links: pl.DataFrame, a: pl.DataFrame, french_forms_only: bool = True) -> pl.DataFrame:
    """links: s1_id, target_id (+ any columns). Adds country and the rule flags r1, r2, r3."""
    q = a.select(s1_id="entity_id", country="country", q_bits="legal_bits", q_hn="hn")
    t = a.select(target_id="entity_id", t_bits="legal_bits", t_hn="hn", t_nm="nm", t_st="st", t_words="words")
    d = links.join(q, on="s1_id", how="left").join(t, on="target_id", how="left")
    mask = FR_MASK if french_forms_only else (1 << 31) - 1
    qb = d["q_bits"].fill_null(0).cast(pl.Int64).to_numpy() & mask
    tb = d["t_bits"].fill_null(0).cast(pl.Int64).to_numpy() & mask
    d = d.with_columns(r1=pl.Series((qb != 0) & (tb != 0) & ((qb & tb) == 0)))

    o = d.select("s1_id", o_id="target_id", o_hn="t_hn", o_nm="t_nm", o_st="t_st", o_words="t_words")
    pr = (d.select("s1_id", "target_id", "q_hn", "t_hn", "t_nm", "t_st", "t_words").join(o, on="s1_id")
          .filter(pl.col("target_id") != pl.col("o_id")))
    fl = pl.DataFrame(schema={"s1_id": pl.String, "target_id": pl.String, "r2": pl.Boolean, "r3": pl.Boolean})
    if pr.height:
        ns = process.cpdist(pr["t_nm"].fill_null("").to_list(), pr["o_nm"].fill_null("").to_list(), scorer=fuzz.token_set_ratio, workers=-1)
        ss = process.cpdist(pr["t_st"].fill_null("").to_list(), pr["o_st"].fill_null("").to_list(), scorer=fuzz.token_set_ratio, workers=-1)
        pr = (pr.with_columns(ns=pl.Series(ns), ss=pl.Series(ss))
              .filter((pl.col("ns") >= 90) & (pl.col("ss") >= 90) & (pl.col("t_st") != "") & (pl.col("o_st") != "")))
        th, qh = pl.col("t_hn").cast(pl.Int64, strict=False), pl.col("q_hn").cast(pl.Int64, strict=False)
        r2 = (pl.col("o_hn") == pl.col("q_hn")) & (pl.col("t_hn") != pl.col("q_hn")) & ((th - qh).abs() <= MAX_SHIFT)
        r3 = ((pl.col("t_hn") == pl.col("o_hn"))
              & (pl.col("o_words").list.set_difference(pl.col("t_words")).list.len() == 0)
              & (pl.col("t_words").list.set_difference(pl.col("o_words")).list.len() > 0))
        fl = pr.group_by("s1_id", "target_id").agg(r2=r2.fill_null(False).any(), r3=r3.fill_null(False).any())
    return (d.join(fl, on=["s1_id", "target_id"], how="left").with_columns(pl.col("r2", "r3").fill_null(False))
            .select(*links.columns, "country", "r1", "r2", "r3"))


RULES = {"R1 legal conflict": pl.col("r1"), "R2 number twin": pl.col("r2"), "R3 extra-word twin": pl.col("r3"),
         "R1+R2+R3": pl.col("r1") | pl.col("r2") | pl.col("r3")}


def main_dev() -> None:
    """Our s2ce answer on the dev panel (US/India, labelled). R1 here uses every legal form: the model already
    rejects most conflicts, so the ones it accepted are mostly the allowed kind (PC/LP/LLP -> LLC/INC)."""
    s2 = C.WORK_DIR / "stage2_ce"
    thr = json.loads((s2 / "stage2ce_meta.json").read_text(encoding="utf-8"))["ce"]["threshold"]
    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    ids = manifest.filter(pl.col("sample") == "audit_panel")["s1_id"]
    truth = load_label_pairs().join(pl.DataFrame({"s1_id": ids}), on="s1_id", how="semi")
    pred = pl.read_parquet(s2 / "pred_ce_dev.parquet").filter(pl.col("p") >= thr).select("s1_id", "target_id", "label")
    d = flag_links(pred, attrs("train"), french_forms_only=False)
    rows = [{"rule": "none", "dropped": 0, "true_among_dropped": 0, "dev_f05": score(d, truth, ids)["macro_f05"]}]
    for name, cond in RULES.items():
        drop = d.filter(cond)
        rows.append({"rule": name, "dropped": drop.height, "true_among_dropped": int(drop["label"].sum()),
                     "dev_f05": score(d.filter(~cond), truth, ids)["macro_f05"]})
    print(f"dev panel, our s2ce answer at threshold {thr}:")
    print(pl.DataFrame(rows))


def main_file(path: str, out_dir: str) -> None:
    pl.Config.set_tbl_rows(30)
    links = read_id_lists(path, "matched_entity_ids")
    a = attrs("test")
    d = flag_links(links, a)
    roster = test_s1_roster()
    n_fr = a.filter(pl.col("entity_id").str.starts_with("S1-") & (pl.col("country") == "France")).height
    print(f"{path}: {links.height:,} links")
    rows = []
    for c in ("France", "US", "India"):
        dc = d.filter(pl.col("country") == c)
        rows.append({"country": c, "links": dc.height, **{name: dc.filter(cond).height for name, cond in RULES.items()}})
    print("links each rule would drop (only France is changed):")
    print(pl.DataFrame(rows))
    for suffix, cond in (("", RULES["R1+R2+R3"]), ("-r1", RULES["R1 legal conflict"])):
        drop = d.filter((pl.col("country") == "France") & cond).select("s1_id", "target_id")
        kept = links.join(drop, on=["s1_id", "target_id"], how="anti")
        out = Path(out_dir + suffix) / "matching_results.tsv"
        write_id_lists(kept, roster, out, "matched_entity_ids")
        fr = kept.join(d.filter(pl.col("country") == "France").select("s1_id").unique(), on="s1_id", how="semi")
        print(f"\n{out}: dropped {drop.height:,} French links over {drop['s1_id'].n_unique():,} businesses; France now "
              f"{fr.height:,} links, {fr.height / n_fr:.4f} per business, empty {1 - fr['s1_id'].n_unique() / n_fr:.4f}")
    print("\nexamples of dropped French links:")
    raw = load_records("test").select("entity_id", "business_name", "business_address")
    ex = d.filter((pl.col("country") == "France") & RULES["R1+R2+R3"])
    ex = ex.sample(min(12, ex.height), seed=C.SEED, shuffle=True)
    print(ex.select("s1_id", "target_id", "r1", "r2", "r3")
          .join(raw.rename({"entity_id": "s1_id", "business_name": "s1_name", "business_address": "s1_addr"}), on="s1_id")
          .join(raw.rename({"entity_id": "target_id", "business_name": "rec_name", "business_address": "rec_addr"}), on="target_id")
          .select("r1", "r2", "r3", "s1_name", "rec_name", "s1_addr", "rec_addr"))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "dev":
        main_dev()
    elif len(sys.argv) == 3:
        main_file(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit("usage: python -m plan1.france_rules dev | <matching_results.tsv> <out_dir>")
