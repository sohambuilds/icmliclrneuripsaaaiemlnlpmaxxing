"""France check on the test set (no labels, CPU, a few minutes). It asks two questions:

  1. How do legal forms change between an S1 and an S2/S3 record that is otherwise the same (same name without
     legal forms, same house number)? Planted copies swap forms at random. A true-record habit would show up as a
     one-way change, like PC -> LLC in the US.
  2. How many France links in the p1-v3-s2 answer have a legal-form change?

Why: in US/India, a legal-form change on an otherwise identical record is a planted copy 99.8% of the time. Features
v2 treat every French form (SARL, SAS, EURL, ...) as ONE form, so a SARL -> SAS copy looks like a perfect match.

  python -m plan1.france_check       reads work/plan1/<run>/stage2/test_scores and the normalized test records
"""
import json

import polars as pl

from . import config as C
from .dataio import load_records
from .decide import one_owner
from .enrich import first_number
from .normalize import DOTTED_FORMS, word_map
from .step2_retrieve import normalized

FORMS = {
    "inc": "INC", "incorporated": "INC", "corp": "CORP", "corporation": "CORP", "co": "CO", "company": "CO",
    "llc": "LLC", "pllc": "PLLC", "llp": "LLP", "lp": "LP", "pc": "PC", "pa": "PA", "ltd": "LTD", "limited": "LTD",
    "pvt": "PVT", "private": "PVT", "plc": "PLC", "opc": "OPC",
    "sarl": "SARL", "sas": "SAS", "sasu": "SASU", "eurl": "EURL", "sa": "SA", "sci": "SCI", "snc": "SNC", "ei": "EI",
}
FR_DOTTED = {"s a": "sa", "e i": "ei"}  # S.A. / E.I. (French records only)


def records() -> pl.DataFrame:
    norm = normalized("test").select("entity_id", "source", "country", "name_cons", "name_red")
    raw = load_records("test").select("entity_id", hn=first_number(pl.col("business_address")))
    fr = pl.col("country") == "France"
    name = word_map(pl.col("name_cons"), DOTTED_FORMS, leftmost=True)
    name = pl.when(fr).then(word_map(name, FR_DOTTED, leftmost=True)).otherwise(name)
    forms = (name.str.split(" ")
             .list.eval(pl.element().replace_strict(list(FORMS), list(FORMS.values()), default=None, return_dtype=pl.String))
             .list.drop_nulls().list.unique().list.sort().list.join("+"))
    core = pl.col("name_red").str.replace_all(r"\b(?:ei|e i|s a)\b", " ").str.replace_all(r"\s+", " ").str.strip_chars()
    return (norm.join(raw, on="entity_id").with_columns(forms=forms, core=core)
            .select("entity_id", "source", "country", "forms", "core", "hn"))


def main() -> None:
    pl.Config.set_tbl_rows(45)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_fmt_str_lengths(60)
    thr = json.loads((C.WORK_DIR / "stage2" / "stage2_meta.json").read_text(encoding="utf-8"))["threshold"]
    rec = records()
    q = rec.filter(pl.col("source") == "S1").select(s1_id="entity_id", country="country", q_forms="forms", q_core="core", q_hn="hn")
    t = rec.filter(pl.col("source") != "S1").select(target_id="entity_id", t_forms="forms", t_core="core", t_hn="hn")
    near, acc = [], []
    for f in sorted((C.WORK_DIR / "stage2" / "test_scores").glob("*.parquet")):
        c = pl.read_parquet(f, columns=["s1_id", "target_id", "p"]).join(q, on="s1_id").join(t, on="target_id")
        near.append(c.filter((pl.col("q_core") == pl.col("t_core")) & (pl.col("q_hn") == pl.col("t_hn"))))
        acc.append(c.filter(pl.col("p") >= thr))
    near, acc = pl.concat(near), pl.concat(acc)
    change = (pl.col("q_forms") != "") & (pl.col("t_forms") != "") & (pl.col("q_forms") != pl.col("t_forms"))

    print(f"threshold {thr}")
    for country in ("France", "US", "India"):
        d = near.filter(pl.col("country") == country)
        print(f"\n{country}: near-identical pairs (same name without legal forms, same house number): {d.height:,}; "
              f"with a legal-form change {d.select(change.mean()).item() or 0:.2%} (accepted {d.filter(change).select((pl.col('p') >= thr).mean()).item() or 0:.1%})")
        print(d.group_by("q_forms", "t_forms").agg(n=pl.len(), accepted=(pl.col("p") >= thr).mean().round(3), mean_p=pl.col("p").mean().round(3))
              .with_columns(share=(pl.col("n") / pl.col("n").sum()).round(4)).sort("n", descending=True).head(40 if country == "France" else 15))

    kept = one_owner(acc.select("s1_id", "target_id", "p"))
    acc1 = acc.join(kept.select("s1_id", "target_id"), on=["s1_id", "target_id"], how="semi")
    print("\nAccepted links with a legal-form change, per country:")
    print(pl.concat([
        d.group_by("country").agg(links=pl.len(), form_change=change.sum(), share=change.mean().round(4),
                                  s1_hit=pl.col("s1_id").filter(change).n_unique()).with_columns(stage=pl.lit(name))
        for name, d in (("accepted", acc), ("after one owner", acc1))
    ]).sort("stage", "country"))
    fr = acc1.filter((pl.col("country") == "France") & change)
    print("\nFrance, after one owner: legal-form changes accepted (S1 form -> record form):")
    print(fr.group_by("q_forms", "t_forms").len().sort("len", descending=True).head(25))

    raw = load_records("test").select("entity_id", "business_name", "business_address")
    ex = fr.sample(min(15, fr.height), seed=C.SEED).select("s1_id", "target_id", "p")
    print("\nexamples:")
    print(ex.join(raw.rename({"entity_id": "s1_id", "business_name": "s1_name", "business_address": "s1_addr"}), on="s1_id")
          .join(raw.rename({"entity_id": "target_id", "business_name": "rec_name", "business_address": "rec_addr"}), on="target_id")
          .select("p", "s1_name", "rec_name", "s1_addr", "rec_addr"))


if __name__ == "__main__":
    main()
