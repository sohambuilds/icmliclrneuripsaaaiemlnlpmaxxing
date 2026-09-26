"""Why the test scores below the panels. No test labels are used.

p1-v3-s2 scored 0.9522 on the leaderboard and 0.8303 on the nofrance probe. So US/India on test ≈ 0.967 (the panels
said 0.979) and France ≈ 0.87. For the p1-v3-s2 answer this script compares, per country:
  1. the KINDS of links accepted on test vs on the dev panel, whose labels say how often each kind is wrong
     -> expected wrong links per 100 S1 on test;
  2. the kinds of plausible links rejected (0.02 <= p < threshold), and how often each kind is true on dev;
  3. how often an S1 has a near-twin S1 (same name without legal forms, same street), train vs test. Twins that
     are planted on the S1 side would be a test-only trap: our panels never contain the twin.

  python -m plan1.test_vs_dev      reads stage2/test_scores and stage2/decide_ef/p2_dev.parquet (CPU, ~10 min)
"""
import json

import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from . import config as C
from .dataio import load_label_pairs, load_records
from .enrich import FR_PLACES, first_number, joined_forms
from .normalize import word_map
from .step2_retrieve import normalized

FORMS = {
    "inc": "INC", "incorporated": "INC", "corp": "CORP", "corporation": "CORP", "co": "CO", "company": "CO",
    "llc": "LLC", "pllc": "PLLC", "llp": "LLP", "lp": "LP", "pc": "PC", "pa": "PA", "ltd": "LTD", "limited": "LTD",
    "pvt": "PVT", "private": "PVT", "plc": "PLC", "opc": "OPC", "public": "PUBLIC",
    "sarl": "SARL", "sas": "SAS", "sasu": "SASU", "eurl": "EURL", "sa": "SA", "sci": "SCI", "snc": "SNC", "ei": "EI",
}
STOP = sorted(set(FORMS) | {"and", "the", "of", "de", "la", "le", "les", "du", "des", "d", "l", "et", "cie", "ets",
                            "etablissements", "www", "com", "net", "org"})
S2_DIR = C.WORK_DIR / "stage2"


def attrs(split: str) -> pl.DataFrame:
    """Per record: legal forms, name words without legal forms/fillers (core), street text without numbers, number."""
    norm = normalized(split).select("entity_id", "source", "country", "name_cons", "addr_norm")
    raw = load_records(split).select("entity_id", hn=first_number(pl.col("business_address")))
    fr = pl.col("country") == "France"
    toks = joined_forms(pl.col("name_cons"), fr).str.split(" ")
    forms = (toks.list.eval(pl.element().replace_strict(list(FORMS), list(FORMS.values()), default=None, return_dtype=pl.String))
             .list.drop_nulls().list.unique())
    words = toks.list.eval(pl.element().filter(~pl.element().is_in(STOP) & (pl.element() != "")))
    street = pl.col("addr_norm").fill_null("").str.replace_all(r"\S*\d\S*", " ").str.replace_all(r"\s+", " ").str.strip_chars()
    street = pl.when(fr).then(word_map(street, FR_PLACES, leftmost=True)).otherwise(street)
    return (norm.join(raw, on="entity_id")
            .select("entity_id", "source", "country", "hn", forms=forms, words=words, street=street,
                    no_addr=pl.col("addr_norm").fill_null("") == "")
            .with_columns(core=pl.col("words").list.join(" ")))


def kinds(pairs: pl.DataFrame, a: pl.DataFrame) -> pl.DataFrame:
    """pairs: s1_id, target_id, ... -> + country and kind = '<name> <address>[ legal_conflict]'."""
    q = a.select(s1_id="entity_id", country="country", q_forms="forms", q_words="words", q_core="core", q_street="street", q_hn="hn")
    t = a.select(target_id="entity_id", t_forms="forms", t_words="words", t_core="core", t_street="street", t_hn="hn", t_no_addr="no_addr")
    d = pairs.join(q, on="s1_id").join(t, on="target_id")
    lev = process.cpdist(d["q_core"].to_list(), d["t_core"].to_list(), scorer=Levenshtein.distance, workers=-1)
    st = process.cpdist(d["q_street"].to_list(), d["t_street"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    d = d.with_columns(lev=pl.Series(lev, dtype=pl.Int32), st=pl.Series(st, dtype=pl.Float32))
    hq, ht = pl.col("q_hn").cast(pl.Int64, strict=False), pl.col("t_hn").cast(pl.Int64, strict=False)
    extra = ((pl.col("t_words").list.set_difference(pl.col("q_words")).list.len() > 0)
             & (pl.col("q_words").list.set_difference(pl.col("t_words")).list.len() == 0))
    name = (pl.when(pl.col("q_core") == pl.col("t_core")).then(pl.lit("same"))
            .when(pl.col("lev") <= 2).then(pl.lit("edit"))
            .when(extra).then(pl.lit("extra_word"))
            .otherwise(pl.lit("other_name")))
    addr = (pl.when(pl.col("t_no_addr")).then(pl.lit("no_addr"))
            .when(pl.col("st") < 80).then(pl.lit("street_diff"))
            .when(pl.col("q_hn").is_null() | pl.col("t_hn").is_null()).then(pl.lit("no_num"))
            .when(pl.col("q_hn") == pl.col("t_hn")).then(pl.lit("num_same"))
            .when((hq - ht).abs() <= 10).then(pl.lit("num_shift10"))
            .otherwise(pl.lit("num_other")))
    legal = (pl.when((pl.col("q_forms").list.len() > 0) & (pl.col("t_forms").list.len() > 0)
                     & (pl.col("q_forms").list.set_intersection(pl.col("t_forms")).list.len() == 0))
             .then(pl.lit(" legal_conflict")).otherwise(pl.lit("")))
    return d.select("s1_id", "target_id", "country", *[c for c in ("p", "label") if c in d.columns],
                    kind=pl.concat_str([name, pl.lit(" "), addr, legal]))


def s1_twins(a: pl.DataFrame) -> pl.DataFrame:
    """Share of S1 (per country) with another S1 of the same core name and street: same number / shifted <= 10."""
    s1 = a.filter((pl.col("source") == "S1") & (pl.col("core") != ""))
    total = s1.group_by("country").agg(s1=pl.len())
    g = s1.with_columns(n=pl.len().over("country", "core")).filter((pl.col("n") > 1) & (pl.col("n") <= 300))
    g = g.select("entity_id", "country", "core", "street", "hn", "forms")
    pr = g.join(g, on=["country", "core"], suffix="_b").filter(pl.col("entity_id") < pl.col("entity_id_b"))
    st = process.cpdist(pr["street"].to_list(), pr["street_b"].to_list(), scorer=fuzz.token_set_ratio, workers=-1)
    hq, hb = pl.col("hn").cast(pl.Int64, strict=False), pl.col("hn_b").cast(pl.Int64, strict=False)
    pr = pr.with_columns(st=pl.Series(st)).filter((pl.col("st") >= 90) & (pl.col("street") != "")).with_columns(
        twin=pl.when(pl.col("hn") == pl.col("hn_b")).then(pl.lit("same_num"))
             .when((hq - hb).abs() <= 10).then(pl.lit("num_shift10")).otherwise(pl.lit("other_num")))
    ends = pl.concat([pr.select("country", "twin", s="entity_id"), pr.select("country", "twin", s="entity_id_b")])
    return (ends.group_by("country", "twin").agg(n=pl.col("s").n_unique()).join(total, on="country")
            .with_columns(share=(pl.col("n") / pl.col("s1")).round(4)).pivot("twin", index="country", values="share").sort("country"))


def main() -> None:
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(200)
    thr = json.loads((S2_DIR / "stage2_meta.json").read_text(encoding="utf-8"))["threshold"]
    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    dev_ids = manifest.filter(pl.col("sample") == "audit_panel").select("s1_id", "country")
    truth = load_label_pairs().join(dev_ids, on="s1_id", how="semi")
    a_tr, a_te = attrs("train"), attrs("test")

    dev = kinds(pl.read_parquet(S2_DIR / "decide_ef" / "p2_dev.parquet").filter(pl.col("p") >= 0.02), a_tr)
    test = kinds(pl.concat([pl.read_parquet(f, columns=["s1_id", "target_id", "p"]).filter(pl.col("p") >= 0.02)
                            for f in sorted((S2_DIR / "test_scores").glob("*.parquet"))]), a_te)
    n_dev = dict(dev_ids.group_by("country").len().iter_rows())
    n_te = dict(a_te.filter(pl.col("source") == "S1").group_by("country").len().iter_rows())

    print(f"threshold {thr}\n\nper S1 (dev has labels):")
    acc_dev, acc_te = dev.filter(pl.col("p") >= thr), test.filter(pl.col("p") >= thr)
    rows = []
    for c in ("US", "India", "France"):
        r = {"country": c, "test_links_per_s1": acc_te.filter(pl.col("country") == c).height / n_te[c],
             "test_empty": 1 - acc_te.filter(pl.col("country") == c)["s1_id"].n_unique() / n_te[c]}
        if c in n_dev:
            ids_c = dev_ids.filter(pl.col("country") == c)
            r.update(dev_links_per_s1=acc_dev.filter(pl.col("country") == c).height / n_dev[c],
                     dev_empty=1 - acc_dev.filter(pl.col("country") == c)["s1_id"].n_unique() / n_dev[c],
                     dev_true_links_per_s1=truth.join(ids_c, on="s1_id", how="semi").height / n_dev[c],
                     dev_no_match_share=1 - truth.join(ids_c, on="s1_id", how="semi")["s1_id"].n_unique() / n_dev[c])
        rows.append(r)
    print(pl.DataFrame(rows))

    for title, cond, stat in (("ACCEPTED links: expected wrong per 100 S1 on test (dev precision of each kind)", pl.col("p") >= thr, "wrong"),
                              (f"REJECTED plausible links (0.02 <= p < {thr}): expected true per 100 S1 on test", pl.col("p") < thr, "true")):
        print(f"\n==== {title}")
        for c in ("US", "India", "France"):
            dref = dev.filter(cond & (pl.col("country") == c)) if c in n_dev else dev.filter(cond)
            nref = n_dev[c] if c in n_dev else sum(n_dev.values())
            dv = dref.group_by("kind").agg(dev_per100=pl.len() / nref * 100, dev_true=pl.col("label").mean())
            te = test.filter(cond & (pl.col("country") == c)).group_by("kind").agg(test_per100=pl.len() / n_te[c] * 100)
            tab = te.join(dv, on="kind", how="full", coalesce=True).fill_null(0)
            rate = (1 - pl.col("dev_true")) if stat == "wrong" else pl.col("dev_true")
            tab = tab.with_columns(**{f"dev_{stat}_per100": pl.col("dev_per100") * rate, f"test_{stat}_per100": pl.col("test_per100") * rate})
            s = tab.select(pl.col(f"dev_{stat}_per100").sum(), pl.col(f"test_{stat}_per100").sum()).row(0)
            ref = c if c in n_dev else "US+India dev as reference"
            print(f"\n{c} ({ref}): {stat} per 100 S1 expected on test {s[1]:.2f} vs dev {s[0]:.2f}")
            print(tab.sort(f"test_{stat}_per100", descending=True).head(15)
                  .with_columns(pl.col(pl.Float64).round(3)))

    print("\n==== S1 with a near-twin S1 (same core name + same street), share of S1")
    print("train:")
    print(s1_twins(a_tr))
    print("test:")
    print(s1_twins(a_te))


if __name__ == "__main__":
    main()
