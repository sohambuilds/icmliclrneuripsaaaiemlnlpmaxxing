"""Trap analysis: how do planted near-copies differ from real duplicates? (training data only)

Run from the repository root:  python -m plan1.trap_analysis
Uses 100k Fit-role S1 (training data; evaluation panels untouched) and, for section 6, the whole training set.
Reuses work/plan1/<run>/ablation/full/features_fit.parquet when present (from `ablate backends`).
Outputs: work/plan1/<run>/trap_analysis/*.tsv

Groups of candidate pairs (S1, S2/S3 record):
  true            the record belongs to this S1
  true_similar    true AND name token-set >= 0.9 AND (address token-set >= 0.8 or record has no address)
  copy_nobiz      wrong, similar as above, the record belongs to NO business (planted distractor)
  copy_otherbiz   wrong, similar as above, the record belongs to ANOTHER S1
"""
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from . import config as C
from .dataio import load_label_pairs, load_records
from .normalize import DOTTED_FORMS, NORMALIZE_VERSION, word_map
from .splits import fit_sample_ids
from .step3_train import labelled_features

OUT = C.WORK_DIR / "trap_analysis"
LEGAL_TAGS = {
    "inc": "INC", "incorporated": "INC", "corp": "CORP", "corporation": "CORP", "co": "CO", "company": "CO",
    "llc": "LLC", "pllc": "PLLC", "llp": "LLP", "lp": "LP", "pc": "PC", "pa": "PA", "ltd": "LTD", "limited": "LTD",
    "pvt": "PVT", "private": "PVT", "plc": "PLC", "opc": "OPC", "public": "PUBLIC", "sarl": "SARL", "sas": "SAS",
    "sasu": "SASU", "eurl": "EURL", "sa": "SA", "sci": "SCI", "snc": "SNC",
}
GROUPS = ["true", "true_similar", "copy_nobiz", "copy_otherbiz"]


def legal_tags(name_cons: pl.Expr) -> pl.Expr:
    """Sorted list of canonical legal-form tags found anywhere in the name ('public' counts only next to LTD)."""
    toks = word_map(name_cons, DOTTED_FORMS).str.split(" ")
    tags = toks.list.eval(pl.element().replace_strict(list(LEGAL_TAGS), list(LEGAL_TAGS.values()), default=None,
                                                      return_dtype=pl.String)).list.drop_nulls().list.unique().list.sort()
    return pl.when(tags.list.contains("LTD")).then(tags).otherwise(tags.list.set_difference(pl.lit(["PUBLIC"])))


def first_number(raw_address: pl.Expr) -> pl.Expr:
    """First run of digits in the raw address, number markers and leading zeros removed."""
    a = raw_address.fill_null("").str.to_lowercase().str.replace_all(r"n°|nº|#", " ")
    return a.str.extract(r"(\d+)").str.replace(r"^0+(\d)", "${1}")


def legal_category(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    la, lb = a.list.len(), b.list.len()
    inter = a.list.set_intersection(b).list.len()
    return (pl.when((la == 0) & (lb == 0)).then(pl.lit("both none"))
            .when(la == 0).then(pl.lit("only record has one"))
            .when(lb == 0).then(pl.lit("only S1 has one"))
            .when((inter == la) & (inter == lb)).then(pl.lit("same"))
            .when(inter == lb).then(pl.lit("record's is a subset of S1's (e.g. Pvt Ltd -> Ltd)"))
            .when(inter == la).then(pl.lit("S1's is a subset of record's"))
            .when(inter == 0).then(pl.lit("conflict (e.g. LLC -> Corp)"))
            .otherwise(pl.lit("partial overlap")))


def number_category(df: pl.DataFrame, a: str, b: str) -> pl.Series:
    sa, sb = df[a].fill_null(""), df[b].fill_null("")
    lev = pl.Series(process.cpdist(sa.to_list(), sb.to_list(), scorer=Levenshtein.distance, workers=-1))
    d = pl.DataFrame({"a": sa, "b": sb, "lev": lev})
    na, nb = pl.col("a").cast(pl.Int64, strict=False), pl.col("b").cast(pl.Int64, strict=False)
    return (d.with_columns(
                both=(pl.col("a") != "") & (pl.col("b") != ""), eq=pl.col("a") == pl.col("b"),
                la=pl.col("a").str.len_chars().cast(pl.Int32), lb=pl.col("b").str.len_chars().cast(pl.Int32),
                small=((na - nb).abs() <= 10).fill_null(False),
                fix=pl.col("a").str.starts_with(pl.col("b")) | pl.col("b").str.starts_with(pl.col("a"))
                | pl.col("a").str.ends_with(pl.col("b")) | pl.col("b").str.ends_with(pl.col("a")))
            .select(pl.when(~pl.col("both")).then(pl.lit("missing on a side"))
                    .when(pl.col("eq")).then(pl.lit("equal"))
                    .when((pl.col("la") == pl.col("lb")) & (pl.col("lev") == 1) & ~pl.col("small")).then(pl.lit("1 digit changed"))
                    .when(((pl.col("la") - pl.col("lb")).abs() == 1) & (pl.col("lev") == 1)).then(pl.lit("1 digit added/dropped"))
                    .when(pl.col("small")).then(pl.lit("small arithmetic change (<=10)"))
                    .when(pl.col("fix")).then(pl.lit("one is prefix/suffix of other"))
                    .when(pl.col("la") == pl.col("lb")).then(pl.lit("2+ digits changed, same length"))
                    .otherwise(pl.lit("different")).alias("cat"))["cat"])


def shares(df: pl.DataFrame, col: str, groups: list[str] = GROUPS) -> pl.DataFrame:
    """Share of each category within each group (columns = groups), plus counts."""
    t = df.filter(pl.col("group_list").list.set_intersection(pl.lit(groups)).list.len() > 0)
    rows = []
    for g in groups:
        sub = t.filter(pl.col("group_list").list.contains(g))
        cnt = sub.group_by(col).len().with_columns(share=pl.col("len") / sub.height).select(col, pl.col("share").alias(g))
        rows.append(cnt)
    out = rows[0]
    for r in rows[1:]:
        out = out.join(r, on=col, how="full", coalesce=True)
    return out.fill_null(0).sort(groups[0], descending=True)


def main() -> None:
    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(260)
    pl.Config.set_fmt_str_lengths(70)
    OUT.mkdir(parents=True, exist_ok=True)
    norm = pl.read_parquet(C.WORK_DIR / f"normalized_train_v{NORMALIZE_VERSION}.parquet")
    raw = load_records("train")
    pairs = load_label_pairs()
    owner = pairs.select("target_id", owner_s1="s1_id")
    manifest = pl.read_parquet(C.SPLIT_DIR / "manifest.parquet")
    ids = fit_sample_ids(manifest, C.ABLATION_FIT_SIZE)

    feats_path = C.WORK_DIR / "ablation" / "full" / "features_fit.parquet"
    if feats_path.exists():
        feats = pl.read_parquet(feats_path)
    else:
        cands = pl.read_parquet(C.WORK_DIR / "candidates_train.parquet").join(pl.DataFrame({"s1_id": ids}), on="s1_id", how="semi")
        feats = labelled_features(cands, pairs, norm, "trap sample")

    rec = (raw.select("entity_id", "source", "country", "business_name", "business_address")
           .join(norm.select("entity_id", "name_red", "name_cons", "addr_norm", "addr_missing"), on="entity_id")
           .with_columns(legal=legal_tags(pl.col("name_cons")), hn=first_number(pl.col("business_address"))))
    q = rec.select(s1_id="entity_id", q_name="business_name", q_addr="business_address", q_red="name_red",
                   q_legal="legal", q_hn="hn", q_addr_norm="addr_norm")
    t = rec.select(target_id="entity_id", t_name="business_name", t_addr="business_address", t_red="name_red",
                   t_legal="legal", t_hn="hn", t_addr_norm="addr_norm", t_addr_missing="addr_missing")

    similar = (pl.col("name_red_token_set") >= 0.9) & ((pl.col("addr_token_set") >= 0.8) | (pl.col("cand_addr_missing") == 1))
    P = (feats.select("s1_id", "target_id", "label", "name_red_token_set", "name_red_ratio", "addr_token_set", "cand_addr_missing")
         .join(owner, on="target_id", how="left").join(q, on="s1_id").join(t, on="target_id")
         .with_columns(similar=similar.fill_null(False)))
    P = P.with_columns(group_list=pl.concat_list(
        pl.when(pl.col("label") == 1).then(pl.lit("true")),
        pl.when((pl.col("label") == 1) & pl.col("similar")).then(pl.lit("true_similar")),
        pl.when((pl.col("label") == 0) & pl.col("similar") & pl.col("owner_s1").is_null()).then(pl.lit("copy_nobiz")),
        pl.when((pl.col("label") == 0) & pl.col("similar") & pl.col("owner_s1").is_not_null()).then(pl.lit("copy_otherbiz")),
    ).list.drop_nulls())
    P = P.filter(pl.col("group_list").list.len() > 0)
    counts = {g: P.filter(pl.col("group_list").list.contains(g)).height for g in GROUPS}
    print(f"sample: {ids.len():,} Fit-role S1; pairs per group: {counts}")
    print("similar wrong candidates per S1:", round((counts["copy_nobiz"] + counts["copy_otherbiz"]) / ids.len(), 3),
          "| similar true candidates per S1:", round(counts["true_similar"] / ids.len(), 3))

    # ---------- 1. legal form ----------
    P = P.with_columns(legal_cat=legal_category(pl.col("q_legal"), pl.col("t_legal")),
                       legal_pair=pl.concat_str(pl.col("q_legal").list.join("+"), pl.lit(" -> "), pl.col("t_legal").list.join("+")))
    t1 = shares(P, "legal_cat")
    t1.write_csv(OUT / "1_legal_category.tsv", separator="\t")
    print("\n1) legal form, S1 vs record (share of pairs in each group):")
    print(t1)
    conflicts = P.filter(pl.col("legal_cat").is_in(["conflict (e.g. LLC -> Corp)", "partial overlap"]))
    for g in ("true_similar", "copy_nobiz"):
        top = conflicts.filter(pl.col("group_list").list.contains(g)).group_by("legal_pair").len().sort("len", descending=True).head(15)
        print(f"   most common legal-form changes in {g}:")
        print(top)

    # ---------- 2. extra words ----------
    tq, tt = pl.col("q_red").str.split(" "), pl.col("t_red").str.split(" ")
    P = P.with_columns(extra_t=tt.list.set_difference(tq), extra_q=tq.list.set_difference(tt))
    P = P.with_columns(n_extra=(pl.col("extra_t").list.len() + pl.col("extra_q").list.len()).clip(0, 3).cast(pl.String)
                       .replace({"3": "3+"}))
    t2 = shares(P, "n_extra")
    t2.write_csv(OUT / "2_extra_word_count.tsv", separator="\t")
    print("\n2) words that differ between the reduced names (count):")
    print(t2)
    words = (P.filter(pl.col("group_list").list.set_intersection(pl.lit(["true_similar", "copy_nobiz", "copy_otherbiz"])).list.len() > 0)
             .with_columns(is_copy=~pl.col("group_list").list.contains("true_similar"))
             .select("is_copy", w=pl.concat_list("extra_t", "extra_q")).explode("w").drop_nulls("w")
             .group_by("w").agg(n=pl.len(), copies=pl.col("is_copy").sum())
             .with_columns(copy_rate=pl.col("copies") / pl.col("n")).filter(pl.col("n") >= 100))
    words.sort("n", descending=True).write_csv(OUT / "2_extra_words.tsv", separator="\t")
    base_rate = (counts["copy_nobiz"] + counts["copy_otherbiz"]) / max(counts["copy_nobiz"] + counts["copy_otherbiz"] + counts["true_similar"], 1)
    print(f"   extra words most tied to COPIES (base copy rate among similar pairs {base_rate:.3f}):")
    print(words.sort("copy_rate", descending=True).head(25))
    print("   extra words most tied to TRUE pairs:")
    print(words.sort("copy_rate").head(25))

    # ---------- 3. house numbers ----------
    P = P.with_columns(num_cat=number_category(P, "q_hn", "t_hn"))
    t3 = shares(P, "num_cat")
    t3.write_csv(OUT / "3_house_number.tsv", separator="\t")
    print("\n3) first number in the address, S1 vs record:")
    print(t3)

    # ---------- 4. records with no address ----------
    na = P.filter(pl.col("t_addr_missing"))
    na = na.with_columns(
        name_rel=pl.when(pl.col("q_red") == pl.col("t_red")).then(pl.lit("reduced name identical"))
        .when(pl.col("name_red_token_set") >= 0.9).then(pl.lit("token set >= 0.9")).otherwise(pl.lit("weaker")),
        same_name_rivals=pl.len().over("s1_id", "t_red"),
        legal_cat=pl.col("legal_cat"),
    )
    t4 = (na.group_by("name_rel", "legal_cat").agg(pairs=pl.len(), true_rate=pl.col("label").mean())
          .sort("name_rel", "pairs", descending=[False, True]))
    t4b = (na.with_columns(rivals=pl.col("same_name_rivals").clip(1, 4).cast(pl.String).replace({"4": "4+"}))
           .group_by("rivals").agg(pairs=pl.len(), true_rate=pl.col("label").mean()).sort("rivals"))
    t4.write_csv(OUT / "4_no_address.tsv", separator="\t")
    print("\n4) similar candidates whose record has NO address: how often true, by name match and legal form:")
    print(t4)
    print("   ...by number of address-less candidates of this S1 with the identical reduced name:")
    print(t4b)

    # ---------- 5. invented single-word names ----------
    vocab = (norm.filter(pl.col("source") == "S1").select(tok=pl.col("name_red").str.split(" ")).explode("tok")
             .unique().with_columns(in_s1=pl.lit(True)))
    allp = (feats.select("s1_id", "target_id", "label", "addr_token_set", "cand_addr_missing", "name_red_token_set")
            .join(t.select("target_id", "t_red", "t_hn", "t_name"), on="target_id").join(q.select("s1_id", "q_hn"), on="s1_id")
            .filter(~pl.col("t_red").str.contains(" ") & (pl.col("t_red").str.len_chars() >= 5)
                    & ~pl.col("t_name").str.contains(r"(?i)www|\.com|\.net|\.org"))
            .join(vocab, left_on="t_red", right_on="tok", how="left").filter(pl.col("in_s1").is_null()))
    allp = allp.with_columns(addr_bucket=pl.when(pl.col("cand_addr_missing") == 1).then(pl.lit("no address"))
                             .when((pl.col("addr_token_set") >= 0.95) & (pl.col("q_hn") == pl.col("t_hn"))).then(pl.lit("address >= 0.95 + same number"))
                             .when(pl.col("addr_token_set") >= 0.9).then(pl.lit("address >= 0.9"))
                             .when(pl.col("addr_token_set") >= 0.7).then(pl.lit("address 0.7-0.9"))
                             .otherwise(pl.lit("address < 0.7")))
    t5 = allp.group_by("addr_bucket").agg(pairs=pl.len(), true_links=pl.col("label").sum(), true_rate=pl.col("label").mean()).sort("addr_bucket")
    t5.write_csv(OUT / "5_invented_names.tsv", separator="\t")
    print(f"\n5) candidates whose record name is ONE word never used in any S1 name ({allp.height:,} pairs, "
          f"{int(allp['label'].sum()):,} true): how often true, by address match:")
    print(t5)

    # ---------- 6. the whole training set: are unmatched records copies of real businesses? ----------
    tg = rec.filter(pl.col("source") != "S1").join(owner, left_on="entity_id", right_on="target_id", how="left")
    s1r = rec.filter(pl.col("source") == "S1")
    name_n = s1r.group_by("country", "name_red").agg(n_s1=pl.len())
    tg = tg.join(name_n, on=["country", "name_red"], how="left")
    print("\n6) whole training set, S2/S3 records:")
    print(tg.group_by(matched=pl.col("owner_s1").is_not_null()).agg(
        records=pl.len(), name_equals_some_S1=pl.col("n_s1").is_not_null().mean(),
        no_address=pl.col("addr_missing").mean()).sort("matched"))
    # nearest same-name S1 for unmatched records (names used by <= 20 S1), vs the owner for matched records
    unm = tg.filter(pl.col("owner_s1").is_null() & pl.col("n_s1").is_not_null() & (pl.col("n_s1") <= 20))
    cl = (unm.select("entity_id", "country", "name_red", t_legal="legal", t_hn="hn", t_addr_norm="addr_norm", t_missing="addr_missing")
          .join(s1r.select("country", "name_red", s1_id="entity_id", q_legal="legal", q_hn="hn", q_addr_norm="addr_norm"), on=["country", "name_red"]))
    cl = cl.with_columns(addr_sim=pl.Series(process.cpdist(cl["t_addr_norm"].to_list(), cl["q_addr_norm"].to_list(),
                                                           scorer=fuzz.token_set_ratio, workers=-1)) / 100)
    cl = cl.sort("entity_id", "addr_sim", descending=[False, True]).unique("entity_id", keep="first", maintain_order=True)
    mt = (tg.filter(pl.col("owner_s1").is_not_null()).sample(min(1_000_000, tg.height), seed=C.SEED)
          .select("entity_id", t_legal="legal", t_hn="hn", t_addr_norm="addr_norm", t_missing="addr_missing", s1_id="owner_s1")
          .join(s1r.select(s1_id="entity_id", q_legal="legal", q_hn="hn", q_addr_norm="addr_norm"), on="s1_id"))
    mt = mt.with_columns(addr_sim=pl.Series(process.cpdist(mt["t_addr_norm"].to_list(), mt["q_addr_norm"].to_list(),
                                                           scorer=fuzz.token_set_ratio, workers=-1)) / 100)
    both = pl.concat([cl.select("t_legal", "q_legal", "t_hn", "q_hn", "addr_sim", "t_missing").with_columns(kind=pl.lit("unmatched vs nearest same-name S1")),
                      mt.select("t_legal", "q_legal", "t_hn", "q_hn", "addr_sim", "t_missing").with_columns(kind=pl.lit("matched vs its own S1"))])
    both = both.with_columns(legal_cat=legal_category(pl.col("q_legal"), pl.col("t_legal")),
                             addr_bucket=pl.when(pl.col("t_missing")).then(pl.lit("no address"))
                             .when(pl.col("addr_sim") >= 0.95).then(pl.lit(">= 0.95")).when(pl.col("addr_sim") >= 0.8).then(pl.lit("0.8-0.95"))
                             .when(pl.col("addr_sim") >= 0.5).then(pl.lit("0.5-0.8")).otherwise(pl.lit("< 0.5")))
    both = both.with_columns(num_cat=number_category(both, "q_hn", "t_hn"), group_list=pl.concat_list(pl.col("kind")))
    kinds = ["matched vs its own S1", "unmatched vs nearest same-name S1"]
    print(f"   unmatched records whose name is used by 1-20 S1: {cl.height:,}")
    for col in ("addr_bucket", "legal_cat", "num_cat"):
        tb = shares(both, col, kinds)
        tb.write_csv(OUT / f"6_{col}.tsv", separator="\t")
        print(f"   {col}:")
        print(tb)
    copy_like = both.filter((pl.col("kind") == kinds[1]) & (pl.col("addr_sim") >= 0.8))
    print(f"   unmatched records that look like a copy of an S1 (same name, address >= 0.8): {copy_like.height:,}; "
          f"what differs:")
    print(copy_like.group_by("legal_cat", "num_cat").len().sort("len", descending=True).head(20))
    groups = (tg.filter(pl.col("owner_s1").is_null())
              .group_by("country", "name_red", pl.col("legal").list.join("+"), "hn").agg(size=pl.len())
              .group_by(size=pl.col("size").clip(1, 5)).agg(groups=pl.len()).sort("size"))
    print("   unmatched records grouped by (country, name, legal form, first number): group sizes (5 = 5+):")
    print(groups)

    # ---------- 7. no-match S1 ----------
    per = (feats.group_by("s1_id").agg(n_true=pl.col("label").sum())
           .join(P.filter(pl.col("group_list").list.set_intersection(pl.lit(["copy_nobiz", "copy_otherbiz"])).list.len() > 0)
                 .group_by("s1_id").agg(n_copies=pl.len()), on="s1_id", how="left").with_columns(pl.col("n_copies").fill_null(0)))
    print("\n7) S1 with a similar wrong candidate (copy), by whether the S1 has any true match in its candidates:")
    print(per.group_by(has_true=pl.col("n_true") > 0).agg(s1=pl.len(), with_copy=(pl.col("n_copies") > 0).mean(),
                                                           mean_copies=pl.col("n_copies").mean()).sort("has_true"))

    # ---------- 8. examples ----------
    cols = ["legal_cat", "num_cat", "q_name", "t_name", "q_addr", "t_addr"]
    for g, n in (("copy_nobiz", 30), ("copy_otherbiz", 15), ("true_similar", 25)):
        sub = P.filter(pl.col("group_list").list.contains(g))
        ex = sub.sample(min(n, sub.height), seed=C.SEED).select(cols)
        ex.write_csv(OUT / f"8_examples_{g}.tsv", separator="\t")
        print(f"\n8) examples: {g}")
        print(ex)
    ex = P.filter(pl.col("group_list").list.contains("true_similar") & (pl.col("legal_cat") == "conflict (e.g. LLC -> Corp)")).head(15).select(cols)
    print("\n   true pairs whose legal forms conflict (if any):")
    print(ex)
    print("\nsaved to", OUT)


if __name__ == "__main__":
    main()
