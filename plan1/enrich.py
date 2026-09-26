"""Per-record extras for the trap features (computed once per split, cached):

  legal_bits          canonical legal forms in the name as a bitmask ('public' only counts next to LTD)
  hn                  first number of the raw address (markers and leading zeros removed)
  name_ntok           number of words in the reduced name
  name_unknown_share  share of the reduced name's words never used in any S1 name of the split (invented aliases)
  q_name_s1_rate      S1 rows: how common the name is among S1 of the country (per 100k)
  t_name_rate / t_addr_rate   S2/S3 rows: how common the name / address is among S2+S3 of the country (per 100k)
  raw_name / raw_addr  the text exactly as read (v3: planted copies get re-formatted, some true records keep the
                       S1's exact address)

Features v3 (French legal forms are told apart, and the addresses used by the pair features are cleaner):
  legal_bits  every French form is its own tag (SARL, SAS, SASU, EURL, SA, SCI, SNC, EI), only on French records;
              v2 had one "FR" tag, so a SARL -> SAS planted copy looked like an exact match
  name_red    rebuilt with dotted forms joined longest-first (normalize v4 turned "S.A.S.U." into "sas u"); French
              records also drop S.A. / E.I. / EI like the other legal forms
  addr_norm   number markers dropped ("no 12" -> "12"; French "n 12" too); French region and department names
              dropped (S1 always writes the region, S2/S3 a third of the time and often a department instead, and
              there are only 3 regions). Search keeps the step-2 text; only the pair features see these columns.
"""
import polars as pl

from . import config as C
from .dataio import load_records
from .normalize import (DOTTED_FORMS, FUNCTION_WORDS, FUNCTION_WORDS_EXTRA, LEGAL_FORMS, LEGAL_FORMS_EXTRA,
                        NORMALIZE_VERSION, _ws, word_map)

FEATURES_VERSION = 3  # bump when enrich / build_features outputs change (feature caches carry it)

FR_TAGS = ["SARL", "SAS", "SASU", "EURL", "SA", "SCI", "SNC", "EI"]
LEGAL_TAGS = {
    "inc": "INC", "incorporated": "INC", "corp": "CORP", "corporation": "CORP", "co": "CO", "company": "CO",
    "llc": "LLC", "pllc": "PLLC", "llp": "LLP", "lp": "LP", "pc": "PC", "pa": "PA", "ltd": "LTD", "limited": "LTD",
    "pvt": "PVT", "private": "PVT", "plc": "PLC", "opc": "OPC", "public": "PUBLIC",
    **{t.lower(): t for t in FR_TAGS},
}
TAG_ORDER = ["INC", "CORP", "CO", "LLC", "PLLC", "LLP", "LP", "PC", "PA", "LTD", "PVT", "PLC", "OPC", "PUBLIC"] + FR_TAGS
TAG_BIT = {t: 1 << i for i, t in enumerate(TAG_ORDER)}
TAG_MASK = {**TAG_BIT, "FR": sum(TAG_BIT[t] for t in FR_TAGS)}  # the per-side flag "FR" = any French form
FR_DOTTED = {"s a": "sa", "e i": "ei"}  # S.A. / E.I.: French records only (elsewhere usually initials)
FR_PLACES = {k: "" for k in ("hauts de france", "nouvelle aquitaine", "pays de la loire", "nord", "pas de calais",
                             "gironde", "loire atlantique")}


DROP_RE = r"\b(?:" + "|".join(LEGAL_FORMS + FUNCTION_WORDS + LEGAL_FORMS_EXTRA + FUNCTION_WORDS_EXTRA) + r")\b"  # = normalize v4


def joined_forms(name_cons: pl.Expr, is_fr: pl.Expr) -> pl.Expr:
    """Dotted legal forms joined: L.L.C. -> llc, S.A.S.U. -> sasu (longest key wins); French records also S.A. -> sa,
    E.I. -> ei."""
    name = word_map(name_cons.fill_null(""), DOTTED_FORMS, leftmost=True)
    return pl.when(is_fr).then(word_map(name, FR_DOTTED, leftmost=True)).otherwise(name)


def legal_bits(name_cons: pl.Expr, is_fr: pl.Expr) -> pl.Expr:
    tags = joined_forms(name_cons, is_fr).str.split(" ").list.eval(
        pl.element().replace_strict(list(LEGAL_TAGS), list(LEGAL_TAGS.values()), default=None, return_dtype=pl.String)
    ).list.drop_nulls().list.unique()
    tags = pl.when(is_fr).then(tags).otherwise(tags.list.set_difference(pl.lit(FR_TAGS)))
    tags = pl.when(tags.list.contains("LTD")).then(tags).otherwise(tags.list.set_difference(pl.lit(["PUBLIC"])))
    bits = tags.list.eval(pl.element().replace_strict(list(TAG_BIT), list(TAG_BIT.values()), return_dtype=pl.UInt32)).list.sum()
    return bits.fill_null(0).cast(pl.UInt32)


def feature_text(df: pl.DataFrame) -> pl.DataFrame:
    """Features v3 rewrites of name_red / addr_norm (see the module doc); df has country, name_cons, addr_norm."""
    fr = pl.col("country") == "France"
    red = _ws(joined_forms(pl.col("name_cons"), fr).str.replace_all(DROP_RE, " "))
    red = pl.when(fr).then(_ws(red.str.replace_all(r"\bei\b", " "))).otherwise(red)
    addr = pl.col("addr_norm").fill_null("").str.replace_all(r"\bno (\d)", "${1}")
    addr = pl.when(fr).then(_ws(word_map(addr, FR_PLACES, leftmost=True).str.replace_all(r"\bn (\d)", "${1}"))).otherwise(addr)
    return df.with_columns(name_red=pl.when(red != "").then(red).otherwise(pl.col("name_cons")),
                           addr_norm=addr).with_columns(addr_missing=pl.col("addr_norm") == "")


def first_number(raw_address: pl.Expr) -> pl.Expr:
    a = raw_address.fill_null("").str.to_lowercase().str.replace_all(r"n°|nº|#", " ")
    return a.str.extract(r"(\d+)").str.replace(r"^0+(\d)", "${1}")


def enrich_frame(norm: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    """norm: normalized records of one split; raw: the same split's raw records."""
    df = feature_text(norm.join(raw.select("entity_id", "business_name", "business_address"), on="entity_id", how="left"))
    df = df.with_columns(legal_bits=legal_bits(pl.col("name_cons"), pl.col("country") == "France"),
                         hn=first_number(pl.col("business_address")),
                         name_ntok=pl.col("name_red").str.split(" ").list.len().cast(pl.Float32),
                         raw_name=pl.col("business_name").fill_null("").str.strip_chars(),
                         raw_addr=pl.col("business_address").fill_null("").str.strip_chars()).drop("business_name", "business_address")
    vocab = (df.filter(pl.col("source") == "S1").select(tok=pl.col("name_red").str.split(" "))
             .explode("tok", empty_as_null=True).drop_nulls().unique().with_columns(known=pl.lit(1.0)))
    unknown = (df.select("entity_id", tok=pl.col("name_red").str.split(" ")).explode("tok", empty_as_null=True).drop_nulls()
               .join(vocab, on="tok", how="left").group_by("entity_id")
               .agg(name_unknown_share=(1.0 - pl.col("known").fill_null(0.0).mean()).cast(pl.Float32)))
    df = df.join(unknown, on="entity_id", how="left")

    s1, tg = pl.col("source") == "S1", pl.col("source") != "S1"
    s1_tot = df.filter(s1).group_by("country").agg(n=pl.len())
    tg_tot = df.filter(tg).group_by("country").agg(n=pl.len())
    rate = (pl.col("c") / pl.col("n") * 1e5).cast(pl.Float32)
    q = df.filter(s1).group_by("country", "name_red").agg(c=pl.len()).join(s1_tot, on="country").select("country", "name_red", q_name_s1_rate=rate)
    tn = df.filter(tg).group_by("country", "name_red").agg(c=pl.len()).join(tg_tot, on="country").select("country", "name_red", t_name_rate=rate)
    ta = (df.filter(tg & (pl.col("addr_norm") != "")).group_by("country", "addr_norm").agg(c=pl.len()).join(tg_tot, on="country")
          .select("country", "addr_norm", t_addr_rate=rate))
    df = df.join(q, on=["country", "name_red"], how="left").join(tn, on=["country", "name_red"], how="left").join(ta, on=["country", "addr_norm"], how="left")
    return df.with_columns(  # each rate only on the side it describes
        q_name_s1_rate=pl.when(s1).then(pl.col("q_name_s1_rate")),
        t_name_rate=pl.when(tg).then(pl.col("t_name_rate")),
        t_addr_rate=pl.when(tg).then(pl.col("t_addr_rate")),
    )


def enriched(split: str) -> pl.DataFrame:
    """Normalized records of a split plus the extras above (cached)."""
    from .step2_retrieve import normalized

    path = C.WORK_DIR / f"enriched_{split}_v{NORMALIZE_VERSION}_f{FEATURES_VERSION}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    out = enrich_frame(normalized(split), load_records(split))
    out.write_parquet(path)
    return out
