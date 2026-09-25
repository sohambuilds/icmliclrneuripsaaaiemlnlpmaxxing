"""Per-record extras for the trap features (computed once per split, cached):

  legal_bits          canonical legal forms in the name as a bitmask ('public' only counts next to LTD)
  hn                  first number of the raw address (markers and leading zeros removed)
  name_ntok           number of words in the reduced name
  name_unknown_share  share of the reduced name's words never used in any S1 name of the split (invented aliases)
  q_name_s1_rate      S1 rows: how common the name is among S1 of the country (per 100k)
  t_name_rate / t_addr_rate   S2/S3 rows: how common the name / address is among S2+S3 of the country (per 100k)
"""
import polars as pl

from . import config as C
from .dataio import load_records
from .normalize import DOTTED_FORMS, NORMALIZE_VERSION, word_map

FEATURES_VERSION = 2  # bump when enrich / build_features outputs change (feature caches carry it)

LEGAL_TAGS = {
    "inc": "INC", "incorporated": "INC", "corp": "CORP", "corporation": "CORP", "co": "CO", "company": "CO",
    "llc": "LLC", "pllc": "PLLC", "llp": "LLP", "lp": "LP", "pc": "PC", "pa": "PA", "ltd": "LTD", "limited": "LTD",
    "pvt": "PVT", "private": "PVT", "plc": "PLC", "opc": "OPC", "public": "PUBLIC", "sarl": "FR", "sas": "FR",
    "sasu": "FR", "eurl": "FR", "sa": "FR", "sci": "FR", "snc": "FR",
}
TAG_ORDER = ["INC", "CORP", "CO", "LLC", "PLLC", "LLP", "LP", "PC", "PA", "LTD", "PVT", "PLC", "OPC", "PUBLIC", "FR"]
TAG_BIT = {t: 1 << i for i, t in enumerate(TAG_ORDER)}


def legal_bits(name_cons: pl.Expr) -> pl.Expr:
    toks = word_map(name_cons, DOTTED_FORMS).str.split(" ")
    tags = toks.list.eval(pl.element().replace_strict(list(LEGAL_TAGS), list(LEGAL_TAGS.values()), default=None,
                                                      return_dtype=pl.String)).list.drop_nulls().list.unique()
    tags = pl.when(tags.list.contains("LTD")).then(tags).otherwise(tags.list.set_difference(pl.lit(["PUBLIC"])))
    bits = tags.list.eval(pl.element().replace_strict(list(TAG_BIT), list(TAG_BIT.values()), return_dtype=pl.UInt32)).list.sum()
    return bits.fill_null(0).cast(pl.UInt32)


def first_number(raw_address: pl.Expr) -> pl.Expr:
    a = raw_address.fill_null("").str.to_lowercase().str.replace_all(r"n°|nº|#", " ")
    return a.str.extract(r"(\d+)").str.replace(r"^0+(\d)", "${1}")


def enrich_frame(norm: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    """norm: normalized records of one split; raw: the same split's raw records."""
    df = norm.join(raw.select("entity_id", "business_address"), on="entity_id", how="left")
    df = df.with_columns(legal_bits=legal_bits(pl.col("name_cons")), hn=first_number(pl.col("business_address")),
                         name_ntok=pl.col("name_red").str.split(" ").list.len().cast(pl.Float32)).drop("business_address")
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
