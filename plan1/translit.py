"""Indian-script names -> English letters in S1's own spelling.

1. A word dictionary learned from training pairs of the Fit ROLE only (no Tune/C-select/Audit labels): for each
   true pair whose S2/S3 name has Indian script, each Indian-script word is lined up with the S1 word it most
   resembles after generic conversion (anyascii); mappings seen >= 3 times with >= 60% agreement are kept.
   The result is S1's spelling (e.g. लक्ष्मी -> whatever S1 writes, Lakshmi or Laxmi).
2. Words not in the dictionary: anyascii, letters/digits only, known suffix spellings fixed, final "mg" -> "ng".
The same dictionary is applied to train and test.
"""
import re

import numpy as np
import polars as pl
from anyascii import anyascii
from rapidfuzz import fuzz, process

INDIC_RE = r"[\x{0900}-\x{0DFF}]"  # Devanagari ... Malayalam (all Indian scripts in the data)
_INDIC = re.compile("[ऀ-෿]")
FALLBACK_FIXES = {  # anyascii spellings of the legal words (seen in the EDA)
    "praivet": "private", "praibhet": "private", "piraivet": "private", "praivrr": "private", "pra": "private",
    "limitet": "limited", "limirrd": "limited", "limtid": "limited", "li": "limited", "elelpi": "llp",
}


def _ascii(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", anyascii(token).lower())


def fallback(token: str) -> str:
    a = _ascii(token)
    a = FALLBACK_FIXES.get(a, a)
    return re.sub(r"mg$", "ng", a)


def make_converter(mapping: dict[str, str]):
    """Returns f(cleaned text) -> text with every Indian-script word replaced (dictionary, else fallback)."""
    def convert(text: str) -> str:
        out = []
        for tok in text.split(" "):
            if _INDIC.search(tok):
                tok = mapping.get(tok) or fallback(tok)
            if tok:
                out.append(tok)
        return " ".join(out)
    return convert


def build_dictionary(train: pl.DataFrame, pairs: pl.DataFrame, fit_role_ids: pl.Series,
                     min_count: int = 3, min_share: float = 0.6, min_sim: float = 50.0) -> tuple[dict, dict]:
    from .normalize import basic_clean

    s1 = (train.filter(pl.col("source") == "S1").join(pl.DataFrame({"entity_id": fit_role_ids}), on="entity_id", how="semi")
          .select(s1_id="entity_id", s1_name=basic_clean(pl.col("business_name"))))
    tg = (train.filter((pl.col("source") != "S1") & pl.col("business_name").fill_null("").str.contains(INDIC_RE))
          .select(target_id="entity_id", t_name=basic_clean(pl.col("business_name"))))
    p = pairs.join(s1, on="s1_id").join(tg, on="target_id").with_row_index("pair")
    nat = (p.select("pair", tok=pl.col("t_name").str.split(" ")).explode("tok", empty_as_null=True)
           .filter(pl.col("tok").is_not_null() & pl.col("tok").str.contains(INDIC_RE)).unique(["pair", "tok"]))
    lat = (p.select("pair", lat=pl.col("s1_name").str.split(" ")).explode("lat", empty_as_null=True)
           .filter(pl.col("lat").is_not_null() & (pl.col("lat") != "")).unique(["pair", "lat"]))
    uniq = nat["tok"].unique()
    asc = pl.Series([_ascii(t) for t in uniq.to_list()], dtype=pl.String)
    cross = nat.join(lat, on="pair").with_columns(asc=pl.col("tok").replace_strict(uniq, asc))
    cross = cross.with_columns(sim=pl.Series(process.cpdist(cross["asc"].to_list(), cross["lat"].to_list(),
                                                            scorer=fuzz.ratio, workers=-1, dtype=np.float32)))
    best = (cross.sort(["pair", "tok", "sim", "lat"], descending=[False, False, True, False])
            .unique(["pair", "tok"], keep="first", maintain_order=True)
            .filter(pl.col("sim") >= min_sim))
    counts = (best.group_by("tok", "lat").agg(n=pl.len())
              .with_columns(total=pl.col("n").sum().over("tok"))
              .with_columns(share=pl.col("n") / pl.col("total"))
              .sort(["tok", "n", "lat"], descending=[False, True, False])
              .unique("tok", keep="first", maintain_order=True)
              .filter((pl.col("n") >= min_count) & (pl.col("share") >= min_share)))
    mapping = dict(zip(counts["tok"].to_list(), counts["lat"].to_list()))
    stats = {"training_pairs_used": p.height, "distinct_indic_words_seen": uniq.len(), "dictionary_size": len(mapping),
             "min_count": min_count, "min_share": min_share, "min_sim": min_sim}
    return mapping, stats


def coverage(records: pl.DataFrame, mapping: dict) -> float:
    """Share of Indian-script word occurrences in S2/S3 names that the dictionary covers."""
    from .normalize import basic_clean

    toks = (records.filter(pl.col("source") != "S1").select(tok=basic_clean(pl.col("business_name")).str.split(" "))
            .explode("tok", empty_as_null=True).filter(pl.col("tok").is_not_null() & pl.col("tok").str.contains(INDIC_RE)))
    known = pl.DataFrame({"tok": list(mapping)}, schema={"tok": pl.String}).with_columns(known=pl.lit(True))
    return float(toks.join(known, on="tok", how="left")["known"].is_not_null().mean()) if toks.height else float("nan")
