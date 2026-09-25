"""Indian-script names -> English letters in S1's own spelling.

1. A word dictionary learned from training pairs of the Fit ROLE only (no Tune/C-select/Audit labels): for each
   true pair whose S2/S3 name has Indian script, each Indian-script word is lined up with the S1 word it most
   resembles after generic conversion (anyascii). Resemblance is the better of the plain spelling ratio and a
   sound-skeleton ratio (so anyascii "phud" lines up with "foods", "tekanalaji" with "technology").
   A mapping is kept when it is seen >= 3 times, wins >= 50% of ALL appearances of the word (not just the
   aligned ones, which let rare accidental alignments through), and its average similarity is >= 60.
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
_SKELETON_MAP = str.maketrans({"c": "k", "q": "k", "z": "j", "w": "v"})


def _ascii(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", anyascii(token).lower())


def skeleton(word: str) -> str:
    """Rough sound skeleton: ph->f, c/q->k, z->j, w->v, x->ks, drop h and vowels, merge doubled letters,
    drop a final s (plural)."""
    s = word.lower().replace("ph", "f").translate(_SKELETON_MAP).replace("x", "ks")
    s = re.sub(r"[haeiouy]", "", s)
    s = re.sub(r"(.)\1+", r"\1", s)
    return s[:-1] if len(s) > 2 and s.endswith("s") else s


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


def _map_series(values: pl.Series, fn) -> pl.Series:
    uniq = values.unique()
    return values.replace_strict(uniq, pl.Series([fn(v) for v in uniq.to_list()], dtype=pl.String))


def build_dictionary(train: pl.DataFrame, pairs: pl.DataFrame, fit_role_ids: pl.Series, min_count: int = 3,
                     min_share: float = 0.5, min_sim: float = 50.0, min_mean_sim: float = 60.0) -> tuple[dict, dict]:
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
    occurrences = nat.group_by("tok").agg(occ=pl.len())

    cross = nat.join(lat, on="pair")
    cross = cross.with_columns(asc=_map_series(cross["tok"], _ascii))
    cross = cross.with_columns(sk_a=_map_series(cross["asc"], skeleton), sk_l=_map_series(cross["lat"], skeleton))
    plain = process.cpdist(cross["asc"].to_list(), cross["lat"].to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    skel = process.cpdist(cross["sk_a"].to_list(), cross["sk_l"].to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    short = ((cross["sk_a"].str.len_chars() < 2) | (cross["sk_l"].str.len_chars() < 2)).to_numpy()
    skel[short] = 0  # 1-letter skeletons match too easily
    cross = cross.with_columns(plain=pl.Series(plain), sim=pl.Series(np.maximum(plain, skel)))

    best = (cross.sort(["pair", "tok", "sim", "lat"], descending=[False, False, True, False])
            .unique(["pair", "tok"], keep="first", maintain_order=True)
            .filter(pl.col("sim") >= min_sim))
    top = (best.group_by("tok", "lat").agg(n=pl.len(), mean_sim=pl.col("sim").mean(), mean_plain=pl.col("plain").mean())
           .join(occurrences, on="tok")
           .with_columns(share=pl.col("n") / pl.col("occ"))
           .sort(["tok", "n", "lat"], descending=[False, True, False])
           .unique("tok", keep="first", maintain_order=True))
    ok = (pl.col("n") >= min_count) & (pl.col("share") >= min_share) & (pl.col("mean_sim") >= min_mean_sim)
    kept = top.filter(ok)
    mapping = dict(zip(kept["tok"].to_list(), kept["lat"].to_list()))

    def rows(df: pl.DataFrame) -> list[dict]:
        return df.select("tok", "lat", "n", "occ", pl.col("share").round(2), pl.col("mean_sim").round(1),
                         pl.col("mean_plain").round(1)).to_dicts()

    stats = {
        "training_pairs_used": p.height, "distinct_indic_words_seen": occurrences.height, "dictionary_size": len(mapping),
        "min_count": min_count, "min_share": min_share, "min_sim": min_sim, "min_mean_sim": min_mean_sim,
        # for eyeballing: frequent words that did NOT get a mapping, and kept mappings that rely on the skeleton
        "dropped_most_frequent": rows(top.filter(~ok).sort("occ", descending=True).head(25)),
        "kept_by_sound_only": rows(kept.filter(pl.col("mean_plain") < 50).sort("n", descending=True).head(25)),
    }
    return mapping, stats


def coverage(records: pl.DataFrame, mapping: dict) -> float:
    """Share of Indian-script word occurrences in S2/S3 names that the dictionary covers."""
    from .normalize import basic_clean

    toks = (records.filter(pl.col("source") != "S1").select(tok=basic_clean(pl.col("business_name")).str.split(" "))
            .explode("tok", empty_as_null=True).filter(pl.col("tok").is_not_null() & pl.col("tok").str.contains(INDIC_RE)))
    known = pl.DataFrame({"tok": list(mapping)}, schema={"tok": pl.String}).with_columns(known=pl.lit(True))
    return float(toks.join(known, on="tok", how="left")["known"].is_not_null().mean()) if toks.height else float("nan")
