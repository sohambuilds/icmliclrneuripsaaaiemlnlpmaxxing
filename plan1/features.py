"""Pair features. Similarities are scaled to 0-1.

Undefined values stay missing, never a perfect score: a similarity is undefined when either text is empty, the
number Jaccard when both number sets are empty, the length ratio when either address is empty.

Run p1-v2: the 12 Plan 1 features plus
  - more direct similarities (token sort, partial, Jaro-Winkler, no-space name ratio, address ratio, shared numbers),
  - features relative to the S1's other candidates (share of best retrieval score, gap to best, ...),
  - similarity to the S1's anchor: its best other candidate by retrieval score (duplicates of one business
    tend to resemble each other).
The relative and anchor features need all candidates of an S1 in the same call.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

BASE_FEATURES = [
    "name_red_ratio",      # reduced-name string ratio
    "name_red_token_set",  # reduced-name token-set ratio
    "name_cons_ratio",     # conservative-name string ratio
    "addr_token_set",      # address token-set ratio
    "addr_partial",        # address partial-string ratio
    "addr_len_ratio",      # shorter/longer address length
    "cand_addr_missing",   # candidate address missing
    "num_jaccard",         # number-token Jaccard
    "num_conflict",        # both number sets present but disjoint
    "retrieval_score",     # word retrieval cosine
    "retrieval_rank",      # rank within source
    "source_s3",           # S2 = 0, S3 = 1
]
EXTRA_FEATURES = [
    "name_red_token_sort", "name_red_partial", "name_red_jw", "name_nospace_ratio", "addr_ratio",
    "cand_name_nonlatin", "num_shared",
    "score_rel_max", "score_gap_max", "rank_overall", "n_close", "n_name_close", "name_set_gap", "addr_set_gap",
    "anchor_name_set", "anchor_addr_set",
]
FEATURES = BASE_FEATURES + EXTRA_FEATURES
_TEXT_COLS = ["name_red", "name_cons", "addr_norm", "addr_nums", "addr_missing", "name_nonlatin"]
F32 = pl.Float32


def _sim(a: pl.Series, b: pl.Series, scorer, scale: float = 100.0) -> np.ndarray:
    """rapidfuzz scorer row by row (multi-threaded), 0-1, NaN where either side is empty."""
    a, b = a.fill_null(""), b.fill_null("")
    out = process.cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=-1, dtype=np.float32) / np.float32(scale)
    out[((a == "") | (b == "")).to_numpy()] = np.nan
    return out


def text_lookup(norm: pl.DataFrame, s1_ids: pl.Series, target_ids: pl.Series) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The normalized text of the S1s and targets that appear in a candidate table (keeps later joins small)."""
    q = norm.join(pl.DataFrame({"entity_id": s1_ids.unique()}), on="entity_id", how="semi")
    t = norm.join(pl.DataFrame({"entity_id": target_ids.unique()}), on="entity_id", how="semi")
    q = q.select(pl.col("entity_id").alias("s1_id"), *[pl.col(c).alias(f"q_{c}") for c in _TEXT_COLS])
    t = t.select(pl.col("entity_id").alias("target_id"), *[pl.col(c).alias(f"t_{c}") for c in _TEXT_COLS])
    return q, t


def _add_anchor(cands: pl.DataFrame) -> pl.DataFrame:
    """anchor_id = the S1's best candidate by retrieval score, or its second best for the best one itself."""
    top = (cands.select("s1_id", "target_id", "score")
           .sort(["s1_id", "score", "target_id"], descending=[False, True, False])
           .with_columns(k=pl.int_range(pl.len()).over("s1_id"))
           .filter(pl.col("k") < 2))
    a1 = top.filter(pl.col("k") == 0).select("s1_id", a1="target_id")
    a2 = top.filter(pl.col("k") == 1).select("s1_id", a2="target_id")
    return (cands.join(a1, on="s1_id", how="left", maintain_order="left")
            .join(a2, on="s1_id", how="left", maintain_order="left")
            .with_columns(anchor_id=pl.when(pl.col("target_id") == pl.col("a1")).then(pl.col("a2")).otherwise(pl.col("a1")))
            .drop("a1", "a2"))


def build_features(cands: pl.DataFrame, q: pl.DataFrame, t: pl.DataFrame, batch: int = 2_000_000) -> pl.DataFrame:
    """cands: s1_id, target_id, source, score, rank (ALL candidates of each S1 present). q/t from text_lookup.

    Returns s1_id, target_id, source + FEATURES (Float32), in the order of `cands`.
    """
    cands = _add_anchor(cands.select("s1_id", "target_id", "source", "score", "rank"))
    anchor_text = t.select(anchor_id="target_id", a_name_red="t_name_red", a_addr_norm="t_addr_norm")
    parts = []
    for start in range(0, cands.height, batch):
        b = (
            cands.slice(start, batch)
            .join(q, on="s1_id", how="left", maintain_order="left")
            .join(t, on="target_id", how="left", maintain_order="left")
            .join(anchor_text, on="anchor_id", how="left", maintain_order="left")
        )
        if b["q_name_cons"].is_null().any() or b["t_name_cons"].is_null().any():
            raise ValueError("candidate ids missing from the normalized records")
        qn, tn = b["q_name_red"], b["t_name_red"]
        qa, ta = b["q_addr_norm"].fill_null(""), b["t_addr_norm"].fill_null("")
        addr_empty = ((qa == "") | (ta == "")).to_numpy()
        la = qa.str.len_chars().cast(F32).to_numpy()
        lb = ta.str.len_chars().cast(F32).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            len_ratio = np.minimum(la, lb) / np.maximum(la, lb)
        len_ratio[addr_empty] = np.nan

        nq, nt = b["q_addr_nums"], b["t_addr_nums"]
        inter = nq.list.set_intersection(nt).list.len().cast(F32).to_numpy()
        union = nq.list.set_union(nt).list.len().cast(F32).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard = np.where(union > 0, inter / union, np.nan).astype(np.float32)
        conflict = ((nq.list.len() > 0) & (nt.list.len() > 0)).to_numpy() & (inter == 0)

        parts.append(pl.DataFrame({
            "s1_id": b["s1_id"],
            "target_id": b["target_id"],
            "source": b["source"],
            "name_red_ratio": _sim(qn, tn, fuzz.ratio),
            "name_red_token_set": _sim(qn, tn, fuzz.token_set_ratio),
            "name_cons_ratio": _sim(b["q_name_cons"], b["t_name_cons"], fuzz.ratio),
            "addr_token_set": _sim(qa, ta, fuzz.token_set_ratio),
            "addr_partial": _sim(qa, ta, fuzz.partial_ratio),
            "addr_len_ratio": len_ratio.astype(np.float32),
            "cand_addr_missing": (ta == "").cast(F32),
            "num_jaccard": jaccard,
            "num_conflict": conflict.astype(np.float32),
            "retrieval_score": b["score"].cast(F32),
            "retrieval_rank": b["rank"].cast(F32),
            "source_s3": (b["source"] == "S3").cast(F32),
            "name_red_token_sort": _sim(qn, tn, fuzz.token_sort_ratio),
            "name_red_partial": _sim(qn, tn, fuzz.partial_ratio),
            "name_red_jw": _sim(qn, tn, JaroWinkler.normalized_similarity, scale=1.0),
            "name_nospace_ratio": _sim(qn.fill_null("").str.replace_all(" ", ""), tn.fill_null("").str.replace_all(" ", ""), fuzz.ratio),
            "addr_ratio": _sim(qa, ta, fuzz.ratio),
            "cand_name_nonlatin": b["t_name_nonlatin"].cast(F32),
            "num_shared": inter,
            "anchor_name_set": _sim(tn, b["a_name_red"], fuzz.token_set_ratio),
            "anchor_addr_set": _sim(ta, b["a_addr_norm"], fuzz.token_set_ratio),
        }))
    out = pl.concat(parts)
    floats = [c for c in out.columns if c not in ("s1_id", "target_id", "source")]
    out = out.with_columns(pl.col(floats).fill_nan(None))  # NaN -> null: one kind of "missing"

    # ---- relative to the S1's other candidates (whole candidate set of each S1) ----
    s, nset, aset = pl.col("retrieval_score"), pl.col("name_red_token_set"), pl.col("addr_token_set")
    out = out.with_columns(
        score_rel_max=(s / s.max()).over("s1_id").cast(F32),
        score_gap_max=(s.max() - s).over("s1_id").cast(F32),
        rank_overall=s.rank("min", descending=True).over("s1_id").cast(F32),
        n_close=(s >= 0.9 * s.max()).sum().over("s1_id").cast(F32),
        n_name_close=(nset >= 0.9).sum().over("s1_id").cast(F32),
        name_set_gap=(nset.max() - nset).over("s1_id").cast(F32),
        addr_set_gap=(aset.max() - aset).over("s1_id").cast(F32),
    )
    out = out.select("s1_id", "target_id", "source", *FEATURES)
    return out


def matrix(feats: pl.DataFrame) -> np.ndarray:
    """Feature matrix in the fixed FEATURES order; nulls become NaN (LightGBM's missing value)."""
    return feats.select(FEATURES).cast(pl.Float32).fill_null(np.nan).to_numpy()
