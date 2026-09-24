"""The 12 pair features of Plan 1 section 5. Similarities are scaled to 0-1.

Undefined values stay missing (NaN), never a perfect score: a similarity is undefined when either text is
empty, the number Jaccard when both number sets are empty, the length ratio when either address is empty.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

FEATURES = [
    "name_red_ratio",      # 01 reduced-name string ratio
    "name_red_token_set",  # 02 reduced-name token-set ratio
    "name_cons_ratio",     # 03 conservative-name string ratio
    "addr_token_set",      # 04 address token-set ratio
    "addr_partial",        # 05 address partial-string ratio
    "addr_len_ratio",      # 06 shorter/longer address length
    "cand_addr_missing",   # 07 candidate address missing
    "num_jaccard",         # 08 number-token Jaccard
    "num_conflict",        # 09 both number sets present but disjoint
    "retrieval_score",     # 10 word retrieval cosine
    "retrieval_rank",      # 11 rank within source
    "source_s3",           # 12 S2 = 0, S3 = 1
]
_TEXT_COLS = ["name_red", "name_cons", "addr_norm", "addr_nums", "addr_missing"]


def _sim(a: pl.Series, b: pl.Series, scorer) -> np.ndarray:
    """rapidfuzz scorer row by row (multi-threaded), 0-1, NaN where either side is empty."""
    a, b = a.fill_null(""), b.fill_null("")
    out = process.cpdist(a.to_list(), b.to_list(), scorer=scorer, workers=-1, dtype=np.float32) / np.float32(100)
    out[((a == "") | (b == "")).to_numpy()] = np.nan
    return out


def text_lookup(norm: pl.DataFrame, s1_ids: pl.Series, target_ids: pl.Series) -> tuple[pl.DataFrame, pl.DataFrame]:
    """The normalized text of the S1s and targets that appear in a candidate table (keeps later joins small)."""
    q = norm.join(pl.DataFrame({"entity_id": s1_ids.unique()}), on="entity_id", how="semi")
    t = norm.join(pl.DataFrame({"entity_id": target_ids.unique()}), on="entity_id", how="semi")
    q = q.select(pl.col("entity_id").alias("s1_id"), *[pl.col(c).alias(f"q_{c}") for c in _TEXT_COLS])
    t = t.select(pl.col("entity_id").alias("target_id"), *[pl.col(c).alias(f"t_{c}") for c in _TEXT_COLS])
    return q, t


def build_features(cands: pl.DataFrame, q: pl.DataFrame, t: pl.DataFrame, batch: int = 2_000_000) -> pl.DataFrame:
    """cands: s1_id, target_id, source, score, rank. q/t from text_lookup.

    Returns s1_id, target_id, source + the 12 FEATURES (Float32), in the order of `cands`.
    """
    parts = []
    for start in range(0, cands.height, batch):
        b = (
            cands.slice(start, batch)
            .select("s1_id", "target_id", "source", "score", "rank")
            .join(q, on="s1_id", how="left", maintain_order="left")
            .join(t, on="target_id", how="left", maintain_order="left")
        )
        if b["q_name_cons"].is_null().any() or b["t_name_cons"].is_null().any():
            raise ValueError("candidate ids missing from the normalized records")
        qa, ta = b["q_addr_norm"].fill_null(""), b["t_addr_norm"].fill_null("")
        addr_empty = ((qa == "") | (ta == "")).to_numpy()
        la = qa.str.len_chars().cast(pl.Float32).to_numpy()
        lb = ta.str.len_chars().cast(pl.Float32).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            len_ratio = np.minimum(la, lb) / np.maximum(la, lb)
        len_ratio[addr_empty] = np.nan

        nq, nt = b["q_addr_nums"], b["t_addr_nums"]
        inter = nq.list.set_intersection(nt).list.len().cast(pl.Float32).to_numpy()
        union = nq.list.set_union(nt).list.len().cast(pl.Float32).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            jaccard = np.where(union > 0, inter / union, np.nan).astype(np.float32)
        conflict = ((nq.list.len() > 0) & (nt.list.len() > 0)).to_numpy() & (inter == 0)

        parts.append(pl.DataFrame({
            "s1_id": b["s1_id"],
            "target_id": b["target_id"],
            "source": b["source"],
            "name_red_ratio": _sim(b["q_name_red"], b["t_name_red"], fuzz.ratio),
            "name_red_token_set": _sim(b["q_name_red"], b["t_name_red"], fuzz.token_set_ratio),
            "name_cons_ratio": _sim(b["q_name_cons"], b["t_name_cons"], fuzz.ratio),
            "addr_token_set": _sim(qa, ta, fuzz.token_set_ratio),
            "addr_partial": _sim(qa, ta, fuzz.partial_ratio),
            "addr_len_ratio": len_ratio.astype(np.float32),
            "cand_addr_missing": (ta == "").cast(pl.Float32),
            "num_jaccard": jaccard,
            "num_conflict": conflict.astype(np.float32),
            "retrieval_score": b["score"].cast(pl.Float32),
            "retrieval_rank": b["rank"].cast(pl.Float32),
            "source_s3": (b["source"] == "S3").cast(pl.Float32),
        }))
    out = pl.concat(parts).with_columns(pl.col(FEATURES).fill_nan(None))  # NaN -> null: one kind of "missing"
    assert out.columns[3:] == FEATURES
    return out


def matrix(feats: pl.DataFrame) -> np.ndarray:
    """Feature matrix in the fixed FEATURES order; nulls become NaN (LightGBM's missing value)."""
    return feats.select(FEATURES).cast(pl.Float32).fill_null(np.nan).to_numpy()
