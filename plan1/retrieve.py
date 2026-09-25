"""Candidate search. Queries only ever search targets of their own country string (any label works).

Main channel (Plan 1): word-unigram TF-IDF over reduced name + normalized address, one index per
(country, target source) fitted on that partition's target texts; top 50 per source by cosine.

Fallback channel (p1-v3): records with NO address can only be matched on their name, and they were the weakest
slice (85.8% found). A letter-group (char 3-4 gram) TF-IDF index over the names of those records is searched
with the S1's reduced name; its top 10 per source are added. Records found only this way get their main-channel
cosine computed directly and rank TOP_K + 1; fb_score (the fallback cosine) is null for everything else.
"""
import os
import time

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from . import config as C

TFIDF_PARAMS = dict(
    analyzer=str.split,  # texts are already cleaned into space-separated [\p{L}\p{M}\p{N}]+ tokens
    min_df=2,
    max_df=1.0,
    max_features=300_000,
    sublinear_tf=True,
    smooth_idf=True,
    norm="l2",
    dtype=np.float32,
)
FALLBACK_TFIDF_PARAMS = dict(
    analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_features=500_000,
    sublinear_tf=True, smooth_idf=True, norm="l2", dtype=np.float32,
)
TOP_K = 50
FALLBACK_TOP_K = C.FALLBACK_TOP_K
QUERY_BATCH = 20_000
CAND_SCHEMA = {"s1_id": pl.String, "target_id": pl.String, "source": pl.String, "score": pl.Float32,
               "rank": pl.Int32, "fb_score": pl.Float32}


def _topk(qm, by_term, q_ids: pl.Series, t_ids: pl.Series, k: int, n_threads: int | None) -> pl.DataFrame:
    res = sp_matmul_topn(qm, by_term, top_n=k, sort=True, n_threads=n_threads or os.cpu_count()).tocsr()
    rows = np.repeat(np.arange(res.shape[0], dtype=np.uint32), np.diff(res.indptr))
    return pl.DataFrame({
        "s1_id": q_ids.gather(pl.Series(rows)),
        "target_id": t_ids.gather(pl.Series(res.indices.astype(np.uint32))),
        "score": pl.Series(res.data.astype(np.float32)),
    }).filter(pl.col("score") > 0)


class TargetIndex:
    """Main channel for one (country, source) partition."""

    def __init__(self, targets: pl.DataFrame):
        self.ids = targets["entity_id"]
        self.vec = TfidfVectorizer(**TFIDF_PARAMS)
        self.matrix = self.vec.fit_transform(targets["retrieval_text"].to_list()).tocsr()  # targets x vocabulary
        self.by_term = self.matrix.T.tocsr()  # vocabulary x targets
        self.row_of = pl.DataFrame({"target_id": self.ids, "t_row": pl.int_range(self.ids.len(), eager=True, dtype=pl.UInt32)})

    @property
    def vocabulary_size(self) -> int:
        return len(self.vec.vocabulary_)

    def transform(self, texts: pl.Series):
        return self.vec.transform(texts.to_list())

    def topk(self, qm, q_ids: pl.Series, k: int = TOP_K, n_threads: int | None = None) -> pl.DataFrame:
        return _topk(qm, self.by_term, q_ids, self.ids, k, n_threads)

    def pair_cosine(self, qm, q_rows: np.ndarray, t_rows: np.ndarray) -> np.ndarray:
        return np.asarray(qm[q_rows].multiply(self.matrix[t_rows]).sum(axis=1)).ravel().astype(np.float32)


class NameFallbackIndex:
    """Fallback channel: letter groups of the names of records without an address."""

    def __init__(self, targets: pl.DataFrame):
        self.ids = targets["entity_id"]
        self.vec = TfidfVectorizer(**FALLBACK_TFIDF_PARAMS)
        self.by_term = self.vec.fit_transform(targets["name_red"].to_list()).T.tocsr()

    @classmethod
    def build(cls, targets: pl.DataFrame):
        t = targets.filter(pl.col("addr_missing") & (pl.col("name_red") != ""))
        if t.height < 2:
            return None
        try:
            return cls(t)
        except ValueError:  # empty vocabulary
            return None

    def query(self, queries: pl.DataFrame, k: int = FALLBACK_TOP_K, n_threads: int | None = None) -> pl.DataFrame:
        qm = self.vec.transform(queries["name_red"].to_list())
        return _topk(qm, self.by_term, queries["entity_id"], self.ids, k, n_threads).rename({"score": "fb_score"})


def build_indexes(norm: pl.DataFrame, country: str) -> dict:
    """{source: (TargetIndex, NameFallbackIndex or None)} for one country."""
    out = {}
    for src in C.TARGET_SOURCES:
        t = norm.filter((pl.col("source") == src) & (pl.col("country") == country))
        if t.height:
            out[src] = (TargetIndex(t), NameFallbackIndex.build(t))
    return out


def search_batch(qb: pl.DataFrame, indexes: dict, top_k: int = TOP_K, fb_k: int = FALLBACK_TOP_K,
                 n_threads: int | None = None) -> pl.DataFrame:
    """qb: S1 rows (entity_id, retrieval_text, name_red). Returns CAND_SCHEMA columns.

    rank = 1..top_k within (S1, source) for main-channel results (ties by target id); top_k + 1 for records
    found only by the fallback.
    """
    qpos = qb.select(s1_id="entity_id").with_row_index("q_row")
    parts = []
    for src, (main, fb) in indexes.items():
        qm = main.transform(qb["retrieval_text"])
        m = main.topk(qm, qb["entity_id"], top_k, n_threads)
        m = m.sort(["s1_id", "score", "target_id"], descending=[False, True, False]).with_columns(
            rank=(pl.int_range(pl.len()).over("s1_id") + 1).cast(pl.Int32)
        )
        if fb is not None:
            f = fb.query(qb, fb_k, n_threads)
            extra = (f.join(m, on=["s1_id", "target_id"], how="anti")
                     .join(qpos, on="s1_id").join(main.row_of, on="target_id"))
            if extra.height:
                cos = main.pair_cosine(qm, extra["q_row"].to_numpy(), extra["t_row"].to_numpy())
                m = pl.concat([m, extra.select("s1_id", "target_id").with_columns(
                    score=pl.Series(cos, dtype=pl.Float32), rank=pl.lit(top_k + 1, dtype=pl.Int32))])
            m = m.join(f, on=["s1_id", "target_id"], how="left")
        else:
            m = m.with_columns(fb_score=pl.lit(None, dtype=pl.Float32))
        parts.append(m.with_columns(source=pl.lit(src)))
    if not parts:
        return pl.DataFrame(schema=CAND_SCHEMA)
    return pl.concat(parts).select(list(CAND_SCHEMA)).cast(CAND_SCHEMA)


def retrieve(norm: pl.DataFrame, query_ids: pl.Series, top_k: int = TOP_K, batch: int = QUERY_BATCH,
             n_threads: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """norm: normalized records of ONE split. Returns (candidates with a country column, timings)."""
    queries = norm.filter(pl.col("source") == "S1").join(pl.DataFrame({"entity_id": query_ids}), on="entity_id", how="semi")
    parts, timings = [], []
    for country in sorted(queries["country"].unique().to_list()):
        q = queries.filter(pl.col("country") == country)
        t0 = time.time()
        indexes = build_indexes(norm, country)
        build_s = time.time() - t0
        for start in range(0, q.height, batch):
            qb = q.slice(start, batch)
            t1 = time.time()
            parts.append(search_batch(qb, indexes, top_k, n_threads=n_threads).with_columns(country=pl.lit(country)))
            timings.append({"country": country, "batch_start": start, "queries": qb.height,
                            "index_build_s": build_s if start == 0 else 0.0, "query_s": time.time() - t1})
        sizes = ", ".join(f"{s}: {m.ids.len():,} (+{fb.ids.len() if fb else 0:,} no-address)" for s, (m, fb) in indexes.items())
        print(f"  {country}: {q.height:,} queries | {sizes} | build {build_s:.0f}s, "
              f"query {sum(x['query_s'] for x in timings if x['country'] == country):.0f}s")
    return pl.concat(parts), pl.DataFrame(timings)


def candidate_report(cands: pl.DataFrame, truth: pl.DataFrame, universe: pl.Series, norm: pl.DataFrame) -> dict:
    """Candidate recall (overall, by country/source, Indic names, missing addresses, by rank, fallback) and the
    candidate-oracle macro F0.5 (predict exactly the true links that were retrieved)."""
    from .metric import score

    uni = pl.DataFrame({"s1_id": universe})
    t = (
        truth.join(uni, on="s1_id", how="semi")
        .join(norm.select(target_id="entity_id", source="source", country="country",
                          name_nonlatin="name_nonlatin", addr_missing="addr_missing"), on="target_id")
        .join(cands.select("s1_id", "target_id", "rank"), on=["s1_id", "target_id"], how="left")
        .with_columns(found=pl.col("rank").is_not_null())
    )
    c = cands.join(uni, on="s1_id", how="semi")
    per_s1 = uni.join(c.group_by("s1_id").agg(n=pl.len()), on="s1_id", how="left").with_columns(pl.col("n").fill_null(0))
    out = {
        "n_s1": uni.height,
        "true_links": t.height,
        "candidate_recall": t["found"].mean(),
        "true_links_found_only_by_fallback": t.select((pl.col("rank") == TOP_K + 1).fill_null(False).mean()).item(),
        "mean_candidates_per_s1": per_s1["n"].mean(),
        "candidates_from_fallback_only_share": c.select((pl.col("rank") == TOP_K + 1).mean()).item() if c.height else 0.0,
        "s1_without_candidates": int((per_s1["n"] == 0).sum()),
        "candidate_oracle": score(t.filter(pl.col("found")).select("s1_id", "target_id"), truth, universe),
        "recall_at_rank": {k: t.select((pl.col("rank") <= k).fill_null(False).mean()).item() for k in (1, 5, 10, 20, 50)},
    }
    for col in ("country", "source", "name_nonlatin", "addr_missing"):
        g = t.group_by(col).agg(links=pl.len(), recall=pl.col("found").mean()).sort(col)
        out[f"recall_by_{col}"] = {str(r[col]): {"links": r["links"], "recall": r["recall"]} for r in g.iter_rows(named=True)}
    return out
