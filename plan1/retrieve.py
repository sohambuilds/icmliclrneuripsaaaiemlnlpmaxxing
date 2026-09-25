"""One retrieval channel (Plan 1 section 4): word-unigram TF-IDF over reduced name + normalized address.

One index per (country, target source), fitted on that partition's target texts only. Each S1 query gets its
top 50 S2 and top 50 S3 candidates by cosine; zero-score results are never kept and lists are never padded.
Queries only ever search targets of their own country string (any country label works, France included).
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
TOP_K = 50
QUERY_BATCH = 20_000


class TargetIndex:
    """TF-IDF fitted on one (country, source) partition's target texts; returns top-k targets by cosine."""

    def __init__(self, targets: pl.DataFrame):
        self.ids = targets["entity_id"]
        self.vec = TfidfVectorizer(**TFIDF_PARAMS)
        self.by_term = self.vec.fit_transform(targets["retrieval_text"].to_list()).T.tocsr()  # vocabulary x targets

    @property
    def vocabulary_size(self) -> int:
        return len(self.vec.vocabulary_)

    def query(self, queries: pl.DataFrame, top_k: int = TOP_K, n_threads: int | None = None) -> pl.DataFrame:
        """queries: entity_id, retrieval_text. Returns s1_id, target_id, score (positive scores only)."""
        qm = self.vec.transform(queries["retrieval_text"].to_list())
        res = sp_matmul_topn(qm, self.by_term, top_n=top_k, sort=True, n_threads=n_threads or os.cpu_count()).tocsr()
        rows = np.repeat(np.arange(res.shape[0], dtype=np.uint32), np.diff(res.indptr))
        return pl.DataFrame({
            "s1_id": queries["entity_id"].gather(pl.Series(rows)),
            "target_id": self.ids.gather(pl.Series(res.indices.astype(np.uint32))),
            "score": pl.Series(res.data.astype(np.float32)),
        }).filter(pl.col("score") > 0)


def add_rank(cands: pl.DataFrame) -> pl.DataFrame:
    """rank 1 = best within (s1_id, source); ties broken by target id."""
    return cands.sort(["s1_id", "source", "score", "target_id"], descending=[False, False, True, False]).with_columns(
        rank=(pl.int_range(pl.len()).over("s1_id", "source") + 1).cast(pl.Int32)
    )


def retrieve(norm: pl.DataFrame, query_ids: pl.Series, top_k: int = TOP_K, batch: int = QUERY_BATCH,
             n_threads: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """norm: normalized records of ONE split (S1 queries and S2/S3 targets).

    Returns (candidates, timings). candidates: s1_id, target_id, source, country, score, rank.
    """
    queries = norm.filter(pl.col("source") == "S1").join(pl.DataFrame({"entity_id": query_ids}), on="entity_id", how="semi")
    parts, timings = [], []
    for country in sorted(queries["country"].unique().to_list()):
        q = queries.filter(pl.col("country") == country)
        for src in C.TARGET_SOURCES:
            t = norm.filter((pl.col("source") == src) & (pl.col("country") == country))
            if t.height == 0:
                continue
            t0 = time.time()
            index = TargetIndex(t)
            build_s = time.time() - t0
            for start in range(0, q.height, batch):
                qb = q.slice(start, batch)
                t1 = time.time()
                parts.append(index.query(qb, top_k, n_threads).with_columns(source=pl.lit(src), country=pl.lit(country)))
                timings.append({"country": country, "source": src, "targets": t.height, "vocabulary": index.vocabulary_size,
                                "batch_start": start, "queries": qb.height, "index_build_s": build_s if start == 0 else 0.0,
                                "query_s": time.time() - t1})
            print(f"  {country}/{src}: {t.height:,} targets, vocab {index.vocabulary_size:,}, {q.height:,} queries, "
                  f"build {build_s:.0f}s, query {sum(x['query_s'] for x in timings if x['country'] == country and x['source'] == src):.0f}s")
    return add_rank(pl.concat(parts)), pl.DataFrame(timings)


def candidate_report(cands: pl.DataFrame, truth: pl.DataFrame, universe: pl.Series, norm: pl.DataFrame) -> dict:
    """Candidate recall (overall, by country/source, Indic names, missing addresses, by rank) and the
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
        "mean_candidates_per_s1": per_s1["n"].mean(),
        "s1_without_candidates": int((per_s1["n"] == 0).sum()),
        "candidate_oracle": score(t.filter(pl.col("found")).select("s1_id", "target_id"), truth, universe),
        "recall_at_rank": {k: t.select((pl.col("rank") <= k).fill_null(False).mean()).item() for k in (1, 5, 10, 20, 50)},
    }
    for col in ("country", "source", "name_nonlatin", "addr_missing"):
        g = t.group_by(col).agg(links=pl.len(), recall=pl.col("found").mean()).sort(col)
        out[f"recall_by_{col}"] = {str(r[col]): {"links": r["links"], "recall": r["recall"]} for r in g.iter_rows(named=True)}
    return out
