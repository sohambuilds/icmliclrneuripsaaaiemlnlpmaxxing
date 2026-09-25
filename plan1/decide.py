"""Turning pair scores into answers."""
import polars as pl


def one_owner(accepted: pl.DataFrame) -> pl.DataFrame:
    """Keep each target only for its highest-scoring S1 (ties -> smallest S1 id).

    In the training labels every S2/S3 record belongs to at most one S1, so all but one claim on a record
    accepted for several S1s are wrong. accepted: s1_id, target_id, p (+ any other columns).
    """
    return (accepted.sort(["target_id", "p", "s1_id"], descending=[False, True, False])
            .unique("target_id", keep="first", maintain_order=True))


def drop_crowded(accepted: pl.DataFrame, limit: int) -> pl.DataFrame:
    """Drop every claim on a record accepted for `limit` or more S1s (such records are generic names or
    address-less records the model accepts for everyone; even the best claim is likely wrong)."""
    claims = accepted.group_by("target_id").agg(claimants=pl.col("s1_id").n_unique())
    return accepted.join(claims.filter(pl.col("claimants") >= limit), on="target_id", how="anti")
