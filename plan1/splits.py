"""The shared split manifest: S1 roles (Fit/Tune/Calibrate/Audit) and the fixed samples drawn from them.

Roles, the Calibrate subdivision and sampling each use their own salted hash, so no ordering is reused.
Samples are proportional by country and true-match-count band (0..5, 6+), zero matches included.
"""
import hashlib
import json

import numpy as np
import polars as pl

from . import config as C


def stable_unit(ids: list[str], salt: str) -> np.ndarray:
    """Deterministic value in [0, 1) per id (blake2b of salt|id); independent of library versions."""
    vals = np.fromiter(
        (int.from_bytes(hashlib.blake2b(f"{salt}|{i}".encode(), digest_size=8).digest(), "big") for i in ids),
        dtype=np.uint64,
        count=len(ids),
    )
    return vals / float(2**64)


def stratified_take(pool: pl.DataFrame, n: int) -> pl.DataFrame:
    """Proportional sample of n rows by (country, band); within a stratum take the smallest sample hash."""
    n = min(n, pool.height)
    strata = pool.group_by("country", "band").agg(size=pl.len()).sort("country", "band")
    exact = strata["size"].to_numpy().astype(np.float64) * n / pool.height  # float: uint32 * n overflows
    quota = np.floor(exact).astype(np.int64)
    # largest remainder so the quotas add up to exactly n
    for i in np.argsort(-(exact - quota), kind="stable")[: n - quota.sum()]:
        quota[i] += 1
    strata = strata.with_columns(quota=pl.Series(quota))
    return (
        pool.join(strata.select("country", "band", "quota"), on=["country", "band"])
        .sort("sample_u")
        .with_columns(rank=pl.int_range(pl.len()).over("country", "band"))
        .filter(pl.col("rank") < pl.col("quota"))
        .drop("rank", "quota")
    )


def build_manifest(s1: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """s1: training S1 records (entity_id, country). pairs: label pairs (s1_id, target_id)."""
    base = (
        s1.select(s1_id="entity_id", country="country")
        .join(pairs.group_by("s1_id").agg(n_true=pl.len()), on="s1_id", how="left")
        .with_columns(pl.col("n_true").fill_null(0).cast(pl.Int32))
        .with_columns(band=pl.col("n_true").clip(0, C.MAX_BAND))
        .sort("s1_id")
    )
    ids = base["s1_id"].to_list()
    role_u = stable_unit(ids, C.ROLE_SALT)
    calib_u = stable_unit(ids, C.CALIB_SALT)
    sample_u = stable_unit(ids, C.SAMPLE_SALT)

    role = np.full(len(ids), "", dtype=object)
    lower = 0.0
    for name, upper in C.ROLE_BOUNDS:
        role[(role_u >= lower) & (role_u < upper)] = name
        lower = upper
    pool = role.copy()
    is_cal = role == "calibrate"
    pool[is_cal & (calib_u < 0.5)] = "c_prob"
    pool[is_cal & (calib_u >= 0.5)] = "c_select"

    base = base.with_columns(
        role=pl.Series(role.tolist(), dtype=pl.String),
        pool=pl.Series(pool.tolist(), dtype=pl.String),
        sample_u=pl.Series(sample_u),
    )
    parts = []
    for sample, (pool_name, size) in C.SAMPLE_SIZES.items():
        taken = stratified_take(base.filter(pl.col("pool") == pool_name), size)
        parts.append(taken.select("s1_id", sample=pl.lit(sample)))
    membership = pl.concat(parts)
    if membership["s1_id"].n_unique() != membership.height:
        raise RuntimeError("an S1 landed in two samples")
    return base.join(membership, on="s1_id", how="left").drop("sample_u")


def load_or_build_manifest(s1: pl.DataFrame, pairs: pl.DataFrame, force: bool = False) -> pl.DataFrame:
    """Saved once and reused; pass force=True only to deliberately rebuild it."""
    path = C.SPLIT_DIR / "manifest.parquet"
    if path.exists() and not force:
        return pl.read_parquet(path)
    manifest = build_manifest(s1, pairs)
    C.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.write_parquet(path)
    for sample in C.SAMPLE_SIZES:
        manifest.filter(pl.col("sample") == sample).select("s1_id").write_csv(C.SPLIT_DIR / f"{sample}.tsv", separator="\t")
    with open(C.SPLIT_DIR / "manifest_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {"role_salt": C.ROLE_SALT, "calib_salt": C.CALIB_SALT, "sample_salt": C.SAMPLE_SALT,
             "role_bounds": C.ROLE_BOUNDS, "sample_sizes": C.SAMPLE_SIZES, "max_band": C.MAX_BAND,
             "n_s1": manifest.height, "polars": pl.__version__},
            f, indent=2,
        )
    return manifest


def sample_ids(manifest: pl.DataFrame, sample: str) -> pl.Series:
    return manifest.filter(pl.col("sample") == sample)["s1_id"]
