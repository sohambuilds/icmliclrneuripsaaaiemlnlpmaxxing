"""Reading the competition TSVs (with header checks and parquet caching) and writing the two output files."""
import polars as pl

from . import config as C


def read_tsv(path, expected_columns: list[str]) -> pl.DataFrame:
    """Tab-separated, no quoting, every column kept as a string (IDs must never become numbers)."""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    if df.columns != expected_columns:
        raise ValueError(f"{path}: header {df.columns} != expected {expected_columns}")
    return df


def load_records(split: str) -> pl.DataFrame:
    """All three sources of one split in one table: entity_id, business_name, business_address, country, source.

    Raw text is kept as read (empty fields stay null). Cached as parquet after the first read.
    """
    cache = C.RAW_DIR / f"{split}_records.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    frames = []
    for src in C.SOURCES:
        path = C.DATA_DIR / split / f"{split}_source{src[1]}.tsv"
        df = read_tsv(path, C.SOURCE_COLUMNS)
        bad_prefix = df.filter(~pl.col("entity_id").str.starts_with(f"{src}-")).height
        if bad_prefix:
            raise ValueError(f"{path}: {bad_prefix} IDs without the {src}- prefix")
        frames.append(df.with_columns(source=pl.lit(src)))
    out = pl.concat(frames)
    if out["entity_id"].n_unique() != out.height:
        raise ValueError(f"{split}: duplicate entity_id values")
    if out["country"].is_null().any():
        raise ValueError(f"{split}: records with an empty country")
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    out.write_parquet(cache)
    return out


def load_label_pairs() -> pl.DataFrame:
    """Training labels as one row per true link: s1_id, target_id. Empty label fields give no rows."""
    cache = C.RAW_DIR / "train_label_pairs.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    raw = read_tsv(C.DATA_DIR / "train" / "train_ground_truth.tsv", C.LABEL_COLUMNS)
    if raw["source1_entity_id"].n_unique() != raw.height:
        raise ValueError("ground truth: duplicate source1_entity_id rows")
    pairs = (
        raw.select(s1_id="source1_entity_id", target_id=pl.col("matched_entity_ids").fill_null("").str.split(","))
        .explode("target_id", empty_as_null=True)
        .with_columns(pl.col("target_id").str.strip_chars())
        .filter(pl.col("target_id").is_not_null() & (pl.col("target_id") != ""))
    )
    dup = pairs.height - pairs.unique().height
    if dup:
        raise ValueError(f"ground truth: {dup} duplicate IDs inside answer lists")
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    pairs.write_parquet(cache)
    return pairs


def label_roster() -> pl.DataFrame:
    """Every training S1 ID from the ground-truth file (including ones with no matches)."""
    raw = read_tsv(C.DATA_DIR / "train" / "train_ground_truth.tsv", C.LABEL_COLUMNS)
    return raw.select(s1_id="source1_entity_id")


def write_id_lists(links: pl.DataFrame, roster: pl.Series, path, list_column: str) -> None:
    """Write one row per S1 in `roster`: s1 id <TAB> comma-joined target ids (empty when none).

    links: (s1_id, target_id) rows; duplicates are removed, ids sorted for a stable file.
    """
    lists = (
        links.select("s1_id", "target_id").unique()
        .sort("s1_id", "target_id")
        .group_by("s1_id", maintain_order=True)
        .agg(pl.col("target_id").str.join(",").alias(list_column))
    )
    out = (
        pl.DataFrame({"source1_entity_id": roster})
        .join(lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left")
        .with_columns(pl.col(list_column).fill_null(""))
    )
    if out["source1_entity_id"].n_unique() != out.height:
        raise ValueError("roster has duplicate S1 ids")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")


def read_id_lists(path, list_column: str) -> pl.DataFrame:
    """Read a file written by write_id_lists back into (s1_id, target_id) rows."""
    df = read_tsv(path, ["source1_entity_id", list_column])
    return (
        df.select(s1_id="source1_entity_id", target_id=pl.col(list_column).fill_null("").str.split(","))
        .explode("target_id", empty_as_null=True)
        .filter(pl.col("target_id").is_not_null() & (pl.col("target_id") != ""))
    )
