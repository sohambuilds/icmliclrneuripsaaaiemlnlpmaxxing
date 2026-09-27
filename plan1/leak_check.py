"""Does the test data itself reveal which records belong together? Uses only the provided test files.

Takes confident s2ce links (p >= 0.99, one owner per record) and checks whether their S1 and record sit unusually
close to each other in the files, or have related ID numbers, compared with the same links after shuffling:
  - relative row position of the record in its file vs the S1's position in test_source1
  - ID numbers: correlation, same last 3 digits, difference below 1M
  - records of the same S1 in one source: how many rows apart they are
On train there was no such signal. If the test shows one, it can place address-less and look-alike records.

  python -m plan1.leak_check        (CPU, a few minutes)
"""
import numpy as np
import polars as pl

from . import config as C
from .dataio import load_records
from .decide import one_owner


def stats(d: pl.DataFrame, name: str) -> dict:
    dr = (d["q_rel"] - d["t_rel"]).abs()
    return {
        "pairs": name, "n": d.height,
        "corr_position": float(np.corrcoef(d["q_rel"].to_numpy(), d["t_rel"].to_numpy())[0, 1]),
        "median_abs_pos_diff": float(dr.median()),
        "share_pos_diff_lt_0.001": float((dr < 0.001).mean()),
        "corr_id": float(np.corrcoef(d["q_num"].to_numpy().astype(float), d["t_num"].to_numpy().astype(float))[0, 1]),
        "share_same_last3": float(((d["q_num"] % 1000) == (d["t_num"] % 1000)).mean()),
        "share_id_diff_lt_1M": float(((d["q_num"] - d["t_num"]).abs() < 1_000_000).mean()),
    }


def main() -> None:
    pl.Config.set_tbl_width_chars(220)
    rec = load_records("test").with_columns(
        pos=pl.int_range(pl.len()).over("source"), n=pl.len().over("source"),
        num=pl.col("entity_id").str.extract(r"-(\d+)$").cast(pl.Int64),
    ).with_columns(rel=pl.col("pos") / pl.col("n"))
    sc = pl.read_parquet(str(C.WORK_DIR / "stage2_ce" / "test_scores" / "*.parquet"), columns=["s1_id", "target_id", "p"])
    links = one_owner(sc.filter(pl.col("p") >= 0.99))
    q = rec.filter(pl.col("source") == "S1").select(s1_id="entity_id", country="country", q_rel="rel", q_num="num")
    t = rec.filter(pl.col("source") != "S1").select(target_id="entity_id", source="source", t_rel="rel", t_pos="pos", t_num="num")
    d = links.join(q, on="s1_id").join(t, on="target_id")
    rng = np.random.default_rng(0)
    shuf = d.with_columns(t_rel=pl.Series(rng.permutation(d["t_rel"].to_numpy())), t_num=pl.Series(rng.permutation(d["t_num"].to_numpy())))
    rows = [stats(d, "confident links"), stats(shuf, "same links shuffled")]
    for c in ("US", "India", "France"):
        rows.append(stats(d.filter(pl.col("country") == c), f"confident links {c}"))
    print("S1 vs its records:")
    print(pl.DataFrame(rows))

    # records of one S1 inside one source: rows between consecutive records vs a shuffled placement
    def gaps(x: pl.DataFrame) -> pl.Series:
        return (x.sort("s1_id", "source", "t_pos").with_columns(gap=pl.col("t_pos").diff().over("s1_id", "source"))
                .filter(pl.col("gap").is_not_null())["gap"])
    g = gaps(d)
    gs = gaps(d.with_columns(pl.col("t_pos").shuffle(seed=1).over("source")))
    print("\nrecords of the same S1 in one source, rows apart:")
    print(pl.DataFrame({"pairs": ["same S1", "shuffled"], "n": [g.len(), gs.len()],
                        "median_gap": [g.median(), gs.median()], "share_gap_le_10": [(g <= 10).mean(), (gs <= 10).mean()],
                        "share_gap_le_1000": [(g <= 1000).mean(), (gs <= 1000).mean()]}))


if __name__ == "__main__":
    main()
