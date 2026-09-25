"""Rewrite an existing output folder's two TSVs in test_source1.tsv row order (content unchanged).

Run from the repository root:  python -m plan1.rewrite_outputs output/plan1/p1-v2
Checks that every S1 keeps exactly the same ids, then prints size and the first lines of each file.
"""
import sys
from pathlib import Path

import polars as pl

from .dataio import read_id_lists, test_s1_roster, write_id_lists

FILES = (("matching_results.tsv", "matched_entity_ids"), ("candidate_pairs.tsv", "candidate_entity_ids"))


def main(folder: str) -> None:
    roster = test_s1_roster()
    for name, col in FILES:
        path = Path(folder) / name
        before = read_id_lists(path, col)
        write_id_lists(before, roster, path, col)
        after = read_id_lists(path, col)
        assert before.sort("s1_id", "target_id").equals(after.sort("s1_id", "target_id")), f"{name}: content changed"
        raw = path.read_bytes()
        assert b"\r" not in raw and b'"' not in raw and not raw.startswith(b"\xef\xbb\xbf"), f"{name}: bad bytes"
        lines = raw.split(b"\n")
        print(f"{path}: {len(raw) / 1e6:.1f} MB, {len(lines) - 2:,} data rows, {after.height:,} ids")
        for line in lines[:4]:
            print("   ", repr(line.decode("utf-8")[:120]))
    first = pl.read_csv(Path(folder) / FILES[0][0], separator="\t", quote_char=None, infer_schema=False)["source1_entity_id"]
    assert first.equals(roster.rename("source1_entity_id")), "row order still differs from test_source1.tsv"
    print("row order = test_source1.tsv: ok")


if __name__ == "__main__":
    main(sys.argv[1])
