"""Step 5: full test inference with the frozen configuration, in checkpointed batches; writes both output files.

Run from the repository root:  python -m plan1.step5_infer
Restartable: finished batches are saved under work/plan1/<run>/test_chunks/ and skipped on rerun.
Outputs: output/plan1/<run>/candidate_pairs.tsv and matching_results.tsv (every test S1 has a row).
matching_results.tsv applies one owner per record; the version without it is kept in the work dir.

Test indexes are fitted on the same-country TEST targets with the same settings (unlabelled text, no labels).
Features are streamed per batch; each chunk keeps (s1_id, target_id, source, retrieval score, rank, model p),
so a later model can reuse the test candidates without repeating retrieval.
"""
import json
import re
import time

import lightgbm as lgb
import polars as pl

from . import config as C
from .dataio import read_id_lists, write_id_lists
from .decide import one_owner
from .features import FEATURES, build_features, matrix, text_lookup
from .normalize import NORMALIZE_VERSION
from .retrieve import QUERY_BATCH, TargetIndex, add_rank
from .step2_retrieve import normalized
from .step4_threshold import sha256

CHUNK_DIR = C.WORK_DIR / "test_chunks"
CHUNK_SCHEMA = {"s1_id": pl.String, "target_id": pl.String, "source": pl.String, "score": pl.Float32,
                "rank": pl.Int32, "p": pl.Float32}


def chunk_path(country: str, batch_no: int):
    return CHUNK_DIR / f"{re.sub(r'[^A-Za-z0-9]+', '_', country)}_{batch_no:05d}.parquet"


def score_batch(qb: pl.DataFrame, indexes: dict, norm: pl.DataFrame, booster: lgb.Booster) -> pl.DataFrame:
    parts = [idx.query(qb).with_columns(source=pl.lit(src)) for src, idx in indexes.items()]
    cands = pl.concat(parts) if parts else pl.DataFrame(schema={"s1_id": pl.String, "target_id": pl.String, "score": pl.Float32, "source": pl.String})
    if cands.height == 0:
        return pl.DataFrame(schema=CHUNK_SCHEMA)
    cands = add_rank(cands)
    q, t = text_lookup(norm, cands["s1_id"], cands["target_id"])
    feats = build_features(cands, q, t)
    return feats.select(
        "s1_id", "target_id", "source",
        score=pl.col("retrieval_score").cast(pl.Float32), rank=pl.col("retrieval_rank").cast(pl.Int32),
    ).with_columns(p=pl.Series(booster.predict(matrix(feats)), dtype=pl.Float32))


def main() -> None:
    frozen = json.loads((C.WORK_DIR / "frozen_config.json").read_text(encoding="utf-8"))
    if frozen["normalize_version"] != NORMALIZE_VERSION:
        raise RuntimeError("frozen configuration used a different normalization version")
    if sha256(C.WORK_DIR / "model.txt") != frozen["model_sha256"]:
        raise RuntimeError("model.txt differs from the frozen model")
    booster = lgb.Booster(model_file=str(C.WORK_DIR / "model.txt"))
    thr = frozen["threshold"]
    print(f"frozen threshold {thr}, model {frozen['model_sha256'][:12]}")

    norm = normalized("test")
    queries = norm.filter(pl.col("source") == "S1").sort("entity_id")
    roster = queries["entity_id"]
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    # chunks hold model scores: never mix chunks from a different model or normalization
    chunk_meta = CHUNK_DIR / "chunk_meta.json"
    this_run = {"model_sha256": frozen["model_sha256"], "normalize_version": NORMALIZE_VERSION, "batch": QUERY_BATCH,
                "features": FEATURES}
    if chunk_meta.exists():
        if json.loads(chunk_meta.read_text(encoding="utf-8")) != this_run:
            raise RuntimeError(f"{CHUNK_DIR} holds chunks from another configuration; move it away first")
    else:
        chunk_meta.write_text(json.dumps(this_run), encoding="utf-8")
    t_all = time.time()

    for country in sorted(queries["country"].unique().to_list()):
        q = queries.filter(pl.col("country") == country)
        starts = list(range(0, q.height, QUERY_BATCH))
        todo = [s for s in starts if not chunk_path(country, s // QUERY_BATCH).exists()]
        print(f"\n{country}: {q.height:,} S1 in {len(starts)} batches, {len(todo)} to do")
        if not todo:
            continue
        t0 = time.time()
        indexes = {}
        for src in C.TARGET_SOURCES:
            t = norm.filter((pl.col("source") == src) & (pl.col("country") == country))
            if t.height:
                indexes[src] = TargetIndex(t)
                print(f"  index {src}: {t.height:,} targets, vocab {indexes[src].vocabulary_size:,}")
        print(f"  indexes built in {time.time() - t0:.0f}s")
        for i, start in enumerate(todo, 1):
            t1 = time.time()
            out = score_batch(q.slice(start, QUERY_BATCH), indexes, norm, booster)
            path = chunk_path(country, start // QUERY_BATCH)
            tmp = path.with_suffix(".tmp")
            out.write_parquet(tmp)
            tmp.replace(path)  # a chunk exists only once it is complete
            print(f"  batch {i}/{len(todo)}: {out.height:,} candidates, {int((out['p'] >= thr).sum()):,} accepted, "
                  f"{time.time() - t1:.0f}s (elapsed {(time.time() - t_all) / 60:.1f} min)")

    # ---- assemble: both files from the same in-memory table ----
    scored = pl.read_parquet(str(CHUNK_DIR / "*.parquet"))
    expected = sum(len(range(0, n, QUERY_BATCH)) for n in queries.group_by("country").len()["len"].to_list())
    n_chunks = len(list(CHUNK_DIR.glob("*.parquet")))
    if n_chunks != expected:
        raise RuntimeError(f"{n_chunks} chunk files, expected {expected}")
    if scored.height != scored.select("s1_id", "target_id").unique().height:
        raise RuntimeError("duplicate (s1, target) candidates")
    if scored.join(pl.DataFrame({"s1_id": roster}), on="s1_id", how="anti").height:
        raise RuntimeError("candidates for S1 ids outside the test roster")
    accepted = scored.filter(pl.col("p") >= thr)
    matches = one_owner(accepted)
    collisions = accepted.group_by("target_id").agg(n=pl.col("s1_id").n_unique()).filter(pl.col("n") > 1).height
    print(f"\naccepted {accepted.height:,} links; {collisions:,} records accepted for 2+ S1; "
          f"one owner per record keeps {matches.height:,} (drops {accepted.height - matches.height:,})")

    cand_path = C.OUT_DIR / "candidate_pairs.tsv"
    match_path = C.OUT_DIR / "matching_results.tsv"
    write_id_lists(scored, roster, cand_path, "candidate_entity_ids")
    write_id_lists(matches, roster, match_path, "matched_entity_ids")
    write_id_lists(accepted, roster, C.WORK_DIR / "matching_results_no_owner.tsv", "matched_entity_ids")

    # ---- internal checks on the written files (the official validator only warns on some of these) ----
    cand_back = read_id_lists(cand_path, "candidate_entity_ids")
    match_back = read_id_lists(match_path, "matched_entity_ids")
    for name, path, col in (("candidates", cand_path, "candidate_entity_ids"), ("matches", match_path, "matched_entity_ids")):
        rows = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
        assert rows.height == roster.len() and rows["source1_entity_id"].n_unique() == roster.len(), f"{name}: row count"
        assert set(rows["source1_entity_id"].to_list()) == set(roster.to_list()), f"{name}: roster mismatch"
    assert match_back.join(cand_back, on=["s1_id", "target_id"], how="anti").height == 0, "a match is not in candidates"
    assert cand_back.height == cand_back.unique().height, "duplicate ids inside a list"
    countries = norm.select("entity_id", "country", "source")
    chk = (cand_back.join(countries.rename({"entity_id": "s1_id", "country": "q_country", "source": "q_source"}), on="s1_id", how="left")
           .join(countries.rename({"entity_id": "target_id", "country": "t_country", "source": "t_source"}), on="target_id", how="left"))
    assert chk["t_country"].is_null().sum() == 0, "candidate ids missing from test S2/S3"
    assert chk.filter(~pl.col("t_source").is_in(list(C.TARGET_SOURCES))).height == 0, "non S2/S3 id in a list"
    assert chk.filter(pl.col("q_country") != pl.col("t_country")).height == 0, "cross-country candidate"
    print("\ninternal checks: ok (one row per S1, matches within candidates, no duplicates, same-country S2/S3 ids)")

    # ---- diagnostics ----
    per = (pl.DataFrame({"s1_id": roster}).join(queries.select(s1_id="entity_id", country="country"), on="s1_id")
           .join(scored.group_by("s1_id").agg(n_cand=pl.len()), on="s1_id", how="left")
           .join(accepted.group_by("s1_id").agg(n_accepted=pl.len()), on="s1_id", how="left")
           .join(matches.group_by("s1_id").agg(n_match=pl.len()), on="s1_id", how="left")
           .with_columns(pl.col("n_cand", "n_accepted", "n_match").fill_null(0)))
    diag = per.group_by("country").agg(
        s1=pl.len(), mean_candidates=pl.col("n_cand").mean(), no_candidates=(pl.col("n_cand") == 0).sum(),
        mean_accepted=pl.col("n_accepted").mean(), mean_matches=pl.col("n_match").mean(),
        empty_before_owner=(pl.col("n_accepted") == 0).mean(), empty_answer_rate=(pl.col("n_match") == 0).mean(),
    ).sort("country")
    print(diag)
    print(f"final links {matches.height:,} | total time {(time.time() - t_all) / 60:.1f} min")
    (C.WORK_DIR / "test_inference_meta.json").write_text(json.dumps({
        "threshold": thr, "model_sha256": frozen["model_sha256"], "candidates": scored.height,
        "accepted": accepted.height, "matches_after_one_owner": matches.height,
        "targets_accepted_by_multiple_s1": collisions, "diagnostics": diag.to_dicts(),
    }, indent=2, default=float), encoding="utf-8")
    print("\nwritten:", cand_path, match_path, sep="\n  ")


if __name__ == "__main__":
    main()
