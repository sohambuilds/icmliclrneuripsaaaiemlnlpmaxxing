"""The competition score: F0.5 per S1, averaged over every S1 in the evaluated population.

Per S1 with true count g, predicted count k and correct count t:
    F0.5 = 1.25 * t / (0.25 * g + k)
    empty truth and empty prediction -> 1.0; any other case with t == 0 -> 0.0
"""
import polars as pl


def per_s1(pred: pl.DataFrame, truth: pl.DataFrame, universe: pl.Series) -> pl.DataFrame:
    """pred, truth: (s1_id, target_id) rows. universe: every S1 id that counts (queries with no candidates included).

    Predictions for S1s outside the universe are ignored. Truth for S1s in the universe is always counted,
    including true targets that retrieval never found.
    """
    uni = pl.DataFrame({"s1_id": universe}).unique()
    p = pred.select("s1_id", "target_id").unique().join(uni, on="s1_id", how="semi")
    t = truth.select("s1_id", "target_id").unique().join(uni, on="s1_id", how="semi")
    correct = p.join(t, on=["s1_id", "target_id"], how="semi").group_by("s1_id").agg(t=pl.len())
    return (
        uni.join(t.group_by("s1_id").agg(g=pl.len()), on="s1_id", how="left")
        .join(p.group_by("s1_id").agg(k=pl.len()), on="s1_id", how="left")
        .join(correct, on="s1_id", how="left")
        .with_columns(pl.col("g", "k", "t").fill_null(0).cast(pl.Int64))
        .with_columns(
            f=pl.when((pl.col("g") == 0) & (pl.col("k") == 0)).then(1.0)
            .when(pl.col("t") == 0).then(0.0)
            .otherwise(1.25 * pl.col("t") / (0.25 * pl.col("g") + pl.col("k")))
        )
    )


def summarize(scores: pl.DataFrame) -> dict:
    """Headline numbers from per_s1() output."""
    n = scores.height
    g, k, t = scores["g"].sum(), scores["k"].sum(), scores["t"].sum()
    zero_truth = scores.filter(pl.col("g") == 0)
    return {
        "n_s1": n,
        "macro_f05": float(scores["f"].mean()) if n else float("nan"),
        "link_precision": t / k if k else float("nan"),
        "link_recall": t / g if g else float("nan"),
        "true_links": int(g),
        "predicted_links": int(k),
        "correct_links": int(t),
        "empty_answer_rate": float((scores["k"] == 0).mean()) if n else float("nan"),
        "singletons": zero_truth.height,
        "singleton_false_positive_rate": float((zero_truth["k"] > 0).mean()) if zero_truth.height else float("nan"),
        "mean_predicted_per_s1": k / n if n else float("nan"),
    }


def score(pred: pl.DataFrame, truth: pl.DataFrame, universe: pl.Series) -> dict:
    return summarize(per_s1(pred, truth, universe))


def self_check() -> None:
    """Edge cases from the problem statement; raises if the metric is wrong."""
    truth = pl.DataFrame({"s1_id": ["a", "a", "c"], "target_id": ["x", "z", "q"]})
    uni = pl.Series(["a", "b", "c", "d"])
    pred = pl.DataFrame({"s1_id": ["a", "a", "a", "b", "e"], "target_id": ["x", "y", "z", "w", "v"]})
    s = per_s1(pred, truth, uni).sort("s1_id")
    f = dict(zip(s["s1_id"], s["f"]))
    assert abs(f["a"] - 0.7142857) < 1e-6, f  # worked example: 2 correct of 3 predicted, 2 true
    assert f["b"] == 0.0  # prediction on an S1 with no true matches
    assert f["c"] == 0.0  # true matches, nothing predicted
    assert f["d"] == 1.0  # no truth, no prediction
    assert "e" not in f  # outside the universe: ignored
    assert abs(summarize(s)["macro_f05"] - (0.7142857 + 0 + 0 + 1) / 4) < 1e-6
    empty = pred.head(0)
    assert summarize(per_s1(empty, truth, uni))["macro_f05"] == 0.5  # only b and d are right
