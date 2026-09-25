"""Training and using the matcher: LightGBM (CPU, reference), XGBoost or CatBoost (GPU). All MIT/Apache licensed.

A model is passed around as a (kind, booster) tuple so every caller works with either backend.
"""
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from . import config as C


def train_lgb(x_fit: np.ndarray, y_fit: np.ndarray, x_tune: np.ndarray, y_tune: np.ndarray, feature_names: list[str],
              max_rounds: int = C.LGB_MAX_ROUNDS, log_every: int = 200) -> tuple[lgb.Booster, float]:
    d_fit = lgb.Dataset(x_fit, y_fit, feature_name=feature_names, free_raw_data=False)
    d_tune = lgb.Dataset(x_tune, y_tune, feature_name=feature_names, reference=d_fit, free_raw_data=False)
    t0 = time.time()
    booster = lgb.train(
        C.LGB_PARAMS, d_fit, num_boost_round=max_rounds, valid_sets=[d_tune], valid_names=["tune"],
        callbacks=[lgb.early_stopping(C.LGB_EARLY_STOPPING, verbose=True), lgb.log_evaluation(log_every)],
    )
    return booster, time.time() - t0


def train_model(x_fit: np.ndarray, y_fit: np.ndarray, x_tune: np.ndarray, y_tune: np.ndarray, feature_names: list[str],
                backend: str | None = None, max_rounds: int = C.LGB_MAX_ROUNDS, log_every: int = 200) -> tuple[tuple, float]:
    """Early stopping on Tune (log-loss) for both backends. Returns ((kind, booster), seconds)."""
    backend = backend or C.MODEL_BACKEND
    if backend == "lightgbm":
        booster, secs = train_lgb(x_fit, y_fit, x_tune, y_tune, feature_names, max_rounds, log_every)
        return ("lightgbm", booster), secs
    if backend == "xgboost":
        import xgboost as xgb

        d_fit = xgb.QuantileDMatrix(x_fit, label=y_fit, feature_names=feature_names)
        d_tune = xgb.QuantileDMatrix(x_tune, label=y_tune, feature_names=feature_names, ref=d_fit)
        t0 = time.time()
        booster = xgb.train(C.XGB_PARAMS, d_fit, num_boost_round=max_rounds, evals=[(d_tune, "tune")],
                            early_stopping_rounds=C.LGB_EARLY_STOPPING, verbose_eval=log_every)
        secs = time.time() - t0
        best = booster.best_iteration
        booster = booster[: best + 1]  # keep only the trees up to the best round
        booster.set_attr(best_iteration=str(best))
        return ("xgboost", booster), secs
    if backend == "catboost":
        from catboost import CatBoostClassifier, Pool

        model = CatBoostClassifier(iterations=max_rounds, **C.CAT_PARAMS)
        t0 = time.time()
        model.fit(Pool(x_fit, y_fit, feature_names=feature_names), eval_set=Pool(x_tune, y_tune, feature_names=feature_names),
                  early_stopping_rounds=C.LGB_EARLY_STOPPING, use_best_model=True, verbose=log_every)
        return ("catboost", model), time.time() - t0
    raise ValueError(f"unknown backend {backend!r}")


def best_iteration(model: tuple) -> int:
    kind, m = model
    if kind == "lightgbm":
        return int(m.best_iteration)
    if kind == "catboost":
        return int(m.get_best_iteration() or m.tree_count_ - 1)
    return int(m.attr("best_iteration") or m.num_boosted_rounds() - 1)


def predict(model: tuple, x: np.ndarray) -> np.ndarray:
    kind, m = model
    if kind == "lightgbm":
        n = m.best_iteration if m.best_iteration and m.best_iteration > 0 else None
        return m.predict(x, num_iteration=n)
    if kind == "catboost":
        return m.predict_proba(x)[:, 1]
    import xgboost as xgb

    return m.predict(xgb.DMatrix(x, feature_names=m.feature_names))


def importance(model: tuple, feature_names: list[str]) -> dict[str, float]:
    kind, m = model
    if kind == "lightgbm":
        return dict(zip(feature_names, m.feature_importance("gain").astype(float)))
    if kind == "catboost":
        return dict(zip(feature_names, m.get_feature_importance().astype(float)))
    gains = m.get_score(importance_type="total_gain")
    return {f: float(gains.get(f, 0.0)) for f in feature_names}


def save_model(model: tuple, stem: Path) -> Path:
    kind, m = model
    if kind == "lightgbm":
        path = stem.parent / f"{stem.name}.lgb.txt"
        m.save_model(str(path), num_iteration=m.best_iteration if m.best_iteration > 0 else None)
    elif kind == "catboost":
        path = stem.parent / f"{stem.name}.cbm"
        m.save_model(str(path))
    else:
        path = stem.parent / f"{stem.name}.xgb.json"
        m.save_model(str(path))
    return path


def load_model(path: Path) -> tuple:
    path = Path(path)
    if path.suffix == ".cbm":
        from catboost import CatBoostClassifier

        return ("catboost", CatBoostClassifier().load_model(str(path)))
    if path.name.endswith(".xgb.json"):
        import xgboost as xgb

        m = xgb.Booster()
        m.load_model(str(path))
        m.set_param({"device": "cuda"})
        return ("xgboost", m)
    return ("lightgbm", lgb.Booster(model_file=str(path)))
