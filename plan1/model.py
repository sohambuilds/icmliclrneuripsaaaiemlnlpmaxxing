"""Training the single LightGBM matcher (fixed parameters, early stopping on Tune)."""
import time

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
