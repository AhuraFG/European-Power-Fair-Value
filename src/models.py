# Seasonal Naive, Ridge, RF, LightGBM
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from . import features as feat

VAL_HOURS = 720  # holdout for LGBM early stopping

class SeasonalNaive:
    def fit(self, X, y): pass
    def predict(self, X): return X["price_lag_168h"].values.astype(float)

class RidgeModel:
    def __init__(self): self.s = StandardScaler(); self.m = Ridge(alpha=1); self.cols = feat.COLS_FULL
    def fit(self, X, y): self.s.fit(X[self.cols]); self.m.fit(self.s.transform(X[self.cols]), y)
    def predict(self, X): return self.m.predict(self.s.transform(X[self.cols]))

class RFModel:
    def __init__(self): self.cols = feat.COLS_FULL; self.m = None
    def fit(self, X, y):
        self.m = RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1)
        self.m.fit(X[self.cols], y)
    def predict(self, X): return self.m.predict(X[self.cols])

def _lgb_params(n_est):
    return dict(objective="regression_l1", metric="mae", n_estimators=n_est, learning_rate=0.04,
                num_leaves=31, min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.2, reg_lambda=1.0, verbosity=-1, random_state=42)

class LGBMModel:
    def __init__(self): self.cols = feat.COLS_FULL; self.m = None
    def fit(self, X, y):
        n = len(X)
        if n >= 2 * VAL_HOURS:
            self.m = lgb.LGBMRegressor(**_lgb_params(1200))
            self.m.fit(X[self.cols].iloc[:-VAL_HOURS], y.iloc[:-VAL_HOURS],
                       eval_set=[(X[self.cols].iloc[-VAL_HOURS:], y.iloc[-VAL_HOURS:])],
                       callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            self.m = lgb.LGBMRegressor(**_lgb_params(400))
            self.m.fit(X[self.cols], y)
    def predict(self, X): return self.m.predict(X[self.cols])
