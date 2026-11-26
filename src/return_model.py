# -*- coding: utf-8 -*-
"""Part 1：收益建模（H / B / BQ 的拟合与预测）。

ver3 改动（相对 ver2）：
- 训练目标为横截面相对收益（股票收益 − 当月池等权收益），由 assemble_train_panel 完成中心化；
- 回退股票预测填 fallback_mu（默认 0 = 零相对收益），不再用历史均值；
- alpha 按验证期 MSE 选择（逻辑在 05_pipeline 的 tune_alpha）。
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


def historical_mean_prediction(R_window):
    """H 模型：训练窗口内每股票月收益均值（绝对收益，作基准）。R_window: (month × asset)。"""
    return R_window.mean(axis=0)


def fit_ridge(train_X, train_y, sample_weight, alpha):
    """合并岭回归。损失 min ||y-Xβ||^2 + α||β||^2（fit_intercept=True）。"""
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(train_X, train_y, sample_weight=sample_weight)
    return model


def fit_predict_return(train_panel, X_score, alpha, fields, fallback_mu=0.0):
    """训练岭回归并预测目标截面相对收益。回退股票填 fallback_mu（默认 0 = 零相对收益）。"""
    X = train_panel[fields].values
    y = train_panel["y"].values
    w = train_panel["w"].values
    model = fit_ridge(X, y, w, alpha)
    mu = model.predict(X_score[fields].values)
    mu = pd.Series(mu, index=X_score.index)
    mu = mu.where(~X_score["fallback"], fallback_mu)
    return mu, model


def cross_sectional_spearman(mu, R_realized):
    """单月横截面 Spearman IC（预测 mu 与已实现收益 R_realized）。"""
    both = pd.concat([mu.rename("mu"), R_realized.rename("r")], axis=1).dropna()
    if len(both) < 3:
        return np.nan
    return spearmanr(both["mu"], both["r"]).correlation


def relative_mse(mu, R_realized):
    """预测相对收益 vs 真实相对收益的 MSE。

    真实相对收益 = R_realized − R_realized 的池等权均值（当月股票池等权收益）。
    """
    r_rel = R_realized - R_realized.mean()
    both = pd.concat([mu.rename("mu"), r_rel.rename("r_rel")], axis=1).dropna()
    if len(both) < 3:
        return np.nan
    return float((both["mu"] - both["r_rel"]).pow(2).mean())
