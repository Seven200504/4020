# -*- coding: utf-8 -*-
"""Part 4：绩效指标、bootstrap 与因子对照。"""
import numpy as np
import pandas as pd

from .config import CONFIG


def performance_metrics(r_net, rf=None):
    """返回年化收益/波动、Sharpe、最大回撤。rf 缺省 0，实际用 monthly_risk_free。"""
    r = r_net if rf is None else (r_net - rf)
    n = len(r)
    ann_ret = float((1 + r_net).prod() ** (12.0 / n) - 1.0)
    ann_vol = float(r_net.std(ddof=1) * np.sqrt(12))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(12))
    nav = (1 + r_net).cumprod()
    running_peak = nav.cummax().clip(lower=1.0)
    max_dd = float((1 - nav / running_peak).max())
    return {"ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd}


def circular_block_bootstrap(a, b, n=CONFIG["bootstrap_n"],
                             block=CONFIG["bootstrap_block_months"],
                             seed=CONFIG["seed"], rf=None):
    """配对循环块 bootstrap。收益差用净收益，Sharpe 差用超额收益（a-rf, b-rf）。"""
    rng = np.random.default_rng(seed)
    T = len(a)
    a, b = np.asarray(a), np.asarray(b)
    rf_arr = np.zeros(T) if rf is None else np.asarray(rf)
    mean_d, sharpe_d = [], []
    for _ in range(n):
        starts = rng.integers(0, T, size=int(np.ceil(T / block)))
        idx = np.concatenate([np.arange(s, s + block) % T for s in starts])[:T]
        da, db = a[idx], b[idx]
        rf_i = rf_arr[idx]
        mean_d.append(da.mean() - db.mean())
        ea, eb = da - rf_i, db - rf_i
        sa = ea.mean() / ea.std(ddof=1) * np.sqrt(12)
        sb = eb.mean() / eb.std(ddof=1) * np.sqrt(12)
        sharpe_d.append(sa - sb)
    ci = lambda x: (np.percentile(x, 2.5), np.percentile(x, 97.5))
    return ci(mean_d), ci(sharpe_d)


def factor_regression(r_excess, factors):
    """OLS 因子回归（FF5 / HXZ 对照），返回 (alpha 年化, beta, t, R2)。"""
    X = np.column_stack([np.ones(len(r_excess)), factors.values])
    y = r_excess.values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = (resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid @ resid) / ss_tot
    return {"alpha_annual": float(beta[0] * 12), "beta": beta[1:], "t": t, "r2": float(r2)}
