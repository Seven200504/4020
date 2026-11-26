# -*- coding: utf-8 -*-
"""Part 3：组合构建与交易记账。

ver3 改动（相对 ver2）：新增 solve_gmv（显式最小方差组合，相同股票池与约束）。
"""
import numpy as np
import pandas as pd

from .config import CONFIG


def solve_portfolio(mu, Sigma, industry, gamma, w_prev, tradable, cfg=CONFIG):
    """受约束 MVO：maximize mu^T w - gamma/2 * w^T Σ w，约束预算/单股/行业上限。"""
    import cvxpy as cp  # 惰性导入：仅求解时需要；未安装则运行 `pip install cvxpy`
    N = len(mu)
    w = cp.Variable(N)
    obj = mu.values @ w - (gamma / 2.0) * cp.quad_form(w, cp.psd_wrap(Sigma))
    cons = [cp.sum(w) == 1.0, w >= 0.0, w <= cfg["stock_weight_cap"]]
    for ic in industry.unique():
        mask = (industry.values == ic)
        cons.append(cp.sum(w[mask]) <= cfg["industry_weight_cap"])
    # TODO(ver1): 用 tradable 细化不可成交持仓的保留约束
    prob = cp.Problem(cp.Maximize(obj), cons)
    prob.solve()
    w_opt = pd.Series(w.value, index=mu.index)
    return w_opt, prob.status, abs(w_opt.sum() - 1.0)


def solve_gmv(Sigma, industry, cfg=CONFIG):
    """显式最小方差组合：minimize w^T Σ w，约束同 solve_portfolio（预算/单股/行业上限）。"""
    import cvxpy as cp
    N = Sigma.shape[0]
    w = cp.Variable(N)
    obj = cp.quad_form(w, cp.psd_wrap(Sigma))
    cons = [cp.sum(w) == 1.0, w >= 0.0, w <= cfg["stock_weight_cap"]]
    for ic in industry.unique():
        mask = (industry.values == ic)
        cons.append(cp.sum(w[mask]) <= cfg["industry_weight_cap"])
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve()
    w_opt = pd.Series(w.value, index=industry.index)
    return w_opt, prob.status, abs(w_opt.sum() - 1.0)


def drift_weights(w_target_prev, R_prev):
    """w_pre = w_target*(1+R) / (1+R_port_gross)。"""
    one_r = 1.0 + R_prev
    port_gross = (w_target_prev * one_r).sum()
    return w_target_prev * one_r / port_gross


def execute_and_account(w_target, w_pre, R_m, cost=CONFIG["one_way_trade_cost"]):
    """返回单边换手、成本率与净值乘子。D_m = Σ|w_target-w_pre|，TO=D_m/2，C=cost*D_m。"""
    D_m = (w_target - w_pre).abs().sum()
    TO_m = D_m / 2.0
    C_m = cost * D_m
    nav_mult = (1.0 - C_m) * (1.0 + float(w_target @ R_m))
    return {"turnover_one_way": TO_m, "cost_rate": C_m, "nav_mult": nav_mult}
