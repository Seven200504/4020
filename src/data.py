# -*- coding: utf-8 -*-
"""Part 0：数据加载、单月面板预处理与股票池。

ver3 改动（相对 ver2）：
- 只转换 DATA_FIELDS（11 字段 = 10 模型 + amount_21）；
- 回退门槛配置化（base_min_valid / enh_min_valid），针对 7+3 字段；
- 训练标签中心化：assemble_train_panel 把 y 减当月池等权收益（横截面相对收益）。
"""
import os
import pickle

import numpy as np
import pandas as pd

from .config import CONFIG

DATA = CONFIG["data_dir"]
OUT = CONFIG["output_dir"]
ALL_FIELDS = CONFIG["BASE_FIELDS"] + CONFIG["ENHANCED_FIELDS"]   # 10 个模型字段


def load_descriptors():
    """读取 descriptors.pkl -> OrderedDict[str, Series]，Series 索引为 (date, asset)。"""
    with open(os.path.join(DATA, "descriptors.pkl"), "rb") as f:
        return pickle.load(f)


def load_prices():
    """读取 daily_adj_close_post_20260630.pkl -> DataFrame(date × asset)。"""
    return pd.read_pickle(os.path.join(DATA, "daily_adj_close_post_20260630.pkl"))


def monthly_total_return(prices):
    """R[i,m] = P[i,m]/P[i,m-1] - 1；P 取每月最后一条有价收盘。

    停牌整月前向沿用（0 收益）、复牌月反映跳变；退市末月按最后成交价结算、之后记 NaN。
    注：退市用反向 cummax 推断（全样本代理），有真实退市/停牌数据时应改为事件驱动。
    """
    last = prices.resample("ME").last()
    valid = last.notna()
    filled = last.ffill()
    after_last = valid.iloc[::-1].cummax().iloc[::-1]
    P = filled.where(after_last, np.nan)
    R = P / P.shift(1) - 1.0
    R.index = R.index.to_period("M")
    return R


def load_industry():
    """读取 ind_code.pkl -> Series(asset -> first_industry_code)，静态一级行业。"""
    return pd.read_pickle(os.path.join(DATA, "ind_code.pkl"))


def monthly_risk_free(rf_csv=CONFIG["risk_free_csv"]):
    """r_f[m] = prod_{d in m}(1 + RF_d) - 1，日度复利成月度。

    注：risk_free.csv 目前只覆盖到 2024-09，早于测试期终点 2025-12。
    将 rf 前向填充（ffill）到 test_end——无风险利率短期近似不变，避免超额
    收益/Sharpe 在最后 15 个月出现 NaN（numpy 不跳过 NaN、pandas 会静默跳过，
    导致主表与 bootstrap 口径不一致）。
    """
    rf = pd.read_csv(rf_csv, parse_dates=["date"]).set_index("date")["RF"]
    monthly = (1 + rf).resample("ME").prod() - 1.0
    monthly.index = monthly.index.to_period("M")
    full_idx = pd.period_range(monthly.index.min(), CONFIG["test_end"], freq="M")
    monthly = monthly.reindex(full_idx).ffill()
    return monthly


def to_monthly_snapshot(series):
    """把 (date, asset) Series 转成 (Period[M], asset)，每月取最后一条已可获得值。"""
    s = series.copy().sort_index(level=0)
    month = s.index.get_level_values(0).to_period("M")
    s.index = pd.MultiIndex.from_arrays(
        [month, s.index.get_level_values(1)], names=["month", "asset"])
    return s.groupby(level=["month", "asset"]).last()


def _cross_section(D_field, month, candidates):
    """取某字段在 month 的横截面并按候选资产对齐（缺失记 NaN）。"""
    try:
        s = D_field.loc[month]
    except KeyError:
        s = pd.Series(dtype="float64")
    return s.reindex(candidates)


def preprocess_cross_section(X, base_fields=CONFIG["BASE_FIELDS"],
                             enh_fields=CONFIG["ENHANCED_FIELDS"],
                             cfg=CONFIG):
    """单个信号截面预处理：无穷->缺失、total_mv取log、1%/99%截尾（向量化）、
    中位数填缺失、有限均值/标准差标准化。返回 (Xz, fallback)。

    fallback 判定（ver3 配置化）：基础字段有效数 >= base_min_valid 且
    增强字段有效数 >= enh_min_valid，否则回退（用零相对收益/历史均值）。
    """
    X = X.replace([np.inf, -np.inf], np.nan)
    if "total_mv" in X.columns:
        X.loc[X["total_mv"] <= 0, "total_mv"] = np.nan
        X["total_mv"] = np.log(X["total_mv"])
    # 向量化 1%/99% 截尾（按列对齐上下限）
    limits = X.quantile([0.01, 0.99])
    X = X.clip(lower=limits.loc[0.01], upper=limits.loc[0.99], axis="columns")
    mu = X.mean()
    sd = X.std(ddof=0)
    med = X.median()
    Xz = (X.fillna(med) - mu) / sd.replace(0, np.nan)
    Xz = Xz.fillna(0.0)
    base_ok = X[base_fields].notna().sum(axis=1) >= cfg["base_min_valid"]
    enh_ok = X[enh_fields].notna().sum(axis=1) >= cfg["enh_min_valid"]
    fallback = ~(base_ok & enh_ok)
    return Xz, fallback


def select_universe(year, cutoff, R, industry, amount_21, total_mv, cfg=CONFIG):
    """年度股票池：计划 §4.1 规则 1-8。R/amount_21/total_mv 为 month×asset DataFrame。"""
    fin = cfg["financial_industry_codes"]
    hist_months = cfg["min_return_history_months"]
    if year < 2007:  # 2007 年前持有年份放宽（价格数据自 2001-01）
        hist_months = cfg["min_return_history_pre2007_months"]
    hist = R[R.index <= cutoff].tail(hist_months)
    eligible = hist.notna().all(axis=0)
    eligible = eligible[eligible].index
    non_fin = industry.index[~industry.isin(fin)]
    a = amount_21[amount_21.index <= cutoff].tail(12)
    liq_ok = a.notna().all(axis=0)
    liq_mean = a.mean()
    lo = liq_mean.quantile(cfg["liquidity_exclude_quantile"])
    liq_keep = liq_mean[liq_ok & (liq_mean > lo)].index
    mv = total_mv.loc[cutoff].dropna()
    mv = mv[mv > 0]
    elig = set(eligible) & set(non_fin) & set(liq_keep) & set(mv.index)
    mv = mv[mv.index.isin(elig)].sort_values(ascending=False, kind="mergesort")
    cnt, picked = {}, []
    for asset in mv.index:
        ic = industry.get(asset)
        if cnt.get(ic, 0) >= cfg["industry_count_cap"]:
            continue
        picked.append(asset)
        cnt[ic] = cnt.get(ic, 0) + 1
        if len(picked) >= cfg["n_assets"]:
            break
    return pd.DataFrame({"industry": industry.reindex(picked)}, index=picked)


def build_data(force=False):
    """只转换 DATA_FIELDS（11 字段），构建 R/universe/industry/rf，缓存到磁盘。

    返回 dict：R（month×asset）、D（10 模型字段的月快照）、universe、industry、rf。
    """
    cache_file = os.path.join(OUT, "cache", "bundle.pkl")
    if not force and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    desc = load_descriptors()
    prices = load_prices()
    industry = load_industry()
    rf = monthly_risk_free()
    R = monthly_total_return(prices)
    # 只转换需要的 11 字段
    D = {f: to_monthly_snapshot(desc[f]) for f in CONFIG["DATA_FIELDS"]}
    total_mv = D["total_mv"].unstack()
    amount_21 = D[CONFIG["LIQUIDITY_FIELD"]].unstack()
    frames = []
    for Y in range(2004, 2026):
        cutoff = pd.Period(f"{Y - 1}-11", freq="M")
        uni = select_universe(Y, cutoff, R, industry, amount_21, total_mv)
        uni["year"] = Y
        frames.append(uni)
    universe = pd.concat(frames, keys=range(2004, 2026), names=["year"])
    # 缓存只保留模型字段（10 个）；amount_21 只用于股票池构建
    D_model = {f: D[f] for f in ALL_FIELDS}
    slim = {"R": R, "D": D_model, "universe": universe, "industry": industry, "rf": rf}
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(slim, f)
    return slim


def build_monthly_panel(q, D, R, universe, cfg=CONFIG):
    """单个历史月的标准化特征快照 + 标签 + 回退标记（供训练与预测共用）。

    列：10 字段（标准化）+ 'fallback' + 'y'（标签 R[q]，未中心化；中心化在 assemble_train_panel）。
    """
    lag = cfg["signal_lag_vs_target"]
    sig = q - lag
    cand = universe.xs(q.year, level="year").index
    X_raw = pd.DataFrame({f: _cross_section(D[f], sig, cand) for f in ALL_FIELDS},
                         index=cand)
    Xz, fallback = preprocess_cross_section(X_raw, cfg["BASE_FIELDS"], cfg["ENHANCED_FIELDS"], cfg)
    panel = Xz.copy()
    panel["fallback"] = fallback
    panel["y"] = R.loc[q].reindex(cand)
    return panel


def precompute_panels(D, R, universe, months, cache_dir, force=False):
    """预计算所有历史月的单月面板，磁盘缓存（每个历史月只预处理一次）。"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "panels.pkl")
    if not force and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    panels = {q: build_monthly_panel(q, D, R, universe) for q in months}
    with open(cache_file, "wb") as f:
        pickle.dump(panels, f)
    return panels


def assemble_train_panel(train_months, panels, cfg=CONFIG):
    """由预计算单月面板拼出训练面板（只保留有效行，权重 1/N_q）。

    ver3：y 中心化为横截面相对收益 = 股票收益 − 当月池等权收益（等权 w=1/N_q，
    故 y.mean() 即池等权收益）。仅用已实现历史收益，预测阶段不再减未来等权收益。
    """
    frames = []
    for q in train_months:
        p = panels[q]
        keep = ~p["fallback"]
        if not keep.any():
            continue
        N_q = int(keep.sum())
        sub = p.loc[keep, ALL_FIELDS].copy()
        sub["y"] = p.loc[keep, "y"]
        sub["y"] = sub["y"] - sub["y"].mean()   # 横截面相对收益中心化
        sub["w"] = 1.0 / N_q
        sub["target_month"] = q
        sub["asset"] = sub.index
        frames.append(sub.reset_index(drop=True))
    if not frames:
        return pd.DataFrame(columns=ALL_FIELDS + ["y", "w", "target_month", "asset"])
    return pd.concat(frames, ignore_index=True)


def score_from_panel(panel, cand):
    """从预计算面板取目标截面特征 + 回退标记（预测用，不含标签）。"""
    Xz = panel[ALL_FIELDS].reindex(cand)
    Xz["fallback"] = panel["fallback"].reindex(cand)
    return Xz
