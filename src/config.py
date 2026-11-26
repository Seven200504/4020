# -*- coding: utf-8 -*-
"""STA4020 ver3 共享配置。

ver3 相对 ver2：
- 字段收敛为 7 基础 + 3 增强（共 10 个模型字段），研究问题聚焦"控制基础特征后，
  应计质量/盈利稳定性/盈利趋势是否提供额外预测信息"；
- 收益模型训练目标改为横截面相对收益（见 src/data.py 中心化、src/return_model.py）；
- 回退门槛配置化（base_min_valid / enh_min_valid），避免 7+3 字段下全部触发回退；
- 组合层新增显式最小方差对照 GMV；gamma 用基础模型 B+LW 在验证期选择。

时间轴（与 ver1/ver2 一致）：
    训练窗口 120 个月，风险窗口 60 个月
    调参期 2015-01—2020-11（71 个月），冻结月 2020-12
    测试期 2021-01—2025-12（60 个月 = 5 年）
"""
CONFIG = {
    "root": "e:/4020prj",
    "data_dir": "e:/4020prj/data",
    "output_dir": "e:/4020prj/project_outputs_ver3",

    # 股票池
    "n_assets": 100,
    "industry_count_cap": 20,
    "industry_weight_cap": 0.25,
    "stock_weight_cap": 0.05,
    "liquidity_exclude_quantile": 0.20,
    "min_return_history_months": 60,
    "min_return_history_pre2007_months": 24,   # 2007 年前股票池放宽规则

    # 时间对齐
    "return_window_months": 120,               # 收益模型训练窗口
    "risk_window_months": 60,                  # 风险模型窗口（不与训练绑定）
    "signal_lag_vs_target": 2,                 # X[i,m] = D[i,m-2]

    # 时间划分
    "validation_start": "2015-01",
    "validation_end": "2020-11",
    "freeze_month": "2020-12",
    "test_start": "2021-01",
    "test_end": "2025-12",

    # 超参数（有限网格，不随测试表现扩大）
    "ridge_grid": [0.1, 1, 10, 100, 1000],
    "gamma_grid": [1, 3, 10, 30],

    # 成本与无风险
    "one_way_trade_cost": 0.001,               # 单边 10bps
    "use_actual_risk_free": True,
    "risk_free_csv": "e:/4020prj/data/risk_free.csv",
    "factor_benchmarks": ["e:/4020prj/data/FF5.csv", "e:/4020prj/data/HXZ.csv"],

    # 行业（静态一级代码，待核实金融代码）
    "industry_static": True,
    "financial_industry_codes": [40, 41],

    # 统计
    "bootstrap_n": 2000,
    "bootstrap_block_months": 6,
    "seed": 4020,

    # 字段名单（7 基础 + 3 增强）
    "BASE_FIELDS": [
        "total_mv", "book_value_to_total_mktcap_mrq",
        "gross_profit_to_asset_ttm", "asset_growth_mrq",
        "momentum_academic_252_21", "illiq_252", "total_vol_21",
    ],
    "ENHANCED_FIELDS": [
        "accrual_to_asset_ttm", "net_profit_to_asset_q_var",
        "gross_profit_to_asset_q_trend",
    ],
    # 股票池流动性代理（不属于模型字段）
    "LIQUIDITY_FIELD": "amount_21",

    # 回退门槛（7 基础至少 6 有效、3 增强至少 2 有效才不用历史回退）
    "base_min_valid": 6,
    "enh_min_valid": 2,
}

# 数据转换只需要的字段：模型 10 字段 + 股票池流动性代理 1 字段 = 11 字段
CONFIG["DATA_FIELDS"] = CONFIG["BASE_FIELDS"] + CONFIG["ENHANCED_FIELDS"] + [CONFIG["LIQUIDITY_FIELD"]]
