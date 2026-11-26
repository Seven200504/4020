# -*- coding: utf-8 -*-
"""Part 2：风险建模（样本协方差 S / Ledoit-Wolf）。"""
import numpy as np
from sklearn.covariance import LedoitWolf


def sample_covariance(R_window):
    """去均值样本协方差，MLE 分母 T。R_window: (T, N)。"""
    Rc = R_window - R_window.mean(axis=0)
    T = R_window.shape[0]
    return (Rc.T @ Rc) / T


def ledoit_wolf_covariance(R_window):
    """LedoitWolf(assume_centered=False)；返回 (covariance, shrinkage)。"""
    lw = LedoitWolf(assume_centered=False).fit(R_window)
    return lw.covariance_, lw.shrinkage_


def stabilize_and_diagnose(Sigma):
    """对称化 + eps 对角（eps = 1e-8 * trace(Sigma)/N）；返回 (Sigma_eps, 诊断)。"""
    Sigma = (Sigma + Sigma.T) / 2.0
    N = Sigma.shape[0]
    eps = 1e-8 * np.trace(Sigma) / N
    Sigma_eps = Sigma + eps * np.eye(N)
    ev = np.linalg.eigvalsh(Sigma_eps)
    diag = {
        "eps": eps,
        "min_eigenvalue": float(ev.min()),
        "cond_before": float(np.linalg.cond(Sigma)),
        "cond_after": float(np.linalg.cond(Sigma_eps)),
    }
    return Sigma_eps, diag


def estimate_covariance(R_window, method):
    """统一风险估计入口，返回 (Sigma_eps, diag)。"""
    if method == "S":
        Sigma = sample_covariance(R_window)
    elif method == "LW":
        Sigma, _ = ledoit_wolf_covariance(R_window)
    else:
        raise ValueError(method)
    return stabilize_and_diagnose(Sigma)
