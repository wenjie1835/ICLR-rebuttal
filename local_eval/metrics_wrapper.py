# local_eval/metrics_wrapper.py
from typing import Dict, Tuple

import numpy as np

from sae_amortization.run_exp5 import (
    nmse,
    dead_and_dense_rates,
    pseudo_labels_from_norm,
    k_sparse_probe_corr,
    absorption_from_corr,
)


def evaluate_metrics_on_codes(
    X: np.ndarray,
    Z: np.ndarray,
    D: np.ndarray,
    dense_thresholds=(0.1, 0.2),
    dead_thr: float = 1e-6,
    absorption_topk: int = 5,
) -> Dict[str, float]:
    """
    按照 run_exp5.py 中的 unified metrics 协议，在给定的 X, Z 上计算：
      - NMSE
      - DeadRate
      - DenseRate@0.1
      - DenseRate@0.2
      - F1@1
      - F1@2
      - ΔF1
      - AbsorptionRate
    """
    # 1) 重构
    X_hat = Z @ D.T  # 与 run_exp5 中一致，D: [d_in, n_lat]

    # 2) NMSE（用 Z 原值，不 ReLU）
    NMSE = nmse(X, X_hat)

    # 3) 统一 metrics 用非负 codes：Z_relu = max(Z, 0) 
    Z_relu = np.maximum(Z, 0.0).astype(np.float32, copy=False)

    # 4) Dead & Dense rates
    dead_rate, dense_dict = dead_and_dense_rates(
        Z_relu,
        dense_thresholds=dense_thresholds,
        dead_thr=dead_thr,
    )
    dense_01 = dense_dict.get("dense@0.1", float("nan"))
    dense_02 = dense_dict.get("dense@0.2", float("nan"))

    # 5) 伪标签（基于范数）用于 probing
    y = pseudo_labels_from_norm(X)

    # 6) k-sparse probe，取 k=1,2
    f1s, order = k_sparse_probe_corr(Z_relu, y, ks=(1, 2))
    F1_1 = f1s.get("F1@1", float("nan"))
    F1_2 = f1s.get("F1@2", float("nan"))
    dF1 = f1s.get("dF1", float("nan"))

    # 7) Absorption
    absorption = absorption_from_corr(Z_relu, y, order, topk=absorption_topk)

    # 8) 整理成你论文中的名字
    metrics = {
        "NMSE": float(NMSE),
        "DeadRate": float(dead_rate),
        "DenseRate@0.1": float(dense_01),
        "DenseRate@0.2": float(dense_02),
        "F1@1": float(F1_1),
        "F1@2": float(F1_2),
        "ΔF1": float(dF1),
        "AbsorptionRate": float(absorption),
    }
    return metrics
