# local_eval/inference_modes.py
from typing import Literal, Dict, Any, Optional

import numpy as np

from sae_amortization.amortization import ista_lasso, calibrate_lambda_by_dense_target
from sae_amortization.run_exp5 import encode_amortized


AmortMode = Literal["amortized", "semi_amortized", "non_amortized"]


def calibrate_lambda_for_X(
    X: np.ndarray,
    D: np.ndarray,
    target_dense: float,
    lam_grid=None,
    steps: int = 100,
    sample_size: int = 5000,
) -> float:
    """
    用 amortization.calibrate_lambda_by_dense_target 在 X 的子集上校准 L1 系数，
    使平均激活密度接近 target_dense。
    """
    n = X.shape[0]
    if n > sample_size:
        X_sub = X[:sample_size]
    else:
        X_sub = X
    lam = calibrate_lambda_by_dense_target(
        X_sub,
        D,
        target_dense=target_dense,
        steps=steps,
        lam_grid=lam_grid,
    )
    return float(lam)


def encode_with_mode(
    X: np.ndarray,
    W_enc: np.ndarray,
    b_enc: np.ndarray,
    D: np.ndarray,
    variant: str,
    mode: AmortMode,
    ista_steps_semi: int,
    ista_steps_non: int,
    lam: float,
    target_dense: float,
) -> np.ndarray:
    """
    统一入口：
      X: [N, d_in]
      W_enc, b_enc, D: 从 checkpoint 中加载和对齐的权重
      variant: "batch_topk", "matryoshka_batch_topk" 等（用于 TopK vs standard）
      mode: "amortized" / "semi_amortized" / "non_amortized"

    返回:
      Z: [N, n_lat]
    """
    # 1) 先算 amortized codes
    Z_amort = encode_amortized(
        X,
        W_enc,
        b_enc,
        variant=variant,
        target_dense=target_dense,
        guessed_k=None,   # 若 checkpoint 中有 k，encode_amortized 自己会使用；否则用 target_dense*m
    )  # Z_amort: [N, n_lat]

    if mode == "amortized":
        return Z_amort.astype(np.float32, copy=False)

    # 注意：amortization.ista_lasso 的 D 约定为 [d, m]，与我们从 _canon_orientations 得到的一致
    if mode == "semi_amortized":
        Z_semi = ista_lasso(
            X,
            D,
            lam=lam,
            steps=ista_steps_semi,
            step_size=None,
            z0=Z_amort,
        )
        return Z_semi.astype(np.float32, copy=False)

    if mode == "non_amortized":
        Z_non = ista_lasso(
            X,
            D,
            lam=lam,
            steps=ista_steps_non,
            step_size=None,
            z0=None,
        )
        return Z_non.astype(np.float32, copy=False)

    raise ValueError(f"Unknown amortization mode: {mode}")
