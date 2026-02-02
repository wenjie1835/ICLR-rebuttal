# encode_latents.py: unified encoding for standard, topk, batch_topk, matryoshka_batch_topk, gated SAEs
from typing import Literal, Dict, Optional
import numpy as np
import torch
from .sae_io import load_sae_state
from .amortization import ista_lasso, calibrate_lambda_by_dense_target

EncodeMode = Literal["amortized", "semi", "unamortized", "topk_omp"]


def _relu(x): return np.maximum(x, 0.0)


def _topk_mask_rows(scores: np.ndarray, k: int) -> np.ndarray:
    """Keep top-k per row; fallback to full if k <= 0 or k >= m."""
    n, m = scores.shape
    if k <= 0 or k >= m:
        return np.ones_like(scores, dtype=bool)
    idx = np.argpartition(scores, kth=m - k, axis=1)[:, -k:]
    mask = np.zeros_like(scores, dtype=bool)
    row_idx = np.arange(n)[:, None]
    mask[row_idx, idx] = True
    return mask

def _power_iter_spectral_norm_sq(D: np.ndarray, iters: int = 50) -> float:
    # Estimate L = ||D||_2^2 (for ISTA step-size).
    u = np.random.randn(D.shape[0])
    u /= (np.linalg.norm(u) + 1e-8)
    for _ in range(iters):
        v = D.T @ u
        v /= (np.linalg.norm(v) + 1e-8)
        u = D @ v
        u /= (np.linalg.norm(u) + 1e-8)
    sigma = float(u @ (D @ v))
    return sigma ** 2

def _ista_from(
    X: np.ndarray, D: np.ndarray, lam: float, steps: int,
    z0: Optional[np.ndarray] = None
) -> np.ndarray:
    #  ISTA， z0； ista_lasso 
    if steps <= 0:
        return z0 if z0 is not None else np.zeros((X.shape[0], D.shape[1]), dtype=np.float32)
    if z0 is None:
        Z = np.zeros((X.shape[0], D.shape[1]), dtype=np.float32)
    else:
        Z = z0.astype(np.float32, copy=True)
    L = _power_iter_spectral_norm_sq(D) ** 0.5 + 1e-8  #  1/L
    Dt = D.T
    for _ in range(steps):
        R = X - Z @ D.T                       # 
        G = Z + (R @ D) / L                   # 
        Z = _relu(G - lam / L)                # soft-threshold + （）
    return Z

def _omp_k_sparse(X: np.ndarray, D: np.ndarray, k: int) -> np.ndarray:
    #  OMP（TopK  k-），
    N, d = X.shape
    m = D.shape[1]
    Z = np.zeros((N, m), dtype=np.float32)
    Dt = D.T
    for i in range(N):
        r = X[i].copy()
        S = []
        for _ in range(k):
            j = int(np.argmax(np.abs(Dt @ r)))
            if j in S: break
            S.append(j)
            Ds = D[:, S]                      # [d, |S|]
            # 
            coef, *_ = np.linalg.lstsq(Ds, X[i], rcond=None)
            r = X[i] - Ds @ coef
        if len(S) > 0:
            Z[i, S] = np.clip(coef, 0.0, None)  # （）
    return Z

def encode_latents(
    X: np.ndarray,
    sae_ckpt_path: str,
    mode: EncodeMode = "unamortized",
    ista_steps: int = 200,
    semi_steps: int = 10,
    target_dense: float = 0.10,
    k_for_topk: int = 20,
    variant: str = "standard",
    guessed_k: Optional[int] = None,
) -> np.ndarray:
    """
    variant: "standard" | "topk" | "batch_topk" | "matryoshka_batch_topk" | "gated" (handled via gate in D).
    For batch_topk / matryoshka_batch_topk, amortized uses Top-K per row; guessed_k from ckpt or target_dense*m.
    """
    state: Dict = load_sae_state(sae_ckpt_path)
    D = state["W_dec"].cpu().numpy().astype(np.float32)  # [d, m]
    gate = None
    for k in ["gate", "gates", "gate_vec"]:
        if k in state:
            gate = state[k].cpu().numpy().astype(np.float32).ravel()
            break
    if gate is not None and gate.shape[0] == D.shape[1]:
        D = D * gate.reshape(1, -1)

    lam = calibrate_lambda_by_dense_target(X, D, target_dense=target_dense, steps=60)

    if mode == "amortized":
        W_enc = state.get("W_enc", None)
        b_enc = state.get("b_enc", None)
        if W_enc is None:
            raise ValueError("Amortized mode requires W_enc in checkpoint.")
        W = W_enc.cpu().numpy().astype(np.float32)  # [m, d]
        b = np.zeros(W.shape[0], dtype=np.float32)
        if b_enc is not None:
            b = b_enc.cpu().numpy().astype(np.float32).ravel()
            if b.size != W.shape[0]:
                b = b[: W.shape[0]]
        Z_lin = X @ W.T + b
        if "topk" in variant.lower():
            m = Z_lin.shape[1]
            k = guessed_k if guessed_k is not None else max(1, int(round(target_dense * m)))
            mask = _topk_mask_rows(Z_lin, k)
            Z = np.where(mask, Z_lin, 0.0).astype(np.float32)
            return Z
        return _relu(Z_lin)

    if mode == "semi":
        W_enc = state.get("W_enc", None)
        if W_enc is not None:
            W = W_enc.cpu().numpy().astype(np.float32)
            b_enc = state.get("b_enc", None)
            b = np.zeros(W.shape[0], dtype=np.float32) if b_enc is None else b_enc.cpu().numpy().astype(np.float32).ravel()[: W.shape[0]]
            z0_lin = X @ W.T + b
            if "topk" in variant.lower():
                m = z0_lin.shape[1]
                k = guessed_k if guessed_k is not None else max(1, int(round(target_dense * m)))
                mask = _topk_mask_rows(z0_lin, k)
                z0 = np.where(mask, z0_lin, 0.0).astype(np.float32)
            else:
                z0 = _relu(z0_lin)
        else:
            L = (_power_iter_spectral_norm_sq(D) ** 0.5) + 1e-8
            z0 = _relu(X @ D / L - lam / L)
        return _ista_from(X, D, lam=lam, steps=semi_steps, z0=z0)

    if mode == "unamortized":
        return _ista_from(X, D, lam=lam, steps=ista_steps, z0=None)

    if mode == "topk_omp":
        return _omp_k_sparse(X, D, k=k_for_topk)

    raise ValueError(f"unknown encode mode: {mode}")
