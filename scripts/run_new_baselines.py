"""
Evaluate a single SAE (batch_topk or matryoshka_batch_topk) under three amortization modes:
amortized, semi_amortized, non_amortized. Run from repo root: python scripts/run_new_baselines.py ...
"""
import argparse
import json
import os
import sys
from typing import Dict

# Ensure repo root is on path when running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from local_eval.config import (
    SAE_PATHS,
    MODEL_CONFIGS,
    SEMI_STEPS,
    NON_STEPS,
)
from local_eval.activations_stream import collect_resid_activations
from local_eval.sae_utils import load_sae_weights

from sae_amortization.amortization import ista_lasso, calibrate_lambda_by_dense_target
from sae_amortization.run_exp5 import (
    nmse,
    dead_and_dense_rates,
    pseudo_labels_from_norm,
    k_sparse_probe_corr,
    absorption_from_corr,
    encode_amortized,
)


def compute_all_metrics(X: np.ndarray, Z: np.ndarray, D: np.ndarray) -> Dict[str, float]:
    """Unified 8 metrics: NMSE, DeadRate, DenseRate@0.1/0.2, F1@1, F1@2, ΔF1, AbsorptionRate."""
    X_hat = Z @ D.T
    nmse_val = nmse(X, X_hat)
    Z_relu = np.maximum(Z, 0.0).astype(np.float32, copy=False)
    dead_rate, dense_dict = dead_and_dense_rates(
        Z_relu, dense_thresholds=(0.1, 0.2), dead_thr=1e-6,
    )
    dense_01 = dense_dict.get("dense@0.1", float("nan"))
    dense_02 = dense_dict.get("dense@0.2", float("nan"))
    y = pseudo_labels_from_norm(X)
    f1s, order = k_sparse_probe_corr(Z_relu, y, ks=(1, 2))
    F1_1 = f1s.get("F1@1", float("nan"))
    F1_2 = f1s.get("F1@2", float("nan"))
    dF1 = f1s.get("dF1", float("nan"))
    absorption = absorption_from_corr(Z_relu, y, order, topk=5)
    return {
        "NMSE": float(nmse_val),
        "DeadRate": float(dead_rate),
        "DenseRate@0.1": float(dense_01),
        "DenseRate@0.2": float(dense_02),
        "F1@1": float(F1_1),
        "F1@2": float(F1_2),
        "ΔF1": float(dF1),
        "AbsorptionRate": float(absorption),
    }


def evaluate_single_sae(
    model_name: str,
    sae_variant: str,
    sae_path: str,
    out_dir: str,
) -> Dict[str, Dict[str, float]]:
    """Run amortized, semi_amortized, non_amortized and save 8 metrics per mode."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== Evaluating {model_name} / {sae_variant} ===")
    print(f"Checkpoint: {sae_path}")

    print("[1/4] Collecting resid_post activations ...")
    X = collect_resid_activations(model_name)
    N, d_in = X.shape
    print(f"  -> X shape = {X.shape}")

    print("[2/4] Loading SAE weights ...")
    W_enc, D, b_enc = load_sae_weights(sae_path, d_in_expected=d_in)
    n_lat = W_enc.shape[1]
    print(f"  -> latent dim = {n_lat}, d_in = {d_in}")

    print("[3/4] Computing amortized codes & metrics ...")
    Z_amort = encode_amortized(
        X, W_enc, b_enc, variant=sae_variant, target_dense=0.1, guessed_k=None,
    )
    metrics_amort = compute_all_metrics(X, Z_amort, D)
    print("  Amortized metrics:")
    for k, v in metrics_amort.items():
        print(f"    {k:15s}: {v:.6f}")

    target_dense = metrics_amort["DenseRate@0.1"]
    print(f"  -> Using DenseRate@0.1(amortized) = {target_dense:.6f} as ISTA target density")

    print("[4/4] Calibrating lambda for ISTA ...")
    X_sub = X[: min(5000, N)]
    lam = calibrate_lambda_by_dense_target(
        X_sub, D, target_dense=target_dense, steps=100, lam_grid=None,
    )
    print(f"  -> calibrated lambda = {lam:.4e}")

    results: Dict[str, Dict[str, float]] = {}
    results["amortized"] = metrics_amort

    print("\n  >> Mode: semi_amortized")
    Z_semi = ista_lasso(X, D, lam=lam, steps=SEMI_STEPS, step_size=None, z0=Z_amort)
    metrics_semi = compute_all_metrics(X, Z_semi, D)
    for k, v in metrics_semi.items():
        print(f"    {k:15s}: {v:.6f}")
    results["semi_amortized"] = metrics_semi

    print("\n  >> Mode: non_amortized")
    Z_non = ista_lasso(X, D, lam=lam, steps=NON_STEPS, step_size=None, z0=None)
    metrics_non = compute_all_metrics(X, Z_non, D)
    for k, v in metrics_non.items():
        print(f"    {k:15s}: {v:.6f}")
    results["non_amortized"] = metrics_non

    out_path = os.path.join(out_dir, f"{model_name}_{sae_variant}_3modes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Saved metrics to {out_path}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", choices=list(MODEL_CONFIGS.keys()), required=True)
    ap.add_argument("--sae_variant", choices=["batch_topk", "matryoshka_batch_topk"], required=True)
    ap.add_argument("--sae_path", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="new_baseline_results")
    args = ap.parse_args()

    if args.sae_path is None:
        key = (args.model_name, args.sae_variant)
        if key not in SAE_PATHS:
            raise KeyError(
                f"No default SAE path for {key}. Set local_eval/config.SAE_PATHS or pass --sae_path."
            )
        sae_path = SAE_PATHS[key]
    else:
        sae_path = args.sae_path

    evaluate_single_sae(
        model_name=args.model_name,
        sae_variant=args.sae_variant,
        sae_path=sae_path,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
