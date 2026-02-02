"""
Semi-amortized step sweep: amortized baseline + ISTA from Z_amort for various steps.
Run from repo root: python scripts/run_semi_step_sweep.py --model_name pythia-160m-deduped ...
"""
import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from local_eval.config import MODEL_CONFIGS, SAE_PATHS
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


def compute_all_metrics(
    X: np.ndarray, Z: np.ndarray, D: np.ndarray,
) -> Dict[str, float]:
    """Unified 8 metrics."""
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


def semi_step_sweep_for_model(
    model_name: str,
    sae_path: str,
    steps_list: List[int],
    out_dir: str,
) -> Dict[str, Dict[str, float]]:
    """Amortized baseline + semi-amortized for each step count."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== Semi-amortized step sweep for {model_name} / BatchTopK SAE ===")
    print(f"SAE checkpoint: {sae_path}")

    print("[1/4] Collecting resid_post activations (first 10k tokens) ...")
    X = collect_resid_activations(model_name)
    N, d_in = X.shape
    print(f"  -> X shape = {X.shape}")

    print("[2/4] Loading SAE weights ...")
    W_enc, D, b_enc = load_sae_weights(sae_path, d_in_expected=d_in)
    n_lat = W_enc.shape[1]
    print(f"  -> latent dim = {n_lat}, d_in = {d_in}")

    print("[3/4] Computing amortized baseline ...")
    Z_amort = encode_amortized(
        X, W_enc, b_enc, variant="batch_topk", target_dense=0.1, guessed_k=None,
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

    for steps in steps_list:
        print(f"\n  >> Semi-amortized with ISTA steps = {steps}")
        Z_semi = ista_lasso(X, D, lam=lam, steps=steps, step_size=None, z0=Z_amort)
        metrics = compute_all_metrics(X, Z_semi, D)
        key = f"semi_steps_{steps}"
        results[key] = metrics
        for k, v in metrics.items():
            print(f"    {k:15s}: {v:.6f}")

    out_path = os.path.join(out_dir, f"{model_name}_batch_topk_semi_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Saved sweep results to {out_path}")
    return results


def parse_steps_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", choices=list(MODEL_CONFIGS.keys()), required=True)
    ap.add_argument("--sae_path", type=str, default=None)
    ap.add_argument(
        "--steps",
        type=str,
        default="5,10,15,20,25,30,35,40,45,50",
        help="Comma-separated ISTA steps, e.g. '5,10,15,20,25,30,35,40,45,50'",
    )
    ap.add_argument("--out_dir", type=str, default="semi_sweep_results")
    args = ap.parse_args()

    steps_list = parse_steps_list(args.steps)

    if args.sae_path is None:
        key = (args.model_name, "batch_topk")
        if key not in SAE_PATHS:
            raise KeyError(
                f"No default SAE path for {key}. Set local_eval/config.SAE_PATHS or pass --sae_path."
            )
        sae_path = SAE_PATHS[key]
    else:
        sae_path = args.sae_path

    semi_step_sweep_for_model(
        model_name=args.model_name,
        sae_path=sae_path,
        steps_list=steps_list,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
