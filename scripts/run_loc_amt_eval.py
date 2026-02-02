"""
Evaluate Loca-SAE (local amortization SAE) with amortized inference only.
Run from repo root: python scripts/run_loc_amt_eval.py --model_name pythia-160m-deduped ...
"""
import argparse
import json
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from local_eval.config import MODEL_CONFIGS
from local_eval.activations_stream import collect_resid_activations
from local_eval.sae_utils import load_sae_weights

from sae_amortization.run_exp5 import (
    nmse,
    dead_and_dense_rates,
    pseudo_labels_from_norm,
    k_sparse_probe_corr,
    absorption_from_corr,
    encode_amortized,
)

# After training with loca_sae and running convert_multi_to_saebench, ae.pt is here
DEFAULT_LOC_AMT_SAE_PATHS = {
    "pythia-160m-deduped": "loc-amt-sae/pythia-160m/ae.pt",
    "gemma-2-2b": "loc-amt-sae/gemma-2-2b/ae.pt",
}


def compute_all_metrics(X: np.ndarray, Z: np.ndarray, D: np.ndarray) -> Dict[str, float]:
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


def evaluate_loc_amt_sae(
    model_name: str,
    sae_path: str,
    variant: str,
    out_dir: str,
) -> Dict[str, float]:
    """Evaluate Loca-SAE with amortized inference only."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== Evaluating local-amortized SAE for {model_name} ===")
    print(f"Checkpoint: {sae_path}")

    print("[1/3] Collecting resid_post activations (first 10k tokens) ...")
    X = collect_resid_activations(model_name)
    N, d_in = X.shape
    print(f"  -> X shape = {X.shape}")

    print("[2/3] Loading SAE weights ...")
    W_enc, D, b_enc = load_sae_weights(sae_path, d_in_expected=d_in)
    n_lat = W_enc.shape[1]
    print(f"  -> latent dim = {n_lat}, d_in = {d_in}")

    print("[3/3] Computing amortized codes & metrics ...")
    Z = encode_amortized(
        X, W_enc, b_enc, variant=variant, target_dense=0.1, guessed_k=None,
    )
    metrics = compute_all_metrics(X, Z, D)

    print("\n[Results] Local-amortized SAE (amortized inference only):")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.6f}")

    out_path = os.path.join(out_dir, f"{model_name}_loc_amt_sae_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Saved metrics to {out_path}")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_name",
        choices=list(MODEL_CONFIGS.keys()),
        required=True,
        help="LM: pythia-160m-deduped or gemma-2-2b",
    )
    ap.add_argument("--sae_path", type=str, default=None, help="Loca-SAE ae.pt path")
    ap.add_argument(
        "--variant",
        type=str,
        default="loc_amortized",
        help="Variant for encoder; use 'loc_amt_topk' if Top-K encoder.",
    )
    ap.add_argument("--out_dir", type=str, default="loc_amt_eval_results")
    args = ap.parse_args()

    if args.sae_path is None:
        if args.model_name not in DEFAULT_LOC_AMT_SAE_PATHS:
            raise KeyError(
                f"No default Loca-SAE path for {args.model_name}. Pass --sae_path."
            )
        sae_path = DEFAULT_LOC_AMT_SAE_PATHS[args.model_name]
    else:
        sae_path = args.sae_path

    evaluate_loc_amt_sae(
        model_name=args.model_name,
        sae_path=sae_path,
        variant=args.variant,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
