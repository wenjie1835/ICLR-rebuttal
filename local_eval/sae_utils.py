# local_eval/sae_utils.py
from typing import Tuple, Any, Dict
import os

import numpy as np
import torch

from sae_amortization.run_exp5 import _load_pt_or_safetensors, _extract_sae_weights, _canon_orientations


def load_sae_weights(ckpt_path: str, d_in_expected: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用 run_exp5.py 中的工具函数，从 checkpoint 中抽取:
      - W_enc: [d_in, n_lat]
      - D    : [d_in, n_lat]
      - b_enc: [1, n_lat] 或 [1,1] 截断成匹配长度

    这里假设 ckpt_path 为 .pt 或 .safetensors，与原代码兼容。
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

    raw = _load_pt_or_safetensors(ckpt_path)
    W_enc_raw, W_dec_raw, b_enc_raw = _extract_sae_weights(raw)
    W_enc, D = _canon_orientations(W_enc_raw, W_dec_raw, d_in_expected)

    if b_enc_raw is None:
        b_enc = np.zeros((1, W_enc.shape[1]), dtype=np.float32)
    else:
        b = np.asarray(b_enc_raw, dtype=np.float32).reshape(1, -1)
        if b.shape[1] != W_enc.shape[1]:
            # 尝试截断到相同 latent 数
            b = b[:, : W_enc.shape[1]]
        b_enc = b
    return W_enc.astype(np.float32, copy=False), D.astype(np.float32, copy=False), b_enc
