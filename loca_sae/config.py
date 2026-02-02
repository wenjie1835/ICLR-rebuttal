# config.py
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import torch


@dataclass
class SAEConfig:
    # --- 基础模型相关 ---
    model_name: str              # HF 模型名
    d_model: int                 # resid_post 维度
    target_layer: int            # 取第几层 resid_post（hidden_states 索引）

    # --- SAE 结构超参（按你给的保持不变）---
    dict_size: int = 16384
    n_sub_encoders: int = 8
    global_k: int = 20
    group_ks: tuple = (6, 5, 4, 3, 3, 2, 2, 1)

    # --- 数据相关 ---
    dataset_name: str = "monology/pile-uncopyrighted"
    dataset_split: str = "train"
    max_tokens: int = 10_000_000       # 使用的 token 上限
    max_seq_len: int = 128             # LM 前向时的最大长度

    # ✅ 新增：是否对 resid_post 做 L2 归一化
    normalize_activations: bool = True

    # --- 训练超参 ---
    lr: float = 3e-4
    batch_size: int = 2048
    phase1_steps: int = 50_000         # Standard SAE 的最大 step 数（上限）
    phase4_steps: int = 50_000         # Multi-Encoder SAE 的最大 step 数（上限）
    angular_profile_batches: int = 200 # 角方差统计时最多多少个 batch

    # ✅ 早停阈值
    target_loss: float = 0.1

    # --- 设备与精度 ---
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lm_dtype: str = "bfloat16"         # LM 前向的 dtype: "bfloat16" or "float16"

    # --- 输出路径 ---
    output_dir: str = "loc-amt-sae/pythia-160m/"

    # --- 杂项 ---
    seed: int = 42

    def ensure_output_dir(self):
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, filename: str = "config.json"):
        out_dir = self.ensure_output_dir()
        cfg_path = out_dir / filename
        with cfg_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        print(f"[Config] Saved to {cfg_path}")
