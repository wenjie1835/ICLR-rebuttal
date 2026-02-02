# local_eval/config.py
import torch

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用的数据集
DATASET_NAME = "monology/pile-uncopyrighted"
DATASET_SPLIT = "train"

# 仅使用前 N 个 token 的 resid_post 激活
MAX_TOKENS = 10_000

# 单个样本最大长度，与 batch_size（只影响激活采集）
MAX_SEQ_LEN = 128
BATCH_SIZE = 4

# 模型和对应层（与 SAEBench 对应）
MODEL_CONFIGS = {
    "pythia-160m-deduped": {
        "hf_name": "EleutherAI/pythia-160m-deduped",
        "target_layer": 8,     # 对应 resid_post layer 8，近似为 hidden_states[8]
    },
    "gemma-2-2b": {
        "hf_name": "google/gemma-2-2b",
        "target_layer": 12,    # resid_post layer 12
    },
}

# Default SAE checkpoint paths (override with --sae_path in scripts)
# Place your batch_topk / matryoshka_batch_topk ae.pt under these paths or pass --sae_path
SAE_PATHS = {
    ("pythia-160m-deduped", "batch_topk"): "batch_topk/pythia-160m/ae.pt",
    ("pythia-160m-deduped", "matryoshka_batch_topk"): "matryoshka_batch_topk/pythia-160m/ae.pt",
    ("gemma-2-2b", "batch_topk"): "batch_topk/gemma-2-2b/ae.pt",
    ("gemma-2-2b", "matryoshka_batch_topk"): "matryoshka_batch_topk/gemma-2-2b/ae.pt",
}

# ISTA 超参数：与你在问题中描述保持一致
SEMI_STEPS = 30       # Semi 摊销：encoder 初始化 + 30 步 ISTA
NON_STEPS = 200       # Non 摊销：从 0 初始化 + 200 步 ISTA

# L1 正则（会通过 grid search 校准）
TARGET_DENSE = 0.1    # 目标平均激活密度，可按需要调
LAMBDA_GRID = None    # 若为 None 则用 amortization.calibrate_lambda_by_dense_target 默认网格
