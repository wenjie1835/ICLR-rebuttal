# train_sae.py
#
# 训练流程：
#   Phase 1: StandardTopKSAE (全局摊销) 预训练，主要追求重建好
#   Phase 2: 用 Standard SAE 计算角方差 + 分组
#   Phase 3: 用分组信息初始化 MultiEncoderTopKSAE (局部摊销)
#   Phase 4: Multi-Encoder 微调：
#              - 更严格的 target_loss_phase4
#              - 加 usage regularizer，减小 dense latent / 均衡 feature 负载

from typing import List

import torch
import torch.nn.functional as F
from torch import optim

from tqdm import tqdm

from .config import SAEConfig
from .data import LMActivationConfig, create_dataloader
from .models import (
    StandardTopKSAE,
    MultiEncoderTopKSAE,
    compute_angular_variance_and_sort,
)


# ---------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------
def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_lm_activation_config(cfg: SAEConfig) -> LMActivationConfig:
    """
    把 SAEConfig 映射到 LMActivationConfig。
    注意这里会把 cfg.normalize_activations 一起传下去。
    """
    # 对老版本 config 做兼容：如果没有 normalize_activations，就默认 True
    normalize_activations = getattr(cfg, "normalize_activations", True)

    return LMActivationConfig(
        model_name=cfg.model_name,
        target_layer=cfg.target_layer,
        d_model=cfg.d_model,
        dataset_name=cfg.dataset_name,
        dataset_split=cfg.dataset_split,
        max_tokens=cfg.max_tokens,
        max_seq_len=cfg.max_seq_len,
        device=cfg.device,
        lm_dtype=cfg.lm_dtype,
        normalize_activations=normalize_activations,
    )


# ---------------------------------------------------------------------
# Phase 1: Standard SAE 预训练
# ---------------------------------------------------------------------
def train_standard_sae(cfg: SAEConfig) -> StandardTopKSAE:
    """
    Phase 1: 训练 StandardTopKSAE（全局摊销编码）

    使用两重条件：
      - step < cfg.phase1_steps （上限）
      - loss >= phase1_target_loss

    一旦 loss < phase1_target_loss，就提前停止。
    """
    print("\n[Phase 1] Training StandardTopKSAE (global amortization)")
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    lm_cfg = build_lm_activation_config(cfg)
    dataloader = create_dataloader(lm_cfg, cfg.batch_size)

    model = StandardTopKSAE(
        d_model=cfg.d_model,
        dict_size=cfg.dict_size,
        k=cfg.global_k,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Phase 1 的目标 loss（如果 cfg 没有单独字段，就用 cfg.target_loss）
    base_target = getattr(cfg, "target_loss", 0.01)
    phase1_target_loss = getattr(cfg, "target_loss_phase1", base_target)

    step = 0
    model.train()
    converged = False

    print(f"[Phase 1] target_loss_phase1 = {phase1_target_loss:.6f}, "
          f"max_steps = {cfg.phase1_steps}")

    while step < cfg.phase1_steps and not converged:
        for x in dataloader:
            step += 1
            x = x.to(device)

            recon, _ = model(x)
            loss = F.mse_loss(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"[Phase 1] step={step} loss={loss.item():.6f}")

            if loss.item() < phase1_target_loss:
                print(
                    f"[Phase 1] Reached target_loss_phase1={phase1_target_loss:.4f} "
                    f"at step={step}, early stopping."
                )
                converged = True
                break

            if step >= cfg.phase1_steps:
                print(
                    f"[Phase 1] Reached max steps={cfg.phase1_steps}, "
                    f"last loss={loss.item():.6f}"
                )
                break

    return model


# ---------------------------------------------------------------------
# Phase 2: 分组（角方差）
# ---------------------------------------------------------------------
def build_groups_indices(cfg: SAEConfig,
                         sorted_indices: torch.Tensor) -> List[torch.Tensor]:
    """
    按角方差排序结果，把 dict_size 个 latent 均分成 n_sub_encoders 组。

    groups_indices[0]: 角方差最大的那一组
    groups_indices[-1]: 角方差最小的一组
    """
    features_per_group = cfg.dict_size // cfg.n_sub_encoders
    groups_indices: List[torch.Tensor] = []

    for i in range(cfg.n_sub_encoders):
        start = i * features_per_group
        end = (i + 1) * features_per_group
        idx_subset = sorted_indices[start:end]
        groups_indices.append(idx_subset)

    return groups_indices


# ---------------------------------------------------------------------
# Phase 3: 用 Standard SAE 初始化 Multi-Encoder SAE
# ---------------------------------------------------------------------
def init_multi_encoder_sae_from_standard(cfg: SAEConfig,
                                         standard_sae: StandardTopKSAE,
                                         groups_indices: List[torch.Tensor]
                                         ) -> MultiEncoderTopKSAE:
    """
    Phase 3: 用 Standard SAE 的权重 + 分组信息初始化 MultiEncoderTopKSAE。
    """
    print("\n[Phase 3] Initializing MultiEncoderTopKSAE from Standard SAE...")

    device = torch.device(cfg.device)
    features_per_group = cfg.dict_size // cfg.n_sub_encoders
    sub_sizes = [features_per_group] * cfg.n_sub_encoders

    multi_sae = MultiEncoderTopKSAE(
        d_model=cfg.d_model,
        n_sub_encoders=cfg.n_sub_encoders,
        sub_dict_sizes=sub_sizes,
        group_ks=cfg.group_ks,
    ).to(device)

    with torch.no_grad():
        old_enc_w = standard_sae.encoder.weight.data   # [dict_size, d_model]
        old_enc_b = standard_sae.encoder.bias.data     # [dict_size]
        old_dec_w = standard_sae.decoder.weight.data   # [d_model, dict_size]

        for i in range(cfg.n_sub_encoders):
            idx = groups_indices[i].to(old_enc_w.device)  # [features_per_group]

            multi_sae.encoders[i].weight.data = old_enc_w[idx].clone()
            multi_sae.encoders[i].bias.data = old_enc_b[idx].clone()
            multi_sae.decoders[i].weight.data = old_dec_w[:, idx].clone()

    print("[Phase 3] Weight transfer complete.")
    return multi_sae


# ---------------------------------------------------------------------
# Phase 4: 微调 Multi-Encoder SAE (局部摊销 + usage regularizer)
# ---------------------------------------------------------------------
def finetune_multi_encoder_sae(cfg: SAEConfig,
                               multi_sae: MultiEncoderTopKSAE):
    """
    Phase 4: 在相同数据分布上微调 Multi-Encoder SAE（局部摊销）。

    新增：
      - 使用更严格的 phase4_target_loss
      - 加一个 usage regularizer，减少 dense latents / 均衡 feature usage

    usage_regularizer:
      - 对每个 batch 计算 usage_batch_j = mean_x[ 1_{a_j(x) > 0} ]
      - target_freq = sum(group_ks) / dict_size  (期望每个 latent 被使用的频率)
      - L_usage = λ * mean_j (usage_batch_j - target_freq)^2
    """
    print("\n[Phase 4] Fine-tuning MultiEncoderTopKSAE (local amortization)")
    device = torch.device(cfg.device)

    lm_cfg = build_lm_activation_config(cfg)
    dataloader = create_dataloader(lm_cfg, cfg.batch_size)

    # 微调使用较小学习率
    base_lr = cfg.lr
    optimizer = optim.Adam(
        multi_sae.parameters(), lr=base_lr * 0.5
    )

    # target_loss 设置：默认比 Phase 1 严格一些
    base_target = getattr(cfg, "target_loss", 0.01)
    # 如果没有专门的 target_loss_phase4，则用 base_target * 0.7
    phase4_target_loss = getattr(cfg, "target_loss_phase4", base_target * 0.7)

    # usage regularizer 参数（可在 cfg 里加字段；没有的话用默认值）
    usage_reg_strength = getattr(cfg, "usage_reg_strength", 1e-3)
    usage_reg_warmup_steps = getattr(cfg, "usage_reg_warmup_steps", 1000)
    usage_ema_alpha = getattr(cfg, "usage_ema_alpha", 0.1)

    total_k = sum(cfg.group_ks)
    target_freq = total_k / cfg.dict_size

    print(f"[Phase 4] target_loss_phase4 = {phase4_target_loss:.6f}, "
          f"max_steps = {cfg.phase4_steps}")
    print(f"[Phase 4] usage_reg_strength = {usage_reg_strength:.2e}, "
          f"usage_reg_warmup_steps = {usage_reg_warmup_steps}, "
          f"target_freq ≈ {target_freq:.6e}")

    step = 0
    multi_sae.train()
    converged = False

    # 用一个 running 估计记录 usage 分布（非必须，但便于调参）
    usage_running = torch.zeros(cfg.dict_size, device=device)

    while step < cfg.phase4_steps and not converged:
        for x in dataloader:
            step += 1
            x = x.to(device)

            recon, acts = multi_sae(x)
            loss_mse = F.mse_loss(recon, x)

            # 计算 batch 内的 activation usage
            binary_acts = (acts > 0).float()               # [B, dict_size]
            batch_usage = binary_acts.mean(dim=0)          # [dict_size]
            usage_running = (1 - usage_ema_alpha) * usage_running + \
                usage_ema_alpha * batch_usage

            # usage regularizer（简单的 L2 偏离）
            if step > usage_reg_warmup_steps and usage_reg_strength > 0.0:
                usage_reg = ((batch_usage - target_freq) ** 2).mean()
                loss = loss_mse + usage_reg_strength * usage_reg
            else:
                usage_reg = torch.tensor(0.0, device=device)
                loss = loss_mse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                avg_active = binary_acts.sum(dim=1).float().mean().item()
                print(
                    f"[Phase 4] step={step} "
                    f"loss={loss.item():.6f} "
                    f"(mse={loss_mse.item():.6f}, usage_reg={usage_reg.item():.6e}) "
                    f"avg_active={avg_active:.2f}"
                )

            # 早停逻辑：用 mse_loss 和总 loss 两个角度看
            if loss_mse.item() < phase4_target_loss:
                print(
                    f"[Phase 4] Reached target_loss_phase4={phase4_target_loss:.4f} "
                    f"(mse_loss) at step={step}, early stopping."
                )
                converged = True
                break

            if step >= cfg.phase4_steps:
                print(
                    f"[Phase 4] Reached max steps={cfg.phase4_steps}, "
                    f"last mse_loss={loss_mse.item():.6f}, "
                    f"last total_loss={loss.item():.6f}"
                )
                break

    # 训练结束时打印一个大致的 usage 概览
    with torch.no_grad():
        usage_cpu = usage_running.detach().cpu()
        print("[Phase 4] Approx. usage stats (EMA over training):")
        print(f"  usage.mean   = {usage_cpu.mean().item():.6e}")
        print(f"  usage.max    = {usage_cpu.max().item():.6e}")
        print(f"  usage.min    = {usage_cpu.min().item():.6e}")

    return multi_sae


# ---------------------------------------------------------------------
# 保存与总管函数
# ---------------------------------------------------------------------
def save_all(cfg: SAEConfig,
             standard_sae: StandardTopKSAE,
             multi_sae: MultiEncoderTopKSAE,
             groups_indices: List[torch.Tensor],
             angular_variance: torch.Tensor):
    """
    保存：
      - config.json
      - standard_sae.pt
      - multi_sae.pt
      - loc_angular_meta.pt（分组索引 + 角方差）
    """
    out_dir = cfg.ensure_output_dir()

    cfg.save_json("config.json")

    std_path = out_dir / "standard_sae.pt"
    multi_path = out_dir / "multi_encoder_sae.pt"
    meta_path = out_dir / "loc_angular_meta.pt"

    torch.save(standard_sae.state_dict(), std_path)
    torch.save(multi_sae.state_dict(), multi_path)

    meta = {
        "groups_indices": [g.cpu() for g in groups_indices],
        "angular_variance": angular_variance.cpu(),
        "group_ks": list(cfg.group_ks),
    }
    torch.save(meta, meta_path)

    print(f"[Save] Standard SAE saved to {std_path}")
    print(f"[Save] Multi-Encoder SAE saved to {multi_path}")
    print(f"[Save] Meta info saved to {meta_path}")


def run_pipeline(cfg: SAEConfig):
    """
    总管函数：从 LM activation -> Standard SAE -> Angular Var -> Multi-Encoder SAE
    """
    print("=" * 80)
    print(f"Starting LOC-AMT-SAE pipeline for model: {cfg.model_name}")
    print(f"Output dir: {cfg.output_dir}")
    print(f"dict_size={cfg.dict_size}, n_sub_encoders={cfg.n_sub_encoders}, "
          f"global_k={cfg.global_k}, group_ks={cfg.group_ks}")
    print(f"target_loss (base)={getattr(cfg, 'target_loss', 0.01)}")
    print("=" * 80)

    device = torch.device(cfg.device)

    # Phase 1: Standard SAE
    standard_sae = train_standard_sae(cfg)

    # Phase 2: Angular variance + 分组
    print("\n[Phase 2] Computing angular variance and grouping...")
    lm_cfg = build_lm_activation_config(cfg)
    profiling_loader = create_dataloader(lm_cfg, cfg.batch_size)
    sorted_indices, angular_variance = compute_angular_variance_and_sort(
        standard_sae,
        profiling_loader,
        device=device,
        max_batches=cfg.angular_profile_batches,
    )
    groups_indices = build_groups_indices(cfg, sorted_indices)

    # Phase 3: 初始化 Multi-Encoder SAE
    multi_sae = init_multi_encoder_sae_from_standard(cfg, standard_sae, groups_indices)

    # Phase 4: 微调 Multi-Encoder SAE
    multi_sae = finetune_multi_encoder_sae(cfg, multi_sae)

    # 简单检查分组稀疏性
    print("\n[Check] Group sparsity on a small batch:")
    lm_cfg = build_lm_activation_config(cfg)
    test_loader = create_dataloader(lm_cfg, batch_size=32)
    multi_sae.eval()
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            _, acts = multi_sae(x)
            features_per_group = cfg.dict_size // cfg.n_sub_encoders

            g0 = acts[:, :features_per_group]
            g7 = acts[:, -features_per_group:]
            g0_active = (g0 > 0).sum(dim=1).float().mean().item()
            g7_active = (g7 > 0).sum(dim=1).float().mean().item()
            print(f"  Group 0 (K={cfg.group_ks[0]}) avg active: {g0_active:.2f}")
            print(f"  Group 7 (K={cfg.group_ks[-1]}) avg active: {g7_active:.2f}")
            break

    # 保存权重 & 配置
    save_all(cfg, standard_sae, multi_sae, groups_indices, angular_variance)
