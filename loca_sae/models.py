# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class StandardTopKSAE(nn.Module):
    """
    标准 Top-K SAE，用于 Phase 1 预训练 + 角方差 profiling。
    """
    def __init__(self, d_model: int, dict_size: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.k = k

        self.encoder = nn.Linear(d_model, dict_size)
        self.decoder = nn.Linear(dict_size, d_model, bias=False)

        # 初始化：encoder.bias = 0, decoder 列向量归一化
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            w = self.decoder.weight.data
            w = w / (w.norm(dim=0, keepdim=True) + 1e-8)
            self.decoder.weight.data = w

    def forward(self, x):
        """
        x: [B, d_model]
        return:
            recon: [B, d_model]
            acts:  [B, dict_size]
        """
        pre_acts = self.encoder(x)                      # [B, dict_size]
        topk_values, topk_indices = torch.topk(
            pre_acts, k=self.k, dim=-1
        )
        mask = torch.zeros_like(pre_acts)
        mask.scatter_(-1, topk_indices, 1.0)
        acts = torch.relu(pre_acts) * mask
        recon = self.decoder(acts)
        return recon, acts


class MultiEncoderTopKSAE(nn.Module):
    """
    Multi-Encoder / Grouped Top-K SAE
    将大字典拆成 8 组，每组独立 encoder/decoder 和 Top-K（局部摊销）。
    """
    def __init__(self, d_model: int, n_sub_encoders: int, sub_dict_sizes, group_ks):
        super().__init__()
        assert n_sub_encoders == len(sub_dict_sizes)
        assert n_sub_encoders == len(group_ks)

        self.d_model = d_model
        self.n_sub_encoders = n_sub_encoders
        self.sub_dict_sizes = list(sub_dict_sizes)
        self.group_ks = list(group_ks)
        self.dict_size = sum(self.sub_dict_sizes)

        self.encoders = nn.ModuleList(
            [nn.Linear(d_model, size) for size in self.sub_dict_sizes]
        )
        self.decoders = nn.ModuleList(
            [nn.Linear(size, d_model, bias=False) for size in self.sub_dict_sizes]
        )

        # 初始化：decoder 列向量归一化
        with torch.no_grad():
            for dec in self.decoders:
                w = dec.weight.data
                w = w / (w.norm(dim=0, keepdim=True) + 1e-8)
                dec.weight.data = w

    def forward(self, x):
        """
        x: [B, d_model]
        return:
            final_recon: [B, d_model]
            full_acts:   [B, dict_size]  (按组拼接)
        """
        recons = []
        all_acts = []

        for i in range(self.n_sub_encoders):
            pre_acts = self.encoders[i](x)             # [B, sub_dict_size]
            k = self.group_ks[i]
            vals, idxs = torch.topk(pre_acts, k=k, dim=-1)

            mask = torch.zeros_like(pre_acts)
            mask.scatter_(-1, idxs, 1.0)

            acts = torch.relu(pre_acts) * mask
            recon = self.decoders[i](acts)

            recons.append(recon)
            all_acts.append(acts)

        final_recon = torch.stack(recons, dim=0).sum(dim=0)
        full_acts = torch.cat(all_acts, dim=-1)
        return final_recon, full_acts


@torch.no_grad()
def compute_angular_variance_and_sort(model: StandardTopKSAE,
                                      dataloader,
                                      device: torch.device,
                                      max_batches: int = None):
    """
    角方差 (Angular Variance) 指标：
        对每个 latent j，收集所有激活它的输入 x，归一化后求均值向量 μ_j，
        角方差 = 1 - || μ_j ||

    返回：
        sorted_indices: [dict_size]，按角方差从大到小排序
        angular_variance: [dict_size]
    """
    print(f"[AngularVar] dict_size={model.dict_size}, d_model={model.d_model}")
    model.eval()

    sum_vecs = torch.zeros(model.dict_size, model.d_model, device=device)
    counts = torch.zeros(model.dict_size, device=device)

    for batch_idx, x in enumerate(tqdm(dataloader, desc="Profiling angular variance")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        _, acts = model(x)                              # acts: [B, dict_size]

        active_mask = (acts > 0).float()               # [B, dict_size]
        x_norm = F.normalize(x, p=2, dim=-1)           # [B, d_model]

        # [dict_size, B] @ [B, d_model] -> [dict_size, d_model]
        sum_vecs += active_mask.t() @ x_norm
        counts += active_mask.sum(dim=0)

    safe_counts = counts.unsqueeze(1).clamp(min=1e-6)
    mean_vecs = sum_vecs / safe_counts
    mean_norms = torch.norm(mean_vecs, dim=1)          # [dict_size]
    angular_variance = 1.0 - mean_norms                # [dict_size]

    sorted_indices = torch.argsort(angular_variance, descending=True)

    print("[AngularVar] Example stats:")
    print("  Top-5 high variance:", angular_variance[sorted_indices[:5]].tolist())
    print("  Top-5 low variance :", angular_variance[sorted_indices[-5:]].tolist())

    return sorted_indices, angular_variance
