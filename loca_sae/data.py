# data.py
from typing import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class LMActivationConfig:
    model_name: str
    target_layer: int
    d_model: int
    dataset_name: str
    dataset_split: str
    max_tokens: int
    max_seq_len: int
    device: str
    lm_dtype: str  # "bfloat16" or "float16"
    normalize_activations: bool  # ✅ 是否对 resid_post 做 L2 归一化


class ResidualStreamDataset(IterableDataset):
    """
    通过 HF streaming + LM 前向，在线产生 resid_post token 向量。

    每次 __iter__:
        - 重新从数据集开头流式读取
        - 截断到 max_tokens 个有效 token
    """

    def __init__(self, cfg: LMActivationConfig):
        super().__init__()
        self.cfg = cfg

        print(f"[LM] Loading model {cfg.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        torch_dtype = torch.bfloat16 if cfg.lm_dtype == "bfloat16" else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch_dtype,
        ).to(cfg.device)
        self.model.eval()

    def _activation_iterator(self) -> Iterator[torch.Tensor]:
        """
        内部生成器：yield 单个 token 的 resid_post 向量 [d_model]
        """
        ds_iter = iter(
            load_dataset(
                self.cfg.dataset_name,
                split=self.cfg.dataset_split,
                streaming=True,
            )
        )

        tokens_seen = 0
        for sample in ds_iter:
            if tokens_seen >= self.cfg.max_tokens:
                break

            text = sample.get("text", None)

            # 防御：跳过非字符串 / 空字符串 / 纯空白
            if not isinstance(text, str):
                continue
            if len(text.strip()) == 0:
                continue

            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.max_seq_len,
            )

            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            # 防御：编码结果长度为 0 的样本直接跳过
            if input_ids.numel() == 0 or attention_mask.sum().item() == 0:
                continue

            input_ids = input_ids.to(self.cfg.device)
            attention_mask = attention_mask.to(self.cfg.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states  # tuple(len = n_layers+1)

            # resid_post 近似为 hidden_states[target_layer]
            acts = hidden_states[self.cfg.target_layer]  # [B, L, d_model]
            mask = attention_mask.bool()                # [B, L]
            acts = acts[mask]                           # [N_tokens, d_model]

            # 防御：极端情况下，mask 后可能没有 token
            if acts.numel() == 0:
                continue

            # ✅ 关键：按 token 对 resid_post 做 L2 归一化
            if self.cfg.normalize_activations:
                acts = F.normalize(acts, p=2, dim=-1)

            for vec in acts:
                if tokens_seen >= self.cfg.max_tokens:
                    break
                # 统一用 float32 训练 SAE，数值更稳定
                yield vec.detach().to("cpu", dtype=torch.float32)
                tokens_seen += 1

            if tokens_seen >= self.cfg.max_tokens:
                break

    def __iter__(self):
        # DataLoader 每个 epoch 调一次 __iter__，每次都重新流式遍历前 max_tokens 个 token。
        return self._activation_iterator()


def create_dataloader(cfg_lm: LMActivationConfig,
                      batch_size: int) -> DataLoader:
    dataset = ResidualStreamDataset(cfg_lm)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,      # 避免多进程重复加载大模型
        pin_memory=True,
    )
    return loader
