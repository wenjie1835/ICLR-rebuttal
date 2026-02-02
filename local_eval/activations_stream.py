# local_eval/activations_stream.py
from typing import Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import DEVICE, DATASET_NAME, DATASET_SPLIT, MAX_TOKENS, MAX_SEQ_LEN, BATCH_SIZE, MODEL_CONFIGS


def build_tokenizer_and_model(model_name: str):
    cfg = MODEL_CONFIGS[model_name]
    hf_name = cfg["hf_name"]

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model, cfg["target_layer"]


def collect_resid_activations(
    model_name: str,
    max_tokens: int = MAX_TOKENS,
) -> np.ndarray:
    """
    流式加载 monology/pile-uncopyrighted，调用 HF 模型，
    收集指定层的 hidden_states，近似视作 resid_post。

    返回:
      X: [N_tokens, d_model] np.float32
    """
    tokenizer, model, target_layer = build_tokenizer_and_model(model_name)
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)

    X_list = []
    total_tokens = 0

    for example in ds:
        text = example["text"]
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=False,
        )["input_ids"][0]  # [L]

        # 切成多个长度 <= MAX_SEQ_LEN 的片段
        for start in range(0, len(tokens), MAX_SEQ_LEN):
            chunk = tokens[start : start + MAX_SEQ_LEN]
            if len(chunk) < 2:
                continue

            # 构造一个 batch_size=1 的输入，为了方便我们凑成 BATCH_SIZE 再前向
            # 这里简单地 accumulating
            # 为了逻辑简单，我们直接每 BATCH_SIZE 个 chunk 一起前向
            # 用一个局部缓冲
            pass

        # 为了避免写复杂的多层循环，简单一点：每个文本单独切 batch
        # 当 token 总数超过 max_tokens 时终止
        # —— 上面写了一半，直接用下一个实现覆盖
        break  # 先跳出，实际逻辑在下面实现

    # 重新写一版更直接的：用一个单独 generator 收集，避免上面的半成品逻辑
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
    total_tokens = 0
    batch_token_ids = []

    for example in ds:
        text = example["text"]
        token_ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]

        # 切 chunk
        for start in range(0, len(token_ids), MAX_SEQ_LEN):
            chunk = token_ids[start : start + MAX_SEQ_LEN]
            if len(chunk) < 2:
                continue
            batch_token_ids.append(chunk)

            if len(batch_token_ids) >= BATCH_SIZE:
                X_batch, n_tokens = _forward_collect_batch(batch_token_ids, tokenizer, model, target_layer)
                X_list.append(X_batch)
                total_tokens += n_tokens
                batch_token_ids = []

                if total_tokens >= max_tokens:
                    X = np.concatenate(X_list, axis=0)[:max_tokens]
                    return X.astype(np.float32, copy=False)

        if total_tokens >= max_tokens:
            break

    # 处理尾部 batch
    if batch_token_ids and total_tokens < max_tokens:
        X_batch, n_tokens = _forward_collect_batch(batch_token_ids, tokenizer, model, target_layer)
        X_list.append(X_batch)
        total_tokens += n_tokens

    if not X_list:
        raise RuntimeError("No activations collected from dataset.")

    X = np.concatenate(X_list, axis=0)[:max_tokens]
    return X.astype(np.float32, copy=False)


def _forward_collect_batch(
    batch_token_ids,
    tokenizer,
    model,
    target_layer: int,
) -> Tuple[np.ndarray, int]:
    """
    将若干 token 序列 padding 成一个 batch，前向，取指定层 hidden_states。
    返回:
      X_flat: [N_tokens_in_batch, d_model] np.ndarray
      n_tokens: int
    """
    # pad
    max_len = max(len(ids) for ids in batch_token_ids)
    padded = []
    attn_masks = []
    for ids in batch_token_ids:
        pad_len = max_len - len(ids)
        padded_ids = torch.cat(
            [ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)]
        )
        padded.append(padded_ids)
        mask = torch.ones_like(padded_ids)
        if pad_len > 0:
            mask[-pad_len:] = 0
        attn_masks.append(mask)

    input_ids = torch.stack(padded, dim=0).to(DEVICE)        # [B, L]
    attention_mask = torch.stack(attn_masks, dim=0).to(DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        # 近似 resid_post
        x_layer = hidden_states[target_layer]                # [B, L, D]
        valid_mask = attention_mask.bool()                   # [B, L]
        B, L, D = x_layer.shape
        x_flat = x_layer[valid_mask.view(B, L)].view(-1, D)  # [N_tokens, D]

    X_np = x_flat.detach().cpu().numpy()
    return X_np, X_np.shape[0]
