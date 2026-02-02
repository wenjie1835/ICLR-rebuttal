import os
import json
import torch

from .config import SAEConfig


def main():
    # Trained Loca-SAE output dir (e.g. after python -m loca_sae.train_pythia160m)
    sae_dir = "loc-amt-sae/pythia-160m"

    # 1. 读 config.json，拿到结构参数
    cfg_path = os.path.join(sae_dir, "config.json")
    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)
    cfg = SAEConfig(**cfg_dict)

    n_sub = cfg.n_sub_encoders
    d_model = cfg.d_model

    print(f"[Info] n_sub_encoders={n_sub}, d_model={d_model}")

    # 2. 读 multi_encoder_sae.pt
    ckpt_path = os.path.join(sae_dir, "multi_encoder_sae.pt")
    print(f"[Info] Loading multi-encoder checkpoint from {ckpt_path}")

    state = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,   # ✅ PyTorch 2.6 以后必须显式关掉这个
    )

    # 简单 sanity check
    keys_preview = list(state.keys())[:10]
    print(f"[Info] First 10 keys in state_dict: {keys_preview}")

    # 3. 从 8 个 encoder/decoder 拼出 W_enc / W_dec
    enc_blocks = []
    dec_blocks = []
    for i in range(n_sub):
        enc_key = f"encoders.{i}.weight"
        dec_key = f"decoders.{i}.weight"

        if enc_key not in state or dec_key not in state:
            raise KeyError(f"Missing key {enc_key} or {dec_key} in checkpoint.")

        enc_w = state[enc_key]   # [2048, d_model]
        dec_w = state[dec_key]   # [d_model, 2048]

        enc_blocks.append(enc_w)
        dec_blocks.append(dec_w)

    W_enc = torch.cat(enc_blocks, dim=0)  # [16384, d_model]
    W_dec = torch.cat(dec_blocks, dim=1)  # [d_model, 16384]

    print("[Info] W_enc shape:", tuple(W_enc.shape))
    print("[Info] W_dec shape:", tuple(W_dec.shape))

    # 4. 存成 SAEBench 风格的 checkpoint
    out_path = os.path.join(sae_dir, "ae.pt")
    torch.save({"W_enc": W_enc, "W_dec": W_dec}, out_path)
    print(f"[Done] Saved SAEBench-style checkpoint to {out_path}")


if __name__ == "__main__":
    main()
