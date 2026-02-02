# train_gemma2_2b.py
from .config import SAEConfig
from .train_sae import run_pipeline


def main():
    cfg = SAEConfig(
        model_name="google/gemma-2-2b",
        d_model=2304,                   # Gemma-2-2B hidden size
        target_layer=12,                # resid_post of layer 12 ~ hidden_states[12]
        output_dir="loc-amt-sae/gemma-2-2b/",
        lm_dtype="bfloat16",            # Gemma-2 默认使用 bf16
        max_seq_len=1024,               # Gemma 支持更长上下文，这里取 1024
        max_tokens=10_000_000,
        lr=3e-4,
        phase1_steps=50_000,
        phase4_steps=50_000,
        batch_size=1024,                # Gemma 更大，可以适当减小 batch
        normalize_activations=True,
        target_loss=0.001,
    )

    run_pipeline(cfg)


if __name__ == "__main__":
    main()