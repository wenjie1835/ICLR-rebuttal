# train_pythia160m.py
from .config import SAEConfig
from .train_sae import run_pipeline


def main():
    cfg = SAEConfig(
        model_name="EleutherAI/pythia-160m-deduped",
        d_model=768,
        target_layer=8,
        output_dir="loc-amt-sae/pythia-160m/",
        lm_dtype="bfloat16",
        max_seq_len=128,
        max_tokens=10_000_000,
        lr=3e-4,
        phase1_steps=50_000,
        phase4_steps=50_000,
        batch_size=2048,
        normalize_activations=True,
#        target_loss=0.001,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()


