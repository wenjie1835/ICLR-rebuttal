
import os, re, argparse, yaml, pandas as pd
from glob import glob
from typing import List, Dict
from tqdm import tqdm
try:
    from .utils import ensure_dir
except ImportError:
    from utils import ensure_dir

def scan_roots(roots: List[str]) -> List[str]:
    files = []
    for r in roots:
        for ext in ("*.safetensors", "*.pt", "*.bin", "*.ckpt"):
            files.extend(glob(os.path.join(r, "**", ext), recursive=True))
    return sorted(set(files))

def infer_meta_from_path(path: str) -> Dict:
    s = {"trainer": None, "step": None, "sparsity": None}
    m = re.search(r"trainer[_\-]?(\d+)", path);  s["trainer"] = int(m.group(1)) if m else None
    m = re.search(r"step[_\-]?(\d+)", path);     s["step"] = int(m.group(1)) if m else None
    m = re.search(r"(sparsity|l0|lambda)[_\-]?(\d+)", path, re.IGNORECASE)
    s["sparsity"] = int(m.group(2)) if m else None
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg["output_dir"]; ensure_dir(out_dir)

    rows = []
    # Support all variant keys: standard, topk, batch_topk, matryoshka_batch_topk, gated, jump_relu, etc.
    sae_ckpts = cfg.get("sae_checkpoints") or {}
    for variant_key, var_cfg in sae_ckpts.items():
        if not var_cfg or not isinstance(var_cfg, dict): continue
        name = var_cfg.get("name", variant_key)
        roots = var_cfg.get("roots")
        if not roots: continue
        if isinstance(roots, str):
            roots = [roots]
        files = scan_roots(roots)
        for fpath in tqdm(files, desc=f"scan_{variant_key}"):
            meta = infer_meta_from_path(fpath)
            rows.append({"variant": name, "path": fpath, **meta})
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(out_dir, "checkpoint_index.parquet"))
    print(f"[index] found {len(df)} files -> {out_dir}/checkpoint_index.parquet")

if __name__ == "__main__":
    main()
