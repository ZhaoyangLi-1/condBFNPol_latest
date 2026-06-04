"""Build a self-contained inference bundle for the real-robot PushT-xarm
policies (BFN-hybrid + DDPM-onehot) and push to HuggingFace Hub.

Usage:
    python scripts/push_pusht_xarm_to_hf.py \\
        --bfn_run outputs/2026.05.11/21.21.05_train_bfn_pusht_xarm_top \\
        --ddpm_run outputs/2026.05.12/08.29.06_train_ddpm_pusht_xarm_top \\
        --repo borueihuang/bfn_pusht_xarm_top \\
        [--private] [--dry_run]
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


INFERENCE_PY = r'''"""Standalone real-robot inference for PushT-xarm BFN-hybrid policy.

Inputs each step:
    cam0: HxWx3 uint8 RGB image from the top camera (any size; will be resized to 224x224)
    cam0_prev: same, one step earlier (n_obs_steps=2)

Output per step (predicts horizon=16, returns next n_action_steps=8):
    direction: int in {0..7}
    distance:  float in [0, 50]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from bfn_hybrid_image_policy import BFNHybridImagePolicy  # noqa: E402


def load_bfn_policy(ckpt_path: str, config_path: str, device: str = "cuda"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    pcfg = cfg["policy"]
    policy = BFNHybridImagePolicy(
        shape_meta=cfg["shape_meta"],
        horizon=cfg["horizon"],
        n_action_steps=cfg["n_action_steps"],
        n_obs_steps=cfg["n_obs_steps"],
        num_discrete_actions=pcfg.get("num_discrete_actions", 8),
        continuous_param_dim=pcfg.get("continuous_param_dim", 1),
        sigma_1=pcfg.get("sigma_1", 0.001),
        beta_1=pcfg.get("beta_1", 0.2),
        n_timesteps=pcfg.get("n_timesteps", 20),
        crop_shape=tuple(pcfg.get("crop_shape", [216, 216])),
        obs_encoder_group_norm=pcfg.get("obs_encoder_group_norm", True),
        eval_fixed_crop=pcfg.get("eval_fixed_crop", True),
        diffusion_step_embed_dim=pcfg.get("diffusion_step_embed_dim", 128),
        down_dims=tuple(pcfg.get("down_dims", [256, 512, 1024])),
        kernel_size=pcfg.get("kernel_size", 5),
        n_groups=pcfg.get("n_groups", 8),
        cond_predict_scale=pcfg.get("cond_predict_scale", True),
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["state_dicts"]["model"] if "state_dicts" in ckpt else ckpt
    policy.load_state_dict(state)
    policy.to(device).eval()
    return policy


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Resize HxWx3 uint8 -> 3x224x224 float32 in [0,1]."""
    if img.shape[:2] != (224, 224):
        img = np.array(Image.fromarray(img).resize((224, 224), Image.BILINEAR))
    img = img.astype(np.float32) / 255.0
    return img.transpose(2, 0, 1)


def infer_step(policy, cam0_now: np.ndarray, cam0_prev: np.ndarray, device: str = "cuda"):
    """One inference call. Returns a list of dicts [{direction, distance}, ...] of length n_action_steps."""
    a = preprocess_image(cam0_prev)
    b = preprocess_image(cam0_now)
    obs = torch.from_numpy(np.stack([a, b])).unsqueeze(0).to(device)  # [1, 2, 3, 224, 224]
    with torch.no_grad():
        out = policy.predict_action({"camera_0": obs})
    actions = out["action"][0].cpu().numpy()  # [n_action_steps, 2] = [direction, distance]
    return [
        {"direction": int(round(a[0])) % 8, "distance": float(np.clip(a[1], 0, 50))}
        for a in actions
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    print(f"Loading policy from {args.ckpt}...")
    policy = load_bfn_policy(args.ckpt, args.config, args.device)
    print("Policy loaded.")

    # Dummy roundtrip test
    dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    t0 = time.time()
    actions = infer_step(policy, dummy, dummy, args.device)
    dt = (time.time() - t0) * 1000
    print(f"Smoke test: {len(actions)} actions in {dt:.1f} ms")
    print(f"First action: {actions[0]}")


if __name__ == "__main__":
    main()
'''


README_MD = r'''---
license: mit
tags:
- robotics
- bfn
- bayesian-flow-networks
- diffusion-policy
- pusht
- xarm
- hybrid-action
---

# PushT-xarm Real-Robot Policies (BFN-Hybrid vs DDPM-OneHot)

Real-robot push-T policies trained on [borueihuang/pusht_xarm_merged](https://huggingface.co/datasets/borueihuang/pusht_xarm_merged).
Both policies were trained on 144 episodes / 7835 frames at 30 Hz, top camera only, 200 epochs.

## Action space

- **Discrete:** 8 push directions (`action.direction` in {0..7})
- **Continuous:** push distance (`action.distance` in `[0, 50]`)

## Observation space

- `camera_0`: top-view RGB image, 3x224x224, two-step history (`n_obs_steps=2`)

## Policies (4 checkpoints)

| Folder | Method | Dataset | Action treatment | Inference steps |
|--------|--------|---------|------------------|-----------------|
| `bfn_merged/` | **BFN-Hybrid** | merged (7835 frames) | true hybrid | 20 |
| `bfn_orig/` | **BFN-Hybrid** | original (9229 frames) | true hybrid | 20 |
| `ddpm_merged/` | **DDPM** | merged | one-hot continuous (9D) | 100 |
| `ddpm_orig/` | **DDPM** | original | one-hot continuous (9D) | 100 |

The **merged** dataset combines consecutive same-direction segments into single actions.
The **original** dataset keeps the operator's raw per-step input.

## Quick start

```bash
pip install -r requirements.txt
pip install -e ./diffusion-policy    # required for DDPM ckpts only

# Validate the model on training frames (should match ground-truth direction)
python sanity_check.py --ckpt bfn_merged/latest.ckpt --config bfn_merged/policy_config.yaml

# Run inference
python inference.py --ckpt bfn_merged/latest.ckpt --config bfn_merged/policy_config.yaml
```

## Sanity check

Before integrating with the real robot, run `sanity_check.py`. It loads 8 known
training frames + ground-truth directions, runs inference, and prints whether
predictions match. If they do, the model + image pipeline are fine on this machine
— any robot-side miss is then a capture-side bug (most commonly **BGR vs RGB**:
OpenCV gives BGR but the policy was trained on RGB).

The script also saves `sample_train_frame.png` — compare it side-by-side with what
your robot camera captures to catch any view/color/resolution mismatch.

Programmatic use:

```python
from inference import load_bfn_policy, infer_step

policy = load_bfn_policy("bfn_merged/latest.ckpt", "bfn_merged/policy_config.yaml", "cuda")
actions = infer_step(policy, cam0_now, cam0_prev, "cuda")
# actions: List[{"direction": int 0..7, "distance": float 0..50}], len = n_action_steps (8)
```

## DDPM checkpoints

DDPM uses `diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy`
from the `diffusion-policy` library. Action is a 9D continuous vector: `[one_hot(8), distance]`.
At inference time, take `argmax` of the first 8 dims for the direction, and the 9th dim for distance.

## Files

```
bfn_merged/
  latest.ckpt
  policy_config.yaml
bfn_orig/
  latest.ckpt
  policy_config.yaml
ddpm_merged/
  latest.ckpt
  policy_config.yaml
ddpm_orig/
  latest.ckpt
  policy_config.yaml
bfn_hybrid_image_policy.py   # standalone BFN policy class
policies/base.py             # BasePolicy abstract class
networks/base.py             # BFNetwork wrapper
diffusion-policy/            # patched diffusion-policy library (pip install -e .)
inference.py                 # example loader + inference
sanity_check.py              # validates model against 8 training frames
sanity_samples.npz           # 8 (prev, curr) frame pairs + ground-truth actions
requirements.txt
```
'''

REQUIREMENTS_TXT = r'''torch>=2.0
torchvision
diffusers>=0.18
einops
numpy
zarr
pyyaml
pillow
hydra-core
robomimic
# The patched diffusion-policy library used to train these ckpts is bundled in
# `diffusion-policy/`. Install it from the bundle:
#   pip install -e ./diffusion-policy
# (Required for the DDPM ckpts. The BFN ckpts work without it.)
'''


SANITY_CHECK_PY = r'''"""Sanity-check the model: run inference on a few training frames and compare to ground truth.

If predictions match (or are close to) ground truth actions, the model + ckpt + image
pipeline are fine, and the bug in your robot rollout is on the robot side (most commonly
RGB vs BGR, image scale, or wrong camera view).

If predictions are random/chaotic, something is wrong with the model loading or
the dataset-side image format here.

Usage:
    python sanity_check.py --ckpt bfn_merged/latest.ckpt --config bfn_merged/policy_config.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from inference import load_bfn_policy, infer_step  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--samples", default="sanity_samples.npz")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    samples_path = THIS_DIR / args.samples
    if not samples_path.exists():
        raise SystemExit(f"sanity_samples.npz not found at {samples_path}")

    data = np.load(samples_path)
    frames = data["frames"]            # [N, 2, 224, 224, 3] uint8 (prev, now)
    gt_directions = data["directions"]  # [N] int
    gt_distances = data["distances"]    # [N] float

    print(f"Loading policy from {args.ckpt}...")
    policy = load_bfn_policy(args.ckpt, args.config, args.device)
    print(f"Loaded. Running {len(frames)} sanity samples.\n")

    # Save first sample as PNG for visual comparison
    Image.fromarray(frames[0, 1]).save(THIS_DIR / "sample_train_frame.png")
    print(f"Saved sample_train_frame.png — compare this with what your robot camera sees.\n")

    n_match = 0
    for i, (frame_pair, dir_gt, dist_gt) in enumerate(zip(frames, gt_directions, gt_distances)):
        cam0_prev, cam0_now = frame_pair[0], frame_pair[1]
        actions = infer_step(policy, cam0_now, cam0_prev, args.device)
        a = actions[0]  # first predicted action of the 8-step plan
        match_dir = (a["direction"] == int(dir_gt))
        if match_dir:
            n_match += 1
        print(
            f"  [{i}] pred dir={a['direction']}, dist={a['distance']:5.1f}  | "
            f"truth dir={int(dir_gt)}, dist={float(dist_gt):5.1f}  | "
            f"{'OK' if match_dir else 'MISMATCH'}"
        )

    print(f"\nDirection match: {n_match}/{len(frames)} = {100.0*n_match/len(frames):.0f}%")
    if n_match >= 0.6 * len(frames):
        print("Model loading + image pipeline look fine on this machine.")
        print("If robot rollout is chaotic, check on robot side:")
        print("  - cv2 frames are BGR; convert to RGB before policy")
        print("  - image dtype: uint8 in [0,255], NOT pre-normalized float")
        print("  - resolution: pass 224x224 in same view as training")
    else:
        print("WARNING: Predictions don't match training labels.")
        print("Check that ckpt and config came from the same training run.")


if __name__ == "__main__":
    main()
'''


def _build_sanity_samples(merged_zarr: Path, dst: Path, n_samples: int = 8):
    """Pick a few (prev_frame, curr_frame, direction, distance) tuples from the merged zarr."""
    import numpy as np
    import zarr
    if not merged_zarr.exists():
        print(f"WARNING: merged zarr not found at {merged_zarr}; skipping sanity samples")
        return
    r = zarr.open(str(merged_zarr), mode="r")
    episode_ends = r["meta/episode_ends"][:]

    # Pick frames from a handful of episodes, mid-episode (skip first 2 of each)
    rng = np.random.default_rng(0)
    pick_idxs = []
    starts = np.concatenate([[0], episode_ends[:-1]])
    for ep_idx in rng.choice(len(episode_ends), size=n_samples, replace=False):
        s, e = starts[ep_idx], episode_ends[ep_idx]
        if e - s < 3:
            continue
        t = rng.integers(s + 2, e)
        pick_idxs.append(int(t))

    frames = np.zeros((len(pick_idxs), 2, 224, 224, 3), dtype=np.uint8)
    directions = np.zeros(len(pick_idxs), dtype=np.int64)
    distances = np.zeros(len(pick_idxs), dtype=np.float32)

    cam0 = r["data/camera_0"]
    dirs = r["data/action_direction"]
    dists = r["data/action_distance"]

    for i, t in enumerate(pick_idxs):
        frames[i, 0] = cam0[t - 1]
        frames[i, 1] = cam0[t]
        directions[i] = dirs[t]
        distances[i] = dists[t]

    np.savez_compressed(dst, frames=frames, directions=directions, distances=distances)
    print(f"Saved sanity samples ({len(pick_idxs)} pairs): {dst}")


def build_bundle(runs: dict, dst: Path):
    """runs: dict mapping subdir-name -> Path(hydra run dir). Each gets its own subfolder."""
    dst.mkdir(parents=True, exist_ok=True)

    import torch
    for name, run_dir in runs.items():
        sub = dst / name
        sub.mkdir(exist_ok=True)
        src_ckpt = run_dir / "checkpoints" / "latest.ckpt"
        ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
        slim = {"state_dicts": {"model": ckpt["state_dicts"]["model"]}}
        if "ema_model" in ckpt["state_dicts"]:
            slim["state_dicts"]["ema_model"] = ckpt["state_dicts"]["ema_model"]
        if "cfg" in ckpt:
            slim["cfg"] = ckpt["cfg"]
        torch.save(slim, sub / "latest.ckpt")
        cfg_src = run_dir / ".hydra" / "config.yaml"
        if cfg_src.exists():
            shutil.copy(cfg_src, sub / "policy_config.yaml")
        else:
            print(f"WARNING: no hydra config at {cfg_src}")

    # Copy source files
    shutil.copy(REPO_ROOT / "policies" / "bfn_hybrid_image_policy.py", dst / "bfn_hybrid_image_policy.py")
    (dst / "policies").mkdir(exist_ok=True)
    shutil.copy(REPO_ROOT / "policies" / "base.py", dst / "policies" / "base.py")
    (dst / "policies" / "__init__.py").write_text("")
    (dst / "networks").mkdir(exist_ok=True)
    shutil.copy(REPO_ROOT / "networks" / "base.py", dst / "networks" / "base.py")
    (dst / "networks" / "__init__.py").write_text("")
    (dst / "utils").mkdir(exist_ok=True)
    shutil.copy(REPO_ROOT / "utils" / "bfn_utils.py", dst / "utils" / "bfn_utils.py")
    (dst / "utils" / "__init__.py").write_text("")

    # Bundle the patched diffusion-policy library (needed for DDPM ckpts + matches training)
    dp_src = REPO_ROOT / "src" / "diffusion-policy"
    dp_dst = dst / "diffusion-policy"
    if dp_src.exists():
        shutil.copytree(
            dp_src,
            dp_dst,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".git*", "data", "outputs", "wandb",
                "*.egg-info", ".pytest_cache", "tests",
            ),
        )

    # Sanity samples + script
    if "bfn_merged" in runs:
        merged_zarr = REPO_ROOT / "data" / "pusht_xarm_merged" / "replay.zarr"
        _build_sanity_samples(merged_zarr, dst / "sanity_samples.npz", n_samples=8)
        (dst / "sanity_check.py").write_text(SANITY_CHECK_PY)

    (dst / "inference.py").write_text(INFERENCE_PY)
    (dst / "README.md").write_text(README_MD)
    (dst / "requirements.txt").write_text(REQUIREMENTS_TXT)

    print(f"Bundle built at: {dst}")
    print("Contents:")
    for p in sorted(dst.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(dst)}  ({p.stat().st_size / 1e6:.1f} MB)")


def push_to_hf(bundle: Path, repo: str, private: bool, dry_run: bool):
    if dry_run:
        print(f"[dry run] would push {bundle} -> {repo} (private={private})")
        return
    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(bundle),
        repo_id=repo,
        repo_type="model",
        commit_message="Initial upload of BFN-hybrid + DDPM PushT-xarm policies",
    )
    print(f"Uploaded to https://huggingface.co/{repo}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bfn_run", help="Hydra run dir for BFN training (merged dataset)")
    p.add_argument("--ddpm_run", help="Hydra run dir for DDPM training (merged dataset)")
    p.add_argument("--bfn_orig_run", help="Hydra run dir for BFN training (orig dataset)")
    p.add_argument("--ddpm_orig_run", help="Hydra run dir for DDPM training (orig dataset)")
    p.add_argument("--repo", required=True, help="HF repo id e.g. user/bfn_pusht_xarm_top")
    p.add_argument("--bundle_dir", default="data/pusht_xarm_bundle")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Build bundle but don't push")
    args = p.parse_args()

    runs = {}
    if args.bfn_run:
        runs["bfn_merged"] = Path(args.bfn_run)
    if args.ddpm_run:
        runs["ddpm_merged"] = Path(args.ddpm_run)
    if args.bfn_orig_run:
        runs["bfn_orig"] = Path(args.bfn_orig_run)
    if args.ddpm_orig_run:
        runs["ddpm_orig"] = Path(args.ddpm_orig_run)
    if not runs:
        raise SystemExit("provide at least one --*_run argument")

    bundle = REPO_ROOT / args.bundle_dir
    if bundle.exists():
        shutil.rmtree(bundle)
    build_bundle(runs, bundle)
    push_to_hf(bundle, args.repo, args.private, args.dry_run)


if __name__ == "__main__":
    main()
