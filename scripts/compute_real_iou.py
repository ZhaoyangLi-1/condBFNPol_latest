"""Compute real-robot Push-T IoU metrics (plateau-max IoU per channel proposal).

For each method run / episode:
  - initial_full_mask = red-T segmentation of start_top.png
  - partial_masks     = red-T segmentation of each video_top.mp4 frame
  - goal_mask         = goal_mask_ref.png (filled green target T, 945 px)
Occluded frames are handled by recover_full_mask (warp the full start mask onto
the visible partial blob's pose). Episode metric = plateau_max_iou (primary),
with final_iou and naive_max_iou as secondaries.
"""
import os, sys, glob, json
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from compute_iou import evaluate_episode

# Red-T HSV segmentation (red wraps hue 0/180)
LO1, HI1 = np.array([0, 80, 60]),   np.array([10, 255, 255])
LO2, HI2 = np.array([170, 80, 60]), np.array([180, 255, 255])
K = np.ones((3, 3), np.uint8)


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, LO1, HI1) | cv2.inRange(hsv, LO2, HI2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K)
    return m


def video_masks(path):
    cap = cv2.VideoCapture(path)
    masks = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        masks.append(red_mask(fr))
    cap.release()
    return masks


def ci95(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


# method label from run dir name
def method_of(run):
    name = os.path.basename(run)
    if "bfn" in name and "infer-20" in name:  return "BFN-20"
    if "bfn" in name and "infer-10" in name:  return "BFN-10"
    if "ddpm" in name:                         return "DDPM"
    if "ddim" in name:                         return "DDIM"
    if "consistency" in name and "infer-3" in name: return "Cons-3"
    if "consistency" in name and "infer-1" in name: return "Cons-1"
    return name


def main():
    root = "eval_output/runs"
    goal = cv2.imread("goal_mask_ref.png", cv2.IMREAD_GRAYSCALE)
    runs = sorted(glob.glob(f"{root}/*"))
    all_results = {}
    for run in runs:
        eps = sorted(glob.glob(f"{run}/episodes/episode_*"))
        if not eps:
            continue
        m = method_of(run)
        per_ep = []
        for ep in eps:
            start = cv2.imread(f"{ep}/start_top.png")
            if start is None:
                continue
            init_mask = red_mask(start)
            partials = video_masks(f"{ep}/video_top.mp4")
            if not partials:
                continue
            r = evaluate_episode(init_mask, partials, goal,
                                 warmup_ratio=0.2, window_size=5)
            per_ep.append({k: r[k] for k in ("plateau_max_iou", "final_iou", "naive_max_iou")})
        if not per_ep:
            continue
        plat = [e["plateau_max_iou"] for e in per_ep]
        fin = [e["final_iou"] for e in per_ep]
        nmax = [e["naive_max_iou"] for e in per_ep]
        all_results[m] = {
            "n": len(per_ep),
            "plateau_mean": float(np.mean(plat)), "plateau_ci": float(ci95(plat)),
            "final_mean": float(np.mean(fin)), "final_ci": float(ci95(fin)),
            "naive_mean": float(np.mean(nmax)), "naive_ci": float(ci95(nmax)),
            "per_episode": per_ep,
        }
        print(f"{m:8s} (n={len(per_ep):2d}): plateau={np.mean(plat):.3f}±{ci95(plat):.3f}  "
              f"final={np.mean(fin):.3f}±{ci95(fin):.3f}  naive_max={np.mean(nmax):.3f}±{ci95(nmax):.3f}",
              flush=True)

    with open("results/real_iou_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved results/real_iou_metrics.json")


if __name__ == "__main__":
    main()
