"""Real-robot Push-T IoU from the FRONT camera (occlusion-resistant).

Rationale: the top camera is heavily occluded by the gripper during pushing
(median visible block area ~25%), making top-down masks unreliable. The front
camera keeps the red T-block visible throughout (visible area 1185-2481 px of a
~2150 px block), so we compute IoU directly on the raw red-block mask with no
geometric occlusion recovery.

Goal mask: the placed T-block from BFN-20 episode 2 (a clean success), extracted
from its end_front.png -> goal_mask_front.png. This calibrates the metric so ep2
itself scores ~0.95 (mp4 compression vs the still PNG which scores 1.0).

Episode metrics:
  plateau_max_iou : max moving-average IoU after a warmup (primary; captures
                    reaching+holding alignment, robust to "push past goal" late decay).
  last_k_iou      : mean IoU over the last K frames (did it hold at the end).
  naive_max_iou, final_iou : references.
"""
import os, sys, glob, json
import cv2
import numpy as np

K = np.ones((3, 3), np.uint8)
LO1, HI1 = np.array([0, 70, 50]),   np.array([12, 255, 255])
LO2, HI2 = np.array([165, 70, 50]), np.array([180, 255, 255])


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, LO1, HI1) | cv2.inRange(hsv, LO2, HI2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        o = np.zeros_like(m)
        cv2.drawContours(o, [c], -1, 255, -1)
        return o
    return m


def video_frames(path):
    cap = cv2.VideoCapture(path)
    fs = []
    while True:
        r, f = cap.read()
        if not r:
            break
        fs.append(f)
    cap.release()
    return fs


def iou(a, b):
    a = a > 0; b = b > 0
    return np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)


def plateau_max(curve, warmup_ratio=0.2, window=5):
    c = np.asarray(curve, np.float32)
    if len(c) == 0:
        return 0.0
    c = c[int(len(c) * warmup_ratio):]
    if len(c) == 0:
        return 0.0
    if len(c) < window:
        return float(c.max())
    return float(np.convolve(c, np.ones(window) / window, "valid").max())


def last_k_mean(curve, k=10):
    c = np.asarray(curve, np.float32)
    return float(c[-k:].mean()) if len(c) else 0.0


def ci95(x):
    x = np.asarray(x, float)
    return 0.0 if len(x) < 2 else 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def method_of(run):
    n = os.path.basename(run)
    if "bfn" in n and "infer-20" in n: return "BFN-20"
    if "bfn" in n and "infer-10" in n: return "BFN-10"
    if "ddpm" in n: return "DDPM"
    if "ddim" in n: return "DDIM"
    if "consistency" in n and "infer-3" in n: return "Cons-3"
    if "consistency" in n and "infer-1" in n: return "Cons-1"
    return n


def main():
    goal = cv2.imread("goal_mask_front.png", cv2.IMREAD_GRAYSCALE)
    out = {}
    for run in sorted(glob.glob("eval_output/runs/*")):
        eps = sorted(glob.glob(f"{run}/episodes/episode_*"))
        if not eps:
            continue
        m = method_of(run)
        plat, lastk, nmax, fin = [], [], [], []
        for ep in eps:
            fs = video_frames(f"{ep}/video_front.mp4")
            if not fs:
                continue
            c = np.array([iou(red_mask(f), goal) for f in fs])
            plat.append(plateau_max(c)); lastk.append(last_k_mean(c, 10))
            nmax.append(float(c.max())); fin.append(float(c[-1]))
        out[m] = dict(n=len(plat),
                      plateau_mean=float(np.mean(plat)), plateau_ci=ci95(plat),
                      last10_mean=float(np.mean(lastk)), last10_ci=ci95(lastk),
                      naive_mean=float(np.mean(nmax)), naive_ci=ci95(nmax),
                      final_mean=float(np.mean(fin)), final_ci=ci95(fin),
                      plateau=plat, last10=lastk)
        print(f"{m:8s} n={len(plat):2d}  plateau={np.mean(plat):.3f}±{ci95(plat):.3f}  "
              f"last10={np.mean(lastk):.3f}±{ci95(lastk):.3f}  "
              f"naive_max={np.mean(nmax):.3f}±{ci95(nmax):.3f}  final={np.mean(fin):.3f}±{ci95(fin):.3f}", flush=True)
    json.dump(out, open("results/real_iou_front.json", "w"), indent=2)
    print("\nSaved results/real_iou_front.json")


if __name__ == "__main__":
    main()