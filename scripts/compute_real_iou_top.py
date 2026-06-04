"""Real-robot Push-T IoU from the TOP camera (geometrically correct planar coverage).

The top-down view measures true table-plane overlap (unlike the front view, whose
perspective makes image-space IoU ambiguous in depth). The gripper occludes the
block during pushing, so we DROP occluded frames (visible red area < VIS_FRAC of the
per-episode reference block area) and compute IoU only on visible frames. The block
is visible when it settles (between/after pushes), so plateau-max over the kept
frames captures reach-and-hold alignment.

Goal mask: goal_mask_ref.png (filled target T from the top view; calibrated so
BFN-20 ep2, a clean success, scores ~0.93).

Per episode (over KEPT/visible frames only):
  plateau_max_iou : max 5-frame moving-average IoU after 20% warmup (primary).
  last_k_iou      : mean over last 10 kept frames.
  end_iou         : IoU at the last kept (visible) frame.
  naive_max_iou   : max single-frame IoU.
  n_kept          : number of visible frames used.
"""
import os, sys, glob, json
import cv2
import numpy as np

K = np.ones((3, 3), np.uint8)
LO1, HI1 = np.array([0, 70, 50]),   np.array([12, 255, 255])
LO2, HI2 = np.array([165, 70, 50]), np.array([180, 255, 255])
VIS_FRAC = 0.80


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


def ci95(x):
    x = np.asarray(x, float)
    return 0.0 if len(x) < 2 else 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def kept_iou_curve(frames, ref_area, goal):
    """IoU on visible (non-occluded) frames only, in temporal order."""
    out = []
    for f in frames:
        pm = red_mask(f)
        if (pm > 0).sum() >= VIS_FRAC * ref_area:
            out.append(iou(pm, goal))
    return np.array(out, np.float32)


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
    goal = cv2.imread("goal_mask_ref.png", cv2.IMREAD_GRAYSCALE)
    out = {}
    for run in sorted(glob.glob("eval_output/runs/*")):
        eps = sorted(glob.glob(f"{run}/episodes/episode_*"))
        if not eps:
            continue
        m = method_of(run)
        plat, lastk, endf, nmax, nkept = [], [], [], [], []
        for ep in eps:
            start = cv2.imread(f"{ep}/start_top.png")
            frames = video_frames(f"{ep}/video_top.mp4")
            if start is None or not frames:
                continue
            ref_area = max((red_mask(start) > 0).sum(), 1)
            c = kept_iou_curve(frames, ref_area, goal)
            if len(c) == 0:
                plat.append(0.0); lastk.append(0.0); endf.append(0.0); nmax.append(0.0); nkept.append(0)
                continue
            plat.append(plateau_max(c))
            lastk.append(float(c[-10:].mean()))
            endf.append(float(c[-1]))
            nmax.append(float(c.max()))
            nkept.append(int(len(c)))
        out[m] = dict(n=len(plat),
                      plateau_mean=float(np.mean(plat)), plateau_ci=ci95(plat),
                      last10_mean=float(np.mean(lastk)), last10_ci=ci95(lastk),
                      end_mean=float(np.mean(endf)), end_ci=ci95(endf),
                      naive_mean=float(np.mean(nmax)), naive_ci=ci95(nmax),
                      mean_kept=float(np.mean(nkept)), min_kept=int(np.min(nkept)),
                      plateau=plat, last10=lastk, end=endf)
        print(f"{m:8s} n={len(plat):2d}  plateau={np.mean(plat):.3f}±{ci95(plat):.3f}  "
              f"last10={np.mean(lastk):.3f}±{ci95(lastk):.3f}  end={np.mean(endf):.3f}±{ci95(endf):.3f}  "
              f"kept(mean/min)={np.mean(nkept):.0f}/{np.min(nkept)}", flush=True)
    json.dump(out, open("results/real_iou_top.json", "w"), indent=2)
    print("\nSaved results/real_iou_top.json")


if __name__ == "__main__":
    main()