"""Real-robot Push-T IoU with tracked template-based occlusion recovery.

Per frame:
  - Segment the red T-block (HSV).
  - If the block is well-visible (area >= VIS_FRAC * ref_area): use the raw mask.
  - If occluded: recover the full block by rigidly aligning the known reference
    T (from the first frame) to the visible partial, searching rotation/translation
    in a window around the *previous* frame's pose (temporal tracking) and
    maximizing visible overlap (cv2.matchTemplate, TM_CCORR on binary = intersection).
IoU is computed against the filled green goal mask. Episode metrics:
  - plateau_max_iou : max moving-average IoU after a warmup, over a small window.
  - last_k_iou      : mean IoU over the last K frames (did the block hold at the end).
  - naive_max_iou, final_iou : references.
"""
import os, sys, glob, json
import cv2
import numpy as np

LO1, HI1 = np.array([0, 70, 50]),   np.array([12, 255, 255])
LO2, HI2 = np.array([165, 70, 50]), np.array([180, 255, 255])
K = np.ones((3, 3), np.uint8)
VIS_FRAC = 0.85          # >= this fraction of ref area => treat as fully visible
ANGLE_WIN = 25           # +/- degrees searched around previous angle
ANGLE_STEP = 2
TRANS_WIN = 40           # +/- px window around previous location for tracking


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, LO1, HI1) | cv2.inRange(hsv, LO2, HI2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, K)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, K)
    # keep largest blob only (drop stray red noise / fingertips)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        out = np.zeros_like(m)
        cv2.drawContours(out, [c], -1, 255, -1)
        return out
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


def centroid(m):
    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.mean(), ys.mean()], np.float32)


def iou(a, b):
    a = a > 0; b = b > 0
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / max(u, 1)


def align_full_T(ref_bin, partial_bin, angles, prev_loc=None):
    """Place the full reference T over the visible partial; return (full_mask, angle, loc)."""
    H, W = partial_bin.shape
    pf = (partial_bin > 0).astype(np.float32)
    if pf.sum() < 15:
        return None, None, None
    rc = centroid(ref_bin)
    best = -1; out = (None, None, None)
    for a in angles:
        M = cv2.getRotationMatrix2D(tuple(rc), float(a), 1.0)
        rot = cv2.warpAffine(ref_bin, M, (W, H), flags=cv2.INTER_NEAREST)
        ys, xs = np.where(rot > 0)
        if len(xs) == 0:
            continue
        tmpl = (rot[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 0).astype(np.float32)
        th, tw = tmpl.shape
        if th >= H or tw >= W:
            continue
        res = cv2.matchTemplate(pf, tmpl, cv2.TM_CCORR)
        if prev_loc is not None:  # restrict to a window around previous top-left
            mask = np.full(res.shape, -1.0, np.float32)
            px, py = prev_loc
            x0 = max(0, px - TRANS_WIN); x1 = min(res.shape[1], px + TRANS_WIN)
            y0 = max(0, py - TRANS_WIN); y1 = min(res.shape[0], py + TRANS_WIN)
            mask[y0:y1, x0:x1] = res[y0:y1, x0:x1]
            res = mask
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best:
            best = mx
            full = np.zeros((H, W), np.uint8)
            x0, y0 = loc
            full[y0:y0 + th, x0:x0 + tw][tmpl > 0.5] = 255
            out = (full, a, loc)
    return out


def episode_iou_curve(frames, ref_mask, goal, return_masks=False):
    ref_area = max((ref_mask > 0).sum(), 1)
    prev_a, prev_loc = 0.0, None
    curve = []; recs = []
    for i, fr in enumerate(frames):
        pm = red_mask(fr)
        area = (pm > 0).sum()
        if area >= VIS_FRAC * ref_area:
            # fully visible: trust the raw mask, but refresh tracking state
            curve.append(iou(pm, goal))
            recs.append(pm if return_masks else None)
            ca = centroid(pm)
            # leave prev_a as-is (raw gives no clean angle); keep loc near centroid
            continue
        angs = (range(int(prev_a) - ANGLE_WIN, int(prev_a) + ANGLE_WIN + 1, ANGLE_STEP)
                if i > 0 else range(-90, 91, 3))
        warp, a, loc = align_full_T(ref_mask, pm, angs, prev_loc if i > 0 else None)
        if warp is None:
            curve.append(0.0); recs.append(None); continue
        prev_a, prev_loc = a, loc
        curve.append(iou(warp, goal))
        recs.append(warp if return_masks else None)
    return (np.array(curve), recs) if return_masks else np.array(curve)


def plateau_max(curve, warmup_ratio=0.2, window=5):
    c = np.asarray(curve, np.float32)
    if len(c) == 0:
        return 0.0
    c = c[int(len(c) * warmup_ratio):]
    if len(c) == 0:
        return 0.0
    if len(c) < window:
        return float(c.max())
    ker = np.ones(window) / window
    return float(np.convolve(c, ker, "valid").max())


def last_k_mean(curve, k=10):
    c = np.asarray(curve, np.float32)
    return float(c[-k:].mean()) if len(c) else 0.0


def method_of(run):
    n = os.path.basename(run)
    if "bfn" in n and "infer-20" in n: return "BFN-20"
    if "bfn" in n and "infer-10" in n: return "BFN-10"
    if "ddpm" in n: return "DDPM"
    if "ddim" in n: return "DDIM"
    if "consistency" in n and "infer-3" in n: return "Cons-3"
    if "consistency" in n and "infer-1" in n: return "Cons-1"
    return n


def ci95(x):
    x = np.asarray(x, float)
    return 0.0 if len(x) < 2 else 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    goal = cv2.imread("goal_mask_ref.png", cv2.IMREAD_GRAYSCALE)
    out = {}
    for run in sorted(glob.glob("eval_output/runs/*")):
        eps = sorted(glob.glob(f"{run}/episodes/episode_*"))
        if not eps:
            continue
        m = method_of(run)
        plat, lastk, nmax, fin = [], [], [], []
        for ep in eps:
            start = cv2.imread(f"{ep}/start_top.png")
            frames = video_frames(f"{ep}/video_top.mp4")
            if start is None or not frames:
                continue
            ref = red_mask(start)
            curve = episode_iou_curve(frames, ref, goal)
            plat.append(plateau_max(curve)); lastk.append(last_k_mean(curve))
            nmax.append(float(curve.max())); fin.append(float(curve[-1]))
        out[m] = dict(n=len(plat),
                      plateau_mean=float(np.mean(plat)), plateau_ci=ci95(plat),
                      lastk_mean=float(np.mean(lastk)), lastk_ci=ci95(lastk),
                      naive_mean=float(np.mean(nmax)), naive_ci=ci95(nmax),
                      final_mean=float(np.mean(fin)), final_ci=ci95(fin),
                      plateau=plat, lastk=lastk)
        print(f"{m:8s} n={len(plat):2d}  plateau={np.mean(plat):.3f}±{ci95(plat):.3f}  "
              f"last10={np.mean(lastk):.3f}±{ci95(lastk):.3f}  "
              f"naive_max={np.mean(nmax):.3f}±{ci95(nmax):.3f}  final={np.mean(fin):.3f}±{ci95(fin):.3f}",
              flush=True)
    json.dump(out, open("results/real_iou_tracked.json", "w"), indent=2)
    print("\nSaved results/real_iou_tracked.json")


if __name__ == "__main__":
    main()
