import cv2
import numpy as np


def binarize_mask(mask):
    """
    Convert mask to binary uint8 0/255.
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return ((mask > 0).astype(np.uint8)) * 255


def compute_iou(mask_a, mask_b):
    """
    Compute IoU between two binary masks.
    """
    a = mask_a > 0
    b = mask_b > 0

    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    return inter / max(union, 1)


def get_largest_contour(mask, min_area=20):
    """
    Get largest contour from binary mask.
    """
    mask = binarize_mask(mask)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < min_area:
        return None

    return cnt


def get_mask_pose(mask):
    """
    Estimate 2D pose of mask:
    center = centroid
    angle = contour orientation from minAreaRect
    """
    cnt = get_largest_contour(mask)

    if cnt is None:
        return None

    moments = cv2.moments(cnt)

    if moments["m00"] == 0:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    rect = cv2.minAreaRect(cnt)
    angle = rect[-1]

    center = np.array([cx, cy], dtype=np.float32)

    return center, angle


def recover_full_mask(initial_full_mask, current_partial_mask):
    """
    Recover full T-block mask under top-down planar motion.

    Assumption:
    - Camera is top-down
    - T-block only has 2D translation and rotation
    - Object shape is rigid
    """
    initial_full_mask = binarize_mask(initial_full_mask)
    current_partial_mask = binarize_mask(current_partial_mask)

    init_pose = get_mask_pose(initial_full_mask)
    curr_pose = get_mask_pose(current_partial_mask)

    if init_pose is None or curr_pose is None:
        return None

    c0, angle0 = init_pose
    c1, angle1 = curr_pose

    delta_angle = angle1 - angle0

    H, W = current_partial_mask.shape[:2]

    transform = cv2.getRotationMatrix2D(
        center=tuple(c0),
        angle=delta_angle,
        scale=1.0
    )

    transform[:, 2] += c1 - c0

    recovered_mask = cv2.warpAffine(
        initial_full_mask,
        transform,
        (W, H),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )

    recovered_mask = binarize_mask(recovered_mask)

    return recovered_mask


def compute_iou_curve(initial_full_mask, partial_masks, goal_mask):
    """
    For each frame:
    partial mask -> recovered full mask -> IoU with goal mask
    """
    goal_mask = binarize_mask(goal_mask)

    iou_curve = []
    recovered_masks = []

    for partial_mask in partial_masks:
        recovered = recover_full_mask(
            initial_full_mask,
            partial_mask
        )

        if recovered is None:
            iou = 0.0
            recovered_masks.append(None)
        else:
            iou = compute_iou(recovered, goal_mask)
            recovered_masks.append(recovered)

        iou_curve.append(iou)

    return np.array(iou_curve), recovered_masks


def plateau_max_iou_from_curve(
    iou_curve,
    warmup_ratio=0.2,
    window_size=5
):
    """
    Plateau max IoU:
    1. Ignore early warmup frames
    2. Compute moving average IoU
    3. Take the maximum moving-average value
    """
    iou_curve = np.array(iou_curve, dtype=np.float32)

    if len(iou_curve) == 0:
        return 0.0

    start = int(len(iou_curve) * warmup_ratio)
    valid_iou = iou_curve[start:]

    if len(valid_iou) == 0:
        return 0.0

    if len(valid_iou) < window_size:
        return float(valid_iou.max())

    kernel = np.ones(window_size) / window_size

    moving_avg = np.convolve(
        valid_iou,
        kernel,
        mode="valid"
    )

    return float(moving_avg.max())


def evaluate_episode(
    initial_full_mask,
    partial_masks,
    goal_mask,
    warmup_ratio=0.2,
    window_size=5
):
    """
    Full episode evaluation.
    """
    iou_curve, recovered_masks = compute_iou_curve(
        initial_full_mask,
        partial_masks,
        goal_mask
    )

    plateau_score = plateau_max_iou_from_curve(
        iou_curve,
        warmup_ratio=warmup_ratio,
        window_size=window_size
    )

    final_iou = float(iou_curve[-1]) if len(iou_curve) > 0 else 0.0
    naive_max_iou = float(iou_curve.max()) if len(iou_curve) > 0 else 0.0

    results = {
        "plateau_max_iou": plateau_score,
        "final_iou": final_iou,
        "naive_max_iou": naive_max_iou,
        "iou_curve": iou_curve,
        "recovered_masks": recovered_masks,
    }

    return results
