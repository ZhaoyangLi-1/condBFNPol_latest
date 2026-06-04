"""Roll out BFN / DDPM / DDIM / Consistency on a PAMDP env with the same seed,
record per-step RGB frames, and stitch into a 2x2 grid mp4.

Usage:
  python scripts/render_rollouts.py --env hard_move_n4 --n_actuators 4 --seed 7 \
      --bfn_ckpt /path/bfn.ckpt --ddpm_ckpt /path/ddpm.ckpt \
      --consistency_ckpt /path/cp.ckpt --out videos/hard_move_n4.mp4

DDIM reuses the DDPM checkpoint with the DDIM sampler at inference time.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

PANEL_LABELS = ["BFN-20", "DDIM-10", "DDPM-100", "Cons-1"]

_FIG_CACHE = {}


def make_env(env_name, n_actuators=0):
    if env_name.startswith("hard_move"):
        from environments.hard_move import HardMoveEnv
        return HardMoveEnv(n_actuators=n_actuators)
    if env_name == "catch_point":
        from environments.catch_point import CatchPointEnv
        return CatchPointEnv()
    if env_name == "platform":
        from environments.platform_env import PlatformEnv
        return PlatformEnv()
    if env_name == "goal":
        from environments.goal_env import GoalEnv
        return GoalEnv()
    if env_name == "hard_goal":
        from environments.hard_goal import HardGoalEnv
        return HardGoalEnv()
    raise NotImplementedError(f"env {env_name} not wired yet")


def _get_fig(size=(4, 4), dpi=90):
    key = (size, dpi)
    if key not in _FIG_CACHE:
        _FIG_CACHE[key] = plt.subplots(figsize=size, dpi=dpi)
    return _FIG_CACHE[key]


def _render_hard_move(env, trail):
    fig, ax = _get_fig()
    ax.clear()
    w = env._env.world
    agent_pos = w.agents[0].state.p_pos
    land_pos = w.landmarks[0].state.p_pos
    radius = getattr(w.agents[0], "target_distance", 0.1)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.add_patch(Circle(land_pos, radius, facecolor="#FFC9C9", edgecolor="#D55E00",
                        alpha=0.6, linewidth=1.2))
    ax.plot(*land_pos, marker="*", markersize=18, color="#D55E00", zorder=3)
    if len(trail) > 1:
        tx, ty = zip(*trail)
        ax.plot(tx, ty, "-", color="#0072B2", linewidth=1.4, alpha=0.6, zorder=2)
    ax.plot(*agent_pos, marker="o", markersize=13, color="#0072B2",
            markeredgecolor="white", markeredgewidth=1.2, zorder=4)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _render_dlpa_2d(env, trail, title):
    """Paper-style DLPA visualization: agent dot, target star, success ring,
    actuator arrows (Hard Move), and fading trail of past positions."""
    fig, ax = _get_fig()
    ax.clear()
    w = env._env.world
    a_pos = w.agents[0].state.p_pos
    l_pos = w.landmarks[0].state.p_pos
    radius = getattr(w.agents[0], "target_distance", 0.1)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.25, linewidth=0.5)
    # Match the DLPA paper figure exactly:
    # - Hard Move: target is a filled orange disc with a flag inside; agent has no ring
    # - Catch Point: target is a 4-pointed star; agent sits inside a large blue
    #   "valid catch distance" ring
    if title.startswith("hard_move"):
        # Filled target disc
        ax.add_patch(Circle(l_pos, max(radius, 0.2),
                            facecolor="#F5B97D", edgecolor="#D55E00",
                            alpha=0.85, linewidth=1.5, zorder=3))
        # Flag inside the disc (pole + triangular fabric)
        pole_h = 0.16
        fx, fy = l_pos[0] - 0.02, l_pos[1] - pole_h * 0.4
        ax.plot([fx, fx], [fy, fy + pole_h], color="#222", linewidth=2.0, zorder=5)
        flag_pts = np.array([
            [fx, fy + pole_h],
            [fx + 0.10, fy + pole_h * 0.75],
            [fx, fy + pole_h * 0.5],
        ])
        ax.fill(flag_pts[:, 0], flag_pts[:, 1],
                color="#A0522D", edgecolor="#222", linewidth=1.0, zorder=5)
    else:
        # Catch Point: catch-radius ring around the AGENT
        ax.add_patch(Circle(a_pos, max(radius, 0.22),
                            facecolor="#D6E5F8", edgecolor="#0072B2",
                            alpha=0.55, linewidth=1.2, zorder=2))
        # 4-pointed star at target
        ax.plot(*l_pos, marker=(4, 1, 0), markersize=22, color="#D55E00",
                markeredgecolor="#222", markeredgewidth=0.8, zorder=4)
    # Hard Move only: actuator direction arrows around the agent
    if title.startswith("hard_move"):
        n_actuators = getattr(env, "n_actuators", 0)
        if n_actuators > 0:
            arrow_len = 0.18
            for k in range(n_actuators):
                theta = 2 * np.pi * k / n_actuators
                dx, dy = arrow_len * np.cos(theta), arrow_len * np.sin(theta)
                ax.annotate("", xy=(a_pos[0] + dx, a_pos[1] + dy),
                            xytext=(a_pos[0], a_pos[1]),
                            arrowprops=dict(arrowstyle="->", color="#888",
                                            lw=1.0, alpha=0.7), zorder=3)
    # Fading trail of past agent positions
    if len(trail) > 1:
        tx, ty = zip(*trail)
        ax.plot(tx, ty, "-", color="#0072B2", linewidth=1.4, alpha=0.5, zorder=2)
        n = len(trail) - 1
        for i, (px, py) in enumerate(trail[:-1]):
            alpha = 0.15 + 0.55 * (i / max(1, n - 1)) if n > 1 else 0.4
            ax.plot(px, py, marker="o", markersize=10, color="#0072B2",
                    markeredgecolor="white", markeredgewidth=0.8,
                    alpha=alpha, zorder=3)
    # Current agent position (on top)
    ax.plot(*a_pos, marker="o", markersize=13, color="#0072B2",
            markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _render_platform(env, trail):
    """Platformer: 3 platforms, player, 2 enemies, finish at right."""
    fig, ax = _get_fig(size=(7, 2.2), dpi=90)
    ax.clear()
    u = env._env.unwrapped
    p1, p2, p3 = u.platform1, u.platform2, u.platform3
    player_pos = u.player.position
    e1, e2 = u.enemy1.position, u.enemy2.position
    plat_h = 40.0
    total_w = p3.position[0] + p3.size[0]
    ax.set_xlim(-10, total_w + 20)
    ax.set_ylim(0, max(p1.position[1], p2.position[1], p3.position[1]) + 200)
    ax.set_aspect("equal"); ax.set_facecolor("#EAF2FB")
    for plat in (p1, p2, p3):
        ax.add_patch(plt.Rectangle((plat.position[0], plat.position[1]),
                                   plat.size[0], plat_h,
                                   facecolor="#888", edgecolor="#444"))
    if len(trail) > 1:
        tx, ty = zip(*trail)
        ax.plot(tx, ty, "-", color="#0072B2", linewidth=1.4, alpha=0.6)
    ax.plot(player_pos[0], player_pos[1] + 20, "o", markersize=11,
            color="#0072B2", markeredgecolor="white", markeredgewidth=1.2)
    for ep in (e1, e2):
        ax.plot(ep[0], ep[1] + 20, "o", markersize=10,
                color="#D55E00", markeredgecolor="white", markeredgewidth=1.0)
    ax.plot(total_w, p3.position[1] + plat_h, marker="*", markersize=22,
            color="#009E73", zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _render_goal(env, trail):
    """Soccer-like: player, ball, goalie on a pitch with right-edge goal."""
    fig, ax = _get_fig(size=(5, 3.5), dpi=90)
    ax.clear()
    u = env._env.unwrapped
    player = u.player.position
    ball = u.ball.position
    goalie = u.goalie.position
    ax.set_xlim(-15, 60); ax.set_ylim(-30, 30)
    ax.set_aspect("equal"); ax.set_facecolor("#C9E5C2")
    ax.add_patch(plt.Rectangle((52.5, -7.01), 2.5, 14.02,
                               facecolor="white", edgecolor="#444", linewidth=1.5))
    ax.axvline(x=52.5, color="white", linewidth=1.0, alpha=0.7)
    if len(trail) > 1:
        tx, ty = zip(*trail)
        ax.plot(tx, ty, "-", color="#0072B2", linewidth=1.3, alpha=0.55)
    ax.plot(*goalie, "o", markersize=12, color="#D55E00",
            markeredgecolor="white", markeredgewidth=1.2, zorder=4)
    ax.plot(*player, "o", markersize=12, color="#0072B2",
            markeredgecolor="white", markeredgewidth=1.2, zorder=5)
    ax.plot(*ball, "o", markersize=8, color="white",
            markeredgecolor="black", markeredgewidth=1.0, zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


USE_NATIVE_RENDERER = os.environ.get("BFN_NATIVE_RENDER", "0") == "1"


def _render_native(env):
    """Use the env's own renderer (paper-style). Caller must set SDL_VIDEODRIVER=dummy."""
    out = env._env.unwrapped.render(mode="rgb_array")
    return np.asarray(out, dtype=np.uint8)


def _render_native_dlpa(env, trail=None):
    """DLPA multiagent env: native pygame-ported MPE renderer (paper-style) +
    fading trail of past agent positions overlaid on top."""
    out = env._env.render(mode="rgb_array")
    if isinstance(out, list):
        out = out[0]
    arr = np.asarray(out, dtype=np.uint8).copy()
    if trail is None or len(trail) < 2:
        return arr
    # MPE viewers use 700x700 with cam_range=1 → world [-1,1] maps to [0,700]
    H, W = arr.shape[:2]
    from PIL import Image, ImageDraw
    img = Image.fromarray(arr).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    n = len(trail) - 1
    for i, (wx, wy) in enumerate(trail[:-1]):
        # World [-1, 1] -> screen pixels; Y flipped (screen down)
        sx = int(round((wx + 1) * 0.5 * W))
        sy = int(round((1 - (wy + 1) * 0.5) * H))
        alpha = int(40 + (180 - 40) * (i / max(1, n - 1))) if n > 1 else 100
        r = 9
        draw.ellipse([sx - r, sy - r, sx + r, sy + r],
                     fill=(120, 120, 120, alpha), outline=(60, 60, 60, alpha))
    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"), dtype=np.uint8)


def render_frame(env, env_name, trail):
    if USE_NATIVE_RENDERER and env_name in ("platform", "goal", "hard_goal"):
        return _render_native(env)
    if env_name.startswith("hard_move") or env_name == "catch_point":
        return _render_dlpa_2d(env, trail, env_name)
    if env_name == "platform":
        return _render_platform(env, trail)
    if env_name in ("goal", "hard_goal"):
        return _render_goal(env, trail)
    raise NotImplementedError(env_name)


def _agent_xy(env, env_name):
    if env_name.startswith("hard_move") or env_name == "catch_point":
        return tuple(env._env.world.agents[0].state.p_pos)
    if env_name == "platform":
        return tuple(env._env.unwrapped.player.position)
    if env_name in ("goal", "hard_goal"):
        return tuple(env._env.unwrapped.player.position)
    raise NotImplementedError(env_name)


def load_policy_by_type(env_name, policy_type, ckpt, n_actuators, device):
    """Dispatch to the per-env eval script's load_policy."""
    if env_name.startswith("hard_move"):
        from scripts.eval_hard_move import load_policy
        return load_policy(ckpt, policy_type, n_actuators, device)
    if env_name == "catch_point":
        from scripts.eval_catch_point import load_policy
        return load_policy(ckpt, policy_type, device)
    if env_name == "platform":
        from scripts.eval_platform import load_policy
        return load_policy(ckpt, policy_type, device)
    if env_name == "goal":
        from scripts.eval_goal import load_policy
        return load_policy(ckpt, policy_type, device)
    if env_name == "hard_goal":
        from scripts.eval_hard_goal import load_policy
        return load_policy(ckpt, policy_type, device)
    raise NotImplementedError(env_name)


def get_action_fn(env_name, policy_type, n_actuators):
    """Return a callable (policy, obs_history, device) -> env action."""
    if env_name.startswith("hard_move"):
        num_discrete = 2 ** n_actuators
        from scripts.eval_hard_move import get_action_bfn, get_action_continuous
        if policy_type.startswith("bfn"):
            return lambda p, o, d: get_action_bfn(p, o, num_discrete, d)
        return lambda p, o, d: get_action_continuous(p, o, num_discrete, d)
    if env_name == "catch_point":
        from scripts.eval_catch_point import get_action_bfn, get_action_continuous
        if policy_type.startswith("bfn"):
            return get_action_bfn
        return get_action_continuous
    if env_name == "platform":
        # eval_platform get_action signature: (policy, obs_history) -- no device
        from scripts.eval_platform import get_action_bfn, get_action_continuous
        fn = get_action_bfn if policy_type.startswith("bfn") else get_action_continuous
        return lambda p, o, d: fn(p, o)
    if env_name == "goal":
        from scripts.eval_goal import get_action_bfn, get_action_continuous
        fn = get_action_bfn if policy_type.startswith("bfn") else get_action_continuous
        return lambda p, o, d: fn(p, o)
    if env_name == "hard_goal":
        from scripts.eval_hard_goal import get_action_bfn, get_action_continuous
        if policy_type.startswith("bfn"):
            return get_action_bfn
        return get_action_continuous
    raise NotImplementedError(env_name)


SUCCESS_THRESHOLD = {
    "platform": ("gt", 0.9),       # R > 0.9
    "goal": ("ge", 40.0),          # R >= 40
    "hard_goal": ("ge", 40.0),     # R >= 40
    "catch_point": ("gt", 0.0),    # R > 0
    # hard_move_n*: R > 0 (default)
}


def _is_success(env_name, reward):
    op, thr = SUCCESS_THRESHOLD.get(env_name, ("gt", 0.0))
    return reward >= thr if op == "ge" else reward > thr


def rollout_one_episode(env, policy, env_name, policy_type, n_actuators, seed,
                        max_steps, device):
    """Run one episode with frame capture. Returns (frames, total_reward, success).
    Seeds torch + numpy so the policy's stochastic sampling is reproducible across runs."""
    get_action = get_action_fn(env_name, policy_type, n_actuators)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    obs, _ = env.reset(seed=seed)
    obs_np = np.asarray(obs, dtype=np.float32)
    obs_history = torch.from_numpy(np.stack([obs_np, obs_np])).unsqueeze(0).to(device)

    trail = [_agent_xy(env, env_name)]
    frames = [render_frame(env, env_name, trail)]
    total_reward, done, steps = 0.0, False, 0
    while not done and steps < max_steps:
        action = get_action(policy, obs_history, device)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        steps += 1
        trail.append(_agent_xy(env, env_name))
        frames.append(render_frame(env, env_name, trail))
        obs_np = np.asarray(obs, dtype=np.float32)
        new_obs = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0).to(device)
        obs_history = torch.cat([obs_history[:, 1:], new_obs], dim=1)
    # For goal/hard_goal, append a synchronized ball-flight animation in every
    # panel so all 4 videos have the same motion timing. Successes reach the
    # goal; failures stop ~30% of the way, visually conveying a shot that
    # didn't make it.
    if env_name in ("goal", "hard_goal"):
        try:
            u = env._env.unwrapped
            end_ball = np.asarray(u.ball.position, dtype=float).copy()
            goal_center = np.array([53.75, 0.0])  # midpoint of goal mouth
            direction = goal_center - end_ball
            scale = 1.0 if _is_success(env_name, total_reward) else 0.3
            target = end_ball + scale * direction
            for alpha in (0.25, 0.5, 0.75, 1.0):
                u.ball.position[:] = end_ball * (1 - alpha) + target * alpha
                frames.append(render_frame(env, env_name, trail))
        except Exception:
            pass
    env.close()
    return frames, total_reward, _is_success(env_name, total_reward)


def label_frame(frame, text, font=None):
    """Burn a label into the top-left of a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    if font is None:
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 24)
        except OSError:
            font = ImageFont.load_default()
    pad = 6
    bbox = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle([bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2], fill=(0, 0, 0))
    draw.text((pad, pad), text, font=font, fill=(255, 255, 255))
    return np.asarray(img)


def pad_and_stack_grid(frame_streams, labels, sep_px=6, sep_color=(20, 20, 20)):
    """Pad each stream to common length, label, stack 2x2 with visible separators."""
    n = max(len(s) for s in frame_streams)
    h, w = frame_streams[0][0].shape[:2]
    padded = []
    for stream, lab in zip(frame_streams, labels):
        s = [label_frame(f, lab) for f in stream]
        if len(s) < n:
            s = s + [s[-1]] * (n - len(s))
        padded.append(s)
    v_sep = np.full((h, sep_px, 3), sep_color, dtype=np.uint8)        # vertical separator
    h_sep = np.full((sep_px, 2 * w + sep_px, 3), sep_color, dtype=np.uint8)  # horizontal separator
    grid_frames = []
    for t in range(n):
        top = np.concatenate([padded[0][t], v_sep, padded[1][t]], axis=1)
        bot = np.concatenate([padded[2][t], v_sep, padded[3][t]], axis=1)
        grid_frames.append(np.concatenate([top, h_sep, bot], axis=0))
    return grid_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--n_actuators", type=int, default=0)
    ap.add_argument("--bfn_ckpt", required=True)
    ap.add_argument("--ddpm_ckpt", required=True)
    ap.add_argument("--ddim_ckpt", default=None, help="defaults to ddpm_ckpt")
    ap.add_argument("--consistency_ckpt", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--cons_steps", type=int, default=1, choices=[1, 3],
                    help="Consistency Policy inference steps")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cons_label = "Cons-3" if a.cons_steps == 3 else "Cons-1"
    cons_type = "consistency3" if a.cons_steps == 3 else "consistency"
    policies = [
        ("bfn",       "BFN-20",   a.bfn_ckpt),
        ("ddim",      "DDIM-10",  a.ddim_ckpt or a.ddpm_ckpt),
        ("ddpm",      "DDPM-100", a.ddpm_ckpt),
        (cons_type,   cons_label, a.consistency_ckpt),
    ]

    streams, summaries = [], []
    for ptype, label, ckpt in policies:
        print(f"[{label}] loading {ckpt}", flush=True)
        env = make_env(a.env, a.n_actuators)
        max_steps = a.max_steps or env.max_steps
        policy = load_policy_by_type(a.env, ptype, ckpt, a.n_actuators, device)
        frames, rew, succ = rollout_one_episode(
            env, policy, a.env, ptype, a.n_actuators, a.seed, max_steps, device)
        print(f"[{label}] steps={len(frames)} reward={rew:.2f} success={succ}", flush=True)
        streams.append(frames)
        summaries.append((label, len(frames), rew, succ))

    labels = [f"{s[0]}  R={s[2]:.1f}  {'✓' if s[3] else '✗'}" for s in summaries]
    grid = pad_and_stack_grid(streams, labels)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print(f"writing {len(grid)} frames to {a.out}", flush=True)
    imageio.mimwrite(a.out, grid, fps=a.fps, macro_block_size=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
