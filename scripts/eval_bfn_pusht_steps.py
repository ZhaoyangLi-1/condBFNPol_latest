"""Evaluate the retrained BFN sim-PushT checkpoint at 20 and 10 inference steps.

BFN-10 = same weights, n_timesteps=10. Reports max-coverage (mean +/- 95% CI),
success@0.95, and inference time, matching the Q2 sim Push-T table.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "src/diffusion-policy")
import numpy as np
import torch
from scripts.benchmark_policies import load_policy, run_rollout_evaluation, measure_inference_time

CKPT = "outputs/2026.05.25/18.15.44_bfn_seed42/checkpoints/latest.ckpt"


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    policy, cfg = load_policy(CKPT, device=dev)
    # dummy obs for timing
    import collections
    out = {}
    for steps in [20, 10]:
        policy.n_timesteps = steps
        print(f"\n===== BFN n_timesteps={steps} =====", flush=True)
        roll = run_rollout_evaluation(policy, cfg, n_test=50, device=dev)
        scores = np.array(roll["all_scores"])
        cov_ci = 1.96 * scores.std(ddof=1) / np.sqrt(len(scores))
        succ = float((scores >= 0.95).mean() * 100)
        succ_ci = 1.96 * np.sqrt((succ / 100) * (1 - succ / 100) / len(scores)) * 100
        # inference timing: build one obs_dict from a fresh env
        from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
        from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
        env = MultiStepWrapper(PushTImageEnv(render_size=96), n_obs_steps=cfg.n_obs_steps,
                               n_action_steps=cfg.n_action_steps, max_episode_steps=300)
        env.seed(0); obs = env.reset()
        obs_dict = {k: torch.from_numpy(v).unsqueeze(0).to(dev) for k, v in obs.items()}
        tinfo = measure_inference_time(policy, obs_dict)
        it_ci = 1.96 * (tinfo["std_ms"] / np.sqrt(20))
        out[f"bfn_{steps}"] = dict(
            n=len(scores), coverage_mean=float(scores.mean()), coverage_ci=float(cov_ci),
            success95=succ, success95_ci=float(succ_ci),
            inf_mean_ms=tinfo["mean_ms"], inf_ci_ms=float(it_ci), nfe=steps)
        print(f"BFN-{steps}: coverage={scores.mean():.3f}+/-{cov_ci:.3f}  "
              f"succ@0.95={succ:.1f}+/-{succ_ci:.1f}  inf={tinfo['mean_ms']:.2f}+/-{it_ci:.2f}ms", flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/bfn_pusht_steps.json", "w"), indent=2)
    print("\nSaved results/bfn_pusht_steps.json")


if __name__ == "__main__":
    main()