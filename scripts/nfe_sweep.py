"""NFE sweep for BFN: evaluate one trained BFN checkpoint at several inference-step
counts n (no retraining), recording success rate and cumulative reward. Demonstrates
BFN's step-count flexibility (the belief is a valid posterior at every n).

One env per process (the PAMDP env modules add conflicting paths, so we never import
two of them together).
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)          # hard_move_n4/6/8/10, goal, hard_goal, catch_point
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_actuators", type=int, default=0)  # for hard_move only
    ap.add_argument("--steps", default="1 2 4 10 20 50")
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    steps = [int(s) for s in a.steps.split()]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if a.env.startswith("hard_move"):
        from scripts.eval_hard_move import load_policy, evaluate_policy
        policy = load_policy(a.ckpt, "bfn", a.n_actuators, dev)
        run = lambda: evaluate_policy(policy, "bfn", a.n_actuators, a.n_episodes, dev)
    elif a.env == "goal":
        from scripts.eval_goal import load_policy, evaluate
        policy = load_policy(a.ckpt, "bfn", dev)
        run = lambda: evaluate(policy, "bfn", a.n_episodes, dev)
    elif a.env == "hard_goal":
        from scripts.eval_hard_goal import load_policy, evaluate
        policy = load_policy(a.ckpt, "bfn", dev)
        run = lambda: evaluate(policy, "bfn", a.n_episodes, dev)
    elif a.env == "catch_point":
        from scripts.eval_catch_point import load_policy, evaluate
        policy = load_policy(a.ckpt, "bfn", dev)
        run = lambda: evaluate(policy, "bfn", a.n_episodes, dev)
    else:
        raise ValueError(a.env)

    res = {}
    for n in steps:
        policy.n_timesteps = n
        r = run()
        res[str(n)] = {"nfe": n,
                       "success_rate": float(r["success_rate"]),
                       "mean_reward": float(r["mean_reward"]),
                       "std_reward": float(r.get("std_reward", 0.0))}
        print(f"{a.env} n={n:2d}: success={r['success_rate']:.1f}%  reward={r['mean_reward']:.2f}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"env": a.env, "n_episodes": a.n_episodes, "sweep": res}, open(a.out, "w"), indent=2)
    print(f"Saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
