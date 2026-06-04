"""Plot BFN NFE sweep: success rate and cumulative reward vs. number of inference
steps n (no retraining), one line per environment. Reads results/nfe_sweep_*.json."""
import glob, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

DISP = {"hard_move_n4": "Hard Move-4", "hard_move_n6": "Hard Move-6",
        "hard_move_n8": "Hard Move-8", "hard_move_n10": "Hard Move-10",
        "goal": "Goal", "hard_goal": "Hard Goal", "catch_point": "Catch Point"}
ORDER = ["hard_move_n4", "hard_move_n6", "hard_move_n8", "hard_move_n10",
         "goal", "hard_goal", "catch_point"]
# Colorblind-safe (Wong) palette + distinct markers, assigned by ORDER.
STYLE = {"hard_move_n4":  ("#0072B2", "o"),
         "hard_move_n6":  ("#E69F00", "s"),
         "hard_move_n8":  ("#009E73", "^"),
         "hard_move_n10": ("#CC79A7", "v"),
         "goal":          ("#D55E00", "D"),
         "hard_goal":     ("#56B4E9", "P"),
         "catch_point":   ("#000000", "X")}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 0.8, "figure.dpi": 120,
})


def main():
    files = sorted(glob.glob("results/nfe_sweep_*.json"))
    data = {}
    for f in files:
        d = json.load(open(f))
        sw = d["sweep"]
        ns = sorted(int(k) for k in sw)
        data[d["env"]] = (ns,
                          [sw[str(n)]["success_rate"] for n in ns],
                          [sw[str(n)]["mean_reward"] for n in ns])

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    envs = [e for e in ORDER if e in data] + [e for e in data if e not in ORDER]
    handles = []
    for env in envs:
        ns, succ, rew = data[env]
        color, marker = STYLE.get(env, ("#444444", "o"))
        kw = dict(color=color, marker=marker, markersize=6, linewidth=2.0,
                  markeredgecolor="white", markeredgewidth=0.6, clip_on=False)
        ln, = axes[0].plot(ns, succ, label=DISP.get(env, env), **kw)
        axes[1].plot(ns, rew, **kw)
        handles.append(ln)

    ticks = [1, 2, 5, 10, 20, 50]
    for ax in axes:
        ax.set_xlabel("Number of inference steps $n$")
        ax.set_xscale("log")
        ax.set_xlim(0.9, 55)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=3)

    axes[0].set_ylabel("Success rate (\\%)" if plt.rcParams["text.usetex"]
                       else "Success rate (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("(a) Success rate", fontsize=11)
    axes[1].set_ylabel("Cumulative reward")
    axes[1].set_title("(b) Cumulative reward", fontsize=11)

    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.02),
               handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/nfe_sweep.pdf", bbox_inches="tight")
    fig.savefig("figures/nfe_sweep.png", dpi=200, bbox_inches="tight")
    print("Saved figures/nfe_sweep.pdf and .png")


if __name__ == "__main__":
    main()
