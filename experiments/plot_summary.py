"""Summary dot plot: cycle-1 forward travel time T/T* (mean ± sd over 5 seeds) for each ablation."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
r = {**json.load(open("experiments/results/round5_seeds.json")), **json.load(open("experiments/results/round6_seeds.json"))}
order = [("no_pheromone", "No pheromone (goal steering + noise only)"),
         ("pher_no_ctrl", "Pheromone, no adjoint control"),
         ("ctrl_reversed_time", "Pheromone + control, costate re-timed"),
         ("ctrl_flipped", "Pheromone + control, sign flipped"),
         ("ctrl_x2", "Pheromone + control ×2"),
         ("pher_ctrl_n2", "Pheromone + control, 2 agents"),
         ("pher_ctrl", "Pheromone + control, 8 agents"),
         ("pher_ctrl_n32", "Pheromone + control, 32 agents")]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for i, (k, lab) in enumerate(order):
    v = np.array(r[k]["ratio"]); xc = np.nanmean(r[k]["xcross"])
    ax.errorbar(np.nanmean(v), i, xerr=np.nanstd(v), fmt="o", color="#2a6fdb", ms=7, capsize=3, lw=1.5)
    ax.text(2.22, i, f"crosses at x = {xc:.2f}", va="center", fontsize=8, color="#555")
ax.axvline(1.0, color="#1a7f37", lw=1, ls="--"); ax.text(1.0, len(order) - 0.35, " Snell optimum (crosses at 0.96)", fontsize=8, color="#1a7f37")
ax.axvline(1.275, color="#888", lw=1, ls=":"); ax.text(1.275, -0.75, " straight line (crosses at 0.50)", fontsize=8, color="#666")
ax.set_yticks(range(len(order))); ax.set_yticklabels([l for _, l in order], fontsize=9); ax.invert_yaxis()
ax.set_xlim(0.95, 2.6); ax.set_ylim(len(order) - 0.3, -1.1)
ax.set_xlabel("Travel time of next-cycle forward trip, T / T*   (lower is better; mean ± sd, 5 seeds)")
ax.set_title("Which component makes the colony find the faster (refracted) path?", fontsize=11, loc="left")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#eee"); plt.tight_layout(); plt.savefig("experiments/results/ablation_summary.png", dpi=150)
