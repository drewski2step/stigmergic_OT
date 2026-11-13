"""Fig. 3-style comparison: ants (Oettler et al. 2013) vs the calibrated model, per condition."""
import json, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "results/oettler/"
ants = json.load(open(R + "oettler_ants_fig3.json"))
mods = {"Hindsight learning": json.load(open(R + "replication_hindsight_eta3_full.json")),
        "Pure stigmergy": json.load(open(R + "replication_stigmergy.json"))}
pairs = ["glass-rough", "glass-smooth", "smooth-rough"]
conds = [(p, a) for a in (60.0, 30.0) for p in pairs]
col = {"Ants": "#222222", "Hindsight learning": "#2a6fdb", "Pure stigmergy": "#e07b39"}
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=False)
for ax, key, lab in [(axs[0], "deg", "Angle score (%)"), (axs[1], "time", "Time score (%)")]:
    ax.axhspan(-200, 0, color="#f2f2f2", zorder=0)
    ax.axhline(0, color="#555", lw=1); ax.axhline(100, color="#555", lw=1, ls="--")
    ax.text(5.45, -12, "Fermat optimum", fontsize=8, va="top", ha="right", color="#555")
    ax.text(2.45, 104, "Direct path", fontsize=8, va="bottom", ha="right", color="#555")
    for i, (p, a) in enumerate(conds):
        k = "large_alpha" if a == 60 else "small_alpha"
        v = ants[k][p][key]
        ax.scatter(np.full(3, i - 0.22), v, s=36, color=col["Ants"], zorder=3, label="Ants (3 colonies)" if i == 0 else None)
        for j, (name, rows) in enumerate(mods.items()):
            y = [r[key + ("_score" if key == "time" else "_score")] for r in rows if r["pair"] == p and r["alpha"] == a]
            ax.scatter(np.full(len(y), i + 0.0 + 0.22 * j), y, s=36, color=col[name], marker="s" if j else "D", zorder=3,
                       label=f"Model: {name.lower()}{', η = 0.3' if j == 0 else ''} (3 seeds)" if i == 0 else None)
    ax.set_xticks(range(6)); ax.set_xticklabels([f"{p}\nα={a:.0f}°" for p, a in conds], fontsize=8)
    ax.axvline(2.5, color="#bbb", lw=1)
    ax.set_ylim(-170, 230); ax.set_ylabel(lab)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_title("Trail angle: 0% = Fermat, 100% = straight line", fontsize=10)
axs[1].set_title("Travel time on that trail, same scale", fontsize=10)
axs[0].legend(fontsize=8, loc="upper left", frameon=False)
fig.suptitle("Oettler et al. (2013) design: fire ants vs calibrated OT-model agents", fontsize=12)
fig.tight_layout(); fig.savefig(R + "oettler_replication_eta03.png", dpi=160)
print("saved")
