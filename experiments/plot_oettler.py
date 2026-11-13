"""Crossing point by mechanism at Oettler's glass/rough-felt speed ratio (2.83), Deneubourg-calibrated ants."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "experiments/results/oettler/"
first = lambda f: list(json.load(open(R + f)).values())[0]
rows = [("OT model as implemented (adjoint on return trip)", np.array(json.load(open(R + "glass_ctrl.json"))["fwd_xcross"][1:]).ravel()),
        ("Pure stigmergy, speed-weighted deposits", first("grid_long_g3.json")["xcross"]),
        ("Pure stigmergy, amplified speed weighting", np.array(json.load(open(R + "grid_long_g3.json"))["gg3.0_tw4.0"]["xcross"])),
        ("+ individual hindsight (η = −0.05)", first("learn_eta-0.05.json")["xcross"]),
        ("+ hindsight with memory (η = −0.15, 90% recall)", first("learn_mem0.9.json")["xcross"])]
fig, ax = plt.subplots(figsize=(8, 3.6))
for i, (lab, v) in enumerate(rows):
    v = np.array(v, float); v = v[~np.isnan(v)]
    ax.errorbar(v.mean(), i, xerr=v.std(), fmt="o", color="#2a6fdb", ms=7, capsize=3)
ax.axvline(0.841, color="#1a7f37", ls="--", lw=1); ax.text(0.836, -0.75, "Fermat / Oettler's ants (0.84)", fontsize=8, color="#1a7f37", ha="right")
ax.axvline(0.5, color="#888", ls=":", lw=1); ax.text(0.502, -0.75, "straight line (0.50)", fontsize=8, color="#666")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9); ax.invert_yaxis()
ax.set_xlim(0.45, 0.95); ax.set_ylim(len(rows) - 0.4, -1.1)
ax.set_xlabel("Where the colony's trail crosses into the slow substrate (x)")
ax.set_title("Refraction at speed ratio 2.83 with Deneubourg-calibrated ants", fontsize=10.5, loc="left")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#eee"); plt.tight_layout(); plt.savefig("experiments/results/oettler_refraction_summary.png", dpi=150)
