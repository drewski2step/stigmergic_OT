"""Fig. 3-style comparison: % of passages on the selected branch vs cumulative passages."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "experiments/results/bridge/"
ld = lambda f, k: np.array(json.load(open(R + f))[k]["curve"])
x = np.arange(50, 2000, 100)
fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(x, ld("authors_kappa8e-5.json", "deneubourg_n2_k20"), color="#111", lw=2.5, label="Deneubourg model n=2, k=20 (fit to the ant data)")
ax.plot(x, ld("authors_kappa8e-5.json", "authors_kappa8e-5"), color="#d1495b", lw=2, ls="--", label="Authors' agent model, authors' parameters")
ax.plot(x, ld("kappa50_beta2.json", "kappa50_beta2"), color="#2a6fdb", lw=2, label="Authors' agent model, floor = 50 passages, β = 2")
ax.plot(x, ld("kappa20.json", "kappa20"), color="#8a8a8a", lw=1.5, ls=":", label="Authors' agent model, floor = 20 passages, β = 1")
ax.set_ylim(50, 101); ax.set_xlim(0, 2000)
ax.set_xlabel("Cumulative ant passages"); ax.set_ylabel("% of passages on the selected branch")
ax.set_title("Diamond bridge: does the authors' steering law reproduce Deneubourg (1990)?", fontsize=10.5, loc="left")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.grid(color="#eee"); ax.legend(fontsize=8, frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig("experiments/results/bridge_vs_deneubourg.png", dpi=150)
