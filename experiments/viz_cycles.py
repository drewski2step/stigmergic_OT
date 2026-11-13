"""Plot forward (red) and return (blue) paths per cycle for one ablation config."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import intelligence_ablation as ia

cfg = json.loads(sys.argv[1]); out = sys.argv[2]; ncyc = cfg.get("num_cycles", 4)
fig, axes = plt.subplots(1, ncyc, figsize=(3.2 * ncyc, 3.4), sharex=True, sharey=True)
orig_walk = ia.walk; store = []
def rec(*a, **k):
    tr = orig_walk(*a, **k); store.append(np.asarray(tr)); return tr
ia.walk = rec
ia.run(cfg, verbose=False)
xs = np.linspace(0, 1, 200); Tt = np.hypot(.5, xs) + 10 * np.hypot(.5, 1 - xs); xo = xs[np.argmin(Tt)]
for c in range(ncyc):
    ax = axes[c]; f, b = store[2 * c], store[2 * c + 1]
    ax.axhspan(0.5, 1.15, color="0.9")
    for i in range(f.shape[1]):
        ax.plot(f[:, i, 0], f[:, i, 1], "r", lw=0.7, alpha=0.7)
        ax.plot(b[:, i, 0], b[:, i, 1], "b", lw=0.7, alpha=0.5)
    ax.plot([0, xo, 1], [0, .5, 1], "k--", lw=1)
    ax.plot(0, 0, "ko"); ax.plot(1, 1, "g*", ms=10)
    ax.set_title(f"cycle {c}"); ax.set_xlim(-.15, 1.15); ax.set_ylim(-.15, 1.15); ax.set_aspect("equal")
fig.suptitle(f"{cfg}  —  red: forward, blue: return, dashed: Snell optimum, gray: slow medium (nu=10)", fontsize=8)
plt.tight_layout(); plt.savefig(out, dpi=130)
