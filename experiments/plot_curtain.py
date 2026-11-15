"""Pheromone 'curtain': height = scent laid per metre = r / v, drawn over three nest-to-food paths.
The curtain's area is r x travel time, so the least-time path is the least-area curtain."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
d = np.tan(np.deg2rad(60)); n1, n2 = 1.0, 2.83
paths = [("Straight line", [(0, 0), (d / 2, .5), (d, 1)], "#2a6fdb"),
         ("Worst: longest stretch on slow ground", [(0, 0), (0, .5), (0, 1), (d, 1)], "#8a8a8a"),
         ("Least-time path", [(0, 0), (1.553, .5), (d, 1)], "#c0392b")]
fig = plt.figure(figsize=(15, 4.8))
for i, (name, P, col) in enumerate(paths):
    ax = fig.add_subplot(1, 3, i + 1, projection="3d"); area = 0
    for (x0, y0), (x1, y1) in zip(P[:-1], P[1:]):
        hgt = n1 if (y0 + y1) / 2 < 0.5 else n2
        area += hgt * np.hypot(x1 - x0, y1 - y0)
        ax.add_collection3d(Poly3DCollection([[(x0, y0, 0), (x1, y1, 0), (x1, y1, hgt), (x0, y0, hgt)]], facecolor=col, alpha=0.55, edgecolor=col))
    ax.plot([0, d], [.5, .5], [0, 0], color="#555", lw=1, ls="--")
    xs, ys = zip(*P); ax.plot(xs, ys, 0, color="#222", lw=1.5)
    ax.scatter([0, d], [0, 1], [0, 0], color="#222", s=18)
    ax.text(0, -0.12, 0, "nest", fontsize=8); ax.text(d, 1.05, 0, "food", fontsize=8)
    ax.set_xlim(0, d); ax.set_ylim(0, 1); ax.set_zlim(0, 3); ax.set_box_aspect((d, 1, 1.1))
    ax.set_xlabel("x", labelpad=-8); ax.set_ylabel("y", labelpad=-8); ax.set_zlabel("scent per metre", labelpad=-6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([1, 2.83]); ax.set_zticklabels(["1", "2.83"], fontsize=8)
    ax.view_init(elev=24, azim=-62)
    ax.set_title(f"{name}\ncurtain area = travel time = {area:.2f}", fontsize=10)
fig.suptitle("Scent laid at a steady rate per second: height = 1 / speed, area = travel time", fontsize=12)
fig.tight_layout(); fig.savefig("results/theory/pheromone_curtain.png", dpi=160)
