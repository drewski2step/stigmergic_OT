"""Travel time T(x_c) as a function of the border-crossing point, Oettler geometry (alpha = 60 deg, glass -> rough felt)."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
a = b = 0.5; d = np.tan(np.deg2rad(60)); n1, n2 = 1.0, 2.83
T = lambda x: n1 * np.hypot(a, x) + n2 * np.hypot(b, d - x)
dT = lambda x: n1 * x / np.hypot(a, x) - n2 * (d - x) / np.hypot(b, d - x)
x = np.linspace(0, d, 2001); xo = x[np.argmin(T(x))]; xs = d / 2
ink, mute, opt, st = "#222222", "#8a8a8a", "#c0392b", "#2a6fdb"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
# arena
A = ax[0]; A.axhspan(0.5, 1, color="#eee6da"); A.axhspan(0, 0.5, color="#f6f9fc"); A.axhline(0.5, color=mute, lw=1)
for xc in np.linspace(0.1, 1.7, 9): A.plot([0, xc, d], [0, 0.5, 1], color="#c9c9c9", lw=1)
A.plot([0, xs, d], [0, 0.5, 1], color=st, lw=2.2); A.plot([0, xo, d], [0, 0.5, 1], color=opt, lw=2.2)
A.plot([0, d], [0, 1], "o", color=ink, ms=6); A.text(0.03, -0.07, "nest", fontsize=10); A.text(d - 0.02, 1.04, "food", fontsize=10, ha="right")
A.text(0.03, 0.42, "fast surface", fontsize=9, color=mute); A.text(0.03, 0.92, "slow surface (2.83× slower)", fontsize=9, color=mute)
A.annotate("crossing point $x_c$", (xo, 0.5), (0.95, 0.27), fontsize=9, arrowprops=dict(arrowstyle="-", color=mute))
A.set_xlim(-0.08, d + 0.08); A.set_ylim(-0.12, 1.12); A.set_aspect("equal"); A.axis("off"); A.set_title("Every path is fixed by where it crosses the border", fontsize=11)
# T
B = ax[1]; B.plot(x, T(x), color=ink, lw=2.2)
B.plot(xs, T(xs), "o", color=st, ms=8); B.plot(xo, T(xo), "o", color=opt, ms=8)
B.annotate(f"straight line\n$x_c$ = {xs:.2f}, T = {T(xs):.2f}", (xs, T(xs)), (0.25, 3.35), fontsize=9, color=st, arrowprops=dict(arrowstyle="-", color=st))
B.annotate(f"least-time path\n$x_c$ = {xo:.2f}, T = {T(xo):.2f}", (xo, T(xo)), (0.95, 4.6), fontsize=9, color=opt, arrowprops=dict(arrowstyle="-", color=opt))
B.set_xlabel("crossing point $x_c$"); B.set_ylabel("travel time T"); B.set_title("Travel time: one bowl, one minimum", fontsize=11)
# slope
C = ax[2]; C.plot(x, dT(x), color=ink, lw=2.2); C.axhline(0, color=mute, lw=1)
C.plot(xs, dT(xs), "o", color=st, ms=8); C.plot(xo, 0, "o", color=opt, ms=8)
C.annotate(f"slope = {dT(xs):.2f}\n(push toward larger $x_c$)", (xs, dT(xs)), (0.15, -0.4), fontsize=9, color=st, arrowprops=dict(arrowstyle="-", color=st))
C.annotate("slope = 0: Snell's law", (xo, 0), (0.55, 0.45), fontsize=9, color=opt, arrowprops=dict(arrowstyle="-", color=opt))
C.set_xlabel("crossing point $x_c$"); C.set_ylabel(r"slope  $\nu_1\sin\theta_1-\nu_2\sin\theta_2$"); C.set_title("Slope of T = the Snell mismatch", fontsize=11)
for q in (B, C):
    for s in ("top", "right"): q.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("results/theory/T_of_xc.png", dpi=160)
print(xo, T(xo), T(xs), dT(xs))
