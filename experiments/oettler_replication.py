"""
Replicate Oettler et al. (2013) trial design with the calibrated agent model.

Conditions: three surface pairs x two food positions (alpha = 30 and 60 deg), food on the
slower surface. Speed ratios from the paper's mean walking speeds (mm/s): glass 4.89,
smooth felt 2.97, rough felt 1.73. Each seed stands in for one colony.

Scores follow the paper's Fig. 3: 0% = Fermat optimum, 100% = straight (direct) path,
for both the crossing angle (rho' vs rho vs alpha) and the travel time of the realized path.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, intelligence_ablation as ia

SPEED = {"glass": 4.89, "smooth": 2.97, "rough": 1.73}
PAIRS = [("glass", "rough"), ("glass", "smooth"), ("smooth", "rough")]


def scores(x, bx, jump):
    """Angle and time scores for a trail crossing the border (y = 0.5) at x."""
    ang = lambda c: np.degrees(np.arctan((bx - c) / 0.5))         # angle from the normal on the slow side
    T = lambda c: np.hypot(0.5, c) + jump * np.hypot(0.5, bx - c)
    xs = np.linspace(0, bx, 200001); xo = xs[np.argmin(T(xs))]
    a, r = ang(bx / 2), ang(xo)
    return 100 * (ang(x) - r) / (a - r), 100 * (T(x) - T(xo)) / (T(bx / 2) - T(xo)), a, r, ang(x)


if __name__ == "__main__":
    base = json.loads(sys.argv[1]); nseed = int(sys.argv[2]); out = sys.argv[3]
    res = []
    for s1, s2 in PAIRS:
        for alpha in (60.0, 30.0):
            jump = SPEED[s1] / SPEED[s2]
            for seed in range(nseed):
                h, _, _ = ia.run(dict(base, nu_jump=jump, alpha_deg=alpha), seed=seed, verbose=False)
                x = float(np.nanmean([r["fwd_xcross"] for r in h[-5:]]))
                bx = float(np.tan(np.deg2rad(alpha)))
                deg, tim, a, r, rp = scores(x, bx, jump)
                row = dict(pair=f"{s1}-{s2}", alpha=alpha, seed=seed, ratio=jump, x=x, alpha_meas=a, rho=r, rho_prime=rp,
                           deg_score=deg, time_score=tim, reached=float(np.mean([q["fwd_success"] for q in h[-5:]])))
                res.append(row)
                print(f"{s1}-{s2} alpha={alpha:.0f} seed={seed}: rho={r:.1f} rho'={rp:.1f} alpha={a:.1f}  "
                      f"angle score={deg:.0f}%  time score={tim:.0f}%  reached={row['reached']:.2f}", flush=True)
                json.dump(res, open(out, "w"), indent=1)
