"""
Deneubourg's diamond bridge, run with the Krishnan-Mahadevan agent model.

Experiment (Deneubourg, Aron, Goss & Pasteels 1990, J. Insect Behav. 3:159):
two identical branches leave a fork at +-30 deg (60 deg apart); explorers cross one
at a time and mark the branch they take; Fig. 3 plots the % of ants on the
eventually-selected branch vs cumulative passages. Their choice model
    P_A = (k+A)^n / ((k+A)^n + (k+B)^n),  n = 2, k = 20
reproduces the averaged data.

Agent model (authors' Eq. 1-2, as in src/apic.py, eps_theta = 1, unit speed):
    x' = cos th,  y' = sin th,
    th' = beta * grad log phi . (-sin th, cos th) + sqrt(2 D) xi
Pheromone field (authors' KDE: Gaussians of width sigma on every deposited point,
inside log(c + sum)). A passage along a branch deposits points every dt_dep, so a
branch marked A times has ridge height A*q with q = sigma*sqrt(2 pi)/dt_dep. Then
    phi = c + q * [ (A+B) f_stem + A f_upper + B f_lower ]
where f is the normalized line-of-Gaussians profile. kappa = c/q plays the role of
Deneubourg's k (in passages). The authors' values (c=0.01, sigma=0.05, dt=0.001)
give q = 125, kappa = 8e-5.

Bridge constraint: an ant commits to the branch on whose side it is once it is
R_COMMIT past the fork (the bridge allows no other way through).
Lengths in units of the fork-to-fork distance (15 cm in the experiment).
"""
import sys, json, time
import numpy as np
from math import erf

SQ2 = np.sqrt(2.0)
ANG = np.deg2rad(30.0)
U_UP = np.array([np.cos(ANG), np.sin(ANG)]); U_LO = np.array([np.cos(ANG), -np.sin(ANG)])
ARM = 0.5 / np.cos(ANG)          # branch length to the diamond's side corner
X_START, R_COMMIT = -0.2, 0.1
verf = np.vectorize(erf)
from scipy.special import erf as serf


def line_profile(x, y, p0, u, L, sigma):
    """Sum of Gaussians along a segment, normalized to height 1 mid-segment."""
    rx, ry = x - p0[0], y - p0[1]
    s = rx * u[0] + ry * u[1]
    d = rx * u[1] - ry * u[0]
    return np.exp(-d**2 / (2 * sigma**2)) * 0.5 * (serf((L - s) / (SQ2 * sigma)) + serf(s / (SQ2 * sigma)))


def log_phi(x, y, A, B, sigma, kappa):
    stem = line_profile(x, y, (X_START - 0.3, 0.0), (1.0, 0.0), 0.3 - X_START, sigma)
    up = line_profile(x, y, (0.0, 0.0), U_UP, ARM, sigma)
    lo = line_profile(x, y, (0.0, 0.0), U_LO, ARM, sigma)
    return np.log(kappa + (A + B) * stem + A * up + B * lo)   # log phi up to the constant log q


def run_bridge(n_pass=2000, runs=200, beta=1.0, D=0.5, sigma=0.05, kappa=8e-5,
               dt=0.002, goal_gain=0.0, seed=0, record=False):
    rng = np.random.default_rng(seed)
    A = np.zeros(runs); B = np.zeros(runs)
    choices = np.zeros((n_pass, runs), dtype=np.int8)       # 1 = upper branch
    rec = []
    h = 1e-4
    for i in range(n_pass):
        x = np.full(runs, X_START); y = np.zeros(runs); th = rng.normal(0, 0.001, runs)
        active = np.ones(runs, bool); pick = np.zeros(runs, np.int8)
        for _ in range(int(3 * (R_COMMIT - X_START) / dt)):
            gx = (log_phi(x + h, y, A, B, sigma, kappa) - log_phi(x - h, y, A, B, sigma, kappa)) / (2 * h)
            gy = (log_phi(x, y + h, A, B, sigma, kappa) - log_phi(x, y - h, A, B, sigma, kappa)) / (2 * h)
            w = beta * (-gx * np.sin(th) + gy * np.cos(th))
            w += goal_gain * np.arctan2(np.sin(-th), np.cos(-th))       # optional pull toward +x (the arena)
            th = th + w * dt + np.sqrt(2 * D * dt) * rng.normal(size=runs) * active
            x = x + dt * np.cos(th) * active; y = y + dt * np.sin(th) * active
            done = active & (np.hypot(np.maximum(x, 0), y) >= R_COMMIT) & (x > 0)
            pick[done] = (y[done] > 0); active &= ~done
            # ants that wander back behind the start are re-released (the bridge is a corridor)
            back = active & (x < X_START - 0.1)
            x[back] = X_START; y[back] = 0; th[back] = 0
            if not active.any():
                break
        pick[active] = (y[active] > 0)          # rare stragglers: side they are on
        if record:
            rec.append(np.stack([A.copy(), B.copy(), pick.astype(float)], 1))
        choices[i] = pick
        A += pick; B += 1 - pick
    return choices, (np.concatenate(rec) if record else None)


def deneubourg_mc(n_pass=2000, runs=2000, n=2.0, k=20.0, seed=1):
    rng = np.random.default_rng(seed)
    A = np.zeros(runs); B = np.zeros(runs); ch = np.zeros((n_pass, runs), np.int8)
    for i in range(n_pass):
        p = (k + A)**n / ((k + A)**n + (k + B)**n)
        c = rng.random(runs) < p; ch[i] = c; A += c; B += ~c
    return ch


def selected_curve(ch, bin_=100):
    """% of passages on the eventually selected branch, in bins of `bin_` passages (as in Fig. 3)."""
    win = ch.mean(0) > 0.5
    on = np.where(win[None, :], ch, 1 - ch)
    nb = ch.shape[0] // bin_
    return 100 * on[:nb * bin_].reshape(nb, bin_, -1).mean(1).mean(1)


def fit_nk(rec):
    """Maximum-likelihood fit of Deneubourg's (n, k) to recorded (A, B, choice)."""
    A, B, c = rec[:, 0], rec[:, 1], rec[:, 2]
    best = (-np.inf, None, None)
    for n in np.arange(0.5, 12.01, 0.25):
        for k in np.geomspace(0.01, 500, 120):
            la = n * np.log(k + A); lb = n * np.log(k + B)
            lp = la - np.logaddexp(la, lb); lq = lb - np.logaddexp(la, lb)
            ll = np.sum(c * lp + (1 - c) * lq)
            if ll > best[0]:
                best = (ll, n, k)
    return best[1], best[2]


if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1]); out = {}
    ref = selected_curve(deneubourg_mc())
    out["deneubourg_n2_k20"] = dict(curve=ref.tolist())
    print("Deneubourg n=2,k=20 :", np.round(ref, 1), flush=True)
    for name, cfg in cfgs.items():
        t0 = time.time()
        ch, rec = run_bridge(record=True, **cfg)
        cur = selected_curve(ch, bin_=max(1, cfg.get("n_pass", 2000) // 20))
        n, k = fit_nk(rec)
        out[name] = dict(cfg=cfg, curve=cur.tolist(), n_fit=n, k_fit=k)
        print(f"{name:22s}: {np.round(cur, 1)}  fitted n={n:.2f}, k={k:.3g}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
