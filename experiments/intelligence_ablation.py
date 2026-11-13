"""
Where does the "intelligence" live in the stigmergic OT model?

Ablation harness around src/apic.py (Algorithm 1). Each run is a few cycles with
a small batch of agents on the Snell problem (A=(0,0) -> B=(1,1), nu = 1 below
y = 0.5, nu = 10 above). Components can be switched off one at a time:

    goal_gain   global "steer toward B" stabilizer (not in the paper's Eq. 4)
    beta        trail following (pheromone memory is read only through this)
    ctrl_scale  adjoint control on the return trip (Eq. 7); 0 = off, -1 = flipped sign
    reverse_lambda  feed the costate to the return trip in reversed order
    D_theta     angular noise (exploration); also sets gamma = beta * D_theta

Metric: travel time T = sum(nu * ds) from A until the path first comes within
R_HIT of B, divided by the Snell optimum T* = 6.098 (straight line = 7.778, ratio 1.275).
"""
import sys, os, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import jax, jax.numpy as jnp
from jax import random, vmap, lax
from src.apic import (smooth_piecewise_nu, make_init_fn, simulate_forward_batch,
                      integrate_costate, compute_weighted_pheromone_gradient,
                      _step_heading)

A = jnp.array([0.0, 0.0]); B = jnp.array([1.0, 1.0])
T_STAR, T_STRAIGHT, X_STAR = 6.0982, 7.7782, 0.9555
R_HIT = 0.05


def walk(key, final_states, lam, pts, wts, sigma, num_steps, dt, goal,
         beta, D_theta, eps_theta, goal_gain, ctrl_scale, reverse_lambda, nu_on_pher=True):
    """Copy of the authors' forward / controlled backward pass with ablation switches.
    Difference from the authors' code: an agent STOPS once it is within R_HIT of its goal
    (in the original, agents keep walking for a fixed number of steps)."""
    B_ = final_states.shape[0]
    keys = random.split(key, num_steps)
    gamma = beta * D_theta if beta * D_theta > 0 else 1.0
    if lam is None:
        lam = jnp.zeros((num_steps, B_, 3))
    lam_seq = lam[::-1] if reverse_lambda else lam

    def step(state, inp):
        lam_t, k = inp
        x, y, th = state[:, 0], state[:, 1], state[:, 2]
        done = jnp.hypot(x - goal[0], y - goal[1]) < R_HIT
        g = vmap(lambda a, b: compute_weighted_pheromone_gradient(pts, wts, a, b, sigma))(x, y)
        n = jnp.stack([-jnp.sin(th), jnp.cos(th)], 1)
        u_pher = beta * jnp.sum(g * n, 1)
        tg = jnp.arctan2(goal[1] - y, goal[0] - x)
        u_goal = jnp.arctan2(jnp.sin(tg - th), jnp.cos(tg - th))
        # (skip entirely when off: 0 * inf from an exploding costate would give NaN)
        u_ctrl = ctrl_scale * (-lam_t[:, 2] / gamma) if ctrl_scale != 0 else jnp.zeros_like(th)
        nu = vmap(smooth_piecewise_nu)(x, y)
        if not nu_on_pher:   # remove the nu amplification of trail following in slow ground
            u_pher = u_pher / nu
        th2 = _step_heading(th, u_pher, u_goal, u_ctrl, nu, random.normal(k, (B_,)),
                            dt, beta, D_theta, eps_theta, goal_gain)
        s2 = jnp.stack([x + dt * jnp.cos(th2), y + dt * jnp.sin(th2), th2], 1)
        s2 = jnp.where(done[:, None], state, s2)
        return s2, s2

    _, tr = lax.scan(step, final_states, (lam_seq, keys))
    return tr


def travel_times(trajs, start, end, dt):
    """Per-agent travel time from start until first within R_HIT of end (nan if never)."""
    tr = np.asarray(trajs)
    out, xc = [], []
    for b in range(tr.shape[1]):
        x, y = tr[:, b, 0], tr[:, b, 1]
        d = np.hypot(x - float(end[0]), y - float(end[1]))
        hit = np.where(d < R_HIT)[0]
        if len(hit) == 0:
            out.append(np.nan); xc.append(np.nan); continue
        i = hit[0]
        nu = np.asarray(vmap(smooth_piecewise_nu)(jnp.array(x[:i]), jnp.array(y[:i])))
        out.append(float(np.sum(nu * dt)))
        c = np.where(y[:i] > 0.5)[0]
        xc.append(float(x[c[0]]) if len(c) else np.nan)
    return np.array(out), np.array(xc)


def run(cfg, seed=0, verbose=True):
    p = dict(num_cycles=6, batch=8, num_steps=1600, dt=0.001, sigma=0.05,
             beta=1.0, D_theta=0.5, eps_theta=1.0, goal_gain=10.0,
             ctrl_scale=1.0, reverse_lambda=False, stride=8, max_trajs=200, deposit="all", nu_on_pher=True)
    p.update(cfg)
    key = random.PRNGKey(seed)
    init_fn = make_init_fn(A, B, p["batch"])
    pts = jnp.empty((0, 2)); wts = jnp.empty((0,))
    pool = []
    hist = []
    for c in range(p["num_cycles"]):
        key, k0, k1, k2, k3 = random.split(key, 5)
        fwd = walk(k1, init_fn(k0), None, pts, wts, p["sigma"], p["num_steps"], p["dt"], B,
                   p["beta"], p["D_theta"], p["eps_theta"], p["goal_gain"], 0.0, False, p["nu_on_pher"])
        Tf, xf = travel_times(fwd, A, B, p["dt"])
        lam = integrate_costate(fwd, pts, wts, p["sigma"], p["dt"], beta=p["beta"], eps_theta=p["eps_theta"])
        fs = fwd[-1]
        fs = jnp.stack([fs[:, 0], fs[:, 1], jnp.mod(fs[:, 2] + jnp.pi, 2 * jnp.pi)], 1)
        bwd = walk(k2, fs, lam, pts, wts, p["sigma"], p["num_steps"], p["dt"], A,
                   p["beta"], p["D_theta"], p["eps_theta"], p["goal_gain"],
                   p["ctrl_scale"], p["reverse_lambda"], p["nu_on_pher"])
        Tb, xb = travel_times(bwd, B, A, p["dt"])   # return trip B -> A
        # Which return paths lay pheromone?
        #   "all":     every return path, full length (authors' choice)
        #   "trimmed": every return path, cut at arrival (no pile-up at the nest)
        #   "success": only agents that made it home, cut at arrival (real-ant style)
        bw = np.asarray(bwd)
        for i in range(bw.shape[1]):
            xy = bw[:, i, :2]
            hit = np.where(np.hypot(xy[:, 0] - float(A[0]), xy[:, 1] - float(A[1])) < R_HIT)[0]
            if p["deposit"] == "all":
                pool.append((c, xy))
            elif len(hit):
                pool.append((c, xy[:hit[0] + 1]))
            elif p["deposit"] == "trimmed":
                pool.append((c, xy))
        if pool:
            ids = np.array([ci for ci, _ in pool])
            pr = (ids + 1.0) ** 0.5; pr = pr / pr.sum()
            n = min(p["max_trajs"], len(pool))
            idx = np.random.default_rng(seed * 100 + c).choice(len(pool), n, replace=False, p=pr)
            pts = jnp.array(np.concatenate([pool[j][1][::p["stride"]] for j in idx]))
            wts = jnp.full((pts.shape[0],), float(p["stride"]))
        row = dict(cycle=c,
                   fwd_ratio=float(np.nanmean(Tf)) / T_STAR, fwd_success=float(np.mean(~np.isnan(Tf))),
                   fwd_xcross=float(np.nanmean(xf)),
                   ret_ratio=float(np.nanmean(Tb)) / T_STAR if np.any(~np.isnan(Tb)) else float("nan"),
                   ret_success=float(np.mean(~np.isnan(Tb))))
        hist.append(row)
        if verbose:
            print(f"  cycle {c}: forward T/T*={row['fwd_ratio']:.3f} (hit {row['fwd_success']:.2f}, x_cross={row['fwd_xcross']:.3f})"
                  f" | return T/T*={row['ret_ratio']:.3f} (hit {row['ret_success']:.2f})", flush=True)
    return hist, fwd, bwd


if __name__ == "__main__":
    configs = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"full": {}}
    out = {}
    for name, cfg in configs.items():
        t0 = time.time()
        print(f"== {name}: {cfg}", flush=True)
        h, _, _ = run(cfg)
        out[name] = h
        print(f"   ({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "ablation_results.json", "w"), indent=1)
