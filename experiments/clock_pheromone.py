"""
Local-only stigmergy on Oettler's two-surface arena: no adjoint control, no hindsight, no goal pull.

Each agent shuttles nest -> food -> nest forever (a continuous, asynchronous stream).
Two scents, mirror-image rules:
    outbound (searching for food):  lays NEST scent,  steers up grad log(c + FOOD scent)
    inbound  (carrying food home):  lays FOOD scent,  steers up grad log(c + NEST scent)
Heading rule (OT paper's Eq. 2, eps = 1, no slowness factor):
    dtheta = beta * grad log(c + S) . (-sin th, cos th) dt + sqrt(2 D dt) xi
Speed 1/nu(x): nu = 1 on the nest (fast) half, nu = ratio on the food (slow) half.

"Clock" deposit: an agent lays exp(-t/tau) per unit length, t = time since it left its source.
tau = inf is ordinary traffic-counting pheromone (the control case).
Scents evaporate at rate `evap` and are read through a Gaussian blur of width sigma.

Arena: box around nest A = (0, 0) and food B = (tan alpha, 1), border y = 0.5, reflecting walls.
"""
import sys, json, time
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def fermat(bx, ratio):
    xs = np.linspace(0, bx, 200001)
    T = np.hypot(0.5, xs) + ratio * np.hypot(0.5, bx - xs)
    i = np.argmin(T)
    return xs[i], T[i], np.hypot(0.5, bx / 2) * (1 + ratio)


def run(alpha=60.0, ratio=2.83, tau=np.inf, n_agents=200, T_total=400.0, dt=0.01, beta=2.0, D=0.5,
        sigma=0.04, h=0.01, evap=0.05, kappa=50.0, margin=0.15, r_hit=0.05, seed=0, snap_every=None, seed_trail=0.0, record=0, k_ref=0.0, delta=0.02):
    rng = np.random.default_rng(seed)
    floor = kappa / (sigma * np.sqrt(2 * np.pi))     # kappa passages' worth of ridge height (Deneubourg-calibrated: 50)
    bx = float(np.tan(np.deg2rad(alpha)))
    x0, x1, y0, y1 = -margin, bx + margin, -margin, 1 + margin
    nx, ny = int(round((x1 - x0) / h)) + 1, int(round((y1 - y0) / h)) + 1
    S = np.zeros((2, nx, ny))                       # 0 = nest scent, 1 = food scent
    A, B = np.array([0.0, 0.0]), np.array([bx, 1.0])

    nu_at = lambda y: np.where(y < 0.5, 1.0, ratio)
    # Speed field with a border of finite width delta (an ant's body straddles the edge).
    v_at = lambda y: 1.0 + (1.0 / ratio - 1.0) * 0.5 * (1 + np.tanh((y - 0.5) / delta))
    dv_at = lambda y: (1.0 / ratio - 1.0) * 0.5 / delta / np.cosh((y - 0.5) / delta) ** 2
    if seed_trail > 0:
        # Bootstrap: an established straight trail worth `seed_trail` passages each way, as if the
        # first route had been set by path integration. After this, only local rules act.
        u = np.linspace(0, 1, 4001); P = A[None] + u[:, None] * (B - A)[None]
        ds = np.hypot(*(B - A)) / (len(u) - 1)
        tA = np.concatenate([[0], np.cumsum(nu_at(P[1:, 1]) * ds)]); tB = tA[-1] - tA
        gi = np.round((P[:, 0] - x0) / h).astype(int); gj = np.round((P[:, 1] - y0) / h).astype(int)
        clk = (lambda tt: np.exp(-tt / tau)) if np.isfinite(tau) else (lambda tt: np.ones_like(tt))
        np.add.at(S[0], (gi, gj), seed_trail * clk(tA) * ds / h**2)
        np.add.at(S[1], (gi, gj), seed_trail * clk(tB) * ds / h**2)
    half = n_agents // 2 if seed_trail > 0 else n_agents
    pos = np.vstack([np.tile(A, (half, 1)), np.tile(B, (n_agents - half, 1))]) + rng.normal(0, 0.01, (n_agents, 2))
    state = np.r_[np.zeros(half, int), np.ones(n_agents - half, int)]   # 0 = outbound, 1 = inbound
    th = np.where(state == 0, np.arctan2(*(B - A)[::-1]), np.arctan2(*(A - B)[::-1])) + rng.normal(0, 0.3, n_agents)
    t_leg = np.zeros(n_agents)                      # time since leaving current source
    start = rng.uniform(0, 20.0, n_agents)          # staggered departures
    xcross = np.full(n_agents, np.nan)              # border crossing of current leg
    trips = []                                      # (time, direction, leg time, crossing x)
    blur = None
    rec = []
    nsteps = int(T_total / dt)
    sv = sigma / h
    for k in range(nsteps):
        t = k * dt
        if k % 5 == 0:
            blur = np.stack([gaussian_filter(S[i], sv, mode="constant") for i in range(2)])
            L = np.log(floor + blur)
            gxF = np.gradient(L, h, axis=1); gyF = np.gradient(L, h, axis=2)
        live = t >= start
        if not live.any():
            continue
        p = pos[live]; s = state[live]; a = th[live]
        ix = (p[:, 0] - x0) / h; iy = (p[:, 1] - y0) / h
        follow = 1 - s                               # outbound follows food scent (1), inbound nest scent (0)
        gx = np.where(follow == 1, map_coordinates(gxF[1], [ix, iy], order=1), map_coordinates(gxF[0], [ix, iy], order=1))
        gy = np.where(follow == 1, map_coordinates(gyF[1], [ix, iy], order=1), map_coordinates(gyF[0], [ix, iy], order=1))
        w = beta * (-gx * np.sin(a) + gy * np.cos(a))
        if k_ref:
            # Local Fermat rule: turn toward the slower side, omega = -k * n . grad v  (grad v = (0, v'(y)))
            w = w - k_ref * np.cos(a) * dv_at(p[:, 1])
            step = dt * v_at(p[:, 1])
        else:
            step = dt / np.where(p[:, 1] < 0.5, 1.0, ratio)
        a = a + w * dt + np.sqrt(2 * D * dt) * rng.normal(size=len(a))
        newp = p + step[:, None] * np.c_[np.cos(a), np.sin(a)]
        # reflecting walls
        for d, lo, hi in ((0, x0 + h, x1 - h), (1, y0 + h, y1 - h)):
            lo_m, hi_m = newp[:, d] < lo, newp[:, d] > hi
            newp[lo_m, d] = 2 * lo - newp[lo_m, d]; newp[hi_m, d] = 2 * hi - newp[hi_m, d]
            if d == 0:
                a[lo_m | hi_m] = np.pi - a[lo_m | hi_m]
            else:
                a[lo_m | hi_m] = -a[lo_m | hi_m]
        # border crossing record (first crossing of y = 0.5 on this leg)
        idx = np.where(live)[0]
        cr = (np.sign(p[:, 1] - 0.5) != np.sign(newp[:, 1] - 0.5)) & np.isnan(xcross[idx])
        if cr.any():
            f = (0.5 - p[cr, 1]) / (newp[cr, 1] - p[cr, 1])
            xcross[idx[cr]] = p[cr, 0] + f * (newp[cr, 0] - p[cr, 0])
        tl = t_leg[live] + dt
        # clock deposit: exp(-t/tau) per unit length, into the scent this leg lays
        amt = (np.exp(-tl / tau) if np.isfinite(tau) else np.ones_like(tl)) * step / (h * h)
        gi = np.clip(np.round((newp[:, 0] - x0) / h).astype(int), 0, nx - 1)
        gj = np.clip(np.round((newp[:, 1] - y0) / h).astype(int), 0, ny - 1)
        np.add.at(S[0], (gi[s == 0], gj[s == 0]), amt[s == 0])   # outbound lays nest scent
        np.add.at(S[1], (gi[s == 1], gj[s == 1]), amt[s == 1])   # inbound lays food scent
        # arrivals: turn around, restart the clock
        goal = np.where(s[:, None] == 0, B, A)
        arr = np.hypot(*(newp - goal).T) < r_hit
        for j in np.where(arr)[0]:
            g = idx[j]
            trips.append((t, int(s[j]), float(tl[j]), float(xcross[g])))
            s[j] = 1 - s[j]; tl[j] = 0.0; a[j] = a[j] + np.pi; xcross[g] = np.nan
        pos[live] = newp; th[live] = a; state[live] = s; t_leg[live] = tl
        S *= (1 - evap * dt)
        if record and k % 20 == 0:
            rec.append(np.c_[pos[:record], state[:record]].copy())
    if record:
        return np.array(trips), blur, (x0, x1, y0, y1), bx, np.array(rec)
    return np.array(trips), blur, (x0, x1, y0, y1), bx


def summarize(trips, bx, ratio, frac=0.5, T_total=None):
    xs, Ts, Tst = fermat(bx, ratio)
    if len(trips) == 0:
        return dict(n=0)
    late = trips[trips[:, 0] > (1 - frac) * trips[:, 0].max()]
    out = {}
    for d, name in ((0, "out"), (1, "in")):
        m = late[late[:, 1] == d]
        x = m[:, 3][np.isfinite(m[:, 3])]
        if len(x) == 0:
            continue
        xmed = float(np.median(x))
        ang = lambda c: np.degrees(np.arctan((bx - c) / 0.5))
        out[name] = dict(n=int(len(m)), x_med=xmed, angle_score=float(100 * (ang(xmed) - ang(xs)) / (ang(bx / 2) - ang(xs))),
                         T_ratio=float(np.median(m[:, 2]) / Ts))
    out.update(x_fermat=float(xs), x_straight=bx / 2, total_trips=int(len(trips)))
    return out


if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1]); res = {}
    for name, cfg in cfgs.items():
        t0 = time.time()
        trips, field, box, bx = run(**cfg)
        s = summarize(trips, bx, cfg.get("ratio", 2.83))
        res[name] = dict(cfg=cfg, summary=s, trips=trips.tolist())
        print(f"{name}: {json.dumps(s)}  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(sys.argv[2], "w"))
        np.save(sys.argv[2].replace(".json", f"_{name}_field.npy"), field)
