"""Repeat configs over several seeds; report cycle-1 forward travel time T/T* (mean ± sd) and crossing x."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, intelligence_ablation as ia
configs = json.loads(sys.argv[1]); seeds = range(int(sys.argv[2])); out = {}
for name, cfg in configs.items():
    r, xc, hit = [], [], []
    for s in seeds:
        h, _, _ = ia.run(dict(cfg, num_cycles=2), seed=s, verbose=False)
        r.append(h[1]["fwd_ratio"]); xc.append(h[1]["fwd_xcross"]); hit.append(h[1]["fwd_success"])
    out[name] = dict(ratio=r, xcross=xc, hit=hit)
    print(f"{name:14s} T/T* = {np.nanmean(r):.3f} ± {np.nanstd(r):.3f}   x_cross = {np.nanmean(xc):.3f}   reached = {np.mean(hit):.2f}", flush=True)
json.dump(out, open(sys.argv[3], "w"), indent=1)
