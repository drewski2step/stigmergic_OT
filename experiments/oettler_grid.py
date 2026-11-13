"""Pure-stigmergy refraction test at an Oettler speed ratio: sweep goal pull and time weighting.
Reports the forward crossing point averaged over the last 5 cycles and over seeds."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, intelligence_ablation as ia
gg_list = json.loads(sys.argv[1]); tw_list = json.loads(sys.argv[2]); base = json.loads(sys.argv[3]); out = sys.argv[4]
seeds = range(int(base.pop("seeds", 3))); res = {}
for gg in gg_list:
    for tw in tw_list:
        xc, rt, hit = [], [], []
        for s in seeds:
            h, _, _ = ia.run(dict(base, goal_gain=gg, time_weight=tw), seed=s, verbose=False)
            xc.append(np.nanmean([r["fwd_xcross"] for r in h[-5:]])); rt.append(np.nanmean([r["fwd_ratio"] for r in h[-5:]]))
            hit.append(np.mean([r["fwd_success"] for r in h[-5:]]))
        res[f"gg{gg}_tw{tw}"] = dict(goal_gain=gg, time_weight=tw, xcross=xc, ratio=rt, hit=hit)
        print(f"goal_gain={gg:5} time_weight={tw:4}: crossing x = {np.nanmean(xc):.3f} ± {np.nanstd(xc):.3f} (Fermat {ia.X_STAR:.3f}),"
              f" T/T* = {np.nanmean(rt):.3f}, reached = {np.mean(hit):.2f}", flush=True)
        json.dump(res, open(out, "w"), indent=1)
