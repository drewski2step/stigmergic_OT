"""Run the Snell (OT) experiment with Deneubourg-calibrated parameters over several seeds; print per-cycle means."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, intelligence_ablation as ia
name, cfg, nseed, out = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
runs = [ia.run(cfg, seed=s, verbose=False)[0] for s in range(nseed)]
res = {}
for key in ["fwd_ratio", "fwd_xcross", "fwd_success", "ret_success", "costate_max"]:
    res[key] = [[r[c][key] for r in runs] for c in range(len(runs[0]))]
json.dump(dict(cfg=cfg, **res), open(out, "w"), indent=1)
for c in range(len(runs[0])):
    print(f"{name} cycle {c}: T/T*={np.nanmean(res['fwd_ratio'][c]):.3f}±{np.nanstd(res['fwd_ratio'][c]):.3f}"
          f"  x_cross={np.nanmean(res['fwd_xcross'][c]):.3f}  reached={np.mean(res['fwd_success'][c]):.2f}"
          f"  returns={np.mean(res['ret_success'][c]):.2f}  max|Gamma|={np.nanmax(res['costate_max'][c]):.2g}", flush=True)
