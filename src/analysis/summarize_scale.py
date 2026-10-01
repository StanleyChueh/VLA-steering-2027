"""S1 summary: greedy vs released-code SCALE on LIBERO (episodes.jsonl from scripts/run_scale/run_scale_eval.py).

  envs/openpi-cf/bin/python src/analysis/summarize_scale.py results/S1_minimal > results/S1_minimal/summary.json

Exception episodes are excluded from the behavioural denominators and reported separately.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.stats import wilson  # noqa: E402


def load(d):
    return [json.loads(x) for x in (Path(d) / "episodes.jsonl").read_text().splitlines() if x]


def rate(recs):
    ok = [r for r in recs if not r.get("exception", r["error"] is not None)]
    k = sum(r["success"] for r in ok)
    p, lo, hi = wilson(k, len(ok))
    return {"k": k, "n": len(ok), "p": p, "lo": lo, "hi": hi, "exceptions": len(recs) - len(ok)}


def step_stats(recs, key):
    v = [s[key] for r in recs for s in r["steps"] if s.get(key) is not None]
    if not v:
        return None
    v = np.asarray(v, float)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p05": float(np.quantile(v, 0.05)),
            "p95": float(np.quantile(v, 0.95)), "n": int(len(v))}


def main(root):
    root = Path(root)
    runs = {m: load(root / m) for m in ("greedy", "scale") if (root / m / "episodes.jsonl").exists()}
    out = {"per_task": defaultdict(dict), "pooled": {}, "mechanism": {}, "cost": {}, "geometry": {}}
    for m, recs in runs.items():
        by_task = defaultdict(list)
        for r in recs:
            by_task[r["task_id"]].append(r)
        for t, rs in sorted(by_task.items()):
            out["per_task"][t][m] = rate(rs)
        out["pooled"][m] = rate(recs)
        out["mechanism"][m] = {
            "u_step": step_stats(recs, "u_step"),
            "ema_u": step_stats(recs, "ema_u"),
            "gamma_applied": step_stats(recs, "gamma_applied"),
            "tau_token_mean": (lambda v: None if not v else {"mean": float(np.mean(v)), "median": float(np.median(v)),
                                                            "frac_below_0.1": float(np.mean(np.asarray(v) < 0.1))})(
                [x for r in recs for s in r["steps"] for x in s.get("tau_tokens", [])]),
        }
        lat = [s["latency_ms"] for r in recs for s in r["steps"][1:]]
        out["cost"][m] = {
            "latency_ms_median": float(np.median(lat)),
            "latency_ms_p95": float(np.quantile(lat, 0.95)),
            "peak_cuda_GiB": max(r["peak_cuda_bytes"] for r in recs) / 2**30,
            "episode_length_mean": float(np.mean([r["episode_length"] for r in recs])),
        }
        out["mechanism"][m]["success_vs_mean_u"] = (
            None if m != "scale" else {
                "mean_u_success": float(np.mean([np.mean([s["u_step"] for s in r["steps"]]) for r in recs if r["success"]] or [np.nan])),
                "mean_u_failure": float(np.mean([np.mean([s["u_step"] for s in r["steps"]]) for r in recs if not r["success"] and r["steps"]] or [np.nan])),
            }
        )
    if len(runs) == 2:
        g = {(r["task_id"], r["initial_state_index"]): r for r in runs["greedy"]}
        s = {(r["task_id"], r["initial_state_index"]): r for r in runs["scale"]}
        keys = sorted(set(g) & set(s))
        out["geometry"] = {
            "matched_pairs": len(keys),
            "static_geometry_identical": sum(
                g[k]["geometry"]["static_body_pos_sha256"] == s[k]["geometry"]["static_body_pos_sha256"] for k in keys),
            "init_sim_state_identical": sum(
                g[k]["geometry"]["sim_state_sha256"] == s[k]["geometry"]["sim_state_sha256"] for k in keys),
        }
        trans = defaultdict(int)
        for k in keys:
            trans[f"greedy={'S' if g[k]['success'] else 'F'} -> scale={'S' if s[k]['success'] else 'F'}"] += 1
        out["state_matched_transitions_not_noise_controlled"] = dict(trans)
    out["per_task"] = {str(k): v for k, v in out["per_task"].items()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
