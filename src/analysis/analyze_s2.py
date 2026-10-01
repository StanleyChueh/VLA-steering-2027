"""S2 pilot analysis: B (vanilla pi0.5) vs S (CAG-TF) on LIBERO-CF, per configs/cag/S2_pilot_prereg.yaml.

  envs/openpi-cf/bin/python src/analysis/analyze_s2.py results/S2_pilot > results/S2_pilot/summary.json

Wilson 95% intervals per task and pooled for the four primary outcomes; state-matched B->S transition tables
(descriptive, NOT noise-controlled); guidance magnitude; cost; geometry/initial-state matching checks.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.stats import wilson  # noqa: E402

OUTCOMES = ("faithful_touch", "biased_touch", "faithful_success", "biased_success")


def load(d):
    return [json.loads(x) for x in (Path(d) / "episodes.jsonl").read_text().splitlines() if x]


def rates(recs):
    ok = [r for r in recs if not r.get("exception", r["error"] is not None)]
    out = {"n": len(ok), "exceptions": len(recs) - len(ok)}
    for m in OUTCOMES:
        k = sum(bool(r[m]) for r in ok)
        p, lo, hi = wilson(k, len(ok))
        out[m] = {"k": k, "p": p, "lo": lo, "hi": hi}
    out["neither_touched"] = sum((not r["faithful_touch"]) and (not r["biased_touch"]) for r in ok)
    out["both_touched"] = sum(r["faithful_touch"] and r["biased_touch"] for r in ok)
    out["episode_length_mean"] = float(np.mean([r["episode_length"] for r in ok])) if ok else None
    out["policy_calls_mean"] = float(np.mean([r["policy_call_count"] for r in ok])) if ok else None
    return out


def outcome_class(r):
    if r["faithful_success"]:
        return "F-success"
    if r["biased_success"]:
        return "B-success"
    if r["biased_touch"] and not r["faithful_touch"]:
        return "B-touch-only"
    if r["faithful_touch"]:
        return "F-touch-no-success"
    return "no-touch"


def guidance_summary(recs):
    keys = ("translation_norm", "rotation_norm", "gripper_abs_diff", "normalized_combined_norm")
    acc = {k: [] for k in keys}
    per_ep = []
    for r in recs:
        ep = []
        for c in r["policy_calls"]:
            g = c.get("diag", {}).get("guidance")
            if not g:
                continue
            for k in keys:
                acc[k].append(g[k])
            ep.append(np.mean(g["normalized_combined_norm"]))
        if ep:
            per_ep.append({"task_id": r["task_id"], "init": r["initial_state_index"], "class": outcome_class(r),
                           "mean_norm_combined": float(np.mean(ep))})
    if not acc["translation_norm"]:
        return None
    out = {}
    for k in keys:
        a = np.asarray(acc[k])  # [calls, horizon]
        out[k] = {"mean_by_k": a.mean(0).round(5).tolist(), "median_all": float(np.median(a)),
                  "p95_all": float(np.quantile(a, 0.95)), "executed_k0_4_mean": float(a[:, :5].mean())}
    by_class = defaultdict(list)
    for e in per_ep:
        by_class[e["class"]].append(e["mean_norm_combined"])
    out["episode_mean_normalized_combined_by_outcome"] = {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in by_class.items()}
    recon = [c["diag"]["guidance"]["reconstruction_max_abs_err"] for r in recs for c in r["policy_calls"] if c.get("diag", {}).get("guidance")]
    out["reconstruction_max_abs_err"] = float(max(recon))
    return out


def cost(recs):
    calls = [c for r in recs for c in r["policy_calls"]]
    infer = [c["policy_timing"]["infer_ms"] for c in calls[1:]]
    tot = [c["server_info"].get("total_wall_ms", c["server_info"].get("wall_ms")) for c in calls[1:]]
    x0 = [b.get("x0_bit_identical") for c in calls for b in c.get("diag", {}).get("branches", {}).values()]
    return {
        "policy_infer_ms_median": float(np.median(infer)),
        "server_total_ms_median_incl_diagnostics": float(np.median(tot)),
        "client_rtt_ms_median": float(np.median([c["client_rtt_ms"] for c in calls[1:]])),
        "peak_gpu_GiB_incl_diagnostics": max(c["server_info"]["peak_gpu_bytes"] for c in calls) / 2**30,
        "diag_branch_checks": len(x0),
        "diag_x0_bit_identical": sum(bool(v) for v in x0),
    }


def main(root):
    root = Path(root)
    runs = {c: load(root / c) for c in ("B", "S")}
    out = {"label": "PILOT", "rng_protocol": runs["B"][0].get("rng_protocol"), "per_task": {}, "pooled": {}}
    tasks = sorted({r["task_id"] for r in runs["B"]})
    for t in tasks:
        out["per_task"][str(t)] = {c: rates([r for r in recs if r["task_id"] == t]) for c, recs in runs.items()}
    out["pooled"] = {c: rates(recs) for c, recs in runs.items()}

    b = {(r["task_id"], r["initial_state_index"]): r for r in runs["B"]}
    s = {(r["task_id"], r["initial_state_index"]): r for r in runs["S"]}
    keys = sorted(set(b) & set(s))
    trans = {"faithful_success": Counter(), "biased_touch": Counter(), "outcome_class": Counter()}
    per_task_trans = defaultdict(Counter)
    for k in keys:
        for m in ("faithful_success", "biased_touch"):
            trans[m][f"B={int(b[k][m])} -> S={int(s[k][m])}"] += 1
        trans["outcome_class"][f"{outcome_class(b[k])} -> {outcome_class(s[k])}"] += 1
        per_task_trans[k[0]][f"B={int(b[k]['faithful_success'])} -> S={int(s[k]['faithful_success'])}"] += 1
    fb = [k for k in keys if not b[k]["faithful_success"]]
    fs = [k for k in keys if b[k]["faithful_success"]]
    out["state_matched_descriptive_NOT_noise_controlled"] = {
        "pairs": len(keys),
        "faithful_success_transitions": dict(trans["faithful_success"]),
        "biased_touch_transitions": dict(trans["biased_touch"]),
        "outcome_class_transitions": dict(trans["outcome_class"]),
        "faithful_success_transitions_per_task": {str(t): dict(v) for t, v in per_task_trans.items()},
        "rescue_rate": f"{sum(s[k]['faithful_success'] for k in fb)}/{len(fb)}",
        "harm_rate": f"{sum(not s[k]['faithful_success'] for k in fs)}/{len(fs)}",
    }
    out["matching_checks"] = {
        "static_geometry_identical": f"{sum(b[k]['geometry']['static_body_pos_sha256'] == s[k]['geometry']['static_body_pos_sha256'] for k in keys)}/{len(keys)}",
        "init_sim_state_identical": f"{sum(b[k]['geometry']['sim_state_sha256'] == s[k]['geometry']['sim_state_sha256'] for k in keys)}/{len(keys)}",
        "first_call_rng_key_B_eq_S": runs["B"][0]["policy_calls"][0]["server_info"]["rng_key_before"]
        == runs["S"][0]["policy_calls"][0]["server_info"]["rng_key_before"],
        "first_call_noise_sha_B_eq_S": runs["B"][0]["policy_calls"][0]["diag"]["branches"]["cond"]["noise_sha"]
        == runs["S"][0]["policy_calls"][0]["diag"]["branches"]["cond"]["noise_sha"],
    }
    out["guidance"] = guidance_summary(runs["S"])
    out["cost"] = {c: cost(recs) for c, recs in runs.items()}
    out["exceptions"] = {c: [(r["task_id"], r["initial_state_index"], r["error"]) for r in recs if r["error"]] for c, recs in runs.items()}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
