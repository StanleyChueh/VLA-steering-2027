"""Summarise LIBERO-CF runs (episodes.jsonl from scripts/run_cag/run_cf_eval.py).

  python3 src/analysis/summarize_cf.py results/S2_smoke/B results/S2_smoke/S

Prints per-condition faithful/biased touch/success counts, latency and memory, and - for every later condition vs
the first (baseline) - paired rescue/harm over episodes matched on (task_id, initial_state_index):
  rescue = P(method faithful-succeeds | baseline fails)   harm = P(method fails | baseline faithful-succeeds)
"""

import json
import statistics
import sys
from pathlib import Path

METRICS = ("faithful_touch", "biased_touch", "faithful_success", "biased_success")


def load(run_dir):
    return [json.loads(line) for line in (Path(run_dir) / "episodes.jsonl").read_text().splitlines() if line]


def summarize(records):
    n = len(records)
    out = {"n": n}
    for m in METRICS:
        k = sum(bool(r[m]) for r in records)
        out[m] = f"{k}/{n}"
    calls = [c for r in records for c in r["policy_calls"]]
    infer = [c["policy_timing"]["infer_ms"] for c in calls if c.get("policy_timing")]
    rtt = [c["client_rtt_ms"] for c in calls]
    peaks = [c["server_info"]["peak_gpu_bytes"] for c in calls if c.get("server_info")]
    out["episode_length_mean"] = statistics.mean(r["episode_length"] for r in records)
    out["policy_calls_mean"] = statistics.mean(r["policy_call_count"] for r in records)
    out["server_infer_ms_median"] = statistics.median(infer[1:] or infer)  # first call includes JIT compile
    out["client_rtt_ms_median"] = statistics.median(rtt[1:] or rtt)
    out["first_call_ms"] = infer[0] if infer else None
    out["peak_gpu_GiB"] = max(peaks) / 2**30 if peaks else None
    out["rng_key_first_call"] = calls[0]["server_info"]["rng_key_before"] if calls else None
    out["errors"] = sum(r["error"] is not None for r in records)
    return out


def paired(base, method, outcome="faithful_success"):
    b = {(r["task_id"], r["initial_state_index"]): bool(r[outcome]) for r in base}
    m = {(r["task_id"], r["initial_state_index"]): bool(r[outcome]) for r in method}
    keys = sorted(set(b) & set(m))
    b_fail = [k for k in keys if not b[k]]
    b_succ = [k for k in keys if b[k]]
    return {
        "pairs": len(keys),
        "rescue": f"{sum(m[k] for k in b_fail)}/{len(b_fail)}",
        "harm": f"{sum(not m[k] for k in b_succ)}/{len(b_succ)}",
        "discordant_b_only": sum(b[k] and not m[k] for k in keys),
        "discordant_m_only": sum(m[k] and not b[k] for k in keys),
    }


def main(run_dirs):
    runs = {Path(d).name: load(d) for d in run_dirs}
    for name, recs in runs.items():
        print(name, json.dumps(summarize(recs), indent=1))
    names = list(runs)
    for other in names[1:]:
        print(f"paired {other} vs {names[0]} (faithful_success):", json.dumps(paired(runs[names[0]], runs[other])))


if __name__ == "__main__":
    main(sys.argv[1:])
