"""S2b B/S pair validity checks under common random numbers (runs in envs/analysis or any numpy env).

For every (task, init) present in both <root>/B/episodes.jsonl and <root>/S/episodes.jsonl:
  * static scene geometry and initial simulator state identical;
  * every call carries the CRN tuple (master_seed, task, init, call index) and the B and S noise fingerprints are
    identical at every call index both episodes reached;
  * call 0 sees the identical observation (client-side hash of both images + state);
  * B's call-0 action chunk equals S's call-0 conditioned branch (pre-CAG) bit-for-bit (env action space);
  * per-call diagnostic noise fingerprints equal the CRN fingerprint, diagnostic re-runs bit-identical,
    CAG reconstruction error small.
Optionally (--replicate <root2>) a second B run is compared for whole-episode determinism.

  envs/analysis/bin/python src/analysis/s2b_pair_checks.py results/S2b/seed0 [--replicate other/root]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    return {(r["task_id"], r["initial_state_index"]): r for r in map(json.loads, Path(path).read_text().splitlines()) if r}


def geometry_key(g):
    return (g.get("static_sha256") or g.get("sha256"), g.get("sim_state_sha256"))


def check_pair(b, s):
    out = {"task": b["task_id"], "init": b["initial_state_index"]}
    out["geometry_match"] = b["geometry"] == s["geometry"] if not isinstance(b["geometry"], dict) else (
        {k: v for k, v in b["geometry"].items() if "sha" in k} == {k: v for k, v in s["geometry"].items() if "sha" in k}
    )
    bc, sc = b["policy_calls"], s["policy_calls"]
    seed = b.get("crn_master_seed")
    tuples_ok = True
    for cond, calls in (("B", bc), ("S", sc)):
        for i, c in enumerate(calls):
            t = c.get("crn") or {}
            tuples_ok &= (t.get("master_seed"), t.get("task_id"), t.get("init_id"), t.get("call_index")) == (
                seed, b["task_id"], b["initial_state_index"], i)
            for br in c["diag"]["branches"].values():
                tuples_ok &= br["noise_sha"] == t.get("noise_sha")
    out["crn_tuples_and_fingerprints_ok"] = bool(tuples_ok)
    n = min(len(bc), len(sc))
    out["common_calls"] = n
    out["noise_identical_all_common_calls"] = all(bc[i]["crn"]["noise_sha"] == sc[i]["crn"]["noise_sha"] for i in range(n))
    out["call0_obs_identical"] = bc[0]["obs_sha"] == sc[0]["obs_sha"]
    out["call0_t_identical"] = bc[0]["t"] == sc[0]["t"]
    b0 = np.asarray(b["predicted_chunks"][0], np.float64)
    s0c = np.asarray(sc[0]["diag"]["guidance"]["a_conditioned"], np.float64)
    out["call0_B_action_eq_S_cond_bit"] = b0.shape == s0c.shape and b0.tobytes() == s0c.tobytes()
    out["call0_B_vs_S_cond_max_abs_diff"] = float(np.max(np.abs(b0 - s0c)))
    # first call index where observations differ (diagnostic of divergence; 1 expected since S executes CAG at call 0)
    out["first_obs_divergence_call"] = next((i for i in range(n) if bc[i]["obs_sha"] != sc[i]["obs_sha"]), None)
    x0 = [br.get("x0_bit_identical", True) for c in bc + sc for br in c["diag"]["branches"].values()]
    out["diag_x0_bit_identical_all"] = all(x0)
    out["cag_reconstruction_max_abs_err"] = max(c["diag"]["guidance"]["reconstruction_max_abs_err"] for c in sc)
    out["exception"] = bool(b.get("exception") or s.get("exception"))
    out["valid"] = (
        out["geometry_match"] and out["crn_tuples_and_fingerprints_ok"] and out["noise_identical_all_common_calls"]
        and out["call0_obs_identical"] and out["call0_B_action_eq_S_cond_bit"] and out["diag_x0_bit_identical_all"]
        and out["cag_reconstruction_max_abs_err"] < 1e-5 and not out["exception"]
    )
    return out


def check_replicate(a, b):
    ea, eb = np.asarray(a["executed_actions"]), np.asarray(b["executed_actions"])
    return {
        "task": a["task_id"], "init": a["initial_state_index"],
        "executed_actions_bit_identical": ea.shape == eb.shape and ea.tobytes() == eb.tobytes(),
        "outcome_identical": all(a[k] == b[k] for k in ("faithful_success", "biased_success", "faithful_touch", "biased_touch", "episode_length")),
        "obs_sha_sequence_identical": [c["obs_sha"] for c in a["policy_calls"]] == [c["obs_sha"] for c in b["policy_calls"]],
    }


def run(root, replicate=None):
    root = Path(root)
    B, S = load(root / "B/episodes.jsonl"), load(root / "S/episodes.jsonl")
    keys = sorted(set(B) & set(S))
    pairs = [check_pair(B[k], S[k]) for k in keys]
    res = {
        "root": str(root), "n_B": len(B), "n_S": len(S), "n_pairs": len(pairs),
        "unpaired": sorted(set(B) ^ set(S)),
        "n_valid": sum(p["valid"] for p in pairs),
        "counts": {k: sum(bool(p[k]) for p in pairs) for k in (
            "geometry_match", "crn_tuples_and_fingerprints_ok", "noise_identical_all_common_calls", "call0_obs_identical",
            "call0_B_action_eq_S_cond_bit", "diag_x0_bit_identical_all", "exception")},
        "max_cag_reconstruction_err": max((p["cag_reconstruction_max_abs_err"] for p in pairs), default=None),
        "first_obs_divergence_call_hist": {str(v): sum(p["first_obs_divergence_call"] == v for p in pairs)
                                           for v in sorted({p["first_obs_divergence_call"] for p in pairs}, key=str)},
        "invalid_pairs": [p for p in pairs if not p["valid"]],
    }
    if replicate:
        R = load(Path(replicate) / "B/episodes.jsonl")
        reps = [check_replicate(B[k], R[k]) for k in sorted(set(B) & set(R))]
        res["replicate"] = {"n": len(reps), "all_bit_identical": all(r["executed_actions_bit_identical"] for r in reps),
                            "rows": reps}
    return res, pairs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--replicate", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    res, pairs = run(a.root, a.replicate)
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": res, "pairs": pairs}, indent=1))
    print(json.dumps(res, indent=1)[:4000])
