"""S2d oracle-reference analysis, exactly as preregistered in reports/S2D_ORACLE_REFERENCE_PREREG.md.

  envs/analysis/bin/python src/analysis/analyze_s2d.py --fig-dir reports/figures

B and MASK episodes come from results/S2b/seed{m}/{B,S}; SOURCE-NORM / OTHER / SOURCE-RAW from
results/S2d/seed{m}/{SN,OT,SR}. Writes results/S2d/s2d_summary.json.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as st

REPO = Path(__file__).resolve().parents[2]
G0 = [1, 4, 5, 10, 11]
CTRL = [0, 2, 3, 6, 7, 8, 9, 12]
TASKS = list(range(13))
SEEDS = (0, 1)
CONDS = {"B": ("S2b", "B"), "MASK": ("S2b", "S"), "SN": ("S2d", "SN"), "OT": ("S2d", "OT"), "SR": ("S2d", "SR")}
PRIMARY = ["faithful_success", "faithful_first", "biased_first", "biased_success"]
SECONDARY = ["episode_length", "timeout", "post_contact_failure", "policy_calls", "latency_ms", "clip_rate", "roughness"]


def ground(r):
    ft, bt = r.get("faithful_touch_t"), r.get("biased_touch_t")
    if ft is None and bt is None:
        return "neither"
    if bt is None or (ft is not None and ft < bt):
        return "faithful"
    if ft is None or bt < ft:
        return "biased"
    return "both"


def episode_outcomes(r):
    ex = np.asarray(r["executed_actions"], float)
    g = ground(r)
    lat = [c["server_info"].get("policy_wall_ms") or c["server_info"].get("wall_ms") for c in r["policy_calls"][1:]]
    return {
        "faithful_success": int(r["faithful_success"]), "biased_success": int(r["biased_success"]),
        "faithful_first": int(g == "faithful"), "biased_first": int(g == "biased"), "ground": g,
        "episode_length": r["episode_length"], "timeout": int(r["episode_length"] >= 220 and not r["success_any"]),
        "post_contact_failure": int(r["faithful_touch"] and not r["faithful_success"]),
        "policy_calls": r["policy_call_count"],
        "latency_ms": float(np.median([x for x in lat if x is not None])) if lat else np.nan,
        "clip_rate": float(np.mean(np.abs(ex[:, :6]) > 1.0)) if len(ex) else np.nan,
        "clip_n": int(np.sum(np.abs(ex[:, :6]) > 1.0)), "clip_den": int(ex[:, :6].size),
        "roughness": float(np.linalg.norm(np.diff(ex, axis=0), axis=1).mean()) if len(ex) > 1 else np.nan,
    }


def load(root):
    eps = {}
    for cname, (src, d) in CONDS.items():
        for seed in SEEDS:
            p = REPO / f"results/{src}/seed{seed}/{d}/episodes.jsonl"
            if not p.exists():
                continue
            for line in open(p):
                r = json.loads(line)
                eps[(cname, seed, r["task_id"], r["initial_state_index"])] = r
    return eps


def validity(eps):
    """S2d episodes vs the S2b B episode of the same (seed, task, init)."""
    rows = []
    for (c, seed, t, i), r in eps.items():
        if c not in ("SN", "OT", "SR"):
            continue
        b = eps[("B", seed, t, i)]
        geo = {k: v for k, v in r["geometry"].items() if "sha" in k} == {k: v for k, v in b["geometry"].items() if "sha" in k}
        calls = r["policy_calls"]
        c0 = calls[0]
        a0 = np.asarray(c0["diag"]["s2d"]["a_conditioned"], np.float64)
        bc0 = np.asarray(b["predicted_chunks"][0], np.float64)
        tup_ok = all((cc["crn"]["master_seed"], cc["crn"]["task_id"], cc["crn"]["init_id"], cc["crn"]["call_index"]) == (seed, t, i, k)
                     and cc["crn"]["noise_sha"] == cc["diag"]["s2d"]["noise_sha"] for k, cc in enumerate(calls))
        n = min(len(calls), len(b["policy_calls"]))
        noise_eq = all(calls[k]["crn"]["noise_sha"] == b["policy_calls"][k]["crn"]["noise_sha"] for k in range(n))
        norm_err = max((abs(cc["diag"]["s2d"]["intervention_norm"] - cc["diag"]["s2d"]["s_mask"]) for cc in calls
                        if "s_mask" in cc["diag"]["s2d"]), default=0.0)
        ref_ok = all(cc["diag"]["s2d"]["ref_prompt"] == r["s2d_ref_prompt"] for cc in calls)
        row = {"cond": c, "seed": seed, "task": t, "init": i, "geometry": geo, "call0_obs": c0["obs_sha"] == b["policy_calls"][0]["obs_sha"],
               "call0_faithful_eq_B": a0.shape == bc0.shape and a0.tobytes() == bc0.tobytes(), "crn_tuples": tup_ok,
               "noise_eq_B_common_calls": noise_eq, "norm_match_err": norm_err, "ref_prompt_ok": ref_ok,
               "exception": bool(r.get("exception"))}
        row["valid"] = (geo and row["call0_obs"] and row["call0_faithful_eq_B"] and tup_ok and noise_eq and ref_ok
                        and norm_err < 1e-4 and not row["exception"])
        rows.append(row)
    return rows


def paired(out, X, Y, outcome, keys):
    x = np.array([out[(X,) + k][outcome] for k in keys], float)
    y = np.array([out[(Y,) + k][outcome] for k in keys], float)
    d = x - y
    states = sorted({(k[1], k[2]) for k in keys})
    sidx = {s: [j for j, k in enumerate(keys) if (k[1], k[2]) == s] for s in states}
    dstate = np.array([d[sidx[s]].sum() for s in states])
    rng = np.random.default_rng(0)
    flips = rng.choice([-1, 1], size=(10000, len(states)))
    null = (flips * dstate).sum(1)
    p_perm = float(np.mean(np.abs(null) >= abs(dstate.sum()))) if dstate.any() else 1.0
    bs = []
    for _ in range(2000):
        pick = rng.integers(0, len(states), len(states))
        idx = np.concatenate([sidx[states[q]] for q in pick])
        bs.append(d[idx].mean())
    res = {"n": len(keys), "X_rate": float(x.mean()), "Y_rate": float(y.mean()), "diff": float(d.mean()),
           "gain": int((d > 0).sum()), "loss": int((d < 0).sum()), "p_perm": p_perm,
           "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}
    for s in SEEDS:
        ds = np.array([d[j] for j, k in enumerate(keys) if k[0] == s])
        g, l = int((ds > 0).sum()), int((ds < 0).sum())
        res[f"seed{s}"] = {"gain": g, "loss": l, "mcnemar_p": float(st.binomtest(min(g, l), g + l, 0.5).pvalue) if g + l else 1.0}
    res["per_task_net"] = {t: int(sum(d[j] for j, k in enumerate(keys) if k[1] == t)) for t in sorted({k[1] for k in keys})}
    both = Counter()
    for s in states:
        v = [d[j] for j in sidx[s]]
        if len(v) == 2:
            both["both_gain" if min(v) > 0 else "both_loss" if max(v) < 0 else "mixed_or_zero"] += 1
    res["seed_consistency"] = dict(both)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig-dir", default=str(REPO / "reports/figures"))
    ap.add_argument("--out", default=str(REPO / "results/S2d/s2d_summary.json"))
    a = ap.parse_args()
    eps = load(REPO)
    vrows = [r for r in validity(eps) if sum(k[0] == r["cond"] for k in eps) == 260]
    invalid = {(r["seed"], r["task"], r["init"]) for r in vrows if not r["valid"]}
    have = [c for c in CONDS if sum(k[0] == c for k in eps) == 260]  # only conditions with complete data
    keys_all = sorted({(s, t, i) for (c, s, t, i) in eps if all((cc, s, t, i) in eps for cc in have)} - invalid)
    out = {(c,) + k: episode_outcomes(eps[(c,) + k]) for c in have for k in keys_all}
    res = {"conditions_present": have, "n_pairs_valid": len(keys_all),
           "validity": {"n_checked": len(vrows), "n_valid": sum(r["valid"] for r in vrows),
                        "counts": {k: sum(bool(r[k]) for r in vrows) for k in ("geometry", "call0_obs", "call0_faithful_eq_B",
                                                                               "crn_tuples", "noise_eq_B_common_calls", "ref_prompt_ok", "exception")},
                        "max_norm_match_err": max(r["norm_match_err"] for r in vrows), "invalid": sorted(invalid)}}
    groups = {"G0": [k for k in keys_all if k[1] in G0], "control": [k for k in keys_all if k[1] in CTRL], "all": keys_all}
    # rates
    res["rates"] = {g: {c: {o: float(np.mean([out[(c,) + k][o] for k in ks])) for o in PRIMARY + SECONDARY}
                        for c in have} for g, ks in groups.items()}
    res["rates_by_task"] = {t: {c: {o: float(np.mean([out[(c,) + k][o] for k in keys_all if k[1] == t])) for o in PRIMARY}
                                for c in have} for t in TASKS}
    res["clip_pooled"] = {g: {c: sum(out[(c,) + k]["clip_n"] for k in ks) / max(sum(out[(c,) + k]["clip_den"] for k in ks), 1)
                              for c in have} for g, ks in groups.items()}
    comps = [("SN", "MASK"), ("SN", "OT"), ("SR", "SN"), ("SN", "B"), ("OT", "MASK"), ("OT", "B"), ("SR", "MASK"),
             ("SR", "B"), ("MASK", "B")]
    res["paired"] = {g: {f"{x}-{y}": {o: paired(out, x, y, o, ks) for o in PRIMARY}
                         for x, y in comps if x in have and y in have} for g, ks in groups.items()}
    # grounding transitions MASK -> SN in G0
    res["transitions_G0"] = {f"{x}->{y}": dict(Counter(f"{out[(x,)+k]['ground']} -> {out[(y,)+k]['ground']}" for k in groups["G0"]))
                             for x, y in (("MASK", "SN"), ("MASK", "OT"), ("B", "SN")) if x in have and y in have}
    # D(i): per-call intervention norms in G0: SN logged vs MASK (S2b, from logged model-space chunks)
    def call_norms(c):
        v = []
        for k in groups["G0"]:
            for cc in eps[(c,) + k]["policy_calls"]:
                if c == "MASK":
                    g = cc["diag"]["guidance"]
                    v.append(float(np.linalg.norm(0.5 * (np.asarray(g["a_conditioned_model"]) - np.asarray(g["a_unconditioned_model"])))))
                else:
                    v.append(cc["diag"]["s2d"]["intervention_norm"])
        return np.array(v)
    norms = {c: call_norms(c) for c in ("MASK", "SN", "OT", "SR") if c in have}
    res["intervention_norm_G0"] = {c: {"median": float(np.median(v)), "mean": float(v.mean()), "n_calls": len(v)} for c, v in norms.items()}

    # gate
    gate = {}
    if all(c in have for c in ("SN", "MASK", "OT")):
        pa = res["paired"]["G0"]["SN-MASK"]
        A_out = [o for o in ("faithful_success", "faithful_first")
                 if pa[o]["diff"] > 0 and pa[o]["p_perm"] < 0.025 and pa[o]["ci"][0] > 0]
        gate["A"] = {"passing_outcomes": A_out, "PASS": bool(A_out)}
        B_out = [o for o in A_out if sum(v >= 2 for v in pa[o]["per_task_net"].values()) >= 2]
        gate["B"] = {"passing_outcomes": B_out, "PASS": bool(B_out),
                     "tasks_net_ge2": {o: [t for t, v in pa[o]["per_task_net"].items() if v >= 2] for o in ("faithful_success", "faithful_first")}}
        po = res["paired"]["G0"]["SN-OT"]
        C_out = [o for o in (B_out or A_out) if po[o]["diff"] > 0 and po[o]["p_perm"] < 0.05 and po[o]["ci"][0] > 0]
        gate["C"] = {"passing_outcomes": C_out, "PASS": bool(C_out)}
        sn_err = np.median([abs(cc["diag"]["s2d"]["intervention_norm"] - cc["diag"]["s2d"]["s_mask"])
                            for k in groups["G0"] for cc in eps[("SN",) + k]["policy_calls"]])
        ratio = res["intervention_norm_G0"]["SN"]["median"] / res["intervention_norm_G0"]["MASK"]["median"]
        clip_diff = res["clip_pooled"]["G0"]["SN"] - res["clip_pooled"]["G0"]["MASK"]
        gate["D"] = {"median_norm_err": float(sn_err), "median_norm_ratio_SN_over_MASK": float(ratio),
                     "clip_rate_diff_pp": float(100 * clip_diff), "replanning": "h = 5 by construction",
                     "calls_per_episode_G0": {c: res["rates"]["G0"][c]["policy_calls"] for c in have}}
        gate["D"]["PASS"] = bool(sn_err < 1e-4 and 0.9 <= ratio <= 1.1 and clip_diff <= 0.01)
        pc = res["paired"]["control"]["SN-MASK"]["faithful_success"]
        gate["E"] = {"diff": pc["diff"], "p_perm": pc["p_perm"], "PASS": not (pc["diff"] <= -0.05 and pc["p_perm"] < 0.05)}
        if not gate["A"]["PASS"]:
            decision = "C. FAIL"
        elif all(gate[g]["PASS"] for g in "BCDE"):
            decision = "A. PASS"
        else:
            decision = "B. PARTIAL"
        gate["decision"] = decision
    res["gate"] = gate
    res["validity_rows"] = vrows
    Path(a.out).write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    figures(res, have, Path(a.fig_dir))
    print(json.dumps({"n_pairs_valid": len(keys_all), "validity": {k: v for k, v in res["validity"].items() if k != "invalid"},
                      "gate": gate}, indent=1, default=str))


def figures(res, have, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"B": "#8a8984", "MASK": "#2a78d6", "SN": "#eb6834", "OT": "#1baf7a", "SR": "#4a3aa7"}
    MK = {"B": "o", "MASK": "s", "SN": "D", "OT": "^", "SR": "v"}
    NAME = {"B": "B", "MASK": "MASK (CAG-TF)", "SN": "SOURCE-NORM", "OT": "OTHER (norm-matched)", "SR": "SOURCE-RAW"}
    INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.labelcolor": INK})
    fig_dir.mkdir(parents=True, exist_ok=True)
    order = G0 + CTRL
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, o, ttl in zip(axes, ("faithful_success", "faithful_first"), ("faithful success", "faithful object contacted first")):
        for j, c in enumerate([c for c in CONDS if c in have]):
            v = [res["rates_by_task"][t][c][o] for t in order]
            ax.scatter(np.arange(len(order)) + (j - 2) * 0.12, v, marker=MK[c], color=COL[c], s=22, label=NAME[c], zorder=3)
        ax.axvspan(-0.5, len(G0) - 0.5, color=GRID, alpha=0.5, lw=0)
        ax.set_xticks(range(len(order)), [str(t) for t in order])
        ax.set_xlabel("task (shaded: S2b zero-rescue group G0)")
        ax.set_title(ttl + " (20 pairs per task)", loc="left", color=INK, fontsize=10)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("rate")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", ncol=1, bbox_to_anchor=(0.12, 1.0))
    fig.tight_layout()
    fig.savefig(fig_dir / "S2d_rates_by_task.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
