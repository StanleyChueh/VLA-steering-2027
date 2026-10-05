"""S2b POST-HOC / EXPLORATORY checks (NOT preregistered; written after the preregistered analysis was run).

They probe how the preregistered results should be read; they cannot change any gate.
  E1  Is the H1 rescue signal treatment-effect heterogeneity or baseline semantic-failure risk?
      Call-0 features are pre-treatment and identical in both arms, so the same features can be scored against
      B failure / B biased-first; and rescue is re-scored within the at-risk sets (B failed; B biased-first).
  E2  Is J (H2) specific to the counterfactual residual or generic chunk roughness? Roughness of the conditioned
      and unconditioned chunks themselves (same J formula on A_cond / A_uncond), partial correlations, joint
      control for L_mean + call index.

  envs/analysis/bin/python src/analysis/s2b_exploratory.py results/S2b > results/S2b/s2b_exploratory.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis import analyze_s2b as A  # noqa: E402
from src.analysis.s2b_data import PRIMARY, load_episodes  # noqa: E402
from src.analysis.stats import auroc  # noqa: E402

FEATS = [f"C_exec|{PRIMARY}@W0", f"C_mean|{PRIMARY}@W0", f"L_early|{PRIMARY}@W0", f"d1|{PRIMARY}@W0",
         f"C_mean|{PRIMARY}@W5", f"C_exec|{PRIMARY}@W5", "m_obs@W0", "m_lang@W0", "H_img@W0", "A_cond_mag@W0"]


def score(pairs, y, f):
    x = np.array([p["feat"][f] for p in pairs], float)
    y = np.asarray(y, bool)
    tasks = np.array([p["task"] for p in pairs])
    seeds = np.array([p["seed"] for p in pairs])
    if y.sum() < 3 or (~y).sum() < 3:
        return None
    r = A.h1_eval_feature(x, y, tasks, seeds, np.zeros(len(y)), [], full_stats=False)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
            if k in ("auroc", "within_task_auroc_signadj", "loto_auroc", "seed0to1_auroc", "seed1to0_auroc")}


def task_id(pairs, y):
    out = {}
    for a, b in ((0, 1), (1, 0)):
        rate = {t: np.mean([yy for p, yy in zip(pairs, y) if p["seed"] == a and p["task"] == t] or [0]) for t in range(13)}
        sb = [(p, yy) for p, yy in zip(pairs, y) if p["seed"] == b]
        out[f"seed{a}to{b}"] = round(auroc([rate[p["task"]] for p, _ in sb], [yy for _, yy in sb]), 3)
    return out


def roughness(chunk_norm):
    return float(np.linalg.norm(np.diff(chunk_norm, axis=0), axis=1).mean())


def main():
    root = Path(sys.argv[1])
    eps = load_episodes(root)
    keys = set()
    for seed in (0, 1):
        _, prs = A.s2b_pair_checks.run(root / f"seed{seed}")
        keys |= {(seed, p["task"], p["init"]) for p in prs if p["valid"]}
    pairs = A.build_pairs(eps, keys)
    res = {"label": "POST-HOC EXPLORATORY; not preregistered; cannot change any gate"}

    # E1
    e1 = {}
    targets = {
        "rescue (all pairs, = preregistered)": (pairs, [p["rescue"] for p in pairs]),
        "B failure (all pairs; baseline risk)": (pairs, [1 - p["succ_B"] for p in pairs]),
        "B biased-first (all pairs; baseline semantic risk)": (pairs, [p["ground_B"] == "biased" for p in pairs]),
    }
    at_risk = [p for p in pairs if p["succ_B"] == 0]
    targets["rescue | B failed (at-risk set)"] = (at_risk, [p["rescue"] for p in at_risk])
    sem_risk = [p for p in pairs if p["ground_B"] == "biased"]
    targets["semantic correction | B biased-first"] = (sem_risk, [p["semcorr"] for p in sem_risk])
    for name, (sub, y) in targets.items():
        e1[name] = {"n": len(sub), "n_pos": int(sum(y)), "n_tasks_with_pos": len({p["task"] for p, yy in zip(sub, y) if yy}),
                    "task_id_seed_transfer": task_id(sub, y), "features": {f: score(sub, y, f) for f in FEATS}}
    # per-task rescue counts among B failures: is rescue a task property?
    e1["rescue_rate_by_task_among_B_failures"] = {
        t: [int(sum(p["rescue"] for p in at_risk if p["task"] == t)), int(sum(p["task"] == t for p in at_risk))] for t in range(13)}
    res["E1"] = e1

    # E2
    rows = []
    for ei, p in enumerate(pairs):
        s = p["S"]
        dr = A.disagreement_rows(s)
        by_call = {}
        for r in dr:
            by_call.setdefault(r["call"], []).append(r["D_comb"])
        for c, ds in by_call.items():
            g = s["policy_calls"][c]["diag"]["guidance"]
            ac, au = np.asarray(g["a_conditioned_model"]), np.asarray(g["a_unconditioned_model"])
            f = p["rows_S"][c]
            rows.append({"ep": ei, "seed": p["seed"], "task": p["task"], "call": c, "D": np.mean(ds),
                         "J": f[f"J|{PRIMARY}"], "L": f[f"L_mean|{PRIMARY}"], "J_cond": roughness(ac),
                         "J_uncond": roughness(au), "J_exec_chunk": roughness(ac + 0.5 * (ac - au))})  # executed CAG chunk (model space)})
    col = lambda k: np.array([r[k] for r in rows], float)  # noqa: E731
    D, J, L, call = col("D"), col("J"), col("L"), col("call")
    e2 = {"n_calls": len(rows)}
    for k in ("J", "J_cond", "J_uncond", "J_exec_chunk", "L"):
        e2[f"rho_{k}_D"] = round(A.spearman(col(k), D), 3)
    e2["partial_J_given_J_cond"] = round(A.partial_spearman(J, D, col("J_cond")), 3)
    e2["partial_J_cond_given_J"] = round(A.partial_spearman(col("J_cond"), D, J), 3)
    # J given L and call index jointly (rank residualisation on both)
    from scipy import stats as st
    rx, ry = st.rankdata(J), st.rankdata(D)
    Z = np.c_[np.ones(len(J)), st.rankdata(L), st.rankdata(call), st.rankdata(col("J_cond"))]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    e2["partial_J_given_L_call_Jcond"] = round(float(np.corrcoef(ex, ey)[0, 1]), 3)
    e2["per_seed_partial_J_given_J_cond"] = {
        s: round(A.partial_spearman(J[col("seed") == s], D[col("seed") == s], col("J_cond")[col("seed") == s]), 3) for s in (0, 1)}
    # decile table: D by J decile
    q = np.quantile(J, np.linspace(0, 1, 11))
    e2["D_by_J_decile"] = [round(float(np.mean(D[(J >= q[i]) & (J <= q[i + 1])])), 3) for i in range(10)]
    res["E2"] = e2
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
