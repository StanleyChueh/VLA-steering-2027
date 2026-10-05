"""S2b causal-discovery analysis, exactly as preregistered in configs/cag/S2b_prereg.yaml.

  envs/analysis/bin/python src/analysis/analyze_s2b.py results/S2b --fig-dir reports/figures

Writes results/S2b/s2b_summary.json (+ pair_checks_seed{m}.json). Sections: validity, effects, consistency, h1, h2,
phase. Nothing is tuned on outcomes; every threshold below is copied from the preregistration.
"""

import argparse
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis import s2b_pair_checks  # noqa: E402
from src.analysis.s2b_data import (  # noqa: E402
    COMPONENTS, PRIMARY, SCALARS, SEEDS, TASKS, UNITS, call_table, disagreement_rows, grounding_class,
    load_episodes, outcome_class, to_norm,
)
from src.analysis.stats import auprc, auroc  # noqa: E402

N_BOOT = 2000
GATE_AUC, GATE_RHO = 0.65, 0.20
STUDY_MIN_POS, STUDY_MIN_TASKS = 15, 3
WINDOWS = ("W0", "W5", "Wall")
COMPARATORS_H1 = ("H_img", "H_vlm", "m_lang", "m_vision", "m_obs", "A_cond_mag")
H2_FEATURES = ("C_mean", "C_exec", "J", "J_exec", "G")
warnings.filterwarnings("ignore", category=RuntimeWarning)


def fnum(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


# ---------------------------------------------------------------------------------------------------------------
# generic statistics
def cluster_boot_idx(clusters, n_boot=N_BOOT, seed=0):
    """clusters: array of cluster labels per row -> list of row-index arrays (resampled clusters)."""
    labels = np.asarray(clusters)
    uniq, inv = np.unique(labels, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    rows = dict(enumerate(np.split(order, np.cumsum(np.bincount(inv))[:-1])))
    uniq = list(range(len(uniq)))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(np.concatenate([rows[uniq[i]] for i in pick]))
    return out


def ci(vals):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)])
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) > 10 else (None, None)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 5 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(st.spearmanr(x, y).statistic)


def partial_spearman(x, y, z):
    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if m.sum() < 10:
        return np.nan
    rx, ry, rz = (st.rankdata(a[m]) for a in (x, y, z))
    A = np.c_[np.ones(m.sum()), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def auroc_nan(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    m = np.isfinite(s)
    return auroc(s[m], y[m])


def within_group_auroc(s, y, g):
    s, y, g = np.asarray(s, float), np.asarray(y, bool), np.asarray(g)
    num = den = 0.0
    for grp in np.unique(g):
        m = (g == grp) & np.isfinite(s)
        pos, neg = s[m & y], s[m & ~y]
        if len(pos) and len(neg):
            num += ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
            den += len(pos) * len(neg)
    return num / den if den else np.nan


# ---------------------------------------------------------------------------------------------------------------
def build_pairs(eps, valid_keys):
    pairs = []
    for (seed, task, init) in sorted(valid_keys):
        b, s = eps[(seed, "B", task, init)], eps[(seed, "S", task, init)]
        rows_s, ev_s = call_table(s)
        rows_b, ev_b = call_table(b)
        p = {"seed": seed, "task": task, "init": init, "state": f"{task}_{init}", "B": b, "S": s,
             "rows_S": rows_s, "rows_B": rows_b, "events_S": ev_s, "events_B": ev_b}
        for c in ("B", "S"):
            r = p[c]
            p[f"succ_{c}"] = int(r["faithful_success"])
            p[f"ft_{c}"], p[f"bt_{c}"] = int(r["faithful_touch"]), int(r["biased_touch"])
            p[f"bsucc_{c}"] = int(r["biased_success"])
            p[f"ground_{c}"], p[f"oclass_{c}"] = grounding_class(r), outcome_class(r)
        p["TE_success"] = p["succ_S"] - p["succ_B"]
        p["TE_faithful_touch"] = p["ft_S"] - p["ft_B"]
        p["TE_biased_touch"] = p["bt_S"] - p["bt_B"]
        p["TE_biased_success"] = p["bsucc_S"] - p["bsucc_B"]
        p["rescue"], p["harm"] = int(p["TE_success"] == 1), int(p["TE_success"] == -1)
        p["semcorr"] = int(p["ground_B"] == "biased" and p["ground_S"] == "faithful")
        # window features (S calls)
        feats = {}
        keys = [k for k in rows_s[0] if "|" in k] + [k for k in COMPARATORS_H1 if k in rows_s[0]]
        for w, sel in (("W0", rows_s[:1]), ("W5", rows_s[:5]), ("Wall", rows_s)):
            for k in keys:
                feats[f"{k}@{w}"] = float(np.nanmean([r[k] for r in sel if k in r]))
        p["feat"] = feats
        pairs.append(p)
    return pairs


# ---------------------------------------------------------------------------------------------------------------
def effects_section(pairs, eps):
    res = {}
    def summarise(sub):
        n = len(sub)
        out = {"n_pairs": n}
        for c in ("B", "S"):
            out[f"success_{c}"] = sum(p[f"succ_{c}"] for p in sub)
            out[f"faithful_touch_{c}"] = sum(p[f"ft_{c}"] for p in sub)
            out[f"biased_touch_{c}"] = sum(p[f"bt_{c}"] for p in sub)
            out[f"biased_success_{c}"] = sum(p[f"bsucc_{c}"] for p in sub)
            out[f"biased_first_{c}"] = sum(p[f"ground_{c}"] == "biased" for p in sub)
        out["rescue"] = sum(p["rescue"] for p in sub)
        out["harm"] = sum(p["harm"] for p in sub)
        out["semantic_correction"] = sum(p["semcorr"] for p in sub)
        out["semantic_regression"] = sum(p["ground_B"] == "faithful" and p["ground_S"] == "biased" for p in sub)
        out["B_failures"] = n - out["success_B"]
        out["B_successes"] = out["success_B"]
        for t in ("TE_faithful_touch", "TE_biased_touch", "TE_biased_success"):
            out[t] = dict(Counter(p[t] for p in sub))
        return out
    for seed in SEEDS:
        sub = [p for p in pairs if p["seed"] == seed]
        res[f"seed{seed}"] = summarise(sub)
        res[f"seed{seed}"]["by_task"] = {t: summarise([p for p in sub if p["task"] == t]) for t in TASKS}
        # McNemar exact (CRN-paired)
        mc = {}
        for name, key in (("success", "succ"), ("faithful_touch", "ft"), ("biased_touch", "bt"), ("biased_success", "bsucc")):
            b10 = sum(p[f"{key}_B"] == 1 and p[f"{key}_S"] == 0 for p in sub)
            b01 = sum(p[f"{key}_B"] == 0 and p[f"{key}_S"] == 1 for p in sub)
            pv = st.binomtest(min(b10, b01), b10 + b01, 0.5).pvalue if b10 + b01 else 1.0
            mc[name] = {"B1_S0": b10, "B0_S1": b01, "p_exact": float(pv)}
        res[f"seed{seed}"]["mcnemar"] = mc
    res["pooled"] = summarise(pairs)
    res["pooled"]["by_task"] = {t: summarise([p for p in pairs if p["task"] == t]) for t in TASKS}
    # pooled state-cluster sign-flip permutation
    rng = np.random.default_rng(0)
    perm = {}
    states = sorted({p["state"] for p in pairs})
    for name, key in (("success", "succ"), ("faithful_touch", "ft"), ("biased_touch", "bt"), ("biased_success", "bsucc")):
        dstate = np.array([sum(p[f"{key}_S"] - p[f"{key}_B"] for p in pairs if p["state"] == s) for s in states])
        obs = dstate.sum()
        flips = rng.choice([-1, 1], size=(10000, len(states)))
        null = (flips * dstate).sum(1)
        perm[name] = {"sum_S_minus_B": int(obs), "p_two_sided": float(np.mean(np.abs(null) >= abs(obs)))}
    res["pooled"]["cluster_signflip"] = perm
    # transitions
    for kind in ("ground", "oclass"):
        res[f"transitions_{kind}"] = {
            f"seed{seed}": dict(Counter(f"{p[f'{kind}_B']} -> {p[f'{kind}_S']}" for p in pairs if p["seed"] == seed))
            for seed in SEEDS
        }
        res[f"transitions_{kind}"]["pooled"] = dict(Counter(f"{p[f'{kind}_B']} -> {p[f'{kind}_S']}" for p in pairs))
    # success-effect x semantic-transition cross table (kept separate but cross-tabulated)
    res["TE_success_by_grounding_transition"] = {
        k: dict(Counter(p["TE_success"] for p in pairs if f"{p['ground_B']} -> {p['ground_S']}" == k))
        for k in sorted({f"{p['ground_B']} -> {p['ground_S']}" for p in pairs})
    }
    # secondary: length / calls
    res["episode_length_mean"] = {c: float(np.mean([p[c]["episode_length"] for p in pairs])) for c in ("B", "S")}
    res["policy_calls_mean"] = {c: float(np.mean([p[c]["policy_call_count"] for p in pairs])) for c in ("B", "S")}
    return res


def consistency_section(pairs, eps):
    by_state = defaultdict(dict)
    for p in pairs:
        by_state[(p["task"], p["init"])][p["seed"]] = p
    classes, detail = Counter(), Counter()
    rows = []
    for (task, init), d in sorted(by_state.items()):
        if len(d) < 2:
            continue
        a, b = d[0]["TE_success"], d[1]["TE_success"]
        if a == b == 1:
            cl = "stable_positive"
        elif a == b == -1:
            cl = "stable_negative"
        elif a == b == 0:
            cl = "neutral"
        else:
            cl = "inconsistent"
        classes[cl] += 1
        detail[f"({a:+d},{b:+d})"] += 1
        rows.append({"task": task, "init": init, "TE_seed0": a, "TE_seed1": b, "class": cl,
                     "sem_seed0": f"{d[0]['ground_B']}->{d[0]['ground_S']}", "sem_seed1": f"{d[1]['ground_B']}->{d[1]['ground_S']}"})
    # expected under independence of seeds given task-level marginals
    exp = Counter()
    for task in TASKS:
        rs = [r for r in rows if r["task"] == task]
        if not rs:
            continue
        p0 = {v: np.mean([r["TE_seed0"] == v for r in rs]) for v in (-1, 0, 1)}
        p1 = {v: np.mean([r["TE_seed1"] == v for r in rs]) for v in (-1, 0, 1)}
        n = len(rs)
        exp["stable_positive"] += n * p0[1] * p1[1]
        exp["stable_negative"] += n * p0[-1] * p1[-1]
        exp["neutral"] += n * p0[0] * p1[0]
    exp["inconsistent"] = len(rows) - sum(exp.values())
    # semantic transitions consistency
    sem = Counter()
    for r in rows:
        c0, c1 = r["sem_seed0"] == "biased->faithful", r["sem_seed1"] == "biased->faithful"
        sem["both" if c0 and c1 else "one" if c0 or c1 else "none"] += 1
    # noise floor: B seed0 vs B seed1 (same scenes, different noise) and S0 vs S1
    def disc(cond):
        out = []
        for (task, init), d in by_state.items():
            if len(d) == 2:
                out.append(int(d[0][f"succ_{cond}"] != d[1][f"succ_{cond}"]))
        return out
    bb, ss = disc("B"), disc("S")
    bs = [int(p["succ_B"] != p["succ_S"]) for p in pairs]
    # B0 vs B1 geometry identity
    geo = []
    for (task, init), d in by_state.items():
        if len(d) == 2:
            g0 = {k: v for k, v in d[0]["B"]["geometry"].items() if "sha" in k}
            g1 = {k: v for k, v in d[1]["B"]["geometry"].items() if "sha" in k}
            geo.append(g0 == g1)
    # B0 vs B1 first-call identity is NOT expected (different noise); report divergence
    return {
        "n_states_both_seeds": len(rows), "classes": dict(classes), "detail": dict(detail),
        "expected_if_seed_independent": {k: float(v) for k, v in exp.items()},
        "semantic_correction_consistency": dict(sem),
        "noise_floor": {
            "B_seed0_vs_B_seed1_success_discordance": [int(sum(bb)), len(bb)],
            "S_seed0_vs_S_seed1_success_discordance": [int(sum(ss)), len(ss)],
            "B_vs_S_within_seed_success_discordance": [int(sum(bs)), len(bs)],
            "B_seed0_vs_B_seed1_grounding_discordance": [
                int(sum(d[0]["ground_B"] != d[1]["ground_B"] for d in by_state.values() if len(d) == 2)), len(bb)],
        },
        "B_seed0_vs_seed1_geometry_identical": [int(sum(geo)), len(geo)],
        "rows": rows,
    }


# ---------------------------------------------------------------------------------------------------------------
def h1_feature_lists(pairs):
    f0 = pairs[0]["feat"]
    primary = [f"{s}|{PRIMARY}@{w}" for w in ("W0", "W5") for s in SCALARS] + [f"d{k}|{PRIMARY}@W0" for k in range(10)]
    comparators = [f"{c}@{w}" for w in ("W0", "W5") for c in COMPARATORS_H1 if f"{c}@{w}" in f0]
    secondary = [f"{s}|{u}|{c}@{w}" for w in ("W0", "W5") for s in SCALARS for u in UNITS for c in COMPONENTS
                 if f"{u}|{c}" != PRIMARY]
    descriptive = [f"{s}|{PRIMARY}@Wall" for s in SCALARS]
    return primary, comparators, secondary, descriptive


def h1_eval_feature(x, y, tasks, seeds, te, boot, full_stats=True):
    x, y, tasks, seeds = np.asarray(x, float), np.asarray(y, bool), np.asarray(tasks), np.asarray(seeds)
    out = {"n": int(np.isfinite(x).sum()), "n_nan": int((~np.isfinite(x)).sum())}
    a = auroc_nan(x, y)
    out["auroc"] = fnum(a)
    sign = 1.0 if (np.isfinite(a) and a >= 0.5) else -1.0
    out["sign"] = int(sign)
    out["auroc_signadj"] = fnum(max(a, 1 - a)) if np.isfinite(a) else None
    m = np.isfinite(x)
    out["auprc"] = fnum(auprc((sign * x)[m], y[m]))
    out["within_task_auroc_signadj"] = fnum(within_group_auroc(sign * x, y, tasks))
    # LOTO
    oof_s, oof_y, signs = [], [], []
    for t in np.unique(tasks):
        tr, tes = (tasks != t), (tasks == t)
        at = auroc_nan(x[tr], y[tr])
        sg = 1.0 if (np.isfinite(at) and at >= 0.5) else -1.0
        signs.append(sg)
        mm = tes & np.isfinite(x)
        oof_s.extend(sg * x[mm])
        oof_y.extend(y[mm])
    out["loto_auroc"] = fnum(auroc(np.asarray(oof_s), np.asarray(oof_y)))
    out["loto_sign_agree"] = int(max(sum(s > 0 for s in signs), sum(s < 0 for s in signs)))
    out["loto_majority_sign"] = int(1 if sum(s > 0 for s in signs) >= sum(s < 0 for s in signs) else -1)
    # seed transfer
    for a_, b_ in ((0, 1), (1, 0)):
        tr, tes = seeds == a_, seeds == b_
        at = auroc_nan(x[tr], y[tr])
        sg = 1.0 if (np.isfinite(at) and at >= 0.5) else -1.0
        out[f"seed{a_}to{b_}_sign"] = int(sg)
        out[f"seed{a_}to{b_}_auroc"] = fnum(auroc_nan(sg * x[tes], y[tes]))
    out["per_seed_auroc"] = {s: fnum(auroc_nan(x[seeds == s], y[seeds == s])) for s in SEEDS}
    if full_stats:
        bs = [auroc_nan(x[i], y[i]) for i in boot]
        lo, hi = ci([v if sign > 0 else 1 - v for v in bs])  # sign fixed at the full-data direction
        out["auroc_signadj_ci"] = [lo, hi]
        out["spearman_TE"] = fnum(spearman(x, te))
        out["spearman_TE_ci"] = list(ci([spearman(x[i], np.asarray(te)[i]) for i in boot[:1000]]))
    return out


def h1_gate(r, comps):
    g = {
        "a_loto": r["loto_auroc"] is not None and r["loto_auroc"] >= GATE_AUC,
        "b_seeds": all(r[f"seed{a}to{b}_auroc"] is not None and r[f"seed{a}to{b}_auroc"] >= GATE_AUC for a, b in ((0, 1), (1, 0))),
        "c_sign": r["loto_sign_agree"] >= 11 and r["seed0to1_sign"] == r["seed1to0_sign"] == r["loto_majority_sign"],
        "d_ci": r.get("auroc_signadj_ci", [None])[0] is not None and r["auroc_signadj_ci"][0] > 0.5,
    }
    g["PASS"] = all(g.values())
    held = [r["loto_auroc"], r["seed0to1_auroc"], r["seed1to0_auroc"]]
    spec = True
    for c in comps:
        if c is None:
            continue
        ch = [c["loto_auroc"], c["seed0to1_auroc"], c["seed1to0_auroc"]]
        spec &= all(h is not None and (cc is None or h > cc) for h, cc in zip(held, ch))
    g["e_specific"] = bool(spec)
    g["between_task_flag"] = r["within_task_auroc_signadj"] is not None and r["within_task_auroc_signadj"] < 0.55
    return g


def multivariate(pairs, y, cols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, export_text

    X = np.array([[p["feat"][c] for c in cols] for p in pairs], float)
    y = np.asarray(y, int)
    tasks = np.array([p["task"] for p in pairs])
    seeds = np.array([p["seed"] for p in pairs])

    def models():
        return {
            "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
            "tree_d3": DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, class_weight="balanced", random_state=0),
        }

    def fit_pred(m, Xtr, ytr, Xte):
        med = np.nanmedian(Xtr, axis=0)
        Xtr, Xte = np.where(np.isfinite(Xtr), Xtr, med), np.where(np.isfinite(Xte), Xte, med)
        if len(np.unique(ytr)) < 2:
            return np.full(len(Xte), np.nan)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]

    out = {}
    for name in ("logreg", "tree_d3"):
        oof = np.full(len(y), np.nan)
        for t in np.unique(tasks):
            tr, te = tasks != t, tasks == t
            oof[te] = fit_pred(models()[name], X[tr], y[tr], X[te])
        r = {"loto_auroc": fnum(auroc_nan(oof, y))}
        for a, b in ((0, 1), (1, 0)):
            pr = fit_pred(models()[name], X[seeds == a], y[seeds == a], X[seeds == b])
            r[f"seed{a}to{b}_auroc"] = fnum(auroc_nan(pr, y[seeds == b]))
        out[name] = r
    m = models()["tree_d3"]
    Xf = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
    if len(np.unique(y)) == 2:
        m.fit(Xf, y)
        out["tree_d3"]["full_data_tree"] = export_text(m, feature_names=[c.replace("|", "_") for c in cols])
        lr = models()["logreg"].fit(Xf, y)
        out["logreg"]["full_data_coef_standardised"] = dict(zip(cols, map(float, lr[-1].coef_[0])))
    return out


def h1_section(pairs):
    primary, comparators, secondary, descriptive = h1_feature_lists(pairs)
    tasks = [p["task"] for p in pairs]
    seeds = [p["seed"] for p in pairs]
    te = [p["TE_success"] for p in pairs]
    boot = cluster_boot_idx([p["state"] for p in pairs])
    boot_task = cluster_boot_idx(tasks, seed=1)
    res = {"features": {"primary": primary, "comparators": comparators, "n_secondary": len(secondary),
                        "descriptive": descriptive}, "targets": {}}
    for target in ("rescue", "harm", "semcorr"):
        y = np.array([p[target] for p in pairs], bool)
        pos_tasks = sorted({p["task"] for p in pairs if p[target]})
        tr = {"n_pos": int(y.sum()), "n": len(y), "base_rate": float(y.mean()), "tasks_with_pos": pos_tasks,
              "studyable": bool(y.sum() >= STUDY_MIN_POS and len(pos_tasks) >= STUDY_MIN_TASKS)}
        if y.sum() == 0 or y.all():
            res["targets"][target] = tr
            continue
        feats = {}
        for f in primary + comparators + descriptive:
            feats[f] = h1_eval_feature([p["feat"][f] for p in pairs], y, tasks, seeds, te, boot)
        for f in secondary:
            feats[f] = h1_eval_feature([p["feat"][f] for p in pairs], y, tasks, seeds, te, boot, full_stats=False)
        # task-ID baseline (seed transfer only)
        tid = {}
        for a, b in ((0, 1), (1, 0)):
            rate = {t: np.mean([p[target] for p in pairs if p["seed"] == a and p["task"] == t] or [0]) for t in TASKS}
            sb = [p for p in pairs if p["seed"] == b]
            tid[f"seed{a}to{b}_auroc"] = fnum(auroc([rate[p["task"]] for p in sb], [p[target] for p in sb]))
        tr["task_id_baseline"] = tid
        # gates on primary features
        gates = {}
        for f in primary:
            w = f.split("@")[1]
            comps = [feats.get(f"A_cond_mag@{w}"), feats.get(f"H_img@{w}")]
            gates[f] = h1_gate(feats[f], comps)
        tr["gates"] = gates
        tr["passing_features"] = [f for f, g in gates.items() if g["PASS"]]
        # top features by LOTO among primary: task-cluster bootstrap robustness
        top = sorted(primary, key=lambda f: -(feats[f]["loto_auroc"] or 0))[:5]
        for f in top:
            x = np.array([p["feat"][f] for p in pairs], float)
            sg = feats[f]["sign"]
            bs = [auroc_nan(sg * x[i], y[i]) for i in boot_task]
            feats[f]["auroc_signadj_task_cluster_ci"] = list(ci(bs))
        tr["top_primary_by_loto"] = top
        tr["features"] = feats
        tr["best_secondary_by_loto"] = sorted(
            [(f, feats[f]["loto_auroc"], feats[f]["seed0to1_auroc"], feats[f]["seed1to0_auroc"]) for f in secondary],
            key=lambda r: -(r[1] or 0))[:8]
        tr["multivariate"] = multivariate(pairs, y, [f"{s}|{PRIMARY}@W0" for s in SCALARS])
        res["targets"][target] = tr
    rs, hm = res["targets"]["rescue"], res["targets"]["harm"]
    passing = {t: res["targets"][t].get("passing_features", []) for t in res["targets"]
               if res["targets"][t].get("studyable")}
    res["gate"] = {
        "rescue_studyable": rs["studyable"], "harm_studyable": hm["studyable"],
        "passing_by_target": passing,
        "PASS": bool(rs["studyable"] and hm["studyable"] and any(passing.values())),
    }
    return res


# ---------------------------------------------------------------------------------------------------------------
def h2_tables(pairs):
    calls, rows = [], []
    for ei, p in enumerate(pairs):
        s = p["S"]
        ct = p["rows_S"]
        dr = disagreement_rows(s)
        by_call = defaultdict(list)
        for r in dr:
            by_call[r["call"]].append(r)
        ch = np.asarray(s["predicted_chunks"], float)
        for c, rr in by_call.items():
            f = ct[c]
            row = {"ep": ei, "seed": p["seed"], "task": p["task"], "call": c,
                   "Dbar": np.mean([r["D_comb"] for r in rr]), "Dbar_cond": np.mean([r["D_cond"] for r in rr]),
                   "Dbar_uncond": np.mean([r["D_uncond"] for r in rr]),
                   "Dbar_trans": np.mean([r["D_trans"] for r in rr]), "Dbar_rot": np.mean([r["D_rot"] for r in rr]),
                   "Dbar_grip": np.mean([r["D_grip"] for r in rr]),
                   "A_exec_mag": float(np.linalg.norm(to_norm(ch[c][5:]), axis=1).mean()),
                   "H_img": f.get("H_img_late", np.nan), "H_vlm": f.get("H_vlm_late", np.nan)}
            for k, v in f.items():
                if "|" in k:
                    row[k] = v
            calls.append(row)
            for r in rr:
                rows.append({"ep": ei, "seed": p["seed"], "task": p["task"], "call": c, "k": r["k"],
                             "D": r["D_comb"], "D_cond": r["D_cond"], "d": f[f"d{r['k']}|{PRIMARY}"],
                             "L_mean": f[f"L_mean|{PRIMARY}"],
                             "H_img_k": float(f["H_img_k"][r["k"]]) if "H_img_k" in f else np.nan})
    return calls, rows


def demean(v, groups):
    v, groups = np.asarray(v, float), np.asarray(groups)
    out = v.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = v[m] - np.nanmean(v[m])
    return out


def h2_eval(x, D, ep, seed, task, Lm, call, Dcond, ep_boot, task_boot, full=True):
    x, D = np.asarray(x, float), np.asarray(D, float)
    out = {"n": int(np.isfinite(x).sum()), "rho": fnum(spearman(x, D))}
    out["per_seed"] = {s: fnum(spearman(x[seed == s], D[seed == s])) for s in SEEDS}
    pt = {}
    for t in np.unique(task):
        m = task == t
        if m.sum() >= 20:
            pt[int(t)] = fnum(spearman(x[m], D[m]))
    out["per_task"] = pt
    sgn = np.sign(out["rho"] or 0)
    out["tasks_same_sign"] = [int(sum(np.sign(v) == sgn for v in pt.values() if v is not None)), len(pt)]
    if full:
        out["ci_episode"] = list(ci([spearman(x[i], D[i]) for i in ep_boot]))
        out["ci_task"] = list(ci([spearman(x[i], D[i]) for i in task_boot]))
        xd, Dd = demean(x, ep), demean(D, ep)
        out["rho_within_episode"] = fnum(spearman(xd, Dd))
        out["rho_within_episode_ci"] = list(ci([spearman(xd[i], Dd[i]) for i in ep_boot[:1000]]))
        out["partial_given_L_mean"] = fnum(partial_spearman(x, D, Lm))
        out["partial_given_call_index"] = fnum(partial_spearman(x, D, call))
        out["rho_vs_D_cond"] = fnum(spearman(x, Dcond))
        out["rho_vs_D_cond_ci"] = list(ci([spearman(x[i], np.asarray(Dcond)[i]) for i in ep_boot[:1000]]))
    return out


def h2_gate(r):
    g = {
        "a_rho_ci": r["rho"] is not None and abs(r["rho"]) >= GATE_RHO and r["ci_episode"][0] is not None
        and (r["ci_episode"][0] > 0 or r["ci_episode"][1] < 0),
        "b_seeds": r["rho"] is not None and all(v is not None and np.sign(v) == np.sign(r["rho"]) for v in r["per_seed"].values()),
        "b_tasks": r["tasks_same_sign"][0] >= 10,
    }
    g["PASS"] = all(g.values())
    if r.get("partial_given_L_mean") is not None:
        g["magnitude_driven"] = abs(r["partial_given_L_mean"]) < 0.10
    if r.get("rho_vs_D_cond_ci") and r["rho_vs_D_cond_ci"][0] is not None:
        lo, hi = r["rho_vs_D_cond_ci"]
        g["mechanical"] = lo <= 0 <= hi
    return g


def h2_section(pairs):
    calls, rows = h2_tables(pairs)
    col = lambda k, src=calls: np.array([r.get(k, np.nan) for r in src], float)  # noqa: E731
    ep, seed, task, call = (np.array([r[k] for r in calls]) for k in ("ep", "seed", "task", "call"))
    D, Dc, Lm = col("Dbar"), col("Dbar_cond"), col(f"L_mean|{PRIMARY}")
    ep_boot = cluster_boot_idx(ep, seed=2)
    task_boot = cluster_boot_idx(task, seed=3)
    res = {"n_call_rows": len(calls), "n_episodes": int(len(np.unique(ep))), "n_row_level": len(rows),
           "D_mean": fnum(np.mean(D)), "D_cond_mean": fnum(np.nanmean(Dc)), "features": {}, "comparators": {}}
    for f in H2_FEATURES:
        res["features"][f] = h2_eval(col(f"{f}|{PRIMARY}"), D, ep, seed, task, Lm, call, Dc, ep_boot, task_boot)
        res["features"][f]["gate"] = h2_gate(res["features"][f])
    for name, x in (("L_mean", Lm), ("H_img (negative control)", col("H_img")), ("H_vlm", col("H_vlm")),
                    ("A_exec_mag", col("A_exec_mag")), ("call_index", call.astype(float))):
        res["comparators"][name] = h2_eval(x, D, ep, seed, task, Lm, call, Dc, ep_boot, task_boot)
    # secondary components / units (point estimates)
    sec = {}
    for f in H2_FEATURES + ("L_mean",):
        for u in UNITS:
            for c in COMPONENTS:
                if f"{u}|{c}" == PRIMARY and f != "L_mean":
                    continue
                sec[f"{f}|{u}|{c}"] = {k: fnum(spearman(col(f"{f}|{u}|{c}"), col(k))) for k in
                                       ("Dbar", "Dbar_trans", "Dbar_rot", "Dbar_grip", "Dbar_cond")}
    res["secondary"] = sec
    # d[k] within-horizon stratification (row level)
    rk = np.array([r["k"] for r in rows])
    rep, rseed, rtask = (np.array([r[k] for r in rows]) for k in ("ep", "seed", "task"))
    rd, rD, rDc, rL = (np.array([r[k] for r in rows], float) for k in ("d", "D", "D_cond", "L_mean"))
    rH = np.array([r["H_img_k"] for r in rows], float)

    def mean_within_k(idx, x=rd, y=rD):
        vals = [spearman(x[idx][rk[idx] == k], y[idx][rk[idx] == k]) for k in range(5, 10)]
        return float(np.nanmean(vals))
    allidx = np.arange(len(rows))
    rboot = cluster_boot_idx(rep, seed=4)
    dk = {"per_k": {int(k): fnum(spearman(rd[rk == k], rD[rk == k])) for k in range(5, 10)},
          "per_k_vs_D_cond": {int(k): fnum(spearman(rd[rk == k], rDc[rk == k])) for k in range(5, 10)},
          "per_k_H_img_negative_control": {int(k): fnum(spearman(rH[rk == k], rD[rk == k])) for k in range(5, 10)},
          "rho": mean_within_k(allidx)}
    dk["ci_episode"] = list(ci([mean_within_k(i) for i in rboot[:1000]]))
    dk["per_seed"] = {s: mean_within_k(np.flatnonzero(rseed == s)) for s in SEEDS}
    pt = {int(t): mean_within_k(np.flatnonzero(rtask == t)) for t in np.unique(rtask) if (rtask == t).sum() >= 100}
    dk["per_task"] = pt
    dk["tasks_same_sign"] = [int(sum(np.sign(v) == np.sign(dk["rho"]) for v in pt.values())), len(pt)]
    rdd, rDd = demean(rd, rep * 100 + rk), demean(rD, rep * 100 + rk)
    dk["rho_within_episode"] = mean_within_k(allidx, rdd, rDd)
    dk["partial_given_L_mean_per_k"] = {int(k): fnum(partial_spearman(rd[rk == k], rD[rk == k], rL[rk == k])) for k in range(5, 10)}
    dk["rho_vs_D_cond"] = mean_within_k(allidx, rd, rDc)
    dk["rho_vs_D_cond_ci"] = list(ci([mean_within_k(i, rd, rDc) for i in rboot[:500]]))
    dk["gate"] = {
        "a_rho_ci": abs(dk["rho"]) >= GATE_RHO and dk["ci_episode"][0] is not None and (dk["ci_episode"][0] > 0 or dk["ci_episode"][1] < 0),
        "b_seeds": all(np.sign(v) == np.sign(dk["rho"]) for v in dk["per_seed"].values()),
        "b_tasks": dk["tasks_same_sign"][0] >= 10,
    }
    dk["gate"]["PASS"] = all(dk["gate"].values())
    dk["gate"]["magnitude_driven"] = "n/a (d[k] is itself a magnitude)"
    lo, hi = dk["rho_vs_D_cond_ci"]
    dk["gate"]["mechanical"] = lo is not None and lo <= 0 <= hi
    res["d_k"] = dk
    passing = [f for f in H2_FEATURES if res["features"][f]["gate"]["PASS"]] + (["d[k]"] if dk["gate"]["PASS"] else [])
    res["gate"] = {"passing": passing, "PASS": bool(passing)}
    return res, calls, rows


# ---------------------------------------------------------------------------------------------------------------
PHASE_Q = {"L_mean": f"L_mean|{PRIMARY}", "C_mean": f"C_mean|{PRIMARY}", "J": f"J|{PRIMARY}",
           "lang_mass": "m_lang", "guidance_mag": "guidance_mag", "H_img": "H_img", "H_vlm": "H_vlm"}
PHASES = ("P0_pre_contact", "P1_contact_pre_close", "P2_post_close")


def phase_section(pairs):
    res = {"phase_call_counts": {}, "episode_means": {}, "paired": {}, "by_TE": {}, "by_outcome": {}}
    cnt = Counter(r["phase"] for p in pairs for r in p["rows_S"])
    res["phase_call_counts"]["S"] = dict(cnt)
    res["phase_call_counts"]["B"] = dict(Counter(r["phase"] for p in pairs for r in p["rows_B"]))
    res["contact_before_close_S"] = dict(Counter(
        "no_contact" if p["events_S"]["t_contact"] is None else "no_close" if p["events_S"]["t_close"] is None
        else ("contact_first" if p["events_S"]["t_contact"] <= p["events_S"]["t_close"] else "close_first") for p in pairs))
    ep_means = []
    for p in pairs:
        m = {"seed": p["seed"], "TE": p["TE_success"], "oclass_S": p["oclass_S"]}
        for ph in PHASES:
            sel = [r for r in p["rows_S"] if r["phase"] == ph]
            for q, key in PHASE_Q.items():
                m[f"{q}@{ph}"] = float(np.nanmean([r[key] for r in sel if key in r])) if sel else np.nan
        selb = {ph: [r for r in p["rows_B"] if r["phase"] == ph] for ph in PHASES}
        for ph in PHASES:
            m[f"B_lang_mass@{ph}"] = float(np.nanmean([r["m_lang"] for r in selb[ph] if "m_lang" in r])) if selb[ph] else np.nan
        ep_means.append(m)
    for q in list(PHASE_Q) + ["B_lang_mass"]:
        res["episode_means"][q] = {}
        for ph in PHASES:
            v = np.array([m[f"{q}@{ph}"] for m in ep_means], float)
            res["episode_means"][q][ph] = {"mean": fnum(np.nanmean(v)), "n_episodes": int(np.isfinite(v).sum())}
        res["paired"][q] = {}
        for ph in PHASES[1:]:
            a = np.array([m[f"{q}@{PHASES[0]}"] for m in ep_means], float)
            b = np.array([m[f"{q}@{ph}"] for m in ep_means], float)
            ok = np.isfinite(a) & np.isfinite(b)
            d = b[ok] - a[ok]
            if len(d) < 5:
                continue
            rng = np.random.default_rng(5)
            bs = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(N_BOOT)]
            per_seed = {}
            seeds_ok = np.array([m["seed"] for m in ep_means])[ok]
            for s in SEEDS:
                ds = d[seeds_ok == s]
                per_seed[s] = {"mean_diff": fnum(np.mean(ds)) if len(ds) else None, "n": int(len(ds))}
            res["paired"][q][f"{ph}_minus_P0"] = {
                "n_episodes": int(len(d)), "mean_diff": fnum(np.mean(d)), "median_diff": fnum(np.median(d)),
                "ci": list(ci(bs)), "wilcoxon_p": fnum(st.wilcoxon(d).pvalue) if np.any(d != 0) else None,
                "rel_change_of_P0_mean": fnum(np.mean(d) / np.mean(a[ok])) if np.mean(a[ok]) else None,
                "per_seed": per_seed,
            }
        for grp_name, key, vals in (("by_TE", "TE", (-1, 0, 1)), ("by_outcome", "oclass_S", sorted({m["oclass_S"] for m in ep_means}))):
            res[grp_name].setdefault(q, {})
            for v in vals:
                sel = [m for m in ep_means if m[key] == v]
                res[grp_name][q][str(v)] = {ph: fnum(np.nanmean([m[f"{q}@{ph}"] for m in sel])) for ph in PHASES}
                res[grp_name][q][str(v)]["n"] = len(sel)
    return res


# ---------------------------------------------------------------------------------------------------------------
def figures(res, pairs, calls, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
    INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF})
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. rescue / harm per task, per seed
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(len(TASKS))
    w = 0.2
    for i, (seed, tgt, col, hatch) in enumerate(((0, "rescue", C1, None), (1, "rescue", C1, "//"),
                                                  (0, "harm", C2, None), (1, "harm", C2, "//"))):
        v = [sum(p[tgt] for p in pairs if p["seed"] == seed and p["task"] == t) for t in TASKS]
        ax.bar(x + (i - 1.5) * w, v, w * 0.9, color=col, hatch=hatch, edgecolor=SURF, linewidth=0.5,
               label=f"{tgt}, seed {seed}")
    ax.set_xticks(x, [str(t) for t in TASKS])
    ax.set_xlabel("CF-Spatial task")
    ax.set_ylabel("pairs (of 10)")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
    ax.set_title("Rescue (S succeeds, B fails) and harm (B succeeds, S fails) per task", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "S2b_rescue_harm_by_task.png", dpi=160)
    plt.close(fig)

    # 2. H1 held-out AUROC (LOTO) for primary W0 features, rescue vs harm
    h1 = res["h1"]["targets"]
    feats = [f"{s}|{PRIMARY}@W0" for s in SCALARS] + [f"{s}|{PRIMARY}@W5" for s in SCALARS] + \
            [f"{c}@W0" for c in ("A_cond_mag", "H_img", "H_vlm", "m_lang")]
    labels = [f.replace(f"|{PRIMARY}", "") for f in feats]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.6), sharey=True)
    for ax, tgt in zip(axes, ("rescue", "harm", "semcorr")):
        t = h1.get(tgt, {})
        ff = t.get("features", {})
        yy = np.arange(len(feats))[::-1]
        for f, yv in zip(feats, yy):
            r = ff.get(f)
            if not r:
                continue
            vals = [v for v in (r["loto_auroc"], r["seed0to1_auroc"], r["seed1to0_auroc"]) if v is not None]
            if vals:
                ax.plot([min(vals), max(vals)], [yv, yv], color=GRID, lw=2)
            for key, mk, col, lab, ms in (("loto_auroc", "o", C1, "LOTO", 5), ("seed0to1_auroc", "s", C2, "seed 0→1", 4),
                                          ("seed1to0_auroc", "^", C3, "seed 1→0", 4)):
                if r[key] is not None:
                    ax.plot(r[key], yv, mk, color=col, ms=ms, label=lab if f == feats[0] else None)
        ax.axvline(0.5, color=INK2, lw=0.8)
        ax.axvline(GATE_AUC, color=INK2, lw=0.8, ls="--")
        ax.set_xlim(0.2, 0.9)
        ax.set_yticks(yy, labels)
        ax.set_title(f"{tgt} (n+={t.get('n_pos', 0)})", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("held-out AUROC")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("H1: held-out AUROC of preregistered features (sign learned on the training fold); "
                 "solid = 0.5, dashed = 0.65 gate", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "S2b_h1_heldout_auroc.png", dpi=160)
    plt.close(fig)

    # 3. H2 rho with CI
    h2 = res["h2"]
    items = [(f, h2["features"][f]) for f in H2_FEATURES] + [("d[k] (within-k)", h2["d_k"])] + \
            list(h2["comparators"].items())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    yy = np.arange(len(items))[::-1]
    for (name, r), yv in zip(items, yy):
        lo, hi = r["ci_episode"]
        if lo is not None:
            ax.plot([lo, hi], [yv, yv], color=C1, lw=2)
        if r["rho"] is not None and np.isfinite(r["rho"]):
            ax.plot(r["rho"], yv, "o", color=C1, ms=5)
        for s, mk in zip(SEEDS, ("s", "^")):
            v = r["per_seed"].get(s)
            if v is not None:
                ax.plot(v, yv - 0.25, mk, color=INK2, ms=3, label=f"seed {s}" if name == items[0][0] else None)
    ax.axvline(0, color=INK2, lw=0.8)
    for g in (-GATE_RHO, GATE_RHO):
        ax.axvline(g, color=INK2, lw=0.8, ls="--")
    ax.set_yticks(yy, [n for n, _ in items])
    ax.set_xlabel("Spearman ρ with future replanning disagreement D")
    ax.set_title("H2: ρ, episode-bootstrap 95% CI; grey = per seed; dashed = ±0.20 gate",
                 color=INK, fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "S2b_h2_rho.png", dpi=160)
    plt.close(fig)

    # 4. d[k] profile by phase (S calls)
    fig, ax = plt.subplots(figsize=(5, 3))
    for ph, col in zip(PHASES, (C1, C2, C3)):
        prof = np.array([[r[f"d{k}|{PRIMARY}"] for k in range(10)] for p in pairs for r in p["rows_S"] if r["phase"] == ph])
        if len(prof):
            ax.plot(range(10), prof.mean(0), "-o", color=col, ms=4, lw=2, label=f"{ph} (n={len(prof)})")
    ax.set_xlabel("horizon index k")
    ax.set_ylabel("mean ‖R[k]‖ (normalized units)")
    ax.grid(color=GRID, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "S2b_residual_profile_by_phase.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--fig-dir", default="reports/figures")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    eps = load_episodes(root)
    validity = {}
    valid_keys = set()
    for seed in SEEDS:
        summ, prs = s2b_pair_checks.run(root / f"seed{seed}")
        (root / f"pair_checks_seed{seed}.json").write_text(json.dumps({"summary": summ, "pairs": prs}, indent=1))
        validity[f"seed{seed}"] = {k: v for k, v in summ.items() if k != "invalid_pairs"}
        validity[f"seed{seed}"]["invalid_pairs"] = [(p["task"], p["init"]) for p in summ["invalid_pairs"]]
        valid_keys |= {(seed, p["task"], p["init"]) for p in prs if p["valid"]}
    # every state / task must be in the frozen design
    assert all(20 <= k[2] <= 29 and k[1] in TASKS for k in valid_keys), "data outside the S2b design"
    pairs = build_pairs(eps, valid_keys)
    # NaN counts of primary features
    nan_counts = {f: int(sum(not np.isfinite(r.get(f, np.nan)) for p in pairs for r in p["rows_S"]))
                  for f in [f"{s}|{PRIMARY}" for s in SCALARS]}
    res = {"validity": validity, "n_valid_pairs": len(pairs),
           "n_S_calls": int(sum(len(p["rows_S"]) for p in pairs)), "primary_feature_nan_calls": nan_counts}
    res["effects"] = effects_section(pairs, eps)
    res["consistency"] = consistency_section(pairs, eps)
    res["h1"] = h1_section(pairs)
    res["h2"], calls, rows = h2_section(pairs)
    res["phase"] = phase_section(pairs)
    # descriptive feature distributions
    res["feature_distribution_W0"] = {
        f: {"median": fnum(np.nanmedian([p["feat"][f"{f}@W0"] for p in pairs])),
            "iqr": [fnum(np.nanquantile([p["feat"][f"{f}@W0"] for p in pairs], q)) for q in (0.25, 0.75)]}
        for f in [f"{s}|{PRIMARY}" for s in SCALARS] + [f"d{k}|{PRIMARY}" for k in range(10)]}
    res["feature_distribution_all_calls"] = {
        f: {"median": fnum(np.nanmedian([r[f] for p in pairs for r in p["rows_S"]]))}
        for f in [f"{s}|{PRIMARY}" for s in SCALARS] + [f"d{k}|{PRIMARY}" for k in range(10)]}
    figures(res, pairs, calls, a.fig_dir)
    out = Path(a.out) if a.out else root / "s2b_summary.json"
    out.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(json.dumps({"n_valid_pairs": len(pairs), "H1": res["h1"]["gate"], "H2": res["h2"]["gate"]}, indent=1))


if __name__ == "__main__":
    main()
