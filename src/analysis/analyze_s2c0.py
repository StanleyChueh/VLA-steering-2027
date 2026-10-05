"""S2c-0 semantic-controllability analysis, exactly as preregistered in configs/cag/S2c0_prereg.yaml.

  envs/analysis/bin/python src/analysis/analyze_s2c0.py --fig-dir reports/figures

Inputs: results/S2c0/chunks/task*_init*.npz (pre-guidance chunks per seed x condition, model space), S2b episode
records (labels only). Output: results/S2c0/s2c0_summary.json and five figures.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as st

REPO = Path(__file__).resolve().parents[2]
COMPS = {"full": slice(0, 7), "trans": slice(0, 3), "rot": slice(3, 6), "grip": slice(6, 7)}
EPS, NZ, OMEGA = 1e-6, 1e-3, 1.5
G0, G1, EXCL = [1, 4, 5, 10, 11], [0, 2, 3, 7, 8, 9, 12], [6]
TASKS = list(range(13))
F, B, O1, O2, U = range(5)
N_BOOT = 5000
LOWER_IS_WORSE = ["SCNR", "S_fb", "ALIGN", "P_cov", "sep_snr", "spread_snr"]


def nrm(x):
    return float(np.linalg.norm(np.asarray(x).ravel()))


def cos(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.nan if na < NZ or nb < NZ else float(a @ b / (na * nb))


def scene_metrics(m):
    """m: [seeds, 5 conds, 10, 7] model space; seeds index 0, 1 = primary CRN seeds."""
    out = {}
    for cname, sl in COMPS.items():
        A = m[..., sl]
        nf = nrm(A[0, F] - A[1, F])
        sfb = np.mean([nrm(A[s, F] - A[s, B]) for s in (0, 1)])
        out[f"N_f|{cname}"], out[f"S_fb|{cname}"] = nf, float(sfb)
        out[f"SCNR|{cname}"] = float(sfb / (nf + EPS))
        nK = np.mean([nrm(A[i, F] - A[j, F]) for i in range(len(A)) for j in range(i + 1, len(A))])
        out[f"SCNR_K|{cname}"] = float(np.mean([nrm(A[i, F] - A[i, B]) for i in range(len(A))]) / (nK + EPS))
        out[f"N_f_K|{cname}"] = float(nK)
    # alignment
    rc = [m[s, F] - m[s, U] for s in (0, 1)]
    rfb = [m[s, F] - m[s, B] for s in (0, 1)]
    def mcos(fn):
        v = [cos(fn(rc[s]), fn(rfb[s])) for s in (0, 1)]
        return float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    out["ALIGN|full"] = mcos(lambda r: r)
    out["ALIGN|exec"] = mcos(lambda r: r[:5])
    out["ALIGN|trans"] = mcos(lambda r: r[:, 0:3])
    out["ALIGN|rot"] = mcos(lambda r: r[:, 3:6])
    for k in range(10):
        out[f"ALIGN|k{k}"] = mcos(lambda r, k=k: r[k])
    out["ALIGN_nan_seeds"] = int(sum(not np.isfinite(cos(rc[s], rfb[s])) for s in (0, 1)))
    out["P_cov"] = float(np.mean([(OMEGA - 1) * (rc[s].ravel() @ rfb[s].ravel()) / max(nrm(rfb[s]) ** 2, NZ ** 2)
                                  for s in (0, 1)]))
    out["L_cag"] = float(np.mean([nrm(r) for r in rc]))
    out["G_mag"] = float(np.mean([np.linalg.norm(m[s, F], axis=1).mean() for s in (0, 1)]))
    # manifold over the 4 instructions
    sv, er, ev1, pair, sepm, sepmin, du = [], [], [], [], [], [], []
    for s in (0, 1):
        X = m[s, :4].reshape(4, -1)
        Xc = X - X.mean(0)
        sig = np.linalg.svd(Xc, compute_uv=False)[:3]
        p = sig ** 2 / max((sig ** 2).sum(), 1e-30)
        sv.append(sig)
        er.append(float(np.exp(-np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)))))
        ev1.append(float(p[0]))
        d = [nrm(X[i] - X[j]) for i in range(4) for j in range(i + 1, 4)]
        pair.append(d)
        sepm.append(np.mean([nrm(X[0] - X[j]) for j in (1, 2, 3)]))
        sepmin.append(np.min([nrm(X[0] - X[j]) for j in (1, 2, 3)]))
        du.append(nrm(m[s, U].ravel() - X.mean(0)))
    nf = out["N_f|full"]
    out["sv"] = np.mean(sv, 0).tolist()
    out["sv_over_Nf"] = (np.mean(sv, 0) / (nf + EPS)).tolist()
    out["eff_rank"] = float(np.mean(er))
    out["PC1_EV"] = float(np.mean(ev1))
    out["pairwise_mean"] = float(np.mean(pair))
    out["pairwise"] = np.mean(pair, 0).tolist()  # order f-b, f-o1, f-o2, b-o1, b-o2, o1-o2
    out["sep_mean"], out["sep_min"] = float(np.mean(sepm)), float(np.mean(sepmin))
    out["sep_snr"] = float(np.mean(sepm) / (nf + EPS))
    out["spread_snr"] = float(np.mean(pair) / (nf + EPS))
    out["u_to_centroid"] = float(np.mean(du))
    out["u_to_centroid_snr"] = float(np.mean(du) / (nf + EPS))
    # nearest instruction to A_f other than itself, and whether f is closer to b than the noise floor
    out["d_f_b_over_d_f_o"] = float(np.mean([nrm(m[s, F] - m[s, B]) / np.mean([nrm(m[s, F] - m[s, O1]), nrm(m[s, F] - m[s, O2])])
                                            for s in (0, 1)]))
    return out


def load_s2b_labels():
    lab = {}
    for seed in (0, 1):
        recs = {}
        for cond in ("B", "S"):
            for line in open(REPO / f"results/S2b/seed{seed}/{cond}/episodes.jsonl"):
                r = json.loads(line)
                recs[(cond, r["task_id"], r["initial_state_index"])] = r
        for (cond, t, i), r in recs.items():
            if cond != "B":
                continue
            s = recs[("S", t, i)]
            def ground(x):
                ft, bt = x.get("faithful_touch_t"), x.get("biased_touch_t")
                if ft is None and bt is None:
                    return "neither"
                if bt is None or (ft is not None and ft < bt):
                    return "faithful"
                if ft is None or bt < ft:
                    return "biased"
                return "both"
            lab[(seed, t, i)] = {"succ_B": int(r["faithful_success"]), "succ_S": int(s["faithful_success"]),
                                 "ground_B": ground(r), "ground_S": ground(s)}
    for v in lab.values():
        v["rescue"] = int(v["succ_S"] == 1 and v["succ_B"] == 0)
        v["semcorr"] = int(v["ground_B"] == "biased" and v["ground_S"] == "faithful")
    return lab


def cliffs(x0, x1):
    x0, x1 = np.asarray(x0, float), np.asarray(x1, float)
    gt = (x0[:, None] > x1[None, :]).sum()
    lt = (x0[:, None] < x1[None, :]).sum()
    return float((gt - lt) / (len(x0) * len(x1)))


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    m = np.isfinite(s)
    s, y = s[m], y[m]
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    r = st.rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def within_auroc(s, y, g):
    s, y, g = np.asarray(s, float), np.asarray(y, bool), np.asarray(g)
    num = den = 0.0
    for t in np.unique(g):
        mm = g == t
        pos, neg = s[mm & y], s[mm & ~y]
        if len(pos) and len(neg):
            num += (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
            den += len(pos) * len(neg)
    return float(num / den) if den else np.nan


def group_stats(task_val, rescue_rate):
    v0 = np.array([task_val[t] for t in G0])
    v1 = np.array([task_val[t] for t in G1])
    rng = np.random.default_rng(0)
    bs = [np.median(rng.choice(v0, len(v0))) - np.median(rng.choice(v1, len(v1))) for _ in range(N_BOOT)]
    lo, hi = np.quantile(bs, [0.025, 0.975])
    loto = []
    for t in G0 + G1:
        a = [task_val[x] for x in G0 if x != t]
        b = [task_val[x] for x in G1 if x != t]
        loto.append(cliffs(a, b))
    tasks12 = G0 + G1
    x = np.array([task_val[t] for t in tasks12])
    y = np.array([rescue_rate[t] for t in tasks12])
    if len(np.unique(x)) < 2:
        rho = float("nan")
    else:
        rho = float(st.spearmanr(x, y).statistic)
    bsr = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, 12, 12)
        if len(np.unique(x[idx])) > 1 and len(np.unique(y[idx])) > 1:
            bsr.append(st.spearmanr(x[idx], y[idx]).statistic)
    return {
        "G0_values": dict(zip(map(str, G0), map(float, v0))), "G1_values": dict(zip(map(str, G1), map(float, v1))),
        "G0_median": float(np.median(v0)), "G1_median": float(np.median(v1)),
        "cliffs_delta": cliffs(v0, v1), "median_diff": float(np.median(v0) - np.median(v1)),
        "median_diff_ci": [float(lo), float(hi)],
        "loto_delta_range": [float(min(loto)), float(max(loto))], "loto_delta_all_negative": bool(all(d < 0 for d in loto)),
        "loto_delta_all_positive": bool(all(d > 0 for d in loto)),
        "G0_below_G1_median": int(sum(v < np.median(v1) for v in v0)),
        "G0_below_G1_min": [t for t in G0 if task_val[t] < v1.min()],
        "spearman_rescue_rate": rho,
        "spearman_ci": [float(np.quantile(bsr, 0.025)), float(np.quantile(bsr, 0.975))] if bsr else [None, None],
    }


def deficiency(g):
    return (g["cliffs_delta"] <= -0.60 and g["median_diff_ci"][1] < 0 and g["loto_delta_all_negative"]
            and g["G0_below_G1_median"] >= 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(REPO / "results/S2c0/chunks"))
    ap.add_argument("--fig-dir", default=str(REPO / "reports/figures"))
    ap.add_argument("--out", default=str(REPO / "results/S2c0/s2c0_summary.json"))
    a = ap.parse_args()
    checks = json.loads((Path(a.chunks) / "query_checks.json").read_text())["summary"]
    scenes = {}
    for t in TASKS:
        for i in range(20, 30):
            d = np.load(Path(a.chunks) / f"task{t:02d}_init{i:03d}.npz")
            scenes[(t, i)] = scene_metrics(d["model"])
    lab = load_s2b_labels()
    resc = {t: sum(lab[(s, t, i)]["rescue"] for s in (0, 1) for i in range(20, 30)) for t in TASKS}
    bfail = {t: sum(1 - lab[(s, t, i)]["succ_B"] for s in (0, 1) for i in range(20, 30)) for t in TASKS}
    rate = {t: (resc[t] / bfail[t] if bfail[t] else np.nan) for t in TASKS}

    scalar_keys = [k for k, v in scenes[(0, 20)].items() if isinstance(v, float) or isinstance(v, int)]
    task_med = {k: {t: float(np.nanmedian([scenes[(t, i)][k] for i in range(20, 30)])) for t in TASKS} for k in scalar_keys}
    res = {"query_checks": checks, "rescue": {"rescues": resc, "B_failures": bfail, "rate": rate},
           "task_medians": task_med, "groups": {"G0": G0, "G1": G1, "excluded": EXCL}}
    res["nan_counts"] = {k: int(sum(not np.isfinite(scenes[s][k]) for s in scenes)) for k in scalar_keys}
    # group statistics for every scalar metric
    gs = {k: group_stats(task_med[k], rate) for k in scalar_keys}
    res["group_stats"] = gs

    # state-level: AUROC for rescue (B-failed pairs) and semantic correction (B biased-first pairs)
    state = {}
    feats = ["SCNR|full", "S_fb|full", "N_f|full", "ALIGN|full", "ALIGN|exec", "P_cov", "sep_snr", "spread_snr",
             "eff_rank", "G_mag", "L_cag", "SCNR_K|full"]
    for target, cond in (("rescue", lambda v: v["succ_B"] == 0), ("semcorr", lambda v: v["ground_B"] == "biased")):
        keys = [k for k, v in lab.items() if cond(v)]
        y = np.array([lab[k][target] for k in keys])
        tasks = np.array([k[1] for k in keys])
        seeds = np.array([k[0] for k in keys])
        st_t = {"n": len(keys), "n_pos": int(y.sum())}
        for f in feats:
            x = np.array([scenes[(k[1], k[2])][f] for k in keys])
            st_t[f] = {"auroc": auroc(x, y), "within_task": within_auroc(x, y, tasks)}
        tid = {}
        for s0, s1 in ((0, 1), (1, 0)):
            r_ = {t: np.mean([y[j] for j in range(len(y)) if seeds[j] == s0 and tasks[j] == t] or [0]) for t in TASKS}
            mm = seeds == s1
            tid[f"{s0}to{s1}"] = auroc([r_[t] for t in tasks[mm]], y[mm])
        st_t["task_id_seed_transfer"] = tid
        # secondary logistic regression, LOTO
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        X = np.array([[scenes[(k[1], k[2])][f] for f in ("SCNR|full", "ALIGN|full", "P_cov", "sep_snr")] for k in keys])
        X = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
        oof = np.full(len(y), np.nan)
        for t in np.unique(tasks):
            tr = tasks != t
            if len(np.unique(y[tr])) == 2:
                mdl = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(X[tr], y[tr])
                oof[tasks == t] = mdl.predict_proba(X[tasks == t])[:, 1]
        st_t["logreg_loto_auroc"] = auroc(oof, y)
        state[target] = st_t
    res["state_level"] = state

    # preregistered decision
    def comparable(k):
        return gs[k]["cliffs_delta"] > -0.33
    dG, dN = gs["G_mag"]["cliffs_delta"], gs["N_f|full"]["cliffs_delta"]
    def generic_ok(k):
        ok = abs(gs[k]["cliffs_delta"]) > abs(dG) and abs(gs[k]["cliffs_delta"]) > abs(dN)
        if k == "SCNR|full":
            ok &= gs["S_fb|full"]["cliffs_delta"] <= -0.33
        return bool(ok)
    D = {k: deficiency(gs[k]) for k in ("SCNR|full", "sep_snr", "ALIGN|full", "P_cov", "S_fb|full", "spread_snr", "eff_rank")}
    collapse = (D["SCNR|full"] and generic_ok("SCNR|full")) or (D["sep_snr"] and generic_ok("sep_snr"))
    misorient = comparable("SCNR|full") and D["ALIGN|full"] and generic_ok("ALIGN|full")
    amplitude = comparable("SCNR|full") and comparable("ALIGN|full") and D["P_cov"] and generic_ok("P_cov")
    g1min = {k: min(task_med[k][t] for t in G1) for k in ("SCNR|full", "ALIGN|full", "P_cov")}
    low = {k: [t for t in G0 if task_med[k][t] < g1min[k]] for k in g1min}
    patA = low["SCNR|full"]
    patB = [t for t in G0 if t not in patA and (t in low["ALIGN|full"] or t in low["P_cov"])]
    mixed = len(patA) >= 2 and len(patB) >= 2
    if amplitude:
        decision = "A"
    elif collapse or misorient:
        decision = "B"
    elif mixed:
        decision = "C"
    else:
        decision = "D"
    coherent = any(D.values()) or any(abs(gs[k]["spearman_rescue_rate"]) >= 0.5 and
                                      gs[k]["spearman_ci"][0] is not None and
                                      (gs[k]["spearman_ci"][0] > 0 or gs[k]["spearman_ci"][1] < 0)
                                      for k in ("SCNR|full", "ALIGN|full", "P_cov", "sep_snr", "spread_snr", "eff_rank"))
    generic = (dG > -0.6 and dN > -0.6) if decision == "C" else (
        (collapse and (generic_ok("SCNR|full") or generic_ok("sep_snr"))) or
        (misorient and generic_ok("ALIGN|full")) or (amplitude and generic_ok("P_cov")))
    res["decision"] = {
        "deficiency": D, "comparable_SCNR": comparable("SCNR|full"), "comparable_ALIGN": comparable("ALIGN|full"),
        "generic_delta": {"G_mag": dG, "N_f": dN},
        "generic_ok": {k: generic_ok(k) for k in ("SCNR|full", "sep_snr", "ALIGN|full", "P_cov")},
        "collapse": bool(collapse), "misorientation": bool(misorient), "amplitude": bool(amplitude),
        "per_task_low": low, "mixed_patterns": {"low_SCNR": patA, "adequate_SCNR_low_ALIGN_or_Pcov": patB}, "mixed": bool(mixed),
        "final": decision,
        "gate": {"1_coherent": bool(coherent), "2_deficiency_pattern": decision in ("A", "B", "C"),
                 "3_generic_control": bool(generic)},
    }
    res["decision"]["gate"]["GO_S2c1"] = all(res["decision"]["gate"].values())
    res["scenes"] = {f"{t}_{i}": {k: v for k, v in m.items()} for (t, i), m in scenes.items()}
    Path(a.out).write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    figures(scenes, task_med, rate, gs, Path(a.fig_dir))
    print(json.dumps(res["decision"], indent=1))


def figures(scenes, task_med, rate, gs, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C2, GREY = "#2a78d6", "#eb6834", "#8a8984"
    INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.labelcolor": INK})
    fig_dir.mkdir(parents=True, exist_ok=True)
    col = {t: (C1 if t in G1 else C2 if t in G0 else GREY) for t in TASKS}
    rng = np.random.default_rng(0)

    def style(ax):
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    def legend(ax, loc="upper right"):
        from matplotlib.lines import Line2D
        ax.legend([Line2D([], [], marker="o", ls="", color=c) for c in (C1, C2, GREY)],
                  ["rescued tasks (G1)", "zero-rescue tasks (G0)", "task 6 (no B failures)"], frameon=False, fontsize=8, loc=loc)

    def per_task(ax, key, ylabel, logy=False, ref=None):
        for t in TASKS:
            v = [scenes[(t, i)][key] for i in range(20, 30)]
            ax.scatter(t + rng.uniform(-0.15, 0.15, len(v)), v, s=10, color=col[t], alpha=0.45, lw=0)
            ax.plot([t - 0.3, t + 0.3], [task_med[key][t]] * 2, color=col[t], lw=2.5)
        if ref is not None:
            ax.axhline(ref, color=INK2, lw=0.8, ls="--")
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(TASKS)
        ax.set_ylabel(ylabel)
        style(ax)

    # 1. SCNR by task + rescue rate (stacked panels, shared x; no dual axis)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    per_task(a1, "SCNR|full", "SCNR = ‖A_f − A_b‖ / ‖A_f(ξ0) − A_f(ξ1)‖", logy=True, ref=1.0)
    a1.set_title("Semantic control-to-noise ratio per scene (bar = task median); dashed = 1", loc="left", color=INK, fontsize=10)
    from matplotlib.lines import Line2D
    a1.set_ylim(0.5, 60)
    a1.legend([Line2D([], [], marker="o", ls="", color=c) for c in (C1, C2, GREY)],
              ["rescued tasks (G1)", "zero-rescue tasks (G0)", "task 6 (no B failures)"], frameon=False, fontsize=8,
              loc="upper center", ncol=3)
    a1.set_title("Semantic control-to-noise ratio per scene (bar = task median); dashed = 1", loc="left", color=INK,
                 fontsize=10)
    a2.bar(TASKS, [0 if np.isnan(rate[t]) else rate[t] for t in TASKS], color=[col[t] for t in TASKS], width=0.6)
    a2.set_ylabel("S2b rescue rate\n(rescues / B failures)")
    a2.set_xlabel("CF-Spatial task")
    style(a2)
    fig.tight_layout()
    fig.savefig(fig_dir / "S2c0_scnr_by_task.png", dpi=160)
    plt.close(fig)

    # 2. ALIGN by task
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, key, ttl in zip(axes, ("ALIGN|full", "ALIGN|exec"), ("full chunk (k = 0..9)", "executed prefix (k = 0..4)")):
        per_task(ax, key, "cos(R_cag, R_fb)", ref=0.0)
        ax.set_title(f"CAG direction alignment, {ttl}", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("CF-Spatial task")
    legend(axes[0], "lower left")
    fig.tight_layout()
    fig.savefig(fig_dir / "S2c0_align_by_task.png", dpi=160)
    plt.close(fig)

    # 3. rescued vs zero-rescue comparison (small multiples of task medians)
    keys = [("SCNR|full", "SCNR"), ("S_fb|full", "S_fb"), ("N_f|full", "N_f (noise)"), ("ALIGN|full", "ALIGN"),
            ("P_cov", "P_cov"), ("sep_snr", "faithful sep. / N_f"), ("eff_rank", "effective rank"), ("G_mag", "‖A_f‖ (generic)")]
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.8))
    for ax, (k, name) in zip(axes.ravel(), keys):
        for gi, (grp, c) in enumerate(((G1, C1), (G0, C2))):
            v = [task_med[k][t] for t in grp]
            ax.scatter(np.full(len(v), gi) + rng.uniform(-0.08, 0.08, len(v)), v, color=c, s=22)
            ax.plot([gi - 0.25, gi + 0.25], [np.median(v)] * 2, color=c, lw=2.5)
        ax.set_xticks([0, 1], ["G1", "G0"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{name}\nδ = {gs[k]['cliffs_delta']:+.2f}", fontsize=9, color=INK, loc="left")
        style(ax)
    fig.suptitle("Task medians: rescued (G1, blue) vs zero-rescue (G0, orange); δ = Cliff's delta G0 vs G1",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(fig_dir / "S2c0_rescued_vs_zero_rescue.png", dpi=160)
    plt.close(fig)

    # 4. singular-value profiles (relative to policy noise)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for t in TASKS:
        sv = np.median([scenes[(t, i)]["sv_over_Nf"] for i in range(20, 30)], 0)
        ax.plot([1, 2, 3], sv, "-o", color=col[t], ms=4, lw=1.5, alpha=0.9)
        ax.annotate(str(t), (1, sv[0]), xytext=(-12, 0), textcoords="offset points", fontsize=7, color=INK2, va="center")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3], ["σ1", "σ2", "σ3"])
    ax.set_ylabel("singular value / N_f (task median)")
    ax.set_title("Language-conditioned action manifold (4 benchmark instructions)", loc="left", fontsize=10, color=INK)
    style(ax)
    legend(ax)
    fig.tight_layout()
    fig.savefig(fig_dir / "S2c0_manifold_singular_values.png", dpi=160)
    plt.close(fig)

    # 5. faithful-vs-biased distance vs policy-noise distance
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for t in TASKS:
        x = [scenes[(t, i)]["N_f|full"] for i in range(20, 30)]
        y = [scenes[(t, i)]["S_fb|full"] for i in range(20, 30)]
        ax.scatter(x, y, s=14, color=col[t], alpha=0.7, lw=0)
    lim = [0.15, 6]
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("policy-noise distance ‖A_f(ξ0) − A_f(ξ1)‖")
    ax.set_ylabel("faithful-vs-biased distance ‖A_f − A_b‖")
    ax.set_title("Per scene; dashed = equal (SCNR = 1)", loc="left", fontsize=10, color=INK)
    ax.grid(color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    legend(ax, "upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "S2c0_fb_vs_noise_distance.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
