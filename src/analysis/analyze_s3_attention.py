"""S3 attention pilot analysis (DIAGNOSTIC ONLY) on the S2 pilot episodes, per configs/cag/S2_pilot_prereg.yaml.

  envs/openpi-cf/bin/python src/analysis/analyze_s3_attention.py results/S2_pilot \
      --fig-dir reports/figures > results/S2_pilot/s3_summary.json

Primary per-call quantity: obs normalized entropy at the FINAL denoising step, mean over layers and heads, per
horizon index k (H[call, k]). All full-resolution tensors stay on disk; nothing is thresholded.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.stats import auprc, auroc, bootstrap_ci, spearman  # noqa: E402

Q01 = np.array([-0.7474, -0.7961, -0.9375, -0.1158, -0.1694, -0.1945, -1.0])
Q99 = np.array([0.9371, 0.8595, 0.9371, 0.1402, 0.1810, 0.3115, 0.9996])
HALF_RANGE = (Q99 - Q01) / 2
REPLAN = 5
LN968 = np.log(968)
COLORS = {"B": "#2a78d6", "S": "#eb6834"}  # categorical slots 1, 2 (dataviz reference palette, light)
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def load_condition(root, cond):
    recs = [json.loads(x) for x in (root / cond / "episodes.jsonl").read_text().splitlines() if x]
    eps = []
    for r in recs:
        if r.get("exception") or not r.get("attention_file"):
            continue
        a = np.load(r["attention_file"])
        e = {
            "cond": cond, "task": r["task_id"], "init": r["initial_state_index"],
            "faithful_success": r["faithful_success"], "biased_success": r["biased_success"],
            "faithful_touch": r["faithful_touch"], "biased_touch": r["biased_touch"],
            "chunks": np.asarray(r["predicted_chunks"], float),  # [calls, 10, 7] executed-condition output
            "t": np.asarray([c["t"] for c in r["policy_calls"]]),
            # [calls, steps, L, H, k] -> primary per-call H[call, k]
            "obs_full": a["cond/obs_normalized_entropy"],
            "H": a["cond/obs_normalized_entropy"][:, -1].mean(axis=(1, 2)),
            "H_step0": a["cond/obs_normalized_entropy"][:, 0].mean(axis=(1, 2)),
            "H_vlm": a["cond/vlm_normalized_entropy"][:, -1].mean(axis=(1, 2)),
            "obs_mass": a["cond/obs_mass"][:, -1].mean(axis=(1, 2)),
            "kwts": a["cond/kwts_E"][:, -1] / LN968,
            "obs_raw": a["cond/obs_raw_entropy"][:, -1].mean(axis=(1, 2)),
            "valid_obs": int(np.unique(a["cond/obs_valid_key_count"])[0]),
            "valid_vlm": np.unique(a["cond/vlm_valid_key_count"]).tolist(),
        }
        if "uncond/obs_normalized_entropy" in a:
            e["H_uncond"] = a["uncond/obs_normalized_entropy"][:, -1].mean(axis=(1, 2))
        e["D"] = disagreement(e)
        eps.append(e)
    return eps


def disagreement(e):
    """Rows (call c, k in 5..9): a(t+k|t) from call c vs a(t+k|t+5) from call c+1."""
    rows = []
    ch, t = e["chunks"], e["t"]
    for c in range(len(ch) - 1):
        if t[c + 1] - t[c] != REPLAN:
            continue
        for k in range(REPLAN, ch.shape[1]):
            a, b = ch[c, k], ch[c + 1, k - REPLAN]
            d = a - b
            rows.append({
                "call": c, "k": k,
                "trans": float(np.linalg.norm(d[0:3])), "rot": float(np.linalg.norm(d[3:6])),
                "grip": float(abs(d[6])), "comb": float(np.linalg.norm(d / HALF_RANGE)),
                "H": float(e["H"][c, k]), "H_vlm": float(e["H_vlm"][c, k]), "kwts": float(e["kwts"][c, k]),
            })
    return rows


def horizon(eps):
    H = np.concatenate([e["H"] for e in eps])  # [calls, k]
    per_ep_rho = [spearman(np.tile(np.arange(10), len(e["H"])), e["H"].ravel()) for e in eps]
    delta = [float((e["H"][:, 9] - e["H"][:, 0]).mean()) for e in eps]
    lo, hi = bootstrap_ci(lambda g: float(np.mean(g)), delta)
    per_call_rho = [spearman(np.arange(10), h) for e in eps for h in e["H"]]
    out = {
        "n_episodes": len(eps), "n_calls": int(H.shape[0]),
        "H_mean_by_k": H.mean(0).round(4).tolist(),
        "H_std_by_k": H.std(0).round(4).tolist(),
        "delta_k9_minus_k0": {"mean": float(np.mean(delta)), "ci95_episode_bootstrap": [lo, hi]},
        "per_episode_spearman_k_H": {"median": float(np.nanmedian(per_ep_rho)), "frac_positive": float(np.mean(np.asarray(per_ep_rho) > 0))},
        "per_call_spearman_k_H": {"median": float(np.nanmedian(per_call_rho)), "frac_positive": float(np.nanmean(np.asarray(per_call_rho) > 0))},
        "secondary": {
            "H_vlm_mean_by_k": np.concatenate([e["H_vlm"] for e in eps]).mean(0).round(4).tolist(),
            "kwts_Enorm_mean_by_k": np.concatenate([e["kwts"] for e in eps]).mean(0).round(4).tolist(),
            "kwts_Enorm_max_observed": float(np.concatenate([e["kwts"] for e in eps]).max()),
            "obs_mass_mean_by_k": np.concatenate([e["obs_mass"] for e in eps]).mean(0).round(4).tolist(),
            "H_first_denoise_step_mean_by_k": np.concatenate([e["H_step0"] for e in eps]).mean(0).round(4).tolist(),
            "obs_raw_entropy_mean_by_k": np.concatenate([e["obs_raw"] for e in eps]).mean(0).round(4).tolist(),
        },
    }
    full = np.concatenate([e["obs_full"][:, -1] for e in eps])  # [calls, L, H, k]
    d_layer = (full[..., 9] - full[..., 0]).mean(axis=(0, 2))
    out["per_layer_delta_k9_minus_k0"] = d_layer.round(4).tolist()
    out["layers_with_positive_delta"] = int((d_layer > 0).sum())
    return out


def replanning(eps):
    rows_by_ep = [e["D"] for e in eps if e["D"]]
    rows = [r for g in rows_by_ep for r in g]
    out = {"n_pairs": len(rows), "n_episodes": len(rows_by_ep)}
    for dkey in ("comb", "trans", "rot", "grip"):
        for hkey in ("H", "H_vlm", "kwts"):
            rho = spearman([r[hkey] for r in rows], [r[dkey] for r in rows])
            ci = bootstrap_ci(lambda g: spearman([r[hkey] for x in g for r in x], [r[dkey] for x in g for r in x]), rows_by_ep, n_boot=1000)
            out[f"spearman_{hkey}_vs_D_{dkey}"] = {"rho": rho, "ci95": ci}
    # exploratory: within-k (removes the shared dependence of H and D on k) and within-task
    out["exploratory_within_k_spearman_H_vs_D_comb"] = {
        k: spearman([r["H"] for r in rows if r["k"] == k], [r["comb"] for r in rows if r["k"] == k]) for k in range(5, 10)}
    out["D_comb_mean_by_k"] = {k: float(np.mean([r["comb"] for r in rows if r["k"] == k])) for k in range(5, 10)}
    return out


def episode_scalars(e):
    H = e["H"]
    slope = np.polyfit(np.arange(10), H.T, 1)[0]
    return {
        "mean_entropy": float(H.mean()),
        "max_entropy": float(H.mean(1).max()),
        "late_horizon_entropy": float(H[:, 5:].mean()),
        "entropy_slope_over_k": float(slope.mean()),
        "first_call_entropy": float(H[0].mean()),
        "early_entropy_first5_calls_EXPLORATORY": float(H[:5].mean()),
    }


def failure_prediction(eps):
    feats = [episode_scalars(e) for e in eps]
    d_ep = np.array([np.mean([r["comb"] for r in e["D"]]) if e["D"] else np.nan for e in eps])
    q75 = np.nanquantile(d_ep, 0.75)
    targets = {
        "faithful_failure": [not e["faithful_success"] for e in eps],
        "biased_touch": [e["biased_touch"] for e in eps],
        "failed_faithful_grounding": [not e["faithful_touch"] for e in eps],
        "high_disagreement_top_quartile": list(d_ep >= q75),
    }
    tasks = np.array([e["task"] for e in eps])
    out = {}
    for tname, y in targets.items():
        y = np.asarray(y, bool)
        out[tname] = {"base_rate": float(y.mean()), "n_pos": int(y.sum()), "n": int(len(y))}
        for f in feats[0]:
            x = np.array([v[f] for v in feats])
            pairs = list(zip(x, y))
            ci = bootstrap_ci(lambda g: auroc([a for a, _ in g], [b for _, b in g]), pairs, n_boot=1000)
            within = [auroc(x[tasks == t], y[tasks == t]) for t in np.unique(tasks)]
            out[tname][f] = {
                "auroc": auroc(x, y), "auroc_ci95": ci, "auprc": auprc(x, y),
                "auroc_within_task_mean": float(np.nanmean(within)) if np.any(np.isfinite(within)) else float("nan"),
            }
    return out


def by_outcome(eps):
    groups = defaultdict(list)
    for e in eps:
        cls = ("F-success" if e["faithful_success"] else "B-success" if e["biased_success"]
               else "B-touch-only" if e["biased_touch"] and not e["faithful_touch"]
               else "F-touch-no-success" if e["faithful_touch"] else "no-touch")
        groups[cls].append(episode_scalars(e))
    return {k: {"n": len(v), "mean_entropy": float(np.mean([x["mean_entropy"] for x in v])),
                "first_call_entropy": float(np.mean([x["first_call_entropy"] for x in v]))} for k, v in groups.items()}


def figures(all_eps, fig_dir):
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted({e["task"] for e in all_eps})
    k = np.arange(10)

    # 1. H(k): small multiples per task, B vs S; obs (top) and KWTS reference (bottom)
    fig, axes = plt.subplots(2, len(tasks), figsize=(4 * len(tasks), 6.4), sharey="row", squeeze=False)
    for j, t in enumerate(tasks):
        for i, (key, lab) in enumerate((("H", "obs normalized entropy"), ("kwts", "KWTS E / ln 968"))):
            ax = axes[i, j]
            for c in ("B", "S"):
                H = np.concatenate([e[key] for e in all_eps if e["task"] == t and e["cond"] == c])
                m, se = H.mean(0), H.std(0) / np.sqrt(len(H))
                ax.plot(k, m, color=COLORS[c], marker="o", markersize=4, label=c)
                ax.fill_between(k, m - 1.96 * se, m + 1.96 * se, color=COLORS[c], alpha=0.15, linewidth=0)
            ax.set_xticks(k)
            if i == 0:
                ax.set_title(f"task {t}", color=INK)
            if i == 1:
                ax.set_xlabel("action horizon index k")
            if j == 0:
                ax.set_ylabel(lab)
    axes[0, 0].legend(frameon=False, title="condition")
    fig.suptitle("Attention entropy vs horizon k (final denoising step, layer/head mean; band = ±1.96 SE over calls, not episode-clustered)", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "S3_entropy_vs_horizon.png", dpi=150)
    plt.close(fig)

    # 2. layer x k heatmap of obs entropy deviation from each layer's own mean (B, all tasks)
    full = np.concatenate([e["obs_full"][:, -1] for e in all_eps if e["cond"] == "B"]).mean(axis=(0, 2))  # [L, k]
    dev = full - full.mean(1, keepdims=True)
    lim = np.abs(dev).max()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(dev, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xlabel("action horizon index k")
    ax.set_ylabel("layer")
    ax.set_xticks(k)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("H(layer, k) - mean_k H(layer)")
    ax.set_title("Per-layer horizon profile (B, tasks pooled)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "S3_layer_horizon_heatmap.png", dpi=150)
    plt.close(fig)

    # 3. binned H vs D_combined (deciles of H), per condition
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for c in ("B", "S"):
        rows = [r for e in all_eps if e["cond"] == c for r in e["D"]]
        h = np.array([r["H"] for r in rows])
        d = np.array([r["comb"] for r in rows])
        edges = np.quantile(h, np.linspace(0, 1, 11))
        idx = np.clip(np.searchsorted(edges, h, side="right") - 1, 0, 9)
        xm = [h[idx == b].mean() for b in range(10)]
        ym = [d[idx == b].mean() for b in range(10)]
        ys = [d[idx == b].std() / np.sqrt(max(1, (idx == b).sum())) for b in range(10)]
        ax.errorbar(xm, ym, yerr=1.96 * np.asarray(ys), color=COLORS[c], marker="o", markersize=5, capsize=0, label=c)
    ax.set_xlabel("obs normalized entropy H(t, k)  (decile means)")
    ax.set_ylabel("replanning disagreement D_combined")
    ax.legend(frameon=False, title="condition")
    ax.set_title("Entropy vs replanning disagreement (k = 5..9)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "S3_entropy_vs_disagreement.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--fig-dir", default=None)
    args = p.parse_args()
    root = Path(args.root)
    eps = {c: load_condition(root, c) for c in ("B", "S")}
    out = {"label": "PILOT / EXPLORATORY (diagnostic only; no threshold)", "valid_key_counts": {}}
    for c, es in eps.items():
        out["valid_key_counts"][c] = {"obs": sorted({e["valid_obs"] for e in es}),
                                      "vlm": sorted({v for e in es for v in e["valid_vlm"]})}
        out[c] = {
            "horizon": horizon(es),
            "horizon_per_task": {str(t): horizon([e for e in es if e["task"] == t])["H_mean_by_k"] for t in sorted({e["task"] for e in es})},
            "replanning": replanning(es),
            "failure_prediction": failure_prediction(es),
            "entropy_by_outcome": by_outcome(es),
        }
        if es and "H_uncond" in es[0]:
            hc = np.concatenate([e["H"] for e in es])
            hu = np.concatenate([e["H_uncond"] for e in es])
            out[c]["cond_minus_uncond_H_mean_by_k"] = (hc - hu).mean(0).round(4).tolist()
    if args.fig_dir:
        figures(eps["B"] + eps["S"], Path(args.fig_dir))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
