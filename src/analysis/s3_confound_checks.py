"""EXPLORATORY confound checks for the S3 pilot (written after seeing s3_summary.json; not preregistered).

  envs/openpi-cf/bin/python src/analysis/s3_confound_checks.py results/S2_pilot > results/S2_pilot/s3_confound_checks.json

1. Is the entropy-slope / failure association explained by episode length (failures run to the 220-step timeout and
   accumulate late-phase calls)? -> slope restricted to the first 5 calls (before any outcome), Spearman with length,
   and AUROC of episode length itself.
2. Is the H_vlm / replanning-disagreement correlation a between-episode or episode-phase effect? -> Spearman after
   demeaning H and D within episode, and within normalised episode-phase bins.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.analyze_s3_attention import load_condition  # noqa: E402
from src.analysis.stats import auroc, bootstrap_ci, spearman  # noqa: E402


def slope(H):
    return float(np.polyfit(np.arange(10), H.T, 1)[0].mean())


def main(root):
    out = {"label": "EXPLORATORY (post hoc)"}
    for c in ("B", "S"):
        eps = load_condition(Path(root), c)
        y = np.array([not e["faithful_success"] for e in eps])
        length = np.array([len(e["t"]) for e in eps], float)
        s_all = np.array([slope(e["H"]) for e in eps])
        s_early = np.array([slope(e["H"][:5]) for e in eps])
        # slope as a function of episode phase (all episodes): call index bins
        r = {
            "auroc_faithful_failure": {
                "slope_all_calls": auroc(s_all, y),
                "slope_first5_calls": auroc(s_early, y),
                "slope_first5_ci95": bootstrap_ci(lambda g: auroc([a for a, _ in g], [b for _, b in g]), list(zip(s_early, y)), n_boot=1000),
                "episode_length_calls": auroc(length, y),
            },
            "spearman_slope_vs_length": spearman(s_all, length),
        }
        # per-call slope vs normalised phase
        ph, sl = [], []
        for e in eps:
            n = len(e["H"])
            for i, h in enumerate(e["H"]):
                ph.append(i / max(1, n - 1))
                sl.append(float(np.polyfit(np.arange(10), h, 1)[0]))
        r["spearman_call_slope_vs_episode_phase"] = spearman(ph, sl)
        r["spearman_call_slope_vs_call_index"] = spearman([i for e in eps for i in range(len(e["H"]))], sl)
        # H_vlm vs D: within-episode demeaned, and within absolute call-index bins
        rows = [(e_i, d["call"], d["H_vlm"], d["H"], d["comb"]) for e_i, e in enumerate(eps) for d in e["D"]]
        rows = np.array(rows, float)
        dm = rows.copy()
        for e_i in np.unique(rows[:, 0]):
            m = rows[:, 0] == e_i
            dm[m, 2:] -= rows[m, 2:].mean(0)
        r["H_vlm_vs_D_comb"] = {
            "pooled": spearman(rows[:, 2], rows[:, 4]),
            "within_episode_demeaned": spearman(dm[:, 2], dm[:, 4]),
            "within_call_index_bins_mean": float(np.nanmean([spearman(rows[rows[:, 1] == b, 2], rows[rows[:, 1] == b, 4])
                                                             for b in np.unique(rows[:, 1]) if (rows[:, 1] == b).sum() > 30])),
        }
        r["H_obs_vs_D_comb"] = {
            "pooled": spearman(rows[:, 3], rows[:, 4]),
            "within_episode_demeaned": spearman(dm[:, 3], dm[:, 4]),
        }
        out[c] = r
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
