"""S2c-0 POST-HOC / EXPLORATORY checks (not preregistered; cannot change the preregistered decision).

  X1  What is G_mag? Per-component faithful-chunk magnitude per task (translation / rotation / gripper).
  X2  Scene-matched task pairs: CF tasks 0/1, 3/11, 4/12 share the identical source LIBERO-Spatial scene, init-state
      file and 4-instruction set and differ only in which instruction is faithful. Metric differences within each
      pair (rescued member minus zero-rescue member).
  X3  Task-level partial Spearman (n = 12) of metrics vs S2b rescue rate, controlling for G_mag.

  envs/analysis/bin/python src/analysis/s2c0_exploratory.py > results/S2c0/s2c0_exploratory.json
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats as st

REPO = Path(__file__).resolve().parents[2]
G0, G1 = [1, 4, 5, 10, 11], [0, 2, 3, 7, 8, 9, 12]
PAIRS = [(0, 1), (3, 11), (12, 4)]  # (rescued member, zero-rescue member), same source scene


def main():
    d = json.loads((REPO / "results/S2c0/s2c0_summary.json").read_text())
    tm, rate = d["task_medians"], d["rescue"]["rate"]
    res = {"label": "POST-HOC EXPLORATORY; not preregistered; cannot change the preregistered decision"}
    # X1
    comp = {}
    for t in range(13):
        vals = {"trans": [], "rot": [], "grip": [], "full": []}
        for i in range(20, 30):
            m = np.load(REPO / f"results/S2c0/chunks/task{t:02d}_init{i:03d}.npz")["model"][:2, 0]  # seeds 0,1, faithful
            vals["trans"].append(np.linalg.norm(m[..., 0:3], axis=-1).mean())
            vals["rot"].append(np.linalg.norm(m[..., 3:6], axis=-1).mean())
            vals["grip"].append(np.abs(m[..., 6]).mean())
            vals["full"].append(np.linalg.norm(m, axis=-1).mean())
        comp[t] = {k: round(float(np.median(v)), 4) for k, v in vals.items()}
    res["X1_G_mag_components"] = comp
    for k in ("trans", "rot", "grip", "full"):
        a, b = [comp[t][k] for t in G0], [comp[t][k] for t in G1]
        gt = sum(x > y for x in a for y in b)
        lt = sum(x < y for x in a for y in b)
        res.setdefault("X1_cliffs_delta", {})[k] = round((gt - lt) / (len(a) * len(b)), 3)
    # X2
    keys = ["SCNR|full", "S_fb|full", "N_f|full", "ALIGN|full", "ALIGN|exec", "ALIGN|trans", "P_cov", "sep_snr", "L_cag", "G_mag"]
    res["X2_scene_pairs"] = {
        f"{r}-{z}": {"rescue_rate": [rate[str(r)], rate[str(z)]],
                     **{k: [round(tm[k][str(r)], 3), round(tm[k][str(z)], 3)] for k in keys}}
        for r, z in PAIRS}
    res["X2_rescued_member_higher"] = {k: sum(tm[k][str(r)] > tm[k][str(z)] for r, z in PAIRS) for k in keys}
    # X3
    tasks = G0 + G1
    y = np.array([rate[str(t)] for t in tasks])
    g = np.array([tm["G_mag"][str(t)] for t in tasks])
    def partial(x):
        rx, ry, rz = st.rankdata(x), st.rankdata(y), st.rankdata(g)
        A = np.c_[np.ones(len(x)), rz]
        ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
        ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        return float(np.corrcoef(ex, ey)[0, 1])
    res["X3_partial_spearman_given_G_mag"] = {
        k: {"rho": round(float(st.spearmanr([tm[k][str(t)] for t in tasks], y).statistic), 3),
            "partial": round(partial(np.array([tm[k][str(t)] for t in tasks])), 3)}
        for k in keys if k != "G_mag"}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
