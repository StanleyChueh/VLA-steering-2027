"""S2b data loading and preregistered counterfactual-action features (configs/cag/S2b_prereg.yaml).

Runs in envs/analysis. Per S policy call: R[k] = A_cond[k] - A_uncond[k] (k = 0..9) in normalized (model quantile
space, dims 0-6) and env units, per component, and the F1-F5 features. Per call and branch: attention summaries.
"""

import json
from pathlib import Path

import numpy as np

Q01 = np.array([-0.7474, -0.7961, -0.9375, -0.1158, -0.1694, -0.1945, -1.0])
Q99 = np.array([0.9371, 0.8595, 0.9371, 0.1402, 0.1810, 0.3115, 0.9996])
HALF_RANGE = (Q99 - Q01) / 2  # S3 D_combined scaling
OMEGA = 1.5
REPLAN = 5
WAIT = 10
EPS = 1e-3  # near-zero rule, normalized units
COMPONENTS = {"comb": slice(0, 7), "trans": slice(0, 3), "rot": slice(3, 6), "grip": slice(6, 7)}
UNITS = ("norm", "env")
SCALARS = ("L_mean", "L_early", "L_exec", "G", "C_mean", "C_exec", "J", "J_exec")
PRIMARY = "norm|comb"
TASKS = list(range(13))
SEEDS = (0, 1)


def to_norm(a_env):
    """Env action -> model quantile-normalised space (inverse of openpi Unnormalize, use_quantiles=True)."""
    return 2.0 * (np.asarray(a_env, float) - Q01) / (Q99 - Q01 + 1e-6) - 1.0


def _cos_mean(V, Vn, ks):
    """mean_k cos(V[k], mean_k V) over ks; near-zero decided on the normalized-unit vectors Vn."""
    ref, refn = V[ks].mean(0), Vn[ks].mean(0)
    if np.linalg.norm(refn) < EPS:
        return np.nan
    vals = []
    for k in ks:
        if np.linalg.norm(Vn[k]) < EPS:
            continue
        vals.append(float(V[k] @ ref / (np.linalg.norm(V[k]) * np.linalg.norm(ref))))
    return float(np.mean(vals)) if vals else np.nan


def residual_features(Rn, Re):
    """Rn, Re: [10, 7] residual in normalized / env units -> {f'{name}|{unit}|{comp}': value} incl. d0..d9."""
    out = {}
    kk = np.arange(Rn.shape[0])
    for unit, R in (("norm", Rn), ("env", Re)):
        for comp, sl in COMPONENTS.items():
            V, Vn = R[:, sl], Rn[:, sl]
            d = np.linalg.norm(V, axis=1)
            tag = f"{unit}|{comp}"
            for k in kk:
                out[f"d{k}|{tag}"] = float(d[k])
            out[f"L_mean|{tag}"] = float(d.mean())
            out[f"L_early|{tag}"] = float(d[:3].mean())
            out[f"L_exec|{tag}"] = float(d[:5].mean())
            out[f"G|{tag}"] = float(np.polyfit(kk, d, 1)[0])
            out[f"C_mean|{tag}"] = _cos_mean(V, Vn, list(kk))
            out[f"C_exec|{tag}"] = _cos_mean(V, Vn, list(range(5)))
            dif = np.linalg.norm(np.diff(V, axis=0), axis=1)
            out[f"J|{tag}"] = float(dif.mean())
            out[f"J_exec|{tag}"] = float(dif[:4].mean())
    return out


def _attn_call_summaries(path, branch="cond"):
    """[calls, k] summaries at the final denoising step, mean over layers and heads."""
    if not path or not Path(path).exists():
        return None
    a = np.load(path)
    out = {}
    for name, key in (("H_img", "obs_normalized_entropy"), ("H_vlm", "vlm_normalized_entropy"),
                      ("m_vision", "obs_mass"), ("m_lang", "lang_mass"), ("m_obs", "prefix_mass")):
        full = f"{branch}/{key}"
        if full in a:
            out[name] = a[full][:, -1].mean(axis=(1, 2))  # [calls, k]
    return out


def load_episodes(root):
    """root/seed{m}/{B,S}/episodes.jsonl -> dict[(seed, cond, task, init)] = record (+ attention summaries)."""
    eps = {}
    for seed in SEEDS:
        for cond in ("B", "S"):
            p = Path(root) / f"seed{seed}" / cond / "episodes.jsonl"
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                if not line:
                    continue
                r = json.loads(line)
                r["seed"], r["cond"] = seed, cond
                r["attn"] = _attn_call_summaries(r.get("attention_file"))
                eps[(seed, cond, r["task_id"], r["initial_state_index"])] = r
    return eps


def grounding_class(r):
    ft, bt = r.get("faithful_touch_t"), r.get("biased_touch_t")
    if ft is None and bt is None:
        return "neither"
    if bt is None:
        return "faithful"
    if ft is None:
        return "biased"
    return "faithful" if ft < bt else "biased" if bt < ft else "both"


def outcome_class(r):
    if r["faithful_success"]:
        return "F-success"
    if r["biased_success"]:
        return "B-success"
    g = grounding_class(r)
    return {"faithful": "F-first-no-success", "biased": "B-first-no-success", "both": "both-no-success"}.get(g, "neither")


def first_close_t(r):
    for i, a in enumerate(r["executed_actions"]):
        if a[6] > 0:
            return WAIT + i
    return None


def call_table(r):
    """Per-call rows for one episode: features (S only), attention, phase, call index, t."""
    calls = r["policy_calls"]
    attn = r["attn"] or {}
    t_contact = min([x for x in (r.get("faithful_touch_t"), r.get("biased_touch_t")) if x is not None], default=None)
    t_close = first_close_t(r)
    rows = []
    for c, pc in enumerate(calls):
        row = {"call": c, "t": pc["t"]}
        g = pc["diag"].get("guidance")
        if g is not None:
            Rn = np.asarray(g["a_conditioned_model"]) - np.asarray(g["a_unconditioned_model"])
            Re = np.asarray(g["a_conditioned"]) - np.asarray(g["a_unconditioned"])
            row.update(residual_features(Rn, Re))
            row["guidance_mag"] = float((OMEGA - 1) * np.linalg.norm(Re[:5, 0:3], axis=1).mean())
            row["A_cond_mag"] = float(np.linalg.norm(np.asarray(g["a_conditioned_model"]), axis=1).mean())
            row["Rn"] = Rn
        else:
            row["A_cond_mag"] = float(np.linalg.norm(to_norm(r["predicted_chunks"][c]), axis=1).mean())
        for name, arr in attn.items():
            if c < len(arr):
                row[name] = float(arr[c].mean())
                row[name + "_late"] = float(arr[c][5:].mean())
                row[name + "_k"] = arr[c]
        t = pc["t"]
        if t_contact is None:
            row["phase"] = "no_contact"
        elif t < t_contact:
            row["phase"] = "P0_pre_contact"
        elif t_close is not None and t < t_close:
            row["phase"] = "P1_contact_pre_close"
        else:
            row["phase"] = "P2_post_close"
        rows.append(row)
    return rows, {"t_contact": t_contact, "t_close": t_close}


def disagreement_rows(r):
    """S3 replanning disagreement rows (c, k=5..9) on executed chunks, plus cond/uncond-branch versions (S)."""
    ch = np.asarray(r["predicted_chunks"], float)
    t = [pc["t"] for pc in r["policy_calls"]]
    g = [pc["diag"].get("guidance") for pc in r["policy_calls"]]
    rows = []
    for c in range(len(ch) - 1):
        if t[c + 1] - t[c] != REPLAN:
            continue
        for k in range(REPLAN, ch.shape[1]):
            d = ch[c, k] - ch[c + 1, k - REPLAN]
            row = {"call": c, "k": k, "D_comb": float(np.linalg.norm(d / HALF_RANGE)),
                   "D_trans": float(np.linalg.norm(d[0:3])), "D_rot": float(np.linalg.norm(d[3:6])),
                   "D_grip": float(abs(d[6]))}
            if g[c] is not None and g[c + 1] is not None:
                for br, key in (("cond", "a_conditioned"), ("uncond", "a_unconditioned")):
                    dd = np.asarray(g[c][key][k]) - np.asarray(g[c + 1][key][k - REPLAN])
                    row[f"D_{br}"] = float(np.linalg.norm(dd / HALF_RANGE))
            rows.append(row)
    return rows
