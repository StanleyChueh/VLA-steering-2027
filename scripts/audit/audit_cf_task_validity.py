"""Freeze the LIBERO-CF task manifest: textual + physical validity audit of every task of a suite.

Runs in envs/libero-cf-client, before any pilot outcome exists. For every task:
  * text  : prompt actually sent (filename-derived) vs BDDL `:language`; faithful/biased labels from
            src/metrics/libero_cf_labels.py; the faithful object the BDDL `:language` would imply instead.
  * physics (every stored initial state, env built exactly as main_cf.py with seed 7):
      - faithful and biased objects resolve to simulator bodies;
      - no listed condition is already satisfied at the initial state;
      - for prompts with a spatial qualifier ("black bowl next to the plate", ...), the labelled faithful instance is
        the one that physically satisfies the qualifier (closer to the reference object than the other instance,
        and above it for "on ...").
Writes configs/cag/<suite>_task_validity.csv and a JSON with the per-init-state details.

  MUJOCO_GL=egl envs/libero-cf-client/bin/python scripts/audit/audit_cf_task_validity.py
"""

import argparse
import csv
import importlib.util
import json
import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO_ROOT / "configs/libero/cf"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402

from src.metrics.libero_cf_labels import CATEGORY_BY_PHRASE, parse_bddl, resolve_labels  # noqa: E402
from src.wrappers.geometry import scene_fingerprint  # noqa: E402

# Prompt qualifier -> (reference instance, relation). "on" additionally requires the object to be above the reference.
QUALIFIER_REFERENCE = {
    "on the cookie box": ("cookies_1", "on"),
    "next to the ramekin": ("glazed_rim_porcelain_ramekin_1", "next_to"),
    "next to the plate": ("plate_1", "next_to"),
    "on the wooden cabinet": ("wooden_cabinet_1", "on"),
}


def _load_main_cf():
    spec = importlib.util.spec_from_file_location("main_cf", REPO_ROOT / "external/libero-cf/eval/main_cf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _faithful_from_text(text, info):
    """Faithful instance implied by an arbitrary instruction text (used for the BDDL :language)."""
    t = text.lower().replace("akita ", "").replace("pick up the ", "pick the ")
    noun = t.split("pick the ", 1)[1].split(" and place it", 1)[0] if "pick the " in t else t
    for phrase, category in CATEGORY_BY_PHRASE.items():
        if noun.startswith(phrase):
            t = noun
            cands = sorted(i for i, c in info["instances"].items() if c == category)
            if len(cands) == 1:
                return cands[0]
            if "top layer" in t or "in the wooden cabinet" in t:
                hits = [i for i in cands if "top_region" in info["init"].get(i, "")]
            elif "on the wooden cabinet" in t:
                hits = [i for i in cands if "top_side" in info["init"].get(i, "")]
            elif "on the cookie box" in t:
                hits = [i for i in cands if info["init"].get(i) == "cookies_1"]
            elif "next to the ramekin" in t:
                hits = [i for i in cands if "next_to_ramekin" in info["init"].get(i, "")]
            elif "next to the plate" in t:
                hits = [i for i in cands if "next_to_plate" in info["init"].get(i, "")]
            else:
                hits = []
            return hits[0] if len(hits) == 1 else "AMBIGUOUS:" + "|".join(cands)
    return "UNRESOLVED"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_cf_spatial")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-init-states", type=int, default=20)
    p.add_argument("--out-csv", default=None)
    args = p.parse_args()
    out_csv = pathlib.Path(args.out_csv or REPO_ROOT / f"configs/cag/{args.suite}_task_validity.csv")
    out_json = out_csv.with_suffix(".details.json")

    cf = _load_main_cf()
    suite = benchmark.get_benchmark_dict()[args.suite]()
    db = cf._load_custom_conditions_db(args.suite)
    rows, details = [], {}
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        key = task.problem_folder + "/" + task.bddl_file
        bddl = pathlib.Path(get_libero_path("bddl_files")) / key
        conditions = cf._get_conditions_for_task(db, key)
        lab = resolve_labels(bddl, conditions)
        info = parse_bddl(bddl)
        faithful_from_bddl_language = _faithful_from_text(lab["bddl_language"], info)

        env, prompt = cf._get_libero_env(task, 256, args.seed)
        assert prompt == lab["prompt"]
        # fixtures (e.g. wooden_cabinet_1) are not in the BDDL :objects section but can be qualifier references
        names = set(info["instances"]) | {r for r, _ in QUALIFIER_REFERENCE.values()}
        bodies = {s: cf._resolve_instance_root_body_name(env, s) for s in names}
        init_states = suite.get_task_init_states(task_id)
        phrase = lab["prompt"].split("pick up the ")[1].split(" and place it")[0]
        qualifier = next((phrase[len(ph):].strip() for ph in CATEGORY_BY_PHRASE if phrase.startswith(ph)), "")
        ref = QUALIFIER_REFERENCE.get(qualifier)
        same_cat = sorted(
            i for i, c in info["instances"].items() if c == info["instances"][lab["faithful_subject"]] and i != lab["faithful_subject"]
        )
        per_init = []
        for i, state in enumerate(init_states):
            env.reset()
            env.set_init_state(state)
            sim = env.sim
            pos = {s: np.array(sim.data.body_xpos[sim.model.body_name2id(b)]) for s, b in bodies.items() if b}
            initially_true = [c for c, v in env.evaluate_conditions(conditions).items() if v]
            qual_ok = None
            if ref is not None:
                r, rel = ref
                d_f = float(np.linalg.norm(pos[lab["faithful_subject"]][:2] - pos[r][:2]))
                d_o = [float(np.linalg.norm(pos[o][:2] - pos[r][:2])) for o in same_cat]
                qual_ok = all(d_f < d for d in d_o)
                if rel == "on":
                    qual_ok = qual_ok and bool(pos[lab["faithful_subject"]][2] > pos[r][2])
            per_init.append(
                {
                    "init": i,
                    "initially_true_conditions": initially_true,
                    "qualifier_physically_holds": qual_ok,
                    "static_geometry_sha": scene_fingerprint(env)["static_body_pos_sha256"],
                }
            )
        env.close()

        n_init = len(init_states)
        bodies_ok = bool(bodies.get(lab["faithful_subject"])) and bool(bodies.get(lab["biased_subject"]))
        trivial = [d["init"] for d in per_init if d["initially_true_conditions"]]
        qual_fail = [d["init"] for d in per_init if d["qualifier_physically_holds"] is False]
        match = not lab["prompt_language_mismatch"]
        reasons = []
        if not match:
            reasons.append(
                f"prompt != BDDL :language (BDDL language implies faithful={faithful_from_bddl_language}"
                + (", identical to the biased object" if faithful_from_bddl_language == lab["biased_subject"] else "")
                + ")"
            )
        if not bodies_ok:
            reasons.append("faithful/biased object not resolvable in simulator")
        if trivial:
            reasons.append(f"condition already true at init states {trivial}")
        if qual_fail:
            reasons.append(f"prompt qualifier physically violated at init states {qual_fail}")
        if n_init < args.min_init_states:
            reasons.append(f"only {n_init} initial states")
        if lab["faithful_subject"] == lab["biased_subject"]:
            reasons.append("faithful == biased")
        include = not reasons
        confidence = "high" if include else ("low" if not match or qual_fail else "medium")
        rows.append(
            {
                "task_id": task_id,
                "bddl_language": lab["bddl_language"],
                "actual_policy_prompt": lab["prompt"],
                "faithful_object": lab["faithful_subject"],
                "biased_object": lab["biased_subject"],
                "prompt_bddl_match": match,
                "label_confidence": confidence,
                "include_primary": include,
                "exclusion_reason": "; ".join(reasons),
            }
        )
        details[task_id] = {
            "bddl_file": key,
            "faithful_condition": lab["faithful_condition"],
            "biased_condition": lab["biased_condition"],
            "faithful_from_bddl_language": faithful_from_bddl_language,
            "qualifier": qualifier,
            "qualifier_reference": ref,
            "n_init_states": n_init,
            "bodies": bodies,
            "per_init": per_init,
        }
        print(rows[-1], flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    valid = [r["task_id"] for r in rows if r["include_primary"]]
    selection = [valid[0], valid[(len(valid) - 1) // 2], valid[-1]] if valid else []
    out_json.write_text(
        json.dumps(
            {
                "suite": args.suite,
                "env_seed": args.seed,
                "valid_tasks": valid,
                "selection_rule": "first, median, and last task among the valid clean tasks (median = lower median)",
                "primary_pilot_tasks": selection,
                "tasks": details,
            },
            indent=1,
        )
    )
    print("valid:", valid, "-> primary pilot tasks:", selection)


if __name__ == "__main__":
    main()
