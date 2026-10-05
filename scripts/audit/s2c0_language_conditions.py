"""S2c-0: resolve the benchmark-defined language conditions of every valid CF-Spatial scene (no paraphrases).

Pure python (any interpreter). Writes configs/cag/S2c0_language_conditions.json BEFORE any language query.

Instruction pool (benchmark strings only, verbatim as the policy receives them, i.e. from BDDL file names):
  * every libero_cf_spatial prompt (benchmark.grab_language_from_filename of the 15 CF BDDL files);
  * every libero_spatial instruction (the 10 pi05_libero training instructions of the source scenes).
Resolution of a pool instruction in a scene: the category noun selects candidate instances; a spatial qualifier
must match the instance's BDDL `:init` relation target EXACTLY (table below). An instruction is valid for the scene
iff it resolves to exactly one instance whose `on <instance> plate_1` condition is listed for the task.

Per scene:
  l_f : the CF prompt (faithful instruction; S2b B/S used exactly this string);
  l_b : the instruction of the source LIBERO-Spatial training scene (the scene whose BDDL :init block is identical);
        its subject is the BDDL :goal subject (biased object). Cross-checked: the CF init-state file is
        byte-identical to that training scene's init file;
  l_o : every other valid pool instruction (one per remaining condition object);
  l_u : CAG's language-masked branch (no string; tokens and mask zeroed).

  python3 scripts/audit/s2c0_language_conditions.py
"""

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.metrics.libero_cf_labels import parse_bddl, prompt_from_bddl_filename  # noqa: E402

LIBERO = REPO / "external/libero-cf/libero/libero"
BDDL = LIBERO / "bddl_files"
INIT = LIBERO / "init_files"
VALID_TASKS = list(range(13))

CATEGORY = {"cookie box": "cookies", "ramekin": "glazed_rim_porcelain_ramekin", "black bowl": "akita_black_bowl"}
QUALIFIER_TARGET = {  # exact :init relation target
    "between the plate and the ramekin": "main_table_between_plate_ramekin_region",
    "from table center": "main_table_table_center",
    "in the top drawer of the wooden cabinet": "wooden_cabinet_1_top_region",
    "next to the cookie box": "main_table_next_to_box_region",
    "next to the plate": "main_table_next_to_plate_region",
    "next to the ramekin": "main_table_next_to_ramekin_region",
    "on the cookie box": "cookies_1",
    "on the ramekin": "glazed_rim_porcelain_ramekin_1",
    "on the stove": "flat_stove_1_cook_region",
    "on the wooden cabinet": "wooden_cabinet_1_top_side",
}


def resolve(prompt, info):
    m = re.match(r"pick up the (.+?) and place it on the plate$", prompt)
    if not m:
        return None
    phrase = m.group(1)
    noun = next((n for n in CATEGORY if phrase.startswith(n)), None)
    if noun is None:
        return None
    cands = sorted(i for i, c in info["instances"].items() if c == CATEGORY[noun])
    qual = phrase[len(noun):].strip()
    if qual:
        if qual not in QUALIFIER_TARGET:
            raise ValueError(f"unknown qualifier {qual!r}")
        cands = [i for i in cands if info["init"].get(i) == QUALIFIER_TARGET[qual]]
    return cands[0] if len(cands) == 1 else None


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def init_block(info):
    return sorted(info["init"].items())


def main():
    cf_files = sorted((BDDL / "libero_cf_spatial").glob("*.bddl"))
    sp_files = sorted((BDDL / "libero_spatial").glob("*.bddl"))
    pool = sorted({prompt_from_bddl_filename(str(f)) for f in cf_files} | {prompt_from_bddl_filename(str(f)) for f in sp_files})
    conditions_db = json.loads((LIBERO / "conditions/libero_cf_spatial.json").read_text())
    sp_info = {f: parse_bddl(f) for f in sp_files}
    out = {"pool": pool, "qualifier_targets": QUALIFIER_TARGET, "tasks": {}}
    for tid in VALID_TASKS:
        f = cf_files[tid]
        info = parse_bddl(f)
        conds = conditions_db[f"libero_cf_spatial/{f.name}"]
        cond_subjects = [c.split()[1] for c in conds]
        l_f = prompt_from_bddl_filename(str(f))
        src = [s for s, si in sp_info.items() if init_block(si) == init_block(info)]
        assert len(src) == 1, (tid, src)
        src = src[0]
        l_b = prompt_from_bddl_filename(str(src))
        biased = info["goal"][0][1]
        assert resolve(l_b, info) == biased, (tid, l_b)
        cf_init = next((INIT / "libero_cf_spatial").glob(f.stem + ".*init"))
        sp_init = INIT / "libero_spatial" / (src.stem + ".pruned_init")
        valid = {}
        for p in pool:
            inst = resolve(p, info)
            if inst is not None and inst in cond_subjects:
                valid.setdefault(inst, []).append(p)
        faithful = resolve(l_f, info)
        others = [p for inst, ps in sorted(valid.items()) for p in ps if p not in (l_f, l_b)]
        out["tasks"][str(tid)] = {
            "cf_bddl": f.name, "source_libero_spatial_bddl": src.name,
            "init_file_identical_to_source": md5(cf_init) == md5(sp_init),
            "conditions": conds, "faithful_subject": faithful, "biased_subject": biased,
            "l_f": l_f, "l_b": l_b, "l_o": others,
            "valid_instructions_by_subject": valid,
            "all_conditions_have_exactly_one_instruction": all(len(valid.get(s, [])) == 1 for s in cond_subjects),
        }
    dst = REPO / "configs/cag/S2c0_language_conditions.json"
    dst.write_text(json.dumps(out, indent=1))
    for t, d in out["tasks"].items():
        print(t, d["faithful_subject"], "|", d["l_b"][12:-26], "|", [p[12:-26] for p in d["l_o"]],
              d["init_file_identical_to_source"], d["all_conditions_have_exactly_one_instruction"])


if __name__ == "__main__":
    main()
