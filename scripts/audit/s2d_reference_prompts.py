"""S2d: freeze the per-task reference instructions BEFORE any S2d rollout (outcome-independent rule).

source : l_b of configs/cag/S2c0_language_conditions.json (the instruction of the identical LIBERO-Spatial training
         scene; subject = biased object akita_black_bowl_1).
other  : the lexicographically first valid benchmark instruction of the scene that is neither l_f nor l_b
         (Python str ordering of the exact prompt strings).
  python3 scripts/audit/s2d_reference_prompts.py   -> configs/cag/S2d_reference_prompts.json
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
lang = json.loads((REPO / "configs/cag/S2c0_language_conditions.json").read_text())["tasks"]
out = {"rule": {"source": "l_b (oracle source/training instruction)",
                "other": "lexicographically first valid instruction that is neither l_f nor l_b"}, "tasks": {}}
for t, d in lang.items():
    alts = sorted(p for ps in d["valid_instructions_by_subject"].values() for p in ps if p not in (d["l_f"], d["l_b"]))
    assert sorted(d["l_o"]) == alts and len(alts) == 2
    other_subject = next(s for s, ps in d["valid_instructions_by_subject"].items() if alts[0] in ps)
    out["tasks"][t] = {"faithful": d["l_f"], "source": d["l_b"], "other": alts[0],
                       "source_subject": d["biased_subject"], "other_subject": other_subject}
    print(t, "| source:", d["l_b"][12:-26], "| other:", alts[0][12:-26], f"({other_subject})")
(REPO / "configs/cag/S2d_reference_prompts.json").write_text(json.dumps(out, indent=1))
