"""Faithful / biased condition labels for LIBERO-CF tasks.

LIBERO-CF evaluates every condition listed in libero/libero/conditions/<suite>.json but never records which
condition is the *instructed* (faithful) one and which is the *original training task* (biased). We derive
both from the task files, with no reference to rollout outcomes:

  biased   := subject of the BDDL `:goal` predicate. LIBERO-CF keeps the original LIBERO training goal there
              (e.g. `(On akita_black_bowl_1 plate_1)`) while the instruction is counterfactual.
  faithful := the object named in the prompt the policy actually receives. The prompt is derived from the BDDL
              *file name* (benchmark.grab_language_from_filename), NOT from the BDDL `:language` field.
              When the named category has several instances, the instance is resolved through the spatial
              phrase in the prompt against the BDDL `:init` relations; any ambiguity raises.

Python 3.8 compatible (runs in the LIBERO-CF client env).
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Prompt noun phrase -> BDDL object category (libero_cf_spatial vocabulary).
CATEGORY_BY_PHRASE = {
    "cookie box": "cookies",
    "ramekin": "glazed_rim_porcelain_ramekin",
    "black bowl": "akita_black_bowl",
}
# Spatial qualifier in the prompt -> substrings that must appear in the instance's `:init` relation target.
QUALIFIER_TARGETS = {
    "on the cookie box": ("cookies_1",),
    "next to the ramekin": ("next_to_ramekin_region",),
    "next to the plate": ("next_to_plate_region",),
    "on the wooden cabinet": ("wooden_cabinet_1_top_side",),
    "between the plate and the ramekin": ("between_plate_ramekin_region",),
}


def prompt_from_bddl_filename(bddl_file: str) -> str:
    """Re-implementation of LIBERO-CF benchmark.grab_language_from_filename for non-LIBERO-100 names."""
    name = Path(bddl_file).name
    language = " ".join(name.split("_")).split("-")[-1]
    return language[: language.find(".bddl")]


def _section(text: str, header: str) -> str:
    start = text.index(f"(:{header}")
    depth = 0
    for i in range(start, len(text)):
        depth += text[i] == "("
        depth -= text[i] == ")"
        if depth == 0:
            return text[start : i + 1]
    raise ValueError(f"unterminated :{header}")


def parse_bddl(bddl_path: Path) -> Dict[str, object]:
    text = Path(bddl_path).read_text()
    goal = re.findall(r"\((On|In)\s+(\S+)\s+(\S+)\)", _section(text, "goal"))
    init = re.findall(r"\((On|In)\s+(\S+)\s+(\S+)\)", _section(text, "init"))
    objects = re.findall(r"^\s*([\w\s]+?)\s+-\s+(\w+)\s*$", _section(text, "objects"), flags=re.M)
    instances = {}
    for names, category in objects:
        for inst in names.split():
            instances[inst] = category
    language = re.search(r"\(:language\s+([^)]*)\)", text).group(1).strip()
    return {"goal": goal, "init": {subj: tgt for _, subj, tgt in init}, "instances": instances, "language": language}


def resolve_labels(bddl_path: Path, conditions: List[str]) -> Dict[str, Optional[str]]:
    """Return {'prompt', 'bddl_language', 'faithful_subject', 'biased_subject', 'faithful_condition',
    'biased_condition', 'prompt_language_mismatch'} for one task."""
    info = parse_bddl(bddl_path)
    prompt = prompt_from_bddl_filename(str(bddl_path))
    if len(info["goal"]) != 1:
        raise ValueError(f"{bddl_path}: expected exactly one goal predicate, got {info['goal']}")
    biased = info["goal"][0][1]

    match = re.match(r"pick up the (.+?) and place it", prompt)
    if not match:
        raise ValueError(f"unrecognised prompt form: {prompt!r}")
    phrase = match.group(1)
    category = next((c for p, c in CATEGORY_BY_PHRASE.items() if phrase.startswith(p)), None)
    if category is None:
        raise ValueError(f"no category for phrase {phrase!r}")
    candidates = sorted(i for i, c in info["instances"].items() if c == category)
    if len(candidates) > 1:
        qualifier = phrase[len(next(p for p in CATEGORY_BY_PHRASE if phrase.startswith(p))) :].strip()
        targets = QUALIFIER_TARGETS.get(qualifier)
        if targets is None:
            raise ValueError(f"{bddl_path}: unresolvable qualifier {qualifier!r} for {candidates}")
        candidates = [i for i in candidates if any(t in info["init"].get(i, "") for t in targets)]
    if len(candidates) != 1:
        raise ValueError(f"{bddl_path}: faithful instance ambiguous: {candidates}")
    faithful = candidates[0]

    def cond_for(subject: str) -> Optional[str]:
        hits = [c for c in conditions if c.split()[1] == subject]
        return hits[0] if len(hits) == 1 else None

    bddl_language = info["language"]
    return {
        "prompt": prompt,
        "bddl_language": bddl_language,
        "prompt_language_mismatch": _normalise(bddl_language) != _normalise(prompt),
        "faithful_subject": faithful,
        "biased_subject": biased,
        "faithful_condition": cond_for(faithful),
        "biased_condition": cond_for(biased),
    }


def _normalise(text: str) -> str:
    text = text.lower().replace("pick up ", "pick ").replace("akita ", "")
    return re.sub(r"\s+", " ", text).strip()


def label_suite(libero_root: Path, suite: str, conditions_db: Dict[str, List[str]]) -> List[Tuple[int, dict]]:
    """Labels for every task of a suite in benchmark order (keys of the conditions DB follow the task map)."""
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map  # noqa: PLC0415

    out = []
    for task_id, name in enumerate(libero_task_map[suite]):
        key = f"{suite}/{name}.bddl"
        bddl = Path(libero_root) / "bddl_files" / key
        out.append((task_id, resolve_labels(bddl, conditions_db[key])))
    return out
