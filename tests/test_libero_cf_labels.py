import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics.libero_cf_labels import resolve_labels  # noqa: E402

LIBERO = ROOT / "external/libero-cf/libero/libero"
SUITE = "libero_cf_spatial"

# Expected faithful subject per task id, written by hand from the prompt + BDDL :init (see S0 report).
EXPECTED_FAITHFUL = {
    0: "cookies_1", 1: "glazed_rim_porcelain_ramekin_1", 2: "cookies_1", 3: "cookies_1",
    4: "glazed_rim_porcelain_ramekin_1", 5: "glazed_rim_porcelain_ramekin_1", 6: "akita_black_bowl_2",
    7: "cookies_1", 8: "cookies_1", 9: "akita_black_bowl_2", 10: "glazed_rim_porcelain_ramekin_1",
    11: "akita_black_bowl_2", 12: "akita_black_bowl_2", 13: "akita_black_bowl_2",
    14: "glazed_rim_porcelain_ramekin_1",
}


def _labels():
    task_map = runpy.run_path(str(LIBERO / "benchmark/libero_suite_task_map.py"))["libero_task_map"][SUITE]
    db = json.loads((LIBERO / "conditions" / f"{SUITE}.json").read_text())
    return {
        tid: resolve_labels(LIBERO / "bddl_files" / f"{SUITE}/{name}.bddl", db[f"{SUITE}/{name}.bddl"])
        for tid, name in enumerate(task_map)
    }


def test_cf_spatial_faithful_and_biased():
    labels = _labels()
    assert {tid: lab["faithful_subject"] for tid, lab in labels.items()} == EXPECTED_FAITHFUL
    for lab in labels.values():
        assert lab["biased_subject"] == "akita_black_bowl_1"
        assert lab["faithful_condition"] and lab["biased_condition"]
        assert lab["faithful_condition"] != lab["biased_condition"]


def test_known_prompt_language_mismatches():
    # Upstream data inconsistencies (BDDL :language disagrees with the filename-derived prompt that is sent):
    #   13: prompt "black bowl on the wooden cabinet" vs :language "... in the top layer of the wooden cabinet"
    #   14: prompt "ramekin" vs :language "akita black bowl next to the plate"
    labels = _labels()
    assert [tid for tid, lab in labels.items() if lab["prompt_language_mismatch"]] == [13, 14]


def test_frozen_task_manifest_matches_labels():
    import csv  # noqa: PLC0415

    rows = list(csv.DictReader(open(ROOT / "configs/cag/libero_cf_spatial_task_validity.csv")))
    labels = _labels()
    assert len(rows) == 15
    for r in rows:
        lab = labels[int(r["task_id"])]
        assert (r["faithful_object"], r["biased_object"]) == (lab["faithful_subject"], lab["biased_subject"])
        assert r["actual_policy_prompt"] == lab["prompt"]
        assert (r["prompt_bddl_match"] == "True") == (not lab["prompt_language_mismatch"])
    valid = [int(r["task_id"]) for r in rows if r["include_primary"] == "True"]
    assert valid == list(range(13))
    assert [valid[0], valid[(len(valid) - 1) // 2], valid[-1]] == [0, 6, 12]
