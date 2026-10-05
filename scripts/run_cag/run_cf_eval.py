"""LIBERO-CF evaluation client with per-episode faithful/biased records (runs in envs/libero-cf-client).

The episode loop mirrors external/libero-cf/eval/main_cf.py::eval_libero line for line (same env construction,
seeding, wait steps, preprocessing, replanning, touch detection and condition evaluation) and calls LIBERO-CF's
own helper functions; differences are additive only:
  * --task-ids selects tasks (main_cf.py can only start at a task id and run to the end of the suite);
  * per-episode JSONL records: faithful/biased touch + success (labels from src/metrics/libero_cf_labels.py),
    episode length, policy-call count, client round-trip latency, server_info (RNG-key fingerprint, server
    latency, peak GPU memory), executed actions and full predicted chunks;
  * a run manifest; videos are optional;
  * static scene-geometry fingerprint after set_init_state (src/wrappers/geometry.py), logging only;
  * with --crn-master-seed (S2b): every request carries the CRN tuple (master_seed, task, init, call index); the
    server derives the flow noise from it (src/wrappers/crn.py). Per call the client also logs an observation
    hash; per episode the first env step at which each labelled object was touched;
  * with a --diagnostics server: per-call guidance magnitude / branch noise fingerprints (in the JSONL) and
    action-expert attention entropies (one <out>/attn/task<T>_init<I>.npz per episode); raw attention probs for the
    first --raw-attention-episodes episodes per task, written by the server to <out>/raw/.

Usage (server already running, see src/wrappers/serve_cf_policy.py):
  MUJOCO_GL=egl envs/libero-cf-client/bin/python scripts/run_cag/run_cf_eval.py \
      --suite libero_cf_spatial --task-ids 0 --episodes 2 --condition B --port 8000 --out results/S2_smoke/B
"""

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LIBERO_CF = REPO_ROOT / "external/libero-cf"
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO_ROOT / "configs/libero/cf"))
sys.path.insert(0, str(REPO_ROOT))

import imageio  # noqa: E402
import numpy as np  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy as _websocket_client_policy  # noqa: E402

from src.metrics.libero_cf_labels import resolve_labels  # noqa: E402
from src.wrappers.geometry import scene_fingerprint  # noqa: E402
from src.wrappers.manifest import write_manifest  # noqa: E402

RNG_PROTOCOL = "matched initial states under native policy stochasticity (fresh server per condition; not noise-paired)"
RNG_PROTOCOL_CRN = (
    "S2b common random numbers: flow noise of every policy call = f(master_seed, task_id, init_id, call_index) "
    "(src/wrappers/crn.py), identical across conditions for the same tuple"
)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = {  # identical to main_cf.py
    "libero_cf_spatial": 220,
    "libero_cf_spatial_focused": 220,
    "libero_cf_ood": 220,
    "libero_cf_object": 280,
    "libero_cf_long": 500,
}


def _load_main_cf():
    spec = importlib.util.spec_from_file_location("libero_cf_main_cf", LIBERO_CF / "eval/main_cf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_episode(cf, env, client, init_state, conditions, subject_to_body, args, max_steps, record_video, raw_prefix):
    env.reset()
    action_plan = collections.deque()
    obs = env.set_init_state(init_state)
    geometry = scene_fingerprint(env, include_values=True)
    attention = collections.defaultdict(list)
    touched = {c: False for c in conditions}
    touch_t = {c: None for c in conditions}  # first env step t whose action produced gripper contact
    t, done = 0, False
    executed, chunks, calls, replay = [], [], [], []
    error = None
    while t < max_steps + args.num_steps_wait:
        try:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )
            if record_video:
                replay.append(img)
            if not action_plan:
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (obs["robot0_eef_pos"], cf._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                    "prompt": str(args._task_description),
                    "diag_meta": {
                        "t": int(t),
                        "call": len(calls),
                        "save_raw_path": None if raw_prefix is None else f"{raw_prefix}_call{len(calls):03d}",
                    },
                }
                if args.crn_master_seed is not None:
                    element["crn"] = {
                        "master_seed": args.crn_master_seed, "task_id": args._task_id,
                        "init_id": args._init_id, "call_index": len(calls),
                    }
                obs_sha = hashlib.sha256(
                    img.tobytes() + wrist_img.tobytes() + np.asarray(element["observation/state"], np.float64).tobytes()
                ).hexdigest()[:16]
                start = time.monotonic()
                result = client.infer(element)
                rtt_ms = (time.monotonic() - start) * 1000
                action_chunk = result["actions"]
                if action_chunk.ndim == 1:
                    action_chunk = action_chunk.reshape(1, -1)
                assert len(action_chunk) >= args.replan_steps
                action_plan.extend(action_chunk[: args.replan_steps])
                chunks.append(np.asarray(action_chunk, dtype=float).tolist())
                calls.append(
                    {
                        "t": int(t),
                        "client_rtt_ms": rtt_ms,
                        "policy_timing": result.get("policy_timing"),
                        "server_info": result.get("server_info"),
                        "crn": result.get("crn"),
                        "obs_sha": obs_sha,
                        "diag": {k: v for k, v in result.get("diag", {}).items() if k != "attention"},
                    }
                )
                for branch, summ in result.get("diag", {}).get("attention", {}).items():
                    for k, v in summ.items():
                        attention[f"{branch}/{k}"].append(np.asarray(v))
                    attention[f"{branch}/rollout_step"].append(np.asarray(t))
            action = action_plan.popleft()
            obs, reward, done, info = env.step(action.tolist())
            executed.append(np.asarray(action, dtype=float).tolist())
            contact_names = cf._get_gripper_contact_body_names(env)
            for cond in conditions:
                if not touched[cond]:
                    resolved = subject_to_body.get(cf._condition_subject_token(cond))
                    if resolved and resolved in contact_names:
                        touched[cond] = True
                        touch_t[cond] = int(t)
            if done:
                break
            t += 1
        except Exception as e:  # mirrors main_cf.py: log and end the episode
            error = repr(e)
            break
    try:
        cond_results = env.evaluate_conditions(conditions)
    except Exception:  # noqa: BLE001
        cond_results = {}
    return {
        "done": bool(done),
        "t_final": int(t),
        "env_steps_executed": len(executed),
        "touched": touched,
        "touch_t": touch_t,
        "cond_success": {c: bool(cond_results.get(c, False)) for c in conditions},
        "executed_actions": executed,
        "predicted_chunks": chunks,
        "policy_calls": calls,
        "error": error,
        "replay": replay,
        "geometry": geometry,
        "attention": attention,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_cf_spatial")
    p.add_argument("--task-ids", required=True, help="comma-separated task ids")
    p.add_argument("--episodes", type=int, required=True, help="episodes per task (init states S..S+N-1)")
    p.add_argument("--init-start", type=int, default=0, help="first initial-state index S (default 0)")
    p.add_argument("--condition", required=True, help="label stored in records, e.g. B or S")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--seed", type=int, default=7, help="main_cf.py default; used for np.random.seed and env.seed")
    p.add_argument("--video", action="store_true")
    p.add_argument("--raw-attention-episodes", type=int, default=2, help="episodes/task with raw attention probs")
    p.add_argument("--task-manifest", default=str(REPO_ROOT / "configs/cag/libero_cf_spatial_task_validity.csv"))
    p.add_argument("--crn-master-seed", type=int, default=None, help="S2b common random numbers (server must match)")
    p.add_argument("--forbid-init-from", type=int, default=None,
                   help="refuse to run if any initial-state index >= this value (S2b: 30; states 30-49 are reserved)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.forbid_init_from is not None and args.init_start + args.episodes > args.forbid_init_from:
        raise SystemExit(f"refusing: init states {args.init_start}..{args.init_start + args.episodes - 1} "
                         f"reach the reserved range >= {args.forbid_init_from}")
    cf = _load_main_cf()
    np.random.seed(args.seed)
    out = pathlib.Path(args.out).resolve()  # the server writes raw attention files to paths under it
    out.mkdir(parents=True, exist_ok=True)
    task_ids = [int(x) for x in args.task_ids.split(",")]
    suite = benchmark.get_benchmark_dict()[args.suite]()
    conditions_db = cf._load_custom_conditions_db(args.suite)
    task_manifest = {}
    if args.task_manifest and pathlib.Path(args.task_manifest).exists():
        import csv  # noqa: PLC0415

        with open(args.task_manifest) as f:
            task_manifest = {int(r["task_id"]): (r["faithful_object"], r["biased_object"]) for r in csv.DictReader(f)}
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    server_metadata = client.get_server_metadata()
    if args.crn_master_seed is not None:
        assert server_metadata.get("crn_master_seed") == args.crn_master_seed, server_metadata.get("crn_master_seed")
    rng_protocol = RNG_PROTOCOL if args.crn_master_seed is None else RNG_PROTOCOL_CRN

    write_manifest(
        out,
        config=vars(args),
        checkpoint=args.checkpoint,
        condition=args.condition,
        seed={
            "np_random_seed": args.seed,
            "env_seed": args.seed,
            "policy_rng": "server-side jax.random.key(0) stream" if args.crn_master_seed is None
            else f"CRN master_seed {args.crn_master_seed}",
            "protocol": rng_protocol,
        },
        task={"suite": args.suite, "task_ids": task_ids},
        initial_state_indices=range(args.init_start, args.init_start + args.episodes),
        extra={"server_metadata": server_metadata},
    )

    records_path = out / "episodes.jsonl"
    with open(records_path, "w") as records:
        for task_id in task_ids:
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            env, task_description = cf._get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            args._task_description = task_description
            bddl_key = task.problem_folder + "/" + task.bddl_file
            conditions = cf._get_conditions_for_task(conditions_db, bddl_key)
            labels = resolve_labels(pathlib.Path(get_libero_path("bddl_files")) / bddl_key, conditions)
            assert labels["prompt"] == task_description, (labels["prompt"], task_description)
            # main_cf.py calls env.set_custom_conditions() inside try/except-pass, but that method is commented out in
            # LIBERO-CF's ControlEnv (envs/env_wrapper.py L109), so upstream it is a silent no-op. Conditions are
            # passed explicitly to evaluate_conditions() instead, exactly as main_cf.py does.
            env.set_success_any_conditions(conditions)  # OR-termination: done = any listed condition holds
            subject_to_body = {}
            for cond in conditions:
                subj = cf._condition_subject_token(cond)
                subject_to_body[subj] = cf._resolve_instance_root_body_name(env, subj)

            frozen = task_manifest.get(task_id)
            if frozen is not None:  # labels must equal the frozen, committed task manifest
                assert (labels["faithful_subject"], labels["biased_subject"]) == frozen, (task_id, labels, frozen)
            for ep in range(args.init_start, args.init_start + args.episodes):
                args._task_id, args._init_id = task_id, ep
                start = time.monotonic()
                raw_prefix = (
                    str(out / "raw" / f"task{task_id:02d}_init{ep:03d}")
                    if ep - args.init_start < args.raw_attention_episodes
                    else None
                )
                r = run_episode(
                    cf, env, client, initial_states[ep], conditions, subject_to_body, args,
                    MAX_STEPS[args.suite], args.video, raw_prefix,
                )
                attn_file = None
                if r["attention"]:
                    attn_file = out / "attn" / f"task{task_id:02d}_init{ep:03d}.npz"
                    attn_file.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        attn_file,
                        dims=np.asarray(["policy_call", "denoising_step", "layer", "head", "action_horizon_index"]),
                        **{k: np.stack(v) for k, v in r["attention"].items()},
                    )
                fc, bc = labels["faithful_condition"], labels["biased_condition"]
                rec = {
                    "suite": args.suite,
                    "task_id": task_id,
                    "condition": args.condition,
                    "instruction": task_description,
                    "bddl_language": labels["bddl_language"],
                    "initial_state_index": ep,
                    "env_seed": args.seed,
                    "rng_protocol": rng_protocol,
                    "crn_master_seed": args.crn_master_seed,
                    "faithful_condition": fc,
                    "biased_condition": bc,
                    "success_any": r["done"],
                    "faithful_touch": r["touched"][fc],
                    "biased_touch": r["touched"][bc],
                    "faithful_success": r["cond_success"][fc],
                    "biased_success": r["cond_success"][bc],
                    "touched": r["touched"],
                    "faithful_touch_t": r["touch_t"][fc],
                    "biased_touch_t": r["touch_t"][bc],
                    "cond_success": r["cond_success"],
                    "episode_length": r["env_steps_executed"],
                    "policy_call_count": len(r["policy_calls"]),
                    "episode_wall_s": time.monotonic() - start,
                    "policy_calls": r["policy_calls"],
                    "executed_actions": r["executed_actions"],
                    "predicted_chunks": r["predicted_chunks"],
                    "error": r["error"],
                    "exception": r["error"] is not None,
                    "subject_to_body": subject_to_body,
                    "geometry": r["geometry"],
                    "attention_file": None if attn_file is None else str(attn_file),
                    "raw_attention_prefix": raw_prefix,
                }
                records.write(json.dumps(rec) + "\n")
                records.flush()
                if args.video and r["replay"]:
                    imageio.mimwrite(out / f"task{task_id:02d}_ep{ep:03d}.mp4", r["replay"], fps=10)
                print(
                    f"[{args.condition}] task {task_id} ep {ep}: any={r['done']} "
                    f"F(touch={rec['faithful_touch']}, succ={rec['faithful_success']}) "
                    f"B(touch={rec['biased_touch']}, succ={rec['biased_success']}) "
                    f"len={rec['episode_length']} calls={rec['policy_call_count']} err={r['error']}",
                    flush=True,
                )
            env.close()
    print(f"wrote {records_path}")


if __name__ == "__main__":
    main()
