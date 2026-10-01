"""SCALE / greedy OpenVLA evaluation on LIBERO with per-step logging (runs in envs/scale).

Mirrors external/scale/experiments/robot/libero/run_libero_eval.py::eval_libero exactly (same seeding,
checkpoint auto-selection, SCALE defaults from configs/scale.yaml, env construction, wait steps, max steps,
gripper post-processing) and calls SCALE's own functions. Additions are logging only:
  * per step: token-level self-uncertainty u_k (the `uncertainty_history[1]` SCALE returns), the decoding
    temperature tau_k = T0 * sigmoid(u_k) implied by Eq. 4, the visual-attention temperature gamma actually
    applied to the vision encoder (read back from the patched attention scale), latency;
  * per episode: success, length, peak CUDA memory; a run manifest.

Usage:
  CUDA_VISIBLE_DEVICES=0 envs/scale/bin/python scripts/run_scale/run_scale_eval.py \
      --task-suite libero_10 --task-ids 0 --episodes 2 --decoding-mode scale --out results/S1_smoke/scale
"""

import argparse
import json
import math
import os
import pathlib
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCALE_ROOT = REPO_ROOT / "external/scale"
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO_ROOT / "configs/libero/upstream"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(SCALE_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from experiments.robot.libero import run_libero_eval as scale_eval  # noqa: E402
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import get_processor  # noqa: E402
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from libero.libero import benchmark  # noqa: E402

from src.wrappers.manifest import write_manifest  # noqa: E402

# Copied from eval_libero() (a local there, so not importable).
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def applied_gamma(model):
    """gamma actually in effect: SCALE sets attn.scale = head_dim**-0.5 / gamma on every vision block."""
    attn = model.vision_backbone.featurizer.blocks[0].attn
    if not hasattr(attn, "_original_forward"):
        return None  # vision encoder never patched (greedy/baselines)
    return (attn.head_dim ** -0.5) / attn.scale


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-ids", required=True)
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--decoding-mode", choices=scale_eval.VALID_DECODING_MODES, required=True)
    p.add_argument("--seed", type=int, default=7, help="SCALE default (GenerateConfig.seed)")
    p.add_argument("--load-in-8bit", action="store_true", help="NON-PAPER-FAITHFUL smoke mode only")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = scale_eval.GenerateConfig(
        task_suite=args.task_suite,
        decoding_mode=args.decoding_mode,
        task_ids=args.task_ids,
        num_trials_per_task=args.episodes,
        seed=args.seed,
        load_in_8bit=args.load_in_8bit,
        save_video=False,
    )
    # Same preamble as eval_libero().
    cfg.pretrained_checkpoint = f"openvla/openvla-7b-finetuned-{cfg.task_suite.replace('_', '-')}"
    scale_eval._apply_scale_defaults(cfg)
    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite
    torch.cuda.reset_peak_memory_stats()
    model = get_model(cfg)
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in model.norm_stats
    processor = get_processor(cfg)
    load_peak = torch.cuda.max_memory_allocated()

    out = pathlib.Path(args.out)
    write_manifest(
        out,
        config={k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(cfg).items()},
        checkpoint=cfg.pretrained_checkpoint,
        condition=args.decoding_mode,
        seed={"set_seed_everywhere": cfg.seed, "env_seed": 0},
        task={"suite": cfg.task_suite, "task_ids": cfg.task_ids},
        initial_state_indices=range(args.episodes),
        extra={"precision": "8bit NON-PAPER-FAITHFUL" if args.load_in_8bit else "bf16", "load_peak_bytes": load_peak},
    )

    suite = benchmark.get_benchmark_dict()[cfg.task_suite]()
    resize_size = get_image_resize_size(cfg)
    max_steps = MAX_STEPS[cfg.task_suite]
    task_ids = [int(t) for t in cfg.task_ids.split(",")]
    with open(out / "episodes.jsonl", "w") as records:
        for task_id in task_ids:
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=256)
            for episode_idx in range(cfg.num_trials_per_task):
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                t, done, error = 0, False, None
                uncertainty_history = None
                steps = []
                torch.cuda.reset_peak_memory_stats()
                ep_start = time.monotonic()
                while t < max_steps + cfg.num_steps_wait:
                    try:
                        if t < cfg.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                            t += 1
                            continue
                        img = get_libero_image(obs, resize_size)
                        observation = {
                            "full_image": img,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }
                        torch.cuda.synchronize()
                        start = time.monotonic()
                        action, step_info = get_action(
                            cfg, model, observation, task_description,
                            processor=processor, uncertainty_history=uncertainty_history,
                        )
                        torch.cuda.synchronize()
                        latency_ms = (time.monotonic() - start) * 1000
                        uncertainty_history = step_info.get("uncertainty_history", uncertainty_history)
                        step = {"t": t, "latency_ms": latency_ms, "action_raw": np.asarray(action).tolist()}
                        if cfg.decoding_mode == "scale":
                            u = list(uncertainty_history[1])
                            step.update(
                                u_tokens=u,
                                u_step=float(np.mean(u)),
                                ema_u=float(np.mean(uncertainty_history[0])),
                                tau_tokens=[cfg.T0 / (1.0 + math.exp(-x)) for x in u],
                                gamma_applied=applied_gamma(model),
                            )
                        action = normalize_gripper_action(action, binarize=True)
                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)
                        obs, reward, done, info = env.step(action.tolist())
                        step["action_env"] = np.asarray(action).tolist()
                        steps.append(step)
                        if done:
                            break
                        t += 1
                    except Exception as e:  # mirrors run_libero_eval.py
                        error = repr(e)
                        break
                rec = {
                    "suite": cfg.task_suite,
                    "task_id": task_id,
                    "instruction": task_description,
                    "decoding_mode": cfg.decoding_mode,
                    "initial_state_index": episode_idx,
                    "seed": cfg.seed,
                    "success": bool(done),
                    "episode_length": len(steps),
                    "episode_wall_s": time.monotonic() - ep_start,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                    "mean_latency_ms": float(np.mean([s["latency_ms"] for s in steps])) if steps else None,
                    "error": error,
                    "steps": steps,
                }
                records.write(json.dumps(rec) + "\n")
                records.flush()
                print(
                    f"[{cfg.decoding_mode}] task {task_id} ep {episode_idx}: success={done} len={len(steps)} "
                    f"lat={rec['mean_latency_ms'] or float('nan'):.1f}ms peak={rec['peak_cuda_bytes'] / 2**30:.2f}GiB err={error}",
                    flush=True,
                )
            env.close()


if __name__ == "__main__":
    main()
