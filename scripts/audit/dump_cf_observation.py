"""Save the first policy observation of a LIBERO-CF episode (exact main_cf.py preprocessing) for offline tests.

Runs in envs/libero-cf-client:
  envs/libero-cf-client/bin/python scripts/audit/dump_cf_observation.py --task-id 0 --init 0 \
      --out results/S0C/obs_cf_spatial_t0_i0.npz
"""

import argparse
import importlib.util
import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO_ROOT / "configs/libero/cf"))

import numpy as np  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from openpi_client import image_tools  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_cf_spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--init", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    spec = importlib.util.spec_from_file_location("main_cf", REPO_ROOT / "external/libero-cf/eval/main_cf.py")
    cf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cf)

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    env, prompt = cf._get_libero_env(task, 256, args.seed)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(args.task_id)[args.init])
    for _ in range(10):  # num_steps_wait
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224)
    )
    state = np.concatenate((obs["robot0_eef_pos"], cf._quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, image=img, wrist_image=wrist, state=state, prompt=np.asarray(prompt))
    print("saved", args.out, prompt, img.shape, state)


if __name__ == "__main__":
    main()
