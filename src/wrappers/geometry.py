"""Static scene-geometry fingerprint of a LIBERO / LIBERO-CF environment (logging only, never mutates the env).

LIBERO envs use hard_reset=True: every env.reset() rebuilds the MuJoCo model, and fixture placement is resampled
from the env's RNG (main_cf.py: "seed seems to affect object positions even when using fixed initial state").
set_init_state() restores qpos/qvel only, so welded fixtures keep whatever `model.body_pos` the reset produced,
while free-joint objects are moved by the restored qpos.

Fingerprints (all after set_init_state):
  static_body_pos_sha256 : model.body_pos/body_quat of every body NOT in a free-joint subtree (fixtures, table,
                           robot base) - the scene geometry that set_init_state cannot restore;
  body_pos_sha256        : model.body_pos of all bodies (free-object XML placements too; these are overridden by
                           qpos and so differ even when the effective scene is identical);
  sim_state_sha256       : flattened MjSimState (time, qpos, qvel).

numpy-only, Python 3.8 compatible (imported from the LIBERO-CF client env and the SCALE env).
"""

import hashlib

import numpy as np

_MJ_JNT_FREE = 0


def _sha(arr):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes()).hexdigest()


def _free_subtree_mask(model):
    """True for bodies whose top-level ancestor (child of world) carries a free joint."""
    parent = np.asarray(model.body_parentid)
    free_root = np.zeros(model.nbody, dtype=bool)
    for b in range(1, model.nbody):
        start, n = int(model.body_jntadr[b]), int(model.body_jntnum[b])
        if n > 0 and any(int(model.jnt_type[j]) == _MJ_JNT_FREE for j in range(start, start + n)):
            free_root[b] = True
    mask = np.zeros(model.nbody, dtype=bool)
    for b in range(1, model.nbody):
        a = b
        while a > 0:
            if free_root[a]:
                mask[b] = True
                break
            a = int(parent[a])
    return mask


def scene_fingerprint(env, include_values=False):
    sim = env.sim
    model = sim.model
    names = [model.body_id2name(i) for i in range(model.nbody)]
    body_pos = np.array(model.body_pos, dtype=np.float64)
    body_quat = np.array(model.body_quat, dtype=np.float64)
    static = ~_free_subtree_mask(model)
    state = np.asarray(sim.get_state().flatten(), dtype=np.float64)
    out = {
        "nbody": int(model.nbody),
        "n_static_bodies": int(static.sum()),
        "body_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "static_body_pos_sha256": _sha(np.concatenate([body_pos[static], body_quat[static]], axis=1)),
        "body_pos_sha256": _sha(body_pos),
        "sim_state_sha256": _sha(state),
    }
    if include_values:
        out["static_body_names"] = [n for n, s in zip(names, static) if s]
        out["static_body_pos"] = body_pos[static].round(6).tolist()
        out["static_body_quat"] = body_quat[static].round(6).tolist()
    return out
