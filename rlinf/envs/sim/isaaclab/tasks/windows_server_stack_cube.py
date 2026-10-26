"""Windows side: the logic of RLinf's make_env_isaaclab() (stack_cube.py) and
_torch_worker() (venv.py), driven over ZeroMQ instead of mp pipes/queues.

Run with your Isaac Lab Windows install, e.g.:
    isaaclab.bat -p windows_server.py --port 5555

One server process = one Isaac Sim app = one env. RLinf may build a train env
AND an eval env per env worker, so start one server per port you configure.

Commands (mirror SubProcIsaacLabEnv):
    ping | init | reset | step | close
"""
import argparse
import time
import traceback

import torch
import zmq

from bridge_protocol import flatten, recv, send

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--bind", default="127.0.0.1", help="use 0.0.0.0 if WSL cannot reach localhost")
parser.add_argument("--exit-on-close", action="store_true",
                    help="shut the sim app down when the client sends 'close' (default: keep it for the next run)")
args, _unknown = parser.parse_known_args()

# Same call RLinf makes in stack_cube.py (Windows needs no DISPLAY handling).
from isaaclab.app import AppLauncher  # noqa: E402

sim_app = AppLauncher(headless=True, enable_cameras=True).app


PRINT_EVERY_NTH = 200


def make_env(init: dict):
    """Copy of make_env_isaaclab() from stack_cube.py, parameterised by the init message."""
    import gymnasium as gym
    from isaaclab_tasks.utils import load_cfg_from_registry

    env_id = init["env_id"]
    cfg = load_cfg_from_registry(env_id, "env_cfg_entry_point")
    cfg.seed = init["seed"]
    cfg.scene.num_envs = init["num_envs"]
    for cam_name, (height, width) in (init.get("cameras") or {}).items():
        cam = getattr(cfg.scene, cam_name)
        cam.height, cam.width = height, width
    return gym.make(env_id, cfg=cfg, render_mode="rgb_array").unwrapped


def obs_arrays(obs, obs_keys):
    flat = flatten(obs)
    if obs_keys:
        missing = [k for k in obs_keys if k not in flat]
        if missing:
            raise KeyError(f"obs_keys not in observation: {missing}; available: {sorted(flat)}")
        flat = {k: flat[k] for k in obs_keys}
    return {f"obs/{k}": v for k, v in flat.items()}


def main():
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://{args.bind}:{args.port}")
    print(f"[bridge] listening on {args.bind}:{args.port}", flush=True)

    env, device, init_key, obs_keys = None, None, None, None
    n_steps, t_env, t_pack = 0, 0.0, 0.0
    t_env_old, t_pack_old = t_env, t_pack

    while sim_app.is_running():
        if not sock.poll(1000):
            continue
        meta, arrays = recv(sock)
        cmd = meta.get("cmd")
        try:
            if cmd == "ping":
                send(sock, {"ok": True})

            elif cmd == "init":
                key = {k: meta[k] for k in ("env_id", "num_envs", "cameras", "obs_keys")}  # seed may differ
                if env is None:
                    print(f"[bridge] creating env: {key}", flush=True)
                    env = make_env(meta)
                    device, init_key = env.device, key
                    print(f"[bridge] env ready on {device}", flush=True)
                elif key != init_key:
                    raise RuntimeError(f"server already holds a different env {init_key}; restart it")
                obs_keys = meta.get("obs_keys")
                send(sock, {"ok": True, "device": str(device)})

            elif cmd == "reset":
                # Same calls as _torch_worker.
                ids = arrays.get("env_ids")
                if ids is None:
                    obs, _info = env.reset(seed=meta.get("seed"))
                else:
                    obs, _info = env.reset(seed=meta.get("seed"), env_ids=torch.from_numpy(ids.copy()).to(device))
                send(sock, {"ok": True}, obs_arrays(obs, obs_keys))

            elif cmd == "step":
                action = torch.from_numpy(arrays["action"].copy()).to(device)
                t0 = time.perf_counter()
                obs, reward, terminated, truncated, _info = env.step(action)
                t1 = time.perf_counter()
                out = obs_arrays(obs, obs_keys)
                out.update(reward=reward, terminated=terminated, truncated=truncated)
                send(sock, {"ok": True}, out)
                t2 = time.perf_counter()
                n_steps, t_env, t_pack = n_steps + 1, t_env + (t1 - t0), t_pack + (t2 - t1)
                if n_steps % PRINT_EVERY_NTH == 0:
                    print(f"[bridge] {n_steps} steps"
                          f" | env.step {1e3 * (t_env-t_env_old) / PRINT_EVERY_NTH:.1f}"
                          f" ({1e3 * t_env / n_steps:.1f}) ms"
                          f" | pack+send {1e3 * (t_pack-t_pack_old) / PRINT_EVERY_NTH:.1f}"
                          f" ({1e3 * t_pack / n_steps:.1f}) ms",
                          flush=True)
                    t_env_old = t_env
                    t_pack_old = t_pack

            elif cmd == "close":
                send(sock, {"ok": True})
                if args.exit_on_close:
                    break

            else:
                send(sock, {"ok": False, "error": f"unknown cmd {cmd!r}"})

        except Exception as exc:  # REP socket must always answer
            traceback.print_exc()
            send(sock, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    if env is not None:
        env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
