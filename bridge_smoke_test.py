"""Run in WSL (no RLinf needed) while windows_server.py is running:

    python bridge_smoke_test.py --env-id <ISAAC_TASK_ID> --num-envs 4 --action-dim 7 \
        --cameras wrist_cam:128x128 table_cam:128x128

Prints the observation layout (keys, shapes, DTYPES - note uint8 vs float for
images, that decides your bandwidth) and the per-step round-trip time.
"""
import argparse
import time

import torch

from rlinf.envs.sim.isaaclab.remote_venv import RemoteIsaacLabEnv

p = argparse.ArgumentParser()
p.add_argument("--env-id", required=True)
p.add_argument("--num-envs", type=int, default=4)
p.add_argument("--action-dim", type=int, default=7)
p.add_argument("--steps", type=int, default=30)
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, default=5555)
p.add_argument("--cameras", nargs="*", default=[], help="name:HxW, e.g. wrist_cam:128x128")
args = p.parse_args()

cameras = {}
for spec in args.cameras:
    name, size = spec.split(":")
    h, w = size.split("x")
    cameras[name] = [int(h), int(w)]

env = RemoteIsaacLabEnv(
    {"cmd": "init", "env_id": args.env_id, "num_envs": args.num_envs, "seed": 0,
     "cameras": cameras, "obs_keys": None},
    host=args.host, port=args.port,
)
dev = env.device()
print("local device:", dev)


def describe(tree, prefix=""):
    total = 0
    for k, v in tree.items():
        if isinstance(v, dict):
            total += describe(v, f"{prefix}{k}/")
        else:
            nbytes = v.numel() * v.element_size()
            total += nbytes
            print(f"  {prefix}{k:<28} {tuple(v.shape)!s:<22} {v.dtype!s:<16} {nbytes / 1e6:8.2f} MB")
    return total


obs, _ = env.reset(seed=0)
print("observation after reset:")
print(f"  total {describe(obs):.0f} bytes per step")

times = []
for _ in range(args.steps):
    action = torch.zeros(args.num_envs, args.action_dim, device=dev)
    t0 = time.perf_counter()
    obs, reward, term, trunc, _ = env.step(action)
    times.append(time.perf_counter() - t0)
print(f"step round trip: mean {1e3 * sum(times) / len(times):.1f} ms, "
      f"max {1e3 * max(times):.1f} ms over {len(times)} steps")
print("reward", reward.shape, reward.dtype, "| terminated", term.dtype, "| truncated", trunc.dtype)

env.reset(seed=1, env_ids=torch.arange(min(2, args.num_envs), device=dev))
print("partial reset OK")
env.close()
