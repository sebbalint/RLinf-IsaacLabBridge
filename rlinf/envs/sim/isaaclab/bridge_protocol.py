"""Wire protocol shared by the Windows Isaac Lab server and the WSL RLinf client.

Keep an IDENTICAL copy of this file on both sides (Windows next to
windows_server.py, WSL in rlinf/envs/sim/isaaclab/).

A message is a ZeroMQ multipart: frame 0 is a JSON header, frames 1..n are raw
array buffers. No pickle. Nested dicts of tensors (Isaac Lab observations) are
flattened to "a/b/c" keys, so dict keys must not contain "/".

Both sides: pip install pyzmq numpy   (torch is already there)
"""
from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import zmq


def to_numpy(x) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(x):
            x = x.detach()
            if x.dtype == torch.bfloat16:  # numpy has no bfloat16
                x = x.float()
            return x.cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def flatten(tree: dict, prefix: str = "") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in tree.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, full + "/"))
        else:
            out[full] = to_numpy(value)
    return out


def unflatten(flat: dict[str, Any], convert: Callable[[Any], Any] | None = None) -> dict:
    tree: dict = {}
    for key, value in flat.items():
        *parents, leaf = key.split("/")
        node = tree
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = convert(value) if convert else value
    return tree


def pack(meta: dict[str, Any], arrays: dict[str, Any] | None = None) -> list[bytes]:
    specs, bufs = [], []
    for name, arr in (arrays or {}).items():
        arr = np.ascontiguousarray(to_numpy(arr))
        specs.append({"name": name, "dtype": str(arr.dtype), "shape": list(arr.shape)})
        bufs.append(arr.tobytes())
    return [json.dumps({"meta": meta, "arrays": specs}).encode(), *bufs]


def unpack(frames: list[bytes]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    header = json.loads(frames[0].decode())
    arrays = {
        spec["name"]: np.frombuffer(buf, dtype=spec["dtype"]).reshape(spec["shape"])
        for spec, buf in zip(header["arrays"], frames[1:])
    }
    return header["meta"], arrays


def send(sock: zmq.Socket, meta: dict[str, Any], arrays: dict[str, Any] | None = None) -> None:
    sock.send_multipart(pack(meta, arrays), copy=False)


def recv(sock: zmq.Socket) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    return unpack([bytes(f) for f in sock.recv_multipart()])
