"""Drop-in replacement for SubProcIsaacLabEnv (venv.py), talking to the Windows server.

Interface is identical to the original, so IsaaclabBaseEnv (metrics, chunk_step,
auto-reset, _wrap_obs in the task classes) stays untouched:

    reset(seed=None, env_ids=None) -> (obs, info)
    step(action)                   -> (obs, reward, terminated, truncated, info)
    close()
    device()                       -> torch.device       (a METHOD, as in the original)

obs is the same nested dict Isaac Lab returns (obs["policy"]["wrist_cam"], ...),
with every leaf a torch tensor on the local CUDA device, like before.

Place in rlinf/envs/sim/isaaclab/ next to bridge_protocol.py.
"""
import numpy as np
import torch
import zmq

try:  # inside the rlinf package
    from .bridge_protocol import recv, send, unflatten
except ImportError:  # standalone (bridge_smoke_test.py)
    from bridge_protocol import recv, send, unflatten


class RemoteIsaacLabEnv:
    def __init__(
        self,
        init_msg: dict,
        host: str = "127.0.0.1",
        port: int = 5555,
        local_device: str | None = None,
        init_timeout_s: float = 1800,  # first start: Isaac Sim boot + shader compile + env creation
        step_timeout_s: float = 300,
    ):
        if local_device is None:
            local_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(local_device)
        self._addr = f"tcp://{host}:{port}"
        self._step_timeout_s = step_timeout_s
        self._connect()
        self._call(init_msg, timeout_s=init_timeout_s)

    # ---- transport -------------------------------------------------------
    def _connect(self):
        self._sock = zmq.Context.instance().socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self._addr)

    def _call(self, meta, arrays=None, timeout_s=None):
        send(self._sock, meta, arrays)
        if not self._sock.poll(int(1000 * (timeout_s or self._step_timeout_s))):
            self._sock.close(linger=0)  # a REQ socket is unusable after a lost reply
            raise TimeoutError(f"no reply from Isaac Lab server at {self._addr} for {meta.get('cmd')!r}")
        reply, arrays_out = recv(self._sock)
        if not reply.get("ok", False):
            raise RuntimeError(f"Isaac Lab server error: {reply.get('error')}")
        return reply, arrays_out

    # ---- helpers ---------------------------------------------------------
    def _to_device(self, a) -> torch.Tensor:
        return torch.from_numpy(np.array(a)).to(self._device)  # np.array copies -> writable

    def _obs(self, arrays):
        flat = {k[len("obs/"):]: v for k, v in arrays.items() if k.startswith("obs/")}
        return unflatten(flat, self._to_device)

    # ---- SubProcIsaacLabEnv interface ------------------------------------
    def reset(self, seed=None, env_ids=None):
        arrays = {}
        if env_ids is not None:
            arrays["env_ids"] = env_ids.detach().cpu().numpy().astype(np.int64)
        _, out = self._call({"cmd": "reset", "seed": seed}, arrays)
        return self._obs(out), {}

    def step(self, action: torch.Tensor):
        """action : (bs, action_dim)"""
        _, out = self._call({"cmd": "step"}, {"action": action.detach().cpu().numpy()})
        return (
            self._obs(out),
            self._to_device(out["reward"]),
            self._to_device(out["terminated"]),
            self._to_device(out["truncated"]),
            {},  # RLinf's base env discards the Isaac Lab infos anyway
        )

    def close(self):
        try:
            self._call({"cmd": "close"}, timeout_s=30)
        except Exception as exc:  # server may already be gone
            print(f"[bridge] close: {exc}")
        finally:
            self._sock.close(linger=0)

    def device(self):
        return self._device
