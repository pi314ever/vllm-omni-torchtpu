# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared TPU runtime patches for vLLM-Omni platform, worker, and engine lifecycle."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from vllm_torchtpu.logger import init_logger

from vllm_omni_torchtpu.patches.registry import Stage, register_patch

if TYPE_CHECKING:
    import torch

logger = init_logger(__name__)


@register_patch(Stage.OMNI_PLATFORM)
def _extend_async_output_timeout() -> None:
    """Extend the async output timeout for TPU compilation and execution."""

    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # 1. Lazy execution attributes failure to sync point: In PyTorch/XLA
    #    and torch_tpu, forward passes are queued lazily in the computation
    #    graph. diffusion_worker emits COMPUTE_DONE prematurely before the TPU
    #    hardware finishes execution. When the async output background loop
    #    triggers host synchronization (.cpu()), it blocks on multi-step
    #    diffusion graph execution and initial compilation.
    # 2. Recompilation storms & JIT compile latency: Initial JIT/AOT
    #    compilation and weight distribution for large diffusion models (e.g.
    #    Wan 14B) take >600s, exceeding vLLM-Omni's default 600s timeout.
    #
    # REMOVAL / FIX CONDITION:
    # This patch can be removed/reduced once:
    # 1. OmniTpuPlatform.record_device_event() implements hardware
    #    synchronization (torch.accelerator.synchronize()) to ensure execution
    #    completes before COMPUTE_DONE.
    # 2. AOT compilation caching stabilizes startup compile times.
    # -------------------------------------------------------------------------
    os.environ.setdefault("VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT", "1800.0")
    logger.info("Applied TPU patch: extended VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT to 1800s.")


_IPC_CPU_PACK_PATCHED = "_tpu_ipc_cpu_pack_patched"


@register_patch(Stage.DIFFUSION_WORKER)
def _patch_diffusion_ipc_cpu_pack() -> None:
    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # 1. Pickling a device tensor aborts process: In torch_tpu, calling
    #    pickle.dumps on a device tensor hits
    #    ABSL_CHECK(tensor.storage().allocator() == nullptr) in
    #    csrc/eager/tensor_to_buffer.cc:260-261, terminating the worker process
    #    with SIGABRT (exit code 134) with no Python traceback.
    # 2. libtpu per-process lockfile prevents cross-process init:
    #    Holding TPU tensors across process boundaries causes the CPU-only
    #    orchestrator process to trigger PJRT backend initialization during
    #    unpickling, colliding with the worker on /tmp/libtpu_lockfile.
    #
    # REMOVAL / FIX CONDITION:
    # This patch can be removed once torch_tpu supports safe inter-process tensor
    # serialization / proxy unpickling without triggering PJRT initialization
    # in host processes that do not own TPU hardware.
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion import ipc

    orig_pack_tensor = ipc._pack_tensor_if_large
    if getattr(orig_pack_tensor, _IPC_CPU_PACK_PATCHED, False):
        return

    def _safe_pack_tensor_if_large(val: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return orig_pack_tensor(val.cpu(), *args, **kwargs)

    setattr(_safe_pack_tensor_if_large, _IPC_CPU_PACK_PATCHED, True)
    ipc._pack_tensor_if_large = _safe_pack_tensor_if_large
    logger.info("Applied TPU patch: safe CPU serialization for diffusion IPC tensors.")


_D2H_STREAM_PATCHED = "_tpu_d2h_stream_patched"


@register_patch(Stage.DIFFUSION_WORKER)
def _patch_diffusion_worker_d2h_stream() -> None:
    # -------------------------------------------------------------------------
    # TODO: Remove once vLLM-Omni creates the D2H stream of
    # WorkerProc._async_output_loop
    # (vllm_omni/diffusion/worker/diffusion_worker.py) on the current device
    # index instead of
    # torch.device(torch.accelerator.current_accelerator().type, self.gpu_id).
    #
    # Problem: WorkerProc._async_output_loop creates its D2H stream on device
    #   index self.gpu_id. On TPU, OmniTpuPlatform.set_device selects device
    #   index 0 in every worker process, so a worker with gpu_id >= 1 asks for
    #   a stream on a device index that its process did not select.
    # Fix: run the loop with a proxy as self whose gpu_id is the current
    #   device index; other attribute reads and writes go to the worker.
    # Effect: the output thread of a worker with gpu_id >= 1 gets its stream
    #   instead of failing at startup.
    # Scope: the proxy is live for the whole output-thread lifetime, not only
    #   while the stream is created. Only attribute reads/writes are
    #   forwarded, so identity/type checks on self inside the loop would see
    #   the proxy (the upstream loop does none).
    # -------------------------------------------------------------------------
    from vllm_omni.diffusion.worker.diffusion_worker import WorkerProc

    orig_async_output_loop = WorkerProc._async_output_loop
    if getattr(orig_async_output_loop, _D2H_STREAM_PATCHED, False):
        return

    def _tpu_async_output_loop(self: Any) -> None:
        orig_async_output_loop(_CurrentDeviceWorkerProxy(self))

    setattr(_tpu_async_output_loop, _D2H_STREAM_PATCHED, True)
    WorkerProc._async_output_loop = _tpu_async_output_loop
    logger.info("Applied TPU patch: diffusion D2H stream on the current device.")


class _CurrentDeviceWorkerProxy:
    """Forward attribute access to a worker, reporting the current device as
    ``gpu_id``."""

    def __init__(self, worker: Any) -> None:
        object.__setattr__(self, "_worker", worker)

    def __getattr__(self, name: str) -> Any:
        if name == "gpu_id":
            import torch

            return torch.accelerator.current_device_index()
        return getattr(self._worker, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._worker, name, value)


_WARMUP_SYNC_PATCHED = "_tpu_warmup_sync_patched"


# TODO: Remove once vLLM-Omni synchronizes the device at the end of its
# startup warmup (e.g. self.executor.collective_rpc("synchronize_device") in
# _dummy_run).
@register_patch(Stage.DIFFUSION_WORKER)
def _patch_diffusion_startup_warmup_sync() -> None:
    """Synchronize the device after vLLM-Omni's diffusion startup warmup.

    Root cause: ``DiffusionEngine._dummy_run`` (startup warmup) drops its output
    without a host sync. Under torch_tpu DEFER_AND_FUSE the warmup's tail ops
    (VAE decode) stay pending in torch_tpu's process-global deferred-op queue and
    are materialized together with the first real request, which then gets cut
    into different fused programs than later identical requests (those start
    from an empty queue because the previous output's ``.cpu()`` flushed it).
    The first request after startup therefore differs numerically from later
    same-seed requests.

    Scope: this only takes effect when the engine runs its worker in-process
    (the uniproc executor, the single-device default). With the mp executor,
    only the worker processes install this patch, and they never call
    run_startup_warmup.
    """
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine

    orig_run_startup_warmup = DiffusionEngine.run_startup_warmup
    if getattr(orig_run_startup_warmup, _WARMUP_SYNC_PATCHED, False):
        return

    def _synced_run_startup_warmup(self: Any) -> None:
        import torch

        orig_run_startup_warmup(self)
        # TODO: Once the vllm-omni pin is >= 0.30, sync every worker with
        # self.executor.collective_rpc("synchronize_device") and install this
        # patch at platform init, so that mp engines are covered too.
        torch.accelerator.synchronize()

    setattr(_synced_run_startup_warmup, _WARMUP_SYNC_PATCHED, True)
    DiffusionEngine.run_startup_warmup = _synced_run_startup_warmup
    logger.info("Applied TPU patch: device sync after diffusion startup warmup.")
