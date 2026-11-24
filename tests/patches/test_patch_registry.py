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
"""CPU unit tests for the Omni TPU patch registry lifecycle and isolation."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import logging
import multiprocessing
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "src/vllm_omni_torchtpu/patches/registry.py"
)


@pytest.fixture(autouse=True)
def _sync_tpu() -> None:
    """Registry lifecycle tests run on CPU without a TPU accelerator."""


def _load_registry() -> ModuleType:
    name = "_test_omni_patch_registry"
    spec = importlib.util.spec_from_file_location(name, _REGISTRY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def registry():
    module = _load_registry()
    yield module
    sys.modules.pop(module.__name__, None)


@contextlib.contextmanager
def captured_records(logger: logging.Logger):
    """Collect INFO records emitted on ``logger`` even when propagate=False."""
    records: list[logging.LogRecord] = []
    handler = logging.Handler(logging.INFO)
    handler.emit = records.append
    old_level = logger.level
    old_disable = logging.root.manager.disable
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logging.disable(logging.NOTSET)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logging.disable(old_disable)


def test_order_and_idempotence_across_stages(registry) -> None:
    seen = []

    @registry.register_patch(registry.Stage.OMNI_PLATFORM, order=10)
    def second() -> None:
        seen.append("second")

    @registry.register_patch(
        registry.Stage.OMNI_PLATFORM, registry.Stage.DIFFUSION_WORKER, order=-1
    )
    def first() -> None:
        seen.append("first")

    @registry.register_patch(registry.Stage.DIFFUSION_WORKER)
    def worker() -> None:
        seen.append("worker")

    with captured_records(registry.logger) as records:
        assert registry.apply(registry.Stage.OMNI_PLATFORM) is True
        assert registry.apply("diffusion_worker") is True
        assert registry.apply(registry.Stage.DIFFUSION_WORKER) is True
    assert seen == ["first", "second", "worker"]
    assert len(records) == 2


def test_failed_patch_retries_without_repeating_completed_patches(registry) -> None:
    first = Mock()
    failing = Mock(side_effect=[RuntimeError("unavailable"), None])
    last = Mock()
    for cb in (first, failing, last):
        registry.register_patch(registry.Stage.DIFFUSION_WORKER)(cb)
    with pytest.raises(RuntimeError, match="unavailable"):
        registry.apply(registry.Stage.DIFFUSION_WORKER)
    last.assert_not_called()
    registry.apply(registry.Stage.DIFFUSION_WORKER)
    first.assert_called_once_with()
    assert failing.call_count == 2
    last.assert_called_once_with()


def test_nonmatching_model_configs_do_not_consume_patch(registry) -> None:
    ar_model = Mock()
    diffusion_model = Mock()
    registry.register_patch(
        registry.Stage.AR_MODEL_LOAD,
        architectures={"Qwen3OmniMoeForConditionalGeneration"},
    )(ar_model)
    registry.register_patch(
        registry.Stage.DIFFUSION_MODEL_LOAD,
        pipelines={"WanPipeline"},
    )(diffusion_model)

    assert (
        registry.apply(
            registry.Stage.AR_MODEL_LOAD,
            model_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        )
        is False
    )
    assert (
        registry.apply(
            registry.Stage.DIFFUSION_MODEL_LOAD,
            od_config=SimpleNamespace(model_class_name="FluxPipeline"),
        )
        is False
    )
    assert registry._applied == set()
    ar_model.assert_not_called()
    diffusion_model.assert_not_called()

    ar_cfg = SimpleNamespace(architectures=["Qwen3OmniMoeForConditionalGeneration"])
    diff_cfg = SimpleNamespace(model_class_name="WanPipeline")
    assert registry.apply(registry.Stage.AR_MODEL_LOAD, model_config=ar_cfg) is True
    assert registry.apply(registry.Stage.AR_MODEL_LOAD, model_config=ar_cfg) is True
    assert (
        registry.apply(registry.Stage.DIFFUSION_MODEL_LOAD, od_config=diff_cfg) is True
    )
    assert (
        registry.apply(registry.Stage.DIFFUSION_MODEL_LOAD, od_config=diff_cfg) is True
    )

    ar_model.assert_called_once_with()
    diffusion_model.assert_called_once_with()


def test_registration_validates_stages_and_filters(registry) -> None:
    with pytest.raises(ValueError, match="At least one"):
        registry.register_patch()
    with pytest.raises(ValueError, match="Unknown Omni TPU patch stage"):
        registry.register_patch("worker")
    with pytest.raises(ValueError, match="DIFFUSION_MODEL_LOAD"):
        registry.register_patch(
            registry.Stage.DIFFUSION_WORKER, pipelines={"WanPipeline"}
        )
    with pytest.raises(ValueError, match="AR_MODEL_LOAD"):
        registry.register_patch(
            registry.Stage.DIFFUSION_MODEL_LOAD,
            architectures={"Qwen3OmniMoeForConditionalGeneration"},
        )


def test_refreshes_late_bindings_without_reinstalling(registry) -> None:
    install = Mock()
    bindings = {}
    registry.register_patch(
        registry.Stage.OMNI_PLATFORM,
        registry.Stage.DIFFUSION_WORKER,
        refresh=lambda: bindings.update(late="patched"),
    )(install)
    registry.apply(registry.Stage.OMNI_PLATFORM)
    bindings["late"] = "original"
    registry.apply(registry.Stage.DIFFUSION_WORKER)
    install.assert_called_once_with()
    assert bindings["late"] == "patched"


def test_refresh_failure_does_not_reinstall_wrapper(registry) -> None:
    install = Mock()
    refresh = Mock(side_effect=[RuntimeError("refresh"), None])
    registry.register_patch(registry.Stage.DIFFUSION_WORKER, refresh=refresh)(install)
    registry.apply(registry.Stage.DIFFUSION_WORKER)
    with pytest.raises(RuntimeError, match="refresh"):
        registry.apply(registry.Stage.DIFFUSION_WORKER)
    registry.apply(registry.Stage.DIFFUSION_WORKER)
    install.assert_called_once_with()
    assert refresh.call_count == 2


def test_import_reentry_preserves_order(registry) -> None:
    seen = []

    @registry.register_patch(registry.Stage.DIFFUSION_WORKER)
    def first() -> None:
        seen.append("start")
        assert registry.apply(registry.Stage.DIFFUSION_WORKER) is False
        seen.append("end")

    @registry.register_patch(registry.Stage.DIFFUSION_WORKER)
    def second() -> None:
        seen.append("second")

    assert registry.apply(registry.Stage.DIFFUSION_WORKER) is True
    assert seen == ["start", "end", "second"]


def test_concurrent_application_installs_once(registry) -> None:
    install = Mock()
    registry.register_patch(registry.Stage.DIFFUSION_WORKER)(install)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(registry.apply, [registry.Stage.DIFFUSION_WORKER] * 16))
    install.assert_called_once_with()


def test_invalid_stage_is_rejected(registry) -> None:
    with pytest.raises(ValueError, match="Unknown Omni TPU patch stage"):
        registry.apply("worker")


def test_has_entered_reports_applied_stages(registry) -> None:
    flaky = Mock(side_effect=[RuntimeError("boom"), None])
    registry.register_patch(registry.Stage.DIFFUSION_WORKER)(flaky)
    with pytest.raises(RuntimeError, match="boom"):
        registry.apply(registry.Stage.DIFFUSION_WORKER)
    assert registry.has_entered(registry.Stage.DIFFUSION_WORKER) is False
    registry.apply(registry.Stage.DIFFUSION_WORKER)
    assert registry.has_entered(registry.Stage.DIFFUSION_WORKER) is True
    assert registry.has_entered("diffusion_worker") is True
    assert registry.has_entered(registry.Stage.DIFFUSION_MODEL_LOAD) is False


def test_fork_during_application_retains_completed_patches(registry) -> None:
    import os
    from threading import Event, Thread

    started = Event()
    release = Event()
    parent_pid = os.getpid()
    seen = []

    @registry.register_patch(registry.Stage.DIFFUSION_WORKER)
    def completed() -> None:
        seen.append("completed")

    @registry.register_patch(registry.Stage.DIFFUSION_WORKER)
    def pending() -> None:
        if os.getpid() == parent_pid:
            started.set()
            assert release.wait(10)
        seen.append("pending")

    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe(duplex=False)

    def in_child() -> None:
        registry.apply(registry.Stage.DIFFUSION_WORKER)
        registry.apply(registry.Stage.DIFFUSION_WORKER)
        child.send(seen)
        child.close()

    thread = Thread(target=registry.apply, args=(registry.Stage.DIFFUSION_WORKER,))
    process = context.Process(target=in_child)
    thread.start()
    try:
        assert started.wait(5)
        process.start()
        child.close()
        release.set()
        assert parent.poll(5), "Fork child is blocked on registry state"
        assert parent.recv() == ["completed", "pending"]
        process.join(5)
        assert process.exitcode == 0
    finally:
        release.set()
        thread.join(5)
        if process.pid is not None and process.is_alive():
            process.terminate()
            process.join(5)
        parent.close()
        child.close()
    assert seen == ["completed", "pending"]


def test_package_discovery_registers_all_patch_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm_omni_torchtpu.patches import registry as pkg_registry

    for name in (
        "vllm_omni.diffusion.models",
        "vllm_omni.model_executor.models",
    ):
        monkeypatch.setitem(sys.modules, name, None)

    pkg_registry._discover_patches()
    targets = {patch.target for patch in pkg_registry._patches}
    assert {
        "vllm_omni_torchtpu.patches.worker:_extend_async_output_timeout",
        "vllm_omni_torchtpu.patches.worker:_patch_diffusion_ipc_cpu_pack",
        "vllm_omni_torchtpu.patches.layers:_patch_wan_pair_swap_rope",
        "vllm_omni_torchtpu.patches.layers:_patch_sdpa_fallback_dispatch",
        "vllm_omni_torchtpu.patches.models.qwen_image:_patch_qwen_image_rope_txt_freqs_clone",
        "vllm_omni_torchtpu.patches.models.wan:_patch_wan_ffn_gelu_fence",
        "vllm_omni_torchtpu.patches.models.ltx2:_patch_ltx2_vocoder_cpu_init",
        "vllm_omni_torchtpu.patches.models.qwen3_omni:patch_qwen3_omni_thinker",
    } <= targets
