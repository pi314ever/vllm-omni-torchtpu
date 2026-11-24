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
"""Decorator-based, process-local patch registry for vLLM-Omni on TPU.

Patch submodules under ``vllm_omni_torchtpu.patches`` register callbacks with
``@register_patch(...)`` and are discovered lazily on the first ``apply()``.
Registration validates stage and matcher contracts up front, while heavy
``torch`` and ``vllm_omni`` imports stay inside the patch functions until a
matching lifecycle stage runs. Spawned processes start with an empty registry;
forked processes retain installed patches and reset process-local locks and
in-progress tracking.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, TypeVar

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm_omni.diffusion.data import OmniDiffusionConfig

_F = TypeVar("_F", bound=Callable[[], None])


class Stage(StrEnum):
    """Lifecycle hooks at which Omni TPU patches can be activated."""

    OMNI_PLATFORM = "omni_platform"
    DIFFUSION_WORKER = "diffusion_worker"
    DIFFUSION_MODEL_LOAD = "diffusion_model_load"
    AR_MODEL_LOAD = "ar_model_load"


@dataclass(frozen=True)
class Patch:
    """Declares a registered patch callback, its stages, and optional matchers."""

    fn: Callable[[], None]
    stages: tuple[Stage, ...]
    target: str
    pipelines: frozenset[str] | None = None
    architectures: frozenset[str] | None = None
    order: int = 0
    refresh: Callable[[], None] | None = None


_patches: list[Patch] = []
_discovered = False
_applied: set[Patch] = set()
_applying: set[Patch] = set()
_active_stages: set[Stage] = set()
_entered_stages: set[Stage] = set()
_lock = RLock()
logger = init_logger(__name__)


def _reset_after_fork() -> None:
    """Retains installed patches and resets process-local execution state."""
    global _lock
    _lock = RLock()
    _applying.clear()
    _active_stages.clear()


os.register_at_fork(after_in_child=_reset_after_fork)


def _resolve_stage(stage: Stage | str) -> Stage:
    try:
        return Stage(stage)
    except ValueError as exc:
        raise ValueError(f"Unknown Omni TPU patch stage: {stage}") from exc


def _callable_target(fn: Callable[[], None]) -> str:
    module = getattr(fn, "__module__", type(fn).__module__)
    for attr in ("__qualname__", "__name__"):
        value = getattr(fn, attr, None)
        if isinstance(value, str):
            return f"{module}:{value}"
    return f"{module}:{type(fn).__name__}"


def register_patch(
    *stages: Stage | str,
    pipelines: Iterable[str] | None = None,
    architectures: Iterable[str] | None = None,
    order: int = 0,
    refresh: Callable[[], None] | None = None,
) -> Callable[[_F], _F]:
    """Register a patch callback to run at one or more lifecycle stages."""
    if not stages:
        raise ValueError("At least one Omni TPU patch stage is required.")
    resolved_stages = tuple(dict.fromkeys(_resolve_stage(stage) for stage in stages))
    if pipelines is not None and Stage.DIFFUSION_MODEL_LOAD not in resolved_stages:
        raise ValueError("pipelines filter requires Stage.DIFFUSION_MODEL_LOAD.")
    if architectures is not None and Stage.AR_MODEL_LOAD not in resolved_stages:
        raise ValueError("architectures filter requires Stage.AR_MODEL_LOAD.")
    resolved_pipelines = None if pipelines is None else frozenset(pipelines)
    resolved_architectures = None if architectures is None else frozenset(architectures)

    def _decorator(fn: _F) -> _F:
        patch = Patch(
            fn=fn,
            stages=resolved_stages,
            target=_callable_target(fn),
            pipelines=resolved_pipelines,
            architectures=resolved_architectures,
            order=order,
            refresh=refresh,
        )
        with _lock:
            if not any(existing.fn is fn for existing in _patches):
                _patches.append(patch)
                _patches.sort(key=lambda entry: entry.order)
        return fn

    return _decorator


def _discover_patches() -> None:
    """Import patch modules under this package once so decorators register."""
    global _discovered
    if _discovered or not __package__:
        return
    package_dir = str(Path(__file__).resolve().parent)
    module_names = sorted(
        name
        for _, name, is_pkg in pkgutil.walk_packages(
            [package_dir], prefix=f"{__package__}."
        )
        if not is_pkg
        and name != f"{__package__}.registry"
        and not any(part.startswith("_") for part in name.split("."))
    )
    for name in module_names:
        importlib.import_module(name)
    _discovered = True


def _matches_patch(
    patch: Patch,
    *,
    model_config: ModelConfig | None,
    od_config: OmniDiffusionConfig | None,
) -> bool:
    if patch.pipelines is not None:
        model_class_name = None if od_config is None else od_config.model_class_name
        if model_class_name not in patch.pipelines:
            return False
    if patch.architectures is not None:
        architectures = None if model_config is None else model_config.architectures
        if not architectures or not any(
            arch in patch.architectures for arch in architectures
        ):
            return False
    return True


def apply(
    stage: Stage | str,
    *,
    model_config: ModelConfig | None = None,
    od_config: OmniDiffusionConfig | None = None,
) -> bool:
    """Apply matching patches for ``stage`` once per process and refresh bindings.

    Returns whether at least one registered patch matched for ``stage``.
    Import-triggered re-entry into the same stage returns ``False`` immediately:
    the outer invocation already owns that ordered pass. Failures propagate and
    leave the failing patch eligible for retry, retaining completed patches.
    """
    resolved_stage = _resolve_stage(stage)

    with _lock:
        if resolved_stage in _active_stages:
            return False
        _active_stages.add(resolved_stage)
        _discover_patches()
        applied: list[str] = []
        matched = False
        try:
            for patch in tuple(_patches):
                if resolved_stage not in patch.stages or patch in _applying:
                    continue
                if not _matches_patch(
                    patch, model_config=model_config, od_config=od_config
                ):
                    continue
                matched = True
                if patch in _applied:
                    if patch.refresh is not None:
                        patch.refresh()
                    continue
                _applying.add(patch)
                try:
                    patch.fn()
                finally:
                    _applying.discard(patch)
                _applied.add(patch)
                applied.append(patch.target)
            _entered_stages.add(resolved_stage)
        finally:
            _active_stages.discard(resolved_stage)
            if applied:
                logger.info(
                    "Completed Omni TPU patch callbacks at %s: %s",
                    resolved_stage,
                    ", ".join(applied),
                )
        return matched


def has_entered(stage: Stage | str) -> bool:
    """Return whether ``apply`` has completed for ``stage`` in this process."""
    return stage in _entered_stages
