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
"""TPU runtime patches and lifecycle registry for vLLM-Omni models.

Patches register via ``@register_patch`` in ``patches/registry.py`` and are
grouped into shared infrastructure (``worker.py``, ``layers.py``) and
model-specific modules under ``models/`` (``ltx2.py``, ``qwen_image.py``,
``qwen3_omni.py``, ``wan.py``). Keep this module lightweight so importing it
does not eagerly import torch or vllm_omni model modules.
"""

from __future__ import annotations

from vllm_omni_torchtpu.patches.registry import (
    Patch,
    Stage,
    apply,
    has_entered,
    register_patch,
)

__all__ = [
    "Patch",
    "Stage",
    "apply",
    "has_entered",
    "register_patch",
]
