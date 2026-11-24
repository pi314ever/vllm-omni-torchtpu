<!-- markdownlint-disable -->
PLEASE FILL IN THE PR DESCRIPTION HERE.

## Contributor intent (required human answers)

All four questions must be answered by the human contributor before submission.
An agent may reuse explicit human answers with a public-safe source reference;
it must not invent, infer, or prefill missing answers. If an answer is missing
or materially ambiguous, ask the human and wait. Do not publish private
conversation content without authorization.

### 1. What outcome do you want to achieve?

<!-- Required: the human contributor's answer, or a public-safe reference to their explicit answer. -->

### 2. What is the highest-leverage variable for that outcome?

<!-- Required: the human contributor's answer, or a public-safe reference to their explicit answer. -->

### 3. How will this work move that variable?

<!-- Required: the human contributor's answer, or a public-safe reference to their explicit answer. -->

### 4. Whose customer or user needs does this address, and what are those needs?

<!-- Required: the human contributor's answer, or a public-safe reference to their explicit answer. -->

## Purpose

<!-- Describe the behavior before and after this change. Link the affected
module/feature contract and docs, or explain why unchanged. -->

## Type of Change

<!-- Select the category matching your PR title prefix (multi-tagging is permitted, e.g., [Bugfix][Worker]): -->
- [ ] `[Bugfix]`: Bug fixes (please include regression tests)
- [ ] `[Worker]`: Omni TPU workers and model runners (`TPUARWorker`, `TPUARModelRunner`, `OmniTPUModelRunner`, `OmniTpuPlatform`)
- [ ] `[Patch]`: Runtime patches for `vllm-omni` (`src/vllm_omni_torchtpu/patches/`)
- [ ] `[Attention]`: Diffusion attention backends, RoPE helpers, and profiler integration
- [ ] `[Model]`: Adding or improving model support (model name must be in PR title, e.g., `[Model][Qwen3-Omni] ...`)
- [ ] `[CI/Build]`: Docker images, Buildkite pipelines, GitHub Actions, dependency pins, pre-commit configuration
- [ ] `[Doc]`: Documentation improvements, guides, docstrings
- [ ] `[Misc]`: Miscellaneous changes not fitting above (please use sparingly)

## Test Plan

**vLLM-Omni Version:**

**vllm-torchtpu-core Version/Commit:**

**vllm-omni-torchtpu Commit:**

<!-- Describe the tests and reproduction commands for CPU or TPU. -->

## Test Result

<!-- Required for full review: paste pytest runs of relevant tests on actual TPU hardware (v7x required, v6e optional), with exact commands and actual output at the reported commit. Explicitly list relevant checks not run (or why hardware tests are N/A for docs/CI-only PRs). -->

**Required for full review:** Pytest runs of relevant tests on actual TPU hardware (`v7x` required, `v6e` optional).

**BEFORE SUBMITTING:** read [CONTRIBUTING.md](https://github.com/vllm-project/vllm-omni-torchtpu/blob/main/CONTRIBUTING.md) and [AGENTS.md](https://github.com/vllm-project/vllm-omni-torchtpu/blob/main/AGENTS.md).
(anything written below this line will be removed by GitHub Actions)
