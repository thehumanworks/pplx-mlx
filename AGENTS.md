# AGENTS.md

## Scope

- Applies to this monorepo.
- Use Pixi for all workspace commands. Do not introduce a second package manager unless the task explicitly requires it.

## Purpose

- This repo converts Perplexity embedding models from Hugging Face into MLX format for efficient Apple Silicon inference.
- Initial target models:
  - `perplexity-ai/pplx-embed-context-v1-4b`
  - `perplexity-ai/pplx-embed-context-v1-0.6b`
- Published MLX artifact repos:
  - `agentmish/pplx-embed-context-v1-4b-mlx`
  - `agentmish/pplx-embed-context-v1-0.6b-mlx`

## Layout

- `pixi.toml`: root Pixi workspace manifest, dependencies, and tasks.
- `packages/pplx-mlx-convert/`: Python conversion package.
- `tests/`: workspace tests.
- `.cursor/rules/`: persistent rule files; use this file as the map and load the relevant rule before edits.
- `.agents/`: durable repo notes and workflows.
- `artifacts/`, `models/`, and `data/`: local/generated content; keep out of git.

## Rule Index

- `.cursor/rules/pixi-mlx-workspace.mdc`: always-on Pixi, platform, dependency, and artifact rules.
- `.cursor/rules/pplx-context-model-compatibility.mdc`: load before implementing converter or inference code for Perplexity contextual embedding models.
- `.cursor/rules/pplx-validation-research.mdc`: load before researching HF metadata, downloading weights, validating conversion parity, or editing conversion tests.

## Commands

- Install/update the environment: `pixi install`
- List target models: `pixi run list-models`
- Convert the smaller model: `pixi run pplx-mlx-convert convert pplx-embed-context-v1-0.6b --overwrite`
- Smoke-validate the smaller artifact: `pixi run pplx-mlx-convert smoke-validate artifacts/mlx/pplx-embed-context-v1-0.6b`
- Convert the 4B model: `pixi run pplx-mlx-convert convert pplx-embed-context-v1-4b --overwrite`
- Smoke-validate the 4B artifact: `pixi run pplx-mlx-convert smoke-validate artifacts/mlx/pplx-embed-context-v1-4b`
- Publish artifacts: `pixi run pplx-mlx-convert publish <slug> --namespace agentmish`
- Format: `pixi run format`
- Lint: `pixi run lint`
- Typecheck: `pixi run typecheck`
- Test: `pixi run test`
- Full local quality gate: `pixi run quality`

## Development Rules

- Load the relevant `.cursor/rules/*.mdc` file before changing code in its scope.
- Keep source model metadata centralized in `pplx_mlx_convert.models`.
- Keep format/lint tasks scoped to `packages` and `tests`; do not run Ruff over `.pixi`.
- Keep generated MLX outputs under `artifacts/mlx/` unless the user specifies another destination.
- Prefer conda-forge dependencies in `[dependencies]`; use `[pypi-dependencies]` mainly for local editable packages or packages unavailable on conda-forge.
- Preserve `osx-arm64` as the primary platform unless adding non-MLX tooling that is intentionally cross-platform.
- Keep Python and macOS requirements aligned with the conda-forge MLX build matrix. The initial environment uses Python 3.10 and macOS 14.5 because MLX `0.31.x` currently requires that combination on `osx-arm64`.
- Do not start converter implementation by calling vanilla `mlx_lm.convert` on the target repos; research found the target `bidirectional_pplx_qwen3` model type is unsupported by installed MLX-LM.
- Default conversion dtype is bfloat16. Source weights are float32; float16 caused NaNs during local contextual embedding smoke validation.
- Both target models have passed local bfloat16 MLX smoke validation and sample comparison against the Transformers remote-code float32 reference.
- Published Hugging Face repos include a generated model card and embedded `pplx_mlx_convert` loader package. They are not vanilla `mlx_lm.load()` artifacts.
