# Perplexity MLX

Monorepo for converting Perplexity embedding models into MLX format so they can run efficiently on Apple Silicon.

## Target Models

| Model | Source |
| --- | --- |
| `pplx-embed-v1-4b` | <https://huggingface.co/perplexity-ai/pplx-embed-v1-4b> |
| `pplx-embed-v1-0.6b` | <https://huggingface.co/perplexity-ai/pplx-embed-v1-0.6b> |
| `pplx-embed-context-v1-4b` | <https://huggingface.co/perplexity-ai/pplx-embed-context-v1-4b> |
| `pplx-embed-context-v1-0.6b` | <https://huggingface.co/perplexity-ai/pplx-embed-context-v1-0.6b> |

## Published MLX Artifacts

| Model | MLX Artifact |
| --- | --- |
| `pplx-embed-v1-4b` | <https://huggingface.co/agentmish/pplx-embed-v1-4b-mlx> |
| `pplx-embed-v1-0.6b` | <https://huggingface.co/agentmish/pplx-embed-v1-0.6b-mlx> |
| `pplx-embed-context-v1-4b` | <https://huggingface.co/agentmish/pplx-embed-context-v1-4b-mlx> |
| `pplx-embed-context-v1-0.6b` | <https://huggingface.co/agentmish/pplx-embed-context-v1-0.6b-mlx> |

## Layout

```text
.
├── packages/
│   └── pplx-mlx-convert/    # Python package for conversion tooling
├── tests/                   # Workspace-level tests
├── .agents/                 # Durable repo workflow notes
└── pixi.toml                # Pixi workspace manifest
```

Generated model artifacts should stay out of git under `artifacts/`, `models/`, or `data/`.

## Setup

```sh
pixi install
```

## Commands

```sh
pixi run list-models
pixi run pplx-mlx-convert convert pplx-embed-v1-0.6b --overwrite
pixi run pplx-mlx-convert smoke-validate artifacts/mlx/pplx-embed-v1-0.6b
pixi run pplx-mlx-convert convert pplx-embed-v1-4b --overwrite
pixi run pplx-mlx-convert smoke-validate artifacts/mlx/pplx-embed-v1-4b
pixi run pplx-mlx-convert convert pplx-embed-context-v1-0.6b --overwrite
pixi run pplx-mlx-convert smoke-validate artifacts/mlx/pplx-embed-context-v1-0.6b
pixi run pplx-mlx-convert convert pplx-embed-context-v1-4b --overwrite
pixi run pplx-mlx-convert smoke-validate artifacts/mlx/pplx-embed-context-v1-4b
pixi run pplx-mlx-convert publish pplx-embed-v1-0.6b --namespace agentmish
pixi run pplx-mlx-convert publish pplx-embed-v1-4b --namespace agentmish
pixi run pplx-mlx-convert publish pplx-embed-context-v1-0.6b --namespace agentmish
pixi run pplx-mlx-convert publish pplx-embed-context-v1-4b --namespace agentmish
pixi run format
pixi run lint
pixi run typecheck
pixi run test
pixi run quality
```

The workspace is intentionally scoped to `osx-arm64`; MLX execution is only meaningful on Apple Silicon. The current Pixi environment uses Python 3.10 and declares macOS 14.5 because conda-forge's current MLX `0.31.x` builds for `osx-arm64` require that combination.

Conversion defaults to bfloat16 for saved floating weights. The source Perplexity safetensors are float32, and float16 caused NaNs in local contextual embedding smoke validation.
