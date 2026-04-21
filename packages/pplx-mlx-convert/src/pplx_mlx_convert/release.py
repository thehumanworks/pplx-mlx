import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi

from .models import ModelSpec, get_model_spec

DEFAULT_NAMESPACE = "agentmish"
SOURCE_REPOSITORY_URL = "https://github.com/thehumanworks/pplx-mlx"

VALIDATION_SUMMARY = {
    "pplx-embed-v1-0.6b": {
        "smoke_shapes": "[[2, 1024]]",
        "reference_cosines": "0.9997950, 0.9997987, 0.9997995",
        "reference_delta": "max absolute int8 delta 1; mean absolute int8 delta 0.191",
    },
    "pplx-embed-v1-4b": {
        "smoke_shapes": "[[2, 2560]]",
        "reference_cosines": "0.9998998, 0.9998891, 0.9999021",
        "reference_delta": "max absolute int8 delta 2; mean absolute int8 delta 0.259",
    },
    "pplx-embed-context-v1-0.6b": {
        "smoke_shapes": "[[2, 1024], [1, 1024]]",
        "reference_cosines": "0.9995409, 0.9995418, 0.9997643",
        "reference_delta": "max absolute int8 delta 2; mean absolute int8 delta 0.209 and 0.145",
    },
    "pplx-embed-context-v1-4b": {
        "smoke_shapes": "[[2, 2560], [1, 2560]]",
        "reference_cosines": "0.9998859, 0.9998759, 0.9998548",
        "reference_delta": "max absolute int8 delta 2; mean absolute int8 delta 0.281 and 0.429",
    },
}


@dataclass(frozen=True, slots=True)
class PreparedArtifact:
    slug: str
    repo_id: str
    artifact_path: Path


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    slug: str
    repo_id: str
    url: str


def default_repo_id(slug: str, *, namespace: str = DEFAULT_NAMESPACE) -> str:
    return f"{namespace}/{slug}-mlx"


def prepare_artifact_for_hub(
    slug: str,
    *,
    artifact_path: Path | None = None,
    repo_id: str | None = None,
) -> PreparedArtifact:
    spec = get_model_spec(slug)
    resolved_artifact_path = (artifact_path or Path(spec.default_output_dir)).resolve()
    resolved_repo_id = repo_id or default_repo_id(slug)

    if not resolved_artifact_path.exists():
        msg = f"Artifact path does not exist: {resolved_artifact_path}"
        raise FileNotFoundError(msg)

    _copy_loader_package(resolved_artifact_path)
    (resolved_artifact_path / "README.md").write_text(
        generate_model_card(spec, repo_id=resolved_repo_id)
    )

    return PreparedArtifact(
        slug=slug,
        repo_id=resolved_repo_id,
        artifact_path=resolved_artifact_path,
    )


def publish_artifact(
    slug: str,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    repo_id: str | None = None,
    artifact_path: Path | None = None,
    private: bool = False,
) -> PublishedArtifact:
    resolved_repo_id = repo_id or default_repo_id(slug, namespace=namespace)
    prepared = prepare_artifact_for_hub(
        slug,
        artifact_path=artifact_path,
        repo_id=resolved_repo_id,
    )
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN is not set.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=prepared.repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=prepared.artifact_path,
        repo_id=prepared.repo_id,
        repo_type="model",
        token=token,
        commit_message=f"Upload {slug} MLX embedding artifact",
    )

    return PublishedArtifact(
        slug=slug,
        repo_id=prepared.repo_id,
        url=f"https://huggingface.co/{prepared.repo_id}",
    )


def generate_model_card(spec: ModelSpec, *, repo_id: str) -> str:
    validation = VALIDATION_SUMMARY[spec.slug]
    contextual = spec.kind == "contextual"
    usage = _contextual_usage(repo_id, spec) if contextual else _independent_usage(repo_id, spec)
    input_summary = (
        "It takes a list of documents where each document is a list of chunks, and returns one "
        "embedding matrix per document."
        if contextual
        else "It takes a list of texts and returns one embedding matrix for the batch."
    )
    smoke_label = "contextual output" if contextual else "embedding output"
    reference_label = "sample contextual inputs" if contextual else "sample text inputs"
    tag_block = "- contextual-embeddings\n" if contextual else "- mteb\n"
    return f"""---
license: mit
base_model: {spec.huggingface_repo}
library_name: mlx
pipeline_tag: feature-extraction
tags:
- mlx
- apple-silicon
- feature-extraction
- sentence-similarity
{tag_block.rstrip()}
- perplexity
- qwen3
---

# {spec.slug}-mlx

MLX conversion of [{spec.huggingface_repo}](https://huggingface.co/{spec.huggingface_repo})
for Apple Silicon.

This is a {"contextual" if contextual else "standard"} embedding model. {input_summary}

## Important Loading Note

This artifact is not loadable through vanilla `mlx_lm.load()` because MLX-LM does not
natively support Perplexity's custom `bidirectional_pplx_qwen3` model type. The repository
includes a small `pplx_mlx_convert` loader package for this artifact.

## Source Code

Conversion and validation code lives in [{SOURCE_REPOSITORY_URL}]({SOURCE_REPOSITORY_URL}).

## Install

```bash
pip install mlx mlx-lm transformers huggingface_hub numpy
```

## Usage

```python
{usage}
```

The model natively produces unnormalized int8 embeddings by default. Use cosine similarity
for comparison. `embedder.encode(..., quantization="none")` returns float32 pooled embeddings,
and `embedder.encode(..., quantization="binary")` returns binary tanh embeddings.

## Conversion Details

- Source model: `{spec.huggingface_repo}`
- Source revision: see `conversion.json`
- Converted dtype: `bfloat16`
- Embedding dimension: `{spec.embedding_dimension}`
- Output root expected by this workspace: `{spec.default_output_dir}`

## Validation

Local MLX smoke validation passed with finite raw float embeddings and int8 {smoke_label}
shapes `{validation["smoke_shapes"]}`.

Compared against the original Transformers remote-code float32 model on {reference_label}:

- cosine similarities: {validation["reference_cosines"]}
- int8 delta: {validation["reference_delta"]}

The MLX artifact is bfloat16 while the reference path used float32, so int8 values are not
expected to be bit-identical.

## License

The source model is MIT licensed. This conversion preserves the MIT license.
"""


def _copy_loader_package(artifact_path: Path) -> None:
    source_package = Path(__file__).resolve().parent
    destination_package = artifact_path / "pplx_mlx_convert"
    if destination_package.exists():
        shutil.rmtree(destination_package)

    destination_package.mkdir(parents=True)
    for filename in ["__init__.py", "architecture.py", "embeddings.py", "models.py"]:
        shutil.copy2(source_package / filename, destination_package / filename)


def _loader_preamble(repo_id: str) -> str:
    return f'''import sys
from huggingface_hub import snapshot_download

repo_path = snapshot_download("{repo_id}")
sys.path.insert(0, repo_path)

from pplx_mlx_convert import load_embedder

embedder = load_embedder(repo_path)'''


def _independent_usage(repo_id: str, spec: ModelSpec) -> str:
    return f"""{_loader_preamble(repo_id)}
texts = [
    "Scientists explore the universe driven by curiosity.",
    "Children learn through curious exploration.",
    "Historical discoveries began with curious questions.",
]

embeddings = embedder.encode(texts)
print(embeddings.shape)  # (3, {spec.embedding_dimension})
print(embeddings.dtype)  # int8"""


def _contextual_usage(repo_id: str, spec: ModelSpec) -> str:
    return f"""{_loader_preamble(repo_id)}
doc_chunks = [
    [
        "Curiosity begins in childhood with endless questions about the world.",
        "As we grow, curiosity drives us to explore new ideas.",
        "Scientific breakthroughs often start with a curious question.",
    ],
    [
        "The curiosity rover explores Mars searching for ancient life.",
        "Each discovery on Mars sparks new questions about the universe.",
    ],
]

embeddings = embedder.encode(doc_chunks)
print(embeddings[0].shape)  # (3, {spec.embedding_dimension})
print(embeddings[1].shape)  # (2, {spec.embedding_dimension})
print(embeddings[0].dtype)  # int8"""
