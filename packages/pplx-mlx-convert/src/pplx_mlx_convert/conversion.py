import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlx.core as mx
from huggingface_hub import HfApi, snapshot_download
from mlx.utils import tree_map
from mlx_lm.utils import load_model, save_config, save_model
from transformers import Qwen2Tokenizer

from .architecture import get_pplx_model_classes
from .models import ModelSpec, get_model_spec

ConversionDType = Literal["float16", "bfloat16", "float32"]


@dataclass(frozen=True, slots=True)
class ConversionResult:
    slug: str
    source_repo: str
    source_revision: str
    output_path: Path
    dtype: str


def convert_model(
    slug: str,
    *,
    output_path: Path | None = None,
    revision: str | None = None,
    dtype: ConversionDType = "bfloat16",
    overwrite: bool = False,
) -> ConversionResult:
    spec = get_model_spec(slug)
    destination = output_path or Path(spec.default_output_dir)
    destination = destination.expanduser().resolve()

    if destination.exists():
        if not overwrite:
            msg = f"Output path already exists: {destination}"
            raise FileExistsError(msg)
        shutil.rmtree(destination)

    api = HfApi()
    model_info = api.model_info(spec.huggingface_repo, revision=revision)
    source_revision = model_info.sha
    if source_revision is None:
        msg = f"Hugging Face did not return a source revision for {spec.huggingface_repo}."
        raise ValueError(msg)

    source_path = Path(
        snapshot_download(
            spec.huggingface_repo,
            revision=revision,
            allow_patterns=[
                "*.json",
                "*.py",
                "*.txt",
                "*.model",
                "*.safetensors",
                "*.jinja",
            ],
        )
    )

    model, config = load_model(
        source_path,
        lazy=True,
        get_model_classes=get_pplx_model_classes,
    )
    _cast_floating_parameters(model, dtype)

    destination.mkdir(parents=True)
    save_model(destination, model, donate_model=True)

    config = dict(config)
    config["mlx_embedding"] = {
        "source_repo": spec.huggingface_repo,
        "source_revision": source_revision,
        "converter": "pplx-mlx-convert",
        "dtype": dtype,
        "kind": spec.kind,
    }
    save_config(config, config_path=destination / "config.json")

    tokenizer = Qwen2Tokenizer.from_pretrained(source_path)
    tokenizer.save_pretrained(destination)
    _copy_support_files(source_path, destination, spec)
    _write_conversion_metadata(spec, destination, source_revision, dtype)

    return ConversionResult(
        slug=slug,
        source_repo=spec.huggingface_repo,
        source_revision=source_revision,
        output_path=destination,
        dtype=dtype,
    )


def _cast_floating_parameters(model: object, dtype: ConversionDType) -> None:
    mlx_dtype = getattr(mx, dtype)

    def maybe_cast(value: object) -> object:
        if isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.floating):
            return value.astype(mlx_dtype)
        return value

    model.update(tree_map(maybe_cast, model.parameters()))  # type: ignore[attr-defined]


def _copy_support_files(source_path: Path, destination: Path, spec: ModelSpec) -> None:
    readme_source = source_path / "README.md"
    if readme_source.exists():
        readme_text = readme_source.read_text()
        (destination / "README.md").write_text(_patch_readme_for_artifact(readme_text, spec))

    modules_source = source_path / "modules.json"
    if modules_source.exists():
        shutil.copy2(modules_source, destination / "modules.json")

    pooling_source = source_path / "1_Pooling"
    if pooling_source.exists():
        shutil.copytree(pooling_source, destination / "1_Pooling")


def _patch_readme_for_artifact(readme_text: str, spec: ModelSpec) -> str:
    uppercase_repo_alias = spec.huggingface_repo.removesuffix("b") + "B"
    patched = readme_text.replace(f'"{uppercase_repo_alias}"', f'"{spec.huggingface_repo}"')
    return patched.replace(
        "# embeddings[0].shape = (3, 1024), embeddings[1].shape = (2, 1024)",
        (
            f"# embeddings[0].shape = (3, {spec.embedding_dimension}), "
            f"embeddings[1].shape = (2, {spec.embedding_dimension})"
        ),
    )


def _write_conversion_metadata(
    spec: ModelSpec,
    destination: Path,
    source_revision: str,
    dtype: ConversionDType,
) -> None:
    metadata = {
        "slug": spec.slug,
        "source_repo": spec.huggingface_repo,
        "source_revision": source_revision,
        "dtype": dtype,
        "artifact_type": f"mlx-{spec.kind}-embedding",
        "kind": spec.kind,
    }
    (destination / "conversion.json").write_text(json.dumps(metadata, indent=2) + "\n")
