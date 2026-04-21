from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .conversion import ConversionDType, convert_model
from .embeddings import load_embedder
from .models import MODEL_SPECS, get_model_spec
from .release import DEFAULT_NAMESPACE, prepare_artifact_for_hub, publish_artifact

app = typer.Typer(
    no_args_is_help=True,
    help="Perplexity-to-MLX conversion workspace helpers.",
)
console = Console(width=140)


@app.command("list-models")
def list_models() -> None:
    """List the Hugging Face models configured for conversion."""
    table = Table(title="Perplexity MLX conversion targets")
    table.add_column("Slug", no_wrap=True)
    table.add_column("Hugging Face repo", no_wrap=True)
    table.add_column("Parameters", no_wrap=True)
    table.add_column("Dim", no_wrap=True)
    table.add_column("Default output", no_wrap=True)

    for spec in MODEL_SPECS:
        table.add_row(
            spec.slug,
            spec.huggingface_repo,
            spec.parameter_count,
            str(spec.embedding_dimension),
            spec.default_output_dir,
        )

    console.print(table)


@app.command("show-model")
def show_model(
    slug: Annotated[str, typer.Argument(help="Model slug from `list-models`.")],
) -> None:
    """Show metadata for one configured source model."""
    try:
        spec = get_model_spec(slug)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print_json(
        data={
            "slug": spec.slug,
            "huggingface_repo": spec.huggingface_repo,
            "huggingface_url": spec.huggingface_url,
            "parameter_count": spec.parameter_count,
            "embedding_dimension": spec.embedding_dimension,
            "default_output_dir": spec.default_output_dir,
        }
    )


@app.command("convert", help="Convert a configured Hugging Face model to a local MLX artifact.")
def convert_command(
    slug: Annotated[str, typer.Argument(help="Model slug from `list-models`.")],
    output_path: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Destination directory for the converted artifact."),
    ] = None,
    revision: Annotated[
        str | None,
        typer.Option("--revision", help="Optional Hugging Face revision, tag, or commit SHA."),
    ] = None,
    dtype: Annotated[
        ConversionDType,
        typer.Option(help="Floating dtype to save non-quantized model weights."),
    ] = "bfloat16",
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace the output directory if it already exists."),
    ] = False,
) -> None:
    result = convert_model(
        slug,
        output_path=None if output_path is None else Path(output_path),
        revision=revision,
        dtype=dtype,
        overwrite=overwrite,
    )
    console.print_json(
        data={
            "slug": result.slug,
            "source_repo": result.source_repo,
            "source_revision": result.source_revision,
            "output_path": str(result.output_path),
            "dtype": result.dtype,
        }
    )


@app.command(
    "smoke-validate",
    help="Load a converted MLX artifact and run contextual embedding smoke validation.",
)
def smoke_validate(
    model_path: Annotated[str, typer.Argument(help="Path to a converted MLX artifact.")],
) -> None:
    result = load_embedder(model_path).smoke_validate()
    console.print_json(
        data={
            "documents": result.documents,
            "chunk_counts": result.chunk_counts,
            "shapes": result.shapes,
            "dtypes": result.dtypes,
            "raw_float_finite": result.raw_float_finite,
        }
    )


@app.command("prepare-hub", help="Prepare a converted artifact for Hugging Face upload.")
def prepare_hub(
    slug: Annotated[str, typer.Argument(help="Model slug from `list-models`.")],
    repo_id: Annotated[
        str | None,
        typer.Option("--repo-id", help="Destination Hugging Face repo id."),
    ] = None,
) -> None:
    result = prepare_artifact_for_hub(slug, repo_id=repo_id)
    console.print_json(
        data={
            "slug": result.slug,
            "repo_id": result.repo_id,
            "artifact_path": str(result.artifact_path),
        }
    )


@app.command("publish", help="Upload a prepared artifact to Hugging Face.")
def publish(
    slug: Annotated[str, typer.Argument(help="Model slug from `list-models`.")],
    namespace: Annotated[
        str,
        typer.Option("--namespace", help="Hugging Face namespace for the default repo id."),
    ] = DEFAULT_NAMESPACE,
    repo_id: Annotated[
        str | None,
        typer.Option("--repo-id", help="Explicit Hugging Face repo id."),
    ] = None,
    private: Annotated[
        bool,
        typer.Option("--private", help="Create or update the repo as private."),
    ] = False,
) -> None:
    result = publish_artifact(slug, namespace=namespace, repo_id=repo_id, private=private)
    console.print_json(
        data={
            "slug": result.slug,
            "repo_id": result.repo_id,
            "url": result.url,
        }
    )
