from typer.testing import CliRunner

from pplx_mlx_convert.cli import app


def test_list_models_reports_requested_sources() -> None:
    result = CliRunner().invoke(app, ["list-models"])

    assert result.exit_code == 0
    assert "perplexity-ai/pplx-embed-context-v1-4b" in result.output
    assert "perplexity-ai/pplx-embed-context-v1-0.6b" in result.output


def test_show_model_reports_default_output_dir() -> None:
    result = CliRunner().invoke(app, ["show-model", "pplx-embed-context-v1-4b"])

    assert result.exit_code == 0
    assert '"huggingface_repo": "perplexity-ai/pplx-embed-context-v1-4b"' in result.output
    assert '"default_output_dir": "artifacts/mlx/pplx-embed-context-v1-4b"' in result.output


def test_convert_help_exposes_conversion_command() -> None:
    result = CliRunner().invoke(app, ["convert", "--help"])

    assert result.exit_code == 0
    assert "Convert a configured Hugging Face model to a local MLX artifact" in result.output


def test_smoke_validate_help_exposes_artifact_validation_command() -> None:
    result = CliRunner().invoke(app, ["smoke-validate", "--help"])

    assert result.exit_code == 0
    assert (
        "Load a converted MLX artifact and run contextual embedding smoke validation"
        in result.output
    )


def test_prepare_hub_help_exposes_release_command() -> None:
    result = CliRunner().invoke(app, ["prepare-hub", "--help"])

    assert result.exit_code == 0
    assert "Prepare a converted artifact for Hugging Face upload" in result.output


def test_publish_help_exposes_upload_command() -> None:
    result = CliRunner().invoke(app, ["publish", "--help"])

    assert result.exit_code == 0
    assert "Upload a prepared artifact to Hugging Face" in result.output
