from pplx_mlx_convert.conversion import _patch_readme_for_artifact
from pplx_mlx_convert.models import get_model_spec


def test_patch_readme_for_artifact_uses_model_specific_dimension() -> None:
    source = "\n".join(
        [
            '    "perplexity-ai/pplx-embed-context-v1-4B",',
            "# embeddings[0].shape = (3, 1024), embeddings[1].shape = (2, 1024)",
        ]
    )

    patched = _patch_readme_for_artifact(source, get_model_spec("pplx-embed-context-v1-4b"))

    assert '    "perplexity-ai/pplx-embed-context-v1-4b",' in patched
    assert "# embeddings[0].shape = (3, 2560), embeddings[1].shape = (2, 2560)" in patched
