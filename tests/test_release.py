from pplx_mlx_convert.models import get_model_spec
from pplx_mlx_convert.release import default_repo_id, generate_model_card


def test_default_repo_id_uses_agentmish_namespace() -> None:
    assert default_repo_id("pplx-embed-context-v1-4b") == "agentmish/pplx-embed-context-v1-4b-mlx"


def test_generated_model_card_uses_custom_loader_and_validation_details() -> None:
    card = generate_model_card(
        get_model_spec("pplx-embed-context-v1-4b"),
        repo_id="agentmish/pplx-embed-context-v1-4b-mlx",
    )

    assert "not loadable through vanilla `mlx_lm.load()`" in card
    assert "from pplx_mlx_convert import load_embedder" in card
    assert 'snapshot_download("agentmish/pplx-embed-context-v1-4b-mlx")' in card
    assert "print(embeddings[0].shape)  # (3, 2560)" in card
    assert "0.9998859" in card
