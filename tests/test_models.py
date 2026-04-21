import pytest

from pplx_mlx_convert import MODEL_SPECS, get_model_spec


def test_registry_contains_initial_conversion_targets() -> None:
    repos = {spec.huggingface_repo for spec in MODEL_SPECS}

    assert repos == {
        "perplexity-ai/pplx-embed-v1-4b",
        "perplexity-ai/pplx-embed-v1-0.6b",
        "perplexity-ai/pplx-embed-context-v1-4b",
        "perplexity-ai/pplx-embed-context-v1-0.6b",
    }


def test_model_slugs_are_unique() -> None:
    slugs = [spec.slug for spec in MODEL_SPECS]

    assert len(slugs) == len(set(slugs))


def test_model_dimensions_match_requested_targets() -> None:
    dimensions = {spec.slug: spec.embedding_dimension for spec in MODEL_SPECS}

    assert dimensions == {
        "pplx-embed-v1-4b": 2560,
        "pplx-embed-v1-0.6b": 1024,
        "pplx-embed-context-v1-4b": 2560,
        "pplx-embed-context-v1-0.6b": 1024,
    }


def test_model_kinds_match_requested_targets() -> None:
    kinds = {spec.slug: spec.kind for spec in MODEL_SPECS}

    assert kinds == {
        "pplx-embed-v1-4b": "independent",
        "pplx-embed-v1-0.6b": "independent",
        "pplx-embed-context-v1-4b": "contextual",
        "pplx-embed-context-v1-0.6b": "contextual",
    }


def test_get_model_spec_rejects_unknown_slug() -> None:
    with pytest.raises(ValueError, match="Unknown model slug"):
        get_model_spec("missing")
