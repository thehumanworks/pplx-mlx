import mlx.core as mx
import pytest

from pplx_mlx_convert.architecture import (
    PPLXQwen3Model,
    get_pplx_model_classes,
    make_bidirectional_padding_mask,
)


def test_get_model_classes_accepts_pplx_model_type() -> None:
    model_class, args_class = get_pplx_model_classes({"model_type": "bidirectional_pplx_qwen3"})

    assert model_class is PPLXQwen3Model
    assert args_class.__name__ == "ModelArgs"


def test_get_model_classes_rejects_other_model_type() -> None:
    with pytest.raises(ValueError, match="Unsupported Perplexity model type"):
        get_pplx_model_classes({"model_type": "qwen3"})


def test_bidirectional_padding_mask_allows_only_valid_keys() -> None:
    attention_mask = mx.array([[1, 1, 0], [1, 0, 0]])

    mask = make_bidirectional_padding_mask(attention_mask)

    assert mask is not None
    assert mask.shape == (2, 1, 1, 3)
    assert mask.tolist() == [[[[True, True, False]]], [[[True, False, False]]]]
