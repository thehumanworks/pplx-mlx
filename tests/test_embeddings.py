import numpy as np

from pplx_mlx_convert.embeddings import (
    extract_chunk_token_spans,
    mean_pool,
    quantize_int8_tanh,
)


def test_extract_chunk_token_spans_splits_on_valid_separator_tokens() -> None:
    token_ids = np.array([10, 11, 99, 20, 21, 99, 30, 0, 99])
    attention_mask = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0])

    spans = extract_chunk_token_spans(
        token_ids=token_ids,
        attention_mask=attention_mask,
        sep_token_id=99,
    )

    assert spans == [(0, 2), (3, 5), (6, 7)]


def test_mean_pool_ignores_padding_rows() -> None:
    token_embeddings = np.array(
        [
            [1.0, 1.0],
            [3.0, 5.0],
            [100.0, 100.0],
        ],
        dtype=np.float32,
    )
    attention_mask = np.array([1, 1, 0])

    pooled = mean_pool(token_embeddings, attention_mask)

    np.testing.assert_allclose(pooled, np.array([2.0, 3.0], dtype=np.float32))


def test_quantize_int8_tanh_matches_perplexity_formula() -> None:
    values = np.array([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=np.float32)

    quantized = quantize_int8_tanh(values)

    assert quantized.dtype == np.int8
    assert quantized.tolist() == [-127, -97, 0, 97, 127]
