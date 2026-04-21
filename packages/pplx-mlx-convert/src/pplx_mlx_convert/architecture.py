from typing import Any

import mlx.core as mx
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

SUPPORTED_MODEL_TYPES = frozenset({"bidirectional_pplx_qwen3", "qwen3"})


def make_bidirectional_padding_mask(attention_mask: mx.array | None) -> mx.array | None:
    """Create a full-attention key padding mask from a tokenizer attention mask."""
    if attention_mask is None:
        return None

    return attention_mask.astype(mx.bool_)[:, None, None, :]


class PPLXQwen3Model(Qwen3Model):
    """Qwen3 encoder variant used by Perplexity contextual embedding models."""

    def __call__(
        self,
        inputs: mx.array,
        attention_mask: mx.array | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        mask = make_bidirectional_padding_mask(attention_mask)

        for layer in self.layers:
            hidden_states = layer(hidden_states, mask, None)

        return self.norm(hidden_states)


def get_pplx_model_classes(config: dict[str, Any]) -> tuple[type[PPLXQwen3Model], type[ModelArgs]]:
    model_type = config.get("model_type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_TYPES))
        msg = f"Unsupported Perplexity model type {model_type!r}; expected one of: {supported}."
        raise ValueError(msg)

    return PPLXQwen3Model, ModelArgs
