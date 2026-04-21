import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np
import numpy.typing as npt
from huggingface_hub import snapshot_download
from mlx_lm.utils import load_model
from transformers import PreTrainedTokenizerBase, Qwen2Tokenizer

from .architecture import get_pplx_model_classes

Quantization = Literal["int8", "binary", "ubinary", "none"]


def extract_chunk_token_spans(
    *,
    token_ids: npt.NDArray[np.integer],
    attention_mask: npt.NDArray[np.integer],
    sep_token_id: int,
) -> list[tuple[int, int]]:
    valid_positions = attention_mask.astype(bool)
    sep_positions = np.flatnonzero((token_ids == sep_token_id) & valid_positions)
    last_valid_pos = int(attention_mask.sum())

    spans: list[tuple[int, int]] = []
    start_pos = 0
    for sep_pos in sep_positions:
        spans.append((start_pos, int(sep_pos)))
        start_pos = int(sep_pos) + 1

    spans.append((start_pos, last_valid_pos))
    return spans


def mean_pool(
    token_embeddings: npt.NDArray[np.floating],
    attention_mask: npt.NDArray[np.integer],
) -> npt.NDArray[np.float32]:
    if token_embeddings.shape[0] == 0:
        return np.zeros(token_embeddings.shape[-1], dtype=np.float32)

    mask = attention_mask.astype(np.float32)[:, None]
    denominator = np.clip(mask.sum(axis=0), a_min=1e-9, a_max=None)
    return ((token_embeddings * mask).sum(axis=0) / denominator).astype(np.float32)


def quantize_int8_tanh(values: npt.NDArray[np.floating]) -> npt.NDArray[np.int8]:
    rounded = np.round(np.tanh(values) * 127)
    return np.clip(rounded, -128, 127).astype(np.int8)


def quantize_binary_tanh(values: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    return np.where(values >= 0, 1.0, -1.0).astype(np.float32)


def quantize_ubinary_tanh(values: npt.NDArray[np.floating]) -> npt.NDArray[np.uint8]:
    return np.packbits(values >= 0, axis=-1)


@dataclass(frozen=True, slots=True)
class SmokeValidationResult:
    documents: int
    chunk_counts: tuple[int, ...]
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    raw_float_finite: bool


class ContextualEmbedder:
    def __init__(self, model_path: Path | str, *, revision: str | None = None) -> None:
        self.model_path = _resolve_model_path(model_path, revision=revision)
        self.model, self.config = load_model(
            self.model_path,
            lazy=False,
            get_model_classes=get_pplx_model_classes,
        )
        self.tokenizer: PreTrainedTokenizerBase = Qwen2Tokenizer.from_pretrained(self.model_path)

        if self.tokenizer.sep_token_id is None:
            raise ValueError("Tokenizer must define sep_token_id for contextual chunk extraction.")

    def encode(
        self,
        documents: Sequence[Sequence[str]],
        *,
        batch_size: int = 4,
        quantization: Quantization = "int8",
        normalize_embeddings: bool = False,
        dimensions: int | None = None,
    ) -> list[npt.NDArray[np.generic]]:
        _validate_documents(documents)
        if quantization not in {"int8", "binary", "none"}:
            msg = f"Unsupported quantization {quantization!r}."
            raise ValueError(msg)

        encoded: list[npt.NDArray[np.generic]] = []
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start : start + batch_size]
            encoded.extend(
                self._encode_batch(
                    batch_docs,
                    quantization=quantization,
                    normalize_embeddings=normalize_embeddings,
                    dimensions=dimensions,
                )
            )

        return encoded

    def smoke_validate(self) -> SmokeValidationResult:
        sample_documents = [
            [
                "Curiosity begins in childhood with questions about the world.",
                "Scientific breakthroughs often start with a curious question.",
            ],
            [
                "The Mars rover searches for signs of ancient life.",
            ],
        ]
        raw_embeddings = self.encode(sample_documents, quantization="none")
        raw_float_finite = all(bool(np.isfinite(embedding).all()) for embedding in raw_embeddings)
        embeddings = self.encode(sample_documents, quantization="int8")
        return SmokeValidationResult(
            documents=len(embeddings),
            chunk_counts=tuple(int(embedding.shape[0]) for embedding in embeddings),
            shapes=tuple(tuple(int(dim) for dim in embedding.shape) for embedding in embeddings),
            dtypes=tuple(str(embedding.dtype) for embedding in embeddings),
            raw_float_finite=raw_float_finite,
        )

    def _encode_batch(
        self,
        documents: Sequence[Sequence[str]],
        *,
        quantization: Quantization,
        normalize_embeddings: bool,
        dimensions: int | None,
    ) -> list[npt.NDArray[np.generic]]:
        sep_token = self.tokenizer.sep_token
        if sep_token is None:
            raise ValueError("Tokenizer must define sep_token for contextual chunk joining.")

        doc_strings = [sep_token.join(chunks) for chunks in documents]
        inputs = self.tokenizer(
            doc_strings,
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        input_ids = np.asarray(inputs["input_ids"])
        attention_mask = np.asarray(inputs["attention_mask"])

        token_embeddings = self.model(
            mx.array(input_ids),
            attention_mask=mx.array(attention_mask),
        )
        token_embeddings = token_embeddings.astype(mx.float32)
        mx.eval(token_embeddings)
        token_embeddings_np = np.asarray(token_embeddings)

        batch_embeddings: list[npt.NDArray[np.generic]] = []
        for batch_index in range(len(documents)):
            spans = extract_chunk_token_spans(
                token_ids=input_ids[batch_index],
                attention_mask=attention_mask[batch_index],
                sep_token_id=int(self.tokenizer.sep_token_id),
            )
            chunk_embeddings = [
                mean_pool(
                    token_embeddings_np[batch_index, span_start:span_end],
                    attention_mask[batch_index, span_start:span_end],
                )
                for span_start, span_end in spans
            ]
            stacked = np.stack(chunk_embeddings, axis=0)
            if dimensions is not None:
                stacked = stacked[..., :dimensions]
            batch_embeddings.append(
                _finalize_embeddings(
                    stacked,
                    quantization=quantization,
                    normalize_embeddings=normalize_embeddings,
                )
            )

        return batch_embeddings


class IndependentEmbedder:
    def __init__(self, model_path: Path | str, *, revision: str | None = None) -> None:
        self.model_path = _resolve_model_path(model_path, revision=revision)
        self.model, self.config = load_model(
            self.model_path,
            lazy=False,
            get_model_classes=get_pplx_model_classes,
        )
        self.tokenizer: PreTrainedTokenizerBase = Qwen2Tokenizer.from_pretrained(self.model_path)

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 4,
        quantization: Quantization = "int8",
        normalize_embeddings: bool = False,
        dimensions: int | None = None,
    ) -> npt.NDArray[np.generic]:
        _validate_texts(texts)
        if quantization not in {"int8", "binary", "ubinary", "none"}:
            msg = f"Unsupported quantization {quantization!r}."
            raise ValueError(msg)

        encoded_batches: list[npt.NDArray[np.generic]] = []
        for start in range(0, len(texts), batch_size):
            encoded_batches.append(
                self._encode_batch(
                    texts[start : start + batch_size],
                    quantization=quantization,
                    normalize_embeddings=normalize_embeddings,
                    dimensions=dimensions,
                )
            )
        return np.concatenate(encoded_batches, axis=0)

    def smoke_validate(self) -> SmokeValidationResult:
        sample_texts = [
            "Scientists explore the universe driven by curiosity.",
            "Children learn through curious exploration.",
        ]
        raw_embeddings = self.encode(sample_texts, quantization="none")
        raw_float_finite = bool(np.isfinite(raw_embeddings).all())
        embeddings = self.encode(sample_texts, quantization="int8")
        return SmokeValidationResult(
            documents=int(embeddings.shape[0]),
            chunk_counts=(),
            shapes=(tuple(int(dim) for dim in embeddings.shape),),
            dtypes=(str(embeddings.dtype),),
            raw_float_finite=raw_float_finite,
        )

    def _encode_batch(
        self,
        texts: Sequence[str],
        *,
        quantization: Quantization,
        normalize_embeddings: bool,
        dimensions: int | None,
    ) -> npt.NDArray[np.generic]:
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="np",
        )
        input_ids = np.asarray(inputs["input_ids"])
        attention_mask = np.asarray(inputs["attention_mask"])
        token_embeddings = self.model(
            mx.array(input_ids),
            attention_mask=mx.array(attention_mask),
        )
        token_embeddings = token_embeddings.astype(mx.float32)
        mx.eval(token_embeddings)
        token_embeddings_np = np.asarray(token_embeddings)
        pooled = np.stack(
            [
                mean_pool(token_embeddings_np[row_index], attention_mask[row_index])
                for row_index in range(len(texts))
            ],
            axis=0,
        )
        if dimensions is not None:
            pooled = pooled[..., :dimensions]
        return _finalize_embeddings(
            pooled,
            quantization=quantization,
            normalize_embeddings=normalize_embeddings,
        )


Embedder = ContextualEmbedder | IndependentEmbedder


def load_embedder(model_path: Path | str, *, revision: str | None = None) -> Embedder:
    resolved_path = _resolve_model_path(model_path, revision=revision)
    config = json.loads((resolved_path / "config.json").read_text())
    if _is_independent_embedding_config(config):
        return IndependentEmbedder(resolved_path)
    return ContextualEmbedder(resolved_path)


def _is_independent_embedding_config(config: dict[str, object]) -> bool:
    metadata = config.get("mlx_embedding")
    if isinstance(metadata, dict) and metadata.get("kind") == "independent":
        return True

    auto_map = config.get("auto_map")
    if isinstance(auto_map, dict):
        auto_model = auto_map.get("AutoModel")
        return auto_model == "modeling.PPLXQwen3Model"

    return False


def _resolve_model_path(model_path: Path | str, *, revision: str | None = None) -> Path:
    path = Path(model_path).expanduser()
    if path.exists():
        return path

    return Path(
        snapshot_download(
            str(model_path),
            revision=revision,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.py",
                "*.txt",
                "pplx_mlx_convert/*.py",
            ],
        )
    )


def _validate_documents(documents: Sequence[Sequence[str]]) -> None:
    if not documents:
        raise ValueError("documents must contain at least one document.")
    for document in documents:
        if not document:
            raise ValueError("Each document must contain at least one chunk.")
        for chunk in document:
            if not isinstance(chunk, str) or not chunk:
                raise ValueError("Each chunk must be a non-empty string.")


def _validate_texts(texts: Sequence[str]) -> None:
    if not texts:
        raise ValueError("texts must contain at least one text.")
    for text in texts:
        if not isinstance(text, str) or not text:
            raise ValueError("Each text must be a non-empty string.")


def _finalize_embeddings(
    embeddings: npt.NDArray[np.floating],
    *,
    quantization: Quantization,
    normalize_embeddings: bool,
) -> npt.NDArray[np.generic]:
    if quantization == "int8":
        finalized: npt.NDArray[np.generic] = quantize_int8_tanh(embeddings)
    elif quantization == "binary":
        finalized = quantize_binary_tanh(embeddings)
    elif quantization == "ubinary":
        finalized = quantize_ubinary_tanh(embeddings)
    else:
        finalized = embeddings.astype(np.float32)

    if normalize_embeddings:
        float_embeddings = finalized.astype(np.float32)
        norms = np.linalg.norm(float_embeddings, axis=-1, keepdims=True)
        finalized = float_embeddings / np.clip(norms, a_min=1e-12, a_max=None)

    return finalized
