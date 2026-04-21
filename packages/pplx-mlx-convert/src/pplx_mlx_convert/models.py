from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Metadata for one source model that this workspace will convert."""

    slug: str
    huggingface_repo: str
    parameter_count: str
    embedding_dimension: int

    @property
    def huggingface_url(self) -> str:
        return f"https://huggingface.co/{self.huggingface_repo}"

    @property
    def default_output_dir(self) -> str:
        return f"artifacts/mlx/{self.slug}"


MODEL_SPECS: Final[tuple[ModelSpec, ...]] = (
    ModelSpec(
        slug="pplx-embed-context-v1-4b",
        huggingface_repo="perplexity-ai/pplx-embed-context-v1-4b",
        parameter_count="4b",
        embedding_dimension=2560,
    ),
    ModelSpec(
        slug="pplx-embed-context-v1-0.6b",
        huggingface_repo="perplexity-ai/pplx-embed-context-v1-0.6b",
        parameter_count="0.6b",
        embedding_dimension=1024,
    ),
)

MODEL_SPECS_BY_SLUG: Final[dict[str, ModelSpec]] = {spec.slug: spec for spec in MODEL_SPECS}


def get_model_spec(slug: str) -> ModelSpec:
    try:
        return MODEL_SPECS_BY_SLUG[slug]
    except KeyError as exc:
        available = ", ".join(MODEL_SPECS_BY_SLUG)
        msg = f"Unknown model slug {slug!r}. Available slugs: {available}."
        raise ValueError(msg) from exc
