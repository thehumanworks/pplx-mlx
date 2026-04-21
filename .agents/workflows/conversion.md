# Conversion Workflow Notes

## Current Scope

- Source models:
  - `perplexity-ai/pplx-embed-v1-4b`
  - `perplexity-ai/pplx-embed-v1-0.6b`
  - `perplexity-ai/pplx-embed-context-v1-4b`
  - `perplexity-ai/pplx-embed-context-v1-0.6b`
- Default generated output root: `artifacts/mlx/`.
- Package entrypoint: `pplx-mlx-convert`.
- Source repository: `https://github.com/thehumanworks/pplx-mlx`
- Published MLX repos:
  - `agentmish/pplx-embed-v1-0.6b-mlx`
  - `agentmish/pplx-embed-v1-4b-mlx`
  - `agentmish/pplx-embed-context-v1-0.6b-mlx`
  - `agentmish/pplx-embed-context-v1-4b-mlx`

## Research Findings

- Both target models are `feature-extraction` Transformers repos with custom code.
- The context models use `AutoModel = modeling.PPLXQwen3ContextualModel`.
- The independent models use `AutoModel = modeling.PPLXQwen3Model`.
- Installed MLX-LM supports `qwen3` but does not support `bidirectional_pplx_qwen3`.
- The contextual path requires nested document chunks, separator-token late chunking, mean pooling, and tanh int8 or binary quantization.
- The independent path requires flat text batches, mean pooling, and tanh int8, binary, or packed-binary quantization.
- Expected full embedding dimensions are 1024 for `0.6b` and 2560 for `4b`.
- Default conversion should use bfloat16 or float32; float16 produced NaNs in local smoke validation.
- 0.6B bfloat16 artifact smoke validation passed against the Transformers remote-code reference:
  shapes matched, int8 value ranges matched, and cosine similarity was above 0.9995 on sample chunks.
- 4B bfloat16 artifact smoke validation passed against the Transformers remote-code reference:
  shapes matched, int8 value ranges matched, and cosine similarity was above 0.99985 on sample chunks.
- Independent 0.6B bfloat16 artifact smoke validation passed against the SentenceTransformers remote-code reference:
  shapes matched, int8 value ranges matched, and cosine similarity was above 0.99979 on sample texts.
- Independent 4B bfloat16 artifact smoke validation passed against the SentenceTransformers remote-code reference:
  shapes matched, int8 value ranges matched, and cosine similarity was above 0.99988 on sample texts.
- Both Hugging Face model cards include a `Using Transformers` example with `AutoModel.from_pretrained(..., trust_remote_code=True)` and `model.encode(doc_chunks)`.
  The examples execute successfully for both target repos when PyTorch/Transformers are available using canonical lowercase repo IDs.
  The 4B card's inline shape comment is stale and says 1024, but actual output dimension is 2560.
  The upstream snippets also use uppercase `0.6B`/`4B`; patch copied artifact READMEs to lowercase canonical repo IDs because Transformers remote-code cache paths are case-sensitive.
- Published MLX artifact repos use generated model cards and include the local `pplx_mlx_convert` package so users can load after `snapshot_download`.
- Perplexity's API also exposes Matryoshka dimensions and base64 int8/binary encodings; implement full-dimension parity first.

## Expected Conversion Shape

1. Resolve the source model from `pplx_mlx_convert.models`.
2. Inspect lightweight Hugging Face metadata and record the source commit SHA.
3. Download weights only for the selected model and explicit conversion output.
4. Convert weights/tokenizer/config into an MLX-compatible contextual embedding directory.
5. Validate by loading the converted model on `osx-arm64` and comparing reference embeddings.
6. Record conversion parameters and source revision with the artifact.

Do not commit downloaded Hugging Face weights or generated MLX artifacts.
