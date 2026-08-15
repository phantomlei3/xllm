# Codex 0.147.0 Responses wire fixtures

These fixtures freeze the xLLM phase 0–5 Responses subset as of 2026-08-14.
Captured samples came from `codex-cli 0.147.0` talking to a local scripted
Responses endpoint with a local model catalog entry for `deepseek-v4`. The
scripted endpoint caused one function call and one `apply_patch` custom call;
Codex executed both and emitted both output items in later requests.

`requests/codex-tool-loop-start.json` preserves the observed top-level fields
and the complete `apply_patch` description and Lark grammar. The other request
fixtures isolate individual contract paths, and the non-stream request is a
normative projection because Codex used SSE during capture.

The committed samples are minimized and sanitized. Session, installation,
thread, turn, message, and generated output IDs were replaced with deterministic
fixture IDs. System instructions, local paths, timestamps, environment details,
and unrelated tool definitions were removed. No authorization headers, keys,
user source code, or sensitive reasoning are present.

The capture established three Codex-required no-effect fields that differ from
the initial profile table: `reasoning.summary=auto`,
`include=["reasoning.encrypted_content"]`, and `prompt_cache_key`. The local
0.147.0 build also sent `client_metadata` and `text.verbosity` when its injected
model catalog enabled those features. `client_metadata` is accepted as a
bounded opaque object because its flat telemetry projections evolve with the
client; other no-effect fields remain restricted to the frozen forms in the
compatibility matrix. Unknown top-level request fields remain fail-closed.
The xLLM limit is one MiB of compact UTF-8 JSON, including object delimiters,
keys, values, and JSON escaping; values at the limit are accepted and larger
objects are rejected with `request_too_large` at `client_metadata`. The raw
request remains independently bounded by the Responses body limit.

Sources:

- Codex CLI tag `rust-v0.147.0`, commit `be6e8eac029b183056b7e4402879f15d2c85f61b`.
- DeepSeek Responses API documentation, captured 2026-08-14.
- OpenAI Responses create, streaming, and function-calling documentation,
  frozen 2026-08-14.
- `agent-workspace/openai-responses-api-phase0-5-spec.md` for the xLLM subset.

The non-stream request and all expected server outputs are normative golden
data, not claims of model execution. Model prompt/token/raw-output facts belong
to the DeepSeek V4 and GLM-5.2 characterization fixtures delivered separately.
