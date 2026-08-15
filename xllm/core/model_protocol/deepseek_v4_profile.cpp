/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/model_protocol/deepseek_v4_profile.h"

#include <memory>
#include <utility>

namespace xllm::model_protocol {
namespace {

ReasoningEffort map_effort(ReasoningEffort effort) {
  switch (effort) {
    case ReasoningEffort::NONE:
      return ReasoningEffort::NONE;
    case ReasoningEffort::MINIMAL:
    case ReasoningEffort::LOW:
    case ReasoningEffort::MEDIUM:
      return ReasoningEffort::LOW;
    case ReasoningEffort::HIGH:
    case ReasoningEffort::XHIGH:
      return ReasoningEffort::HIGH;
    case ReasoningEffort::MAX:
      return ReasoningEffort::MAX;
  }
  return ReasoningEffort::HIGH;
}

}  // namespace

DeepseekV4Profile::DeepseekV4Profile()
    : identity_{.profile_id = "deepseek_v4_responses",
                .model_type = "deepseek_v4",
                .tokenizer_fingerprint =
                    "sha256:"
                    "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c"
                    "8633cf",
                .template_fingerprint =
                    "sha256:"
                    "a06d98abfe8e4d78d2505ca28464e6b4af203731d6c07ee1ca9c4ab1"
                    "bd77f95b"},
      capabilities_{.preserves_reasoning = true,
                    .supports_function_tools = true,
                    .supports_apply_patch = true,
                    .supports_reasoning_stream = true,
                    .supports_raw_decode = true},
      raw_decoding_{
          .policy = OutputDecodingPolicy::PROTOCOL_RAW,
          .parser_dialect = "deepseek_v4_dsml",
          .preserved_token_ids = {0, 1, 128803, 128804, 128821, 128822, 128825},
          .preserved_token_sequences = {"</think>",
                                        "<｜DSML｜tool_calls>",
                                        "</｜DSML｜tool_calls>",
                                        "<｜DSML｜invoke name=\"",
                                        "<｜DSML｜parameter name=\"",
                                        "<｜end▁of▁sentence｜>"},
          .max_marker_bytes = 23} {}

const ModelProtocolIdentity& DeepseekV4Profile::identity() const {
  return identity_;
}

const ModelProtocolCapabilities& DeepseekV4Profile::capabilities() const {
  return capabilities_;
}

const RawDecodingRequirements& DeepseekV4Profile::raw_decoding() const {
  return raw_decoding_;
}

ReasoningEffort DeepseekV4Profile::default_effort() const {
  return ReasoningEffort::MEDIUM;
}

SamplingPolicy DeepseekV4Profile::resolve_sampling(ReasoningEffort effort,
                                                   float temperature,
                                                   float top_p) const {
  const ReasoningEffort mapped_effort = map_effort(effort);
  if (mapped_effort != ReasoningEffort::NONE) {
    return {.effort = mapped_effort, .temperature = 1.0f, .top_p = 0.95f};
  }
  return {.effort = mapped_effort, .temperature = temperature, .top_p = top_p};
}

TemplatePolicy DeepseekV4Profile::resolve_template(
    ThinkingHistoryPolicy thinking) const {
  if (thinking == ThinkingHistoryPolicy::PRESERVE) {
    return {.thinking_history = thinking, .drop_thinking = false};
  }
  return {.thinking_history = thinking};
}

std::unique_ptr<ModelOutputParser> DeepseekV4Profile::new_parser() const {
  TextReasoningGrammar grammar;
  grammar.reasoning_end = "</think>";
  grammar.reasoning_end_tokens = {128822};
  grammar.text_end = "<｜end▁of▁sentence｜>";
  grammar.text_end_tokens = {1};
  grammar.tool_open_tokens = {30, 128825, 72461, 4941, 12548, 1018};
  grammar.tool_done_tokens = {1718, 128825, 72461, 4941, 12548, 32, 1};
  grammar.max_marker_bytes = raw_decoding_.max_marker_bytes;
  grammar.tool.dialect = ToolGrammarDialect::DEEPSEEK_DSML;
  return make_text_reasoning_parser(std::move(grammar));
}

std::shared_ptr<const ModelProtocolProfile> make_deepseek_v4_profile() {
  return std::make_shared<DeepseekV4Profile>();
}

}  // namespace xllm::model_protocol
