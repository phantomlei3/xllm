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

#include "core/model_protocol/glm_moe_dsa_profile.h"

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
    case ReasoningEffort::HIGH:
      return ReasoningEffort::HIGH;
    case ReasoningEffort::XHIGH:
    case ReasoningEffort::MAX:
      return ReasoningEffort::MAX;
  }
  return ReasoningEffort::HIGH;
}

}  // namespace

GlmMoeDsaProfile::GlmMoeDsaProfile()
    : identity_{.profile_id = "glm_moe_dsa_responses",
                .model_type = "glm_moe_dsa",
                .tokenizer_fingerprint =
                    "sha256:"
                    "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f"
                    "333a82d",
                .template_fingerprint =
                    "sha256:"
                    "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec957165"
                    "4489679"},
      capabilities_{.preserves_reasoning = true,
                    .supports_function_tools = true,
                    .supports_apply_patch = true,
                    .supports_reasoning_stream = true,
                    .supports_raw_decode = true},
      raw_decoding_{.policy = OutputDecodingPolicy::PROTOCOL_RAW,
                    .parser_dialect = "glm_moe_dsa_native",
                    .preserved_token_ids = {154820,
                                            154822,
                                            154824,
                                            154826,
                                            154827,
                                            154828,
                                            154829,
                                            154841,
                                            154842,
                                            154843,
                                            154844,
                                            154847,
                                            154848,
                                            154849,
                                            154850},
                    .preserved_token_sequences = {"</think>",
                                                  "<tool_call>",
                                                  "</tool_call>",
                                                  "<arg_key>",
                                                  "</arg_key>",
                                                  "<arg_value>",
                                                  "</arg_value>",
                                                  "<|observation|>"},
                    .max_marker_bytes = 15} {}

const ModelProtocolIdentity& GlmMoeDsaProfile::identity() const {
  return identity_;
}

const ModelProtocolCapabilities& GlmMoeDsaProfile::capabilities() const {
  return capabilities_;
}

const RawDecodingRequirements& GlmMoeDsaProfile::raw_decoding() const {
  return raw_decoding_;
}

ReasoningEffort GlmMoeDsaProfile::default_effort() const {
  return ReasoningEffort::HIGH;
}

SamplingPolicy GlmMoeDsaProfile::resolve_sampling(ReasoningEffort effort,
                                                  float temperature,
                                                  float top_p) const {
  const ReasoningEffort mapped_effort = map_effort(effort);
  if (mapped_effort != ReasoningEffort::NONE) {
    return {.effort = mapped_effort, .temperature = 1.0f, .top_p = 0.95f};
  }
  return {.effort = mapped_effort, .temperature = temperature, .top_p = top_p};
}

TemplatePolicy GlmMoeDsaProfile::resolve_template(
    ThinkingHistoryPolicy thinking) const {
  if (thinking == ThinkingHistoryPolicy::PRESERVE) {
    return {.thinking_history = thinking, .clear_thinking = false};
  }
  return {.thinking_history = thinking};
}

std::unique_ptr<ModelOutputParser> GlmMoeDsaProfile::new_parser() const {
  TextReasoningGrammar grammar;
  grammar.reasoning_end = "</think>";
  grammar.reasoning_end_token = 154842;
  grammar.text_end = "<|user|>";
  grammar.text_end_token = 154827;
  for (int64_t token_id : raw_decoding_.preserved_token_ids) {
    grammar.reserved_control_tokens.emplace_back(
        static_cast<int32_t>(token_id));
  }
  grammar.max_marker_bytes = raw_decoding_.max_marker_bytes;
  grammar.tool.dialect = ToolGrammarDialect::GLM_NATIVE;
  return make_text_reasoning_parser(std::move(grammar));
}

std::shared_ptr<const ModelProtocolProfile> make_glm_moe_dsa_profile() {
  return std::make_shared<GlmMoeDsaProfile>();
}

}  // namespace xllm::model_protocol
