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

#include "core/model_protocol/glm_5_2_profile.h"

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

Glm52Profile::Glm52Profile()
    : identity_{.profile_id = "glm_5_2_responses",
                .canonical_model_id = "GLM-5-W4A8",
                .model_aliases = {"/ds_models/GLM-5.2-W4A8"},
                .model_type = "glm_moe_dsa",
                .model_fingerprint =
                    "sha256:"
                    "0e4c1649072dda2d1c3f91196b33f390a3605e3732466dfb9e28ac4c"
                    "dd2e91f5",
                .tokenizer_id = "/ds_models/GLM-5.2-W4A8",
                .tokenizer_fingerprint =
                    "sha256:"
                    "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f"
                    "333a82d",
                .tokenizer_config_fingerprint =
                    "sha256:"
                    "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213"
                    "c6fd5fc",
                .template_id = "/ds_models/GLM-5.2-W4A8/chat_template.jinja",
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
                    .parser_dialect = "glm_5_2_native",
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

const ModelProtocolIdentity& Glm52Profile::identity() const {
  return identity_;
}

const ModelProtocolCapabilities& Glm52Profile::capabilities() const {
  return capabilities_;
}

const RawDecodingRequirements& Glm52Profile::raw_decoding() const {
  return raw_decoding_;
}

ReasoningEffort Glm52Profile::default_effort() const {
  return ReasoningEffort::HIGH;
}

SamplingPolicy Glm52Profile::resolve_sampling(ReasoningEffort effort,
                                              float temperature,
                                              float top_p) const {
  const ReasoningEffort mapped_effort = map_effort(effort);
  if (mapped_effort != ReasoningEffort::NONE) {
    return {.effort = mapped_effort, .temperature = 1.0f, .top_p = 0.95f};
  }
  return {.effort = mapped_effort, .temperature = temperature, .top_p = top_p};
}

TemplatePolicy Glm52Profile::resolve_template(
    ThinkingHistoryPolicy thinking) const {
  if (thinking == ThinkingHistoryPolicy::PRESERVE) {
    return {.thinking_history = thinking, .clear_thinking = false};
  }
  return {.thinking_history = thinking};
}

std::unique_ptr<ModelOutputParser> Glm52Profile::new_parser() const {
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
  return make_text_reasoning_parser(std::move(grammar));
}

std::shared_ptr<const ModelProtocolProfile> make_glm_5_2_profile() {
  return std::make_shared<Glm52Profile>();
}

}  // namespace xllm::model_protocol
