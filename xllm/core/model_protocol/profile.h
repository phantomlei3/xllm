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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/model_protocol/output_parser.h"
#include "core/model_protocol/policy.h"

namespace xllm::model_protocol {

struct ModelProtocolIdentity {
  std::string profile_id;
  std::string canonical_model_id;
  std::vector<std::string> model_aliases;
  std::string tokenizer_id;
  std::string template_id;
  std::string template_fingerprint;
};

struct LoadedModelContext {
  std::string model_id;
  std::string tokenizer_id;
  std::string template_id;
  std::string template_fingerprint;
};

struct ModelProtocolCapabilities {
  bool preserves_reasoning = false;
  bool supports_function_tools = false;
  bool supports_apply_patch = false;
  bool supports_reasoning_stream = false;
  bool supports_raw_decode = false;
};

struct RawDecodingRequirements {
  OutputDecodingPolicy policy = OutputDecodingPolicy::PROTOCOL_RAW;
  std::vector<int64_t> preserved_token_ids;
  std::vector<std::string> preserved_token_sequences;
  uint32_t max_marker_bytes = 0;
};

class ModelProtocolProfile {
 public:
  virtual ~ModelProtocolProfile() = default;

  virtual const ModelProtocolIdentity& identity() const = 0;
  virtual const ModelProtocolCapabilities& capabilities() const = 0;
  virtual const RawDecodingRequirements& raw_decoding() const = 0;
  virtual ReasoningEffort default_effort() const = 0;
  virtual SamplingPolicy resolve_sampling(ReasoningEffort effort,
                                          float temperature,
                                          float top_p) const = 0;
  virtual TemplatePolicy resolve_template(
      ThinkingHistoryPolicy thinking) const = 0;
  virtual std::unique_ptr<ModelOutputParser> new_parser() const = 0;
};

}  // namespace xllm::model_protocol
