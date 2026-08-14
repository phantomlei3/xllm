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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm::model_protocol {

enum class ParseFailureCode : uint8_t {
  INVALID_ORDINAL = 0,
  CUMULATIVE_PREFIX_MISMATCH = 1,
  DUPLICATE_CALLBACK = 2,
  INVALID_UTF8 = 3,
  UNKNOWN_CONTROL_TOKEN = 4,
  SEQUENCE_MISMATCH = 5,
  DELTA_AFTER_FINISH = 6,
  BACKEND_ERROR = 7,
  CONTROL_TOKEN_MISMATCH = 8,
  INVALID_TOOL_ARGUMENTS = 9,
  TOOL_ARGUMENTS_TOO_LARGE = 10,
  CUSTOM_INPUT_TOO_LARGE = 11,
  UNKNOWN_CUSTOM_TOOL = 12,
  UNKNOWN_TOOL_GRAMMAR = 13,
  UNCLOSED_TOOL_CALL = 14,
};

struct ParseFailure {
  ParseFailureCode code = ParseFailureCode::INVALID_ORDINAL;
  std::string message;
};

inline bool operator==(const ParseFailure& left, const ParseFailure& right) {
  return left.code == right.code && left.message == right.message;
}

struct GenerationUsage {
  int32_t input_tokens = 0;
  int32_t cached_input_tokens = 0;
  int32_t output_tokens = 0;
  std::optional<int32_t> reasoning_tokens;
  int32_t total_tokens = 0;
};

struct BackendError {
  std::string code;
  std::string message;
};

struct GenerationDelta {
  size_t sequence_index = 0;
  uint64_t generation_ordinal = 0;
  std::string text_delta;
  std::vector<int32_t> token_id_delta;
  bool finished = false;
  std::optional<std::string> finish_reason;
  std::optional<GenerationUsage> final_usage;
  std::optional<BackendError> backend_error;
};

struct CumulativeGeneration {
  size_t sequence_index = 0;
  uint64_t generation_ordinal = 0;
  std::string text;
  std::vector<int32_t> token_ids;
  bool finished = false;
  std::optional<std::string> finish_reason;
  std::optional<GenerationUsage> final_usage;
  std::optional<BackendError> backend_error;
};

class NormalizationResult final {
 public:
  explicit NormalizationResult(GenerationDelta delta);
  explicit NormalizationResult(ParseFailure failure);

  bool ok() const { return delta_.has_value(); }
  const GenerationDelta& delta() const { return *delta_; }
  const ParseFailure& failure() const { return *failure_; }

 private:
  std::optional<GenerationDelta> delta_;
  std::optional<ParseFailure> failure_;
};

class RequestOutputNormalizer final {
 public:
  NormalizationResult normalize(const CumulativeGeneration& output);

 private:
  struct SequenceState {
    uint64_t ordinal = 0;
    std::string text;
    std::vector<int32_t> token_ids;
    bool initialized = false;
    bool finished = false;
  };

  std::unordered_map<size_t, SequenceState> sequences_;
};

}  // namespace xllm::model_protocol
