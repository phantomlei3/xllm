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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/model_protocol/generation_delta.h"

namespace xllm::model_protocol {

enum class OutputSegmentKind : uint8_t {
  REASONING_DELTA = 0,
  REASONING_DONE = 1,
  TEXT_DELTA = 2,
  TEXT_DONE = 3,
  FUNCTION_CALL_START = 4,
  ARGUMENTS_DELTA = 5,
  FUNCTION_CALL_DONE = 6,
  CUSTOM_CALL_START = 7,
  CUSTOM_INPUT_DELTA = 8,
  CUSTOM_CALL_DONE = 9,
  PARSE_FAILURE = 10,
};

struct OutputSegment {
  OutputSegmentKind kind = OutputSegmentKind::REASONING_DELTA;
  std::string raw;
  std::string text;
  std::string name;
  std::optional<std::string> call_id;
  bool incomplete = false;
  std::optional<ParseFailure> failure;
};

inline bool operator==(const OutputSegment& left, const OutputSegment& right) {
  return left.kind == right.kind && left.raw == right.raw &&
         left.text == right.text && left.name == right.name &&
         left.call_id == right.call_id && left.incomplete == right.incomplete &&
         left.failure == right.failure;
}

enum class ToolGrammarDialect : uint8_t {
  NONE = 0,
  DEEPSEEK_DSML = 1,
  GLM_NATIVE = 2,
};

struct ToolGrammar {
  ToolGrammarDialect dialect = ToolGrammarDialect::NONE;
  size_t max_arguments_bytes = 1024 * 1024;
  size_t max_custom_input_bytes = 4 * 1024 * 1024;
  size_t max_json_depth = 64;
};

struct TextReasoningGrammar {
  std::string reasoning_end;
  int32_t reasoning_end_token = 0;
  std::string text_end;
  int32_t text_end_token = 0;
  std::vector<int32_t> reserved_control_tokens;
  size_t max_marker_bytes = 0;
  ToolGrammar tool;
};

class ModelOutputParser {
 public:
  virtual ~ModelOutputParser() = default;

  virtual std::vector<OutputSegment> consume(const GenerationDelta& delta) = 0;
};

std::unique_ptr<ModelOutputParser> make_text_reasoning_parser(
    TextReasoningGrammar grammar);

}  // namespace xllm::model_protocol
