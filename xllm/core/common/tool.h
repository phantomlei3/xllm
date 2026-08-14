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
#include <optional>
#include <string>
#include <variant>

#include "common.pb.h"
#include "nlohmann/json.hpp"

namespace xllm {

struct FunctionTool {
  std::string name;
  std::string description;
  nlohmann::json parameters = nlohmann::json::object();
};

struct CustomTool {
  std::string name;
};

using Tool = std::variant<FunctionTool, CustomTool>;

struct FunctionCall {
  std::string id;
  std::string name;
  std::string arguments;
};

struct CustomToolCall {
  std::string id;
  std::string name;
  std::string input;
};

using ToolCall = std::variant<FunctionCall, CustomToolCall>;

struct FunctionCallOutput {
  std::string id;
  std::string call_id;
  std::string output;
};

struct CustomToolCallOutput {
  std::string id;
  std::string call_id;
  std::string output;
};

using ToolOutput = std::variant<FunctionCallOutput, CustomToolCallOutput>;

enum class ToolOutputKind : uint8_t {
  UNSPECIFIED = 0,
  FUNCTION = 1,
  CUSTOM = 2,
};

enum class ToolConversionError : uint8_t {
  NONE = 0,
  TYPE_PAYLOAD_MISMATCH = 1,
  UNSUPPORTED_TYPE = 2,
};

class ToolConversion final {
 public:
  explicit ToolConversion(Tool value);
  explicit ToolConversion(ToolConversionError error);

  bool ok() const { return value_.has_value(); }
  const std::optional<Tool>& value() const { return value_; }
  ToolConversionError error() const { return error_; }

 private:
  std::optional<Tool> value_;
  ToolConversionError error_ = ToolConversionError::NONE;
};

class ToolCallConversion final {
 public:
  explicit ToolCallConversion(ToolCall value);
  explicit ToolCallConversion(ToolConversionError error);

  bool ok() const { return value_.has_value(); }
  const std::optional<ToolCall>& value() const { return value_; }
  ToolConversionError error() const { return error_; }

 private:
  std::optional<ToolCall> value_;
  ToolConversionError error_ = ToolConversionError::NONE;
};

class ToolOutputConversion final {
 public:
  explicit ToolOutputConversion(ToolOutput value);
  explicit ToolOutputConversion(ToolConversionError error);

  bool ok() const { return value_.has_value(); }
  const std::optional<ToolOutput>& value() const { return value_; }
  ToolConversionError error() const { return error_; }

 private:
  std::optional<ToolOutput> value_;
  ToolConversionError error_ = ToolConversionError::NONE;
};

ToolConversion tool_from_proto(const proto::Tool& tool);
ToolCallConversion tool_call_from_proto(const proto::ToolCall& call);
ToolOutputConversion tool_output_from_proto(const proto::ToolOutput& output);

}  // namespace xllm
