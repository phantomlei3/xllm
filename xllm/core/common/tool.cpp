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

#include "core/common/tool.h"

#include <cstddef>
#include <utility>

namespace xllm {
namespace {

nlohmann::json value_to_json(const google::protobuf::Value& value);

nlohmann::json struct_to_json(const google::protobuf::Struct& value) {
  nlohmann::json result = nlohmann::json::object();
  for (const auto& field : value.fields()) {
    result[field.first] = value_to_json(field.second);
  }
  return result;
}

nlohmann::json value_to_json(const google::protobuf::Value& value) {
  switch (value.kind_case()) {
    case google::protobuf::Value::kNullValue:
      return nullptr;
    case google::protobuf::Value::kNumberValue:
      return value.number_value();
    case google::protobuf::Value::kStringValue:
      return value.string_value();
    case google::protobuf::Value::kBoolValue:
      return value.bool_value();
    case google::protobuf::Value::kStructValue:
      return struct_to_json(value.struct_value());
    case google::protobuf::Value::kListValue: {
      nlohmann::json result = nlohmann::json::array();
      result.get_ref<nlohmann::json::array_t&>().reserve(
          static_cast<size_t>(value.list_value().values_size()));
      for (const google::protobuf::Value& item : value.list_value().values()) {
        result.emplace_back(value_to_json(item));
      }
      return result;
    }
    case google::protobuf::Value::KIND_NOT_SET:
      return nullptr;
  }
  return nullptr;
}

}  // namespace

ToolConversion::ToolConversion(Tool value) : value_(std::move(value)) {}

ToolConversion::ToolConversion(ToolConversionError error) : error_(error) {}

ToolCallConversion::ToolCallConversion(ToolCall value)
    : value_(std::move(value)) {}

ToolCallConversion::ToolCallConversion(ToolConversionError error)
    : error_(error) {}

ToolOutputConversion::ToolOutputConversion(ToolOutput value)
    : value_(std::move(value)) {}

ToolOutputConversion::ToolOutputConversion(ToolConversionError error)
    : error_(error) {}

ToolConversion tool_from_proto(const proto::Tool& tool) {
  if (tool.type() == "function") {
    if (tool.payload_case() != proto::Tool::kFunction) {
      return ToolConversion(ToolConversionError::TYPE_PAYLOAD_MISMATCH);
    }
    const proto::Function& function = tool.function();
    nlohmann::json parameters = nlohmann::json::object();
    if (function.has_parameters()) {
      parameters = struct_to_json(function.parameters());
    }
    return ToolConversion(FunctionTool{.name = function.name(),
                                       .description = function.description(),
                                       .parameters = std::move(parameters)});
  }
  if (tool.type() == "custom") {
    if (tool.payload_case() != proto::Tool::kCustom) {
      return ToolConversion(ToolConversionError::TYPE_PAYLOAD_MISMATCH);
    }
    return ToolConversion(CustomTool{.name = tool.custom().name()});
  }
  return ToolConversion(ToolConversionError::UNSUPPORTED_TYPE);
}

ToolCallConversion tool_call_from_proto(const proto::ToolCall& call) {
  if (call.type() == "function") {
    if (call.payload_case() != proto::ToolCall::kFunction) {
      return ToolCallConversion(ToolConversionError::TYPE_PAYLOAD_MISMATCH);
    }
    return ToolCallConversion(
        FunctionCall{.id = call.id(),
                     .name = call.function().name(),
                     .arguments = call.function().arguments()});
  }
  if (call.type() == "custom") {
    if (call.payload_case() != proto::ToolCall::kCustom) {
      return ToolCallConversion(ToolConversionError::TYPE_PAYLOAD_MISMATCH);
    }
    return ToolCallConversion(CustomToolCall{.id = call.id(),
                                             .name = call.custom().name(),
                                             .input = call.custom().input()});
  }
  return ToolCallConversion(ToolConversionError::UNSUPPORTED_TYPE);
}

ToolOutputConversion tool_output_from_proto(const proto::ToolOutput& output) {
  if (output.payload_case() == proto::ToolOutput::kFunction) {
    return ToolOutputConversion(
        FunctionCallOutput{.call_id = output.function().call_id(),
                           .output = output.function().output()});
  }
  if (output.payload_case() == proto::ToolOutput::kCustom) {
    return ToolOutputConversion(
        CustomToolCallOutput{.call_id = output.custom().call_id(),
                             .output = output.custom().output()});
  }
  return ToolOutputConversion(ToolConversionError::TYPE_PAYLOAD_MISMATCH);
}

}  // namespace xllm
