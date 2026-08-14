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
#include <vector>

#include "core/common/tool.h"
#include "core/model_protocol/policy.h"

namespace xllm::responses {

struct MessageItem {
  std::string id;
  std::string role;
  std::string content;
};

struct ReasoningItem {
  std::string id;
  std::string content;
};

struct FunctionCallItem {
  std::string id;
  FunctionCall call;
};

struct CustomToolCallItem {
  std::string id;
  CustomToolCall call;
};

using InputItem = std::variant<MessageItem,
                               ReasoningItem,
                               FunctionCallItem,
                               FunctionCallOutput,
                               CustomToolCallItem,
                               CustomToolCallOutput>;

using OutputItem = std::
    variant<MessageItem, ReasoningItem, FunctionCallItem, CustomToolCallItem>;

struct ResponsesOptions {
  bool stream = false;
  bool store = false;
  bool parallel_tool_calls = true;
  std::optional<uint32_t> max_output_tokens;
  model_protocol::ReasoningEffort reasoning_effort =
      model_protocol::ReasoningEffort::MEDIUM;
};

struct RequestContext {
  std::string request_id;
  std::string trace_id;
};

struct ResponsesRequest {
  std::string model;
  std::optional<std::string> instructions;
  std::vector<InputItem> input;
  std::vector<Tool> tools;
  ResponsesOptions options;
};

struct PreparedRequest {
  std::vector<InputItem> canonical_input;
  ResponsesOptions options;
  std::string profile_id;
  std::string canonical_model_id;
  RequestContext context;
};

}  // namespace xllm::responses
