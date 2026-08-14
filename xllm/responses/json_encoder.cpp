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

#include "responses/json_encoder.h"

#include <string>
#include <type_traits>

namespace xllm::responses {
namespace {

const char* status_name(ResponseStatus status) {
  switch (status) {
    case ResponseStatus::IN_PROGRESS:
      return "in_progress";
    case ResponseStatus::COMPLETED:
      return "completed";
    case ResponseStatus::INCOMPLETE:
      return "incomplete";
    case ResponseStatus::FAILED:
      return "failed";
    case ResponseStatus::CANCELLED:
      return "cancelled";
  }
  return "failed";
}

const char* item_status_name(ItemStatus status) {
  switch (status) {
    case ItemStatus::IN_PROGRESS:
      return "in_progress";
    case ItemStatus::COMPLETED:
      return "completed";
    case ItemStatus::INCOMPLETE:
      return "incomplete";
  }
  return "incomplete";
}

const char* effort_name(model_protocol::ReasoningEffort effort) {
  using model_protocol::ReasoningEffort;
  switch (effort) {
    case ReasoningEffort::NONE:
      return "none";
    case ReasoningEffort::MINIMAL:
      return "minimal";
    case ReasoningEffort::LOW:
      return "low";
    case ReasoningEffort::MEDIUM:
      return "medium";
    case ReasoningEffort::HIGH:
      return "high";
    case ReasoningEffort::XHIGH:
      return "xhigh";
    case ReasoningEffort::MAX:
      return "max";
  }
  return "medium";
}

const char* error_code_name(ErrorCode code) {
  switch (code) {
    case ErrorCode::INVALID_TOOL_ARGUMENTS:
      return "invalid_tool_arguments";
    case ErrorCode::REQUEST_TOO_LARGE:
      return "request_too_large";
    case ErrorCode::GENERATION_FAILED:
      return "generation_failed";
    case ErrorCode::REQUEST_CANCELLED:
      return "request_cancelled";
    case ErrorCode::REQUEST_TIMEOUT:
      return "request_timeout";
    default:
      return "invalid_request";
  }
}

nlohmann::json encode_usage(const model_protocol::GenerationUsage& usage) {
  return {
      {"input_tokens", usage.input_tokens},
      {"input_tokens_details", {{"cached_tokens", usage.cached_input_tokens}}},
      {"output_tokens", usage.output_tokens},
      {"output_tokens_details", {{"reasoning_tokens", usage.reasoning_tokens}}},
      {"total_tokens", usage.total_tokens}};
}

}  // namespace

nlohmann::json encode_item(const OutputItem& item) {
  return std::visit(
      [](const auto& value) -> nlohmann::json {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, OutputMessageItem>) {
          return {{"id", value.id},
                  {"type", "message"},
                  {"status", item_status_name(value.status)},
                  {"role", "assistant"},
                  {"content",
                   nlohmann::json::array(
                       {{{"type", "output_text"},
                         {"text", value.content},
                         {"annotations", nlohmann::json::array()}}})}};
        } else if constexpr (std::is_same_v<Value, OutputReasoningItem>) {
          nlohmann::json result = {
              {"id", value.id},
              {"type", "reasoning"},
              {"content",
               nlohmann::json::array(
                   {{{"type", "reasoning_text"}, {"text", value.content}}})}};
          if (value.status == ItemStatus::INCOMPLETE) {
            result["status"] = "incomplete";
          }
          return result;
        } else if constexpr (std::is_same_v<Value, OutputFunctionCallItem>) {
          return {{"id", value.id},
                  {"type", "function_call"},
                  {"call_id", value.call.id},
                  {"name", value.call.name},
                  {"arguments", value.call.arguments},
                  {"status", item_status_name(value.status)}};
        } else {
          return {{"id", value.id},
                  {"type", "custom_tool_call"},
                  {"call_id", value.call.id},
                  {"name", value.call.name},
                  {"input", value.call.input},
                  {"status", item_status_name(value.status)}};
        }
      },
      item);
}

nlohmann::json encode_response(const FinalResponse& response) {
  nlohmann::json output = nlohmann::json::array();
  output.get_ref<nlohmann::json::array_t&>().reserve(response.output.size());
  for (const OutputItem& item : response.output) {
    output.emplace_back(encode_item(item));
  }
  nlohmann::json error = nullptr;
  if (response.error.has_value()) {
    error = {
        {"message", response.error->message},
        {"type", response.error->type},
        {"param",
         response.error->param.empty() ? nlohmann::json(nullptr)
                                       : nlohmann::json(response.error->param)},
        {"code", error_code_name(response.error->code)}};
  }
  nlohmann::json incomplete_details = nullptr;
  if (response.incomplete_reason.has_value()) {
    incomplete_details = {{"reason", *response.incomplete_reason}};
  }
  nlohmann::json result = {
      {"id", response.id},
      {"object", response.object},
      {"created_at", response.created_at},
      {"status", status_name(response.status)},
      {"error", std::move(error)},
      {"incomplete_details", std::move(incomplete_details)},
      {"model", response.model},
      {"output", std::move(output)},
      {"parallel_tool_calls", response.parallel_tool_calls},
      {"previous_response_id", nullptr},
      {"reasoning",
       {{"effort", effort_name(response.reasoning_effort)},
        {"summary", nullptr}}},
      {"store", response.store},
      {"usage",
       response.usage.has_value() ? encode_usage(*response.usage)
                                  : nlohmann::json(nullptr)}};
  return result;
}

}  // namespace xllm::responses
