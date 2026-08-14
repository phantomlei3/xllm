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

#include "responses/sse_encoder.h"

#include <type_traits>

#include "responses/json_encoder.h"

namespace xllm::responses {
namespace {

template <typename Value>
constexpr std::string_view type_name() {
  if constexpr (std::is_same_v<Value, ResponseCreatedEvent>) {
    return "response.created";
  } else if constexpr (std::is_same_v<Value, ResponseInProgressEvent>) {
    return "response.in_progress";
  } else if constexpr (std::is_same_v<Value, ResponseCompletedEvent>) {
    return "response.completed";
  } else if constexpr (std::is_same_v<Value, ResponseIncompleteEvent>) {
    return "response.incomplete";
  } else if constexpr (std::is_same_v<Value, ResponseFailedEvent>) {
    return "response.failed";
  } else if constexpr (std::is_same_v<Value, OutputItemAddedEvent>) {
    return "response.output_item.added";
  } else if constexpr (std::is_same_v<Value, OutputItemDoneEvent>) {
    return "response.output_item.done";
  } else if constexpr (std::is_same_v<Value, ContentPartAddedEvent>) {
    return "response.content_part.added";
  } else if constexpr (std::is_same_v<Value, ContentPartDoneEvent>) {
    return "response.content_part.done";
  } else if constexpr (std::is_same_v<Value, OutputTextDeltaEvent>) {
    return "response.output_text.delta";
  } else if constexpr (std::is_same_v<Value, OutputTextDoneEvent>) {
    return "response.output_text.done";
  } else if constexpr (std::is_same_v<Value, ReasoningTextDeltaEvent>) {
    return "response.reasoning_text.delta";
  } else if constexpr (std::is_same_v<Value, ReasoningTextDoneEvent>) {
    return "response.reasoning_text.done";
  } else if constexpr (std::is_same_v<Value, FunctionArgumentsDeltaEvent>) {
    return "response.function_call_arguments.delta";
  } else if constexpr (std::is_same_v<Value, FunctionArgumentsDoneEvent>) {
    return "response.function_call_arguments.done";
  } else if constexpr (std::is_same_v<Value, CustomInputDeltaEvent>) {
    return "response.custom_tool_call_input.delta";
  } else {
    return "response.custom_tool_call_input.done";
  }
}

nlohmann::json base_event(const ResponseEvent& event) {
  return {{"type", event_type(event)},
          {"sequence_number", event.sequence_number}};
}

}  // namespace

std::string_view event_type(const ResponseEvent& event) {
  return std::visit(
      [](const auto& value) {
        using Value = std::decay_t<decltype(value)>;
        return type_name<Value>();
      },
      event.data);
}

nlohmann::json encode_event(const ResponseEvent& event) {
  nlohmann::json encoded = base_event(event);
  std::visit(
      [&encoded](const auto& value) {
        using Value = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Value, ResponseCreatedEvent> ||
                      std::is_same_v<Value, ResponseInProgressEvent> ||
                      std::is_same_v<Value, ResponseCompletedEvent> ||
                      std::is_same_v<Value, ResponseIncompleteEvent> ||
                      std::is_same_v<Value, ResponseFailedEvent>) {
          encoded["response"] = encode_response(value.response);
        } else if constexpr (std::is_same_v<Value, OutputItemAddedEvent> ||
                             std::is_same_v<Value, OutputItemDoneEvent>) {
          encoded["output_index"] = value.output_index;
          encoded["item"] = encode_item(value.item);
        } else if constexpr (std::is_same_v<Value, ContentPartAddedEvent>) {
          encoded["output_index"] = value.output_index;
          encoded["content_index"] = 0;
          encoded["item_id"] = value.item_id;
          encoded["part"] = {{"type", "output_text"},
                             {"text", ""},
                             {"annotations", nlohmann::json::array()}};
        } else if constexpr (std::is_same_v<Value, ContentPartDoneEvent>) {
          encoded["output_index"] = value.output_index;
          encoded["content_index"] = 0;
          encoded["item_id"] = value.item_id;
          encoded["part"] = {{"type", "output_text"},
                             {"text", value.text},
                             {"annotations", nlohmann::json::array()}};
        } else {
          encoded["output_index"] = value.output_index;
          encoded["item_id"] = value.item_id;
          if constexpr (std::is_same_v<Value, OutputTextDeltaEvent> ||
                        std::is_same_v<Value, ReasoningTextDeltaEvent>) {
            encoded["content_index"] = 0;
            encoded["delta"] = value.delta;
          } else if constexpr (std::is_same_v<Value, OutputTextDoneEvent> ||
                               std::is_same_v<Value, ReasoningTextDoneEvent>) {
            encoded["content_index"] = 0;
            encoded["text"] = value.text;
          } else if constexpr (std::is_same_v<Value,
                                              FunctionArgumentsDeltaEvent> ||
                               std::is_same_v<Value, CustomInputDeltaEvent>) {
            encoded["delta"] = value.delta;
          } else if constexpr (std::is_same_v<Value,
                                              FunctionArgumentsDoneEvent>) {
            encoded["arguments"] = value.arguments;
          } else {
            encoded["input"] = value.input;
          }
        }
      },
      event.data);
  return encoded;
}

std::string encode_sse(const ResponseEvent& event) {
  return "event: " + std::string(event_type(event)) +
         "\ndata: " + encode_event(event).dump() + "\n\n";
}

}  // namespace xllm::responses
