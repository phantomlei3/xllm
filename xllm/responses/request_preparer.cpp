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

#include "responses/request_preparer.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_set>
#include <utility>

#include "nlohmann/json.hpp"

namespace xllm::responses {
namespace {

using Json = nlohmann::json;

constexpr size_t kMaxIdBytes = 256;

ResponsesError make_error(ErrorCode code,
                          std::string param,
                          std::string message) {
  return {.message = std::move(message),
          .type = "invalid_request_error",
          .param = std::move(param),
          .code = code};
}

ResponsesError invalid(std::string param, std::string message) {
  return make_error(
      ErrorCode::INVALID_REQUEST, std::move(param), std::move(message));
}

ResponsesError unsupported(std::string param, std::string message) {
  return make_error(
      ErrorCode::UNSUPPORTED_PARAMETER, std::move(param), std::move(message));
}

bool exceeds_depth(const std::string& body, uint32_t max_depth) {
  uint32_t depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (char byte : body) {
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (byte == '\\') {
        escaped = true;
      } else if (byte == '"') {
        in_string = false;
      }
      continue;
    }
    if (byte == '"') {
      in_string = true;
    } else if (byte == '{' || byte == '[') {
      ++depth;
      if (depth > max_depth) {
        return true;
      }
    } else if ((byte == '}' || byte == ']') && depth > 0) {
      --depth;
    }
  }
  return false;
}

bool known_field(const std::string& field,
                 std::initializer_list<std::string_view> allowed) {
  return std::any_of(
      allowed.begin(), allowed.end(), [&field](std::string_view value) {
        return field == value;
      });
}

std::optional<ResponsesError> reject_unknown(
    const Json& object,
    std::initializer_list<std::string_view> allowed,
    const std::string& prefix) {
  for (auto it = object.begin(); it != object.end(); ++it) {
    const std::string& field = it.key();
    if (!known_field(field, allowed)) {
      const std::string param = prefix.empty() ? field : prefix + "." + field;
      return unsupported(param, "unsupported field");
    }
  }
  return std::nullopt;
}

bool valid_id(const std::string& id) {
  if (id.empty() || id.size() > kMaxIdBytes) {
    return false;
  }
  return std::all_of(id.begin(), id.end(), [](char byte) {
    const unsigned char value = static_cast<unsigned char>(byte);
    return std::isalnum(value) != 0 || byte == '_' || byte == '-' ||
           byte == '.';
  });
}

std::optional<ResponsesError> read_text(const Json& value,
                                        const std::string& param,
                                        const ResponsesLimits& limits,
                                        std::string* output) {
  if (!value.is_string()) {
    return invalid(param, "expected a string");
  }
  const std::string& text = value.get_ref<const std::string&>();
  if (text.size() > limits.max_text_bytes) {
    return make_error(ErrorCode::REQUEST_TOO_LARGE,
                      param,
                      "text exceeds the configured limit");
  }
  *output = text;
  return std::nullopt;
}

std::optional<ResponsesError> read_id(const Json& item,
                                      const std::string& param,
                                      std::unordered_set<std::string>* item_ids,
                                      std::string* id) {
  if (!item.contains("id")) {
    return std::nullopt;
  }
  if (!item["id"].is_string()) {
    return invalid(param, "item ID must be a string");
  }
  *id = item["id"].get<std::string>();
  if (!valid_id(*id) || !item_ids->emplace(*id).second) {
    return invalid(param, "item ID is invalid or duplicated");
  }
  return std::nullopt;
}

std::optional<ResponsesError> read_parts(const Json& content,
                                         const std::string& param,
                                         std::string_view expected_type,
                                         const ResponsesLimits& limits,
                                         std::string* text) {
  if (content.is_string()) {
    return read_text(content, param, limits, text);
  }
  if (!content.is_array()) {
    return invalid(param, "content must be a string or array");
  }
  std::string joined;
  for (size_t index = 0; index < content.size(); ++index) {
    const Json& part = content[index];
    const std::string part_param = param + "[" + std::to_string(index) + "]";
    if (!part.is_object() || !part.contains("type") ||
        !part["type"].is_string()) {
      return invalid(part_param, "content part must have a string type");
    }
    if (part["type"] != expected_type) {
      return make_error(ErrorCode::UNSUPPORTED_CONTENT_TYPE,
                        part_param + ".type",
                        "unsupported content type");
    }
    if (std::optional<ResponsesError> error =
            reject_unknown(part, {"type", "text"}, part_param)) {
      return error;
    }
    if (!part.contains("text")) {
      return invalid(part_param + ".text", "content text is required");
    }
    std::string part_text;
    if (std::optional<ResponsesError> error =
            read_text(part["text"], part_param + ".text", limits, &part_text)) {
      return error;
    }
    if (joined.size() > limits.max_text_bytes - part_text.size()) {
      return make_error(ErrorCode::REQUEST_TOO_LARGE,
                        param,
                        "text exceeds the configured limit");
    }
    joined += part_text;
  }
  *text = std::move(joined);
  return std::nullopt;
}

class RequestAssembler final {
 public:
  RequestAssembler(const model_protocol::ModelProtocolIdentity& profile,
                   const RequestContext& context,
                   const ResponsesLimits& limits)
      : profile_(profile), context_(context), limits_(limits) {}

  PrepareResult assemble(const Json& body) {
    if (std::optional<ResponsesError> error =
            reject_unknown(body,
                           {"model",
                            "input",
                            "instructions",
                            "stream",
                            "store",
                            "previous_response_id",
                            "parallel_tool_calls",
                            "reasoning"},
                           "")) {
      return PrepareResult(std::move(*error));
    }
    if (!body.contains("model") || !body["model"].is_string() ||
        body["model"].get_ref<const std::string&>().empty()) {
      return PrepareResult(invalid("model", "model is required"));
    }
    const std::string model = body["model"].get<std::string>();
    if (!model_matches(model)) {
      return PrepareResult(make_error(
          ErrorCode::MODEL_MISMATCH, "model", "model does not match profile"));
    }
    if (!body.contains("input") && !body.contains("instructions")) {
      return PrepareResult(
          invalid("input", "input or instructions is required"));
    }

    PreparedRequest prepared;
    prepared.profile_id = profile_.profile_id;
    prepared.canonical_model_id = profile_.canonical_model_id;
    prepared.context = context_;
    init_chat(&prepared);

    if (std::optional<ResponsesError> error = parse_options(body, &prepared)) {
      return PrepareResult(std::move(*error));
    }
    if (body.contains("instructions")) {
      std::string instructions;
      if (std::optional<ResponsesError> error = read_text(
              body["instructions"], "instructions", limits_, &instructions)) {
        return PrepareResult(std::move(*error));
      }
      if (instructions.empty()) {
        return PrepareResult(
            invalid("instructions", "instructions must not be empty"));
      }
      add_message("", "system", instructions, &prepared);
    }
    if (body.contains("input")) {
      if (std::optional<ResponsesError> error =
              parse_input(body["input"], &prepared)) {
        return PrepareResult(std::move(*error));
      }
    }
    if (prepared.canonical_input.empty()) {
      return PrepareResult(invalid("input", "input must not be empty"));
    }
    return PrepareResult(std::move(prepared));
  }

 private:
  enum class AssistantState : uint8_t {
    NONE = 0,
    REASONING = 1,
    MESSAGE = 2,
  };

  bool model_matches(const std::string& model) const {
    if (model == profile_.canonical_model_id) {
      return true;
    }
    return std::find(profile_.model_aliases.begin(),
                     profile_.model_aliases.end(),
                     model) != profile_.model_aliases.end();
  }

  void init_chat(PreparedRequest* prepared) const {
    prepared->chat_request.set_model(profile_.canonical_model_id);
    prepared->chat_request.set_n(1);
    prepared->chat_request.set_thinking_history_policy(proto::PRESERVE);
    prepared->chat_request.set_output_decoding_policy(proto::PROTOCOL_RAW);
    prepared->chat_request.set_request_id(context_.request_id);
  }

  std::optional<ResponsesError> parse_options(const Json& body,
                                              PreparedRequest* prepared) const {
    if (body.contains("stream")) {
      if (!body["stream"].is_boolean()) {
        return invalid("stream", "stream must be a boolean");
      }
      prepared->options.stream = body["stream"].get<bool>();
    }
    prepared->chat_request.set_stream(prepared->options.stream);
    if (body.contains("store")) {
      if (!body["store"].is_boolean()) {
        return invalid("store", "store must be a boolean");
      }
      if (body["store"].get<bool>()) {
        return unsupported("store", "stored responses are not supported");
      }
    }
    if (body.contains("previous_response_id") &&
        !body["previous_response_id"].is_null()) {
      return unsupported("previous_response_id",
                         "previous responses are not supported");
    }
    if (body.contains("parallel_tool_calls")) {
      if (!body["parallel_tool_calls"].is_boolean()) {
        return invalid("parallel_tool_calls",
                       "parallel_tool_calls must be a boolean");
      }
      if (!body["parallel_tool_calls"].get<bool>()) {
        return unsupported("parallel_tool_calls",
                           "parallel tool calls are fixed to true");
      }
    }
    if (body.contains("reasoning")) {
      return parse_reasoning(body["reasoning"], &prepared->options);
    }
    return std::nullopt;
  }

  std::optional<ResponsesError> parse_reasoning(
      const Json& reasoning,
      ResponsesOptions* options) const {
    if (!reasoning.is_object()) {
      return invalid("reasoning", "reasoning must be an object");
    }
    if (std::optional<ResponsesError> error =
            reject_unknown(reasoning, {"effort"}, "reasoning")) {
      return error;
    }
    if (!reasoning.contains("effort") || !reasoning["effort"].is_string()) {
      return invalid("reasoning.effort", "reasoning effort is required");
    }
    const std::string effort = reasoning["effort"].get<std::string>();
    if (effort == "none") {
      options->reasoning_effort = model_protocol::ReasoningEffort::NONE;
    } else if (effort == "minimal") {
      options->reasoning_effort = model_protocol::ReasoningEffort::MINIMAL;
    } else if (effort == "low") {
      options->reasoning_effort = model_protocol::ReasoningEffort::LOW;
    } else if (effort == "medium") {
      options->reasoning_effort = model_protocol::ReasoningEffort::MEDIUM;
    } else if (effort == "high") {
      options->reasoning_effort = model_protocol::ReasoningEffort::HIGH;
    } else if (effort == "xhigh") {
      options->reasoning_effort = model_protocol::ReasoningEffort::XHIGH;
    } else if (effort == "max") {
      options->reasoning_effort = model_protocol::ReasoningEffort::MAX;
    } else {
      return invalid("reasoning.effort", "unsupported reasoning effort");
    }
    return std::nullopt;
  }

  std::optional<ResponsesError> parse_input(const Json& input,
                                            PreparedRequest* prepared) {
    if (input.is_string()) {
      if (limits_.max_input_items == 0) {
        return make_error(
            ErrorCode::TOO_MANY_ITEMS, "input", "too many input items");
      }
      std::string text;
      if (std::optional<ResponsesError> error =
              read_text(input, "input", limits_, &text)) {
        return error;
      }
      if (text.empty()) {
        return invalid("input", "input must not be empty");
      }
      add_message("", "user", text, prepared);
      return std::nullopt;
    }
    if (!input.is_array()) {
      return invalid("input", "input must be a string or item array");
    }
    if (input.size() > limits_.max_input_items) {
      return make_error(
          ErrorCode::TOO_MANY_ITEMS, "input", "too many input items");
    }
    for (size_t index = 0; index < input.size(); ++index) {
      if (std::optional<ResponsesError> error =
              parse_item(input[index], index, prepared)) {
        return error;
      }
    }
    return std::nullopt;
  }

  std::optional<ResponsesError> parse_item(const Json& item,
                                           size_t index,
                                           PreparedRequest* prepared) {
    const std::string param = "input[" + std::to_string(index) + "]";
    if (!item.is_object() || !item.contains("type") ||
        !item["type"].is_string()) {
      return invalid(param + ".type", "item type is required");
    }
    const std::string type = item["type"].get<std::string>();
    if (type == "message") {
      return parse_message(item, param, prepared);
    }
    if (type == "reasoning") {
      return parse_reasoning_item(item, param, prepared);
    }
    return make_error(ErrorCode::UNSUPPORTED_ITEM_TYPE,
                      param + ".type",
                      "unsupported input item type");
  }

  std::optional<ResponsesError> parse_message(const Json& item,
                                              const std::string& param,
                                              PreparedRequest* prepared) {
    if (std::optional<ResponsesError> error =
            reject_unknown(item, {"type", "id", "role", "content"}, param)) {
      return error;
    }
    if (!item.contains("role") || !item["role"].is_string()) {
      return invalid(param + ".role", "message role is required");
    }
    const std::string role = item["role"].get<std::string>();
    if (role != "user" && role != "assistant" && role != "system" &&
        role != "developer") {
      return invalid(param + ".role", "unsupported message role");
    }
    if (!item.contains("content")) {
      return invalid(param + ".content", "message content is required");
    }
    std::string id;
    if (std::optional<ResponsesError> error =
            read_id(item, param + ".id", &item_ids_, &id)) {
      return error;
    }
    const std::string_view part_type =
        role == "assistant" ? "output_text" : "input_text";
    std::string content;
    if (std::optional<ResponsesError> error = read_parts(item["content"],
                                                         param + ".content",
                                                         part_type,
                                                         limits_,
                                                         &content)) {
      return error;
    }
    if (content.empty()) {
      return invalid(param + ".content", "message content must not be empty");
    }
    if (role == "assistant") {
      if (assistant_state_ == AssistantState::MESSAGE) {
        return make_error(ErrorCode::INVALID_ITEM_ORDER,
                          param,
                          "assistant message is duplicated in its group");
      }
      prepared->canonical_input.emplace_back(
          MessageItem{.id = id, .role = role, .content = content});
      if (assistant_state_ == AssistantState::REASONING) {
        prepared->chat_request
            .mutable_messages(prepared->chat_request.messages_size() - 1)
            ->set_content(content);
      } else {
        proto::ChatMessage* message = prepared->chat_request.add_messages();
        message->set_role("assistant");
        message->set_content(content);
      }
      assistant_state_ = AssistantState::MESSAGE;
      return std::nullopt;
    }
    assistant_state_ = AssistantState::NONE;
    add_message(id, role == "developer" ? "system" : role, content, prepared);
    return std::nullopt;
  }

  std::optional<ResponsesError> parse_reasoning_item(
      const Json& item,
      const std::string& param,
      PreparedRequest* prepared) {
    if (std::optional<ResponsesError> error = reject_unknown(
            item,
            {"type", "id", "content", "summary", "encrypted_content"},
            param)) {
      return error;
    }
    if (assistant_state_ != AssistantState::NONE) {
      return make_error(ErrorCode::INVALID_ITEM_ORDER,
                        param,
                        "reasoning must begin an assistant group");
    }
    if (item.contains("summary") &&
        (!item["summary"].is_array() || !item["summary"].empty())) {
      return unsupported(param + ".summary",
                         "reasoning summary is unsupported");
    }
    if (item.contains("encrypted_content") &&
        !item["encrypted_content"].is_null()) {
      return unsupported(param + ".encrypted_content",
                         "encrypted reasoning is unsupported");
    }
    if (!item.contains("content")) {
      return invalid(param + ".content", "reasoning content is required");
    }
    std::string id;
    if (std::optional<ResponsesError> error =
            read_id(item, param + ".id", &item_ids_, &id)) {
      return error;
    }
    std::string content;
    if (std::optional<ResponsesError> error = read_parts(item["content"],
                                                         param + ".content",
                                                         "reasoning_text",
                                                         limits_,
                                                         &content)) {
      return error;
    }
    if (content.empty()) {
      return invalid(param + ".content", "reasoning content must not be empty");
    }
    prepared->canonical_input.emplace_back(
        ReasoningItem{.id = id, .content = content});
    proto::ChatMessage* message = prepared->chat_request.add_messages();
    message->set_role("assistant");
    message->set_reasoning_content(content);
    assistant_state_ = AssistantState::REASONING;
    return std::nullopt;
  }

  void add_message(const std::string& id,
                   const std::string& role,
                   const std::string& content,
                   PreparedRequest* prepared) const {
    prepared->canonical_input.emplace_back(
        MessageItem{.id = id, .role = role, .content = content});
    proto::ChatMessage* message = prepared->chat_request.add_messages();
    message->set_role(role);
    message->set_content(content);
  }

  const model_protocol::ModelProtocolIdentity& profile_;
  const RequestContext& context_;
  const ResponsesLimits& limits_;
  std::unordered_set<std::string> item_ids_;
  AssistantState assistant_state_ = AssistantState::NONE;
};

}  // namespace

PrepareResult::PrepareResult(PreparedRequest value)
    : value_(std::move(value)) {}

PrepareResult::PrepareResult(ResponsesError error) : error_(std::move(error)) {}

PrepareResult prepare_request(
    const std::string& body,
    const model_protocol::ModelProtocolIdentity& profile,
    const RequestContext& context,
    const ResponsesLimits& limits) {
  if (body.size() > limits.max_body_bytes) {
    return PrepareResult(make_error(ErrorCode::REQUEST_TOO_LARGE,
                                    "",
                                    "request body exceeds configured limit"));
  }
  if (exceeds_depth(body, limits.max_json_depth)) {
    return PrepareResult(make_error(ErrorCode::MAX_DEPTH_EXCEEDED,
                                    "",
                                    "JSON nesting exceeds configured limit"));
  }
  Json parsed;
  try {
    parsed = Json::parse(body);
  } catch (const Json::exception&) {
    return PrepareResult(make_error(
        ErrorCode::INVALID_JSON, "", "request body is invalid JSON"));
  }
  if (!parsed.is_object()) {
    return PrepareResult(invalid("", "request body must be an object"));
  }
  return RequestAssembler(profile, context, limits).assemble(parsed);
}

}  // namespace xllm::responses
