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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "nlohmann/json.hpp"

namespace xllm::responses {
namespace {

using Json = nlohmann::json;

constexpr size_t kMaxIdBytes = 256;
constexpr std::string_view kApplyPatchGrammar =
    R"grammar(start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF
)grammar";

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

bool valid_number(const Json& value, float min, float max) {
  if (!value.is_number()) {
    return false;
  }
  const double number = value.get<double>();
  return std::isfinite(number) && number >= min && number <= max;
}

void json_to_value(const Json& json, google::protobuf::Value* value) {
  if (json.is_null()) {
    value->set_null_value(google::protobuf::NULL_VALUE);
  } else if (json.is_boolean()) {
    value->set_bool_value(json.get<bool>());
  } else if (json.is_number()) {
    value->set_number_value(json.get<double>());
  } else if (json.is_string()) {
    value->set_string_value(json.get<std::string>());
  } else if (json.is_array()) {
    for (const Json& item : json) {
      json_to_value(item, value->mutable_list_value()->add_values());
    }
  } else {
    for (auto it = json.begin(); it != json.end(); ++it) {
      json_to_value(
          it.value(),
          &(*value->mutable_struct_value()->mutable_fields())[it.key()]);
    }
  }
}

void json_to_struct(const Json& json, google::protobuf::Struct* value) {
  for (auto it = json.begin(); it != json.end(); ++it) {
    json_to_value(it.value(), &(*value->mutable_fields())[it.key()]);
  }
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
                   const std::string& model_id,
                   const RequestContext& context,
                   const ResponsesLimits& limits)
      : profile_(profile),
        model_id_(model_id),
        context_(context),
        limits_(limits) {}

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
                            "reasoning",
                            "max_output_tokens",
                            "temperature",
                            "top_p",
                            "tools",
                            "tool_choice",
                            "include",
                            "prompt_cache_key",
                            "client_metadata",
                            "text"},
                           "")) {
      return PrepareResult(std::move(*error));
    }
    if (!body.contains("model") || !body["model"].is_string() ||
        body["model"].get_ref<const std::string&>().empty()) {
      return PrepareResult(invalid("model", "model is required"));
    }
    const std::string model = body["model"].get<std::string>();
    if (!model_matches(model)) {
      return PrepareResult(
          make_error(ErrorCode::MODEL_MISMATCH,
                     "model",
                     "model does not match deployment binding"));
    }
    if (!body.contains("input") && !body.contains("instructions")) {
      return PrepareResult(
          invalid("input", "input or instructions is required"));
    }

    PreparedRequest prepared;
    prepared.profile_id = profile_.profile_id;
    prepared.model_id = model_id_;
    prepared.context = context_;
    init_chat(&prepared);

    if (std::optional<ResponsesError> error = parse_no_effect(body)) {
      return PrepareResult(std::move(*error));
    }
    if (std::optional<ResponsesError> error = parse_options(body, &prepared)) {
      return PrepareResult(std::move(*error));
    }
    if (std::optional<ResponsesError> error = parse_tools(body, &prepared)) {
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
    CALLS = 2,
    MESSAGE = 3,
  };

  enum class CallKind : uint8_t {
    FUNCTION = 0,
    CUSTOM = 1,
  };

  struct CallState {
    CallKind kind = CallKind::FUNCTION;
    bool completed = false;
  };

  bool model_matches(const std::string& model) const {
    return model == model_id_;
  }

  void init_chat(PreparedRequest* prepared) const {
    prepared->chat_request.set_model(model_id_);
    prepared->chat_request.set_n(1);
    prepared->chat_request.set_thinking_history_policy(proto::PRESERVE);
    prepared->chat_request.set_output_decoding_policy(proto::PROTOCOL_RAW);
    prepared->chat_request.set_request_id(context_.request_id);
    prepared->chat_request.set_parallel_tool_calls(true);
  }

  std::optional<ResponsesError> parse_no_effect(const Json& body) const {
    if (body.contains("include")) {
      const Json& include = body["include"];
      if (!include.is_array() || include.size() != 1 ||
          !include[0].is_string() ||
          include[0] != "reasoning.encrypted_content") {
        return unsupported("include", "unsupported response include field");
      }
    }
    if (body.contains("prompt_cache_key")) {
      std::string cache_key;
      if (std::optional<ResponsesError> error =
              read_text(body["prompt_cache_key"],
                        "prompt_cache_key",
                        limits_,
                        &cache_key)) {
        return error;
      }
      if (cache_key.empty()) {
        return unsupported("prompt_cache_key",
                           "prompt cache key must not be empty");
      }
    }
    if (body.contains("client_metadata")) {
      if (std::optional<ResponsesError> error =
              parse_client_metadata(body["client_metadata"])) {
        return error;
      }
    }
    if (body.contains("text")) {
      const Json& text = body["text"];
      if (!text.is_object()) {
        return invalid("text", "text options must be an object");
      }
      if (std::optional<ResponsesError> error =
              reject_unknown(text, {"verbosity"}, "text")) {
        return error;
      }
      if (!text.contains("verbosity") || !text["verbosity"].is_string() ||
          text["verbosity"] != "low") {
        return unsupported("text.verbosity",
                           "only captured low verbosity is accepted");
      }
    }
    return std::nullopt;
  }

  std::optional<ResponsesError> parse_client_metadata(
      const Json& metadata) const {
    if (!metadata.is_object()) {
      return invalid("client_metadata", "client metadata must be an object");
    }
    if (metadata.dump().size() > limits_.max_client_metadata_bytes) {
      return make_error(ErrorCode::REQUEST_TOO_LARGE,
                        "client_metadata",
                        "client metadata exceeds configured limit");
    }
    return std::nullopt;
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
      if (std::optional<ResponsesError> error =
              parse_reasoning(body["reasoning"], &prepared->options)) {
        return error;
      }
    }
    if (body.contains("temperature") &&
        !valid_number(body["temperature"], 0.0f, 2.0f)) {
      return invalid("temperature", "temperature must be between 0 and 2");
    }
    if (body.contains("top_p") && !valid_number(body["top_p"], 0.0f, 1.0f)) {
      return invalid("top_p", "top_p must be between 0 and 1");
    }
    const bool thinking = prepared->options.reasoning_effort !=
                          model_protocol::ReasoningEffort::NONE;
    prepared->options.temperature = thinking || !body.contains("temperature")
                                        ? 1.0f
                                        : body["temperature"].get<float>();
    prepared->options.top_p = thinking || !body.contains("top_p")
                                  ? 0.95f
                                  : body["top_p"].get<float>();
    prepared->chat_request.set_temperature(prepared->options.temperature);
    prepared->chat_request.set_top_p(prepared->options.top_p);
    prepared->chat_request.set_reasoning_effort(
        static_cast<proto::ReasoningEffort>(
            prepared->options.reasoning_effort));
    if (body.contains("max_output_tokens")) {
      if (!body["max_output_tokens"].is_number_unsigned() ||
          body["max_output_tokens"].get<uint64_t>() == 0 ||
          body["max_output_tokens"].get<uint64_t>() >
              std::numeric_limits<uint32_t>::max()) {
        return invalid("max_output_tokens",
                       "max_output_tokens must be a positive integer");
      }
      prepared->options.max_output_tokens =
          body["max_output_tokens"].get<uint32_t>();
      prepared->chat_request.set_max_tokens(
          *prepared->options.max_output_tokens);
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
            reject_unknown(reasoning, {"effort", "summary"}, "reasoning")) {
      return error;
    }
    if (reasoning.contains("summary") &&
        (!reasoning["summary"].is_string() || reasoning["summary"] != "auto")) {
      return unsupported("reasoning.summary",
                         "only captured automatic summary is accepted");
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

  std::optional<ResponsesError> parse_tools(const Json& body,
                                            PreparedRequest* prepared) {
    if (body.contains("tools")) {
      const Json& tools = body["tools"];
      if (!tools.is_array()) {
        return invalid("tools", "tools must be an array");
      }
      if (tools.size() > limits_.max_tools) {
        return make_error(ErrorCode::TOO_MANY_ITEMS, "tools", "too many tools");
      }
      for (size_t index = 0; index < tools.size(); ++index) {
        if (std::optional<ResponsesError> error =
                parse_tool(tools[index], index, prepared)) {
          return error;
        }
      }
    }
    return parse_tool_choice(body, prepared);
  }

  std::optional<ResponsesError> parse_tool(const Json& tool,
                                           size_t index,
                                           PreparedRequest* prepared) {
    const std::string param = "tools[" + std::to_string(index) + "]";
    if (!tool.is_object() || !tool.contains("type") ||
        !tool["type"].is_string()) {
      return invalid(param + ".type", "tool type is required");
    }
    const uint64_t tool_bytes = tool.dump().size();
    if (tool_bytes > limits_.max_tool_bytes - tool_bytes_) {
      return make_error(ErrorCode::REQUEST_TOO_LARGE,
                        "tools",
                        "tool definitions exceed configured limit");
    }
    tool_bytes_ += tool_bytes;
    const std::string type = tool["type"].get<std::string>();
    if (type == "function") {
      return parse_function_tool(tool, param, prepared);
    }
    if (type == "custom") {
      return parse_custom_tool(tool, param, prepared);
    }
    return make_error(
        ErrorCode::UNKNOWN_TOOL, param + ".type", "unsupported tool type");
  }

  std::optional<ResponsesError> parse_function_tool(const Json& tool,
                                                    const std::string& param,
                                                    PreparedRequest* prepared) {
    if (std::optional<ResponsesError> error = reject_unknown(
            tool,
            {"type", "name", "description", "parameters", "strict"},
            param)) {
      return error;
    }
    if (!tool.contains("name") || !tool["name"].is_string() ||
        tool["name"].get_ref<const std::string&>().empty()) {
      return invalid(param + ".name", "function name is required");
    }
    const std::string name = tool["name"].get<std::string>();
    if (!tool_names_.emplace("function:" + name).second) {
      return invalid(param + ".name", "tool is duplicated");
    }
    if (tool.contains("description") && !tool["description"].is_string()) {
      return invalid(param + ".description", "description must be a string");
    }
    if (tool.contains("strict")) {
      if (!tool["strict"].is_boolean()) {
        return invalid(param + ".strict", "strict must be a boolean");
      }
      if (tool["strict"].get<bool>()) {
        return unsupported(param + ".strict", "strict tools are unsupported");
      }
    }
    if (!tool.contains("parameters") || !tool["parameters"].is_object()) {
      return invalid(param + ".parameters",
                     "function parameters must be an object");
    }
    const std::string schema = tool["parameters"].dump();
    if (schema.size() > limits_.max_function_schema_bytes) {
      return make_error(ErrorCode::REQUEST_TOO_LARGE,
                        param + ".parameters",
                        "function schema exceeds configured limit");
    }
    if (exceeds_depth(schema, limits_.max_json_depth)) {
      return make_error(ErrorCode::MAX_DEPTH_EXCEEDED,
                        param + ".parameters",
                        "function schema nesting exceeds configured limit");
    }

    FunctionTool function{.name = name,
                          .description = tool.value("description", ""),
                          .parameters = tool["parameters"]};
    proto::Tool* proto_tool = prepared->chat_request.add_tools();
    proto_tool->set_type("function");
    proto_tool->mutable_function()->set_name(function.name);
    proto_tool->mutable_function()->set_description(function.description);
    json_to_struct(function.parameters,
                   proto_tool->mutable_function()->mutable_parameters());
    tools_.emplace_back(std::move(function));
    return std::nullopt;
  }

  std::optional<ResponsesError> parse_custom_tool(const Json& tool,
                                                  const std::string& param,
                                                  PreparedRequest* prepared) {
    if (std::optional<ResponsesError> error = reject_unknown(
            tool, {"type", "name", "description", "format"}, param)) {
      return error;
    }
    if (!tool.contains("name") || !tool["name"].is_string()) {
      return invalid(param + ".name", "custom tool name is required");
    }
    const std::string name = tool["name"].get<std::string>();
    if (name != "apply_patch") {
      return make_error(
          ErrorCode::UNKNOWN_TOOL, param + ".name", "unknown custom tool");
    }
    if (tool.contains("description")) {
      std::string description;
      if (std::optional<ResponsesError> error =
              read_text(tool["description"],
                        param + ".description",
                        limits_,
                        &description)) {
        return error;
      }
    }
    if (tool.contains("format")) {
      const Json& format = tool["format"];
      if (!format.is_object()) {
        return invalid(param + ".format",
                       "custom tool format must be an object");
      }
      if (std::optional<ResponsesError> error = reject_unknown(
              format, {"type", "syntax", "definition"}, param + ".format")) {
        return error;
      }
      if (!format.contains("type") || !format["type"].is_string() ||
          format["type"] != "grammar") {
        return unsupported(param + ".format.type",
                           "only the captured grammar format is accepted");
      }
      if (!format.contains("syntax") || !format["syntax"].is_string() ||
          format["syntax"] != "lark") {
        return unsupported(param + ".format.syntax",
                           "only the captured Lark syntax is accepted");
      }
      if (!format.contains("definition") || !format["definition"].is_string() ||
          format["definition"].get_ref<const std::string&>() !=
              kApplyPatchGrammar) {
        return unsupported(param + ".format.definition",
                           "only the captured apply_patch grammar is accepted");
      }
    }
    if (!tool_names_.emplace("custom:" + name).second) {
      return invalid(param + ".name", "tool is duplicated");
    }
    proto::Tool* proto_tool = prepared->chat_request.add_tools();
    proto_tool->set_type("custom");
    proto_tool->mutable_custom()->set_name(name);
    tools_.emplace_back(CustomTool{.name = name});
    return std::nullopt;
  }

  std::optional<ResponsesError> parse_tool_choice(const Json& body,
                                                  PreparedRequest* prepared) {
    const Json choice = body.value("tool_choice", Json("auto"));
    if (choice.is_string()) {
      const std::string value = choice.get<std::string>();
      if (value != "none" && value != "auto" && value != "required") {
        return make_error(
            ErrorCode::UNKNOWN_TOOL, "tool_choice", "unknown tool choice");
      }
      if (value == "required" && tools_.empty()) {
        return invalid("tool_choice", "required needs at least one tool");
      }
      if (value == "none") {
        prepared->options.tool_choice.kind =
            model_protocol::ToolChoiceKind::NONE;
        prepared->chat_request.set_protocol_tool_choice(
            proto::TOOL_CHOICE_NONE);
      } else if (value == "required") {
        prepared->options.tool_choice.kind =
            model_protocol::ToolChoiceKind::REQUIRED;
        prepared->chat_request.set_protocol_tool_choice(
            proto::TOOL_CHOICE_REQUIRED);
      } else {
        prepared->chat_request.set_protocol_tool_choice(
            proto::TOOL_CHOICE_AUTO);
      }
      prepared->chat_request.set_tool_choice(value);
      return std::nullopt;
    }
    if (!choice.is_object()) {
      return invalid("tool_choice", "tool_choice must be a string or object");
    }
    if (std::optional<ResponsesError> error =
            reject_unknown(choice, {"type", "name"}, "tool_choice")) {
      return error;
    }
    if (!choice.contains("type") || !choice["type"].is_string() ||
        !choice.contains("name") || !choice["name"].is_string()) {
      return invalid("tool_choice", "named tool choice needs type and name");
    }
    const std::string type = choice["type"].get<std::string>();
    const std::string name = choice["name"].get<std::string>();
    if (type != "function" && type != "custom") {
      return make_error(
          ErrorCode::UNKNOWN_TOOL, "tool_choice.type", "unknown tool type");
    }
    if (type == "custom" && name != "apply_patch") {
      return make_error(
          ErrorCode::UNKNOWN_TOOL, "tool_choice.name", "unknown custom tool");
    }
    if (tool_names_.count(type + ":" + name) == 0) {
      return make_error(
          ErrorCode::UNKNOWN_TOOL, "tool_choice.name", "tool is not declared");
    }
    prepared->options.tool_choice = {
        .kind = type == "function" ? model_protocol::ToolChoiceKind::FUNCTION
                                   : model_protocol::ToolChoiceKind::CUSTOM,
        .name = name};
    prepared->chat_request.set_protocol_tool_choice(
        type == "function" ? proto::TOOL_CHOICE_FUNCTION
                           : proto::TOOL_CHOICE_CUSTOM);
    prepared->chat_request.set_protocol_tool_name(name);
    prepared->chat_request.set_tool_choice(choice.dump());
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
    if (type == "function_call") {
      return parse_function_call(item, param, prepared);
    }
    if (type == "custom_tool_call") {
      return parse_custom_call(item, param, prepared);
    }
    if (type == "function_call_output") {
      return parse_call_output(item, param, CallKind::FUNCTION, prepared);
    }
    if (type == "custom_tool_call_output") {
      return parse_call_output(item, param, CallKind::CUSTOM, prepared);
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
      if (assistant_state_ == AssistantState::REASONING ||
          assistant_state_ == AssistantState::CALLS) {
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
    if (!pending_calls_.empty()) {
      return make_error(ErrorCode::INVALID_ITEM_ORDER,
                        param,
                        "pending tool calls must be completed first");
    }
    assistant_state_ = AssistantState::NONE;
    add_message(id, role == "developer" ? "system" : role, content, prepared);
    return std::nullopt;
  }

  std::optional<ResponsesError> parse_function_call(const Json& item,
                                                    const std::string& param,
                                                    PreparedRequest* prepared) {
    if (std::optional<ResponsesError> error = reject_unknown(
            item,
            {"type", "id", "call_id", "name", "arguments", "status"},
            param)) {
      return error;
    }
    if (assistant_state_ == AssistantState::MESSAGE) {
      return make_error(ErrorCode::INVALID_ITEM_ORDER,
                        param,
                        "tool calls must precede assistant text");
    }
    if (std::optional<ResponsesError> error =
            validate_call_status(item, param)) {
      return error;
    }
    std::string item_id;
    if (std::optional<ResponsesError> error =
            read_id(item, param + ".id", &item_ids_, &item_id)) {
      return error;
    }
    std::string call_id;
    std::string name;
    std::string arguments;
    if (std::optional<ResponsesError> error =
            read_call_fields(item,
                             param,
                             limits_.max_function_args_bytes,
                             &call_id,
                             &name,
                             &arguments)) {
      return error;
    }
    Json parsed_arguments;
    try {
      parsed_arguments = Json::parse(arguments);
    } catch (const Json::exception&) {
      return make_error(ErrorCode::INVALID_TOOL_ARGUMENTS,
                        param + ".arguments",
                        "function arguments must be a JSON object string");
    }
    if (!parsed_arguments.is_object()) {
      return make_error(ErrorCode::INVALID_TOOL_ARGUMENTS,
                        param + ".arguments",
                        "function arguments must be a JSON object string");
    }
    if (exceeds_depth(arguments, limits_.max_json_depth)) {
      return make_error(ErrorCode::MAX_DEPTH_EXCEEDED,
                        param + ".arguments",
                        "function arguments nesting exceeds configured limit");
    }
    if (std::optional<ResponsesError> error =
            add_call(call_id, CallKind::FUNCTION, param + ".call_id")) {
      return error;
    }
    FunctionCall call{.id = call_id, .name = name, .arguments = arguments};
    prepared->canonical_input.emplace_back(
        FunctionCallItem{.id = item_id, .call = call});
    proto::ToolCall* proto_call = assistant_message(prepared)->add_tool_calls();
    proto_call->set_id(call_id);
    proto_call->set_type("function");
    proto_call->mutable_function()->set_name(name);
    proto_call->mutable_function()->set_arguments(arguments);
    assistant_state_ = AssistantState::CALLS;
    return std::nullopt;
  }

  std::optional<ResponsesError> parse_custom_call(const Json& item,
                                                  const std::string& param,
                                                  PreparedRequest* prepared) {
    if (std::optional<ResponsesError> error =
            reject_unknown(item,
                           {"type", "id", "call_id", "name", "input", "status"},
                           param)) {
      return error;
    }
    if (assistant_state_ == AssistantState::MESSAGE) {
      return make_error(ErrorCode::INVALID_ITEM_ORDER,
                        param,
                        "tool calls must precede assistant text");
    }
    if (std::optional<ResponsesError> error =
            validate_call_status(item, param)) {
      return error;
    }
    std::string item_id;
    if (std::optional<ResponsesError> error =
            read_id(item, param + ".id", &item_ids_, &item_id)) {
      return error;
    }
    if (!item.contains("name") || !item["name"].is_string() ||
        item["name"] != "apply_patch") {
      return make_error(
          ErrorCode::UNKNOWN_TOOL, param + ".name", "unknown custom tool");
    }
    std::string call_id;
    if (std::optional<ResponsesError> error =
            required_id(item, param + ".call_id", "call_id", &call_id)) {
      return error;
    }
    if (!item.contains("input")) {
      return invalid(param + ".input", "custom tool input is required");
    }
    std::string input;
    if (std::optional<ResponsesError> error =
            read_payload(item["input"],
                         param + ".input",
                         limits_.max_custom_payload_bytes,
                         &input)) {
      return error;
    }
    if (std::optional<ResponsesError> error =
            add_call(call_id, CallKind::CUSTOM, param + ".call_id")) {
      return error;
    }
    CustomToolCall call{.id = call_id, .name = "apply_patch", .input = input};
    prepared->canonical_input.emplace_back(
        CustomToolCallItem{.id = item_id, .call = call});
    proto::ToolCall* proto_call = assistant_message(prepared)->add_tool_calls();
    proto_call->set_id(call_id);
    proto_call->set_type("custom");
    proto_call->mutable_custom()->set_name("apply_patch");
    proto_call->mutable_custom()->set_input(input);
    assistant_state_ = AssistantState::CALLS;
    return std::nullopt;
  }

  std::optional<ResponsesError> parse_call_output(const Json& item,
                                                  const std::string& param,
                                                  CallKind kind,
                                                  PreparedRequest* prepared) {
    if (std::optional<ResponsesError> error =
            reject_unknown(item, {"type", "id", "call_id", "output"}, param)) {
      return error;
    }
    std::string item_id;
    if (std::optional<ResponsesError> error =
            read_id(item, param + ".id", &item_ids_, &item_id)) {
      return error;
    }
    std::string call_id;
    if (std::optional<ResponsesError> error =
            required_id(item, param + ".call_id", "call_id", &call_id)) {
      return error;
    }
    auto call = calls_.find(call_id);
    if (call == calls_.end()) {
      return make_error(ErrorCode::UNKNOWN_CALL_ID,
                        param + ".call_id",
                        "output references an unknown call ID");
    }
    if (call->second.kind != kind) {
      return make_error(ErrorCode::TOOL_CALL_TYPE_MISMATCH,
                        param + ".call_id",
                        "output type does not match its call");
    }
    if (call->second.completed) {
      return make_error(ErrorCode::INVALID_ITEM_ORDER,
                        param + ".call_id",
                        "call output is duplicated");
    }
    if (!item.contains("output")) {
      return invalid(param + ".output", "tool output is required");
    }
    std::string output;
    const uint64_t limit = kind == CallKind::FUNCTION
                               ? limits_.max_tool_bytes
                               : limits_.max_custom_payload_bytes;
    if (std::optional<ResponsesError> error =
            read_payload(item["output"], param + ".output", limit, &output)) {
      return error;
    }
    call->second.completed = true;
    pending_calls_.erase(call_id);
    if (kind == CallKind::FUNCTION) {
      prepared->canonical_input.emplace_back(FunctionCallOutput{
          .id = item_id, .call_id = call_id, .output = output});
    } else {
      prepared->canonical_input.emplace_back(CustomToolCallOutput{
          .id = item_id, .call_id = call_id, .output = output});
    }
    proto::ChatMessage* message = prepared->chat_request.add_messages();
    message->set_role("tool");
    message->set_tool_call_id(call_id);
    message->set_content(output);
    message->set_tool_output_type(kind == CallKind::FUNCTION
                                      ? proto::FUNCTION_OUTPUT
                                      : proto::CUSTOM_OUTPUT);
    assistant_state_ = AssistantState::NONE;
    return std::nullopt;
  }

  std::optional<ResponsesError> required_id(const Json& item,
                                            const std::string& param,
                                            std::string_view field,
                                            std::string* id) const {
    if (!item.contains(field) || !item[field].is_string()) {
      return invalid(param, "call ID is required");
    }
    *id = item[field].get<std::string>();
    if (!valid_id(*id)) {
      return invalid(param, "call ID is invalid");
    }
    return std::nullopt;
  }

  std::optional<ResponsesError> validate_call_status(
      const Json& item,
      const std::string& param) const {
    if (!item.contains("status")) {
      return std::nullopt;
    }
    if (!item["status"].is_string() || item["status"] != "completed") {
      return invalid(param + ".status", "historical call must be completed");
    }
    return std::nullopt;
  }

  std::optional<ResponsesError> read_call_fields(const Json& item,
                                                 const std::string& param,
                                                 uint64_t max_payload_bytes,
                                                 std::string* call_id,
                                                 std::string* name,
                                                 std::string* payload) const {
    if (std::optional<ResponsesError> error =
            required_id(item, param + ".call_id", "call_id", call_id)) {
      return error;
    }
    if (!item.contains("name") || !item["name"].is_string() ||
        item["name"].get_ref<const std::string&>().empty()) {
      return invalid(param + ".name", "function name is required");
    }
    *name = item["name"].get<std::string>();
    if (!item.contains("arguments")) {
      return invalid(param + ".arguments", "function arguments are required");
    }
    return read_payload(
        item["arguments"], param + ".arguments", max_payload_bytes, payload);
  }

  std::optional<ResponsesError> read_payload(const Json& value,
                                             const std::string& param,
                                             uint64_t max_bytes,
                                             std::string* payload) const {
    if (!value.is_string()) {
      return invalid(param, "payload must be a string");
    }
    *payload = value.get<std::string>();
    if (payload->size() > max_bytes) {
      return make_error(ErrorCode::REQUEST_TOO_LARGE,
                        param,
                        "payload exceeds configured limit");
    }
    return std::nullopt;
  }

  std::optional<ResponsesError> add_call(const std::string& call_id,
                                         CallKind kind,
                                         const std::string& param) {
    if (!calls_.emplace(call_id, CallState{.kind = kind}).second) {
      return make_error(
          ErrorCode::DUPLICATE_CALL_ID, param, "call ID is duplicated");
    }
    pending_calls_.emplace(call_id);
    return std::nullopt;
  }

  proto::ChatMessage* assistant_message(PreparedRequest* prepared) const {
    if (assistant_state_ == AssistantState::NONE) {
      proto::ChatMessage* message = prepared->chat_request.add_messages();
      message->set_role("assistant");
      return message;
    }
    return prepared->chat_request.mutable_messages(
        prepared->chat_request.messages_size() - 1);
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
  const std::string& model_id_;
  const RequestContext& context_;
  const ResponsesLimits& limits_;
  std::unordered_set<std::string> item_ids_;
  std::unordered_set<std::string> tool_names_;
  std::unordered_map<std::string, CallState> calls_;
  std::unordered_set<std::string> pending_calls_;
  std::vector<Tool> tools_;
  uint64_t tool_bytes_ = 0;
  AssistantState assistant_state_ = AssistantState::NONE;
};

}  // namespace

ModelFieldResult::ModelFieldResult(std::string model)
    : model_(std::move(model)) {}

ModelFieldResult::ModelFieldResult(ResponsesError error)
    : error_(std::move(error)) {}

ModelFieldResult read_model_field(const std::string& body) {
  Json json;
  try {
    json = Json::parse(body);
  } catch (const Json::parse_error&) {
    return ModelFieldResult(make_error(
        ErrorCode::INVALID_JSON, "body", "request body is not valid JSON"));
  }
  if (!json.is_object()) {
    return ModelFieldResult(invalid("body", "request body must be an object"));
  }
  if (!json.contains("model") || !json["model"].is_string() ||
      json["model"].get_ref<const std::string&>().empty()) {
    return ModelFieldResult(
        invalid("model", "model must be a non-empty string"));
  }
  return ModelFieldResult(json["model"].get<std::string>());
}

PrepareResult::PrepareResult(PreparedRequest value)
    : value_(std::move(value)) {}

PrepareResult::PrepareResult(ResponsesError error) : error_(std::move(error)) {}

PrepareResult prepare_request(
    const std::string& body,
    const std::string& model_id,
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
  return RequestAssembler(profile, model_id, context, limits).assemble(parsed);
}

}  // namespace xllm::responses
