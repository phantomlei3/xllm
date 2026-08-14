/* Copyright 2025-2026 The xLLM Authors.

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

#include "jinja_chat_template.h"

#include <glog/logging.h>
#include <unistd.h>

#include <optional>
#include <string>
#include <utility>

namespace xllm {

namespace {
const std::unordered_map<std::string, std::string> type_to_modality = {
    {"video_url", "video"},
    {"image_url", "image"},
    {"audio_url", "audio"},
    {"image_embedding", "image"},
    {"video_embedding", "video"},
    {"audio_embedding", "audio"}};

std::string glm_effort(model_protocol::ReasoningEffort effort) {
  switch (effort) {
    case model_protocol::ReasoningEffort::NONE:
      return "none";
    case model_protocol::ReasoningEffort::XHIGH:
    case model_protocol::ReasoningEffort::MAX:
      return "max";
    case model_protocol::ReasoningEffort::MINIMAL:
    case model_protocol::ReasoningEffort::LOW:
    case model_protocol::ReasoningEffort::MEDIUM:
    case model_protocol::ReasoningEffort::HIGH:
      return "high";
  }
  return "high";
}

bool selected_tool(const Tool& tool, const model_protocol::ToolChoice& choice) {
  if (choice.kind == model_protocol::ToolChoiceKind::NONE) {
    return false;
  }
  if (choice.kind != model_protocol::ToolChoiceKind::FUNCTION &&
      choice.kind != model_protocol::ToolChoiceKind::CUSTOM) {
    return true;
  }
  if (std::holds_alternative<FunctionTool>(tool)) {
    return choice.kind == model_protocol::ToolChoiceKind::FUNCTION &&
           std::get<FunctionTool>(tool).name == choice.name;
  }
  return choice.kind == model_protocol::ToolChoiceKind::CUSTOM &&
         std::get<CustomTool>(tool).name == choice.name;
}

nlohmann::ordered_json protocol_tools_json(
    const std::vector<Tool>& tools,
    const model_protocol::ToolChoice& choice) {
  nlohmann::ordered_json result = nlohmann::json::array();
  for (const Tool& tool : tools) {
    if (!selected_tool(tool, choice)) {
      continue;
    }
    nlohmann::ordered_json function;
    if (std::holds_alternative<FunctionTool>(tool)) {
      const FunctionTool& value = std::get<FunctionTool>(tool);
      function["description"] = value.description;
      function["name"] = value.name;
      function["parameters"] = value.parameters;
    } else {
      const CustomTool& value = std::get<CustomTool>(tool);
      function["description"] =
          "Apply an exact patch. Every added content line must start with a "
          "literal plus sign (+), and the patch must end with a newline.";
      function["name"] = value.name;
      function["parameters"] = {
          {"properties",
           {{"patch",
             {{"description",
               "Complete patch text, preserving every literal character "
               "including leading + and final newline."},
              {"type", "string"}}}}},
          {"required", {"patch"}},
          {"type", "object"}};
    }
    result.emplace_back(nlohmann::ordered_json{
        {"type", "function"}, {"function", std::move(function)}});
  }
  return result;
}

nlohmann::ordered_json protocol_calls_json(
    const Message::ProtocolToolCallVec& calls) {
  nlohmann::ordered_json result = nlohmann::json::array();
  for (const ToolCall& tool_call : calls) {
    nlohmann::ordered_json function;
    std::string id;
    if (std::holds_alternative<FunctionCall>(tool_call)) {
      const FunctionCall& call = std::get<FunctionCall>(tool_call);
      id = call.id;
      function["name"] = call.name;
      function["arguments"] = nlohmann::ordered_json::parse(call.arguments);
    } else {
      const CustomToolCall& call = std::get<CustomToolCall>(tool_call);
      id = call.id;
      function["name"] = call.name;
      std::string input = call.input;
      // The frozen GLM replay prompt omits the patch terminator newline.
      if (input.ends_with('\n')) {
        input.pop_back();
      }
      function["arguments"] = {{"patch", std::move(input)}};
    }
    result.emplace_back(
        nlohmann::ordered_json{{"id", std::move(id)},
                               {"type", "function"},
                               {"function", std::move(function)}});
  }
  return result;
}
}  // namespace

JinjaChatTemplate::JinjaChatTemplate(const TokenizerArgs& args) : args_(args) {
  try {
    template_ = std::make_unique<minja::chat_template>(
        args_.chat_template(), args_.bos_token(), args_.eos_token());
    LOG(INFO) << "Jinja chat template init succeed.";

  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to parse jinja chat template, TokenizerArgs: "
               << args_ << std::endl
               << "Error message: " << e.what();
  }
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages) const {
  const std::vector<xllm::JsonTool> empty_tools;
  const nlohmann::ordered_json chat_template_kwargs;
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  const std::vector<xllm::JsonTool> empty_tools;
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages) const {
  // Call the overloaded method with empty tools
  nlohmann::ordered_json empty_tools = nlohmann::json::array();
  const nlohmann::ordered_json chat_template_kwargs = nlohmann::json::object();
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  // convert the messages to json object
  nlohmann::ordered_json messages_json = nlohmann::json::array();
  for (const auto& message : messages) {
    nlohmann::ordered_json message_json;
    message_json["role"] = message.role;

    if (std::holds_alternative<std::string>(message.content)) {
      message_json["content"] = std::get<std::string>(message.content);
    } else if (std::holds_alternative<MMContentVec>(message.content)) {
      message_json["content"] =
          get_mm_content(std::get<MMContentVec>(message.content));
    }

    if (message.tool_call_id.has_value()) {
      message_json["tool_call_id"] = *message.tool_call_id;
    }

    if (message.reasoning_content.has_value()) {
      message_json["reasoning_content"] = *message.reasoning_content;
    }

    if (message.tool_calls.has_value()) {
      nlohmann::ordered_json tool_calls_json = nlohmann::json::array();
      const auto& tool_calls = *message.tool_calls;

      for (const auto& tool_call : tool_calls) {
        tool_calls_json.emplace_back(nlohmann::ordered_json{
            {"id", tool_call.id},
            {"type", tool_call.type},
            {"function",
             nlohmann::ordered_json{
                 {"name", tool_call.function.name},
                 {"arguments", tool_call.function.arguments}}}});
      }
      message_json["tool_calls"] = std::move(tool_calls_json);
    }

    messages_json.emplace_back(std::move(message_json));
  }

  nlohmann::ordered_json tools_json = nlohmann::json::array();

  for (const auto& json_tool : json_tools) {
    tools_json.emplace_back(nlohmann::ordered_json{
        {"type", json_tool.type},
        {"function",
         nlohmann::ordered_json{
             {"name", json_tool.function.name},
             {"description", json_tool.function.description},
             {"parameters", json_tool.function.parameters}}}});
  }
  // apply the template
  return apply(messages_json, tools_json, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const std::vector<xllm::JsonTool>& json_tools,
    const std::vector<xllm::Tool>& protocol_tools,
    const model_protocol::TemplatePolicy& template_policy,
    model_protocol::ReasoningEffort reasoning_effort,
    const model_protocol::ToolChoice& tool_choice,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  if (template_policy.thinking_history ==
      model_protocol::ThinkingHistoryPolicy::TEMPLATE_DEFAULT) {
    return apply(messages, json_tools, chat_template_kwargs);
  }

  try {
    nlohmann::ordered_json messages_json = nlohmann::json::array();
    for (const Message& message : messages) {
      nlohmann::ordered_json message_json;
      message_json["role"] = message.role;
      if (std::holds_alternative<std::string>(message.content)) {
        message_json["content"] = std::get<std::string>(message.content);
      } else {
        message_json["content"] =
            get_mm_content(std::get<MMContentVec>(message.content));
      }
      if (message.tool_call_id.has_value()) {
        message_json["tool_call_id"] = *message.tool_call_id;
      }
      if (message.reasoning_content.has_value()) {
        message_json["reasoning_content"] = *message.reasoning_content;
      }
      if (message.protocol_tool_calls.has_value()) {
        message_json["tool_calls"] =
            protocol_calls_json(*message.protocol_tool_calls);
      } else if (message.tool_calls.has_value()) {
        nlohmann::ordered_json calls = nlohmann::json::array();
        for (const Message::ToolCall& call : *message.tool_calls) {
          calls.emplace_back(nlohmann::ordered_json{
              {"id", call.id},
              {"type", call.type},
              {"function",
               {{"name", call.function.name},
                {"arguments", call.function.arguments}}}});
        }
        message_json["tool_calls"] = std::move(calls);
      }
      messages_json.emplace_back(std::move(message_json));
    }

    nlohmann::ordered_json tools_json;
    if (!protocol_tools.empty()) {
      tools_json = protocol_tools_json(protocol_tools, tool_choice);
    } else {
      tools_json = nlohmann::json::array();
      if (tool_choice.kind != model_protocol::ToolChoiceKind::NONE) {
        for (const JsonTool& tool : json_tools) {
          tools_json.emplace_back(nlohmann::ordered_json{
              {"type", tool.type},
              {"function",
               {{"name", tool.function.name},
                {"description", tool.function.description},
                {"parameters", tool.function.parameters}}}});
        }
      }
    }

    nlohmann::ordered_json controlled_kwargs = chat_template_kwargs;
    if (template_policy.thinking_history ==
        model_protocol::ThinkingHistoryPolicy::PRESERVE) {
      controlled_kwargs["clear_thinking"] = false;
    } else if (template_policy.clear_thinking.has_value()) {
      controlled_kwargs["clear_thinking"] = *template_policy.clear_thinking;
    }
    controlled_kwargs["enable_thinking"] =
        reasoning_effort != model_protocol::ReasoningEffort::NONE;
    controlled_kwargs["reasoning_effort"] = glm_effort(reasoning_effort);
    return apply(messages_json, tools_json, controlled_kwargs);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to normalize protocol chat template input: "
               << e.what();
    return std::nullopt;
  }
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages,
    const nlohmann::ordered_json& tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  try {
    minja::chat_template_inputs input;
    input.messages = messages;
    input.tools = tools;
    input.add_generation_prompt = true;
    input.extra_context = chat_template_kwargs;
    minja::chat_template_options options;

    return template_->apply(input, options);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to apply chat template: " << e.what();
    return std::nullopt;
  }
}

nlohmann::ordered_json JinjaChatTemplate::get_mm_content(
    const MMContentVec& vec) const {
  nlohmann::ordered_json content_json = nlohmann::json::array();

  for (const auto& item : vec) {
    nlohmann::ordered_json item_json;
    item_json["type"] = item.type;
    if (item.type == "text") {
      item_json["text"] = item.text;
    } else if (auto it = type_to_modality.find(item.type);
               it != type_to_modality.end()) {
      const std::string& modality = it->second;
      item_json[modality] = "mm place holder";
      item_json[item.type] = "mm place holder";
    } else {
      item_json[item.type] = "mm place holder";
    }

    content_json.emplace_back(item_json);
  }

  return content_json;
}

}  // namespace xllm
