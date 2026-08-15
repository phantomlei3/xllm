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

#include "core/framework/request/request_params.h"

#include <google/protobuf/util/json_util.h>
#include <gtest/gtest.h>

#include "anthropic.pb.h"
#include "chat.pb.h"
#include "completion.pb.h"
#include "multimodal.pb.h"

namespace xllm {
namespace {

TEST(RequestParamsTest, IncludeStopStringInOutputDefaultsToFalse) {
  RequestParams completion_params(proto::CompletionRequest(), "", "");
  RequestParams chat_params(proto::ChatRequest(), "", "");
  RequestParams mm_chat_params(proto::MMChatRequest(), "", "");

  EXPECT_FALSE(completion_params.include_stop_str_in_output);
  EXPECT_FALSE(chat_params.include_stop_str_in_output);
  EXPECT_FALSE(mm_chat_params.include_stop_str_in_output);
}

TEST(RequestParamsTest, ChatProtocolPoliciesDefaultToLegacyBehavior) {
  RequestParams params(proto::ChatRequest(), "", "");

  EXPECT_EQ(params.thinking_history_policy,
            model_protocol::ThinkingHistoryPolicy::TEMPLATE_DEFAULT);
  EXPECT_EQ(params.output_decoding_policy,
            model_protocol::OutputDecodingPolicy::VISIBLE_TEXT);
}

TEST(RequestParamsTest, ChatProtocolPoliciesPropagateTypedValues) {
  proto::ChatRequest request;
  request.set_thinking_history_policy(proto::PRESERVE);
  request.set_output_decoding_policy(proto::PROTOCOL_RAW);

  RequestParams params(request, "", "");

  EXPECT_EQ(params.thinking_history_policy,
            model_protocol::ThinkingHistoryPolicy::PRESERVE);
  EXPECT_EQ(params.output_decoding_policy,
            model_protocol::OutputDecodingPolicy::PROTOCOL_RAW);
  EXPECT_FALSE(params.skip_special_tokens);
  EXPECT_TRUE(params.include_stop_str_in_output);
}

TEST(RequestParamsTest, RawPolicyCannotBeOverriddenByDecodeFlag) {
  proto::ChatRequest request;
  request.set_skip_special_tokens(true);
  request.set_include_stop_str_in_output(false);
  request.set_output_decoding_policy(proto::PROTOCOL_RAW);

  RequestParams params(request, "", "");

  EXPECT_FALSE(params.skip_special_tokens);
  EXPECT_TRUE(params.include_stop_str_in_output);
}

TEST(RequestParamsTest, ResponsesGenerationIntentPropagatesTypedValues) {
  proto::ChatRequest request;
  request.set_max_tokens(77);
  request.set_temperature(1.0f);
  request.set_top_p(0.95f);
  request.set_reasoning_effort(proto::REASONING_HIGH);
  request.set_parallel_tool_calls(true);
  request.set_protocol_tool_choice(proto::TOOL_CHOICE_CUSTOM);
  request.set_protocol_tool_name("apply_patch");

  RequestParams params(request, "", "");

  EXPECT_EQ(params.max_tokens, 77);
  EXPECT_FLOAT_EQ(params.temperature, 1.0f);
  EXPECT_FLOAT_EQ(params.top_p, 0.95f);
  EXPECT_EQ(params.reasoning_effort, model_protocol::ReasoningEffort::HIGH);
  EXPECT_TRUE(params.parallel_tool_calls);
  EXPECT_EQ(params.protocol_tool_choice.kind,
            model_protocol::ToolChoiceKind::CUSTOM);
  EXPECT_EQ(params.protocol_tool_choice.name, "apply_patch");
}

TEST(RequestParamsTest, ChatFunctionToolKeepsLegacyAndTypedViews) {
  proto::ChatRequest request;
  proto::Tool* tool = request.add_tools();
  tool->set_type("function");
  tool->mutable_function()->set_name("read_file");

  RequestParams params(request, "", "");

  ASSERT_EQ(params.tools.size(), 1);
  ASSERT_EQ(params.protocol_tools.size(), 1);
  EXPECT_EQ(params.tools[0].function.name, "read_file");
  EXPECT_TRUE(std::holds_alternative<FunctionTool>(params.protocol_tools[0]));
  EXPECT_FALSE(params.tool_conversion_error.has_value());
}

TEST(RequestParamsTest, ChatCustomToolUsesTypedViewWithoutFunctionFallback) {
  proto::ChatRequest request;
  proto::Tool* tool = request.add_tools();
  tool->set_type("custom");
  tool->mutable_custom()->set_name("apply_patch");

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.tools.empty());
  ASSERT_EQ(params.protocol_tools.size(), 1);
  EXPECT_TRUE(std::holds_alternative<CustomTool>(params.protocol_tools[0]));
  EXPECT_FALSE(params.tool_conversion_error.has_value());
}

TEST(RequestParamsTest, ToolChoiceNoneKeepsTypedDefinitionForProjection) {
  proto::ChatRequest request;
  request.set_tool_choice("none");
  proto::Tool* tool = request.add_tools();
  tool->set_type("function");
  tool->mutable_function()->set_name("read_file");

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.tools.empty());
  ASSERT_EQ(params.protocol_tools.size(), 1);
  EXPECT_TRUE(std::holds_alternative<FunctionTool>(params.protocol_tools[0]));
  EXPECT_EQ(params.tool_choice, "none");
}

TEST(RequestParamsTest, ChatToolTypeMismatchIsRetainedAsValidationError) {
  proto::ChatRequest request;
  proto::Tool* tool = request.add_tools();
  tool->set_type("function");
  tool->mutable_custom()->set_name("apply_patch");

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.tools.empty());
  EXPECT_TRUE(params.protocol_tools.empty());
  ASSERT_TRUE(params.tool_conversion_error.has_value());
  EXPECT_EQ(*params.tool_conversion_error,
            ToolConversionError::TYPE_PAYLOAD_MISMATCH);
}

TEST(RequestParamsTest, IncludeStopStringInOutputIsParsedFromRequests) {
  proto::CompletionRequest completion_request;
  completion_request.set_include_stop_str_in_output(true);
  proto::ChatRequest chat_request;
  chat_request.set_include_stop_str_in_output(true);
  proto::MMChatRequest mm_chat_request;
  mm_chat_request.set_include_stop_str_in_output(true);

  RequestParams completion_params(completion_request, "", "");
  RequestParams chat_params(chat_request, "", "");
  RequestParams mm_chat_params(mm_chat_request, "", "");

  EXPECT_TRUE(completion_params.include_stop_str_in_output);
  EXPECT_TRUE(chat_params.include_stop_str_in_output);
  EXPECT_TRUE(mm_chat_params.include_stop_str_in_output);
}

TEST(RequestParamsTest, IncludeStopStringInOutputUsesVllmJsonName) {
  proto::ChatRequest request;
  auto status = google::protobuf::util::JsonStringToMessage(
      R"({"include_stop_str_in_output": true})", &request);
  ASSERT_TRUE(status.ok()) << status.ToString();

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.include_stop_str_in_output);
}

TEST(RequestParamsTest,
     CompletionBeamSearchDefaultsTopLogprobsToBeamWidthWhenUnset) {
  proto::CompletionRequest request;
  request.set_beam_width(3);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 3);
}

TEST(RequestParamsTest, CompletionBeamSearchKeepsExplicitLogprobsWhenSet) {
  proto::CompletionRequest request;
  request.set_beam_width(3);
  request.set_logprobs(5);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 5);
}

TEST(RequestParamsTest, CompletionNonBeamSearchKeepsLogprobsDisabled) {
  proto::CompletionRequest request;
  request.set_beam_width(1);

  RequestParams params(request, "", "");

  EXPECT_FALSE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 0);
}

TEST(RequestParamsTest, ChatBeamSearchDefaultsTopLogprobsToBeamWidthWhenUnset) {
  proto::ChatRequest request;
  request.set_beam_width(4);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 4);
}

TEST(RequestParamsTest, ChatBeamSearchKeepsExplicitLogprobsDisabled) {
  proto::ChatRequest request;
  request.set_beam_width(4);
  request.set_logprobs(false);

  RequestParams params(request, "", "");

  EXPECT_FALSE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 0);
}

TEST(RequestParamsTest, ChatBeamSearchKeepsExplicitTopLogprobs) {
  proto::ChatRequest request;
  request.set_beam_width(4);
  request.set_logprobs(true);
  request.set_top_logprobs(2);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 2);
}

TEST(RequestParamsTest, ChatBeamSearchKeepsExplicitZeroTopLogprobs) {
  proto::ChatRequest request;
  request.set_beam_width(4);
  request.set_logprobs(true);
  request.set_top_logprobs(0);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.logprobs);
  EXPECT_EQ(params.top_logprobs, 0);
}

TEST(RequestParamsTest, AnthropicPreservesIgnoreEos) {
  proto::AnthropicMessagesRequest request;
  request.set_model("claude-3");
  request.set_max_tokens(16);
  request.set_ignore_eos(true);

  RequestParams params(request, "", "");

  EXPECT_TRUE(params.ignore_eos);
}

TEST(RequestParamsTest, AnthropicToolChoiceDefaults) {
  proto::AnthropicMessagesRequest request;
  request.set_model("claude-3");
  request.set_max_tokens(16);

  RequestParams no_tool_params(request, "", "");
  EXPECT_EQ(no_tool_params.tool_choice, "");

  auto* tool = request.add_tools();
  tool->set_name("list_files");

  RequestParams auto_params(request, "", "");
  EXPECT_EQ(auto_params.tool_choice, "auto");

  request.mutable_tool_choice()->set_type("any");
  RequestParams required_params(request, "", "");
  EXPECT_EQ(required_params.tool_choice, "required");

  request.mutable_tool_choice()->set_type("tool");
  request.mutable_tool_choice()->clear_name();
  RequestParams fallback_params(request, "", "");
  EXPECT_EQ(fallback_params.tool_choice, "auto");

  request.mutable_tool_choice()->set_type("unknown");
  RequestParams unknown_params(request, "", "");
  EXPECT_EQ(unknown_params.tool_choice, "auto");
}

TEST(RequestParamsTest, AnthropicToolWithoutSchemaUsesEmptyJson) {
  proto::AnthropicMessagesRequest request;
  request.set_model("claude-3");
  request.set_max_tokens(16);

  auto* tool = request.add_tools();
  tool->set_name("list_files");

  RequestParams params(request, "", "");

  ASSERT_EQ(params.tools.size(), 1);
  EXPECT_TRUE(params.tools[0].function.parameters.is_object());
  EXPECT_TRUE(params.tools[0].function.parameters.empty());
}

TEST(RequestParamsTest, AnthropicToolSchemaUsesPlainJson) {
  proto::AnthropicMessagesRequest request;
  request.set_model("claude-3");
  request.set_max_tokens(16);

  auto* tool = request.add_tools();
  tool->set_name("list_files");
  tool->set_description("List files under a folder");
  const std::string schema = R"({
    "type": "object",
    "properties": {
      "path": {
        "type": "string",
        "description": "Folder path"
      },
      "recursive": {
        "type": "boolean",
        "default": false
      }
    },
    "required": ["path"]
  })";
  auto status = google::protobuf::util::JsonStringToMessage(
      schema, tool->mutable_input_schema());
  ASSERT_TRUE(status.ok()) << status.ToString();

  auto* tool_choice = request.mutable_tool_choice();
  tool_choice->set_type("tool");
  tool_choice->set_name("list_files");

  RequestParams params(request, "", "");

  ASSERT_EQ(params.tools.size(), 1);
  const auto& parsed_tool = params.tools[0];
  EXPECT_EQ(parsed_tool.type, "function");
  EXPECT_EQ(parsed_tool.function.name, "list_files");
  EXPECT_EQ(parsed_tool.function.description, "List files under a folder");

  const nlohmann::json& params_schema = parsed_tool.function.parameters;
  ASSERT_TRUE(params_schema.is_object());
  EXPECT_FALSE(params_schema.contains("fields"));
  EXPECT_EQ(params_schema.at("type"), "object");
  EXPECT_EQ(params_schema.at("properties").at("path").at("type"), "string");
  EXPECT_EQ(params_schema.at("properties").at("recursive").at("type"),
            "boolean");
  EXPECT_EQ(params_schema.at("properties").at("recursive").at("default"),
            false);
  ASSERT_TRUE(params_schema.at("required").is_array());
  EXPECT_EQ(params_schema.at("required").at(0), "path");

  nlohmann::json expected_tool_choice = {
      {"type", "function"}, {"function", {{"name", "list_files"}}}};
  EXPECT_EQ(nlohmann::json::parse(params.tool_choice), expected_tool_choice);
}

}  // namespace
}  // namespace xllm
