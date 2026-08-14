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

#include <gtest/gtest.h>

#include <variant>

#include "common.pb.h"
#include "core/common/tool.h"
#include "responses/error.h"
#include "responses/responses_limits.h"
#include "responses/types.h"

namespace xllm::responses {
namespace {

TEST(TypedProtocolTest, DomainDistinguishesFunctionAndCustomPayloads) {
  Tool function_tool = FunctionTool{.name = "read_file",
                                    .description = "Read a file",
                                    .parameters = {{"type", "object"}}};
  Tool custom_tool = CustomTool{.name = "apply_patch"};
  ToolCall function_call = FunctionCall{
      .id = "call_1", .name = "read_file", .arguments = R"({"path":"a.txt"})"};
  ToolCall custom_call = CustomToolCall{
      .id = "call_2", .name = "apply_patch", .input = "*** Begin Patch"};

  EXPECT_TRUE(std::holds_alternative<FunctionTool>(function_tool));
  EXPECT_TRUE(std::holds_alternative<CustomTool>(custom_tool));
  EXPECT_TRUE(std::holds_alternative<FunctionCall>(function_call));
  EXPECT_TRUE(std::holds_alternative<CustomToolCall>(custom_call));

  InputItem function_output =
      FunctionCallOutput{.call_id = "call_1", .output = "contents"};
  InputItem custom_output =
      CustomToolCallOutput{.call_id = "call_2", .output = "Done"};
  EXPECT_TRUE(std::holds_alternative<FunctionCallOutput>(function_output));
  EXPECT_TRUE(std::holds_alternative<CustomToolCallOutput>(custom_output));
}

TEST(TypedProtocolTest, ProtobufRoundTripsLegacyFunctionFields) {
  EXPECT_EQ(proto::Tool::kFunctionFieldNumber, 2);
  EXPECT_EQ(proto::Tool::kCustomFieldNumber, 3);
  EXPECT_EQ(proto::ToolCall::kFunctionFieldNumber, 4);
  EXPECT_EQ(proto::ToolCall::kCustomFieldNumber, 5);
  proto::Tool original;
  original.set_type("function");
  original.mutable_function()->set_name("read_file");
  original.mutable_function()->set_description("Read a file");

  proto::Tool restored;
  ASSERT_TRUE(restored.ParseFromString(original.SerializeAsString()));
  ToolConversion result = tool_from_proto(restored);

  ASSERT_TRUE(result.ok());
  ASSERT_TRUE(std::holds_alternative<FunctionTool>(*result.value()));
  EXPECT_EQ(std::get<FunctionTool>(*result.value()).name, "read_file");
  EXPECT_EQ(restored.function().name(), "read_file");
}

TEST(TypedProtocolTest, ProtobufConvertsCustomToolAndCall) {
  proto::Tool custom_tool;
  custom_tool.set_type("custom");
  custom_tool.mutable_custom()->set_name("apply_patch");
  ToolConversion tool_result = tool_from_proto(custom_tool);
  ASSERT_TRUE(tool_result.ok());
  EXPECT_EQ(std::get<CustomTool>(*tool_result.value()).name, "apply_patch");

  proto::ToolCall custom_call;
  custom_call.set_type("custom");
  custom_call.set_id("call_2");
  custom_call.mutable_custom()->set_name("apply_patch");
  custom_call.mutable_custom()->set_input("*** Begin Patch");
  ToolCallConversion call_result = tool_call_from_proto(custom_call);
  ASSERT_TRUE(call_result.ok());
  EXPECT_EQ(std::get<CustomToolCall>(*call_result.value()).input,
            "*** Begin Patch");

  proto::ToolOutput output;
  output.mutable_custom()->set_call_id("call_2");
  output.mutable_custom()->set_output("Done");
  ToolOutputConversion output_result = tool_output_from_proto(output);
  ASSERT_TRUE(output_result.ok());
  EXPECT_EQ(std::get<CustomToolCallOutput>(*output_result.value()).output,
            "Done");
}

TEST(TypedProtocolTest, RejectsTypeAndPayloadMismatch) {
  proto::Tool mismatched_tool;
  mismatched_tool.set_type("function");
  mismatched_tool.mutable_custom()->set_name("apply_patch");
  ToolConversion tool_result = tool_from_proto(mismatched_tool);
  ASSERT_FALSE(tool_result.ok());
  EXPECT_EQ(tool_result.error(), ToolConversionError::TYPE_PAYLOAD_MISMATCH);

  proto::ToolCall mismatched_call;
  mismatched_call.set_type("custom");
  mismatched_call.mutable_function()->set_name("read_file");
  ToolCallConversion call_result = tool_call_from_proto(mismatched_call);
  ASSERT_FALSE(call_result.ok());
  EXPECT_EQ(call_result.error(), ToolConversionError::TYPE_PAYLOAD_MISMATCH);
}

TEST(TypedProtocolTest, FoundationCarriesTypedErrorLimitsAndPreparedRequest) {
  ResponsesError error{.message = "unsupported model",
                       .type = "invalid_request_error",
                       .param = "model",
                       .code = ErrorCode::UNSUPPORTED_MODEL_CAPABILITY};
  ResponsesLimits limits;
  ResponsesRequest request;
  request.model = "fake/model";
  PreparedRequest prepared;
  prepared.profile_id = "fake_responses";
  prepared.canonical_model_id = "fake/model";

  EXPECT_EQ(error.code, ErrorCode::UNSUPPORTED_MODEL_CAPABILITY);
  EXPECT_EQ(limits.max_input_items, 4096);
  EXPECT_EQ(request.model, "fake/model");
  EXPECT_EQ(prepared.profile_id, "fake_responses");

  model_protocol::ModelProtocolError protocol_error(
      model_protocol::ModelProtocolErrorCode::UNSUPPORTED_MODEL_CAPABILITY,
      "model has no profile");
  ResponsesError mapped_error = from_protocol_error(protocol_error);
  EXPECT_EQ(mapped_error.code, ErrorCode::UNSUPPORTED_MODEL_CAPABILITY);
  EXPECT_EQ(mapped_error.param, "model");
}

}  // namespace
}  // namespace xllm::responses
