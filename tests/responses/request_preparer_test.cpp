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

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include "responses/fixture_loader.h"

namespace xllm::responses {
namespace {

const std::filesystem::path kFixtureRoot =
    std::filesystem::path(XLLM_SOURCE_DIR) / "tests/fixtures/responses";

model_protocol::ModelProtocolIdentity fixture_identity() {
  return {.profile_id = "fixture_responses",
          .model_type = "fixture-type",
          .tokenizer_fingerprint = "fixture-tokenizer-fingerprint",
          .template_fingerprint = "fixture-fingerprint"};
}

PrepareResult prepare(const std::string& body,
                      ResponsesLimits limits = ResponsesLimits()) {
  return prepare_request(
      body,
      "fixture-model",
      fixture_identity(),
      {.request_id = "req_fixture", .trace_id = "trace_fixture"},
      limits);
}

void expect_error(const std::string& body,
                  ErrorCode code,
                  const std::string& param,
                  ResponsesLimits limits = ResponsesLimits()) {
  PrepareResult result = prepare(body, limits);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error().code, code);
  EXPECT_EQ(result.error().param, param);
}

TEST(RequestPreparerTest, NormalizesStringInputToGoldenSingleSequence) {
  testing::FixtureCatalog fixtures =
      testing::FixtureCatalog::load(kFixtureRoot);
  const nlohmann::json& request = fixtures.wire_input("plain_text_nonstream");
  const nlohmann::json& expected = fixtures.wire_expected(
      "plain_text_nonstream", testing::ExpectedArtifact::PREPARED_REQUEST);

  PrepareResult result = prepare(request.dump());

  ASSERT_TRUE(result.ok()) << result.error().message;
  const PreparedRequest& prepared = result.value();
  ASSERT_EQ(prepared.canonical_input.size(), 1);
  const MessageItem& item = std::get<MessageItem>(prepared.canonical_input[0]);
  EXPECT_EQ(item.role,
            expected["canonical_items"][0]["role"].get<std::string>());
  EXPECT_EQ(
      item.content,
      expected["canonical_items"][0]["content"][0]["text"].get<std::string>());
  EXPECT_EQ(prepared.chat_request.messages_size(), 1);
  EXPECT_EQ(prepared.chat_request.messages(0).role(), "user");
  EXPECT_EQ(prepared.chat_request.messages(0).content(), item.content);
  EXPECT_EQ(prepared.sequence_count,
            expected["sequence_count"].get<uint32_t>());
  EXPECT_EQ(prepared.chat_request.n(), 1);
  EXPECT_EQ(prepared.chat_request.thinking_history_policy(), proto::PRESERVE);
  EXPECT_EQ(prepared.chat_request.output_decoding_policy(),
            proto::PROTOCOL_RAW);
  EXPECT_EQ(prepared.profile_id, "fixture_responses");
  EXPECT_EQ(prepared.model_id, expected["model"].get<std::string>());
  EXPECT_EQ(prepared.context.request_id, "req_fixture");
}

TEST(RequestPreparerTest, AcceptsEveryCapturedCodexRequest) {
  testing::FixtureCatalog fixtures =
      testing::FixtureCatalog::load(kFixtureRoot);
  const std::vector<std::string> paths = {
      "requests/apply-patch-stream.json",
      "requests/codex-tool-loop-start.json",
      "requests/function-tool-output.json",
      "requests/function-tool-stream.json",
      "requests/plain-text-nonstream.json",
      "requests/tool-loop-replay.json",
  };

  for (const std::string& path : paths) {
    PrepareResult result = prepare(fixtures.wire(path).dump());
    EXPECT_TRUE(result.ok()) << path << ": " << result.error().message;
  }
}

TEST(RequestPreparerTest, AcceptsOnlyCapturedNoEffectShapes) {
  const std::vector<std::pair<std::string, std::string>> rejected = {
      {R"({"model":"fixture-model","input":"x","reasoning":{"effort":"medium","summary":"detailed"}})",
       "reasoning.summary"},
      {R"({"model":"fixture-model","input":"x","include":["reasoning.summary"]})",
       "include"},
      {R"({"model":"fixture-model","input":"x","prompt_cache_key":""})",
       "prompt_cache_key"},
      {R"({"model":"fixture-model","input":"x","client_metadata":{"other":"value"}})",
       "client_metadata.other"},
      {R"({"model":"fixture-model","input":"x","text":{"verbosity":"high"}})",
       "text.verbosity"},
      {R"({"model":"fixture-model","input":"x","tools":[{"type":"custom","name":"apply_patch","format":{"type":"grammar","syntax":"lark","definition":"other"}}]})",
       "tools[0].format.definition"},
  };

  for (const auto& [body, param] : rejected) {
    expect_error(body, ErrorCode::UNSUPPORTED_PARAMETER, param);
  }
}

TEST(RequestPreparerTest, KeepsInstructionsAndDeveloperAsSystemMessages) {
  PrepareResult result = prepare(R"({
    "model":"fixture-model",
    "instructions":"first",
    "input":[
      {"type":"message","role":"developer","content":"second"},
      {"type":"message","role":"system","content":[{"type":"input_text","text":"third"}]},
      {"type":"message","role":"user","content":"你好 🌍"}
    ]
  })");

  ASSERT_TRUE(result.ok()) << result.error().message;
  const PreparedRequest& prepared = result.value();
  ASSERT_EQ(prepared.canonical_input.size(), 4);
  EXPECT_EQ(std::get<MessageItem>(prepared.canonical_input[0]).role, "system");
  EXPECT_EQ(std::get<MessageItem>(prepared.canonical_input[1]).role, "system");
  ASSERT_EQ(prepared.chat_request.messages_size(), 4);
  EXPECT_EQ(prepared.chat_request.messages(0).role(), "system");
  EXPECT_EQ(prepared.chat_request.messages(1).role(), "system");
  EXPECT_EQ(prepared.chat_request.messages(2).role(), "system");
  EXPECT_EQ(prepared.chat_request.messages(3).content(), "你好 🌍");
}

TEST(RequestPreparerTest, AggregatesCanonicalAssistantGroupsWithoutEmptyItems) {
  PrepareResult result = prepare(R"({
    "model":"fixture-model",
    "input":[
      {"type":"message","role":"user","content":"question"},
      {"type":"reasoning","id":"rs_1","content":[{"type":"reasoning_text","text":"think"}]},
      {"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"answer"}]},
      {"type":"message","role":"user","content":"next"},
      {"type":"reasoning","id":"rs_2","content":"only thought"}
    ]
  })");

  ASSERT_TRUE(result.ok()) << result.error().message;
  const PreparedRequest& prepared = result.value();
  ASSERT_EQ(prepared.canonical_input.size(), 5);
  EXPECT_EQ(std::get<ReasoningItem>(prepared.canonical_input[1]).id, "rs_1");
  EXPECT_EQ(std::get<MessageItem>(prepared.canonical_input[2]).id, "msg_1");
  ASSERT_EQ(prepared.chat_request.messages_size(), 4);
  EXPECT_EQ(prepared.chat_request.messages(1).reasoning_content(), "think");
  EXPECT_EQ(prepared.chat_request.messages(1).content(), "answer");
  EXPECT_EQ(prepared.chat_request.messages(3).reasoning_content(),
            "only thought");
  EXPECT_FALSE(prepared.chat_request.messages(3).has_content());
}

TEST(RequestPreparerTest, RejectsInvalidAssistantOrderDeterministically) {
  const std::vector<std::string> bodies = {
      R"({"model":"fixture-model","input":[{"type":"message","role":"assistant","content":"answer"},{"type":"reasoning","content":"late"}]})",
      R"({"model":"fixture-model","input":[{"type":"reasoning","content":"one"},{"type":"reasoning","content":"two"}]})",
      R"({"model":"fixture-model","input":[{"type":"message","role":"assistant","content":"one"},{"type":"message","role":"assistant","content":"two"}]})",
  };
  for (const std::string& body : bodies) {
    expect_error(body, ErrorCode::INVALID_ITEM_ORDER, "input[1]");
  }
}

TEST(RequestPreparerTest, RejectsUnsupportedFieldsAndPayloads) {
  struct Case {
    std::string body;
    ErrorCode code;
    std::string param;
  };
  const std::vector<Case> cases = {
      {R"({"model":"fixture-model","input":"x","unknown":1})",
       ErrorCode::UNSUPPORTED_PARAMETER,
       "unknown"},
      {R"({"model":"fixture-model","input":[{"type":"other"}]})",
       ErrorCode::UNSUPPORTED_ITEM_TYPE,
       "input[0].type"},
      {R"({"model":"fixture-model","input":[{"type":"additional_tools"}]})",
       ErrorCode::UNSUPPORTED_ITEM_TYPE,
       "input[0].type"},
      {R"({"model":"fixture-model","input":[{"type":"image_generation_call"}]})",
       ErrorCode::UNSUPPORTED_ITEM_TYPE,
       "input[0].type"},
      {R"({"model":"fixture-model","input":[{"type":"web_search_call"}]})",
       ErrorCode::UNSUPPORTED_ITEM_TYPE,
       "input[0].type"},
      {R"({"model":"fixture-model","input":[{"type":"computer_call"}]})",
       ErrorCode::UNSUPPORTED_ITEM_TYPE,
       "input[0].type"},
      {R"({"model":"fixture-model","input":[{"type":"file_search_call"}]})",
       ErrorCode::UNSUPPORTED_ITEM_TYPE,
       "input[0].type"},
      {R"({"model":"fixture-model","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"x"}]}]})",
       ErrorCode::UNSUPPORTED_CONTENT_TYPE,
       "input[0].content[0].type"},
      {R"({"model":"fixture-model","input":[{"type":"message","role":"user","content":[{"type":"input_file","file_id":"x"}]}]})",
       ErrorCode::UNSUPPORTED_CONTENT_TYPE,
       "input[0].content[0].type"},
      {R"({"model":"fixture-model","input":[{"type":"message","role":"user","content":[{"type":"input_audio","audio":"x"}]}]})",
       ErrorCode::UNSUPPORTED_CONTENT_TYPE,
       "input[0].content[0].type"},
      {R"({"model":"fixture-model","input":[{"type":"message","role":"user","content":[{"type":"video","video_url":"x"}]}]})",
       ErrorCode::UNSUPPORTED_CONTENT_TYPE,
       "input[0].content[0].type"},
      {R"({"model":"fixture-model","input":[{"type":"message","role":"user","content":[{"type":"refusal","refusal":"x"}]}]})",
       ErrorCode::UNSUPPORTED_CONTENT_TYPE,
       "input[0].content[0].type"},
      {R"({"model":"fixture-model","input":[{"type":"reasoning","summary":[{"text":"summary"}],"content":"x"}]})",
       ErrorCode::UNSUPPORTED_PARAMETER,
       "input[0].summary"},
      {R"({"model":"fixture-model","input":[{"type":"reasoning","encrypted_content":"cipher","content":"x"}]})",
       ErrorCode::UNSUPPORTED_PARAMETER,
       "input[0].encrypted_content"},
      {R"({"model":"fixture-model","input":[{"type":"message","role":"user","content":[{"type":"output_text","text":"x"}]}]})",
       ErrorCode::UNSUPPORTED_CONTENT_TYPE,
       "input[0].content[0].type"},
      {R"({"model":"fixture-model","input":[{"type":"message","role":"tool","content":"x"}]})",
       ErrorCode::INVALID_REQUEST,
       "input[0].role"},
      {R"({"model":"fixture-model","input":"x","reasoning":{"effort":"medium","summary":"detailed"}})",
       ErrorCode::UNSUPPORTED_PARAMETER,
       "reasoning.summary"},
  };
  for (const Case& test_case : cases) {
    expect_error(test_case.body, test_case.code, test_case.param);
  }
}

TEST(RequestPreparerTest, AcceptsOnlyEmptyReasoningReplayMetadata) {
  PrepareResult result = prepare(R"({
    "model":"fixture-model",
    "input":[{
      "type":"reasoning",
      "id":"rs_replay",
      "summary":[],
      "encrypted_content":null,
      "content":[{"type":"reasoning_text","text":"preserved"}]
    }]
  })");

  ASSERT_TRUE(result.ok()) << result.error().message;
  ASSERT_EQ(result.value().canonical_input.size(), 1);
  EXPECT_EQ(std::get<ReasoningItem>(result.value().canonical_input[0]).content,
            "preserved");
}

TEST(RequestPreparerTest, DoesNotCreateCanonicalItemsFromEmptyInput) {
  expect_error(R"({"model":"fixture-model","input":[]})",
               ErrorCode::INVALID_REQUEST,
               "input");
}

TEST(RequestPreparerTest, EnforcesStatelessPresenceAndModelContract) {
  expect_error(R"({"model":"fixture-model","input":"x","store":true})",
               ErrorCode::UNSUPPORTED_PARAMETER,
               "store");
  expect_error(
      R"({"model":"fixture-model","input":"x","previous_response_id":"resp_old"})",
      ErrorCode::UNSUPPORTED_PARAMETER,
      "previous_response_id");
  expect_error(
      R"({"model":"fixture-model"})", ErrorCode::INVALID_REQUEST, "input");
  expect_error(
      R"({"model":"" ,"input":"x"})", ErrorCode::INVALID_REQUEST, "model");
  expect_error(
      R"({"model":"other","input":"x"})", ErrorCode::MODEL_MISMATCH, "model");

  EXPECT_TRUE(
      prepare(
          R"({"model":"fixture-model","instructions":"only","store":false,"previous_response_id":null})")
          .ok());
}

TEST(RequestPreparerTest, ValidatesIdsUtf8DepthBodyAndTextLimits) {
  expect_error(R"({"model":"fixture-model","input":[
    {"type":"reasoning","id":"rs_dup","content":"one"},
    {"type":"message","id":"rs_dup","role":"assistant","content":"two"}
  ]})",
               ErrorCode::INVALID_REQUEST,
               "input[1].id");
  expect_error(R"({"model":"fixture-model","input":[
    {"type":"reasoning","id":"bad id","content":"one"}
  ]})",
               ErrorCode::INVALID_REQUEST,
               "input[0].id");

  ResponsesLimits tiny;
  tiny.max_body_bytes = 20;
  expect_error(R"({"model":"fixture-model","input":"x"})",
               ErrorCode::REQUEST_TOO_LARGE,
               "",
               tiny);
  tiny = ResponsesLimits();
  tiny.max_text_bytes = 3;
  expect_error(R"({"model":"fixture-model","input":"four"})",
               ErrorCode::REQUEST_TOO_LARGE,
               "input",
               tiny);
  tiny = ResponsesLimits();
  tiny.max_json_depth = 2;
  expect_error(
      R"({"model":"fixture-model","input":[{"type":"message","role":"user","content":"x"}]})",
      ErrorCode::MAX_DEPTH_EXCEEDED,
      "",
      tiny);

  std::string invalid_utf8 =
      std::string("{\"model\":\"fixture-model\",\"input\":\"") +
      static_cast<char>(0xff) + "\"}";
  expect_error(invalid_utf8, ErrorCode::INVALID_JSON, "");

  tiny = ResponsesLimits();
  tiny.max_input_items = 1;
  expect_error(R"({"model":"fixture-model","input":[
    {"type":"message","role":"user","content":"one"},
    {"type":"message","role":"user","content":"two"}
  ]})",
               ErrorCode::TOO_MANY_ITEMS,
               "input",
               tiny);
  tiny.max_input_items = 0;
  expect_error(R"({"model":"fixture-model","input":"one"})",
               ErrorCode::TOO_MANY_ITEMS,
               "input",
               tiny);
}

TEST(RequestPreparerTest, ProducesDeterministicRequestAndError) {
  const std::string valid =
      R"({"model":"fixture-model","input":"same","stream":true})";
  PrepareResult first = prepare(valid);
  PrepareResult second = prepare(valid);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(first.value().chat_request.SerializeAsString(),
            second.value().chat_request.SerializeAsString());
  EXPECT_EQ(std::get<MessageItem>(first.value().canonical_input[0]).content,
            std::get<MessageItem>(second.value().canonical_input[0]).content);

  const std::string invalid_body =
      R"({"model":"fixture-model","input":"x","z":1,"a":2})";
  PrepareResult first_error = prepare(invalid_body);
  PrepareResult second_error = prepare(invalid_body);
  ASSERT_FALSE(first_error.ok());
  ASSERT_FALSE(second_error.ok());
  EXPECT_EQ(first_error.error().code, second_error.error().code);
  EXPECT_EQ(first_error.error().param, "a");
  EXPECT_EQ(first_error.error().param, second_error.error().param);
  EXPECT_EQ(first_error.error().message, second_error.error().message);
}

TEST(RequestPreparerTest, PreparesTypedToolsCallsAndOutputs) {
  PrepareResult result = prepare(R"({
    "model":"fixture-model",
    "input":[
      {"type":"message","role":"user","content":"inspect"},
      {"type":"reasoning","id":"rs_1","content":"plan"},
      {"type":"function_call","id":"fc_1","call_id":"call_read","name":"read_file","arguments":"{\"path\":\"a.txt\"}"},
      {"type":"custom_tool_call","id":"ctc_1","status":"completed","call_id":"call_patch","name":"apply_patch","input":"*** Begin Patch\n*** End Patch"},
      {"type":"function_call_output","id":"fco_1","call_id":"call_read","output":"old"},
      {"type":"custom_tool_call_output","id":"ctco_1","call_id":"call_patch","output":"Done"},
      {"type":"message","role":"user","content":"continue"}
    ],
    "tools":[
      {"type":"function","name":"read_file","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}}},"strict":false},
      {"type":"custom","name":"apply_patch"}
    ],
    "tool_choice":{"type":"custom","name":"apply_patch"}
  })");

  ASSERT_TRUE(result.ok()) << result.error().message;
  const PreparedRequest& prepared = result.value();
  ASSERT_EQ(prepared.canonical_input.size(), 7);
  EXPECT_TRUE(
      std::holds_alternative<FunctionCallItem>(prepared.canonical_input[2]));
  EXPECT_TRUE(
      std::holds_alternative<CustomToolCallItem>(prepared.canonical_input[3]));
  EXPECT_TRUE(
      std::holds_alternative<FunctionCallOutput>(prepared.canonical_input[4]));
  EXPECT_TRUE(std::holds_alternative<CustomToolCallOutput>(
      prepared.canonical_input[5]));
  EXPECT_EQ(std::get<FunctionCallOutput>(prepared.canonical_input[4]).id,
            "fco_1");
  EXPECT_EQ(std::get<CustomToolCallOutput>(prepared.canonical_input[5]).id,
            "ctco_1");
  ASSERT_EQ(prepared.chat_request.tools_size(), 2);
  EXPECT_EQ(prepared.chat_request.tools(0).type(), "function");
  EXPECT_EQ(prepared.chat_request.tools(1).type(), "custom");
  EXPECT_EQ(prepared.chat_request.protocol_tool_choice(),
            proto::TOOL_CHOICE_CUSTOM);
  EXPECT_EQ(prepared.chat_request.protocol_tool_name(), "apply_patch");
  ASSERT_EQ(prepared.chat_request.messages_size(), 5);
  ASSERT_EQ(prepared.chat_request.messages(1).tool_calls_size(), 2);
  EXPECT_EQ(prepared.chat_request.messages(1).tool_calls(0).type(), "function");
  EXPECT_EQ(prepared.chat_request.messages(1).tool_calls(1).type(), "custom");
  EXPECT_EQ(prepared.chat_request.messages(2).tool_output_type(),
            proto::FUNCTION_OUTPUT);
  EXPECT_EQ(prepared.chat_request.messages(3).tool_output_type(),
            proto::CUSTOM_OUTPUT);
}

TEST(RequestPreparerTest, ValidatesToolDefinitionsChoicesAndLimits) {
  struct Case {
    std::string suffix;
    ErrorCode code;
    std::string param;
  };
  const std::vector<Case> cases = {
      {R"(,"tools":[{"type":"function","name":"f","parameters":{"type":"object"},"strict":true}]})",
       ErrorCode::UNSUPPORTED_PARAMETER,
       "tools[0].strict"},
      {R"(,"tools":[{"type":"custom","name":"other"}]})",
       ErrorCode::UNKNOWN_TOOL,
       "tools[0].name"},
      {R"(,"tools":[{"type":"web_search"}]})",
       ErrorCode::UNKNOWN_TOOL,
       "tools[0].type"},
      {R"(,"tools":[{"type":"file_search"}]})",
       ErrorCode::UNKNOWN_TOOL,
       "tools[0].type"},
      {R"(,"tools":[{"type":"code_interpreter"}]})",
       ErrorCode::UNKNOWN_TOOL,
       "tools[0].type"},
      {R"(,"tools":[{"type":"computer_use"}]})",
       ErrorCode::UNKNOWN_TOOL,
       "tools[0].type"},
      {R"(,"tools":[{"type":"mcp"}]})",
       ErrorCode::UNKNOWN_TOOL,
       "tools[0].type"},
      {R"(,"tools":[{"type":"function","name":"f","parameters":[]}]})",
       ErrorCode::INVALID_REQUEST,
       "tools[0].parameters"},
      {R"(,"tool_choice":"required"})",
       ErrorCode::INVALID_REQUEST,
       "tool_choice"},
      {R"(,"tools":[{"type":"function","name":"f","parameters":{}}],"tool_choice":{"type":"function","name":"missing"}})",
       ErrorCode::UNKNOWN_TOOL,
       "tool_choice.name"},
      {R"(,"tools":[{"type":"custom","name":"apply_patch"}],"tool_choice":{"type":"function","name":"apply_patch"}})",
       ErrorCode::UNKNOWN_TOOL,
       "tool_choice.name"},
  };
  for (const Case& test_case : cases) {
    const std::string body =
        R"({"model":"fixture-model","input":"x")" + test_case.suffix;
    expect_error(body, test_case.code, test_case.param);
  }

  ResponsesLimits tiny;
  tiny.max_tools = 0;
  expect_error(
      R"({"model":"fixture-model","input":"x","tools":[{"type":"custom","name":"apply_patch"}]})",
      ErrorCode::TOO_MANY_ITEMS,
      "tools",
      tiny);
  tiny = ResponsesLimits();
  tiny.max_function_schema_bytes = 1;
  expect_error(
      R"({"model":"fixture-model","input":"x","tools":[{"type":"function","name":"f","parameters":{}}]})",
      ErrorCode::REQUEST_TOO_LARGE,
      "tools[0].parameters",
      tiny);
  tiny = ResponsesLimits();
  tiny.max_tool_bytes = 90;
  expect_error(
      R"({"model":"fixture-model","input":"x","tools":[{"type":"function","name":"first","parameters":{}},{"type":"function","name":"second","parameters":{}}]})",
      ErrorCode::REQUEST_TOO_LARGE,
      "tools",
      tiny);
}

TEST(RequestPreparerTest, AcceptsEverySupportedToolChoice) {
  const std::vector<std::string> choices = {
      R"("none")",
      R"("auto")",
      R"("required")",
      R"({"type":"function","name":"read_file"})",
      R"({"type":"custom","name":"apply_patch"})",
  };
  for (const std::string& choice : choices) {
    PrepareResult result = prepare(
        R"({"model":"fixture-model","input":"x","tools":[{"type":"function","name":"read_file","parameters":{}},{"type":"custom","name":"apply_patch"}],"tool_choice":)" +
        choice + "}");
    EXPECT_TRUE(result.ok()) << result.error().message;
  }
}

TEST(RequestPreparerTest, ValidatesCallLinkageAndAssistantGrammar) {
  struct Case {
    std::string input;
    ErrorCode code;
    std::string param;
  };
  const std::vector<Case> cases = {
      {R"([{"type":"function_call_output","call_id":"missing","output":"x"}])",
       ErrorCode::UNKNOWN_CALL_ID,
       "input[0].call_id"},
      {R"([{"type":"function_call","call_id":"dup","name":"a","arguments":"{}"},{"type":"custom_tool_call","call_id":"dup","name":"apply_patch","input":"p"}])",
       ErrorCode::DUPLICATE_CALL_ID,
       "input[1].call_id"},
      {R"([{"type":"function_call","call_id":"c","name":"a","arguments":"{}"},{"type":"custom_tool_call_output","call_id":"c","output":"x"}])",
       ErrorCode::TOOL_CALL_TYPE_MISMATCH,
       "input[1].call_id"},
      {R"([{"type":"function_call","call_id":"c","name":"a","arguments":"{}"},{"type":"function_call_output","call_id":"c","output":"x"},{"type":"function_call_output","call_id":"c","output":"y"}])",
       ErrorCode::INVALID_ITEM_ORDER,
       "input[2].call_id"},
      {R"([{"type":"function_call","call_id":"c","name":"a","arguments":"{}"},{"type":"message","role":"user","content":"too soon"}])",
       ErrorCode::INVALID_ITEM_ORDER,
       "input[1]"},
      {R"([{"type":"function_call","call_id":"c","name":"a","arguments":"[]"}])",
       ErrorCode::INVALID_TOOL_ARGUMENTS,
       "input[0].arguments"},
      {R"([{"type":"message","role":"assistant","content":"done"},{"type":"function_call","call_id":"c","name":"a","arguments":"{}"}])",
       ErrorCode::INVALID_ITEM_ORDER,
       "input[1]"},
      {R"([{"type":"custom_tool_call","status":"in_progress","call_id":"c","name":"apply_patch","input":"p"}])",
       ErrorCode::INVALID_REQUEST,
       "input[0].status"},
  };
  for (const Case& test_case : cases) {
    expect_error(R"({"model":"fixture-model","input":)" + test_case.input + "}",
                 test_case.code,
                 test_case.param);
  }
}

TEST(RequestPreparerTest, CompletesParallelCallsInAnyOutputOrder) {
  PrepareResult result = prepare(R"({
    "model":"fixture-model",
    "input":[
      {"type":"function_call","call_id":"call_a","name":"a","arguments":"{}"},
      {"type":"function_call","call_id":"call_b","name":"b","arguments":"{}"},
      {"type":"function_call_output","call_id":"call_b","output":"b"},
      {"type":"function_call_output","call_id":"call_a","output":"a"},
      {"type":"message","role":"user","content":"continue"}
    ]
  })");

  ASSERT_TRUE(result.ok()) << result.error().message;
  ASSERT_EQ(result.value().chat_request.messages_size(), 4);
  EXPECT_EQ(result.value().chat_request.messages(0).tool_calls(0).id(),
            "call_a");
  EXPECT_EQ(result.value().chat_request.messages(0).tool_calls(1).id(),
            "call_b");
  EXPECT_EQ(result.value().chat_request.messages(1).tool_call_id(), "call_b");
  EXPECT_EQ(result.value().chat_request.messages(2).tool_call_id(), "call_a");
}

TEST(RequestPreparerTest, AcceptsEveryReasoningEffort) {
  struct Case {
    std::string effort;
    proto::ReasoningEffort expected;
  };
  const std::vector<Case> cases = {
      {"none", proto::REASONING_NONE},
      {"minimal", proto::REASONING_MINIMAL},
      {"low", proto::REASONING_LOW},
      {"medium", proto::REASONING_MEDIUM},
      {"high", proto::REASONING_HIGH},
      {"xhigh", proto::REASONING_XHIGH},
      {"max", proto::REASONING_MAX},
  };
  for (const Case& test_case : cases) {
    PrepareResult result = prepare(
        R"({"model":"fixture-model","input":"x","reasoning":{"effort":")" +
        test_case.effort + R"("}})");
    ASSERT_TRUE(result.ok()) << result.error().message;
    EXPECT_EQ(result.value().chat_request.reasoning_effort(),
              test_case.expected);
  }
}

TEST(RequestPreparerTest, AppliesGenerationIntentToRequestParams) {
  PrepareResult thinking = prepare(R"({
    "model":"fixture-model","input":"x","reasoning":{"effort":"high"},
    "temperature":0.2,"top_p":0.3,"max_output_tokens":77,
    "parallel_tool_calls":true
  })");
  ASSERT_TRUE(thinking.ok()) << thinking.error().message;
  EXPECT_EQ(thinking.value().chat_request.max_tokens(), 77);
  EXPECT_FLOAT_EQ(thinking.value().chat_request.temperature(), 1.0f);
  EXPECT_FLOAT_EQ(thinking.value().chat_request.top_p(), 0.95f);
  EXPECT_EQ(thinking.value().chat_request.reasoning_effort(),
            proto::REASONING_HIGH);
  EXPECT_TRUE(thinking.value().chat_request.parallel_tool_calls());

  PrepareResult no_thinking = prepare(R"({
    "model":"fixture-model","input":"x","reasoning":{"effort":"none"},
    "temperature":0.2,"top_p":0.3
  })");
  ASSERT_TRUE(no_thinking.ok()) << no_thinking.error().message;
  EXPECT_FLOAT_EQ(no_thinking.value().chat_request.temperature(), 0.2f);
  EXPECT_FLOAT_EQ(no_thinking.value().chat_request.top_p(), 0.3f);

  const std::vector<std::string> unsupported_fields = {"n",
                                                       "best_of",
                                                       "beam_width",
                                                       "max_tokens",
                                                       "presence_penalty",
                                                       "frequency_penalty",
                                                       "metadata",
                                                       "conversation",
                                                       "background",
                                                       "truncation",
                                                       "user",
                                                       "top_k",
                                                       "repetition_penalty"};
  for (const std::string& field : unsupported_fields) {
    expect_error(
        R"({"model":"fixture-model","input":"x",")" + field + R"(":1})",
        ErrorCode::UNSUPPORTED_PARAMETER,
        field);
  }
  expect_error(R"({"model":"fixture-model","input":"x","temperature":2.1})",
               ErrorCode::INVALID_REQUEST,
               "temperature");
  expect_error(R"({"model":"fixture-model","input":"x","top_p":-0.1})",
               ErrorCode::INVALID_REQUEST,
               "top_p");
  expect_error(R"({"model":"fixture-model","input":"x","max_output_tokens":0})",
               ErrorCode::INVALID_REQUEST,
               "max_output_tokens");
}

TEST(RequestPreparerTest, EnforcesFunctionAndCustomPayloadLimits) {
  ResponsesLimits tiny;
  tiny.max_function_args_bytes = 1;
  expect_error(
      R"({"model":"fixture-model","input":[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"}]})",
      ErrorCode::REQUEST_TOO_LARGE,
      "input[0].arguments",
      tiny);

  tiny = ResponsesLimits();
  tiny.max_custom_payload_bytes = 1;
  expect_error(
      R"({"model":"fixture-model","input":[{"type":"custom_tool_call","call_id":"c","name":"apply_patch","input":"xx"}]})",
      ErrorCode::REQUEST_TOO_LARGE,
      "input[0].input",
      tiny);
  expect_error(
      R"({"model":"fixture-model","input":[{"type":"custom_tool_call","call_id":"c","name":"apply_patch","input":"x"},{"type":"custom_tool_call_output","call_id":"c","output":"xx"}]})",
      ErrorCode::REQUEST_TOO_LARGE,
      "input[1].output",
      tiny);
}

}  // namespace
}  // namespace xllm::responses
