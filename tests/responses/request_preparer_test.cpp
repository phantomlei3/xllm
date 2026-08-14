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

#include "responses/fixture_loader.h"

namespace xllm::responses {
namespace {

const std::filesystem::path kFixtureRoot =
    std::filesystem::path(XLLM_SOURCE_DIR) / "tests/fixtures/responses";

model_protocol::ModelProtocolIdentity fixture_identity() {
  return {.profile_id = "fixture_responses",
          .canonical_model_id = "fixture-model",
          .model_aliases = {"fixture-alias"},
          .tokenizer_id = "fixture-tokenizer",
          .template_id = "fixture-template",
          .template_fingerprint = "fixture-fingerprint"};
}

PrepareResult prepare(const std::string& body,
                      ResponsesLimits limits = ResponsesLimits()) {
  return prepare_request(
      body,
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
  EXPECT_EQ(prepared.canonical_model_id, expected["model"].get<std::string>());
  EXPECT_EQ(prepared.context.request_id, "req_fixture");
}

TEST(RequestPreparerTest, KeepsInstructionsAndDeveloperAsSystemMessages) {
  PrepareResult result = prepare(R"({
    "model":"fixture-alias",
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
      {R"({"model":"fixture-model","input":"x","reasoning":{"effort":"medium","summary":"auto"}})",
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

}  // namespace
}  // namespace xllm::responses
