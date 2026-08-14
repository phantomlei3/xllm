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

#include "responses/output_processor.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/model_protocol/deepseek_v4_profile.h"
#include "core/model_protocol/glm_5_2_profile.h"
#include "core/model_protocol/output_parser.h"
#include "responses/fixture_loader.h"
#include "responses/json_encoder.h"

namespace xllm::responses {
namespace {

using model_protocol::OutputSegment;
using model_protocol::OutputSegmentKind;

class FixtureIdProvider final : public IdProvider {
 public:
  std::string next(std::string_view prefix) override {
    uint32_t& count = counts_[std::string(prefix)];
    ++count;
    return std::string(prefix) + "fixture_" + std::to_string(count);
  }

 private:
  std::unordered_map<std::string, uint32_t> counts_;
};

ProcessorConfig config() {
  ProcessorConfig value;
  value.model = "fixture-model";
  value.created_at = 123;
  value.reasoning_effort = model_protocol::ReasoningEffort::LOW;
  return value;
}

std::filesystem::path fixture_root() {
  return std::filesystem::path(XLLM_SOURCE_DIR) / "tests" / "fixtures" /
         "responses";
}

std::unique_ptr<model_protocol::ModelOutputParser> parser_for(
    const std::string& profile) {
  if (profile == "deepseek_v4_responses") {
    return model_protocol::make_deepseek_v4_profile()->new_parser();
  }
  return model_protocol::make_glm_5_2_profile()->new_parser();
}

nlohmann::json normalized_items(nlohmann::json items) {
  for (nlohmann::json& item : items) {
    if (item.at("type") == "message" || item.at("type") == "custom_tool_call") {
      item.erase("status");
    }
  }
  return items;
}

TEST(ResponsesProcessorTest, BuildsReasoningThenTextWithoutEmptyItems) {
  ResponsesProcessor processor(config(), std::make_unique<FixtureIdProvider>());
  EXPECT_TRUE(processor.consume(
      {.kind = OutputSegmentKind::REASONING_DELTA, .text = "Think"}));
  EXPECT_TRUE(processor.consume({.kind = OutputSegmentKind::REASONING_DONE}));
  EXPECT_TRUE(processor.consume(
      {.kind = OutputSegmentKind::TEXT_DELTA, .text = "Answer"}));
  EXPECT_TRUE(processor.consume({.kind = OutputSegmentKind::TEXT_DONE}));

  const FinalResponse& response = processor.finish();
  ASSERT_EQ(response.output.size(), 2);
  const OutputReasoningItem& reasoning =
      std::get<OutputReasoningItem>(response.output[0]);
  EXPECT_EQ(reasoning.id, "rs_fixture_1");
  EXPECT_EQ(reasoning.content, "Think");
  EXPECT_EQ(reasoning.status, ItemStatus::COMPLETED);
  const OutputMessageItem& message =
      std::get<OutputMessageItem>(response.output[1]);
  EXPECT_EQ(message.id, "msg_fixture_1");
  EXPECT_EQ(message.content, "Answer");
  EXPECT_EQ(response.id, "resp_fixture_1");
  EXPECT_EQ(response.model, "fixture-model");
  EXPECT_FALSE(response.store);
  EXPECT_TRUE(response.parallel_tool_calls);
  EXPECT_EQ(response.reasoning_effort, model_protocol::ReasoningEffort::LOW);

  const nlohmann::json encoded = encode_response(response);
  EXPECT_EQ(encoded["object"], "response");
  EXPECT_EQ(encoded["created_at"], 123);
  EXPECT_EQ(encoded["previous_response_id"], nullptr);
  EXPECT_EQ(encoded["output"][0]["type"], "reasoning");
  EXPECT_EQ(encoded["output"][1]["content"][0]["text"], "Answer");
}

TEST(ResponsesProcessorTest, MatchesBothModelItemGoldens) {
  const testing::FixtureCatalog catalog =
      testing::FixtureCatalog::load(fixture_root());
  for (const std::string& profile : catalog.profiles()) {
    for (const std::string& scenario : catalog.scenario_ids(profile)) {
      const nlohmann::json& raw = catalog.expected(
          profile, scenario, testing::ExpectedArtifact::RAW_GENERATION);
      model_protocol::GenerationDelta delta{
          .sequence_index = 0,
          .generation_ordinal = 0,
          .text_delta = raw.at("text").get<std::string>(),
          .token_id_delta = raw.at("token_ids").get<std::vector<int32_t>>(),
          .finished = true,
          .finish_reason = raw.at("finish_reason").get<std::string>()};
      std::unique_ptr<model_protocol::ModelOutputParser> parser =
          parser_for(profile);
      ResponsesProcessor processor(config(),
                                   std::make_unique<FixtureIdProvider>());
      for (const OutputSegment& segment : parser->consume(delta)) {
        processor.consume(segment);
      }
      const nlohmann::json encoded = encode_response(processor.finish());
      EXPECT_EQ(normalized_items(encoded.at("output")),
                normalized_items(catalog.expected(
                    profile, scenario, testing::ExpectedArtifact::ITEMS)))
          << profile << "/" << scenario;
    }
  }
}

TEST(ResponsesProcessorTest, PreservesRawPatchAndValidModelCallId) {
  ResponsesProcessor processor(config(), std::make_unique<FixtureIdProvider>());
  EXPECT_TRUE(processor.consume({.kind = OutputSegmentKind::CUSTOM_CALL_START,
                                 .name = "apply_patch",
                                 .call_id = "call_from_model"}));
  const std::string patch =
      "*** Begin Patch\n*** Add File: a.txt\n+{}\\n\n*** End Patch\n";
  EXPECT_TRUE(processor.consume(
      {.kind = OutputSegmentKind::CUSTOM_INPUT_DELTA, .text = patch}));
  EXPECT_TRUE(processor.consume({.kind = OutputSegmentKind::CUSTOM_CALL_DONE}));
  const nlohmann::json item = encode_item(processor.finish().output[0]);
  EXPECT_EQ(item["call_id"], "call_from_model");
  EXPECT_EQ(item["input"], patch);
}

TEST(ResponsesProcessorTest, AvoidsReplayAndModelIdConflicts) {
  ProcessorConfig replay_config = config();
  replay_config.replayed_items = {
      FunctionCallItem{.id = "fc_fixture_1",
                       .call = FunctionCall{.id = "call_from_model",
                                            .name = "old",
                                            .arguments = "{}"}},
      ReasoningItem{.id = "rs_fixture_1", .content = "old"}};
  ResponsesProcessor processor(std::move(replay_config),
                               std::make_unique<FixtureIdProvider>());
  EXPECT_TRUE(processor.consume({.kind = OutputSegmentKind::FUNCTION_CALL_START,
                                 .name = "next",
                                 .call_id = "call_from_model"}));
  EXPECT_TRUE(processor.consume(
      {.kind = OutputSegmentKind::ARGUMENTS_DELTA, .text = "{}"}));
  EXPECT_TRUE(
      processor.consume({.kind = OutputSegmentKind::FUNCTION_CALL_DONE}));
  const OutputFunctionCallItem& item =
      std::get<OutputFunctionCallItem>(processor.finish().output[0]);
  EXPECT_EQ(item.id, "fc_fixture_2");
  EXPECT_EQ(item.call.id, "call_fixture_1");
}

TEST(ResponsesProcessorTest, FailsMalformedArgumentsBeforeCallDone) {
  ResponsesProcessor processor(config(), std::make_unique<FixtureIdProvider>());
  ASSERT_TRUE(processor.consume(
      {.kind = OutputSegmentKind::FUNCTION_CALL_START, .name = "read_file"}));
  ASSERT_TRUE(processor.consume(
      {.kind = OutputSegmentKind::ARGUMENTS_DELTA, .text = "[1]"}));
  EXPECT_FALSE(
      processor.consume({.kind = OutputSegmentKind::FUNCTION_CALL_DONE}));
  const FinalResponse before_finish = processor.response();
  EXPECT_EQ(before_finish.status, ResponseStatus::FAILED);
  ASSERT_TRUE(before_finish.error.has_value());
  EXPECT_EQ(before_finish.error->code, ErrorCode::INVALID_TOOL_ARGUMENTS);
  EXPECT_EQ(std::get<OutputFunctionCallItem>(before_finish.output[0]).status,
            ItemStatus::IN_PROGRESS);
  EXPECT_EQ(encode_response(before_finish)["error"]["code"],
            "invalid_tool_arguments");
}

TEST(ResponsesProcessorTest, RejectsLateDeltaAndLeavesTerminalUnchanged) {
  ResponsesProcessor processor(config(), std::make_unique<FixtureIdProvider>());
  ASSERT_TRUE(processor.consume(
      {.kind = OutputSegmentKind::TEXT_DELTA, .text = "done"}));
  ASSERT_TRUE(processor.consume({.kind = OutputSegmentKind::TEXT_DONE}));
  EXPECT_FALSE(processor.consume(
      {.kind = OutputSegmentKind::TEXT_DELTA, .text = "late"}));
  const FinalResponse failed = processor.response();
  EXPECT_EQ(failed.status, ResponseStatus::FAILED);
  EXPECT_EQ(std::get<OutputMessageItem>(failed.output[0]).content, "done");
  EXPECT_FALSE(processor.consume(
      {.kind = OutputSegmentKind::REASONING_DELTA, .text = "later"}));
  EXPECT_EQ(encode_response(processor.finish()), encode_response(failed));
}

TEST(ResponsesProcessorTest, DoesNotCompleteUnclosedCall) {
  ResponsesProcessor processor(config(), std::make_unique<FixtureIdProvider>());
  ASSERT_TRUE(processor.consume(
      {.kind = OutputSegmentKind::CUSTOM_CALL_START, .name = "apply_patch"}));
  ASSERT_TRUE(processor.consume({.kind = OutputSegmentKind::CUSTOM_INPUT_DELTA,
                                 .text = "*** Begin Patch"}));
  const FinalResponse& response = processor.finish();
  EXPECT_EQ(response.status, ResponseStatus::FAILED);
  EXPECT_EQ(std::get<OutputCustomToolCallItem>(response.output[0]).status,
            ItemStatus::IN_PROGRESS);
}

TEST(ResponsesProcessorTest, CreatesNoItemForEmptyReasoningBoundary) {
  ResponsesProcessor processor(config(), std::make_unique<FixtureIdProvider>());
  ASSERT_TRUE(processor.consume({.kind = OutputSegmentKind::REASONING_DONE}));
  EXPECT_TRUE(processor.finish().output.empty());
}

TEST(ResponsesProcessorTest, EnforcesPayloadLimitsWithoutChecks) {
  ProcessorConfig limited = config();
  limited.limits.max_custom_payload_bytes = 3;
  ResponsesProcessor processor(std::move(limited),
                               std::make_unique<FixtureIdProvider>());
  ASSERT_TRUE(processor.consume(
      {.kind = OutputSegmentKind::CUSTOM_CALL_START, .name = "apply_patch"}));
  EXPECT_FALSE(processor.consume(
      {.kind = OutputSegmentKind::CUSTOM_INPUT_DELTA, .text = "four"}));
  ASSERT_TRUE(processor.response().error.has_value());
  EXPECT_EQ(processor.response().error->code, ErrorCode::REQUEST_TOO_LARGE);
}

TEST(RandomIdProviderTest, ProducesPrefixedUniqueIds) {
  RandomIdProvider provider;
  const std::string first = provider.next("resp_");
  const std::string second = provider.next("resp_");
  EXPECT_EQ(first.rfind("resp_", 0), 0);
  EXPECT_EQ(second.rfind("resp_", 0), 0);
  EXPECT_NE(first, second);
}

TEST(JsonEncoderTest, MatchesFrozenNonStreamResponse) {
  FinalResponse response;
  response.id = "resp_fixture";
  response.status = ResponseStatus::COMPLETED;
  response.model = "fixture-model";
  response.output.emplace_back(
      OutputMessageItem{.id = "msg_fixture",
                        .content = "Fixture response.",
                        .status = ItemStatus::COMPLETED});
  response.usage = model_protocol::GenerationUsage{
      .input_tokens = 8, .output_tokens = 3, .total_tokens = 11};
  const testing::FixtureCatalog catalog =
      testing::FixtureCatalog::load(fixture_root());
  EXPECT_EQ(encode_response(response),
            catalog.wire("expected-responses/plain-text.json"));
}

}  // namespace
}  // namespace xllm::responses
