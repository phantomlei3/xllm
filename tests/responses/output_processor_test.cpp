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
#include "responses/sse_encoder.h"

namespace xllm::responses {
namespace {

using model_protocol::OutputSegment;
using model_protocol::OutputSegmentKind;

class ScriptedParser final : public model_protocol::ModelOutputParser {
 public:
  explicit ScriptedParser(
      std::vector<std::vector<OutputSegment>> callbacks,
      std::optional<int32_t> reasoning_tokens = std::nullopt)
      : callbacks_(std::move(callbacks)), reasoning_tokens_(reasoning_tokens) {}

  std::vector<OutputSegment> consume(
      const model_protocol::GenerationDelta& /*unused*/) override {
    if (next_ >= callbacks_.size()) {
      return {};
    }
    return callbacks_[next_++];
  }

  std::optional<int32_t> reasoning_tokens() const override {
    return reasoning_tokens_;
  }

 private:
  std::vector<std::vector<OutputSegment>> callbacks_;
  std::optional<int32_t> reasoning_tokens_;
  size_t next_ = 0;
};

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

model_protocol::GenerationUsage usage(int32_t input_tokens,
                                      int32_t output_tokens,
                                      int32_t reasoning_tokens) {
  return {.input_tokens = input_tokens,
          .cached_input_tokens = 2,
          .output_tokens = output_tokens,
          .reasoning_tokens = reasoning_tokens,
          .total_tokens = input_tokens + output_tokens};
}

model_protocol::GenerationDelta terminal_delta(
    std::string finish_reason,
    std::vector<int32_t> token_ids,
    model_protocol::GenerationUsage final_usage) {
  return {.sequence_index = 0,
          .generation_ordinal = 0,
          .token_id_delta = std::move(token_ids),
          .finished = true,
          .finish_reason = std::move(finish_reason),
          .final_usage = std::move(final_usage)};
}

TEST(ResponsesProcessorTest, BuildsReasoningThenTextWithoutEmptyItems) {
  ResponsesProcessor processor(config(), std::make_unique<FixtureIdProvider>());
  EXPECT_TRUE(processor.consume(
      {.kind = OutputSegmentKind::REASONING_DELTA, .text = "Think"}));
  EXPECT_TRUE(processor.consume({.kind = OutputSegmentKind::REASONING_DONE}));
  EXPECT_TRUE(processor.consume(
      {.kind = OutputSegmentKind::TEXT_DELTA, .text = "Answer"}));
  EXPECT_TRUE(processor.consume({.kind = OutputSegmentKind::TEXT_DONE}));

  const FinalResponse& response = processor.response();
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
      const int32_t output_tokens =
          static_cast<int32_t>(delta.token_id_delta.size());
      delta.final_usage =
          model_protocol::GenerationUsage{.input_tokens = 7,
                                          .cached_input_tokens = 2,
                                          .output_tokens = output_tokens,
                                          .total_tokens = 7 + output_tokens};
      std::unique_ptr<model_protocol::ModelOutputParser> parser =
          parser_for(profile);
      ResponsesProcessor processor(config(),
                                   std::make_unique<FixtureIdProvider>());
      ASSERT_TRUE(processor.consume(delta, *parser))
          << profile << "/" << scenario;
      const nlohmann::json encoded = encode_response(processor.response());
      EXPECT_EQ(normalized_items(encoded.at("output")),
                normalized_items(catalog.expected(
                    profile, scenario, testing::ExpectedArtifact::ITEMS)))
          << profile << "/" << scenario;
      const std::vector<ResponseEvent> events = processor.take_events();
      ASSERT_FALSE(events.empty()) << profile << "/" << scenario;
      for (size_t index = 0; index < events.size(); ++index) {
        EXPECT_EQ(events[index].sequence_number, index)
            << profile << "/" << scenario;
      }
      const nlohmann::json terminal = encode_event(events.back());
      EXPECT_EQ(terminal.at("response"), encoded) << profile << "/" << scenario;
      EXPECT_EQ(
          terminal.at("type"),
          catalog
              .expected(
                  profile, scenario, testing::ExpectedArtifact::SSE_EVENTS)
              .back()
              .at("type"))
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
  const nlohmann::json item = encode_item(processor.response().output[0]);
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
      std::get<OutputFunctionCallItem>(processor.response().output[0]);
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
  EXPECT_TRUE(processor.response().output.empty());
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

TEST(ResponsesProcessorTerminalTest, CompletesEosStopAndToolStopWithUsage) {
  const std::vector<std::pair<std::string, std::vector<OutputSegment>>> cases =
      {{"stop",
        {{.kind = OutputSegmentKind::TEXT_DELTA, .text = "done"},
         {.kind = OutputSegmentKind::TEXT_DONE}}},
       {"eos",
        {{.kind = OutputSegmentKind::TEXT_DELTA, .text = "done"},
         {.kind = OutputSegmentKind::TEXT_DONE}}},
       {"tool_calls",
        {{.kind = OutputSegmentKind::FUNCTION_CALL_START, .name = "read_file"},
         {.kind = OutputSegmentKind::ARGUMENTS_DELTA, .text = "{}"},
         {.kind = OutputSegmentKind::FUNCTION_CALL_DONE}}}};
  for (const auto& [reason, segments] : cases) {
    ResponsesProcessor processor(config(),
                                 std::make_unique<FixtureIdProvider>());
    ScriptedParser parser({segments}, /*reasoning_tokens=*/0);
    EXPECT_TRUE(processor.consume(
        terminal_delta(reason, {10, 11}, usage(8, 2, 0)), parser));
    const FinalResponse& response = processor.response();
    EXPECT_EQ(response.status, ResponseStatus::COMPLETED) << reason;
    ASSERT_TRUE(response.usage.has_value());
    EXPECT_EQ(response.usage->cached_input_tokens, 2);
    EXPECT_EQ(response.usage->total_tokens, 10);
    ASSERT_EQ(response.output.size(), 1);
    EXPECT_EQ(std::visit([](const auto& item) { return item.status; },
                         response.output[0]),
              ItemStatus::COMPLETED);
  }
}

TEST(ResponsesProcessorTerminalTest,
     TokenLimitKeepsDoneItemsAndLeavesOpenToolUndone) {
  ProcessorConfig limited = config();
  limited.max_output_tokens = 4;
  ResponsesProcessor processor(std::move(limited),
                               std::make_unique<FixtureIdProvider>());
  ScriptedParser parser(
      {{{.kind = OutputSegmentKind::REASONING_DELTA, .text = "thought"},
        {.kind = OutputSegmentKind::REASONING_DONE},
        {.kind = OutputSegmentKind::CUSTOM_CALL_START, .name = "apply_patch"},
        {.kind = OutputSegmentKind::CUSTOM_INPUT_DELTA,
         .text = "*** Begin Patch"}}},
      /*reasoning_tokens=*/1);

  EXPECT_TRUE(processor.consume(
      terminal_delta("length", {10, 20, 30, 40}, usage(6, 4, 1)), parser));
  const FinalResponse& response = processor.response();
  EXPECT_EQ(response.status, ResponseStatus::INCOMPLETE);
  EXPECT_EQ(response.incomplete_reason, "max_output_tokens");
  ASSERT_EQ(response.output.size(), 2);
  EXPECT_EQ(std::get<OutputReasoningItem>(response.output[0]).status,
            ItemStatus::COMPLETED);
  EXPECT_EQ(std::get<OutputCustomToolCallItem>(response.output[1]).status,
            ItemStatus::IN_PROGRESS);
  const nlohmann::json encoded = encode_response(response);
  EXPECT_EQ(encoded["error"], nullptr);
  EXPECT_EQ(encoded["incomplete_details"]["reason"], "max_output_tokens");
  EXPECT_EQ(encoded["usage"]["output_tokens"], 4);
}

TEST(ResponsesProcessorTerminalTest, MaxTokensCountsEveryGeneratedToken) {
  ProcessorConfig limited = config();
  limited.max_output_tokens = 3;
  ResponsesProcessor processor(std::move(limited),
                               std::make_unique<FixtureIdProvider>());
  ScriptedParser parser(
      {{{.kind = OutputSegmentKind::REASONING_DELTA, .text = "r"},
        {.kind = OutputSegmentKind::REASONING_DONE}},
       {{.kind = OutputSegmentKind::TEXT_DELTA, .text = "t"}}},
      /*reasoning_tokens=*/1);
  EXPECT_TRUE(processor.consume(
      {.sequence_index = 0, .generation_ordinal = 0, .token_id_delta = {101}},
      parser));
  EXPECT_TRUE(processor.consume({.sequence_index = 0,
                                 .generation_ordinal = 1,
                                 .token_id_delta = {102, 103},
                                 .final_usage = usage(5, 3, 1)},
                                parser));
  EXPECT_EQ(processor.response().status, ResponseStatus::INCOMPLETE);
  EXPECT_EQ(processor.response().incomplete_reason, "max_output_tokens");
}

TEST(ResponsesProcessorTerminalTest, MapsBackendTimeoutAndDisconnect) {
  ResponsesProcessor backend(config(), std::make_unique<FixtureIdProvider>());
  ScriptedParser unused({});
  EXPECT_FALSE(backend.consume(
      {.sequence_index = 0,
       .generation_ordinal = 0,
       .backend_error =
           model_protocol::BackendError{.code = "worker_failed",
                                        .message = "worker unavailable"}},
      unused));
  EXPECT_EQ(backend.response().status, ResponseStatus::FAILED);
  EXPECT_EQ(backend.response().error->code, ErrorCode::GENERATION_FAILED);

  ResponsesProcessor timeout(config(), std::make_unique<FixtureIdProvider>());
  timeout.timeout();
  EXPECT_EQ(timeout.response().status, ResponseStatus::FAILED);
  EXPECT_EQ(timeout.response().error->code, ErrorCode::REQUEST_TIMEOUT);

  ResponsesProcessor disconnected(config(),
                                  std::make_unique<FixtureIdProvider>());
  disconnected.cancel();
  EXPECT_EQ(disconnected.response().status, ResponseStatus::CANCELLED);
  EXPECT_EQ(disconnected.response().error, std::nullopt);

  ResponsesProcessor parser_failure(config(),
                                    std::make_unique<FixtureIdProvider>());
  ScriptedParser failed_parser(
      {{{.kind = OutputSegmentKind::PARSE_FAILURE,
         .failure =
             model_protocol::ParseFailure{
                 .code =
                     model_protocol::ParseFailureCode::UNKNOWN_CONTROL_TOKEN,
                 .message = "unknown marker"}}}},
      /*reasoning_tokens=*/0);
  EXPECT_FALSE(parser_failure.consume(
      terminal_delta("stop", {1}, usage(2, 1, 0)), failed_parser));
  EXPECT_EQ(parser_failure.response().status, ResponseStatus::FAILED);
  EXPECT_EQ(parser_failure.response().error->code,
            ErrorCode::GENERATION_FAILED);
  EXPECT_EQ(encode_response(parser_failure.response())["error"]["code"],
            "generation_failed");
}

TEST(ResponsesProcessorTerminalTest, RejectsSequenceAndOrdinalViolations) {
  ResponsesProcessor sequence(config(), std::make_unique<FixtureIdProvider>());
  ScriptedParser sequence_parser({{}, {}});
  ASSERT_TRUE(sequence.consume({.sequence_index = 0, .generation_ordinal = 0},
                               sequence_parser));
  EXPECT_FALSE(sequence.consume({.sequence_index = 1, .generation_ordinal = 1},
                                sequence_parser));
  EXPECT_EQ(sequence.response().error->code, ErrorCode::GENERATION_FAILED);

  ResponsesProcessor ordinal(config(), std::make_unique<FixtureIdProvider>());
  ScriptedParser ordinal_parser({{}, {}});
  ASSERT_TRUE(ordinal.consume({.sequence_index = 0, .generation_ordinal = 2},
                              ordinal_parser));
  EXPECT_FALSE(ordinal.consume({.sequence_index = 0, .generation_ordinal = 2},
                               ordinal_parser));
  EXPECT_EQ(ordinal.response().error->code, ErrorCode::GENERATION_FAILED);

  ResponsesProcessor reversed(config(), std::make_unique<FixtureIdProvider>());
  ScriptedParser reversed_parser({{}, {}});
  ASSERT_TRUE(reversed.consume({.sequence_index = 0, .generation_ordinal = 3},
                               reversed_parser));
  EXPECT_FALSE(reversed.consume({.sequence_index = 0, .generation_ordinal = 1},
                                reversed_parser));
  EXPECT_EQ(reversed.response().error->code, ErrorCode::GENERATION_FAILED);
}

TEST(ResponsesProcessorTerminalTest, FailsClosedOnUnreliableUsage) {
  ResponsesProcessor missing(config(), std::make_unique<FixtureIdProvider>());
  ScriptedParser missing_parser(
      {{{.kind = OutputSegmentKind::TEXT_DELTA, .text = "done"},
        {.kind = OutputSegmentKind::TEXT_DONE}}},
      /*reasoning_tokens=*/0);
  EXPECT_FALSE(missing.consume({.sequence_index = 0,
                                .generation_ordinal = 0,
                                .token_id_delta = {1},
                                .finished = true,
                                .finish_reason = "stop"},
                               missing_parser));
  EXPECT_EQ(missing.response().status, ResponseStatus::FAILED);
  EXPECT_EQ(missing.response().usage, std::nullopt);

  ResponsesProcessor mismatch(config(), std::make_unique<FixtureIdProvider>());
  ScriptedParser mismatch_parser({{}}, /*reasoning_tokens=*/2);
  EXPECT_FALSE(mismatch.consume(
      terminal_delta("stop", {1, 2, 3}, usage(4, 3, 1)), mismatch_parser));
  EXPECT_EQ(mismatch.response().status, ResponseStatus::FAILED);
  EXPECT_EQ(mismatch.response().error->code, ErrorCode::GENERATION_FAILED);
}

TEST(ResponsesProcessorTerminalTest, UsesVerifiedParserReasoningAttribution) {
  ResponsesProcessor processor(config(), std::make_unique<FixtureIdProvider>());
  ScriptedParser parser(
      {{{.kind = OutputSegmentKind::REASONING_DELTA, .text = "thought"},
        {.kind = OutputSegmentKind::REASONING_DONE}}},
      /*reasoning_tokens=*/2);
  model_protocol::GenerationUsage backend_usage{.input_tokens = 5,
                                                .cached_input_tokens = 1,
                                                .output_tokens = 3,
                                                .total_tokens = 8};
  ASSERT_TRUE(processor.consume(
      terminal_delta("stop", {1, 2, 3}, backend_usage), parser));
  ASSERT_TRUE(processor.response().usage.has_value());
  EXPECT_EQ(processor.response().usage->reasoning_tokens, 2);
  EXPECT_EQ(
      encode_response(processor.response())["usage"]["output_tokens_details"]
                                           ["reasoning_tokens"],
      2);
}

TEST(ResponsesProcessorTerminalTest, FirstTerminalCauseWinsInEitherOrder) {
  ResponsesProcessor finish_first(config(),
                                  std::make_unique<FixtureIdProvider>());
  ScriptedParser finish_parser(
      {{{.kind = OutputSegmentKind::TEXT_DELTA, .text = "done"},
        {.kind = OutputSegmentKind::TEXT_DONE}}},
      /*reasoning_tokens=*/0);
  ASSERT_TRUE(finish_first.consume(terminal_delta("stop", {1}, usage(2, 1, 0)),
                                   finish_parser));
  const nlohmann::json completed = encode_response(finish_first.response());
  finish_first.timeout();
  EXPECT_EQ(encode_response(finish_first.response()), completed);

  ResponsesProcessor timeout_first(config(),
                                   std::make_unique<FixtureIdProvider>());
  timeout_first.timeout();
  const nlohmann::json failed = encode_response(timeout_first.response());
  ScriptedParser late_parser({{}});
  EXPECT_FALSE(timeout_first.consume(
      terminal_delta("stop", {1}, usage(2, 1, 0)), late_parser));
  EXPECT_EQ(encode_response(timeout_first.response()), failed);

  ResponsesProcessor cancel_first(config(),
                                  std::make_unique<FixtureIdProvider>());
  cancel_first.cancel();
  const nlohmann::json cancelled = encode_response(cancel_first.response());
  ScriptedParser cancelled_parser({{}});
  EXPECT_FALSE(cancel_first.consume(terminal_delta("stop", {1}, usage(2, 1, 0)),
                                    cancelled_parser));
  EXPECT_EQ(encode_response(cancel_first.response()), cancelled);
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
