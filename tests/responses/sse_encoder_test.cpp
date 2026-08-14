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

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/model_protocol/output_parser.h"
#include "responses/json_encoder.h"
#include "responses/output_processor.h"

namespace xllm::responses {
namespace {

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

class ScriptedParser final : public model_protocol::ModelOutputParser {
 public:
  explicit ScriptedParser(std::vector<model_protocol::OutputSegment> segments)
      : segments_(std::move(segments)) {}

  std::vector<model_protocol::OutputSegment> consume(
      const model_protocol::GenerationDelta& /*unused*/) override {
    return std::move(segments_);
  }

  std::optional<int32_t> reasoning_tokens() const override { return 0; }

 private:
  std::vector<model_protocol::OutputSegment> segments_;
};

ProcessorConfig config() {
  ProcessorConfig value;
  value.model = "fixture-model";
  value.created_at = 123;
  return value;
}

TEST(SseEncoderTest, ProjectsTextLifecycleAndEscapesFrame) {
  ResponsesProcessor processor(config(), std::make_unique<FixtureIdProvider>());
  ScriptedParser parser(
      {{.kind = model_protocol::OutputSegmentKind::TEXT_DELTA,
        .text = "line 1\n\"雪\""},
       {.kind = model_protocol::OutputSegmentKind::TEXT_DONE}});
  const model_protocol::GenerationUsage usage{.input_tokens = 2,
                                              .output_tokens = 1,
                                              .reasoning_tokens = 0,
                                              .total_tokens = 3};
  ASSERT_TRUE(processor.consume({.sequence_index = 0,
                                 .generation_ordinal = 0,
                                 .token_id_delta = {1},
                                 .finished = true,
                                 .finish_reason = "stop",
                                 .final_usage = usage},
                                parser));

  const std::vector<ResponseEvent> events = processor.take_events();
  const std::vector<std::string> expected = {"response.created",
                                             "response.in_progress",
                                             "response.output_item.added",
                                             "response.content_part.added",
                                             "response.output_text.delta",
                                             "response.output_text.done",
                                             "response.content_part.done",
                                             "response.output_item.done",
                                             "response.completed"};
  ASSERT_EQ(events.size(), expected.size());
  for (size_t index = 0; index < events.size(); ++index) {
    EXPECT_EQ(events[index].sequence_number, index);
    EXPECT_EQ(event_type(events[index]), expected[index]);
  }

  const nlohmann::json delta = encode_event(events[4]);
  EXPECT_EQ(delta["output_index"], 0);
  EXPECT_EQ(delta["content_index"], 0);
  EXPECT_EQ(delta["item_id"], "msg_fixture_1");
  EXPECT_EQ(delta["delta"], "line 1\n\"雪\"");
  const std::string frame = encode_sse(events[4]);
  EXPECT_EQ(frame.find("event: response.output_text.delta\n"), 0);
  EXPECT_NE(frame.find("line 1\\n\\\"雪\\\""), std::string::npos);
  EXPECT_EQ(frame.substr(frame.size() - 2), "\n\n");
  EXPECT_EQ(frame.find("[DONE]"), std::string::npos);

  EXPECT_TRUE(processor.take_events().empty());
  EXPECT_FALSE(processor.consume(
      {.kind = model_protocol::OutputSegmentKind::TEXT_DELTA, .text = "late"}));
  EXPECT_TRUE(processor.take_events().empty());
}

TEST(SseEncoderTest, ProjectsReasoningAndBothToolLifecycles) {
  ResponsesProcessor processor(config(), std::make_unique<FixtureIdProvider>());
  ScriptedParser parser(
      {{.kind = model_protocol::OutputSegmentKind::REASONING_DELTA,
        .text = "think"},
       {.kind = model_protocol::OutputSegmentKind::REASONING_DONE},
       {.kind = model_protocol::OutputSegmentKind::FUNCTION_CALL_START,
        .name = "lookup",
        .call_id = "call_model_1"},
       {.kind = model_protocol::OutputSegmentKind::ARGUMENTS_DELTA,
        .text = R"({"path":"a\\\"雪"})"},
       {.kind = model_protocol::OutputSegmentKind::FUNCTION_CALL_DONE},
       {.kind = model_protocol::OutputSegmentKind::CUSTOM_CALL_START,
        .name = "apply_patch",
        .call_id = "call_model_2"},
       {.kind = model_protocol::OutputSegmentKind::CUSTOM_INPUT_DELTA,
        .text = "*** Begin Patch\n+line\n*** End Patch\n"},
       {.kind = model_protocol::OutputSegmentKind::CUSTOM_CALL_DONE}});
  const model_protocol::GenerationUsage usage{.input_tokens = 2,
                                              .output_tokens = 2,
                                              .reasoning_tokens = 0,
                                              .total_tokens = 4};
  ASSERT_TRUE(processor.consume({.sequence_index = 0,
                                 .generation_ordinal = 0,
                                 .token_id_delta = {1, 2},
                                 .finished = true,
                                 .finish_reason = "tool_calls",
                                 .final_usage = usage},
                                parser));

  const std::vector<ResponseEvent> events = processor.take_events();
  std::vector<std::string> types;
  types.reserve(events.size());
  for (const ResponseEvent& event : events) {
    types.emplace_back(event_type(event));
  }
  EXPECT_EQ(types,
            (std::vector<std::string>{"response.created",
                                      "response.in_progress",
                                      "response.output_item.added",
                                      "response.reasoning_text.delta",
                                      "response.reasoning_text.done",
                                      "response.output_item.done",
                                      "response.output_item.added",
                                      "response.function_call_arguments.delta",
                                      "response.function_call_arguments.done",
                                      "response.output_item.done",
                                      "response.output_item.added",
                                      "response.custom_tool_call_input.delta",
                                      "response.custom_tool_call_input.done",
                                      "response.output_item.done",
                                      "response.completed"}));
  EXPECT_EQ(encode_event(events[7])["item_id"], "fc_fixture_1");
  EXPECT_EQ(encode_event(events[7])["output_index"], 1);
  EXPECT_EQ(encode_event(events[11])["item_id"], "ctc_fixture_1");
  EXPECT_EQ(encode_event(events[11])["output_index"], 2);
}

TEST(SseEncoderTest, DoesNotCloseOpenItemsOnFailureOrTruncation) {
  ResponsesProcessor failed(config(), std::make_unique<FixtureIdProvider>());
  ASSERT_TRUE(failed.consume(
      {.kind = model_protocol::OutputSegmentKind::FUNCTION_CALL_START,
       .name = "lookup"}));
  ASSERT_TRUE(failed.consume(
      {.kind = model_protocol::OutputSegmentKind::ARGUMENTS_DELTA,
       .text = "{"}));
  EXPECT_FALSE(failed.consume(
      {.kind = model_protocol::OutputSegmentKind::FUNCTION_CALL_DONE}));
  const std::vector<ResponseEvent> failed_events = failed.take_events();
  EXPECT_EQ(event_type(failed_events.back()), "response.failed");
  for (const ResponseEvent& event : failed_events) {
    EXPECT_NE(event_type(event), "response.function_call_arguments.done");
    EXPECT_NE(event_type(event), "response.output_item.done");
  }

  ProcessorConfig limited = config();
  limited.max_output_tokens = 1;
  ResponsesProcessor incomplete(std::move(limited),
                                std::make_unique<FixtureIdProvider>());
  ScriptedParser parser(
      {{.kind = model_protocol::OutputSegmentKind::CUSTOM_CALL_START,
        .name = "apply_patch"},
       {.kind = model_protocol::OutputSegmentKind::CUSTOM_INPUT_DELTA,
        .text = "*** Begin Patch"}});
  const model_protocol::GenerationUsage usage{.input_tokens = 2,
                                              .output_tokens = 1,
                                              .reasoning_tokens = 0,
                                              .total_tokens = 3};
  ASSERT_TRUE(incomplete.consume({.sequence_index = 0,
                                  .generation_ordinal = 0,
                                  .token_id_delta = {1},
                                  .finish_reason = "length",
                                  .final_usage = usage},
                                 parser));
  const std::vector<ResponseEvent> incomplete_events = incomplete.take_events();
  EXPECT_EQ(event_type(incomplete_events.back()), "response.incomplete");
  for (const ResponseEvent& event : incomplete_events) {
    EXPECT_NE(event_type(event), "response.custom_tool_call_input.done");
    EXPECT_NE(event_type(event), "response.output_item.done");
  }
}

}  // namespace
}  // namespace xllm::responses
