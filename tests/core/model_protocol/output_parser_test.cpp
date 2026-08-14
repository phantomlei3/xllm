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

#include "core/model_protocol/output_parser.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/model_protocol/deepseek_v4_profile.h"
#include "core/model_protocol/generation_delta.h"
#include "core/model_protocol/glm_5_2_profile.h"
#include "responses/fixture_loader.h"

namespace xllm::model_protocol {
namespace {

using xllm::testing::ExpectedArtifact;
using xllm::testing::FixtureCatalog;

struct RawFixture {
  std::string text;
  std::vector<int32_t> token_ids;
  std::vector<std::string> token_text;
  std::string finish_reason;
};

std::filesystem::path fixture_root() {
  return std::filesystem::path(XLLM_SOURCE_DIR) / "tests" / "fixtures" /
         "responses";
}

RawFixture load_raw(const FixtureCatalog& catalog,
                    const std::string& profile,
                    const std::string& scenario = "reasoning_text_stop") {
  const nlohmann::json& raw =
      catalog.expected(profile, scenario, ExpectedArtifact::RAW_GENERATION);
  return {.text = raw.at("text").get<std::string>(),
          .token_ids = raw.at("token_ids").get<std::vector<int32_t>>(),
          .token_text = raw.value("token_text", std::vector<std::string>{}),
          .finish_reason = raw.at("finish_reason").get<std::string>()};
}

std::vector<OutputSegment> call_segments(
    const std::vector<OutputSegment>& segments) {
  std::vector<OutputSegment> calls;
  for (const OutputSegment& segment : segments) {
    if (segment.kind == OutputSegmentKind::FUNCTION_CALL_START ||
        segment.kind == OutputSegmentKind::ARGUMENTS_DELTA ||
        segment.kind == OutputSegmentKind::FUNCTION_CALL_DONE ||
        segment.kind == OutputSegmentKind::CUSTOM_CALL_START ||
        segment.kind == OutputSegmentKind::CUSTOM_INPUT_DELTA ||
        segment.kind == OutputSegmentKind::CUSTOM_CALL_DONE) {
      calls.emplace_back(segment);
    }
  }
  return calls;
}

std::vector<OutputSegment> load_segments(const FixtureCatalog& catalog,
                                         const std::string& profile) {
  const nlohmann::json& expected = catalog.expected(
      profile, "reasoning_text_stop", ExpectedArtifact::SEGMENTS);
  std::vector<OutputSegment> segments;
  segments.reserve(expected.size());
  for (const nlohmann::json& segment : expected) {
    const std::string kind = segment.at("kind").get<std::string>();
    OutputSegmentKind segment_kind = OutputSegmentKind::REASONING_DELTA;
    if (kind == "reasoning_done") {
      segment_kind = OutputSegmentKind::REASONING_DONE;
    } else if (kind == "text_delta") {
      segment_kind = OutputSegmentKind::TEXT_DELTA;
    } else if (kind == "text_done") {
      segment_kind = OutputSegmentKind::TEXT_DONE;
    }
    segments.emplace_back(
        OutputSegment{.kind = segment_kind,
                      .raw = segment.at("raw").get<std::string>(),
                      .text = segment.value("text", ""),
                      .incomplete = segment.value("incomplete", false)});
  }
  return segments;
}

void append_segments(std::vector<OutputSegment>* output,
                     std::vector<OutputSegment> next) {
  for (OutputSegment& segment : next) {
    if (!output->empty() && segment.kind == output->back().kind &&
        (segment.kind == OutputSegmentKind::REASONING_DELTA ||
         segment.kind == OutputSegmentKind::TEXT_DELTA)) {
      output->back().raw += segment.raw;
      output->back().text += segment.text;
      output->back().incomplete = segment.incomplete;
      continue;
    }
    output->emplace_back(std::move(segment));
  }
}

std::vector<OutputSegment> parse_chunks(
    std::unique_ptr<ModelOutputParser> parser,
    const std::vector<std::pair<std::string, std::vector<int32_t>>>& chunks,
    const std::string& finish_reason) {
  std::vector<OutputSegment> output;
  for (size_t index = 0; index < chunks.size(); ++index) {
    const bool finished = index + 1 == chunks.size();
    GenerationDelta delta{
        .sequence_index = 0,
        .generation_ordinal = index,
        .text_delta = chunks[index].first,
        .token_id_delta = chunks[index].second,
        .finished = finished,
        .finish_reason = finished ? std::optional<std::string>(finish_reason)
                                  : std::nullopt};
    append_segments(&output, parser->consume(delta));
  }
  return output;
}

std::vector<std::pair<std::string, std::vector<int32_t>>> token_chunks(
    const RawFixture& raw) {
  std::vector<std::pair<std::string, std::vector<int32_t>>> chunks;
  chunks.reserve(raw.token_ids.size());
  for (size_t index = 0; index < raw.token_ids.size(); ++index) {
    chunks.emplace_back(raw.token_text[index],
                        std::vector<int32_t>{raw.token_ids[index]});
  }
  return chunks;
}

std::vector<std::pair<std::string, std::vector<int32_t>>> utf8_chunks(
    const std::string& text) {
  std::vector<std::pair<std::string, std::vector<int32_t>>> chunks;
  size_t index = 0;
  while (index < text.size()) {
    const uint8_t first = static_cast<uint8_t>(text[index]);
    size_t width = 1;
    if ((first & 0xe0) == 0xc0) {
      width = 2;
    } else if ((first & 0xf0) == 0xe0) {
      width = 3;
    } else if ((first & 0xf8) == 0xf0) {
      width = 4;
    }
    chunks.emplace_back(text.substr(index, width), std::vector<int32_t>{});
    index += width;
  }
  return chunks;
}

std::vector<std::pair<std::string, std::vector<int32_t>>> byte_chunks(
    const std::string& text) {
  std::vector<std::pair<std::string, std::vector<int32_t>>> chunks;
  chunks.reserve(text.size());
  for (char byte : text) {
    chunks.emplace_back(std::string(1, byte), std::vector<int32_t>{});
  }
  return chunks;
}

void expect_fixture_chunk_invariance(
    const FixtureCatalog& catalog,
    const std::shared_ptr<const ModelProtocolProfile>& profile) {
  const RawFixture raw = load_raw(catalog, profile->identity().profile_id);
  const std::vector<OutputSegment> whole = parse_chunks(
      profile->new_parser(), {{raw.text, raw.token_ids}}, raw.finish_reason);
  const std::vector<OutputSegment> by_token =
      parse_chunks(profile->new_parser(), token_chunks(raw), raw.finish_reason);
  const std::vector<OutputSegment> by_character = parse_chunks(
      profile->new_parser(), utf8_chunks(raw.text), raw.finish_reason);

  EXPECT_EQ(whole, by_token);
  EXPECT_EQ(whole, by_character);
  EXPECT_EQ(whole, load_segments(catalog, profile->identity().profile_id));
  ASSERT_EQ(whole.size(), 4);
  EXPECT_EQ(whole[0].kind, OutputSegmentKind::REASONING_DELTA);
  EXPECT_EQ(whole[1].kind, OutputSegmentKind::REASONING_DONE);
  EXPECT_EQ(whole[2].kind, OutputSegmentKind::TEXT_DELTA);
  EXPECT_EQ(whole[3].kind, OutputSegmentKind::TEXT_DONE);
}

TEST(ModelOutputParserTest, FrozenTextFixturesAreChunkInvariant) {
  const FixtureCatalog catalog = FixtureCatalog::load(fixture_root());

  expect_fixture_chunk_invariance(catalog, make_deepseek_v4_profile());
  expect_fixture_chunk_invariance(catalog, make_glm_5_2_profile());
}

TEST(ModelOutputParserTest, FrozenToolFixturesHaveTypedLifecycles) {
  const FixtureCatalog catalog = FixtureCatalog::load(fixture_root());
  for (const std::shared_ptr<const ModelProtocolProfile>& profile :
       {make_deepseek_v4_profile(), make_glm_5_2_profile()}) {
    const RawFixture function_raw = load_raw(
        catalog, profile->identity().profile_id, "reasoning_function_call");
    const std::vector<OutputSegment> function = call_segments(
        parse_chunks(profile->new_parser(),
                     {{function_raw.text, function_raw.token_ids}},
                     function_raw.finish_reason));
    ASSERT_EQ(function.size(), 3);
    EXPECT_EQ(function[0].kind, OutputSegmentKind::FUNCTION_CALL_START);
    EXPECT_EQ(function[0].name, "get_weather");
    EXPECT_FALSE(function[0].call_id.has_value());
    EXPECT_EQ(function[1].kind, OutputSegmentKind::ARGUMENTS_DELTA);
    EXPECT_EQ(function[1].text, R"({"city":"Beijing"})");
    EXPECT_EQ(function[2].kind, OutputSegmentKind::FUNCTION_CALL_DONE);

    const RawFixture patch_raw = load_raw(
        catalog, profile->identity().profile_id, "reasoning_apply_patch");
    const std::vector<OutputSegment> patch =
        call_segments(parse_chunks(profile->new_parser(),
                                   {{patch_raw.text, patch_raw.token_ids}},
                                   patch_raw.finish_reason));
    ASSERT_EQ(patch.size(), 3);
    EXPECT_EQ(patch[0].kind, OutputSegmentKind::CUSTOM_CALL_START);
    EXPECT_EQ(patch[0].name, "apply_patch");
    EXPECT_EQ(patch[1].kind, OutputSegmentKind::CUSTOM_INPUT_DELTA);
    EXPECT_EQ(patch[1].text,
              "*** Begin Patch\n*** Add File: fixture.txt\n+hello\n"
              "*** End Patch\n");
    EXPECT_EQ(patch[2].kind, OutputSegmentKind::CUSTOM_CALL_DONE);
  }
}

TEST(ModelOutputParserTest, ParallelCallsAndToolContinuationsStayTyped) {
  const FixtureCatalog catalog = FixtureCatalog::load(fixture_root());
  for (const std::shared_ptr<const ModelProtocolProfile>& profile :
       {make_deepseek_v4_profile(), make_glm_5_2_profile()}) {
    const RawFixture parallel = load_raw(
        catalog, profile->identity().profile_id, "parallel_function_calls");
    const std::vector<OutputSegment> calls =
        call_segments(parse_chunks(profile->new_parser(),
                                   {{parallel.text, parallel.token_ids}},
                                   parallel.finish_reason));
    ASSERT_EQ(calls.size(), 6);
    EXPECT_EQ(calls[0].name, "get_weather");
    EXPECT_EQ(calls[3].name, "get_time");

    for (const std::string& scenario :
         {"function_output_continue", "custom_output_continue"}) {
      const RawFixture continuation =
          load_raw(catalog, profile->identity().profile_id, scenario);
      const std::vector<OutputSegment> output =
          parse_chunks(profile->new_parser(),
                       {{continuation.text, continuation.token_ids}},
                       continuation.finish_reason);
      EXPECT_TRUE(call_segments(output).empty());
      EXPECT_NE(std::find_if(output.begin(),
                             output.end(),
                             [](const auto& item) {
                               return item.kind ==
                                      OutputSegmentKind::TEXT_DELTA;
                             }),
                output.end());
    }
  }
}

TEST(ModelOutputParserTest, ToolFixturesAreCharacterChunkInvariant) {
  const FixtureCatalog catalog = FixtureCatalog::load(fixture_root());
  for (const std::shared_ptr<const ModelProtocolProfile>& profile :
       {make_deepseek_v4_profile(), make_glm_5_2_profile()}) {
    for (const std::string& scenario :
         {"parallel_function_calls", "reasoning_apply_patch"}) {
      const RawFixture raw =
          load_raw(catalog, profile->identity().profile_id, scenario);
      const std::vector<OutputSegment> whole =
          parse_chunks(profile->new_parser(),
                       {{raw.text, raw.token_ids}},
                       raw.finish_reason);
      const std::vector<OutputSegment> by_character = parse_chunks(
          profile->new_parser(), utf8_chunks(raw.text), raw.finish_reason);
      const std::vector<OutputSegment> by_byte = parse_chunks(
          profile->new_parser(), byte_chunks(raw.text), raw.finish_reason);
      EXPECT_EQ(whole, by_character);
      EXPECT_EQ(whole, by_byte);
    }
  }
}

TEST(ModelOutputParserTest, PreservesModelCallIdWithoutGeneratingOne) {
  const std::string raw =
      "</think><｜DSML｜tool_calls><｜DSML｜invoke name=\"lookup\" "
      "id=\"call_model_7\"><｜DSML｜parameter name=\"query\" "
      "string=\"true\">x</｜DSML｜parameter></｜DSML｜invoke>"
      "</｜DSML｜tool_calls><｜end▁of▁sentence｜>";
  const std::vector<OutputSegment> calls = call_segments(parse_chunks(
      make_deepseek_v4_profile()->new_parser(), {{raw, {}}}, "tool_calls"));
  ASSERT_EQ(calls.size(), 3);
  ASSERT_TRUE(calls[0].call_id.has_value());
  EXPECT_EQ(*calls[0].call_id, "call_model_7");
}

TEST(ModelOutputParserTest, FunctionJsonKeepsEscapeUnicodeAndNesting) {
  const std::string value = R"({"label":"北\"京","nested":{"items":[1,true]}})";
  const std::string raw =
      "</think><tool_call>lookup<arg_key>config</arg_key><arg_value>" + value +
      "</arg_value></tool_call><|observation|>";
  const std::vector<OutputSegment> whole = call_segments(parse_chunks(
      make_glm_5_2_profile()->new_parser(), {{raw, {}}}, "tool_calls"));
  const std::vector<OutputSegment> split = call_segments(parse_chunks(
      make_glm_5_2_profile()->new_parser(), byte_chunks(raw), "tool_calls"));
  ASSERT_EQ(whole.size(), 3);
  EXPECT_EQ(whole, split);
  EXPECT_EQ(whole[1].text,
            R"({"config":{"label":"北\"京","nested":{"items":[1,true]}}})");
}

TEST(ModelOutputParserTest, PatchKeepsControlLikeTextVerbatim) {
  const std::string input =
      "line <tool_call>x</tool_call>\n<arg_key>literal</arg_key> end\n";
  const std::string raw =
      "</think><tool_call>apply_patch<arg_key>patch</arg_key><arg_value>" +
      input + "</arg_value></tool_call><|observation|>";
  const std::vector<OutputSegment> calls = call_segments(parse_chunks(
      make_glm_5_2_profile()->new_parser(), {{raw, {}}}, "tool_calls"));
  ASSERT_EQ(calls.size(), 3);
  EXPECT_EQ(calls[1].kind, OutputSegmentKind::CUSTOM_INPUT_DELTA);
  EXPECT_EQ(calls[1].text, input);

  const std::string ds_input =
      "line </｜DSML｜parameter> text </｜DSML｜invoke> end\n";
  const std::string ds_raw =
      "</think><｜DSML｜tool_calls><｜DSML｜invoke name=\"apply_patch\">"
      "<｜DSML｜parameter name=\"patch\" string=\"true\">" +
      ds_input +
      "</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"
      "<｜end▁of▁sentence｜>";
  const std::vector<OutputSegment> ds_calls = call_segments(parse_chunks(
      make_deepseek_v4_profile()->new_parser(), {{ds_raw, {}}}, "tool_calls"));
  ASSERT_EQ(ds_calls.size(), 3);
  EXPECT_EQ(ds_calls[1].text, ds_input);
}

TEST(ModelOutputParserTest, RejectsMalformedAndUnclosedCallsWithoutDone) {
  const std::string malformed =
      "</think><tool_call>lookup<arg_key>config</arg_key><arg_value>"
      "{\"broken\"</arg_value></tool_call><|observation|>";
  std::vector<OutputSegment> invalid = parse_chunks(
      make_glm_5_2_profile()->new_parser(), {{malformed, {}}}, "tool_calls");
  ASSERT_EQ(invalid.back().kind, OutputSegmentKind::PARSE_FAILURE);
  EXPECT_EQ(invalid.back().failure->code,
            ParseFailureCode::INVALID_TOOL_ARGUMENTS);
  EXPECT_EQ(std::count_if(invalid.begin(),
                          invalid.end(),
                          [](const auto& item) {
                            return item.kind ==
                                   OutputSegmentKind::FUNCTION_CALL_DONE;
                          }),
            0);

  const std::string unclosed =
      "</think><tool_call>lookup<arg_key>city</arg_key><arg_value>Beijing";
  std::vector<OutputSegment> unfinished = parse_chunks(
      make_glm_5_2_profile()->new_parser(), {{unclosed, {}}}, "stop");
  ASSERT_EQ(unfinished.back().kind, OutputSegmentKind::PARSE_FAILURE);
  EXPECT_EQ(unfinished.back().failure->code,
            ParseFailureCode::UNCLOSED_TOOL_CALL);
  EXPECT_EQ(std::count_if(unfinished.begin(),
                          unfinished.end(),
                          [](const auto& item) {
                            return item.kind ==
                                   OutputSegmentKind::FUNCTION_CALL_DONE;
                          }),
            0);
}

TEST(ModelOutputParserTest, RejectsUnknownCustomAndExcessiveJsonDepth) {
  const std::string unknown_custom =
      "</think><｜DSML｜tool_calls><｜DSML｜invoke name=\"other_patch\" "
      "type=\"custom\"><｜DSML｜parameter name=\"patch\" "
      "string=\"true\">x</｜DSML｜parameter></｜DSML｜invoke>"
      "</｜DSML｜tool_calls><｜end▁of▁sentence｜>";
  std::vector<OutputSegment> custom =
      parse_chunks(make_deepseek_v4_profile()->new_parser(),
                   {{unknown_custom, {}}},
                   "tool_calls");
  ASSERT_EQ(custom.back().kind, OutputSegmentKind::PARSE_FAILURE);
  EXPECT_EQ(custom.back().failure->code, ParseFailureCode::UNKNOWN_CUSTOM_TOOL);

  TextReasoningGrammar grammar{
      .reasoning_end = "</think>",
      .text_end = "<|user|>",
      .max_marker_bytes = 15,
      .tool = {.dialect = ToolGrammarDialect::GLM_NATIVE, .max_json_depth = 3}};
  const std::string nested =
      "</think><tool_call>lookup<arg_key>config</arg_key><arg_value>"
      "{\"outer\":{\"inner\":1}}</arg_value></tool_call>"
      "<|observation|>";
  std::vector<OutputSegment> depth = parse_chunks(
      make_text_reasoning_parser(grammar), {{nested, {}}}, "tool_calls");
  ASSERT_EQ(depth.back().kind, OutputSegmentKind::PARSE_FAILURE);
  EXPECT_EQ(depth.back().failure->code,
            ParseFailureCode::INVALID_TOOL_ARGUMENTS);
}

TEST(ModelOutputParserTest, EnforcesFunctionAndCustomPayloadLimits) {
  TextReasoningGrammar function_grammar{
      .reasoning_end = "</think>",
      .text_end = "<|user|>",
      .max_marker_bytes = 15,
      .tool = {.dialect = ToolGrammarDialect::GLM_NATIVE,
               .max_arguments_bytes = 8}};
  const std::string function =
      "</think><tool_call>lookup<arg_key>city</arg_key><arg_value>Beijing"
      "</arg_value></tool_call><|observation|>";
  std::vector<OutputSegment> arguments =
      parse_chunks(make_text_reasoning_parser(function_grammar),
                   {{function, {}}},
                   "tool_calls");
  ASSERT_EQ(arguments.back().kind, OutputSegmentKind::PARSE_FAILURE);
  EXPECT_EQ(arguments.back().failure->code,
            ParseFailureCode::TOOL_ARGUMENTS_TOO_LARGE);

  TextReasoningGrammar custom_grammar{
      .reasoning_end = "</think>",
      .text_end = "<|user|>",
      .max_marker_bytes = 15,
      .tool = {.dialect = ToolGrammarDialect::GLM_NATIVE,
               .max_custom_input_bytes = 4}};
  const std::string custom =
      "</think><tool_call>apply_patch<arg_key>patch</arg_key><arg_value>"
      "12345</arg_value></tool_call><|observation|>";
  std::vector<OutputSegment> patch = parse_chunks(
      make_text_reasoning_parser(custom_grammar), {{custom, {}}}, "tool_calls");
  ASSERT_EQ(patch.back().kind, OutputSegmentKind::PARSE_FAILURE);
  EXPECT_EQ(patch.back().failure->code,
            ParseFailureCode::CUSTOM_INPUT_TOO_LARGE);
}

TEST(ModelOutputParserTest, HandlesReasoningTextAndEmptyShapes) {
  std::vector<OutputSegment> reasoning =
      parse_chunks(make_deepseek_v4_profile()->new_parser(),
                   {{"only reasoning", {42}}},
                   "length");
  ASSERT_EQ(reasoning.size(), 1);
  EXPECT_EQ(reasoning[0].kind, OutputSegmentKind::REASONING_DELTA);
  EXPECT_TRUE(reasoning[0].incomplete);

  std::vector<OutputSegment> text =
      parse_chunks(make_glm_5_2_profile()->new_parser(),
                   {{"</think>visible<|user|>", {154842, 100, 154827}}},
                   "stop");
  ASSERT_EQ(text.size(), 3);
  EXPECT_EQ(text[0].kind, OutputSegmentKind::REASONING_DONE);
  EXPECT_EQ(text[1].kind, OutputSegmentKind::TEXT_DELTA);
  EXPECT_EQ(text[1].text, "visible");
  EXPECT_EQ(text[2].kind, OutputSegmentKind::TEXT_DONE);

  std::vector<OutputSegment> empty = parse_chunks(
      make_deepseek_v4_profile()->new_parser(), {{"", {}}}, "stop");
  EXPECT_TRUE(empty.empty());
}

TEST(ModelOutputParserTest, RejectsInvalidUtf8AndUnknownControls) {
  GenerationDelta invalid_utf8{.sequence_index = 0,
                               .generation_ordinal = 0,
                               .text_delta = std::string("\xc3\x28", 2)};
  std::vector<OutputSegment> invalid =
      make_deepseek_v4_profile()->new_parser()->consume(invalid_utf8);
  ASSERT_EQ(invalid.size(), 1);
  ASSERT_EQ(invalid[0].kind, OutputSegmentKind::PARSE_FAILURE);
  EXPECT_EQ(invalid[0].failure->code, ParseFailureCode::INVALID_UTF8);

  GenerationDelta unknown{.sequence_index = 0,
                          .generation_ordinal = 0,
                          .text_delta = "prefix<|unknown_control|>",
                          .finished = true,
                          .finish_reason = "stop"};
  std::vector<OutputSegment> rejected =
      make_glm_5_2_profile()->new_parser()->consume(unknown);
  ASSERT_EQ(rejected.back().kind, OutputSegmentKind::PARSE_FAILURE);
  EXPECT_EQ(rejected.back().failure->code,
            ParseFailureCode::UNKNOWN_CONTROL_TOKEN);

  GenerationDelta unknown_tool_grammar{
      .sequence_index = 0,
      .generation_ordinal = 0,
      .text_delta = "</think><tool_call></tool_call><|observation|>",
      .finished = true,
      .finish_reason = "tool_calls"};
  std::vector<OutputSegment> grammar_rejected =
      make_glm_5_2_profile()->new_parser()->consume(unknown_tool_grammar);
  ASSERT_EQ(grammar_rejected.back().kind, OutputSegmentKind::PARSE_FAILURE);
  EXPECT_EQ(grammar_rejected.back().failure->code,
            ParseFailureCode::UNKNOWN_TOOL_GRAMMAR);
}

TEST(ModelOutputParserTest, TokenBoundariesAreAuthoritative) {
  GenerationDelta spoofed{.sequence_index = 0,
                          .generation_ordinal = 0,
                          .text_delta = "payload</think>",
                          .token_id_delta = {42},
                          .finished = true,
                          .finish_reason = "stop"};
  std::vector<OutputSegment> output =
      make_deepseek_v4_profile()->new_parser()->consume(spoofed);

  ASSERT_EQ(output.back().kind, OutputSegmentKind::PARSE_FAILURE);
  EXPECT_EQ(output.back().failure->code,
            ParseFailureCode::CONTROL_TOKEN_MISMATCH);
}

TEST(ModelOutputParserTest, RejectsSequenceMixingAndBackendErrors) {
  std::unique_ptr<ModelOutputParser> parser =
      make_deepseek_v4_profile()->new_parser();
  std::vector<OutputSegment> first = parser->consume(
      {.sequence_index = 0, .generation_ordinal = 0, .text_delta = "first"});
  ASSERT_EQ(first.size(), 1);
  EXPECT_EQ(first[0].kind, OutputSegmentKind::REASONING_DELTA);
  std::vector<OutputSegment> mixed = parser->consume(
      {.sequence_index = 1, .generation_ordinal = 1, .text_delta = "second"});
  ASSERT_EQ(mixed.size(), 1);
  EXPECT_EQ(mixed[0].failure->code, ParseFailureCode::SEQUENCE_MISMATCH);

  GenerationDelta backend{.sequence_index = 0,
                          .generation_ordinal = 0,
                          .backend_error = BackendError{.code = "worker_failed",
                                                        .message = "failed"}};
  std::vector<OutputSegment> failed =
      make_glm_5_2_profile()->new_parser()->consume(backend);
  ASSERT_EQ(failed.size(), 1);
  EXPECT_EQ(failed[0].failure->code, ParseFailureCode::BACKEND_ERROR);
}

TEST(ModelOutputParserTest, AttributesReasoningFromRawTokenBoundaries) {
  std::unique_ptr<ModelOutputParser> parser =
      make_deepseek_v4_profile()->new_parser();
  parser->consume({.sequence_index = 0,
                   .generation_ordinal = 0,
                   .text_delta = "two tokens</think>",
                   .token_id_delta = {101, 102, 128822}});
  parser->consume({.sequence_index = 0,
                   .generation_ordinal = 1,
                   .text_delta = "answer<｜end▁of▁sentence｜>",
                   .token_id_delta = {103, 1},
                   .finished = true,
                   .finish_reason = "stop"});
  EXPECT_EQ(parser->reasoning_tokens(), 2);
}

TEST(GenerationNormalizerTest, ConvertsCumulativeCallbacksToUniqueDeltas) {
  RequestOutputNormalizer normalizer;
  CumulativeGeneration first{.sequence_index = 0,
                             .generation_ordinal = 0,
                             .text = "alpha",
                             .token_ids = {10}};
  NormalizationResult first_result = normalizer.normalize(first);
  ASSERT_TRUE(first_result.ok());
  EXPECT_EQ(first_result.delta().text_delta, "alpha");
  EXPECT_EQ(first_result.delta().token_id_delta, std::vector<int32_t>({10}));

  CumulativeGeneration second{
      .sequence_index = 0,
      .generation_ordinal = 1,
      .text = "alpha beta",
      .token_ids = {10, 11},
      .finished = true,
      .finish_reason = "stop",
      .final_usage = GenerationUsage{
          .input_tokens = 3, .output_tokens = 2, .total_tokens = 5}};
  NormalizationResult second_result = normalizer.normalize(second);
  ASSERT_TRUE(second_result.ok());
  EXPECT_EQ(second_result.delta().text_delta, " beta");
  EXPECT_EQ(second_result.delta().token_id_delta, std::vector<int32_t>({11}));
  ASSERT_TRUE(second_result.delta().final_usage.has_value());
  EXPECT_EQ(second_result.delta().final_usage->total_tokens, 5);
}

TEST(GenerationNormalizerTest, RejectsRollbackDuplicateAndOrdinalErrors) {
  RequestOutputNormalizer rollback;
  ASSERT_TRUE(rollback
                  .normalize({.sequence_index = 0,
                              .generation_ordinal = 0,
                              .text = "prefix",
                              .token_ids = {1, 2}})
                  .ok());
  NormalizationResult rollback_result =
      rollback.normalize({.sequence_index = 0,
                          .generation_ordinal = 1,
                          .text = "pre",
                          .token_ids = {1}});
  ASSERT_FALSE(rollback_result.ok());
  EXPECT_EQ(rollback_result.failure().code,
            ParseFailureCode::CUMULATIVE_PREFIX_MISMATCH);

  RequestOutputNormalizer duplicate;
  ASSERT_TRUE(duplicate
                  .normalize({.sequence_index = 0,
                              .generation_ordinal = 0,
                              .text = "same",
                              .token_ids = {1}})
                  .ok());
  NormalizationResult duplicate_result =
      duplicate.normalize({.sequence_index = 0,
                           .generation_ordinal = 1,
                           .text = "same",
                           .token_ids = {1}});
  ASSERT_FALSE(duplicate_result.ok());
  EXPECT_EQ(duplicate_result.failure().code,
            ParseFailureCode::DUPLICATE_CALLBACK);

  RequestOutputNormalizer ordinal;
  ASSERT_TRUE(
      ordinal
          .normalize(
              {.sequence_index = 0, .generation_ordinal = 2, .text = "first"})
          .ok());
  NormalizationResult ordinal_result = ordinal.normalize(
      {.sequence_index = 0, .generation_ordinal = 2, .text = "first again"});
  ASSERT_FALSE(ordinal_result.ok());
  EXPECT_EQ(ordinal_result.failure().code, ParseFailureCode::INVALID_ORDINAL);
}

}  // namespace
}  // namespace xllm::model_protocol
