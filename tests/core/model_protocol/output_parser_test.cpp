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

RawFixture load_raw(const FixtureCatalog& catalog, const std::string& profile) {
  const nlohmann::json& raw = catalog.expected(
      profile, "reasoning_text_stop", ExpectedArtifact::RAW_GENERATION);
  return {.text = raw.at("text").get<std::string>(),
          .token_ids = raw.at("token_ids").get<std::vector<int32_t>>(),
          .token_text = raw.at("token_text").get<std::vector<std::string>>(),
          .finish_reason = raw.at("finish_reason").get<std::string>()};
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

  GenerationDelta unknown_token{.sequence_index = 0,
                                .generation_ordinal = 0,
                                .text_delta = "<tool_call>",
                                .token_id_delta = {154843}};
  std::vector<OutputSegment> token_rejected =
      make_glm_5_2_profile()->new_parser()->consume(unknown_token);
  ASSERT_EQ(token_rejected.size(), 1);
  EXPECT_EQ(token_rejected[0].failure->code,
            ParseFailureCode::UNKNOWN_CONTROL_TOKEN);
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
