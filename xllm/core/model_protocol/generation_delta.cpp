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

#include "core/model_protocol/generation_delta.h"

#include <algorithm>
#include <iterator>
#include <utility>

namespace xllm::model_protocol {
namespace {

ParseFailure failure(ParseFailureCode code, std::string message) {
  return {.code = code, .message = std::move(message)};
}

}  // namespace

NormalizationResult::NormalizationResult(GenerationDelta delta)
    : delta_(std::move(delta)) {}

NormalizationResult::NormalizationResult(ParseFailure failure)
    : failure_(std::move(failure)) {}

NormalizationResult RequestOutputNormalizer::normalize(
    const CumulativeGeneration& output) {
  SequenceState& state = sequences_[output.sequence_index];
  if (state.finished) {
    return NormalizationResult(failure(ParseFailureCode::DELTA_AFTER_FINISH,
                                       "callback arrived after finish"));
  }
  if (state.initialized && output.generation_ordinal <= state.ordinal) {
    return NormalizationResult(failure(ParseFailureCode::INVALID_ORDINAL,
                                       "generation ordinal is not increasing"));
  }
  if (output.text.size() < state.text.size() ||
      output.text.compare(/*pos=*/0, state.text.size(), state.text) != 0 ||
      output.token_ids.size() < state.token_ids.size() ||
      !std::equal(state.token_ids.begin(),
                  state.token_ids.end(),
                  output.token_ids.begin())) {
    return NormalizationResult(
        failure(ParseFailureCode::CUMULATIVE_PREFIX_MISMATCH,
                "cumulative generation does not extend the previous callback"));
  }

  const std::string text_delta = output.text.substr(state.text.size());
  std::vector<int32_t> token_delta;
  token_delta.reserve(output.token_ids.size() - state.token_ids.size());
  std::copy(output.token_ids.begin() +
                static_cast<std::ptrdiff_t>(state.token_ids.size()),
            output.token_ids.end(),
            std::back_inserter(token_delta));
  if (state.initialized && text_delta.empty() && token_delta.empty() &&
      !output.finished && !output.backend_error.has_value() &&
      !output.final_usage.has_value()) {
    return NormalizationResult(failure(ParseFailureCode::DUPLICATE_CALLBACK,
                                       "callback has no new generation data"));
  }

  state.ordinal = output.generation_ordinal;
  state.text = output.text;
  state.token_ids = output.token_ids;
  state.initialized = true;
  state.finished = output.finished || output.backend_error.has_value();
  return NormalizationResult(
      GenerationDelta{.sequence_index = output.sequence_index,
                      .generation_ordinal = output.generation_ordinal,
                      .text_delta = text_delta,
                      .token_id_delta = std::move(token_delta),
                      .finished = output.finished,
                      .finish_reason = output.finish_reason,
                      .final_usage = output.final_usage,
                      .backend_error = output.backend_error});
}

}  // namespace xllm::model_protocol
