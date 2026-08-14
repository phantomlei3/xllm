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

#include <algorithm>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace xllm::model_protocol {
namespace {

enum class ParseState : uint8_t { REASONING = 0, TEXT = 1, DONE = 2 };

bool valid_utf8(const std::string& text) {
  size_t index = 0;
  while (index < text.size()) {
    const uint8_t first = static_cast<uint8_t>(text[index]);
    if (first <= 0x7f) {
      ++index;
      continue;
    }
    size_t continuation_count = 0;
    uint32_t code_point = 0;
    uint32_t minimum = 0;
    if ((first & 0xe0) == 0xc0) {
      continuation_count = 1;
      code_point = first & 0x1f;
      minimum = 0x80;
    } else if ((first & 0xf0) == 0xe0) {
      continuation_count = 2;
      code_point = first & 0x0f;
      minimum = 0x800;
    } else if ((first & 0xf8) == 0xf0) {
      continuation_count = 3;
      code_point = first & 0x07;
      minimum = 0x10000;
    } else {
      return false;
    }
    if (index + continuation_count >= text.size()) {
      return false;
    }
    for (size_t offset = 1; offset <= continuation_count; ++offset) {
      const uint8_t byte = static_cast<uint8_t>(text[index + offset]);
      if ((byte & 0xc0) != 0x80) {
        return false;
      }
      code_point = (code_point << 6) | (byte & 0x3f);
    }
    if (code_point < minimum || code_point > 0x10ffff ||
        (code_point >= 0xd800 && code_point <= 0xdfff)) {
      return false;
    }
    index += continuation_count + 1;
  }
  return true;
}

bool starts_control(const std::string& text, size_t pos) {
  constexpr std::string_view kPrefixes[] = {
      "<|", "<｜", "</think", "<think", "<tool_", "</tool_", "<arg_", "</arg_"};
  for (std::string_view candidate : kPrefixes) {
    const size_t available = text.size() - pos;
    const size_t length = std::min(available, candidate.size());
    if (text.compare(pos, length, candidate.data(), length) == 0) {
      return true;
    }
  }
  return false;
}

class TextReasoningParser final : public ModelOutputParser {
 public:
  explicit TextReasoningParser(TextReasoningGrammar grammar)
      : grammar_(std::move(grammar)),
        reserved_tokens_(grammar_.reserved_control_tokens.begin(),
                         grammar_.reserved_control_tokens.end()) {}

  std::vector<OutputSegment> consume(const GenerationDelta& delta) override {
    std::vector<OutputSegment> output;
    if (failed_) {
      return output;
    }
    if (done_) {
      return fail(ParseFailureCode::DELTA_AFTER_FINISH,
                  "generation delta arrived after finish");
    }
    if (sequence_initialized_ && delta.sequence_index != sequence_index_) {
      return fail(ParseFailureCode::SEQUENCE_MISMATCH,
                  "parser instance received another sequence");
    }
    if (ordinal_initialized_ && delta.generation_ordinal <= ordinal_) {
      return fail(ParseFailureCode::INVALID_ORDINAL,
                  "generation ordinal is not increasing");
    }
    sequence_index_ = delta.sequence_index;
    sequence_initialized_ = true;
    ordinal_ = delta.generation_ordinal;
    ordinal_initialized_ = true;
    if (delta.backend_error.has_value()) {
      return fail(ParseFailureCode::BACKEND_ERROR,
                  delta.backend_error->message);
    }
    if (!valid_utf8(delta.text_delta)) {
      return fail(ParseFailureCode::INVALID_UTF8,
                  "generation text is not valid UTF-8");
    }
    for (int32_t token_id : delta.token_id_delta) {
      token_boundaries_ = true;
      if (reserved_tokens_.contains(token_id) &&
          token_id != grammar_.reasoning_end_token &&
          token_id != grammar_.text_end_token) {
        return fail(ParseFailureCode::UNKNOWN_CONTROL_TOKEN,
                    "generation contains an unsupported control token");
      }
      if (token_id == grammar_.reasoning_end_token) {
        ++reasoning_markers_;
      } else if (token_id == grammar_.text_end_token) {
        ++text_markers_;
      }
    }

    pending_ += delta.text_delta;
    const bool incomplete = delta.finished && delta.finish_reason.has_value() &&
                            *delta.finish_reason == "length";
    parse_pending(&output, /*flush=*/delta.finished, incomplete);
    if (failed_) {
      output.emplace_back(failure_segment_);
      return output;
    }
    if (delta.finished) {
      if (token_boundaries_ &&
          (reasoning_markers_ != 0 || text_markers_ != 0)) {
        output.emplace_back(fail(ParseFailureCode::CONTROL_TOKEN_MISMATCH,
                                 "control token has no matching text marker")
                                .front());
        return output;
      }
      if (!pending_.empty()) {
        emit_payload(&output, pending_, incomplete);
        pending_.clear();
      }
      done_ = true;
    }
    return output;
  }

 private:
  std::vector<OutputSegment> fail(ParseFailureCode code,
                                  const std::string& message) {
    failed_ = true;
    failure_segment_ = {
        .kind = OutputSegmentKind::PARSE_FAILURE,
        .failure = ParseFailure{.code = code, .message = message}};
    return {failure_segment_};
  }

  void set_failure(ParseFailureCode code, const std::string& message) {
    failed_ = true;
    failure_segment_ = {
        .kind = OutputSegmentKind::PARSE_FAILURE,
        .failure = ParseFailure{.code = code, .message = message}};
  }

  void emit_payload(std::vector<OutputSegment>* output,
                    const std::string& text,
                    bool incomplete) const {
    if (text.empty()) {
      return;
    }
    const OutputSegmentKind kind = state_ == ParseState::REASONING
                                       ? OutputSegmentKind::REASONING_DELTA
                                       : OutputSegmentKind::TEXT_DELTA;
    output->emplace_back(OutputSegment{
        .kind = kind, .raw = text, .text = text, .incomplete = incomplete});
  }

  size_t ambiguous_suffix() const {
    const size_t marker_bytes = lookbehind_size();
    const size_t lower =
        pending_.size() > marker_bytes ? pending_.size() - marker_bytes : 0;
    for (size_t pos = lower; pos < pending_.size(); ++pos) {
      if (starts_control(pending_, pos)) {
        return pos;
      }
    }
    return pending_.size();
  }

  bool reject_unknown_control() {
    for (size_t pos = 0; pos < pending_.size(); ++pos) {
      if (!starts_control(pending_, pos)) {
        continue;
      }
      const size_t close = pending_.find('>', pos);
      if (close == std::string::npos) {
        if (pending_.size() - pos >= lookbehind_size()) {
          set_failure(ParseFailureCode::UNKNOWN_CONTROL_TOKEN,
                      "control marker exceeds the profile lookbehind");
          return true;
        }
        return false;
      }
      set_failure(ParseFailureCode::UNKNOWN_CONTROL_TOKEN,
                  "generation contains an unsupported control marker");
      return true;
    }
    return false;
  }

  size_t lookbehind_size() const {
    return std::max({grammar_.max_marker_bytes,
                     grammar_.reasoning_end.size(),
                     grammar_.text_end.size()});
  }

  void parse_pending(std::vector<OutputSegment>* output,
                     bool flush,
                     bool incomplete) {
    while (!pending_.empty()) {
      const std::string& marker = state_ == ParseState::REASONING
                                      ? grammar_.reasoning_end
                                      : grammar_.text_end;
      const size_t marker_pos = pending_.find(marker);
      if (marker_pos != std::string::npos) {
        size_t& marker_count = state_ == ParseState::REASONING
                                   ? reasoning_markers_
                                   : text_markers_;
        if (token_boundaries_ && marker_count == 0) {
          set_failure(ParseFailureCode::CONTROL_TOKEN_MISMATCH,
                      "text marker has no matching control token");
          return;
        }
        if (token_boundaries_) {
          --marker_count;
        }
        emit_payload(
            output, pending_.substr(0, marker_pos), /*incomplete=*/false);
        output->emplace_back(
            OutputSegment{.kind = state_ == ParseState::REASONING
                                      ? OutputSegmentKind::REASONING_DONE
                                      : OutputSegmentKind::TEXT_DONE,
                          .raw = marker});
        pending_.erase(0, marker_pos + marker.size());
        if (state_ == ParseState::REASONING) {
          state_ = ParseState::TEXT;
        } else {
          state_ = ParseState::DONE;
          if (!pending_.empty()) {
            set_failure(ParseFailureCode::UNKNOWN_CONTROL_TOKEN,
                        "generation has data after its terminal marker");
          }
          return;
        }
        continue;
      }
      if (reject_unknown_control()) {
        return;
      }
      const size_t safe_size = flush ? pending_.size() : ambiguous_suffix();
      if (safe_size == 0) {
        return;
      }
      emit_payload(output, pending_.substr(0, safe_size), incomplete);
      pending_.erase(0, safe_size);
      if (!flush) {
        return;
      }
    }
  }

  TextReasoningGrammar grammar_;
  std::unordered_set<int32_t> reserved_tokens_;
  ParseState state_ = ParseState::REASONING;
  std::string pending_;
  size_t sequence_index_ = 0;
  uint64_t ordinal_ = 0;
  bool sequence_initialized_ = false;
  bool ordinal_initialized_ = false;
  bool done_ = false;
  bool failed_ = false;
  bool token_boundaries_ = false;
  size_t reasoning_markers_ = 0;
  size_t text_markers_ = 0;
  OutputSegment failure_segment_;
};

}  // namespace

std::unique_ptr<ModelOutputParser> make_text_reasoning_parser(
    TextReasoningGrammar grammar) {
  return std::make_unique<TextReasoningParser>(std::move(grammar));
}

}  // namespace xllm::model_protocol
