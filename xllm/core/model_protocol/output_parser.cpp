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
#include <cctype>
#include <initializer_list>
#include <nlohmann/json.hpp>
#include <string_view>
#include <utility>

namespace xllm::model_protocol {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kDsCallsOpen = "<｜DSML｜tool_calls>";
constexpr std::string_view kDsCallsClose = "</｜DSML｜tool_calls>";
constexpr std::string_view kDsInvokeOpen = "<｜DSML｜invoke";
constexpr std::string_view kDsInvokeClose = "</｜DSML｜invoke>";
constexpr std::string_view kDsParamOpen = "<｜DSML｜parameter";
constexpr std::string_view kDsParamClose = "</｜DSML｜parameter>";
constexpr std::string_view kGlmCallOpen = "<tool_call>";
constexpr std::string_view kGlmCallClose = "</tool_call>";
constexpr std::string_view kGlmKeyOpen = "<arg_key>";
constexpr std::string_view kGlmKeyClose = "</arg_key>";
constexpr std::string_view kGlmValueOpen = "<arg_value>";
constexpr std::string_view kGlmValueClose = "</arg_value>";
constexpr std::string_view kGlmCallsDone = "<|observation|>";

enum class ParseState : uint8_t {
  REASONING = 0,
  TEXT = 1,
  TOOL = 2,
  DONE = 3,
};

enum class MarkerKind : uint8_t {
  REASONING_END = 0,
  TEXT_END = 1,
  TOOL_OPEN = 2,
  TOOL_DONE = 3,
};

bool utf8_prefix(const std::string& text, bool final, size_t* complete) {
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
      if (final) {
        return false;
      }
      *complete = index;
      return true;
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
  *complete = text.size();
  return true;
}

bool whitespace_only(std::string_view text) {
  return std::all_of(text.begin(), text.end(), [](char value) {
    return std::isspace(static_cast<unsigned char>(value)) != 0;
  });
}

std::string trim(std::string_view text) {
  size_t begin = 0;
  while (begin < text.size() &&
         std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
    ++begin;
  }
  size_t end = text.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
    --end;
  }
  return std::string(text.substr(begin, end - begin));
}

bool valid_call_id(const std::string& call_id) {
  if (call_id.empty() || call_id.size() > 128) {
    return false;
  }
  return std::all_of(call_id.begin(), call_id.end(), [](char value) {
    const unsigned char byte = static_cast<unsigned char>(value);
    return std::isalnum(byte) != 0 || value == '_' || value == '-';
  });
}

size_t json_depth(const Json& value) {
  if (!value.is_array() && !value.is_object()) {
    return 1;
  }
  size_t depth = 1;
  for (const auto& child : value) {
    depth = std::max(depth, 1 + json_depth(child));
  }
  return depth;
}

bool structured_prefix(const std::string& value) {
  const std::string cleaned = trim(value);
  if (cleaned.empty()) {
    return false;
  }
  const char first = cleaned.front();
  return first == '{' || first == '[' || first == '"';
}

bool parse_value(const std::string& value, Json* output) {
  try {
    *output = Json::parse(value);
    return true;
  } catch (const Json::exception&) {
    if (structured_prefix(value)) {
      return false;
    }
    *output = value;
    return true;
  }
}

std::optional<std::string> attribute(std::string_view tag,
                                     std::string_view name) {
  const std::string prefix = std::string(name) + "=\"";
  const size_t begin = tag.find(prefix);
  if (begin == std::string_view::npos) {
    return std::nullopt;
  }
  const size_t value_begin = begin + prefix.size();
  const size_t end = tag.find('"', value_begin);
  if (end == std::string_view::npos) {
    return std::nullopt;
  }
  return std::string(tag.substr(value_begin, end - value_begin));
}

size_t boundary_close(std::string_view text,
                      std::string_view marker,
                      size_t start,
                      std::initializer_list<std::string_view> continuations,
                      bool allow_end) {
  size_t pos = text.find(marker, start);
  while (pos != std::string_view::npos) {
    size_t next = pos + marker.size();
    while (next < text.size() &&
           std::isspace(static_cast<unsigned char>(text[next])) != 0) {
      ++next;
    }
    if (allow_end && next == text.size()) {
      return pos;
    }
    for (std::string_view continuation : continuations) {
      if (text.substr(next).starts_with(continuation)) {
        return pos;
      }
    }
    pos = text.find(marker, pos + marker.size());
  }
  return std::string_view::npos;
}

class TextReasoningParser final : public ModelOutputParser {
 public:
  explicit TextReasoningParser(TextReasoningGrammar grammar)
      : grammar_(std::move(grammar)) {}

  std::vector<OutputSegment> consume(const GenerationDelta& delta) override {
    std::vector<OutputSegment> output;
    if (failed_) {
      return output;
    }
    if (finalized_) {
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
    utf8_pending_ += delta.text_delta;
    size_t complete_bytes = 0;
    if (!utf8_prefix(utf8_pending_, /*final=*/false, &complete_bytes)) {
      return fail(ParseFailureCode::INVALID_UTF8,
                  "generation text is not valid UTF-8");
    }
    inspect_tokens(delta.token_id_delta);
    if (failed_) {
      return {failure_segment_};
    }

    pending_ += utf8_pending_.substr(0, complete_bytes);
    utf8_pending_.erase(0, complete_bytes);
    parse_pending(&output, /*flush=*/false, /*incomplete=*/false);
    if (failed_) {
      output.emplace_back(failure_segment_);
    }
    return output;
  }

  std::vector<OutputSegment> finalize(ParserTerminalReason reason) override {
    std::vector<OutputSegment> output;
    if (failed_) {
      return output;
    }
    if (finalized_) {
      return fail(ParseFailureCode::DELTA_AFTER_FINISH,
                  "parser terminal arrived more than once");
    }
    finalized_ = true;
    size_t complete_bytes = 0;
    const bool incomplete = reason == ParserTerminalReason::TOKEN_LIMIT;
    if (!utf8_prefix(utf8_pending_, /*final=*/!incomplete, &complete_bytes)) {
      return fail(ParseFailureCode::INVALID_UTF8,
                  "generation text is not valid UTF-8");
    }
    pending_ += utf8_pending_.substr(0, complete_bytes);
    utf8_pending_.erase(0, complete_bytes);
    parse_pending(&output, /*flush=*/true, incomplete);
    if (!failed_) {
      finish(&output, incomplete);
    } else {
      output.emplace_back(failure_segment_);
    }
    return output;
  }

  std::optional<int32_t> reasoning_tokens() const override {
    if (!token_boundaries_) {
      return std::nullopt;
    }
    return reasoning_tokens_;
  }

 private:
  std::vector<OutputSegment> fail(ParseFailureCode code,
                                  const std::string& message) {
    set_failure(code, message);
    return {failure_segment_};
  }

  void set_failure(ParseFailureCode code, const std::string& message) {
    failed_ = true;
    failure_segment_ = {
        .kind = OutputSegmentKind::PARSE_FAILURE,
        .failure = ParseFailure{.code = code, .message = message}};
  }

  bool ends_with(const std::vector<int32_t>& suffix) const {
    return suffix.size() <= token_tail_.size() &&
           std::equal(suffix.rbegin(), suffix.rend(), token_tail_.rbegin());
  }

  void record_marker(MarkerKind kind, const std::vector<int32_t>& sequence) {
    token_markers_.emplace_back(kind);
    if (kind == MarkerKind::REASONING_END) {
      reasoning_tokens_done_ = true;
      reasoning_tokens_ -= static_cast<int32_t>(sequence.size());
    }
    token_tail_.clear();
  }

  void inspect_tokens(const std::vector<int32_t>& tokens) {
    const size_t max_sequence = std::max({grammar_.reasoning_end_tokens.size(),
                                          grammar_.text_end_tokens.size(),
                                          grammar_.tool_open_tokens.size(),
                                          grammar_.tool_done_tokens.size(),
                                          size_t{1}});
    for (int32_t token_id : tokens) {
      token_boundaries_ = true;
      token_tail_.emplace_back(token_id);
      if (!reasoning_tokens_done_) {
        ++reasoning_tokens_;
      }
      if (!grammar_.tool_done_tokens.empty() &&
          ends_with(grammar_.tool_done_tokens)) {
        record_marker(MarkerKind::TOOL_DONE, grammar_.tool_done_tokens);
      } else if (!grammar_.tool_open_tokens.empty() &&
                 ends_with(grammar_.tool_open_tokens)) {
        record_marker(MarkerKind::TOOL_OPEN, grammar_.tool_open_tokens);
      } else if (!grammar_.reasoning_end_tokens.empty() &&
                 ends_with(grammar_.reasoning_end_tokens)) {
        record_marker(MarkerKind::REASONING_END, grammar_.reasoning_end_tokens);
      } else if (!grammar_.text_end_tokens.empty() &&
                 ends_with(grammar_.text_end_tokens)) {
        record_marker(MarkerKind::TEXT_END, grammar_.text_end_tokens);
      } else if (token_tail_.size() >= max_sequence) {
        token_tail_.erase(token_tail_.begin());
      }
    }
  }

  bool accept_marker(MarkerKind expected, bool allow_text_only) {
    if (!token_boundaries_) {
      return allow_text_only;
    }
    if (token_markers_.empty()) {
      if (finalized_) {
        set_failure(ParseFailureCode::CONTROL_TOKEN_MISMATCH,
                    "text marker has no matching token sequence");
      }
      return false;
    }
    if (token_markers_.front() != expected) {
      set_failure(ParseFailureCode::CONTROL_TOKEN_MISMATCH,
                  "text and token control markers are out of order");
      return false;
    }
    token_markers_.erase(token_markers_.begin());
    return true;
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

  std::string tool_open() const {
    if (grammar_.tool.dialect == ToolGrammarDialect::DEEPSEEK_DSML) {
      return std::string(kDsCallsOpen);
    }
    if (grammar_.tool.dialect == ToolGrammarDialect::GLM_NATIVE) {
      return std::string(kGlmCallOpen);
    }
    return "";
  }

  std::string tool_done() const {
    if (grammar_.tool.dialect == ToolGrammarDialect::GLM_NATIVE) {
      return std::string(kGlmCallsDone);
    }
    return grammar_.text_end;
  }

  std::string marker_text(MarkerKind kind) const {
    switch (kind) {
      case MarkerKind::REASONING_END:
        return grammar_.reasoning_end;
      case MarkerKind::TEXT_END:
        return grammar_.text_end;
      case MarkerKind::TOOL_OPEN:
        return tool_open();
      case MarkerKind::TOOL_DONE:
        return tool_done();
    }
    return "";
  }

  std::vector<size_t> anchor_positions() const {
    std::vector<size_t> positions(token_markers_.size(), std::string::npos);
    size_t search_end = pending_.size();
    // Newest anchors claim the rightmost matching marker text.
    for (size_t index = token_markers_.size(); index > 0; --index) {
      const std::string marker = marker_text(token_markers_[index - 1]);
      if (marker.empty() || marker.size() > search_end) {
        return {};
      }
      const size_t pos = pending_.rfind(marker, search_end - marker.size());
      if (pos == std::string::npos) {
        return {};
      }
      positions[index - 1] = pos;
      search_end = pos;
    }
    return positions;
  }

  size_t anchored_pos() const {
    const std::vector<size_t> positions = anchor_positions();
    return positions.empty() ? std::string::npos : positions.front();
  }

  size_t unresolved_pos() const {
    size_t first_pos = std::string::npos;
    for (MarkerKind kind : {MarkerKind::REASONING_END,
                            MarkerKind::TEXT_END,
                            MarkerKind::TOOL_OPEN,
                            MarkerKind::TOOL_DONE}) {
      const std::string marker = marker_text(kind);
      if (!marker.empty()) {
        first_pos = std::min(first_pos, pending_.find(marker));
      }
    }
    return first_pos;
  }

  size_t ambiguous_suffix() const {
    const size_t marker_bytes = std::max({grammar_.max_marker_bytes,
                                          grammar_.reasoning_end.size(),
                                          grammar_.text_end.size(),
                                          tool_open().size(),
                                          tool_done().size()});
    const size_t lower =
        pending_.size() > marker_bytes ? pending_.size() - marker_bytes : 0;
    for (size_t pos = lower; pos < pending_.size(); ++pos) {
      if (pending_[pos] != '<') {
        continue;
      }
      return pos;
    }
    return pending_.size();
  }

  bool consume_marker(size_t pos,
                      const std::string& marker,
                      MarkerKind marker_kind,
                      bool allow_text_only,
                      std::vector<OutputSegment>* output) {
    if (!accept_marker(marker_kind, allow_text_only)) {
      return false;
    }
    emit_payload(output, pending_.substr(0, pos), /*incomplete=*/false);
    output->emplace_back(
        OutputSegment{.kind = state_ == ParseState::REASONING
                                  ? OutputSegmentKind::REASONING_DONE
                                  : OutputSegmentKind::TEXT_DONE,
                      .raw = marker});
    pending_.erase(0, pos + marker.size());
    state_ =
        state_ == ParseState::REASONING ? ParseState::TEXT : ParseState::DONE;
    return true;
  }

  bool emit_call(const std::string& name,
                 const std::optional<std::string>& call_id,
                 const Json& arguments,
                 const std::string& custom_input,
                 std::vector<OutputSegment>* output) {
    if (call_id.has_value() && !valid_call_id(*call_id)) {
      set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                  "model call ID is invalid");
      return false;
    }
    if (name == "apply_patch") {
      if (custom_input.size() > grammar_.tool.max_custom_input_bytes) {
        set_failure(ParseFailureCode::CUSTOM_INPUT_TOO_LARGE,
                    "apply_patch input exceeds the profile limit");
        return false;
      }
      output->emplace_back(
          OutputSegment{.kind = OutputSegmentKind::CUSTOM_CALL_START,
                        .name = name,
                        .call_id = call_id});
      if (!custom_input.empty()) {
        output->emplace_back(
            OutputSegment{.kind = OutputSegmentKind::CUSTOM_INPUT_DELTA,
                          .raw = custom_input,
                          .text = custom_input});
      }
      output->emplace_back(OutputSegment{
          .kind = OutputSegmentKind::CUSTOM_CALL_DONE, .name = name});
      return true;
    }
    if (!custom_input.empty()) {
      set_failure(ParseFailureCode::UNKNOWN_CUSTOM_TOOL,
                  "only apply_patch may use raw custom input");
      return false;
    }
    if (!arguments.is_object() ||
        json_depth(arguments) > grammar_.tool.max_json_depth) {
      set_failure(ParseFailureCode::INVALID_TOOL_ARGUMENTS,
                  "function arguments must be a bounded JSON object");
      return false;
    }
    const std::string encoded = arguments.dump();
    if (encoded.size() > grammar_.tool.max_arguments_bytes) {
      set_failure(ParseFailureCode::TOOL_ARGUMENTS_TOO_LARGE,
                  "function arguments exceed the profile limit");
      return false;
    }
    output->emplace_back(
        OutputSegment{.kind = OutputSegmentKind::FUNCTION_CALL_START,
                      .name = name,
                      .call_id = call_id});
    output->emplace_back(OutputSegment{
        .kind = OutputSegmentKind::ARGUMENTS_DELTA, .text = encoded});
    output->emplace_back(OutputSegment{
        .kind = OutputSegmentKind::FUNCTION_CALL_DONE, .name = name});
    return true;
  }

  bool parse_dsml(std::string_view body, std::vector<OutputSegment>* output) {
    if (!body.starts_with(kDsCallsOpen) || !body.ends_with(kDsCallsClose)) {
      set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                  "invalid DeepSeek tool call envelope");
      return false;
    }
    size_t pos = kDsCallsOpen.size();
    while (pos < body.size() - kDsCallsClose.size()) {
      while (pos < body.size() &&
             std::isspace(static_cast<unsigned char>(body[pos])) != 0) {
        ++pos;
      }
      if (body.substr(pos).starts_with(kDsCallsClose)) {
        break;
      }
      if (!body.substr(pos).starts_with(kDsInvokeOpen)) {
        set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                    "invalid DeepSeek invocation marker");
        return false;
      }
      const size_t tag_end = body.find('>', pos);
      const size_t call_end = boundary_close(body,
                                             kDsInvokeClose,
                                             tag_end,
                                             {kDsInvokeOpen, kDsCallsClose},
                                             /*allow_end=*/false);
      if (tag_end == std::string_view::npos ||
          call_end == std::string_view::npos) {
        set_failure(ParseFailureCode::UNCLOSED_TOOL_CALL,
                    "DeepSeek tool call is not closed");
        return false;
      }
      const std::string_view invoke_tag = body.substr(pos, tag_end - pos + 1);
      const std::optional<std::string> name = attribute(invoke_tag, "name");
      const std::optional<std::string> call_id = attribute(invoke_tag, "id");
      const std::optional<std::string> call_type =
          attribute(invoke_tag, "type");
      if (!name.has_value() || name->empty()) {
        set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                    "DeepSeek tool call has no name");
        return false;
      }
      if (call_type.has_value() && *call_type != "function" &&
          *call_type != "custom") {
        set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                    "DeepSeek tool call type is invalid");
        return false;
      }
      if (call_type == "custom" && *name != "apply_patch") {
        set_failure(ParseFailureCode::UNKNOWN_CUSTOM_TOOL,
                    "only apply_patch is a supported custom tool");
        return false;
      }
      if (call_type == "function" && *name == "apply_patch") {
        set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                    "apply_patch cannot be classified as a function");
        return false;
      }
      Json arguments = Json::object();
      std::string patch;
      size_t param_pos = tag_end + 1;
      while (param_pos < call_end) {
        while (param_pos < call_end &&
               std::isspace(static_cast<unsigned char>(body[param_pos])) != 0) {
          ++param_pos;
        }
        if (param_pos == call_end) {
          break;
        }
        if (!body.substr(param_pos).starts_with(kDsParamOpen)) {
          set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                      "invalid DeepSeek parameter marker");
          return false;
        }
        const size_t param_tag_end = body.find('>', param_pos);
        const size_t param_end = boundary_close(body,
                                                kDsParamClose,
                                                param_tag_end,
                                                {kDsParamOpen, kDsInvokeClose},
                                                /*allow_end=*/false);
        if (param_tag_end == std::string_view::npos ||
            param_end == std::string_view::npos || param_end > call_end) {
          set_failure(ParseFailureCode::UNCLOSED_TOOL_CALL,
                      "DeepSeek parameter is not closed");
          return false;
        }
        const std::string_view param_tag =
            body.substr(param_pos, param_tag_end - param_pos + 1);
        const std::optional<std::string> key = attribute(param_tag, "name");
        const std::optional<std::string> string_type =
            attribute(param_tag, "string");
        if (!key.has_value() || !string_type.has_value() || key->empty()) {
          set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                      "DeepSeek parameter attributes are invalid");
          return false;
        }
        const std::string value(
            body.substr(param_tag_end + 1, param_end - param_tag_end - 1));
        if (*name == "apply_patch") {
          if (*key != "patch" || !patch.empty() || *string_type != "true") {
            set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                        "apply_patch requires one raw patch parameter");
            return false;
          }
          patch = value;
        } else if (*string_type == "true") {
          arguments[*key] = value;
        } else if (*string_type == "false") {
          Json parsed;
          try {
            parsed = Json::parse(value);
          } catch (const Json::exception&) {
            set_failure(ParseFailureCode::INVALID_TOOL_ARGUMENTS,
                        "DeepSeek function argument is invalid JSON");
            return false;
          }
          arguments[*key] = std::move(parsed);
        } else {
          set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                      "DeepSeek parameter string flag is invalid");
          return false;
        }
        param_pos = param_end + kDsParamClose.size();
      }
      if (!emit_call(*name, call_id, arguments, patch, output)) {
        return false;
      }
      pos = call_end + kDsInvokeClose.size();
    }
    return true;
  }

  bool parse_glm(std::string_view body, std::vector<OutputSegment>* output) {
    size_t pos = 0;
    while (pos < body.size()) {
      if (!body.substr(pos).starts_with(kGlmCallOpen)) {
        set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                    "invalid GLM tool call marker");
        return false;
      }
      const size_t call_end = boundary_close(body,
                                             kGlmCallClose,
                                             pos + kGlmCallOpen.size(),
                                             {kGlmCallOpen},
                                             /*allow_end=*/true);
      if (call_end == std::string_view::npos) {
        set_failure(ParseFailureCode::UNCLOSED_TOOL_CALL,
                    "GLM tool call is not closed");
        return false;
      }
      const size_t name_begin = pos + kGlmCallOpen.size();
      size_t field_pos = body.find(kGlmKeyOpen, name_begin);
      if (field_pos == std::string_view::npos || field_pos > call_end) {
        field_pos = call_end;
      }
      const std::string name =
          trim(body.substr(name_begin, field_pos - name_begin));
      if (name.empty()) {
        set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                    "GLM tool call has no name");
        return false;
      }
      Json arguments = Json::object();
      std::string patch;
      while (field_pos < call_end) {
        const size_t key_end = body.find(kGlmKeyClose, field_pos);
        if (key_end == std::string_view::npos || key_end > call_end) {
          set_failure(ParseFailureCode::UNCLOSED_TOOL_CALL,
                      "GLM argument key is not closed");
          return false;
        }
        const std::string key =
            trim(body.substr(field_pos + kGlmKeyOpen.size(),
                             key_end - field_pos - kGlmKeyOpen.size()));
        const size_t value_open = key_end + kGlmKeyClose.size();
        if (!body.substr(value_open).starts_with(kGlmValueOpen)) {
          set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                      "GLM argument value marker is missing");
          return false;
        }
        const size_t value_begin = value_open + kGlmValueOpen.size();
        const size_t value_end = boundary_close(body,
                                                kGlmValueClose,
                                                value_begin,
                                                {kGlmKeyOpen, kGlmCallClose},
                                                /*allow_end=*/false);
        if (value_end == std::string_view::npos || value_end > call_end) {
          set_failure(ParseFailureCode::UNCLOSED_TOOL_CALL,
                      "GLM argument value is not closed");
          return false;
        }
        const std::string value(
            body.substr(value_begin, value_end - value_begin));
        if (name == "apply_patch") {
          if (key != "patch" || !patch.empty()) {
            set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                        "apply_patch requires one raw patch parameter");
            return false;
          }
          patch = value;
        } else {
          Json parsed;
          if (!parse_value(value, &parsed)) {
            set_failure(ParseFailureCode::INVALID_TOOL_ARGUMENTS,
                        "GLM function argument is invalid JSON");
            return false;
          }
          arguments[key] = std::move(parsed);
        }
        field_pos = value_end + kGlmValueClose.size();
      }
      if (!emit_call(name, std::nullopt, arguments, patch, output)) {
        return false;
      }
      pos = call_end + kGlmCallClose.size();
    }
    return true;
  }

  bool parse_tools(std::vector<OutputSegment>* output) {
    const std::string done_marker = tool_done();
    size_t done_pos = pending_.find(done_marker);
    size_t done_index = 0;
    if (token_boundaries_) {
      const auto done = std::find(
          token_markers_.begin(), token_markers_.end(), MarkerKind::TOOL_DONE);
      const std::vector<size_t> positions = anchor_positions();
      if (done == token_markers_.end() || positions.empty()) {
        done_pos = std::string::npos;
      } else {
        done_index =
            static_cast<size_t>(std::distance(token_markers_.begin(), done));
        if (std::any_of(token_markers_.begin(), done, [](MarkerKind kind) {
              return kind != MarkerKind::TOOL_OPEN;
            })) {
          set_failure(ParseFailureCode::CONTROL_TOKEN_MISMATCH,
                      "tool control markers are out of order");
          return false;
        }
        done_pos = positions[done_index];
      }
    }
    if (done_pos == std::string::npos) {
      return false;
    }
    if (token_boundaries_) {
      token_markers_.erase(token_markers_.begin(),
                           token_markers_.begin() + done_index + 1);
    } else if (!accept_marker(MarkerKind::TOOL_DONE,
                              finalized_ && !token_boundaries_)) {
      return false;
    }
    const std::string body = pending_.substr(0, done_pos);
    // Validate the complete block before exposing any call lifecycle.
    std::vector<OutputSegment> parsed_output;
    bool parsed = false;
    if (grammar_.tool.dialect == ToolGrammarDialect::DEEPSEEK_DSML) {
      parsed = parse_dsml(body, &parsed_output);
    } else if (grammar_.tool.dialect == ToolGrammarDialect::GLM_NATIVE) {
      parsed = parse_glm(body, &parsed_output);
    }
    if (!parsed) {
      return false;
    }
    for (OutputSegment& segment : parsed_output) {
      output->emplace_back(std::move(segment));
    }
    pending_.erase(0, done_pos + done_marker.size());
    state_ = ParseState::DONE;
    if (!pending_.empty()) {
      set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                  "generation has data after tool terminal");
      return false;
    }
    return true;
  }

  void parse_pending(std::vector<OutputSegment>* output,
                     bool flush,
                     bool incomplete) {
    while (!pending_.empty() && !failed_) {
      if (state_ == ParseState::TOOL) {
        if (pending_.size() > grammar_.tool.max_tool_block_bytes) {
          set_failure(ParseFailureCode::TOOL_BLOCK_TOO_LARGE,
                      "tool block exceeds the profile limit");
          return;
        }
        parse_tools(output);
        return;
      }
      if (state_ == ParseState::DONE) {
        set_failure(ParseFailureCode::UNKNOWN_TOOL_GRAMMAR,
                    "generation has data after its terminal marker");
        return;
      }
      const std::string marker = state_ == ParseState::REASONING
                                     ? grammar_.reasoning_end
                                     : grammar_.text_end;
      size_t marker_pos = pending_.find(marker);
      size_t tool_pos = std::string::npos;
      if (token_boundaries_) {
        marker_pos = std::string::npos;
        if (!token_markers_.empty()) {
          const MarkerKind next = token_markers_.front();
          const size_t anchor_pos = anchored_pos();
          if ((state_ == ParseState::REASONING &&
               next == MarkerKind::REASONING_END) ||
              (state_ == ParseState::TEXT && next == MarkerKind::TEXT_END)) {
            marker_pos = anchor_pos;
          } else if (state_ == ParseState::TEXT &&
                     next == MarkerKind::TOOL_OPEN) {
            tool_pos = anchor_pos;
          }
        }
      }
      if (state_ == ParseState::TEXT &&
          grammar_.tool.dialect != ToolGrammarDialect::NONE) {
        const std::string open = tool_open();
        if (!token_boundaries_) {
          tool_pos = pending_.find(open);
        }
        if (tool_pos != std::string::npos &&
            (marker_pos == std::string::npos || tool_pos < marker_pos)) {
          if (!accept_marker(MarkerKind::TOOL_OPEN,
                             flush && !token_boundaries_)) {
            return;
          }
          const std::string prefix = pending_.substr(0, tool_pos);
          const bool has_text = text_started_ || !whitespace_only(prefix);
          if (!prefix.empty() && !whitespace_only(prefix)) {
            emit_payload(output, prefix, /*incomplete=*/false);
          }
          if (has_text) {
            output->emplace_back(
                OutputSegment{.kind = OutputSegmentKind::TEXT_DONE});
          }
          pending_.erase(0, tool_pos);
          state_ = ParseState::TOOL;
          continue;
        }
      }
      if (marker_pos != std::string::npos) {
        const MarkerKind marker_kind = state_ == ParseState::REASONING
                                           ? MarkerKind::REASONING_END
                                           : MarkerKind::TEXT_END;
        if (!consume_marker(marker_pos,
                            marker,
                            marker_kind,
                            flush && !token_boundaries_,
                            output)) {
          return;
        }
        text_started_ = false;
        continue;
      }
      if (state_ == ParseState::TEXT &&
          grammar_.tool.dialect != ToolGrammarDialect::NONE) {
        if (!text_started_ && whitespace_only(pending_) && !flush) {
          return;
        }
        if (!text_started_) {
          size_t first = 0;
          while (first < pending_.size() &&
                 std::isspace(static_cast<unsigned char>(pending_[first])) !=
                     0) {
            ++first;
          }
          const std::string candidate = pending_.substr(first);
          const std::string open = tool_open();
          if (!candidate.empty() && candidate.size() < open.size() &&
              open.starts_with(candidate)) {
            return;
          }
        }
      }
      size_t safe_size = flush ? pending_.size() : ambiguous_suffix();
      if (token_boundaries_) {
        safe_size = std::min(safe_size, unresolved_pos());
      }
      if (safe_size == 0) {
        return;
      }
      const std::string payload = pending_.substr(0, safe_size);
      emit_payload(output, payload, incomplete);
      if (state_ == ParseState::TEXT && !payload.empty()) {
        text_started_ = true;
      }
      pending_.erase(0, safe_size);
      if (!flush) {
        return;
      }
    }
  }

  void finish(std::vector<OutputSegment>* output, bool incomplete) {
    if (failed_) {
      return;
    }
    if (token_boundaries_ && !token_markers_.empty()) {
      set_failure(ParseFailureCode::CONTROL_TOKEN_MISMATCH,
                  "control token has no matching text marker");
      output->emplace_back(failure_segment_);
      return;
    }
    if (state_ == ParseState::TOOL) {
      if (incomplete) {
        return;
      }
      set_failure(ParseFailureCode::UNCLOSED_TOOL_CALL,
                  "generation finished with an unclosed tool call");
      output->emplace_back(failure_segment_);
      return;
    }
    if (!pending_.empty()) {
      emit_payload(output, pending_, incomplete);
      pending_.clear();
    }
  }

  TextReasoningGrammar grammar_;
  ParseState state_ = ParseState::REASONING;
  std::string pending_;
  std::string utf8_pending_;
  size_t sequence_index_ = 0;
  uint64_t ordinal_ = 0;
  bool sequence_initialized_ = false;
  bool ordinal_initialized_ = false;
  bool finalized_ = false;
  bool failed_ = false;
  bool token_boundaries_ = false;
  bool text_started_ = false;
  bool reasoning_tokens_done_ = false;
  int32_t reasoning_tokens_ = 0;
  std::vector<int32_t> token_tail_;
  std::vector<MarkerKind> token_markers_;
  OutputSegment failure_segment_;
};

}  // namespace

std::unique_ptr<ModelOutputParser> make_text_reasoning_parser(
    TextReasoningGrammar grammar) {
  return std::make_unique<TextReasoningParser>(std::move(grammar));
}

}  // namespace xllm::model_protocol
