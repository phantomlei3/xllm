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

#include <openssl/rand.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "nlohmann/json.hpp"

namespace xllm::responses {
namespace {

constexpr size_t kMaxIdBytes = 256;
constexpr size_t kRandomIdBytes = 24;
constexpr size_t kIdAttempts = 64;
constexpr std::string_view kIdAlphabet =
    "23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

bool valid_id(const std::string& id) {
  if (id.empty() || id.size() > kMaxIdBytes) {
    return false;
  }
  return std::all_of(id.begin(), id.end(), [](char byte) {
    const unsigned char value = static_cast<unsigned char>(byte);
    return std::isalnum(value) != 0 || byte == '_' || byte == '-' ||
           byte == '.';
  });
}

bool valid_call_id(const std::string& id) {
  return id.rfind("call_", 0) == 0 && valid_id(id);
}

bool valid_utf8(const std::string& text) {
  size_t index = 0;
  while (index < text.size()) {
    const uint8_t first = static_cast<uint8_t>(text[index]);
    if (first <= 0x7f) {
      ++index;
      continue;
    }
    size_t count = 0;
    uint32_t code_point = 0;
    uint32_t minimum = 0;
    if ((first & 0xe0) == 0xc0) {
      count = 1;
      code_point = first & 0x1f;
      minimum = 0x80;
    } else if ((first & 0xf0) == 0xe0) {
      count = 2;
      code_point = first & 0x0f;
      minimum = 0x800;
    } else if ((first & 0xf8) == 0xf0) {
      count = 3;
      code_point = first & 0x07;
      minimum = 0x10000;
    } else {
      return false;
    }
    if (index + count >= text.size()) {
      return false;
    }
    for (size_t offset = 1; offset <= count; ++offset) {
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
    index += count + 1;
  }
  return true;
}

bool exceeds_depth(const std::string& text, uint32_t limit) {
  uint32_t depth = 0;
  bool quoted = false;
  bool escaped = false;
  for (char byte : text) {
    if (quoted) {
      if (escaped) {
        escaped = false;
      } else if (byte == '\\') {
        escaped = true;
      } else if (byte == '"') {
        quoted = false;
      }
      continue;
    }
    if (byte == '"') {
      quoted = true;
    } else if (byte == '{' || byte == '[') {
      ++depth;
      if (depth > limit) {
        return true;
      }
    } else if ((byte == '}' || byte == ']') && depth > 0) {
      --depth;
    }
  }
  return false;
}

}  // namespace

std::string RandomIdProvider::next(std::string_view prefix) {
  std::string id(prefix);
  id.reserve(prefix.size() + kRandomIdBytes);
  constexpr size_t kRandomBatchBytes = 64;
  const uint32_t acceptance_limit =
      256 - (256 % static_cast<uint32_t>(kIdAlphabet.size()));
  while (id.size() < prefix.size() + kRandomIdBytes) {
    std::array<unsigned char, kRandomBatchBytes> bytes{};
    if (RAND_priv_bytes(bytes.data(), static_cast<int32_t>(bytes.size())) !=
        1) {
      return {};
    }
    for (unsigned char byte : bytes) {
      if (byte >= acceptance_limit) {
        continue;
      }
      id.push_back(kIdAlphabet[byte % kIdAlphabet.size()]);
      if (id.size() == prefix.size() + kRandomIdBytes) {
        break;
      }
    }
  }
  return id;
}

ResponsesProcessor::ResponsesProcessor(ProcessorConfig config,
                                       std::unique_ptr<IdProvider> id_provider)
    : config_(std::move(config)), id_provider_(std::move(id_provider)) {
  if (id_provider_ == nullptr) {
    id_provider_ = std::make_unique<RandomIdProvider>();
  }
  reserve_replayed_ids();
  response_.id = new_id("resp_");
  response_.created_at = config_.created_at;
  response_.model = config_.model;
  response_.reasoning_effort = config_.reasoning_effort;
  if (response_.id.empty()) {
    fail(ErrorCode::GENERATION_FAILED, "could not allocate response ID");
  }
}

bool ResponsesProcessor::consume(const model_protocol::OutputSegment& segment) {
  if (response_.status != ResponseStatus::IN_PROGRESS) {
    return false;
  }
  using model_protocol::OutputSegmentKind;
  switch (segment.kind) {
    case OutputSegmentKind::REASONING_DELTA:
      return append_reasoning(segment);
    case OutputSegmentKind::REASONING_DONE:
      return close_text_item(ActiveKind::REASONING);
    case OutputSegmentKind::TEXT_DELTA:
      return append_text(segment);
    case OutputSegmentKind::TEXT_DONE:
      return close_text_item(ActiveKind::TEXT);
    case OutputSegmentKind::FUNCTION_CALL_START:
      return start_function(segment);
    case OutputSegmentKind::ARGUMENTS_DELTA:
      return append_arguments(segment);
    case OutputSegmentKind::FUNCTION_CALL_DONE:
      return done_function();
    case OutputSegmentKind::CUSTOM_CALL_START:
      return start_custom(segment);
    case OutputSegmentKind::CUSTOM_INPUT_DELTA:
      return append_custom(segment);
    case OutputSegmentKind::CUSTOM_CALL_DONE:
      return done_custom();
    case OutputSegmentKind::PARSE_FAILURE:
      return fail(
          segment.failure.has_value() &&
                  segment.failure->code ==
                      model_protocol::ParseFailureCode::INVALID_TOOL_ARGUMENTS
              ? ErrorCode::INVALID_TOOL_ARGUMENTS
              : ErrorCode::GENERATION_FAILED,
          segment.failure.has_value() ? segment.failure->message
                                      : "model output parse failed");
  }
  return fail(ErrorCode::GENERATION_FAILED, "unknown output segment");
}

bool ResponsesProcessor::consume(const model_protocol::GenerationDelta& delta,
                                 model_protocol::ModelOutputParser& parser) {
  if (response_.status != ResponseStatus::IN_PROGRESS) {
    return false;
  }
  if (delta.sequence_index != 0) {
    return fail(ErrorCode::GENERATION_FAILED,
                "responses generation only accepts sequence index 0");
  }
  if (ordinal_initialized_ && delta.generation_ordinal <= last_ordinal_) {
    return fail(ErrorCode::GENERATION_FAILED,
                "generation ordinal is not increasing");
  }
  ordinal_initialized_ = true;
  last_ordinal_ = delta.generation_ordinal;
  generated_tokens_ += delta.token_id_delta.size();

  if (delta.backend_error.has_value()) {
    const std::string message = delta.backend_error->message.empty()
                                    ? "backend generation failed"
                                    : delta.backend_error->message;
    return fail(ErrorCode::GENERATION_FAILED, message);
  }

  const bool token_limit = config_.max_output_tokens.has_value() &&
                           generated_tokens_ >= *config_.max_output_tokens;
  const bool terminal = delta.finished || token_limit;
  if (!terminal && delta.final_usage.has_value()) {
    return fail(ErrorCode::GENERATION_FAILED,
                "final usage arrived before generation finished");
  }

  const std::vector<model_protocol::OutputSegment> segments =
      parser.consume(delta);
  if (terminal) {
    if (!delta.final_usage.has_value()) {
      return fail(ErrorCode::GENERATION_FAILED,
                  "generation finished without reliable usage");
    }
    if (!set_usage(*delta.final_usage, parser)) {
      return false;
    }
  }
  for (const model_protocol::OutputSegment& segment : segments) {
    if (!consume(segment)) {
      return false;
    }
  }
  if (!terminal) {
    return true;
  }
  return finish_delta(delta, token_limit);
}

bool ResponsesProcessor::append_reasoning(
    const model_protocol::OutputSegment& segment) {
  if (reasoning_closed_ || phase_ == OutputPhase::CALLS ||
      phase_ == OutputPhase::TEXT ||
      (active_kind_ != ActiveKind::NONE &&
       active_kind_ != ActiveKind::REASONING)) {
    return fail(ErrorCode::GENERATION_FAILED,
                "reasoning delta is out of order");
  }
  if (!valid_utf8(segment.text)) {
    return fail(ErrorCode::GENERATION_FAILED, "reasoning delta is not UTF-8");
  }
  if (active_kind_ == ActiveKind::NONE) {
    const std::string id = new_id("rs_");
    if (id.empty()) {
      return fail(ErrorCode::GENERATION_FAILED, "could not allocate item ID");
    }
    response_.output.emplace_back(OutputReasoningItem{.id = id});
    active_index_ = response_.output.size() - 1;
    active_kind_ = ActiveKind::REASONING;
    phase_ = OutputPhase::REASONING;
  }
  OutputReasoningItem& item =
      std::get<OutputReasoningItem>(response_.output[active_index_]);
  if (item.content.size() + segment.text.size() >
      config_.limits.max_text_bytes) {
    return fail(ErrorCode::REQUEST_TOO_LARGE,
                "reasoning output exceeds configured limit");
  }
  item.content += segment.text;
  if (segment.incomplete) {
    item.status = ItemStatus::INCOMPLETE;
  }
  return true;
}

bool ResponsesProcessor::append_text(
    const model_protocol::OutputSegment& segment) {
  if (text_closed_ ||
      (active_kind_ != ActiveKind::NONE && active_kind_ != ActiveKind::TEXT)) {
    return fail(ErrorCode::GENERATION_FAILED, "text delta is out of order");
  }
  if (!valid_utf8(segment.text)) {
    return fail(ErrorCode::GENERATION_FAILED, "text delta is not UTF-8");
  }
  if (active_kind_ == ActiveKind::NONE) {
    const std::string id = new_id("msg_");
    if (id.empty()) {
      return fail(ErrorCode::GENERATION_FAILED, "could not allocate item ID");
    }
    response_.output.emplace_back(OutputMessageItem{.id = id});
    active_index_ = response_.output.size() - 1;
    active_kind_ = ActiveKind::TEXT;
    phase_ = OutputPhase::TEXT;
  }
  OutputMessageItem& item =
      std::get<OutputMessageItem>(response_.output[active_index_]);
  if (item.content.size() + segment.text.size() >
      config_.limits.max_text_bytes) {
    return fail(ErrorCode::REQUEST_TOO_LARGE,
                "text output exceeds configured limit");
  }
  item.content += segment.text;
  if (segment.incomplete) {
    item.status = ItemStatus::INCOMPLETE;
  }
  return true;
}

bool ResponsesProcessor::start_function(
    const model_protocol::OutputSegment& segment) {
  if (active_kind_ != ActiveKind::NONE || phase_ == OutputPhase::TEXT ||
      segment.name.empty()) {
    return fail(ErrorCode::GENERATION_FAILED,
                "function call start is out of order");
  }
  const std::string item_id = new_id("fc_");
  const std::string linkage_id = call_id(segment.call_id);
  if (item_id.empty() || linkage_id.empty()) {
    return fail(ErrorCode::GENERATION_FAILED, "could not allocate call ID");
  }
  response_.output.emplace_back(OutputFunctionCallItem{
      .id = item_id,
      .call = FunctionCall{.id = linkage_id, .name = segment.name}});
  active_index_ = response_.output.size() - 1;
  active_kind_ = ActiveKind::FUNCTION;
  phase_ = OutputPhase::CALLS;
  return true;
}

bool ResponsesProcessor::append_arguments(
    const model_protocol::OutputSegment& segment) {
  if (active_kind_ != ActiveKind::FUNCTION) {
    return fail(ErrorCode::GENERATION_FAILED,
                "function arguments delta has no open call");
  }
  if (!valid_utf8(segment.text)) {
    return fail(ErrorCode::INVALID_TOOL_ARGUMENTS,
                "function arguments are not UTF-8");
  }
  OutputFunctionCallItem& item =
      std::get<OutputFunctionCallItem>(response_.output[active_index_]);
  if (item.call.arguments.size() + segment.text.size() >
      config_.limits.max_function_args_bytes) {
    return fail(ErrorCode::REQUEST_TOO_LARGE,
                "function arguments exceed configured limit");
  }
  item.call.arguments += segment.text;
  return true;
}

bool ResponsesProcessor::done_function() {
  if (active_kind_ != ActiveKind::FUNCTION) {
    return fail(ErrorCode::GENERATION_FAILED,
                "function call done has no open call");
  }
  OutputFunctionCallItem& item =
      std::get<OutputFunctionCallItem>(response_.output[active_index_]);
  try {
    const nlohmann::json arguments = nlohmann::json::parse(item.call.arguments);
    if (!arguments.is_object() ||
        exceeds_depth(item.call.arguments, config_.limits.max_json_depth)) {
      return fail(ErrorCode::INVALID_TOOL_ARGUMENTS,
                  "function arguments must be a JSON object");
    }
  } catch (const nlohmann::json::exception&) {
    return fail(ErrorCode::INVALID_TOOL_ARGUMENTS,
                "function arguments must be a JSON object");
  }
  item.status = ItemStatus::COMPLETED;
  active_kind_ = ActiveKind::NONE;
  return true;
}

bool ResponsesProcessor::start_custom(
    const model_protocol::OutputSegment& segment) {
  if (active_kind_ != ActiveKind::NONE || phase_ == OutputPhase::TEXT ||
      segment.name != "apply_patch") {
    return fail(ErrorCode::GENERATION_FAILED,
                "custom call start is invalid or out of order");
  }
  const std::string item_id = new_id("ctc_");
  const std::string linkage_id = call_id(segment.call_id);
  if (item_id.empty() || linkage_id.empty()) {
    return fail(ErrorCode::GENERATION_FAILED, "could not allocate call ID");
  }
  response_.output.emplace_back(OutputCustomToolCallItem{
      .id = item_id,
      .call = CustomToolCall{.id = linkage_id, .name = "apply_patch"}});
  active_index_ = response_.output.size() - 1;
  active_kind_ = ActiveKind::CUSTOM;
  phase_ = OutputPhase::CALLS;
  return true;
}

bool ResponsesProcessor::append_custom(
    const model_protocol::OutputSegment& segment) {
  if (active_kind_ != ActiveKind::CUSTOM) {
    return fail(ErrorCode::GENERATION_FAILED,
                "custom input delta has no open call");
  }
  if (!valid_utf8(segment.text)) {
    return fail(ErrorCode::GENERATION_FAILED, "custom input is not UTF-8");
  }
  OutputCustomToolCallItem& item =
      std::get<OutputCustomToolCallItem>(response_.output[active_index_]);
  if (item.call.input.size() + segment.text.size() >
      config_.limits.max_custom_payload_bytes) {
    return fail(ErrorCode::REQUEST_TOO_LARGE,
                "custom input exceeds configured limit");
  }
  item.call.input += segment.text;
  return true;
}

bool ResponsesProcessor::done_custom() {
  if (active_kind_ != ActiveKind::CUSTOM) {
    return fail(ErrorCode::GENERATION_FAILED,
                "custom call done has no open call");
  }
  OutputCustomToolCallItem& item =
      std::get<OutputCustomToolCallItem>(response_.output[active_index_]);
  item.status = ItemStatus::COMPLETED;
  active_kind_ = ActiveKind::NONE;
  return true;
}

bool ResponsesProcessor::close_text_item(ActiveKind kind) {
  if (active_kind_ == ActiveKind::NONE && kind == ActiveKind::REASONING &&
      phase_ == OutputPhase::START) {
    reasoning_closed_ = true;
    return true;
  }
  if (active_kind_ != kind) {
    return fail(ErrorCode::GENERATION_FAILED, "item done is out of order");
  }
  if (kind == ActiveKind::REASONING) {
    OutputReasoningItem& item =
        std::get<OutputReasoningItem>(response_.output[active_index_]);
    if (item.status == ItemStatus::IN_PROGRESS) {
      item.status = ItemStatus::COMPLETED;
    }
    reasoning_closed_ = true;
  } else {
    OutputMessageItem& item =
        std::get<OutputMessageItem>(response_.output[active_index_]);
    if (item.status == ItemStatus::IN_PROGRESS) {
      item.status = ItemStatus::COMPLETED;
    }
    text_closed_ = true;
  }
  active_kind_ = ActiveKind::NONE;
  return true;
}

bool ResponsesProcessor::fail(ErrorCode code, const std::string& message) {
  if (response_.status != ResponseStatus::IN_PROGRESS) {
    return false;
  }
  response_.status = ResponseStatus::FAILED;
  response_.error =
      ResponsesError{.message = message, .type = "server_error", .code = code};
  return false;
}

bool ResponsesProcessor::set_usage(
    const model_protocol::GenerationUsage& usage,
    const model_protocol::ModelOutputParser& parser) {
  const int64_t expected_total = static_cast<int64_t>(usage.input_tokens) +
                                 static_cast<int64_t>(usage.output_tokens);
  if (usage.input_tokens < 0 || usage.cached_input_tokens < 0 ||
      usage.output_tokens < 0 || usage.total_tokens < 0 ||
      (usage.reasoning_tokens.has_value() && *usage.reasoning_tokens < 0) ||
      usage.cached_input_tokens > usage.input_tokens ||
      static_cast<int64_t>(usage.total_tokens) != expected_total ||
      static_cast<uint64_t>(usage.output_tokens) != generated_tokens_) {
    return fail(ErrorCode::GENERATION_FAILED,
                "backend returned inconsistent generation usage");
  }
  const std::optional<int32_t> attributed = parser.reasoning_tokens();
  if (usage.reasoning_tokens.has_value() && attributed.has_value() &&
      *usage.reasoning_tokens != *attributed) {
    return fail(ErrorCode::GENERATION_FAILED,
                "backend and parser reasoning usage disagree");
  }
  if (!usage.reasoning_tokens.has_value() && !attributed.has_value()) {
    return fail(ErrorCode::GENERATION_FAILED,
                "reasoning token usage is not reliable");
  }
  model_protocol::GenerationUsage reliable = usage;
  if (!reliable.reasoning_tokens.has_value()) {
    reliable.reasoning_tokens = *attributed;
  }
  if (*reliable.reasoning_tokens > reliable.output_tokens) {
    return fail(ErrorCode::GENERATION_FAILED,
                "reasoning usage exceeds generated tokens");
  }
  response_.usage = reliable;
  return true;
}

void ResponsesProcessor::mark_incomplete() {
  if (active_kind_ == ActiveKind::REASONING) {
    std::get<OutputReasoningItem>(response_.output[active_index_]).status =
        ItemStatus::INCOMPLETE;
  } else if (active_kind_ == ActiveKind::TEXT) {
    std::get<OutputMessageItem>(response_.output[active_index_]).status =
        ItemStatus::INCOMPLETE;
  }
}

bool ResponsesProcessor::finish_delta(
    const model_protocol::GenerationDelta& delta,
    bool token_limit) {
  const std::string reason = delta.finish_reason.value_or("");
  const bool incomplete = token_limit || reason == "length";
  if (incomplete) {
    mark_incomplete();
    response_.status = ResponseStatus::INCOMPLETE;
    response_.incomplete_reason = "max_output_tokens";
    return true;
  }
  if (!delta.finished ||
      (reason != "stop" && reason != "eos" && reason != "tool_calls")) {
    return fail(ErrorCode::GENERATION_FAILED,
                "generation ended with an unsupported finish reason");
  }
  if (active_kind_ == ActiveKind::FUNCTION ||
      active_kind_ == ActiveKind::CUSTOM) {
    return fail(ErrorCode::GENERATION_FAILED,
                "generation ended with an open call");
  }
  if (active_kind_ == ActiveKind::REASONING) {
    close_text_item(ActiveKind::REASONING);
  } else if (active_kind_ == ActiveKind::TEXT) {
    close_text_item(ActiveKind::TEXT);
  }
  response_.status = ResponseStatus::COMPLETED;
  return true;
}

std::string ResponsesProcessor::new_id(std::string_view prefix) {
  for (size_t attempt = 0; attempt < kIdAttempts; ++attempt) {
    std::string id = id_provider_->next(prefix);
    if (id.rfind(prefix, 0) == 0 && valid_id(id) && ids_.emplace(id).second) {
      return id;
    }
  }
  return {};
}

std::string ResponsesProcessor::call_id(
    const std::optional<std::string>& candidate) {
  if (candidate.has_value() && valid_call_id(*candidate) &&
      ids_.emplace(*candidate).second) {
    return *candidate;
  }
  return new_id("call_");
}

void ResponsesProcessor::reserve_replayed_ids() {
  for (const InputItem& item : config_.replayed_items) {
    std::visit(
        [this](const auto& value) {
          if (!value.id.empty()) {
            ids_.emplace(value.id);
          }
          using Value = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<Value, FunctionCallItem> ||
                        std::is_same_v<Value, CustomToolCallItem>) {
            ids_.emplace(value.call.id);
          }
        },
        item);
  }
}

const FinalResponse& ResponsesProcessor::finish() {
  if (response_.status != ResponseStatus::IN_PROGRESS) {
    return response_;
  }
  fail(ErrorCode::GENERATION_FAILED,
       "generation finished without reliable usage");
  return response_;
}

const FinalResponse& ResponsesProcessor::timeout() {
  fail(ErrorCode::REQUEST_TIMEOUT, "request deadline exceeded");
  return response_;
}

const FinalResponse& ResponsesProcessor::cancel() {
  if (response_.status == ResponseStatus::IN_PROGRESS) {
    response_.status = ResponseStatus::CANCELLED;
  }
  return response_;
}

}  // namespace xllm::responses
