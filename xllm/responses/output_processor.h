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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "core/model_protocol/output_parser.h"
#include "responses/error.h"
#include "responses/responses_limits.h"
#include "responses/types.h"

namespace xllm::responses {

enum class ResponseStatus : uint8_t {
  IN_PROGRESS = 0,
  COMPLETED = 1,
  INCOMPLETE = 2,
  FAILED = 3,
  CANCELLED = 4,
};

struct FinalResponse {
  std::string id;
  std::string object = "response";
  int64_t created_at = 0;
  ResponseStatus status = ResponseStatus::IN_PROGRESS;
  std::optional<ResponsesError> error;
  std::optional<std::string> incomplete_reason;
  std::string model;
  std::vector<OutputItem> output;
  bool parallel_tool_calls = true;
  model_protocol::ReasoningEffort reasoning_effort =
      model_protocol::ReasoningEffort::MEDIUM;
  bool store = false;
  std::optional<model_protocol::GenerationUsage> usage;
};

struct ResponseCreatedEvent {
  FinalResponse response;
};
struct ResponseInProgressEvent {
  FinalResponse response;
};
struct ResponseCompletedEvent {
  FinalResponse response;
};
struct ResponseIncompleteEvent {
  FinalResponse response;
};
struct ResponseFailedEvent {
  FinalResponse response;
};
struct OutputItemAddedEvent {
  size_t output_index;
  OutputItem item;
};
struct OutputItemDoneEvent {
  size_t output_index;
  OutputItem item;
};
struct ContentPartAddedEvent {
  size_t output_index;
  std::string item_id;
};
struct ContentPartDoneEvent {
  size_t output_index;
  std::string item_id;
  std::string text;
};
struct OutputTextDeltaEvent {
  size_t output_index;
  std::string item_id;
  std::string delta;
};
struct OutputTextDoneEvent {
  size_t output_index;
  std::string item_id;
  std::string text;
};
struct ReasoningTextDeltaEvent {
  size_t output_index;
  std::string item_id;
  std::string delta;
};
struct ReasoningTextDoneEvent {
  size_t output_index;
  std::string item_id;
  std::string text;
};
struct FunctionArgumentsDeltaEvent {
  size_t output_index;
  std::string item_id;
  std::string delta;
};
struct FunctionArgumentsDoneEvent {
  size_t output_index;
  std::string item_id;
  std::string arguments;
};
struct CustomInputDeltaEvent {
  size_t output_index;
  std::string item_id;
  std::string delta;
};
struct CustomInputDoneEvent {
  size_t output_index;
  std::string item_id;
  std::string input;
};

using ResponseEventData = std::variant<ResponseCreatedEvent,
                                       ResponseInProgressEvent,
                                       ResponseCompletedEvent,
                                       ResponseIncompleteEvent,
                                       ResponseFailedEvent,
                                       OutputItemAddedEvent,
                                       OutputItemDoneEvent,
                                       ContentPartAddedEvent,
                                       ContentPartDoneEvent,
                                       OutputTextDeltaEvent,
                                       OutputTextDoneEvent,
                                       ReasoningTextDeltaEvent,
                                       ReasoningTextDoneEvent,
                                       FunctionArgumentsDeltaEvent,
                                       FunctionArgumentsDoneEvent,
                                       CustomInputDeltaEvent,
                                       CustomInputDoneEvent>;

struct ResponseEvent {
  uint64_t sequence_number;
  ResponseEventData data;
};

class IdProvider {
 public:
  virtual ~IdProvider() = default;
  virtual std::string next(std::string_view prefix) = 0;
};

class RandomIdProvider final : public IdProvider {
 public:
  std::string next(std::string_view prefix) override;
};

struct ProcessorConfig {
  std::string model;
  int64_t created_at = 0;
  model_protocol::ReasoningEffort reasoning_effort =
      model_protocol::ReasoningEffort::MEDIUM;
  std::vector<InputItem> replayed_items;
  std::optional<uint32_t> max_output_tokens;
  ResponsesLimits limits;
};

// One serialized request context owns each processor instance.
class ResponsesProcessor final {
 public:
  explicit ResponsesProcessor(ProcessorConfig config,
                              std::unique_ptr<IdProvider> id_provider =
                                  std::make_unique<RandomIdProvider>());

  bool consume(const model_protocol::OutputSegment& segment);
  bool consume(const model_protocol::GenerationDelta& delta,
               model_protocol::ModelOutputParser& parser);
  const FinalResponse& finish();
  const FinalResponse& timeout();
  const FinalResponse& fail_request(ErrorCode code, const std::string& message);
  const FinalResponse& cancel();
  const FinalResponse& response() const { return response_; }
  std::vector<ResponseEvent> take_events();

 private:
  enum class ActiveKind : uint8_t {
    NONE = 0,
    REASONING = 1,
    TEXT = 2,
    FUNCTION = 3,
    CUSTOM = 4,
  };

  enum class OutputPhase : uint8_t {
    START = 0,
    REASONING = 1,
    TEXT_CANDIDATE = 2,
    CALLS_WITHOUT_PRE_TEXT = 3,
    CALLS_AFTER_PRE_TEXT = 4,
  };

  bool append_reasoning(const model_protocol::OutputSegment& segment);
  bool append_text(const model_protocol::OutputSegment& segment);
  bool start_function(const model_protocol::OutputSegment& segment);
  bool append_arguments(const model_protocol::OutputSegment& segment);
  bool done_function();
  bool start_custom(const model_protocol::OutputSegment& segment);
  bool append_custom(const model_protocol::OutputSegment& segment);
  bool done_custom();
  bool close_text_item(ActiveKind kind);
  bool fail(ErrorCode code, const std::string& message);
  bool finish_delta(const model_protocol::GenerationDelta& delta,
                    bool token_limit);
  bool set_usage(const model_protocol::GenerationUsage& usage,
                 const model_protocol::ModelOutputParser& parser);
  void mark_incomplete();
  std::string new_id(std::string_view prefix);
  std::string call_id(const std::optional<std::string>& candidate);
  void reserve_replayed_ids();
  void emit(ResponseEventData data);
  void emit_terminal();

  ProcessorConfig config_;
  std::unique_ptr<IdProvider> id_provider_;
  FinalResponse response_;
  ActiveKind active_kind_ = ActiveKind::NONE;
  OutputPhase phase_ = OutputPhase::START;
  size_t active_index_ = 0;
  bool reasoning_closed_ = false;
  bool text_closed_ = false;
  bool ordinal_initialized_ = false;
  uint64_t last_ordinal_ = 0;
  uint64_t generated_tokens_ = 0;
  uint64_t next_event_sequence_ = 0;
  std::unordered_set<std::string> ids_;
  std::vector<ResponseEvent> events_;
};

}  // namespace xllm::responses
