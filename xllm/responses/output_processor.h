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
  ResponsesLimits limits;
};

// One serialized request context owns each processor instance.
class ResponsesProcessor final {
 public:
  explicit ResponsesProcessor(ProcessorConfig config,
                              std::unique_ptr<IdProvider> id_provider =
                                  std::make_unique<RandomIdProvider>());

  bool consume(const model_protocol::OutputSegment& segment);
  const FinalResponse& finish();
  const FinalResponse& response() const { return response_; }

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
    CALLS = 2,
    TEXT = 3,
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
  std::string new_id(std::string_view prefix);
  std::string call_id(const std::optional<std::string>& candidate);
  void reserve_replayed_ids();

  ProcessorConfig config_;
  std::unique_ptr<IdProvider> id_provider_;
  FinalResponse response_;
  ActiveKind active_kind_ = ActiveKind::NONE;
  OutputPhase phase_ = OutputPhase::START;
  size_t active_index_ = 0;
  bool reasoning_closed_ = false;
  bool text_closed_ = false;
  std::unordered_set<std::string> ids_;
};

}  // namespace xllm::responses
