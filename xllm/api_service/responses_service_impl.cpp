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

#include "api_service/responses_service_impl.h"

#include <glog/logging.h>
#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

#include "core/common/message.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/framework/config/model_config.h"
#include "core/framework/request/request_params.h"
#include "core/model_protocol/deepseek_v4_profile.h"
#include "core/model_protocol/glm_5_2_profile.h"
#include "core/util/threadpool.h"
#include "responses/json_encoder.h"
#include "responses/output_processor.h"
#include "responses/sse_encoder.h"

namespace xllm {
namespace {

using responses::ErrorCode;
using responses::ResponsesError;

ResponsesHttpResult error_result(int32_t status_code, ResponsesError error) {
  return {.status_code = status_code, .body = responses::encode_error(error)};
}

const char* terminal_status(responses::ResponseStatus status) {
  switch (status) {
    case responses::ResponseStatus::IN_PROGRESS:
      return "in_progress";
    case responses::ResponseStatus::COMPLETED:
      return "completed";
    case responses::ResponseStatus::INCOMPLETE:
      return "incomplete";
    case responses::ResponseStatus::FAILED:
      return "failed";
    case responses::ResponseStatus::CANCELLED:
      return "cancelled";
  }
  return "failed";
}

struct RequestLogContext {
  responses::RequestContext ids;
  std::string model;
  std::string profile;
};

RequestLogContext request_log_context(
    const responses::PreparedRequest& request) {
  return {.ids = request.context,
          .model = request.canonical_model_id,
          .profile = request.profile_id};
}

void log_rejection(const responses::RequestContext& context,
                   const std::string& model,
                   const std::string& profile,
                   size_t body_bytes,
                   ErrorCode code) {
  const nlohmann::json encoded = responses::encode_error({.code = code});
  LOG(INFO) << "Responses request trace_id=" << context.trace_id
            << " request_id=" << context.request_id << " model=" << model
            << " profile=" << profile
            << " status=" << encoded["error"]["code"].get<std::string>()
            << " body_bytes=" << body_bytes << " terminal=rejected";
}

void log_terminal(const RequestLogContext& request,
                  const responses::FinalResponse& response,
                  size_t body_bytes) {
  LOG(INFO) << "Responses request trace_id=" << request.ids.trace_id
            << " request_id=" << request.ids.request_id
            << " response_id=" << response.id << " model=" << request.model
            << " profile=" << request.profile
            << " status=" << terminal_status(response.status)
            << " body_bytes=" << body_bytes
            << " output_items=" << response.output.size() << " input_tokens="
            << (response.usage.has_value() ? response.usage->input_tokens : 0)
            << " output_tokens="
            << (response.usage.has_value() ? response.usage->output_tokens : 0)
            << " total_tokens="
            << (response.usage.has_value() ? response.usage->total_tokens : 0)
            << " terminal=" << terminal_status(response.status);
}

bool is_json_content_type(const std::string& content_type) {
  const size_t delimiter = content_type.find(';');
  const std::string media_type = content_type.substr(0, delimiter);
  return media_type == "application/json";
}

std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return "";
  }
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), &EVP_MD_CTX_free);
  if (context == nullptr ||
      EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    return "";
  }
  std::array<char, 64 * 1024> buffer;
  while (input) {
    input.read(buffer.data(), buffer.size());
    const std::streamsize count = input.gcount();
    if (count > 0) {
      if (EVP_DigestUpdate(
              context.get(), buffer.data(), static_cast<size_t>(count)) != 1) {
        return "";
      }
    }
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest;
  uint32_t digest_size = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &digest_size) != 1) {
    return "";
  }
  std::ostringstream encoded;
  encoded << "sha256:" << std::hex << std::setfill('0');
  for (uint32_t index = 0; index < digest_size; ++index) {
    encoded << std::setw(2) << static_cast<uint32_t>(digest[index]);
  }
  return encoded.str();
}

std::vector<Message> to_messages(const proto::ChatRequest& request) {
  std::vector<Message> messages;
  messages.reserve(request.messages_size());
  for (const proto::ChatMessage& source : request.messages()) {
    messages.emplace_back(source.role(), source.content());
    Message& message = messages.back();
    if (source.has_reasoning_content()) {
      message.reasoning_content = source.reasoning_content();
    }
    if (source.has_tool_call_id()) {
      message.tool_call_id = source.tool_call_id();
    }
    if (source.tool_calls_size() == 0) {
      continue;
    }
    Message::ProtocolToolCallVec calls;
    calls.reserve(source.tool_calls_size());
    for (const proto::ToolCall& source_call : source.tool_calls()) {
      ToolCallConversion converted = tool_call_from_proto(source_call);
      if (converted.ok()) {
        calls.emplace_back(*converted.value());
      }
    }
    message.protocol_tool_calls = std::move(calls);
    if (source.has_tool_output_type()) {
      message.tool_output_kind =
          source.tool_output_type() == proto::CUSTOM_OUTPUT
              ? ToolOutputKind::CUSTOM
              : ToolOutputKind::FUNCTION;
    }
  }
  return messages;
}

model_protocol::GenerationUsage to_usage(const Usage& usage) {
  return {.input_tokens = usage.num_prompt_tokens,
          .cached_input_tokens = usage.num_cached_tokens,
          .output_tokens = usage.num_generated_tokens,
          .total_tokens = usage.num_total_tokens};
}

class PoolSerialExecutor final : public ResponsesSerialExecutor {
 public:
  PoolSerialExecutor(std::shared_ptr<ThreadPool> pool, size_t thread_id)
      : pool_(std::move(pool)), thread_id_(thread_id) {}

  void post(std::function<void()> task) override {
    pool_->schedule_with_tid(std::move(task), thread_id_);
  }

 private:
  std::shared_ptr<ThreadPool> pool_;
  size_t thread_id_;
};

class StreamSession final : public ResponsesStreamControl,
                            public std::enable_shared_from_this<StreamSession> {
 public:
  StreamSession(
      const responses::PreparedRequest& request,
      std::shared_ptr<const model_protocol::ModelProtocolProfile> profile,
      ResponsesExecutor* executor,
      std::shared_ptr<ResponsesStreamWriter> writer,
      std::shared_ptr<ResponsesSerialExecutor> serial_executor,
      const responses::ResponsesLimits& limits,
      size_t body_bytes)
      : request_(request),
        parser_(profile->new_parser()),
        processor_({.model = request.canonical_model_id,
                    .created_at =
                        std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count(),
                    .reasoning_effort = request.options.reasoning_effort,
                    .replayed_items = request.canonical_input,
                    .max_output_tokens = request.options.max_output_tokens,
                    .limits = limits}),
        executor_(executor),
        writer_(std::move(writer)),
        serial_executor_(std::move(serial_executor)),
        max_pending_bytes_(limits.max_sse_buffer_bytes),
        body_bytes_(body_bytes) {}

  void start() {
    self_keepalive_ = shared_from_this();
    std::weak_ptr<StreamSession> weak = shared_from_this();
    serial_executor_->post([weak]() {
      if (std::shared_ptr<StreamSession> self = weak.lock()) {
        self->start_serial();
      }
    });
  }

  void deadline() override {
    post([](StreamSession& self) { self.fail_timeout(); });
  }

  void disconnect() override {
    post([](StreamSession& self) { self.cancel_disconnected(); });
  }

  uint64_t pending_bytes() const override {
    return pending_bytes_.load(std::memory_order_relaxed);
  }

 private:
  using Action = std::function<void(StreamSession&)>;

  void post(Action action) {
    std::weak_ptr<StreamSession> weak = shared_from_this();
    serial_executor_->post([weak, action = std::move(action)]() mutable {
      if (std::shared_ptr<StreamSession> self = weak.lock()) {
        action(*self);
      }
    });
  }

  void start_serial() {
    if (!writer_->open()) {
      cancel_internal();
      finalize();
      return;
    }
    opened_ = true;
    if (!queue_events(processor_.take_events())) {
      return;
    }
    pump();
    if (terminal_) {
      return;
    }
    std::weak_ptr<StreamSession> weak = shared_from_this();
    executor_->execute(request_, [weak](RequestOutput output) {
      std::shared_ptr<StreamSession> self = weak.lock();
      if (self == nullptr ||
          self->terminal_requested_.load(std::memory_order_relaxed)) {
        return false;
      }
      self->post([output = std::move(output)](StreamSession& session) mutable {
        session.consume(std::move(output));
      });
      return !self->terminal_requested_.load(std::memory_order_relaxed);
    });
  }

  void consume(RequestOutput output) {
    if (terminal_) {
      return;
    }
    if (output.cancelled) {
      cancel_disconnected();
      return;
    }
    if (output.status.has_value() && !output.status->ok()) {
      processor_.fail_request(responses::ErrorCode::GENERATION_FAILED,
                              output.status->message());
      if (!queue_events(processor_.take_events())) {
        return;
      }
      terminal_ = true;
      terminal_requested_.store(true, std::memory_order_relaxed);
      cancel_once();
      pump();
      return;
    }
    if (output.outputs.empty()) {
      if (output.finished) {
        model_protocol::GenerationDelta delta{.generation_ordinal = ++ordinal_,
                                              .finished = true,
                                              .finish_reason = "stop"};
        if (output.usage.has_value()) {
          delta.final_usage = to_usage(*output.usage);
        }
        processor_.consume(delta, *parser_);
        if (!queue_events(processor_.take_events())) {
          return;
        }
        terminal_ = true;
        finish_once();
        pump();
      }
      return;
    }
    auto unexpected = std::find_if(
        output.outputs.begin(),
        output.outputs.end(),
        [](const SequenceOutput& sequence) { return sequence.index != 0; });
    if (unexpected != output.outputs.end()) {
      model_protocol::GenerationDelta delta{.sequence_index = unexpected->index,
                                            .generation_ordinal = ++ordinal_};
      processor_.consume(delta, *parser_);
      if (!queue_events(processor_.take_events())) {
        return;
      }
      terminal_ = true;
      terminal_requested_.store(true, std::memory_order_relaxed);
      cancel_once();
      pump();
      return;
    }
    for (const SequenceOutput& sequence : output.outputs) {
      merge_output(sequence, output.finished);
      model_protocol::CumulativeGeneration cumulative{
          .sequence_index = sequence.index,
          .generation_ordinal = ++ordinal_,
          .text = cumulative_text_,
          .token_ids = cumulative_token_ids_,
          .finished = output.finished,
          .finish_reason = sequence.finish_reason};
      if (output.finished && output.usage.has_value()) {
        cumulative.final_usage = to_usage(*output.usage);
      }
      model_protocol::NormalizationResult normalized =
          normalizer_.normalize(cumulative);
      if (!normalized.ok()) {
        processor_.fail_request(responses::ErrorCode::GENERATION_FAILED,
                                normalized.failure().message);
      } else {
        processor_.consume(normalized.delta(), *parser_);
      }
      if (processor_.response().status !=
          responses::ResponseStatus::IN_PROGRESS) {
        const responses::ResponseStatus status = processor_.response().status;
        if (!queue_events(processor_.take_events())) {
          return;
        }
        terminal_ = true;
        if (status == responses::ResponseStatus::COMPLETED ||
            status == responses::ResponseStatus::INCOMPLETE) {
          finish_once();
        } else {
          terminal_requested_.store(true, std::memory_order_relaxed);
          cancel_once();
        }
        pump();
        return;
      }
      if (!queue_events(processor_.take_events())) {
        return;
      }
      pump();
    }
  }

  void merge_output(const SequenceOutput& sequence, bool finished) {
    const bool full_text =
        finished && sequence.text.starts_with(cumulative_text_);
    const bool full_tokens =
        finished && sequence.token_ids.size() >= cumulative_token_ids_.size() &&
        std::equal(cumulative_token_ids_.begin(),
                   cumulative_token_ids_.end(),
                   sequence.token_ids.begin());
    if (full_text && full_tokens) {
      cumulative_text_ = sequence.text;
      cumulative_token_ids_ = sequence.token_ids;
      return;
    }
    cumulative_text_.append(sequence.text);
    cumulative_token_ids_.insert(cumulative_token_ids_.end(),
                                 sequence.token_ids.begin(),
                                 sequence.token_ids.end());
  }

  bool queue_events(std::vector<responses::ResponseEvent> events) {
    for (const responses::ResponseEvent& event : events) {
      std::string frame = responses::encode_sse(event);
      const uint64_t next_bytes = pending_bytes() + frame.size();
      if (next_bytes > max_pending_bytes_) {
        fail_slow_client(event.sequence_number);
        return false;
      }
      pending_bytes_.store(next_bytes, std::memory_order_relaxed);
      frames_.emplace_back(std::move(frame));
    }
    return true;
  }

  void pump() {
    if (write_in_flight_) {
      return;
    }
    if (frames_.empty()) {
      if (terminal_) {
        finalize();
      }
      return;
    }
    write_in_flight_ = true;
    const uint64_t frame_size = frames_.front().size();
    in_flight_bytes_ = frame_size;
    std::string frame = std::move(frames_.front());
    frames_.pop_front();
    std::weak_ptr<StreamSession> weak = shared_from_this();
    writer_->write(std::move(frame), [weak, frame_size](bool success) {
      if (std::shared_ptr<StreamSession> self = weak.lock()) {
        self->post([success, frame_size](StreamSession& session) {
          session.write_done(success, frame_size);
        });
      }
    });
  }

  void write_done(bool success, uint64_t frame_size) {
    if (finalized_) {
      return;
    }
    write_in_flight_ = false;
    in_flight_bytes_ = 0;
    pending_bytes_.fetch_sub(frame_size, std::memory_order_relaxed);
    if (!success) {
      frames_.clear();
      pending_bytes_.store(0, std::memory_order_relaxed);
      processor_.fail_request(responses::ErrorCode::GENERATION_FAILED,
                              "stream write failed");
      terminal_ = true;
      terminal_requested_.store(true, std::memory_order_relaxed);
      cancel_once();
      finalize();
      return;
    }
    pump();
  }

  void fail_slow_client(uint64_t sequence_number) {
    if (finalized_) {
      return;
    }
    const std::string message = "client cannot consume the response stream";
    processor_.fail_request(responses::ErrorCode::CLIENT_TOO_SLOW, message);
    terminal_ = true;
    terminal_requested_.store(true, std::memory_order_relaxed);
    cancel_once();
    frames_.clear();
    pending_bytes_.store(in_flight_bytes_, std::memory_order_relaxed);
    if (writer_->writable()) {
      responses::FinalResponse failed = processor_.response();
      failed.status = responses::ResponseStatus::FAILED;
      failed.output.clear();
      failed.error = responses::ResponsesError{
          .message = message,
          .type = "server_error",
          .code = responses::ErrorCode::CLIENT_TOO_SLOW};
      responses::ResponseEvent event{
          .sequence_number = sequence_number,
          .data = responses::ResponseFailedEvent{.response = failed}};
      std::string frame = responses::encode_sse(event);
      if (pending_bytes() + frame.size() <= max_pending_bytes_) {
        pending_bytes_.fetch_add(frame.size(), std::memory_order_relaxed);
        frames_.emplace_back(std::move(frame));
      }
    }
    pump();
  }

  void fail_timeout() {
    if (terminal_) {
      return;
    }
    processor_.timeout();
    if (!queue_events(processor_.take_events())) {
      return;
    }
    terminal_ = true;
    terminal_requested_.store(true, std::memory_order_relaxed);
    cancel_once();
    pump();
  }

  void cancel_disconnected() {
    if (terminal_) {
      return;
    }
    processor_.cancel();
    terminal_ = true;
    terminal_requested_.store(true, std::memory_order_relaxed);
    cancel_once();
    frames_.clear();
    pending_bytes_.store(0, std::memory_order_relaxed);
    finalize();
  }

  void cancel_internal() {
    terminal_ = true;
    terminal_requested_.store(true, std::memory_order_relaxed);
    processor_.cancel();
    cancel_once();
  }

  void finish_once() {
    terminal_requested_.store(true, std::memory_order_relaxed);
    if (!finish_called_) {
      finish_called_ = true;
      executor_->finish_request(request_.context.request_id);
    }
  }

  void cancel_once() {
    if (!cancel_called_) {
      cancel_called_ = true;
      executor_->cancel(request_.context.request_id);
    }
  }

  void finalize() {
    if (finalized_) {
      return;
    }
    finalized_ = true;
    log_terminal(
        request_log_context(request_), processor_.response(), body_bytes_);
    if (opened_) {
      writer_->close();
    }
    writer_->complete_http();
    std::shared_ptr<StreamSession> keepalive = std::move(self_keepalive_);
    (void)keepalive;
  }

  responses::PreparedRequest request_;
  std::unique_ptr<model_protocol::ModelOutputParser> parser_;
  model_protocol::RequestOutputNormalizer normalizer_;
  responses::ResponsesProcessor processor_;
  ResponsesExecutor* executor_;
  std::shared_ptr<ResponsesStreamWriter> writer_;
  std::shared_ptr<ResponsesSerialExecutor> serial_executor_;
  std::shared_ptr<StreamSession> self_keepalive_;
  uint64_t max_pending_bytes_;
  size_t body_bytes_;
  std::atomic<uint64_t> pending_bytes_{0};
  std::atomic<bool> terminal_requested_{false};
  std::deque<std::string> frames_;
  std::string cumulative_text_;
  std::vector<int32_t> cumulative_token_ids_;
  uint64_t ordinal_ = 0;
  bool opened_ = false;
  bool write_in_flight_ = false;
  uint64_t in_flight_bytes_ = 0;
  bool terminal_ = false;
  bool finish_called_ = false;
  bool cancel_called_ = false;
  bool finalized_ = false;
};

class NonStreamSession final {
 public:
  NonStreamSession(
      const responses::PreparedRequest& request,
      std::shared_ptr<const model_protocol::ModelProtocolProfile> profile,
      ResponsesServiceImpl::Completion completion,
      const responses::ResponsesLimits& limits,
      size_t body_bytes)
      : log_context_(request_log_context(request)),
        parser_(profile->new_parser()),
        processor_({.model = request.canonical_model_id,
                    .created_at =
                        std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count(),
                    .reasoning_effort = request.options.reasoning_effort,
                    .replayed_items = request.canonical_input,
                    .max_output_tokens = request.options.max_output_tokens,
                    .limits = limits}),
        completion_(std::move(completion)),
        body_bytes_(body_bytes) {}

  bool consume(RequestOutput output) {
    if (completed_) {
      return false;
    }
    if (output.status.has_value() && !output.status->ok()) {
      model_protocol::GenerationDelta delta{
          .generation_ordinal = ++ordinal_,
          .backend_error = model_protocol::BackendError{
              .code = "generation_failed",
              .message = output.status->message()}};
      processor_.consume(delta, *parser_);
      complete();
      return false;
    }
    if (output.outputs.empty()) {
      if (output.finished) {
        model_protocol::GenerationDelta delta{.generation_ordinal = ++ordinal_,
                                              .finished = true,
                                              .finish_reason = "stop"};
        if (output.usage.has_value()) {
          delta.final_usage = to_usage(*output.usage);
        }
        processor_.consume(delta, *parser_);
        complete();
      }
      return !completed_;
    }
    auto unexpected = std::find_if(
        output.outputs.begin(),
        output.outputs.end(),
        [](const SequenceOutput& sequence) { return sequence.index != 0; });
    if (unexpected != output.outputs.end()) {
      model_protocol::GenerationDelta delta{.sequence_index = unexpected->index,
                                            .generation_ordinal = ++ordinal_};
      processor_.consume(delta, *parser_);
      complete();
      return false;
    }
    for (const SequenceOutput& sequence : output.outputs) {
      model_protocol::CumulativeGeneration cumulative{
          .sequence_index = sequence.index,
          .generation_ordinal = ++ordinal_,
          .text = sequence.text,
          .token_ids = sequence.token_ids,
          .finished = output.finished,
          .finish_reason = sequence.finish_reason};
      if (output.finished && output.usage.has_value()) {
        cumulative.final_usage = to_usage(*output.usage);
      }
      model_protocol::NormalizationResult normalized =
          normalizer_.normalize(cumulative);
      if (!normalized.ok()) {
        model_protocol::GenerationDelta failed{
            .generation_ordinal = ++ordinal_,
            .backend_error = model_protocol::BackendError{
                .code = "generation_failed",
                .message = normalized.failure().message}};
        processor_.consume(failed, *parser_);
        complete();
        return false;
      }
      processor_.consume(normalized.delta(), *parser_);
      if (processor_.response().status !=
          responses::ResponseStatus::IN_PROGRESS) {
        complete();
        return false;
      }
    }
    if (output.finished) {
      complete();
    }
    return !completed_;
  }

 private:
  void complete() {
    if (completed_) {
      return;
    }
    completed_ = true;
    log_terminal(log_context_, processor_.response(), body_bytes_);
    completion_({.status_code = 200,
                 .body = responses::encode_response(processor_.response())});
  }

  RequestLogContext log_context_;
  std::unique_ptr<model_protocol::ModelOutputParser> parser_;
  model_protocol::RequestOutputNormalizer normalizer_;
  responses::ResponsesProcessor processor_;
  ResponsesServiceImpl::Completion completion_;
  uint64_t ordinal_ = 0;
  size_t body_bytes_;
  bool completed_ = false;
};

}  // namespace

model_protocol::LoadedModelContext inspect_responses_model(
    const std::string& model_id,
    const LLMMaster& master) {
  const std::filesystem::path model_path = master.options().model_path();
  const bool cpp_template =
      ModelConfig::get_instance().use_cpp_chat_template() &&
      master.model_type() == "deepseek_v4";
  std::string template_id;
  std::filesystem::path template_path;
  if (cpp_template) {
    template_id =
        "xllm/core/framework/chat_template/deepseek_v4_cpp_template.cpp";
    template_path =
        std::filesystem::path(__FILE__).parent_path().parent_path() /
        "core/framework/chat_template/"
        "deepseek_v4_cpp_template.cpp";
  } else {
    template_path = model_path / "chat_template.jinja";
    template_id = template_path.string();
  }
  return {.model_id = model_id,
          .model_type = master.model_type(),
          .model_fingerprint = sha256_file(model_path / "config.json"),
          .tokenizer_id = model_path.string(),
          .tokenizer_fingerprint = sha256_file(model_path / "tokenizer.json"),
          .tokenizer_config_fingerprint =
              sha256_file(model_path / "tokenizer_config.json"),
          .template_id = std::move(template_id),
          .template_fingerprint = sha256_file(template_path)};
}

LLMMasterResponsesExecutor::LLMMasterResponsesExecutor(LLMMaster* master)
    : master_(master) {}

void LLMMasterResponsesExecutor::execute(
    const responses::PreparedRequest& request,
    OutputCallback callback) {
  if (master_->get_rate_limiter()->is_limited()) {
    callback(RequestOutput(
        Status(StatusCode::RESOURCE_EXHAUSTED, "request is rate limited")));
    return;
  }
  RequestParams params(request.chat_request,
                       request.context.request_id,
                       /*x_rtime=*/"");
  master_->handle_request(to_messages(request.chat_request),
                          std::nullopt,
                          std::move(params),
                          std::nullopt,
                          std::move(callback));
}

ResponsesServiceImpl::ResponsesServiceImpl(
    responses::ResponsesLimits limits,
    SerialExecutorFactory executor_factory)
    : limits_(std::move(limits)),
      executor_factory_(std::move(executor_factory)) {
  if (!executor_factory_) {
    constexpr size_t kStreamThreads = 4;
    stream_pool_ = std::make_shared<ThreadPool>(
        kStreamThreads, /*cpu_binding=*/false, "responses_stream");
    auto next_thread = std::make_shared<std::atomic<uint64_t>>(0);
    std::shared_ptr<ThreadPool> pool = stream_pool_;
    executor_factory_ = [pool, next_thread]() {
      const size_t thread_id = static_cast<size_t>(
          next_thread->fetch_add(1, std::memory_order_relaxed) % pool->size());
      return std::make_shared<PoolSerialExecutor>(pool, thread_id);
    };
  }
  model_protocol::ModelProtocolError error =
      registry_.add(model_protocol::make_deepseek_v4_profile());
  if (error.ok()) {
    error = registry_.add(model_protocol::make_glm_5_2_profile());
  }
  if (!error.ok()) {
    deployment_error_ = error.message();
  }
}

bool ResponsesServiceImpl::add_model(
    const model_protocol::LoadedModelContext& context,
    ResponsesExecutor* executor) {
  if (executor == nullptr) {
    deployment_error_ = "Responses executor is missing";
    return false;
  }
  model_protocol::ProfileResult result = registry_.resolve(context);
  if (!result.ok()) {
    if (result.error().code() ==
        model_protocol::ModelProtocolErrorCode::PROFILE_IDENTITY_MISMATCH) {
      deployment_error_ = result.error().message();
    }
    return false;
  }
  Backend backend{.profile = result.profile(), .executor = executor};
  const model_protocol::ModelProtocolIdentity& identity =
      result.profile()->identity();
  backends_.insert_or_assign(identity.canonical_model_id, backend);
  for (const std::string& alias : identity.model_aliases) {
    backends_.insert_or_assign(alias, backend);
  }
  return true;
}

bool ResponsesServiceImpl::add_model(
    const model_protocol::LoadedModelContext& context,
    LLMMaster* master) {
  auto executor = std::make_unique<LLMMasterResponsesExecutor>(master);
  if (!add_model(context, executor.get())) {
    return false;
  }
  owned_executors_.emplace_back(std::move(executor));
  return true;
}

void ResponsesServiceImpl::process_non_stream(const std::string& body,
                                              const std::string& content_type,
                                              responses::RequestContext context,
                                              Completion completion) const {
  if (body.size() > limits_.max_body_bytes) {
    completion(error_result(
        /*status_code=*/413,
        {.message = "request body exceeds the configured limit",
         .param = "body",
         .code = ErrorCode::REQUEST_TOO_LARGE}));
    return;
  }
  if (!is_json_content_type(content_type)) {
    completion(error_result(
        /*status_code=*/415,
        {.message = "Content-Type must be application/json",
         .param = "Content-Type",
         .code = ErrorCode::UNSUPPORTED_CONTENT_TYPE}));
    return;
  }
  responses::ModelFieldResult model = responses::read_model_field(body);
  if (!model.ok()) {
    completion(error_result(/*status_code=*/400, model.error()));
    return;
  }
  auto backend = backends_.find(model.model());
  if (backend == backends_.end()) {
    completion(error_result(
        /*status_code=*/400,
        {.message = "model has no Responses protocol profile",
         .param = "model",
         .code = ErrorCode::UNSUPPORTED_MODEL_CAPABILITY}));
    return;
  }
  responses::PrepareResult prepared = responses::prepare_request(
      body, backend->second.profile->identity(), context, limits_);
  if (!prepared.ok()) {
    const model_protocol::ModelProtocolIdentity& identity =
        backend->second.profile->identity();
    log_rejection(context,
                  identity.canonical_model_id,
                  identity.profile_id,
                  body.size(),
                  prepared.error().code);
    completion(error_result(/*status_code=*/400, prepared.error()));
    return;
  }
  if (prepared.value().options.stream) {
    completion(error_result(
        /*status_code=*/400,
        {.message = "streaming Responses are not available on this adapter",
         .param = "stream",
         .code = ErrorCode::UNSUPPORTED_PARAMETER}));
    return;
  }
  auto session = std::make_shared<NonStreamSession>(prepared.value(),
                                                    backend->second.profile,
                                                    std::move(completion),
                                                    limits_,
                                                    body.size());
  backend->second.executor->execute(
      prepared.value(), [session](RequestOutput output) {
        return session->consume(std::move(output));
      });
}

std::shared_ptr<ResponsesStreamControl> ResponsesServiceImpl::process_stream(
    const std::string& body,
    const std::string& content_type,
    responses::RequestContext context,
    std::shared_ptr<ResponsesStreamWriter> writer,
    Completion early_completion) const {
  if (body.size() > limits_.max_body_bytes) {
    early_completion(error_result(
        /*status_code=*/413,
        {.message = "request body exceeds the configured limit",
         .param = "body",
         .code = ErrorCode::REQUEST_TOO_LARGE}));
    return nullptr;
  }
  if (!is_json_content_type(content_type)) {
    early_completion(error_result(
        /*status_code=*/415,
        {.message = "Content-Type must be application/json",
         .param = "Content-Type",
         .code = ErrorCode::UNSUPPORTED_CONTENT_TYPE}));
    return nullptr;
  }
  responses::ModelFieldResult model = responses::read_model_field(body);
  if (!model.ok()) {
    early_completion(error_result(/*status_code=*/400, model.error()));
    return nullptr;
  }
  auto backend = backends_.find(model.model());
  if (backend == backends_.end()) {
    early_completion(error_result(
        /*status_code=*/400,
        {.message = "model has no Responses protocol profile",
         .param = "model",
         .code = ErrorCode::UNSUPPORTED_MODEL_CAPABILITY}));
    return nullptr;
  }
  responses::PrepareResult prepared = responses::prepare_request(
      body, backend->second.profile->identity(), context, limits_);
  if (!prepared.ok()) {
    const model_protocol::ModelProtocolIdentity& identity =
        backend->second.profile->identity();
    log_rejection(context,
                  identity.canonical_model_id,
                  identity.profile_id,
                  body.size(),
                  prepared.error().code);
    early_completion(error_result(/*status_code=*/400, prepared.error()));
    return nullptr;
  }
  if (!prepared.value().options.stream) {
    early_completion(error_result(
        /*status_code=*/400,
        {.message = "stream must be true on the streaming adapter",
         .param = "stream",
         .code = ErrorCode::UNSUPPORTED_PARAMETER}));
    return nullptr;
  }
  if (writer == nullptr) {
    early_completion(error_result(
        /*status_code=*/500,
        {.message = "response stream writer is unavailable",
         .code = ErrorCode::GENERATION_FAILED}));
    return nullptr;
  }
  auto session = std::make_shared<StreamSession>(prepared.value(),
                                                 backend->second.profile,
                                                 backend->second.executor,
                                                 std::move(writer),
                                                 executor_factory_(),
                                                 limits_,
                                                 body.size());
  session->start();
  return session;
}

}  // namespace xllm
