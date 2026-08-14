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

#include <openssl/evp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
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
#include "responses/json_encoder.h"
#include "responses/output_processor.h"

namespace xllm {
namespace {

using responses::ErrorCode;
using responses::ResponsesError;

ResponsesHttpResult error_result(int32_t status_code, ResponsesError error) {
  return {.status_code = status_code, .body = responses::encode_error(error)};
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

class NonStreamSession final {
 public:
  NonStreamSession(
      const responses::PreparedRequest& request,
      std::shared_ptr<const model_protocol::ModelProtocolProfile> profile,
      ResponsesServiceImpl::Completion completion,
      const responses::ResponsesLimits& limits)
      : parser_(profile->new_parser()),
        processor_({.model = request.canonical_model_id,
                    .created_at =
                        std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count(),
                    .reasoning_effort = request.options.reasoning_effort,
                    .replayed_items = request.canonical_input,
                    .max_output_tokens = request.options.max_output_tokens,
                    .limits = limits}),
        completion_(std::move(completion)) {}

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
    completion_({.status_code = 200,
                 .body = responses::encode_response(processor_.response())});
  }

  std::unique_ptr<model_protocol::ModelOutputParser> parser_;
  model_protocol::RequestOutputNormalizer normalizer_;
  responses::ResponsesProcessor processor_;
  ResponsesServiceImpl::Completion completion_;
  uint64_t ordinal_ = 0;
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

ResponsesServiceImpl::ResponsesServiceImpl(responses::ResponsesLimits limits)
    : limits_(std::move(limits)) {
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
                                                    limits_);
  backend->second.executor->execute(
      prepared.value(), [session](RequestOutput output) {
        return session->consume(std::move(output));
      });
}

}  // namespace xllm
