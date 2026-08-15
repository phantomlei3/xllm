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
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "core/model_protocol/deepseek_v4_profile.h"
#include "core/model_protocol/glm_moe_dsa_profile.h"

namespace xllm {
namespace {

model_protocol::LoadedModelContext context_for(
    const model_protocol::ModelProtocolIdentity& identity,
    const std::string& model_id) {
  return {.model_id = model_id,
          .model_type = identity.model_type,
          .tokenizer_fingerprint = identity.tokenizer_fingerprint,
          .template_fingerprint = identity.template_fingerprint};
}

class FakeResponsesExecutor final : public ResponsesExecutor {
 public:
  explicit FakeResponsesExecutor(std::vector<RequestOutput> outputs)
      : outputs_(std::move(outputs)) {}

  void execute(const responses::PreparedRequest& request,
               OutputCallback callback) override {
    request_ = request;
    for (const RequestOutput& output : outputs_) {
      if (!callback(output)) {
        break;
      }
    }
  }

  void cancel(const std::string& /*request_id*/) override { ++cancel_count; }
  void finish_request(const std::string& /*request_id*/) override {
    ++finish_count;
  }

  const responses::PreparedRequest& request() const { return request_; }
  int32_t cancel_count = 0;
  int32_t finish_count = 0;

 private:
  std::vector<RequestOutput> outputs_;
  responses::PreparedRequest request_;
};

class InlineSerialExecutor final : public ResponsesSerialExecutor {
 public:
  void post(std::function<void()> task) override { task(); }
};

class SafeLogCapture final : public google::LogSink {
 public:
  SafeLogCapture() { google::AddLogSink(this); }
  ~SafeLogCapture() override { google::RemoveLogSink(this); }

  void send(google::LogSeverity /*severity*/,
            const char* /*full_filename*/,
            const char* /*base_filename*/,
            int /*line*/,
            const google::LogMessageTime& /*logmsgtime*/,
            const char* message,
            size_t message_len) override {
    std::lock_guard<std::mutex> lock(mutex_);
    messages_.append(message, message_len);
    messages_.push_back('\n');
  }

  std::string messages() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return messages_;
  }

 private:
  mutable std::mutex mutex_;
  std::string messages_;
};

ResponsesServiceImpl::SerialExecutorFactory inline_executor_factory() {
  return []() { return std::make_shared<InlineSerialExecutor>(); };
}

class FakeStreamWriter final : public ResponsesStreamWriter {
 public:
  bool open() override {
    ++open_count;
    return true;
  }

  void write(std::string frame, WriteCompletion completion) override {
    frames.emplace_back(std::move(frame));
    completions.emplace_back(std::move(completion));
    if (auto_complete) {
      acknowledge(write_success);
    }
  }

  void acknowledge(bool success) {
    ASSERT_FALSE(completions.empty());
    WriteCompletion completion = std::move(completions.front());
    completions.erase(completions.begin());
    completion(success);
  }

  bool writable() const override { return is_writable; }
  void close() override { ++close_count; }
  void complete_http() override { ++http_completion_count; }

  std::vector<std::string> frames;
  std::vector<WriteCompletion> completions;
  bool auto_complete = true;
  bool write_success = true;
  bool is_writable = true;
  int32_t open_count = 0;
  int32_t close_count = 0;
  int32_t http_completion_count = 0;
};

class DeferredResponsesExecutor final : public ResponsesExecutor {
 public:
  void execute(const responses::PreparedRequest& request,
               OutputCallback callback) override {
    request_ = request;
    callback_ = std::move(callback);
  }

  bool emit(RequestOutput output) { return callback_(std::move(output)); }
  void cancel(const std::string& /*request_id*/) override { ++cancel_count; }
  void finish_request(const std::string& /*request_id*/) override {
    ++finish_count;
  }

  responses::PreparedRequest request_;
  OutputCallback callback_;
  int32_t cancel_count = 0;
  int32_t finish_count = 0;
};

std::string joined_frames(const FakeStreamWriter& writer) {
  std::string wire;
  for (const std::string& frame : writer.frames) {
    wire.append(frame);
  }
  return wire;
}

nlohmann::json terminal_response(const FakeStreamWriter& writer) {
  for (auto it = writer.frames.rbegin(); it != writer.frames.rend(); ++it) {
    const size_t data = it->find("\ndata: ");
    if (data == std::string::npos) {
      continue;
    }
    nlohmann::json event =
        nlohmann::json::parse(it->substr(data + 7, it->size() - data - 9));
    if (event.contains("response") && (event["type"] == "response.completed" ||
                                       event["type"] == "response.incomplete" ||
                                       event["type"] == "response.failed")) {
      return event["response"];
    }
  }
  return nullptr;
}

void erase_dynamic_fields(nlohmann::json* value) {
  if (value->is_array()) {
    for (nlohmann::json& child : *value) {
      erase_dynamic_fields(&child);
    }
    return;
  }
  if (!value->is_object()) {
    return;
  }
  value->erase("id");
  value->erase("call_id");
  value->erase("item_id");
  value->erase("created_at");
  for (auto& [key, child] : value->items()) {
    (void)key;
    erase_dynamic_fields(&child);
  }
}

RequestOutput text_output(const std::string& text,
                          const std::string& finish_reason = "stop") {
  RequestOutput output;
  output.outputs.emplace_back(
      SequenceOutput{.index = 0,
                     .text = "</think>" + text + "<｜end▁of▁sentence｜>",
                     .token_ids = {128822, 7, 1},
                     .finish_reason = finish_reason});
  output.usage = Usage{.num_prompt_tokens = 2,
                       .num_generated_tokens = 3,
                       .num_total_tokens = 5,
                       .num_cached_tokens = 0};
  output.finished = true;
  return output;
}

RequestOutput fixture_output(const std::string& profile,
                             const std::string& scenario) {
  const std::filesystem::path path = std::filesystem::path(XLLM_SOURCE_DIR) /
                                     "tests/fixtures/responses/"
                                     "model-protocol" /
                                     (profile + ".json");
  std::ifstream input(path);
  nlohmann::json fixture = nlohmann::json::parse(input);
  for (const nlohmann::json& candidate : fixture["scenarios"]) {
    if (candidate["scenario_id"] != scenario) {
      continue;
    }
    const nlohmann::json& raw = candidate["raw_generation"];
    const std::vector<int32_t> token_ids =
        raw["token_ids"].get<std::vector<int32_t>>();
    RequestOutput output;
    output.outputs.emplace_back(SequenceOutput{
        .index = 0,
        .text = raw["text"].get<std::string>(),
        .token_ids = token_ids,
        .finish_reason = raw["finish_reason"].get<std::string>()});
    const int32_t output_tokens = static_cast<int32_t>(token_ids.size());
    output.usage = Usage{.num_prompt_tokens = 2,
                         .num_generated_tokens = output_tokens,
                         .num_total_tokens = 2 + output_tokens,
                         .num_cached_tokens = 0};
    output.finished = true;
    return output;
  }
  return {};
}

TEST(ResponsesServiceImplTest, CompletesThroughSharedPipeline) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  FakeResponsesExecutor executor({text_output("hello")});
  ResponsesServiceImpl service(responses::ResponsesLimits(),
                               inline_executor_factory());
  ASSERT_TRUE(service.add_model(context_for(profile->identity(), "deepseek-v4"),
                                &executor));

  ResponsesHttpResult result;
  service.process_non_stream(
      R"({"model":"deepseek-v4","input":"hi","stream":false})",
      "application/json",
      {.request_id = "request-1", .trace_id = "trace-1"},
      [&result](ResponsesHttpResult value) { result = std::move(value); });

  EXPECT_EQ(result.status_code, 200);
  EXPECT_EQ(result.body["status"], "completed");
  EXPECT_EQ(result.body["model"], "deepseek-v4");
  ASSERT_EQ(result.body["output"].size(), 1);
  EXPECT_EQ(result.body["output"][0]["content"][0]["text"], "hello");
  EXPECT_EQ(executor.request().chat_request.thinking_history_policy(),
            proto::PRESERVE);
  EXPECT_EQ(executor.request().chat_request.output_decoding_policy(),
            proto::PROTOCOL_RAW);
  EXPECT_EQ(executor.request().sequence_count, 1);
}

TEST(ResponsesServiceImplTest, BindsAndEchoesDeploymentModelId) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  FakeResponsesExecutor executor({text_output("hello")});
  ResponsesServiceImpl service(responses::ResponsesLimits(),
                               inline_executor_factory());
  ASSERT_TRUE(service.add_model(
      context_for(profile->identity(), "deepseek-v4-flash-w8a8"), &executor));

  ResponsesHttpResult result;
  service.process_non_stream(
      R"({"model":"deepseek-v4-flash-w8a8","input":"hi"})",
      "application/json",
      {.request_id = "request-deployment", .trace_id = "trace-deployment"},
      [&result](ResponsesHttpResult value) { result = std::move(value); });

  ASSERT_EQ(result.status_code, 200) << result.body.dump();
  EXPECT_EQ(result.body["model"], "deepseek-v4-flash-w8a8");
  EXPECT_EQ(executor.request().chat_request.model(), "deepseek-v4-flash-w8a8");

  auto writer = std::make_shared<FakeStreamWriter>();
  ASSERT_NE(
      service.process_stream(
          R"({"model":"deepseek-v4-flash-w8a8","input":"hi","stream":true})",
          "application/json",
          {},
          writer,
          [](ResponsesHttpResult /*unused*/) {}),
      nullptr);
  EXPECT_EQ(terminal_response(*writer)["model"], "deepseek-v4-flash-w8a8");

  ResponsesHttpResult historical_name;
  service.process_non_stream(R"({"model":"deepseek-v4","input":"hi"})",
                             "application/json",
                             {},
                             [&historical_name](ResponsesHttpResult value) {
                               historical_name = std::move(value);
                             });
  EXPECT_EQ(historical_name.status_code, 400);
  EXPECT_EQ(historical_name.body["error"]["code"],
            "unsupported_model_capability");
}

TEST(ResponsesServiceImplTest, BindsGlmMoeDsaDeploymentModelId) {
  const auto profile = model_protocol::make_glm_moe_dsa_profile();
  FakeResponsesExecutor executor({text_output("hello")});
  ResponsesServiceImpl service(responses::ResponsesLimits(),
                               inline_executor_factory());
  ASSERT_TRUE(service.add_model(
      context_for(profile->identity(), "glm-5.2-flash-w4a8"), &executor));

  ResponsesHttpResult result;
  service.process_non_stream(
      R"({"model":"glm-5.2-flash-w4a8","input":"hi"})",
      "application/json",
      {},
      [&result](ResponsesHttpResult value) { result = std::move(value); });

  ASSERT_EQ(result.status_code, 200) << result.body.dump();
  EXPECT_EQ(result.body["model"], "glm-5.2-flash-w4a8");
  EXPECT_EQ(executor.request().profile_id, "glm_moe_dsa_responses");
}

TEST(ResponsesServiceImplTest, SuccessLogsOnlySafeResponseMetadata) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  FakeResponsesExecutor executor({text_output("GENERATED_SECRET")});
  ResponsesServiceImpl service(responses::ResponsesLimits(),
                               inline_executor_factory());
  ASSERT_TRUE(service.add_model(context_for(profile->identity(), "deepseek-v4"),
                                &executor));
  SafeLogCapture logs;
  ResponsesHttpResult result;

  service.process_non_stream(
      R"({"model":"deepseek-v4","input":[{"type":"message","role":"user","content":"PROMPT_SECRET"},{"type":"reasoning","id":"rs_safe","content":"REASONING_SECRET"},{"type":"function_call","id":"fc_safe","call_id":"call_safe","name":"read_fixture","arguments":"{\"secret\":\"ARGUMENT_SECRET\"}"},{"type":"function_call_output","call_id":"call_safe","output":"TOOL_OUTPUT_SECRET"},{"type":"custom_tool_call","id":"ctc_safe","call_id":"call_patch","name":"apply_patch","input":"PATCH_SECRET"},{"type":"custom_tool_call_output","call_id":"call_patch","output":"PATCH_OUTPUT_SECRET"}],"tools":[{"type":"function","name":"read_fixture","parameters":{}},{"type":"custom","name":"apply_patch"}]})",
      "application/json",
      {.request_id = "request-safe", .trace_id = "trace-safe"},
      [&result](ResponsesHttpResult value) { result = std::move(value); });

  ASSERT_EQ(result.status_code, 200);
  const std::string captured = logs.messages();
  EXPECT_NE(captured.find("trace_id=trace-safe"), std::string::npos);
  EXPECT_NE(captured.find("request_id=request-safe"), std::string::npos);
  EXPECT_NE(captured.find("model=deepseek-v4"), std::string::npos);
  EXPECT_NE(captured.find("profile=deepseek_v4_responses"), std::string::npos);
  EXPECT_NE(captured.find("status=completed"), std::string::npos);
  EXPECT_NE(captured.find("input_tokens=2"), std::string::npos);
  for (const std::string& secret : {"PROMPT_SECRET",
                                    "REASONING_SECRET",
                                    "ARGUMENT_SECRET",
                                    "TOOL_OUTPUT_SECRET",
                                    "PATCH_SECRET",
                                    "PATCH_OUTPUT_SECRET",
                                    "GENERATED_SECRET"}) {
    EXPECT_EQ(captured.find(secret), std::string::npos) << secret;
  }
}

TEST(ResponsesServiceImplTest, ErrorLogsOnlySafeRequestMetadata) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  FakeResponsesExecutor executor({text_output("unused")});
  ResponsesServiceImpl service(responses::ResponsesLimits(),
                               inline_executor_factory());
  ASSERT_TRUE(service.add_model(context_for(profile->identity(), "deepseek-v4"),
                                &executor));
  SafeLogCapture logs;
  ResponsesHttpResult result;

  service.process_non_stream(
      R"({"model":"deepseek-v4","input":"ERROR_PROMPT_SECRET","forbidden":"ERROR_BODY_SECRET"})",
      "application/json",
      {.request_id = "request-error", .trace_id = "trace-error"},
      [&result](ResponsesHttpResult value) { result = std::move(value); });

  ASSERT_EQ(result.status_code, 400);
  const std::string captured = logs.messages();
  EXPECT_NE(captured.find("trace_id=trace-error"), std::string::npos);
  EXPECT_NE(captured.find("request_id=request-error"), std::string::npos);
  EXPECT_NE(captured.find("model=deepseek-v4"), std::string::npos);
  EXPECT_NE(captured.find("profile=deepseek_v4_responses"), std::string::npos);
  EXPECT_NE(captured.find("status=unsupported_parameter"), std::string::npos);
  EXPECT_EQ(captured.find("ERROR_PROMPT_SECRET"), std::string::npos);
  EXPECT_EQ(captured.find("ERROR_BODY_SECRET"), std::string::npos);
}

TEST(ResponsesServiceImplTest, StreamsFormalTerminalAndClosesExactlyOnce) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  FakeResponsesExecutor executor({text_output("hello")});
  ResponsesServiceImpl service(responses::ResponsesLimits(),
                               inline_executor_factory());
  ASSERT_TRUE(service.add_model(context_for(profile->identity(), "deepseek-v4"),
                                &executor));
  auto writer = std::make_shared<FakeStreamWriter>();
  ResponsesHttpResult early_error;

  std::shared_ptr<ResponsesStreamControl> stream = service.process_stream(
      R"({"model":"deepseek-v4","input":"hi","stream":true})",
      "application/json",
      {.request_id = "request-stream", .trace_id = "trace-stream"},
      writer,
      [&early_error](ResponsesHttpResult value) {
        early_error = std::move(value);
      });

  ASSERT_NE(stream, nullptr);
  ASSERT_FALSE(writer->frames.empty());
  const std::string wire = joined_frames(*writer);
  EXPECT_NE(wire.find("event: response.created"), std::string::npos);
  EXPECT_NE(wire.find("event: response.completed"), std::string::npos);
  EXPECT_EQ(wire.find("[DONE]"), std::string::npos);
  EXPECT_TRUE(early_error.body.is_null());
  EXPECT_EQ(writer->open_count, 1);
  EXPECT_EQ(writer->close_count, 1);
  EXPECT_EQ(writer->http_completion_count, 1);
  EXPECT_EQ(executor.finish_count, 1);
  EXPECT_EQ(executor.cancel_count, 0);
}

TEST(ResponsesServiceImplTest, SeparatesEarlyJsonFromOpenedStreamFailure) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  DeferredResponsesExecutor executor;
  ResponsesServiceImpl service(responses::ResponsesLimits(),
                               inline_executor_factory());
  ASSERT_TRUE(service.add_model(context_for(profile->identity(), "deepseek-v4"),
                                &executor));

  auto early_writer = std::make_shared<FakeStreamWriter>();
  ResponsesHttpResult early_error;
  EXPECT_EQ(service.process_stream("{",
                                   "application/json",
                                   {},
                                   early_writer,
                                   [&early_error](ResponsesHttpResult value) {
                                     early_error = std::move(value);
                                   }),
            nullptr);
  EXPECT_EQ(early_error.status_code, 400);
  EXPECT_EQ(early_error.body["error"]["code"], "invalid_json");
  EXPECT_EQ(early_writer->open_count, 0);

  auto stream_writer = std::make_shared<FakeStreamWriter>();
  ASSERT_NE(service.process_stream(
                R"({"model":"deepseek-v4","input":"hi","stream":true})",
                "application/json",
                {},
                stream_writer,
                [](ResponsesHttpResult /*unused*/) {}),
            nullptr);
  EXPECT_FALSE(executor.emit(
      RequestOutput(Status(StatusCode::UNKNOWN, "backend unavailable"))));
  const std::string wire = joined_frames(*stream_writer);
  const size_t failed = wire.find("event: response.failed");
  ASSERT_NE(failed, std::string::npos);
  EXPECT_EQ(wire.find("event: response.failed", failed + 1), std::string::npos);
  EXPECT_EQ(wire.find("event: response.completed"), std::string::npos);
  EXPECT_EQ(executor.cancel_count, 1);
  EXPECT_EQ(stream_writer->close_count, 1);
  EXPECT_EQ(stream_writer->http_completion_count, 1);
}

TEST(ResponsesServiceImplTest, SlowClientFailsOnceAndCancelsGeneration) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  FakeResponsesExecutor executor({text_output(std::string(8192, 'x'))});
  responses::ResponsesLimits limits;
  limits.max_sse_buffer_bytes = 4096;
  ResponsesServiceImpl service(limits, inline_executor_factory());
  ASSERT_TRUE(service.add_model(context_for(profile->identity(), "deepseek-v4"),
                                &executor));
  auto writer = std::make_shared<FakeStreamWriter>();
  writer->auto_complete = false;

  std::shared_ptr<ResponsesStreamControl> stream = service.process_stream(
      R"({"model":"deepseek-v4","input":"hi","stream":true})",
      "application/json",
      {},
      writer,
      [](ResponsesHttpResult /*unused*/) {});

  ASSERT_NE(stream, nullptr);
  EXPECT_LE(stream->pending_bytes(), limits.max_sse_buffer_bytes);
  EXPECT_EQ(executor.cancel_count, 1);
  while (!writer->completions.empty()) {
    writer->acknowledge(true);
  }
  const std::string wire = joined_frames(*writer);
  EXPECT_NE(wire.find("event: response.failed"), std::string::npos);
  EXPECT_NE(wire.find("client_too_slow"), std::string::npos);
  EXPECT_EQ(wire.find("event: response.completed"), std::string::npos);
  EXPECT_EQ(writer->close_count, 1);
  EXPECT_EQ(writer->http_completion_count, 1);
}

TEST(ResponsesServiceImplTest, WriteFailureCancelsWithoutAnotherWrite) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  DeferredResponsesExecutor executor;
  ResponsesServiceImpl service(responses::ResponsesLimits(),
                               inline_executor_factory());
  ASSERT_TRUE(service.add_model(context_for(profile->identity(), "deepseek-v4"),
                                &executor));
  auto writer = std::make_shared<FakeStreamWriter>();
  writer->write_success = false;
  writer->is_writable = false;

  std::shared_ptr<ResponsesStreamControl> stream = service.process_stream(
      R"({"model":"deepseek-v4","input":"hi","stream":true})",
      "application/json",
      {},
      writer,
      [](ResponsesHttpResult /*unused*/) {});

  ASSERT_NE(stream, nullptr);
  EXPECT_EQ(writer->frames.size(), 1);
  EXPECT_EQ(executor.cancel_count, 1);
  EXPECT_EQ(writer->close_count, 1);
  EXPECT_EQ(writer->http_completion_count, 1);
}

TEST(ResponsesServiceImplTest, DisconnectAndLastChunkHonorArrivalOrder) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  DeferredResponsesExecutor finish_first_executor;
  ResponsesServiceImpl finish_first_service(responses::ResponsesLimits(),
                                            inline_executor_factory());
  ASSERT_TRUE(finish_first_service.add_model(
      context_for(profile->identity(), "deepseek-v4"), &finish_first_executor));
  auto finish_first_writer = std::make_shared<FakeStreamWriter>();
  std::shared_ptr<ResponsesStreamControl> finish_first =
      finish_first_service.process_stream(
          R"({"model":"deepseek-v4","input":"hi","stream":true})",
          "application/json",
          {},
          finish_first_writer,
          [](ResponsesHttpResult /*unused*/) {});
  ASSERT_NE(finish_first, nullptr);
  EXPECT_FALSE(finish_first_executor.emit(text_output("done")));
  finish_first->disconnect();
  EXPECT_NE(
      joined_frames(*finish_first_writer).find("event: response.completed"),
      std::string::npos);
  EXPECT_EQ(finish_first_executor.finish_count, 1);
  EXPECT_EQ(finish_first_executor.cancel_count, 0);

  DeferredResponsesExecutor disconnect_first_executor;
  ResponsesServiceImpl disconnect_first_service(responses::ResponsesLimits(),
                                                inline_executor_factory());
  ASSERT_TRUE(disconnect_first_service.add_model(
      context_for(profile->identity(), "deepseek-v4"),
      &disconnect_first_executor));
  auto disconnect_first_writer = std::make_shared<FakeStreamWriter>();
  std::shared_ptr<ResponsesStreamControl> disconnect_first =
      disconnect_first_service.process_stream(
          R"({"model":"deepseek-v4","input":"hi","stream":true})",
          "application/json",
          {},
          disconnect_first_writer,
          [](ResponsesHttpResult /*unused*/) {});
  ASSERT_NE(disconnect_first, nullptr);
  disconnect_first->disconnect();
  EXPECT_FALSE(disconnect_first_executor.emit(text_output("late")));
  EXPECT_EQ(
      joined_frames(*disconnect_first_writer).find("event: response.completed"),
      std::string::npos);
  EXPECT_EQ(disconnect_first_executor.finish_count, 0);
  EXPECT_EQ(disconnect_first_executor.cancel_count, 1);
}

TEST(ResponsesServiceImplTest, TimeoutAndLastChunkHonorArrivalOrder) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  DeferredResponsesExecutor timeout_first_executor;
  ResponsesServiceImpl timeout_first_service(responses::ResponsesLimits(),
                                             inline_executor_factory());
  ASSERT_TRUE(timeout_first_service.add_model(
      context_for(profile->identity(), "deepseek-v4"),
      &timeout_first_executor));
  auto timeout_first_writer = std::make_shared<FakeStreamWriter>();
  std::shared_ptr<ResponsesStreamControl> timeout_first =
      timeout_first_service.process_stream(
          R"({"model":"deepseek-v4","input":"hi","stream":true})",
          "application/json",
          {},
          timeout_first_writer,
          [](ResponsesHttpResult /*unused*/) {});
  ASSERT_NE(timeout_first, nullptr);

  timeout_first->deadline();
  EXPECT_FALSE(timeout_first_executor.emit(text_output("late")));
  const std::string timeout_wire = joined_frames(*timeout_first_writer);
  EXPECT_NE(timeout_wire.find("event: response.failed"), std::string::npos);
  EXPECT_NE(timeout_wire.find("request_timeout"), std::string::npos);
  EXPECT_EQ(timeout_wire.find("event: response.completed"), std::string::npos);
  EXPECT_EQ(timeout_first_executor.cancel_count, 1);

  DeferredResponsesExecutor finish_first_executor;
  ResponsesServiceImpl finish_first_service(responses::ResponsesLimits(),
                                            inline_executor_factory());
  ASSERT_TRUE(finish_first_service.add_model(
      context_for(profile->identity(), "deepseek-v4"), &finish_first_executor));
  auto finish_first_writer = std::make_shared<FakeStreamWriter>();
  std::shared_ptr<ResponsesStreamControl> finish_first =
      finish_first_service.process_stream(
          R"({"model":"deepseek-v4","input":"hi","stream":true})",
          "application/json",
          {},
          finish_first_writer,
          [](ResponsesHttpResult /*unused*/) {});
  ASSERT_NE(finish_first, nullptr);

  EXPECT_FALSE(finish_first_executor.emit(text_output("done")));
  finish_first->deadline();
  const std::string finish_wire = joined_frames(*finish_first_writer);
  EXPECT_NE(finish_wire.find("event: response.completed"), std::string::npos);
  EXPECT_EQ(finish_wire.find("event: response.failed"), std::string::npos);
  EXPECT_EQ(finish_first_executor.finish_count, 1);
  EXPECT_EQ(finish_first_executor.cancel_count, 0);
}

TEST(ResponsesServiceImplTest, FixtureStreamsReaggregateLikeNonStream) {
  struct Scenario {
    std::string profile;
    std::string model;
    std::string scenario;
    std::string expected_event;
  };
  const std::vector<Scenario> scenarios = {
      {.profile = "deepseek-v4",
       .model = "deepseek-v4",
       .scenario = "reasoning_text_stop",
       .expected_event = "response.output_text.delta"},
      {.profile = "deepseek-v4",
       .model = "deepseek-v4",
       .scenario = "reasoning_function_call",
       .expected_event = "response.function_call_arguments.delta"},
      {.profile = "deepseek-v4",
       .model = "deepseek-v4",
       .scenario = "parallel_function_calls",
       .expected_event = "response.function_call_arguments.delta"},
      {.profile = "deepseek-v4",
       .model = "deepseek-v4",
       .scenario = "reasoning_apply_patch",
       .expected_event = "response.custom_tool_call_input.delta"},
      {.profile = "deepseek-v4",
       .model = "deepseek-v4",
       .scenario = "reasoning_truncated",
       .expected_event = "response.incomplete"},
      {.profile = "deepseek-v4",
       .model = "deepseek-v4",
       .scenario = "function_output_continue",
       .expected_event = "response.output_text.delta"},
      {.profile = "deepseek-v4",
       .model = "deepseek-v4",
       .scenario = "custom_output_continue",
       .expected_event = "response.output_text.delta"},
      {.profile = "glm-5.2",
       .model = "GLM-5-W4A8",
       .scenario = "reasoning_text_stop",
       .expected_event = "response.reasoning_text.delta"},
      {.profile = "glm-5.2",
       .model = "GLM-5-W4A8",
       .scenario = "reasoning_function_call",
       .expected_event = "response.function_call_arguments.delta"},
      {.profile = "glm-5.2",
       .model = "GLM-5-W4A8",
       .scenario = "parallel_function_calls",
       .expected_event = "response.function_call_arguments.delta"},
      {.profile = "glm-5.2",
       .model = "GLM-5-W4A8",
       .scenario = "reasoning_apply_patch",
       .expected_event = "response.custom_tool_call_input.delta"},
      {.profile = "glm-5.2",
       .model = "GLM-5-W4A8",
       .scenario = "function_output_continue",
       .expected_event = "response.output_text.delta"},
      {.profile = "glm-5.2",
       .model = "GLM-5-W4A8",
       .scenario = "custom_output_continue",
       .expected_event = "response.output_text.delta"},
      {.profile = "glm-5.2",
       .model = "GLM-5-W4A8",
       .scenario = "reasoning_truncated",
       .expected_event = "response.incomplete"}};

  for (const Scenario& scenario : scenarios) {
    std::shared_ptr<const model_protocol::ModelProtocolProfile> profile =
        scenario.profile == "deepseek-v4"
            ? model_protocol::make_deepseek_v4_profile()
            : model_protocol::make_glm_moe_dsa_profile();
    const RequestOutput output =
        fixture_output(scenario.profile, scenario.scenario);
    FakeResponsesExecutor non_stream_executor({output});
    ResponsesServiceImpl non_stream_service(responses::ResponsesLimits(),
                                            inline_executor_factory());
    ASSERT_TRUE(non_stream_service.add_model(
        context_for(profile->identity(), scenario.model),
        &non_stream_executor));
    const std::string body =
        "{\"model\":\"" + scenario.model + "\",\"input\":\"hi\"}";
    ResponsesHttpResult non_stream;
    non_stream_service.process_non_stream(
        body, "application/json", {}, [&non_stream](ResponsesHttpResult value) {
          non_stream = std::move(value);
        });

    FakeResponsesExecutor stream_executor({output});
    ResponsesServiceImpl stream_service(responses::ResponsesLimits(),
                                        inline_executor_factory());
    ASSERT_TRUE(stream_service.add_model(
        context_for(profile->identity(), scenario.model), &stream_executor));
    auto writer = std::make_shared<FakeStreamWriter>();
    const std::string stream_body = "{\"model\":\"" + scenario.model +
                                    "\",\"input\":\"hi\",\"stream\":true}";
    ASSERT_NE(
        stream_service.process_stream(stream_body,
                                      "application/json",
                                      {},
                                      writer,
                                      [](ResponsesHttpResult /*unused*/) {}),
        nullptr)
        << scenario.profile << "/" << scenario.scenario;
    EXPECT_NE(joined_frames(*writer).find(scenario.expected_event),
              std::string::npos)
        << scenario.profile << "/" << scenario.scenario;
    nlohmann::json streamed = terminal_response(*writer);
    ASSERT_FALSE(streamed.is_null())
        << scenario.profile << "/" << scenario.scenario;
    nlohmann::json expected = non_stream.body;
    erase_dynamic_fields(&streamed);
    erase_dynamic_fields(&expected);
    EXPECT_EQ(streamed, expected)
        << scenario.profile << "/" << scenario.scenario;
  }
}

TEST(ResponsesServiceImplTest, RejectsBeforeExecutionWithStableEnvelope) {
  const auto profile = model_protocol::make_glm_moe_dsa_profile();
  FakeResponsesExecutor executor({text_output("unused")});
  responses::ResponsesLimits limits;
  limits.max_body_bytes = 8;
  ResponsesServiceImpl service(limits);
  ASSERT_TRUE(service.add_model(context_for(profile->identity(), "GLM-5-W4A8"),
                                &executor));

  ResponsesHttpResult result;
  service.process_non_stream(
      "123456789",
      "application/json",
      {.request_id = "request-2", .trace_id = "trace-2"},
      [&result](ResponsesHttpResult value) { result = std::move(value); });

  EXPECT_EQ(result.status_code, 413);
  EXPECT_EQ(result.body["error"]["code"], "request_too_large");
  EXPECT_EQ(result.body["error"]["param"], "body");

  responses::ResponsesLimits parse_limits;
  ResponsesServiceImpl parse_service(parse_limits);
  ASSERT_TRUE(parse_service.add_model(
      context_for(profile->identity(), "GLM-5-W4A8"), &executor));
  ResponsesHttpResult invalid_json;
  parse_service.process_non_stream(
      "{", "application/json", {}, [&invalid_json](ResponsesHttpResult value) {
        invalid_json = std::move(value);
      });
  EXPECT_EQ(invalid_json.status_code, 400);
  EXPECT_EQ(invalid_json.body["error"]["code"], "invalid_json");
}

TEST(ResponsesServiceImplTest, UnknownModelDoesNotAffectRegisteredModel) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  FakeResponsesExecutor executor({text_output("ok")});
  ResponsesServiceImpl service;
  ASSERT_TRUE(service.add_model(context_for(profile->identity(), "deepseek-v4"),
                                &executor));

  ResponsesHttpResult unknown;
  service.process_non_stream(
      R"({"model":"unknown","input":"hi"})",
      "application/json",
      {},
      [&unknown](ResponsesHttpResult value) { unknown = std::move(value); });
  EXPECT_EQ(unknown.status_code, 400);
  EXPECT_EQ(unknown.body["error"]["code"], "unsupported_model_capability");

  ResponsesHttpResult known;
  service.process_non_stream(
      R"({"model":"deepseek-v4","input":"hi"})",
      "application/json",
      {},
      [&known](ResponsesHttpResult value) { known = std::move(value); });
  EXPECT_EQ(known.status_code, 200);
}

TEST(ResponsesServiceImplTest, RejectsIdentityMismatchAsDeploymentError) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  FakeResponsesExecutor executor({text_output("unused")});
  model_protocol::LoadedModelContext context =
      context_for(profile->identity(), "deepseek-v4");
  context.template_fingerprint = "sha256:mismatch";
  ResponsesServiceImpl service;

  EXPECT_FALSE(service.add_model(context, &executor));
  EXPECT_TRUE(service.deployment_error().has_value());
}

TEST(ResponsesServiceImplTest, RejectsContentTypeAndStreamingRequest) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  FakeResponsesExecutor executor({text_output("unused")});
  ResponsesServiceImpl service;
  ASSERT_TRUE(service.add_model(context_for(profile->identity(), "deepseek-v4"),
                                &executor));

  ResponsesHttpResult content_type;
  service.process_non_stream(R"({"model":"deepseek-v4","input":"hi"})",
                             "text/plain",
                             {},
                             [&content_type](ResponsesHttpResult value) {
                               content_type = std::move(value);
                             });
  EXPECT_EQ(content_type.status_code, 415);
  EXPECT_EQ(content_type.body["error"]["code"], "unsupported_content_type");

  ResponsesHttpResult stream;
  service.process_non_stream(
      R"({"model":"deepseek-v4","input":"hi","stream":true})",
      "application/json",
      {},
      [&stream](ResponsesHttpResult value) { stream = std::move(value); });
  EXPECT_EQ(stream.status_code, 400);
  EXPECT_EQ(stream.body["error"]["code"], "unsupported_parameter");
  EXPECT_EQ(stream.body["error"]["param"], "stream");

  ResponsesHttpResult semantic;
  service.process_non_stream(
      R"({"model":"deepseek-v4","input":42})",
      "application/json",
      {},
      [&semantic](ResponsesHttpResult value) { semantic = std::move(value); });
  EXPECT_EQ(semantic.status_code, 400);
  EXPECT_EQ(semantic.body["error"]["code"], "invalid_request");
  EXPECT_EQ(semantic.body["error"]["param"], "input");

  ResponsesHttpResult client_profile;
  service.process_non_stream(
      R"({"model":"deepseek-v4","input":"hi","profile":"glm-5.2"})",
      "application/json",
      {},
      [&client_profile](ResponsesHttpResult value) {
        client_profile = std::move(value);
      });
  EXPECT_EQ(client_profile.status_code, 400);
  EXPECT_EQ(client_profile.body["error"]["code"], "unsupported_parameter");
  EXPECT_EQ(client_profile.body["error"]["param"], "profile");
}

TEST(ResponsesServiceImplTest, ProjectsIncompleteAndBackendFailure) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  FakeResponsesExecutor incomplete_executor({text_output("partial", "length")});
  ResponsesServiceImpl incomplete_service;
  ASSERT_TRUE(incomplete_service.add_model(
      context_for(profile->identity(), "deepseek-v4"), &incomplete_executor));
  ResponsesHttpResult incomplete;
  incomplete_service.process_non_stream(
      R"({"model":"deepseek-v4","input":"hi","max_output_tokens":3})",
      "application/json",
      {},
      [&incomplete](ResponsesHttpResult value) {
        incomplete = std::move(value);
      });
  EXPECT_EQ(incomplete.status_code, 200);
  EXPECT_EQ(incomplete.body["status"], "incomplete");
  EXPECT_EQ(incomplete.body["incomplete_details"]["reason"],
            "max_output_tokens");

  RequestOutput failure(Status(StatusCode::UNKNOWN, "backend unavailable"));
  FakeResponsesExecutor failed_executor({std::move(failure)});
  ResponsesServiceImpl failed_service;
  ASSERT_TRUE(failed_service.add_model(
      context_for(profile->identity(), "deepseek-v4"), &failed_executor));
  ResponsesHttpResult failed;
  failed_service.process_non_stream(
      R"({"model":"deepseek-v4","input":"hi"})",
      "application/json",
      {},
      [&failed](ResponsesHttpResult value) { failed = std::move(value); });
  EXPECT_EQ(failed.status_code, 200);
  EXPECT_EQ(failed.body["status"], "failed");
  EXPECT_EQ(failed.body["error"]["code"], "generation_failed");
}

TEST(ResponsesServiceImplTest, FailsClosedOnMultipleSequences) {
  const auto profile = model_protocol::make_deepseek_v4_profile();
  RequestOutput output = text_output("first");
  output.outputs.emplace_back(SequenceOutput{
      .index = 1, .text = "second", .token_ids = {8}, .finish_reason = "stop"});
  FakeResponsesExecutor executor({std::move(output)});
  ResponsesServiceImpl service;
  ASSERT_TRUE(service.add_model(context_for(profile->identity(), "deepseek-v4"),
                                &executor));
  ResponsesHttpResult result;
  service.process_non_stream(
      R"({"model":"deepseek-v4","input":"hi"})",
      "application/json",
      {},
      [&result](ResponsesHttpResult value) { result = std::move(value); });
  EXPECT_EQ(result.status_code, 200);
  EXPECT_EQ(result.body["status"], "failed");
  EXPECT_EQ(result.body["error"]["code"], "generation_failed");
}

TEST(ResponsesServiceImplTest, SupportsBothProfilesAndTypedTools) {
  const auto glm = model_protocol::make_glm_moe_dsa_profile();
  FakeResponsesExecutor glm_executor(
      {fixture_output("glm-5.2", "reasoning_text_stop")});
  ResponsesServiceImpl glm_service;
  ASSERT_TRUE(glm_service.add_model(context_for(glm->identity(), "GLM-5-W4A8"),
                                    &glm_executor));
  ResponsesHttpResult glm_result;
  glm_service.process_non_stream(R"({"model":"GLM-5-W4A8","input":"hi"})",
                                 "application/json",
                                 {},
                                 [&glm_result](ResponsesHttpResult value) {
                                   glm_result = std::move(value);
                                 });
  EXPECT_EQ(glm_result.body["status"], "completed") << glm_result.body.dump();
  EXPECT_EQ(glm_result.body["model"], "GLM-5-W4A8");

  const auto deepseek = model_protocol::make_deepseek_v4_profile();
  FakeResponsesExecutor function_executor(
      {fixture_output("deepseek-v4", "reasoning_function_call")});
  ResponsesServiceImpl function_service;
  ASSERT_TRUE(function_service.add_model(
      context_for(deepseek->identity(), "deepseek-v4"), &function_executor));
  ResponsesHttpResult function_result;
  function_service.process_non_stream(
      R"({"model":"deepseek-v4","input":"read","tools":[{"type":"function","name":"get_weather","description":"read","parameters":{"type":"object"}}]})",
      "application/json",
      {},
      [&function_result](ResponsesHttpResult value) {
        function_result = std::move(value);
      });
  EXPECT_EQ(function_result.body["status"], "completed")
      << function_result.body.dump();
  ASSERT_EQ(function_result.body["output"].size(), 2);
  EXPECT_EQ(function_result.body["output"][1]["type"], "function_call");

  FakeResponsesExecutor patch_executor(
      {fixture_output("deepseek-v4", "reasoning_apply_patch")});
  ResponsesServiceImpl patch_service;
  ASSERT_TRUE(patch_service.add_model(
      context_for(deepseek->identity(), "deepseek-v4"), &patch_executor));
  ResponsesHttpResult patch_result;
  patch_service.process_non_stream(
      R"({"model":"deepseek-v4","input":"patch","tools":[{"type":"custom","name":"apply_patch"}]})",
      "application/json",
      {},
      [&patch_result](ResponsesHttpResult value) {
        patch_result = std::move(value);
      });
  EXPECT_EQ(patch_result.body["status"], "completed");
  ASSERT_EQ(patch_result.body["output"].size(), 2);
  EXPECT_EQ(patch_result.body["output"][1]["type"], "custom_tool_call");
}

}  // namespace
}  // namespace xllm
