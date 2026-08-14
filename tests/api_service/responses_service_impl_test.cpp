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

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/model_protocol/deepseek_v4_profile.h"
#include "core/model_protocol/glm_5_2_profile.h"

namespace xllm {
namespace {

model_protocol::LoadedModelContext context_for(
    const model_protocol::ModelProtocolIdentity& identity,
    const std::string& model_id) {
  return {.model_id = model_id,
          .model_type = identity.model_type,
          .model_fingerprint = identity.model_fingerprint,
          .tokenizer_id = identity.tokenizer_id,
          .tokenizer_fingerprint = identity.tokenizer_fingerprint,
          .tokenizer_config_fingerprint = identity.tokenizer_config_fingerprint,
          .template_id = identity.template_id,
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

  const responses::PreparedRequest& request() const { return request_; }

 private:
  std::vector<RequestOutput> outputs_;
  responses::PreparedRequest request_;
};

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
  ResponsesServiceImpl service;
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

TEST(ResponsesServiceImplTest, RejectsBeforeExecutionWithStableEnvelope) {
  const auto profile = model_protocol::make_glm_5_2_profile();
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
  const auto glm = model_protocol::make_glm_5_2_profile();
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
