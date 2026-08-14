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

#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <unordered_map>

#include "core/framework/request/request_output.h"
#include "core/model_protocol/profile_registry.h"
#include "responses/request_preparer.h"

namespace xllm {

class LLMMaster;

model_protocol::LoadedModelContext inspect_responses_model(
    const std::string& model_id,
    const LLMMaster& master);

struct ResponsesHttpResult {
  int32_t status_code = 200;
  nlohmann::json body;
};

class ResponsesExecutor {
 public:
  virtual ~ResponsesExecutor() = default;
  virtual void execute(const responses::PreparedRequest& request,
                       OutputCallback callback) = 0;
};

class LLMMasterResponsesExecutor final : public ResponsesExecutor {
 public:
  explicit LLMMasterResponsesExecutor(LLMMaster* master);
  void execute(const responses::PreparedRequest& request,
               OutputCallback callback) override;

 private:
  LLMMaster* master_;
};

class ResponsesServiceImpl final {
 public:
  using Completion = std::function<void(ResponsesHttpResult)>;

  explicit ResponsesServiceImpl(
      responses::ResponsesLimits limits = responses::ResponsesLimits());

  bool add_model(const model_protocol::LoadedModelContext& context,
                 ResponsesExecutor* executor);
  bool add_model(const model_protocol::LoadedModelContext& context,
                 LLMMaster* master);
  const std::optional<std::string>& deployment_error() const {
    return deployment_error_;
  }

  void process_non_stream(const std::string& body,
                          const std::string& content_type,
                          responses::RequestContext context,
                          Completion completion) const;

 private:
  struct Backend {
    std::shared_ptr<const model_protocol::ModelProtocolProfile> profile;
    ResponsesExecutor* executor;
  };

  model_protocol::ProfileRegistry registry_;
  responses::ResponsesLimits limits_;
  std::unordered_map<std::string, Backend> backends_;
  std::vector<std::unique_ptr<ResponsesExecutor>> owned_executors_;
  std::optional<std::string> deployment_error_;
};

}  // namespace xllm
