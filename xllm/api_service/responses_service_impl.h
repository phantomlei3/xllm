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
#include <vector>

#include "core/framework/request/request_output.h"
#include "core/model_protocol/profile_registry.h"
#include "responses/request_preparer.h"

namespace xllm {

class LLMMaster;
class ThreadPool;

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
  virtual void cancel(const std::string& /*request_id*/) {}
  virtual void finish_request(const std::string& /*request_id*/) {}
};

class ResponsesSerialExecutor {
 public:
  virtual ~ResponsesSerialExecutor() = default;
  virtual void post(std::function<void()> task) = 0;
};

class ResponsesStreamWriter {
 public:
  using WriteCompletion = std::function<void(bool)>;

  virtual ~ResponsesStreamWriter() = default;
  virtual bool open() = 0;
  virtual void write(std::string frame, WriteCompletion completion) = 0;
  virtual bool writable() const = 0;
  virtual void close() = 0;
  virtual void complete_http() = 0;
};

class ResponsesStreamControl {
 public:
  virtual ~ResponsesStreamControl() = default;
  virtual void deadline() = 0;
  virtual void disconnect() = 0;
  virtual uint64_t pending_bytes() const = 0;
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
  using SerialExecutorFactory =
      std::function<std::shared_ptr<ResponsesSerialExecutor>()>;

  explicit ResponsesServiceImpl(
      responses::ResponsesLimits limits = responses::ResponsesLimits(),
      SerialExecutorFactory executor_factory = {});

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

  std::shared_ptr<ResponsesStreamControl> process_stream(
      const std::string& body,
      const std::string& content_type,
      responses::RequestContext context,
      std::shared_ptr<ResponsesStreamWriter> writer,
      Completion early_completion) const;

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
  std::shared_ptr<ThreadPool> stream_pool_;
  SerialExecutorFactory executor_factory_;
};

}  // namespace xllm
