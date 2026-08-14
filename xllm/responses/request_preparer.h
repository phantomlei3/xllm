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

#include <optional>
#include <string>

#include "core/model_protocol/profile.h"
#include "responses/error.h"
#include "responses/responses_limits.h"
#include "responses/types.h"

namespace xllm::responses {

class ModelFieldResult final {
 public:
  explicit ModelFieldResult(std::string model);
  explicit ModelFieldResult(ResponsesError error);

  bool ok() const { return !model_.empty(); }
  const std::string& model() const { return model_; }
  const ResponsesError& error() const { return error_; }

 private:
  std::string model_;
  ResponsesError error_;
};

class PrepareResult final {
 public:
  explicit PrepareResult(PreparedRequest value);
  explicit PrepareResult(ResponsesError error);

  bool ok() const { return value_.has_value(); }
  const PreparedRequest& value() const { return *value_; }
  const ResponsesError& error() const { return error_; }

 private:
  std::optional<PreparedRequest> value_;
  ResponsesError error_;
};

PrepareResult prepare_request(
    const std::string& body,
    const model_protocol::ModelProtocolIdentity& profile,
    const RequestContext& context,
    const ResponsesLimits& limits = ResponsesLimits());

ModelFieldResult read_model_field(const std::string& body);

}  // namespace xllm::responses
