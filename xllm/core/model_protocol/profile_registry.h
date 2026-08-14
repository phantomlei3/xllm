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
#include <string>
#include <unordered_map>

#include "core/model_protocol/profile.h"

namespace xllm::model_protocol {

enum class ModelProtocolErrorCode : uint8_t {
  NONE = 0,
  UNSUPPORTED_MODEL_CAPABILITY = 1,
  PROFILE_IDENTITY_MISMATCH = 2,
  DUPLICATE_PROFILE_IDENTITY = 3,
};

class ModelProtocolError final {
 public:
  ModelProtocolError() = default;
  ModelProtocolError(ModelProtocolErrorCode code, std::string message);

  bool ok() const { return code_ == ModelProtocolErrorCode::NONE; }
  ModelProtocolErrorCode code() const { return code_; }
  const std::string& message() const { return message_; }

 private:
  ModelProtocolErrorCode code_ = ModelProtocolErrorCode::NONE;
  std::string message_;
};

class ProfileResult final {
 public:
  explicit ProfileResult(std::shared_ptr<const ModelProtocolProfile> profile);
  explicit ProfileResult(ModelProtocolError error);

  bool ok() const { return profile_ != nullptr; }
  const std::shared_ptr<const ModelProtocolProfile>& profile() const {
    return profile_;
  }
  const ModelProtocolError& error() const { return error_; }

 private:
  std::shared_ptr<const ModelProtocolProfile> profile_;
  ModelProtocolError error_;
};

class ProfileRegistry final {
 public:
  ModelProtocolError add(std::shared_ptr<const ModelProtocolProfile> profile);
  ProfileResult resolve(const LoadedModelContext& context) const;

 private:
  std::unordered_map<std::string, std::shared_ptr<const ModelProtocolProfile>>
      profiles_;
};

}  // namespace xllm::model_protocol
