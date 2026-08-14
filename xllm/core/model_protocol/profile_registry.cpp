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

#include "core/model_protocol/profile_registry.h"

#include <utility>

namespace xllm::model_protocol {
namespace {

bool identity_matches(const ModelProtocolIdentity& identity,
                      const LoadedModelContext& context) {
  return identity.tokenizer_id == context.tokenizer_id &&
         identity.template_id == context.template_id &&
         identity.template_fingerprint == context.template_fingerprint;
}

}  // namespace

ModelProtocolError::ModelProtocolError(ModelProtocolErrorCode code,
                                       std::string message)
    : code_(code), message_(std::move(message)) {}

ProfileResult::ProfileResult(
    std::shared_ptr<const ModelProtocolProfile> profile)
    : profile_(std::move(profile)) {}

ProfileResult::ProfileResult(ModelProtocolError error)
    : error_(std::move(error)) {}

ModelProtocolError ProfileRegistry::add(
    std::shared_ptr<const ModelProtocolProfile> profile) {
  const ModelProtocolIdentity& identity = profile->identity();
  if (profiles_.contains(identity.canonical_model_id)) {
    return {ModelProtocolErrorCode::DUPLICATE_PROFILE_IDENTITY,
            "duplicate canonical model identity"};
  }
  for (const std::string& alias : identity.model_aliases) {
    if (profiles_.contains(alias)) {
      return {ModelProtocolErrorCode::DUPLICATE_PROFILE_IDENTITY,
              "duplicate model alias"};
    }
  }

  profiles_.emplace(identity.canonical_model_id, profile);
  for (const std::string& alias : identity.model_aliases) {
    profiles_.emplace(alias, profile);
  }
  return {};
}

ProfileResult ProfileRegistry::resolve(
    const LoadedModelContext& context) const {
  auto it = profiles_.find(context.model_id);
  if (it == profiles_.end()) {
    return ProfileResult(
        ModelProtocolError(ModelProtocolErrorCode::UNSUPPORTED_MODEL_CAPABILITY,
                           "model has no Responses protocol profile"));
  }
  if (!identity_matches(it->second->identity(), context)) {
    return ProfileResult(
        ModelProtocolError(ModelProtocolErrorCode::PROFILE_IDENTITY_MISMATCH,
                           "loaded tokenizer or template identity mismatch"));
  }
  return ProfileResult(it->second);
}

}  // namespace xllm::model_protocol
