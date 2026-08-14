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

#include <memory>

#include "core/model_protocol/profile.h"

namespace xllm::model_protocol {

class Glm52Profile final : public ModelProtocolProfile {
 public:
  Glm52Profile();

  const ModelProtocolIdentity& identity() const override;
  const ModelProtocolCapabilities& capabilities() const override;
  const RawDecodingRequirements& raw_decoding() const override;
  ReasoningEffort default_effort() const override;
  SamplingPolicy resolve_sampling(ReasoningEffort effort,
                                  float temperature,
                                  float top_p) const override;
  TemplatePolicy resolve_template(
      ThinkingHistoryPolicy thinking) const override;
  std::unique_ptr<ModelOutputParser> new_parser() const override;

 private:
  ModelProtocolIdentity identity_;
  ModelProtocolCapabilities capabilities_;
  RawDecodingRequirements raw_decoding_;
};

std::shared_ptr<const ModelProtocolProfile> make_glm_5_2_profile();

}  // namespace xllm::model_protocol
