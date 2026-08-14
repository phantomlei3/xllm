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

#include <gtest/gtest.h>

#include <memory>
#include <string>

namespace xllm::model_protocol {
namespace {

class FakeParser final : public ModelOutputParser {};

class FakeProfile final : public ModelProtocolProfile {
 public:
  FakeProfile()
      : identity_{.profile_id = "fake_responses",
                  .canonical_model_id = "fake/model",
                  .model_aliases = {"fake-alias"},
                  .tokenizer_id = "fake-tokenizer",
                  .template_id = "fake-template",
                  .template_fingerprint = "sha256:fake"},
        capabilities_{.preserves_reasoning = true,
                      .supports_function_tools = true,
                      .supports_apply_patch = true,
                      .supports_reasoning_stream = true,
                      .supports_raw_decode = true} {}

  const ModelProtocolIdentity& identity() const override { return identity_; }

  const ModelProtocolCapabilities& capabilities() const override {
    return capabilities_;
  }

  const RawDecodingRequirements& raw_decoding() const override {
    return raw_decoding_;
  }

  ReasoningEffort default_effort() const override {
    return ReasoningEffort::MEDIUM;
  }

  SamplingPolicy resolve_sampling(ReasoningEffort effort,
                                  float temperature,
                                  float top_p) const override {
    return {.effort = effort, .temperature = temperature, .top_p = top_p};
  }

  TemplatePolicy resolve_template(
      ThinkingHistoryPolicy thinking) const override {
    return {.thinking_history = thinking};
  }

  std::unique_ptr<ModelOutputParser> new_parser() const override {
    return std::make_unique<FakeParser>();
  }

 private:
  ModelProtocolIdentity identity_;
  ModelProtocolCapabilities capabilities_;
  RawDecodingRequirements raw_decoding_;
};

LoadedModelContext make_context(const std::string& model_id) {
  return {.model_id = model_id,
          .tokenizer_id = "fake-tokenizer",
          .template_id = "fake-template",
          .template_fingerprint = "sha256:fake"};
}

TEST(ProfileRegistryTest, ResolvesCanonicalModelAndControlledAlias) {
  ProfileRegistry registry;
  ASSERT_TRUE(registry.add(std::make_shared<FakeProfile>()).ok());

  ProfileResult canonical = registry.resolve(make_context("fake/model"));
  ASSERT_TRUE(canonical.ok());
  EXPECT_EQ(canonical.profile()->identity().profile_id, "fake_responses");

  ProfileResult alias = registry.resolve(make_context("fake-alias"));
  ASSERT_TRUE(alias.ok());
  EXPECT_EQ(alias.profile()->identity().canonical_model_id, "fake/model");
  EXPECT_TRUE(alias.profile()->capabilities().preserves_reasoning);
  EXPECT_TRUE(alias.profile()->capabilities().supports_function_tools);
  EXPECT_TRUE(alias.profile()->capabilities().supports_apply_patch);
  EXPECT_TRUE(alias.profile()->capabilities().supports_reasoning_stream);
  EXPECT_TRUE(alias.profile()->capabilities().supports_raw_decode);
  EXPECT_EQ(alias.profile()->raw_decoding().policy,
            OutputDecodingPolicy::PROTOCOL_RAW);
  EXPECT_NE(alias.profile()->new_parser(), nullptr);
  EXPECT_EQ(alias.profile()->default_effort(), ReasoningEffort::MEDIUM);
  EXPECT_EQ(alias.profile()
                ->resolve_template(ThinkingHistoryPolicy::PRESERVE)
                .thinking_history,
            ThinkingHistoryPolicy::PRESERVE);
  SamplingPolicy sampling = alias.profile()->resolve_sampling(
      ReasoningEffort::HIGH, /*temperature=*/0.4f, /*top_p=*/0.8f);
  EXPECT_EQ(sampling.effort, ReasoningEffort::HIGH);
  EXPECT_FLOAT_EQ(sampling.temperature, 0.4f);
  EXPECT_FLOAT_EQ(sampling.top_p, 0.8f);
}

TEST(ProfileRegistryTest, UnknownModelReturnsTypedCapabilityError) {
  ProfileRegistry registry;
  ASSERT_TRUE(registry.add(std::make_shared<FakeProfile>()).ok());

  ProfileResult result = registry.resolve(make_context("unknown/model"));

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error().code(),
            ModelProtocolErrorCode::UNSUPPORTED_MODEL_CAPABILITY);
}

TEST(ProfileRegistryTest, IdentityMismatchReturnsDeploymentError) {
  ProfileRegistry registry;
  ASSERT_TRUE(registry.add(std::make_shared<FakeProfile>()).ok());
  LoadedModelContext context = make_context("fake/model");
  context.template_fingerprint = "sha256:wrong";

  ProfileResult result = registry.resolve(context);

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error().code(),
            ModelProtocolErrorCode::PROFILE_IDENTITY_MISMATCH);
}

}  // namespace
}  // namespace xllm::model_protocol
