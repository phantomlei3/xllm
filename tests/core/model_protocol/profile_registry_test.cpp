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
#include <vector>

#include "core/model_protocol/deepseek_v4_profile.h"
#include "core/model_protocol/glm_moe_dsa_profile.h"

namespace xllm::model_protocol {
namespace {

class FakeParser final : public ModelOutputParser {
 public:
  std::vector<OutputSegment> consume(
      const GenerationDelta& /*delta*/) override {
    return {};
  }
};

class FakeProfile final : public ModelProtocolProfile {
 public:
  FakeProfile()
      : identity_{.profile_id = "fake_responses",
                  .model_type = "fake-type",
                  .tokenizer_fingerprint = "sha256:fake-tokenizer",
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
          .model_type = "fake-type",
          .tokenizer_fingerprint = "sha256:fake-tokenizer",
          .template_fingerprint = "sha256:fake"};
}

TEST(ProfileRegistryTest, ResolvesByModelTypeRegardlessOfPublicId) {
  ProfileRegistry registry;
  ASSERT_TRUE(registry.add(std::make_shared<FakeProfile>()).ok());

  ProfileResult result = registry.resolve(make_context("deployment-name"));
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.profile()->identity().profile_id, "fake_responses");
  EXPECT_TRUE(result.profile()->capabilities().preserves_reasoning);
  EXPECT_TRUE(result.profile()->capabilities().supports_function_tools);
  EXPECT_TRUE(result.profile()->capabilities().supports_apply_patch);
  EXPECT_TRUE(result.profile()->capabilities().supports_reasoning_stream);
  EXPECT_TRUE(result.profile()->capabilities().supports_raw_decode);
  EXPECT_EQ(result.profile()->raw_decoding().policy,
            OutputDecodingPolicy::PROTOCOL_RAW);
  EXPECT_NE(result.profile()->new_parser(), nullptr);
  EXPECT_EQ(result.profile()->default_effort(), ReasoningEffort::MEDIUM);
  EXPECT_EQ(result.profile()
                ->resolve_template(ThinkingHistoryPolicy::PRESERVE)
                .thinking_history,
            ThinkingHistoryPolicy::PRESERVE);
  SamplingPolicy sampling = result.profile()->resolve_sampling(
      ReasoningEffort::HIGH, /*temperature=*/0.4f, /*top_p=*/0.8f);
  EXPECT_EQ(sampling.effort, ReasoningEffort::HIGH);
  EXPECT_FLOAT_EQ(sampling.temperature, 0.4f);
  EXPECT_FLOAT_EQ(sampling.top_p, 0.8f);
}

TEST(ProfileRegistryTest, UnknownModelTypeReturnsTypedCapabilityError) {
  ProfileRegistry registry;
  ASSERT_TRUE(registry.add(std::make_shared<FakeProfile>()).ok());
  LoadedModelContext context = make_context("deployment-name");
  context.model_type = "unknown-type";

  ProfileResult result = registry.resolve(context);

  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error().code(),
            ModelProtocolErrorCode::UNSUPPORTED_MODEL_CAPABILITY);
}

TEST(ProfileRegistryTest, RejectsDuplicateModelType) {
  ProfileRegistry registry;
  ASSERT_TRUE(registry.add(std::make_shared<FakeProfile>()).ok());

  ModelProtocolError error = registry.add(std::make_shared<FakeProfile>());

  ASSERT_FALSE(error.ok());
  EXPECT_EQ(error.code(), ModelProtocolErrorCode::DUPLICATE_PROFILE_IDENTITY);
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

LoadedModelContext make_deepseek_context() {
  return {
      .model_id = "deepseek-v4",
      .model_type = "deepseek_v4",
      .tokenizer_fingerprint =
          "sha256:"
          "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
      .template_fingerprint =
          "sha256:"
          "a06d98abfe8e4d78d2505ca28464e6b4af203731d6c07ee1ca9c4ab1bd77f95b"};
}

TEST(ProfileRegistryTest, DeepseekRequiresCompleteFrozenIdentity) {
  ProfileRegistry registry;
  ASSERT_TRUE(registry.add(make_deepseek_v4_profile()).ok());

  ProfileResult result = registry.resolve(make_deepseek_context());

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.profile()->identity().profile_id, "deepseek_v4_responses");
  EXPECT_EQ(result.profile()->raw_decoding().policy,
            OutputDecodingPolicy::PROTOCOL_RAW);
  EXPECT_EQ(result.profile()->raw_decoding().parser_dialect,
            "deepseek_v4_dsml");
  EXPECT_EQ(result.profile()->raw_decoding().max_marker_bytes, 23);
  EXPECT_NE(result.profile()->new_parser(), nullptr);
  TemplatePolicy preserve =
      result.profile()->resolve_template(ThinkingHistoryPolicy::PRESERVE);
  ASSERT_TRUE(preserve.drop_thinking.has_value());
  EXPECT_FALSE(*preserve.drop_thinking);

  LoadedModelContext mismatched = make_deepseek_context();
  mismatched.tokenizer_fingerprint = "sha256:wrong";
  ProfileResult rejected = registry.resolve(mismatched);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code(),
            ModelProtocolErrorCode::PROFILE_IDENTITY_MISMATCH);
}

TEST(ProfileRegistryTest, DeepseekMapsEffortAndSampling) {
  std::shared_ptr<const ModelProtocolProfile> profile =
      make_deepseek_v4_profile();

  SamplingPolicy low = profile->resolve_sampling(
      ReasoningEffort::MINIMAL, /*temperature=*/0.2f, /*top_p=*/0.4f);
  EXPECT_EQ(low.effort, ReasoningEffort::LOW);
  EXPECT_FLOAT_EQ(low.temperature, 1.0f);
  EXPECT_FLOAT_EQ(low.top_p, 0.95f);

  SamplingPolicy xhigh = profile->resolve_sampling(
      ReasoningEffort::XHIGH, /*temperature=*/0.2f, /*top_p=*/0.4f);
  EXPECT_EQ(xhigh.effort, ReasoningEffort::HIGH);

  SamplingPolicy none = profile->resolve_sampling(
      ReasoningEffort::NONE, /*temperature=*/0.2f, /*top_p=*/0.4f);
  EXPECT_EQ(none.effort, ReasoningEffort::NONE);
  EXPECT_FLOAT_EQ(none.temperature, 0.2f);
  EXPECT_FLOAT_EQ(none.top_p, 0.4f);
}

LoadedModelContext make_glm_context() {
  return {
      .model_id = "GLM-5-W4A8",
      .model_type = "glm_moe_dsa",
      .tokenizer_fingerprint =
          "sha256:"
          "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
      .template_fingerprint =
          "sha256:"
          "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679"};
}

TEST(ProfileRegistryTest, GlmRequiresCompleteFrozenIdentity) {
  ProfileRegistry registry;
  ASSERT_TRUE(registry.add(make_glm_moe_dsa_profile()).ok());

  ProfileResult result = registry.resolve(make_glm_context());

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.profile()->identity().profile_id, "glm_moe_dsa_responses");
  EXPECT_EQ(result.profile()->raw_decoding().policy,
            OutputDecodingPolicy::PROTOCOL_RAW);
  EXPECT_EQ(result.profile()->raw_decoding().parser_dialect,
            "glm_moe_dsa_native");
  EXPECT_EQ(result.profile()->raw_decoding().max_marker_bytes, 15);
  EXPECT_EQ(result.profile()->raw_decoding().preserved_token_ids,
            std::vector<int64_t>({154820,
                                  154822,
                                  154824,
                                  154826,
                                  154827,
                                  154828,
                                  154829,
                                  154841,
                                  154842,
                                  154843,
                                  154844,
                                  154847,
                                  154848,
                                  154849,
                                  154850}));
  EXPECT_NE(result.profile()->new_parser(), nullptr);
  TemplatePolicy preserve =
      result.profile()->resolve_template(ThinkingHistoryPolicy::PRESERVE);
  ASSERT_TRUE(preserve.clear_thinking.has_value());
  EXPECT_FALSE(*preserve.clear_thinking);

  LoadedModelContext mismatched = make_glm_context();
  mismatched.template_fingerprint = "sha256:wrong";
  ProfileResult rejected = registry.resolve(mismatched);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().code(),
            ModelProtocolErrorCode::PROFILE_IDENTITY_MISMATCH);
}

TEST(ProfileRegistryTest, GlmMapsEffortAndSampling) {
  std::shared_ptr<const ModelProtocolProfile> profile =
      make_glm_moe_dsa_profile();

  SamplingPolicy low = profile->resolve_sampling(
      ReasoningEffort::MINIMAL, /*temperature=*/0.2f, /*top_p=*/0.4f);
  EXPECT_EQ(low.effort, ReasoningEffort::HIGH);
  EXPECT_FLOAT_EQ(low.temperature, 1.0f);
  EXPECT_FLOAT_EQ(low.top_p, 0.95f);

  SamplingPolicy xhigh = profile->resolve_sampling(
      ReasoningEffort::XHIGH, /*temperature=*/0.2f, /*top_p=*/0.4f);
  EXPECT_EQ(xhigh.effort, ReasoningEffort::MAX);

  SamplingPolicy none = profile->resolve_sampling(
      ReasoningEffort::NONE, /*temperature=*/0.2f, /*top_p=*/0.4f);
  EXPECT_EQ(none.effort, ReasoningEffort::NONE);
  EXPECT_FLOAT_EQ(none.temperature, 0.2f);
  EXPECT_FLOAT_EQ(none.top_p, 0.4f);
}

}  // namespace
}  // namespace xllm::model_protocol
