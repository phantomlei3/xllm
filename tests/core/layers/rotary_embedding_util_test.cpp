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

#include "layers/common/rotary_embedding_util.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace xllm {
namespace layer {
namespace {

constexpr int64_t kMaxPositions = 64;
constexpr int64_t kRotaryDim = 64;
constexpr int64_t kHalfRotaryDim = kRotaryDim / 2;

const std::vector<int64_t> kMropeSection = {11, 11, 10};
const std::vector<int64_t> kAxisByHalfDim = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
                                             2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
                                             1, 2, 0, 1, 2, 0, 1, 2, 0, 1};

torch::Tensor make_cache() {
  constexpr int64_t kCacheWidth = kRotaryDim * 2;
  return torch::arange(kMaxPositions * kCacheWidth, torch::kFloat32)
      .reshape({kMaxPositions, kCacheWidth})
      .contiguous();
}

torch::Tensor make_mrope_positions() {
  return torch::tensor({{1, 2, 3}, {11, 12, 13}, {21, 22, 23}},
                       torch::TensorOptions().dtype(torch::kLong));
}

std::pair<torch::Tensor, torch::Tensor> make_mrope_ref(
    const torch::Tensor& cos_sin_cache,
    const torch::Tensor& positions) {
  torch::Tensor selected = cos_sin_cache.index({positions});
  std::vector<torch::Tensor> chunks = selected.chunk(/*chunks=*/2, /*dim=*/-1);
  torch::Tensor expected_cos =
      torch::empty({positions.size(1), kRotaryDim}, cos_sin_cache.options());
  torch::Tensor expected_sin =
      torch::empty({positions.size(1), kRotaryDim}, cos_sin_cache.options());

  for (int64_t column = 0; column < kRotaryDim; ++column) {
    int64_t axis = kAxisByHalfDim[column % kHalfRotaryDim];
    expected_cos.select(/*dim=*/1, column)
        .copy_(chunks[0].select(/*dim=*/0, axis).select(/*dim=*/1, column));
    expected_sin.select(/*dim=*/1, column)
        .copy_(chunks[1].select(/*dim=*/0, axis).select(/*dim=*/1, column));
  }
  return {expected_cos, expected_sin};
}

TEST(RotaryEmbeddingUtilTest, RealSectionsProduceInterleavedMrope) {
  torch::Tensor cos_sin_cache = make_cache();
  torch::Tensor positions = make_mrope_positions();

  auto [cos, sin] =
      rotary::apply_mrope(cos_sin_cache, positions, kMropeSection);
  auto [expected_cos, expected_sin] = make_mrope_ref(cos_sin_cache, positions);

  EXPECT_EQ(cos.sizes(), torch::IntArrayRef({positions.size(1), kRotaryDim}));
  EXPECT_EQ(sin.sizes(), torch::IntArrayRef({positions.size(1), kRotaryDim}));
  EXPECT_TRUE(torch::equal(cos, expected_cos));
  EXPECT_TRUE(torch::equal(sin, expected_sin));
  EXPECT_EQ(cos.scalar_type(), cos_sin_cache.scalar_type());
  EXPECT_EQ(sin.scalar_type(), cos_sin_cache.scalar_type());
  EXPECT_EQ(cos.device(), cos_sin_cache.device());
  EXPECT_EQ(sin.device(), cos_sin_cache.device());
  EXPECT_TRUE(cos.is_contiguous());
  EXPECT_TRUE(sin.is_contiguous());
}

TEST(RotaryEmbeddingUtilTest, OneDimensionalPositionsMatchExpandedPositions) {
  torch::Tensor cos_sin_cache = make_cache();
  torch::Tensor positions = torch::tensor({1, 2, 3}, torch::kLong);
  torch::Tensor expanded_positions = positions.expand({3, -1});

  auto [cos, sin] =
      rotary::apply_mrope(cos_sin_cache, positions, kMropeSection);
  auto [expanded_cos, expanded_sin] =
      rotary::apply_mrope(cos_sin_cache, expanded_positions, kMropeSection);

  EXPECT_TRUE(torch::equal(cos, expanded_cos));
  EXPECT_TRUE(torch::equal(sin, expanded_sin));
}

TEST(RotaryEmbeddingUtilTest, RejectsUndefinedCache) {
  torch::Tensor positions = make_mrope_positions();
  torch::Tensor undefined_cache;

  EXPECT_DEATH(
      (void)rotary::apply_mrope(undefined_cache, positions, kMropeSection),
      "mRoPE cos_sin_cache must be defined");
}

TEST(RotaryEmbeddingUtilTest, RejectsInvalidCacheRank) {
  torch::Tensor cos_sin_cache =
      torch::zeros({2, kMaxPositions, kRotaryDim * 2});
  torch::Tensor positions = make_mrope_positions();

  EXPECT_DEATH(
      (void)rotary::apply_mrope(cos_sin_cache, positions, kMropeSection),
      "mRoPE cos_sin_cache must be 2D");
}

TEST(RotaryEmbeddingUtilTest, RejectsUndefinedPositions) {
  torch::Tensor cos_sin_cache = make_cache();
  torch::Tensor undefined_positions;

  EXPECT_DEATH((void)rotary::apply_mrope(
                   cos_sin_cache, undefined_positions, kMropeSection),
               "mRoPE positions must be defined");
}

TEST(RotaryEmbeddingUtilTest, RejectsInvalidPositionRank) {
  torch::Tensor cos_sin_cache = make_cache();
  torch::Tensor positions = torch::zeros({1, 3, 1}, torch::kLong);

  EXPECT_DEATH(
      (void)rotary::apply_mrope(cos_sin_cache, positions, kMropeSection),
      "mRoPE positions must be 1D or 2D");
}

TEST(RotaryEmbeddingUtilTest, RejectsInvalidAxisCount) {
  torch::Tensor cos_sin_cache = make_cache();
  torch::Tensor positions = torch::zeros({2, 3}, torch::kLong);

  EXPECT_DEATH(
      (void)rotary::apply_mrope(cos_sin_cache, positions, kMropeSection),
      "mRoPE positions must have 3 axes");
}

TEST(RotaryEmbeddingUtilTest, RejectsInvalidSectionCount) {
  torch::Tensor cos_sin_cache = make_cache();
  torch::Tensor positions = make_mrope_positions();
  const std::vector<int64_t> sections = {16, 16};

  EXPECT_DEATH((void)rotary::apply_mrope(cos_sin_cache, positions, sections),
               "mRoPE requires exactly 3 sections");
}

TEST(RotaryEmbeddingUtilTest, RejectsNegativeSection) {
  torch::Tensor cos_sin_cache = make_cache();
  torch::Tensor positions = make_mrope_positions();
  const std::vector<int64_t> sections = {11, -1, 22};

  EXPECT_DEATH((void)rotary::apply_mrope(cos_sin_cache, positions, sections),
               "mRoPE sections must be non-negative");
}

TEST(RotaryEmbeddingUtilTest, RejectsInvalidCacheWidth) {
  torch::Tensor cos_sin_cache =
      torch::zeros({kMaxPositions, kRotaryDim * 2 - 2});
  torch::Tensor positions = make_mrope_positions();

  EXPECT_DEATH(
      (void)rotary::apply_mrope(cos_sin_cache, positions, kMropeSection),
      "mRoPE cache width must be positive and divisible by 4");
}

TEST(RotaryEmbeddingUtilTest, RejectsSectionDimensionMismatch) {
  torch::Tensor cos_sin_cache = make_cache();
  torch::Tensor positions = make_mrope_positions();
  const std::vector<int64_t> sections = {10, 11, 10};

  EXPECT_DEATH((void)rotary::apply_mrope(cos_sin_cache, positions, sections),
               "mRoPE section sum must equal half rotary dimension");
}

}  // namespace
}  // namespace layer
}  // namespace xllm
