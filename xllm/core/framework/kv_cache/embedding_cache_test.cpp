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

#include "embedding_cache.h"

#include <gtest/gtest.h>

#include "platform/device.h"

namespace xllm {

namespace {

bool tensor_equal(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  return lhs.defined() && rhs.defined() && torch::equal(lhs, rhs);
}

}  // namespace

TEST(EmbeddingCacheTest, WriteAndClear) {
  // use init device to trigger the loading of torch backend for different
  // devices
  //  since the allocation of pinnned memory on cpu is still backend-dependent.
  torch::Device device(Device::type_torch(), 0);
  EmbeddingCache cache(/*total_nums=*/4);

  std::vector<int32_t> ids = {3, 2};
  auto cached_tokens = torch::tensor({31, 41}, torch::kInt);
  auto cached_embeddings = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto cached_probs = torch::tensor({{0.1f, 0.9f}, {0.4f, 0.6f}});

  cache.write(ids, cached_tokens, cached_embeddings, cached_probs);

  auto output = cache.read_for_decode(ids);
  EXPECT_TRUE(torch::equal(output.sample_output.next_tokens.to(torch::kInt),
                           cached_tokens));
  EXPECT_TRUE(tensor_equal(output.sample_output.embeddings, cached_embeddings));
  EXPECT_TRUE(tensor_equal(output.sample_output.probs, cached_probs));

  cache.clear(ids);
  auto updated_tokens = torch::tensor({51, 61}, torch::kInt);
  auto updated_embeddings = torch::tensor({{5.0f, 6.0f}, {7.0f, 8.0f}});
  auto updated_probs = torch::tensor({{0.2f, 0.8f}, {0.3f, 0.7f}});
  cache.write(ids, updated_tokens, updated_embeddings, updated_probs);
  output = cache.read_for_decode(ids);
  EXPECT_TRUE(torch::equal(output.sample_output.next_tokens.to(torch::kInt),
                           updated_tokens));
  EXPECT_TRUE(
      tensor_equal(output.sample_output.embeddings, updated_embeddings));
  EXPECT_TRUE(tensor_equal(output.sample_output.probs, updated_probs));
}

TEST(EmbeddingCacheTest, WriteSelectedOnlyProbs) {
  // use init device to trigger the loading of torch backend for different
  // devices
  //  since the allocation of pinnned memory on cpu is still backend-dependent.
  torch::Device device(Device::type_torch(), 0);
  EmbeddingCache cache(/*total_nums=*/2);
  std::vector<int32_t> ids = {0, 1};
  auto cached_tokens = torch::tensor({11, 12}, torch::kInt);
  auto cached_embeddings = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
  auto cached_probs = torch::tensor({0.2f, 0.8f});

  cache.write(ids, cached_tokens, cached_embeddings, cached_probs);
  auto output = cache.read_for_decode(ids);
  EXPECT_TRUE(torch::equal(output.sample_output.next_tokens.to(torch::kInt),
                           cached_tokens));
  EXPECT_TRUE(tensor_equal(output.sample_output.embeddings, cached_embeddings));
  EXPECT_TRUE(tensor_equal(output.sample_output.probs, cached_probs));
}

TEST(EmbeddingCacheTest, BuildSeedMaskMixedRows) {
  torch::Device device(Device::type_torch(), 0);
  EmbeddingCache cache(/*total_nums=*/3);

  std::vector<int32_t> write_ids = {0, 2};
  torch::Tensor cached_tokens = torch::tensor({11, 13}, torch::kInt);
  torch::Tensor cached_embeddings = torch::tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
  torch::Tensor cached_probs = torch::tensor({{0.1f, 0.9f}, {0.4f, 0.6f}});

  cache.write(write_ids, cached_tokens, cached_embeddings, cached_probs);

  std::vector<int32_t> read_ids = {0, 1, 2};
  std::vector<uint8_t> mask = cache.build_seed_mask(read_ids);
  EXPECT_EQ(mask, (std::vector<uint8_t>{1, 0, 1}));

  cache.clear({2});
  mask = cache.build_seed_mask(read_ids);
  EXPECT_EQ(mask, (std::vector<uint8_t>{1, 0, 0}));

  cache.write({2},
              torch::tensor({23}, torch::kInt),
              torch::tensor({{5.0f, 6.0f}}),
              torch::tensor({{0.3f, 0.7f}}));
  mask = cache.build_seed_mask(read_ids);
  EXPECT_EQ(mask, (std::vector<uint8_t>{1, 0, 1}));
}

}  // namespace xllm
