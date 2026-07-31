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

#include "runtime/rec_beam_utils.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm::runtime::detail {
namespace {

TEST(RecBeamUtilsTest, MluTwoStageTracksDecodeStepAndBlockIds) {
  const auto options = torch::TensorOptions().dtype(torch::kInt32);
  auto block_table = torch::tensor({2, 0, 3, 1}, options).view({4, 1});
  auto slot_mapping = torch::empty({4}, options);
  auto unshared_seq_lens = torch::empty({4}, options);

  update_mlu_two_stage(block_table,
                       /*decode_step=*/0,
                       /*max_decode_steps=*/2,
                       slot_mapping,
                       unshared_seq_lens);
  EXPECT_TRUE(slot_mapping.equal(torch::tensor({4, 0, 6, 2}, options)));
  EXPECT_TRUE(unshared_seq_lens.equal(torch::tensor({1, 1, 1, 1}, options)));

  update_mlu_two_stage(block_table,
                       /*decode_step=*/1,
                       /*max_decode_steps=*/2,
                       slot_mapping,
                       unshared_seq_lens);
  EXPECT_TRUE(slot_mapping.equal(torch::tensor({5, 1, 7, 3}, options)));
  EXPECT_TRUE(unshared_seq_lens.equal(torch::tensor({2, 2, 2, 2}, options)));
}

TEST(RecBeamUtilsTest, MluTwoStageUsesExpandedBeamLayout) {
  constexpr int32_t kBatchSize = 2;
  constexpr int32_t kBeamWidth = 3;
  constexpr int32_t kMaxDecodeSteps = 2;
  const auto options = torch::TensorOptions().dtype(torch::kInt32);
  auto block_table =
      torch::arange(kBatchSize * kBeamWidth, options).view({-1, 1});
  auto slot_mapping = torch::empty({kBatchSize * kBeamWidth}, options);
  auto unshared_seq_lens = torch::empty_like(slot_mapping);
  auto q_cu_seq_lens =
      torch::arange(0, (kBatchSize + 1) * kBeamWidth, kBeamWidth, options);

  update_mlu_two_stage(block_table,
                       /*decode_step=*/0,
                       kMaxDecodeSteps,
                       slot_mapping,
                       unshared_seq_lens);
  EXPECT_TRUE(slot_mapping.equal(torch::tensor({0, 2, 4, 6, 8, 10}, options)));
  EXPECT_TRUE(unshared_seq_lens.equal(torch::ones({6}, options)));
  EXPECT_TRUE(q_cu_seq_lens.equal(torch::tensor({0, 3, 6}, options)));

  update_mlu_two_stage(block_table,
                       /*decode_step=*/1,
                       kMaxDecodeSteps,
                       slot_mapping,
                       unshared_seq_lens);
  EXPECT_TRUE(slot_mapping.equal(torch::tensor({1, 3, 5, 7, 9, 11}, options)));
  EXPECT_TRUE(unshared_seq_lens.equal(torch::full({6}, 2, options)));
  EXPECT_TRUE(q_cu_seq_lens.equal(torch::tensor({0, 3, 6}, options)));
}

TEST(RecBeamUtilsTest, WriteFirstRoundBeamOutputsReshapesBatchBeamSlice) {
  constexpr int32_t kBatchSize = 2;
  constexpr int32_t kBeamWidth = 64;
  constexpr int32_t kTotalRounds = 3;
  const auto int_options = torch::TensorOptions().dtype(torch::kInt32);
  const auto fp32_options = torch::TensorOptions().dtype(torch::kFloat32);

  auto top_tokens =
      torch::arange(kBatchSize * kBeamWidth, int_options).view({-1, 1});
  auto top_logprobs =
      torch::arange(kBatchSize * kBeamWidth, fp32_options).view({-1, 1});
  auto out_token_ids = torch::zeros({kBatchSize * kBeamWidth, 1}, int_options);
  auto out_log_probs = torch::zeros({kBatchSize * kBeamWidth, 1}, fp32_options);
  auto out_seqgroup =
      torch::zeros({kBatchSize, kBeamWidth, kTotalRounds}, int_options);

  write_first_round_beam_outputs(top_tokens,
                                 top_logprobs,
                                 kBatchSize,
                                 out_token_ids,
                                 out_log_probs,
                                 out_seqgroup);

  EXPECT_TRUE(out_token_ids.equal(top_tokens));
  EXPECT_TRUE(out_log_probs.equal(top_logprobs));
  EXPECT_TRUE(out_seqgroup.select(/*dim=*/2, /*index=*/0)
                  .equal(top_tokens.view({kBatchSize, kBeamWidth})));
  EXPECT_EQ(out_seqgroup.select(/*dim=*/2, /*index=*/1).sum().item<int32_t>(),
            0);
  EXPECT_EQ(out_seqgroup.select(/*dim=*/2, /*index=*/2).sum().item<int32_t>(),
            0);
}

}  // namespace
}  // namespace xllm::runtime::detail
