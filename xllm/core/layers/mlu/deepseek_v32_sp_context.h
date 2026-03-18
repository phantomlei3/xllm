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

#include <torch/torch.h>

#include <cstdint>
#include <numeric>
#include <optional>
#include <vector>

#include "core/common/global_flags.h"
#include "framework/batch/batch_forward_type.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/parallel_state/process_group.h"
#include "layers/mlu/deepseek_v32_sp_metadata.h"
#include "layers/mlu/deepseek_v32_sp_plan.h"

namespace xllm::layer::v32_sp {

using PaddedGatherHandle = xllm::parallel_state::GatherAsyncCtx;

struct DeepseekV32SPContext {
  DeepseekV32SPCommPlan comm_plan;
  AttentionMetadata local_attn_metadata;
  DeepseekV32SPMetadata sp_meta;
  BatchForwardType batch_forward_type;

  torch::Tensor gathered_reorder_index;
  torch::Tensor gathered_slot_mapping;

  int32_t total_tokens = 0;
  int32_t rank = 0;
  ProcessGroup* process_group = nullptr;
};

inline torch::Tensor reorder_by_index(const torch::Tensor& tensor,
                                      const torch::Tensor& reorder_index);

inline std::optional<DeepseekV32SPContext> build_deepseek_v32_sp_context(
    const AttentionMetadata& base_attn_metadata,
    BatchForwardType batch_forward_type,
    const torch::Tensor& tokens,
    ProcessGroup* sp_group,
    int32_t curr_rank,
    int32_t world_size) {
  if (!FLAGS_enable_prefill_sp || !batch_forward_type.no_decode() ||
      world_size <= 1) {
    return std::nullopt;
  }

  CHECK(sp_group != nullptr)
      << "deepseek_v32 sequence parallel requires sp_group.";
  CHECK_EQ(tokens.dim(), 1)
      << "deepseek_v32 sequence parallel expects 1D tokens.";
  CHECK_GE(curr_rank, 0) << "curr_rank must be non-negative.";
  CHECK_LT(curr_rank, world_size) << "curr_rank must be less than world_size.";

  const std::vector<int32_t> q_seq_lens =
      extract_q_seq_lens(base_attn_metadata);
  const std::vector<int32_t> ctx_seq_lens =
      extract_ctx_seq_lens(base_attn_metadata);
  CHECK(!q_seq_lens.empty())
      << "deepseek_v32 sequence parallel requires non-empty prefill requests.";
  for (int32_t seq_len : q_seq_lens) {
    if (seq_len < world_size) {
      return std::nullopt;
    }
  }

  const int32_t total_tokens = static_cast<int32_t>(tokens.numel());
  CHECK_EQ(total_tokens,
           std::accumulate(q_seq_lens.begin(), q_seq_lens.end(), int32_t{0}))
      << "tokens.numel() must match total prefill tokens.";

  DeepseekV32SPContext context;
  const auto all_segments =
      build_all_sp_segments(world_size, q_seq_lens, ctx_seq_lens);
  const auto local_segments = build_local_sp_segments(curr_rank, all_segments);
  const auto runtime_artifacts = build_sp_runtime_artifacts(
      curr_rank, world_size, all_segments, total_tokens);
  context.comm_plan = runtime_artifacts.comm_plan;
  context.local_attn_metadata = build_local_prefill_attention_metadata(
      base_attn_metadata, local_segments);
  context.sp_meta =
      build_sp_metadata(base_attn_metadata, local_segments, q_seq_lens);
  context.batch_forward_type = batch_forward_type;
  context.total_tokens = total_tokens;
  context.rank = curr_rank;
  context.process_group = sp_group;

  const auto int64_options =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  const torch::Device device = tokens.device();
  context.gathered_reorder_index =
      torch::tensor(runtime_artifacts.gathered_reorder_index_cpu, int64_options)
          .to(device);
  if (base_attn_metadata.slot_mapping.defined()) {
    context.gathered_slot_mapping = reorder_by_index(
        base_attn_metadata.slot_mapping, context.gathered_reorder_index);
  }
  return context;
}

inline torch::Tensor reorder_by_index(const torch::Tensor& tensor,
                                      const torch::Tensor& reorder_index) {
  if (!tensor.defined()) {
    return tensor;
  }
  CHECK_GT(tensor.dim(), 0)
      << "deepseek_v32 sequence parallel expects non-scalar tensors.";

  torch::Tensor index = reorder_index;
  if (index.scalar_type() != torch::kLong) {
    index = index.to(torch::kLong);
  }
  if (index.device() != tensor.device()) {
    index = index.to(tensor.device());
  }
  return tensor.index_select(0, index);
}

inline torch::Tensor reorder_to_local_shard(
    const torch::Tensor& tensor,
    const DeepseekV32SPContext& context) {
  const int32_t token_num = context.comm_plan.tokens_per_rank.at(context.rank);
  return reorder_by_index(
      tensor,
      context.gathered_reorder_index.narrow(
          0, context.comm_plan.token_num_offset, token_num));
}

inline torch::Tensor all_gather_across_ranks(
    const torch::Tensor& local_tensor,
    const DeepseekV32SPContext& context) {
  if (!local_tensor.defined()) {
    return local_tensor;
  }
  return xllm::parallel_state::gather(
      local_tensor, context.process_group, context.comm_plan.tokens_per_rank);
}

inline torch::Tensor restore_gathered_to_global_order(
    const torch::Tensor& gathered_tensor,
    const DeepseekV32SPContext& context) {
  if (!gathered_tensor.defined()) {
    return gathered_tensor;
  }
  CHECK_GT(gathered_tensor.dim(), 0)
      << "deepseek_v32 sequence parallel expects non-scalar tensors.";

  std::vector<int64_t> output_sizes;
  output_sizes.reserve(gathered_tensor.dim());
  output_sizes.push_back(context.total_tokens);
  for (int64_t dim = 1; dim < gathered_tensor.dim(); ++dim) {
    output_sizes.push_back(gathered_tensor.size(dim));
  }
  torch::Tensor restored =
      torch::zeros(output_sizes, gathered_tensor.options());

  CHECK_EQ(gathered_tensor.size(0), context.gathered_reorder_index.size(0))
      << "unexpected packed tensor length for sequence parallel restore.";
  torch::Tensor restore_index = context.gathered_reorder_index;

  if (restore_index.device() != gathered_tensor.device()) {
    restore_index = restore_index.to(gathered_tensor.device());
  }
  if (restore_index.scalar_type() != torch::kLong) {
    restore_index = restore_index.to(torch::kLong);
  }
  restored.index_copy_(0, restore_index, gathered_tensor);
  return restored;
}

inline torch::Tensor gather_and_restore_global(
    const torch::Tensor& local_tensor,
    const DeepseekV32SPContext& context) {
  return restore_gathered_to_global_order(
      all_gather_across_ranks(local_tensor, context), context);
}

inline torch::Tensor slice_local_packed(const torch::Tensor& packed_tensor,
                                        const DeepseekV32SPContext& context) {
  if (!packed_tensor.defined()) {
    return packed_tensor;
  }
  CHECK_GT(packed_tensor.dim(), 0)
      << "deepseek_v32 sequence parallel expects non-scalar tensors.";
  const int32_t start = context.comm_plan.token_num_offset;
  const int32_t token_num = context.comm_plan.tokens_per_rank.at(context.rank);
  CHECK_LE(start + token_num, packed_tensor.size(0))
      << "packed tensor rows are smaller than local slice range.";
  return packed_tensor.slice(0, start, start + token_num);
}

inline torch::Tensor pad_to_sp_rows(const torch::Tensor& local_tensor,
                                    const DeepseekV32SPContext& context) {
  if (!local_tensor.defined()) {
    return local_tensor;
  }
  const int32_t target_rows =
      context.comm_plan.padded_tokens_per_rank.at(context.rank);
  CHECK_GT(local_tensor.dim(), 0)
      << "deepseek_v32 sequence parallel expects non-scalar tensors.";
  CHECK_LE(local_tensor.size(0), target_rows)
      << "local tensor rows exceed padded target.";
  if (local_tensor.size(0) == target_rows) {
    return local_tensor.contiguous();
  }
  auto pad_shape = local_tensor.sizes().vec();
  pad_shape[0] = target_rows - local_tensor.size(0);
  torch::Tensor pad_tensor = torch::zeros(pad_shape, local_tensor.options());
  return torch::cat({local_tensor.contiguous(), pad_tensor}, /*dim=*/0);
}

}  // namespace xllm::layer::v32_sp
