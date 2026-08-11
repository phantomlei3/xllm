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

#include "runtime/decode_graph_warmup_plan.h"

#include <array>
#include <cstddef>
#include <utility>

#include "platform/platform.h"
#include "runtime/options.h"

namespace xllm::runtime {
namespace {

constexpr int64_t kGraphTokenStep = 16;
constexpr std::array<int32_t, 5> kCompatibilitySmallBatchSizes = {1,
                                                                  2,
                                                                  4,
                                                                  8,
                                                                  16};

}  // namespace

int64_t get_decode_graph_token_bucket(int64_t num_tokens,
                                      bool enable_no_padding) {
  if (enable_no_padding) {
    return num_tokens;
  }
  if (num_tokens <= 1) {
    return 1;
  }
  if (num_tokens <= 2) {
    return 2;
  }
  if (num_tokens <= 4) {
    return 4;
  }
  if (num_tokens <= 8) {
    return 8;
  }

  return ((num_tokens + kGraphTokenStep - 1) / kGraphTokenStep) *
         kGraphTokenStep;
}

DecodeGraphWarmupPlan get_compatibility_decode_graph_warmup_plan(
    int32_t max_global_batch_size,
    int32_t dp_size) {
  DecodeGraphWarmupPlan plan;
  if (max_global_batch_size <= 0 || dp_size <= 0) {
    return plan;
  }

  const size_t max_bucket_count = kCompatibilitySmallBatchSizes.size() +
                                  static_cast<size_t>(max_global_batch_size) /
                                      static_cast<size_t>(kGraphTokenStep) +
                                  1;
  plan.batch_sizes.reserve(max_bucket_count);
  for (int32_t batch_size : kCompatibilitySmallBatchSizes) {
    if (batch_size >= dp_size && batch_size <= max_global_batch_size) {
      plan.batch_sizes.emplace_back(batch_size);
    }
  }

  for (int32_t batch_size = static_cast<int32_t>(kGraphTokenStep * 2);
       batch_size <= max_global_batch_size;
       batch_size += static_cast<int32_t>(kGraphTokenStep)) {
    if (batch_size >= dp_size) {
      plan.batch_sizes.emplace_back(batch_size);
    }
  }

  if (max_global_batch_size >= dp_size &&
      (plan.batch_sizes.empty() ||
       plan.batch_sizes.back() != max_global_batch_size)) {
    plan.batch_sizes.emplace_back(max_global_batch_size);
  }

  return plan;
}

DecodeGraphWarmupPlan build_decode_graph_warmup_plan(
    const Options& options,
    int32_t max_global_batch_size,
    int32_t dp_size) {
  DecodeGraphWarmupPlan plan = get_compatibility_decode_graph_warmup_plan(
      max_global_batch_size, dp_size);
  plan.execution_shape.num_decoding_tokens = options.num_decoding_tokens();
  plan.execution_shape.num_speculative_tokens =
      options.num_speculative_tokens();
  plan.execution_shape.enable_graph_mode_decode_no_padding =
      options.enable_graph_mode_decode_no_padding();

  // MTP emits num_decoding_tokens rows per sequence. On supporting backends,
  // the graph cache is keyed by the padded number of rows rather than the
  // sequence count. Therefore the compatibility schedule can miss a graph
  // bucket (for example, batch size 9 with four decode tokens starts the
  // 48-token bucket). Platform owns this capability decision so this generic
  // plan does not depend on a device build macro. A backend must not opt in
  // until its graph keying and MTP replay behavior are covered by tests.
  if (!::xllm::Platform::supports_mtp_decode_graph_warmup()) {
    return plan;
  }

  const bool use_mtp_batches =
      !plan.execution_shape.enable_graph_mode_decode_no_padding &&
      plan.execution_shape.num_decoding_tokens > 1 &&
      max_global_batch_size >= dp_size && dp_size > 0;
  if (!use_mtp_batches) {
    return plan;
  }

  const int32_t max_local_batch_size = max_global_batch_size / dp_size;
  std::vector<int32_t> batch_sizes;
  batch_sizes.reserve(static_cast<size_t>(max_local_batch_size) + 1);
  int64_t last_token_bucket = 0;
  for (int32_t local_batch_size = 1; local_batch_size <= max_local_batch_size;
       ++local_batch_size) {
    const int64_t num_tokens = static_cast<int64_t>(local_batch_size) *
                               plan.execution_shape.num_decoding_tokens;
    const int64_t token_bucket = get_decode_graph_token_bucket(
        num_tokens, plan.execution_shape.enable_graph_mode_decode_no_padding);
    if (batch_sizes.empty() || token_bucket != last_token_bucket) {
      batch_sizes.emplace_back(local_batch_size * dp_size);
      last_token_bucket = token_bucket;
    }
  }

  const int32_t max_full_global_batch_size = max_local_batch_size * dp_size;
  if (max_full_global_batch_size < max_global_batch_size) {
    batch_sizes.emplace_back(max_global_batch_size);
  }
  plan.batch_sizes = std::move(batch_sizes);

  return plan;
}

}  // namespace xllm::runtime
