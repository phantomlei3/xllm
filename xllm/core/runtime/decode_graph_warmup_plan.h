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
#include <vector>

namespace xllm::runtime {

struct Options;

struct DecodeGraphExecutionShape {
  int64_t num_decoding_tokens = 1;
  int32_t num_speculative_tokens = 0;
  bool enable_graph_mode_decode_no_padding = false;
};

struct DecodeGraphWarmupPlan {
  DecodeGraphExecutionShape execution_shape;
  std::vector<int32_t> batch_sizes;
};

int64_t get_decode_graph_token_bucket(int64_t num_tokens,
                                      bool enable_no_padding);

DecodeGraphWarmupPlan get_compatibility_decode_graph_warmup_plan(
    int32_t max_global_batch_size,
    int32_t dp_size);

DecodeGraphWarmupPlan build_decode_graph_warmup_plan(
    const Options& options,
    int32_t max_global_batch_size,
    int32_t dp_size);

}  // namespace xllm::runtime
