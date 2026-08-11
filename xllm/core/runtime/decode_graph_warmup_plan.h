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

// Decode graph shape fields supplied by the owning Engine. For MTP,
// num_decoding_tokens is the number of token rows produced per sequence in a
// decode step; it is normally num_speculative_tokens + 1.
struct DecodeGraphExecutionShape {
  int64_t num_decoding_tokens = 1;
  int32_t num_speculative_tokens = 0;
  bool enable_graph_mode_decode_no_padding = false;
};

// A graph warmup schedule for decode. batch_sizes contains global scheduler
// batch sizes, not per-DP-rank sizes; the plan builder accounts for dp_size
// when selecting them. The profile manager executes these entries to capture
// the graph shapes an executor can later replay.
struct DecodeGraphWarmupPlan {
  DecodeGraphExecutionShape execution_shape;
  std::vector<int32_t> batch_sizes;
};

// Returns the padded token-row bucket shared by decode graph executors. When
// no-padding mode is enabled each exact token count is its own graph shape.
int64_t get_decode_graph_token_bucket(int64_t num_tokens,
                                      bool enable_no_padding);

// Returns the legacy, backend-neutral decode schedule. Keep this schedule for
// a backend until it explicitly advertises a more precise graph warmup
// capability through Platform.
DecodeGraphWarmupPlan get_compatibility_decode_graph_warmup_plan(
    int32_t max_global_batch_size,
    int32_t dp_size);

// Builds the decode graph warmup schedule from the Engine's effective runtime
// options. Backends that support MTP token-bucket warmup replace the legacy
// schedule only for padded multi-token decode; all other cases preserve the
// compatibility schedule above.
DecodeGraphWarmupPlan build_decode_graph_warmup_plan(
    const Options& options,
    int32_t max_global_batch_size,
    int32_t dp_size);

}  // namespace xllm::runtime
