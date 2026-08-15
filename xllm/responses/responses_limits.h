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

namespace xllm::responses {

struct ResponsesLimits {
  uint64_t max_body_bytes = 16 * 1024 * 1024;
  uint32_t max_json_depth = 64;
  uint64_t max_client_metadata_bytes = 1024 * 1024;
  uint32_t max_input_items = 4096;
  uint64_t max_text_bytes = 4 * 1024 * 1024;
  uint32_t max_tools = 128;
  uint64_t max_function_schema_bytes = 1024 * 1024;
  uint64_t max_tool_bytes = 4 * 1024 * 1024;
  uint64_t max_function_args_bytes = 1024 * 1024;
  uint64_t max_custom_payload_bytes = 4 * 1024 * 1024;
  uint64_t max_sse_buffer_bytes = 4 * 1024 * 1024;
};

}  // namespace xllm::responses
