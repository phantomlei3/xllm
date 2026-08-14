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

namespace xllm::model_protocol {

enum class ThinkingHistoryPolicy : uint8_t {
  TEMPLATE_DEFAULT = 0,
  PRESERVE = 1,
};

enum class OutputDecodingPolicy : uint8_t {
  VISIBLE_TEXT = 0,
  PROTOCOL_RAW = 1,
};

enum class ReasoningEffort : uint8_t {
  NONE = 0,
  MINIMAL = 1,
  LOW = 2,
  MEDIUM = 3,
  HIGH = 4,
  XHIGH = 5,
  MAX = 6,
};

struct SamplingPolicy {
  ReasoningEffort effort = ReasoningEffort::MEDIUM;
  float temperature = 1.0f;
  float top_p = 0.95f;
};

struct TemplatePolicy {
  ThinkingHistoryPolicy thinking_history =
      ThinkingHistoryPolicy::TEMPLATE_DEFAULT;
};

}  // namespace xllm::model_protocol
