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

#include <string_view>

namespace xllm {
namespace util {

inline bool is_deepseek_v3_model_type(std::string_view model_type) {
  return model_type == "deepseek_v3" || model_type == "deepseek-v3" ||
         model_type == "deepseek_v3_mtp" || model_type == "deepseek-v3-mtp";
}

inline bool should_enable_prefill_mqa(std::string_view model_type) {
  return !is_deepseek_v3_model_type(model_type) &&
         model_type.find("joyai") == std::string_view::npos;
}

}  // namespace util
}  // namespace xllm
