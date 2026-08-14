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
#include <string>

#include "core/model_protocol/profile_registry.h"

namespace xllm::responses {

enum class ErrorCode : uint8_t {
  INVALID_JSON = 0,
  INVALID_REQUEST = 1,
  UNSUPPORTED_PARAMETER = 2,
  UNSUPPORTED_ITEM_TYPE = 3,
  UNSUPPORTED_CONTENT_TYPE = 4,
  UNSUPPORTED_MODEL_CAPABILITY = 5,
  MODEL_NOT_FOUND = 6,
  MODEL_MISMATCH = 7,
  UNKNOWN_TOOL = 8,
  UNKNOWN_CALL_ID = 9,
  DUPLICATE_CALL_ID = 10,
  TOOL_CALL_TYPE_MISMATCH = 11,
  INVALID_ITEM_ORDER = 12,
  INVALID_TOOL_ARGUMENTS = 13,
  REQUEST_TOO_LARGE = 14,
  TOO_MANY_ITEMS = 15,
  MAX_DEPTH_EXCEEDED = 16,
  GENERATION_FAILED = 17,
  REQUEST_CANCELLED = 18,
  REQUEST_TIMEOUT = 19,
  CLIENT_TOO_SLOW = 20,
};

struct ResponsesError {
  std::string message;
  std::string type = "invalid_request_error";
  std::string param;
  ErrorCode code = ErrorCode::INVALID_REQUEST;
};

ResponsesError from_protocol_error(
    const model_protocol::ModelProtocolError& error);

}  // namespace xllm::responses
