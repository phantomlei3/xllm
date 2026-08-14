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

#include "responses/error.h"

namespace xllm::responses {

ResponsesError from_protocol_error(
    const model_protocol::ModelProtocolError& error) {
  ErrorCode code = ErrorCode::INVALID_REQUEST;
  if (error.code() ==
      model_protocol::ModelProtocolErrorCode::UNSUPPORTED_MODEL_CAPABILITY) {
    code = ErrorCode::UNSUPPORTED_MODEL_CAPABILITY;
  } else if (error.code() == model_protocol::ModelProtocolErrorCode::
                                 PROFILE_IDENTITY_MISMATCH) {
    code = ErrorCode::MODEL_MISMATCH;
  }
  return {.message = error.message(),
          .type = "invalid_request_error",
          .param = "model",
          .code = code};
}

}  // namespace xllm::responses
