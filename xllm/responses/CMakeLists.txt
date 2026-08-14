include(cc_library)

cc_library(
  NAME
    xllm_responses
  HDRS
    error.h
    responses_limits.h
    types.h
  SRCS
    error.cpp
  DEPS
    :xllm_model_protocol
    :xllm_tool_protocol
    nlohmann_json::nlohmann_json
)
