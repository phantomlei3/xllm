include(cc_library)

cc_library(
  NAME
    xllm_model_protocol
  HDRS
    output_parser.h
    policy.h
    profile.h
    profile_registry.h
  SRCS
    profile_registry.cpp
)
