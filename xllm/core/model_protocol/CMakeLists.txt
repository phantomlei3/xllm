include(cc_library)

cc_library(
  NAME
    xllm_model_protocol
  HDRS
    deepseek_v4_profile.h
    glm_5_2_profile.h
    output_parser.h
    policy.h
    profile.h
    profile_registry.h
  SRCS
    deepseek_v4_profile.cpp
    glm_5_2_profile.cpp
    profile_registry.cpp
)
