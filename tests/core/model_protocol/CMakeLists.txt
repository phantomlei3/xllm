include(cc_test)

cc_test(
  NAME
    model_protocol_profile_test
  SRCS
    profile_registry_test.cpp
  DEPS
    :xllm_model_protocol
    GTest::gtest_main
)
