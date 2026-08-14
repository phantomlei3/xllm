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

cc_test(
  NAME
    model_output_parser_test
  SRCS
    output_parser_test.cpp
  DEPS
    :xllm_model_protocol
    GTest::gtest_main
    responses_fixture_loader
)

target_compile_definitions(
  model_output_parser_test
  PRIVATE XLLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
