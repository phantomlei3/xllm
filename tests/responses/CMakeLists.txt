include(cc_test)

add_library(responses_fixture_loader STATIC fixture_loader.cpp)
target_include_directories(
  responses_fixture_loader
  PUBLIC "${PROJECT_SOURCE_DIR}/tests"
)
target_link_libraries(
  responses_fixture_loader
  PUBLIC nlohmann_json::nlohmann_json
)

cc_test(
  NAME
    responses_fixture_conformance_test
  SRCS
    fixture_conformance_test.cpp
  DEPS
    GTest::gtest_main
    responses_fixture_loader
)

target_compile_definitions(
  responses_fixture_conformance_test
  PRIVATE XLLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

cc_test(
  NAME
    responses_typed_protocol_test
  SRCS
    typed_protocol_test.cpp
  DEPS
    :xllm_tool_protocol
    :xllm_responses
    GTest::gtest_main
    proto::xllm_proto
)

cc_test(
  NAME
    responses_request_preparer_test
  SRCS
    request_preparer_test.cpp
  DEPS
    :xllm_responses
    GTest::gtest_main
    proto::xllm_proto
    responses_fixture_loader
)

target_compile_definitions(
  responses_request_preparer_test
  PRIVATE XLLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

cc_test(
  NAME
    responses_output_processor_test
  SRCS
    output_processor_test.cpp
  DEPS
    :xllm_responses
    GTest::gtest_main
    responses_fixture_loader
)

cc_test(
  NAME
    responses_sse_encoder_test
  SRCS
    sse_encoder_test.cpp
  DEPS
    :xllm_responses
    GTest::gtest_main
)

target_compile_definitions(
  responses_output_processor_test
  PRIVATE XLLM_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
