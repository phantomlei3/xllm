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
