include(cc_test)

cc_test(
  NAME
    mlu_beam_search_test
  SRCS
    beam_search_test.cpp
  DEPS
    :mlu_kernels
    :flags
    torch
    torch_mlu
    cnrt
    GTest::gtest_main
    glog::glog
)
