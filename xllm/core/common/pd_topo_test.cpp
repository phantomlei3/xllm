/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "common/pd_topo.h"

#include <gtest/gtest.h>

#include <vector>

namespace xllm {
namespace {

TEST(PdTopoTest, CheckTp4ToTp8) {
  auto check = check_mlu_pd_topo(/*src_world_size=*/4,
                                 /*src_dp_size=*/1,
                                 /*dst_world_size=*/8,
                                 /*dst_dp_size=*/1,
                                 /*dst_cp_size=*/1,
                                 /*enable_mla=*/true);
  EXPECT_TRUE(check.ok());
  EXPECT_EQ(check.topo.src_tp_size, 4);
  EXPECT_EQ(check.topo.dst_tp_size, 8);
}

TEST(PdTopoTest, CheckTp8ToTp4) {
  auto check = check_mlu_pd_topo(/*src_world_size=*/8,
                                 /*src_dp_size=*/1,
                                 /*dst_world_size=*/4,
                                 /*dst_dp_size=*/1,
                                 /*dst_cp_size=*/1,
                                 /*enable_mla=*/true);
  EXPECT_TRUE(check.ok());
  EXPECT_EQ(check.topo.src_tp_size, 8);
  EXPECT_EQ(check.topo.dst_tp_size, 4);
}

TEST(PdTopoTest, CheckSameTp) {
  auto check = check_mlu_pd_topo(/*src_world_size=*/4,
                                 /*src_dp_size=*/1,
                                 /*dst_world_size=*/4,
                                 /*dst_dp_size=*/1,
                                 /*dst_cp_size=*/1,
                                 /*enable_mla=*/true);
  EXPECT_TRUE(check.ok());
}

TEST(PdTopoTest, CheckUnsupDp) {
  auto check = check_mlu_pd_topo(/*src_world_size=*/8,
                                 /*src_dp_size=*/2,
                                 /*dst_world_size=*/8,
                                 /*dst_dp_size=*/1,
                                 /*dst_cp_size=*/1,
                                 /*enable_mla=*/true);
  EXPECT_EQ(check.stat, PdTopoStat::kUnsup);
}

TEST(PdTopoTest, CheckUnsupCp) {
  auto check = check_mlu_pd_topo(/*src_world_size=*/4,
                                 /*src_dp_size=*/1,
                                 /*dst_world_size=*/8,
                                 /*dst_dp_size=*/1,
                                 /*dst_cp_size=*/2,
                                 /*enable_mla=*/true);
  EXPECT_EQ(check.stat, PdTopoStat::kUnsup);
}

TEST(PdTopoTest, CheckInvalid) {
  auto check = check_mlu_pd_topo(/*src_world_size=*/3,
                                 /*src_dp_size=*/2,
                                 /*dst_world_size=*/8,
                                 /*dst_dp_size=*/1,
                                 /*dst_cp_size=*/1,
                                 /*enable_mla=*/true);
  EXPECT_EQ(check.stat, PdTopoStat::kInvalid);
}

TEST(PdTopoTest, GetPullMapTp4ToTp8) {
  auto check = check_mlu_pd_topo(/*src_world_size=*/4,
                                 /*src_dp_size=*/1,
                                 /*dst_world_size=*/8,
                                 /*dst_dp_size=*/1,
                                 /*dst_cp_size=*/1,
                                 /*enable_mla=*/true);
  ASSERT_TRUE(check.ok());

  const std::vector<int32_t> want = {0, 1, 2, 3, 0, 1, 2, 3};
  for (int32_t tp_rank = 0; tp_rank < static_cast<int32_t>(want.size());
       ++tp_rank) {
    PdPullMap pull_map;
    ASSERT_TRUE(get_pd_pull_map(check.topo,
                                /*src_dp_rank=*/0,
                                /*dst_dp_rank=*/0,
                                tp_rank,
                                &pull_map));
    EXPECT_EQ(pull_map.dst_worker_rank, tp_rank);
    EXPECT_EQ(pull_map.src_worker_rank, want[tp_rank]);
  }
}

TEST(PdTopoTest, GetPullMapTp8ToTp4) {
  auto check = check_mlu_pd_topo(/*src_world_size=*/8,
                                 /*src_dp_size=*/1,
                                 /*dst_world_size=*/4,
                                 /*dst_dp_size=*/1,
                                 /*dst_cp_size=*/1,
                                 /*enable_mla=*/true);
  ASSERT_TRUE(check.ok());

  for (int32_t tp_rank = 0; tp_rank < 4; ++tp_rank) {
    PdPullMap pull_map;
    ASSERT_TRUE(get_pd_pull_map(check.topo,
                                /*src_dp_rank=*/0,
                                /*dst_dp_rank=*/0,
                                tp_rank,
                                &pull_map));
    EXPECT_EQ(pull_map.dst_worker_rank, tp_rank);
    EXPECT_EQ(pull_map.src_worker_rank, tp_rank);
  }
}

TEST(PdTopoTest, GetPullMapRejectsBadRank) {
  auto check = check_mlu_pd_topo(/*src_world_size=*/4,
                                 /*src_dp_size=*/1,
                                 /*dst_world_size=*/8,
                                 /*dst_dp_size=*/1,
                                 /*dst_cp_size=*/1,
                                 /*enable_mla=*/true);
  ASSERT_TRUE(check.ok());

  PdPullMap pull_map;
  EXPECT_FALSE(get_pd_pull_map(check.topo,
                               /*src_dp_rank=*/0,
                               /*dst_dp_rank=*/0,
                               /*dst_tp_rank=*/8,
                               &pull_map));
}

TEST(PdTopoTest, CheckNonMlaSameTp) {
  auto check = check_mlu_pd_topo(/*src_world_size=*/4,
                                 /*src_dp_size=*/1,
                                 /*dst_world_size=*/4,
                                 /*dst_dp_size=*/1,
                                 /*dst_cp_size=*/2,
                                 /*enable_mla=*/false);
  EXPECT_TRUE(check.ok());
}

TEST(PdTopoTest, CheckNonMlaHeteroTp) {
  auto check = check_mlu_pd_topo(/*src_world_size=*/4,
                                 /*src_dp_size=*/1,
                                 /*dst_world_size=*/8,
                                 /*dst_dp_size=*/1,
                                 /*dst_cp_size=*/1,
                                 /*enable_mla=*/false);
  EXPECT_EQ(check.stat, PdTopoStat::kUnsup);
}

TEST(PdTopoTest, CheckTopoStatStrByMode) {
  EXPECT_STREQ("unsupported mla pd topology",
               pd_topo_stat_str(PdTopoStat::kUnsup, /*enable_mla=*/true));
  EXPECT_STREQ("unsupported pd topology",
               pd_topo_stat_str(PdTopoStat::kUnsup, /*enable_mla=*/false));
}

}  // namespace
}  // namespace xllm
