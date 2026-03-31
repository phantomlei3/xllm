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

#pragma once

#include <cstdint>

namespace xllm {

enum class PdTopoStat : uint8_t {
  kOk = 0,
  kInvalid = 1,
  kUnsup = 2,
};

struct PdTopoInfo {
  int32_t src_world_size = 0;
  int32_t src_dp_size = 0;
  int32_t src_tp_size = 0;
  int32_t dst_world_size = 0;
  int32_t dst_dp_size = 0;
  int32_t dst_tp_size = 0;
  int32_t dst_cp_size = 1;
};

struct PdTopoCheck {
  PdTopoStat stat = PdTopoStat::kInvalid;
  PdTopoInfo topo;

  bool ok() const { return stat == PdTopoStat::kOk; }
};

struct PdPullMap {
  int32_t dst_worker_rank = 0;
  int32_t src_dp_worker_rank = 0;
  int32_t src_worker_rank = 0;
};

inline const char* pd_topo_stat_str(PdTopoStat stat, bool enable_mla) {
  switch (stat) {
    case PdTopoStat::kOk:
      return "ok";
    case PdTopoStat::kInvalid:
      return "topology invalid";
    case PdTopoStat::kUnsup:
      return enable_mla ? "unsupported mla pd topology"
                        : "unsupported pd topology";
  }
  return "unknown";
}

inline PdTopoCheck check_mlu_pd_topo(int32_t src_world_size,
                                     int32_t src_dp_size,
                                     int32_t dst_world_size,
                                     int32_t dst_dp_size,
                                     int32_t dst_cp_size,
                                     bool enable_mla) {
  PdTopoCheck check;
  if (src_world_size <= 0 || dst_world_size <= 0 || src_dp_size <= 0 ||
      dst_dp_size <= 0 || dst_cp_size <= 0 ||
      src_world_size % src_dp_size != 0 || dst_world_size % dst_dp_size != 0) {
    return check;
  }

  check.topo.src_world_size = src_world_size;
  check.topo.src_dp_size = src_dp_size;
  check.topo.src_tp_size = src_world_size / src_dp_size;
  check.topo.dst_world_size = dst_world_size;
  check.topo.dst_dp_size = dst_dp_size;
  check.topo.dst_tp_size = dst_world_size / dst_dp_size;
  check.topo.dst_cp_size = dst_cp_size;

  if (check.topo.src_tp_size <= 0 || check.topo.dst_tp_size <= 0) {
    return check;
  }

  if (enable_mla) {
    // Source cp_size is not carried in the current PD protocol. V1 only
    // validates the destination runtime and relies on the deployment contract
    // that source cp_size is also 1.
    if (check.topo.src_dp_size != 1 || check.topo.dst_dp_size != 1 ||
        check.topo.dst_cp_size != 1) {
      check.stat = PdTopoStat::kUnsup;
      return check;
    }
  } else if (check.topo.src_dp_size != check.topo.dst_dp_size ||
             check.topo.src_tp_size != check.topo.dst_tp_size) {
    check.stat = PdTopoStat::kUnsup;
    return check;
  }

  check.stat = PdTopoStat::kOk;
  return check;
}

inline bool get_pd_pull_map(const PdTopoInfo& topo,
                            int32_t src_dp_rank,
                            int32_t dst_dp_rank,
                            int32_t dst_tp_rank,
                            PdPullMap* pull_map) {
  if (pull_map == nullptr || topo.src_world_size <= 0 ||
      topo.dst_world_size <= 0 || topo.src_dp_size <= 0 ||
      topo.dst_dp_size <= 0 || topo.src_tp_size <= 0 || topo.dst_tp_size <= 0 ||
      src_dp_rank < 0 || dst_dp_rank < 0 || src_dp_rank >= topo.src_dp_size ||
      dst_dp_rank >= topo.dst_dp_size || dst_tp_rank < 0 ||
      dst_tp_rank >= topo.dst_tp_size) {
    return false;
  }

  pull_map->dst_worker_rank = dst_dp_rank * topo.dst_tp_size + dst_tp_rank;
  pull_map->src_dp_worker_rank = pull_map->dst_worker_rank % topo.src_tp_size;
  pull_map->src_worker_rank =
      src_dp_rank * topo.src_tp_size + pull_map->src_dp_worker_rank;

  return pull_map->dst_worker_rank >= 0 &&
         pull_map->dst_worker_rank < topo.dst_world_size &&
         pull_map->src_worker_rank >= 0 &&
         pull_map->src_worker_rank < topo.src_world_size;
}

}  // namespace xllm
