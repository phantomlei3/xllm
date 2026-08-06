/* Copyright 2025-2026 The xLLM Authors.

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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace xllm {

class RateLimiter final {
  class SharedState;

 public:
  // Special value indicating sleep state.
  static constexpr int32_t kSleeping = INT32_MIN;

  class AdmissionSlot final {
   public:
    explicit AdmissionSlot(std::shared_ptr<SharedState> state);
    ~AdmissionSlot();

    AdmissionSlot(const AdmissionSlot&) = delete;
    AdmissionSlot& operator=(const AdmissionSlot&) = delete;
    AdmissionSlot(AdmissionSlot&&) = delete;
    AdmissionSlot& operator=(AdmissionSlot&&) = delete;

    // Returns this slot's capacity at most once.
    void release_once();

   private:
    std::shared_ptr<SharedState> state_;
    std::atomic<bool> released_{false};
  };

  using AdmissionSlotPtr = std::shared_ptr<AdmissionSlot>;

  RateLimiter();

  ~RateLimiter();

  RateLimiter(const RateLimiter&) = delete;
  RateLimiter& operator=(const RateLimiter&) = delete;
  RateLimiter(RateLimiter&&) = delete;
  RateLimiter& operator=(RateLimiter&&) = delete;

  // Atomically acquires request capacity. Returns nullptr if the concurrency
  // limit has been reached or the service is sleeping.
  [[nodiscard]] AdmissionSlotPtr try_acquire();

  // Returns true if request is rate-limited or sleeping.
  // If not limited and not sleeping, increments the counter.
  bool is_limited();

  // Releases an acquired slot. A release at zero or while sleeping is ignored.
  void decrease_one_request();

  // Releases up to decrease_requests_num acquired slots without underflowing.
  void decrease_requests(size_t decrease_requests_num);

  int32_t get_num_concurrent_requests() const;

  // CAS: only succeeds if num_concurrent_requests == 0.
  // Sets to kSleeping on success. Returns true on success.
  bool try_set_sleeping();

  // CAS: only succeeds if num_concurrent_requests == kSleeping.
  // Sets to 0 on success. Returns true on success.
  bool try_wakeup();

  bool is_sleeping() const;

 private:
  std::shared_ptr<SharedState> state_;
};

}  // namespace xllm
