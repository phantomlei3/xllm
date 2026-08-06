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

#include "core/common/rate_limiter.h"

#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <thread>
#include <vector>

#include "core/framework/config/service_config.h"

namespace xllm {

TEST(RequestLimiterTest, Basic) {
  // Set the maximum number of concurrent requests to 1.
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter rate_limiter;
  // The current number of concurrent requests is 0, no rate limiting is
  // applied.
  EXPECT_EQ(rate_limiter.is_limited(), false);
  // The current number of concurrent requests is 1, rate limiting is applied.
  EXPECT_EQ(rate_limiter.is_limited(), true);
  // Decrease the number of concurrent requests by one, changing the concurrency
  // from 1 to 0.
  rate_limiter.decrease_one_request();
  // The current number of concurrent requests is 0, no rate limiting is
  // applied.
  EXPECT_EQ(rate_limiter.is_limited(), false);
}

TEST(RequestLimiterTest, ConcurrentAdmissionDoesNotExceedCapacity) {
  constexpr int32_t kMaxConcurrentRequests = 8;
  constexpr int32_t kNumThreads = 128;
  ServiceConfig::get_instance().max_concurrent_requests(kMaxConcurrentRequests);
  RateLimiter rate_limiter;
  std::barrier start_barrier(kNumThreads);
  std::atomic<int32_t> admitted_requests{0};
  std::vector<RateLimiter::AdmissionSlotPtr> admission_slots(kNumThreads);

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (size_t i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&, i]() {
      start_barrier.arrive_and_wait();
      admission_slots[i] = rate_limiter.try_acquire();
      if (admission_slots[i] != nullptr) {
        admitted_requests.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(admitted_requests.load(std::memory_order_relaxed),
            kMaxConcurrentRequests);
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), kMaxConcurrentRequests);
}

TEST(RequestLimiterTest, AdmissionSlotReleaseIsIdempotent) {
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter rate_limiter;
  RateLimiter::AdmissionSlotPtr admission_slot = rate_limiter.try_acquire();

  ASSERT_NE(admission_slot, nullptr);
  admission_slot->release_once();
  admission_slot->release_once();

  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);
}

TEST(RequestLimiterTest, ConcurrentReleaseReturnsCapacityOnce) {
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter rate_limiter;
  RateLimiter::AdmissionSlotPtr admission_slot = rate_limiter.try_acquire();
  ASSERT_NE(admission_slot, nullptr);

  constexpr int32_t kNumThreads = 128;
  std::barrier start_barrier(kNumThreads);
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int32_t i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&]() {
      start_barrier.arrive_and_wait();
      admission_slot->release_once();
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);
}

TEST(RequestLimiterTest, AdmissionSlotDestructorReturnsCapacity) {
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter rate_limiter;
  {
    RateLimiter::AdmissionSlotPtr admission_slot = rate_limiter.try_acquire();
    ASSERT_NE(admission_slot, nullptr);
    EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 1);
  }

  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);
}

TEST(RequestLimiterTest, AdmissionSlotCanOutliveRateLimiter) {
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter::AdmissionSlotPtr admission_slot;
  {
    RateLimiter rate_limiter;
    admission_slot = rate_limiter.try_acquire();
    ASSERT_NE(admission_slot, nullptr);
  }

  admission_slot->release_once();
}

TEST(RequestLimiterTest, SleepingRejectsAdmissionWithoutChangingState) {
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter rate_limiter;
  ASSERT_TRUE(rate_limiter.try_set_sleeping());

  EXPECT_EQ(rate_limiter.try_acquire(), nullptr);
  EXPECT_TRUE(rate_limiter.is_sleeping());
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), RateLimiter::kSleeping);
}

TEST(RequestLimiterTest, UnmatchedReleaseDoesNotUnderflow) {
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter rate_limiter;

  rate_limiter.decrease_one_request();
  rate_limiter.decrease_requests(4);
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);

  ASSERT_TRUE(rate_limiter.try_set_sleeping());
  rate_limiter.decrease_one_request();
  rate_limiter.decrease_requests(4);
  EXPECT_TRUE(rate_limiter.is_sleeping());
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), RateLimiter::kSleeping);
}

}  // namespace xllm
