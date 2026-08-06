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

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int32_t i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&]() {
      start_barrier.arrive_and_wait();
      if (!rate_limiter.is_limited()) {
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
