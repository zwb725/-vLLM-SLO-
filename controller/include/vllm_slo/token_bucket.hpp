#pragma once

#include <chrono>
#include <mutex>

namespace vllm_slo {

// Thread-safe token bucket based on std::chrono::steady_clock.
class TokenBucket {
public:
    TokenBucket(double capacity, double refill_rate_per_second);

    [[nodiscard]] bool try_consume(double tokens = 1.0);
    [[nodiscard]] double available_tokens();
    [[nodiscard]] double capacity() const noexcept;
    [[nodiscard]] double refill_rate_per_second() const noexcept;

private:
    using Clock = std::chrono::steady_clock;

    void refill_unlocked(Clock::time_point now);

    const double capacity_;
    const double refill_rate_per_second_;
    mutable std::mutex mutex_;
    double tokens_;
    Clock::time_point last_refill_;
};

}  // namespace vllm_slo
