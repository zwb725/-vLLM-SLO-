#include "vllm_slo/token_bucket.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vllm_slo {

TokenBucket::TokenBucket(double capacity, double refill_rate_per_second)
    : capacity_(capacity),
      refill_rate_per_second_(refill_rate_per_second),
      tokens_(capacity),
      last_refill_(Clock::now()) {
    if (!std::isfinite(capacity_) || capacity_ <= 0.0) {
        throw std::invalid_argument("TokenBucket capacity must be finite and positive");
    }
    if (!std::isfinite(refill_rate_per_second_) || refill_rate_per_second_ <= 0.0) {
        throw std::invalid_argument("TokenBucket refill rate must be finite and positive");
    }
}

bool TokenBucket::try_consume(double tokens) {
    if (!std::isfinite(tokens) || tokens <= 0.0) {
        throw std::invalid_argument("consumed tokens must be finite and positive");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    refill_unlocked(Clock::now());

    if (tokens_ < tokens) {
        return false;
    }

    tokens_ -= tokens;
    return true;
}

double TokenBucket::available_tokens() {
    std::lock_guard<std::mutex> lock(mutex_);
    refill_unlocked(Clock::now());
    return tokens_;
}

double TokenBucket::capacity() const noexcept {
    return capacity_;
}

double TokenBucket::refill_rate_per_second() const noexcept {
    return refill_rate_per_second_;
}

void TokenBucket::refill_unlocked(Clock::time_point now) {
    if (now <= last_refill_) {
        return;
    }

    const auto elapsed = std::chrono::duration<double>(now - last_refill_).count();
    tokens_ = std::min(capacity_, tokens_ + elapsed * refill_rate_per_second_);
    last_refill_ = now;
}

}  // namespace vllm_slo
