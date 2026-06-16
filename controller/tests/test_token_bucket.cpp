#include "vllm_slo/token_bucket.hpp"

#include <chrono>
#include <stdexcept>
#include <thread>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    vllm_slo::TokenBucket bucket(2.0, 20.0);

    require(bucket.capacity() == 2.0, "capacity should match constructor");
    require(bucket.refill_rate_per_second() == 20.0, "refill rate should match constructor");
    require(bucket.try_consume(), "first token should be available");
    require(bucket.try_consume(), "second token should be available");
    require(!bucket.try_consume(), "third token should not be available immediately");

    std::this_thread::sleep_for(std::chrono::milliseconds(80));
    require(bucket.available_tokens() > 0.0, "tokens should refill over steady_clock time");
    require(bucket.try_consume(), "refilled token should be consumable");

    bool threw = false;
    try {
        vllm_slo::TokenBucket invalid(0.0, 1.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "invalid constructor argument should throw");

    threw = false;
    try {
        (void)bucket.try_consume(0.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "invalid consume amount should throw");

    return 0;
}
