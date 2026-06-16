#include "vllm_slo/inflight_limiter.hpp"

#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    vllm_slo::InflightLimiter limiter(1, 1);
    int release_callbacks = 0;
    limiter.set_on_release([&release_callbacks]() { ++release_callbacks; });

    auto first = limiter.try_acquire_or_queue();
    require(first.decision == vllm_slo::InflightDecision::Admit, "first request should admit");
    require(first.permit.valid(), "admitted request should own a permit");

    auto queued = limiter.try_acquire_or_queue();
    require(queued.decision == vllm_slo::InflightDecision::Queue, "second request should queue");
    require(queued.queue_position == 1U, "queued request should report position");

    auto rejected = limiter.try_acquire_or_queue();
    require(rejected.decision == vllm_slo::InflightDecision::Reject, "full queue should reject");

    first.permit.release();
    require(release_callbacks == 1, "release callback should run once");

    auto promoted = limiter.try_promote_queued();
    require(promoted.valid(), "queued request should promote after release");
    require(limiter.snapshot().queued_requests == 0U, "promotion should decrement queue");

    promoted.release();
    require(limiter.snapshot().current_inflight == 0U, "permit release should clear inflight");

    auto active = limiter.try_acquire_or_queue();
    require(active.decision == vllm_slo::InflightDecision::Admit, "new active request should admit");
    auto queued_for_cancel = limiter.try_acquire_or_queue();
    require(queued_for_cancel.decision == vllm_slo::InflightDecision::Queue, "request should queue before cancel");
    require(limiter.cancel_queued(), "cancel should remove queued request");
    require(limiter.snapshot().queued_requests == 0U, "cancel should leave no queued request");
    active.permit.release();

    {
        auto raii = limiter.try_acquire_or_queue();
        require(raii.decision == vllm_slo::InflightDecision::Admit, "RAII request should admit");
    }
    require(limiter.snapshot().current_inflight == 0U, "RAII destructor should release slot");

    limiter.close();
    auto after_close = limiter.try_acquire_or_queue();
    require(after_close.decision == vllm_slo::InflightDecision::Closed, "closed limiter should not accept");
    require(limiter.snapshot().closed, "snapshot should mark closed");

    bool threw = false;
    try {
        vllm_slo::InflightLimiter invalid(0, 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "invalid limiter config should throw");

    return 0;
}
