#include "vllm_slo/inflight_limiter.hpp"

#include <stdexcept>
#include <utility>

namespace vllm_slo {

struct InflightLimiter::Permit::State {
    State(std::size_t max_inflight_value, std::size_t max_queue_size_value)
        : max_inflight(max_inflight_value), max_queue_size(max_queue_size_value) {}

    mutable std::mutex mutex;
    std::size_t current_inflight{0};
    std::size_t queued_requests{0};
    std::size_t max_inflight{0};
    std::size_t max_queue_size{0};
    bool closed{false};
    std::function<void()> on_release;
};

InflightLimiter::Permit::Permit(std::weak_ptr<State> state)
    : state_(std::move(state)), owns_slot_(true) {}

InflightLimiter::Permit::Permit(Permit&& other) noexcept
    : state_(std::move(other.state_)), owns_slot_(other.owns_slot_) {
    other.owns_slot_ = false;
}

InflightLimiter::Permit& InflightLimiter::Permit::operator=(Permit&& other) noexcept {
    if (this != &other) {
        release();
        state_ = std::move(other.state_);
        owns_slot_ = other.owns_slot_;
        other.owns_slot_ = false;
    }
    return *this;
}

InflightLimiter::Permit::~Permit() {
    release();
}

void InflightLimiter::Permit::release() {
    if (!owns_slot_) {
        return;
    }
    owns_slot_ = false;

    auto state = state_.lock();
    if (!state) {
        return;
    }

    std::function<void()> callback;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->current_inflight == 0U) {
            return;
        }
        --state->current_inflight;
        callback = state->on_release;
    }

    if (callback) {
        callback();
    }
}

bool InflightLimiter::Permit::valid() const noexcept {
    return owns_slot_;
}

InflightLimiter::InflightLimiter(std::size_t max_inflight, std::size_t max_queue_size)
    : state_(std::make_shared<State>(max_inflight, max_queue_size)) {
    if (max_inflight == 0U) {
        throw std::invalid_argument("max_inflight must be greater than zero");
    }
}

InflightLimiter::AcquireResult InflightLimiter::try_acquire_or_queue() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->closed) {
        return {InflightDecision::Closed, Permit{}, 0U};
    }

    if (state_->current_inflight < state_->max_inflight) {
        ++state_->current_inflight;
        return {InflightDecision::Admit, Permit{state_}, 0U};
    }

    if (state_->queued_requests < state_->max_queue_size) {
        ++state_->queued_requests;
        return {InflightDecision::Queue, Permit{}, state_->queued_requests};
    }

    return {InflightDecision::Reject, Permit{}, 0U};
}

InflightLimiter::Permit InflightLimiter::try_promote_queued() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->closed || state_->queued_requests == 0U ||
        state_->current_inflight >= state_->max_inflight) {
        return Permit{};
    }

    --state_->queued_requests;
    ++state_->current_inflight;
    return Permit{state_};
}

bool InflightLimiter::cancel_queued() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->queued_requests == 0U) {
        return false;
    }
    --state_->queued_requests;
    return true;
}

void InflightLimiter::close() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->closed = true;
    state_->queued_requests = 0U;
}

void InflightLimiter::set_on_release(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->on_release = std::move(callback);
}

InflightLimiterSnapshot InflightLimiter::snapshot() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return InflightLimiterSnapshot{
        state_->current_inflight,
        state_->queued_requests,
        state_->max_inflight,
        state_->max_queue_size,
        state_->closed,
    };
}

const char* to_string(InflightDecision decision) noexcept {
    switch (decision) {
        case InflightDecision::Admit:
            return "Admit";
        case InflightDecision::Queue:
            return "Queue";
        case InflightDecision::Reject:
            return "Reject";
        case InflightDecision::Closed:
            return "Closed";
    }
    return "Unknown";
}

}  // namespace vllm_slo
