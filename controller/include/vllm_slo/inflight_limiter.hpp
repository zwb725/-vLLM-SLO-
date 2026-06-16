#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>

namespace vllm_slo {

enum class InflightDecision {
    Admit,
    Queue,
    Reject,
    Closed
};

struct InflightLimiterSnapshot {
    std::size_t current_inflight{0};
    std::size_t queued_requests{0};
    std::size_t max_inflight{0};
    std::size_t max_queue_size{0};
    bool closed{false};
};

// Thread-safe fixed in-flight limiter with a bounded queue counter.
//
// The limiter owns only counts. Callers own queued request objects and must call
// cancel_queued() when a queued request times out or disconnects before promote.
class InflightLimiter {
public:
    class Permit {
    public:
        Permit() = default;
        Permit(const Permit&) = delete;
        Permit& operator=(const Permit&) = delete;
        Permit(Permit&& other) noexcept;
        Permit& operator=(Permit&& other) noexcept;
        ~Permit();

        void release();
        [[nodiscard]] bool valid() const noexcept;

    private:
        friend class InflightLimiter;
        struct State;

        explicit Permit(std::weak_ptr<State> state);
        std::weak_ptr<State> state_;
        bool owns_slot_{false};
    };

    struct AcquireResult {
        InflightDecision decision{InflightDecision::Reject};
        Permit permit;
        std::size_t queue_position{0};
    };

    InflightLimiter(std::size_t max_inflight, std::size_t max_queue_size);

    [[nodiscard]] AcquireResult try_acquire_or_queue();
    [[nodiscard]] Permit try_promote_queued();
    [[nodiscard]] bool cancel_queued();
    void close();
    void set_on_release(std::function<void()> callback);
    [[nodiscard]] InflightLimiterSnapshot snapshot() const;

private:
    using State = Permit::State;
    std::shared_ptr<State> state_;
};

[[nodiscard]] const char* to_string(InflightDecision decision) noexcept;

}  // namespace vllm_slo
