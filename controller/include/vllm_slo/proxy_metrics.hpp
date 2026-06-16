#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>

#include "vllm_slo/sliding_window.hpp"

namespace vllm_slo {

struct ProxyMetricsSnapshot {
    std::uint64_t total_requests{0};
    std::uint64_t completed_requests{0};
    std::uint64_t failed_requests{0};
    std::uint64_t first_token_events{0};
    std::optional<double> last_ttft_ms;
    std::optional<double> p95_ttft_ms;
};

// Thread-safe request metrics for the proxy. Durations are stored in milliseconds.
class ProxyMetrics {
public:
    explicit ProxyMetrics(std::size_t ttft_window_capacity);

    void record_request_started();
    void record_request_completed();
    void record_request_failed();
    void record_ttft(
        std::chrono::steady_clock::time_point request_start,
        std::chrono::steady_clock::time_point first_data_event);

    [[nodiscard]] ProxyMetricsSnapshot snapshot() const;

private:
    mutable std::mutex mutex_;
    std::uint64_t total_requests_{0};
    std::uint64_t completed_requests_{0};
    std::uint64_t failed_requests_{0};
    std::uint64_t first_token_events_{0};
    std::optional<double> last_ttft_ms_;
    SlidingWindow ttft_window_;
};

}  // namespace vllm_slo
