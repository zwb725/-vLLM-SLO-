#include "vllm_slo/proxy_metrics.hpp"

#include <stdexcept>

namespace vllm_slo {

ProxyMetrics::ProxyMetrics(std::size_t ttft_window_capacity)
    : ttft_window_(ttft_window_capacity) {}

void ProxyMetrics::record_request_started() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++total_requests_;
}

void ProxyMetrics::record_request_completed() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++completed_requests_;
}

void ProxyMetrics::record_request_failed() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++failed_requests_;
}

void ProxyMetrics::record_ttft(
    std::chrono::steady_clock::time_point request_start,
    std::chrono::steady_clock::time_point first_data_event) {
    if (first_data_event < request_start) {
        throw std::invalid_argument("first_data_event must be >= request_start");
    }

    const auto ttft_ms =
        std::chrono::duration<double, std::milli>(first_data_event - request_start).count();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++first_token_events_;
        last_ttft_ms_ = ttft_ms;
    }
    ttft_window_.add_sample(ttft_ms);
}

ProxyMetricsSnapshot ProxyMetrics::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ProxyMetricsSnapshot{
        total_requests_,
        completed_requests_,
        failed_requests_,
        first_token_events_,
        last_ttft_ms_,
        ttft_window_.p95(),
    };
}

}  // namespace vllm_slo
