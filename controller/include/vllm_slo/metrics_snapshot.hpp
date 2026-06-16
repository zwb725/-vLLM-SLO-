#pragma once

#include <chrono>

namespace vllm_slo {

// One sampled view of service state. Ratio fields use the inclusive range [0.0, 1.0].
struct MetricsSnapshot {
    std::chrono::steady_clock::time_point timestamp{std::chrono::steady_clock::now()};
    double p95_ttft_ms{0.0};
    int running_requests{0};
    int waiting_requests{0};
    double kv_cache_usage_ratio{0.0};
    double gpu_memory_usage_ratio{0.0};
    double gpu_utilization_ratio{0.0};
    double error_rate{0.0};
};

}  // namespace vllm_slo
