#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include "vllm_slo/admission_policy.hpp"

namespace {

void print_metrics(const vllm_slo::MetricsSnapshot& metrics) {
    std::cout << "p95_ttft_ms=" << metrics.p95_ttft_ms
              << " running=" << metrics.running_requests
              << " waiting=" << metrics.waiting_requests
              << " kv_cache=" << std::fixed << std::setprecision(2) << metrics.kv_cache_usage_ratio
              << " gpu_memory=" << metrics.gpu_memory_usage_ratio
              << " gpu_util=" << metrics.gpu_utilization_ratio
              << " error_rate=" << metrics.error_rate;
}

}  // namespace

int main() {
    std::cout << "controller_demo: simulation only; no vLLM, HTTP/SSE, CUDA, GPU, or NVML connection.\n";

    vllm_slo::AdmissionPolicyConfig config;
    config.min_inflight = 2;
    config.max_inflight = 8;
    config.max_queue_size = 4;
    config.target_p95_ttft_ms = 800.0;
    config.critical_p95_ttft_ms = 1600.0;
    config.max_kv_cache_usage_ratio = 0.85;
    config.max_gpu_memory_usage_ratio = 0.90;
    config.max_waiting_requests = 8;
    config.recovery_ratio = 0.75;
    config.cooldown_samples = 2;

    vllm_slo::AdmissionPolicy policy(config);

    const std::vector<vllm_slo::MetricsSnapshot> samples = {
        {std::chrono::steady_clock::now(), 420.0, 2, 0, 0.30, 0.40, 0.45, 0.00},
        {std::chrono::steady_clock::now(), 760.0, 6, 1, 0.70, 0.75, 0.80, 0.00},
        {std::chrono::steady_clock::now(), 980.0, 8, 3, 0.88, 0.82, 0.92, 0.00},
        {std::chrono::steady_clock::now(), 1720.0, 8, 4, 0.92, 0.95, 0.98, 0.02},
        {std::chrono::steady_clock::now(), 650.0, 3, 1, 0.55, 0.60, 0.58, 0.00},
        {std::chrono::steady_clock::now(), 430.0, 2, 0, 0.35, 0.45, 0.42, 0.00},
        {std::chrono::steady_clock::now(), 390.0, 2, 0, 0.32, 0.38, 0.40, 0.00},
    };

    int inflight = 2;
    int queued = 0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        vllm_slo::AdmissionRequestState state{inflight, queued};
        const auto result = policy.evaluate(samples[i], state);

        std::cout << "sample=" << i << " ";
        print_metrics(samples[i]);
        std::cout << " decision=" << vllm_slo::to_string(result.decision)
                  << " recommended_max_inflight=" << result.recommended_max_inflight
                  << " overloaded=" << std::boolalpha << result.overloaded
                  << " critical=" << result.critical << "\n";

        if (result.decision == vllm_slo::AdmissionDecision::Admit) {
            inflight = std::min(inflight + 1, result.recommended_max_inflight);
        } else if (result.decision == vllm_slo::AdmissionDecision::Queue) {
            queued = std::min(queued + 1, config.max_queue_size);
        }
    }

    return 0;
}
