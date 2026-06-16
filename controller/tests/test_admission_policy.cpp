#include "vllm_slo/admission_policy.hpp"

#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

vllm_slo::AdmissionPolicyConfig test_config() {
    vllm_slo::AdmissionPolicyConfig config;
    config.min_inflight = 2;
    config.max_inflight = 8;
    config.max_queue_size = 2;
    config.target_p95_ttft_ms = 100.0;
    config.critical_p95_ttft_ms = 200.0;
    config.max_kv_cache_usage_ratio = 0.80;
    config.max_gpu_memory_usage_ratio = 0.85;
    config.max_waiting_requests = 4;
    config.recovery_ratio = 0.70;
    config.cooldown_samples = 2;
    return config;
}

vllm_slo::MetricsSnapshot healthy_metrics() {
    vllm_slo::MetricsSnapshot metrics;
    metrics.p95_ttft_ms = 50.0;
    metrics.running_requests = 1;
    metrics.waiting_requests = 0;
    metrics.kv_cache_usage_ratio = 0.30;
    metrics.gpu_memory_usage_ratio = 0.40;
    metrics.gpu_utilization_ratio = 0.50;
    metrics.error_rate = 0.0;
    return metrics;
}

}  // namespace

int main() {
    vllm_slo::AdmissionPolicy policy(test_config());
    const vllm_slo::AdmissionRequestState empty_state{0, 0};

    auto result = policy.evaluate(healthy_metrics(), empty_state);
    require(result.decision == vllm_slo::AdmissionDecision::Admit, "healthy state should admit");
    require(result.recommended_max_inflight == 8, "initial recommendation should be max");

    auto soft = healthy_metrics();
    soft.p95_ttft_ms = 120.0;
    result = policy.evaluate(soft, vllm_slo::AdmissionRequestState{8, 0});
    require(result.decision == vllm_slo::AdmissionDecision::Queue, "soft overload should queue first");
    require(result.recommended_max_inflight == 7, "soft overload should reduce by one");

    auto critical = healthy_metrics();
    critical.p95_ttft_ms = 240.0;
    result = policy.evaluate(critical, vllm_slo::AdmissionRequestState{8, 0});
    require(result.decision == vllm_slo::AdmissionDecision::Reject, "critical overload should reject");
    require(result.recommended_max_inflight == 3, "critical overload should halve recommendation");
    require(result.critical, "critical flag should be set");

    result = policy.evaluate(healthy_metrics(), vllm_slo::AdmissionRequestState{1, 2});
    require(result.decision == vllm_slo::AdmissionDecision::Reject, "full queue should reject");

    result = policy.evaluate(healthy_metrics(), empty_state);
    require(result.decision == vllm_slo::AdmissionDecision::Admit, "healthy cooldown sample can admit");
    require(result.recommended_max_inflight == 3, "cooldown should delay recovery");

    (void)policy.evaluate(healthy_metrics(), empty_state);
    (void)policy.evaluate(healthy_metrics(), empty_state);
    result = policy.evaluate(healthy_metrics(), empty_state);
    require(result.recommended_max_inflight >= 4, "consecutive recovery should increase recommendation");

    bool threw = false;
    try {
        auto bad_config = test_config();
        bad_config.min_inflight = 0;
        vllm_slo::AdmissionPolicy bad_policy(bad_config);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "invalid config should throw");

    return 0;
}
