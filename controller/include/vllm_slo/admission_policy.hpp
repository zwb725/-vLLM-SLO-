#pragma once

#include <cstddef>

#include "vllm_slo/metrics_snapshot.hpp"

namespace vllm_slo {

enum class AdmissionDecision {
    Admit,
    Queue,
    Reject
};

struct AdmissionPolicyConfig {
    int min_inflight{1};
    int max_inflight{8};
    int max_queue_size{16};
    double target_p95_ttft_ms{800.0};
    double critical_p95_ttft_ms{1600.0};
    double max_kv_cache_usage_ratio{0.85};
    double max_gpu_memory_usage_ratio{0.90};
    int max_waiting_requests{16};
    double recovery_ratio{0.75};
    int cooldown_samples{3};
};

struct AdmissionRequestState {
    int current_inflight{0};
    int current_queue_size{0};
};

struct AdmissionResult {
    AdmissionDecision decision{AdmissionDecision::Admit};
    int recommended_max_inflight{1};
    bool overloaded{false};
    bool critical{false};
};

// Threshold policy for the controller-side maximum in-flight request count.
//
// This class does not change vLLM's internal max_num_seqs at runtime. It only
// recommends the C++ admission controller's own in-flight cap.
class AdmissionPolicy {
public:
    explicit AdmissionPolicy(AdmissionPolicyConfig config);

    [[nodiscard]] AdmissionResult evaluate(
        const MetricsSnapshot& metrics,
        const AdmissionRequestState& state);

    [[nodiscard]] int recommended_max_inflight() const noexcept;
    [[nodiscard]] const AdmissionPolicyConfig& config() const noexcept;

private:
    [[nodiscard]] bool is_soft_overload(const MetricsSnapshot& metrics) const noexcept;
    [[nodiscard]] bool is_critical_overload(const MetricsSnapshot& metrics) const noexcept;
    [[nodiscard]] bool is_recovery_sample(const MetricsSnapshot& metrics) const noexcept;
    void update_recommendation(bool soft_overload, bool critical_overload, bool recovery);

    AdmissionPolicyConfig config_;
    int recommended_max_inflight_;
    int cooldown_remaining_{0};
    int consecutive_recovery_samples_{0};
};

[[nodiscard]] const char* to_string(AdmissionDecision decision) noexcept;

}  // namespace vllm_slo
