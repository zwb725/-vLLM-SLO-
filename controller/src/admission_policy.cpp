#include "vllm_slo/admission_policy.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace vllm_slo {

namespace {

void validate_ratio(double value, const char* name) {
    if (value < 0.0 || value > 1.0) {
        throw std::invalid_argument(std::string(name) + " must be in [0, 1]");
    }
}

}  // namespace

AdmissionPolicy::AdmissionPolicy(AdmissionPolicyConfig config)
    : config_(config), recommended_max_inflight_(config.max_inflight) {
    if (config_.min_inflight <= 0) {
        throw std::invalid_argument("min_inflight must be positive");
    }
    if (config_.max_inflight < config_.min_inflight) {
        throw std::invalid_argument("max_inflight must be >= min_inflight");
    }
    if (config_.max_queue_size < 0) {
        throw std::invalid_argument("max_queue_size must be non-negative");
    }
    if (config_.target_p95_ttft_ms <= 0.0) {
        throw std::invalid_argument("target_p95_ttft_ms must be positive");
    }
    if (config_.critical_p95_ttft_ms < config_.target_p95_ttft_ms) {
        throw std::invalid_argument("critical_p95_ttft_ms must be >= target_p95_ttft_ms");
    }
    validate_ratio(config_.max_kv_cache_usage_ratio, "max_kv_cache_usage_ratio");
    validate_ratio(config_.max_gpu_memory_usage_ratio, "max_gpu_memory_usage_ratio");
    if (config_.max_waiting_requests < 0) {
        throw std::invalid_argument("max_waiting_requests must be non-negative");
    }
    if (config_.recovery_ratio <= 0.0 || config_.recovery_ratio >= 1.0) {
        throw std::invalid_argument("recovery_ratio must be in (0, 1)");
    }
    if (config_.cooldown_samples <= 0) {
        throw std::invalid_argument("cooldown_samples must be positive");
    }
}

AdmissionResult AdmissionPolicy::evaluate(
    const MetricsSnapshot& metrics,
    const AdmissionRequestState& state) {
    if (state.current_inflight < 0 || state.current_queue_size < 0) {
        throw std::invalid_argument("request state counters must be non-negative");
    }

    const bool soft_overload = is_soft_overload(metrics);
    const bool critical_overload = is_critical_overload(metrics);
    const bool recovery = is_recovery_sample(metrics);

    update_recommendation(soft_overload, critical_overload, recovery);

    AdmissionDecision decision = AdmissionDecision::Admit;
    if (critical_overload || state.current_queue_size >= config_.max_queue_size) {
        decision = AdmissionDecision::Reject;
    } else if (soft_overload || state.current_inflight >= recommended_max_inflight_) {
        decision = AdmissionDecision::Queue;
    }

    return AdmissionResult{
        decision,
        recommended_max_inflight_,
        soft_overload || critical_overload,
        critical_overload,
    };
}

int AdmissionPolicy::recommended_max_inflight() const noexcept {
    return recommended_max_inflight_;
}

const AdmissionPolicyConfig& AdmissionPolicy::config() const noexcept {
    return config_;
}

bool AdmissionPolicy::is_soft_overload(const MetricsSnapshot& metrics) const noexcept {
    return metrics.p95_ttft_ms >= config_.target_p95_ttft_ms ||
           metrics.kv_cache_usage_ratio >= config_.max_kv_cache_usage_ratio ||
           metrics.gpu_memory_usage_ratio >= config_.max_gpu_memory_usage_ratio ||
           metrics.waiting_requests >= config_.max_waiting_requests;
}

bool AdmissionPolicy::is_critical_overload(const MetricsSnapshot& metrics) const noexcept {
    return metrics.p95_ttft_ms >= config_.critical_p95_ttft_ms ||
           metrics.kv_cache_usage_ratio >= 0.98 ||
           metrics.gpu_memory_usage_ratio >= 0.98 ||
           metrics.error_rate >= 0.10;
}

bool AdmissionPolicy::is_recovery_sample(const MetricsSnapshot& metrics) const noexcept {
    return metrics.p95_ttft_ms < config_.target_p95_ttft_ms * config_.recovery_ratio &&
           metrics.kv_cache_usage_ratio < config_.max_kv_cache_usage_ratio * config_.recovery_ratio &&
           metrics.gpu_memory_usage_ratio < config_.max_gpu_memory_usage_ratio * config_.recovery_ratio &&
           metrics.waiting_requests < config_.max_waiting_requests * config_.recovery_ratio &&
           metrics.error_rate < 0.01;
}

void AdmissionPolicy::update_recommendation(
    bool soft_overload,
    bool critical_overload,
    bool recovery) {
    if (critical_overload) {
        recommended_max_inflight_ = std::max(config_.min_inflight, recommended_max_inflight_ / 2);
        cooldown_remaining_ = config_.cooldown_samples;
        consecutive_recovery_samples_ = 0;
        return;
    }

    if (soft_overload) {
        recommended_max_inflight_ = std::max(config_.min_inflight, recommended_max_inflight_ - 1);
        cooldown_remaining_ = config_.cooldown_samples;
        consecutive_recovery_samples_ = 0;
        return;
    }

    if (cooldown_remaining_ > 0) {
        --cooldown_remaining_;
        return;
    }

    if (!recovery) {
        consecutive_recovery_samples_ = 0;
        return;
    }

    ++consecutive_recovery_samples_;
    if (consecutive_recovery_samples_ >= config_.cooldown_samples) {
        recommended_max_inflight_ = std::min(config_.max_inflight, recommended_max_inflight_ + 1);
        consecutive_recovery_samples_ = 0;
    }
}

const char* to_string(AdmissionDecision decision) noexcept {
    switch (decision) {
        case AdmissionDecision::Admit:
            return "Admit";
        case AdmissionDecision::Queue:
            return "Queue";
        case AdmissionDecision::Reject:
            return "Reject";
    }
    return "Unknown";
}

}  // namespace vllm_slo
