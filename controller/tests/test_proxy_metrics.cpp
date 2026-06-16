#include "vllm_slo/proxy_metrics.hpp"

#include <chrono>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    vllm_slo::ProxyMetrics metrics(4);
    const auto start = std::chrono::steady_clock::time_point(std::chrono::milliseconds(1000));

    metrics.record_request_started();
    metrics.record_ttft(start, start + std::chrono::milliseconds(120));
    metrics.record_request_completed();

    auto snapshot = metrics.snapshot();
    require(snapshot.total_requests == 1U, "started request should be counted");
    require(snapshot.completed_requests == 1U, "completed request should be counted");
    require(snapshot.failed_requests == 0U, "failed request should not be counted");
    require(snapshot.first_token_events == 1U, "first token event should be counted");
    require(snapshot.last_ttft_ms.has_value(), "last TTFT should be present");
    require(*snapshot.last_ttft_ms == 120.0, "TTFT should be deterministic");
    require(snapshot.p95_ttft_ms.has_value(), "P95 TTFT should be present");
    require(*snapshot.p95_ttft_ms == 120.0, "single-sample P95 should match TTFT");

    metrics.record_request_started();
    metrics.record_request_failed();
    snapshot = metrics.snapshot();
    require(snapshot.total_requests == 2U, "second request should be counted");
    require(snapshot.failed_requests == 1U, "failed request should be counted");

    bool threw = false;
    try {
        metrics.record_ttft(start, start - std::chrono::milliseconds(1));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "negative TTFT should throw");

    return 0;
}
