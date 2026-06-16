#pragma once

#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <vector>

namespace vllm_slo {

// Thread-safe fixed-capacity sliding window for double samples.
//
// Percentiles use the nearest-rank method:
//   rank = ceil(percentile / 100 * sample_count)
//   index = clamp(rank - 1, 0, sample_count - 1)
// Samples are copied and sorted for each percentile query.
class SlidingWindow {
public:
    explicit SlidingWindow(std::size_t capacity);

    void add_sample(double value);
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t capacity() const;
    [[nodiscard]] std::optional<double> average() const;
    [[nodiscard]] std::optional<double> percentile(double percentile_value) const;
    [[nodiscard]] std::optional<double> p50() const;
    [[nodiscard]] std::optional<double> p95() const;
    [[nodiscard]] std::optional<double> p99() const;

private:
    [[nodiscard]] std::vector<double> snapshot_unlocked() const;

    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::deque<double> samples_;
};

}  // namespace vllm_slo
