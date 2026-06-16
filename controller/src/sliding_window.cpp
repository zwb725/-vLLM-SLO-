#include "vllm_slo/sliding_window.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace vllm_slo {

SlidingWindow::SlidingWindow(std::size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0U) {
        throw std::invalid_argument("SlidingWindow capacity must be greater than zero");
    }
}

void SlidingWindow::add_sample(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("SlidingWindow sample must be finite");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (samples_.size() == capacity_) {
        samples_.pop_front();
    }
    samples_.push_back(value);
}

std::size_t SlidingWindow::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return samples_.size();
}

std::size_t SlidingWindow::capacity() const {
    return capacity_;
}

std::optional<double> SlidingWindow::average() const {
    const auto values = snapshot_unlocked();
    if (values.empty()) {
        return std::nullopt;
    }

    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

std::optional<double> SlidingWindow::percentile(double percentile_value) const {
    if (!std::isfinite(percentile_value) || percentile_value < 0.0 || percentile_value > 100.0) {
        throw std::invalid_argument("percentile must be in [0, 100]");
    }

    auto values = snapshot_unlocked();
    if (values.empty()) {
        return std::nullopt;
    }

    std::sort(values.begin(), values.end());
    if (percentile_value == 0.0) {
        return values.front();
    }

    const double rank = std::ceil((percentile_value / 100.0) * static_cast<double>(values.size()));
    const auto index = static_cast<std::size_t>(std::max(1.0, rank)) - 1U;
    return values[std::min(index, values.size() - 1U)];
}

std::optional<double> SlidingWindow::p50() const {
    return percentile(50.0);
}

std::optional<double> SlidingWindow::p95() const {
    return percentile(95.0);
}

std::optional<double> SlidingWindow::p99() const {
    return percentile(99.0);
}

std::vector<double> SlidingWindow::snapshot_unlocked() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {samples_.begin(), samples_.end()};
}

}  // namespace vllm_slo
