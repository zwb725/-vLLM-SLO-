#include "vllm_slo/sliding_window.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(double actual, double expected, const char* message) {
    if (std::fabs(actual - expected) > 1e-9) {
        std::cerr << message << ": actual=" << actual << " expected=" << expected << "\n";
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    vllm_slo::SlidingWindow window(5);
    require(window.size() == 0U, "new window should be empty");
    require(!window.average().has_value(), "empty average should be nullopt");
    require(!window.p95().has_value(), "empty percentile should be nullopt");

    window.add_sample(10.0);
    window.add_sample(20.0);
    window.add_sample(30.0);
    window.add_sample(40.0);
    window.add_sample(50.0);

    require(window.size() == 5U, "window should contain five samples");
    require_close(*window.average(), 30.0, "average should match");
    require_close(*window.p50(), 30.0, "p50 should use nearest rank");
    require_close(*window.p95(), 50.0, "p95 should use nearest rank");
    require_close(*window.p99(), 50.0, "p99 should use nearest rank");

    window.add_sample(60.0);
    require(window.size() == 5U, "window should keep fixed capacity");
    require_close(*window.percentile(0.0), 20.0, "oldest sample should be evicted");
    require_close(*window.average(), 40.0, "average should reflect evicted sample");

    bool threw = false;
    try {
        (void)window.percentile(120.0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    require(threw, "invalid percentile should throw");

    return 0;
}
