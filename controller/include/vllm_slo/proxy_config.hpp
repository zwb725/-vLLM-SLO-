#pragma once

#include <chrono>
#include <cstddef>
#include <string>

namespace vllm_slo {

struct ProxyConfig {
    std::string listen_host{"127.0.0.1"};
    unsigned short listen_port{8080};
    std::string upstream_host{"127.0.0.1"};
    unsigned short upstream_port{8000};
    std::chrono::milliseconds connect_timeout{2000};
    std::chrono::milliseconds read_timeout{10000};
    std::chrono::milliseconds request_timeout{30000};
    std::chrono::milliseconds queue_timeout{1000};
    std::size_t max_inflight{2};
    std::size_t max_queue_size{8};
    std::size_t worker_threads{1};
    std::size_t max_sse_buffer_bytes{1024 * 1024};
};

// Phase 1 uses environment variables for the executable. This placeholder keeps
// the JSON config schema documented without adding a JSON dependency yet.

}  // namespace vllm_slo
