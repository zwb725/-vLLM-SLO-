#include "vllm_slo/proxy_config.hpp"

#ifdef VLLM_SLO_HAS_BOOST

#include <boost/asio/io_context.hpp>
#include <boost/asio/signal_set.hpp>

#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "vllm_slo/proxy_server.hpp"

namespace {

std::size_t parse_size_env(const char* name, std::size_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return fallback;
    }
    try {
        return static_cast<std::size_t>(std::stoull(value));
    } catch (...) {
        return fallback;
    }
}

unsigned short parse_port_env(const char* name, unsigned short fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return fallback;
    }
    try {
        return static_cast<unsigned short>(std::stoul(value));
    } catch (...) {
        return fallback;
    }
}

std::string parse_string_env(const char* name, std::string fallback) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::move(fallback) : std::string(value);
}

}  // namespace

int main() {
    vllm_slo::ProxyConfig config;
    config.listen_host = parse_string_env("VLLM_SLO_LISTEN_HOST", config.listen_host);
    config.listen_port = parse_port_env("VLLM_SLO_LISTEN_PORT", config.listen_port);
    config.upstream_host = parse_string_env("VLLM_SLO_UPSTREAM_HOST", config.upstream_host);
    config.upstream_port = parse_port_env("VLLM_SLO_UPSTREAM_PORT", config.upstream_port);
    config.max_inflight = parse_size_env("VLLM_SLO_MAX_INFLIGHT", config.max_inflight);
    config.max_queue_size = parse_size_env("VLLM_SLO_MAX_QUEUE_SIZE", config.max_queue_size);
    config.worker_threads = parse_size_env("VLLM_SLO_WORKER_THREADS", config.worker_threads);

    boost::asio::io_context io;
    auto server = std::make_shared<vllm_slo::ProxyServer>(io, config);

    boost::asio::signal_set signals(io, SIGINT, SIGTERM);
    signals.async_wait([server, &io](boost::system::error_code, int) {
        server->stop();
        io.stop();
    });

    try {
        server->start();
        std::cout << "vllm_slo_proxy listening on " << config.listen_host << ":" << config.listen_port
                  << ", upstream " << config.upstream_host << ":" << config.upstream_port << "\n";

        const auto workers = std::max<std::size_t>(1U, config.worker_threads);
        std::vector<std::thread> threads;
        threads.reserve(workers > 0U ? workers - 1U : 0U);
        for (std::size_t i = 1; i < workers; ++i) {
            threads.emplace_back([&io]() { io.run(); });
        }
        io.run();
        for (auto& thread : threads) {
            thread.join();
        }
    } catch (const std::exception& ex) {
        std::cerr << "proxy failed: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}

#else

#include <iostream>

int main() {
    std::cerr << "vllm_slo_proxy was built without Boost.Asio/Beast support.\n";
    return 1;
}

#endif
