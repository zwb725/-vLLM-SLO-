#pragma once

#include "vllm_slo/proxy_config.hpp"

#ifdef VLLM_SLO_HAS_BOOST

#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>

#include <deque>
#include <memory>

#include "vllm_slo/inflight_limiter.hpp"
#include "vllm_slo/proxy_metrics.hpp"

namespace vllm_slo {

class ProxySession;

class ProxyServer : public std::enable_shared_from_this<ProxyServer> {
public:
    ProxyServer(boost::asio::io_context& io, ProxyConfig config);

    void start();
    void stop();
    void enqueue_session(std::shared_ptr<ProxySession> session);
    void promote_queued_sessions();

    [[nodiscard]] const ProxyConfig& config() const noexcept;
    [[nodiscard]] InflightLimiter& limiter() noexcept;
    [[nodiscard]] ProxyMetrics& metrics() noexcept;
    [[nodiscard]] boost::asio::io_context& io_context() noexcept;

private:
    using Tcp = boost::asio::ip::tcp;

    void do_accept();
    void on_accept(boost::system::error_code ec, Tcp::socket socket);

    boost::asio::io_context& io_;
    ProxyConfig config_;
    Tcp::acceptor acceptor_;
    boost::asio::strand<boost::asio::io_context::executor_type> strand_;
    InflightLimiter limiter_;
    ProxyMetrics metrics_;
    std::deque<std::weak_ptr<ProxySession>> queued_sessions_;
    bool stopping_{false};
};

}  // namespace vllm_slo

#endif  // VLLM_SLO_HAS_BOOST
