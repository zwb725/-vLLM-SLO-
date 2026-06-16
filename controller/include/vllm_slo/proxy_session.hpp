#pragma once

#ifdef VLLM_SLO_HAS_BOOST

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <boost/beast/http.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include "vllm_slo/inflight_limiter.hpp"
#include "vllm_slo/proxy_config.hpp"
#include "vllm_slo/sse_parser.hpp"

namespace vllm_slo {

class ProxyServer;

class ProxySession : public std::enable_shared_from_this<ProxySession> {
public:
    using Tcp = boost::asio::ip::tcp;

    ProxySession(Tcp::socket socket, std::weak_ptr<ProxyServer> server);
    ~ProxySession();

    void start();
    void start_queue_timer();
    bool try_start_from_queue(InflightLimiter::Permit permit);
    [[nodiscard]] bool is_queued() const noexcept;
    void close_client();

private:
    using HttpStringRequest = boost::beast::http::request<boost::beast::http::string_body>;
    using HttpBufferResponseParser =
        boost::beast::http::response_parser<boost::beast::http::buffer_body>;

    void do_read_request();
    void on_read_request(boost::system::error_code ec, std::size_t bytes_transferred);
    void handle_request();
    void send_health();
    void send_json_error(boost::beast::http::status status, const std::string& error);
    void send_text_response(
        boost::beast::http::status status,
        const std::string& content_type,
        std::string body);

    void start_proxy_request(InflightLimiter::Permit permit);
    void start_request_timer();
    void resolve_upstream();
    void connect_upstream(Tcp::resolver::results_type results);
    void write_upstream_request();
    void read_upstream_header();
    void write_downstream_sse_header();
    void read_upstream_body();
    void write_downstream_body(std::string data, bool upstream_done);
    void finish_request();
    void fail_with_status(boost::beast::http::status status, const std::string& error);
    void cancel_timers();
    void close_upstream();
    [[nodiscard]] bool observe_sse_chunk(const std::string& data);

    Tcp::socket client_socket_;
    std::weak_ptr<ProxyServer> server_;
    boost::beast::flat_buffer client_buffer_;
    boost::beast::flat_buffer upstream_buffer_;
    HttpStringRequest client_request_;
    HttpStringRequest upstream_request_;
    HttpBufferResponseParser upstream_parser_;
    Tcp::resolver resolver_;
    boost::beast::tcp_stream upstream_stream_;
    boost::asio::steady_timer request_timer_;
    boost::asio::steady_timer queue_timer_;
    std::array<char, 8192> upstream_body_buffer_{};
    std::optional<InflightLimiter::Permit> permit_;
    std::optional<std::chrono::steady_clock::time_point> request_start_;
    SseParser sse_parser_;
    bool first_sse_data_seen_{false};
    std::atomic_bool queued_{false};
    std::atomic_bool finished_{false};
    std::atomic_bool response_started_{false};
    bool request_counted_{false};
};

}  // namespace vllm_slo

#endif  // VLLM_SLO_HAS_BOOST
