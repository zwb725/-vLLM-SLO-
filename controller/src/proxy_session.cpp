#include "vllm_slo/proxy_session.hpp"

#ifdef VLLM_SLO_HAS_BOOST

#include <boost/asio/connect.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/beast/core/buffers_to_string.hpp>
#include <boost/beast/http/write.hpp>

#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

#include "vllm_slo/proxy_server.hpp"

namespace vllm_slo {

namespace beast = boost::beast;
namespace http = boost::beast::http;
using boost::asio::ip::tcp;

namespace {

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char ch : value) {
        switch (ch) {
            case '"':
                escaped += "\\\"";
                break;
            case '\\':
                escaped += "\\\\";
                break;
            case '\n':
                escaped += "\\n";
                break;
            case '\r':
                escaped += "\\r";
                break;
            case '\t':
                escaped += "\\t";
                break;
            default:
                escaped.push_back(ch);
                break;
        }
    }
    return escaped;
}

bool is_disconnect(boost::system::error_code ec) {
    return ec == boost::asio::error::operation_aborted ||
           ec == boost::asio::error::eof ||
           ec == boost::asio::error::connection_reset ||
           ec == boost::asio::error::broken_pipe ||
           ec == boost::beast::http::error::end_of_stream;
}

}  // namespace

ProxySession::ProxySession(Tcp::socket socket, std::weak_ptr<ProxyServer> server)
    : client_socket_(std::move(socket)),
      server_(std::move(server)),
      resolver_(client_socket_.get_executor()),
      upstream_stream_(client_socket_.get_executor()),
      request_timer_(client_socket_.get_executor()),
      queue_timer_(client_socket_.get_executor()),
      sse_parser_([](const std::weak_ptr<ProxyServer>& weak_server) {
          auto locked = weak_server.lock();
          return locked ? locked->config().max_sse_buffer_bytes : 1024 * 1024;
      }(server_)) {}

ProxySession::~ProxySession() {
    if (queued_.load()) {
        if (auto server = server_.lock()) {
            (void)server->limiter().cancel_queued();
        }
    }
}

void ProxySession::start() {
    do_read_request();
}

void ProxySession::start_queue_timer() {
    queued_.store(true);
    auto server = server_.lock();
    if (!server) {
        close_client();
        return;
    }

    queue_timer_.expires_after(server->config().queue_timeout);
    queue_timer_.async_wait([self = shared_from_this()](boost::system::error_code ec) {
        if (ec || !self->queued_.load()) {
            return;
        }
        if (auto server = self->server_.lock()) {
            (void)server->limiter().cancel_queued();
        }
        self->queued_.store(false);
        self->send_json_error(http::status::too_many_requests, "queue timeout");
    });
}

bool ProxySession::try_start_from_queue(InflightLimiter::Permit permit) {
    if (!queued_.load() || finished_.load()) {
        return false;
    }
    queued_.store(false);
    queue_timer_.cancel();
    start_proxy_request(std::move(permit));
    return true;
}

bool ProxySession::is_queued() const noexcept {
    return queued_.load();
}

void ProxySession::close_client() {
    boost::system::error_code ignored;
    client_socket_.shutdown(tcp::socket::shutdown_both, ignored);
    client_socket_.close(ignored);
}

void ProxySession::do_read_request() {
    http::async_read(
        client_socket_,
        client_buffer_,
        client_request_,
        [self = shared_from_this()](boost::system::error_code ec, std::size_t bytes_transferred) {
            self->on_read_request(ec, bytes_transferred);
        });
}

void ProxySession::on_read_request(boost::system::error_code ec, std::size_t /*bytes_transferred*/) {
    if (ec) {
        if (!is_disconnect(ec)) {
            std::cerr << "client read failed: " << ec.message() << "\n";
        }
        finished_.store(true);
        return;
    }

    handle_request();
}

void ProxySession::handle_request() {
    if (client_request_.target() == "/health") {
        if (client_request_.method() != http::verb::get) {
            send_json_error(http::status::method_not_allowed, "GET required for /health");
            return;
        }
        send_health();
        return;
    }

    if (client_request_.target() != "/v1/chat/completions") {
        send_json_error(http::status::not_found, "unsupported path");
        return;
    }

    if (client_request_.method() != http::verb::post) {
        send_json_error(http::status::method_not_allowed, "POST required for /v1/chat/completions");
        return;
    }

    auto server = server_.lock();
    if (!server) {
        send_json_error(http::status::service_unavailable, "proxy server is stopping");
        return;
    }

    server->metrics().record_request_started();
    request_counted_ = true;
    auto acquire = server->limiter().try_acquire_or_queue();
    if (acquire.decision == InflightDecision::Admit) {
        start_proxy_request(std::move(acquire.permit));
        return;
    }
    if (acquire.decision == InflightDecision::Queue) {
        server->enqueue_session(shared_from_this());
        return;
    }
    if (acquire.decision == InflightDecision::Closed) {
        send_json_error(http::status::service_unavailable, "proxy is shutting down");
        return;
    }
    send_json_error(http::status::too_many_requests, "admission queue is full");
}

void ProxySession::send_health() {
    auto server = server_.lock();
    if (!server) {
        send_json_error(http::status::service_unavailable, "proxy server is stopping");
        return;
    }

    const auto limit = server->limiter().snapshot();
    const auto metrics = server->metrics().snapshot();
    std::ostringstream body;
    body << "{"
         << "\"status\":\"ok\","
         << "\"proxy_healthy\":true,"
         << "\"upstream_healthy\":null,"
         << "\"current_inflight\":" << limit.current_inflight << ","
         << "\"queued_requests\":" << limit.queued_requests << ","
         << "\"max_inflight\":" << limit.max_inflight << ","
         << "\"max_queue_size\":" << limit.max_queue_size << ","
         << "\"total_requests\":" << metrics.total_requests << ","
         << "\"completed_requests\":" << metrics.completed_requests << ","
         << "\"failed_requests\":" << metrics.failed_requests << ","
         << "\"upstream\":\"" << json_escape(server->config().upstream_host) << ":"
         << server->config().upstream_port << "\""
         << "}";

    send_text_response(http::status::ok, "application/json", body.str());
}

void ProxySession::send_json_error(http::status status, const std::string& error) {
    fail_with_status(status, error);
}

void ProxySession::send_text_response(http::status status, const std::string& content_type, std::string body) {
    auto response = std::make_shared<http::response<http::string_body>>(status, client_request_.version());
    response->set(http::field::server, "vllm-slo-proxy");
    response->set(http::field::content_type, content_type);
    response->keep_alive(false);
    response->body() = std::move(body);
    response->prepare_payload();

    http::async_write(
        client_socket_,
        *response,
        [self = shared_from_this(), response](boost::system::error_code ec, std::size_t) {
            if (ec && !is_disconnect(ec)) {
                std::cerr << "client write failed: " << ec.message() << "\n";
            }
            self->finished_.store(true);
            self->close_client();
        });
}

void ProxySession::start_proxy_request(InflightLimiter::Permit permit) {
    permit_ = std::move(permit);
    request_start_ = std::chrono::steady_clock::now();
    start_request_timer();
    resolve_upstream();
}

void ProxySession::start_request_timer() {
    auto server = server_.lock();
    if (!server) {
        return;
    }

    request_timer_.expires_after(server->config().request_timeout);
    request_timer_.async_wait([self = shared_from_this()](boost::system::error_code ec) {
        if (ec || self->finished_.load()) {
            return;
        }
        self->fail_with_status(http::status::gateway_timeout, "request timeout");
    });
}

void ProxySession::resolve_upstream() {
    auto server = server_.lock();
    if (!server) {
        fail_with_status(http::status::service_unavailable, "proxy server is stopping");
        return;
    }

    resolver_.async_resolve(
        server->config().upstream_host,
        std::to_string(server->config().upstream_port),
        [self = shared_from_this()](boost::system::error_code ec, tcp::resolver::results_type results) {
            if (ec) {
                self->fail_with_status(http::status::service_unavailable, "upstream resolve failed");
                return;
            }
            self->connect_upstream(std::move(results));
        });
}

void ProxySession::connect_upstream(tcp::resolver::results_type results) {
    auto server = server_.lock();
    if (!server) {
        fail_with_status(http::status::service_unavailable, "proxy server is stopping");
        return;
    }

    upstream_stream_.expires_after(server->config().connect_timeout);
    upstream_stream_.async_connect(
        results,
        [self = shared_from_this()](boost::system::error_code ec, const tcp::endpoint&) {
            if (ec) {
                self->fail_with_status(http::status::service_unavailable, "upstream connect failed");
                return;
            }
            self->write_upstream_request();
        });
}

void ProxySession::write_upstream_request() {
    auto server = server_.lock();
    if (!server) {
        fail_with_status(http::status::service_unavailable, "proxy server is stopping");
        return;
    }

    upstream_request_.version(11);
    upstream_request_.method(http::verb::post);
    upstream_request_.target("/v1/chat/completions");
    upstream_request_.set(http::field::host, server->config().upstream_host);
    upstream_request_.set(http::field::user_agent, "vllm-slo-proxy");
    upstream_request_.set(http::field::content_type, "application/json");
    upstream_request_.set(http::field::accept, "text/event-stream");
    upstream_request_.body() = client_request_.body();
    upstream_request_.prepare_payload();

    upstream_stream_.expires_after(server->config().read_timeout);
    http::async_write(
        upstream_stream_,
        upstream_request_,
        [self = shared_from_this()](boost::system::error_code ec, std::size_t) {
            if (ec) {
                self->fail_with_status(http::status::service_unavailable, "upstream write failed");
                return;
            }
            self->read_upstream_header();
        });
}

void ProxySession::read_upstream_header() {
    upstream_parser_.body_limit((std::numeric_limits<std::uint64_t>::max)());
    upstream_parser_.get().body().data = upstream_body_buffer_.data();
    upstream_parser_.get().body().size = upstream_body_buffer_.size();

    http::async_read_header(
        upstream_stream_,
        upstream_buffer_,
        upstream_parser_,
        [self = shared_from_this()](boost::system::error_code ec, std::size_t) {
            if (ec) {
                self->fail_with_status(http::status::service_unavailable, "upstream header read failed");
                return;
            }
            const auto status = self->upstream_parser_.get().result();
            if (status != http::status::ok) {
                self->fail_with_status(http::status::service_unavailable, "upstream returned non-200");
                return;
            }
            self->write_downstream_sse_header();
        });
}

void ProxySession::write_downstream_sse_header() {
    response_started_.store(true);
    auto header = std::make_shared<std::string>(
        "HTTP/1.1 200 OK\r\n"
        "Server: vllm-slo-proxy\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "\r\n");

    boost::asio::async_write(
        client_socket_,
        boost::asio::buffer(*header),
        [self = shared_from_this(), header](boost::system::error_code ec, std::size_t) {
            if (ec) {
                self->finish_request();
                return;
            }
            self->read_upstream_body();
        });
}

void ProxySession::read_upstream_body() {
    auto server = server_.lock();
    if (!server) {
        fail_with_status(http::status::service_unavailable, "proxy server is stopping");
        return;
    }

    upstream_parser_.get().body().data = upstream_body_buffer_.data();
    upstream_parser_.get().body().size = upstream_body_buffer_.size();
    upstream_stream_.expires_after(server->config().read_timeout);

    http::async_read_some(
        upstream_stream_,
        upstream_buffer_,
        upstream_parser_,
        [self = shared_from_this()](boost::system::error_code ec, std::size_t bytes_transferred) {
            if (ec == http::error::need_buffer) {
                self->read_upstream_body();
                return;
            }
            if (ec == http::error::end_of_stream || self->upstream_parser_.is_done()) {
                self->write_downstream_body("", true);
                return;
            }
            if (ec) {
                self->fail_with_status(http::status::service_unavailable, "upstream body read failed");
                return;
            }

            const auto remaining = self->upstream_parser_.get().body().size;
            const auto consumed = self->upstream_body_buffer_.size() - remaining;
            std::string chunk(self->upstream_body_buffer_.data(), consumed);
            const bool done = self->observe_sse_chunk(chunk);
            self->write_downstream_body(std::move(chunk), done);
        });
}

void ProxySession::write_downstream_body(std::string data, bool upstream_done) {
    if (data.empty() && upstream_done) {
        finish_request();
        return;
    }

    auto payload = std::make_shared<std::string>(std::move(data));
    boost::asio::async_write(
        client_socket_,
        boost::asio::buffer(*payload),
        [self = shared_from_this(), payload, upstream_done](boost::system::error_code ec, std::size_t) {
            if (ec) {
                self->fail_with_status(http::status::service_unavailable, "downstream write failed");
                return;
            }
            if (upstream_done) {
                self->finish_request();
                return;
            }
            self->read_upstream_body();
        });
}

void ProxySession::finish_request() {
    if (finished_.load()) {
        return;
    }
    finished_.store(true);
    cancel_timers();
    close_upstream();
    if (auto server = server_.lock()) {
        if (request_counted_) {
            server->metrics().record_request_completed();
        }
    }
    permit_.reset();
    close_client();
}

void ProxySession::fail_with_status(http::status status, const std::string& error) {
    if (finished_.load()) {
        return;
    }
    finished_.store(true);
    cancel_timers();
    close_upstream();
    if (auto server = server_.lock()) {
        if (request_counted_) {
            server->metrics().record_request_failed();
        }
    }
    permit_.reset();

    if (response_started_.load()) {
        close_client();
        return;
    }

    const std::string body = "{\"error\":\"" + json_escape(error) + "\"}";
    auto response = std::make_shared<http::response<http::string_body>>(status, client_request_.version());
    response->set(http::field::server, "vllm-slo-proxy");
    response->set(http::field::content_type, "application/json");
    response->keep_alive(false);
    response->body() = body;
    response->prepare_payload();

    http::async_write(
        client_socket_,
        *response,
        [self = shared_from_this(), response](boost::system::error_code ec, std::size_t) {
            if (ec && !is_disconnect(ec)) {
                std::cerr << "error response write failed: " << ec.message() << "\n";
            }
            self->close_client();
        });
}

void ProxySession::cancel_timers() {
    boost::system::error_code ignored;
    request_timer_.cancel(ignored);
    queue_timer_.cancel(ignored);
}

void ProxySession::close_upstream() {
    boost::system::error_code ignored;
    upstream_stream_.socket().shutdown(tcp::socket::shutdown_both, ignored);
    upstream_stream_.socket().close(ignored);
}

bool ProxySession::observe_sse_chunk(const std::string& data) {
    bool done = false;
    try {
        for (const auto& event : sse_parser_.feed(data)) {
            if (!first_sse_data_seen_ && !event.data.empty()) {
                first_sse_data_seen_ = true;
                if (auto server = server_.lock()) {
                    server->metrics().record_ttft(*request_start_, std::chrono::steady_clock::now());
                }
            }
            done = done || event.is_done;
        }
    } catch (const std::exception& ex) {
        std::cerr << "SSE parse warning: " << ex.what() << "\n";
        done = true;
    }
    return done;
}

}  // namespace vllm_slo

#endif  // VLLM_SLO_HAS_BOOST
