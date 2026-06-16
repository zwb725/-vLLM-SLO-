#include "vllm_slo/proxy_server.hpp"

#ifdef VLLM_SLO_HAS_BOOST

#include <boost/asio/bind_executor.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/post.hpp>
#include <boost/system/system_error.hpp>

#include <iostream>
#include <utility>

#include "vllm_slo/proxy_session.hpp"

namespace vllm_slo {

ProxyServer::ProxyServer(boost::asio::io_context& io, ProxyConfig config)
    : io_(io),
      config_(std::move(config)),
      acceptor_(io_),
      strand_(boost::asio::make_strand(io_)),
      limiter_(config_.max_inflight, config_.max_queue_size),
      metrics_(256) {
}

void ProxyServer::start() {
    auto self = shared_from_this();
    boost::system::error_code ec;
    const auto address = boost::asio::ip::make_address(config_.listen_host, ec);
    if (ec) {
        throw boost::system::system_error(ec);
    }

    const Tcp::endpoint endpoint(address, config_.listen_port);
    acceptor_.open(endpoint.protocol(), ec);
    if (ec) {
        throw boost::system::system_error(ec);
    }

    acceptor_.set_option(boost::asio::socket_base::reuse_address(true), ec);
    if (ec) {
        throw boost::system::system_error(ec);
    }

    acceptor_.bind(endpoint, ec);
    if (ec) {
        throw boost::system::system_error(ec);
    }

    acceptor_.listen(boost::asio::socket_base::max_listen_connections, ec);
    if (ec) {
        throw boost::system::system_error(ec);
    }

    limiter_.set_on_release([weak = std::weak_ptr<ProxyServer>(self)]() {
        if (auto server = weak.lock()) {
            boost::asio::post(server->strand_, [server]() { server->promote_queued_sessions(); });
        }
    });

    boost::asio::dispatch(strand_, [self]() { self->do_accept(); });
}

void ProxyServer::stop() {
    auto self = shared_from_this();
    boost::asio::post(strand_, [self]() {
        self->stopping_ = true;
        self->limiter_.close();
        boost::system::error_code ignored;
        self->acceptor_.close(ignored);
        while (!self->queued_sessions_.empty()) {
            if (auto session = self->queued_sessions_.front().lock()) {
                session->close_client();
            }
            self->queued_sessions_.pop_front();
        }
    });
}

void ProxyServer::enqueue_session(std::shared_ptr<ProxySession> session) {
    boost::asio::post(strand_, [self = shared_from_this(), session = std::move(session)]() {
        if (self->stopping_) {
            session->close_client();
            return;
        }
        self->queued_sessions_.push_back(session);
        session->start_queue_timer();
    });
}

void ProxyServer::promote_queued_sessions() {
    while (!queued_sessions_.empty()) {
        auto session = queued_sessions_.front().lock();
        queued_sessions_.pop_front();
        if (!session) {
            continue;
        }
        if (!session->is_queued()) {
            continue;
        }

        auto permit = limiter_.try_promote_queued();
        if (!permit.valid()) {
            if (limiter_.snapshot().queued_requests == 0U) {
                continue;
            }
            queued_sessions_.push_front(session);
            return;
        }

        if (!session->try_start_from_queue(std::move(permit))) {
            continue;
        }
    }
}

const ProxyConfig& ProxyServer::config() const noexcept {
    return config_;
}

InflightLimiter& ProxyServer::limiter() noexcept {
    return limiter_;
}

ProxyMetrics& ProxyServer::metrics() noexcept {
    return metrics_;
}

boost::asio::io_context& ProxyServer::io_context() noexcept {
    return io_;
}

void ProxyServer::do_accept() {
    acceptor_.async_accept(
        boost::asio::make_strand(io_),
        boost::asio::bind_executor(
            strand_,
            [self = shared_from_this()](boost::system::error_code ec, Tcp::socket socket) {
                self->on_accept(ec, std::move(socket));
            }));
}

void ProxyServer::on_accept(boost::system::error_code ec, Tcp::socket socket) {
    if (!ec) {
        std::make_shared<ProxySession>(std::move(socket), weak_from_this())->start();
    } else if (ec != boost::asio::error::operation_aborted) {
        std::cerr << "accept failed: " << ec.message() << "\n";
    }

    if (!stopping_) {
        do_accept();
    }
}

}  // namespace vllm_slo

#endif  // VLLM_SLO_HAS_BOOST
