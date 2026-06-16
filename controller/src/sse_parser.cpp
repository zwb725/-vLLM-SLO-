#include "vllm_slo/sse_parser.hpp"

#include <algorithm>
#include <stdexcept>

namespace vllm_slo {

namespace {

std::string_view strip_single_leading_space(std::string_view value) {
    if (!value.empty() && value.front() == ' ') {
        value.remove_prefix(1);
    }
    return value;
}

}  // namespace

SseParser::SseParser(std::size_t max_buffer_size) : max_buffer_size_(max_buffer_size) {
    if (max_buffer_size_ == 0U) {
        throw std::invalid_argument("SseParser max_buffer_size must be greater than zero");
    }
}

std::vector<SseEvent> SseParser::feed(std::string_view chunk) {
    enforce_limit(chunk.size());
    buffer_.append(chunk.data(), chunk.size());

    std::vector<SseEvent> events;
    std::size_t line_start = 0;

    while (line_start < buffer_.size()) {
        const auto newline = buffer_.find('\n', line_start);
        if (newline == std::string::npos) {
            break;
        }

        auto line = std::string_view(buffer_.data() + line_start, newline - line_start);
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }

        if (line.empty()) {
            if (has_data_) {
                events.push_back(make_event());
                clear_current_event();
            }
        } else if (line.front() == ':') {
            // SSE comment line. It can be used as keepalive and is ignored here.
        } else {
            constexpr std::string_view data_prefix = "data:";
            if (line.size() >= data_prefix.size() &&
                line.substr(0, data_prefix.size()) == data_prefix) {
                append_data_line(strip_single_leading_space(line.substr(data_prefix.size())));
            }
        }

        line_start = newline + 1U;
    }

    if (line_start > 0U) {
        buffer_.erase(0, line_start);
    }

    return events;
}

void SseParser::reset() {
    buffer_.clear();
    clear_current_event();
}

std::size_t SseParser::pending_bytes() const noexcept {
    return buffer_.size();
}

void SseParser::append_data_line(std::string_view value) {
    const auto separator_bytes = has_data_ ? 1U : 0U;
    if (value.size() > max_buffer_size_ ||
        current_data_.size() > max_buffer_size_ - value.size() ||
        current_data_.size() + value.size() > max_buffer_size_ - separator_bytes) {
        throw std::length_error("SseParser event data size limit exceeded");
    }

    if (has_data_) {
        current_data_.push_back('\n');
    }
    current_data_.append(value.data(), value.size());
    has_data_ = true;
}

SseEvent SseParser::make_event() const {
    return SseEvent{current_data_, current_data_ == "[DONE]"};
}

void SseParser::clear_current_event() {
    current_data_.clear();
    has_data_ = false;
}

void SseParser::enforce_limit(std::size_t extra_bytes) const {
    if (extra_bytes > max_buffer_size_ || buffer_.size() > max_buffer_size_ - extra_bytes) {
        throw std::length_error("SseParser buffer size limit exceeded");
    }
}

}  // namespace vllm_slo
