#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace vllm_slo {

struct SseEvent {
    std::string data;
    bool is_done{false};
};

// Incremental parser for Server-Sent Events.
//
// The parser accepts arbitrary network chunks. A feed call may contain a full
// event, part of an event, or multiple events. Blank lines terminate events.
// Only data fields are surfaced; comments and unsupported fields are ignored.
class SseParser {
public:
    explicit SseParser(std::size_t max_buffer_size);

    [[nodiscard]] std::vector<SseEvent> feed(std::string_view chunk);
    void reset();
    [[nodiscard]] std::size_t pending_bytes() const noexcept;

private:
    void append_data_line(std::string_view value);
    [[nodiscard]] SseEvent make_event() const;
    void clear_current_event();
    void enforce_limit(std::size_t extra_bytes) const;

    const std::size_t max_buffer_size_;
    std::string buffer_;
    std::string current_data_;
    bool has_data_{false};
};

}  // namespace vllm_slo
