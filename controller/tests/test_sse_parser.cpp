#include "vllm_slo/sse_parser.hpp"

#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main() {
    {
        vllm_slo::SseParser parser(1024);
        const auto events = parser.feed("data: hello\n\n");
        require(events.size() == 1U, "complete event should parse");
        require(events[0].data == "hello", "data should match");
        require(!events[0].is_done, "normal event should not be done");
    }

    {
        vllm_slo::SseParser parser(1024);
        require(parser.feed("data: he").empty(), "partial event should not emit");
        const auto events = parser.feed("llo\n\n");
        require(events.size() == 1U, "split event should emit after blank line");
        require(events[0].data == "hello", "split data should match");
        require(parser.pending_bytes() == 0U, "complete split event should leave no pending bytes");
    }

    {
        vllm_slo::SseParser parser(1024);
        const auto events = parser.feed("data: one\n\ndata: two\n\n");
        require(events.size() == 2U, "two events in one chunk should parse");
        require(events[0].data == "one", "first event should match");
        require(events[1].data == "two", "second event should match");
    }

    {
        vllm_slo::SseParser parser(1024);
        const auto events = parser.feed(": keepalive\r\ndata: crlf\r\n\r\n");
        require(events.size() == 1U, "CRLF event should parse");
        require(events[0].data == "crlf", "comments should be ignored");
    }

    {
        vllm_slo::SseParser parser(1024);
        const auto events = parser.feed("data: first\ndata: second\n\n");
        require(events.size() == 1U, "multi-line data should parse");
        require(events[0].data == "first\nsecond", "multi-line data should join with newline");
    }

    {
        vllm_slo::SseParser parser(1024);
        const auto events = parser.feed("data: [DONE]\n\n");
        require(events.size() == 1U, "done event should parse");
        require(events[0].is_done, "done event should be marked");
    }

    {
        vllm_slo::SseParser parser(1024);
        require(parser.feed("data: pending").empty(), "incomplete tail should not emit");
        require(parser.pending_bytes() == std::string("data: pending").size(), "tail should be retained");
        parser.reset();
        require(parser.pending_bytes() == 0U, "reset should clear tail");
    }

    {
        vllm_slo::SseParser parser(8);
        bool threw = false;
        try {
            (void)parser.feed("data: too long\n");
        } catch (const std::length_error&) {
            threw = true;
        }
        require(threw, "oversized buffer should throw");
    }

    return 0;
}
