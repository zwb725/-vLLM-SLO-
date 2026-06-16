# 阶段 1 面试文档：C++ 异步 HTTP/SSE 代理

## 本阶段解决了什么工程问题

本阶段把项目从纯策略核心推进到本地可验证的网络链路：客户端请求进入 C++ 代理，代理做固定并发和有界队列控制，再转发到本地 Mock OpenAI-compatible SSE 服务。它解决的是网络代理、SSE 流式转发、TTFT 基础观测和过载保护的工程骨架问题。

## 为什么采用当前设计

- `SseParser` 独立测试，避免假设一次 socket read 就是完整 SSE 事件。
- `InflightLimiter` 用 RAII permit 释放在途槽位，降低异常和断开路径的计数泄漏风险。
- `ProxyMetrics` 只统计基础 TTFT，不伪装真实 vLLM Metrics。
- `ProxySession` 负责单连接生命周期，`ProxyServer` 负责监听和队列唤醒。
- 本阶段不直接接动态 `AdmissionPolicy`，因为还没有真实 vLLM Metrics、NVML 或验证机实验数据。

## 核心调用链

```text
ProxyServer::start
  -> async_accept
  -> ProxySession::start
  -> async_read client request
  -> InflightLimiter::try_acquire_or_queue
  -> resolve/connect upstream
  -> async_write upstream request
  -> async_read upstream SSE chunks
  -> SseParser::feed
  -> ProxyMetrics::record_ttft
  -> async_write downstream chunk
  -> Permit RAII release
```

## 关键类和源码位置

- `SseParser`：`controller/include/vllm_slo/sse_parser.hpp`，`controller/src/sse_parser.cpp`
- `InflightLimiter`：`controller/include/vllm_slo/inflight_limiter.hpp`，`controller/src/inflight_limiter.cpp`
- `ProxyMetrics`：`controller/include/vllm_slo/proxy_metrics.hpp`，`controller/src/proxy_metrics.cpp`
- `ProxyServer`：`controller/include/vllm_slo/proxy_server.hpp`，`controller/src/proxy_server.cpp`
- `ProxySession`：`controller/include/vllm_slo/proxy_session.hpp`，`controller/src/proxy_session.cpp`
- Mock 上游：`tools/mock_vllm/mock_vllm_server.py`

## 面试问题

### 1. 同步 I/O 和异步 I/O 有什么区别？

30 秒回答：同步 I/O 是调用线程等待 I/O 完成后再继续执行；异步 I/O 是发起操作后立即返回，完成时由事件循环调用回调。这个项目里 Mock 上游可以同步，因为它只是测试服务；C++ 代理用异步 I/O，因为它要同时管理多个客户端、上游连接、超时和流式转发。

深入追问：异步 I/O 如何避免每个连接一个阻塞线程？

代码位置：`controller/src/proxy_session.cpp`

常见错误回答：异步 I/O 就一定更快，或者异步 I/O 等于多线程。

应能画出：客户端 read 回调 -> 上游 connect/write/read -> 下游 write 的链路。

### 2. Boost.Asio 的 `io_context` 负责什么？

30 秒回答：`io_context` 是异步事件循环，负责分发异步操作完成事件、定时器事件和信号处理。没有 `io_context.run()`，异步操作不会推进。

深入追问：多个线程调用同一个 `io_context.run()` 时会发生什么？

代码位置：`controller/src/proxy_main.cpp`

常见错误回答：`io_context` 是线程池本身。

应能画出：main 创建 `io_context` -> server 注册 accept -> `run()` 分发回调。

### 3. 异步 I/O 是否等于多线程？

30 秒回答：不是。异步 I/O 是编程模型，多线程是执行资源。一个线程也可以跑异步事件循环，多个线程只是让多个回调可以并行执行。

深入追问：多线程下共享状态怎么保护？

代码位置：`controller/src/proxy_main.cpp`，`controller/src/inflight_limiter.cpp`

常见错误回答：用了 async 就自动多线程安全。

应能画出：单线程事件循环和多 worker 事件循环差异。

### 4. 为什么 Session 通常通过 `shared_from_this` 管理生命周期？

30 秒回答：异步回调可能在发起函数返回后才执行，如果 Session 是栈对象或裸指针，回调可能访问已经销毁的对象。`shared_from_this` 让回调持有 Session，保证异步链路完成前对象活着。

深入追问：什么时候会形成循环引用？

代码位置：`controller/include/vllm_slo/proxy_session.hpp`，`controller/src/proxy_session.cpp`

常见错误回答：因为 shared_ptr 总是最安全。

应能画出：Session 发起 async_read，回调捕获 `self`。

### 5. 什么情况下需要 strand？

30 秒回答：当多个线程运行同一个 `io_context`，且某个对象的回调会访问共享状态时，需要 strand 串行化这些回调。当前 `ProxyServer` 用 strand 管理 accept、停止和队列唤醒。

深入追问：`InflightLimiter` 已有 mutex，为什么 server 队列还需要串行化？

代码位置：`controller/src/proxy_server.cpp`

常见错误回答：所有 Asio 程序都必须用 strand。

应能画出：多个 worker 同时完成回调，但 server 队列操作被 strand 排队。

### 6. HTTP Keep-Alive 是什么？

30 秒回答：Keep-Alive 允许一个 TCP 连接复用多个 HTTP 请求。当前阶段代理为了简化流式响应和关闭流程，返回时使用 `Connection: close`，不做连接复用。

深入追问：生产环境开启 Keep-Alive 有什么收益和复杂度？

代码位置：`controller/src/proxy_session.cpp`

常见错误回答：Keep-Alive 等于 SSE。

应能画出：一个 TCP 连接承载多个请求 vs 当前请求结束关闭。

### 7. SSE 与 WebSocket 有什么区别？

30 秒回答：SSE 是基于 HTTP 的服务端到客户端单向事件流，文本格式，适合 token 流式输出；WebSocket 是双向连接协议，适合双向实时通信。本项目只需要转发大模型流式输出，所以 SSE 足够。

深入追问：SSE 如何表示事件结束？

代码位置：`controller/src/sse_parser.cpp`

常见错误回答：SSE 和 WebSocket 都是一样的长连接。

应能画出：`data: ...` + 空行结束事件。

### 8. 为什么 SSE 解析器必须处理半包和粘包？

30 秒回答：TCP 是字节流，不保留应用层消息边界。一次 read 可能只读到半个事件，也可能读到多个事件，所以解析器必须增量处理并保留未完成尾部。

深入追问：如果把一次 read 当成一个事件会出什么 bug？

代码位置：`controller/src/sse_parser.cpp`，`controller/tests/test_sse_parser.cpp`

常见错误回答：HTTP chunk 和 SSE event 是同一个边界。

应能画出：`data: he` 和 `llo\n\n` 两次 chunk 合成一个事件。

### 9. 什么是 Chunked Transfer Encoding？

30 秒回答：HTTP/1.1 中当响应长度未知时，可以用 chunked 把 body 分块发送，每块有长度前缀。SSE 常见于 chunked 响应，但 SSE 事件边界仍然由 `data:` 和空行决定。

深入追问：chunk 边界和 SSE 事件边界为什么不能混用？

代码位置：`controller/src/proxy_session.cpp`，`controller/src/sse_parser.cpp`

常见错误回答：一个 HTTP chunk 就是一个 SSE event。

应能画出：HTTP chunk 包着任意字节，SSE parser 在字节流上找空行。

### 10. 为什么同一 socket 不能随意并发执行多个 write？

30 秒回答：多个未协调的异步 write 可能交错写入，破坏 HTTP/SSE 字节顺序。必须保证前一次 write 完成后再发下一次。

深入追问：如何设计写队列？

代码位置：`controller/src/proxy_session.cpp`

常见错误回答：TCP 会自动帮应用层消息排序，所以并发 write 无所谓。

应能画出：read upstream chunk -> write downstream -> write 完成后再 read 下一块。

### 11. 有界队列解决什么问题？

30 秒回答：有界队列限制等待请求数量，避免过载时内存和延迟无限增长。满了就快速拒绝，让客户端尽早重试或降载。

深入追问：队列过大为什么会伤害 TTFT？

代码位置：`controller/src/inflight_limiter.cpp`

常见错误回答：队列越大越能扛流量。

应能画出：in-flight 满 -> queue 未满 -> queue 满返回 429。

### 12. 排队、429 和 503 分别适用于什么场景？

30 秒回答：排队适合短暂超过并发但还有容量；429 表示本地限流或队列满；503 表示上游不可用或代理停止。它们表达的是不同失败来源。

深入追问：上游超时为什么更适合 504？

代码位置：`controller/src/proxy_session.cpp`

常见错误回答：所有失败都返回 500。

应能画出：本地容量问题和上游可用性问题分支。

### 13. 什么是背压？

30 秒回答：背压是系统过载时向上游调用方反馈容量不足，避免继续吞入超过处理能力的请求。本项目通过最大在途数、有界队列、429 和队列超时体现背压。

深入追问：背压和限流有什么区别？

代码位置：`controller/src/inflight_limiter.cpp`

常见错误回答：背压就是把请求全部拒绝。

应能画出：负载进入代理后被准入、排队或拒绝。

### 14. TTFT 应从哪个时间点统计到哪个时间点？

30 秒回答：本阶段 TTFT 从代理决定开始处理请求并转发上游时计时，到收到上游第一个有效 `data:` SSE 事件时结束。它是代理视角的 TTFT，不是 vLLM 内部 engine TTFT。

深入追问：排队时间要不要计入 TTFT？

代码位置：`controller/src/proxy_session.cpp`，`controller/src/proxy_metrics.cpp`

常见错误回答：TTFT 从程序启动算到响应结束。

应能画出：request accepted -> upstream first data event。

### 15. 客户端中途断开时代理应如何处理上游请求？

30 秒回答：代理应停止继续写客户端，关闭上游连接，释放 in-flight permit，并记录失败或完成状态，避免资源泄漏。

深入追问：如何保证断开路径也释放槽位？

代码位置：`controller/src/proxy_session.cpp`，`controller/src/inflight_limiter.cpp`

常见错误回答：客户端断开可以忽略，上游自然会结束。

应能画出：client write/read error -> finish/fail -> permit reset。

### 16. RAII 在网络连接和槽位释放中如何体现？

30 秒回答：`InflightLimiter::Permit` 在析构时释放在途槽位，即使请求失败或异常路径提前结束，只要 permit 被销毁就能回收计数。socket 和 timer 也通过对象生命周期管理。

深入追问：RAII 是否能替代所有错误处理？

代码位置：`controller/include/vllm_slo/inflight_limiter.hpp`，`controller/src/inflight_limiter.cpp`

常见错误回答：RAII 就是不需要显式 close。

应能画出：Permit 持有 slot，析构调用 release。

### 17. 为什么当前阶段不直接接入动态 AdmissionPolicy？

30 秒回答：动态策略需要真实或至少可信的 Metrics 输入。当前阶段没有 vLLM Metrics、NVML 和验证机数据，所以只做固定并发和队列控制，避免伪装自适应算法。

深入追问：后续怎么把 Metrics 接进来？

代码位置：`controller/include/vllm_slo/admission_policy.hpp`，`controller/src/inflight_limiter.cpp`

常见错误回答：可以随便模拟几个指标就说动态控制有效。

应能画出：阶段 2 的 Metrics Provider -> AdmissionPolicy -> limiter cap。

### 18. C++ 代理增加了一层网络跳转，为什么仍可能有价值？

30 秒回答：它增加了少量网络开销，但可以在进入 vLLM 前做准入、排队、拒绝、TTFT 观测和后续 SLO 控制，避免过载请求把 vLLM 队列和 KV Cache 压爆。

深入追问：什么时候这层代理不值得？

代码位置：`controller/src/proxy_session.cpp`，`controller/src/inflight_limiter.cpp`

常见错误回答：多一层代理一定会让系统更快。

应能画出：请求先被代理限流，再进入 vLLM。

### 19. 这个控制器和 vLLM 内部 Scheduler 有什么区别？

30 秒回答：控制器在 vLLM 外部，决定请求是否进入、等待或被拒绝；vLLM Scheduler 在引擎内部，负责已经进入的请求如何 batch、prefill/decode 和管理 KV Cache。两者控制层级不同。

深入追问：控制器能否替代 `max_num_seqs`？

代码位置：`docs/01_architecture.md`，`controller/include/vllm_slo/admission_policy.hpp`

常见错误回答：代理就是重写 vLLM Scheduler。

应能画出：Client -> Proxy admission -> vLLM Scheduler。

### 20. 为什么不直接用 Nginx 或 Envoy？

30 秒回答：Nginx/Envoy 的生产能力更成熟，本项目 C++ 代理不是为了取代它们。这个项目重点是实现与推理指标关联的准入控制，并训练 Linux/C++ 网络、并发和系统设计能力。后续可以把控制策略做成独立服务，或集成到成熟网关体系。

深入追问：生产落地时你会如何选型？

代码位置：`README.md`，`controller/src/inflight_limiter.cpp`

常见错误回答：自己写的 C++ 代理一定比 Nginx/Envoy 更强。

应能画出：成熟网关负责通用流量能力，策略服务负责推理 SLO 决策。

## 用户必须能够手写或解释的代码

- `SseParser::feed` 如何保留尾部并按空行产出事件。
- `InflightLimiter::Permit` 如何用 RAII 释放槽位。
- `ProxyMetrics::record_ttft` 如何用两个 `steady_clock::time_point` 算毫秒。
- `ProxySession` 的基本异步调用链。
- 429、503、504 的错误分支。

## 本阶段自测清单

- 能解释为什么本阶段是 Mock 验证，不是 vLLM 验证。
- 能解释 SSE 事件和 TCP read 边界的关系。
- 能解释固定并发、有界队列和快速拒绝。
- 能指出 TTFT 是代理视角指标。
- 能指出当前代码还没有动态 SLO 闭环和 NVML/vLLM Metrics。
