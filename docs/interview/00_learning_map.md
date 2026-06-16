# 阶段学习地图

本文件用于把阶段 1 的代码和面试知识点对应起来。当前阶段只完成本地 Mock 链路，不代表真实 vLLM 或 GPU 实验已经完成。

| 主线 | 当前项目源码 | 优先级 | 是否实现 | 面试追问深度 |
| --- | --- | --- | --- | --- |
| 同步 I/O | `tools/mock_vllm/mock_vllm_server.py` | 了解 | 已实现 Mock | 能说明同步服务为何足够做本地上游 |
| 异步 I/O | `controller/src/proxy_session.cpp` | 必须 | Boost 可用时构建 | 能说出异步回调链和错误处理 |
| `io_context` 与事件循环 | `controller/src/proxy_main.cpp` | 必须 | Boost 可用时构建 | 能解释 run、worker thread、signal stop |
| Session 生命周期 | `controller/include/vllm_slo/proxy_session.hpp` | 必须 | Boost 可用时构建 | 能解释 `shared_from_this` 为什么必要 |
| HTTP | `controller/src/proxy_session.cpp` | 必须 | MVP | 能区分 `/health`、POST 转发、错误码 |
| SSE | `controller/src/sse_parser.cpp` | 必须 | 已实现并可独立测试 | 能解释半包、粘包、空行结束事件 |
| 并发控制 | `controller/src/inflight_limiter.cpp` | 必须 | 已实现并可独立测试 | 能解释 in-flight 槽位和 RAII 释放 |
| 有界队列 | `controller/src/inflight_limiter.cpp` | 必须 | 已实现计数模型 | 能解释 Queue、Reject 和取消队列 |
| 超时与取消 | `controller/src/proxy_session.cpp` | 重要 | 代理骨架已实现 | 能说明请求超时、队列超时、客户端断开 |
| 背压 | `controller/src/inflight_limiter.cpp` | 必须 | 固定策略已实现 | 能解释为什么队列必须有上限 |
| TTFT | `controller/src/proxy_metrics.cpp` | 必须 | 已实现基础统计 | 能说明起点和终点 |
| vLLM 服务边界 | `docs/01_architecture.md` | 必须 | 文档化 | 能区分 C++ 代理和 vLLM Scheduler |

## 建议阅读顺序

1. `controller/include/vllm_slo/sse_parser.hpp`
2. `controller/src/sse_parser.cpp`
3. `controller/include/vllm_slo/inflight_limiter.hpp`
4. `controller/src/inflight_limiter.cpp`
5. `controller/src/proxy_metrics.cpp`
6. `controller/include/vllm_slo/proxy_session.hpp`
7. `controller/src/proxy_session.cpp`
8. `controller/src/proxy_server.cpp`
9. `tools/mock_vllm/mock_vllm_server.py`

## 本阶段解决的工程问题

阶段 1 解决的是：在没有真实 vLLM、GPU 和模型权重的项目机上，先把“客户端 -> C++ 代理 -> 本地 SSE 上游”的网络形态搭起来，并把 SSE 解析、固定并发、有界队列、TTFT 统计这些后续动态 SLO 控制需要的基础能力做成可测试模块。

## 当前设计原因

- SSE 解析器独立出来，避免把协议边界藏在 socket read 回调里。
- 固定并发限制器独立于动态 `AdmissionPolicy`，避免在没有真实 Metrics 时伪装自适应控制。
- Mock 服务使用 Python 标准库，避免引入大型 Python 依赖。
- Boost 网络目标和无 Boost 支持库分离，便于缺依赖机器仍能测试核心控制逻辑。

## 本阶段自测清单

- `SseParser` 能处理完整事件、半包、粘包、CRLF、注释、`[DONE]` 和缓存上限。
- `InflightLimiter` 能处理 Admit、Queue、Reject、释放、取消和关闭。
- `ProxyMetrics` 能用确定性时间点统计 TTFT。
- Mock 服务明确标注不是 vLLM。
- README 不展示真实 GPU 性能数据。
