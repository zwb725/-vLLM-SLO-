# 架构说明

目标架构：

```text
Client / vllm bench
        |
        v
C++ Admission Controller
        |
        v
vLLM OpenAI-compatible Server
        |
        v
CUDA / RTX 4060
```

## 组件职责

- `vllm bench serve` 负责产生标准化负载，并提供 TTFT、TPOT、吞吐和 Goodput 等指标口径。
- C++ 控制器负责准入、限流、排队、拒绝和指标窗口统计。
- vLLM 负责模型推理、Prefill/Decode、Continuous Batching、Paged KV Cache 和内部请求调度。
- NVML 与 vLLM Metrics 负责监控显存、GPU 利用率、功耗、运行/等待请求数和 KV Cache 使用率。

## 当前阶段

阶段 0 只实现控制器的纯策略核心：

- `MetricsSnapshot`
- `SlidingWindow`
- `TokenBucket`
- `AdmissionPolicy`
- `controller_demo`

当前代码不连接 vLLM，不实现 HTTP/SSE 转发，不采集 NVML，也不运行 CUDA。

## 控制边界

`AdmissionPolicy` 推荐的是控制器侧最大在途请求数。它不是 vLLM 内部的 `max_num_seqs`，也不会在运行时修改 vLLM 调度参数。
