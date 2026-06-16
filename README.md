# 基于 vLLM 的单卡大模型推理服务与 SLO 感知性能优化

本仓库用于实现一个面向单卡 RTX 4060 8GB 平台的 vLLM 推理服务性能优化项目。最终目标是搭建 OpenAI-compatible 流式推理服务，并在服务前置层实现 SLO 感知的请求准入、限流、排队、拒绝和性能观测。

当前仓库处于阶段 0：只完成可测试的 C++17 策略核心骨架，不连接 vLLM，不依赖 GPU，不实现真实 HTTP/SSE 代理，也不展示任何未实际运行的性能提升数据。

## 目标架构

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

## 当前已完成

- C++17 核心库 `vllm_slo_controller_core`
- 固定容量线程安全滑动窗口
- 线程安全令牌桶
- 基于阈值、冷却和滞回的准入策略核心
- 内置模拟指标序列的 `controller_demo`
- CTest 测试入口
- 基础文档、配置示例、CI 和双机工作流说明

## 项目机与验证机工作流

- 项目机：Windows，负责编码、CMake 构建、单元测试和 Git 提交。
- 验证机：Windows 11 + WSL2 Ubuntu + RTX 4060 Laptop 8GB，负责后续 vLLM 部署、GPU Benchmark、NVML/Nsight 数据采集和真实结果回填。

## 本地构建

Windows:

```powershell
cmake --preset windows-debug
cmake --build --preset windows-debug
ctest --preset windows-debug --output-on-failure
```

Linux:

```bash
cmake --preset linux-debug
cmake --build --preset linux-debug
ctest --preset linux-debug --output-on-failure
```

## 当前边界

本阶段只实现纯 C++ 策略核心。以下内容尚未实现：

- 真实 OpenAI-compatible HTTP/SSE 代理
- vLLM 服务部署
- vLLM Metrics 解析
- NVML 采集
- Nsight Systems 分析
- CUDA 或 Triton Kernel 实验
- 真实 Benchmark 数据

没有真实实验数据时，不得展示 P95 TTFT、Goodput、吞吐或显存控制等性能提升结论。

## 后续里程碑

1. 接入 Boost.Asio/Beast，实现 HTTP/SSE 流式转发。
2. 接入 vLLM `/metrics` 与 NVML 采集。
3. 在验证机部署 vLLM 与 AWQ 模型。
4. 使用 `vllm bench serve` 固定负载并采集真实基线。
5. 对比低延迟与高吞吐配置，并按实验契约回填结果。
