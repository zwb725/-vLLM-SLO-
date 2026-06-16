# 基于 vLLM 的单卡大模型推理服务与 SLO 感知性能优化

本仓库用于实现一个面向单卡 RTX 4060 8GB 平台的 vLLM 推理服务性能优化项目。最终目标是搭建 OpenAI-compatible 流式推理服务，并在服务前置层实现 SLO 感知的请求准入、限流、排队、拒绝和性能观测。

当前仓库处于阶段 1：在阶段 0 纯 C++17 策略核心基础上，新增本地 Mock OpenAI-compatible SSE 服务、SSE 增量解析器、固定并发/有界队列限制器、TTFT 基础指标和 Boost.Asio/Beast HTTP/SSE 代理骨架。

阶段 1 只验证本地链路，不连接真实 vLLM，不依赖 GPU、CUDA、NVML 或模型权重，也不展示任何未实际运行的性能提升数据。

## 目标架构

本地开发链路：

```text
curl / 本地测试客户端
        |
        v
C++ Async Admission Proxy
        |
        v
Mock OpenAI-compatible SSE Server
```

未来验证机链路：

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
- SSE 增量解析器 `SseParser`
- 固定并发与有界队列限制器 `InflightLimiter`
- TTFT 基础指标 `ProxyMetrics`
- Mock OpenAI-compatible SSE Server
- Boost.Asio/Beast 代理骨架，Boost 可用时构建 `vllm_slo_proxy`
- CTest 测试入口
- 基础文档、配置示例、CI 和双机工作流说明

## 项目机与验证机工作流

- 项目机：Windows，负责编码、CMake 构建、单元测试和 Git 提交。
- 验证机：Windows 11 + WSL2 Ubuntu + RTX 4060 Laptop 8GB，负责后续 vLLM 部署、GPU Benchmark、NVML/Nsight 数据采集和真实结果回填。

## 依赖安装

基础核心测试需要：

- CMake 3.21+
- C++17 编译器
- Ninja 或其他 CMake 生成器

代理网络模块需要 Boost.Asio/Beast。Windows 推荐使用 vcpkg：

```powershell
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
cmake --preset windows-debug -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
```

Linux 示例：

```bash
sudo apt-get update
sudo apt-get install -y cmake ninja-build g++ libboost-dev python3 curl
```

本仓库提供 `vcpkg.json`，只声明阶段 1 需要的 Boost 依赖。没有 Boost 时，CMake 应继续构建 SSE、限流和指标核心测试，但不会构建真实代理可执行文件。

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

## 本地 Mock 验证

启动 Mock 上游：

```powershell
.\scripts\local\start_mock_server.ps1
```

如果 `python` 不在 PATH 中，可以指定：

```powershell
.\scripts\local\start_mock_server.ps1 -PythonExe "C:\path\to\python.exe"
```

启动代理：

```powershell
.\scripts\local\start_proxy.ps1
```

执行 Smoke Test：

```powershell
.\scripts\local\smoke_test_proxy.ps1
```

停止本地进程：

```powershell
.\scripts\local\stop_local_stack.ps1
```

## 当前边界

以下内容尚未实现：

- vLLM 服务部署
- vLLM Metrics 解析
- NVML 采集
- Nsight Systems 分析
- CUDA 或 Triton Kernel 实验
- 真实 Benchmark 数据
- 动态 SLO 闭环控制
- 生产级网关能力

没有真实实验数据时，不得展示 P95 TTFT、Goodput、吞吐或显存控制等性能提升结论。

## 后续里程碑

1. 完善 Boost.Asio/Beast HTTP/SSE 代理的集成测试。
2. 接入 Prometheus Metrics 解析、Mock Metrics Provider 和动态 AdmissionPolicy。
3. 在验证机部署 vLLM 与 AWQ 模型。
4. 使用 `vllm bench serve` 固定负载并采集真实基线。
5. 对比低延迟与高吞吐配置，并按实验契约回填结果。
