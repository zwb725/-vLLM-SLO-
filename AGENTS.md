# 项目协作约定

## 项目目标与边界

本项目目标是构建一个基于 vLLM 的单卡大模型推理服务与 SLO 感知性能优化项目。最终规划包含 OpenAI-compatible 流式推理服务、标准化压测、vLLM Metrics/NVML/Nsight Systems 性能分析，以及 C++17 前置请求准入控制器。

当前阶段仅完成“阶段 0：仓库初始化与可测试核心骨架”。本阶段只实现不依赖 vLLM、GPU、网络、CUDA、NVML、Boost 或 YAML 解析库的纯 C++ 策略核心。不得伪造真实 vLLM、CUDA、GPU 或 Benchmark 实验结果。

## 项目机与验证机职责

- 项目机：Windows 环境，用于 Codex 编写代码、CMake 构建、单元测试和 Git 管理。
- 验证机：Windows 11 + WSL2 Ubuntu + RTX 4060 Laptop 8GB，用于后续部署 vLLM、运行 GPU 实验、采集真实数据并回填仓库。

## C++ 与工程规范

- 使用 C++17 标准。
- CMake 工程应保持简单、可测试、可跨平台。
- Windows/MSVC 使用 `/W4`，GCC/Clang 使用 `-Wall -Wextra -Wpedantic`。
- 修改代码后必须运行格式检查、构建和测试，无法运行时必须明确说明原因。
- 公共接口需要有必要注释，说明行为边界、单位和异常。
- 优先使用 RAII、值语义和明确所有权。
- 避免裸 `new`/`delete`。
- 网络模块未来使用 Boost.Asio/Beast，但本阶段不实现真实 HTTP/SSE 代理。
- 未经允许不得增加生产依赖。

## 文档与命名

- Markdown 文档默认使用中文。
- 代码、类名、函数名、变量名和 Git 提交信息使用英文。
- 配置示例可以使用英文键名，注释使用中文。

## 数据与安全

- 禁止伪造性能数据。
- 禁止提交模型权重、密钥、真实 Token、虚拟环境、构建目录、缓存、原始大文件或二进制大文件。
- Benchmark 结果必须来自真实运行，并记录实验契约中要求的环境、版本、负载和原始结果路径。

## Git 要求

push 前需要展示：

- `git status`
- `git diff --check`
- 主要文件和测试结果摘要

仅当 origin 正确、无敏感文件、构建和测试状态明确、Git 认证可用时才执行 push。禁止 force push、`git reset --hard`、`git clean -fd` 或修改全局 Git 配置。
