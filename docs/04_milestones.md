# 里程碑

## 阶段 0：仓库初始化与可测试核心骨架（已完成）

- 建立 CMake/CTest 工程
- 实现纯 C++17 策略核心
- 提供 demo、配置示例、文档和 CI
- 不连接 vLLM/GPU/网络

## 阶段 1：异步 HTTP/SSE 代理与本地 Mock 验证（进行中）

- 引入 Boost.Asio/Beast
- 实现 OpenAI-compatible 请求转发
- 支持 SSE 流式响应解析和转发
- 实现超时、取消和安全退出
- 提供 Mock OpenAI-compatible SSE Server
- 使用本地 Smoke Test 验证 `/health` 和流式响应
- 不连接真实 vLLM/GPU/CUDA/NVML

## 阶段 2：Metrics 接入与动态策略接线

- Prometheus Metrics 解析
- Mock Metrics Provider
- 动态 `AdmissionPolicy` 接线
- 控制器侧最大在途请求数动态调整
- 仍不声明真实 GPU 实验结果

## 阶段 3：验证机真实实验

- 部署 vLLM 和 AWQ 模型
- 使用 `vllm bench serve` 生成标准化负载
- 记录基线和控制器对比结果
- 接入 NVML
- 使用 Nsight Systems 分析时间线
- 回填真实实验数据

## 阶段 4：性能分析与配置沉淀

- 使用 Nsight Systems 分析时间线
- 对比 `max_num_seqs`、`max_num_batched_tokens`、Prefix Caching 和 Chunked Prefill
- 形成低延迟与高吞吐两套配置
