# 里程碑

## 阶段 0：仓库初始化与可测试核心骨架

- 建立 CMake/CTest 工程
- 实现纯 C++17 策略核心
- 提供 demo、配置示例、文档和 CI
- 不连接 vLLM/GPU/网络

## 阶段 1：HTTP/SSE 前置控制器

- 引入 Boost.Asio/Beast
- 实现 OpenAI-compatible 请求转发
- 支持 SSE 流式响应解析和转发
- 实现超时、取消和安全退出

## 阶段 2：监控接入

- 解析 vLLM Metrics
- 接入 NVML
- 将采样结果送入准入策略

## 阶段 3：验证机真实实验

- 部署 vLLM 和 AWQ 模型
- 使用 `vllm bench serve` 生成标准化负载
- 记录基线和控制器对比结果

## 阶段 4：性能分析与配置沉淀

- 使用 Nsight Systems 分析时间线
- 对比 `max_num_seqs`、`max_num_batched_tokens`、Prefix Caching 和 Chunked Prefill
- 形成低延迟与高吞吐两套配置
