# 项目范围

本项目面向单卡 RTX 4060 8GB 平台，目标是围绕 vLLM 推理服务建立可观测、可压测、可调优的 SLO 感知性能优化流程。

## 包含内容

- vLLM OpenAI-compatible 流式推理服务
- `vllm bench serve` 标准化压测
- vLLM Metrics、NVML 与 Nsight Systems 性能分析
- C++17 前置异步请求准入控制器
- 基于 TTFT、队列、KV Cache 和显存水位的 SLO/背压控制
- 低延迟与高吞吐配置对比

## 不包含内容

- 模型训练、SFT、RAG 或 Agent
- 网页聊天前端
- 多 GPU 或 Kubernetes
- 本阶段的真实 HTTP/SSE 代理
- 本阶段的真实 GPU 实验数据

没有真实实验记录时，不得声明性能提升、吞吐提升或显存优化结果。
