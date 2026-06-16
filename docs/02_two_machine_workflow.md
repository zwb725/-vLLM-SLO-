# 双机工作流

## 项目机

项目机是当前 Windows 项目机，职责包括：

- 编码
- CMake 构建
- 单元测试
- Git 提交
- 文档维护

项目机不负责生成真实 GPU 性能结论。

## 验证机

验证机是 Windows 11 + WSL2 Ubuntu + RTX 4060 Laptop 8GB，职责包括：

- `git pull`
- 安装 WSL2、CUDA、PyTorch、vLLM 等环境
- 部署 vLLM 与模型
- 运行 GPU Benchmark
- 采集 NVML、vLLM Metrics 和 Nsight Systems 数据
- 将真实结果和实验契约回填仓库

## 推荐流程

1. 项目机完成代码和测试。
2. 项目机提交并推送到 GitHub。
3. 验证机拉取相同 commit。
4. 验证机记录环境并运行实验。
5. 验证机将原始结果路径、配置和摘要回填文档。
