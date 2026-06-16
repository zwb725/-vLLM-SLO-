# vLLM 部署说明

本目录用于后续记录验证机上的 vLLM 部署命令和配置。本阶段不安装 vLLM，不下载模型权重，不声明任何真实 GPU 实验已经完成。

验证机预期环境：

- Windows 11 + WSL2 Ubuntu
- RTX 4060 Laptop 8GB
- NVIDIA Driver、CUDA、PyTorch、vLLM

后续部署时需要记录：

- GPU 和驱动版本
- CUDA、PyTorch、vLLM 版本
- 模型名称与量化格式
- vLLM 启动参数
- Benchmark 原始输出路径
