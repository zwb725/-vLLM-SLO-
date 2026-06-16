# Mock vLLM SSE Server

该服务只用于阶段 1 本地开发验证，不是 vLLM，不加载模型，不依赖 CUDA/GPU。

启动：

```powershell
python tools/mock_vllm/mock_vllm_server.py --host 127.0.0.1 --port 8000
```

环境变量：

- `MOCK_VLLM_FIRST_TOKEN_DELAY_SECONDS`：首 Token 延迟。
- `MOCK_VLLM_TOKEN_DELAY_SECONDS`：Token 间延迟。
- `MOCK_VLLM_MODE=ok|http500|drop|hang`：模拟成功、HTTP 500、连接中断或长时间不响应。
- `MOCK_VLLM_HANG_SECONDS`：`hang` 模式睡眠时间。

接口：

- `GET /health`
- `POST /v1/chat/completions`

POST 接口会返回固定 SSE 流：

```text
data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" from"}}]}

data: {"choices":[{"delta":{"content":" mock"}}]}

data: [DONE]
```
