# 第 2 章：Token、Embedding 与 Tensor Shape

## 1. 本章目标

学完本章后，你应该能回答：

- Tokenizer 为什么要把文本变成 Token ID？
- Vocabulary、Token、Token ID、Embedding 之间是什么关系？
- `[B, S, H]` 里的 `B`、`S`、`H` 分别代表什么？
- 为什么 Token ID 不是可以直接做线性计算的“语义特征”？

## 2. 五分钟直觉

Tokenizer（分词器）：把文本切成 Token，并把 Token 映射成整数 ID 的规则系统。模型不能直接处理“你好”“Transformer”这样的字符串，只能处理数字张量，所以文本必须先被编码成 Token ID。

Token（词元）：模型看到的离散文本单元。它不一定等于一个字或一个英文单词，可能是子词、标点、空格片段或字节片段。

Vocabulary（词表）：Token 到 Token ID 的映射表。比如一个模型的词表里可能有：

| Token | Token ID |
| --- | --- |
| `你` | 872 |
| `好` | 1962 |
| `Transformer` | 4242 |

Token ID（词元编号）：Token 在词表里的整数编号。它只是编号，不是数值特征。ID `1000` 不代表比 ID `10` “大 100 倍”，也不代表语义更强。

Embedding（嵌入表）：把 Token ID 查成向量的参数表。可以把它想成一张大表：

```text
Embedding Table: [V, H]
```

其中 `V` 是词表大小，`H` 是每个 Token 的向量维度。输入 Token ID 后，模型从这张表里取出对应行，得到可计算的向量。

## 3. 完整计算或数据流

```mermaid
flowchart LR
    A["文本 Prompt"] --> B["Tokenizer<br/>切分文本"]
    B --> C["Tokens<br/>离散词元"]
    C --> D["Vocabulary Lookup<br/>查词表"]
    D --> E["Token IDs<br/>[B, S]"]
    E --> F["Embedding Table<br/>[V, H]"]
    F --> G["Hidden States<br/>[B, S, H]"]
```

用一句话概括：

```text
文本 -> Tokens -> Token IDs [B, S] -> Embedding Lookup -> Hidden States [B, S, H]
```

## 4. 关键术语

- Tokenizer（分词器）：把文本切成 Token，并转换为模型词表 ID 的规则系统。
- Token（词元）：模型处理文本的最小离散单元，可能是词、子词、字符、标点或字节片段。
- Vocabulary（词表）：Token 与整数 ID 的映射集合。
- Token ID（词元编号）：Token 在词表中的整数索引。
- Embedding（嵌入表）：把 Token ID 映射成连续向量的参数矩阵。
- Batch Size（批大小）：一次送入模型的样本或请求数量，记作 `B`。
- Sequence Length（序列长度）：每条样本中的 Token 数量，记作 `S`。
- Hidden Size（隐藏维度）：每个 Token 向量的长度，记作 `H`。
- Padding（填充）：为了让同一批序列长度一致，在短序列后补特殊 Token。
- Attention Mask（注意力掩码）：告诉模型哪些位置是真实 Token，哪些位置是 Padding。

## 5. Tensor Shape

本章最重要的 Shape：

```text
Token IDs: [B, S]
Embedding Table: [V, H]
Hidden States: [B, S, H]
Attention Mask: [B, S]
```

维度含义：

- `B`：Batch Size，一批有多少条输入。
- `S`：Sequence Length，每条输入有多少个 Token。
- `V`：Vocabulary Size，词表里有多少个 Token。
- `H`：Hidden Size，每个 Token 被表示成多长的向量。

例子：

```text
B = 2
S = 5
H = 4096
V = 32000

Token IDs: [2, 5]
Embedding Table: [32000, 4096]
Hidden States: [2, 5, 4096]
```

解释：

- `[2, 5]` 表示一批有 2 条文本，每条文本被切成 5 个 Token。
- `[32000, 4096]` 表示词表有 32000 行，每行是 4096 维向量。
- `[2, 5, 4096]` 表示每个 Token ID 都查到了一个 4096 维向量。

## 6. 核心公式

Embedding Lookup 可以写成：

```text
hidden[b, s, :] = embedding_table[token_ids[b, s], :]
```

变量含义：

- `token_ids[b, s]`：第 `b` 条样本第 `s` 个位置的 Token ID。
- `embedding_table[id, :]`：嵌入表中第 `id` 行的向量。
- `hidden[b, s, :]`：查表后得到的 Token 向量。

工程意义：

- Embedding Lookup 本质上是查表，不是把 Token ID 当连续数值去乘。
- Token ID 的大小没有直接语义，语义来自它查到的向量。
- 进入 Transformer 主体前，模型需要的是 `[B, S, H]`，不是 `[B, S]`。

## 7. 与推理 Runtime 的联系

在推理 Runtime 里，Tokenizer 和 Embedding 处在请求入口的最前面：

1. 用户输入文本。
2. Tokenizer 把文本变成 Token IDs `[B, S]`。
3. Runtime 根据 `S` 估计请求长度、上下文长度、KV Cache 需求和调度压力。
4. Model Runner 进入模型前，Embedding 把 `[B, S]` 变成 `[B, S, H]`。
5. 后续 Transformer 层都围绕 `[B, S, H]` 继续计算。

所以 Token 数量比“字符数”更接近推理成本。一个 Prompt 看起来短，但如果被 tokenizer 切成很多 Token，就会增加 Prefill 工作量和 KV Cache 占用。

## 8. 易错点

| 易错说法 | 问题 | 正确认知 |
| --- | --- | --- |
| Token ID 越大语义越强 | ID 只是索引 | 语义来自 Embedding 向量 |
| 一个 Token 就是一个字 | 不一定 | Token 可能是子词、字节片段或标点 |
| Embedding 会“计算位置” | 职责混淆 | Embedding 表示 Token，位置信息由位置编码机制提供 |
| `[B, S, H]` 只有训练才用 | 错 | 推理阶段同样使用这些 Shape |
| 文本长度等于 Token 数 | 错 | 不同 tokenizer 切分结果不同 |
| Padding 可以让模型随便看 | 错 | 需要 Attention Mask 告诉模型哪些是填充 |

## 9. 面试回答模板

如果被问“Token、Token ID 和 Embedding 是什么关系”，可以这样答：

1. Tokenizer 先把原始文本切成 Token。
2. Vocabulary 把每个 Token 映射成 Token ID。
3. Token ID 只是离散索引，不能直接当连续语义特征。
4. Embedding Table 根据 Token ID 查表，得到 `[H]` 维向量。
5. 一批输入从 `[B, S]` 变成 `[B, S, H]` 后，才能进入 Transformer 主体。

如果追问推理成本，可以补充：`S` 是 Token 数，直接影响 Prefill 计算量和 KV Cache 长度。

## 10. 真实面试问题

本章暂未收录与 Token、Embedding、Tensor Shape 直接相关的 `VERIFIED` 或 `PARTIAL` 面试问题。

### 未核实候选问题（UNVERIFIED）

以下问题来自本章知识点推导，已按牛客网、知乎、小红书、脉脉、CSDN、GitHub 和公开搜索结果做跨平台复核，但暂时没有可访问的一手面经正文支撑，只能用于自测，不能当作真实面经或高频题。完整候选池见 `面试题/未核实候选问题.md`，复核记录见 `面试题/来源登记.md` 的 I008。

1. 为什么 Token ID 不能直接作为连续语义特征？
   - 对应能力：解释 Token ID 只是离散索引，Embedding 才是可计算向量。
   - 30 秒回答：Token ID 是词表里的编号，只表示“第几个词元”，编号大小没有语义距离。模型不能把 ID 当连续特征直接做线性计算，否则 ID 1000 会被错误理解成比 ID 10 大很多。Embedding Lookup 会用 ID 去查参数表，把离散词元映射成 `[H]` 维向量，后续 Transformer 才能在连续向量空间里计算。
2. Embedding Lookup 的输入输出 Shape 是什么？
   - 对应能力：能把 `[B, S]` 到 `[B, S, H]` 说清楚。
   - 30 秒回答：Tokenizer 输出的 Token IDs 通常是 `[B, S]`，其中 `B` 是 batch size，`S` 是序列长度。Embedding Table 是 `[V, H]`，`V` 是词表大小，`H` 是 hidden size。每个 token id 查表得到一个 `[H]` 向量，所以输出 hidden states 是 `[B, S, H]`。

## 11. 我的回答

待用户回答本章验收问题后填写。

## 12. 纠错记录

暂无。

## 13. 本章验收

请用自己的话回答：

1. 为什么 Token ID 只是索引，不能直接当作浮点语义特征？
2. `Token IDs: [B, S]` 和 `Hidden States: [B, S, H]` 分别表示什么？
3. 假设 `B=4, S=128, H=4096`，那么 Embedding 之后的 hidden states Shape 是什么？其中每个维度是什么意思？

## 14. 参考资料

- 页面标题：Tokenizers
  - 发布者或作者：Hugging Face
  - URL：https://huggingface.co/learn/llm-course/chapter2/4
  - 发布时间：未确认
  - 访问日期：2026-06-18
  - 来源类型：官方课程文档
  - 本文使用内容：Tokenizer、Token、Vocabulary、Token ID 和 encoding 流程。
- 页面标题：Tokenizers
  - 发布者或作者：Hugging Face
  - URL：https://huggingface.co/docs/tokenizers/index
  - 发布时间：未确认
  - 访问日期：2026-06-18
  - 来源类型：官方文档
  - 本文使用内容：Tokenizer 库和分词器定位。
- 页面标题：Embedding - PyTorch 2.12 documentation
  - 发布者或作者：PyTorch
  - URL：https://docs.pytorch.org/docs/2.12/generated/torch.nn.Embedding.html
  - 发布时间：未确认
  - 访问日期：2026-06-18
  - 来源类型：官方文档
  - 本文使用内容：Embedding 作为索引查表得到向量的概念。
- 页面标题：Attention Is All You Need
  - 发布者或作者：Ashish Vaswani 等，arXiv
  - URL：https://arxiv.org/abs/1706.03762
  - 发布时间：2017-06-12
  - 访问日期：2026-06-18
  - 来源类型：论文
  - 本文使用内容：Transformer 使用 input/output embeddings 和位置编码的来源。
