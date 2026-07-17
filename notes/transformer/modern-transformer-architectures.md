# 现代 Transformer 架构与 AI Infra 学习路线

## 1. 结论

现代 Transformer 并不只有一种架构。当前生成式 LLM 最常见的基准结构是：

> Pre-Norm、decoder-only、causal attention、RMSNorm、RoPE、GQA、SwiGLU Transformer。

Llama、Qwen 等模型大体属于这个范式，但 Transformer 还包括 encoder-only、encoder-decoder，以及大量 attention、FFN、位置编码和残差结构变体。

学习时应先掌握一个标准的 Llama-style decoder-only block，再通过与基准结构比较来理解其他设计。

深入理解单个 block 中 Attention、FFN、Norm、Residual 的逻辑、tensor shape 和 FLOPs 推导，见：[Transformer Block 深入理解](transformer-block-deep-dive.md)。

## 2. Transformer 的三种宏观架构

| 架构 | Attention | 主要用途 | 代表模型 |
| --- | --- | --- | --- |
| Encoder-only | 双向 attention，无 causal mask | 表征、分类、检索 | BERT |
| Decoder-only | 单向 causal attention | 自回归生成 | GPT、Llama、Qwen |
| Encoder-decoder | Encoder 双向；Decoder causal + cross-attention | 翻译、摘要、条件生成 | 原始 Transformer、T5 |

### 2.1 Encoder-only

每个 token 可以看到整个输入：

```text
token 1: ✓ ✓ ✓ ✓
token 2: ✓ ✓ ✓ ✓
token 3: ✓ ✓ ✓ ✓
token 4: ✓ ✓ ✓ ✓
```

这种结构适合编码完整序列，但不能直接按照自回归方式生成长文本。

### 2.2 Decoder-only

每个 token 只能看到自己和之前的 token：

```text
token 1: ✓ ✗ ✗ ✗
token 2: ✓ ✓ ✗ ✗
token 3: ✓ ✓ ✓ ✗
token 4: ✓ ✓ ✓ ✓
```

它与语言模型分解直接对应：

$$
P(x)=\prod_tP(x_t\mid x_{<t})
$$

因此 decoder-only 成为现代生成式 LLM 的主流架构。

### 2.3 Encoder-decoder

Decoder block 通常包含：

```text
Masked Self-Attention
        ↓
Cross-Attention：读取 Encoder 输出
        ↓
FFN
```

《Attention Is All You Need》提出的是面向机器翻译的 encoder-decoder Transformer，并不是今天生成式 LLM 常用的纯 decoder-only 架构。

## 3. 推荐作为基准的 Decoder-only Transformer

建议首先掌握下面的 Llama-style block：

```text
Token IDs
   ↓
Token Embedding
   ↓
┌─────────────────────────────────┐
│ x = x + GQA(RMSNorm(x), RoPE)   │
│ x = x + SwiGLU(RMSNorm(x))      │
└─────────────────────────────────┘ × N
   ↓
Final RMSNorm
   ↓
LM Head
   ↓
Vocabulary Logits
```

其中：

- Token Embedding：将 token ID 映射为 hidden vector；
- RMSNorm：控制 hidden state 的数值尺度；
- RoPE：为 Q/K 注入位置信息；
- GQA：多个 Query heads 分组共享较少的 K/V heads；
- SwiGLU：逐 token 的非线性特征变换；
- Residual：保留原始信息，并改善深层网络优化；
- LM Head：将 hidden state 映射为词表 logits。

## 4. Decoder-only 内部的主要变体

### 4.1 Normalization 与 Residual

原始 Transformer 常用 Post-Norm：

$$
x'=\operatorname{Norm}(x+\operatorname{Attention}(x))
$$

现代 LLM 更常使用 Pre-Norm：

$$
x'=x+\operatorname{Attention}(\operatorname{Norm}(x))
$$

$$
y=x'+\operatorname{FFN}(\operatorname{Norm}(x'))
$$

常见 normalization 包括 LayerNorm 和 RMSNorm。Pre-Norm 通常更有利于深层模型的训练稳定性。

### 4.2 Attention Head 组织

| 方法 | Query heads | KV heads | 主要系统影响 |
| --- | ---: | ---: | --- |
| MHA | $H$ | $H$ | 表达能力强，KV Cache 最大 |
| MQA | $H$ | 1 | KV Cache 最小，但共享程度最高 |
| GQA | $H$ | $H_{kv}<H$ | 质量与推理效率之间的折中 |
| MLA | 多头 Query | 压缩/重参数化 KV | 进一步降低 KV 表示和缓存成本 |

这些设计需要从以下角度比较：

- KV Cache 容量；
- Decode 阶段的 HBM 读取量；
- 单卡可承载的并发请求数；
- Tensor Parallel 通信；
- 对模型质量的影响。

### 4.3 Attention 范围与替代方案

Transformer 不一定在每一层执行完整的 $L\times L$ attention：

- Full attention；
- Sliding-window attention；
- Local/global hybrid attention；
- Block-sparse attention；
- Linear attention；
- State-space model 与 attention 的混合架构。

完整 attention 表达能力强，但长上下文 prefill 成本高。局部、稀疏或线性化 attention 可以降低成本，但会改变信息传播路径和模型能力。

### 4.4 位置编码

常见位置编码包括：

- Sinusoidal positional encoding；
- Learned absolute position embedding；
- RoPE；
- ALiBi；
- 各种 RoPE scaling 方法。

现代 decoder-only LLM 经常使用 RoPE，但并非所有 Transformer 都如此。

### 4.5 FFN 与 MoE

原始 Transformer FFN：

$$
\operatorname{FFN}(x)=\operatorname{ReLU}(xW_1)W_2
$$

现代 Llama-style SwiGLU：

$$
\operatorname{FFN}(x)=
\left[
\operatorname{SiLU}(xW_{gate})\odot xW_{up}
\right]W_{down}
$$

FFN 又可分为：

- Dense FFN：所有 token 使用同一组参数；
- MoE：router 将 token 分配给不同 experts，仅激活少数 experts。

MoE 能增加总参数容量，同时控制每个 token 的 active FLOPs，但会引入 routing、load balancing、expert parallelism 和 all-to-all 通信问题。

## 5. 面向 AI Infra 的六层理解方法

学习每个 Transformer 模块时，都应回答下面六类问题：

| 层次 | 核心问题 |
| --- | --- |
| 数学 | 公式是什么？ |
| Shape | 输入、中间状态和输出 tensor 多大？ |
| 语义 | 为什么需要这个模块？ |
| Compute | 参数量和 FLOPs 是多少？ |
| Memory | 需要读写多少字节？缓存什么？ |
| System | 在 GPU 上为什么快或慢？瓶颈在哪里？ |

例如分析 GQA：

- 数学：多个 Query heads 分组共享 K/V；
- Shape：$H_q>H_{kv}$；
- 语义：减少 K/V 冗余；
- Compute：降低 K/V projection 成本；
- Memory：KV Cache 约缩小 $H_q/H_{kv}$ 倍；
- System：降低 decode HBM traffic，提高服务并发。

## 6. 结合 CS336 的学习顺序

### 阶段 1：Resource Accounting

先学习 CS336 Lecture 2，掌握：

- tensor shape；
- parameter count；
- FLOPs；
- activation memory；
- arithmetic intensity；
- GPU compute 与 HBM bandwidth。

目标是看到：

$$
[B,L,d]\times[d,d_{ff}]
$$

就能立即写出输出形状、参数量、FLOPs 和内存读写量。

### 阶段 2：标准架构

重点学习：

- [CS336 2026 Lecture 3 — Architectures](https://www.youtube.com/watch?v=lVynu4bo1rY)
- [CS336 课程材料](https://cs336.stanford.edu/)

建议按以下顺序理解：

```text
Embedding
→ RMSNorm
→ RoPE
→ Single-head Attention
→ Multi-head Attention
→ Causal Mask
→ Residual
→ SwiGLU
→ LM Head
```

### 阶段 3：Attention Alternatives 与 MoE

完成标准架构后再学习：

- [CS336 2026 Lecture 4 — Attention Alternatives](https://www.youtube.com/watch?v=cKSwj_qZ8Jg)

学习每种变体时，重点回答：

> 相比标准 Transformer，它改变了哪个计算、内存或通信瓶颈？

### 阶段 4：实现与验证

至少亲手实现一次：

1. Single-head causal attention；
2. Multi-head attention；
3. RoPE；
4. RMSNorm；
5. SwiGLU；
6. 完整 Transformer block；
7. 自回归 generation；
8. KV Cache。

CS336 Assignment 1 要求实现 tokenizer、标准 Transformer、optimizer 并训练最小语言模型；Assignment 2 要求 profile 模型并实现 FlashAttention2。这两部分与 AI Infra 学习目标高度一致。

### 阶段 5：推理系统

架构掌握后继续学习：

- Prefill 与 Decode；
- KV Cache；
- Continuous Batching；
- PagedAttention；
- FlashAttention；
- Tensor Parallel；
- Quantization；
- Speculative Decoding。

## 7. Infra 面试检查表

### Tensor Shape

给定：

$$
X\in\mathbb R^{B\times L\times d}
$$

应能写出 Q、K、V、attention score、attention output 和 FFN 中间状态的 shape。

### 参数与 FLOPs

应能推导：

- Q/K/V/O projection；
- $QK^T$；
- $AV$；
- SwiGLU FFN；
- LM Head。

### Prefill 与 Decode

应能解释：

- 为什么 prefill 更容易 compute-bound；
- 为什么 decode 更容易 memory-bound；
- 为什么 decode 的矩阵乘法容易退化成 GEMV；
- batch size 如何影响硬件利用率。

### KV Cache

应能推导：

$$
\text{KV bytes}
=
2BNLH_{kv}d_h\times\text{bytes per element}
$$

并解释 MHA、GQA、MQA 对 KV Cache 和并发能力的影响。

### FlashAttention

需要明确：

> FlashAttention 保持标准 attention 的数学结果，核心收益主要来自减少 HBM 读写以及避免物化完整的 $L\times L$ attention matrix，而不是简单减少理论 FLOPs。

### MoE

应掌握：

- Total parameters 与 active parameters；
- Router 与 top-k experts；
- Load balancing；
- Capacity factor；
- Expert Parallelism；
- All-to-all communication。

## 8. 当前学习重点

现阶段先不要追求覆盖所有 Transformer 变体。优先把标准 decoder-only block 理解到以下程度：

1. 能从 token ID 追踪到 vocabulary logits；
2. 能画出每一步 tensor shape；
3. 能推导主要参数量、FLOPs 和内存量；
4. 能区分 prefill 与 decode；
5. 能解释 KV Cache 为什么成为推理瓶颈；
6. 能独立实现一个最小 Transformer 和 KV Cache。

达到这一层之后，再学习 GQA、MLA、sliding-window attention、MoE 和其他替代架构，才不会变成术语堆积。
