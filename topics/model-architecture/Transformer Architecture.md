---
type: topic
status: developing
area: model-architecture
mastery: explain
aliases:
  - 现代 Transformer 架构
  - Modern Transformer Architecture
topics:
  - "[[Transformer Block]]"
---

# Transformer Architecture

> [!abstract] 核心结论
> 当前生成式 LLM 最常见的基准结构是 Pre-Norm、decoder-only、causal attention、RMSNorm、RoPE、GQA、SwiGLU Transformer。学习时先掌握这一基准，再比较其他架构改变了什么能力或系统瓶颈。

## 问题与适用范围

Transformer 并不只有一种架构。Llama、Qwen 等生成模型大体属于上述范式，但 Transformer 还包括 encoder-only、encoder-decoder，以及大量 attention、FFN、位置编码和残差结构变体。

本笔记负责回答：

1. Transformer 有哪些宏观架构？
2. 现代 decoder-only LLM 的基准结构是什么？
3. 常见变体主要改变哪些计算、内存与通信特征？

单个 block 中 Attention、FFN、Norm、Residual 的数据流、tensor shape 与 FLOPs 推导，见：[Transformer Block](Transformer%20Block.md)。整体学习顺序见：[AI Infra Learning Roadmap](../../roadmaps/AI%20Infra%20Learning%20Roadmap.md)。

## 1. Transformer 的三种宏观架构

| 架构 | Attention | 主要用途 | 代表模型 |
| --- | --- | --- | --- |
| Encoder-only | 双向 attention，无 causal mask | 表征、分类、检索 | BERT |
| Decoder-only | 单向 causal attention | 自回归生成 | GPT、Llama、Qwen |
| Encoder-decoder | Encoder 双向；Decoder causal + cross-attention | 翻译、摘要、条件生成 | 原始 Transformer、T5 |

### 1.1 Encoder-only

每个 token 可以看到整个输入：

```text
token 1: ✓ ✓ ✓ ✓
token 2: ✓ ✓ ✓ ✓
token 3: ✓ ✓ ✓ ✓
token 4: ✓ ✓ ✓ ✓
```

这种结构适合编码完整序列，但不能直接按照自回归方式生成长文本。

### 1.2 Decoder-only

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

### 1.3 Encoder-decoder

Decoder block 通常包含：

```text
Masked Self-Attention
        ↓
Cross-Attention：读取 Encoder 输出
        ↓
FFN
```

《Attention Is All You Need》提出的是面向机器翻译的 encoder-decoder Transformer，并不是今天生成式 LLM 常用的纯 decoder-only 架构。

## 2. 基准 Decoder-only Transformer

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

## 3. Decoder-only 内部的主要变体

### 3.1 Normalization 与 Residual

> [!info] CS336 Lecture 3 视频锚点
> - [视频 07:27：Pre-vs-Post Norm 与 residual path](https://www.youtube.com/watch?v=lVynu4bo1rY&t=447s)
> - [视频 10:45：梯度衰减、梯度尖峰与训练稳定性](https://www.youtube.com/watch?v=lVynu4bo1rY&t=645s)
> - [视频 14:10：LayerNorm 与 RMSNorm 公式](https://www.youtube.com/watch?v=lVynu4bo1rY&t=850s)
> - [视频 14:35：为什么现代模型使用 RMSNorm](https://www.youtube.com/watch?v=lVynu4bo1rY&t=875s)

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

这里的关键变化不只是“把 Norm 往前挪”：

- **Post-Norm**：$x$ 与子层更新相加后还必须经过 Norm，主 residual path 不再是纯 identity；
- **Pre-Norm**：$x$ 可以沿 residual stream 直接传到下一层，Norm 只作用于 Attention/FFN 分支的输入；
- 因而 Pre-Norm 的反向传播包含一条不经过 Norm 和子层的 identity gradient path。CS336 展示的经验结果包括更少的 gradient attenuation 和 gradient spikes；早期强调可减少 warmup，现代大模型更看重训练稳定性以及可使用更大的 learning rate；
- 多层 Pre-Norm block 之后通常仍有一个 final Norm，再接 LM Head。

从 Jacobian 角度看，Pre-Norm 的单层更新为

$$
x_{l+1}=x_l+F_l(\operatorname{Norm}(x_l)),
$$

因此

$$
\frac{\partial x_{l+1}}{\partial x_l}
=
I+J_{F_l}J_{\operatorname{Norm}}.
$$

其中的 $I$ 就是没有被 normalization 截断的 residual 梯度通路。它不能保证训练永不发散，但解释了为什么该公式通常比

$$
x_{l+1}=\operatorname{Norm}(x_l+F_l(x_l))
$$

更适合训练很深的网络。

> [!question] Q-CS336-L03-0727：为什么现代 LLM 的 block 通常写成 Pre-Norm？
> - [x] #question 为什么当前常见公式是 $x_{l+1}=x_l+F(\operatorname{Norm}(x_l))$，而不是把 Norm 放在 residual addition 之后？
> - 来源：[视频 07:27](https://www.youtube.com/watch?v=lVynu4bo1rY&t=447s)、[视频 10:45](https://www.youtube.com/watch?v=lVynu4bo1rY&t=645s)
> - 结论：核心不是 Norm 本身更强，而是它不再位于主 residual signal path 上；identity path 得以保留，梯度传播和大规模训练更稳定。

^q-cs336-l03-0727-prenorm

常见 normalization 包括 LayerNorm 和 RMSNorm。**Pre-Norm/Post-Norm 描述 Norm 的位置；LayerNorm/RMSNorm 描述 Norm 的计算方式，二者是两个独立设计维度。** 现代 LLM 常把它们组合成 Pre-RMSNorm。

### 3.2 Attention Head 组织

| 方法 | Query heads | KV heads | 主要系统影响 |
| --- | ---: | ---: | --- |
| MHA | $H$ | $H$ | 表达能力强，KV Cache 最大 |
| MQA | $H$ | 1 | KV Cache 最小，但共享程度最高 |
| GQA | $H$ | $H_{kv}<H$ | 质量与推理效率之间的折中 |
| MLA | 多头 Query | 压缩/重参数化 KV | 进一步降低 KV 表示和缓存成本 |

比较这些设计时需要考虑：

- KV Cache 容量；
- Decode 阶段的 HBM 读取量；
- 单卡可承载的并发请求数；
- Tensor Parallel 通信；
- 对模型质量的影响。

### 3.3 Attention 范围与替代方案

Transformer 不一定在每一层执行完整的 $L\times L$ attention：

- Full attention；
- Sliding-window attention；
- Local/global hybrid attention；
- Block-sparse attention；
- Linear attention；
- State-space model 与 attention 的混合架构。

完整 attention 表达能力强，但长上下文 prefill 成本高。局部、稀疏或线性化 attention 可以降低成本，但会改变信息传播路径和模型能力。

### 3.4 位置编码

常见位置编码包括：

- Sinusoidal positional encoding；
- Learned absolute position embedding；
- RoPE；
- ALiBi；
- 各种 RoPE scaling 方法。

现代 decoder-only LLM 经常使用 RoPE，但并非所有 Transformer 都如此。

### 3.5 FFN 与 MoE

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

## 4. 系统视角

| 设计 | 主要改变 | 重点观察 |
| --- | --- | --- |
| MQA/GQA/MLA | KV 表示与共享方式 | KV Cache、HBM traffic、并发能力 |
| Sliding/Sparse Attention | Token 间连接范围 | 长上下文计算、信息传播路径 |
| RoPE Scaling | 位置表示 | 长上下文质量与外推能力 |
| MoE | 每个 token 激活的参数子集 | Routing、负载均衡、All-to-all |
| Pre-Norm/RMSNorm | 数值路径与归一化 | 训练稳定性、fusion、内存流量 |

> [!warning] 比较架构时
> “理论 FLOPs 更少”不等于端到端速度更快。还需要检查 tensor shape、算术强度、内存访问、kernel 实现、并行通信和实际 workload。

## 5. 当前理解边界

当前先将标准 decoder-only block 掌握到 `derive`，再独立学习：

- GQA、MLA 与 KV Cache；
- Sliding-window attention；
- MoE；
- FlashAttention；
- 分布式训练与推理并行。

## 待解决问题

- [ ] #question MLA 与 GQA 在 KV Cache 容量、decode 计算和 Tensor Parallel 通信上应如何统一比较？
- [ ] #question 不同 attention 范围如何影响多层网络中的有效感受野？

## 关联内容

- 深入推导：[Transformer Block](Transformer%20Block.md)
- 学习顺序：[AI Infra Learning Roadmap](../../roadmaps/AI%20Infra%20Learning%20Roadmap.md)
- 推理主题：[LLM Inference](../inference/LLM%20Inference.md)
