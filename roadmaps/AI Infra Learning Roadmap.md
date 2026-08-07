---
type: roadmap
status: developing
area: ai-infra
aliases:
  - AI Infra 学习路线
topics:
  - "[[Transformer Architecture]]"
  - "[[LLM Inference]]"
---

# AI Infra Learning Roadmap

> [!abstract] 当前策略
> 先把 Llama-style decoder-only Transformer 理解到能够推导、实现和分析性能，再逐步学习 Attention Alternatives、MoE、分布式训练与推理服务。

## 通用分析框架

学习每个模块或系统技术时，依次回答：

| 层次 | 核心问题 |
| --- | --- |
| 语义 | 它解决什么问题，适用边界是什么？ |
| 数学 | 公式和算法是什么？ |
| Shape | 输入、中间状态和输出 tensor 多大？ |
| Compute | 参数量和 FLOPs 是多少？ |
| Memory | 容量、读写量和缓存对象是什么？ |
| Communication | 跨设备传输什么、传输多少？ |
| Runtime | 在具体 workload 和硬件上为什么快或慢？ |
| Evidence | 结论来自课程、论文、代码还是实验？ |

## 阶段 1：Resource Accounting

以 CS336 相关课程为主，掌握：

- Tensor shape；
- Parameter count；
- FLOPs；
- Activation memory；
- Arithmetic intensity；
- GPU compute 与 HBM bandwidth。

目标是看到：

$$
[B,L,d]\times[d,d_{ff}]
$$

就能推导输出 shape、参数量、FLOPs、主要内存量和可能的瓶颈。

## 阶段 2：标准模型架构

推荐顺序：

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

当前材料：

- [Transformer Architecture](../topics/model-architecture/Transformer%20Architecture.md)
- [Transformer Block](../topics/model-architecture/Transformer%20Block.md)

## 阶段 3：实现与验证

至少亲手实现：

1. Single-head causal attention；
2. Multi-head attention；
3. RoPE；
4. RMSNorm；
5. SwiGLU；
6. 完整 Transformer block；
7. 自回归 generation；
8. KV Cache。

对应实验统一记录到 [Experiments](../labs/Experiments.md)。

## 阶段 4：Attention Alternatives 与 MoE

完成标准架构后再学习：

- Sliding-window、local/global 与 block-sparse attention；
- Linear attention 与混合架构；
- MLA；
- MoE routing、load balancing 与 expert parallelism。

> [!question] 比较设计时的核心问题
> 相比标准 Transformer，它改变了哪个计算、内存或通信瓶颈？代价是什么？

## 阶段 5：推理算法与推理系统

课程主线：[CMU 11-763](../courses/cmu-11-763/CMU%2011-763.md)。

重点主题：

- Sampling、search 与 meta-generation；
- Prefill、decode 与 KV Cache；
- Continuous Batching 与 PagedAttention；
- FlashAttention；
- Tensor Parallel；
- Quantization；
- Speculative Decoding；
- Inference-time scaling。

主题入口：[LLM Inference](../topics/inference/LLM%20Inference.md)。

## 掌握程度

主题笔记使用 `mastery` 表示当前能力：

```text
explain → derive → implement → benchmark
```

- `explain`：能够解释目标、机制与适用边界；
- `derive`：能够推导 shape、参数量、FLOPs、内存和通信；
- `implement`：能够独立实现最小正确版本；
- `benchmark`：能够设计 workload、测量指标并解释性能结果。

## 当前重点

1. 从 token ID 追踪到 vocabulary logits；
2. 现场推导 Transformer block 的主要资源开销；
3. 区分 prefill 与 decode 的计算形态；
4. 实现最小 Transformer 和 KV Cache；
5. 建立第一个可复现的性能实验。
