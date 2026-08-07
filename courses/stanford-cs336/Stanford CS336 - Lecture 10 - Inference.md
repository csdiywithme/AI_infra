---
type: course-note
status: developing
course: "[[Stanford CS336]]"
lecture: 10
lecture_date: 2026-04-29
area: inference
topics:
  - "[[LLM Inference]]"
  - "[[KV Cache]]"
  - "[[Serving Systems]]"
aliases:
  - Stanford CS336 Lecture 10
  - CS336 Inference
video_url: https://www.youtube.com/watch?v=EfM546A79aM
---

# Lecture 10：Inference

> [!abstract] 本讲一句话
> LLM inference 的核心不只是“少算一点”，而是针对 prefill 可并行、decode 严格自回归且常受显存带宽限制的 workload，联合优化 KV cache、模型表示、batch 调度和动态内存管理，同时分别守住 TTFT、单请求延迟与总体吞吐。

## 来源与范围

- 课程：Stanford CS336 — Language Modeling from Scratch, Spring 2026
- 讲师：Percy Liang
- 上课日期：2026-04-29
- [课程视频](https://www.youtube.com/watch?v=EfM546A79aM)，时长 1:25:30
- [课程主页](https://cs336.stanford.edu/)
- [官方 Lecture 10 可执行讲义](https://cs336.stanford.edu/lectures/?trace=lecture_10)
- 延伸教材：[How to Scale Your Model — Inference](https://jax-ml.github.io/scaling-book/inference/)
- 本讲覆盖：prefill/decode、arithmetic intensity、KV cache、latency/throughput、GQA/MLA/CLA/local attention、quantization、pruning/distillation、speculative sampling、continuous batching 和 PagedAttention
- 本讲不展开：tensor/pipeline parallel serving 的完整通信模型、admission control、prefix-aware routing 和多租户隔离

> [!warning] 来源边界
> 正文按公开视频完整英文字幕与官方 2026 可执行讲义交叉核对；论文细节以原论文为边界。下表是字幕轨定位出的真实时间点。讲义中的理想性能模型忽略 kernel、通信、cache hierarchy 和调度开销，正文会明确这些假设；生产 SLO 和容量规划属于笔记拓展。

## 视频索引

| 视频位置 | 内容结构 | 对应笔记 |
| --- | --- | --- |
| [00:00](https://www.youtube.com/watch?v=EfM546A79aM&t=0s) | 推理成本与 agent workload | [[#1. 为什么 inference 是独立问题\|1]] |
| [04:23](https://www.youtube.com/watch?v=EfM546A79aM&t=263s) | vLLM、SGLang、TensorRT-LLM、llama.cpp | [[#1. 为什么 inference 是独立问题\|1]] |
| [05:13](https://www.youtube.com/watch?v=EfM546A79aM&t=313s) | TTFT、ITL、throughput | [[#2. “快”至少有三个指标\|2]] |
| [14:23](https://www.youtube.com/watch?v=EfM546A79aM&t=863s) | Arithmetic intensity | [[#3. 用 arithmetic intensity 分析 inference\|3]] |
| [20:44](https://www.youtube.com/watch?v=EfM546A79aM&t=1244s) | 自回归生成与 KV cache | [[#4. 两阶段 inference\|4]] |
| [35:07](https://www.youtube.com/watch?v=EfM546A79aM&t=2107s) | Llama 2 13B/H100 latency–throughput 算例 | [[#6. Batch size 的双刃剑\|6]] |
| [45:47](https://www.youtube.com/watch?v=EfM546A79aM&t=2747s) | GQA/MLA/CLA/local attention | [[#7. 从架构上缩小 KV cache\|7]] |
| [64:29](https://www.youtube.com/watch?v=EfM546A79aM&t=3869s) | Quantization 与 AWQ | [[#8. 压缩模型与表示\|8]] |
| [67:39](https://www.youtube.com/watch?v=EfM546A79aM&t=4059s) | Pruning 与 distillation | [[#8.2 Pruning + Distillation\|8.2]] |
| [71:43](https://www.youtube.com/watch?v=EfM546A79aM&t=4303s) | Speculative decoding | [[#9. Speculative sampling\|9]] |
| [77:09](https://www.youtube.com/watch?v=EfM546A79aM&t=4629s) | Dynamic requests 与 continuous batching | [[#10. 动态 serving workload\|10]] |
| [79:34](https://www.youtube.com/watch?v=EfM546A79aM&t=4774s) | PagedAttention | [[#10.3 PagedAttention\|10.3]] |
| [83:28](https://www.youtube.com/watch?v=EfM546A79aM&t=5008s) | 其他 kernel/runtime 优化 | [[#11. 一个端到端 serving 心智模型\|11]] |

## 视频补充：课堂中的定量例子与判断

- 课堂以 2026-04-29 的数量级作动机：推理 token 消耗在数天内就可能接近一次大型训练；agent 内部 reasoning/tool loop 又缺少人类阅读速度这一自然上限。重点是生命周期成本，而不是背某个公司的瞬时数字。
- 当时的软件判断是 vLLM 可作为常用默认、SGLang 偏 agentic workload、TensorRT-LLM 更高度特化、llama.cpp 面向 CPU/local；这是课堂时点快照，不是长期排名。
- 如果每生成一个 token 都重算整个 prefix，随生成长度累积会出现近似 $O(T^3)$ 的总工作；KV cache 把重复 prefix computation 换成持续增长的 memory footprint。
- Decode MLP 的 arithmetic intensity 可随并发 batch 增长；decode attention 的强度通常仍小于 1，不能仅靠普通 batching 消除 memory-bound。这解释了为什么优化 KV 表示与调度比“再调大 batch”更根本。
- 课堂 Llama 2 13B/H100 算例用“公交车”类比 batch：载客更多提高总体吞吐，却可能增加单请求等待；TTFT 主要受 prefill/queue 影响，decode 通常需要更大的 batch 才能提高权重复用。
- AWQ 的直觉是保留与少数异常大 activation channels 对应的权重为较高精度，其余权重更激进量化；pruning 后还要用 continued training/distillation “healing”。
- 展示的 speculative decoding 实验中，draft 长度约 3–4 tokens 是 sweet spot；draft 越长并不保证越快。

![PagedAttention 的 prefix sharing 与 block-level copy-on-write](../../assets/courses/stanford-cs336/lecture-10/l10-82m55s-pagedattention.png)

> 视频关键帧：[1:22:55](https://www.youtube.com/watch?v=EfM546A79aM&t=4975s)。Logical KV blocks 映射到 physical blocks；共享 prefix 用 ref count，发生分叉时才 copy-on-write，避免为最大长度 1024 等上限连续预分配整块 KV 空间。

## 1. 为什么 inference 是独立问题

Inference 出现在：

- Chatbot、code completion、agent；
- 离线 batch data processing；
- 模型 evaluation；
- RL 训练中的 rollout；
- 生成 synthetic data。

训练通常只发生一次，inference 会在模型生命周期内重复很多次。因此总成本更接近：

$$
C_{\mathrm{life}}
=
C_{\mathrm{train}}
+
\sum_{r=1}^{R}
C_{\mathrm{infer}}(r)
$$

Agent workload 尤其重要：用户只看到简短结果，内部却可能生成长 reasoning trace、调用工具并反复重试。**Generated tokens 直接对应计算与服务成本。**

训练与推理的 workload 不同：

| 维度 | Training | Autoregressive inference |
| --- | --- | --- |
| Token 可见性 | 整段序列已知 | 未来 token 未知 |
| 序列并行 | 可对所有 token 并行 | decode 必须逐 token |
| Batch | 常规则、固定 | 动态到达、长度不同 |
| 持久状态 | parameters、optimizer、activations | parameters、KV cache |
| 常见瓶颈 | compute、communication | decode memory bandwidth、capacity |

## 2. “快”至少有三个指标

### 2.1 Time to First Token

TTFT 是请求到达后，第一个输出 token 出现前的时间：

$$
\operatorname{TTFT}
=
t_{\mathrm{queue}}
+
t_{\mathrm{prefill}}
+
t_{\mathrm{schedule}}
$$

它影响交互系统的响应感，主要与排队和 prompt prefill 有关。

### 2.2 Inter-token latency

也称 time per output token（TPOT）：

$$
\operatorname{TPOT}
\approx
\frac{t_{\mathrm{decode}}}{T_{\mathrm{output}}}
$$

用户感受到的输出速度常以 seconds/token 或 tokens/second/request 表示。

### 2.3 Throughput

服务整体吞吐：

$$
\operatorname{Throughput}
=
\frac{\text{all generated tokens}}
{\text{wall-clock second}}
$$

它关注多请求资源利用率。

> [!warning] 三者不能互换
> 等更久以凑大 batch 可以提高 throughput，却恶化 TTFT；增加 batch 也可能让每轮 decode 更慢，从而恶化 TPOT。

此外生产系统还关心：

- P50/P95/P99 latency；
- requests/second 与 tokens/second；
- goodput：满足 SLO 的有效吞吐；
- GPU-hour/token、energy/token；
- cache hit rate、preemption 和失败率。

## 3. 用 arithmetic intensity 分析 inference

### 3.1 符号

沿用官方讲义：

| 符号 | 含义 |
| --- | --- |
| $B$ | 同时处理的 sequences |
| $S$ | 已有 prefix/context tokens |
| $T$ | 本次并行计算的 query/output positions |
| $D$ | model dimension |
| $F$ | MLP hidden dimension，常约 $4D$ |
| $N$ | query heads |
| $K$ | KV heads |
| $H$ | head dimension，$D=NH$ |
| $L$ | layers |
| $V$ | vocabulary size |

### 3.2 一个矩阵乘法

对 BF16：

$$
X_{B\times D}
W_{D\times F}
\rightarrow
Y_{B\times F}
$$

FLOPs：

$$
2BDF
$$

理想 HBM bytes：

$$
2BD+2DF+2BF
$$

Arithmetic intensity：

$$
I
=
\frac{2BDF}
{2BD+2DF+2BF}
$$

若 $B\ll D,F$，权重读取占主导：

$$
I
\approx
B
\quad
\text{FLOP/byte}
$$

H100 的课堂示例使用：

$$
\frac{989\ \text{TFLOP/s}}
{3.35\ \text{TB/s}}
\approx
295\ \text{FLOP/byte}
$$

因此小 batch matrix-vector-like workload 远低于硬件 knee，常为 memory-bound。

> [!note] Roofline 只是上界
> 实际性能还受 Tensor Core shape、kernel launch、cache、fusion、quantization kernel、并行通信和 scheduler 影响。

## 4. 两阶段 inference

### 4.1 为什么需要 KV cache

若每生成一个 token 都重新对完整 prefix 做 forward，attention 会重复计算旧 token 的 K/V。

不缓存时，单次长度 $t$ 的 forward attention 约为 $O(t^2)$，连续生成到 $T$ 的总和为：

$$
\sum_{t=1}^{T} O(t^2)
=
O(T^3)
$$

KV cache 保存每层历史 token 的 key/value，使旧位置无需重复投影。

### 4.2 Prefill

Prefill 一次处理 prompt 的多个 token：

- prompt tokens 已知；
- 可在 sequence 维并行；
- 矩阵更大，通常更容易 compute-bound；
- 建立后续 decode 所需的 KV cache；
- 决定 TTFT 的主要计算部分。

### 4.3 Decode

每步只产生一个新 token：

- $T=1$；
- 下一步依赖上一步采样结果；
- 每层需要读权重和历史 KV；
- 小 batch 时权重没有被充分复用；
- 常受 HBM bandwidth 限制。

这不是说 decode “FLOPs 很大”，而是每搬一个 byte 做的 FLOPs 太少。

## 5. MLP 与 Attention 的 inference 账本

### 5.1 Gated MLP

官方讲义按三次 projection 核算：

$$
\text{FLOPs}_{\mathrm{MLP}}
=
6BTDF
$$

BF16 bytes：

$$
\text{Bytes}_{\mathrm{MLP}}
=
4BTD
+
4BTF
+
6DF
$$

若 $BT\ll D,F$，权重读取主导：

$$
I_{\mathrm{MLP}}
\approx
BT
$$

于是：

- Prefill：$BT$ 大，易提高 arithmetic intensity；
- Decode：$T=1$，intensity 约为 $B$，需要并发请求摊薄权重读取。

### 5.2 FlashAttention 视角下的 Attention

考虑 $T$ 个 query positions 读取 $S$ 个历史 K/V：

$$
\text{FLOPs}_{\mathrm{attn}}
=
4BSTD
$$

BF16 bytes：

$$
\text{Bytes}_{\mathrm{attn}}
=
4BSD
+
4BTD
$$

所以：

$$
I_{\mathrm{attn}}
=
\frac{ST}{S+T}
$$

Prefill 取 $T=S$：

$$
I_{\mathrm{prefill,attn}}
=
\frac{S}{2}
$$

Decode 取 $T=1$：

$$
I_{\mathrm{decode,attn}}
=
\frac{S}{S+1}
<
1
$$

Attention decode 几乎是极端 memory-bound。

为什么增加 $B$ 不提升这个理想 intensity？

- MLP 权重由整个 batch 共享；
- 每条 sequence 的 KV cache 不同；
- batch 增加时，KV bytes 与 FLOPs 同比例增长。

## 6. Batch size 的双刃剑

### 6.1 KV cache 大小

每条 sequence、每层、每 token 保存 K 和 V。BF16 下：

$$
M_{\mathrm{KV,seq}}
=
S
\times K
\times H
\times L
\times 2_{\mathrm{K,V}}
\times 2_{\mathrm{bytes}}
$$

Batch $B$：

$$
M_{\mathrm{KV,total}}
=
4BSKHL
$$

注意 cache 容量随：

- batch size 线性增长；
- context length 线性增长；
- layers 线性增长；
- KV heads 和 head dimension 线性增长。

### 6.2 理想 memory-bound 模型

若每 decode step 需要读参数和当前 KV：

$$
t_{\mathrm{step}}
\gtrsim
\frac{M_{\mathrm{params}}+M_{\mathrm{KV}}}
\text{HBM bandwidth}
$$

一次并行生成 $B$ 个 token：

$$
\operatorname{throughput}
\lesssim
\frac{B}{t_{\mathrm{step}}}
$$

增大 batch：

- 好处：参数读取被更多请求复用，throughput 提高；
- 代价：KV cache 增大，step latency 和 capacity 压力增加；
- 最终：KV bytes 主导后，throughput 收益递减。

多复制几份模型可以线性增加 capacity，且不改变单副本 latency；模型切分则需要额外通信。

### 6.3 Prefill 和 decode 不应采用同一策略

- Prefill：小 batch 往往更有利于 TTFT；
- Decode：较大 batch 能摊薄权重读取；
- 长 prompt prefill 可能抢占 decode，引发 head-of-line blocking；
- 实际 scheduler 需要同时满足交互延迟和吞吐 SLO。

## 7. 从架构上缩小 KV cache

### 7.1 GQA / MQA

标准 MHA：

$$
K=N
$$

MQA：

$$
K=1
$$

GQA：

$$
1<K<N
$$

$N$ 个 query heads 共享 $K$ 组 K/V，KV cache 相对 MHA 缩小：

$$
\frac{N}{K}
\text{ times}
$$

这能降低 capacity 和 bandwidth，但需要检查质量变化。

### 7.2 Multi-head Latent Attention

普通 attention 缓存投影后的 K/V。MLA 先将 hidden state 压缩：

$$
c
=
W_c h
$$

缓存低维 $c$，使用时再上投影得到 K/V。若 latent dimension $C\ll NH$，cache 大幅缩小。

RoPE 与这种 factorization 的组合需要额外设计；不能只看压缩维度而忽略位置编码分支。

### 7.3 Cross-layer Attention

CLA 在 layer 之间共享 K/V，类似 GQA 在 head 之间共享。

收益：减少按 $L$ 线性增长的 cache。

代价：层间表示自由度减少，且实现和并行布局会变化。

### 7.4 Local / Sliding-window Attention

只保留最近 $W$ 个 token：

$$
M_{\mathrm{KV}}
=
O(BWLKH)
$$

而非随完整 $S$ 增长。多层堆叠仍能扩大 effective receptive field。

风险是丢失远程信息，因此常将 local 与 global layers 交错。

> [!tip] 架构优化要在训练前决定
> GQA、MLA、CLA 和 local attention 直接改变模型结构。最干净的 recipe 是按目标架构从头训练；若从旧模型改造，则通常需要 weight surgery 和 distillation 修复。

## 8. 压缩模型与表示

### 8.1 Quantization

对 affine integer quantization：

$$
q
=
\operatorname{round}
\left(\frac{x}{s}\right)
+
z
$$

$$
\hat{x}
=
(q-z)s
$$

其中 $s$ 是 scale，$z$ 是 zero point。

更低 bit-width 同时减少：

- parameter memory；
- 权重读取 bytes；
- 可能的 KV cache bytes；
- 某些硬件上的计算成本。

但速度收益要求有匹配的 kernel；若频繁 dequantize 或算子不支持，容量下降不一定转化为 latency 改善。

#### QAT 与 PTQ

QAT：

- 训练 forward 中模拟量化误差；
- 模型可适应误差；
- 需要重新训练，成本高。

PTQ：

- 在训练后用 calibration data 决定 scale；
- 成本低；
- accuracy 对 outlier、group size 和量化对象敏感。

AWQ 的直觉是：大 activation channel 对应的部分权重更重要，应获得更高保护或更合适的缩放。

### 8.2 Pruning + Distillation

课程示例流程：

1. 用 calibration set 估计 layer、head、hidden dimension 的重要性；
2. 删除不重要结构；
3. 用原模型作为 teacher 对 pruned student 做 distillation。

结构化 pruning 更容易获得真实硬件 speedup；非结构化 sparsity 若没有硬件和 kernel 支持，可能只减少理论参数而不变快。

## 9. Speculative sampling

### 9.1 为什么可能加速

Decode 逐 token 生成慢，但 target model 对一段已知候选 token 做验证可以并行。于是：

1. 小 draft model $p$ 快速提出 $k$ 个 tokens；
2. 大 target model $q$ 一次并行计算这些位置的概率；
3. 接受与 $q$ 一致的前缀；
4. 在首次拒绝处做修正采样。

### 9.2 Exactness

对 draft 候选 $x$，接受概率：

$$
a(x)
=
\min\left(1,\frac{q(x)}{p(x)}\right)
$$

若拒绝，从 normalized residual 分布采样：

$$
r(x)
\propto
\max(q(x)-p(x),0)
$$

这个 modified rejection-sampling construction 保证最终 token 的边际分布仍为 $q$，因此在数学算法正确、随机数和数值误差可忽略时，它是 **lossless acceleration**，不是把 draft 输出直接当结果。

### 9.3 何时有效

Speedup 取决于：

- draft/target 成本比；
- acceptance rate；
- speculation length $k$；
- target 验证的并行效率；
- sampling 参数和 batch；
- serving scheduler 是否能容纳额外 draft 工作。

Draft 太弱会频繁拒绝；太强又太贵。Distillation、Medusa、EAGLE 等方向都在提高便宜预测与 target 的一致性。

## 10. 动态 serving workload

### 10.1 为什么 static batching 浪费

线上请求：

- 到达时间不同；
- prompt/output 长度不同；
- 有的共享 system prompt；
- 有的从同一 prompt 采样多个答案。

若等待整个 batch 全部完成：

- 短请求被长请求拖住；
- 已完成 slot 空闲；
- padding 做无效计算；
- 新请求必须等待。

### 10.2 Continuous batching

Iteration-level scheduling 在每个 decode iteration 重组 batch：

```text
执行当前 batch 的一个 token
→ 移除完成请求
→ 接纳新请求
→ 更新 active sequences
→ 执行下一 iteration
```

Selective batching 处理 ragged sequences：

- attention 需要尊重各序列边界；
- non-attention operators 可把所有 active tokens 拼成二维 tensor；
- 用 metadata 描述每条序列的 offsets 和 lengths。

这提高 utilization，却让调度、数据结构和 kernel 更复杂。

### 10.3 PagedAttention

传统方式按最大可能长度为每个请求预留连续 KV 区域，会产生：

- internal fragmentation：请求提前结束，预留未使用；
- external fragmentation：空洞无法被新大块使用。

PagedAttention 借鉴虚拟内存：

- 将每条序列的逻辑 KV 切成固定大小 blocks；
- 物理 blocks 可以不连续；
- block table 完成逻辑到物理映射；
- prefix sharing 时多个序列引用相同 blocks；
- 分叉后用 copy-on-write。

收益：

- 更接近按实际 token 使用量分配；
- 容纳更大 batch；
- 高效共享 system prompt 和多样本 prefix。

代价：

- block lookup 和 metadata；
- attention kernel 必须支持分块读取；
- block size 在碎片率与管理开销之间取舍。

## 11. 一个端到端 serving 心智模型

```text
request queue → admission / scheduler → prefill batch
              → KV block allocator → decode batch
                 ├─ unfinished → 回到 scheduler 组成下一轮 batch
                 └─ finished   → release KV blocks
```

对每个 optimization，先问它改变哪一项：

| 技术 | Compute | Memory capacity | Bandwidth | Runtime |
| --- | --- | --- | --- | --- |
| GQA / MLA / CLA | 可能增加 projection | 降 KV | 降 KV IO | 架构需训练 |
| Quantization | kernel-dependent | 降参数/KV | 降 IO | calibration |
| Pruning | 降 | 降 | 降 | 需修复质量 |
| Speculative sampling | 增加 draft、减少 target steps | 增少量状态 | 更好利用 target pass | 接受率决定收益 |
| Continuous batching | 基本不改单 token 算法 | batch 变大 | 提高权重复用 | scheduler 复杂 |
| PagedAttention | 少量寻址开销 | 降碎片 | 支持共享 | block 管理 |

## 12. 我的推导与易错点

### 12.1 KV cache 与模型权重的 crossover

设 BF16 参数量为 $P$，参数 bytes 为 $2P$。KV 为：

$$
4BSKHL
$$

当：

$$
4BSKHL
\approx
2P
$$

即：

$$
BS
\approx
\frac{P}{2KHL}
$$

KV cache 与权重容量相当。长 context、大 batch 即使模型本身能装下，也可能迅速撞上 cache wall。

### 12.2 Decode throughput 的饱和

理想模型：

$$
\operatorname{TPS}(B)
\approx
\frac{B\cdot BW}
{M_P+B M_{KV,1}}
$$

当 $B$ 很小时，$M_P$ 主导，TPS 随 $B$ 近似线性。

当 $B$ 很大时：

$$
\lim_{B\to\infty}
\operatorname{TPS}(B)
=
\frac{BW}{M_{KV,1}}
$$

这解释了为什么 batch 不会无限提高吞吐。

### 12.3 常见误区

> [!danger] 易错点
> - 把 TTFT、TPOT、end-to-end latency 和 throughput 混成一个“速度”；
> - 认为 prefill 与 decode 有相同瓶颈；
> - 只算 KV 容量，不算每步读取的 bandwidth；
> - 认为 batch 越大越好，忽略 queueing 和 tail latency；
> - 把量化后文件变小直接等同于运行更快；
> - 认为所有 pruning 都能获得 dense-kernel speedup；
> - 把 speculative decoding 当近似采样；正确算法可保持 target 分布；
> - 把 PagedAttention 误解成一种新的 attention 数学公式；它首先是 KV memory-management 方法。

## 13. 本讲结论

1. Prefill 可在 sequence 维并行，decode 必须自回归，二者应该分别建模和调度。
2. KV cache 避免重复计算，但把 decode 转化成 memory-capacity 和 bandwidth 问题。
3. MLP decode 的 arithmetic intensity 约随 batch 增长；attention decode 的理想 intensity 小于 1。
4. 大 batch 提高 throughput，却可能恶化 TTFT、TPOT 和显存占用。
5. GQA、MLA、CLA 与 local attention 从架构上减少 KV cache。
6. Quantization 和 pruning 只有配合硬件、kernel 与质量校准才会产生真实收益。
7. Speculative sampling 利用“并行验证比逐 token 生成更高效”的不对称性，并可保持 target 分布。
8. Continuous batching 和 PagedAttention 让动态、ragged 请求更高效地共享计算和显存。
9. Serving 优化必须以 workload、SLO 和 tail latency 为目标，而非只看理论 FLOPs。

## 14. 自测问题

1. TTFT、TPOT 和 throughput 分别由哪些阶段决定？
2. 为什么没有 KV cache 时，自回归生成的 attention 可能达到 $O(T^3)$？
3. 推导 gated MLP 在 $BT\ll D,F$ 时 arithmetic intensity 约为 $BT$。
4. 为什么 attention decode 的 arithmetic intensity 不随 batch size 提高？
5. 写出 BF16 GQA KV cache 的 shape 与 bytes。
6. 为什么增大 decode batch 最终会出现 throughput 饱和？
7. MHA、GQA、MQA 和 MLA 分别缓存什么？
8. QAT、PTQ、AWQ 的目标和成本有何不同？
9. Speculative sampling 被拒绝时，为什么不能直接从 target $q$ 重新采样？
10. Continuous batching 解决了 static batching 的什么浪费？
11. PagedAttention 的 internal fragmentation、external fragmentation 和 copy-on-write 分别是什么？
12. 如果一个优化降低理论 FLOPs 却增加 all-to-all 或 kernel overhead，应该用什么指标判断是否值得？

## 参考资料

- [Stanford CS336 Spring 2026](https://cs336.stanford.edu/)
- [Official Lecture 10 executable notes](https://cs336.stanford.edu/lectures/?trace=lecture_10)
- [How to Scale Your Model — Inference](https://jax-ml.github.io/scaling-book/inference/)
- [Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [Brandon et al., Reducing Transformer Key-Value Cache Size with Cross-Layer Attention](https://arxiv.org/abs/2405.12981)
- [Leviathan et al., Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)
- [Yu et al., Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Kwon et al., Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Lin et al., AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
