---
type: course-note
status: developing
course: "[[Stanford CS336]]"
lecture: 4
lecture_date: 2026-04-08
area: model-architecture
topics:
  - "[[Transformer Architecture]]"
  - "[[Transformer Block]]"
  - "[[LLM Inference]]"
aliases:
  - Stanford CS336 Lecture 04
  - CS336 Attention Alternatives and Mixture of Experts
video_url: https://www.youtube.com/watch?v=cKSwj_qZ8Jg
---

# Lecture 04：Attention Alternatives and Mixture of Experts

> [!abstract] 本讲一句话
> 这一讲讨论两种 conditional computation：attention alternatives 决定“当前 token 读取哪些历史信息”，MoE 决定“当前 token 激活哪些参数”。它们都试图让模型容量或上下文增长得比每 token 计算更快，但也把瓶颈从纯 FLOPs 转移到了状态压缩、路由、负载均衡、显存和跨设备通信。

## 来源与范围

- 课程：Stanford CS336 — Language Modeling from Scratch, Spring 2026
- 讲师：Tatsunori Hashimoto
- 上课日期：2026-04-08
- [课程视频](https://www.youtube.com/watch?v=cKSwj_qZ8Jg)，时长 1:26:20
- [课程主页](https://cs336.stanford.edu/)
- [官方讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf)，60 页
- 本讲覆盖：linear attention、recurrent/state-space 视角、Mamba-2、Gated DeltaNet、hybrid attention、learned sparse attention、MoE routing、load balancing、expert parallelism、upcycling，以及 DeepSeek V3 的 MoE/MLA/MTP 组合
- 本讲不展开：FlashAttention/kernel 的具体实现、GPU 架构、分布式并行细节和 serving engine；这些会在 Lecture 05–10 继续

> [!warning] 来源边界
> 公式、模型表格和实验观察以官方讲义为准；时间点来自 YouTube 自动章节。视频页面当前显示字幕不可用，因此这不是逐字稿，也不会把无法核对的课堂口头补充写成课程原话。模型效果比较受训练数据、规模和系统实现影响，本笔记不会把单个公开模型的结果解释为普遍定律。

## 视频索引

| 时间 | 视频内容 | 对应笔记 |
| --- | --- | --- |
| [00:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=0s) | Introduction to advanced architectures | [[#1. 本讲的统一问题\|1. 本讲的统一问题]] |
| [01:32](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=92s) | Controlling attention costs | [[#2. Full attention 到底贵在哪里\|2. Full attention 到底贵在哪里]] |
| [05:14](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=314s) | Linear time attention | [[#3. Linear attention\|3. Linear attention]] |
| [11:11](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=671s) | State-based model variants | [[#4. 从线性状态到 Mamba-2 和 Gated DeltaNet\|4. 从线性状态到 Mamba-2 和 Gated DeltaNet]] |
| [22:54](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1374s) | Sparse attention mechanisms | [[#5. Learned sparse attention\|5. Learned sparse attention]] |
| [34:23](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2063s) | Mixture of experts basics | [[#6. MoE 的核心机制\|6. MoE 的核心机制]] |
| [47:30](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2850s) | Routing and design choices | [[#7. Routing 与 expert 设计\|7. Routing 与 expert 设计]] |
| [58:31](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3511s) | Training and system challenges | [[#8. MoE 如何训练\|8. MoE 如何训练]]、[[#9. MoE 的系统账本\|9. MoE 的系统账本]] |

## 1. 本讲的统一问题

Lecture 03 已经介绍了现代 dense Transformer 的常见默认配置，也讨论过
[[Stanford CS336 - Lecture 03 - Architectures and Hyperparameters#7.2 MHA、MQA 与 GQA|GQA/MQA、KV Cache 和 decode arithmetic intensity]]。
这一讲继续追问两个更激进的问题：

1. Context 变长时，是否必须让每个 query 与所有历史 token 做 full attention？
2. 模型参数变多时，是否必须让每个 token 都经过所有参数？

对应两类稀疏性：

| 稀疏对象    | 典型方法                                 | 每个 token 做出的选择   |
| ------- | ------------------------------------ | ---------------- |
| 历史信息/连接 | Linear、local、sparse、hybrid attention | 读取或压缩哪些历史状态      |
| 模型参数    | Mixture of Experts                   | 激活哪些 FFN experts |

它们都在追求：

$$
\text{capacity or context}
\uparrow
\quad\text{faster than}\quad
\text{compute per token}
\uparrow
$$

但省掉 dense compute 后，新的问题会变得突出：

- 固定大小的 state 能否保留需要的历史细节？
- 稀疏索引器本身是否足够便宜、足够准确？
- Expert 是否负载均衡？
- 动态 token routing 能否形成高效的大矩阵？
- 跨设备 All-to-All 是否吃掉稀疏计算的收益？
- “总参数多、激活参数少”应怎样核算显存和推理成本？

> [!tip] 学习主线
> 不要只记住 Linear Attention、Mamba、GDN、DSA、MoE 等名称。每看到一种方法，都用四个问题检查：它压缩了什么？保留了什么？新增了什么状态或路由？最终瓶颈转移到了哪里？

## 2. Full attention 到底贵在哪里

令单个 attention head 的：

$$
Q\in\mathbb R^{n\times d_k},\qquad
K\in\mathbb R^{n\times d_k},\qquad
V\in\mathbb R^{n\times d_v}
$$

忽略 causal mask、scale 和数值稳定细节，普通 attention 可写成：

$$
\operatorname{Attn}(Q,K,V)
=
\rho(QK^\top)V
$$

其中 $\rho$ 表示 softmax 等归一化。

> [!question] 视频中 21 分钟左右提问：为什么这个归一化可以忽略？这是 lossy approximation 吗？
> **回答：**这里的 $\rho$ 不是可学习“参数”，而是 softmax 一类归一化算子。“忽略”有两种完全不同的含义：
>
> 1. **只做复杂度记账时忽略。**scale、causal mask 和逐行 softmax 都处理 $n\times n$ 个 score，复杂度约为 $O(n^2)$；两次矩阵乘法是 $O(n^2d)$。当 $d>1$ 时，前者不是主导项，所以可以在渐进复杂度里略写，但真实模型和 kernel 仍必须计算它们。
> 2. **为利用结合律而令 $\rho=I$。**这会把 softmax attention 换成另一种函数，确实是有损的架构替换，不能保证复现原 attention。普通 softmax 是逐行非线性算子，不能从 $(QK^\top)V$ 中移到右侧。
>
> Kernelized linear attention 不是简单“删掉分母”，而是近似 softmax kernel
> $\exp(q^\top k/\sqrt d)\approx\phi(q)^\top\phi(k)$，并保留相应分母：
> $$
> y_i\approx
> \frac{\phi(q_i)^\top\sum_{j\le i}\phi(k_j)v_j^\top}
> {\phi(q_i)^\top\sum_{j\le i}\phi(k_j)}.
> $$
> 若 $\phi$ 是有限维随机/确定性特征，这是对 softmax kernel 的 lossy approximation；若模型从头使用另一种 feature map 训练，则更准确地说是**学习一种新的 attention 规则**，而不是近似一个已经训练好的 softmax 模型。

### 2.1 计算复杂度

先计算 score matrix：

$$
QK^\top:
\quad
(n\times d_k)(d_k\times n)
\rightarrow
(n\times n)
$$

计算量：

$$
O(n^2d_k)
$$

再计算 attention-weighted values：

$$
A V:
\quad
(n\times n)(n\times d_v)
\rightarrow
(n\times d_v)
$$

计算量：

$$
O(n^2d_v)
$$

因此 attention mixing 部分合计约为：

$$
O\left(n^2(d_k+d_v)\right)
$$

### 2.2 三种成本不要混在一起

| 问题 | Full attention 的典型成本 |
| --- | --- |
| Training/prefill compute | score/value mixing 随 $n^2$ 增长 |
| 中间 attention matrix | 朴素实现是 $O(n^2)$；FlashAttention 可以避免完整写回 HBM，但不会消除二次方计算 |
| Autoregressive decode | 每个新 token 读取 $O(n)$ 历史 K/V；KV Cache 容量也随 $n$ 增长 |

所以“FlashAttention 解决了 attention memory”与“attention 已不是二次复杂度”不是同一件事：

- FlashAttention 主要改变数据搬运和中间 materialization；
- local/sparse attention 改变实际连接数；
- linear/recurrent attention 改变历史信息的表示方式；
- MLA/GQA 主要压缩需要缓存或搬运的状态。

### 2.3 Lecture 03 的 basic toolkit

在更激进的方案之前，课程先提醒我们已有一组稳健工具：

- Sliding-window/local attention：只看最近 $w$ 个 token；
- Full + local interleave：少量 full layers 提供全局通道；
- GQA/MQA：减少 KV heads；
- FlashAttention、KV Cache、量化和系统优化。

它们保留了标准 softmax attention 的大部分结构，风险较低。Lecture 04 关注的是：如果还要
进一步降低长上下文成本，能否改变 attention 的数学形式或使用 learned sparsity？

## 3. Linear attention

### 3.1 最简单的推导：利用矩阵乘法结合律

先假设 $\rho$ 是 identity：

$$
(QK^\top)V=Q(K^\top V)
$$

左结合：

$$
(QK^\top)V
$$

需要：

$$
O(n^2d_k+n^2d_v)
$$

右结合时先计算：

$$
K^\top V:
\quad
(d_k\times n)(n\times d_v)
\rightarrow
(d_k\times d_v)
$$

再计算：

$$
Q(K^\top V):
\quad
(n\times d_k)(d_k\times d_v)
\rightarrow
(n\times d_v)
$$

两步合计：

$$
O(2nd_kd_v)
$$

当 $d_k,d_v\ll n$ 时，复杂度从关于 $n$ 的二次方降为一次方，并且不需要生成
$n\times n$ 的 attention matrix。

> [!warning] 结合律不能直接穿过 softmax
> 普通 softmax attention 不是 $QK^\top V$，而是
> $\operatorname{softmax}(QK^\top)V$。softmax 是非线性操作，所以不能简单写成
> $Q\operatorname{softmax}(K^\top V)$。课程先用 identity case 展示代数核心，实际 linear attention 通常需要 kernel feature map 或改用其他归一化。

### 3.2 Kernelized linear attention

如果能找到 feature map $\phi$，使 attention kernel 近似满足：

$$
\kappa(q,k)\approx\phi(q)^\top\phi(k)
$$

则 causal attention 可以写成：

$$
y_t
=
\frac{
\phi(q_t)^\top
\left(
\sum_{j\le t}\phi(k_j)v_j^\top
\right)
}{
\phi(q_t)^\top
\left(
\sum_{j\le t}\phi(k_j)
\right)
}
$$

定义两个 prefix states：

$$
S_t
=
S_{t-1}+\phi(k_t)v_t^\top
$$

$$
z_t
=
z_{t-1}+\phi(k_t)
$$

便得到：

$$
y_t
=
\frac{\phi(q_t)^\top S_t}
{\phi(q_t)^\top z_t}
$$

讲义为突出主线，省略归一化 state $z_t$，写成纯线性形式：

$$
S_t=S_{t-1}+k_tv_t^\top,
\qquad
y_t=q_t^\top S_t
$$

### 3.3 为什么它既像 attention，又像 RNN

从 attention 视角：

- $k_tv_t^\top$ 把当前位置的 key-value association 写入 state；
- $q_t$ 用当前 query 从 state 中读出结果。

从 RNN 视角：

- $S_t$ 是一个随序列递推的 hidden state；
- 推理只需要上一步 $S_{t-1}$，不必保留所有历史 K/V；
- state shape 与序列长度无关。

对于纯线性形式：

| 项目 | Full causal attention | Linear recurrent form |
| --- | --- | --- |
| 历史状态 | $O(n(d_k+d_v))$ KV | $O(d_kd_v)$ matrix state |
| 第 $t$ 步读取 | $O(t(d_k+d_v))$ | $O(d_kd_v)$ |
| 能否逐 token recurrent update | 需要 KV Cache | 可以 |
| 历史信息 | 保留每个 token 的 K/V | 压缩进固定大小 state |

这里最重要的 tradeoff 是：

> Full attention 保留历史 token 的细粒度地址；linear recurrent model 把历史压缩进固定大小的 associative memory。

固定 state 带来成本上限，也带来信息瓶颈。两个不同历史如果被压缩成相同 $S_t$，后续 query
就无法再区分它们。

### 3.4 Parallel–recurrent duality

同一个模型可以有两种等价或近似等价的执行形式：

- 训练/prefill：使用 matrix/chunk/parallel-scan form，一次处理许多 token；
- decode：使用 recurrent form，只维护 state 并逐 token 更新。

这叫 duality。它不是说训练一定要物化 $n^2$ attention matrix，而是说模型的 operator
既可以表达成便于 GPU 并行的 block computation，也可以表达成便于 decode 的 recurrence。

这一性质非常重要，因为：

- 纯 RNN recurrence 的序列依赖难以并行训练；
- 纯 attention form 在 decode 时又需要不断读取历史；
- dual form 试图同时获得 training parallelism 与 recurrent decoding。

### 3.5 为什么近期模型常用 hybrid

纯 linear state 对精确 retrieval、copying 和复杂长距离关联可能不足。一个常见折中是：

```text
Linear → Linear → Linear → Linear → Linear → Linear → Linear → Full → ...
```

课程列举的公开趋势包括：

| 模型 | 课程中的简化描述 | 设计直觉 |
| --- | --- | --- |
| MiniMax M1 / MiniMax-Text-01 | 约 7 个 linear layers 配 1 个 full-attention layer | 主要用 linear 控制长上下文成本，周期性 full attention 恢复精细全局访问 |
| Nemotron 3 | 约 3:1 的 Mamba/attention hybrid | recurrent state 提供高效推理，attention 层弥补状态压缩 |
| Qwen 3.5 / Qwen Next | 约 3:1 的 Gated DeltaNet/attention hybrid | 将可更新 state 与标准 attention 组合 |

课程也强调：公开模型的 end-to-end 结果很多，但严格控制其他变量的 ablation 仍不充分。
“Hybrid 模型表现好”不能自动证明其中某个 layer ratio 普遍最优。

## 4. 从线性状态到 Mamba-2 和 Gated DeltaNet

纯累加 state：

$$
S_t=S_{t-1}+k_tv_t^\top
$$

存在一个明显问题：旧信息只能不断累积，模型缺少主动遗忘或覆盖的机制。

### 4.1 Mamba-2：给旧状态加 data-dependent decay

课程用下面的简化形式连接 linear attention 与 Mamba-2：

$$
S_t
=
\gamma_t S_{t-1}
+k_tv_t^\top
$$

$$
y_t
=
q_t^\top S_t+v_t^\top D,
\qquad
\gamma_t=f(x_t)
$$

其中：

- $\gamma_t$ 控制旧 state 保留多少；
- $\gamma_t$ 小时，模型快速遗忘历史；
- $\gamma_t$ 接近 1 时，历史保留更久；
- $D$ 是直接从当前输入/值到输出的 skip term。

真实 Mamba-2/SSM 参数化和高效算法更复杂；课程这里的重点是 mechanics：

> 给线性 state 加上由当前输入决定的 gate，可以让有效 memory timescale 随 token 变化。

Mamba-2 的 State Space Duality 进一步说明，一类 structured state-space models 与
structured attention matrices 可以在统一框架中理解，并可使用 block/parallel algorithms
高效训练。

### 4.2 Gated DeltaNet：不仅遗忘，还定向覆盖

课程给出的简化更新为：

$$
S_t
=
\gamma_t
\left(I-\beta_tk_tk_t^\top\right)S_{t-1}
+\beta_tk_tv_t^\top
$$

$$
y_t=q_t^\top S_t,
\qquad
\gamma_t=f_\gamma(x_t),
\quad
\beta_t=f_\beta(x_t)
$$

分三步理解：

1. **Global decay**

$$
\gamma_t S_{t-1}
$$

控制旧状态整体保留多少。

2. **Selective erase**

$$
\left(I-\beta_tk_tk_t^\top\right)S_{t-1}
$$

$k_tk_t^\top$ 取出 state 在当前 key 方向上的分量，然后按 $\beta_t$ 擦除。若 $k_t$
已归一化且 $\beta_t=1$，这一项近似把当前 key 所指向的旧 association 清掉。

3. **Write**

$$
\beta_tk_tv_t^\top
$$

把新的 key-value association 写回同一方向。

因此 delta rule 不只是“继续累加”，而更接近：

> 先删除当前地址上的旧值，再写入新值。

$\beta_t=0$ 时既不 erase 也不 write，相当于一个 “no input operation” gate。课程把这类
机制与 fast-weight programming、test-time state update 联系起来。

### 4.3 Linear、Mamba-2 与 GDN 的关系

| 方法 | State update 的主要能力 | 主要信息瓶颈 |
| --- | --- | --- |
| Linear attention | 累加 key-value associations | 旧信息不能主动删除，state 容量固定 |
| Mamba-2 简化视角 | 根据输入整体衰减旧 state | 不同地址之间仍可能发生干扰 |
| Gated DeltaNet | 整体衰减 + key-directed erase/write | 定向更新仍受 key 表示和 state rank 限制 |

这些方法不是普通 softmax attention 的无损加速版，而是改变了模型表达历史的方式。

## 5. Learned sparse attention

Linear/state methods 把全部历史压缩进固定 state。另一条路线是继续保留 token-level KV，
但只读取少量相关位置。

### 5.1 基本形式

对 query $q_t$，先由轻量 indexer 给历史位置打分：

$$
a_{t,j}
=
\operatorname{Indexer}(q_t,k_j)
$$

选出 $k$ 个候选位置：

$$
\mathcal I_t
=
\operatorname{TopK}_{j\le t}(a_{t,j},k)
$$

再只在这些位置上计算精确 attention：

$$
y_t
=
\sum_{j\in\mathcal I_t}
\operatorname{softmax}_{j\in\mathcal I_t}
\left(
\frac{q_t^\top k_j}{\sqrt{d_k}}
\right)v_j
$$

如果 $k\ll n$ 且 indexer 足够便宜，精确 attention 部分可从：

$$
O(n^2d)
$$

降到近似：

$$
O(nkd)
$$

### 5.2 真正的难点在 indexer

如果为了找 Top-$k$ 仍对所有 query-key pairs 做完整高维打分，索引器本身还是
$O(n^2d)$，就失去了主要收益。因此 learned sparse attention 必须同时解决：

- 用低维或结构化表示廉价地产生候选；
- Top-$k$ selection 的 kernel 与数据布局；
- 候选 recall：不要漏掉真正重要的远距离 token；
- causal mask、variable length 和 batch 的实现；
- 稀疏 pattern 变化带来的 irregular memory access。

### 5.3 DSA 的课程定位

课程以 DeepSeek Sparse Attention（DSA）为代表：

- 使用轻量 learned indexer 选择少数历史位置；
- 对选中位置执行更精确的 attention；
- 可在 dense short-context pretraining 后做 post-hoc sparse adaptation；
- DeepSeek V3.2、GLM-5 等公开模型采用了这一方向。

> [!note] 拓展：DeepSeek 的 MoE 架构演进
> DeepSeek 的早期工作不是只把 dense FFN 换成更多 experts，而是在三个层面持续探索：
>
> 1. **专家粒度：**DeepSeekMoE 把较大的 expert 拆成更多细粒度 experts，并相应增加每 token 的激活数；在 active compute 近似不变时，token 可组合出更丰富的专家集合。见 [[#7.4 Fine-grained experts|Fine-grained experts]]。
> 2. **知识分工：**增加始终激活的 shared experts，让公共知识走共享路径，减少 routed experts 重复学习相同能力。见 [[#7.5 Shared experts|Shared experts]]。
> 3. **负载与拓扑：**DeepSeek-V2/V3 继续加入 device/node-aware routing、受限路由范围和在线 expert bias。V3 用 bias 调节“谁被选中”，避免强 balancing loss 直接扭曲输出 gate，但仍保留较弱的 sequence-wise auxiliary loss。见 [[#8.3 Per-device balance|Per-device balance]]、[[#8.4 DeepSeek V3 的 per-expert bias|Per-expert bias]]。
>
> 后续若单独成文，可以按“专家专业化 → shared/routed 分工 → load balance → topology-aware dispatch → dense-to-MoE upcycling”组织，而不是只罗列各代模型配置。原始论文见 [DeepSeekMoE](https://arxiv.org/abs/2401.06066)、[DeepSeek-V2](https://arxiv.org/abs/2405.04434) 与 [DeepSeek-V3](https://arxiv.org/abs/2412.19437)。

这条路线与 linear attention 的区别是：

| Linear/state | Learned sparse |
| --- | --- |
| 历史被压缩到固定 state | 历史 token-level KV 通常仍保留 |
| 每步读取固定大小 state | 每步读取选中的 $k$ 个位置 |
| 风险是 state compression | 风险是 indexer 漏检 |
| 访问模式较规则 | Top-$k$ gather 更不规则 |

> [!warning] Sparse compute 不一定等于 sparse KV memory
> 如果系统仍保存所有历史 K/V，只是每步少读一部分，attention compute 和 HBM traffic 可以下降，但 KV Cache capacity 仍可能是 $O(n)$. 是否同时压缩 cache，要看具体架构和实现。

## 6. MoE 的核心机制

### 6.1 从一个 FFN 变成多个 experts

Dense Transformer 的每个 token 都经过同一个 FFN：

$$
y_t=x_t+\operatorname{FFN}(x_t)
$$

MoE 把这一层替换成 $E$ 个 FFN experts 和一个 router：

$$
r_t=x_tW_r\in\mathbb R^E
$$

$$
p_t=\operatorname{softmax}(r_t)
$$

选择 Top-$K$ experts：

$$
\mathcal E_t
=
\operatorname{TopK}(p_t,K)
$$

输出：

$$
y_t
=
x_t
+
\sum_{i\in\mathcal E_t}
\widetilde p_{t,i}\operatorname{FFN}_i(x_t)
$$

$\widetilde p$ 表示选中 experts 的 gate，部分实现会在 Top-$K$ 后重新归一化。

典型结构是只替换 MLP/FFN，而保留 attention 为 dense 或其他共享结构。把 attention heads
本身做成 MoE 也存在，但远不如 FFN MoE 常见。

### 6.2 为什么可以增加参数而不同比例增加 FLOPs

令每个 expert 有 $N_e$ 个参数，其余共享模型有 $N_s$ 个参数：

$$
N_{\mathrm{total}}
\approx
N_s+E N_e
$$

每个 token 只走 $K$ 个 routed experts：

$$
N_{\mathrm{active/token}}
\approx
N_s+K N_e
$$

当 $K$ 和 expert size 固定、只增加 $E$ 时：

- 总参数容量增加；
- 每个 token 的 expert FLOPs 近似不变；
- 每个 token 可以根据内容选择不同参数。

这就是 MoE 的核心价值：**把 parameter count 与 active compute 部分解耦**。

> [!warning] “FLOPs 不变”不等于“成本不变”
> 总权重显存、checkpoint、optimizer state、跨设备 dispatch、router、负载不均和小矩阵效率仍会随 expert 数量或分布变化。MoE 省的是 active arithmetic，不是所有系统资源。

### 6.3 为什么 MoE 近期越来越常见

课程总结的经验动机：

- 固定 active FLOPs 时，更多总参数往往带来更低 loss；
- 在达到相近质量时，MoE 可能比 dense 模型训练得更快；
- 多个 experts 天然可以分布到不同设备；
- 近年的 kernel、expert parallel library 和网络硬件让实现更成熟；
- 越来越多高性能公开模型验证了这条路线。

但它长期没有完全取代 dense 模型，是因为：

- 多节点优势更明显，单设备或小规模场景不一定划算；
- routing objective 带有大量 heuristic；
- 训练稳定性、load balance 和可复现性更复杂；
- inference 必须加载大量 total weights，即使每个 token 只用少量 experts；
- All-to-All communication 可能成为主要瓶颈。

## 7. Routing 与 expert 设计

课程把 MoE 设计拆成三类变量：

1. routing function；
2. expert size、数量和 active 数；
3. training objective。

### 7.1 三类 routing

| Routing | 谁做选择 | 优点 | 主要问题 |
| --- | --- | --- | --- |
| Token-choice Top-$K$ | 每个 token 选择 experts | 简单、主流、每 token active compute 明确 | experts 可能严重不均衡 |
| Expert-choice | 每个 expert 选择 tokens | 容易固定每个 expert 的 quota | token 可能被选零次或多次，语义与 serving 更复杂 |
| Global assignment | 全局解 matching/transport | 可同时优化匹配质量与负载 | 求解和同步成本高 |

公开 LLM 绝大多数仍使用 token-choice Top-$K$。历史上也探索过 hashing、RL routing 和
linear assignment，但没有成为主流默认值。

### 7.2 Top-$K$ 的几个细节

Top-$K$ routing 至少有三种容易混淆的步骤：

1. 计算 raw router logits；
2. 用 softmax 或 sigmoid 得到 affinity；
3. 选择 Top-$K$，并决定是否对选中的 weights 再归一化。

常见变体包括：

- 先对所有 experts softmax，再取 Top-$K$；
- 先按 raw/sigmoid score 取 Top-$K$，再只在选中集合内归一化；
- selection score 加 load-balancing bias，但用于加权输出的 gate 不一定包含该 bias。

这些做法会改变：

- experts 之间的竞争方式；
- 未选中 experts 是否收到梯度；
- router score 的数值尺度；
- load-balancing bias 是否直接污染模型输出。

### 7.3 Top-$K$ 不是可微的

在选中集合固定的局部区域内，loss 可以对选中 experts 的 gate 和参数反向传播；但
Top-$K$ 集合在跨过排序边界时会离散跳变：

$$
\frac{\partial\,\operatorname{TopK}}{\partial r}
$$

不能像普通连续函数一样稳定求导。直接后果是：

- 未被选中的 experts 通常收不到该 token 的 expert gradient；
- router 容易早期锁定少数 experts；
- 很难直接用任务 loss 学到兼顾质量和系统平衡的全局路由。

课程回顾了三类处理方法：

| 方法 | 思路 | 实际地位 |
| --- | --- | --- |
| REINFORCE/RL | 把 routing 当离散 policy | 原理直接，但 gradient variance 和系统复杂度高 |
| Stochastic perturbation | 给 router logits 加 Gaussian noise 或 multiplicative jitter | 增强探索与鲁棒性，但增加噪声 |
| Heuristic auxiliary loss | 用连续 router probability 约束负载 | 最常见的工程方案 |

### 7.4 Fine-grained experts

假设原本有 $E$ 个较大的 experts、每 token 选 $K$ 个。可以把每个 expert 沿 FFN
intermediate dimension 拆成 $m$ 个小 experts，同时每 token 选约 $mK$ 个：

$$
E\rightarrow mE,
\qquad
K\rightarrow mK
$$

如果每个小 expert 的宽度约为原来的 $1/m$，active compute 可以近似不变，但组合数增大：

- token 可以组合更细粒度的能力；
- 小 expert 更容易专业化；
- router 和 dispatch 数量增加；
- 单个 GEMM 更小，硬件利用率和通信可能变差。

课程中的公开 ablation 对这一点相对正面：DeepSeekMoE 和 OLMoE 都观察到 fine-grained
experts 的收益。

### 7.5 Shared experts

Shared experts 对所有 token 都激活，routed experts 仍由 Top-$K$ 选择：

$$
y_t
=
x_t
+
\sum_{i\in\mathcal E_t}g_{t,i}E_i(x_t)
+
\sum_{j=1}^{S}E_j^{\mathrm{shared}}(x_t)
$$

设计直觉：

- shared experts 学习所有 token 都需要的公共知识；
- routed experts 减少重复学习公共模式，专注于差异化能力；
- 即使 router 暂时不理想，也有稳定的共享计算路径。

代价是 shared experts 始终增加 active FLOPs。课程也展示了相互不完全一致的 ablation：

- DeepSeek 的实验支持 shared experts；
- OLMoE 的实验没有观察到明显收益。

因此 shared expert 是一个可实验的架构选择，不是已经完全收敛的定律。

### 7.6 代表性配置

下面按课程讲义的统一口径摘录几个配置，目的是建立数量感：

| 模型 | Routed experts | 每 token routed active | Shared | 备注 |
| --- | ---: | ---: | ---: | --- |
| Switch Transformer | 64 | 1 | 0 | 极简 Top-1 |
| Mixtral | 8 | 2 | 0 | 较粗粒度 experts |
| DBRX | 16 | 4 | 0 | Top-4 |
| DeepSeek V3 | 256 | 8 | 1 | Fine-grained + load-balancing bias |
| OLMoE | 64 | 8 | 0 | Fine-grained |
| Llama 4 Maverick | 128 | 1 | 1 | Top-1 routed + shared |

不要只用 “total/active parameter ratio” 判断好坏；还要同时比较 expert width、layer 数、
shared experts、attention 架构、训练 tokens 和硬件实现。

## 8. MoE 如何训练

### 8.1 Load balancing 为什么既是模型问题，也是系统问题

如果大量 token 都选同一个 expert：

- 这个 expert 的设备成为 straggler；
- 其他 expert/device 空闲；
- capacity overflow 可能导致 token dropping；
- 少数 experts 过度训练，其他 experts 学不到东西；
- router 进一步偏向已有强 experts，形成正反馈。

因此系统希望每个 expert 接收近似相同数量的 token，但模型希望每个 token 去最合适的
expert。这两个目标不完全一致。

### 8.2 Switch Transformer 的 balancing loss

对一个 batch 中的 $T$ 个 tokens 和 $E$ 个 experts，定义：

$$
f_i
=
\frac1T
\sum_{x\in\mathcal B}
\mathbf 1[\arg\max p(x)=i]
$$

$f_i$ 是真正 dispatch 到 expert $i$ 的 token fraction。再定义：

$$
P_i
=
\frac1T
\sum_{x\in\mathcal B}p_i(x)
$$

$P_i$ 是 router 分给 expert $i$ 的平均 probability。辅助 loss：

$$
\mathcal L_{\mathrm{balance}}
=
\alpha E\sum_{i=1}^{E}f_iP_i
$$

把 discrete 的 $f_i$ 当常数，关于某个 $p_i(x)$ 的梯度近似为：

$$
\frac{\partial\mathcal L_{\mathrm{balance}}}
{\partial p_i(x)}
=
\frac{\alpha E}{T}f_i
$$

使用越频繁的 expert，$f_i$ 越大，收到的向下压力越强；较空闲的 experts 相对更容易
获得 token。

这个 loss 是 heuristic：

- 它间接优化离散 assignment，而不是直接对吞吐求导；
- $\alpha$ 太小无法平衡，太大会强迫 router 牺牲语义匹配；
- batch/sequence 粒度会影响观察到的负载；
- Top-$K$ 时需要按 dispatch fraction 调整定义。

### 8.3 Per-device balance

Expert-level balance 不一定等于 communication balance。假设两个高流量 experts 恰好在
同一张 GPU，即使其他 experts 较均匀，那张 GPU 仍会过载。

因此可以对 device 聚合：

$$
f_d
=
\sum_{i\in\mathcal E(d)}f_i,
\qquad
P_d
=
\sum_{i\in\mathcal E(d)}P_i
$$

并增加 device-level balance objective。DeepSeek V1/V2 的路线不仅考虑 expert balance，
也逐步加入 communication/device-aware routing。

### 8.4 DeepSeek V3 的 per-expert bias

DeepSeek V3 为每个 expert 维护 routing bias $b_i$，selection 使用：

$$
\mathcal E_t
=
\operatorname{TopK}_i(s_{t,i}+b_i,K)
$$

训练过程中根据观测负载在线更新：

- expert 过载：降低 $b_i$；
- expert 欠载：提高 $b_i$。

关键思想是让 load balancing 通过独立 bias 改变**谁被选中**，而尽量不让强 auxiliary
loss 直接扭曲模型用于组合输出的 affinity。

课程提醒，“auxiliary-loss-free balancing”这个名称并不表示训练中完全没有 auxiliary
loss：DeepSeek V3 仍保留较弱的 sequence-wise balance loss，只是主要全局平衡机制改为
online bias。

### 8.5 Router exploration

早期 MoE 工作会给 router 加噪声：

$$
\widetilde r_{t,i}
=
r_{t,i}+\epsilon_{t,i}
$$

或使用 multiplicative jitter：

$$
\widetilde x_t
=
x_t\odot u_t
$$

其目的不是“数据增强”，而是防止路由边界过早变得脆弱，让 experts 在训练早期接触更
多样的 tokens。噪声也会增加方差，因此后续工作有时会移除或减弱它。

## 9. MoE 的系统账本

### 9.1 Expert parallelism 的执行流程

如果每个 device 放一部分 experts，一个 MoE layer 通常经历：

```text
hidden states
    ↓
router + Top-K
    ↓
pack / permute by destination expert
    ↓
All-to-All dispatch
    ↓
local expert GEMMs
    ↓
All-to-All combine
    ↓
unpermute + weighted sum
```

这提供了一种新的 parallel axis：expert parallelism。增加 experts 时，可以把新增参数
放到更多 devices，而不让单个 token 经过全部 experts。

### 9.2 通信量的第一层估算

令：

- $T$：本层处理的 tokens；
- $K$：每个 token 激活的 routed experts；
- $d$：传给 expert 的 hidden width；
- $s$：每个 element 的 bytes。

忽略本地 expert、metadata 和网络拓扑，dispatch payload 约为：

$$
T K d s
$$

expert output 返回原 device 还有一次近似同量通信，所以：

$$
\text{MoE communication bytes}
\approx
2TKds
$$

这说明：

- Top-$K$ 越大，communication 近似线性增加；
- hidden width 越大，All-to-All payload 越大；
- fine-grained experts 即使保持 active FFN FLOPs，也可能增加 dispatch fan-out；
- 跨节点带宽和 tail latency 会直接限制收益。

### 9.3 为什么 expert GEMM 容易低效

Dense FFN 把大量 tokens 放进一个规则的大 GEMM。MoE routing 后，每个 expert 收到的
token 数不同：

```text
expert 0: 192 tokens
expert 1: 17 tokens
expert 2: 0 tokens
expert 3: 81 tokens
...
```

直接逐 expert 启动 GEMM 会产生：

- 很多 skinny/small GEMMs；
- padding 到统一 capacity 的浪费；
- 动态 shape 和大量 kernel launch；
- 最慢 expert 决定整个 layer 完成时间。

### 9.4 Capacity、padding 与 token dropping

传统实现常给每个 expert 分配容量：

$$
C
\approx
\left\lceil
\text{capacity factor}
\cdot
\frac{TK}{E}
\right\rceil
$$

- 容量大：少 drop tokens，但 padding 和显存浪费更多；
- 容量小：效率高，但 overflow tokens 需要丢弃或 reroute；
- token dropping 会改变模型函数，而且依赖同 batch 中其他 token 的路由。

这也是 MoE 可能产生额外随机性的原因：同一个请求与不同请求一起 batch 时，expert
capacity 占用不同，它的 token 是否 overflow 也可能不同。

现代 drop-free 或动态 block-sparse 实现可以减轻这一问题，但 irregular workload
并不会自动消失。

### 9.5 MegaBlocks 的思路

MegaBlocks 不把所有 experts 强行 pad 成同样大的 dense batch，而把 routing 后的运算
组织为 block-sparse matrix multiplication：

- 只为实际存在的 token-expert blocks 做计算；
- 避免 token dropping；
- 减少 padding；
- 用专门 sparse kernels 保持 GPU 利用率。

它解决的是 “数学上 sparse，但硬件只擅长 dense regular GEMM” 之间的表示鸿沟。

### 9.6 降维后再 dispatch

课程还展示了 Nemotron 3 的方向：在跨设备发送前先把 activation 从 $d$ 下投影到
$d_c<d$：

$$
x_t^{(c)}
=
x_tW_{\mathrm{down}}
$$

通信量由：

$$
O(TKds)
$$

降为：

$$
O(TKd_cs)
$$

expert 侧再在压缩空间计算或恢复。这是典型的 communication–compute–quality tradeoff：

- 好处：All-to-All bytes 更少；
- 成本：增加 projection compute；
- 风险：低维 bottleneck 可能损失信息；
- 是否值得取决于网络带宽是否真是 bottleneck。

## 10. MoE 的稳定性、微调与 upcycling

### 10.1 Router 为什么容易出现数值问题

Router softmax 的 expert 数可能很多，且离散 Top-$K$ 会放大 score 排名差异：

- logits 过大使路由过早接近 one-hot；
- 小数值扰动可能改变 Top-$K$ 集合；
- 某些 experts 被长期饿死；
- 低精度 overflow/rounding 会让 routing 不稳定。

常见工程做法：

- router projection 和 softmax 使用 FP32；
- expert FFN 仍使用 BF16/FP8 等高吞吐 dtype；
- 监控 router logits、entropy、expert counts 和 per-device load；
- 必要时加入 router z-loss。

### 10.2 Router z-loss

对 router logits $r(x)\in\mathbb R^E$：

$$
\mathcal L_z^{\mathrm{router}}
=
\frac1{|\mathcal B|}
\sum_{x\in\mathcal B}
\left(
\log\sum_{i=1}^{E}\exp r_i(x)
\right)^2
$$

它与 Lecture 03 的 output z-loss 使用同一思想：主 routing objective 对某些整体 logit
漂移约束不足，z-loss 把 log-normalizer 锚在较安全范围。

它不能直接解决 load imbalance，但可以降低 router logits 无约束增长带来的训练脆弱性。

### 10.3 微调为什么可能更容易过拟合

大 total-parameter MoE 在小 SFT dataset 上会遇到：

- 每个 expert 实际看到的 tokens 更少；
- 某些 task/domain 只更新少数 experts；
- router 可能快速重排，破坏预训练 specialization；
- 总容量相对数据量过大。

课程列举的应对方向：

- 只微调非-MoE/shared dense layers；
- 冻结或降低 router/expert learning rate；
- 使用更大、更丰富的 SFT data；
- 监控 expert usage 是否在微调时突然塌缩。

没有单一策略对所有模型都最优；关键是区分“模型能力不足”与“每个 expert 获得的数据不足”。

### 10.4 Upcycling：从 dense checkpoint 初始化 MoE

训练大 dense model 已经花费大量 compute。Upcycling 将一个 dense FFN 复制或拆分成多个
experts，再继续预训练：

```text
Dense FFN checkpoint
       ↓ replicate / split
Expert 1  Expert 2  ...  Expert E
       ↓ initialize router
continued pretraining
       ↓
experts gradually specialize
```

如果所有 experts 初始完全相同，且选中 gates 归一化到和为 1，MoE layer 初始可以近似
保持 dense FFN 的函数：

$$
\sum_{i\in\mathcal E_t}
\widetilde p_{t,i}\operatorname{FFN}(x_t)
=
\operatorname{FFN}(x_t)
$$

优势：

- 复用 dense pretraining 的 sunk cost；
- 起始 loss 明显低于随机初始化大 MoE；
- 可以从已有 model family 平滑扩容。

挑战：

- 完全复制导致 experts 对称，specialization 可能很慢；
- router 随机性和 load balance 会影响谁先分化；
- 长期从 scratch 训练的 MoE 有时仍可能赶上或超过简单 upcycling；
- partial re-initialization、expert splitting 和 router warmup 都是可调设计。

课程列举 MiniCPM、Qwen MoE 作为成功案例。

## 11. DeepSeek MoE V1 → V2 → V3

课程最后用 DeepSeek 展示多个 MoE 技巧如何逐代叠加。

| 版本 | Total / active | Experts | Routing 与 balance |
| --- | --- | --- | --- |
| DeepSeekMoE V1 | 16B / 2.8B | 64 routed、6 active、2 shared | Standard Top-$K$；expert + device auxiliary balance |
| DeepSeek V2 | 236B / 21B | 160 routed、6 active、2 shared | 加强 communication balance；限制到 Top-$M$ devices |
| DeepSeek V3 | 671B / 37B | 256 routed、8 active、1 shared | Per-expert bias 主导 load balance；保留较弱 sequence-wise auxiliary loss |

数字用于建立架构数量感；不同报告对 embedding、MTP module、shared experts 是否计入
active parameters 的口径可能略有差异。

### 11.1 V1：fine-grained + shared experts

核心思路：

- 将大 experts 拆成更多小 experts；
- 每个 token 组合多个 routed experts；
- 额外保留 always-on shared experts；
- 用 expert-level 与 device-level auxiliary loss 保持平衡。

### 11.2 V2：把通信约束写入 routing

Expert parallelism 中，token 若被路由到许多不同 devices，通信 fan-out 很大。V2
进一步考虑：

- 哪些 devices 接收 token；
- device 输入和输出是否平衡；
- 只在少量候选 devices 内选 experts，即 Top-$M$ device routing。

这说明 router 不只是在学习语义 specialization，也在参与 placement-aware system
optimization。

### 11.3 V3：减轻 auxiliary loss 对模型的干扰

V3 的方向：

- 使用 per-expert bias 在线调负载；
- selection 与输出 affinity 尽可能解耦；
- 只保留较弱的 sequence-level balance regularizer；
- 在大规模训练中避免强 auxiliary loss 持续拉扯主语言模型目标。

课程对 “aux-loss-free” 的态度是准确理解其边界：它主要移除了强 expert-level
balancing loss，不代表完全没有额外平衡机制。

## 12. Bonus：MLA 与 MTP

DeepSeek V3 不只有 MoE。课程还用 MLA 和 MTP 说明，真实模型的效率来自多种技术叠加。

### 12.1 MLA：缓存低维 latent，而不是完整 K/V

令 token hidden state 为 $h_t$，先压缩：

$$
c_t^{KV}
=
h_tW^{DKV},
\qquad
c_t^{KV}\in\mathbb R^{d_c},
\quad
d_c\ll d
$$

再从 latent 恢复 content key/value：

$$
k_t^C=c_t^{KV}W^{UK},
\qquad
v_t^C=c_t^{KV}W^{UV}
$$

如果没有 RoPE，attention score 可以重写：

$$
q_s(k_t^C)^\top
=
q_s(W^{UK})^\top(c_t^{KV})^\top
$$

因此 $W^{UK}$ 可以吸收到 query-side projection；decode 时不必先为每个历史 token
显式恢复完整 key，KV Cache 主要保存较小的 $c_t^{KV}$。

### 12.2 为什么 RoPE 与这种吸收有冲突

RoPE 对 query position $s$ 和 key position $t$ 使用不同旋转：

$$
(R_sq_s)(R_tk_t)^\top
$$

$R_t$ 位于 low-rank up-projection 之后，并依赖历史位置 $t$。一般情况下：

$$
R_tW^{UK}
\ne
W^{UK}R_t
$$

所以不能把所有历史位置不同的旋转都预先吸收到当前 query projection。

DeepSeek 的解决思路是把 Q/K 拆成：

- 可由 latent 表示的 content component；
- 少量独立的 decoupled RoPE dimensions。

Cache 保存低维 KV latent，再额外保存较小的 positional key component。这样同时保留
RoPE 位置信息和 MLA 的 KV compression。

> [!note] MLA 与 linear attention 不同
> MLA 主要压缩 K/V 的表示和 cache；如果仍对所有历史位置做 dense attention，score 计算关于 context length 仍是二次/每步线性。Linear attention 则把 token-level 历史本身压缩成 recurrent state。

### 12.3 MTP：增加额外的未来 token 训练信号

Multi-Token Prediction（MTP）在主 next-token head 之外增加轻量模块，让当前位置预测
更远的未来 token。DeepSeek V3 使用一个额外的未来 token 目标。

潜在作用：

- 每段训练文本提供更密集的监督；
- 强化 hidden state 对后续信息的预测能力；
- 训练好的轻量模块可为 speculative decoding 提供候选。

它会增加少量训练 compute 和参数；是否用于推理取决于 serving 实现。

## 13. Attention/MoE 的统一 AI Infra 视角

### 13.1 Shape

- Full attention：score shape 为 $[B,H,L,L]$；
- Linear state：每层 state 约为 $[B,H,d_\phi,d_v]$，不含 $L$；
- Sparse attention：候选 index 常为 $[B,H,L,K]$；
- MoE router logits：$[T,E]$；
- Top-$K$ routing indices/gates：$[T,K]$；
- Expert input：从统一 $[T,d]$ 被动态分桶为每 expert 不同的 $[T_i,d]$；
- MLA cache：低维 $c^{KV}$ 加少量 positional component。

### 13.2 Compute

| 方法 | 主要 attention mixing compute | 备注 |
| --- | --- | --- |
| Full | $O(n^2d)$ | FlashAttention 改 I/O，不改二次 FLOPs |
| Sliding window | $O(nwd)$ | $w$ 为窗口 |
| Linear | $O(nd_\phi d_v)$ | 取决于 feature/state dimension |
| Learned sparse | $O(nkd)$ + indexer | $k$ 为选中位置数 |
| MLA dense attention | 仍可为 $O(n^2d)$ | 主要先压缩 KV cache；结合 DSA 才减少连接数 |

MoE compute 则应按 active experts 核算：

$$
\text{expert FLOPs/token}
\propto
K\cdot N_e
$$

而不是直接使用 total expert parameters。

### 13.3 Memory capacity

- Full/GQA/MLA：KV Cache 通常仍随 context length 线性增长，只是每 token bytes 不同；
- Linear/Mamba/GDN：recurrent state 对 context length 近似常数，但 state matrix 本身可能不小；
- Sparse attention：若保留所有 K/V，capacity 仍为 $O(n)$；
- MoE：weights 和 optimizer state 按 total parameters 增长，不按 active parameters；
- Expert parallelism 用多设备聚合内存，单 device 只保存部分 experts。

### 13.4 Memory bandwidth

- Decode full attention 每步读取历史 KV；
- Linear recurrence 每步读写固定 state；
- Sparse attention 只 gather 选中的 KV，但访问更离散；
- MLA 减少每历史 token 的 cache bytes；
- MoE decode 可能要为少量 tokens 读取多个大型 expert weights，weight reuse 很差。

### 13.5 Communication

- Tensor parallel attention/MLP 通常需要 All-Reduce 或 Reduce-Scatter/All-Gather；
- Expert parallel MoE 引入 token dispatch/combine 的 All-to-All；
- Top-$K$、hidden width、expert placement 和跨节点比例共同决定通信量；
- load balance 不只是平均 token count，还要看每 device、每 node 和网络链路；
- Hybrid attention/MoE 会与 tensor、data、pipeline、sequence parallelism 叠加。

### 13.6 Runtime

- 理论 $O(n)$ operator 可能由 scan、small GEMM 或不成熟 kernel 限制；
- Full attention 虽是 $O(n^2)$，但 FlashAttention 的大规则 tile 可能更接近硬件峰值；
- Sparse Top-$K$ 需要 selection、gather、scatter 和动态 shape；
- MoE 的尾延迟由最慢 expert/device 决定；
- Continuous batching 会不断改变每个 expert 的 token 数和 GEMM shape；
- 训练最省 FLOPs的 architecture 未必是 decode 最快或最容易部署的 architecture。

## 14. 我的理解与推导

### 14.1 Attention alternatives 与 MoE 是同一个思想的两种应用

两者都在做 input-dependent sparsity：

```text
Attention alternatives
token → 选择/压缩历史状态 → 只计算需要的信息连接

MoE
token → 选择 experts → 只计算需要的参数子集
```

它们都要解决：

1. selector/router 的质量；
2. 离散选择的训练；
3. 负载与数据布局；
4. 被跳过部分是否仍占 memory；
5. irregular sparsity 能否映射为高效 kernel。

因此可以把 MoE 理解为 parameter-space sparse attention，也可以把 sparse attention 理解为
context-space routing。

### 14.2 “Linear time” 要分别问训练、prefill 和 decode

| 架构 | Training/prefill | Decode state | 第一步系统风险 |
| --- | --- | --- | --- |
| Full + FlashAttention | 二次 compute，I/O 优化充分 | 线性增长 KV | 长 context compute/KV traffic |
| Linear recurrent | 可用 parallel/chunk algorithms | 固定 state | state capacity、scan/kernel |
| Learned sparse | indexer + 稀疏 attention | 常保留全量 KV | index recall、gather/scatter |
| MLA | dense attention 仍可能二次 | 更小的线性 KV | latent/positional cache 实现 |
| Hybrid | 按 layer ratio 混合 | 混合 state + KV | runtime 同时支持多种 layer |

所以判断 “更高效” 至少要指定：

- context length；
- batch size；
- prefill 还是 decode；
- latency 还是 throughput；
- state/cache dtype；
- kernel 是否成熟；
- 是否跨设备。

### 14.3 Hybrid 是一种 architecture risk control

纯 full attention 成本高但表达路径直接；纯 recurrent/linear 成本可控但 state compression
风险大。Hybrid 把二者组合：

- 大多数便宜层负责持续处理；
- 少数 full layers 定期恢复 token-level 全局访问；
- 无需赌某一种新 operator 在所有能力上都替代 softmax attention。

这与系统中的 fast path + slow path 很像：大部分请求走便宜路径，少数关键位置付出完整
成本。

### 14.4 MoE 至少需要三本参数账

| 账本 | 回答的问题 | 近似口径 |
| --- | --- | --- |
| Total parameters | checkpoint/optimizer/集群总显存多大 | 所有 experts + shared model |
| Active parameters | 一个 token 做多少主要 arithmetic | Top-$K$ experts + shared path |
| Resident parameters | 一次请求实际需要加载/驻留多少权重 | 由 placement、batch 中触及的 experts 和 offload 决定 |

一个 token 只激活 37B 参数，不代表 37B dense 模型的部署成本：

- 小 batch 仍可能在不同 token 上触及大量 experts；
- weights 需要分布式常驻或按需搬运；
- 通信和动态 batching 不体现在 active-parameter 数字中。

### 14.5 Load balancing loss 在优化一个系统代理目标

真正想最小化的是：

$$
\text{step time}
=
\max_d \text{workload}_d
+\text{communication}
+\text{overhead}
$$

但它对离散 routing 和硬件执行不可直接平滑求导。Balancing loss、device loss 和
per-expert bias 都是在构造容易训练的 proxy。

因此 $\alpha$ 不只是一个 regularization hyperparameter；它表达了：

> 愿意牺牲多少 token–expert semantic affinity，换取更规则、更快的系统执行。

## 15. 用户问题与解答

> [!question] Q-CS336-L04-0514：Linear attention 为什么不能直接重排 softmax？
> - [x] #question 已解答
> - 矩阵结合律只适用于双线性的乘加：$(QK^\top)V=Q(K^\top V)$。逐行 softmax 会让第 $i$ 个 query 的每个权重依赖该行**全部 keys**：
>   $$a_{ij}=\frac{e^{q_i^\top k_j}}{\sum_l e^{q_i^\top k_l}}.$$
>   这是非线性且按 query 归一化的操作，既不存在一般的 $\operatorname{softmax}(AB)=A\operatorname{softmax}(B)$，也不能只用一个与序列长度无关的 $K^\top V$ 精确表达。
> - Linear attention 的可重排性来自可分解 kernel：$\kappa(q,k)\approx\phi(q)^\top\phi(k)$。它缓存 $S_t=\sum_{j\le t}\phi(k_j)v_j^\top$ 和 $z_t=\sum_{j\le t}\phi(k_j)$，再计算 $y_t=\phi(q_t)^\top S_t/(\phi(q_t)^\top z_t)$。有限维 $\phi$ 通常是近似或新的架构假设，而不是对 softmax 的纯代数重排。
> - 来源：[视频 05:14](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=314s)

^q-cs336-l04-0514-linear-softmax

> [!question] Q-CS336-L04-1111：固定大小 state 为什么会影响 retrieval？
> - [x] #question 已解答
> - 固定 state 是对任意长度历史的 many-to-one 压缩。以最简单的 $\phi(k)=1$ 为例，历史值 $(1,3)$ 与 $(2,2)$ 都得到 $S=4,z=2$；后续 query 只看到相同 state，无法判断某个具体 token 是多少、出现在哪个位置。
> - 更一般地，$S_t\in\mathbb R^{d_\phi\times d_v}$ 只保存若干聚合统计量，其 rank 和可区分模式数受固定维度限制；full attention 则保留每个历史 token 的 K/V，query 可以逐项寻址。因此 fixed-state 模型擅长聚合、衰减记忆和模式跟踪，却难以对无限增长的任意历史做无损 exact copy/retrieval。
> - Gating、delta update、扩大 state 或与 full attention 混合能减少冲突，但不能从信息论上消除固定容量瓶颈。
> - 来源：[视频 11:11](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=671s)

^q-cs336-l04-1111-state-capacity

> [!question] Q-CS336-L04-4730：Top-$K$ routing 的梯度到底流向哪里？
> - [x] #question 已解答
> - 设 $I=\operatorname{TopK}(z)$，$y=\sum_{e\in I}g_eE_e(x)$。反向传播时，$I$ 在当前排序不变的局部区域被当作常量：选中 experts 的参数和输入收到任务梯度，选中 gates 也通过加权和收到梯度；未选中 experts 对这个 token 没有 expert gradient。
> - **未选中 router logits 是否有任务梯度取决于 gate 定义：**若先对全部 experts 做 softmax 再 mask，选中概率的分母仍依赖未选中 logits，它们可收到间接梯度；若只在 Top-$K$ 集合内 softmax/重归一化，未选中 logits 通常为零任务梯度。它们仍可能收到 balance loss、z-loss 等辅助梯度。
> - 离散 selection boundary 本身没有普通导数。三 expert 的 Top-1 例子中，只要 $z_1>z_2,z_3$，微小变化仍选择 expert 1；越过 $z_1=z_2$ 时选择突然跳变。尤其 Top-1 gate 若选中后归一化为常数 1，任务 loss 甚至无法训练 router，必须依靠保留 gate magnitude、auxiliary loss、噪声或其他 estimator。
> - 来源：[视频 47:30](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2850s)

^q-cs336-l04-4730-routing-gradient

> [!question] Q-CS336-L04-5831：什么时候 MoE 会被 All-to-All 限制？
> - [x] #question 已解答
> - 一个近似判据是比较
>   $$T_{\rm comm}\approx\frac{2TKds\,f_{\rm remote}}{B_{\rm eff}}+T_{\rm latency}$$
>   与 expert 计算时间 $T_{\rm expert}$。这里两倍分别对应 dispatch 和 combine，$f_{\rm remote}$ 是真正跨设备/跨节点的比例，$B_{\rm eff}$ 是考虑拥塞、协议和拓扑后的有效带宽。当 $T_{\rm comm}>T_{\rm expert}$，或最慢 device 的通信/排队时间主导 step 时，就是 communication-bound。
> - 对普通两层 FFN，单 token-expert 前向约 $4dd_{ff}$ FLOPs；SwiGLU 类三矩阵 FFN 约 $6dd_{ff}$ FLOPs。$K$、$d$ 和 element bytes 增大会增加通信，$d_{ff}$ 增大则增加可用来“遮住”通信的计算。
> - 小 microbatch、小 experts、低 Top-$K$ GEMM 利用率、跨节点 InfiniBand、负载不均和随机远端路由最容易受限；大 token batch、NVLink/NVSwitch、同节点/受限路由、通信计算 overlap、降维后 dispatch 与良好负载平衡会改善。平均 bytes 够小仍不代表安全，因为 MoE step 常由 straggler 决定。
> - 来源：[视频 58:31](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3511s)
> - 预计沉淀到：[LLM Inference](../../topics/inference/LLM%20Inference.md)

^q-cs336-l04-5831-moe-alltoall

> [!question] Q-CS336-L04-MLA：MLA 为什么需要 decoupled RoPE dimensions？
> - [x] #question 已解答
> - MLA 能吸收 content key up-projection，是因为它是与位置无关的固定线性映射：若 $k_t^C=W^{UK}c_t$，则 $q_s^\top k_t^C=(W^{UK\top}q_s)^\top c_t$，decode 可直接让变换后的 query 与缓存 latent 做内积。
> - RoPE 引入依赖历史位置 $t$ 的 $R_t$：$(R_sq_s)^\top(R_tW^{UK}c_t)$。不同 token 的 $R_t$ 不同，且一般 $R_tW^{UK}\ne W^{UK}R_t$，所以不存在一个固定的 query-side 矩阵把所有历史位置的旋转一起吸收掉。
> - 因此把维度拆为 $q=[q^C;q^R]$、$k=[k^C;k^R]$，score 为 $q^{C\top}k^C+q^{R\top}k^R$。content 部分继续通过 latent 压缩和 weight absorption 计算；较小的 RoPE key component 按 token 缓存。“Decoupled”不是去掉位置信息，而是隔离那部分无法被固定线性映射吸收的 positional dimensions。
> - 来源：[官方讲义第 58 页](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf)

^q-cs336-l04-mla-rope

## 16. 本讲结论

1. Full attention 的主要长上下文成本包括二次 mixing FLOPs、历史 KV capacity 和 decode KV traffic；三者需要分别优化。
2. Linear attention 利用 matrix associativity 或 kernel feature map，把 token-token interaction 压缩为固定大小 state，并产生 parallel/recurrent dual forms。
3. Mamba-2 用 data-dependent decay 控制记忆时间；Gated DeltaNet 进一步用 delta rule 对 key 方向做 erase/write。
4. 固定 state 会损失 token-level 历史细节，因此近期公开模型常用 recurrent/linear 与 full attention 的 hybrid。
5. Learned sparse attention 保留 token-level KV，通过轻量 indexer 只选择少数位置；收益取决于 indexer 成本、recall 和稀疏 kernel。
6. MoE 通过 Top-$K$ routing 解耦 total parameters 与 active compute，但 total memory、communication 和 runtime 不会同比例下降。
7. Fine-grained experts 增加组合能力，shared experts 提供公共路径；二者的最优设置仍依赖模型与系统。
8. Load balancing 是语义路由与系统吞吐之间的冲突；auxiliary loss、device-aware routing 和 online bias 都是代理优化方法。
9. Expert parallelism 的核心数据流是两次 All-to-All 加动态 expert GEMM；MegaBlocks 等实现负责把 irregular sparsity 映射到 GPU。
10. DeepSeek V3 的效率不是单一 MoE 技巧，而是 MoE routing、MLA KV compression、MTP 和系统协同的组合。

## 17. 自测问题

1. 从 $Q,K,V$ 的 shape 出发，为什么 full attention mixing 是 $O(n^2(d_k+d_v))$？

    **面试回答：** 单头 Q、K 分别为 [n,d_k]，QKᵀ 得到 [n,n]，需约 2n²d_k FLOPs；再与 [n,d_v] 的 V 相乘，需约 2n²d_v FLOPs。因此 mixing 合计 O(n²(d_k+d_v))，未计 Q/K/V projection；causal mask 可减少常数，但不改变二次阶数。

2. 为什么 $(QK^\top)V=Q(K^\top V)$ 不能直接用于普通 softmax attention？

    **面试回答：** 矩阵乘法结合律只适用于中间没有非线性变换的 QKᵀV。普通 attention 是 softmax(QKᵀ)V，逐行指数和归一化依赖整行 score，不能穿过结合律移到 KᵀV 上；linear attention 要改用可分解的 kernel feature map 或另一种规则，不能假定与原 softmax 精确等价。

3. Causal linear attention 的 $S_t$ 和 $z_t$ 分别保存什么？

    **面试回答：** $S_t=\sum_{j\le t}\phi(k_j)v_j^\top$ 保存历史 key-value association 的累积矩阵，$z_t=\sum_{j\le t}\phi(k_j)$ 保存归一化所需的 key 特征和。输出 $y_t=\phi(q_t)^\top S_t/[\phi(q_t)^\top z_t]$；每步只更新这两个前缀状态，无须重读全部历史 K/V。

4. 为什么 linear attention 的 recurrent state 不随 context length 增长，却可能损失精确 retrieval？

    **面试回答：** 它把历史外积加到固定维度的矩阵 S_t，而不为每个 token 保留独立地址，所以状态大小由特征维度决定，与 context length 无关。代价是不同历史可能压缩成相同或近似的状态，关联之间会干扰，后续 query 无法像 full attention 那样精确取回每个历史 token。

5. Parallel–recurrent duality 分别解决训练和 decode 的什么问题？

    **面试回答：** 训练和 prefill 时，已知全部 token，可使用等价的矩阵、分块或 parallel scan 形式，减轻逐步递归的串行依赖并利用 GPU。Decode 时则转为 recurrent update，只读写固定大小状态，避免随历史长度增长的 KV 访问；两种形式服务于同一模型的不同 workload。

6. Mamba-2 的 $\gamma_t$ 与 Gated DeltaNet 的 $\beta_t$ 分别控制什么？

    **面试回答：** 在本讲的简化更新里，Mamba-2 的 γ_t 决定旧状态整体保留多少，相当于控制遗忘速度和记忆时间尺度。Gated DeltaNet 的 β_t 决定当前 key 方向的擦除与新 value 写入强度，γ_t 仍负责全局 decay；β_t=0 时不执行这次定向更新。

7. 怎样从 $(I-\beta_tk_tk_t^\top)S_{t-1}$ 看出定向 erase？

    **面试回答：** 展开可得 $S_{t-1}-\beta_tk_t(k_t^\top S_{t-1})$，也就是先读出沿当前 key 的旧 association，再从同一方向减掉一部分。若 $\lVert k_t\rVert=1$ 且 $\beta_t=1$，该方向的旧分量被清除，正交方向保留；之后加 $\beta_tk_tv_t^\top$ 写入新值，若 key 未归一化则不能直接解释为完全擦除。

8. Learned sparse attention 若仍保存全部 K/V，减少了 compute、bandwidth 和 capacity 中的哪些？

    **面试回答：** 若每个 query 只访问 k≪n 个候选，精确 attention 的计算和所选 KV 的读取量可以下降。但只要全部历史 K/V 仍驻留，cache capacity 仍随 n 增长；总 bandwidth 收益还要扣除 indexer、Top-K 和不规则 gather 的开销，不能只看稀疏矩阵的非零数。

9. 为什么 sparse indexer 如果做完整高维 $QK^\top$ 就失去了主要意义？

    **面试回答：** 因为完整高维 QKᵀ 已经付出了 dense attention 最主要的 O(n²d) 打分成本，再做 Top-K 只省后面的部分计算，还增加选择和 gather 开销。Indexer 应用更低维、分块或其他便宜表示产生候选，并在成本与重要 token 的召回率之间折中。

10. MoE 的 total parameters 与 active parameters 应怎样分别计算？

    **面试回答：** 若 E 个 routed experts 各含 N_e 参数，其余共享部分含 N_s 参数，每 token 激活 K 个 expert，则 N_total≈N_s+EN_e，N_active≈N_s+KN_e。始终激活的 shared experts 和 router 应计入相应共享项；存储和训练状态主要看 total，逐 token 的主要 FFN FLOPs 看 active。

11. 为什么增加 expert 数量、固定 Top-$K$ 时，主要 FFN FLOPs近似不变但部署成本仍会上升？

    **面试回答：** 固定 K 和 expert size 时，每 token 仍只执行 K 个 FFN，因此主要 expert FLOPs 近似不变。但总权重、Adam state 和 checkpoint 随 expert 数增大，router 更大，跨设备路由与负载平衡更复杂；每 expert token 更少还可能使 GEMM 变小、设备利用率下降。

12. Token-choice 与 expert-choice routing 的负载特性有什么不同？

    **面试回答：** Token-choice 让每个 token 选固定 K 个 expert，单 token 计算量可控，但热点 expert 可能过载。Expert-choice 让每个 expert 按 quota 选 token，更容易控制 expert 负载，却允许某 token 被选零次或多次，单 token 计算和服务语义更复杂。

13. Switch balancing loss 中 $f_i$ 与 $P_i$ 的区别是什么？

    **面试回答：** 在 Switch 的 Top-1 定义里，$f_i$ 是实际被 argmax 分到 expert $i$ 的 token 比例，是离散计数；$P_i$ 是所有 token 给 expert $i$ 的平均软概率，可微。辅助项 $\alpha E\sum_i f_iP_i$ 把 $f_i$ 视为常数，通过 $P_i$ 给高负载 expert 更强的下调压力；Top-K 情况需调整派发比例定义。

14. 为什么 balancing coefficient 太大会损害模型质量？

    **面试回答：** Balancing loss 优化的是负载均匀这一系统代理目标，任务 loss 希望 token 进入最适合的 expert，两者并不总一致。系数过大时，router 为满足均匀配额牺牲语义匹配，抑制 specialization，最终损害质量；应结合吞吐、溢出率和验证 loss 联调。

15. Expert parallel MoE 为什么需要 dispatch 和 combine 两次 All-to-All？

    **面试回答：** Token 最初在其所属设备上，选中的 experts 却可能位于其他设备，所以先通过 dispatch All-to-All 把输入送到 expert 所在处。专家计算后，再通过 combine All-to-All 把输出送回原 token 的设备，恢复顺序并按 gate 加权求和；这是 forward 的两次主要交换。

16. 推导 MoE layer 约 $2TKds$ 的 aggregate communication payload。

    **面试回答：** T 个 token 各送给 K 个 expert，每份输入向量含 d 个元素、每元素 s bytes，dispatch payload 为 TKds。若 expert 输出同为 d 维，返回再传 TKds，故 forward 的聚合 payload≈2TKds；该估算忽略本地路由、metadata、padding、链路转发和 backward 通信。

17. Capacity factor、padding 与 token dropping 之间是什么 tradeoff？

    **面试回答：** 每 expert 容量通常设为 C≈⌈capacity factor·TK/E⌉。Capacity factor 大能降低 overflow 和 token dropping，却增加预留显存及 padding 计算；容量小更省资源，但需丢弃或重路由超额 token，可能改变模型函数和质量；动态无丢弃实现仍需处理不规则负载。

18. Router 使用 FP32 和 z-loss 分别在防什么问题？

    **面试回答：** Router 使用 FP32 主要减少低精度舍入、溢出以及临界分数扰动导致的 Top-K 排名不稳定。Router z-loss 约束 (logΣexp r)²，抑制 log-normalizer 和整体 logits 漂移；它不是直接的负载均衡或熵正则，不能单独保证 experts 使用均匀。

19. Upcycling 为什么能保持 dense FFN 的初始函数？为什么 experts 又可能难以分化？

    **面试回答：** 若每个 expert 都复制同一个 dense FFN，且选中 gate 之和为 1，则 $\sum_i g_i\operatorname{FFN}(x)=\operatorname{FFN}(x)$，初始层函数可保持不变。相同初始化也造成对称性，输出缺少差异时 router 分工信号较弱，experts 需要不同路由数据、扰动或继续训练逐步分化；拆分和重初始化不一定严格保函数。

20. MLA 压缩的主要是 KV Cache，为什么它本身不一定消除 dense attention 的二次计算？

    **面试回答：** MLA 把每个历史 token 的 K/V 表示压成较低维 latent，减少的是每个位置的 cache 宽度。只要每个 query 仍与所有历史位置交互，prefill 仍有 O(n²) 对连接、decode 每步仍访问 O(n) 历史，因而不自动具有 linear attention 的固定历史状态和线性总计算。

21. 为什么 RoPE 会妨碍把 MLA 的 key up-projection 完全吸收到 query side？

    **面试回答：** 无位置旋转时 $q^\top W_Kc$ 可改写为 $(W_K^\top q)^\top c$，把 key up-projection 吸收到 query 侧。RoPE 插入位置相关的 $R_s^\top R_t$ 后，一般不能与 $W_K$ 交换，也就无法用一个与历史位置 $t$ 无关的 query 变换完成吸收；decoupled RoPE 用小块独立位置 key 保留该信息。

22. Attention alternatives 与 MoE 为什么都可以视为 conditional computation？

    **面试回答：** 两者都试图按输入把有限计算用于更相关的信息：learned sparse attention 选择历史位置，MoE 选择参数专家，因此分别在上下文和参数维度做条件计算。这个类比对 sparse attention 最直接；linear/state 方法主要通过压缩与门控更新历史提效，并不一定含显式离散选择，不能把所有替代方案都等同于稀疏路由。


## 参考资料

- [Stanford CS336 Spring 2026](https://cs336.stanford.edu/)
- [Lecture 04 video](https://www.youtube.com/watch?v=cKSwj_qZ8Jg)
- [Lecture 04 official slides](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf)
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)
- [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313)
- [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556)
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906)
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts](https://arxiv.org/abs/2211.15841)
- [Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints](https://arxiv.org/abs/2212.05055)
- [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066)
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Transformer Architecture](../../topics/model-architecture/Transformer%20Architecture.md)
- [Transformer Block](../../topics/model-architecture/Transformer%20Block.md)
- [LLM Inference](../../topics/inference/LLM%20Inference.md)
