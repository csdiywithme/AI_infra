---
type: course-note
status: seed
course: "[[Stanford CS336]]"
lecture: 3
lecture_date: 2026-04-06
area: model-architecture
topics:
  - "[[Transformer Architecture]]"
  - "[[Transformer Block]]"
  - "[[LLM Inference]]"
aliases:
  - Stanford CS336 Lecture 03
  - CS336 Architectures and Hyperparameters
video_url: https://www.youtube.com/watch?v=lVynu4bo1rY
---

# Lecture 03：Architectures and Hyperparameters

> [!abstract] 本讲一句话
> 现代 dense LLM 看似型号繁多，实际已经收敛到一组相当稳定的默认设计：非 residual 的 normalization（通常是 Pre-RMSNorm）、GLU FFN、RoPE，以及面向推理效率的 GQA；剩余差异主要来自训练稳定性、长上下文和硬件效率之间的取舍。

## 来源与范围

- 课程：Stanford CS336 — Language Modeling from Scratch, Spring 2026
- 讲师：Tatsunori Hashimoto
- 日期：2026-04-06
- [课程视频](https://www.youtube.com/watch?v=lVynu4bo1rY)，时长 1:29:13
- [课程主页](https://cs336.stanford.edu/)
- [官方讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)，67 页
- 本讲覆盖：dense Transformer 的常见架构选择、经验超参数、训练稳定性技巧和 attention 推理优化
- 本讲不覆盖：attention alternatives、MoE 和 SSM 的详细原理；这些属于下一讲

> [!warning] 初稿说明
> 这版用于快速审阅，以视频自动章节和官方讲义为主。YouTube 转写文稿当前不可用，因此没有把课堂口头补充伪装成逐字记录；视频时间点来自自动生成章节，后续精读时可以再细化。

## 视频索引

| 时间                                                           | 视频内容                          | 对应笔记                                                                              |
| ------------------------------------------------------------ | ----------------------------- | --------------------------------------------------------------------------------- |
| [00:00](https://www.youtube.com/watch?v=lVynu4bo1rY&t=0s)    | Introduction and survey       | [[#1. 课程的观察方法\|1. 课程的观察方法]]                                                          |
| [04:22](https://www.youtube.com/watch?v=lVynu4bo1rY&t=262s)  | Core architecture patterns    | [[#2. Normalization 与 residual path\|2. Normalization 与 residual path]]              |
| [07:27](https://www.youtube.com/watch?v=lVynu4bo1rY&t=447s)  | Normalization techniques      | [[#2. Normalization 与 residual path\|2. Normalization 与 residual path]]              |
| [13:38](https://www.youtube.com/watch?v=lVynu4bo1rY&t=818s)  | Systems and bias optimization | [[#2.4 为什么 RMSNorm 和去 bias 仍能影响运行时间\|2.4 为什么 RMSNorm 和去 bias 仍能影响运行时间]]              |
| [20:11](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1211s) | Activation functions          | [[#3. Activation、GLU 与 block 组织\|3. Activation、GLU 与 block 组织]]                       |
| [27:16](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1636s) | Serial vs. parallel blocks    | [[#3.3 Serial 与 parallel block\|3.3 Serial 与 parallel block]]                        |
| [30:42](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1842s) | Position embeddings           | [[#4. 位置编码与 RoPE\|4. 位置编码与 RoPE]]                                                    |
| [38:53](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2333s) | Hyperparameter rules of thumb | [[#5. 已经收敛的超参数经验\|5. 已经收敛的超参数经验]]                                                    |
| [46:44](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2804s) | Scaling and vocabulary size   | [[#5.3 模型的深宽比\|5.3 模型的深宽比]]、[[#5.4 Vocabulary size\|5.4 Vocabulary size]]               |
| [55:12](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3312s) | Regularization and stability  | [[#5.5 Dropout 与 weight decay\|5.5 Dropout 与 weight decay]]、[[#6. 训练稳定性技巧\|6. 训练稳定性技巧]] |
| [59:23](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3563s) | Attention interventions       | [[#6. 训练稳定性技巧\|6. 训练稳定性技巧]]、[[#7. 面向推理和长上下文的 attention\|7. 面向推理和长上下文的 attention]]       |

## 1. 课程的观察方法

这讲不是从第一性原理推出“唯一正确”的 Transformer，而是横向比较大量公开 dense LLM：

1. 哪些设计几乎所有模型都采用？
2. 哪些设计仍然在变化？
3. 变化是为了模型质量、训练稳定性，还是系统效率？

课程的核心方法论是：

> 架构设计既是表示学习问题，也是经验科学和系统工程问题。一个修改即使几乎不减少总 FLOPs，也可能因为减少数据搬运而显著降低 wall-clock time。

与 2017 年原始 Transformer 相比，课程采用的简单现代基线主要改了四件事：

- LayerNorm 放到子层之前，即 Pre-Norm；
- 使用 RoPE，而不是 additive sinusoidal position embedding；
- FFN 使用 SwiGLU，而不是 ReLU；
- linear layer 和 normalization 通常不使用 bias。

## 2. Normalization 与 residual path

### 2.1 Post-Norm 与 Pre-Norm

原始 Transformer 的 Post-Norm 可简写为：

$$
x_{l+1}=\operatorname{Norm}\left(x_l+F_l(x_l)\right)
$$

现代 LLM 更常见的 Pre-Norm 是：

$$
x_{l+1}=x_l+F_l\left(\operatorname{Norm}(x_l)\right)
$$

Pre-Norm 的关键不是名称中的“pre”，而是 **Norm 不再位于主 residual stream 上**。这样 residual path 保留了一条更接近 identity mapping 的信号和梯度通路。

课程给出的经验解释包括：

- 减轻 gradient attenuation；
- 减少 gradient spikes；
- 早期优势是可以减少对 warmup 的依赖；
- 在更大网络中表现为更好的稳定性，以及可以承受更大的 learning rate。

因此，“现代模型使用 Pre-Norm”更准确地说是：

> 大家基本同意不要让 normalization 截断主 residual signal path。

### 2.2 Double Norm / non-residual Post-Norm

近期模型不一定只有一个 Pre-Norm。Grok、Gemma 2、OLMo 2 等模型会在子层输出侧增加额外 Norm，但让它处于 residual 分支内部，而不是 residual addition 之后：

```text
x ────────────────────────┐
│                         +
└─ Norm → Attention → Norm ┘
```

这类设计试图同时保留 identity residual path，并进一步限制子层更新的数值尺度。

### 2.3 LayerNorm 与 RMSNorm

LayerNorm 对 hidden dimension 同时做中心化和缩放：

$$
\operatorname{LayerNorm}(x)
=
\frac{x-\mathbb E[x]}{\sqrt{\operatorname{Var}(x)+\epsilon}}
\odot\gamma+\beta
$$

RMSNorm 不减去均值，通常也没有 bias：

$$
\operatorname{RMSNorm}(x)
=
\frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon}}
\odot\gamma
$$

公开实验通常显示 RMSNorm 与 LayerNorm 的模型质量相近，但 RMSNorm：

- 少一次 mean reduction；
- 少一个 bias 参数；
- kernel 更简单；
- 需要搬运的数据更少。

因此现代 Llama-style 模型普遍采用 RMSNorm。

### 2.4 为什么 RMSNorm 和去 bias 仍能影响运行时间

“Norm FLOPs 很少，所以优化 Norm 不重要”是错误的推理。Norm、activation 和 bias add 等操作的 arithmetic intensity 很低，常常受 memory bandwidth 或 kernel launch 限制，而不是受算力限制。

同理，很多现代 Transformer 去掉 linear bias，并不主要是为了减少参数总量，而是因为：

- bias 对质量的收益不明确；
- 额外参数和逐元素操作需要读写显存；
- 它可能让 fusion 和优化路径更复杂；
- 有工作报告了更好的优化稳定性。

> [!tip] AI Infra 心智模型
> FLOPs 适合做第一层资源估算，但不能单独预测运行时间。对逐元素操作，应重点看 memory traffic、fusion 和 kernel launch。

## 3. Activation、GLU 与 block 组织

### 3.1 从 ReLU、GeLU 到 gated FFN

非 gated FFN 的通用形式：

$$
\operatorname{FFN}(x)=\sigma(xW_1)W_2
$$

历史上常见的 $\sigma$ 包括：

- ReLU：原始 Transformer、T5、Gopher、Chinchilla、OPT；
- GeLU：GPT-1/2/3、GPT-J、GPT-NeoX、BLOOM。

GLU 类 FFN 增加一条 gate projection：

$$
\operatorname{FFN}_{GLU}(x)
=
\left[\sigma(xW_{gate})\odot(xW_{up})\right]W_{down}
$$

其中：

- GeGLU 使用 GeLU；
- SwiGLU 使用 Swish/SiLU，$\operatorname{SiLU}(z)=z\sigma(z)$。

### 3.2 为什么 gated FFN 会缩小中间维度

普通 FFN 有两块主要权重：

$$
W_1\in\mathbb R^{d\times d_{ff}},\qquad
W_2\in\mathbb R^{d_{ff}\times d}
$$

参数量约为：

$$
2dd_{ff}
$$

GLU 增加第三个 projection，参数量约为：

$$
3dd_{ff}
$$

为了与 $d_{ff}=4d$ 的普通 FFN 保持近似参数量：

$$
3d\,d_{ff}^{GLU}\approx 2d(4d)
$$

得到：

$$
d_{ff}^{GLU}\approx\frac{8}{3}d
$$

这就是课程中“GLU 的 $d_{ff}$ 缩为原来的 $2/3$”的来源。公开实验显示 SwiGLU/GeGLU 的提升相当稳定，因此它们已经成为现代模型的共识选择。

### 3.3 Serial 与 parallel block

常规 Pre-Norm block 串行执行 Attention 和 MLP：

```text
x → Norm → Attention → Add → Norm → MLP → Add
```

写成公式是：

$$
u_l=x_l+\operatorname{Attn}(\operatorname{Norm}_1(x_l))
$$

$$
x_{l+1}=u_l+\operatorname{MLP}(\operatorname{Norm}_2(u_l))
$$

关键点是 MLP 的输入 $u_l$ 已经包含本层 Attention 的输出。Attention 先把其他 token
的信息写入当前位置，MLP 随后可以在同一个 block 内立即加工这份新信息。

GPT-J、PaLM、GPT-NeoX 等尝试让两个分支并行：

```text
                     ┌→ Attention ─┐
x_l → shared Norm ───┤              + → x_{l+1}
  └──────────────────┼──────────────┤
                     └→ MLP ────────┘
```

对应公式：

$$
h_l=\operatorname{Norm}(x_l)
$$

$$
x_{l+1}
=x_l+\operatorname{Attn}(h_l)+\operatorname{MLP}(h_l)
$$

Attention 和 MLP 读取的是同一个 $h_l$，所以本层 MLP 看不到本层 Attention 刚产生的
结果；两条分支直到最后 residual add 时才汇合。

#### 3.3.1 “Lost half of your depth” 是什么意思

这里的 depth 不是 block 数量、参数量或矩阵乘法数量，而是 **计算图最长路径上的串行
子层数，也就是 compositional depth**。

固定堆叠 $L$ 个 block：

| 结构 | 单个 block 的最长依赖路径 | $L$ 个 block 的近似串行深度 |
| --- | --- | ---: |
| Serial | Attention → MLP | $2L$ 个子层 |
| Parallel | Attention 或 MLP | $L$ 个子层 |

在 serial block 中，一条信息路径可以依次经过：

```text
Attn₁ → MLP₁ → Attn₂ → MLP₂ → ... → Attn_L → MLP_L
```

而 parallel block 的两条分支之间没有先后关系。一条计算路径在每层只能经过其中一个
分支，然后在 residual stream 汇合：

```text
┌→ Attn₁ ─┐   ┌→ Attn₂ ─┐         ┌→ Attn_L ─┐
│         + → │         + → ... → │          +
└→ MLP₁ ──┘   └→ MLP₂ ──┘         └→ MLP_L ──┘
```

虽然每层仍然同时计算了 Attention 和 MLP，最长路径却没有连续经过二者。因此 Tatsu
说 “effectively, you've lost half of your depth”。

也可以从 token 信息流理解：

- Serial：第 $l$ 层 Attention 把 token $j$ 的信息传给 token $i$，紧接着第 $l$ 层
  MLP 就能加工这份新信息；
- Parallel：第 $l$ 层 MLP 与 Attention 同时启动，只看旧的 $x_l$。Attention 新写入
  的信息至少要等到第 $l+1$ 层才能被 MLP 加工。

所以 parallel block 没有删除任何模块，但在固定 block 数下减少了“先通信、再加工”
的连续组合次数。这里的“一半”是很有用的子层级近似，并不是严格的表达能力定理；
Attention 和 MLP 内部本身仍包含多步运算，residual sum 也会混合两条分支。

#### 3.3.2 为什么可以共享 Norm

Serial block 必须做两次 Norm：

```text
h_attn = Norm₁(x)
a      = Attention(h_attn)
u      = x + a
h_mlp  = Norm₂(u)       # 必须等 Attention 和 residual add 完成
m      = MLP(h_mlp)
y      = u + m
```

第二次 Norm 的输入包含 Attention 输出，因此不能提前执行。

Parallel block 的两条分支都读取同一个输入：

```text
h = Norm(x)             # 只算一次
a = Attention(h)
m = MLP(h)
y = x + a + m
```

于是 Norm 的 reduction、缩放、输入读写和一次 kernel launch 都可以共享。不过 Norm
只占 block 总工作量的一小部分，因此这通常不是最主要的加速来源。

#### 3.3.3 “融合矩阵乘法”具体融合什么

把 batch 和 sequence 维展平，令：

$$
H\in\mathbb R^{M\times d},\qquad M=B\cdot L
$$

Attention 首先需要 Q/K/V projections：

$$
[Q,K,V]=H W_{QKV}
$$

SwiGLU MLP 首先需要 gate/up projections：

$$
[G,U]=H[W_{gate},W_{up}]
$$

在 parallel block 中，它们的左输入都是同一个 $H$，所以可以把权重沿输出维拼接：

$$
W_{fused}
=
\left[
W_{QKV}\mid W_{gate}\mid W_{up}
\right]
$$

然后用一次更大的 GEMM 计算：

$$
[Q,K,V,G,U]=H W_{fused}
$$

最后只是把输出视图切分给 Attention 和 MLP，不需要复制数据。潜在收益包括：

- $H$ 可以少从显存或 cache 读取一次；
- 多个小 GEMM 变成一个更大的 GEMM，更容易填满 GPU；
- 减少 kernel launch 和调度开销；
- Tensor shape 合适时，更容易与周边 projection/fusion kernel 协同。

这主要能融合两条分支的 **input projections**。Attention 的 output projection 和
MLP 的 down projection 输入不同、且分别依赖各自分支的中间结果，不能一起塞进上面
这次 GEMM。

#### 3.3.4 “并行调度”与 GEMM fusion 的区别

Fusion 是把多个 projection 合成一次大 GEMM；并行调度则保留不同 kernel，但在 shared
Norm 完成后让两条独立分支重叠执行：

```text
time ─────────────────────────────────────────────→

Attention:  QKV GEMM → RoPE/attention → O projection ─┐
                                                       + → residual add
MLP:        gate/up GEMM → SwiGLU → down projection ──┘
```

理论上的 block latency 从：

$$
T_{serial}\approx T_{attn}+T_{mlp}
$$

变成理想情况下的：

$$
T_{parallel}\approx\max(T_{attn},T_{mlp})
$$

但这只是依赖关系给出的上限，真实 GPU 上通常达不到：

- 大 GEMM 单独就可能占满 Tensor Cores，两个 GEMM 同时提交也只能争抢同一批 SM；
- Attention 与 MLP 还会竞争 HBM bandwidth、shared memory 和寄存器；
- training/prefill 的矩阵通常很大，并发执行的额外收益可能有限；
- decode 或小 batch 下矩阵较“瘦”，GPU 利用率不足，融合和重叠更可能有价值；
- runtime、编译器和模型并行的切分方式必须真正支持这种重叠。

因此 parallel block **没有减少主要 FLOPs**。它提供的是更短的关键路径、更少的数据
搬运和更大的调度空间；能否转化成 wall-clock speedup，取决于 workload 和实现。

#### 3.3.5 为什么它没有成为主流

Parallel block 的权衡可以概括为：

| 收益 | 代价 |
| --- | --- |
| 共享 Norm | MLP 不能立即处理本层 Attention 的新信息 |
| 可融合 input projections | 固定 block 数下 compositional depth 近似减半 |
| 两分支可以重叠调度 | 两分支可能争抢同一 GPU 资源，实际加速小于理想值 |
| 缩短 block critical path | 为恢复串行深度而增加 block 会抵消部分系统收益 |

课程对公开模型的观察是：parallel block 一直存在，但没有成为主流；大多数近期 dense
LLM 仍使用 serial block。一个合理的理解是：在现有硬件和 kernel 已经高度优化的
条件下，parallel block 的实际 wall-clock 收益未必足以稳定抵消 compositional depth
与模型质量方面的代价。这个解释是工程推论，不是课程给出的严格因果结论。

## 4. 位置编码与 RoPE

课程比较了四类位置编码：

| 方法 | 注入位置的方式 | 代表模型 |
| --- | --- | --- |
| Sinusoidal | 将固定 sine/cosine 向量加到 token embedding | 原始 Transformer |
| Learned absolute | 将可学习的绝对位置向量加到 embedding | GPT-1/2/3、OPT |
| Relative | 在 attention score 中加入相对位置项 | T5、Gopher、Chinchilla |
| RoPE | 按位置旋转 Q/K 的成对坐标 | GPT-J、PaLM、Llama、多数 2024+ 模型 |

理想的相对位置机制希望：

$$
\langle f(x,i),f(y,j)\rangle
=g(x,y,i-j)
$$

即 attention score 只依赖内容和相对距离，而不依赖绝对位置。

RoPE 将向量的每两维视为一个二维平面，并按位置 $m$ 旋转：

$$
R(m,\theta)=
\begin{bmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)&\cos(m\theta)
\end{bmatrix}
$$

对 Query 和 Key 应用旋转后：

$$
\langle R(i)q,R(j)k\rangle
=
\langle q,R(j-i)k\rangle
$$

于是点积只显式依赖相对位置 $j-i$。与 additive sinusoidal embedding 不同，RoPE 是乘法旋转，不会产生 token embedding 与 position embedding 的加法 cross terms。

> [!note] 边界
> 这讲解释的是 RoPE 的核心相对位置性质，不是长上下文 extrapolation 的完整方案。RoPE scaling、频率选择和 YaRN/NTK-aware scaling 需要单独讨论。

## 5. 已经收敛的超参数经验

### 5.1 FFN expansion ratio

非 gated FFN 的默认值：

$$
d_{ff}\approx4d_{model}
$$

GLU 为保持相近参数量，默认值：

$$
d_{ff}\approx\frac{8}{3}d_{model}\approx2.67d_{model}
$$

实际模型常落在约 $2.5$ 到 $3.5$，PaLM 等模型更大。Kaplan 等实验显示这是一个相当宽的 good basin，而不是精确最优点。

反例同样重要：T5 11B 曾使用 $d_{ff}=64d_{model}$，模型仍能工作；但 T5 v1.1 又回到 GeGLU 的 2.5 倍，说明“能训练”不代表“系统或质量最优”。

### 5.2 Attention head dimension

常见约定：

$$
n_{heads}\cdot d_{head}=d_{model}
$$

这不是数学要求。T5、LaMDA、PaLM 等模型有明显偏离，但多数模型的 ratio 仍接近 1。

课程的态度是：

- 这是强经验共识；
- 公开验证却没有 FFN ratio 那么充分；
- 应把它视为稳定默认值，而不是不可违反的定律。

### 5.3 模型的深宽比

课程用：

$$
\frac{d_{model}}{n_{layers}}
$$

描述模型是“宽而浅”还是“窄而深”。多数公开模型落在大约 100–200，极端例子更低或更高。

深度和宽度的选择不只影响 perplexity：

- 更深：依赖链更长，单请求 latency 更高；
- 更深：层间顺序依赖更多，更难做并行；
- 更宽：单层矩阵更大，通常更有利于设备利用率；
- 具体选择最终受并行策略、设备规模和延迟目标约束。

因此这个超参数往往由系统约束决定，而不是只靠离线 loss 搜索。

### 5.4 Vocabulary size

课程观察到：

- 单语模型通常约 30k–50k；
- 多语或 production 模型常见 100k–250k；
- GPT-4、DeepSeek、Qwen、Gemma 等近年模型普遍使用更大的词表。

大词表的权衡：

- 优点：多语文本和常见字符串可用更少 token 表示，序列更短；
- 成本：embedding 和 LM head 参数量随 $Vd$ 增长；
- 成本：输出 logits、softmax 和 sampling 的工作量随 $V$ 增长；
- 影响：tokenizer 改变训练 token 数，不能只看“每 token loss”比较模型。

### 5.5 Dropout 与 weight decay

旧模型较常使用 dropout；较新的大模型预训练通常关闭 dropout，但保留 weight decay。

> [!question] 已处理：Dropout 和 weight decay 分别在做什么？
> - [x] #ai-review
> - 原始问题：不太了解这两个概念，希望可以展开讲下。
> - 结论：Dropout 给中间激活加入随机噪声；weight decay 持续缩小参数。现代 LLM 预训练中，后者的价值更多体现在优化动力学和稳定性，而不只是传统的“防过拟合”。

#### 5.5.1 Dropout：随机删除一部分激活

训练时，对中间激活 $h$ 采样一个 Bernoulli mask：

$$
m_i\sim\operatorname{Bernoulli}(1-p),\qquad
\widetilde h=\frac{m\odot h}{1-p}
$$

其中 $p$ 是 dropout probability。除以 $1-p$ 叫 **inverted dropout**，它保证：

$$
\mathbb E[\widetilde h]=h
$$

推理时不再随机删除激活，也不需要额外缩放。直观上，每一步训练的都是一个略有不同的
子网络，模型不能过度依赖某几个神经元总是同时出现，因此它是一种对激活施加的随机
正则化。

Dropout 不等于“少算一些 FLOPs”：通用 GPU 实现通常仍计算完整的 dense tensor，只是
额外生成 mask 并做逐元素乘法。它还会增加梯度噪声，使达到同一训练 loss 可能需要更多
训练步。

现代 LLM 预训练常把 $p$ 设为 0，主要是因为：

- 数据量巨大，且通常只训练约一个 epoch，经典过拟合不是首要矛盾；
- mini-batch、数据顺序等已经带来随机性；
- 大规模训练更关心可预测收敛和有效利用固定 compute budget；
- dropout 的收益依赖数据规模和下游任务，不能从“小模型上有效”直接外推。

这不是说 dropout 已经无用：数据少、反复训练多个 epoch、微调或明显过拟合时，它仍是
合理的实验选项。

#### 5.5.2 Weight decay：每一步都把参数向零收缩

以 AdamW 为例，忽略 bias correction 的细节，一步更新可写成：

$$
\theta_{t+1}
=
(1-\eta_t\lambda)\theta_t
-
\eta_t\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}
$$

其中 $\eta_t$ 是 learning rate，$\lambda$ 是 weight-decay coefficient。第一项表示即使
当前梯度为零，参数也会按比例缩小；第二项才是 Adam 的梯度更新。

“AdamW 是 decoupled weight decay”的准确含义是：$\lambda\theta$ 不进入 Adam 的
moment estimation 和逐坐标 preconditioner。它并不意味着 weight decay 可以脱离
learning rate 单独理解，因为每一步实际的收缩因子仍然是 $1-\eta_t\lambda$。

经过 $T$ 步，单看 decay 的累计效果近似为：

$$
\prod_{t=1}^{T}(1-\eta_t\lambda)
\approx
\exp\left(-\lambda\sum_{t=1}^{T}\eta_t\right)
$$

这就解释了它和 cosine schedule 的交互：

- 训练前期 learning rate 大，同一个 $\lambda$ 产生的每步收缩更强；
- cosine decay 后期 learning rate 接近 0，weight decay 的实际作用也随之减弱；
- 改 learning-rate peak、warmup 或训练步数，会同时改变累计 shrinkage，因此不能只固定
  $\lambda$、孤立比较不同 schedule。

#### 5.5.3 为什么课程不把它简单解释为“防过拟合”

课程引用的近单 epoch 语言模型实验里，不同 weight decay 配置的 training loss 和
validation loss 大致沿对角线一起改善：如果只是传统正则化，常见预期是 training loss
变差而 validation loss 变好；图里更像是优化本身找到了更好的解。

因此在这一训练 setting 中，更合适的理解是：

- weight decay 改变参数范数和不同方向上的更新动态；
- 它会与 learning-rate schedule 一起影响训练 loss 和数值稳定性；
- 合适值可能降低 loss、减少发散风险，过大则会压制参数学习；
- 最优 $\lambda$ 依赖 optimizer、learning-rate schedule、训练时长和模型规模。

这个结论针对现代大规模、近单 epoch 训练，不否认 weight decay 在其他 setting 中也能
发挥经典正则化作用。

## 6. 训练稳定性技巧

> [!question] 已处理：stability、图中的尖峰，以及 softmax 之外的嫌疑
> - [x] #ai-review
> - 原始问题：stability 指什么？课上图中的尖峰是什么含义？除了 softmax 还有什么值得怀疑？
> - 结论：稳定不等于 loss 完全光滑，而是 loss、梯度、激活和 logits 长期保持有限、可控，并能持续收敛；尖峰是某个监控量突然远高于邻近趋势，需要结合多个信号定位。

### 6.0.1 什么叫“训练稳定”

一个相对稳定的训练过程通常满足：

- loss 虽有 mini-batch 噪声，但整体趋势可预测地下降；
- gradient norm、activation RMS、parameter norm 和 logits 不持续失控增长；
- 不频繁出现 NaN/Inf、optimizer state 损坏、loss 突然永久抬升或必须回滚 checkpoint；
- 同一配置换少量随机种子或数据顺序，不会频繁从“正常收敛”变成“发散”。

因此不能把每次 loss 波动都叫 instability。困难或异常 batch 也可能造成一次性尖峰；
真正危险的是尖峰越来越高、越来越密，和其他内部量同步恶化，或者尖峰之后无法恢复。

### 6.0.2 课程图里的“尖峰”是什么

讲义的上图是 training loss，下图是 gradient $L_2$ norm。旧 OLMo 7B 曲线中，
gradient norm 会突然跳到远高于局部基线的位置，loss 往往也出现峰值；OLMo 2 的曲线
则明显更平滑。

gradient-norm spike 表示这一批数据产生的整体更新信号突然非常大。它可能：

1. 被 gradient clipping 或 optimizer 吸收，之后自行恢复；
2. 把参数推入更不稳定的区域，引发连续尖峰；
3. 在低精度计算中溢出，最终出现 NaN 或 loss divergence。

图本身展示的是**症状**，不是根因证明。一次尖峰可能来自特殊 batch；需要看 attention
logits、激活、optimizer state 和 batch metadata 是否同步异常，才能判断根因。

### 6.0.3 除了 softmax，还应该怀疑什么

课程把两个 softmax 视为高风险数值位置：

1. attention softmax；
2. output vocabulary softmax。

但“softmax 有指数运算”不是完整诊断。稳定实现会先减去最大 logit；更常见的问题是
上游让 logits、Q/K norm 或整体参数尺度不断增长。排查时可以按下面几类看：

| 类别 | 典型嫌疑 | 优先检查 |
| --- | --- | --- |
| Optimization | peak LR 过大、warmup 太短、Adam $\beta_2/\epsilon$ 或 clipping 不合适 | gradient norm、update/parameter ratio、optimizer moments |
| Architecture/scale | residual 逐层放大、初始化不当、Norm 位置或 $\epsilon$ 不合适 | 各层 activation RMS、parameter norm、Q/K norm |
| Numerical precision | fp16/bf16 overflow/underflow、reduction 精度或 loss scaling | 每个算子的 NaN/Inf、master weights、低精度与 fp32 对照 |
| Data/workload | 异常样本、损坏数据、超长序列、长度或数据分布突变 | spike 对应的 batch ID、长度、数据来源和 token 统计 |
| Distributed system | 某 rank 数据/状态不同步、checkpoint resume 不一致、collective 或 kernel bug | per-rank checksum/norm、optimizer step、恢复前后配置 |

有用的监控信号包括：

- attention logits 的 max/min、Q/K norm 和 attention entropy；
- output logits 的 max、$\log Z$；
- 每层 activation RMS、gradient norm 和 parameter norm；
- update-to-parameter ratio、Adam moments；
- spike 所对应 batch 的来源、长度以及各 rank 是否一致。

先定位“哪个量最早失控”，再选择干预，通常比把所有 stability trick 一起打开更容易得到
可解释结果。下面三个方法分别作用在不同位置。

### 6.1 Output z-loss

> [!question] 已处理：z-loss 到底约束了什么？
> - [x] #ai-review
> - 原始问题：这部分没太看懂。
> - 结论：cross-entropy 只关心 logits 之间的相对差，不约束所有 logits 共同平移；z-loss 给这个“自由漂移方向”加一个很小的锚。

设正确类别为 $y$，输出 logits 为 $u$。普通 cross-entropy 是：

$$
\mathcal L_{\mathrm{CE}}(u,y)
=
-u_y+\log\sum_{v=1}^{|V|}\exp(u_v)
=-u_y+\log Z
$$

它有一个关键的 shift invariance。给所有 logits 同时加常数 $c$：

$$
\mathcal L_{\mathrm{CE}}(u+c\mathbf 1,y)
=
\mathcal L_{\mathrm{CE}}(u,y)
$$

例如 $[1000,999]$ 和 $[1,0]$ 产生相同的 softmax probability 和 cross-entropy，但前者
的数值幅度明显更危险。也就是说，主 loss 没有动力阻止 logits 沿着“共同加一个常数”
的方向漂移。

令：

$$
Z=\sum_{v=1}^{|V|}\exp(u_v)
$$

z-loss 在主 loss 上增加：

$$
\mathcal L_z=\alpha\left(\log Z\right)^2
$$

其梯度为：

$$
\frac{\partial\mathcal L_z}{\partial u_j}
=
2\alpha\log Z\cdot\operatorname{softmax}(u)_j
$$

- $\log Z>0$ 时，梯度下降会把 logits 整体向下推；
- $\log Z<0$ 时，会把 logits 向上推；
- probability 越高的类别收到的校正越大。

因此 z-loss 把 $\log Z$ 轻柔地锚在 0 附近，使 logits 更接近正规化后的 log
probabilities。它主要约束 output softmax 的 log-normalizer 和整体漂移，不直接规定
类别排序，也不是 entropy regularization。

PaLM 使用过 $\alpha=10^{-4}$，但这只是具体模型的经验值，不是通用常数。$\alpha$ 太大
会让辅助项干扰主任务；z-loss 也不能替代数值稳定的 `logsumexp` 实现。

### 6.2 QK Norm

在计算 attention logits 前，对 Query 和 Key 做 LayerNorm 或 RMSNorm：

$$
A=
\operatorname{softmax}
\left(
\frac{\operatorname{Norm}(Q)\operatorname{Norm}(K)^T}{\sqrt{d_h}}
\right)
$$

它直接限制进入 attention softmax 的输入尺度。DCLM、OLMo 2、Gemma 2、Qwen 3、Gemma 4 等近期模型采用了类似设计。

### 6.3 Logit soft-capping

使用 tanh 将 logits 限制在固定范围：

$$
\operatorname{softcap}(z)
=c\tanh\left(\frac{z}{c}\right)
$$

它提供更强的上界，但也可能阻止模型表达非常尖锐的 attention distribution，因此稳定性收益和模型质量之间需要实验权衡。

### 6.4 z-loss、QK Norm 与 soft-cap 怎么选

| 方法 | 直接针对的信号 | 生效位置与开销 | 主要风险 |
| --- | --- | --- | --- |
| z-loss | output $\log Z$ 或 output logits 整体漂移 | 只增加训练 loss 中的小项，推理时可移除 | 系数太大时干扰主 loss |
| QK Norm | Q/K norm 和 attention logits 随深度增长 | 每层 attention、训练和推理都执行 Norm | 改变 attention geometry，并增加少量运行时成本 |
| soft-cap | 少数极端 attention/output logits | 训练和推理都做 tanh | 饱和后梯度变小，cap 太低会限制尖锐分布 |

实践顺序可以是：

1. 先排查 learning rate、warmup、precision、异常 batch 和 optimizer state；
2. output $\log Z$ 漂移时优先试 z-loss；
3. Q/K norm 和 attention logits 逐层失控时试 QK Norm；
4. 仍有少数极端 logits，且明确需要硬上界时再评估 soft-cap；
5. 每次只改变一个主要因素，比较 stability、loss 和 runtime。

## 7. 面向推理和长上下文的 attention

### 7.1 为什么 decode 阶段的瓶颈不同

训练和 prefill 可以对多个 token 并行计算；自回归 decode 必须逐 token 前进。使用 KV Cache 后，新 token 不再重算历史 K/V，但每一步仍要读取历史 cache。

这使 decode attention 常受以下因素限制：

- KV Cache 容量；
- HBM bandwidth；
- batch size；
- sequence length；
- kernel 和调度开销。

换句话说，decode 的问题往往不是“乘法算不动”，而是“每生成一个 token 要搬太多历史状态”。

### 7.2 MHA、MQA 与 GQA

| 方法 | Query heads | KV heads | 主要效果 |
| --- | ---: | ---: | --- |
| MHA | $H$ | $H$ | 最大表达容量，KV Cache 最大 |
| MQA | $H$ | 1 | 最大幅度压缩 KV Cache，可能有小幅质量损失 |
| GQA | $H$ | $H_{kv}<H$ | 用可调共享程度平衡质量和推理效率 |

KV Cache 的核心 shape 可写为：

$$
[B,L,H_{kv},d_h]
$$

因此从 MHA 改成 GQA/MQA，KV Cache 容量和每步读取量近似按：

$$
\frac{H_{kv}}{H}
$$

缩小。课程引用的经验结果是：MQA 有时带来小幅 perplexity 损失，而 GQA 往往可以做到低损失或接近无损。

> [!question] 已处理：GQA/MQA 的 arithmetic intensity 推导
> - [x] #ai-review
> - 原始问题：补充视频里 GQA/MQA 的 arithmetic intensity 计算公式推导。
> - 结论：decode 时，MHA 的累计 KV 读取项是 $O(bn^2d)$；MQA 将其中的 $d$ 降为单个 KV head 的 $k=d/h$，GQA 则降为 $gk$。

#### 7.2.1 先统一符号和口径

令：

- $b$：batch size；
- $n$：处理或生成的序列长度；
- $d$：model hidden dimension；
- $h$：query head 数；
- $k=d/h$：每个 head 的维度；
- $g$：GQA 的 KV head 数，$1\le g\le h$。

Arithmetic intensity（AI）粗略定义为：

$$
\mathrm{AI}
=
\frac{\text{arithmetic operations}}
{\text{memory accesses}}
$$

讲义计算的是数量级，并假设 $n<d$。这里的 memory access 是按元素规模估算，不是精确
HBM transaction；常数、数据类型字节数、kernel fusion、cache residency、
FlashAttention tiling、量化和 tensor-parallel 通信都被省略。

#### 7.2.2 一次处理所有 token：training / prefill

线性 projection 的主要计算量是 $O(bnd^2)$；attention score/value 还有
$O(bn^2d)$。在 $n<d$ 的假设下，前者占主导，因此分子写成：

$$
\text{operations}=O(bnd^2)
$$

讲义把主要 memory access 拆成：

$$
O(\underbrace{bnd}_{\text{activations}}
+\underbrace{bhn^2}_{\text{attention matrix}}
+\underbrace{d^2}_{\text{weights}})
$$

所以：

$$
\begin{aligned}
\mathrm{AI}_{\text{all-token}}
&=
\frac{bnd^2}{bnd+bhn^2+d^2}\\
&=
\left(
\frac1d+\frac{hn}{d^2}+\frac1{bn}
\right)^{-1}\\
&=
\left(
\frac1d+\frac{n}{kd}+\frac1{bn}
\right)^{-1}\\
&=
O\left(
\left(\frac1k+\frac1{bn}\right)^{-1}
\right).
\end{aligned}
$$

最后一步使用 $h=d/k$ 和 $n<d$ 做数量级简化。直觉是：一次同时处理的 token 越多，
同一份 $d^2$ 权重就能被更多 token 复用，$\frac1{bn}$ 项越小，AI 越高；这也是
training/prefill 的大矩阵通常较容易把 GPU 算力喂满的原因。

#### 7.2.3 MHA 的 incremental decoding

decode 不能一次知道未来 $n$ 个 token，只能顺序执行 $n$ 步。第 $t$ 步要读取前
$t$ 个位置的 K/V，约为 $O(btd)$；累计得到：

$$
\sum_{t=1}^{n}O(btd)=O(bn^2d)
$$

每一步还要读取约 $O(d^2)$ 的 projection/MLP weights。权重可以被 batch 中的请求共享，
但下一 token 仍要重新流过模型；累计为 $O(nd^2)$。于是：

$$
\begin{aligned}
\mathrm{AI}_{\mathrm{MHA,decode}}
&=
\frac{bnd^2}{bn^2d+nd^2}\\
&=
\left(\frac nd+\frac1b\right)^{-1}.
\end{aligned}
$$

这里两个瓶颈很直观：

- context 越长，累计 KV 读取的 $\frac nd$ 项越大；
- batch 越小，不能充分摊薄每步权重读取，$\frac1b$ 项越大。

因此 decode 常是 memory-bandwidth-bound，而不是 compute-bound。

#### 7.2.4 MQA 为什么提高 AI

MHA 的 $h$ 个 KV heads 总宽度为 $hk=d$。MQA 让所有 query heads 共享一个 KV head，
所以每个位置的 K/V 宽度从 $d$ 降为 $k=d/h$。累计 KV 读取变为：

$$
O(bn^2k)
$$

加上较小的 activation 项 $O(bnd)$ 和不变的逐步 weight 读取 $O(nd^2)$：

$$
\begin{aligned}
\mathrm{AI}_{\mathrm{MQA,decode}}
&=
\frac{bnd^2}{bnd+bn^2k+nd^2}\\
&=
\left(
\frac1d+\frac{nk}{d^2}+\frac1b
\right)^{-1}\\
&=
\left(
\frac1d+\frac{n}{dh}+\frac1b
\right)^{-1}.
\end{aligned}
$$

与 MHA 的 $\frac nd$ 相比，KV 项变成 $\frac{n}{dh}$，缩小约 $h$ 倍。注意：

- query heads 仍然是 $h$ 个，attention 仍需为每个 query head 计算结果；
- 大部分模型权重和 projection/MLP compute 没有随之缩小；
- 主要收益是 KV Cache 容量和 HBM traffic，而不是训练 FLOPs 同比例减少。

#### 7.2.5 GQA 是 MHA 与 MQA 之间的连续旋钮

GQA 保留 $g$ 个 KV heads，每个宽 $k$，所以累计 KV 读取为：

$$
O(bn^2gk)
$$

对应：

$$
\mathrm{AI}_{\mathrm{GQA,decode}}
\approx
\left(
\frac1d+\frac{gn}{dh}+\frac1b
\right)^{-1}
$$

- $g=h$ 时，$\frac{gn}{dh}=\frac nd$，退化为 MHA；
- $g=1$ 时，得到 MQA；
- $1<g<h$ 时，用更大的 KV Cache 换取比 MQA 更高的表达容量。

所以选择 $g$ 本质上是在调节 **quality / KV capacity / decode bandwidth**，而不是简单
追求“head 越少越快”。当 $\frac1b$ 的 weight-read 项占主导时，再压缩 KV heads 的
边际收益也会下降。

### 7.3 Sliding-window 与 hybrid attention

Full attention 的 prefill 成本随长度近似二次增长。Sliding-window attention（SWA）只允许每个 token 读取局部窗口，降低长上下文成本，但单层不能直接传播全局信息。

近期常见方案是交错使用：

```text
SWA → SWA → SWA → Full → ...
```

> [!note] 已调整范围：最新模型的 attention
> - [x] #ai-review
> - 原始需求：扩展 K3、DeepSeek 等最新开源模型的 attention 特点。
> - 处理：按审阅决定，本讲不展开；移至 Lecture 04 的 attention 专题统一比较，避免在这里重复。

例如每 4 层放一个 full-attention layer。这样：

- 局部层承担大部分计算；
- 周期性 full attention 提供全局信息通道；
- 多层堆叠让局部信息逐步传播；
- 可以组合 NoPE、RoPE 和 SWA，让不同层负责不同距离范围。

课程列举的相关模型包括 Cohere Command A、Llama 4、Gemma 3/4、OLMo 3 和 Qwen 3.5/Next。

## 8. AI Infra 视角

### Shape

- residual stream 始终保持 $[B,L,d]$，否则不能做 residual add；
- FFN 的上投影临时扩展到 $d_{ff}$，GLU 同时产生 gate 和 value 两条支路；
- MHA/GQA/MQA 的主要区别体现在 $H_{kv}$，而不在输出 residual shape；
- vocabulary size 直接决定 embedding/LM head 的 $[V,d]$ 规模。

### Compute

- 普通 FFN 的主要矩阵计算约与 $2BLdd_{ff}$ 成正比，GLU 有三次主要 projection；
- full attention 的 score/value 部分随 $BL^2d$ 增长；
- SWA 将每层 attention 范围从 $L$ 限制到窗口 $W$，对应项约变为 $BLWd$；
- 超参数存在较宽的 good basin，因此可以在质量接近时优先选择硬件友好的 shape。

### Memory

- RMSNorm 和去 bias 的收益主要来自减少低 arithmetic-intensity 的数据搬运；
- decode 时，KV Cache 通常比 attention FLOPs 更直接地决定 batch capacity；
- GQA/MQA 通过减少 $H_{kv}$ 同时降低 cache 容量与 HBM 读取；
- 大 vocabulary 增加 embedding、LM head 和 optimizer state。

### Communication

- 深模型有更长的顺序依赖链，pipeline bubble 和 latency 更难隐藏；
- 宽模型更适合 tensor parallel，但会增加设备间 collective 通信；
- GQA 还可能改变 tensor-parallel 下 KV head 的切分方式和通信粒度。

### Runtime/System

- FLOPs 不是 runtime；Norm、bias、activation 等操作应结合 memory bandwidth 和 fusion 分析；
- “质量相近”时，架构选择应进一步比较 kernel 成熟度、显存占用、并行效率和 serving workload；
- 训练最稳的结构、训练 FLOPs 最低的结构、decode 最快的结构不一定是同一个结构。

## 9. 一个可执行的默认配置

如果目标是先实现一个可靠的现代 dense decoder-only LM，可以从下面的配置开始：

| 设计维度 | 建议基线 | 主要理由 |
| --- | --- | --- |
| Residual/Norm | Pre-RMSNorm | 保留 identity path；实现和 runtime 成熟 |
| Linear bias | 关闭 | 收益不明确，减少逐元素操作和状态 |
| FFN | SwiGLU，$d_{ff}\approx 8d/3$ | 现代模型共识，参数量与 $4d$ 普通 FFN 接近 |
| Position | RoPE | 相对位置性质清晰，生态成熟 |
| Attention | GQA | serving 质量与 KV Cache 成本的稳健折中 |
| Head shape | $H d_h=d$ | 经验稳定的默认值 |
| Dropout | 预训练默认关闭 | 大数据、单 epoch 场景中收益有限 |
| Weight decay | 保留并联调 LR schedule | 主要作为 optimization 旋钮 |
| Stability | 监控 logits/gradients；必要时加 z-loss、QK Norm 或 soft-cap | 直接干预两个 softmax 风险点 |
| Long context | 先 full attention；受成本约束时评估 SWA/full interleave | 避免过早牺牲全局信息路径 |

这不是“最佳架构”，而是一个低风险起点。偏离默认值时，应明确想改善的是：

1. 模型质量；
2. 训练稳定性；
3. 训练吞吐；
4. prefill 延迟；
5. decode 吞吐；
6. KV Cache 容量；
7. 长上下文能力。

## 10. 待审阅 / 待深入问题

> [!question] Q-CS336-L03-3042：RoPE 为什么能产生相对位置依赖？
> - [ ] #question 当前笔记只给出旋转内积恒等式；后续补齐不同频率、pairing 和实际 tensor 实现。
> - 来源：[视频 30:42](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1842s)
> - 预计沉淀到：[Transformer Block](../../topics/model-architecture/Transformer%20Block.md)

^q-cs336-l03-3042-rope

> [!question] Q-CS336-L03-4644：深宽比的质量最优点和系统最优点是否一致？
> - [ ] #question 需要结合具体 parameter budget、parallel strategy 和 latency target 比较，而不是只记录 $d/n_{layers}\approx100$–$200$。
> - 来源：[视频 46:44](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2804s)

^q-cs336-l03-4644-aspect-ratio

> [!question] Q-CS336-L03-5512：Weight decay 如何与 cosine LR schedule 交互？
> - [x] #question 已补充 AdamW 更新式、累计 shrinkage 与 cosine schedule 的交互，并区分优化动力学和传统 regularization 解释。见 [[#5.5 Dropout 与 weight decay]]。
> - 来源：[视频 55:12](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3312s)

^q-cs336-l03-5512-weight-decay

> [!question] Q-CS336-L03-5923：QK Norm、z-loss 与 soft-cap 应该如何选择？
> - [x] #question 已按触发信号、生效位置、额外开销和风险整理选择顺序。见 [[#6.4 z-loss、QK Norm 与 soft-cap 怎么选]]。
> - 来源：[视频 59:23](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3563s)

^q-cs336-l03-5923-stability

## 11. 本讲结论

1. 现代 dense LLM 的主干已经高度收敛：Pre-RMSNorm、无 bias、GLU FFN、RoPE 是可靠基线。
2. 超参数多数存在宽阔的 good basin；系统效率经常比小范围 loss 差异更能决定最终 shape。
3. Stability trick 主要保护 attention 和 output 两个 softmax，代表方法是 z-loss、QK Norm 和 logit soft-capping。
4. 推理阶段的核心约束是 KV Cache 和内存带宽；GQA/MQA 的价值主要来自减少 K/V 状态，而不是改变 Query 数量。
5. 长上下文 attention 正从“所有层都 full”转向 full 与 local/sliding-window 的混合。

## 12. 自测问题

1. 为什么 Pre-Norm 的关键是 Norm 不处于 residual signal path，而不只是“位置靠前”？

    **面试回答：** Pre-Norm 写成 xₗ₊₁=xₗ+F(Norm(xₗ))，主 residual 支路保留直接的 identity path，梯度包含一条不必经过 Norm Jacobian 的通路。Post-Norm 则对相加后的整体归一化，主信号也被变换；关键是计算图里的梯度路径，而不只是 Norm 在代码中的先后位置。

2. RMSNorm 只减少很少的 FLOPs，为什么仍可能明显改善 wall-clock time？

    **面试回答：** Norm 往往是低算术强度算子，时间主要花在 reduction、显存搬运和 kernel launch，FLOPs 占比不能直接代表耗时。RMSNorm 去掉均值中心化、实现更简单，可能减少归约与中间读写并更容易融合；实际收益取决于 kernel 和模型 shape。

3. 为什么普通 FFN 常用 $d_{ff}=4d$，而等参数量 GLU 常用 $d_{ff}\approx8d/3$？

    **面试回答：** 普通 FFN 两个 projection 的参数量约为 2dd_ff，d_ff=4d 时是 8d²；GLU 多一个 gate projection，约为 3dd_ff。要保持相近参数与主要计算量，令 3dd_ff=8d²，得到 d_ff≈8d/3；4d 本身是经验基线，实际还会为硬件对齐调整。

4. 用旋转矩阵说明 RoPE 的 Q/K 点积为何只依赖相对位置。

    **面试回答：** RoPE 在每对坐标上使用按位置旋转的矩阵 R(m)，满足 R(i)ᵀR(j)=R(j−i)。因此 (R(i)q)ᵀ(R(j)k)=qᵀR(j−i)k，位置项只显式依赖相对距离；多频率时各二维块都满足同样性质，但 q、k 的内容仍可能间接包含上下文位置信息。

5. 为什么 $n_{heads}d_{head}=d_{model}$ 是经验约定而不是数学约束？

    **面试回答：** Q/K/V projection 可以把 d_model 映射到任意合理的多头总宽度 H·d_head，最后再用输出 projection 映射回 d_model，残差相加只要求最终维度一致。H·d_head=d_model 是平衡参数量、表达能力和硬件效率的常见默认值，并非 attention 数学定义的限制。

6. 大 vocabulary 如何同时影响序列长度、LM head 计算和 optimizer memory？

    **面试回答：** 在固定原始文本上，大词表通常减少 token 数，从而降低 block、attention 和 KV 的序列长度成本；但 embedding/LM head 参数随 Vd 增加，每个位置的输出 projection 随 dV 增加。新增词表参数还对应梯度和 Adam moments，因此参数与 optimizer memory 一起增长，最终计算取决于 T 和 V 的共同变化。

7. z-loss、QK Norm 和 logit soft-cap 分别控制哪个数值风险？

    **面试回答：** z-loss 用 α(logΣexp u)² 约束输出 softmax 的 log-normalizer 漂移；QK Norm 直接规范 Q/K 尺度，缓解 attention logits 过大。Soft-cap 用 c·tanh(z/c) 限制极端 logits，但可能压低梯度并限制尖锐分布；三者针对的位置不同，不能替代 LR、精度和数据排障。

8. 为什么 GQA 对 decode 的收益通常比对训练 FLOPs 的收益更重要？

    **面试回答：** GQA 让多组 query heads 共享较少的 KV heads，KV Cache 容量与读取量约按 H_kv/H 缩小，直接缓解 decode 的显存和带宽瓶颈。Query heads 及大部分 MLP 计算仍然保留，因此训练 FLOPs 不会同比例下降；KV 不占主要瓶颈时，decode 的收益也会变小。

9. Full/SWA interleave 如何在降低成本的同时保留全局信息路径？

    **面试回答：** 多数层用窗口 W 的局部 attention，把对应 mixing 成本从 O(L²d) 降到 O(LWd)，隔若干层插入 full attention 提供直接的全局信息通道。局部层堆叠也扩大感受野，但这种混合不保证保留全 full attention 的所有能力，仍需评估长距离检索和实际吞吐。

10. Dropout 和 AdamW weight decay 分别作用于激活还是参数？为什么现代 LLM 对二者的取舍不同？

    **面试回答：** Dropout 在训练中随机置零并缩放激活，增加随机正则化；AdamW weight decay 则按 θ←(1−ηλ)θ 持续收缩参数。大数据、近单 epoch 的 LLM 预训练中 dropout 的防过拟合收益可能有限且增加噪声，而适当 weight decay 仍可改善参数尺度和优化动力学，二者都需结合训练场景判断。

11. 从第 $t$ 步读取 $O(btd)$ KV 状态出发，推导 MHA、MQA 和 GQA decode 的 arithmetic intensity。

    **面试回答：** 令 $n$ 为 decode 步数、$h$ 为 query heads、$g$ 为 KV heads、$k=d/h$，按正文假设 $n<d$、忽略常数和 dtype。MHA 累计 KV 读取为 $\sum_{t=1}^nO(btd)=O(bn^2d)$，权重读取为 $O(nd^2)$，计算约 $O(bnd^2)$，故 $\mathrm{AI}\approx(n/d+1/b)^{-1}$；MQA 和 GQA 把 KV 宽度分别改为 $k$ 和 $gk$，保留 $O(bnd)$ 激活项后分别为 $(1/d+n/(dh)+1/b)^{-1}$ 与 $(1/d+gn/(dh)+1/b)^{-1}$。这是 operations/元素访问的数量级，换成 FLOP/byte 还要计入元素字节数。


## 参考资料

- [CS336 Spring 2026 课程主页](https://cs336.stanford.edu/)
- [Lecture 3 视频](https://www.youtube.com/watch?v=lVynu4bo1rY)
- [Lecture 3 官方讲义](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)
- [Why Do We Need Weight Decay in Modern Deep Learning?](https://arxiv.org/abs/2310.04415)
- [Small-scale Proxies for Large-scale Transformer Training Instabilities](https://arxiv.org/abs/2309.14322)
- [PaLM：Pathways Language Model（z-loss）](https://arxiv.org/abs/2204.02311)
- [Fast Transformer Decoding: One Write-Head is All You Need（MQA）](https://arxiv.org/abs/1911.02150)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Transformer Architecture](../../topics/model-architecture/Transformer%20Architecture.md)
- [Transformer Block](../../topics/model-architecture/Transformer%20Block.md)
