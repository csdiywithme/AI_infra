---
type: course-note
status: developing
course: "[[CMU 11-763]]"
lecture: 1
lecture_date: 2025-08-26
area: inference
topics:
  - "[[LLM Inference]]"
  - "[[Transformer Block]]"
aliases:
  - CMU 11-763 Lecture 01
video_url: https://www.youtube.com/watch?v=F-mduXzNcRQ
---

# Lecture 01：Introduction to Language Models and Inference

## 课程信息

- 课程：CMU 11-664/763 — Inference Algorithms for Language Modeling
- 讲师：Graham Neubig
- 日期：2025-08-26
- [课程视频](https://www.youtube.com/watch?v=F-mduXzNcRQ)
- [课程讲义](https://www.phontron.com/class/lminference-fall2025/assets/slides/2025-08-26-lm-intro/index.html)
- 指定阅读：[From Decoding to Meta-Generation: Inference-time Algorithms for Large Language Models](https://arxiv.org/abs/2406.16838)，Sections 1–2

## 视频索引

| 时间 | 视频内容 | 对应笔记 |
| --- | --- | --- |
| [00:00](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=0s) | Introduction and modeling | [本讲目标](#1-本讲目标)、[语言模型](#2-语言模型) |
| [23:20](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=1400s) | Hardware for inference | [Transformer 建模与计算成本](#3-transformer-建模与计算成本) |
| [30:21](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=1821s) | Training vs. inference | [训练与推理](#4-训练与推理) |
| [37:55](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=2275s) | Generation algorithms | [两类基本生成方法](#5-两类基本生成方法)、[Basic Generation 与 Meta-generation](#6-basic-generation-与-meta-generation) |
| [41:52](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=2512s) | Latent variables | [中间变量与推理轨迹](#7-中间变量与推理轨迹) |
| [44:51](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=2691s) | Evaluation and errors | [模型概率与任务质量](#8-模型概率与任务质量)、[Search Error 与 Model Error](#9-search-error-与-model-error) |
| [51:28](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=3088s) | 学生提问：Inference 的目标是 sampling 还是 optimization？ | [Inference 的 Goal](#10-inference-的-goalsample-还是-optimize) |
| [56:04](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=3364s) | Diversity, course goals, and Q&A | [推理的多目标权衡](#11-推理的多目标权衡) |

## 1. 本讲目标

第一讲主要建立后续课程需要的概念边界：

1. 语言模型定义了什么？
2. 训练与推理分别解决什么问题？
3. 什么是 generation algorithm 与 meta-generation algorithm？
4. 模型概率为什么不等于任务质量？
5. 如何区分 search error 与 model error？
6. 推理算法如何权衡质量、延迟、吞吐、多样性和成本？

> [!abstract] 核心心智模型
> 语言模型定义一个 token 序列上的概率分布；推理算法负责在有限计算预算下使用这个模型产生符合应用目标的输出。

## 2. 语言模型

### 2.1 序列概率

对于 token 序列：

$$
x=(x_1,x_2,\ldots,x_L)
$$

语言模型为其分配概率：

$$
P_\theta(x)
$$

现代 LLM 最常见的是自回归语言模型。利用概率链式法则：

$$
P_\theta(x)
=
\prod_{t=1}^{L}P_\theta(x_t\mid x_{<t})
$$

其中：

- $x_{<t}$：当前位置之前的 token；
- $\theta$：模型参数；
- $P_\theta(x_t\mid x_{<t})$：给定历史前缀后的 next-token distribution。

### 2.2 条件生成

实际应用通常给定 prompt $x$，要求模型生成 response $y$：

$$
P_\theta(y\mid x)
=
\prod_{t=1}^{|y|}
P_\theta(y_t\mid x,y_{<t})
$$

Prompt 是 prefix/query，response 是 completion。改变 prompt 会改变条件分布，但不改变模型参数。

### 2.3 序列概率的长度效应

完整序列概率是多个小于 1 的条件概率之积。使用 log probability 时：

$$
\log P_\theta(y\mid x)
=
\sum_t\log P_\theta(y_t\mid x,y_{<t})
$$

序列越长，累积 log probability 通常越负。因此，直接最大化未经校正的完整序列概率容易偏向短而普通的输出。

这说明：

> Maximum-probability sequence 不一定是人类或任务指标认为最好的 sequence。

## 3. Transformer 建模与计算成本

语言模型需要实现：

$$
P_\theta(x_t\mid x_{<t})
$$

当前主流实现是 decoder-only Transformer。每个 block 主要包含：

- Self-Attention：在 token 之间传递信息；
- FFN/MLP：在单个 token 内加工特征；
- Norm：稳定模块输入尺度；
- Residual：保存并累积各模块的更新。

详细的 tensor shape、MHA/GQA、SwiGLU、KV Cache 和 FLOPs 推导见：[Transformer Block](../../topics/model-architecture/Transformer%20Block.md)。

### 3.1 讲义中的计算规律

对于完整长度为 $L$ 的序列：

- Q/K/V/O projections 随 $L$ 线性增长；
- FFN 随 $L$ 线性增长，但矩阵通常很宽；
- 完整 attention 的 $QK^T$ 与 $AV$ 包含 $L^2$ 项；
- 总成本随 Transformer 层数 $N$ 近似线性增长。

因此短上下文中 FFN 常占据主要 FLOPs；上下文足够长时，attention 的 $L^2$ 项逐渐变得显著。

### 3.2 Infra 补充：Prefill 与 Decode

讲义中的完整 $L\times L$ attention costing 主要对应 full-sequence forward/prefill。

使用 KV Cache 的自回归 decode 中，每一步只有一个新 Query：

$$
Q_{new}:[B,H_q,1,d_h]
$$

它读取长度为 $L$ 的历史 K/V：

$$
K_{cache},V_{cache}:[B,H_{kv},L,d_h]
$$

因此单步 attention 随历史长度近似线性增长。但 decode 具有串行依赖、矩阵较瘦，并且反复读取权重和 KV Cache，通常更容易受到显存带宽与调度效率限制。

## 4. 训练与推理

### 4.1 训练

训练希望学习参数 $\theta$，使模型分布逼近数据分布。典型目标是最大似然估计：

$$
\theta^*
=
\arg\max_\theta
\sum_{x\in\mathcal D}\log P_\theta(x)
$$

工程实现通常最小化 negative log-likelihood/cross-entropy，包括：

1. Forward：计算预测与 loss；
2. Backward：计算梯度；
3. Optimizer step：更新参数。

训练改变模型参数和模型分布。

### 4.2 推理

推理假定参数基本固定，目标是根据输入 $x$ 产生输出 $y$。它可能包括：

- 多次调用语言模型；
- 维护生成状态与 KV Cache；
- 选择扩展哪些 token 或候选序列；
- 决定停止条件；
- 使用 verifier、reward model、reranker 或外部工具；
- 在输出质量与计算预算之间做决策。

推理并不等价于单次 Transformer forward；它是围绕模型组织的一套生成和决策过程。

## 5. 两类基本生成方法

### 5.1 Sampling

从模型分布中采样：

$$
y\sim P_\theta(y\mid x)
$$

逐 token 表示为：

$$
y_t\sim P_\theta(\cdot\mid x,y_{<t})
$$

特点：

- 具有随机性和较高多样性；
- 可以探索不同模式；
- 单个样本的质量与可靠性可能较低。

Temperature、top-k 和 top-p 都是在调整或截断采样分布。

### 5.2 Search

近似寻找某个评分函数下的最优输出：

$$
\hat y
\approx
\arg\max_y s_\theta(y\mid x)
$$

评分函数 $s_\theta$ 不一定等于语言模型概率，还可以包含：

- length penalty；
- reward/verifier 分数；
- 格式和语法约束；
- 业务规则；
- 多模型组合分数。

Greedy decoding、beam search、best-first search 与 A* 可以理解为不同的搜索策略。

## 6. Basic Generation 与 Meta-generation

### 6.1 Basic generation

基本生成逐 token 调用模型：

```python
def generate(model, x):
    y = []
    while not done(y):
        distribution = model(x + y)
        y.append(select(distribution))
    return y
```

`select` 可以是 greedy、sampling 或搜索算法的一部分。

### 6.2 Meta-generation

Meta-generation 将“生成部分或完整 token 序列”作为子程序，再在候选之上执行进一步计算。

典型的 generate-and-rerank：

$$
y^{(1)},\ldots,y^{(N)}\sim P_\theta(y\mid x)
$$

$$
\hat y
=
\arg\max_{y^{(i)}}r(x,y^{(i)})
$$

常见方法包括：

- Best-of-N；
- self-consistency；
- Minimum Bayes Risk；
- verifier-guided search；
- iterative refinement；
- tool-use 与 agent workflow。

Meta-generation 体现了 inference-time compute 的核心思想：固定模型并不意味着固定生成质量，可以通过增加候选、反馈、搜索和外部计算提高结果质量。

## 7. 中间变量与推理轨迹

最终答案 $y$ 之外，模型可能生成中间推理过程 $z$：

$$
P(y\mid x)
=
\sum_zP(y,z\mid x)
$$

其中 $z$ 可以是：

- Chain of Thought；
- scratchpad；
- 搜索轨迹；
- 工具调用记录；
- agent 的中间状态。

实际系统通常无法枚举并严格边缘化所有 $z$，而是采样或搜索少量轨迹。这引出：

- reasoning token budget；
- 多分支搜索；
- 中间步骤验证；
- 轨迹选择与答案聚合；
- 质量提升与额外延迟之间的权衡。

## 8. 模型概率与任务质量

分析 inference 时需要分开三个层次：

1. **模型分布** $P_\theta(y\mid x)$：模型认为不同输出的概率；
2. **推理算法** $A(\theta,x,C)$：在计算预算 $C$ 下产生输出或样本；
3. **外部任务目标** $r(y\mid x)$：应用真正关心的质量。

语言模型提供概率或模型分数，但应用真正关心外部任务价值：

$$
r(y\mid x)
$$

评价信号可能来自：

- 人类偏好；
- LLM-as-a-Judge；
- accuracy；
- BLEU、ROUGE；
- 单元测试或程序执行；
- 数学 verifier；
- 安全、格式与业务规则。

理想情况下 $s_\theta(y\mid x)$ 与 $r(y\mid x)$ 一致，但实际中经常错位。

因此推理优化的完整目标不是“更快找到最高概率序列”，而是：

> 在有限计算预算下，找到外部任务价值更高的输出。

## 9. Search Error 与 Model Error

### 9.1 Search error

先定义模型评分下的最优输出：

$$
y_s=\arg\max_y s_\theta(y\mid x)
$$

如果实际算法返回 $y_{alg}$，但没有找到这个最高分输出：

$$
s_\theta(y_{alg}\mid x)
<
s_\theta(y_s\mid x)
$$

那么存在 search error。可以定义 search regret：

$$
R_{search}
=
s_\theta(y_s\mid x)-s_\theta(y_{alg}\mid x)
$$

可能原因：

- greedy 过早选择局部最优 token；
- beam 太小；
- 搜索被错误剪枝；
- token/compute budget 不足；
- 没有探索到更好的推理路径。

主要改进方向是更好的 search/inference algorithm。

一个典型例子是 greedy decoding 的局部选择不等于 sequence-level 全局最优：

```text
起点
├── A：0.6
│   └── A1：0.1    完整序列概率 0.06
└── B：0.4
    └── B1：0.9    完整序列概率 0.36
```

Greedy 第一步选择 A，但全局 MAP 序列是 B1。更大的 beam、best-first search 或合适的 lookahead 可能修复这个问题。

### 9.2 Model error

即使找到模型评分最高的输出，它也不是外部指标下最好的输出：

$$
\hat y=\arg\max_y s_\theta(y\mid x)
$$

但：

$$
r(\hat y\mid x)<\max_y r(y\mid x)
$$

主要改进方向包括：

- 改进训练或对齐；
- reward model/verifier；
- reranking；
- constrained decoding；
- 多样本生成；
- 外部反馈和工具。

### 9.3 诊断表

| 现象 | 主要问题 |
| --- | --- |
| 高质量答案的模型分数高，但搜索没有找到 | Search error |
| 搜索找到模型最高分答案，但答案质量差 | Model error |
| 候选集中有好答案，但 reranker 选错 | Scoring/model error |
| 候选集中没有好答案 | Generation/search coverage error |

必须先确定错误类型。盲目扩大 beam 或增加采样数，可能只是在更昂贵地优化错误目标。

### 9.4 更精确的 Search 不一定提高任务质量

假设：

| 输出 | 模型分数 $s$ | 外部奖励 $r$ |
| --- | ---: | ---: |
| 错误答案 | -1.0 | 0 |
| 正确答案 | -2.0 | 1 |

模型更偏好错误答案。如果近似搜索意外返回正确答案，那么它相对模型分数存在 search error，却获得更高任务奖励。扩大 beam 并找到模型 argmax 后：

- Search error 减少；
- 模型分数提高；
- 外部任务质量反而下降。

因此：

> 减少 search error 只保证更好地优化 $s_\theta$，不保证提高 $r$。

当增加搜索预算使模型分数提高、但外部指标停滞或下降时，主要瓶颈是 score/reward mismatch，而不是搜索能力。

## 10. Inference 的 Goal：Sample 还是 Optimize？

Inference 是上位概念，没有规定必须 sampling 或 optimization。目标取决于应用。

> [!question] Q-CMU763-L01-5128：Inference 的目标是 sampling 还是 optimization？
> - [x] #question 纯 sampling 没有找到 argmax，是否应该被视为 search error？
> - 来源：[视频 51:28](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=3088s)
> - 课堂语境：学生追问 inference 究竟是从分布中采样，还是寻找分布的 mode，以及 sampling 结果为何不应直接套用 search error。
> - 处理结果：需要先声明 inference 的目标；纯 sampling 应评价分布匹配，而不是是否找到 argmax。完整结论见本节。

^q-cmu763-l01-5128-sample-optimize

### 10.1 忠实采样

如果目标是：

$$
y\sim P_\theta(y\mid x)
$$

那么算法需要产生符合目标分布的样本，而不是寻找最高概率输出。样本的分数低于 argmax 是正常现象，不构成 search error。

此时更合适的问题是：算法实际产生的分布 $q$ 与目标分布是否一致，以及 diversity、calibration 和 sample efficiency 如何。

Temperature、top-k 和 top-p 通常从经过修改或截断的分布 $q$ 采样。只要目标就是这个 $q$，没有抽到最高概率 token 不是错误。

### 10.2 优化模型分数

如果目标是：

$$
\hat y=\arg\max_y\log P_\theta(y\mid x)
$$

那么这是 MAP decoding，属于组合搜索问题。此时可以讨论 greedy、beam、A* 是否找到目标，以及是否存在 search error。

但 MAP sequence 不一定是模型分布中具有代表性的样本，也不一定具有最高外部任务质量。

### 10.3 优化外部奖励

很多应用真正希望：

$$
\hat y=\arg\max_y r(y\mid x)
$$

例如代码通过测试、数学答案正确或输出满足 schema。由于外部奖励往往只能在完整生成后计算，常见策略是先生成候选，再使用 verifier/reward 选择。

### 10.4 Sampling 与 Optimization 可以组合

Sampling 可以负责探索，optimization/aggregation 负责最终决策。

**Best-of-N：**

$$
y_1,\ldots,y_N\sim P_\theta(y\mid x)
$$

$$
\hat y=\arg\max_i r(y_i\mid x)
$$

**Self-consistency：**采样多个 reasoning paths，再对最终答案聚合或投票。

**Minimum Bayes Risk：**使用 samples 近似期望风险，再选择期望损失最低的输出：

$$
\hat y
\approx
\arg\min_{y_i}
\frac1N\sum_{j=1}^{N}\Delta(y_i,y_j)
$$

因此，对[视频 51:28](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=3088s)学生问题的准确结论是：

> Sampling 和 optimization 不是互斥答案。Inference 可以只做其中一种，也可以用 sampling 探索输出空间，再用 optimization、reranking 或 aggregation 做决策。Search/model error 的定义主要适用于已经指定 sequence-level 优化目标的场景，不能直接套在纯 sampling 上。

## 11. 推理的多目标权衡

推理系统通常同时关注：

- Quality；
- Latency；
- Throughput；
- Diversity；
- Memory；
- Monetary/energy cost。

不同算法有不同权衡：

| 方法 | 典型特征 |
| --- | --- |
| Greedy | 延迟较低，多样性低，可能局部最优 |
| Sampling | 多样性高，单样本不稳定 |
| Beam search | 扩大搜索范围，但增加计算和状态管理 |
| Best-of-N | 可能提高质量，但生成成本近似随 N 增加 |
| 长 CoT | 可能提高推理质量，但增加 token 成本和尾延迟 |

不存在脱离 workload 的“最佳推理算法”。交互式聊天、代码生成、数学推理和离线批处理的目标函数不同。

## 12. 本讲结论

1. Language model 是概率模型，inference algorithm 是使用模型产生结果的算法。
2. 训练决定模型分布，推理决定如何在固定模型上分配计算。
3. 最高模型概率不等于最高任务质量。
4. 必须区分 search error 与 model error。
5. Generation 可以作为 meta-generation 的子程序。
6. 推理优化是质量、延迟、吞吐、多样性、内存与成本之间的系统性权衡。
7. Sampling 与 optimization 是不同目标，也可以组成 sample-then-optimize 系统。
8. Search error 的减少不保证外部任务质量提高。

> [!tip] 面向 AI Infra 的进一步理解
> LLM inference optimization 不只是优化一次 Transformer forward，而是联合优化模型执行、缓存、内存流量、请求调度、搜索策略、候选管理与任务级评价目标。

## 13. 自测问题

1. 为什么完整序列的最大模型概率可能偏向短输出？— 视频起点：[00:00 Introduction and modeling](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=0s)

    **面试回答：** 完整序列概率是逐 token 条件概率连乘，logprob 则累加非正数，所以直接最大化它常带来长度偏好，尤其模型过早给 EOS 高概率时。比较不同完整序列还要计入 EOS，不能说短文本必然概率更高；长度归一化能缓解偏好，但已改变搜索目标。

2. Sampling 与 search 分别在优化或近似什么？— 视频：[32:44 Sampling 与 Search](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=1964s)

    **面试回答：** Sampling 的目标是按指定分布产生样本，用样本近似分布、期望或答案边缘概率；search 则试图找到指定评分函数的高分或最优输出。评分可取模型概率，也可取 reward；采到的输出不是 argmax，并不意味着 sampling 出错。

3. 为什么搜索分数不一定等于语言模型概率？— 视频起点：[37:55 Generation algorithms](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=2275s)

    **面试回答：** 语言模型概率描述模型如何给序列分配概率质量，而搜索分数可以额外加入长度惩罚、约束、reward model 或 verifier 结果。例如平均 token logprob 与总 logprob 的排序可能不同；评价 search error 前必须先说清实际优化哪一个分数。

4. Generation 与 meta-generation 的边界是什么？— 视频起点：[37:55 Generation algorithms](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=2275s)

    **面试回答：** Basic generation 是用模型和解码规则产生一条候选，例如 greedy、sampling 或 beam search。Meta-generation 在其外层组织多次生成、评分、选择、投票或改写，例如 best-of-N、自一致性和生成—验证循环；边界在于是否额外编排候选与反馈。

5. 为什么 Chain of Thought 可以视为中间变量？— 视频：[41:52 Latent variables](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=2512s)

    **面试回答：** 把推理轨迹记作 z、最终答案记作 y，就有 $P(y\mid x)=\sum_z P(z\mid x)P(y\mid x,z)$。轨迹帮助模型生成答案，但最终任务通常只关心 y，因此 z 可作为被边缘化的中间变量；选最可能的一条轨迹不等于选边缘概率最大的答案。

6. 如何通过候选集判断 search error 与 model error？— 视频：[54:46 课堂追问](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=3286s)

    **面试回答：** 先比较候选的模型分数与外部质量：若存在比返回答案模型分数更高的候选，说明搜索未优化好指定分数；若高质量候选被评分器排低，说明 scoring/model mismatch。候选全差只能说明覆盖或模型能力有问题，有限候选集不能证明全局最优或排除 search error。

7. 为什么扩大 inference-time compute 不一定提高质量？— 视频起点：[44:51 Evaluation and errors](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=2691s)

    **面试回答：** 更多计算可能扩大候选覆盖，却无法自动修正错误的评分目标或 verifier。候选高度相关、评分器偏差或过度优化代理分数，都可能让质量停滞甚至下降；要看预算增加后的外部任务质量、延迟与成本，而不是只看模型分数。

8. Prefill 与使用 KV Cache 的 decode 在计算形态上有何差异？— 课程背景：[23:20 Hardware for inference](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=1400s)；具体回答属于 Infra 延伸

    **面试回答：** Prefill 一次处理多个 prompt token，线性层是较大 GEMM，权重复用充分，dense attention 的总计算随长度平方增长。使用 KV cache 的普通 decode 每请求只处理一个新 token，用它的 Q 读历史 K/V；小 batch 下计算少、反复读权重和 KV，更易受带宽与 launch 开销限制。

9. 为什么纯 sampling 不应按“是否找到 argmax”判断 search error？— 视频：[51:28 学生提问](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=3088s)

    **面试回答：** 纯 sampling 的正确性标准是样本分布是否等于目标分布，而不是每次是否得到 mode。低概率样本本来就应偶尔出现；只有事先定义了最大化某个分数的目标，才适合用未找到 argmax 来讨论 search error。

10. 什么情况下扩大 beam 可能降低外部任务质量？— 视频起点：[44:51 Evaluation and errors](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=2691s)；结论包含扩展推导

    **面试回答：** 当模型分数与外部质量不一致时，扩大 beam 可能更准确地找到模型偏好的短、空泛或错误答案。此时 search error 下降而任务质量下降；应调整评分、约束或模型，而不是继续单纯增加搜索宽度。

11. Best-of-N 中 sampling 和 optimization 分别承担什么职责？— 视频起点：[37:55 Generation algorithms](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=2275s)

    **面试回答：** Sampling 负责生成有一定多样性的候选集，决定搜索覆盖；optimization 负责用 scorer/verifier 从有限集合中选择高分答案。最终质量同时受 generator 的覆盖和 scorer 的排序能力限制，best-of-N 也会改变原始生成分布。


## 14. 关联内容

- 课程索引：[CMU 11-763](CMU%2011-763.md)
- 主题入口：[LLM Inference](../../topics/inference/LLM%20Inference.md)
- 模型执行：[Transformer Block](../../topics/model-architecture/Transformer%20Block.md)
