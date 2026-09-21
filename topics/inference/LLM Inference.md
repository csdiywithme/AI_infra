---
type: topic
status: developing
area: inference
mastery: explain
aliases:
  - LLM 推理
  - Language Model Inference
topics:
  - "[[Transformer Architecture]]"
---

# LLM Inference

> [!abstract] 核心心智模型
> 语言模型定义 token 序列上的概率分布；推理算法负责在有限计算预算下调用模型、管理状态并产生符合应用目标的输出。

## 主题地图

### 模型执行

- [Transformer Block](../model-architecture/Transformer%20Block.md)
- [Prefill 与 Decode](../model-architecture/Transformer%20Block.md#14-prefill-与-decode)
- [KV Cache、Prefix Sharing、GQA/MLA 与压缩](../../courses/cmu-11-763/CMU%2011-763%20-%20Lecture%2020%20-%20Prefix%20Sharing%20and%20KV%20Cache%20Optimizations.md)
- [Sparse Attention、Linear Attention、SSM 与 Hybrid](../../courses/cmu-11-763/CMU%2011-763%20-%20Lecture%2022%20-%20Linearizing%20Attention%20and%20Sparse%20Models.md)

### 生成与决策

- [CMU Lecture 01：Basic Generation 与 Meta-generation](../../courses/cmu-11-763/CMU%2011-763%20-%20Lecture%2001%20-%20Introduction%20to%20Language%20Models%20and%20Inference.md#6-basic-generation-与-meta-generation)
- [Sampling 与 Search](../../courses/cmu-11-763/CMU%2011-763%20-%20Lecture%2001%20-%20Introduction%20to%20Language%20Models%20and%20Inference.md#5-两类基本生成方法)
- [Search Error 与 Model Error](../../courses/cmu-11-763/CMU%2011-763%20-%20Lecture%2001%20-%20Introduction%20to%20Language%20Models%20and%20Inference.md#9-search-error-与-model-error)
- [Temperature 与 Sampling Distribution](../../courses/cmu-11-763/CMU%2011-763%20-%20Lecture%2002%20-%20Probability%20Review%20and%20Code%20Examples.md#4-temperature-改变的是采样分布)
- [Sampling-based Estimation 与 Bias](../../courses/cmu-11-763/CMU%2011-763%20-%20Lecture%2002%20-%20Probability%20Review%20and%20Code%20Examples.md#5-sampling-based-estimation)
- [Reasoning Trace、Marginalization 与 Self-consistency](../../courses/cmu-11-763/CMU%2011-763%20-%20Lecture%2002%20-%20Probability%20Review%20and%20Code%20Examples.md#7-reasoning-trace-是-latent-variable)
- [Reranking、Meta-generation 与 Speculative Decoding 的边界](../../courses/cmu-11-763/CMU%2011-763%20-%20Lecture%2002%20-%20Probability%20Review%20and%20Code%20Examples.md#14-reranking-与-speculative-decoding-的边界)

### Serving 与系统优化

- Continuous Batching
- PagedAttention
- Quantization
- Speculative Decoding
- Tensor Parallel

这些主题将在出现足够课程材料、推导或实验后升级为独立主题笔记；当前不创建空壳文件。

### 从生成分布到系统决策

- [[CMU 11-763 - Lecture 03 - Common Sampling Methods for Modern NLP|Sampling]]：temperature、截断与 typical sampling 改变输出分布。
- [[CMU 11-763 - Lecture 04 - Beam Search and Variants|Beam Search]]：有限搜索预算、停止条件与多样性。
- [[CMU 11-763 - Lecture 05 - A Star and Best First Search|A* 与 Best-First Search]]：启发式、最优性条件与搜索顺序如何影响计算成本。
- [[CMU 11-763 - Lecture 06 - Other Controlled Generation Methods|Controlled Generation]]：语法约束、判别器与逐 token 控制。
- [[CMU 11-763 - Lecture 07 - Chain of Thought and Intermediate Steps|CoT 与 Self-consistency]]：中间计算与答案层面的边缘化。
- [[CMU 11-763 - Lecture 08 - Self-Refine and Self-Correction Methods|Self-correction]]：反馈来源、停止策略与无外部证据时的局限。
- [[CMU 11-763 - Lecture 09 - Reasoning Models|Reasoning Models]]：训练如何使推理时计算更有效。
- [[CMU 11-763 - Lecture 10 - Incorporating Tools|Tool Use]]、[[CMU 11-763 - Lecture 11 - Agents and Multi-Agent Communication|Agents]]：模型外执行、交互状态、预算与评测。
- [[CMU 11-763 - Lecture 12 - Reward Models and Best-of-N|Best-of-N]]、[[CMU 11-763 - Lecture 14 - Minimum Bayes Risk and Multi-Sample Strategies|MBR]]：候选生成与决策效用不是同一件事。
- [[CMU 11-763 - Lecture 15 - Inference Scaling vs Model Size|Inference Scaling]]：在同一资源口径下比较模型大小、候选数和推理策略。
- [[CMU 11-763 - Lecture 16 - Token Budgets and Training-Time Distillation|预算与蒸馏]]：把部分在线搜索成本转移到离线训练。

这些是课程来源入口；公式的适用条件、勘误和材料覆盖范围以各讲笔记为准。

## 分析框架

分析一个推理优化时至少需要明确：

1. 目标是质量、延迟、吞吐、显存还是成本；
2. Workload 是 prefill、decode、在线服务还是离线批处理；
3. 改变了计算量、内存流量、通信量还是调度方式；
4. 理论收益是否能在具体硬件和实现上观察到；
5. 是否改变模型输出分布或任务质量。

## 待解决问题

- [ ] #question [CMU Lecture 01 23:20：Hardware for inference](https://www.youtube.com/watch?v=F-mduXzNcRQ&t=1400s)——什么条件下 decode 会从 memory-bound 转向 compute-bound？这是从课程硬件背景延伸出的系统问题。
- [ ] #question 如何用统一指标比较 speculative decoding、quantization 与 continuous batching？

## 来源课程

- [CMU 11-763](../../courses/cmu-11-763/CMU%2011-763.md)
- [Stanford CS336](../../courses/stanford-cs336/Stanford%20CS336.md)
