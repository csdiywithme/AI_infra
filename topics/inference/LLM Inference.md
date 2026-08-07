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
- KV Cache：待从后续课程和实验继续沉淀。

### 生成与决策

- [CMU Lecture 01：Basic Generation 与 Meta-generation](../../courses/cmu-11-763/CMU%2011-763%20-%20Lecture%2001%20-%20Introduction%20to%20Language%20Models%20and%20Inference.md#6-basic-generation-与-meta-generation)
- [Sampling 与 Search](../../courses/cmu-11-763/CMU%2011-763%20-%20Lecture%2001%20-%20Introduction%20to%20Language%20Models%20and%20Inference.md#5-两类基本生成方法)
- [Search Error 与 Model Error](../../courses/cmu-11-763/CMU%2011-763%20-%20Lecture%2001%20-%20Introduction%20to%20Language%20Models%20and%20Inference.md#9-search-error-与-model-error)

### Serving 与系统优化

- Continuous Batching
- PagedAttention
- Quantization
- Speculative Decoding
- Tensor Parallel

这些主题将在出现足够课程材料、推导或实验后升级为独立主题笔记；当前不创建空壳文件。

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
- Stanford CS336：待添加对应课程笔记。
