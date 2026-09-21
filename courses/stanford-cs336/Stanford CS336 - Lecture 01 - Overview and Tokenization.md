---
type: course-note
status: developing
course: "[[Stanford CS336]]"
lecture: 1
lecture_date: 2026-03-30
area: data
topics:
  - "[[Transformer Architecture]]"
aliases:
  - Stanford CS336 Lecture 01
  - CS336 Overview and Tokenization
video_url: https://www.youtube.com/watch?v=JuoVZkPBiKk
---

# Lecture 01：Overview and Tokenization

> [!abstract] 本讲一句话
> 这门课把语言模型视为一个受资源约束的完整系统：目标不是记住组件名称，而是亲手构建并理解数据、tokenization、模型、训练和系统之间的取舍；第一讲用 BPE 说明了这一方法——tokenizer 本质上是在词表大小和序列长度之间分配计算资源。

## 来源与范围

- 课程：Stanford CS336 — Language Modeling from Scratch, Spring 2026
- 讲师：Percy Liang
- 上课日期：2026-03-30
- [课程视频](https://www.youtube.com/watch?v=JuoVZkPBiKk)，时长 1:19:21
- [课程主页](https://cs336.stanford.edu/)
- [官方可执行讲义](https://cs336.stanford.edu/lectures/?trace=lecture_01)
- [Assignment 1：Basics](https://github.com/stanford-cs336/assignment1-basics)
- 本讲覆盖：课程方法论、语言模型发展脉络、五个课程模块，以及 character/byte/word/BPE tokenization
- 本讲不覆盖：Transformer、训练与系统组件的完整细节；这些内容在后续讲次展开

> [!warning] 来源边界
> 知识结构、代码和示例数值来自官方可执行讲义；视频顺序与时间点来自 YouTube 自动章节。YouTube 转写文稿加载失败，因此这里不是逐字稿，也没有把无法核对的课堂口头补充写成课程原话。

## 视频索引

| 时间 | 视频内容 | 对应笔记 |
| --- | --- | --- |
| [00:00](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=0s) | Course introduction | [[#1. 为什么要从零构建\|1. 为什么要从零构建]] |
| [02:22](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=142s) | Course philosophy and goals | [[#1. 为什么要从零构建\|1. 为什么要从零构建]] |
| [03:24](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=204s) | The research landscape | [[#1. 为什么要从零构建\|1. 为什么要从零构建]] |
| [07:10](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=430s) | Efficiency and scaling | [[#2. 课程的统一视角：效率\|2. 课程的统一视角：效率]] |
| [11:37](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=697s) | Language model history | [[#3. Language Model 的发展脉络\|3. Language Model 的发展脉络]] |
| [19:27](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1167s) | Executable lecture format | [[#4. 课程如何学习\|4. 课程如何学习]] |
| [20:14](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1214s) | Logistics and syllabus | [[#4. 课程如何学习\|4. 课程如何学习]] |
| [27:24](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1644s) | The “Basics” unit | [[#5. 五个课程模块\|5. 五个课程模块]] |
| [35:53](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2153s) | Systems and hardware | [[#5.2 Systems\|5.2 Systems]] |
| [45:11](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2711s) | Scaling laws | [[#5.3 Scaling Laws\|5.3 Scaling Laws]] |
| [53:42](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3222s) | Data engineering | [[#5.4 Data\|5.4 Data]] |
| [1:00:22](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3622s) | Alignment and future units | [[#5.5 Alignment\|5.5 Alignment]] |
| [1:05:06](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3906s) | Tokenization deep dive | [[#6. Tokenizer 的问题定义\|6. Tokenizer 的问题定义]] |
| [1:12:03](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4323s) | BPE algorithm | [[#9. Byte Pair Encoding\|9. Byte Pair Encoding]] |

## 1. 为什么要从零构建

### 1.1 抽象层上移提高了生产力，也拉远了研究者与底层技术的距离

课程用一个简化的时间线描述研究工作方式的变化：

- 2016：研究者通常自己实现、训练模型；
- 2018：下载 BERT 等预训练模型，再做 fine-tuning；
- 今天：通过 API prompt GPT、Claude、Gemini 等模型。

抽象层上移本身是好事，但语言模型的抽象仍然是 **leaky abstraction**：

- 模型的行为会受 tokenizer、context length、训练数据和 decoding 等底层选择影响；
- 质量、延迟、显存和成本无法只通过 API 表面完全理解；
- 想做基础研究，有时必须改变整条技术栈，而不是只调整最上层接口。

因此课程的方法论是：

> **Understanding via building：通过亲手构建获得完整理解。**

### 1.2 Frontier model 带来的教学困难

Frontier model 的训练成本极高，公开技术细节又很少。即使我们可以训练小于 1B 参数的模型，小模型上的直觉也不一定能直接迁移到大模型：

- 模型变大后，Attention 和 MLP 的 FLOPs 占比会变化；
- 某些能力或行为只在足够规模上明显出现；
- 在小模型上合理的超参数和系统实现，放大后可能不稳定或低效。

课程将可以学习的知识分成三类：

| 类型 | 含义 | 向 frontier scale 迁移性 |
| --- | --- | --- |
| Mechanics | Transformer、并行、优化器等如何工作 | 高 |
| Mindset | 资源核算、重视 scaling、榨取硬件效率 | 高 |
| Intuitions | 哪种数据或 modeling choice 能获得更好质量 | 部分可迁移 |

这解释了课程的边界：它能系统教授 mechanics 和 mindset，但不会把小规模实验产生的经验直觉包装成普适规律。

### 1.3 “Bitter Lesson” 的正确理解

课程反对把 Bitter Lesson 简化为：

> 只要 scale 足够大，算法不重要。

更准确的理解是：

> **能够随资源扩展的算法最重要。**

课程给出的概念式是：

$$
\text{accuracy}
=
\text{efficiency}
\times
\text{resources}
$$

它不是严格的数值公式，而是研究框架：

- `resources`：数据、算力、内存、通信带宽；
- `efficiency`：同样资源能转化出多少模型能力；
- scale 越大，低效率造成的绝对浪费越不可接受。

课程最终反复回答的是：

> 给定 compute 和 data budget，怎样训练出最好的模型？

## 2. 课程的统一视角：效率

五个课程模块表面不同，实际都在做资源分配：

| 模块 | 效率问题 |
| --- | --- |
| Tokenization | 直接处理 byte 很通用，但会产生更长序列；怎样用有限词表缩短序列？ |
| Architecture | 怎样用更少 FLOPs、显存或 KV Cache 获得足够表达能力？ |
| Systems | 怎样减少数据搬运，让硬件真正接近峰值性能？ |
| Scaling Laws | 怎样用小规模实验决定大规模训练配置，避免昂贵试错？ |
| Data | 怎样避免把训练算力浪费在低质量、重复或无关数据上？ |
| Alignment | 怎样在 rollout、打分和更新之间获得更高训练信号效率？ |

课程指出，当前通常是 compute-constrained，因此很多设计都围绕算力效率。随着高质量公开数据逐渐耗尽，问题也会越来越 data-constrained。

> [!tip] 我的理解
> 这门课不是把算法与系统拆成两部分，而是把模型质量写成资源约束下的优化问题。一个 modeling choice 如果改变了序列长度、tensor shape 或访存量，它同时也是 system choice。

## 3. Language Model 的发展脉络

本讲不是要背模型年表，而是说明今天的语言模型是如何由多条技术路线汇合而成。

### 3.1 Pre-neural 与神经网络基础

- Shannon：用语言模型研究英语的 entropy；
- N-gram：曾广泛用于机器翻译和语音识别；
- LSTM 与第一批 neural language model；
- Sequence-to-sequence、Attention、Transformer；
- Adam；
- Mixture of Experts；
- Model Parallelism。

### 3.2 Foundation model 与 scaling

- ELMo：LSTM 预训练后再 fine-tune；
- BERT：Transformer 预训练后再 fine-tune；
- T5：将任务统一成 text-to-text；
- GPT-2：流畅生成与早期 zero-shot 迹象；
- GPT-3：scale 带来 in-context learning；
- Scaling laws：让大规模训练变得更可预测；
- PaLM 与 Chinchilla：把“更大模型”推进为“固定 compute 下如何分配模型参数与训练 token”。

### 3.3 Open model 的作用

开放程度可以拆成不同层次：

1. 只有 API；
2. Open weights + paper；
3. Weights + paper + code + data；
4. 连开发过程也开放。

开放模型不只是方便部署。它们让研究者可以检查训练配方、复现实验、修改底层实现，也使 CS336 这样的课程成为可能。

### 3.4 “语言模型”的交互界面在变化

课程概括了四种典型接口：

- 2018：需要 fine-tune 的模型；
- 2020：可以 prompt 的模型；
- 2022：可以对话的模型；
- 2026：可以自主行动的 agent。

接口和 workload 在变，但 Attention、kernel、优化与资源约束等基本问题仍然存在。Agent 带来了更长上下文和更多推理调用，因此 inference efficiency 反而更加重要。

## 4. 课程如何学习

### 4.1 Executable Lecture

Percy 的讲义本身是 Python 程序。程序执行时逐步展示：

- 文本和图像；
- 实际可运行的代码；
- 函数层级；
- 变量中间值；
- 算法执行轨迹。

这与课程方法一致：概念不只通过静态 slide 描述，而是通过可检查的执行过程解释。

### 4.2 作业方式

课程有五个 implementation-heavy assignments：

1. Basics；
2. Systems；
3. Scaling Laws；
4. Data；
5. Alignment。

作业几乎不提供 scaffolding code，但提供 unit tests 和 adapter interface。典型工作流是：

```text
本地实现与 correctness test
→ 放到集群
→ benchmark accuracy / speed
→ 根据资源约束继续优化
```

课程对 AI 工具的立场也与“理解 via building”一致：AI 适合回答问题和辅导，但直接替代实现会绕过最重要的学习过程。

## 5. 五个课程模块

### 5.1 Basics

目标：能够训练一个基本语言模型。

主要组件：

- Tokenization；
- Model architecture；
- Training。

Assignment 1 要求从头实现：

- BPE tokenizer；
- Transformer；
- Cross-entropy loss；
- AdamW；
- Training loop；
- Resource accounting；
- 在 TinyStories 和 OpenWebText 上训练；
- 在固定 B200 时间预算下优化 OpenWebText perplexity。

所有 architecture/training choice 都在平衡：

| 目标           | 问题                 |
| ------------ | ------------------ |
| Expressivity | 能否表示数据中的复杂依赖？      |
| Stability    | 参数和梯度的数值是否处于可训练区间？ |
| Efficiency   | 训练和推理时能否高效使用硬件？    |

### 5.2 Systems

目标：从 GPU/TPU 中获得尽可能多的有效性能。

#### Resource accounting

需要先理解：

- 参数、activation、gradient、optimizer state 占多少内存；
- 算子需要多少 FLOPs；
- 数据从 HBM 移到 compute unit 的成本；
- workload 是 compute-bound 还是 memory-bound。

讲义给 B200 的数量级示例是：

- BF16 峰值约 2.25 PFLOP/s；
- HBM bandwidth 约 8 TB/s。

两者不能分别看。Roofline analysis 关心的是：

$$
\text{arithmetic intensity}
=
\frac{\text{FLOPs}}{\text{bytes moved}}
$$

#### Kernels

PyTorch 的一个 primitive operation 通常会启动一个 kernel。Kernel 优化的核心原则是减少数据搬运：

```text
Naive:
HBM read → A → HBM write
HBM read → B → HBM write

Fused:
HBM read → A → B → HBM write
```

常见方法包括 fusion、tiling、memory coalescing、避免 bank conflict 和提高 occupancy；FlashAttention 是通过 tiling 减少 HBM traffic 的代表。

#### Parallelism

多 GPU 场景不仅要分 computation，也要分 memory：

- Data Parallel；
- Tensor Parallel；
- Pipeline Parallel；
- Sequence Parallel；
- Expert Parallel。

通信带宽远低于芯片内部计算吞吐，因此仍然服从“最小化数据移动”的统一原则。

#### Inference

推理分为：

- Prefill：prompt token 已知，可以并行处理，通常更接近 compute-bound；
- Decode：一次只能生成一个新 token，反复读取模型权重与 KV Cache，通常更接近 memory-bound。

加速方法包括量化、剪枝、蒸馏、speculative decoding、fused kernels 和 continuous batching。

### 5.3 Scaling Laws

如果目标训练要消耗 $10^{25}$ FLOPs，就不可能在目标规模上反复做常规超参数搜索。关键概念从“一个配置”变成：

> **Scaling recipe：从 compute budget 映射到一组超参数。**

工作流是：

1. 在多个较小 compute scale 上训练；
2. 为每个 scaling recipe 拟合 loss；
3. 外推到目标规模；
4. 比较 recipe，选择大规模训练配置。

语言模型训练 FLOPs 的常用一阶估算为：

$$
C\approx 6ND
$$

其中：

- $C$：训练 FLOPs；
- $N$：非 embedding 参数量；
- $D$：训练 token 数量。

Chinchilla-style 经验近似：

$$
D\approx 20N
$$

例如 70B 参数模型对应约 1.4T training tokens。课程同时提醒：这个 compute-optimal 结论没有计入未来 inference cost；如果模型要被调用很多次，训练一个更小但训练更久的模型可能更划算。

Scaling law 不会自动产生。模型参数化、hyperparameter transfer 和训练稳定性会决定小规模实验能否预测大规模结果。课程强调：

> Predictability 至少与单点 optimality 同样重要。

### 5.4 Data

首先要问的不是“有多少数据”，而是“希望模型获得什么能力”：

- Multilingual；
- Conversation；
- Agentic coding；
- Long context；
- 特定知识与安全行为。

数据工作包括：

| 阶段                         | 作用                      |
| -------------------------- | ----------------------- |
| Transformation             | 将 HTML、PDF、代码目录等转成可训练文本 |
| Filtering                  | 保留高质量和相关内容，去除有害内容       |
| Deduplication              | 节约计算并降低 memorization    |
| Mixing                     | 决定不同来源的采样权重             |
| Rewriting / Synthetic Data | 将真实数据改造成更接近目标任务的形式      |

评估也有两个不同目标：

- Internal evaluation：指导研发，更关心跨 scale 的平滑性和相对变化；
- External evaluation：判断真实用途，更关心 absolute quality 和 ecological validity。

#### Perplexity

> [!question] Q-CS336-L01-5342：Perplexity 是什么？
> - [x] #question 课程中举的 perplexity 是什么，应该怎样理解？
> - 来源：[视频 53:42](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3222s)
> - 结论：Perplexity 是 next-token cross-entropy 的指数形式；越低表示模型给真实下一个 token 分配的概率越高，但它主要衡量语言建模能力，不能直接代表推理、事实性或指令遵循质量。

给定 token 序列 $x_1,\ldots,x_T$，模型对真实 token 的平均 negative
log-likelihood/cross-entropy 是：

$$
\mathcal L
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log P_\theta(x_t\mid x_{<t})
$$

如果使用自然对数，perplexity 定义为：

$$
\operatorname{PPL}
=
\exp(\mathcal L)
=
\left(
\prod_{t=1}^{T}
\frac{1}{P_\theta(x_t\mid x_{<t})}
\right)^{1/T}
$$

所以 PPL 与 cross-entropy 是同一个指标的两种刻度：

$$
\operatorname{PPL}=e^{\text{loss}}
$$

- loss 越低，PPL 越低；
- PPL 越低，模型对测试文本中的真实下一个 token 越不“意外”；
- 二者是单调变换，用它们比较同一组模型时排序相同。

一种粗略直觉是：如果 PPL 为 20，模型每一步的不确定程度近似于“在 20 个等可能
选项中选择”。这只是等效不确定度，不表示模型真的只在 20 个 token 之间选择。

例如，两个位置上的真实 token 概率分别为 $0.5$ 和 $0.25$：

$$
\operatorname{PPL}
=
\sqrt{\frac{1}{0.5}\times\frac{1}{0.25}}
=
\sqrt{8}
\approx 2.83
$$

如果模型把这两个正确 token 的概率都提高，PPL 就会下降。

Perplexity 很适合 internal evaluation：

- 直接对应预训练的 next-token prediction objective；
- 计算便宜，不需要人工标注；
- 通常随训练 compute 和模型规模平滑变化；
- 适合发现训练回归、比较数据配方和拟合 scaling law。

课程建议在未公开的 held-out documents 上计算 PPL。若评估文本已经出现在训练语料中，
模型可能只是记住了内容，得到虚假的低 PPL，这就是 data contamination。

但 PPL 有明确边界：

- 它不直接衡量 reasoning、factuality、helpfulness、safety 或 instruction following；
- 最可能的文本不一定是人类最喜欢或任务得分最高的回答；
- 只能在相同 tokenizer、相同测试集和相同预处理条件下直接比较；
- tokenizer 改变后，token 粒度和 $T$ 都会变化，PPL 数值不再处于同一尺度。

> [!tip] 与 Tokenization 的连接
> 这也是为什么不能脱离 tokenizer 比较 PPL。若必须比较使用不同 tokenizer 的模型，
> 可以考虑按原始 byte 数归一化的 negative log-likelihood，例如 bits per byte，但最终
> 仍应结合真实下游任务评估。

数据还可以按训练阶段分为：

- Pretraining：规模大、来源多样；
- Mid-training：质量更高，可能突出 long-context 等能力；
- Post-training：对话、偏好、agent trajectory 和 tool calling。

### 5.5 Alignment

预训练使用 full supervision：每个位置都有下一个 token 作为标签。Alignment 进一步利用 weak supervision，尤其适合：

> **评价一个答案比从零生成答案更容易。**

基本模板：

```text
模型生成 responses
→ human / verifier / LM judge 打分
→ 更新模型，使其偏好高分 response
```

课程涉及：

- PPO；
- DPO；
- GRPO。

Alignment 不只是 loss function 问题。大规模 RL 需要持续生成 rollout，系统必须在 inference throughput 和 on-policyness 之间做权衡，因此会引入新的异步推理基础设施。

## 6. Tokenizer 的问题定义

Raw text 在程序中通常表示为 Unicode string，而语言模型操作的是整数 token 序列。

定义：

$$
\operatorname{encode}:
\text{string}
\rightarrow
[t_1,t_2,\ldots,t_T]
$$

$$
\operatorname{decode}:
[t_1,t_2,\ldots,t_T]
\rightarrow
\text{string}
$$

基本正确性要求是 round-trip：

$$
\operatorname{decode}(\operatorname{encode}(s))=s
$$

课程中的最小接口：

```python
class Tokenizer:
    def encode(self, string: str) -> list[int]:
        ...

    def decode(self, indices: list[int]) -> str:
        ...
```

> [!note] Round-trip 的边界
> 要求是合法输入字符串经过 encode 后能够重建。它不意味着任意 token ID 序列都一定能单独解成合法 Unicode；byte-level BPE 的单个 token 可能只是一个 UTF-8 字符的部分 bytes，通常应先拼接全部 bytes，再统一 decode。

## 7. Tokenizer 要优化什么

### 7.1 Compression ratio

课程定义：

$$
R
=
\frac{\text{UTF-8 bytes 数量}}
{\text{token 数量}}
$$

$R$ 的单位是 bytes/token。值越大，同一文本对应的 token 序列越短。

官方讲义使用 `o200k_base` tokenizer 编码：

```text
Hello, 🌍! 你好!
```

结果为：

```text
[13225, 11, 130321, 235, 0, 220, 177519, 0]
```

该字符串有 20 个 UTF-8 bytes、8 个 tokens，因此：

$$
R=\frac{20}{8}=2.5
$$

该 tokenizer 的词表大小为 200,019。

### 7.2 Vocabulary size 与 sequence length 的矛盾

增大词表 $V$ 通常可以让常见 byte sequence 合并为更长 token，从而降低 $T$；但它也会带来：

- 更大的 embedding table；
- 更大的 LM head；
- 更大的 vocabulary logits；
- 更稀疏的 token frequency，稀有 token 更难学好。

因此 tokenizer 不是单纯追求最高压缩率，而是在 $V$ 与 $T$ 之间寻找系统级折中。

### 7.3 Token 是一种可变粒度的 computation allocation

课程对 tokenizer 提出两个更一般的要求：

1. 模型应操作 sequence 的 chunk/abstraction，而不是永远操作最底层 byte；
2. chunk 应该是可变的，使模型把更多计算容量分给“有意思”的部分。

一个 token 无论包含一个 byte 还是一段常见单词，通常都会消耗一次 Transformer position 的计算。BPE 因此可以理解为：

- 常见模式：压成一个 token，用较少 position；
- 稀有模式：拆成多个 token，用更多 position。

这是一种固定在训练前的数据驱动型 adaptive computation。

## 8. Character、Byte 与 Word Tokenizer

| 方法 | 基本单位 | 词表 | Compression | 核心问题 |
| --- | --- | ---: | --- | --- |
| Character | Unicode code point | 约 150K | 较低 | 词表大，稀有字符多 |
| Byte | UTF-8 byte | 256 | 固定 1 byte/token | 序列很长 |
| Word | 人工/regex 切出的词 | 很大且不易固定 | 通常高 | Rare word、OOV/UNK |
| Byte-level BPE | 学习得到的 byte chunks | $256+M$ | 数据驱动 | 需训练 merges，词表与序列长度折中 |

### 8.1 Character tokenizer

Python 可以使用：

```python
ord("a") == 97
ord("🌍") == 127757
chr(127757) == "🌍"
```

示例字符串：

```text
Hello, 🌍! 你好!
```

会得到 13 个 Unicode code points，compression ratio 为：

$$
\frac{20}{13}\approx1.54
$$

问题：

- 需要覆盖约 150K Unicode characters；
- 很多字符很稀有，embedding 学习低效；
- 相对 byte tokenizer 虽略短，但没有解决“大词表 + 长序列”的矛盾。

### 8.2 Byte tokenizer

UTF-8 将 Unicode string 转换为 bytes：

```python
"a".encode("utf-8") == b"a"
"🌍".encode("utf-8") == b"\xf0\x9f\x8c\x8d"
```

Byte tokenizer 的优点：

- 词表固定为 256；
- 任意合法 UTF-8 文本都可表示；
- 不需要 UNK。

但每个 byte 就是一个 token，因此：

$$
R=1
$$

示例字符串会变成 20 个 token。Transformer 的 context window 以 token 计数，Attention 又与序列长度相关，因此纯 byte sequence 通常太长。

### 8.3 Word tokenizer

课程用正则：

```python
regex.findall(r"\w+|.", string)
```

切分：

```text
I'll say supercalifragilisticexpialidocious!
```

得到：

```text
["I", "'", "ll", " ", "say", " ",
 "supercalifragilisticexpialidocious", "!"]
```

这个示例的 compression ratio 是 5.5，且 token 对人类有一定语义。

问题：

- 训练数据中的 distinct word 数量可能极大；
- 稀有词缺少足够样本；
- 词表大小不容易预先固定；
- 新词会变成 `UNK`，丢失原始信息；
- `UNK` 还会干扰 perplexity：大量不同未知词可能都被映射为同一容易预测的 token。

## 9. Byte Pair Encoding

### 9.1 核心思想

BPE 最初是数据压缩算法，后来被用于 neural machine translation，再被 GPT-2 等语言模型采用。

Byte-level BPE：

1. 从 256 个 byte token 开始；
2. 在训练语料中统计相邻 token pair；
3. 找到最常见 pair；
4. 为这个 pair 创建新 token；
5. 把语料中的该 pair 替换成新 token；
6. 重复 $M$ 次。

最后：

$$
V=256+M
$$

不计额外 special tokens。

直觉是：

> 常见 byte sequence 用一个 token 表示；稀有 sequence 仍退化为多个 byte token，所以不会产生 OOV。

### 9.2 Tokenizer 的最小状态

官方实现把 BPE 参数分成：

```python
vocab: dict[int, bytes]
merges: dict[tuple[int, int], int]
```

- `vocab`：token ID $\rightarrow$ 对应 bytes；
- `merges`：旧 token pair $\rightarrow$ 新 token ID；
- `merges` 的顺序也表示训练时的 merge priority。

训练 tokenizer 与使用 tokenizer 是两个阶段：

```text
Training corpus
→ 学习 vocab + ordered merges

New text
→ UTF-8 bytes
→ 按 learned merge priority 合并
→ token IDs
```

### 9.3 `the cat in the hat` 示例

初始文本有 18 个 ASCII bytes，也就是 18 个 byte tokens。

由于示例实现使用首次出现顺序打破并列，三次 merge 为：

| Merge | 新 ID | 新 token bytes | 序列长度 |
| --- | ---: | --- | ---: |
| `(t, h)` | 256 | `b"th"` | 16 |
| `(256, e)` | 257 | `b"the"` | 14 |
| `(257, space)` | 258 | `b"the "` | 12 |

最终：

$$
R=\frac{18}{12}=1.5
$$

训练后的 merge 表：

```text
(116, 104) → 256       # t + h  → th
(256, 101) → 257       # th + e → the
(257, 32)  → 258       # the + space → "the "
```

这里可以看到，BPE token 不必等于一个“单词”，它可以包含尾随空格。

### 9.4 用训练好的 tokenizer 编码新文本

编码：

```text
the quick brown fox
```

得到：

```text
[258, 113, 117, 105, 99, 107, 32,
 98, 114, 111, 119, 110, 32, 102, 111, 120]
```

新文本中的 `"the "` 使用 token 258，其余未被训练 merge 覆盖的内容退化为 byte token。

解码时：

```python
bytes_list = [vocab[token_id] for token_id in indices]
string = b"".join(bytes_list).decode("utf-8")
```

先拼 bytes，再做 UTF-8 decode，从而保证 round-trip。

### 9.5 `merge` 的行为

给定 token 序列、目标 pair 和新 ID，算法从左到右扫描：

```python
if current_two_tokens == pair:
    output.append(new_id)
    i += 2
else:
    output.append(current_token)
    i += 1
```

这会执行 **互不重叠** 的合并。例如 pair 是 `(a, a)` 时，`a a a` 的第一次 pass 只会把前两个 `a` 合并。

### 9.6 真实实现还要处理什么

讲义中的 `encode` 对所有 merge 依次扫描整个序列，只适合解释，不适合生产使用。Assignment 1 还要求：

- 只处理当前序列中真正可能发生的 merge；
- 识别并原样保留 `<|endoftext|>` 等 special token；
- 使用 GPT-2-style regex 做 pre-tokenization；
- 优化 tokenizer training 和 encoding 的速度。

> [!note] Pre-tokenization 的作用
> Byte-level BPE 具有任意跨边界合并的能力。Pre-tokenization 先用 regex 将文本分块，再限制 BPE 只在块内合并，从而控制空格、标点、数字等模式，并减少训练和编码的搜索空间。

## 10. 常见 tokenizer 现象

课程建议直接观察真实 tokenizer，典型现象包括：

- 单词与前面的空格可能属于同一 token，例如 `" world"`；
- 同一个词出现在字符串开头和中间，tokenization 可能不同；
- 数字经常每隔若干位切成一个 token；
- token 并不稳定对应 character、word 或语义单位。

这些不是展示层的小细节，会实际影响：

- prompt 的 token 数；
- context window 能装下多少原始文本；
- 不同语言的训练与推理成本；
- 数字、拼写和格式任务的难度；
- streaming 输出时可安全解码的边界。

> [!warning] 比较 token 数时
> “多少 tokens”只在指定 tokenizer 后才有意义。不同模型的词表、pre-tokenization、normalization 和 special tokens 不同，不能把 token count 当成与模型无关的文本属性。

## 11. Tokenizer-free 的方向

Character、byte、word tokenizer 都有明显问题，BPE 也只是一个独立于模型训练的 heuristic。更理想的方向是直接从 byte 学习表示与可变粒度 chunk。

课程列举的方向包括：

- ByT5；
- MegaByte；
- Byte Latent Transformer；
- T-Free；
- H-Net。

它们试图让模型端到端学习 chunk 或直接操作 byte，但尚未像 BPE Transformer 一样在 frontier scale 得到充分验证。

课程给出的最终约束比“要不要 BPE”更一般：

1. 模型需要在 chunk/abstraction 上计算；
2. chunk 粒度需要可变，以便把更多模型容量分配给复杂内容。

## 12. AI Infra 视角

### 12.1 Shape

设：

- Batch size：$B$；
- Token sequence length：$T$；
- Vocabulary size：$V$；
- Hidden dimension：$d$。

主要 tensor：

| 对象              | Shape     |
| --------------- | --------- |
| Token IDs       | $[B,T]$   |
| Embedding table | $[V,d]$   |
| Hidden states   | $[B,T,d]$ |
| LM head weight  | $[d,V]$   |
| Logits          | $[B,T,V]$ |

Embedding 和 LM head 可以 weight tying，但 shape tradeoff 不变：增大 $V$ 会线性增大词表相关参数。

### 12.2 Compute

Tokenizer 通过改变 $T$ 影响几乎整张 Transformer 计算图：

- Projection/MLP 等 token-wise 计算：约随 $T$ 线性增长；
- Full attention：约随 $T^2$ 增长；
- LM head：约为 $O(BTdV)$；
- Softmax/cross-entropy：也随 $BTV$ 增长。

因此增大 $V$ 的效果有两面：

```text
V 增大
├─ T 可能减小 → block 与 attention 更便宜
└─ embedding / LM head / logits 更大
```

“压缩率更高”不必然等于端到端更快，需要在真实 corpus 和目标 model shape 上测量。

### 12.3 Memory

主要影响包括：

- 参数内存：embedding/LM head 约为 $Vd$；
- Logits activation：训练时可能达到 $BTV$；
- 中间 activation 和 KV Cache：通常随 $T$ 线性增长；
- Attention score/materialization：naive 实现可随 $T^2$ 增长；
- Padding：batch 内长度差异会造成无效 token 计算。

Tokenizer 的 compression ratio 因而会影响：

- 单请求 KV Cache；
- 可承载并发数；
- batch padding waste；
- 每个训练 batch 能容纳的原始文本量。

### 12.4 Communication

Tokenizer 本身通常在 CPU/data pipeline 上运行，但会间接改变分布式系统：

- $T$ 改变 activation/gradient 的规模；
- sequence parallel 的分片与通信量依赖 $T$；
- 更大的词表可能需要 vocabulary-parallel LM head；
- 所有 training/inference worker 必须使用完全一致的 vocab、merge order 和 special token IDs。

> [!tip] Model ABI
> Tokenizer 配置可以视为模型接口的一部分。只保存 weights 而丢失 tokenizer，或在部署时改动 special token ID，会让同一字符串映射到不同输入，模型语义随之失效。

### 12.5 Runtime / Data Pipeline

真实 tokenizer 需要考虑：

- 大语料上的并行与 streaming；
- Pair count 的内存占用；
- Merge 后只增量更新相邻 pair，而不是每轮重扫全部 corpus；
- Pre-tokenization regex 的吞吐；
- Unicode 与 invalid byte 的处理策略；
- Special token 与普通文本的边界；
- Encode throughput 是否跟得上 GPU training pipeline。

如果 GPU 在等待 CPU tokenization 或数据加载，再高的理论 FLOPs 利用率也没有意义。

## 13. 我的理解与推导

### 13.1 Tokenizer 是模型前置的资源调度器

BPE 不只是在“压缩字符串”。它决定一个原始片段需要消耗多少次 Transformer position：

```text
常见片段 → 较少 token → 较少模型 step
稀有片段 → 较多 token → 较多模型 step
```

所以 BPE 同时改变：

- 表示粒度；
- 有效 context length；
- Attention/MLP FLOPs；
- KV Cache；
- 不同语言和领域的数据效率。

这正是第一讲把 tokenization 放进“efficiency”框架，而不是只当作文本预处理的原因。

### 13.2 BPE 的优势来自可逆 backoff

Word tokenizer 遇到未知词时退化为 `UNK`，信息直接丢失。Byte-level BPE 的最底层词表是全部 256 bytes，因此：

$$
\text{任何合法 byte sequence}
\Rightarrow
\text{总能表示}
$$

常见模式得到压缩，罕见模式退回 byte。这种“learned compression + lossless fallback”是 BPE 成功的关键组合。

### 13.3 Tokenizer 不是越大越好

假设增大词表使 $T$ 缩短：

- block 内大部分计算下降；
- Attention 的下降可能更快；
- 但 embedding、LM head 和 logits 随 $V$ 增长；
- token frequency 变稀，某些 embedding 缺少训练数据；
- 多语言词表还涉及不同语言之间如何分配 merge budget。

因此应该测量的是：

$$
\text{quality under fixed end-to-end resources}
$$

而不是单独最大化 compression ratio。

## 14. 本讲结论

1. CS336 的核心不是复述 LLM 技术栈，而是通过构建理解 mechanics，并用资源约束形成 systems mindset。
2. Scale 不会让算法失去意义；规模越大，算法和系统效率的绝对价值越高。
3. Tokenizer 将 Unicode string 可逆地映射为 integer tokens，它是模型定义的一部分。
4. Character tokenizer 词表大，byte tokenizer 序列长，word tokenizer存在 rare word 和 OOV；byte-level BPE 在这些问题之间取得实用折中。
5. BPE 从 256 bytes 开始，反复合并训练语料中的高频相邻 pair；常见片段压缩，稀有片段无损退回 byte。
6. Tokenizer 同时决定 sequence length、词表参数、Attention/MLP 计算、KV Cache 和数据管线吞吐，因此是 AI Infra 问题。

## 15. 自测问题

1. 为什么课程认为 API abstraction 对语言模型来说是“leaky”的？

    **面试回答：** 因为 API 只隐藏了实现，没有消除底层选择对行为的影响：tokenizer 会影响切词和长度，context window、训练数据与 decoding 会影响能力和输出。质量、延迟、显存和成本仍由这些机制决定，所以研究和排障有时必须穿透接口检查整条技术栈。

2. Mechanics、mindset 和 intuitions 中，哪些更容易从小模型迁移到 frontier scale？

    **面试回答：** Mechanics 和 mindset 更容易迁移：前者是 attention、优化器、并行等工作机制，后者是资源核算、效率与 scaling 的研究方法。具体数据配方、超参数和架构优劣属于 intuitions，可能随规模改变，需要跨规模实验验证。

3. 为什么 Bitter Lesson 不应被理解为“算法不重要”？

    **面试回答：** Bitter Lesson 强调的是能有效利用更多计算和数据的通用方法，而不是算法无关紧要。算法决定资源转化成能力的效率；规模越大，低效率的绝对成本越高，因此应优先寻找可扩展、能随资源持续获益的算法。

4. 为什么 compression ratio 使用 bytes/token，而不是 characters/token？

    **面试回答：** UTF-8 byte 是可直接计量的原始数据单位，而“字符”可能指 code point 或可见字形，不同语言一个字符对应的 byte 数也不同。bytes/token 能统一表达压缩率和原始文本覆盖量，但跨语言比较时仍要注意编码长度差异。

5. Character、byte、word tokenizer 分别卡在哪个 tradeoff 上？

    **面试回答：** Character tokenizer 的序列较长，完整 Unicode 词表又大且长尾严重；byte tokenizer 只有 256 个基础 token、无 OOV，却使序列更长。Word tokenizer 常能缩短序列，但词表难以封闭，新词会成为 UNK，稀有词也难学好。

6. BPE 的 vocab 和 ordered merges 分别记录什么？

    **面试回答：** vocab 记录 token ID 到原始 bytes 的映射，用于还原文本；ordered merges 记录哪些相邻 token pair 可以合并为哪个新 token，以及合并优先级。编码按训练所得顺序应用规则，解码则先拼接各 token 的 bytes，再统一做 UTF-8 解码。

7. 为什么 byte-level BPE 不需要 `UNK`？

    **面试回答：** 只要基础词表包含全部 256 个 byte，任意合法 UTF-8 文本都能退回到 byte 序列表示；学习到的 merge 只是让常见片段更短，不会丢失这种覆盖能力。因此不需要 UNK，但单个 token 可能只包含一个字符的部分 bytes，解码应先拼接再转字符串。

8. `the cat in the hat` 的三次 merge 如何把序列从 18 缩短到 12？

    **面试回答：** 按示例的并列处理顺序，先把两处 t+h 合成 th，长度 18→16；再把两处 th+e 合成 the，16→14；最后把两处 the+空格合成一个 token，14→12。每次合并两处都各省一个 token，因此最终压缩率为 18/12=1.5 bytes/token。

9. 增大 vocabulary size 为什么可能同时让 Transformer block 更便宜、LM head 更昂贵？

    **面试回答：** 固定原始文本时，更大的词表通常能缩短 token 序列 T，使 block 的逐 token 计算约按 T、full attention 约按 T² 降低。但 LM head 每个位置要预测 V 类，成本约为 O(BTdV)，词表参数约为 Vd；T 的下降是否抵消 V 的增长要实测。

10. 为什么 tokenizer 配置应该与 model weights 一起版本化和部署？

    **面试回答：** Token ID 是模型 embedding 和输出权重的索引，vocab、merge 顺序、预切分规则或 special token ID 一变，同一文本就可能映射到不同语义。把 tokenizer 与权重一起版本化，才能保证训练、推理及各 worker 的输入输出约定一致，并支持可靠复现和回滚。


## 参考资料

- [Stanford CS336 Spring 2026](https://cs336.stanford.edu/)
- [Lecture 01 executable lecture](https://cs336.stanford.edu/lectures/?trace=lecture_01)
- [Lecture 01 video](https://www.youtube.com/watch?v=JuoVZkPBiKk)
- [Assignment 1：Basics](https://github.com/stanford-cs336/assignment1-basics)
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models](https://arxiv.org/abs/2105.13626)
- [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871)
