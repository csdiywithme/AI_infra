---
type: course-note
status: developing
course: "[[CMU 11-763]]"
lecture: 2
lecture_date: 2025-08-28
area: inference
topics:
  - "[[LLM Inference]]"
  - "[[Transformer Architecture]]"
aliases:
  - CMU 11-763 Lecture 02
  - CMU LLM Inference Probability Review
  - Probability Review and Code Examples
video_url: https://www.youtube.com/watch?v=UKuPCxozypU
---

# Lecture 02：Probability Review and Code Examples

> [!abstract] 本讲一句话
> 自回归语言模型给出逐 token 条件概率；sampling 把这些局部分布变成序列和 Monte Carlo 估计，temperature 决定采样分布是否仍等于模型分布，而 meta-generation 则在候选生成之后再用额外模型或规则做选择。

## 来源与范围

- 课程：CMU 11-664/763 — Inference Algorithms for Language Modeling, Fall 2025
- 讲师：Graham Neubig
- 上课日期：2025-08-28
- [课程视频](https://www.youtube.com/watch?v=UKuPCxozypU)，时长约 1:12:05
- [课程主页](https://www.phontron.com/class/lminference-fall2025/)
- [本讲官方代码目录](https://github.com/neubig/lminference-fall2025-code/tree/main/02-generation-basics)
- 可执行课件：[Probability](https://github.com/neubig/lminference-fall2025-code/blob/main/02-generation-basics/probability.ipynb)、[Generation](https://github.com/neubig/lminference-fall2025-code/blob/main/02-generation-basics/generation.ipynb)、[Meta-generation](https://github.com/neubig/lminference-fall2025-code/blob/main/02-generation-basics/meta_generation.ipynb)
- 最小 GPT-2 实现：[nanogpt.py](https://github.com/neubig/lminference-fall2025-code/blob/main/02-generation-basics/nanogpt.py)

> [!note] 本讲没有传统 slides
> 讲师在 [00:20](https://www.youtube.com/watch?v=UKuPCxozypU&t=20s) 明确说明，本讲不使用 slides，而是现场运行 notebook。因此本文所说的“课件”指官方 notebook 与配套 Python 文件；正文按公开视频英文字幕与这些代码交叉核对。

> [!warning] 来源边界
> 课堂演示使用人为构造的 bigram model、GPT-2 small/medium 和少量随机样本，目的是建立直觉，不是严谨 benchmark。采样结果每次可能不同；课堂关于 GPT-5、DeepSeek、OpenAI reasoning model 和商业 API 的说法是 2025 年的口头观察或 anecdote，本文不把它们外推为稳定产品事实。

## 视频时间索引

| 时间 | 视频内容 | 对应笔记 |
| --- | --- | --- |
| [00:00](https://www.youtube.com/watch?v=UKuPCxozypU&t=0s) | 课程安排与 notebook 说明 | [[#1. 本讲主线\|1]] |
| [02:08](https://www.youtube.com/watch?v=UKuPCxozypU&t=128s) | 四部分内容概览 | [[#1. 本讲主线\|1]] |
| [03:43](https://www.youtube.com/watch?v=UKuPCxozypU&t=223s) | Conditional probability 与语言模型定义 | [[#2. 条件概率是自回归 LM 的接口\|2]] |
| [08:02](https://www.youtube.com/watch?v=UKuPCxozypU&t=482s) | N-gram、bigram 与 Markov assumption | [[#2.2 Bigram model：最小可观察实验台\|2.2]] |
| [10:53](https://www.youtube.com/watch?v=UKuPCxozypU&t=653s) | 从语言模型采样 | [[#3. 从逐 token 分布采样出完整序列\|3]] |
| [12:34](https://www.youtube.com/watch?v=UKuPCxozypU&t=754s) | Temperature sampling 公式与极限 | [[#4. Temperature 改变的是采样分布\|4]] |
| [18:20](https://www.youtube.com/watch?v=UKuPCxozypU&t=1100s) | 用采样估计未知概率 | [[#5. Sampling-based estimation\|5]] |
| [25:29](https://www.youtube.com/watch?v=UKuPCxozypU&t=1529s) | Q&A：估计是否依赖 temperature | [[#5.3 T=1 与估计偏差\|5.3]] |
| [28:57](https://www.youtube.com/watch?v=UKuPCxozypU&t=1737s) | Marginalization | [[#6. Marginalization：消去不关心的变量\|6]] |
| [33:01](https://www.youtube.com/watch?v=UKuPCxozypU&t=1981s) | Latent variables、CoT 与 self-consistency | [[#7. Reasoning trace 是 latent variable\|7]] |
| [38:00](https://www.youtube.com/watch?v=UKuPCxozypU&t=2280s) | T=1 无偏、T=0.5 误差平台 | [[#8. 先区分 inference 的两个目标\|8]] |
| [42:04](https://www.youtube.com/watch?v=UKuPCxozypU&t=2524s) | 最小 GPT-2 实现 | [[#9. 最小 Transformer 代码的作用\|9]] |
| [43:19](https://www.youtube.com/watch?v=UKuPCxozypU&t=2599s) | GPT-2 不同温度生成 | [[#10. GPT-2 温度生成实验\|10]] |
| [47:19](https://www.youtube.com/watch?v=UKuPCxozypU&t=2839s) | 为什么 temperature=0 仍可能不复现 | [[#11. Greedy 不等于端到端确定性\|11]] |
| [54:54](https://www.youtube.com/watch?v=UKuPCxozypU&t=3294s) | Diversity metric 与 LLM-as-a-judge | [[#12. 怎样评价生成结果\|12]] |
| [1:00:19](https://www.youtube.com/watch?v=UKuPCxozypU&t=3619s) | Meta-generation：small 生成、medium 重排 | [[#13. Meta-generation：先生成候选再评分\|13]] |
| [1:05:00](https://www.youtube.com/watch?v=UKuPCxozypU&t=3900s) | Per-token log probability 与长度归一化 | [[#13.2 为什么使用平均 token log probability\|13.2]] |
| [1:10:25](https://www.youtube.com/watch?v=UKuPCxozypU&t=4225s) | Q&A：meta-generation 的用途 | [[#15. Meta-generation 的实际用途\|15]] |

## 1. 本讲主线

本讲把第一讲的概念落到可运行代码，共四部分：

1. 概率基础：条件概率、联合概率、边缘化、潜变量和采样估计；
2. 一个极简 GPT-2/Transformer 实现；
3. 用 temperature 控制生成，并用确定性指标与 LLM judge 做初步评价；
4. 最简单的 meta-generation：用一个模型生成、另一个模型重排序。

贯穿全讲的问题是：

> 当模型只提供下一 token 的条件分布时，我们怎样生成序列、估计整体性质，并用额外计算改善最终输出？

整体数据流可以写成：

```text
prompt x
  -> LM 给出 P(token | prefix)
  -> decoding / sampling 产生候选 y_1...y_N
  -> metric、judge 或更强模型给候选打分
  -> 选择或聚合最终答案 y*
```

前两步属于 basic generation；后两步开始进入 meta-generation。

## 2. 条件概率是自回归 LM 的接口

### 2.1 从序列概率到 next-token probability

语言模型的宽泛定义是：给 token 序列分配概率。对序列

$$
x_{1:T}=(x_1,x_2,\ldots,x_T)
$$

自回归分解为：

$$
P_\theta(x_{1:T})
=
\prod_{t=1}^{T}P_\theta(x_t\mid x_{<t})
$$

每一步暴露的接口是条件概率：

$$
P(y\mid x)=\frac{P(x,y)}{P(x)}
$$

其中 $x$ 是当前 prefix，$y$ 是下一个 token。Transformer、RNN 和 n-gram model 的内部计算不同，但只要它们是自回归 LM，生成循环看到的都是这一接口。

### 2.2 Bigram model：最小可观察实验台

课程构造了一个只有 17 个符号的 bigram model。它使用一阶 Markov assumption：

$$
P(x_t\mid x_{<t})
\approx
P(x_t\mid x_{t-1})
$$

因此下一词只依赖前一个词。例如课件中：

$$
P(\text{cat}\mid\text{the})=0.25,
\qquad
P(\text{sat}\mid\text{cat})=0.30,
\qquad
P(\text{on}\mid\text{sat})=0.40
$$

完整转移表在 [probability.py](https://github.com/neubig/lminference-fall2025-code/blob/main/02-generation-basics/probability.py) 的 `setup_bigram_model()`。选 bigram 不是因为课程将使用它，而是因为：

- 条件概率矩阵可以完整画出来；
- 采样速度快；
- 真值已知，可以检查估计是否收敛；
- Transformer 上难以解析计算的概念，可先在小模型上验证。

### 2.3 Word2Vec、BERT 为什么不是普通自回归 LM

课堂借学生回答澄清模型类别：

- Word2Vec 的 CBOW/skip-gram 目标用于学习局部词表示，不天然给完整序列分配归一化概率；
- BERT 是 masked language model，能估计被遮蔽位置，却不直接给出标准 left-to-right 序列概率；
- diffusion language model 可以是序列生成模型，但其生成与概率分解不是本讲的自回归形式；
- n-gram、RNN 和 decoder-only Transformer 都可以按 left-to-right 条件概率生成。

关键不是模型是否“处理文本”或“能生成 embedding”，而是它定义了什么概率对象。

## 3. 从逐 token 分布采样出完整序列

### 3.1 基本算法

课程的最小 sampling loop 是：

```python
sequence = [BOS]
while len(sequence) < max_length:
    probs = model.next_token_distribution(sequence)
    next_token = sample(probs)
    sequence.append(next_token)
    if next_token == EOS:
        break
```

bigram 版本只需记住最后一个 token；Transformer 需要对整个 prefix 编码，实际高效实现还会把历史 K/V 写入 KV cache。但概率层面的算法相同。

### 3.2 为什么必须同时有 EOS 与 max length

停止条件通常包括：

1. 采到 EOS；
2. 达到最大生成长度；
3. 应用定义的 stop sequence；
4. 预算、超时或安全策略触发。

即使模型有 EOS，也不能假设一定会在合理长度内生成它。`max_length` 是资源保护边界，可防止异常分布或错误 prompt 让请求无限延长。

> [!tip] AI Infra 视角
> 最大长度不仅影响单请求成本，还影响 KV cache 上限、batch 调度、尾延迟和 admission control。算法参数会直接变成系统容量参数。

## 4. Temperature 改变的是采样分布

### 4.1 公式

给定原始概率 $P(i)$，temperature 后的分布为：

$$
P_T(i)
=
\frac{\exp(\log P(i)/T)}
{\sum_j\exp(\log P(j)/T)}
$$

若模型直接输出 logits $z_i$，等价写成：

$$
P_T(i)=\operatorname{softmax}(z/T)_i
$$

代码核心只有两步：

```python
scaled_logits = logits / temperature
probs = torch.softmax(scaled_logits, dim=-1)
next_token = torch.multinomial(probs, 1)
```

### 4.2 四个重要区域

| Temperature | 分布变化 | 生成行为 |
| --- | --- | --- |
| $T\to0^+$ | 最大 logit 被无限放大 | 趋近 greedy/argmax |
| $0<T<1$ | 分布更尖 | 更保守、重复性更高 |
| $T=1$ | 不改变原分布 | 真正从模型 $P_\theta$ 采样 |
| $T>1$ | 分布变平 | 更多低概率 token、更高多样性与失控风险 |
| $T\to\infty$ | 趋近 vocabulary 上的均匀分布 | 几乎忽略模型偏好 |

代码对 `T=0` 单独执行 `argmax`，因为直接计算 `logits / 0` 没有定义。若多个 token 并列最大，数学极限会把概率分给这些并列项；具体实现的 `argmax` 通常按固定索引规则选一个。

### 4.3 “真实分布”指什么

本讲反复强调 $T=1$ 的特殊性。准确说法是：

> 若目标是描述固定模型本身定义的 $P_\theta$，只有直接按 $T=1$ 的条件分布逐步采样，才不因 temperature 额外扭曲该分布。

$T\ne1$ 不是“错误推理”；它只是从另一个分布 $P_T$ 采样。因此要先说明目标：

- 研究模型自身产生 toxic text 的比例：应尽量从 $P_\theta$ 无偏采样；
- 为用户生成高质量答案：可以有意使用 $T<1$、top-k、top-p、reranking 或其他改变输出分布的方法。

## 5. Sampling-based estimation

### 5.1 为什么用 sampling

对 bigram model，转移矩阵已知；但某些全局量仍不方便解析求解。对 Transformer，序列空间指数增长，直接枚举几乎不可能。

若想估计“给定 prompt 时模型生成 toxic content 的概率”，可以：

1. 从模型采样 $N$ 个输出；
2. 对每个输出计算 indicator $f(y_i)\in\{0,1\}$；
3. 用样本均值估计期望：

$$
\widehat{\mathbb E}[f(Y)]
=
\frac1N\sum_{i=1}^N f(y_i)
$$

当样本独立且确实来自目标分布时，大数定律保证估计随 $N$ 增大而收敛。

### 5.2 用计数恢复 joint 与 conditional probability

课件从采样序列中统计相邻 token：

$$
\widehat P(x,y)
=
\frac{\operatorname{count}(x,y)}
{\text{all observed bigrams}}
$$

$$
\widehat P(y\mid x)
=
\frac{\operatorname{count}(x,y)}
{\operatorname{count}(x)}
$$

这里的 $P(x,y)$ 具体指“从采样语料的全部相邻位置中均匀抽一个 bigram，观察到 $(x,y)$ 的概率”，不是任意两个位置或整条序列的 joint probability。

代码分别对应：

- `estimate_joint_probabilities_from_sequences()`；
- `estimate_conditional_probabilities_from_sequences()`。

实验使用 $10,50,100,500,1000,5000,10000$ 条序列，观察 `the→cat`、`cat→sat`、`on→the`、`sat→on` 等估计。小样本下，低频 bigram 可能一次都未出现，于是估计为 0；样本增多后，均值误差整体下降并逼近真值。

### 5.3 T=1 与估计偏差

如果样本来自 $P_T$，计数估计会收敛到 $P_T$，而不是原模型 $P_\theta$：

$$
\widehat P_N
\xrightarrow[N\to\infty]{}
P_T
$$

因此：

- $T=1$：相对于原模型分布的误差随样本量增加趋向 0；
- $T=0.5$：有限样本造成的随机误差会随样本量降低，但相对于原模型的误差最终停在非零 bias floor；
- greedy：只观察一条确定路径，无法恢复完整分布。

这说的是直接按经验频率计数。若已知 proposal distribution，也可对 $T\ne1$ 的样本做 importance weighting 来估计原目标分布，但会引入权重方差与 support/数值稳定性问题；本讲没有实现这种校正。

> [!question] Q-CMU-L02-2529：为什么增加 T=0.5 的样本量仍不能恢复原分布？
> 因为更多样本只能降低对 $P_{0.5}$ 的估计方差，不能消除 $P_{0.5}$ 与 $P_\theta$ 之间由 temperature 引入的系统偏差。
> 来源：[视频 25:29](https://www.youtube.com/watch?v=UKuPCxozypU&t=1529s)

> [!warning] 课堂运行结果的边界
> [37:28](https://www.youtube.com/watch?v=UKuPCxozypU&t=2248s) 附近，讲师指出现场误差曲线收敛得比预期慢，并怀疑代码可能有 bug、需要课后复查。因此应以大数定律与代码定义理解理论趋势，不把现场某条曲线的数值当作验证过的实验结果。

### 5.4 与训练数据的联系

训练集也可看作对真实文本分布的有限样本：

- 数据越少，稀有事件越可能缺失；
- 经验频率噪声更大，模型更容易过拟合；
- 采集范围、站点选择和过滤规则会让数据产生 selection bias；
- “更多数据”降低 sampling variance，却不会自动消除采集偏差。

课堂把互联网文本称为对语言分布的近似样本。更严格地说，它只代表抓取与过滤流程诱导出的数据分布，不是所有人类语言的无偏样本。

## 6. Marginalization：消去不关心的变量

给定 joint distribution，边缘概率通过求和得到：

$$
P(x)=\sum_yP(x,y),
\qquad
P(y)=\sum_xP(x,y)
$$

“Marginalization”这个名称来自传统概率表：行和、列和写在表格边缘 margin。

在 PyTorch 中只是按维度求和：

```python
marginal_x = joint_prob.sum(dim=1)
marginal_y = joint_prob.sum(dim=0)
```

计算很简单，重要的是建模动作：把当前不需要显式决策的变量积分或求和掉。

## 7. Reasoning trace 是 latent variable

### 7.1 $x$、$z$、$y$

对 reasoning model，可写成：

- $x$：输入 prompt；
- $z$：reasoning trace / Chain of Thought；
- $y$：最终答案。

联合分布分解为：

$$
P(y,z\mid x)
=
P(z\mid x)P(y\mid x,z)
$$

如果最终只关心答案，应边缘化推理轨迹：

$$
P(y\mid x)
=
\sum_zP(y,z\mid x)
=
\sum_zP(z\mid x)P(y\mid x,z)
$$

潜变量不一定是“模型内部 hidden state”。这里 $z$ 是可生成、可采样，但最终不一定展示或评价的随机序列。

### 7.2 Self-consistency 是 Monte Carlo marginalization

Self-consistency 的基本做法是：

1. 对同一输入采样多条 reasoning trace；
2. 从每条 trace 得到答案；
3. 对答案做多数投票或聚合。

若采样对为 $(z_i,y_i)$，某答案 $a$ 的经验概率为：

$$
\widehat P(y=a\mid x)
=
\frac1N\sum_{i=1}^N\mathbf 1[y_i=a]
$$

选择出现次数最多的答案近似 marginal mode：

$$
\hat y
=
\arg\max_a\widehat P(y=a\mid x)
$$

它与“只生成一条 CoT 后直接输出”不同：额外 inference-time compute 用于探索多个 $z$，再消去对具体推理路径的依赖。

> [!note] 与 minimum Bayes risk 的关系
> 课堂只做预告：self-consistency 可以看成离散答案和 0–1 utility 下的简单聚合；MBR 会把候选间效用或风险显式放进决策规则，后续课程再展开。

## 8. 先区分 inference 的两个目标

课程用 temperature bias 引出两个不同目标：

| 目标 | 想回答的问题 | 适合的策略 |
| --- | --- | --- |
| Distribution characterization | 模型本身会以什么概率产生某性质？ | 从目标分布无偏采样，通常 $T=1$ |
| Output optimization | 怎样在预算内得到最符合应用目标的结果？ | 允许 temperature、search、reranking、tools 等改变分布 |

这一区分非常重要。用 $T=0.2$ 的结果测“模型自然毒性率”，估计对象错了；反过来，为用户生成答案时拘泥于 $T=1$ 也没有必要。

> [!abstract] 核心判断
> “Faithfully sample the model” 与 “produce the best application output” 是两个目标；推理算法应先声明自己在优化哪一个。

## 9. 最小 Transformer 代码的作用

讲师在 [42:04](https://www.youtube.com/watch?v=UKuPCxozypU&t=2524s) 快速展示 `nanogpt.py`，没有逐行讲 Transformer。其用途是为后续 decoding 实验提供一个透明、可修改、只有数百行的 GPT-2 实现。

### 9.1 代码结构

```text
token ids
  -> token embedding + position embedding
  -> N × Transformer block
       -> LayerNorm -> causal self-attention -> residual
       -> LayerNorm -> MLP -> residual
  -> final LayerNorm
  -> tied LM head
  -> next-token logits
```

关键模块包括：

- `GPT2Tokenizer`：使用 GPT-2 BPE；
- `GPT2Config`：层数、head 数、embedding dimension、context length；
- `CausalSelfAttention`：一次线性层生成 Q/K/V，使用 causal mask；
- `MLP`：$d\to4d\to d$；
- `Block`：Pre-Norm attention 与 MLP residual；
- `GPT2`：embedding、blocks、final norm 与 LM head。

### 9.2 课堂比较的模型尺寸

| 模型 | Layers | Heads | Embedding dim | 约参数量 |
| --- | ---: | ---: | ---: | ---: |
| GPT-2 small | 12 | 12 | 768 | 117M/124M 量级 |
| GPT-2 medium | 24 | 16 | 1024 | 345M/355M 量级 |
| GPT-2 large | 36 | 20 | 1280 | 774M 量级 |

课程重点不是记住版本命名，而是看清：同一模型家族可用较小模型做 cheap generator，用较大模型做更昂贵的 scorer。

> [!tip] 实现限制
> 课堂 notebook 的生成循环每一步把完整 prefix 再送入模型，没有 KV cache，也没有 batch/continuous batching；它适合教学，不代表生产 inference engine。

## 10. GPT-2 温度生成实验

### 10.1 实验设置

Prompt 固定为：

```text
The future of artificial intelligence is
```

对每个 temperature 生成 3 个 continuation：

| 设置 | 代码行为 | 课堂观察 |
| --- | --- | --- |
| $T=0$ | `argmax` | 三次文本相同、保守、容易泛化成模板 |
| $T=0.5$ | sharpened sampling | 基本流畅，有一些变化 |
| $T=1$ | 原模型采样 | 更多低频选择，质量波动增大 |
| $T=1.5$ | flattened sampling | 词汇更意外，但很快偏离语义和语法 |

`temperature_sample()` 的实现：

```python
if temperature == 0.0:
    return torch.argmax(logits, dim=-1)
probs = F.softmax(logits / temperature, dim=-1)
return torch.multinomial(probs, 1).squeeze()
```

### 10.2 Temperature 不改变 ranking，但改变 probability gap

因为除以正数 $T$ 不改变 logits 的大小顺序：

$$
z_i>z_j
\iff
\frac{z_i}{T}>\frac{z_j}{T}
$$

所以 temperature 不会把原本排名第二的 token 变成 argmax。它改变的是 token 被随机抽到的相对概率。

## 11. Greedy 不等于端到端确定性

课堂从 [47:19](https://www.youtube.com/watch?v=UKuPCxozypU&t=2839s) 花较多时间回答：为什么商业 API 即使设置 `temperature=0`，多次调用仍可能得到不同结果？

### 11.1 数值层原因

浮点加法不满足严格结合律：

$$
(a+b)+c\ne a+(b+c)
$$

并行 reduction、不同 kernel、不同 batch shape 或分布式聚合顺序可能造成极小 logits 差异。如果前两名 token 很接近，argmax 可能翻转；一旦某一步 token 不同，后续 prefix 改变，整条序列会分叉。

### 11.2 系统层放大器

课堂讨论了：

- GPU 并行 reduction 的执行/聚合顺序；
- 多 GPU 执行；
- quantization 带来的额外舍入误差；
- MoE router 的 expert 选择边界；
- agent 的多次模型调用和工具反馈。

更完整地说，动态 batching、不同硬件或 kernel、服务端模型版本、路由与隐藏系统提示也可能影响 API 输出。

> [!warning] 不要把两种随机性混在一起
> `temperature=0` 关闭的是 token sampling randomness；它不保证数值执行、服务调度、模型版本和多步 workflow 都确定。反之，固定随机种子也不足以保证跨硬件、跨 kernel 的 bitwise reproducibility。

### 11.3 课堂 Q&A 延伸

- “硬件扰动等价于多大 empirical temperature？”是一个可实验的问题，但不是一个固定常数；扰动通常与输入、batch、logit margin 和系统配置有关。
- 多 GPU 可能增加聚合路径差异。
- 量化可能放大误差，但是否改变最终 token 取决于 logit margin。
- 不能仅凭少量输出差异可靠反推出闭源服务使用的具体量化格式。

## 12. 怎样评价生成结果

### 12.1 确定性指标

官方 notebook 给出两个简单 diversity metric。

单文本 unique-word ratio：

$$
D_{\text{word}}(y)
=
\frac{|\operatorname{unique\ words}(y)|}
{|\operatorname{words}(y)|}
$$

跨生成 bigram diversity：

$$
D_{\text{bigram}}(Y)
=
\frac{|\operatorname{unique\ bigrams}(Y)|}
{|\operatorname{all\ bigrams}(Y)|}
$$

也可测量 length、repetition、latency、task accuracy 等。优点是便宜、稳定、可复现；缺点是这些表面指标不直接等于语义质量。

> [!note] 指标局限
> Unique n-gram ratio 对文本长度、tokenization 和样本数敏感。不同温度若生成长度不同，直接比较会混入 length effect；正式实验应固定预算、报告置信区间，并搭配任务指标。

### 12.2 LLM-as-a-judge

课件向外部模型发送：

```text
Rate the fluency and coherence of this text on a scale of 0-10.
10 = perfect. Only respond with a number.
```

这使开放式质量评价变得便宜、易扩展，但课程提醒：

1. Judge 不擅长某任务时，也难以可靠评价该任务；
2. Judge 可能偏好自己的模型家族或写作风格；
3. 单一 prompt 和单一分数可能受格式、顺序、长度影响；
4. Judge score 需要用人工标注或可验证指标做校准。

### 12.3 课堂实验结论

少量样本上，greedy 文本的平均 fluency score 约为 5.3/10，多样性最低。随 temperature 上升：

$$
\text{diversity}\uparrow,
\qquad
\text{fluency/coherence}\downarrow
$$

这是 temperature sampling 常见的 quality-diversity trade-off，不是所有 decoding 方法不可突破的铁律。讲师预告后续方法可以在两维上 Pareto-dominate 某些 temperature 设置。

## 13. Meta-generation：先生成候选再评分

### 13.1 实验流程

课程实现最简单的 reranking：

1. GPT-2 small 从 prompt 采样 $N$ 个候选；
2. GPT-2 small 和 GPT-2 medium 分别计算每个候选的 log probability；
3. 比较两个模型偏好；
4. 可按 medium model 分数选择候选。

对 token 序列 $w_{1:n}$：

$$
\log P_\theta(w_{1:n})
=
\sum_{t=1}^{n}
\log P_\theta(w_t\mid w_{<t})
$$

代码先把 logits 与 next-token labels 错开一位：

```python
shift_logits = logits[..., :-1, :]
shift_labels = inputs[..., 1:]
log_probs = F.log_softmax(shift_logits, dim=-1)
token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
```

### 13.2 为什么使用平均 token log probability

完整序列 log probability 会随长度累加更多负数，长文本通常绝对值更大。课件使用：

$$
\bar\ell_\theta(y)
=
\frac{1}{|y|}
\sum_{t=1}^{|y|}\log P_\theta(y_t\mid y_{<t})
$$

它等价于负的 token-level cross-entropy，并与 perplexity 单调对应：

$$
\operatorname{PPL}(y)=\exp(-\bar\ell_\theta(y))
$$

长度归一化让长短候选更可比较，但不是完全消除 length bias：tokenization、EOS、短而通用的文本以及平均分的“稀释效应”仍会影响排序。

### 13.3 为什么 small model 常偏好自己的样本

候选来自 GPT-2 small 自己的分布，形成 proposal bias：

$$
y_i\sim P_{\text{small}}(y\mid x)
$$

因此候选集天然集中在 small model 喜欢的区域，small model 往往给自己的样本更高平均 log probability。课堂只看到少数 medium-preferred cases，并观察它们有时使用更意外但仍基本流畅的词汇。

这个观察不能直接推出“较大模型总是更有趣”；它受候选分布、随机种子、prompt、长度和模型家族影响。

## 14. Reranking 与 speculative decoding 的边界

课堂看到“small model 先生成、large model 再检查”的拓扑，学生联想到 speculative decoding。二者外形相似，但目标不同：

| 方法 | Small/draft model | Large/target model | 结果 |
| --- | --- | --- | --- |
| Reranking | 生成多个完整候选 | 给候选打分并选择 | 主动改变最终选择分布，以提高某种质量 |
| Speculative decoding | 提议若干 future tokens | 并行验证并按接受规则修正 | 在规则正确时保持 target model 分布，同时降低 latency |

标准 speculative decoding 不能简单写成“small 错了就停”。严格实现需要 target/draft probability 与 acceptance/rejection correction，才能保持 target distribution。课堂这里只借此说明共同的 draft-then-verify 结构，后续效率章节再讲完整算法。

## 15. Meta-generation 的实际用途

在 [1:10:25](https://www.youtube.com/watch?v=UKuPCxozypU&t=4225s) 的 Q&A 中，讲师给出几类用途：

### 15.1 昂贵 verifier 只检查少量候选

如果最强 evaluator 是一个会检索、调用工具或核对事实的 agent，逐 token 放进生成循环太贵。可以先用便宜模型生成少量候选，再对每个候选做昂贵验证，例如检查 hallucination。

### 15.2 Best-of-N / reranking

$$
y^*
=
\arg\max_{y_i\in\mathcal C(x)}s(x,y_i)
$$

其中候选集 $\mathcal C(x)$ 来自 generator，$s$ 可来自 reward model、LLM judge、规则、执行结果或人类偏好模型。

### 15.3 Self-consistency

多次生成 reasoning 与答案，再按答案多数投票。它不是按单个候选的 likelihood 取最大，而是用样本频率近似对 latent reasoning path 的 marginalization。

### 15.4 分层计算

```text
cheap generator
  -> medium scorer
  -> expensive verifier/tool agent
  -> final response
```

这种 cascade 把昂贵计算集中在少数有希望的候选上，是 inference-time scaling 的基本模式。

## 16. 面向 AI Infra 的实现含义

### 16.1 Notebook 生成循环为何慢

`generate_with_temperature()` 每生成一个 token 都重新 forward 完整 prefix：

```python
for _ in range(max_length):
    logits, _ = model(input_ids)
    next_token = sample(logits[0, -1])
    input_ids = torch.cat([input_ids, next_token], dim=1)
```

缺少 KV cache 时，历史 attention K/V 被反复计算。生产系统通常：

- prefill 一次处理 prompt；
- decode 每步只处理新 token；
- KV cache 保存历史状态；
- batch 多个请求提高硬件利用率。

### 16.2 Meta-generation 的资源账本

若生成 $N$ 个长度 $L$ 的候选，并用 scorer 全量重算：

- generator decode cost 近似随 $N\times L$ 增长；
- scorer 可把每个完整候选作为 prefill 并行评分；
- 候选共享 prompt，可利用 prefix/KV reuse；
- 并行度提高吞吐，但增大显存、队列与峰值 compute；
- 对在线服务，需要同时观察 TTFT、TPOT、端到端 latency 和 tail latency。

### 16.3 Apple Silicon 与 device 选择

课堂提醒 Mac 用户优先尝试 PyTorch MPS；代码也支持自动选择 CUDA/MPS/CPU。是否更快仍取决于模型、dtype、算子支持和数据搬运，不能仅凭 device 名称判断。

## 17. 本讲最容易混淆的概念

| 易混淆项 | 正确区分 |
| --- | --- |
| LM probability vs decoding quality | 高概率文本不一定最符合外部任务目标 |
| $T=1$ vs “最好” | $T=1$ 忠实采样模型，不保证用户体验最好 |
| $T=0$ vs reproducible | greedy token rule 不保证整个服务栈确定 |
| Sampling variance vs bias | 增加样本降低方差，不能消除错误采样分布造成的 bias |
| CoT vs hidden state | 本讲的 $z$ 是可采样 reasoning trace，不是任意内部 activation |
| Reranking vs speculative decoding | 前者改变选择，后者主要在保持 target distribution 下加速 |
| Per-token normalization vs length neutrality | 平均 logprob 缓解长度效应，但不完全消除 |
| LLM judge vs ground truth | Judge 是可扩展 proxy，需要校准而不是盲信 |

## 18. 本讲结论

1. 自回归 LM 用条件概率定义序列分布，generation loop 把 next-token distribution 递归变成完整序列。
2. Bigram model 是验证 sampling、joint/conditional probability 与 convergence 的透明实验台。
3. Temperature 通过缩放 logits 改变采样分布；$T=1$ 才保持原模型分布。
4. Monte Carlo sampling 可以估计难以解析计算的模型性质；更多样本降低方差，但不消除 sampling bias。
5. Marginalization 把不关心的变量求和掉；reasoning trace 可视为 latent variable。
6. Self-consistency 通过多条 reasoning path 近似答案边缘分布。
7. Greedy decoding 在算法层确定，但 GPU 数值、量化、MoE、分布式与多步 agent 会破坏端到端复现。
8. 生成评价至少应区分确定性任务指标、diversity proxy 与 LLM judge，并报告各自局限。
9. Meta-generation 用 generator 产生候选，再用 scorer/verifier 选择或聚合，是 inference-time compute scaling 的基本结构。
10. 教学代码揭示算法结构，但没有 KV cache、batching 和高性能 serving 优化，不能直接代表生产性能。

## 19. 自测问题

1. 为什么 $P(y\mid x)$ 与 $P(x,y)$ 不是同一个概率对象？— [03:43](https://www.youtube.com/watch?v=UKuPCxozypU&t=223s)
2. Bigram model 的 Markov assumption 丢弃了哪些上下文？— [09:32](https://www.youtube.com/watch?v=UKuPCxozypU&t=572s)
3. 为什么生成必须同时设置 EOS 与 max length？— [12:16](https://www.youtube.com/watch?v=UKuPCxozypU&t=736s)
4. 推导 $T\to0^+$、$T=1$、$T\to\infty$ 时的 temperature distribution。— [12:34](https://www.youtube.com/watch?v=UKuPCxozypU&t=754s)
5. 为什么 $T=0.5$ 上无限采样也不能恢复 $P_\theta$？— [25:29](https://www.youtube.com/watch?v=UKuPCxozypU&t=1529s)
6. 如何用采样估计模型生成 toxic content 的概率，并给出置信区间？— [20:54](https://www.youtube.com/watch?v=UKuPCxozypU&t=1254s)
7. Self-consistency 在边缘化哪个变量？— [33:53](https://www.youtube.com/watch?v=UKuPCxozypU&t=2033s)
8. 什么时候 inference 应忠实采样，什么时候应主动改变分布？— [40:32](https://www.youtube.com/watch?v=UKuPCxozypU&t=2432s)
9. 为什么 `temperature=0` 的 API 仍可能返回不同文本？— [47:19](https://www.youtube.com/watch?v=UKuPCxozypU&t=2839s)
10. LLM-as-a-judge 的 capability ceiling 和 self-preference 分别是什么？— [59:27](https://www.youtube.com/watch?v=UKuPCxozypU&t=3567s)
11. 为什么总 logprob 偏向短文本，per-token logprob 又仍不完全 length-neutral？— [1:05:00](https://www.youtube.com/watch?v=UKuPCxozypU&t=3900s)
12. Reranking 与 speculative decoding 虽然都使用 small/large model，为什么目标不同？— [1:01:54](https://www.youtube.com/watch?v=UKuPCxozypU&t=3714s)
13. 若 expensive verifier 是一个检索 agent，怎样设计候选数、并行度与 latency budget？— [1:10:25](https://www.youtube.com/watch?v=UKuPCxozypU&t=4225s)

## 20. 课件代码地图

| 文件 | 本讲作用 | 关键入口 |
| --- | --- | --- |
| [probability.ipynb](https://github.com/neubig/lminference-fall2025-code/blob/main/02-generation-basics/probability.ipynb) | Bigram、采样估计、边缘化、潜变量、temperature bias | `sample_from_bigram_model()`、`estimate_*()` |
| [generation.ipynb](https://github.com/neubig/lminference-fall2025-code/blob/main/02-generation-basics/generation.ipynb) | GPT-2 temperature generation 与评价 | `temperature_sample()`、`generate_with_temperature()` |
| [meta_generation.ipynb](https://github.com/neubig/lminference-fall2025-code/blob/main/02-generation-basics/meta_generation.ipynb) | small 生成、small/medium 评分 | `calculate_log_probability()`、`run_meta_generation_experiment()` |
| [nanogpt.py](https://github.com/neubig/lminference-fall2025-code/blob/main/02-generation-basics/nanogpt.py) | 最小 GPT-2 实现 | `CausalSelfAttention`、`Block`、`GPT2` |
| [plotting_utils.py](https://github.com/neubig/lminference-fall2025-code/blob/main/02-generation-basics/plotting_utils.py) | 概率热图与收敛曲线 | plotting helpers |

## 21. 关联内容

- 课程索引：[CMU 11-763](CMU%2011-763.md)
- 上一讲：[Lecture 01：Introduction to Language Models and Inference](CMU%2011-763%20-%20Lecture%2001%20-%20Introduction%20to%20Language%20Models%20and%20Inference.md)
- 主题入口：[LLM Inference](../../topics/inference/LLM%20Inference.md)
- 模型结构：[Transformer Architecture](../../topics/model-architecture/Transformer%20Architecture.md)
