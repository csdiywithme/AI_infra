---
type: course-note
status: developing
course: "[[CMU 11-763]]"
lecture: 3
lecture_date: 2025-09-02
area: inference
topics:
  - "[[LLM Inference]]"
aliases:
  - CMU 11-763 Lecture 03
  - Common Sampling Methods for Modern NLP
  - Sampling Methods
video_url: https://www.youtube.com/watch?v=fvbR-9OXUvo
---

# Lecture 03：Common Sampling Methods for Modern NLP

> [!abstract] 本讲一句话
> Sampling 的选择是在逐 token 分布上决定“哪些候选保留、怎样重新分配概率”；temperature 和截断控制长尾，locally typical sampling 则让 token 的信息量接近当前分布的预期信息量，但这些选择都必须按任务验证。

## 来源与范围

- 课程：CMU 11-664/763 — Inference Algorithms for Language Modeling，Fall 2025。
- 讲师：Amanda Bertsch；日期：2025-09-02。
- [课程视频](https://www.youtube.com/watch?v=fvbR-9OXUvo)，时长 1:01:33；本文完整核对英文字幕，结尾转交学生报告，报告本身不在这段录像内。
- [官方 Slides，共 30 页](https://docs.google.com/presentation/d/1fwbyhdYlpKcZ8vyzi0Fvf7zH7yr0l-hsYS14B-5ze9w/edit)；页码按官方 PDF。
- [课程主页](https://www.phontron.com/class/lminference-fall2025/)。
- 课件相关论文：[Locally Typical Sampling](https://arxiv.org/abs/2202.00666)、[A Thorough Examination of Decoding Methods in the Era of LLMs](https://arxiv.org/abs/2402.06925)、[Trading Off Diversity and Quality in Natural Language Generation](https://arxiv.org/abs/2004.10450)。

> [!note] 阅读边界
> 主线、课堂例子与问答来自字幕及课件；标为“推导补充”“实现补充”的内容用于补齐数学和工程细节。课堂的模型/API 默认参数、RLHF 校准图和模型表现均按 2025 年教学背景理解，不当作当前产品规范。

> [!warning] 字幕中的符号需要校正
> 自动字幕把 ergodic、Mirostat、η-sampling 等词识别错误；讲授局部典型性时也有关于 log probability 正负号的现场纠正。本文统一用 surprisal $I(v)=-\log p(v)\ge0$ 与熵 $H\ge0$ 比较，避免把负的 logprob 直接与正的熵相减。

## 视频时间索引

| 时间 | 内容 | 课件 |
| --- | --- | --- |
| [00:44](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=44s) | 课程结构：分布性质、简单采样、典型采样 | 2–3 |
| [01:44](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=104s) | 局部归一化、prefix 概率的单调性 | 4–5 |
| [04:55](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=295s) | Calibration 与 post-training | 6–8 |
| [07:44](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=464s) | Entropy、cross-entropy、perplexity | 9–11 |
| [11:48](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=708s) | Ancestral sampling 与长尾 | 13–14 |
| [13:53](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=833s) | Temperature、top-k、top-p、epsilon | 15–18 |
| [17:29](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=1049s) | Generation config 与隐含默认值 | 19–20 |
| [19:38](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=1178s) | 60% 正面硬币：mode 与 typical set | 21 |
| [21:48](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=1308s) | 平稳性、遍历性、熵率 | 22–23 |
| [28:43](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=1723s) | EOS 吸收态与全局典型性的困难 | 24 |
| [31:14](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=1874s) | Locally typical sampling 算法 | 25–26 |
| [33:06](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=1986s) | 熵、surprisal、排序的课堂讨论 | 26 |
| [45:31](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=2731s) | 人类语言的信息量动机 | 27 |
| [48:44](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=2924s) | Mirostat 与 η-sampling 简介 | 28 |
| [50:05](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=3005s) | 任务选择与评测策略 | 29–30 |
| [55:25](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=3325s) | 按任务调参、熵与错误风险的问答 | 课后讨论 |

## 1. 先说明采样器能看到什么

给定输入 $x$ 和已生成前缀 $y_{<t}$，模型提供词表上的条件分布：

$$
p_t(v)=P_\theta(y_t=v\mid x,y_{<t}),\qquad v\in\mathcal V.
$$

本讲把模型当作这一接口，不讨论分布是 Transformer、RNN 还是其他架构算出的。

采样器通常先得到 logits $z_t$，再形成：

$$
p_t(v)=\frac{e^{z_t(v)}}{\sum_{u\in\mathcal V}e^{z_t(u)}}.
$$

这里需要区分三个对象：

1. 模型原始分布 $p_t$；
2. 解码器修改后的采样分布 $q_t$；
3. 实际采出的一个 token $y_t$。

一次采样只得到一个结果，不会告诉我们分布的全部性质。
而修改 $q_t$ 会沿自回归过程改变后续上下文，因此也改变完整序列分布。

## 2. 局部归一化：为什么早期选择不能靠后文“加回概率”

课件 4–5 页与 [01:44](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=104s) 强调：每一步归一化使下一 token 概率和为 1。

$$
P(y_{1:t}\mid x)=\prod_{i=1}^{t}p_i(y_i).
$$

因为 $0\le p_i\le1$：

$$
P(y_{1:t+1}\mid x)\le P(y_{1:t}\mid x).
$$

logprob 表达为：

$$
s(y_{1:t})=\sum_{i=1}^{t}\log p_i(y_i),
\qquad s(y_{1:t+1})\le s(y_{1:t}).
$$

课堂例子是续写 “The U.S. president in 2023 was …”：

- “Barack Obama’s …” 开头看起来可能不合事实，但完成为 “former VP” 后成立；
- “Joseph Biden’s …” 开头看起来合理，但完成为 “daughter” 后错误。

低概率 prefix 的完整路径仍可能胜过另一条路径，因为另一条也可能继续损失大量概率。
单调性只约束一条路径的 prefix 与其 extension，不禁止不同路径之间交换排序。

> [!example] 推导补充
> 路径 A 的 prefix 概率为 0.2，后续为 0.9，完整概率为 0.18；路径 B 的 prefix 概率为 0.5，后续为 0.1，完整概率为 0.05。A 最终超过 B，但 0.18 依然小于 A 自己此前的 0.2。

局部归一化使训练与逐步生成方便；完整输出是否满足事实、格式或风格等全局要求，则需要后续的搜索、控制或验证。

## 3. Calibration：模型置信度是否能当正确率

课件 6–8 页展示 calibration 曲线，并讨论 post-training 的影响。

理想的校准关系可写成：

$$
P(\text{预测正确}\mid\text{置信度}=c)=c.
$$

例如一组模型报告 0.8 置信度的多选答案中，大约 80% 正确，才符合这一含义。

这里比较的是大量类似预测的统计频率，不是说一个答案“80% 真、20% 假”。
也不能从单条完整自然语言输出的 logprob 直接推出其事实正确率。

课堂对比 GPT-4 技术报告中的 MMLU 校准图：pretraining 后的校准关系较好，RLHF 后发生明显变化。
因此，输出更符合人类偏好与概率更校准不是同一指标。

课件还引用 Kalai 与 Vempala 关于校准和 hallucination 的理论工作。
其意义是：即使训练数据事实正确，也不能把概率建模理解为把所有错误输出概率压成零。
具体不可能性结论依赖论文的统计设置和假设，不应扩张为“任何校准模型一定答错每类事实题”。

## 4. Entropy、cross-entropy 与 perplexity

### 4.1 Entropy 衡量一个分布的预期信息量

$$
H(p_t)=-\sum_{v\in\mathcal V}p_t(v)\log p_t(v).
$$

单个 token 的信息量是：

$$
I_t(v)=-\log p_t(v).
$$

因此：

$$
H(p_t)=\mathbb E_{v\sim p_t}[I_t(v)].
$$

熵属于整个分布；单个候选具有 surprisal，而不是一个独立的“token entropy”。

- 完全集中在一个 token 上，熵为 0；
- $V$ 个 token 均匀分布，熵为 $\log V$；
- 熵越高，平均而言下一 token 越难预测。

若使用 $\log_2$，单位是 bits；使用自然对数，单位是 nats。
比较数值前必须统一对数底。

### 4.2 Cross-entropy 引入参考分布

令真实或目标分布为 $p^*$，模型为 $p_\theta$：

$$
H(p^*,p_\theta)=-\sum_v p^*(v)\log p_\theta(v).
$$

在 next-token prediction 中，单个训练样本的 label 通常是 one-hot，因此 loss 为：

$$
\ell_t=-\log p_\theta(y_t^{\mathrm{gold}}\mid x,y_{<t}^{\mathrm{gold}}).
$$

这不能理解为自然语言真的只有一个合法下文。
One-hot 是我们对这一条观察样本使用的监督信号。

推导补充：

$$
H(p^*,p_\theta)=H(p^*)+D_{KL}(p^*\Vert p_\theta).
$$

目标分布固定时，降低 cross-entropy 等价于降低这一方向的 KL。

### 4.3 Perplexity 是指数化的平均 NLL

$$
\operatorname{PPL}
=\exp\left(-\frac1N\sum_{t=1}^{N}\log p_\theta(y_t\mid y_{<t})\right).
$$

课件写 $2^{\text{cross-entropy}}$，对应以 2 为底的 log；自然对数对应 $\exp$。

“等效多少面骰子”的比喻帮助理解有效不确定性：均匀六选一的 perplexity 是 6。
但一般模型的 $1/\mathrm{PPL}$ 不是其 top-1 accuracy。
它是参考 token 概率的几何平均，不能被误写成模型每六次就恰好答对一次。

还应区分：

- 固定 held-out corpus 上的 PPL：主要评估模型对数据的拟合；
- 模型自己生成文本的 PPL：同时受采样策略选择偏差影响。

低温生成更可预测，并不证明模型对真实数据拟合更好。

## 5. Ancestral sampling 与长尾风险

[11:48](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=708s)，课件 13–14 页。

Ancestral sampling 递归执行：

$$
y_t\sim p_\theta(\cdot\mid x,y_{<t}).
$$

如果每一步都不修改分布，完整路径就服从自回归模型定义的分布。
这个性质对概率估计很重要，但不自动保证输出适合某个任务。

课堂以大词表说明长尾：每个罕见 token 概率很低，很多这样的项加起来仍可能有大量质量。
多次解码又提供了多次进入不合适区域的机会。

> [!example] 推导补充
> 假设每一步进入某个坏事件的条件概率都是 $r=0.01$，独立近似下 100 步至少发生一次的概率为 $1-(1-r)^{100}\approx63.4\%$。实际自回归步骤不独立，坏 token 的后续影响也依赖上下文，这个算式只展示“逐步小风险可以累积”。

长尾中的 token 不全是错误：罕见人名、专业词或代码符号也可能在那里。
所以截断是在任务质量与覆盖度之间做选择。

## 6. Temperature：保持排序，改变相对尖锐程度

课件 15 页：

$$
q_T(v)=\operatorname{softmax}(z(v)/T)
=\frac{p(v)^{1/T}}{\sum_u p(u)^{1/T}},\qquad T>0.
$$

- $T=1$：恢复原始模型分布；
- $0<T<1$：更集中在高概率候选上；
- $T>1$：更平坦，长尾占比可能增加；
- $T\to0^+$：集中到最大 logit 集合，实际 greedy 可另设 tie-break。

有限正温度不会改变 logits 的排序，也不会主动把有限 logit 的 token 概率精确变成零。

例如两个候选原始概率比为 $4:1$，温度 0.5 后变成 $16:1$。
Temperature 控制随机性，但不是直接控制“创造力”或“正确率”的单一旋钮。

## 7. 截断方法：保留集合之后必须重新归一化

对一个保留集 $S$：

$$
q(v)=\frac{p(v)\mathbf1[v\in S]}{\sum_{u\in S}p(u)}.
$$

### 7.1 Top-k：固定候选数量

令 $S$ 为概率最大的 $k$ 个 token。

课堂例子（课件 16 页）：$k=6$ 时，“The” 后的 top-6 只覆盖约 0.68 概率，而 “The car” 后 top-6 覆盖约 0.99。

因此 $k$ 固定，不代表每一步丢弃的概率质量固定。
平坦分布可能删掉很多合理输出；尖锐分布可能保留几乎没有质量的尾部项。

### 7.2 Top-p：固定目标概率覆盖量

将概率从大到小排序为 $p_{(1)},p_{(2)},\ldots$，取最小的 $m$ 满足：

$$
\sum_{i=1}^{m}p_{(i)}\ge p_{\mathrm{cutoff}}.
$$

再从前 $m$ 项按重新归一化后的概率采样。

课件用 $p_{\mathrm{cutoff}}=0.94$：平坦状态保留很多候选，尖锐状态只需少数候选。
这里的 $p$ 是阈值超参数，不是某个 token 的概率。

跨过阈值的那一项通常仍要保留；若代码仅保留累计和小于阈值的项，会漏掉边界 token，甚至得到空集。

### 7.3 Epsilon sampling：固定单项概率下限

$$
S_\epsilon=\{v:p(v)\ge\epsilon\}.
$$

与 top-p 的“累计质量”不同，它判断每个候选是否足够可能。
实现必须处理所有项都低于阈值的情况，例如显式保留最大概率项。

| 方法 | 固定的东西 | 随上下文变化的东西 |
| --- | --- | --- |
| Top-k | 候选数 | 概率覆盖量 |
| Top-p | 累计概率下限 | 候选数 |
| Epsilon | 单项概率下限 | 候选数及覆盖量 |
| Temperature | logit 缩放比例 | 分布熵与各 token 概率 |

### 7.4 一个统一数值例子

以下为推导补充，使用 $p=(0.50,0.25,0.15,0.07,0.03)$：

- Top-k，$k=2$：保留前两项，总质量 0.75；采样概率为 $(2/3,1/3)$。
- Top-p，阈值 0.8：前两项不足，保留前三项，总质量 0.9。
- Epsilon，阈值 0.1：也保留前三项，但这是该分布下的巧合。
- Temperature，$T=0.5$：全部有限概率项仍保留，概率与原概率的平方成正比。

## 8. 默认参数会悄悄改变你以为在做的实验

[17:29](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=1049s)，课件 19–20 页。

课堂展示 `model.generate(do_sample=True, num_beams=1, ...)`，并现场查看模型的 generation config。
重点不是记住某个版本的 top-k/top-p 默认数字，而是理解默认值可能来自多层：

1. 库版本的默认设置；
2. 模型仓库附带的 generation config；
3. 调用方显式覆盖；
4. 更外层 serving 或评测框架。

只设 `temperature=1` 不代表 ancestral sampling：top-k、top-p、repetition penalty 或其他 processor 可能仍有效。

实现补充：记录“最终生效”的完整解码配置，而不是只保存调用代码中改动的两项。
还应记录模型 revision、tokenizer、随机种子、stop rule、最大长度和库版本。

Temperature 与截断的组合顺序也可能影响结果。
温度保持排序，但会改变累计质量，所以先调温再 top-p 与先 top-p 再调温的保留集合可以不同。

## 9. Mode 不等于典型输出：100 次硬币实验

[19:38](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=1178s)，课件 21 页。

假设硬币独立抛掷，正面概率为 0.6，反面为 0.4。
任一有 $k$ 个正面的特定有序序列概率为：

$$
P(y)=0.6^k0.4^{100-k}.
$$

最高概率的单条序列是 100 次全正面，因为每个位置都选择 0.6。
但它的概率只有：

$$
0.6^{100}\approx6.53\times10^{-23}.
$$

典型观察更接近 60 次正面、40 次反面的某种排列。
单条这样的序列概率较低，但这样的排列有 $\binom{100}{60}$ 条。

因此需要区分：

- 最可能的一个点；
- 汇聚大量概率质量的一片集合。

这也是“高概率文本可能过于单调”的信息论动机。
它不直接证明所有任务都应使用 typical sampling；事实题、数学题和开放写作的效用不同。

## 10. 从全局典型集到局部典型性

### 10.1 熵率与典型集

课件 22–23 页引入离散、平稳、遍历随机过程。
对满足适当条件的过程，熵率为：

$$
\mathcal H(Y)=\lim_{T\to\infty}\frac1T H(Y_1,\ldots,Y_T).
$$

典型集可写作：

$$
\mathcal T^{(T)}_\epsilon
=\left\{y_{1:T}:\left|-\frac1T\log P(y_{1:T})-\mathcal H(Y)\right|<\epsilon\right\}.
$$

它要求平均每个符号的信息量接近过程熵率，而非要求每个 token 都是局部最可能。

### 10.2 语言模型为什么不能直接套用全部保证

课堂 [25:12](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=1512s) 讨论“上下文改变导致熵改变，是否违反平稳性”。
一个时间齐次的有限上下文转移规则可以依赖历史状态，同时不显式依赖绝对时间。
但严格的平稳性还涉及初始状态分布；有限窗口本身不是平稳性的充分证明。

更直接的障碍是 EOS：常见生成过程到 EOS 就结束，相当于进入吸收态。
从该状态不能继续访问所有普通 token 状态，所以不能随意援引遍历过程的长序列结论。

讲师据此把全局典型性作为动机，转向容易实施的逐步近似。
“Locally typical”不等价于已证明生成序列属于一个满足全部渐近定理的全局典型集。

## 11. Locally typical sampling 的准确算法

课件 25–26 页，[31:14](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=1874s)。

每步先计算：

$$
H_t=-\sum_v p_t(v)\log p_t(v),\qquad
d_t(v)=|{-\log p_t(v)}-H_t|.
$$

再执行：

1. 按 $d_t(v)$ 从小到大排序；
2. 沿这个顺序累积原始概率，直到覆盖质量达到阈值 $\tau$；
3. 在保留集合内重新归一化；
4. 按概率采样，更新前缀。

它借用了 nucleus 的“累积到目标质量”操作，但排序标准发生了变化。
因此不能实现成先普通 top-p，再在其内部按概率挑选。

### 11.1 为什么它可能丢掉最高概率 token

若某 token 特别可预测，其 $-\log p$ 可能显著小于分布熵。
它携带的信息量低于平均预期，因此未必靠近 $H_t$。

推导例子：

$$
p=(0.4,0.2,0.2,0.2),\qquad H(p)\approx1.332\ \text{nats}.
$$

第一项 surprisal 约为 0.916，距离 0.416；其余项 surprisal 约为 1.609，距离 0.277。
所以后三项在 typical 排序中反而优先。
若质量阈值为 0.6，可能只保留后三项，最高概率 token 被排除。

极尖锐分布则不同：若一项概率接近 1，熵与该项 surprisal 都接近 0，保留最高概率项很自然。
完全均匀时所有 surprisal 都等于熵，全部距离为 0，应明确 tie-break。

### 11.2 最小伪代码

```python
logp = log_softmax(logits, dim=-1)
p = exp(logp)
H = -(p * logp).sum()
distance = abs(-logp - H)
order = argsort(distance)
mass = cumsum(p[order])
m = first_index(mass >= typical_mass) + 1
keep = order[:m]
q = p[keep] / p[keep].sum()
next_token = keep[categorical_sample(q)]
```

这是说明数学步骤的伪代码，不是绑定某个库版本的接口。
对被 mask 的 $p=0$ 项应使用稳定的熵计算，避免 `0 * -inf` 导致 NaN。

## 12. 人类语言动机、Mirostat 与 η-sampling

[45:31](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=2731s)，课件 27 页介绍研究观察：人类文本的 token 信息量与上下文预期信息量有联系。
动机是避免两端：完全意外的词串难理解；永远最可预测的词串信息量太低。

这种观察依赖用于评分的语言模型、数据域和统计方式。
不能从一张图推出“人类每个 token 都精确等于条件熵”。

课堂 [48:44](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=2924s) 简短提到 Mirostat：指定目标 perplexity/信息量水平，依据已生成结果反馈更新截断控制量。
本讲没有展开其完整控制方程，因此不把它写成详细讲授过的算法。

η-sampling 则结合固定概率门槛与 entropy-dependent threshold。
课件 28 页图示强调阈值会随当前分布的不确定性变化。
它与 typical sampling 共享“信息量可指导采样”的动机，但不是同一个排序算法。

实现时应以所用方法的原始定义为准，不能把课件简写的“熵阈值”误读成直接对 token 概率设一个上界。

## 13. 怎样选择采样策略并评估

课件 29–30 页的结论是先看任务，再在目标模型上试验。

| 任务 | 应重点观察 | 容易误用的指标 |
| --- | --- | --- |
| 开放写作 | 连贯性、多样性、重复、偏题 | 只比较生成文本 PPL |
| 事实问答 | 正确率、答案规范化、校准 | 把熵低直接当事实正确 |
| 数学/代码 | 最终可验证结果、pass@k、预算 | 只看措辞丰富程度 |
| 多候选系统 | 候选覆盖率、相关性、后续选择收益 | 只看单个样本分数 |

课堂建议在 validation set 上做参数 sweep，选一个表现稳定的区域，然后固定解码协议比较模型。
若每个 benchmark 都选择其最有利的参数，却把结果汇总成一个“通用默认表现”，会放大评测偏差。

小模型可以快速帮我们发现实现问题，但采样退化行为可能与大模型不同。
最终应在真正要研究或部署的模型上验证。

## 14. 课堂问答与应保留的细节

**Q：模型能按任务自己选采样策略吗？** — [55:25](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=3325s)

模型的原始分布已经会随问题变尖或变平；系统还可以显式按请求类型调整 generation settings。
讲师把后者作为合理系统设计，没有声称所有商业 API 都已如此实现。

**Q：熵高是不是特别容易出错？** — [59:44](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=3584s)

不一定。高熵可能只表示同一个意思有很多表达方式；关键数学分歧可能只有 533 与 534 两项，熵不高却决定正确与否。

**Q：typical sampling 对确定事实是否合理？** — [46:51](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=2811s)

自然语言描述事实仍有多种表述；数学或代码可能更受约束，应具体评测。
不能因信息论动机漂亮就默认它在所有任务优于 top-p。

**Q：为什么 typical sampling 不就是 top-p？** — [44:16](https://www.youtube.com/watch?v=fvbR-9OXUvo&t=2656s)

区别在排序：top-p 按概率大小，typical 按 surprisal 离熵的距离。
排序后的概率累加和重新归一化仍使用原概率质量。

## 15. 面向 AI Infra 的实现含义

本节是从课程算法得到的工程补充。

### 15.1 采样器不是总能忽略的零成本步骤

大词表上，softmax、top-k、排序、前缀和、随机采样和 host/device 同步都可能进入 decode 热路径。
Top-p 与 typical sampling 的一般实现需要排序；后者还要计算熵和距离。
实际瓶颈取决于 batch、词表大小、kernel 融合及模型本身耗时。

### 15.2 观测应保留原始分布与修改后分布的区别

若日志只保存最终采样分布的 logprob，就无法直接研究原模型的校准或熵。
建议明确区分 raw logits、raw logprob、processed logprob 和最终候选集。
对用户报告的“token confidence”也应说明是哪一层。

### 15.3 可复现的配置需要包含执行顺序

同样写着 temperature 0.8、top-p 0.9，不同 processor 顺序可能产生不同候选集。
随机种子、并行 batch 调度、浮点差异又可能影响最终 token。
对研究实验，记录 sampling 实现和配置 revision 与记录模型权重同样重要。

### 15.4 多候选预算要看有效多样性

如果采样分布过尖，生成 20 个候选可能只有几个不同答案。
这会增加 token 成本而几乎不提高覆盖率。
如果过平，候选彼此不同却大多不可用，verifier 同样浪费资源。
合理策略应同时测量成功率、候选重复率、总 token 与端到端 latency。

## 16. 本讲最容易混淆的概念

| 混淆 | 正确区分 |
| --- | --- |
| 局部归一化 / 全局排序固定 | 每条路径自身概率不增加，不禁止路径间排名变化 |
| Entropy / surprisal | 前者是分布期望，后者是单个事件的信息量 |
| PPL / accuracy | 几何平均概率不能直接换算 top-1 正确率 |
| 低熵 / 事实正确 | 分布可以非常自信地犯错 |
| Top-p / typical | 累积规则相似，排序标准不同 |
| 最可能单条 / 典型集合 | 点的概率与集合总质量不同 |
| $T=1$ / 忠实采样 | 还必须关闭其他改变分布的规则 |
| 低生成 PPL / 好模型 | 采样策略可以主动选择容易预测的文本 |

## 17. 自测问题与面试回答

1. 局部归一化是否意味着低概率 prefix 永远不能赢？

    **面试回答：** 不能。它只保证完整序列概率不超过自己的 prefix 概率；另一条路径可能在后续损失更多质量，所以排名仍会交换。搜索要考虑未来续写，greedy 只看当前最高概率可能错过更优完整路径。

2. Entropy 与单个 token 的 surprisal 是什么关系？

    **面试回答：** Surprisal 是 $-\log p(v)$，描述某个候选有多意外；entropy 是按整个分布求其期望。Locally typical sampling 比较的是两者的距离，不是给每个 token 再算一个分布熵。

3. 为什么 PPL=6 不代表模型准确率一定为 1/6？

    **面试回答：** PPL 是平均 NLL 的指数，等于参考 token 概率几何平均的倒数；accuracy 则取决于参考 token 是否位列最大项。只有特定均匀猜测例子才有六面骰子的直接对应，一般不能把两者互换。

4. Top-k、top-p 与 epsilon 最关键的区别是什么？

    **面试回答：** Top-k 固定候选数量，top-p 保留按概率排序后累计质量达到阈值的最小集合，epsilon 保留单项概率达到下限的候选。三者都需重新归一化，并处理边界项和空集问题。

5. 为什么 temperature=1 不足以保证 ancestral sampling？

    **面试回答：** 模型 generation config 或服务层可能仍应用 top-k、top-p、重复惩罚、语法 mask 等。只有每一步实际使用的分布等于原模型分布，并采用同一停止契约，才是在对应序列空间做 ancestral sampling。

6. 100 次偏置硬币的 mode 为什么不典型？

    **面试回答：** 正面概率 0.6 时全正面是概率最高的一条有序序列，但只有一条。约 60 正 40 反的排列有极多条，集合总概率大得多；mode 优化一个点，typicality 描述承载大量概率质量的集合。

7. 写出 locally typical sampling 的排序键。

    **面试回答：** 先算 $H=-\sum_vp(v)\log p(v)$，排序键为 $|{-\log p(v)}-H|$。按距离排序后累积原概率至质量阈值，再重新归一化采样。最高概率候选可能因过于可预测被排除。

8. EOS 给经典全局典型性带来什么困难？

    **面试回答：** 常见生成到 EOS 后停止，相当于吸收态，不能从中再访问所有生成状态。这破坏直接套用遍历随机过程长序列定理所需的条件，所以本讲转向逐 token 的局部近似，而非声称有无条件的全局典型集保证。

9. 高熵是否适合做 reasoning 错误检测器？

    **面试回答：** 可以作为特征，但不能单独判错。高熵可能来自多种同义表达，低熵也可能包含一个影响答案的二选一分歧，或者模型自信的错误。需要结合任务结构、验证信号和校准数据。

10. 采样方法比较怎样避免评测偏差？

    **面试回答：** 在 validation set 上选择参数、在 held-out set 上比较，固定模型和完整解码配置，报告随机种子、停止条件、样本数与计算预算。同时看质量、多样性和失败样本，避免只用生成 PPL 或为每个测试集单独挑最有利参数。

11. Temperature 与 top-p 为什么可能不交换？

    **面试回答：** Temperature 不改变候选排名，却改变各项概率及累计质量。先调温会改变达到 top-p 阈值所需的候选数；先截断则已经删掉部分支持集，后续调温不能恢复，因此两种顺序通常不同。

12. 在多候选推理系统中，采样多样性为什么有系统价值？

    **面试回答：** 多样性决定有限候选预算覆盖多少不同解法或答案，但要与可用质量一起看。过低导致重复计算，过高导致大量坏样本；应联合测量有效答案数、成功率、verifier 成本和端到端延迟来选择采样策略。

## 18. 关联内容

- [课程索引](CMU%2011-763.md)
- [上一讲：Probability Review and Code Examples](CMU%2011-763%20-%20Lecture%2002%20-%20Probability%20Review%20and%20Code%20Examples.md)
- [下一讲：Beam Search and Variants](CMU%2011-763%20-%20Lecture%2004%20-%20Beam%20Search%20and%20Variants.md)
- [LLM Inference](../../topics/inference/LLM%20Inference.md)
