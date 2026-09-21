---
type: course-note
status: developing
course: "[[CMU 11-763]]"
lecture: 7
lecture_date: 2025-09-16
area: inference
topics:
  - "[[LLM Inference]]"
aliases:
  - CMU 11-763 Lecture 07
  - Chain of Thought and Intermediate Steps
  - CMU CoT and Self-Consistency
video_url: https://www.youtube.com/watch?v=pKR3Vr6yg4U
---

# Lecture 07：Chain of Thought and Intermediate Steps

> [!abstract] 本讲一句话
> CoT 用中间 token 为一道题增加串行计算；self-consistency 用多条轨迹增加并行探索，并在答案层面聚合概率质量；两者是否有效，取决于任务、模型分布和额外计算是否真的提供有用信息。

## 来源与范围

- 课程：CMU 11-664/763 — Inference Algorithms for Language Modeling，Fall 2025。
- 讲师：Graham Neubig；日期：2025-09-16。
- [完整视频](https://www.youtube.com/watch?v=pKR3Vr6yg4U)，约 54:55；正文依据完整英文自动字幕。
- [官方 HTML 课件](https://www.phontron.com/class/lminference-fall2025/assets/slides/2025-09-16-chain-of-thought/index.html)。整理时旧路径直连回落到新版网站，已通过搜索索引读取课件完整文本，与字幕逐段交叉核对；未声称取得或逐页检查原 PDF 图像。
- [课程主页](https://www.phontron.com/class/lminference-fall2025/)。当前课程站点与旧课件索引存在迁移差异。
- 课堂涉及 CoT、zero-shot CoT、self-consistency、adaptive consistency、适用范围、faithfulness、complexity-based prompting 和多模态 CoT。

> [!note] 阅读约定
> 时间链接指向课堂实际讨论；公式中的记号澄清、归一化补全及工程推导是整理补充。课堂末尾转入学生报告，但这段公开视频没有收录报告正文，因此不会把课件列出的学生论文展开成“课堂内容”。

## 视频时间索引

| 时间 | 课堂内容 | 阅读位置 |
| --- | --- | --- |
| [00:29](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=29s) | CoT 与后续 reasoning models 的关系 | 第 1 节 |
| [02:05](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=125s) | H100 生成速度问题：知识、建模与计算 | 第 1 节 |
| [07:18](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=438s) | 按问题难度分配计算 | 第 2 节 |
| [08:24](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=504s) | Adaptive Computation Time | 第 2 节 |
| [10:25](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=625s) | CoT 定义与潜变量 | 第 3 节 |
| [12:26](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=746s) | Few-shot prompting 的形式 | 第 4 节 |
| [13:29](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=809s) | 预训练、SFT、RL 与 mid-training | 第 4 节 |
| [18:25](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=1105s) | 早期 CoT 实验任务 | 第 4 节 |
| [20:16](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=1216s) | Zero-shot CoT | 第 4 节 |
| [21:17](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=1277s) | Sampling 与 mode seeking | 第 5 节 |
| [24:15](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=1455s) | 联合最优轨迹不等于最优答案 | 第 5 节 |
| [25:25](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=1525s) | Self-consistency | 第 6 节 |
| [27:05](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=1625s) | Adaptive consistency | 第 7 节 |
| [29:43](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=1783s) | Dirichlet prior 的课堂推导 | 第 7 节 |
| [34:44](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=2084s) | Top-2 Beta approximation | 第 7 节 |
| [37:07](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=2227s) | CoT 并非对所有任务有益 | 第 8 节 |
| [40:44](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=2444s) | Explanation faithfulness | 第 9 节 |
| [43:44](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=2624s) | 长度、复杂度与候选过滤 | 第 10 节 |
| [46:20](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=2780s) | Q&A：CoT 概率记号是否符合边缘化恒等式 | 第 3、11 节 |
| [52:47](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=3167s) | 多模态 CoT 与下讲预告 | 第 12 节 |

## 1. 为什么需要中间步骤

课堂用一个 AI Infra 问题开场：一张 H100 最快能用多久为 Llama 3 8B 生成 100 个 token？

它同时要求三种能力：

1. 查到或记住硬件规格、数值精度、模型层数和各层大小。
2. 建立每个 token 所需 FLOPs、访存和串行依赖的计算模型。
3. 执行计算并检查量纲，必要时编写程序完成运算。

这与直接回答 `2+3=5` 的难度显然不同。把二者都压成“一次固定深度的 forward 之后直接给答案”，相当于给不同难度问题同一种计算路径。

### 1.1 实测性能与理论下界

学生提出“先实测一个 token 多久，再乘 100”。讲师指出实测含框架、调度等开销，不能当成理论最快值；上下文变化也使每一步开销不完全相同。

**整理补充：** 对带 KV cache 的常见自回归 decode，可以先用下式组织思路：

$$
T_{100}\gtrsim\sum_{t=1}^{100}
\max\left(\frac{F_t}{\text{compute throughput}},
\frac{B_t}{\text{memory bandwidth}}\right)
$$

$F_t$ 是第 $t$ 步运算量，$B_t$ 是该步所需数据搬运量。它仍是简化下界，不含所有同步和调度成本。

课堂用“attention 的二次复杂度”解释上下文变化。更精确地说，全序列 attention 是二次的；使用 KV cache 后，单个 decode step 对历史长度通常线性增长，整个生成区间仍需对各步求和。

本讲没有完成 H100 的数值性能测算。这个例子的用途是说明：解决问题需要先组织计算过程，随后才得到答案。

## 2. CoT 之前：Adaptive Computation Time

[08:24](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=504s) 回顾 Graves 的 Adaptive Computation Time（ACT）。

它让 RNN 在一个输入位置上执行若干内部更新，并学习何时停止。简化的停止信号是：

$$
h_n=\sigma(w^Ts_n+b)
$$

$s_n$ 是第 $n$ 次内部计算状态，$h_n$ 是 halting unit 的输出。ACT 使用累计停止质量和 remainder 构造可训练计算过程，并通过 ponder cost 抑制无限“思考”。

**记号注意：** 课件把 ponder cost 简写为概率之和；不要据此直接实现 ACT。原方法的成本涉及内部更新次数及 remainder，本讲只需要理解“任务损失 + 计算代价”的思想。

可以用教学目标表达为：

$$
\mathcal L=\mathcal L_{task}+\lambda\mathcal C_{ponder}
$$

其中 $\lambda$ 越大，模型越有动力尽早停止。

CoT 与 ACT 的联系，是都能让较难问题消耗更多计算；区别是 CoT 把额外计算放在生成中间 token 的自回归步骤中，不要求相同的内部 halting 架构。

### 2.1 增加 token 为什么等于增加计算

每新增一个 reasoning token，模型都再执行一轮条件分布计算；下一个 token 可以使用此前生成的结果。

这同时提供：

- 更多串行网络计算；
- 可被后续 attention 读取的中间状态；
- 分解任务、记录部分结果和修正错误的机会。

但“多做计算”不等于“做了有效计算”。重复问题、无关铺垫和循环反思也会消耗 token。

## 3. CoT 的概率定义，以及本讲最重要的记号澄清

设 $x$ 为问题，$z$ 为推理轨迹，$y$ 为最终答案。

在“先写推理，再写答案”的生成协议下：

$$
P_{cot}(z,y\mid x)
=P_\theta(z\mid c_{cot}(x))
P_\theta(y\mid c_{cot}(x),z)
$$

答案的边缘分布为：

$$
P_{cot}(y\mid x)=\sum_zP_{cot}(z,y\mid x)
$$

$c_{cot}(x)$ 包含 CoT 指令、示例和推理/答案格式。求和要覆盖可能的完整轨迹，包括不同长度与 EOS/阶段终止规则。

### 3.1 Direct 与 CoT 不应被误写成同一个分布

直接回答定义另一个协议：

$$
P_{direct}(y\mid x)=P_\theta(y\mid c_{direct}(x))
$$

一般没有：

$$
P_{direct}(y\mid x)=P_{cot}(y\mid x)
$$

原因不是概率边缘化失效，而是二者的 prompt、生成位置与条件上下文不同。

[49:37](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=2977s) 学生指出课堂板书看似违反全概率公式；讲师随后澄清，答案“跟在 $z$ 后面”与“不跟在 $z$ 后面”具有不同条件概率。以上两套记号把这个区别显式写出来。

### 3.2 CoT 的假设到底是什么

希望 $P_{cot}$ 比 $P_{direct}$ 更适合目标任务，例如使正确答案的概率更高，或使答案分布的 mode 更准确。

这是关于模型和任务的经验假设，不是由概率恒等式直接推出的定理。

生成一条 CoT 已经能改变答案的条件分布；采样很多 CoT 则进一步让我们估计并利用答案边缘分布。二者的贡献应分开分析。

## 4. 如何让模型产生 CoT

### 4.1 Few-shot CoT

普通 few-shot 示例只展示问题与答案；CoT 示例增加问题如何一步步得到答案。

课件的网球题展示了这种结构：已有 5 个球，再买 2 罐，每罐 3 个，因此新增 $2\times3=6$ 个，最后共有 11 个。

关键是把“2 罐”转换成“6 个球”，保留中间量及单位，而非只给模型一个答案字符串。

这种 prompting 不更新模型权重。它通过上下文诱导模型采用已经能够表达的解题模式。

### 4.2 Zero-shot CoT

[20:16](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=1216s) 介绍 Kojima 等人的方法，用简短指令 “Let's think step by step” 诱导逐步输出。

原工作区分生成推理和提取最终答案的阶段。复现时应明确最终答案怎样抽取，不要把“有 reasoning 文本”自动当成“可稳定评测的答案”。

讲师提到这一短语在原实验的多种候选提示中有效。它是特定实验的结果，不是跨所有模型的最优 prompt 定律。

### 4.3 能力来自哪里

课堂给出三条来源：

| 来源 | 机制 | 需要更新权重吗 |
| --- | --- | --- |
| 预训练中已有相关模式 | 代码、证明、教科书和习题解析提供推导结构 | 预训练阶段需要，当前 prompting 不需要 |
| SFT | 用明确的 reasoning traces 监督模型 | 需要 |
| RL | 对完成任务的轨迹给奖励，增强有效行为 | 需要 |

讲师特别强调“足够强”比“参数足够大”更合适。训练数据、训练长度和后训练都会改变相同参数量模型的能力。

### 4.4 Base model 与 mid-training

[15:50](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=950s) 提醒：标成 base 的模型也可能在预训练末期大量使用精选教育、数学或合成数据。

因此不能只根据 checkpoint 名字声称某种能力完全“无监督涌现”。比较模型时，需要尽量检查其数据与训练阶段，无法确认的部分应保留不确定性。

课堂对不同模型家族的数据策略属于当时的研究讨论，不足以证明某一未公开训练配方。

### 4.5 早期实验告诉了我们什么

讲师讨论 GSM8K、StrategyQA 和 last-letter concatenation：它们分别强调算术、多跳常识以及符号操作。

观察到的明显收益集中在需要中间状态的题目。课件列出的百分比来自历史实验设置，不能拼成今天所有模型的性能表，也不等于相同计算预算下的统一比较。

## 5. Sampling 容易，寻找答案 mode 更难

### 5.1 祖先采样直接给出边缘分布的样本

先采样 $z\sim P_{cot}(z\mid x)$，再采样 $y\sim P_{cot}(y\mid x,z)$，丢弃 $z$ 后的 $y$ 就服从答案边缘分布。

若目标是原模型协议下的分布，应保持 $T=1$，且不引入 top-k/top-p 等改变分布的截断。

如果实际使用其他温度或过滤规则，也可以做 self-consistency，但此时估计的是对应解码策略诱导的分布。

### 5.2 联合最优与边缘最优

联合搜索目标：

$$
(z^*,y^*)=\arg\max_{z,y}P_{cot}(z,y\mid x)
$$

答案层面的目标：

$$
\hat y=\arg\max_y\sum_zP_{cot}(z,y\mid x)
$$

两个操作不可交换：$\max$ 取单条最大路径，$\sum$ 汇总所有落到同一答案的路径。

课堂“三英尺是多少英寸”的示意分布为：

| 轨迹概述 | 答案 | 联合概率 |
| --- | --- | --- |
| 用一英尺 12 英寸计算 | 36 | 0.3 |
| 错误地转向厘米问题 | 300 | 0.4 |
| 另一条简短正确路径 | 36 | 0.3 |

最大单条路径给 300；聚合后 $P(36\mid x)=0.6$、$P(300\mid x)=0.4$，答案 mode 是 36。

该表是课堂人为构造的例子，不是实际模型测量结果。

### 5.3 为什么不能枚举全部轨迹

词表大小为 $V$、轨迹最长 100 token，序列空间规模约为 $V^{100}$。即使大量字符串没有意义，显式遍历仍不可行。

Beam search 可以近似寻找若干高概率轨迹，却不会自动把所有等价答案的概率质量充分加总。

## 6. Self-consistency：答案投票近似边缘化

生成 $N$ 个样本 $(z_i,y_i)$，定义答案等价类规范化函数 $g$：

$$
\hat P_N(a\mid x)=\frac1N\sum_{i=1}^N\mathbf1[g(y_i)=a]
$$

最后选：

$$
\hat a=\arg\max_a\hat P_N(a\mid x)
$$

这种方法重视跨轨迹一致的答案，不要求所有推导都长得相似。

### 6.1 规范化是算法的一部分

`36`、`36 inches`、`3 feet = 36 inches` 可以映射到同一答案；但错误的单位转换、数值近似和多答案问题可能被错误合并。

实现时至少区分：

- 可解析且有效的答案；
- 解析失败；
- 生成被截断；
- 题目允许的等价表达；
- 需要保留的单位、符号或选项语义。

解析失败不能被默认为正确答案，也不应把失败字符串当成一个有意义的“大众答案”。

### 6.2 多数票不等于真值

如果模型 70% 的轨迹都犯同一种系统错误，更多样本会让投票更稳定地选错。

Self-consistency 降低的是对模型答案分布 mode 的采样不确定性；它不会从概率论上保证该 mode 等于现实正确答案。

### 6.3 自由文本为何更难

对整数或多项选择题，答案等价类相对清楚。对文章、摘要或解释，同一语义可能有大量不同表达，逐字相同的票数几乎没有意义。

课堂把这种情况留给 minimum Bayes risk 等语义效用方法。也可以用语义归类，但归类器本身会引入成本与判断误差。

### 6.4 最小实现

```python
from collections import Counter

def self_consistency(question, n, generate, parse):
    votes = Counter()
    failures = 0
    for _ in range(n):
        trace = generate(question)
        answer = parse(trace)
        if answer is None:
            failures += 1
        else:
            votes[answer] += 1
    if not votes:
        return None, {"failures": failures}
    ranked = votes.most_common()
    tied = len(ranked) > 1 and ranked[0][1] == ranked[1][1]
    return ranked[0][0], {"votes": votes, "tied": tied,
                           "failures": failures}
```

这是教学伪实现，不是课程官方代码。生产实现还应明确 tie-break、随机种子、最大 token、并发和失败重试规则。

## 7. Adaptive consistency：什么时候停止采样

固定每题采 100 条会浪费大量容易题的预算。[27:05](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=1625s) 的方案是分批采样，当当前领先答案足够可能保持领先时停止。

### 7.1 为什么一次出现不能意味着概率 1

课堂例子：只观察到一次 A，没有 B、C。最大似然估计给出 $(1,0,0)$，但我们显然不能确认以后永远出现 A。

用 Dirichlet prior 表示对答案分布的先验：

$$
(p_1,\ldots,p_K)\sim\operatorname{Dirichlet}(\alpha q_1,\ldots,\alpha q_K)
$$

观察计数 $n_i$ 后，后验仍是 Dirichlet：

$$
p\mid n\sim\operatorname{Dirichlet}(n_1+\alpha q_1,\ldots,n_K+\alpha q_K)
$$

后验均值和下一次观察的 posterior predictive probability 为：

$$
\mathbb E[p_i\mid n]=\frac{n_i+\alpha q_i}{N+\alpha}
$$

课堂取 $q_i=1/3$、$\alpha=3$、$n=(1,0,0)$，得到 $(1/2,1/4,1/4)$。

$\alpha$ 是先验总强度；它越大，相同数量观察对先验的改变越小。这里的 posterior mean 不要误称为 MAP estimate。

### 7.2 用前两名近似胜者稳定性

设 $v_1,v_2$ 为当前第一、第二名的计数。对两者条件概率使用均匀 Beta prior，可得到：

$$
p_2\mid v_1,v_2\sim\operatorname{Beta}(v_2+1,v_1+1)
$$

当前第一名真实概率大于第二名的后验概率：

$$
C=\Pr(p_2<1/2\mid v_1,v_2)
=\frac{\int_0^{1/2}p^{v_2}(1-p)^{v_1}\,dp}
{B(v_2+1,v_1+1)}
$$

当 $C>C_{thresh}$ 时停止，课堂阈值示例是 0.95。

> [!warning] 课件公式修正
> 官方 HTML 文本显示的积分缺少 Beta 归一化因子。分子本身不是概率，不能直接与 0.95 比较；正确实现应使用归一化 Beta CDF。这里补全归一化，也明确它是 top-2 approximation，而非对所有潜在答案的无条件保证。

### 7.3 这个 95% 到底保证什么

它表示在所选统计模型及近似下，对“继续采样后哪一类更常见”的置信程度。

它不等于“答案有 95% 概率正确”，也不自动涵盖未观察类别、相关样本、变化的解码分布或错误答案抽取。

若不同样本共享强烈偏差，增加样本和调高阈值只能得到更稳定的共识。

### 7.4 调度和结果边界

课堂说明在每批生成后更新判断，没有给出固定批大小。实现可设置初始批、后续批和总预算上限，并在验证集选择阈值。

原论文在其 17 个推理/代码数据集、3 个模型的设置中报告最高约 7.9 倍样本预算降低、平均精度损失小于 0.1%。这些数字是特定实验结果，不能当成部署预期。见 [Adaptive-Consistency 原论文](https://arxiv.org/abs/2305.11860)。

## 8. 哪些任务值得花 CoT 预算

[37:07](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=2227s) 讨论 *To CoT or not to CoT?*：论文既分析已有 CoT 文献，也在 20 个数据集与 14 个模型上做实验。

课堂强调，数学与形式逻辑通常受益更明显；常识、知识检索或一些语言任务的收益小得多。

MMLU 的“是否出现等号”分析提供一个有意思的诊断：总体 benchmark 上的提升，可能由其中需要符号运算的题目驱动。

不能把这个分析变成“有等号必须 CoT，没有等号绝不 CoT”的路由规则。等号只是论文用于划分子集的粗代理。

**工程补充：** 比较 direct、单条 CoT 和多样本 CoT 时应报告任务分组，并保持可比较的计算预算，否则难以区分算法改进与多用了算力。

## 9. Faithfulness：解释可信度与答案正确率不同

课堂讨论 Turpin 等人的偏置实验：把 few-shot 示例正确选项全部重排到 A，模型更倾向选择 A，但生成的解释通常不承认选项位置的影响。

Wayne Rooney 的例子中，无偏上下文能正确联系到足球的 18-yard box；偏置上下文则编出另一个看似合理的解释支持错误选项。

这揭示三个不同性质：

| 性质 | 关注的问题 |
| --- | --- |
| 正确性 | 最终答案是否正确 |
| 推导有效性 | 写出的每步推导是否成立 |
| Faithfulness | 写出的解释是否忠实反映影响预测的因素 |

答案正确也可能配有不忠实解释；语言流畅也不证明推导成立。

该研究在 GPT-3.5、Claude 1.0 等当时模型上进行。课堂将它作为方法论警示，没有证明后来所有模型都以相同程度失真。见 [原论文](https://arxiv.org/abs/2305.04388)。

在需要审计的系统中，优先记录可核验的外部证据、运算、工具结果和引用，不应仅凭模型生成的自我说明判断正确性。

## 10. Complexity-based prompting 与 decoding

[43:44](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=2624s) 介绍 Fu 等人的研究。原工作涉及选择更复杂的示例，以及根据采样轨迹复杂度进行投票；课堂着重讲后者。

基本流程：

1. 对同一道题采样多条 reasoning traces。
2. 以推理步骤数量等代理度量复杂度。
3. 舍弃较短的部分轨迹。
4. 对保留轨迹的最终答案做 self-consistency。

它利用“较多推理步骤与较高准确率相关”的观察，不能推出随意增加废话会提高准确率。参见 [Complexity-Based Prompting](https://arxiv.org/abs/2210.00720)。

### 10.1 筛选会改变答案分布

若用固定接受条件 $A(z)=\mathbf1[\operatorname{steps}(z)\ge k]$，接受样本的答案分布为：

$$
P_A(y\mid x)=
\frac{\sum_z A(z)P_{cot}(z,y\mid x)}
{\sum_{z,y'}A(z)P_{cot}(z,y'\mid x)}
$$

这是整理补充，用于解释课堂所说的“类似 rejection sampling”。如果实际按批内相对排名保留最长若干条，接受规则还依赖整批样本，不能直接视为同一个固定阈值分布。

### 10.2 长度本质上是粗糙的打分器

讲师在 [51:26](https://www.youtube.com/watch?v=pKR3Vr6yg4U&t=3086s) 将该方法称为 heuristic：可以经验调参，也可以使用更直接预测推理质量的 reward model。

长度适合作为便宜代理，但它容易选择啰嗦、重复或过度分析的轨迹。收益是否超过筛选代价必须实验检验。

## 11. 课堂问答中的关键澄清

### Q1：轨迹空间更大，为什么一条 CoT 也能有效？

生成 $z$ 会改变后续答案的条件上下文，并提供额外计算。一条样本也能从这个新协议产生有用答案，但不足以精确估计答案分布。

因此“更多计算改善模型的有效预测分布”和“多样本估计该分布”是不同问题。课堂在 46–51 分钟围绕这点展开。

### Q2：如何选择样本数和轨迹长度？

固定设置可以在验证集上调；adaptive consistency 根据票数分歧决定是否继续。长度过滤本身则是经验策略。

实际选择需同时考虑预算、边际准确率收益和长输出尾延迟，不能只追求更大的 $N$ 或更长的 $z$。

### Q3：新模型是否已经解决解释不忠实？

讲师回答是有所改善，但没有解决；多步 agent 和分布外任务仍可能生成不可靠自我说明。

这属于课堂经验判断，最可靠的落实方式是对当前使用的模型和任务做具体验证。

## 12. 多模态 CoT

课堂最后简述：把图像和语言输入一起用于生成 rationale，再基于 rationale 推断答案；也有工作把定位框等视觉信息融入中间表示。

关键不是简单“把图片变成文字”，而是中间步骤能否保留与任务有关的视觉证据。

视频只作概念介绍，没有展开实现或逐项讲解 benchmark。多模态识别错误也可能被后续语言推理放大，需要区分 perception error 与 reasoning error。

## 13. 面向 AI Infra 的资源账本

以下是把课堂算法转成系统设计时的补充分析。

### 13.1 串行深度与并行宽度

设每条轨迹长 $L_z$、答案长 $L_y$、共采样 $N$ 条：

$$
\text{generated tokens}\approx N(L_z+L_y)
$$

CoT 增加单条请求的串行深度；self-consistency 增加可并行的分支数。两者都增加总计算，但 latency 表现不同。

假设有足够并行资源，采样 $N$ 条不一定带来 $N$ 倍墙钟时间；然而显存、总 GPU 时间和吞吐占用仍会增加。

### 13.2 共享前缀与独立后缀

多条轨迹共享问题 prompt，可复用 prefix prefill/KV；分叉之后各条轨迹拥有自己的后缀 KV。

因此 prefix sharing 节省共同前缀成本，却不会消除长 CoT 的后缀显存和逐 token decode 成本。

### 13.3 Adaptive stopping 与 batch 冲突

批越大，GPU 利用率可能越好，但当足够票数已经出现时，剩余在途样本也许不再有价值。

批越小，停止更及时，但调度开销和硬件利用率可能较差。

应联合选择 batch size、并发上限、取消语义和最大预算，而不是把统计停止规则与 serving scheduler 分开优化。

### 13.4 最小实验表

| 方法 | 需要记录的质量指标 | 需要记录的成本指标 |
| --- | --- | --- |
| Direct | 正确率、解析失败率 | 输出 token、延迟 |
| 单条 CoT | 正确率、截断率 | reasoning/answer token 分开计数 |
| 固定 $N$ self-consistency | 最终正确率、票数分布 | 总 token、并发、端到端延迟 |
| Adaptive consistency | 正确率、与固定预算结果分歧 | 每题样本数分布、节省预算 |
| Complexity filtering | 过滤前后准确率 | 被弃轨迹已消耗的 token |

## 14. 自测问题与面试回答

1. CoT 怎样实现 adaptive computation？

    **面试回答：** 每个中间 token 都触发新的自回归 forward，并成为后续步骤可读取的上下文，因此模型可以通过轨迹长度为难题投入更多串行计算。但有效推导、重复文本都要付费，长度本身不是质量保证。

2. 为什么 direct answer 的分布不必等于 CoT 边缘分布？

    **面试回答：** 两者的提示和输出位置不同。应分别定义 $P_{direct}(y\mid x)=P_\theta(y\mid c_d(x))$ 与 $P_{cot}(y\mid x)=\sum_zP_\theta(z,y\mid c_c(x))$。边缘化恒等式在同一协议内部成立，不能跨不同条件上下文强行等同。

3. 联合 MAP 为什么可能返回错误的答案 mode？

    **面试回答：** 联合 MAP 只选概率最高的一条 $(z,y)$；答案 mode 把所有通向同一答案的轨迹概率相加。若正确答案有两条各 0.3 的路径，错误答案有一条 0.4 的路径，联合 MAP 选错，答案边缘 mode 仍选正确答案。

4. Self-consistency 是否必须使用 temperature=1？

    **面试回答：** 不必须。任何固定采样策略都能生成候选做投票，但投票估计的是该策略诱导的答案分布。只有不带改变分布的截断等操作、且 T=1 的祖先采样，才直接对应原模型的指定 CoT 协议分布。

5. 为什么更多 self-consistency 样本不保证答案更正确？

    **面试回答：** 更多样本让经验频率更接近模型分布，减少估计方差；如果模型把最高概率质量放在错误答案上，多数票会更稳定地选错。它解决采样不确定性，不消除知识或推理的系统偏差。

6. Dirichlet prior 的 $\alpha$ 是什么？

    **面试回答：** 在 $Dir(\alpha q)$ 参数化下，$q$ 是先验均值，$\alpha$ 是总先验强度。观察计数 $n$ 后，后验参数为 $n+\alpha q$，后验预测均值为 $(n_i+\alpha q_i)/(N+\alpha)$；$\alpha$ 越大，有限数据越难改变先验。

7. Adaptive consistency 的 0.95 是否表示 95% 正确率？

    **面试回答：** 不是。它是在选定后验模型与 top-2 近似下，当前领先类别的真实采样概率超过竞争者的概率。它不判断答案真伪，也不完整覆盖未观察类别、样本相关性和解析错误。

8. 为什么 Beta 积分必须归一化？

    **面试回答：** $p^{v_2}(1-p)^{v_1}$ 只是密度核，其在 [0,1] 的积分不是 1。除以 $B(v_2+1,v_1+1)$ 后才得到 Beta 密度，积分到 0.5 才能作为可与 0.95 比较的概率。

9. 答案正确、解释有效、解释忠实有何不同？

    **面试回答：** 正确性看结论，推导有效性看写出的步骤是否成立，faithfulness 看解释是否反映实际影响预测的因素。模型可能给出正确结论却遗漏偏置因素，也可能用流畅的推导包装错误答案，三者必须分别验证。

10. Complexity filtering 改变了什么？

    **面试回答：** 它按轨迹长度或步骤数过滤样本，使最终投票来自经过选择的条件分布，不再是原始 CoT 分布的直接估计。其有效性依赖复杂度代理与正确性的相关性，不能把更多字数当成充分条件。

11. 如何公平比较长 CoT 与多条短 CoT？

    **面试回答：** 至少同时比较固定总 token 或 GPU 成本下的准确率，以及固定延迟预算下的准确率。长 CoT 有串行依赖，多条短 CoT 可以并行但占更多并发显存；还要统计答案抽取失败、截断和共享前缀收益。

12. 只调用同一个模型多次为什么也可能得到收益？

    **面试回答：** 不同采样轨迹会探索不同中间步骤，而同一个答案可能通过多条独立路径出现。聚合可以减少单条轨迹偶然失误，但同模型的系统偏差仍存在；需要通过任务评测确定多样性是否真的转化为有效探索。

## 15. 阅读地图与关联

| 阅读 | 本讲对应问题 |
| --- | --- |
| [Wei et al., 2022](https://arxiv.org/abs/2201.11903) | Few-shot CoT 怎样诱导推导 |
| [Kojima et al., 2022](https://arxiv.org/abs/2205.11916) | 没有示例时能否诱导 CoT |
| [Wang et al., 2022](https://arxiv.org/abs/2203.11171) | 多轨迹答案投票 |
| [Graves, 2016](https://arxiv.org/abs/1603.08983) | 显式自适应计算与停止 |
| [Aggarwal et al., 2023](https://arxiv.org/abs/2305.11860) | 什么时候停止继续采样 |
| [Sprague et al., 2024](https://arxiv.org/abs/2409.12183) | CoT 的任务适用范围 |
| [Turpin et al., 2023](https://arxiv.org/abs/2305.04388) | 解释是否遗漏实际偏置因素 |
| [Fu et al., 2022](https://arxiv.org/abs/2210.00720) | 复杂示例与复杂轨迹选择 |
| [Zhang et al., 2023](https://arxiv.org/abs/2302.00923) | 多模态输入如何进入中间推理 |

- 课程索引：[CMU 11-763](CMU%2011-763.md)。
- 概率前置：[Lecture 02](CMU%2011-763%20-%20Lecture%2002%20-%20Probability%20Review%20and%20Code%20Examples.md)。
- 下一讲：[Lecture 08](CMU%2011-763%20-%20Lecture%2008%20-%20Self-Refine%20and%20Self-Correction%20Methods.md)。
- 主题入口：[LLM Inference](../../topics/inference/LLM%20Inference.md)。
