---
type: course-note
status: developing
course: "[[CMU 11-763]]"
lecture: 14
lecture_date: 2025-10-09
area: inference
source_mode: slides
topics:
  - "[[LLM Inference]]"
aliases:
  - CMU 11-763 Lecture 14
  - Minimum Bayes Risk and Multi-Sample Strategies
  - Multi-sample strategies pt 2 MBR and more
slides_url: https://docs.google.com/presentation/d/1WiLhDob25YOb1YU1AVwVPWYvKRK9HJR4v57_pLizndk/edit
---

# Lecture 14：Minimum Bayes Risk and Multi-Sample Strategies

> [!abstract] 本讲一句话
> MBR 不问“哪一条完整输出概率最高”，而问“在可能的参考答案分布下，返回哪一个候选的期望损失最低”；Self-Consistency 是特殊的 MBR，而高效 MBR 的关键在于分开管理候选覆盖、证据质量和评分成本。

## 来源与范围

- 课程：CMU 11-664/763 — Inference Algorithms for Language Modeling，Fall 2025。
- 日期：2025-10-09。
- 官方课件：[Multi-sample strategies pt 2: MBR and more，43 页](https://docs.google.com/presentation/d/1WiLhDob25YOb1YU1AVwVPWYvKRK9HJR4v57_pLizndk/edit)。
- 材料核对：完整读取 43 页文字，并渲染检查原始 PDF 全部页面；公式、Self-Consistency 示意图和实验图均以页面图像补足。
- 本讲没有取得可用的公开视频/字幕，因此以下以课件页码定位，不提供虚构时间戳，也不将自拟问题写成课堂问答。
- “补充推导”“实现补充”“工程延伸”是为理解和落地整理的内容，不声称课件逐字讲过。

### 课件导航

| 页码 | 内容 | 复习时要回答的问题 |
|---|---|---|
| 1–6 | 高概率输出与语义概率质量 | 一个概率最高的字符串，为什么不代表最可信的意思？ |
| 7–11 | 伪参考、风险与 MBR 决策 | 平均的是哪些分数，概率体现在哪里？ |
| 12–16 | 分布、证据集、候选集、指标 | 哪部分要求忠实采样，哪部分可以任意搜索？ |
| 17–22 | Coarse-to-Fine MBR | 怎样减少昂贵 metric 的调用数？ |
| 23–30 | 置信剪枝与 bootstrap | 如何及早排除明显不可能胜出的候选？ |
| 31–37 | Self-Consistency 的 MBR 解释 | 为什么投票只比较最终答案而忽略推理路径？ |
| 38–42 | Universal Self-Consistency | 开放式回答无法 exact match 时怎样聚合？ |
| 43 | 总结 | 多样本推理如何统一为风险最小化？ |

## 1. 字符串的众数，不一定是语义上的共识

### 1.1 课件的六个输出例子（第 3–6 页）

同一个输入可能对应如下完整输出分布：

| 概率 | 输出含义 |
|---:|---|
| 0.300 | 猫坐下了 |
| 0.250 | 猫跑走了 |
| 0.200 | 猫冲走了 |
| 0.149 | 猫离开了这里 |
| 0.100 | 猫很小 |
| 0.001 | 猫长出了翅膀 |

按单条字符串的概率选择，赢家是“猫坐下了”。

但表达“猫离开”的三个输出合计有：

$$
0.250+0.200+0.149=0.599.
$$

课件将它近似描述为 60% 的概率质量。
这种质量被不同措辞分散了，而“坐下”的质量集中在一条字符串上。

所以要分清两件事：

- 一条输出比另一个明显离谱的输出概率高，通常有价值。
- 在几个都很流畅的高概率输出之间，模型概率未必能可靠排序任务质量。

第 4 页的 Model Score–ROUGE-2 散点图正是在强调第二点，不是在声称模型概率完全无用。

### 1.2 从最大概率，转向做决策的损失

MAP 选择：

$$
\hat y_{\mathrm{MAP}}=\arg\max_y p_\theta(y\mid x).
$$

它在“只有整个字符串完全相同才算对”的 0–1 损失下有自然的决策解释。
但摘要、翻译和开放式问答通常允许多种等价措辞，逐字不同不等于完全错误。

因此，真正要明确的是：

> 如果真实可接受参考是某个回答，我返回当前候选，会造成多大损失？

这就是从生成概率问题转向决策问题的关键。

## 2. MBR 的完整数学形式

### 2.1 风险与收益（第 7–11 页；补充推导）

记：

- $x$：输入问题或上下文。
- $r$：可能的参考输出。
- $h$：准备返回给用户的候选输出。
- $p^*(r\mid x)$：真实参考分布，实际通常不可得。
- $L(h,r)$：把 $h$ 与 $r$ 比较得到的损失，越小越好。
- $G(h,r)$：对应的收益/相似度，越大越好。

理想的 Bayes risk 是：

$$
R(h\mid x)=\mathbb E_{r\sim p^*(\cdot\mid x)}[L(h,r)].
$$

理想决策为：

$$
\hat h=\arg\min_{h\in\mathcal H}R(h\mid x).
$$

当 $L(h,r)=C-G(h,r)$，且 $C$ 与候选无关时，等价于最大化期望收益。
对于一般损失，也可以直接令 $G=-L$。

因为拿不到 $p^*$，实际 MBR 使用模型分布或选定的代理分布 $q$：

$$
\hat h_q
=\arg\max_{h\in\mathcal H}
\mathbb E_{r\sim q(\cdot\mid x)}[G(h,r)].
$$

因此 MBR 的“Bayes”不意味着参考是真实答案，也不自动保证事实正确。
它是相对于指定分布和指定损失的最优决策。

### 2.2 用 Monte Carlo 估计（第 8–10 页）

取证据样本列表：

$$
\mathcal E=(r_1,\ldots,r_M),\qquad r_j\overset{\mathrm{iid}}\sim q(\cdot\mid x).
$$

计算每个候选的平均收益：

$$
\hat U(h)=\frac1M\sum_{j=1}^{M}G(h,r_j),
\qquad
\hat h=\arg\max_{h\in\mathcal H}\hat U(h).
$$

课件第 9 页省去共同的 $1/M$，因为每个候选使用同一个证据列表，缩放不改变 argmax。
课件把伪参考写在 $G$ 的第一个参数中；本文统一采用 $G(\text{candidate},\text{reference})$，实际调用必须与 metric 接口一致。
不要默认所有 metric 都对称。

概率没有消失：频繁出现的参考，在 Monte Carlo 求和里会被重复计入。
这正是样本频率对概率的近似。

> [!warning] 不要额外乘一次候选概率
> 第 9 页将“高概率”标在候选集合上，将“低风险”标在求和项上。这是选候选与评估风险的直觉，不是 $p_\theta(h\mid x)\hat U(h)$ 这一新目标。标准 MBR 不额外乘候选概率；若这么做，就改变了决策规则。

### 2.3 为什么重复样本不能随手去掉

若证据为六个 A、三个 B、一个 C，exact-match metric 下：

$$
\hat U(A)=0.6,\quad \hat U(B)=0.3,\quad \hat U(C)=0.1.
$$

把证据转换为集合 $\{A,B,C\}$ 后，三者会变成平局。
原来对分布的估计，被错误替换成了“每个不同字符串一票”。

正确的节省计算方式是把重复证据压缩为计数：

$$
\hat U(h)
=\sum_{u\in\mathrm{unique}(\mathcal E)}\frac{c_u}{M}G(h,u).
$$

每个独特字符串只调用一次 metric，但恢复它出现的权重。
第 20 页提到的缓存，是缓存计算，不是删除概率质量。

### 2.4 为什么不需要再按模型概率加权

若 $r_j$ 已从 $q$ 采样，平均 $G$ 就是在估计 $\mathbb E_q[G]$。
如果再乘 $q(r_j\mid x)$，目标将近似变成：

$$
\sum_r q(r\mid x)^2G(h,r),
$$

这会过度强调高概率参考，不再是原来的期望。
不要把“枚举唯一输出后按概率求和”与“按概率随机采样后求平均”混为一谈。

## 3. 四个独立设计选择

### 3.1 目标分布：到底向谁求共识（第 12 页）

课件推荐以模型学到的分布作为默认目标，使用 ancestral sampling 获取证据。
每一步从原始下一 token 分布中采样，产生对应的完整序列分布。

改变 temperature、截断低概率 token、加重复惩罚，都可能改变证据分布。
这些调整不是绝对不能用，但应明确自己现在最小化的是哪个分布下的风险。

- 为了单条回答更顺畅而设计的解码器，不一定给出忠实的概率证据。
- 为了强行得到独特样本而拒绝重复，会破坏样本频次的意义。
- 经过 alignment 改善的模型分布，可以成为新的目标分布。

第 16 页建议：若证据和候选必须共用一组样本，可以选接近原分布、同时维持一定质量的策略，例如 epsilon sampling。
这是一种实践折中，不是严格无偏性的声明。

### 3.2 重要性采样能不能修复分布偏移（补充推导）

若目标是 $q$，实际采样来自 $s$，且 $s(r)>0$ 覆盖所有 $q(r)>0$ 的地方：

$$
\mathbb E_q[G(h,r)]
=\mathbb E_s\left[\frac{q(r)}{s(r)}G(h,r)\right].
$$

但这不意味着任意采样策略都能便宜地修复：

- 完整序列的概率比可能非常极端，导致高方差。
- top-k/top-p 截断可能令某些 $s(r)=0$，丢失的目标支持不能靠有限权重补回来。
- 自归一化权重在有限样本下通常有偏。

对学习本讲而言，最稳妥的起点仍是：先用忠实 evidence，再独立优化 hypotheses。

### 3.3 Evidence set 与 hypothesis set（第 13–16 页）

| 维度 | Evidence / 伪参考列表 $\mathcal E$ | Hypotheses / 候选集合 $\mathcal H$ |
|---|---|---|
| 用途 | 估计目标分布下的收益 | 决定最终允许返回哪些答案 |
| 来源 | 应来自要积分的目标分布 | 可以混合多个模型、prompt、采样和搜索策略 |
| 重复 | 频次承载概率质量 | 相同候选通常可去重 |
| 扩大后的主要收益 | 降低期望估计噪声 | 提高找到优质输出的机会 |
| 主要成本 | 生成与增加评分列 | 生成与增加评分行 |

课件给出的常见设计是 $|\mathcal H|<|\mathcal E|$，有时 $\mathcal H\subseteq\mathcal E$。
这些关系是实践选择，不是 MBR 定义所强制。

一个可操作配置是：

1. 从目标模型普通采样 128 条作为 evidence。
2. 从普通采样、较低温度、beam 等策略中构造 16 个不同候选。
3. 用同一个 metric 对 16 个候选分别和 128 条 evidence 比较。
4. 返回平均收益最高的候选。

此处数字是示例，不是课件报告的推荐最优参数。

### 3.4 Metric 决定“共识”是什么（第 15–16 页）

根据任务，可以比较：

- 字面片段重叠。
- 语义表示相似度。
- 最终数值答案是否一致。
- 程序在相同输入上的执行结果是否一致。
- 参考条件下的学习型质量分数。

第 15 页展示 CNN/DailyMail 上不同 MBR gain 的结果。
例如课件表中，Greedy 的 R1 是 44.11，MBR ROUGE-1 是 47.35；BERTScore 一列从 88.02 到 MBR BERTScore 的 88.74。
不同 gain 在多个指标上都有收益，说明不一定只对选用的那一个指标有效。

但不能把这个表解读为任意任务上都稳健，或“metric 越高一定越真实”。
优化哪个代理指标，就可能放大哪个代理的盲点。
原始讨论见 [Bertsch、Xie 等：It’s MBR All the Way Down](https://aclanthology.org/2023.bigpicture-1.9/)。

### 3.5 “特征空间找众数”要怎样准确理解

第 11 页把 MBR 描述为在 gain 定义的特征空间里寻找共识，而 beam 更接近在序列概率空间里寻找众数。

如果 $G(h,r)=\mathbf1[f(h)=f(r)]$，这严格对应寻找特征 $f(r)$ 的众数。
如果 $G$ 是连续相似度，MBR 更一般地是选择平均相似度最高的代表，类似 medoid。
此时它不一定对应某个明确概率密度的数学众数。

这一点解释了为什么改变 gain，会改变算法最终认为“代表大家”的答案。

## 4. 高效 MBR：先拆成本，再做优化

### 4.1 朴素成本（第 17–19、30 页）

设候选数为 $N$，证据数为 $M$，一次 metric 调用成本为 $C_G$。
总体可拆为：

$$
C_{\mathrm{total}}
\approx C_{\mathrm{generation}}+NM C_G+C_{\mathrm{aggregate}}.
$$

当 $N=M$ 时，评分调用数是二次的。
最后对每行求和再 argmax，往往比生成和神经 metric 便宜得多。

第 30 页强调三种不同优化对象：

1. 采样：利用通用的高效、并行生成技术。
2. pairwise 评分：预筛或迭代剪枝候选。
3. 聚合选最大值：一般不是主要瓶颈。

不能看到 $O(N^2)$ 就只优化最后的矩阵求和，昂贵的通常是构造矩阵。

### 4.2 Coarse-to-Fine MBR（第 18–22 页）

方法分成两级：

1. 对较大的候选集合，使用便宜的 gain 估计收益。
2. 保留一小批高分候选。
3. 使用更好的、可能更昂贵的 gain 重新运行 MBR。

把 $N$ 个候选筛成 $K$ 个候选后，成本近似：

$$
C_{\mathrm{C2F}}
\approx NM_cC_{\mathrm{cheap}}+KM_fC_{\mathrm{expensive}}.
$$

这里允许粗筛与精筛使用不同证据数量 $M_c,M_f$。
课件并没有要求必须这样做；它是便于分析的推广。

举例：$N=64$、$M_f=128$ 时，直接昂贵评分需 8192 对。
粗筛后只留下 $K=8$，昂贵评分降为 1024 对，再加粗筛的开销。
若便宜 metric 足够轻，整体可能明显划算。

这也允许扩展候选的产生方式，而不必让所有候选都承担昂贵精评分。
方法来源：[Eikema 与 Aziz，Sampling-Based Approximations to MBR](https://aclanthology.org/2022.emnlp-main.754/)。

### 4.3 它的失败模式：粗筛不可逆地扔掉赢家

cheap metric 与真正目标弱相关时，细节正确但措辞不同的好候选可能被误删。
精筛只能从留下来的候选里选，不能挽回已删除的最优解。

因此评估粗筛至少要看：

- 最终质量下降多少。
- full-MBR 赢家在 top-$K$ 中的保留率。
- 昂贵 metric 调用减少多少。
- 端到端时间是否改善，而不仅是理论调用数。

第 22 页报告的质量/效率改善，是具体翻译实验结果，不是对任意粗筛 metric 的保证。

## 5. Confidence-Based Pruning：把证据留给还可能赢的人

### 5.1 与固定粗筛的区别（第 23–25 页）

固定粗筛通常换一个便宜 metric。
置信剪枝则可以保持目标 metric，但一开始只用少量 evidence 来比较候选。

流程是：

1. 初始化较大的候选集合，以及小的证据列表。
2. 计算当前平均收益，找到暂时领先者。
3. 估计每个其他候选有多大机会追上领先者。
4. 删除机会足够低的候选。
5. 扩大证据列表，仅给仍存活的候选计算新增评分。
6. 达到最大证据预算后，对剩余候选做最终 MBR；若只剩一个，也可结束。

本质是：无需把每个明显不佳候选的期望都估得非常精确。

### 5.2 为什么不能只看当前均分（第 26–27 页）

当只有四条伪参考时，0.78 与 0.77 的差距可能只是抽样噪声。
0.78 与 0.25 的差距则可能已经足以做决策。

问题不只是“谁现在低”，而是“谁在更多证据下仍有合理机会变成赢家”。
直接估计一个候选战胜所有其他候选的概率很困难。
课件改用较简单的比较：它能否至少追平当前领先者 $b$。

### 5.3 成对 bootstrap 的数学写法（第 28–29 页；补充推导）

当前评分矩阵为：

$$
S_{ij}=G(h_i,r_j),\qquad S\in\mathbb R^{N\times m}.
$$

当前领先者：

$$
b=\arg\max_i\frac1m\sum_{j=1}^mS_{ij}.
$$

进行 $B$ 次 bootstrap；第 $k$ 次从 $\{1,\ldots,m\}$ 有放回抽取 $m$ 个索引 $I^{(k)}$。
每次比较同一组重采样 reference 上的收益：

$$
\hat p_i=\frac1B\sum_{k=1}^B
\mathbf1\left[
\frac1m\sum_{j\in I^{(k)}}S_{ij}
\ge
\frac1m\sum_{j\in I^{(k)}}S_{bj}
\right].
$$

若 $\hat p_i$ 小于设定阈值 $\tau$，则剪枝。
课件引用的论文用置信参数 $\alpha$ 表达为保留 $\hat p_i>1-\alpha$；注意别把两个符号的方向弄反。

> [!warning] 必须成对重采样
> 同一参考对多个候选有共同影响。候选 $i$ 与领先者 $b$ 必须用同一组 reference 索引进行比较，不能分别、独立抽样两组 scores。否则破坏协方差结构，估计的差值波动会失真。

也可以直接对差值 $D_{ij}=S_{ij}-S_{bj}$ 的列进行 bootstrap。
这与成对比较等价，但领先者变更后需要更新差值。

### 5.4 这个概率不是“答案正确的概率”

它度量的是在现有样本所代表的分布下，该候选追平当前领先者的 bootstrap 频率。
不能解释成：

- 答案有 99% 的事实正确率。
- 被删候选在真实分布下绝不可能获胜。
- 多轮自适应筛选后的整体错误率自动小于 $\tau$。

很小的初始 evidence、分布相关性、多个比较和重复查看，都可能使直觉上的置信度过于乐观。
实际应验证最终质量与 full MBR 的差距，并记录剪枝误差。
原始方法见 [Cheng 与 Vlachos，Faster MBR with Confidence-based Pruning](https://arxiv.org/abs/2311.14919)。

## 6. Self-Consistency 是一种特殊的 MBR

### 6.1 课件中的鸡蛋例子（第 31–35 页）

题目涉及每天 16 个鸡蛋，早餐用 3 个，做松饼用 4 个，剩下每个卖 2 美元。
正确计算是：

$$
(16-3-4)\times2=18.
$$

课件展示贪心路径给出 14；多个采样推理路径给出 18、26、18。
Self-Consistency 不选“最有说服力的一段文字”，而是提取最终答案并投票，返回 18。

不同推理路径可以通向相同答案；投票把这些路径的质量聚合到一起。

### 6.2 精确对应关系（第 36 页）

令完整输出为 $y=(z,a)$，其中 $z$ 是 reasoning chain，$a$ 是最终答案。
定义提取器 $f(y)=a$，以及 gain：

$$
G(h,r)=\mathbf1[f(h)=f(r)].
$$

代入 MBR：

$$
\hat h=\arg\max_{h\in\mathcal H}
\sum_{r\in\mathcal E}\mathbf1[f(h)=f(r)].
$$

因此它就是选择在证据中出现次数最多的最终答案。
在概率层面，估计的是：

$$
p(a\mid x)=\sum_zp(z,a\mid x),
$$

而不是选最大联合概率的那一条 $(z,a)$。
这里的概率分布是已经启用 CoT 的生成协议下的分布。

### 6.3 工程上最容易忽视的是 answer extraction

“18”“18.0”“$18”是否应该算一个答案，取决于任务。
以下情况需要明确规则：

- 单位换算之后等价，但字符串不同。
- 中间步骤含多个数字，最后答案有专用标记。
- 无法解析或输出被截断。
- 多个选项顺序不同但集合相同。
- 数学表达式需要符号等价，而非文本相同。

把所有解析失败都映射成一个 `None` 再投票，会让失败输出形成虚假的“最大共识”。
解析失败应单独统计，不能作为普通答案自然胜出。

### 6.4 第 37 页的实验曲线应怎样读

图中采样推理路径增加后，多种温度和截断策略的准确率提高，之后收益趋缓。
贪心重复执行得到的仍是同一路径，不会凭重复次数获得同样收益。

但该图不是“所有题多采样都更好”的证明。
若错误答案本身占最大的边际概率质量，更多样本会更稳定地选错。

进一步的 CoT、抽样边际化与停止准则，见 [[CMU 11-763 - Lecture 07 - Chain of Thought and Intermediate Steps]]。

## 7. Universal Self-Consistency：把开放式共识识别交给 LLM

### 7.1 为什么 exact match 不够（第 38–41 页）

开放式摘要、自由文本问答、代码，不容易提取统一的单个数值答案。
意思相近的回答可能在长度、结构和措辞上完全不同。

USC 的流程是：

1. 给同一个问题生成多个完整回答。
2. 将问题和带编号的候选交给 LLM。
3. 要求它根据多数共识，选择最一致的已有回答。
4. 返回该候选，而不是默认让 selector 重新写一个新答案。

第 40 页展示同一道数数问题的不同表述，模型将包含 30 的回答聚为共识。
第 41 页展示列举国家的开放式回答：它们部分重叠，不能靠整个字符串相同进行投票。
这些是共识选择示例，不应把示例中未验证的开放式事实直接当成可靠知识。

### 7.2 USC 与标准 pairwise MBR 的关系

两者都利用多个回答之间的一致性来选输出。
但 USC 的一次 listwise LLM 选择，未必显式计算一个固定、可分解的 $\sum_jG(h,r_j)$。

因此“同一思想框架”不等于“任何 USC prompt 都严格等价于某个已知 pairwise metric 的 MBR”。
LLM selector 是另一层估计器，有自己的错误、上下文限制和位置偏好风险。

### 7.3 第 42 页的结果与边界

课件图中，TruthfulQA 从 $k=1$ 的 62.9 提升到 $k=16$ 的 70.6；SummScreen 从 30.2 到 32.2。
这些数字对应图里的具体实验配置，不可跨模型直接比较。

GSM8K 图在 $k=8$ 为 90.2，$k=16$ 为 89.2，说明收益并非严格单调。
图中的括号数字是 USC 相对普通 SC 的差值，不是置信区间。

原论文还研究了数学、代码、长文本摘要和开放问答：[Universal Self-Consistency](https://arxiv.org/abs/2311.17311)。

### 7.4 工程延伸：selector 也必须被评估

- 改变候选顺序，检查选择是否过度依赖位置。
- 固定输出候选 ID，并校验 ID 在合法集合内。
- 将候选视为待比较的数据，不允许其中的指令改变 selector 的任务。
- 分别度量生成器的候选覆盖率与 selector 的选对率。
- 记录上下文截断是否使靠后的候选丢失。
- 承认多数共识可能是共同幻觉；事实任务仍需要外部证据或验证。

## 8. 可运行的最小实现

以下是教学实现，不是课程官方代码，也不是生产级批处理系统。

### 8.1 保留证据重复权重的 MBR

```python
from collections import Counter

def mbr_select(hypotheses, evidence, gain):
    if not hypotheses or not evidence:
        raise ValueError("hypotheses and evidence must be nonempty")
    # 候选可去重；证据只能压缩计算，不能丢掉计数。
    candidates = list(dict.fromkeys(hypotheses))
    counts = Counter(evidence)
    m = len(evidence)
    scores = [
        sum(count * gain(h, r) for r, count in counts.items()) / m
        for h in candidates
    ]
    # 明确规定：平局时按原候选顺序选择。
    winner = max(range(len(candidates)), key=lambda i: scores[i])
    return candidates[winner], scores

winner, scores = mbr_select(
    ["A", "B", "C"],
    ["A"] * 6 + ["B"] * 3 + ["C"],
    lambda h, r: float(h == r),
)
assert winner == "A"
assert scores == [0.6, 0.3, 0.1]
```

实际神经 metric 应把独特的 `(candidate, reference)` 对组织为 batch。
若 metric 依赖原输入 $x$，需要把 $x$ 也传入，并纳入缓存键。
模型版本、tokenizer、截断规则与 metric 配置不同，不能共享同一个无版本缓存。

### 8.2 成对 bootstrap 剪枝

```python
import numpy as np

def paired_bootstrap_keep(scores, threshold=0.05, n_boot=1000, seed=0):
    s = np.asarray(scores, dtype=float)
    if s.ndim != 2 or min(s.shape) == 0:
        raise ValueError("scores must be nonempty [hypothesis, evidence]")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if not 0 <= threshold < 1 or n_boot <= 0:
        raise ValueError("invalid pruning parameters")
    leader = int(s.mean(axis=1).argmax())
    rng = np.random.default_rng(seed)
    # 所有候选共享同一组重采样的列索引。
    index = rng.integers(0, s.shape[1], size=(n_boot, s.shape[1]))
    bootstrap_means = s[:, index].mean(axis=-1)
    p_catch_up = (bootstrap_means >= bootstrap_means[leader]).mean(axis=1)
    keep = p_catch_up >= threshold
    keep[leader] = True
    return keep, p_catch_up

keep, prob = paired_bootstrap_keep([
    [0.9, 0.8, 0.9, 0.8],
    [0.2, 0.1, 0.2, 0.1],
    [0.9, 0.8, 0.9, 0.8],
])
assert keep.tolist() == [True, False, True]
assert prob.tolist() == [1.0, 0.0, 1.0]
```

本实现为了清楚直接构造 $N\times B\times m$ 中间数组，大规模运行应按候选或 bootstrap 批次分块。
当所有候选相等时，所有追平概率都是 1，不应该随意剪到只剩一个。
完整迭代算法还需要保留原候选 ID、证据增长计划和跨轮 metric 缓存。

## 9. 与 AI Infra 的联系

### 9.1 调度粒度不再是单条 completion

一个用户请求可能包含几十条采样、几千对 metric 计算，以及一次 selector 请求。
基础设施需要同时统计用户级延迟和内部任务级成本。

即使 generation 全部并行，聚合仍通常要等待足够多的候选/证据完成。
长尾样本会影响 barrier 时间；提前结束会改变可用样本及可能的估计偏差。

### 9.2 Prompt prefix 可以共享，生成路径不相同

多个 sample 共享同一个输入前缀，可复用 prefill 结果或 prefix KV cache。
但一旦采样分叉，每条分支仍需要自己的后续 token 状态。
提高并行度会增加 KV 容量和调度压力，不是免费增加样本。

### 9.3 同一个 evidence 可以服务多个候选

在评分服务侧，可复用 reference 编码，或缓存确定性的 pair 分数。
能否分开编码，取决于 metric 架构：cross-encoder 不一定允许像双塔模型那样完全复用。
不要用“embedding 可缓存”泛化到所有神经 metric。

### 9.4 建议的评测账本

| 层次 | 必须记录的量 |
|---|---|
| 生成 | 候选/证据数量、解码分布、完整 token 数、失败与截断率 |
| 评分 | metric 版本、实际 pair 调用数、缓存命中、评分耗时 |
| 选择 | full-MBR 一致率、最终任务质量、剪枝误删率 |
| 系统 | 用户级 p50/p95 延迟、总 GPU 时间、显存峰值、美元成本 |
| 统计 | 随机种子、证据重复频次、候选来源、评估集划分 |

只有这样才能区分“算法更省计算”与“只是换了一套不同质量的候选”。

## 10. 课件要点辨析

本节为自拟复习问题，不是课堂实录。

- **MBR 能产生候选里没有的新答案吗？** 标准候选选择形式不能；它只选 $\mathcal H$ 中的输出。加入生成式融合就是另一个步骤。
- **更多候选总能提高真实质量吗？** 不保证。固定 evidence 下，更多候选虽能提高最大经验 gain，也可能更容易挑到被评分噪声高估的候选。
- **更多 evidence 会创造新解法吗？** 如果候选集合固定，不会；它主要改善评估，除非同时允许 evidence 加入候选。
- **共用候选与证据会不会自我加分？** 可能。若所有候选各出现一次且自相似度恒定，统一的对角项不改变排序；重复频次不同或自相似度不恒定时不能直接这么说。
- **独立 evidence 有何价值？** 候选固定且与 evidence 独立时，对其期望收益的 Monte Carlo 估计更容易分析；候选从同一证据中挑出时，需考虑选择依赖。
- **是否应该永远丢弃 diagonal？** 不应机械处理。leave-one-out 改变有限样本估计，尤其要区分删除一个出现位置与删除所有相同字符串。

## 11. 自测与面试式回答

### Q1. 用一分钟解释 MBR 与 MAP 的区别。

**回答：** MAP 选模型概率最高的完整输出；MBR 则定义候选相对于可能参考的损失，并选择期望损失最低的候选。字符串概率可能被多种等价表述分散，MBR 可通过语义或任务指标汇总这部分质量。但它的可靠性取决于参考代理分布和损失函数，并非天然知道真实答案。

### Q2. 为什么采样版 MBR 的公式里不显式出现模型概率？

**回答：** 因为证据按目标分布采样，概率已经体现在出现频率里。对 iid 样本平均 gain 就是在估计期望。若再按样本的模型概率加权，会相当于额外强调一次高概率事件，改变目标。只有从其他 proposal 分布采样时，才应考虑条件成立的重要性权重。

### Q3. Evidence 和 hypotheses 分开有什么实际好处？

**回答：** Evidence 负责忠实估计分布，hypotheses 负责覆盖高质量输出。分开后可以让 evidence 普通采样，同时让候选来自 beam、低温度或多个模型，还可以令候选数远小于证据数以减少 pairwise 评分。两者分别控制估计精度和搜索覆盖率。

### Q4. 证据中重复输出很多，怎样省计算而不改变 MBR？

**回答：** 将相同参考压缩为计数，对每个独特参考只计算一次 gain，随后乘以出现次数并除以总样本数。不能把 evidence 直接变成等权的集合。前者只是缓存计算，后者会改变概率质量；而相同候选通常可以去重，因为候选重复不需要多次参加 argmax。

### Q5. 为什么 temperature 或 top-p 会影响 MBR 的统计解释？

**回答：** 它们改变序列采样分布，最终平均 gain 估计的是修改后分布的期望，而非原模型的期望。这可以是有意的设计，但必须明确。重要性采样也不总能补救，例如截断移除了目标分布的支持时，缺失部分不能靠有限权重恢复。

### Q6. Coarse-to-Fine MBR 的收益和风险各是什么？

**回答：** 它先用便宜 metric 筛掉大量候选，再用昂贵 metric 精选，把昂贵调用从 $NM$ 降到大约 $KM$。风险是粗筛误删真正高质量或 full-MBR 最优候选，之后无法恢复。评估应同时测最终质量、赢家保留率、metric 调用和端到端延迟。

### Q7. 解释置信剪枝的 bootstrap，为什么必须“成对”？

**回答：** 把当前伪参考看作经验分布，有放回重采样 reference 索引，比较每个候选与当前领先者在相同重采样参考上的均分，统计追平频率。必须共享索引，因为参考难度会共同影响两个分数；独立重采样会破坏相关性，误估分差的不确定性。

### Q8. 追平概率小于 1% 是否等于答案错误概率大于 99%？

**回答：** 不等于。它只说明在现有 evidence 的 bootstrap 比较中，这个候选很少追平当前领先者。它不校准事实正确率，也不自动给出多轮剪枝的总体错误率。参考分布错误、metric 有偏或初始样本不足，都会使这个统计量具有误导性。

### Q9. 怎样把 Self-Consistency 写成 MBR？

**回答：** 对每条完整推理输出提取最终答案，定义 gain 为两个最终答案的 exact match。对证据求和就得到该答案的票数，选最大值就是多数投票。该方法聚合了不同推理路径通向同一最终答案的概率质量，而不要求推理文本相同。

### Q10. USC 与普通 SC、pairwise MBR 有何区别？

**回答：** 普通 SC 依赖可规范化的答案提取和显式投票；pairwise MBR 显式计算候选与伪参考的 gain；USC 则把多个回答交给 LLM，要求它选择最符合多数共识的回答。USC 更适合自由文本，但需要评估 selector 的位置偏好、上下文限制和选择错误，也不一定严格等价于固定的 pairwise 求和。

### Q11. 多采样后结果更稳定，是不是说明更可靠？

**回答：** 只能说明决策相对于采样分布更稳定。若模型共同相信同一个错误事实，或者 metric 偏好空泛答案，多采样会更确定地选错。需要区分抽样方差下降与系统偏差下降，事实可靠性仍要由外部证据、验证器或独立评测检验。

### Q12. 设计一个公平的高效 MBR 实验，至少控制哪些变量？

**回答：** 固定模型、prompt、候选和目标 evidence 分布，比较 full MBR、粗筛与置信剪枝。除最终任务质量，还报告实际生成 token、pair 调用、缓存、用户级延迟和总 GPU 时间；记录多随机种子及 full-MBR 赢家保留情况。否则质量提升可能来自更好的候选，速度提升也可能只是更小的工作量，而不是筛选算法本身。

## 12. 最后应形成的心智模型

把一次多样本推理分成四层：

1. **候选生成：** 有哪些值得返回的输出？
2. **参考分布：** 用谁来代表不确定的真实答案？
3. **收益函数：** 什么叫与参考一致、任务上有用？
4. **有限预算估计：** 用多少 evidence、多少评分，才能足够可靠地作出选择？

MBR、Self-Consistency 与 USC 的共同点，是用多个输出之间的关系改善最终决策。
区别在于：共识如何定义、如何估计，以及计算花在了哪一层。

课程索引：[[CMU 11-763]]。
