---
type: course-note
status: developing
course: "[[CMU 11-763]]"
lecture: 5
lecture_date: 2025-09-09
area: inference
topics:
  - "[[LLM Inference]]"
aliases:
  - CMU 11-763 Lecture 05
  - Intro to A* and Best First Search
  - A Star and Best First Search
video_url: https://www.youtube.com/watch?v=Cal4oRoumTw
---

# Lecture 05：A* and Best First Search

> [!abstract] 本讲一句话
> 把生成看成加权路径搜索后，beam、uniform-cost、A* 和 best-first beam 的差别主要在队列排序、剪枝预算、未来成本估计和停止条件；“少扩展节点”必须与“找到什么意义上的最优解”一起说明。

## 来源与范围

- 课程：CMU 11-664/763 — Inference Algorithms for Language Modeling，Fall 2025。
- 讲师：Graham Neubig；日期：2025-09-09。
- [课程视频](https://www.youtube.com/watch?v=Cal4oRoumTw)，时长 59:45；本文完整核对英文字幕。
- [原官方网页课件入口](https://www.phontron.com/class/lminference-fall2025/assets/slides/2025-09-09-best-first-search/index.html)：整理时已被站点新版应用入口替代，不能取得完整原始 HTML；搜索索引仅保留部分旧课件正文。
- 本讲完整主来源是字幕，旧课件索引只交叉核对部分定义；下文不编造课件页码。
- 原论文：[Best-First Beam Search, Meister et al. (2020)](https://arxiv.org/abs/2007.03909)、[Modeling Future Cost for Neural Machine Translation, Duan et al. (2020)](https://arxiv.org/abs/2002.12558)。相关算法按论文对应章节额外核对。
- [课程主页](https://www.phontron.com/class/lminference-fall2025/)。

> [!warning] 三处课堂表述需要校正
> 1. 课堂称为“depth-first search”的演示按累计成本从小到大取节点，准确名称是 uniform-cost / best-first search，不是栈式 DFS。
> 2. “Transformer 没有 admissible heuristic”不能按字面理解：NLL 成本下 $h=0$ 始终 admissible；难的是有用且可保证的非零下界。
> 3. 课堂对剩余成本预测器的泛化讨论，不等于所引 NMT 论文的精确实现；后者通过下一词的 future-context 分布及辅助训练建模未来信息，并非直接回归全部剩余 NLL。

## 视频时间索引

| 时间 | 内容 |
| --- | --- |
| [00:09](https://www.youtube.com/watch?v=Cal4oRoumTw&t=9s) | 为什么重新考虑高模型分数搜索 |
| [04:17](https://www.youtube.com/watch?v=Cal4oRoumTw&t=257s) | Greedy、beam 回顾 |
| [05:28](https://www.youtube.com/watch?v=Cal4oRoumTw&t=328s) | 加权有限状态自动机 |
| [08:00](https://www.youtube.com/watch?v=Cal4oRoumTw&t=480s) | 概率、logprob、负 logprob |
| [11:18](https://www.youtube.com/watch?v=Cal4oRoumTw&t=678s) | Greedy 演示 |
| [13:10](https://www.youtube.com/watch?v=Cal4oRoumTw&t=790s) | Beam 与同长度预算 |
| [16:40](https://www.youtube.com/watch?v=Cal4oRoumTw&t=1000s) | 不剪枝的最小累计成本搜索 |
| [19:54](https://www.youtube.com/watch?v=Cal4oRoumTw&t=1194s) | A* 的 $g+h$ 排序 |
| [23:05](https://www.youtube.com/watch?v=Cal4oRoumTw&t=1385s) | Admissibility 与下界示例 |
| [27:47](https://www.youtube.com/watch?v=Cal4oRoumTw&t=1667s) | 启发函数与最优性的课堂问答 |
| [35:28](https://www.youtube.com/watch?v=Cal4oRoumTw&t=2128s) | Transformer 搜索难点 |
| [37:48](https://www.youtube.com/watch?v=Cal4oRoumTw&t=2268s) | Hypothesis recombination |
| [44:47](https://www.youtube.com/watch?v=Cal4oRoumTw&t=2687s) | 学习 future cost |
| [50:36](https://www.youtube.com/watch?v=Cal4oRoumTw&t=3036s) | 全部成本还是剩余成本？ |
| [54:23](https://www.youtube.com/watch?v=Cal4oRoumTw&t=3263s) | Best-first beam search |
| [56:32](https://www.youtube.com/watch?v=Cal4oRoumTw&t=3392s) | 统一搜索框架 |
| [59:34](https://www.youtube.com/watch?v=Cal4oRoumTw&t=3574s) | 转交下一篇学生报告 |

## 1. 为什么模型变好后还要研究搜索？

讲师开头提出一个研究判断，不是证明所有任务都应改用 A*。
早期模型的高概率输出可能重复、过短或直接结束；精确找到 mode 不自动得到好文本。
采样温度与截断等方法，也因此兼有修正模型输出倾向的作用。

如果模型分数与任务质量的关系改善，高分区域可能更值得仔细搜索。
本讲的问题是：能否把有限推理计算集中到更可能产生高分完整序列的路径？

需要区分三层：

- 模型是否给好答案高分：建模与对齐问题。
- 能否找到该分数下的好路径：搜索问题。
- 找到路径消耗多少时间和显存：系统问题。

开场提到的模型代际、greedy 表现和 API 参数限制，按 2025 年课堂背景理解。
它们不是本笔记对当前服务接口或所有任务的结论。

## 2. 用一个加权图表示生成

### 2.1 图的元素

[05:28](https://www.youtube.com/watch?v=Cal4oRoumTw&t=328s) 用 weighted finite-state automaton 描述玩具空间：

$$
\mathcal A=(S,\Sigma,\delta,s_0,F,w).
$$

- $S$：状态集合。
- $\Sigma$：输出符号表，可以对应 token 词表。
- $\delta$：带标签的状态转移。
- $s_0$：起始状态。
- $F$：接受／终止状态集合。
- $w$：转移权重。

从起点到接受状态的一条路径，对应完整输出。
课堂用 A、B、C、EOS 等符号和出边概率演示。

对有最大长度限制的 LM，可将所有前缀展开成有限但极大的树。
没有长度界时，词表有限不代表前缀状态数有限。
玩具 WFSA 是算法抽象，不表示实际 Transformer 存在一个很小的精确有限图。

### 2.2 LM 的状态不能只看最后一个词

搜索状态至少隐含：

$$
(x,y_{1:t},\text{model/configuration},\text{constraint state}).
$$

KV cache 是前缀的可复用计算表示，不是天然可以合并的“语义标签”。
两个前缀末词相同，通常不意味着下一 token 分布相同。
完整 token 前缀、输入及推理配置相同，在确定性推理条件下才诱导相同未来分布。

## 3. 三种计分约定必须统一

[08:00](https://www.youtube.com/watch?v=Cal4oRoumTw&t=480s) 的核心是固定优化方向。

| 边权 | 路径聚合 | 路径目标 |
| --- | --- | --- |
| $p(v\mid s)$ | 相乘 | 最大概率 |
| $\log p(v\mid s)$ | 相加 | 最大 logprob |
| $-\log p(v\mid s)$ | 相加 | 最小非负成本 |

下文统一采用第三种。前缀 $n=y_{1:t}$ 的已付成本是：

$$
g(n)=-\sum_{i=1}^t\log P(y_i\mid x,y_{<i}).
$$

每条边的 NLL 非负，所以继续扩展不会降低 $g$。
例如路径概率 $0.5\times0.3\times0.2=0.03$：

$$
\log0.03\approx-3.5066,\qquad -\log0.03\approx3.5066.
$$

EOS 若是模型采出的 token，也应计入完整路径成本。
长度惩罚或外部奖励改变了评分函数，不能改分后无条件复用原始 NLL 的单调性。

## 4. Greedy 与 beam 的队列行为

### 4.1 Greedy

[11:18](https://www.youtube.com/watch?v=Cal4oRoumTw&t=678s) 的 greedy 只保留一条路径。
它选当前最便宜出边，并忘掉其他分支。
课堂小图中进行了 4 次扩展，这是该图上的计数，不是一般复杂度。

Greedy 可用容量为 1 的局部结构表达。
但“都用了容量为 1 的队列”不足以说明两个算法相同，还要看比较器和丢弃候选的时机。

### 4.2 Beam

[13:10](https://www.youtube.com/watch?v=Cal4oRoumTw&t=790s) 用 beam width $B=2$。
统一队列视角下，标准 beam 首先比较长度，再比较累计成本：

$$
\text{priority}_{\text{beam}}(n)=(|n|,g(n)).
$$

$B$ 是同一长度的保留／扩展预算，不是任何时刻全局队列最多只有 $B$ 个条目。
逐节点实现里，队列可以同时存未展开的长度 2 前缀和已产生的长度 3 候选。

课堂图中 beam 用 6 次扩展找到成本约 2.43 的路径。
它比 greedy 多保留几种早期选择，但仍可能把最终最优路径剪掉。

> [!tip] 三种计数不要混用
> 节点扩展次数、模型 forward 次数、GPU batch 数不是同一个量；逐层向量化和逐节点队列可能表达相同选择规则，却具有不同执行代价。

## 5. 不剪枝：uniform-cost search

[16:40](https://www.youtube.com/watch?v=Cal4oRoumTw&t=1000s) 把所有候选留在队列，反复取累计成本最小者：

$$
\text{priority}_{\text{UCS}}(n)=g(n).
$$

长而确定的前缀，可以先于短但已付出较多 NLL 的前缀扩展。
这不是栈式 DFS，也不是按长度逐层推进的 beam。

### 5.1 为什么首个弹出的完整路径是最优的？

在非负边成本、有可达目标且搜索满足终止条件时：

1. 前缀成本是所有完整后代成本的下界。
2. 弹出完整路径时，没有成本更低的未决前缀。
3. 因而不存在成本更低的未发现完整后代。

课堂图上 UCS 用 9 次扩展得到最优路径，但多访问了浅层分支。
真实 LM 的大词表会放大这种开销。

### 5.2 正确不等于可计算

长度上限为 $T$ 的前缀树，最坏包含：

$$
1+|\mathcal V|+|\mathcal V|^2+\cdots+|\mathcal V|^T
$$

个状态。
若没有长度界，还存在零成本边或无限低成本延长路径，算法可能迟迟无法返回目标。
实现必须定义最大长度、EOS、无解和预算耗尽的处理方式。

## 6. A*：用 $g+h$ 安排计算

[19:54](https://www.youtube.com/watch?v=Cal4oRoumTw&t=1194s) 引入：

$$
f(n)=g(n)+h(n).
$$

- $g(n)$ 是已真实付出的成本。
- $h(n)$ 是到接受状态的剩余成本估计。
- $f(n)$ 是完整路径成本估计，越小越优先。

某个前缀可能进入高度可预测的固定表达，剩余完成很便宜。
另一个前缀当前成本低，却仍需完成很多不确定选择。
启发函数试图把这一差别提前放进调度。

A* 首先改变下一步探索哪条路径，并不改变原模型概率。
若再加 beam 剪枝或不可靠 heuristic，也可能改变最终找到的结果。

### 6.1 Admissibility 的不等号方向

最优真实剩余成本定义为：

$$
h^*(n)=\min_{y:\ n\preceq y,\ y\text{ complete}}
\bigl(g(y)-g(n)\bigr).
$$

Admissible 要求：

$$
h(n)\le h^*(n).
$$

它是乐观下界，不会把还未发生的成本估得过高。
通常对目标设 $h=0$；NLL 非负时，处处 $h=0$ 合法，此时 A* 就是 UCS。

课堂手工把部分状态的下界设为 0.5 或 1.0，帮助搜索优先深入有希望分支。
同一图中 A* 用 8 次扩展，比 UCS 少一步；这是原理示例，不是普遍加速比例。

### 6.2 最优性证明的核心

以下形式化补齐 [27:47](https://www.youtube.com/watch?v=Cal4oRoumTw&t=1667s) 后的课堂问答。
设最优完整路径成本是 $C^*$，该路径上的一个未决前缀为 $n^*$。
下界条件给出：

$$
f(n^*)=g(n^*)+h(n^*)\le C^*.
$$

如果较差完整路径 $z$ 先被弹出，且目标的 $h(z)=0$：

$$
f(z)=g(z)>C^*\ge f(n^*),
$$

就与“每次弹出最小 $f$”矛盾。
保证依赖最优前缀未被剪掉、队列和停止规则正确，以及搜索能够完成。

### 6.3 从树变成图还需要什么？

若把不同路径合并为同一图状态，必须处理更低成本路径后来到达的情况。
一个常用充分条件是 consistency：

$$
h(n)\le c(n,n')+h(n').
$$

如果 heuristic 不 consistent，图搜索可能需要 reopen 状态。
因此不能只写“admissible”就认为任何 visited-set 实现都正确。
这一点是算法补充，前缀树场景更容易直接应用前述证明。

## 7. 启发函数的课堂问答

### 7.1 Admissibility 是每次找到最优解的必要条件吗？

不是。某个会高估的函数也可能在具体实例上恰好返回最优。
Admissibility 是一组算法假设下的保证条件，不是对一次运行结果的事后必要条件。

### 7.2 低估过多会怎样？

更多路径看起来值得探索，效率下降。
下界越松，对排序的帮助越小；$h=0$ 安全，但不一定实用。

### 7.3 给所有 heuristic 加相同常数，排序不是不变吗？

若所有对象、包括目标都加同一个常数，单纯排序确实不变。
但标准证明依赖目标 $h=0$ 与下界停止规则。
只给非目标或部分状态加常数，就可能改变返回时机。

正确答案是检查完整排序与停止契约，不能由此推出“任意高估也安全”。

### 7.4 有限 beam 下还能直接用 A* 全局保证吗？

不能。只要最优路径前缀被剪掉，后续排序无法把它找回。
Admissible A* 和有 beam 上限的 A* 是不同保证等级。

## 8. Transformer 上的两个困难

### 8.1 状态几乎不能精确复用

[35:28](https://www.youtube.com/watch?v=Cal4oRoumTw&t=2128s) 强调完整历史依赖。
传统小图可能让多条路径到达同一状态；LM 中选一个 token 往往就制造新的前缀状态。
如果每次扩展都要做昂贵神经网络计算，聪明的队列也无法消除指数分支。

### 8.2 有用的非零下界难求

某些完成会进入近乎确定的记忆文本、固定格式或立即 EOS。
最优剩余 NLL 因而可能接近零。
要证明严格正下界，需要排除所有更便宜的合法完成方式。

特别注意方向：

- 很低的未来概率会增加 NLL。
- 强下界的难点是可能存在高概率、低成本的完成。
- “未来不确定”不是一个对所有完成路径成立的正下界证明。

有硬性剩余长度、概率上界或有限状态结构时，可以研究更强下界。
本讲说的是通用 Transformer 上困难，不是数学上绝不可能。

## 9. Hypothesis recombination

[37:48](https://www.youtube.com/watch?v=Cal4oRoumTw&t=2268s) 讨论合并相似状态，只保留较便宜路径。

### 9.1 精确合并的条件

两状态必须有完全相同的合法后续集合，并对所有后续诱导相同转移分布。
这样相同后缀接到较便宜路径上，总成本不会更高。

“下一个 token 相似”比“所有未来等价”弱得多。
约束状态、终止规则也要一并比较。

### 9.2 课堂的三类近似判据

| 判据 | 直觉 | 风险 |
| --- | --- | --- |
| 最近 $n$ 个 token 相同 | 以有限阶上下文近似历史 | 忘掉早期事实、指令、约束 |
| Hidden state 欧氏／余弦接近 | 表示近似则未来可能近似 | 几何距离未必对应输出误差 |
| 下一 token 分布 KL 接近 | 直接比较当前预测行为 | 当前相似不保证多步相似 |

[43:44](https://www.youtube.com/watch?v=Cal4oRoumTw&t=2624s) 讲师澄清 KL 比较的是概率分布，不是原始 hidden vectors：

$$
D_{KL}(p\Vert q)=\sum_vp(v)\log\frac{p(v)}{q(v)}.
$$

KL 通常不对称，需定义方向、平滑和阈值。
课堂以约 16 条 beam 举例：先有 $16\times|\mathcal V|$ 概率表，再做两两比较。
比较本身也是额外成本。

### 9.3 不要与 KV prefix sharing 混为一谈

相同前缀共享 KV 是精确计算复用。
相似前缀合并 hypothesis 是改变搜索空间的近似。
应分别报告 KV bytes 节省、合并数量、质量损失和是否还保留保证。

## 10. 学习 future cost：先分清课堂想法与论文实现

[44:47](https://www.youtube.com/watch?v=Cal4oRoumTw&t=2687s) 提出：严格下界难做时，可以训练更有用但不保证 admissible 的估计。
按最小成本约定，可写成：

$$
f_\lambda(n)=g(n)+\lambda\hat h_\phi(n).
$$

$\lambda=0$ 不使用预测器；较小正系数弱化其排序影响。
但 $0<\lambda<1$ 不自动把任意高估器变成下界。

### 10.1 Remaining-cost head 的教学形式化

[50:36](https://www.youtube.com/watch?v=Cal4oRoumTw&t=3036s) 先讨论完整序列分数，随后纠正为当前状态之后的剩余成本。
对某条完成轨迹 $y_{1:T}$，可构造目标：

$$
R_t=-\sum_{j=t+1}^T\log P_\theta(y_j\mid x,y_{<j}).
$$

一个可能的回归实现是：

$$
\mathcal L_{\text{heuristic}}
=\mathbb E\bigl[(\hat h_\phi(x,y_{1:t})-R_t)^2\bigr].
$$

这是课堂通用想法的形式化补充，不是所引论文的原算法。
若把整个 $g(y)$ 当 $h$ 再加 $g(n)$，会重复计算已经发生的成本。
某条参考或采样 suffix 的成本，也不等于最优剩余成本 $h^*$。
回归平均 rollout 成本，不会自动给出最优成本下界。

### 10.2 训练和推理的选择

课堂讨论了联合训练辅助 head、冻结主模型后再训练，以及用大模型轨迹训练小预测器。
这些方式影响训练代价、调用成本和模型更新后的失配风险。
基础模型、tokenizer 或 rollout 分布变化后，应重新验证。

讲师将其类比 value function，因为两者都估计未来量。
但模型似然成本不天然等于任务正确率、环境奖励或用户偏好。

### 10.3 原论文具体做了什么？

[Modeling Future Cost for Neural Machine Translation](https://arxiv.org/abs/2002.12558) 的 §3–4 是：

1. 由当前 token embedding 和当前上下文构造 future context。
2. 用它预测下一目标词的分布。
3. 加入未来词预测辅助目标；另一变体把 future context 进一步用于下一步生成。

它不是直接用标量 head 回归直到 EOS 的全部剩余 NLL。
论文支持“显式未来信息可改善 NMT”的动机；课堂延伸的 A* heuristic 设计应单独理解。
2020 年翻译实验增益，也不是今天任意 LLM 的保证。

## 11. Best-first beam search

[54:23](https://www.youtube.com/watch?v=Cal4oRoumTw&t=3263s) 引入 Meister、Vieira、Cotterell 的工作。
标准 beam 先处理同一层的 $B$ 条路径，即使有些已经明显很差。
BFBS 跨长度按当前分数排序，但仍保留同长度的 beam 预算。

$$
\text{priority}_{\text{BFBS}}(n)=(g(n),|n|),\qquad
\text{budget at each length}=B.
$$

它不是“整个队列只留 $B$ 个节点”，也不是取消所有剪枝的 UCS。

### 11.1 单调性为何重要？

论文用最大化 score 记号：

$$
\operatorname{score}(y_{1:t})\ge
\operatorname{score}(y_{1:t+1}).
$$

累计 logprob 满足该条件，因为追加 logprob 不大于零。
已经很差的路径不能靠继续生成提高原始累计分数。

当某一长度已有 $B$ 个更好的状态被处理，部分仍排队的短前缀可以被证明无法进入对应 beam。
这些路径就不用继续做模型扩展。
在论文规定的剪枝、比较和停止条件下，可在首个完整候选成为队首时停止求 top-1。

### 11.2 保证的是 beam-equivalent

这里的 $B$-optimal / $k$-optimal 是相对于对应 beam 搜索的结果。
标准 beam 本身可能错过全局最高概率序列。

正确表述：对单调评分和匹配的算法约定，BFBS 可以少扩展，同时返回与标准 beam 相同的目标结果。
加入 learned heuristic、非单调长度归一化或改变 EOS 规则后，必须重新分析。

### 11.3 概念级伪代码

下面省略完整 top-$B$ 返回、EOS 延展表示和额外安全剪枝；复现以论文 Algorithm 2 为准。

```python
queue = min_priority_queue(key=(negative_logprob, length))
queue.push(start)
pops_at_length = Counter()

while queue:
    node = queue.pop()
    t = len(node)
    if t > max_length or pops_at_length[t] >= beam_width:
        continue
    pops_at_length[t] += 1

    if node.ends_with_eos:
        return node  # top-1, monotonic cost, matched stopping rules

    for token in vocabulary:
        child = extend(node, token)
        child.cost = node.cost - log_p(token, node)
        queue.push(child)
```

枚举词表是算法表达，不要求给所有 child 立即分配完整 KV。
可以先保存 token id、累计分数和父节点引用，延迟物化昂贵状态。

## 12. 用四个开关统一算法

将 [56:32](https://www.youtube.com/watch?v=Cal4oRoumTw&t=3392s) 的表统一成最小成本约定：

| 方法 | 首要排序 | 同长度预算 | heuristic | 典型保证 |
| --- | --- | --- | --- | --- |
| Beam | 长度，再 $g$ | 有限 $B$ | 0 | 近似搜索 |
| UCS / best-first | $g$，再长度 | 无限制 | 0 | 非负成本及终止条件下全局最优 |
| A* | $g+h$ | 无限制 | admissible | 完整假设下全局最优 |
| Best-first beam | $g$，再长度 | 有限 $B$ | 0 | 单调分数下可与对应 beam 等价 |
| A* beam | $g+h$ | 有限 $B$ | 手工／学习 | 剪枝可能改变结果 |

停止规则是另一个不能省略的开关。
“遇到 EOS”“EOS 在最优队首”“已有 $B$ 个完整候选”不总能互换。

## 13. AI Infra：少扩展不等于低延迟

### 13.1 Batch 形态

标准 beam 的同长度状态容易形成规则 batch。
Best-first 的前缀长度、KV 长度和执行顺序更不规则。
节省 token 扩展的收益可能被小 batch、调度及 CPU–GPU 同步抵消。

至少同时测量：

- 扩展前缀数、实际 forward 数、处理 token 数。
- Batch occupancy、吞吐、端到端延迟。
- Queue 操作及同步时间。
- 最终候选的模型分数和任务质量。

### 13.2 Frontier 的显存生命周期

逐层 beam 可较早释放旧层；best-first 会同时保留多个深度。
父指针、prefix sharing、延迟物化和块式 KV 管理可节省空间。
缓存共享本身不解决近似状态合并的正确性。

### 13.3 Heuristic 也消耗预算

为少调用一次大模型而对全词表调用昂贵预测器，可能得不偿失。
轻量 head、小模型、候选预筛与 batched scoring 都需算入总 FLOPs、延迟和显存。

### 13.4 先在可穷举任务上验证

固定模型、词表、EOS、长度上限、评分方式、beam、精度和 tie-breaking。
先在短长度小词表上检验 beam 与 BFBS 一致，再接入真实 LM。
论文最高约 10 倍加速来自特定实验，不是现代 GPU serving 的无条件保证。

## 14. 自测与面试回答

### Q1：为什么最大概率生成能转成最短路径？

**回答：** 概率乘积取负对数后变成非负边成本之和，因此最大概率等价于最小 NLL。额外奖励或长度归一化可能改变评分及单调性，必须重新检查搜索保证。

### Q2：Greedy、beam、UCS 的本质差别是什么？

**回答：** Greedy 只保留一个局部选择，beam 每长度保留有限前缀，UCS 在全部未决前缀中按累计成本排序且不做有限 beam 剪枝。区别在比较器、预算和停止规则，而非是否使用队列。

### Q3：为什么课堂的 DFS 演示实际上是 UCS？

**回答：** 演示按累计成本选节点，而非后进先出。普通 DFS 的首个目标不保证最低成本；非负成本下的 UCS 才有相应保证，还需满足终止条件。

### Q4：Admissible heuristic 是什么？

**回答：** 在最小成本约定下，$h(n)\le h^*(n)$，即不高估最优剩余成本。配合正确队列、停止和不误剪路径的规则，A* 可返回最优解；图搜索还要处理一致性或状态重开。

### Q5：$h=0$ 有什么缺点？

**回答：** 正确性没有问题，NLL 非负时它始终 admissible；缺点是没有未来信息，A* 退化成 UCS，可能扩展大量分支。难点是寻找便宜且足够紧的下界。

### Q6：Transformer 的 heuristic 为什么难？

**回答：** 完整历史造成巨大状态空间，且可能存在近乎确定、剩余成本接近零的完成。要证明正下界，就要排除所有更便宜完成；不能把这个困难误说成所有 admissible heuristic 都不存在。

### Q7：给 learned heuristic 乘小于 1 的系数能保证安全吗？

**回答：** 不能。即使缩小后的估计仍可能高于真实最优剩余成本，尤其后者接近零时。除非有误差界及严格修正，否则只有回到零估计才直接恢复这个下界保证。

### Q8：为什么应预测剩余成本而非整个序列成本？

**回答：** $g$ 已计算过去，$h$ 再包含过去会重复计分。即便只预测 suffix，也要区分采样后缀成本、平均成本与最优剩余成本；预测准确不等于 admissible。

### Q9：什么时候 hypothesis 可以精确合并？

**回答：** 两状态必须有相同合法后续和相同未来转移分布，才可让便宜路径支配贵路径。末尾词相同、hidden state 接近或当前 KL 小只是近似判据，不能直接继承最优保证。

### Q10：BFBS 怎样加速且不改变 beam 结果？

**回答：** 利用 logprob 随扩展不增加，跨长度按分数排序并保留同长度 beam 预算，跳过注定掉出 beam 的路径，按证明过的条件提前停止。保证是对应 beam 的结果等价，不是全局最优。

### Q11：节点数少 10 倍为什么不一定快 10 倍？

**回答：** 还存在不规则 batch、跨深度 KV、优先队列、预测器和同步成本。GPU 利用率可能降低，所以必须测实际 forward、吞吐、显存和墙钟时间。

### Q12：最合适的实现练习是什么？

**回答：** 在可穷举的小模型上实现 greedy、beam、UCS、A*、BFBS，比较路径、成本和扩展数。再加入错误 heuristic、非单调分数与不同 EOS 规则做反例，最后才接入真实 LM 与 KV cache。

## 15. 本讲结束时应保留的判断

1. 先定义目标，再决定搜索算法。
2. 最优必须注明是全局、对应 beam，还是实验质量更好。
3. Heuristic 既有预测问题，也有下界及停止条件问题。
4. 精确 prefix 复用和近似 hypothesis 合并应分别报告。
5. 少扩展的理论收益，需要真实系统测量来兑现。

视频在转交 Neurologic A*esque 学生报告后结束；报告正文不在这段录像内。
不能把论文全文或下一讲 recap 当作本段已详细讲授的内容。

## 关联内容

- [课程索引](CMU%2011-763.md)
- [上一讲：Beam Search and Variants](CMU%2011-763%20-%20Lecture%2004%20-%20Beam%20Search%20and%20Variants.md)
- [下一讲：Other Controlled Generation Methods](CMU%2011-763%20-%20Lecture%2006%20-%20Other%20Controlled%20Generation%20Methods.md)
- [LLM Inference](../../topics/inference/LLM%20Inference.md)
