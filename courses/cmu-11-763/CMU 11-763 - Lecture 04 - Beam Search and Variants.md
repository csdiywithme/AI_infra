---
type: course-note
status: developing
course: "[[CMU 11-763]]"
lecture: 4
lecture_date: 2025-09-04
area: inference
topics:
  - "[[LLM Inference]]"
aliases:
  - CMU 11-763 Lecture 04
  - Beam Search and Variants
video_url: https://www.youtube.com/watch?v=2hhyfPYGCmY
---

# Lecture 04：Beam Search and Variants

> [!abstract] 本讲一句话
> Beam search 用有限宽度的逐层搜索近似序列 MAP；diverse beam search 用候选间差异修改目标，stochastic beam search 用条件 Gumbel 技巧实现序列无放回采样，而更好地搜索模型概率不一定带来更好的任务输出。

## 来源与范围

- 课程：CMU 11-664/763 — Inference Algorithms for Language Modeling，Fall 2025。
- 讲师：Amanda Bertsch；日期：2025-09-04。
- [课程视频](https://www.youtube.com/watch?v=2hhyfPYGCmY)，时长 1:11:35，完整核对英文字幕。
- [官方 Slides，共 51 页](https://docs.google.com/presentation/d/1CXm9nEhp66nF1-KYgai0nb5GT2fpnLWaodeP-C25Mq0/edit)。
- [Diverse Beam Search 原论文](https://arxiv.org/abs/1610.02424)，重点核对分组和 diversity function。
- [Stochastic Beams and Where to Find Them 原论文](https://arxiv.org/abs/1903.06059)，重点核对算法 1、条件 Gumbel 变换及实验设置。
- [课程主页](https://www.phontron.com/class/lminference-fall2025/)。

> [!note] 整理方式
> 本文按字幕和完整课件组织，公式图页另行视觉核对；数学推导和 AI Infra 细节作为补充。课堂对 GPT 模型、Hugging Face 或 serving 库的描述保留其 2025 年上下文，不外推为当前 API 行为。

> [!warning] 三处应避免照抄的课堂简写
> 1. 普通 Transformer 上把 beam width 设成词表大小仍不等于穷举；长度增加后前缀数会指数增长。
> 2. Stochastic beam search 的“cap 到父节点”是特定条件 Gumbel 变换，不能用普通 `min(child, parent)` 代替。
> 3. 原论文的精确无放回采样保证不允许随意叠加 length normalization 或未经证明的 early stopping。

## 视频时间索引

| 时间 | 内容 | 课件 |
| --- | --- | --- |
| [00:49](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=49s) | 从 sampling 转向 mode-seeking | 3–4 |
| [02:20](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=140s) | Greedy 的局部最优与序列反例 | 5–6 |
| [03:54](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=234s) | Repetition trap | 7 |
| [11:19](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=679s) | 延后低概率词 | 8 |
| [12:20](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=740s) | Beam search 基本思想 | 9 |
| [13:46](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=826s) | Cat 例子：expand、prune、EOS | 10–17 |
| [18:17](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=1097s) | 长度归一化与长度奖励 | 18–19 |
| [23:09](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=1389s) | Beam 候选相似与 diverse beam search | 20–27 |
| [27:18](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=1638s) | 分组流水线与参数 | 28–30 |
| [30:37](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=1837s) | Diversity 度量与问答 | 31 |
| [37:22](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=2242s) | 无放回采样的需求 | 32–34 |
| [40:06](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=2406s) | Gumbel-Max 与 Gumbel-Top-k | 35–38 |
| [47:43](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=2863s) | 跨层传递 Gumbel score | 39–41 |
| [50:13](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=3013s) | 质量–多样性曲线 | 42 |
| [57:06](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=3426s) | Temperature 与随机 beam 的问答 | 课堂补充 |
| [1:01:29](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=3689s) | Beam width、curse、mode 问题 | 43–47 |
| [1:07:29](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=4049s) | Blessing of beam search | 48 |
| [1:08:45](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=4125s) | 当代用途、成本和课堂总结 | 49–51 |

## 1. 解码的优化视角：找完整序列的 mode

前一讲多数方法问“从某个分布抽什么”；本讲先问“哪条完整输出概率最大”。

$$
y^*=\arg\max_{y\in\mathcal Y}P_\theta(y\mid x)
=\arg\max_y\sum_{t=1}^{|y|}\log P_\theta(y_t\mid x,y_{<t}).
$$

这通常叫 MAP decoding 或 mode-seeking decoding。
条件 $x$ 固定时，目标是输出序列分布的 mode。

这里的“最好”是指定模型概率最高，并未包含事实性、人类偏好或外部 reward。
如果加入长度惩罚或多样性奖励，目标就已经发生改变。

## 2. Greedy 的三个问题

### 2.1 局部 argmax 不等于序列 argmax

Greedy 每一步选：

$$
\hat y_t=\arg\max_vP_\theta(v\mid x,\hat y_{<t}).
$$

单 token 输出时它恰好解决 MAP；长序列时未来分布取决于已选 token。

课件 6 页使用“当前较高概率，但下一步没有高概率 completion”的树状反例。
下面给出同结构的简化数值推导：

| 路径 | 第一个 token | 最佳后续 | 完整概率 |
| --- | --- | --- | --- |
| A | 0.6 | 0.1 | 0.06 |
| B | 0.4 | 0.9 | 0.36 |

Greedy 先选 A，之后无法回头；B 的完整概率反而更高。
这属于 search error：优化算法没找到模型本身最喜欢的序列。

### 2.2 Repetition trap：重复使继续重复更容易

[03:54](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=234s)，课件 7 页展示 GPT-3 写作逐渐陷入重复。

形成机制是：

1. 生成进入重复性较强的上下文；
2. 模型在训练中学过重复结构，所以“再重复一次”变得更可能；
3. Greedy 每步选最高项，使这种模式继续；
4. 更多重复又进一步强化该上下文。

Stochastic sampling 有机会选其他 continuation，但不是逃离循环的保证。
Instruction tuning 或偏好训练也可能降低这种行为，但不能据此断言所有重复都消失。

课堂提醒：重复不只来自低质量数据，CSV、模板和结构化文件也存在合理重复。
因此单纯把重复等同于“坏训练数据”太粗糙。

### 2.3 低概率内容词可能被推迟

课件比较：

- “The dog was seen by Jane”；
- “Jane saw the dog”。

若实体 Jane 当前概率低，逐 token 贪心可能优先生成常见功能词，直到后文不得不补实体。
因此可出现拗口的被动句或延后信息结构。
这是教学直觉，并非所有被动句都能被该机制解释。

## 3. Beam search：把单条承诺变成有限候选集

给定 beam width $B$，每个长度保留一组前缀 $\mathcal B_t$。

$$
\mathcal C_{t+1}
=\{y\circ v:y\in\mathcal B_t,\ v\in\mathcal V\}.
$$

$$
\mathcal B_{t+1}
=\operatorname{TopB}_{y\in\mathcal C_{t+1}}s(y).
$$

普通 raw-logprob score 为：

$$
s(y\circ v)=s(y)+\log p(v\mid x,y).
$$

算法仍是近似：一旦某条前缀被 prune，后面的高概率续写就无法挽回它。

### 3.1 课堂 cat 例子

[13:46](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=826s)，课件 10–17 页。
输入为 “When my cat gets hungry”，$B=3$：

1. 先保留 “she”“he”“it”；
2. 分别扩展，例如 “she meows”“she starts”“he …”；
3. 根据完整 prefix score，从所有扩展中取全局 top-3；
4. 重复，直到完成或达到长度上限。

保留下来的三条路径不必来自三个不同父节点。
它们可以来自一个父节点，也可以来自多个父节点。

这一点很重要：标准 beam 没有“每个父节点留一个名额”的公平配额。
因此一个高概率早期分支可能逐步占满所有 beam。

### 3.2 为什么课件只画 $B^2$ 个扩展

课件用“每个父节点只取 top-B，再从 $B^2$ 项中选 B”简化展示。
在同一长度、同一加法分数规则下，这是安全的局部候选筛选：

如果某个 child 连自己父节点的 top-B 都排不进，就至少有 B 个同父 child 比它高，它也不可能进入全局 top-B。

实现仍通常要先算每条 beam 的整个词表 logits，即 $B\times|\mathcal V|$。
$B^2$ 是后续候选数量，不是神经模型只计算了 $B^2$ 个 logit。
对 group penalty、不同长度比较或特殊 EOS 规则，该筛选条件需重新确认。

## 4. EOS、完成候选与停止准则

课堂中 “she meows EOS” 已完成，而其他 beam 还在继续。
EOS hypothesis 不再执行普通内容扩展，但仍可能参与最后候选比较。

可以有不同实现契约：

- completed hypotheses 与 active hypotheses 共用固定宽度；
- completed hypotheses 放入单独集合；
- 通过保持 EOS 状态让序列在算法层“延长”但不改变概率。

这些设计会影响 beam width 的实际含义和何时结束。
笔记或代码必须说明使用哪一种，不能把所有库的行为当作同一个算法。

Raw logprob 单调不增时，一个 active prefix 的分数是其所有完整后代的上界。
若最佳 completed score 已不小于所有 active prefix 的上界，就可以为 top-1 安全停止。

但是加入长度归一化后，这个简单条件不再自动成立。
平均 logprob 可能随着追加高概率 token 而增加。

## 5. 长度归一化：修正目标，而不只是数值操作

### 5.1 为什么短输出常占优势

每个 token 的 logprob 不大于 0，累加更多项通常使总分更负。
这不说明任何短句都优于任何长句，但使 EOS 的相对校准十分重要。

课堂将原概率转换为 logprob，是为避免长序列连乘下溢。
取 log 保持概率排序；除以长度则会改变排序，二者作用不同。

### 5.2 课件列出的四类调整

课件 18 页：

$$
s_0(y)=\log P(y\mid x).
$$

$$
s_1(y)=\frac{\log P(y\mid x)}{|y|}.
$$

$$
s_\alpha(y)=\frac{\log P(y\mid x)}{|y|^\alpha}.
$$

$$
s_r(y)=\log P(y\mid x)+r|y|.
$$

$$
s_{r,\ell}(y)=\log P(y\mid x)+r\min(|y|,\ell(x)).
$$

$\alpha=0$ 是不归一化，$\alpha=1$ 是平均 logprob。
因分子为负，较大的正 $\alpha$ 通常给较长输出更强相对补偿，不能把其方向反着理解。

最后一个公式把长度奖励限制到期望长度 $\ell(x)$，防止无止境奖励变长。
机器翻译可利用源句长度估计目标长度；开放对话则更难指定这种先验。

### 5.3 数值例子与停止条件陷阱

推导补充：长度 2、总 logprob 为 -2 的候选平均分为 -1。
若追加一个 logprob 为 -0.1 的 token，平均分变为：

$$
\frac{-2.1}{3}=-0.7.
$$

Raw score 变差，归一化 score 却变好。
因此不能在归一化后继续使用 raw-score 的单调上界停止证明。

### 5.4 是否有完全长度中立的 alpha

[20:21](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=1221s) 的课堂问答指出，“让所有长度公平”首先需要定义公平目标。
可以指定长度先验，但这属于用户或任务选择。
在所有正整数长度上不存在可归一化的等概率分布，不能把“1 到无穷全部同等概率”作为默认方案。

## 6. 为什么增加 beam 不一定增加多样性

课件 20 页展示同一张火车图片的多个 captions。
不同 beam 可能仅改变介词或短语，表达同一内容。

高概率区域内部的微小表面变化会占据候选名额。
增加宽度可以搜得更广一些，但并不显式奖励语义覆盖。

这会影响多候选用途：

- 人类需要多个不同草稿；
- reranker 需要覆盖不同解法；
- 搜索系统希望避免所有预算集中在同一错误方向。

## 7. Diverse beam search：组内搜索，组间施加差异奖励

令总 beam 数为 $B$，组数为 $G$，每组宽度为 $B'=B/G$。
课堂在 [36:14](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=2174s) 澄清：$G$ 是组数，不是每组大小。

第 1 组按普通 beam 生成；第 $g$ 组参考更早各组的选择，调整分数：

$$
s_g(y\circ v)=s_{\mathrm{LM}}(y\circ v)-\lambda D_g(y\circ v).
$$

其中 $D_g$ 是与先前组的相似程度，$\lambda\ge0$ 控制惩罚强度。
也可以用正的 diversity reward 写成加号，关键是明确度量方向。

在课堂例子中，前一组已经偏好 “meows”“starts”，下一组就可能更偏向 “yells”“is”。
这仍保留语言模型似然的影响，不是强制所有词都不能重复。

### 7.1 参数极限

| 设置 | 对应算法 |
| --- | --- |
| $G=1$ | 普通 beam search |
| $G=B$ | 每组宽度 1，执行带跨组惩罚的 greedy |
| $1<G<B$ | 每组进行 beam，跨组鼓励差异 |

课堂报告原论文中很小的组宽可有良好效果，但这不是适用于所有模型和任务的最优参数定理。
过大的 $\lambda$ 可能压过语言流畅性，让系统为了不同而生成不合理文本。

### 7.2 为什么不在组内互相惩罚

组内同时选择候选；若每个选择都依赖其他尚未确定的选择，会增加耦合。
分组将依赖限定为“已定的先前组”，使算法易于执行。
组内相似性依然可能存在，所以 $G=B$ 是一种极端的差异控制。

### 7.3 错位流水线

课件 28 页用时间错位展示：第 1 组在时刻 6，第 2 组在时刻 5，第 3 组在时刻 4。
后一组使用前面组已经生成的对应位置作参考。

理想依赖调度下，长度 $T$、组数 $G$ 的流水线有 $T+G-1$ 个时隙。
这表示依赖深度，不表示总 FLOPs 与单条 greedy 相同。
实际设备上吞吐还取决于并行组数、batch packing 和各组结束时间。

## 8. Diversity 的四种定义与命名注意

课件 31 页列出 Hamming、cumulative、n-gram 和 embedding diversity。
其共同目标是惩罚候选与之前组太相似。

1. **位置相关 token 差异**：当前 token 若被先前组在相应位置选过，施加计数惩罚。
2. **累计历史差异**：综合此前各位置的相同/不同程度，允许已经充分分化的路径适当重复普通词。
3. **N-gram 差异**：惩罚已经出现的相同短语，能捕捉比单词更长的结构。
4. **Embedding 差异**：用词向量距离把同义词等软相似也纳入惩罚。

> [!warning] 对照原论文的术语补充
> 课件对 Hamming 与 cumulative 的一句话解释容易让人误以为二者仅是“是否同一时刻”的互换。原论文第 5.1 节中 cumulative 的核心是根据历史累计差异调整惩罚；实现时应读具体公式，而非只按名称写跨全部历史的词频惩罚。

N-gram 匹配与当前时间对齐是两种不同选择。
Embedding 也会增加计算开销，并不必然提高最终任务收益。
课堂指出原论文中简单度量已经有效，因此复杂度量必须用实际收益证明价值。

## 9. Stochastic beam search：为什么需要无放回采样

[37:22](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=2242s)，课件 32–34 页。

若希望候选随机但不重复，直接独立采 $B$ 次可能浪费多个 beam 在同一输出上。

两种直观做法各有代价：

- 每采一个就删除并重新归一化：连续进行 $B$ 次，存在串行依赖；
- 一次多采很多，再删除重复：尖锐分布下可能一直得到同一个结果，数量和耗时不可预测。

Gumbel-Top-k 提供一次生成随机扰动后选 top-k 的方法。
但要注意，单步 token 无放回与完整序列无放回是两个不同层次的问题。

## 10. Gumbel-Max：把 categorical sampling 写成带噪 argmax

标准 Gumbel 的 CDF 为：

$$
F(g)=\exp(-\exp(-g)).
$$

可从 uniform 变量构造：

$$
U\sim\operatorname{Uniform}(0,1),\qquad G=-\log(-\log U).
$$

对每个候选独立采 $G_i$，计算：

$$
I=\arg\max_i(z_i+G_i).
$$

则：

$$
P(I=i)=\frac{e^{z_i}}{\sum_je^{z_j}}.
$$

这不是“近似模拟”softmax sampling，而是在理想连续随机数条件下的分布等价。
Gumbel 的 location 参数是 mode，不是均值；scale 也必须对应所使用的公式。

### 10.1 一个简短推导

以下为对课堂作业动机的推导补充。
令 $X_i=z_i+G_i$，其 CDF 为 $F_i(a)=\exp(-e^{z_i-a})$。

$$
P(I=i)=\int_{-\infty}^{\infty}f_i(a)\prod_{j\ne i}F_j(a)\,da.
$$

将密度和 CDF 代入，令 $S=\sum_j e^{z_j}$：

$$
P(I=i)=\int_{-\infty}^{\infty}e^{z_i-a}\exp(-Se^{-a})\,da
=\frac{e^{z_i}}{S}.
$$

### 10.2 Temperature 放在哪里

课堂 [57:06](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=3426s) 经讨论确认可先缩放 logits：

$$
\arg\max_i\left(\frac{z_i}{T}+G_i\right).
$$

等价地可选 $\arg\max_i(z_i+TG_i)$。
不能把加完噪声的整个向量再除以 $T$ 来改变采样分布，因为正数缩放不改变 argmax。

## 11. Gumbel-Top-k：同一次扰动产生无放回有序样本

对 $z_i+G_i$ 取从大到小的前 $k$ 项，就得到按权重的无放回样本序列。

若原始权重为 $p_i$，有序样本 $(i_1,\ldots,i_k)$ 的概率为：

$$
P(i_1,\ldots,i_k)
=\prod_{j=1}^{k}\frac{p_{i_j}}{1-\sum_{r<j}p_{i_r}}.
$$

这是“每次从剩余项按原权重重新归一化”的联合分布。
各个样本并不独立；第 2 项也不是再次服从未修改的原分布。

```python
u = uniform_open_interval(shape=logits.shape)
g = -log(-log(u))
indices = topk(logits + g, k).indices
```

实现必须避免 $U=0$ 或 $U=1$ 的数值端点，并保证各候选噪声独立。

## 12. 从单步技巧到完整序列：条件 Gumbel 变换

如果能给所有完整序列独立加 Gumbel 再取 top-k，就解决了序列无放回采样。
问题是完整序列空间指数大，不能枚举。

原论文利用“子树最大 Gumbel”结构，从根到叶隐式构造相同分布。
每个搜索节点保留：

- 普通累计 logprob $\phi$；
- 随机优先级 $\widetilde G$；
- 对应 prefix 及其模型状态。

对父节点 $s$ 的每个 child $i$：

$$
\phi_i=\phi_s+\log p(i\mid s),\qquad G_i\sim\operatorname{Gumbel}(\phi_i,1).
$$

令 $Z=\max_iG_i$，父节点已经确定的随机优先级为 $T$，变换为：

$$
\widetilde G_i
=-\log\left(e^{-T}-e^{-Z}+e^{-G_i}\right).
$$

课件 41 页算法 1 第 14 行正是这一式子。
因为 $G_i\le Z$，所以 $\widetilde G_i\le T$；最大 child 的值恰好为 $T$。

这让子树的随机最大值与父节点保持一致，同时维持正确的条件分布。
普通 clipping 会把多个 child 压成同一个数，改变排序与分布，因此不等价。

随后跨所有扩展取 $\widetilde G$ 最大的 B 项，继续扩展到完整序列。
这不是先随机扩展、再按未经扰动的 logprob 全部重新排序；后者会丢掉已有的随机选择信息。

> [!warning] 这里的 score 不全是 log probability
> $\widetilde G$ 是随机优先级，不是归一化 logprob。父子单调性质是该随机过程的一部分，不能把其数值直接报告成模型置信度。

### 12.1 精确性保证的适用范围

标准算法提供目标序列分布的无放回采样；不是“找到模型 MAP”的保证。
如果采用温度或截断，应先明确修改后的目标分布，再对那个分布正确执行算法。

原论文实验明确关闭 length normalization 和不兼容的 early stopping，以保留理论正确性。
把普通 beam 的长度归一化直接塞进随机优先级后，通常不再拥有同一个精确采样结论。

## 13. 怎样读论文中的质量–多样性图

[50:13](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=3013s)，课件 42 页。
横轴是生成候选集的 diversity，纵轴是 BLEU；不同线分别展示候选中的最大值、均值和最小值。

需要区分：

- 最高 BLEU 候选是基于参考答案事后选出的 oracle-best；
- 模型实际返回的 top-1 未必就是这个候选；
- 平均质量高不代表候选覆盖了很多不同内容；
- “在同质量下更有多样性”与“在同多样性下质量更高”是不同的切面。

原论文用 temperature 控制普通 sampling 与 stochastic beam 的多样性，用 diversity strength 调 diverse beam。
课堂对横轴参数有口头不确定，本文以原论文实验部分为准。

这些实验来自当时的机器翻译模型，不能直接推广成“2025/2026 所有 LLM 上随机 beam 恒优”。

## 14. Curse of beam search：搜索更好，输出反而更差

[1:01:29](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=3689s)，课件 43–47 页。

课堂报告一些早期实验中，增加 beam width 会降低 downstream 指标。
两种解释并不互斥：

1. 搜索暴露长度偏好，找到更短但不充分的输出；
2. 模型最高概率区域本身偏离人类期待，较小 beam 的搜索偏差反而有帮助。

第一种可通过适当长度打分缓解；第二种说明需要重新审视目标，而不只是继续加大搜索。

### 14.1 语义质量可能分散在多个表述上

假设 “cat sat down” 的单条概率最高，而“cat ran away”的三个同义表述各略低。
后三者总质量可能更高，单条 MAP 却看不到语义聚合。

这与前一讲硬币的“点 versus 集合”联系起来，并为后面的 self-consistency、MBR 等方法铺垫。

### 14.2 Likelihood trap

课件 47 页引用人类偏好与 logprob 的关系：靠近最高概率区域的文本可能比绝对最高项更受欢迎。
因此“模型概率更高”不是“更优人类评价”的充分条件。

即使模型很好地拟合真实文本分布，概率 mode 也未必最大化用户效用。
要使两者一致，还需要概率目标与效用目标的额外对齐假设。

### 14.3 Blessing of beam search

课件 48 页介绍“较小 beam 偶然施加较均匀的信息密度”的解释。
一个统计量是 token surprisal 的标准差：

$$
\sigma_I=\sqrt{\frac1T\sum_t(I_t-\overline I)^2},\qquad I_t=-\log p(y_t\mid y_{<t}).
$$

较平稳的信息释放可能对应更自然的语言。
这是特定研究对 search bias 的解释，不是“小 beam 总是最好”的普遍结论。

## 15. 课堂问答

**Q：为什么不用更大的 beam 直接解决？** — [1:01:29](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=3689s)

宽度增加算力、显存和搜索范围，但不会修复模型目标错配，也不显式保证语义多样性。
有限宽度搜索的候选集还可能随 B 改变，不能假设所有实现里每次增宽都单调提高最终分数。

**Q：DBS 里相同词还可能出现吗？** — [34:18](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=2058s)

会。惩罚是软分数，如果语言模型优势足够大，仍可能选择同一个词。
具体初始化是否强制不同首 token 还取决于实现契约。

**Q：Gumbel-Max 是否必须显式计算 softmax？** — [44:31](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=2671s)

单次 categorical sampling 不必，logits 加标准 Gumbel 后 argmax 即可。
但完整序列的累计概率比较要使用条件 logprob，不能跨不同父节点直接相加未归一化 raw logits。

**Q：随机 beam 能叠加 top-p 吗？** — [53:04](https://www.youtube.com/watch?v=2hhyfPYGCmY&t=3184s)

可以先定义截断后的条件分布，但这改变了目标序列分布，且要保留足够支持集容纳 B 条不同完整输出。
精确性应相对于新目标讨论。

## 16. AI Infra：beam 的资源账本

本节为工程补充。

### 16.1 候选状态远比一个分数贵

每条活跃 beam 需要 token history、累计分数、结束状态，以及 Transformer KV 状态。
分支共享 prefix 可以通过共享 KV block 减少复制，但分歧后的新 token 会形成独立缓存。

粗略而言，decode 计算随活跃 beam 数增加；完整成本还包含词表打分、top-k、状态重排和 cache 索引维护。
不能只看 Python 中 beam list 的长度估算显存。

### 16.2 Prune 必须同步重排 KV cache

全局 top-B 可能从同一父节点挑出多个 child。
因此要同时更新 token、parent index、position、score 与 KV 引用。
若只重排 token、不重排 KV，模型将继续在错误历史上解码，结果可能仍像语言但算法已经错了。

### 16.3 Streaming 的选择承诺

Beam 中当前最优 prefix 未来可能被其他 beam 替代。
如果每步都把它直接发给用户，后续需要回滚文本。
可以只提交所有保留 beam 的共同前缀，或等候更稳定的边界，但会影响首字及持续输出延迟。

### 16.4 把时间、吞吐与质量一起报告

并行 B 条可能改善总吞吐，却增加单请求显存；串行则减少同时占用但增加延迟。
DBS 的跨组依赖和 SBS 的随机排序也有调度成本。
应报告总生成 token、模型 forward 数、峰值 KV、latency 和候选质量，而非只比较 beam width。

## 17. 本讲最容易混淆的概念

| 混淆 | 正确区分 |
| --- | --- |
| Greedy / MAP | 单步相同，完整序列一般不同 |
| Log transform / length normalization | 前者保持概率排序，后者改变目标 |
| Beam width / 词表大小 | 宽度限制前缀数；词表仅决定每个前缀的分支数 |
| DBS / SBS | DBS 修改差异目标；SBS 精确版本做序列无放回采样 |
| 唯一 token / 唯一完整序列 | 局部不重复不等于全序列采样语义正确 |
| Gumbel priority / logprob | 随机搜索分数不是置信度 |
| Cap / 条件变换 | 普通 clipping 破坏 Gumbel 分布 |
| Oracle-best / 可部署 top-1 | 看参考答案挑最佳不同于模型自己选最佳 |
| 更准搜索 / 更高任务质量 | 还取决于目标与任务效用的关系 |

## 18. 自测问题与面试回答

1. 给出 greedy 找不到序列 MAP 的最小反例。

    **面试回答：** 第一步 A/B 概率为 0.6/0.4，A 的最佳续写概率为 0.1，B 的为 0.9。Greedy 选 A，完整概率 0.06；B 路径是 0.36。未来条件分布依赖当前决策，所以逐步 argmax 不能交换成全序列 argmax。

2. 为什么每个父节点只取 top-B child 通常足够选全局 top-B？

    **面试回答：** 在统一长度和同一分数函数下，一个 child 若在自己父节点下排名超过 B，至少已有 B 个兄弟比它高，所以它不可能进全局前 B。这只减少候选筛选量，模型往往仍计算完整词表 logits；特殊 EOS、分组或长度规则需要单独检查。

3. 长度归一化为什么会破坏 raw logprob 的停止条件？

    **面试回答：** Raw logprob 追加 token 只会下降，但平均 logprob 可因追加高概率 token 上升，例如 -2/2 变成 -2.1/3。因此 active prefix 的当前归一化分数不再是后代上界，需要专门的 bound 或保守终止规则。

4. DBS 的 B、G 和 B/G 分别是什么？

    **面试回答：** B 是总 beam 数，G 是组数，B/G 是每组宽度。组内进行普通有限宽度搜索，后续组按前面组的选择加入差异惩罚。G=1 退化为普通 beam，G=B 为跨组互相影响的多个 greedy 路径。

5. 为什么需要无放回采样？

    **面试回答：** 独立采样在尖锐分布下可能反复得到同一候选，浪费固定候选预算。无放回采样保留随机选择但排除已选项，提高不同候选覆盖；样本之间依赖，所以不能再按独立同分布样本直接解释统计量。

6. 写出 Gumbel-Max 以及带温度的形式。

    **面试回答：** 独立采 $G_i=-\log(-\log U_i)$，则 $\arg\max_i(z_i+G_i)$ 服从 softmax(z)。温度 T 对应 $\arg\max_i(z_i/T+G_i)$；只把整个加噪向量除以 T 不改变 argmax，也就没有改变采样分布。

7. SBS 为什么不能只在每一步加独立噪声然后 ordinary beam prune？

    **面试回答：** 完整序列无放回采样要求不同层的随机子树最大值保持一致。SBS 保留父节点扰动值，并对孩子做特定条件 Gumbel 变换，再按扰动分数筛选。逐步独立噪声或按原始 logprob 重排通常对应另一种分布。

8. 课件说把 child score cap 到 parent，代码能否用 min？

    **面试回答：** 不能。正确变换为 $-\log(e^{-T}-e^{-Z}+e^{-G_i})$，它保持条件 Gumbel 结构、最大 child 等于 T。普通 min 会把多个候选压成同一值，改变相对分布和无放回采样保证。

9. Curse of beam search 揭示哪两类问题？

    **面试回答：** 一类是长度偏好，宽 beam 更容易找到过短序列；另一类是目标错配，模型概率最高的文本未必最符合任务或人类偏好。更强搜索只能优化已有目标，需要长度打分、奖励或其他决策规则处理目标问题。

10. 模型 KV cache 与 beam parent indices 为什么必须一起更新？

    **面试回答：** 每轮全局筛选可复制同一父节点并丢弃另一些父节点，token 行顺序会改变。KV 必须跟随对应历史，否则下一步概率是在别的 prefix 上算的。正确实现要同时重排或共享缓存引用，并维护各分支位置与长度。

11. SBS 图中的最高 BLEU 为什么不能直接报告为系统 top-1？

    **面试回答：** 最高 BLEU 通常是拿参考答案对候选事后评分选出的 oracle-best，部署时并不知道参考答案。它说明候选集覆盖了好答案，但还需要实际 selector 或 verifier 才能把覆盖转化为可实现的 top-1 质量。

12. 为什么 B 等于词表大小也不是精确解码？

    **面试回答：** 长度一有 V 个前缀，长度二已有 V² 个，长度 t 有 V^t 个。固定 B=V 仍会在后面各层丢弃大量前缀；除非能严格合并未来等价状态或用正确 bound 剪枝，否则不能获得全空间最优保证。

## 19. 关联内容

- [课程索引](CMU%2011-763.md)
- [上一讲：Common Sampling Methods for Modern NLP](CMU%2011-763%20-%20Lecture%2003%20-%20Common%20Sampling%20Methods%20for%20Modern%20NLP.md)
- [下一讲：A Star and Best First Search](CMU%2011-763%20-%20Lecture%2005%20-%20A%20Star%20and%20Best%20First%20Search.md)
- [LLM Inference](../../topics/inference/LLM%20Inference.md)
