---
type: course-note
status: developing
course: "[[CMU 11-763]]"
lecture: 12
lecture_date: 2025-10-02
area: inference
topics:
  - "[[LLM Inference]]"
aliases:
  - CMU 11-763 Lecture 12
  - Reward Models and Best-of-N
  - 奖励模型与多候选重排序
video_url: https://www.youtube.com/watch?v=p-MWR625HB8
---

# Lecture 12：Reward Models and Best-of-N

> [!abstract] 本讲一句话
> Best-of-N 用生成模型提出候选、用奖励模型定义“更好”，把额外推理计算变成输出选择；它的上限同时受候选覆盖和评分可靠性限制，多采样不自动等于更接近真实人类偏好。

## 来源与范围

- 课程：CMU 11-664/763 — Inference Algorithms for Language Modeling，Fall 2025。
- 讲师：Amanda Bertsch；日期：2025-10-02。
- [课程视频](https://www.youtube.com/watch?v=p-MWR625HB8)，约 53:01；已完整读取英文自动字幕。
- [官方课件，共 36 页](https://docs.google.com/presentation/d/1PkZJ_nhn4vVL5--gBzltY3vBugYbZa3fMHGJ8Srp31M/edit?usp=sharing)，已读取完整文字，并从 PDF 核对关键公式和图表。
- [课程主页](https://www.phontron.com/class/lminference-fall2025/)。
- 理论核对：[Beirami et al., Theoretical guarantees on the best-of-n alignment policy](https://proceedings.mlr.press/v267/beirami25a.html)。
- 评价核对：[RewardBench 2](https://arxiv.org/abs/2506.01937)。
- 课程日程列有 Monte Carlo Tree Search，但这段公开视频和本份 36 页课件没有展开 MCTS；本文不把额外树搜索教程冒充本讲内容。

> [!warning] 一个需要纠正的课件表述
> 第 16 页左侧写“BoN 与 target distribution 的 KL”，但该页图的纵轴和引用原论文实际讨论 $D_{\mathrm{KL}}(\pi_{\mathrm{BoN}}\|\pi_{\mathrm{ref}})$，即相对**参考生成策略**的偏移。$\log N-(N-1)/N$ 不是 BoN 到理想人类偏好分布的距离保证。本文按原论文纠正，并把“偏好分布”保留为动机层面的直觉。

## 视频时间索引与课件对应

| 时间 | 内容 | Slides |
| --- | --- | --- |
| [00:00](https://www.youtube.com/watch?v=p-MWR625HB8&t=0s) | 作业、AI 使用与反馈安排 | 课前说明 |
| [01:09](https://www.youtube.com/watch?v=p-MWR625HB8&t=69s) | 为什么一整讲讨论 BoN | 1–2 |
| [01:59](https://www.youtube.com/watch?v=p-MWR625HB8&t=119s) | Rejection sampling 的问题设定 | 3–9 |
| [05:35](https://www.youtube.com/watch?v=p-MWR625HB8&t=335s) | Support、包络常数与接受概率 | 10 |
| [07:42](https://www.youtube.com/watch?v=p-MWR625HB8&t=462s) | 提议分布越好，拒绝越少 | 11 |
| [09:19](https://www.youtube.com/watch?v=p-MWR625HB8&t=559s) | 从“按分布采样”转向“选一个好答案” | 12–13 |
| [13:39](https://www.youtube.com/watch?v=p-MWR625HB8&t=819s) | Best-of-N 算法 | 14 |
| [14:32](https://www.youtube.com/watch?v=p-MWR625HB8&t=872s) | Inference-time alignment 与 KL | 15–16 |
| [18:15](https://www.youtube.com/watch?v=p-MWR625HB8&t=1095s) | Bradley–Terry 与 scalar RM | 17–18 |
| [20:51](https://www.youtube.com/watch?v=p-MWR625HB8&t=1251s) | Generative reward model | 19–21 |
| [24:44](https://www.youtube.com/watch?v=p-MWR625HB8&t=1484s) | 偏好是否应当传递 | 21 的问答 |
| [26:40](https://www.youtube.com/watch?v=p-MWR625HB8&t=1600s) | 课堂偏好投票演示 | 22–25 |
| [31:49](https://www.youtube.com/watch?v=p-MWR625HB8&t=1909s) | 身份、风格、帮助与危害 | 26–27 |
| [35:02](https://www.youtube.com/watch?v=p-MWR625HB8&t=2102s) | 人类偏差与长度偏置 | 28–29 |
| [37:11](https://www.youtube.com/watch?v=p-MWR625HB8&t=2231s) | 完整序列 RM 不一定适用于前缀 | 30 |
| [38:38](https://www.youtube.com/watch?v=p-MWR625HB8&t=2318s) | Ties：不总有更好的那一个 | 31 |
| [39:31](https://www.youtube.com/watch?v=p-MWR625HB8&t=2371s) | RewardBench 2 | 32 |
| [41:46](https://www.youtube.com/watch?v=p-MWR625HB8&t=2506s) | 谁提供 preference data | 33–34 |
| [44:11](https://www.youtube.com/watch?v=p-MWR625HB8&t=2651s) | 选择 N 与降低成本 | 35 |
| [48:02](https://www.youtube.com/watch?v=p-MWR625HB8&t=2882s) | 本讲结论 | 36 |
| [48:53](https://www.youtube.com/watch?v=p-MWR625HB8&t=2933s) | 同源模型偏差、评估为何可能比生成容易 | 课后问答 |

## 1. 为什么“采 N 个取最好”值得单独一讲

算法只有三步，但每一步隐含一个问题：

1. 从什么分布采样，才能有机会得到真正好的候选？
2. 用什么指标排序，它是否对应用户希望的质量？
3. 花多少计算才值得，评分误差是否会随搜索加强而放大？

本讲用 rejection sampling 建立概率直觉，再讨论 reward model，最后回到计算预算。
它把前面的多采样方法和后面 inference efficiency 的主题连接起来。

## 2. 标准 rejection sampling 的目标

### 2.1 想采样的分布与能采样的分布不同

记目标分布为 $D(x)$，容易采样的 proposal 为 $P(x)$。
我们能评价某一点的目标密度或概率，却不一定能直接从 $D$ 采样。
于是先从 $P$ 抽样，再按一定概率接受或拒绝。

课件 4–9 页用几何图示说明：proposal 覆盖目标分布所在区域，但抽到的点必须筛选，才会具有目标分布的比例关系。
核心不是简单地删除“看起来差”的样本，而是使用正确的接受概率。

### 2.2 Support 条件

$$
D(x)>0\Rightarrow P(x)>0.
$$

目标可能出现的结果必须也能被 proposal 产生。
如果一个答案在 proposal 中概率为零，拒绝采样无法凭空创造它。
这也是后面 BoN 的候选覆盖限制。

### 2.3 包络常数与接受概率

选取 $c$，满足所有 $x$ 上：

$$
D(x)\le cP(x).
$$

采样 $x\sim P$，再以概率

$$
a(x)=\frac{D(x)}{cP(x)}
$$

接受它；等价地，采 $u\sim\mathrm{Uniform}(0,1)$，检查 $u\le a(x)$。
包络条件保证 $a(x)\in[0,1]$。
不能随意选一个太小的 $c$，然后把大于 1 的比例截断还宣称仍保持原算法的精确分布。

### 2.4 为什么最终分布是目标分布

以下是对课件第 10 页公式的补充推导。
若 $D$、$P$ 均归一化，则：

$$
P(x,\mathrm{accept})=P(x)\frac{D(x)}{cP(x)}=\frac{D(x)}c.
$$

对所有点求和或积分：

$$
P(\mathrm{accept})=\frac1c.
$$

因此：

$$
P(x\mid\mathrm{accept})
=\frac{D(x)/c}{1/c}=D(x).
$$

平均每个接受样本需要 $c$ 次 proposal。
课件“proposal 越接近 target，拒绝越少”的直觉，更精确地与密度比的最大值、能否找到紧包络有关，而不仅是某种平均距离小。

## 3. Best-of-N 改变了问题目标

### 3.1 从恢复分布到选择高质量点

课堂在 [09:19](https://www.youtube.com/watch?v=p-MWR625HB8&t=559s) 转向新目标：不是获得代表整个目标分布的样本集合，而是找到一个相对好的输出。
于是固定预算 $N$，无论候选整体好坏，都从这 $N$ 个里选最好者。

$$
y_1,\ldots,y_N\overset{\mathrm{iid}}{\sim}P_\theta(\cdot\mid x),
$$

$$
y^*=\arg\max_{i=1,\ldots,N}r_\phi(x,y_i).
$$

这里 $r_\phi$ 是评分函数，不要求它是归一化概率。
固定 N 避免“直到满足很难的约束才停止”导致的无限等待风险。
代价是：如果所有候选都不好，算法仍会返回其中相对不坏的一个。

### 3.2 不要把两种算法混成同一统计保证

标准 rejection sampling 使用 $D/(cP)$，目标是精确采样。
BoN 使用 `argmax reward`，目标是更高分的输出。
课堂把 BoN 称为 rejection sampling 的变体，是机制类比；不意味着 BoN 的输出严格服从某个人类偏好分布。

尤其不能把 $r_\phi(x,y)$ 直接当成 $D(y\mid x)$。
一个 reward 通常只是实数效用评分；即使取指数，还需要定义基准分布、温度与归一化，才得到概率分布。

### 3.3 与模型自身概率排序的区别

课件强调使用原模型 log probability 以外的指标。
如果采样后只按原模型概率选最好者，仍在做近似 mode seeking。
reward model 的价值在于改变偏好方向，例如事实性、指令遵循或风格，而非只强化模型原本最可能的文本。

## 4. 候选覆盖与选择质量是两个瓶颈

假设单次采样产生“可接受答案”的概率是 $p$，样本独立且判定器完美。
至少存在一个可接受候选的概率为：

$$
P(\exists\text{ acceptable candidate})=1-(1-p)^N.
$$

例如 $p=0.2$ 时，$N=1$ 为 0.2，$N=4$ 为 0.5904，$N=10$ 约为 0.8926。
这是本文构造的概率例子，用来说明多采样怎样改善覆盖。

但 BoN 最终成功率还受 selector 限制：

$$
P(\text{选中正确答案})
=P(\text{集合中有正确答案})
\,P(\text{选对}\mid\text{集合中有正确答案}).
$$

样本更多可以提高第一项，却可能让第二项更困难。
候选中出现更能骗过 reward model 的错误文本时，最高 proxy reward 不等于最高真实质量。
因此 oracle/pass@N 与实际 selected accuracy 应分开测量。

## 5. BoN 的 KL 理论到底衡量什么

### 5.1 正确的比较对象

课件引用 Beirami et al.，其对象是：

$$
D_{\rm KL}\bigl(\pi_{\rm BoN}^{(N)}(\cdot\mid x)
\,\|\,\pi_{\rm ref}(\cdot\mid x)\bigr).
$$

$\pi_{\rm ref}$ 是产生候选的参考策略，不是我们无法直接知道的理想偏好分布。
它衡量的是：通过选最好候选，输出分布偏离原生成分布多少。

在论文设定下，常见表达式

$$
\log N-\frac{N-1}{N}
$$

是上界，不能对所有离散语言输出都当作精确 KL。
[原论文摘要](https://proceedings.mlr.press/v267/beirami25a.html)明确指出原先当作等式的说法并不普遍成立。

### 5.2 第 16 页图应该怎样读

蓝色虚线是常见上界，随 N 增大继续增长。
黑点是例子中的真实 KL，红线是更紧的 proposed estimator，图中两者很快趋于平台。
纵轴明确写的是 BoN 相对 reference policy 的 KL。

图证明的是“粗上界可能很松”。
它没有证明“每增加一个候选，人类偏好差距就固定缩小”，也不是实际任务质量的通用 scaling law。
红线在图例中是 estimator，不应将其简单改称为一条普适精确等式。

### 5.3 一个可以手算的离散例子

以下为本文构造，不是课件实验。
只有两个输出：$A$ 概率 0.8、reward 0；$B$ 概率 0.2、reward 1。
BoN 只有在所有样本都是 A 时才返回 A，因此：

$$
\pi_N(A)=0.8^N,\qquad
\pi_N(B)=1-0.8^N.
$$

当 N 趋于无穷时，分布集中于 B：

$$
\lim_{N\to\infty}D_{\rm KL}(\pi_N\|\pi_{\rm ref})
=\log\frac1{0.2}=\log5.
$$

真实 KL 有有限极限，常见上界却约为 $\log N-1$ 持续增长。
这个例子展示为什么离散空间上不能盲用上界作为等式。
它也说明 reward-driven mode selection 与“覆盖完整偏好分布”不是同一个任务。

## 6. Scalar reward model 与 Bradley–Terry

### 6.1 Reward score 如何转成两两偏好

课件第 17 页先给出 Bradley–Terry 形式，再写成 reward 的指数比例：

$$
P(y_1\succ y_2\mid x)
=\frac{\exp r_\phi(x,y_1)}
{\exp r_\phi(x,y_1)+\exp r_\phi(x,y_2)}.
$$

等价于：

$$
P(y_1\succ y_2\mid x)
=\sigma\bigl(r_\phi(x,y_1)-r_\phi(x,y_2)\bigr).
$$

reward 相同，偏好概率为 0.5；reward 差为 $\log3$，偏好概率为 0.75。
两者的差值决定相对偏好，不是每个 reward 本身就代表“正确率”。

### 6.2 Pairwise 训练目标

训练数据是 prompt、chosen、rejected 三元组。
模型常由强 LM 初始化，将语言建模输出头改成序列打分头。
标准 pairwise loss 为：

$$
\mathcal L_{\rm BT}
=-\mathbb E_{(x,y_w,y_l)}
\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right).
$$

这是对课件训练描述的公式化，不是把 chosen/rejected 各自独立判成绝对好坏。
一个被 rejected 的答案可能也不错，只是同一对里没有另一个好。

### 6.3 Reward 的可识别性

若给同一 prompt 的所有 reward 加上常数 $b(x)$，两两差值不变。
因此 preference data 不自动确定 reward 的绝对零点。
不同 RM 的分数范围也不能未经校准直接比较。

BoN 只需要同一候选集合内的排序，因此对严格单调变换不敏感。
RL 或概率采样会使用分数大小，reward scale 在那些场景中就很重要。
这是从课堂 scalar reward 定义推导的使用边界。

## 7. Generative reward model：直接让 LM 当评审

### 7.1 输入和输出可以很灵活

可以给模型两份答案，让它选择；也可以给一个候选问是否满足 rubric，或给一组候选要求排名。
不一定需要重新训练专用打分头，也可以使用 in-context examples、SFT 或 RL 提高判断能力。
课堂还提到让模型解释判断理由，有时能改善评价效果。

当评价标准变化时，修改 rubric 或 prompt 比重新训练一个 scalar RM 更方便。
例如答案要求、风格或时效性发生变化，可将新的任务条件写入评审输入。

### 7.2 它与 LLM-as-a-judge 的关系

课堂回答：两者高度重叠。
“Reward model”强调输出作为选择或优化信号；“LLM-as-a-judge”强调用 LM 执行评价任务。
同一个 generative judge 可以在 benchmark 中评分，也可以成为 BoN 的 selector。

## 8. 生成式评审的不一致与非传递

### 8.1 顺序、模板和版本可能影响结果

同一对候选，改一下 prompt，交换 A/B 顺序，或服务端模型更新，都可能改变判断。
课堂指出，相比 temperature=0 下的偶发非确定性，格式与顺序敏感往往更值得留意。

对需要稳定排名的任务，应固定模型版本与 rubric，并测试交换候选顺序后的结果。
这是实验控制，不是说某个模型从来不能可靠评价。

### 8.2 Scalar ranking 与独立 pairwise judging

一个固定 scalar function 满足：若 $r(A)>r(B)$ 且 $r(B)>r(C)$，则 $r(A)>r(C)$。
但三次独立生成式判断可能得到 A>B、B>C、C>A 的循环。
这会使排序结果依赖比较顺序，尤其影响 tournament selector。

这里的“固定 scalar function”指潜在评分本身；根据偏好概率随机抽样出的离散比较结果仍可能偶发不传递。
不能把随机判断与底层 score 的传递性混为一谈。

### 8.3 偏好本来就一定传递吗

学生用石头剪刀布或宝可梦相克指出：任务关系本来可能是循环的。
讲师认可，非传递不总是模型错误。
应先确定评价目标是否适合一个全局 scalar utility，再要求 judge 满足其结构。

## 9. 偏好数据为何容易收集，也为何有噪声

### 9.1 从写 rubric 到二选一

课堂先问“语言模型是怎样工作的，一个好回答应是什么样”，这是开放而难回答的问题。
展示两个答案后，选择更喜欢哪一个通常容易得多。
这正是 pairwise preference data 的实用价值。

课堂用在线竞技场做了一次投票，学生既有分歧，也能根据标题、emoji 等猜测模型风格。
讲师指出：大家没有完整读完长答案，就已经投票了。
这展示了表面风格可能影响标注，而不仅是内容本身。

### 9.2 好答案不只由正确性定义

课件用 `4 * 12` 说明：`48` 与一段完整的分解计算都可能正确。
如果用户要简短答案，前者更适合；如果用户在学习计算，后者可能更有帮助。

因此偏好至少由任务、用户要求和评价人群共同决定。
“更长”或“更会解释”不是所有 prompt 的统一质量尺度。

### 9.3 谁的偏好被模型学到了

课堂列举众包标注者、数据公司承包者、领域专家和真实用户反馈。
标注规范决定回答风格、拒绝边界、身份表达等行为。
用户点击、重试、点赞等隐式信号也可能用于评价，但课堂对具体公司使用哪些信号的部分是推测，本文不将其当作已核实事实。

## 10. 长度偏置：更高 reward 可能只意味着更啰嗦

课件第 29 页展示长度与 reward 的关联。
当标注者倾向把更长的答案理解成更详细、更完整，RM 可能把长度学成便宜的代理特征。
优化该 reward 后，输出可以变长而不变正确。

可以写成一个简单误差模型：

$$
\hat r(x,y)=r_{\rm desired}(x,y)+\alpha\,\mathrm{len}(y)+\epsilon(x,y).
$$

这是本文的解释模型，不是课件拟合方程。
若 $\alpha$ 不必要地偏大，BoN 可能总选冗长候选，RL 也可能主动拉长回答。
可用长度匹配的候选对、明确 rubric 和独立任务指标检查这种问题。

别与序列 log probability 的长度偏置混淆。
概率连乘常偏向短序列；reward model 则可能因偏好数据而偏向长文本。
不能拿同一个长度归一化公式不加验证地处理两者。

## 11. 完整答案 RM 不一定适合评价前缀

课件第 30 页展示一个句子不同前缀的 reward 上下跳动。
一个不完整前缀若被当作完整回答，低分可能合理；但它也可能只是一个正确长答案的中间状态。

因此：

$$
r_{\rm complete}(x,y_{1:t})
\not\equiv
\mathbb E\left[r_{\rm final}(x,Y)\mid Y_{1:t}=y_{1:t}\right].
$$

左边是把前缀直接送给完整序列 RM，右边是“从该前缀继续生成后最终质量”的期望。
两者不是同一个学习目标。

BoN 通常对完整候选打分，较少直接碰到这个错配。
逐 token reward-guided decoding、提前剪枝或过程评估，则需要更合适的训练信号。
Process RM 与 future discriminator 正是为类似对象定义评分的方向，不能随便把 outcome RM 复用过去。

## 12. Ties 与 RewardBench 2

### 12.1 某些候选没有有意义的优劣

课件问“说出彩虹的一种颜色”，Green 和 Blue 都合法。
如果强迫标注者选出严格胜者，会制造不存在的 preference signal。
理想的 pairwise probability 在这种情况下应接近 0.5，而不是非常自信地偏向一边。

注意 0.5 指比较概率，不是要求两个 raw reward 都等于 0.5。
若二者 raw reward 相同，Bradley–Terry 比较概率就是 0.5。

### 12.2 评价需要覆盖不同技能

课堂介绍 RewardBench 2，强调模型在数学、安全、事实与 ties 等维度上可能很不均衡。
[原论文](https://arxiv.org/abs/2506.01937)将其组织为 factuality、precise instruction following、math、safety、focus、ties 六类。
总榜分数可帮助初筛，最终还应在自己的任务、语言和候选分布上验证。

### 12.3 同模型家族不是万能原则

课堂提到 generator 与 RM 的来源匹配可能重要，并在问答中承认同源模型也可能共享盲点。
原论文对“同 lineage 更适合”的明确强结论主要来自 RLHF/PPO 设置；不能直接推成所有 BoN 必须用同家族 RM。
论文也观察到 RM 对同源模型输出有自偏好，因此要区分真正质量提升与风格偏好。

## 13. Best-of-N 的工程成本

### 13.1 生成通常有更长串行路径

自回归生成需要一个 token 接一个 token 地运行。
完整候选已知后，scalar RM 可以将全部 token 一次送入模型并行处理，再输出序列分数。
因此“小 generator + 较大 scalar RM”可能是有用配置。

但 generative RM 如果要写长解释，也会有自己的串行 decode 成本。
所以“scoring 比 generation 便宜”应按具体评分架构理解，不是所有 judge 都成立。

### 13.2 候选数与 batch 边界

课堂例子：如果一个 batch 容纳 32 条候选，33 条需要第二批，则 N=32 可能更合算。
这个例子不是说 N 必须等于 batch size，而是提醒预算和硬件吞吐不一定连续变化。

可用如下近似：

$$
C_{\rm total}\approx N C_{\rm generate}+N C_{\rm scalar\ score}.
$$

墙钟时间则取决于并发容量、长度差异、队列和评分是否能批处理。
相同总 token 数，不代表相同延迟。

### 13.3 重复候选降低有效收益

温度很低时，采 100 条可能只有很少的不同答案。
重复生成消耗成本，却不明显增加覆盖面。
课堂建议改变 decoding 设置或采多个模型以增加多样性。

去重可减少重复评分，但不会返还已经花掉的生成成本。
更好的候选生成策略应在生成前改善多样性与质量的平衡。

### 13.4 多 proposal 时要重新说明统计设定

固定混合权重可以定义合法的 mixture proposal：

$$
P_{\rm mix}(y\mid x)=\sum_k w_kP_k(y\mid x).
$$

如果每次先独立抽模型 k、再抽 y，就可把样本视为从 mixture 独立采样。
但固定配额或根据已有结果自适应换温度、换模型，不再是同一 proposal 的 iid 样本。
课堂说统计性质“不再清楚”，准确的意思是不能原封不动套用前面的单 proposal iid 推导，而不是混合分布数学上无法定义。

## 14. 为什么评估可能比生成容易

课后问答给出几个原因：

- 完整输出已经可见，评审能检查结尾是否完成、前后是否矛盾。
- 评审可以专注于一个明确 rubric，不必同时创造内容。
- 可以额外训练专用 scorer，或给 judge 更多推理预算。

课堂举了 100-token 上限的例子：两个答案前 90 个 token 都合理，一个能在 100 token 内结束，另一个被截断。
生成时很难提前判断全部未来；生成后完整性更容易检查。

这也解释了为什么 RLHF 与 BoN 可以叠加。
训练改善一般输出分布，BoN 在当前预算和要求下再选一次；但专用 RM 是否足够好，仍必须实测。

## 15. 最小实现与必要的评价分离

以下是本文的 BoN 示意，不是官方 SDK 示例：

```python
def best_of_n(prompt, generator, reward_model, n):
    candidates = generator.sample(prompt, n=n)
    scores = reward_model.score(prompt, candidates)
    selected_index = max(range(len(scores)), key=scores.__getitem__)
    return {
        "answer": candidates[selected_index],
        "candidate_count": len(candidates),
        "selected_index": selected_index,
        "scores": scores,
    }
```

训练/选择用的 proxy reward 与最终报告的 evaluation metric 应尽量分开。
否则用 RM 挑出最高分答案，再用同一个 RM 宣称“质量提高”，很容易得到循环证据。

一个有信息量的实验应同时报告：

- N 与生成参数，独特候选数。
- Oracle/pass@N 或候选覆盖指标。
- Selector 真正选中正确答案的比例。
- 独立评价下的质量、总成本、延迟与输出长度。

## 16. 自测与面试回答

### 1）Rejection sampling 为什么需要 support 覆盖？

**回答：** 被接受的样本一定先来自 proposal，因此 proposal 概率为零的点永远不会出现。如果 target 在这些点有正概率，就无法恢复 target 分布。接受概率再聪明也不能创造 proposal 从不生成的候选。

### 2）如何证明标准拒绝采样的接受结果服从 D？

**回答：** 对归一化 D 和 P，联合概率为 $P(x)D(x)/(cP(x))=D(x)/c$，总体接受率为 $1/c$。用联合概率除以接受率，就得到条件分布 D。前提是包络常数保证接受概率不超过 1。

### 3）为什么 BoN 不是精确地从人类偏好分布采样？

**回答：** BoN 在固定候选集中最大化 reward，没有按 $D/(cP)$ 做接受概率校正，而且 reward 通常不是归一化偏好概率。它定义了一个偏向高分输出的新策略，可用于 inference-time alignment，但不自动等于某个理想目标分布。

### 4）$\log N-(N-1)/N$ 衡量什么？

**回答：** 在所引用论文设定下，它上界化 BoN 输出策略相对 reference proposal 的 KL 偏移，不是 BoN 到理想人类偏好的距离。离散空间中常不是精确等式，课件示例中真实 KL 会饱和，而粗上界继续增长。

### 5）Bradley–Terry RM 的 raw score 是正确概率吗？

**回答：** 不是。两个 score 的差经过 sigmoid 才给出模型预测的两两偏好概率。整体加同一常数不改变偏好，说明 raw score 零点并不由 pairwise data 唯一决定。不同 RM 的分数也不能直接横向比较。

### 6）为什么 N 增大未必让实际答案更好？

**回答：** 它提高出现好候选的机会，但也扩大了选择器要区分的集合。如果 RM 存在系统性误差，更多候选会提高找到高 proxy reward 错误答案的机会。因此候选覆盖、selector 能力和真实评价必须分别检查。

### 7）Scalar RM 与 generative RM 的主要取舍是什么？

**回答：** Scalar RM 易批处理、排序一致，但通常需要专门训练且不易临时改变标准；generative RM 可直接使用 rubric、解释和列表输入，灵活却有额外生成成本与顺序、模板、版本敏感性。应按质量与成本共同选择。

### 8）非传递偏好一定意味着 judge 出错吗？

**回答：** 不一定。若评价关系像石头剪刀布，本来就循环；但对预期存在一致质量排序的任务，独立比较产生循环说明 judge 不稳定或标准不一致。需要先确定任务能否合理用 scalar utility 表示。

### 9）为什么 outcome RM 不能直接用来做逐 token 剪枝？

**回答：** 它把输入当完整回答评价，未必学习了从当前前缀继续后能达到的最终质量。不完整文本低分并不代表其所有续写都差。逐步剪枝需要前缀未来价值或过程质量的合适训练目标。

### 10）多模型候选是否完全不能做统计分析？

**回答：** 可以。固定权重、独立先抽模型再抽输出时，有明确定义的 mixture proposal。问题在于固定配额或自适应改变生成策略不满足原单 proposal iid 假设，因此需要重新推导，而不是直接搬用原来的公式。

### 11）为什么不能直接选 RewardBench 总分最高的 RM？

**回答：** 总分是跨领域平均，自身任务的语言、候选分布和评价目标可能不同。generator 与 RM 来源匹配、同源偏好和特定领域盲点也影响下游表现。榜单适合初筛，最终看自己的 BoN 或 RLHF 实验。

### 12）已经做了 RLHF，为什么还可能需要 BoN？

**回答：** 训练调整一般生成分布，BoN 则根据当前任务、预算和完整候选再选择。生成后的完整性和一致性有时更容易检查，例如长度上限导致一个答案完成、另一个被截断。两者可互补，但额外收益必须超过生成与评审成本。

## 与课程主线的联系

[[CMU 11-763 - Lecture 02 - Probability Review and Code Examples|Lecture 02]]中的 meta-generation 已有“生成再评分”的原型。
[[CMU 11-763 - Lecture 11 - Agents and Multi-Agent Communication|Lecture 11]]把候选扩展到完整 agent trajectory。
本讲进一步说明评分器如何学习、偏好数据有什么偏差，以及 N、候选分布与系统成本如何共同限制方法。
