---
type: course-note
status: developing
course: "[[CMU 11-763]]"
lecture: 9
lecture_date: 2025-09-23
area: inference
topics:
  - "[[LLM Inference]]"
aliases:
  - CMU 11-763 Lecture 09
  - Reasoning Models
  - CMU STaR GRPO and Long CoT
video_url: https://www.youtube.com/watch?v=6-mSbIPI4tc
---

# Lecture 09：Reasoning Models

> [!abstract] 本讲一句话
> Reasoning model 的核心是学会有效使用额外推理计算；基础模型的行为先验、奖励、训练预算和长度限制共同决定它能否学会验证、回退与搜索，而推理系统需要把这些能力变成可控的准确率、成本和延迟。

## 来源与范围

- 课程：CMU 11-664/763 — Inference Algorithms for Language Modeling，Fall 2025。
- 讲师：Graham Neubig；日期：2025-09-23。
- [完整视频](https://www.youtube.com/watch?v=6-mSbIPI4tc)，约 1:05:50；已通读完整英文自动字幕。
- [官方 HTML 课件](https://www.phontron.com/class/lminference-fall2025/assets/slides/2025-09-23-reasoning-models/index.html)：旧路径直连在整理时回落到新网站，通过搜索索引取得完整课件正文，交叉核对各章节及公式；未取得完整原 PDF 图像。
- [课程主页](https://www.phontron.com/class/lminference-fall2025/)。
- 正文覆盖 STaR、DeepSeek-R1/GRPO、cognitive behaviors、long-CoT 稳定性、跨域迁移、s1、L1、Stream of Search 与自适应并行推理。

> [!warning] 课堂口误与简写处理
> 课堂把 DeepSeek 大模型多次口头称作约 470B，并把 R1-Zero/R1 的部分描述放在同一小节。下文按原论文区分两条训练路线；DeepSeek-V3/R1 的总参数为 671B、每 token 激活约 37B。课件 GRPO 式是简写，实际 token-level ratio 与 KL 正则另作说明。模型对比均属于所引论文年代的实验，不是当前榜单。

## 视频时间索引

| 时间 | 课堂内容 | 阅读位置 |
| --- | --- | --- |
| [00:00](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=0s) | 作业格式、答案解析与提交问答 | 第 1 节 |
| [03:43](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=223s) | 什么是 reasoning model | 第 1 节 |
| [05:00](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=300s) | STaR、verifiable rewards、rationalization | 第 2 节 |
| [07:27](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=447s) | Policy gradient 解释 | 第 3 节 |
| [08:37](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=517s) | GPT-J scratchpad 与多位数加法 | 第 2 节 |
| [12:28](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=748s) | DeepSeek-R1 系列概述 | 第 4 节 |
| [14:09](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=849s) | GRPO group advantage | 第 5 节 |
| [16:01](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=961s) | Clipped objective | 第 5 节 |
| [19:47](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=1187s) | Q&A：正负 advantage 的 clipping | 第 5 节 |
| [24:21](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=1461s) | 与 PPO 的关系、old policy | 第 5 节 |
| [25:35](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=1535s) | Temperature 与 on/off-policy | 第 5 节 |
| [26:31](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=1591s) | R1-Zero prompt、准确率与长度变化 | 第 4、6 节 |
| [29:45](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=1785s) | Aha moment 与自纠错 | 第 6 节 |
| [30:42](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=1842s) | Distillation 与小模型 | 第 6 节 |
| [32:18](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=1938s) | 推理成本和模型大小的含义 | 第 12 节 |
| [33:58](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=2038s) | 四种 cognitive behaviors | 第 7 节 |
| [36:55](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=2215s) | Mid-training 与 base model 差异 | 第 7 节 |
| [41:18](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=2478s) | 训练曲线的横轴与假平台 | 第 7 节 |
| [42:55](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=2575s) | Demystifying Long CoT | 第 8 节 |
| [45:05](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=2705s) | 长度超限导致训练崩溃 | 第 8 节 |
| [46:04](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=2764s) | Cosine reward | 第 8 节 |
| [48:27](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=2907s) | SimpleRL-Zoo | 第 9 节 |
| [50:27](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=3027s) | 数学推理跨域迁移 | 第 10 节 |
| [54:06](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=3246s) | SFT 与 RL 的 token distribution drift | 第 10 节 |
| [59:58](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=3598s) | s1 与 budget forcing | 第 11 节 |
| [1:01:25](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=3685s) | L1 / LCPO | 第 11 节 |
| [1:02:29](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=3749s) | Stream of Search | 第 11 节 |
| [1:03:56](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=3836s) | 自适应并行推理 | 第 11 节 |

## 1. Reasoning model 的定义

本讲采用操作性定义：经过专门训练，通常包括 RL，能够使用较长推理序列改善任务表现的语言模型。

典型行为包括分解问题、验证中间结果、重新检查假设、回退失败路线、尝试其他策略。

它与“在答案前多写一段文字”有交集，但关键是这些步骤能帮助完成任务。

### 1.1 Long CoT 是预算使用方式

长度常与更高准确率相关，但不是单调定律。短题不需要长推导，长轨迹也可能重复、偏题或在最终答案前耗尽上下文。

所以应把 reasoning length 看成可学习、可控制的资源消耗，而非质量标签。

### 1.2 为什么推理课要讲训练

训练决定模型是否会合理使用推理 token，inference algorithm 再决定给多少预算、是否多采样、是否调用工具和何时停止。

同一个解码方法放到不同训练背景的模型上，不一定有相同效果。

开头作业问答也提醒：答案提取可以使用约束解码、结构化输出或解析；生成正确但无法按评测格式提取，同样可能被判错。

## 2. STaR：从自己做对的轨迹中学习

给定问题 $x_i$ 与已知正确答案 $y_i$：

1. 模型生成 rationale $z_i$ 与预测答案 $\hat y_i$。
2. 检查 $\hat y_i$ 是否正确。
3. 保留成功轨迹作为训练数据。
4. 对失败题，提供正确答案作为提示，重新生成能导出该答案的 rationale。
5. 对筛出的数据做 fine-tuning，再重复。

第 4 步称为 rationalization，是 STaR 的重要机制，但不是所有 reasoning model 的必选步骤。见 [STaR 原论文](https://arxiv.org/abs/2203.14465)。

### 2.1 Verifiable reward 不代表检查完全简单

数学题可比对规范化答案；代码题可执行测试。

课堂举例：`500 cm` 与 `5 m` 等价，但直接字符串比较会判不同。

因此 verifier 至少要定义格式抽取、单位、符号等价、近似容差和失败情况。错误 verifier 会把训练推向错误目标。

### 2.2 Scratchpad 示例：624 + 259

课堂展示结构化逐位加法：

- 个位 $4+9=13$，写 3、进 1。
- 十位 $2+5+1=8$，写 8、不进位。
- 百位 $6+2=8$。
- 最终答案 883。

这类 scratchpad 可以通过规则生成，为早期模型提供清晰中间状态结构。

它与自由文字 CoT 的差别在于格式更明确，状态如 carry 可直接检查。

### 2.3 Rationalization 缓解稀疏奖励

模型若几乎从不做对困难题，只保留成功轨迹就几乎得不到训练数据。

给正确答案作 hint，能帮助模型生成可训练的推导，促进起步。

但“已知答案后能编出一条解释”不必然意味着无提示时能解题；因此训练时应移除答案 hint，并用独立测试判断能力是否迁移。

### 2.4 监督越多不一定越好

课堂讨论加法曲线时提出：强监督可能让起步更快，也可能限制后续探索。

这是关于特定训练设置的经验趋势，不能推出所有 SFT 都降低 RL 的上限；轨迹质量、风格多样性、底座能力和训练量都有关。

## 3. 用 policy gradient 理解“保留做对的轨迹”

设完整输出 $o=(z,y)$，策略为 $\pi_\theta(o\mid x)$，奖励为 $R(x,o)$：

$$
J(\theta)=\mathbb E_{o\sim\pi_\theta}[R(x,o)]
$$

对不显式依赖参数的奖励：

$$
\nabla_\theta J=
\mathbb E[R(x,o)\nabla_\theta\log\pi_\theta(o\mid x)]
$$

若 $R=1$ 表示正确、$R=0$ 表示错误，则没有 baseline 的 REINFORCE 估计会把错误样本的这项梯度置零。

### 3.1 这与 SFT 筛选的关系

两者都可增强成功轨迹，但 STaR 的离线生成、数据筛选、rationalization 与反复 fine-tuning，不应直接说成与任意 on-policy RL 完全相同。

### 3.2 奖励不是过程正确性的完整证明

最终答案正确，可能来自真实推导，也可能来自猜测或错误相互抵消。

Outcome reward 的优势是便宜、可规模化；限制是无法完整定位每个中间步骤的质量。

若引入过程监督，需要额外的步骤判定器和标注/验证成本；本讲主要讨论结果奖励。

## 4. DeepSeek-R1-Zero 与 DeepSeek-R1

### 4.1 两条路线必须分开

| 路线 | 初始阶段 | 后续重点 |
| --- | --- | --- |
| R1-Zero | 直接从已预训练的 DeepSeek-V3-Base 做 RL，不先做专门的 reasoning SFT | 验证仅靠结果/格式奖励能否诱导长 CoT 与自检 |
| R1 | 少量 cold-start reasoning 数据先做 SFT | Reasoning RL、筛选并混合数据的 SFT、面向更多场景的 RL |

R1-Zero 的“Zero”不是随机初始化，也不是没有数据：底座已经有大规模预训练知识。

R1 的多阶段流程旨在改善可读性、语言混杂以及更广泛任务表现。见 [DeepSeek-R1 原论文第 2 节](https://arxiv.org/html/2501.12948v1#S2)。

### 4.2 Minimal prompt 的作用

课堂展示用 `think` 与 `answer` 标签区分推理和最终答案的模板。

模板主要提供输出结构，没有给出详细解题策略；模型仍需从预训练能力中产生一些成功样本，RL 才有可用信号。

格式符合要求与答案正确是不同奖励维度。只学会写标签，不等于学会推理。

### 4.3 参数量与每步成本

DeepSeek-V3/R1 是 MoE。应区分总参数 671B 与每 token 激活约 37B，不能把总参数直接代入 dense 模型的每 token FLOPs 估算。

总参数仍影响权重存储和部署布局，激活参数影响一部分计算量；通信、路由和注意力开销还需单独计入。参数数据见 [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)。

## 5. GRPO：组内相对优势与 clipped update

### 5.1 同一道题生成一组候选

从采样策略 $\pi_{old}$ 对同一问题 $q$ 生成 $G$ 个完整输出 $o_i$，分别计算奖励 $r_i$。

组内均值和标准差：

$$
\mu_r=\frac1G\sum_{i=1}^Gr_i,\qquad
\sigma_r=\sqrt{\frac1G\sum_{i=1}^G(r_i-\mu_r)^2}
$$

归一化 advantage：

$$
A_i=\frac{r_i-\mu_r}{\sigma_r}
$$

实际实现应处理 $\sigma_r=0$，例如加稳定项或跳过无相对奖励差异的组；不要直接除零。

### 5.2 一个数值例子

四个奖励是 $(1,1,0,0)$，则 $\mu_r=0.5$、总体标准差 $\sigma_r=0.5$，advantage 为 $(1,1,-1,-1)$。

重点是相对于同题其他样本的表现。它不需要单独训练一个 value model 来估计该题 baseline。

如果四个都正确或都错误，归一化后的相对信号为零；这也解释题目难度与有效训练样本组的重要性。

### 5.3 课堂简化目标

定义 probability ratio：

$$
\rho_i(\theta)=\frac{\pi_\theta(o_i\mid q)}{\pi_{old}(o_i\mid q)}
$$

课堂展示的最大化目标为：

$$
J_{clip}(\theta)=\mathbb E\left[
\frac1G\sum_i\min\left(\rho_i A_i,
\operatorname{clip}(\rho_i,1-\epsilon,1+\epsilon)A_i\right)
\right]
$$

若代码使用 optimizer 最小化 loss，通常取该目标的负数，并加所需正则项。

### 5.4 正 advantage 为什么只截过度上升的一侧

设 $A>0$，单项可以写成：

$$
f(\rho,A)=A\min(\rho,1+\epsilon)
$$

当 $\rho>1+\epsilon$，继续增大该样本概率不再增加 clipped surrogate；当 $\rho$ 很小，目标仍保留把好样本概率提高的梯度。

它不会在坏方向也“保护性截断”，否则模型错误地降低好样本概率时，可能没有信号拉回来。

### 5.5 负 advantage 为什么只截过度下降的一侧

设 $A<0$：

$$
f(\rho,A)=A\max(\rho,1-\epsilon)
$$

若 $\rho<1-\epsilon$，继续降低坏样本概率不再改善 surrogate；若坏样本概率变大，仍保留惩罚。

这正是课堂 19–24 分钟问答的核心。

### 5.6 带数值检查正负号

取 $\epsilon=0.1$：

| $A$ | $\rho$ | 原项 $\rho A$ | clipped 项 | 二者最小值 |
| --- | --- | --- | --- | --- |
| $+1$ | 1.5 | 1.5 | 1.1 | 1.1：过度提高不再获益 |
| $+1$ | 0.1 | 0.1 | 0.9 | 0.1：保留恢复好样本的梯度 |
| $-1$ | 0.1 | -0.1 | -0.9 | -0.9：过度压低不再获益 |
| $-1$ | 1.5 | -1.5 | -1.1 | -1.5：保留惩罚坏样本的梯度 |

课堂用极大负 advantage 的例子容易把数值大小与梯度方向混在一起。先确定目标是最大化，再比较两个乘积，最不容易出错。

### 5.7 Clipping 不是硬约束

该目标并未保证更新后所有 $\rho$ 都在区间内；参数共享意味着对一个 token 的更新会影响其他 token。

它也不是 gradient norm clipping。概率比值截断与梯度范数截断限制的是不同对象。

### 5.8 实际算法比课堂式更细

DeepSeek 论文使用 token-level probability ratios，并对每条输出的 token 求平均，同时包含 reference policy 的 KL penalty。

可以概括为：

$$
J=\mathbb E\left[\frac1G\sum_i\frac1{|o_i|}
\sum_t\left(f(\rho_{i,t},A_i)-\beta D_{KL,i,t}\right)\right]
$$

其中 $\rho_{i,t}$ 比较新旧策略对同一已生成 prefix 下该 token 的条件概率。

`old policy` 是生成 rollout 的策略快照；`reference policy` 则作为 KL 锚点，二者概念不同。

### 5.9 为什么课堂强调 T=1

若优化对象是未调整的 $\pi_\theta$，T=1 的祖先采样直接来自该策略；其他温度会改变行为策略，形成 off-policy 问题。

可以设计带温度的采样与训练，但要把实际 behavior distribution 和 importance ratio 定义一致。并不是所有工程实现必须机械地使用 T=1，而是不能把改变分布的采样仍当成原策略样本。

## 6. R1 实验对 inference 的启示

### 6.1 训练使模型愿意使用更多 token

课堂展示 R1-Zero 在训练中准确率提高、输出长度从数百增长到数千 token 的曲线。

原论文的 AIME 2024 结果为 pass@1 从 15.6% 增到 71.0%，多数投票可进一步达到 86.7%；这些是固定论文设置的结果。参见 [R1 论文](https://arxiv.org/html/2501.12948v1)。

不能仅从长度增长判断“学到了更多推理”，还要看质量、策略行为与截断。

### 6.2 Aha moment 的准确解读

轨迹出现“Wait”并修正此前推导，是一个可以观察的自检行为。

出现这个词本身不是证据，关键是修正是否有效。还需区分训练后某种行为更频繁与证明该行为在预训练模型中完全不存在。

### 6.3 Distillation 是另一种获得小模型能力的路径

大模型生成高质量轨迹，筛选后用于训练较小的 Qwen/Llama 模型。

在原论文比较中，直接从大模型蒸馏比在相应小模型上直接做 RL 更有效；课堂解释是小模型起始成功率不足时，RL 获得有效奖励更困难。

这种解释合理，但实验结果不构成“小模型 RL 永远不如蒸馏”的普遍定律。

> [!note] 课件数值修正
> HTML 文本把 AIME 55.5% 与 14B 放在一起；原报告该数值对应 Distill-Qwen-7B。本文避免复用这组错配，不把各模型、任务或 pass@1/多数票指标混在一起。

## 7. 哪些 cognitive behaviors 支持自我改进

[33:58](https://www.youtube.com/watch?v=6-mSbIPI4tc&t=2038s) 比较 Qwen 与 Llama 的 RL 轨迹：相近规模和相同训练方法不一定获得相同增长。

论文归纳四类行为：

| 行为 | 功能 | 例子 |
| --- | --- | --- |
| Verification | 检查结果与约束是否一致 | 把结果代回原式 |
| Backtracking | 放弃失败路线，回到先前状态 | “这条路线不行，换一种方法” |
| Subgoal setting | 为复杂任务设中间目标 | 先得到 10 的倍数 |
| Backward chaining | 从目标反推需要满足的条件 | 24 可以由 $8\times3$ 得到 |

预先具有这些行为的模型更容易通过 RL 放大有效策略；用行为示例 priming 或相关数据继续训练，也可帮助原本较少表现这些行为的模型。见 [Cognitive Behaviors 论文](https://arxiv.org/abs/2503.01307)。

### 7.1 Base 并不是统一的训练契约

课堂把 mid-training、精选教育文本、数学数据和合成数据列为可能原因。

这些因素能影响初始策略分布，即模型在 RL 开始时是否会偶尔产生有用轨迹。

但不能因为观察到行为差异，就反向断言某厂商具体使用了哪种未公开数据。

### 7.2 不能只按 token 长度统计行为

较长输出自然有更多机会出现“检查”等词。比较行为频率时应考虑长度、任务难度与行为是否真的产生有效修正。

仅统计某个关键词，会把礼貌套话和无效自我重复也算作 reasoning skill。

### 7.3 看训练图一定要检查横轴

课堂发现一个研究曲线在约 250 steps 附近结束，而另一大规模实验经历更多 steps 和较大 batch。

“前几百步长度没涨”不证明永远不会涨。不同 step 定义、batch size、每题 rollout 数和 token 长度使训练计算不可直接按横轴数字比较。

**工程补充：** 至少同时记录 optimizer steps、问题数、rollout tokens 和 GPU 时间，才容易判断是否真的出现平台。

## 8. Demystifying Long CoT：长度增长也可能毁掉训练

### 8.1 初始轨迹习惯影响后续 RL

课堂展示 short-CoT 与 long-CoT 初始化的比较：在所测设置中，长 CoT 初始化更容易继续从 RL 受益，过多短式指令数据可能阻碍长轨迹发展。

这里的“长”不仅是字数，还包括验证和回退等复杂策略的表达空间。

### 8.2 长度超限造成奖励崩溃

只奖励正确答案，模型可能逐渐倾向更长输出；当长度超过上限，最终答案被截断，verifier 无法找到正确答案，奖励突然下降。

```text
较长推理暂时提高成功率
    → RL 增强长输出
    → 触及最大长度
    → 最终答案未生成
    → 被判错误、有效训练信号下降
```

因此训练崩溃时，不能只查学习率和梯度；还要看 truncation/exceed rate、答案抽取失败与长度分布。

### 8.3 Cosine reward 的方向

课堂描述一种联合考虑正确性与长度的奖励：

- 正确轨迹：在保持正确的前提下，更短更受奖励。
- 错误轨迹：短而错受到较强惩罚，鼓励在预算内继续尝试。
- 超限轨迹：单独处罚，避免把用完预算当成进步。

用端点形式解释余弦插值：

$$
s(L)=\frac{1+\cos(\pi L/L_{max})}{2}
$$

$$
R_c(L)=r_L^c+(r_0^c-r_L^c)s(L),\qquad
R_w(L)=r_L^w+(r_0^w-r_L^w)s(L)
$$

其中正确轨迹通常取 $r_0^c>r_L^c$，错误轨迹取 $r_0^w<r_L^w$，达到硬上限另给 exceed penalty。

以上是按端点重新参数化的解释式，精确复现需采用论文的参数和超限判断。参见 [Demystifying Long CoT](https://arxiv.org/abs/2502.03373) 与 [作者代码](https://github.com/eddycmu/demystify-long-cot)。

### 8.4 不能误解为“错误越长越好”

奖励应保持正确答案总体优于错误答案，同时在各类内部塑造预算使用。

如果长度奖励压过正确性，模型可能通过冗长输出来得分，即 reward hacking。固定长度惩罚也可能让困难题过早结束。

### 8.5 Verifier 质量与能力出现

课堂还讨论模型容量、底座数据、训练计算、规则验证与模型验证等因素。

某些 7B 模型在实验中难以发展复杂能力，不意味着所有 7B 模型存在统一不可突破的边界。

“规则 verifier 更好”也有任务前提：数学等价或程序测试可提供清晰信号时，规则通常更稳定；开放式语义任务未必存在同等可靠的规则。

## 9. SimpleRL-Zoo：跨模型检验 zero RL

该研究把直接从 base model 做 RL 扩展到多个模型家族和规模，检查是否只有 Qwen 能有效增长。

课堂结论是：收益并非只存在于一个模型家族，但需要题目难度、格式奖励与训练预算足以产生有信息的反馈。

不同模型可能发展不同策略，如枚举、子目标或验证；长度增加不必然与某一种行为同步出现。见 [SimpleRL-Zoo](https://arxiv.org/abs/2503.18892)。

这对复现的重要启示是：全零奖励可能是任务或输出格式不匹配，不能立刻归因于优化算法无效。

## 10. 数学训练能否迁移到其他能力

课堂讨论 Qwen3-14B 的控制实验：使用 math-only 数据，比较 SFT 与 GRPO，并在数学之外测试科学问答、代码、agent planning、对话和指令遵循。

SFT 使用 teacher 产生并筛选的轨迹；RL 使用答案正确性奖励。

所测实验中 RL 的跨域保持/迁移较好，SFT 更容易损伤通用能力。这个结果是特定模型与配方的比较，不能简单推广成“RL 总泛化，SFT 总遗忘”。见 [Reasoning Transfer 原论文](https://arxiv.org/abs/2507.00432)。

### 10.1 从 token gradient 理解差别

SFT 对一个目标 token $y$ 的 logit 梯度：

$$
\frac{\partial[-\log p(y)]}{\partial z_j}
=p(j)-\mathbf1[j=y]
$$

它增加目标 token 的相对概率，同时对其他 token 施加归一化竞争。

Group-based RL 使用有正有负的 advantage。多个样本中的共同表达可能在加权梯度中部分抵消，奖励相关差异则被放大。

### 10.2 课堂口头解释的严格边界

讲师用“SFT 下调其他序列，RL 只上调好样本、下调坏样本”说明分布漂移差异。

严格说，RL 的 softmax 也归一化，参数也共享，所以更新同样会影响未采样输出。不能把该描述当成“未采样序列概率保持不变”的数学结论。

更稳妥的结论是：这些实验观察到 RL 的参数/表示/token 分布漂移较小，并与较好跨域保持相关。

### 10.3 为什么需要两个层次的评测

非控制的模型横向比较能看总体趋势，但不同模型的数据、架构和训练量很多因素同时变化。

固定底座和数据，比较不同训练方式，更接近回答“训练方式造成了什么差异”。二者不能混为同一种证据。

## 11. 四类预算与搜索控制方法

### 11.1 s1：Budget forcing

课堂概括两类推理时干预：

- 模型想结束推理时，通过追加 “Wait” 等方式让它继续检查。
- 接近预算上限时，切换到生成最终答案的阶段。

论文用约 1,000 条精选 reasoning 示例对 Qwen2.5-32B 做 SFT，说明小规模高质量数据配合推理时控制可有效。见 [s1](https://arxiv.org/abs/2501.19393)。

这不是把服务进程简单 kill 掉。硬截断可能连答案都没有；budget forcing 需要保留一个可生成答案的结束协议。

### 11.2 L1 / LCPO：学会遵守指定长度

在 prompt 中给出目标 token 数，训练模型同时满足任务正确性与长度要求。

课件用如下简化奖励说明：

$$
r=\mathbf1[y=y_{gold}]-\alpha|n_y-n_{target}|
$$

该式体现 accuracy 与 length adherence 的权衡，不能未经核对就当成所有 LCPO 版本的完整实现。

原工作区分 exact length 与 maximum length 约束；二者对“少于目标长度”是否受罚不同。见 [L1 原论文](https://arxiv.org/abs/2503.04697)。

### 11.3 Stream of Search：把搜索过程写成训练序列

先运行 BFS/DFS 等搜索策略，把探索、失败、回退和成功过程序列化成文本，再训练模型生成这类搜索流。

课堂提到 Countdown 游戏：用给定数字和算术运算组合到目标数；其训练数据包含多种搜索策略产生的大量轨迹。

与只模仿成功路径相比，它让模型看到失败之后如何继续探索。见 [Stream of Search](https://arxiv.org/abs/2404.03683)。

### 11.4 Adaptive Parallel Reasoning

课堂末尾称为 adaptive parallel search；对应工作是 *Learning Adaptive Parallel Reasoning with Language Models*（APR）。

模型能调用 `spawn` 创建子线程，调用 `join` 汇总结果，并对父子线程协调进行训练。

它改变的不是单个序列的最大上下文，而是把搜索分到多个独立上下文中，再把必要信息返回主线程。

这样可以增加总探索量，同时限制单条串行轨迹长度。实验应同时看总 token 和 latency；总计算增加不必然意味着墙钟延迟同比增加。见 [APR 原论文](https://arxiv.org/abs/2504.15466)。

## 12. AI Infra：把推理能力变成可运行的服务

### 12.1 资源向量比单一 token 数更有用

至少考虑：

$$
\text{cost}=f(\text{active model size},\text{prompt tokens},
\text{reasoning tokens},\text{branches},\text{hardware})
$$

同样 10,000 token，串行生成和十条并行 1,000-token 轨迹的延迟与峰值显存不同。

同样参数量，dense 与 MoE 的激活计算、通信和存储也不同。

### 12.2 为最终答案留预算

设置总预算 $B$ 时，可分成 reasoning budget $B_r$ 与 answer budget $B_a$，满足 $B_r+B_a\le B$。

如果只给一个总 max tokens，而模型把预算全部用于推理，正确思路也可能因为没有最终答案而判失败。

### 12.3 训练与 serving 的长度契约要一致

模型在训练时习惯 16K 轨迹，部署却只允许 2K，可能大量截断；只测训练 reward 看不出真实服务效果。

应分别测试不同预算下的准确率、自然结束率、截断率、最终答案抽取率和延迟分位数。

### 12.4 RL rollout 是一个推理系统

训练侧仍需要高吞吐生成、采样策略、旧 logprob、token mask、终止原因、verifier 调用与异常处理。

rollout 服务升级、模板变化或温度改变，如果没有与训练目标同步，会改变实际 behavior policy。

### 12.5 正确预算比较

| 比较 | 需要固定或报告 |
| --- | --- |
| 大模型短推理 vs 小模型长推理 | 总 GPU 成本、延迟、精度、量化、任务 |
| 一条长轨迹 vs 多条短轨迹 | 总 token、并发、聚合方法、峰值显存 |
| SFT vs RL | 底座、数据、训练计算、推理预算 |
| 原始 RL vs 长度奖励 | 正确率、长度分布、截断率、reward hacking |
| 串行搜索 vs APR | 总探索量、critical path、协调开销 |

## 13. 课堂问答与容易混淆之处

### Q1：GRPO 与 PPO 的主要区别是什么？

课堂强调 clipping 继承 PPO 思路，group-normalized advantage 是关键区别之一；不能把 GRPO 解释成新发明了一种 clipping。

### Q2：Old policy 是哪一个 checkpoint？

它应对应产生当前 rollout 的策略。异步生成时，数据可能来自比当前更新更老的版本，因此要保存实际生成版本和 logprob。

### Q3：负 advantage 时取 min 为什么仍会截断？

乘上负数会反转大小关系。比如 $\rho=0.1,A=-1$，原项为 -0.1，clipped 项为 -0.9，min 选 -0.9，正是平坦区。

### Q4：长度曲线平台是否意味着收敛？

不一定。课堂指出短训练曲线可能尚未出现后来增长；需要检查 batch、样本量、总生成量和训练时间。

### Q5：从 base 做 RL 是从零学习推理吗？

不是。base 已经有预训练知识和行为先验。课堂后段还用棋类自我改进作类比；其中“完全从零训练”的严格对应应是 AlphaGo Zero/AlphaZero，而非混用所有 AlphaGo 版本。

### Q6：SFT + RL 是否总比纯 RL 更好？

某些领域内结果可能更好，但必须同时看跨域保持和预算；课堂的迁移实验表明数学分数提高不等于所有能力一起提高。

## 14. 自测问题与面试回答

1. Reasoning model 与普通 CoT prompting 的区别？

    **面试回答：** CoT prompting 在现有模型上诱导中间步骤，reasoning model 则经过专门训练来有效使用长轨迹，常见训练包括 RL。区别不是是否有一段解释，而是是否学会验证、回退等能改善任务表现的策略。

2. STaR 的 rationalization 解决什么问题？

    **面试回答：** 困难题初始正确率低时，成功轨迹筛选几乎得不到数据。Rationalization 提供正确答案作提示，让模型生成可学习推导，再用这些轨迹训练不带答案提示的模型，缓解起步阶段稀疏奖励；泛化仍需独立验证。

3. R1-Zero 的 Zero 是什么含义？

    **面试回答：** 它从已预训练的 base model 开始做 RL，不先做专门的 reasoning SFT。它不是随机初始化或没有知识；正式 R1 还包含 cold-start SFT、reasoning RL、筛选数据 SFT 与后续 RL，不能混为同一路线。

4. 如何计算 group advantage，全部奖励相同怎么办？

    **面试回答：** 以同题候选奖励均值作 baseline，再除标准差，$A_i=(r_i-\mu)/\sigma$。全对或全错时没有组内相对信号，需用稳定项、跳过等方式处理零方差；训练数据难度要让足够多的组同时出现成功与失败。

5. PPO-style clipping 对正负 advantage 分别做什么？

    **面试回答：** 最大化目标时，正 advantage 在 ratio 超过 $1+\epsilon$ 后停止奖励进一步提高；负 advantage 在 ratio 低于 $1-\epsilon$ 后停止奖励进一步降低。相反方向保留纠正梯度，因此不是把 ratio 硬限制在区间里。

6. Old policy 和 reference policy 有什么不同？

    **面试回答：** Old policy 是产生当前 rollout 的行为策略，进入 importance ratio；reference policy 是用于限制分布漂移的 KL 锚点。二者可在某些时刻相同，但更新节奏和作用不同，异步训练要准确保存生成策略版本。

7. 为什么温度改变会影响 on-policy 假设？

    **面试回答：** 温度把 softmax logits 改成 logits/T，生成分布随之改变。若仍优化未调整策略却用另一温度采样，样本来自不同 behavior policy；可以做 off-policy 设计，但 ratio 和采样契约必须一致。

8. 长 CoT 训练为什么会突然准确率崩溃？

    **面试回答：** 只奖励正确性可能鼓励轨迹变长，当长度触及上限，最终答案被截断，verifier 判失败，奖励和训练信号骤降。应检查 exceed rate、答案解析失败和长度分布，而不只查梯度或学习率。

9. Cosine reward 怎样同时鼓励效率与探索？

    **面试回答：** 对正确轨迹，短轨迹有较高奖励；对错误轨迹，短而错受到更强惩罚，允许在预算内继续探索；超限另罚。必须让正确性占主导，并监测模型是否通过无效延长轨迹获取奖励。

10. 为什么 SFT 与 GRPO 的跨域影响可能不同？

    **面试回答：** SFT 直接模仿选定目标序列，可能同时改变大量表达习惯；GRPO 通过组内正负优势强调奖励相关差异，共同部分可能抵消。所述实验中 RL 漂移更小、通用能力保持更好，但参数共享使两者都会影响未采样输出，不能作绝对保证。

11. 小模型长推理胜过大模型短推理，是否证明小模型更省钱？

    **面试回答：** 不自动证明。要把激活参数、总输出长度、量化、硬件利用率、KV 和通信成本一起计入，并比较同样质量下的费用与延迟。论文上的准确率比较只是质量证据，不等于完整服务成本比较。

12. APR 如何突破单条 CoT 的局限？

    **面试回答：** 模型学习创建并行子线程和汇总结果，把探索分配到多个上下文中，增加总计算而不要求一条轨迹装下所有过程。收益取决于任务可分解性、协调质量、资源并行度和通信开销，必须同时测总 token 与 critical-path latency。

## 15. 阅读地图与关联

| 论文 | 对应问题 |
| --- | --- |
| [STaR](https://arxiv.org/abs/2203.14465) | 成功轨迹、rationalization、迭代学习 |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | R1-Zero、R1、GRPO、蒸馏 |
| [Cognitive Behaviors](https://arxiv.org/abs/2503.01307) | 初始行为先验与 RL 可改进性 |
| [Demystifying Long CoT](https://arxiv.org/abs/2502.03373) | 长度稳定性、奖励与 emergence 条件 |
| [SimpleRL-Zoo](https://arxiv.org/abs/2503.18892) | 跨模型 zero RL 的实证检查 |
| [Reasoning Transfer](https://arxiv.org/abs/2507.00432) | 数学训练的跨域保持与遗忘 |
| [s1](https://arxiv.org/abs/2501.19393) | Budget forcing 与小规模 SFT |
| [L1](https://arxiv.org/abs/2503.04697) | 显式 token budget 的 RL 控制 |
| [Stream of Search](https://arxiv.org/abs/2404.03683) | 搜索轨迹的语言化 |
| [Adaptive Parallel Reasoning](https://arxiv.org/abs/2504.15466) | 学习串行/并行的协调 |

- 课程索引：[CMU 11-763](CMU%2011-763.md)。
- 前置：[Lecture 07](CMU%2011-763%20-%20Lecture%2007%20-%20Chain%20of%20Thought%20and%20Intermediate%20Steps.md)、[Lecture 08](CMU%2011-763%20-%20Lecture%2008%20-%20Self-Refine%20and%20Self-Correction%20Methods.md)。
- 主题入口：[LLM Inference](../../topics/inference/LLM%20Inference.md)。
