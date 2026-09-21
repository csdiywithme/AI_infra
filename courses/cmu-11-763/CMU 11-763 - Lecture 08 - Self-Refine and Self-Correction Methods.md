---
type: course-note
status: developing
course: "[[CMU 11-763]]"
lecture: 8
lecture_date: 2025-09-18
area: inference
topics:
  - "[[LLM Inference]]"
aliases:
  - CMU 11-763 Lecture 08
  - Self-Refine and Self-Correction Methods
  - CMU Iterative Refinement
video_url: https://www.youtube.com/watch?v=uaxf9yssDy4
---

# Lecture 08：Self-Refine and Self-Correction Methods

> [!abstract] 本讲一句话
> 把一次生成变成“草稿—检查—修改”的循环，只有在检查能提供有用信号、修改能够利用该信号、停止规则不会持续破坏好答案时，额外推理计算才会转化为质量收益。

## 来源与范围

- 课程：CMU 11-664/763 — Inference Algorithms for Language Modeling，Fall 2025。
- 讲师：Graham Neubig；日期：2025-09-18。
- [课程视频](https://www.youtube.com/watch?v=uaxf9yssDy4)，约 41:34；已读取完整英文自动字幕。
- [官方课程安排](https://www.phontron.com/class/lminference-fall2025/schedule/) 的历史索引列有本讲 HTML/PDF slides 及阅读论文。
- 整理时新版网站的旧资源路由回落到网站主页；本讲未取得可验证的完整原始 slides。正文以完整字幕为主，并用下文链接的论文核对方法定义。不会把未见课件的图片数值写成已核验结果。
- 覆盖课堂的六类生成/修订方法与一个编辑表示方法，以及自纠错失败分析、工程成本和课堂问答。

> [!note] 来源边界
> 代码和额外公式是根据课堂算法写出的教学实现与数学解释，不是官方代码的逐字复刻。公开视频在学生报告开始前结束，学生报告的内容未计入正文。

## 视频时间索引

| 时间 | 课堂内容 | 阅读位置 |
| --- | --- | --- |
| [00:30](https://www.youtube.com/watch?v=uaxf9yssDy4&t=30s) | 本讲与 CoT、reasoning models 的关系 | 第 1 节 |
| [01:01](https://www.youtube.com/watch?v=uaxf9yssDy4&t=61s) | 通用 self-correction loop | 第 1 节 |
| [02:43](https://www.youtube.com/watch?v=uaxf9yssDy4&t=163s) | Critique、training、conditioning、tools 四个维度 | 第 2 节 |
| [04:49](https://www.youtube.com/watch?v=uaxf9yssDy4&t=289s) | Deliberation Networks | 第 3 节 |
| [06:46](https://www.youtube.com/watch?v=uaxf9yssDy4&t=406s) | 中间序列空间与采样近似梯度 | 第 3 节 |
| [11:34](https://www.youtube.com/watch?v=uaxf9yssDy4&t=694s) | Learning to Model Editing Processes | 第 4 节 |
| [14:15](https://www.youtube.com/watch?v=uaxf9yssDy4&t=855s) | 编辑标签与内容生成分解 | 第 4 节 |
| [17:36](https://www.youtube.com/watch?v=uaxf9yssDy4&t=1056s) | Wikipedia/GitHub 编辑历史 | 第 4 节 |
| [19:46](https://www.youtube.com/watch?v=uaxf9yssDy4&t=1186s) | Learning to Represent Edits | 第 5 节 |
| [21:35](https://www.youtube.com/watch?v=uaxf9yssDy4&t=1295s) | 512 维编辑表示瓶颈 | 第 5 节 |
| [25:19](https://www.youtube.com/watch?v=uaxf9yssDy4&t=1519s) | Training-free Self-Refine | 第 6 节 |
| [27:12](https://www.youtube.com/watch?v=uaxf9yssDy4&t=1632s) | 不同任务、不同模型的收益 | 第 6 节 |
| [29:11](https://www.youtube.com/watch?v=uaxf9yssDy4&t=1751s) | Self-Debugging 与执行反馈 | 第 7 节 |
| [30:31](https://www.youtube.com/watch?v=uaxf9yssDy4&t=1831s) | Q&A：为何不直接传 stack trace | 第 7、12 节 |
| [31:52](https://www.youtube.com/watch?v=uaxf9yssDy4&t=1912s) | 与 self-consistency 的样本效率比较 | 第 7 节 |
| [32:49](https://www.youtube.com/watch?v=uaxf9yssDy4&t=1969s) | Reflexion：把反思存成 memory | 第 8 节 |
| [34:18](https://www.youtube.com/watch?v=uaxf9yssDy4&t=2058s) | CRITIC：检查者使用工具 | 第 9 节 |
| [35:59](https://www.youtube.com/watch?v=uaxf9yssDy4&t=2159s) | 小模型是否能自纠错 | 第 12 节 |
| [37:10](https://www.youtube.com/watch?v=uaxf9yssDy4&t=2230s) | Intrinsic self-correction 的失败 | 第 10 节 |
| [40:22](https://www.youtube.com/watch?v=uaxf9yssDy4&t=2422s) | 为什么不把所有组件全部加上 | 第 11、12 节 |

## 1. 从单次生成到修订循环

传统自回归生成从左到右推进。早期 token 一旦出错，后面的文字通常只能在其基础上继续；人写文章或代码时却会回头修改前文。

本讲把这种修订显式拆成几个组件：

$$
y_0=G(x),\qquad
c_t=C(x,y_t,h_t),\qquad
y_{t+1}=R(x,y_t,c_t,h_t)
$$

- $x$：任务和约束。
- $y_t$：当前版本。
- $c_t$：本轮检查结果或 critique。
- $h_t$：可选的历史、记忆和工具观测。
- $G,C,R$：生成、检查、修改组件，可以由同一模型承担。

```text
问题 x → 初始版本 y0 → 检查 c0 → 满足停止条件？
                             ├─ 是 → 返回当前版本
                             └─ 否 → 修订 y1 → 再检查
```

课堂明确要求有最大迭代次数，避免无限循环；“检查者认为足够好”只是软停止条件。

### 1.1 CoT 与 self-refinement 的联系

CoT 在一个输出序列中展开中间步骤；显式 refinement 则把当前完整答案作为下一轮输入，再生成新版本。

Reasoning model 可能把检查和回退写进一条长轨迹，所以它们在行为上有交集；但显式多轮系统还需要管理版本、调用和停止规则。

### 1.2 “修改了”不代表“改善了”

必须评价 $y_{t+1}$ 相对 $y_t$ 是否更满足任务。更长、更自信、更详细，可能只改变风格。

若任务只要求返回一个整数，新增三段解释不一定有用；若任务要求调整语气，事实与格式保持不变才是合理修订。

## 2. 四个设计维度

### 2.1 Explicit 与 implicit critique

显式 critique 先生成“哪里有问题、为什么、怎样改”的检查结果。

隐式 critique 不一定产生单独评语；模型直接预测编辑操作或新版本，若输出 no-op/不作修改，就被视为停止信号。

二者不能简单按优劣排序。显式反馈可观察、易定位；隐式编辑少一次生成，并可能更便宜。

### 2.2 是否专门训练

早期方法通过训练学会修订；Self-Refine 等方法调用已有强模型，通过 prompt 完成多种角色。

Training-free 的意思是方法使用时不额外更新权重，不代表底座没有经过预训练、SFT 或其他训练。

### 2.3 修订依赖哪些信息

至少有以下选择：原始任务、当前版本、当前 critique、之前所有版本、历次反馈、外部观测。

增加历史可减少重复错误，也会增加上下文成本，并可能把过时结论带入下一轮。

### 2.4 是否有外部工具

执行器、测试、检索和专门的分类 API 可以给模型提供新的证据。

如果只是把同一问题再问一遍，新增的信息主要来自重新计算与不同生成模式；工具则可能改变系统实际掌握的信息。

### 2.5 方法对照

| 方法 | Critique / 编辑方式 | 训练 | 主要反馈或状态 |
| --- | --- | --- | --- |
| Deliberation Networks | 第二遍直接改写 | 专门训练 | 输入 + 初稿全局上下文 |
| Editing Processes | 操作标签 + 局部内容 | 专门训练 | 当前文本 + 编辑历史 |
| Representing Edits | 把变化编码成向量 | 专门训练 | before/after 的编辑表示 |
| Self-Refine | 显式自然语言反馈 | 方法本身无需额外训练 | 输入 + 当前输出 + 反馈 |
| Self-Debugging | 解释代码、检查执行结果 | 课堂强调 prompting 方案 | 代码 + 执行观测/测试 |
| Reflexion | 将反馈转成文字记忆 | 推理时不更新权重 | 当前任务 + 历次反思 |
| CRITIC | Critic 主动调用工具验证 | 方法本身无需额外训练 | 待检查输出 + 工具证据 |

编辑表示方法主要研究如何表示变化，并非完整的自纠错循环；课堂也明确指出七篇中有一篇属于相关研究。

## 3. Deliberation Networks：第二遍获得全局视野

[04:49](https://www.youtube.com/watch?v=uaxf9yssDy4&t=289s) 介绍 2017 年的双遍解码：第一遍生成草稿 $y'$，第二遍读取 $x,y'$，输出修订版 $y$。

$$
y'\sim P_\phi(y'\mid x),\qquad
y\sim P_\psi(y\mid x,y')
$$

第一遍生成早期词时，看不到未来词；第二遍却能观察整条草稿，因而有机会修正一致性、缺失和措辞。

原方法使用 encoder-decoder 架构。今天也可以让同一个模型用不同提示承担两个角色，但这只是实现选择，不是原论文结构的原样复现。见 [作者发表页](https://www.microsoft.com/en-us/research/publication/deliberation-networks-sequence-generation-beyond-one-pass-decoding/)。

### 3.1 为什么训练中的求和难算

把草稿作为潜变量，最终输出概率是：

$$
P(y\mid x)=\sum_{y'}P_\phi(y'\mid x)P_\psi(y\mid x,y')
$$

在监督数据 $(x,y)$ 中，目标 $y$ 已知，但中间 $y'$ 的空间是所有可能序列。

无长度限制时它可数无限；设最长 100 token 后仍约为 $V^{100}$，不能枚举。

课堂学生首先想到“离散所以不能求梯度”。讲师强调，在讨论梯度前，连对所有草稿求和的函数值都算不起。

### 3.2 用采样解释训练信号

采样若干草稿 $y'_1,\ldots,y'_K$，让第二遍评估已知正确目标在这些草稿条件下的 log probability。

直觉上，能帮助第二遍生成正确答案的草稿应被强化。

为了理解梯度，可以考虑以下教学目标：

$$
\mathcal J(\phi,\psi)=
\mathbb E_{y'\sim P_\phi(\cdot\mid x)}
[\log P_\psi(y\mid x,y')]
$$

固定 $\psi$ 时的 score-function gradient：

$$
\nabla_\phi\mathcal J=
\mathbb E[(r(y')-b)\nabla_\phi\log P_\phi(y'\mid x)]
$$

其中 $r(y')=\log P_\psi(y\mid x,y')$，$b$ 是不依赖本次草稿的 baseline，可降低方差。

对第二遍模型，直接对监督目标做条件 log-likelihood 梯度。

> [!note] 两种目标不要混淆
> $\mathbb E[\log P_\psi(y\mid x,y')]$ 与 $\log\mathbb E[P_\psi(y\mid x,y')]$ 一般不同，前者是后者的 Jensen 下界。上式用于解释课堂的“采样中间态，以最终目标似然提供训练信号”，不冒充完整原论文训练细节。

### 3.3 它与 GRPO 的联系有限

讲师类比“采多个候选，增强效果好的候选”，并明确说不完全等同于 GRPO。

共享这种思想不代表有相同的 group normalization、PPO clipping、KL 项或更新规则。后续 Lecture 09 再介绍 GRPO。

## 4. Learning to Model Editing Processes

该方法不要求每轮重写完整文本，而是先判断哪里需要变动，再生成变动内容。

### 4.1 操作与内容分开

对当前版本 $s_t$，预测操作标签 $e_t$：

- KEEP：保留当前位置。
- DELETE：删除当前位置。
- INSERT：在指定位置插入内容。
- REPLACE：替换指定片段。

再对需要新增文字的操作生成内容 $u_t$，最后由确定性函数应用修改：

$$
e_t\sim P(e_t\mid s_t,h_t),\quad
u_t\sim P(u_t\mid e_t,s_t,h_t),\quad
s_{t+1}=\operatorname{Apply}(s_t,e_t,u_t)
$$

这里 $e_t$ 只是“做什么、在哪里做”的标签，并未包含新文本；这正是课堂学生在 [16:37](https://www.youtube.com/watch?v=uaxf9yssDy4&t=997s) 追问的要点。

### 4.2 为什么需要单独生成内容

DELETE/KEEP 可以直接应用；INSERT/REPLACE 还不知道具体插入或替换成哪些词。

课堂以关于 domestic dog 和 wolf 的句子展示多个位置的局部修订。不同待填片段可并行生成，各自以 EOS 表示局部片段结束。

如果编辑标签已经包含最终文本，就不需要这一步概率分解；而原方法特意把结构决策与内容预测分开。

### 4.3 长文档上的效率意义

每轮重写长度为 $L$ 的文档，需要生成约 $L$ 个 token；如果只有 $m\ll L$ 个 token 要改，标签预测加局部生成可以减少串行 decode。

不过仍需读取和编码原文。节省的是完整重写的生成成本，不是“只需处理修改的几个字”。

### 4.4 编辑历史是额外条件

历史 $h_t$ 可包含旧版本、旧编辑和编辑意图。

例如某段 Wikipedia 内容被加入垃圾文本，之后很可能有人删除；只有当前版本无法充分表达这种动态，而历史提供了可预测下一步的线索。

这并不保证每次编辑都提高真实性：编辑战、回滚和垃圾内容都是数据本身的一部分。

### 4.5 数据如何获得

课堂介绍两类自然存在的监督来源：

| 来源 | before/after | 编辑意图线索 |
| --- | --- | --- |
| Wikipedia revisions | 相邻文章版本 | revision comment |
| GitHub revisions | commit 前后代码 | commit message |

模型可以学习单步编辑，也可利用过去若干步预测下一次变化。论文强调了多步编辑过程与数据构造，而不只是最终生成文本的质量。见 [原论文](https://arxiv.org/abs/2205.12374)。

**整理补充：** 如果建立同类数据集，应按文档或仓库划分训练/测试集，避免相邻版本同时进入两边造成泄漏。

## 5. Learning to Represent Edits

[19:46](https://www.youtube.com/watch?v=uaxf9yssDy4&t=1186s) 介绍把变化本身编码成向量：

$$
e=E_\phi(s_{before},s_{after}),\qquad
P_\psi(s_{after}\mid s_{before},e)
$$

训练用重构目标：

$$
\mathcal L=-\log P_\psi(s_{after}\mid s_{before},
E_\phi(s_{before},s_{after}))
$$

编码器和编辑器联合训练，因为 edit vector 连续且影响后续输出似然。

### 5.1 512 维瓶颈的用途

课堂特别指出 edit representation 被限制为 512 维，是为了鼓励模型表示变化类型，而非机械保存完整 after 文本。

它是归纳偏置，不是严格的信息论保证。实数向量和强解码器仍可保存复杂信息；是否学到可迁移编辑语义需要实验。

### 5.2 一次修改可以迁移给另一个输入

在新输入 $s'_{before}$ 上，使用相同 $e$ 生成 $s'_{after}$，尝试迁移“相同类型的修改”。

例如某种代码 API 更新、变量风格变换或安全缺陷修复，可能在不同代码片段中表现不同，但共享编辑意图。

### 5.3 评价的是表示，不只是编辑质量

课堂说明该工作主要检查“能否找出相似编辑”，并与 bag-of-words 表示比较。

部分评测使用代码转换工具构造已知类型的变更，测试同类变化是否在表示空间更接近。不要把相似编辑检索准确率误读成真实漏洞自动修复成功率。见 [Learning to Represent Edits](https://arxiv.org/abs/1810.13337)。

讲师还提出把这种表示思路用于分析 reasoning steps，是研究方向讨论，并非本讲已经验证的结果。

## 6. Self-Refine：同一模型的三个角色

[25:19](https://www.youtube.com/watch?v=uaxf9yssDy4&t=1519s) 开始讨论 training-free Self-Refine。

### 6.1 生成、反馈、修订

同一个 LLM 执行：

1. 按任务提示生成初稿。
2. 按特定质量维度评价初稿，给出反馈。
3. 读取原输入、初稿和反馈，生成改进版。
4. 根据反馈或迭代上限停止。

关键不在循环语法，而在反馈是否具体、可执行且与目标一致。

例如“代码不好”提供的方向很少；“变量名无法区分税前总额和税后总额，且缺少空列表处理”能指向具体编辑。

### 6.2 质量目标必须显式

课堂举例包括代码可读性、对话友好程度和情感调整。

同一输出可在多个维度上得分不同：提高简洁性可能损失完整性，提高礼貌程度可能增加冗余。

Critique prompt 应明确本轮优先改什么，并保留其他必须满足的约束。

### 6.3 为什么强模型更受益

反馈生成与修订本身都是任务。弱模型可能无法发现问题，或者找到问题后不会修复；更强模型能够更稳定地承担这两个角色。

课堂将“更大模型更有效”修正为“更强模型更有效”，避免把参数规模当成唯一决定因素。

### 6.4 数学题的收益为何不一定突出

讲师观察原工作在某些语言风格类任务上收益明显，但数学推理不如部分其他任务突出。

如果模型不了解某一步算术或逻辑，它重复审视也可能产生同一个错误；风格调整则通常更容易被模型自身判断。

这不是数学不能自纠错的普遍结论。模型训练、题目难度和可用反馈决定实际表现。参见 [Self-Refine 原论文](https://arxiv.org/abs/2303.17651)。

## 7. Self-Debugging：给修订加入执行观测

典型流程是：

```text
生成代码 → 执行/测试 → 获取结果 → 解释问题 → 修订代码 → 再执行
```

执行反馈提供模型单靠文字预测不一定知道的信息，例如异常、错误返回值或不通过的测试。

### 7.1 测试通过与没有异常不同

代码能够运行，不表示实现正确；通过若干公开测试，也不表示所有输入都正确。

要区分 syntax/runtime success、visible test success 和 held-out correctness。

若允许模型同时修改测试和答案，还必须保证评价契约没有被改掉，否则“通过”可能只是测试变容易。

### 7.2 并非所有任务都有 unit tests

课堂主要用单元测试解释外部反馈；原论文也讨论没有单元测试的 text-to-SQL 场景，用执行结果和自然语言解释进行检查。

因此 Self-Debugging 不能缩写为“失败测试修复”这一种形式。见 [Teaching Large Language Models to Self-Debug](https://arxiv.org/abs/2304.05128)。

### 7.3 为什么还要解释 stack trace

学生在 [30:31](https://www.youtube.com/watch?v=uaxf9yssDy4&t=1831s) 问：为什么不直接把 stack trace 传回模型？

讲师提出可能是为了压缩信息，并指出更新、更熟悉异常文本的模型未必需要同样步骤。这是当场解释和实现选择，不是原论文已证明的唯一原因。

工程上可以直接输入结构化执行结果，也可先压缩；压缩时不能丢掉报错行、期望值、实际值等关键证据。

### 7.4 与 self-consistency 怎样比较

课堂提到一种对照：self-consistency 采样 16 个候选，而 self-debugging 从一个初始候选出发，通过反馈修复。

“一个初始候选”不等于“一次调用”或“只生成一份代码”。多轮修订仍有模型与工具成本。

它的启示是：与其只增加独立候选数，也可以把预算用于从失败样本中获取信息。

## 8. Reflexion：把失败经验留给下一次

Reflexion 在评估当前尝试后生成反思，将其加入 episodic memory，随后新尝试能读取这些文字经验。

$$
m_{t+1}=\operatorname{UpdateMemory}(m_t,
\operatorname{Reflect}(x,y_t,feedback_t))
$$

与只看当前 critique 的循环相比，差别是反馈会跨尝试保留。

### 8.1 Verbal reinforcement learning 的含义

该方法通过文字反馈改变后续行为，不是在每次推理时对模型权重做 policy-gradient update。记忆进入 prompt，权重保持不变。见 [Reflexion 原论文](https://arxiv.org/abs/2303.11366)。

### 8.2 记忆应保留可复用的信息

有效记忆可包含：失败原因、约束、已排除方案、工具实际返回的事实。

“下次更仔细”通常不可操作；“上次把题目要求的闭区间误写成半开区间”则能改变下一次实现。

**工程补充：** 记忆有预算且可能含错结论，应保留来源与适用范围；随着历史增长，可以合并重复条目，但不能把未经验证的反思升级成事实。

## 9. CRITIC：让检查者主动使用工具

Self-Debugging 的课堂示例使用固定执行流程；CRITIC 更一般地让 critic 自己调用适合的工具。

例如用搜索或知识库核对事实，用代码解释器检查计算，用专门的分类 API 检查指定属性。

$$
c_t=C(x,y_t,\operatorname{ToolObservations}(x,y_t))
$$

工具结果再进入修订步骤，而不是只由初始 generator 使用。

### 9.1 工具在流程里的位置也是设计变量

| 位置 | 作用 |
| --- | --- |
| 生成前 | 补齐生成需要的事实与状态 |
| 生成中 | 随任务进展获取新信息或执行计算 |
| Critique 中 | 对已形成的具体断言进行有针对性的验证 |
| 修订后 | 确認最终候选满足约束 |

有针对性的检查有时比提前检索所有信息便宜；但草稿如果遗漏重要方面，critic 也可能没意识到该检查什么。

### 9.2 课堂术语修正

字幕把 Perspective API 说成检查“positive sentiment”。其典型用途是 toxicity 等评论属性检测，不应将 toxicity score 与情感正负混为一谈。原方法把不同工具配置到相应任务，见 [CRITIC 原论文](https://arxiv.org/abs/2305.11738)。

## 10. Intrinsic self-correction 为什么可能退步

[37:10](https://www.youtube.com/watch?v=uaxf9yssDy4&t=2230s) 介绍负面结果：只依赖模型自身能力、没有外部反馈的修订，在推理任务上可能无效甚至降低准确率。

主要失败方式包括：

- 生成与检查共享知识缺口，无法补出未知事实。
- 检查者把原来正确的步骤误判为错误。
- 修订过度迎合“你应该再改改”的提示。
- 确认偏差使模型继续替原判断辩护。
- 多步任务中，局部修复破坏此前已满足的其他约束。

论文针对其测试模型和 intrinsic setting 的结果，不能被提升为“所有语言模型永远不能自纠错”的定理。见 [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)。

### 10.1 必须同时统计修复与破坏

设初始正确率为 $a$，错误答案被修好的概率为 $r$，正确答案被改错的概率为 $d$。

一轮后的正确率：

$$
a'=a(1-d)+(1-a)r
$$

净收益条件是：

$$
(1-a)r>ad
$$

**整理示例：** 初始 $a=0.9$，能修好 30% 的错误，即 $r=0.3$；若改坏 5% 的正确答案，$d=0.05$，则 $a'=0.885$，整体反而下降。

这解释了为什么在很强的 baseline 上，“能修一些错”仍不足以证明 refinement 有用。

### 10.2 停止规则也会出错

模型说“没有问题”可能是假阴性；模型持续挑错可能是假阳性。最大轮数保证有限成本，却不保证返回最佳版本。

若有可靠外部 scorer，可保留历史最佳版本；如果没有，不能假装能够识别“最好的一版”，应如实报告选择规则。

## 11. AI Infra：循环背后的成本

### 11.1 单请求延迟通常串行累加

若进行 $K$ 轮修订：

$$
T\approx T_G+\sum_{t=0}^{K-1}(T_{critique,t}+T_{tool,t}+T_{refine,t})
$$

下一轮依赖上一轮结果，这部分很难像独立采样那样完全并行化。

可以并行执行独立检查项，但仍需等待足以支持本轮修订的结果。

### 11.2 全历史上下文可能快速变贵

若每轮都把之前所有长度约 $L$ 的版本和反馈放进 prompt，总输入 token 量可能随轮数约二次增长。

Prefix cache 能减少相同前缀的重复计算，但动态版本、编辑位置和截断策略会影响实际复用。

可考虑只保留当前版本和关键反馈、使用局部编辑、压缩已解决问题；这些优化需要确认不会丢失必要约束。

### 11.3 工具不是零成本函数

最后的课堂 Q&A 特别指出，代码 sandbox 比单纯调用 LLM API 多出部署、隔离、超时、依赖和故障处理。

因此是否加入工具由质量收益、延迟、费用和运维负担共同决定，不是看组件名称是否先进。

### 11.4 推荐记录的实验字段

| 字段 | 作用 |
| --- | --- |
| 初始/最终正确率 | 看总收益 |
| wrong→right、right→wrong | 判断修复和破坏的平衡 |
| 每轮质量曲线 | 检查过度迭代 |
| 每轮 input/output tokens | 记录真实模型成本 |
| 执行次数与失败类型 | 区分模型错误、工具错误 |
| 停止原因 | 成功、no-op、轮数、超时、预算分别计数 |
| 初稿与最后版本差异 | 确认约束没有被悄悄改变 |

### 11.5 教学控制循环

```python
def refine_loop(task, generate, critique, revise, max_rounds):
    current = generate(task)
    history = []
    for round_id in range(max_rounds):
        feedback = critique(task, current)
        history.append({"round": round_id,
                        "candidate": current,
                        "feedback": feedback})
        if feedback["stop"]:
            return current, "critic_stop", history
        updated = revise(task, current, feedback)
        if updated == current:
            return current, "no_change", history
        current = updated
    return current, "round_limit", history
```

这是流程示意，没有把 `critic_stop` 命名成 `correct`，因为模型主观通过检查不等于客观正确。

## 12. 课堂问答

### Q1：为什么监督目标固定，而中间序列要采样？

监督数据给出了想得到的最终 $y$，但没有给出唯一正确的草稿 $y'$。采样多个草稿能比较哪些中间态更利于产生同一个目标。

### Q2：编辑操作已经预测出来，为什么还需内容模型？

操作标签决定插入、删除、保留或替换的位置；新增文本尚未确定。只有把标签与内容合起来，才可确定下一版文本。

### Q3：edit encoder 与 decoder 是否联合训练？

是。编码器产生连续 edit vector，decoder 的重构似然依赖该向量，所以梯度可以反向更新两个部分。

### Q4：这些简单流程是否只对强模型有效？

讲师区分：早期弱模型方法通常需要专门训练；后来的强模型能通过 prompting 承担反馈和修订。若弱模型缺乏这种能力，训练可能要在流程中的某处补上。

### Q5：为什么不同时加显式 critique、长历史与所有工具？

讲师明确回答计算成本与 operational complexity。额外步骤需要提供足够收益，才值得支付服务延迟与系统复杂度。

## 13. 自测问题与面试回答

1. Self-refinement 的三个核心组件是什么？

    **面试回答：** 生成器产生初稿，检查者给出反馈或停止信号，修订器基于任务、当前版本和反馈生成新版本。三者可共享同一个模型，但有不同输入输出契约；控制器还负责轮数、预算与版本状态。

2. Explicit critique 与 implicit critique 怎样区分？

    **面试回答：** 显式方法单独生成问题描述或评分，隐式方法直接预测改写或编辑操作，可能用 no-op 表示停止。显式方法更容易观察定位，隐式方法可减少反馈生成成本，效果取决于任务和模型能力。

3. 双遍 decoding 给第二遍增加了什么信息？

    **面试回答：** 第二遍可以看到第一遍完整草稿，包括第一遍生成早期 token 时尚不可见的未来词，因此能利用全局一致性修正输出。它增加输入上下文和一次生成成本，不保证所有任务都受益。

4. 为什么中间草稿需要采样近似？

    **面试回答：** 中间草稿是离散序列潜变量，无限长时空间可数无限，限制长度后仍随词表和长度指数增长。采样少量草稿可估计目标或梯度，再通过最终目标条件似然判断草稿是否有帮助。

5. 为什么 $\mathbb E\log p$ 不等于 $\log\mathbb E p$？

    **面试回答：** 对数是非线性凹函数，Jensen 给出 $\mathbb E\log p\le\log\mathbb E p$。前者强调采样草稿下的平均 log-likelihood，后者对应边缘概率的对数；训练和推导时必须明确采用哪一个。

6. 局部编辑为什么可能比整篇重写快？

    **面试回答：** 标签模型可一次编码预测所有编辑位置，再只为插入/替换片段生成内容；多个片段还可并行。它减少串行生成量，但仍要读原文并维护位置映射，文档编码成本不会消失。

7. Self-Debugging 中一个初始样本是否等于一次推理调用？

    **面试回答：** 不等于。一个初始程序还可经历多次执行、解释和修订，每轮都消耗模型 token 与工具时间。与多候选方法比较时应统计完整流程成本，不能只比较初始样本数。

8. Reflexion 为什么称为 verbal RL，却不一定更新权重？

    **面试回答：** 它把反馈转换为反思文本并存入 episodic memory，后续尝试在 prompt 中读取这些经验。行为改变来自上下文状态，而非推理时做梯度更新，因此不能与参数层面的 policy optimization 混淆。

9. 如何判断一次修订是否有净收益？

    **面试回答：** 同时测错误修复率 r 和正确破坏率 d。初始正确率 a 时，修订后为 $a(1-d)+(1-a)r$；只有 $(1-a)r>ad$ 才净改善。高准确率 baseline 尤其要控制正确答案被改坏的比例。

10. 外部反馈为什么经常有帮助？

    **面试回答：** 它提供模型未必能内部可靠判断的信息，例如执行结果、测试反例和检索证据，使修订有更明确方向。但反馈自身也有覆盖率和噪声，测试通过不等于全局正确，工具失败也不应误判成答案失败。

11. 怎么设计停止条件？

    **面试回答：** 硬上限控制轮数、token 和时间，软条件可以是可靠测试通过、没有有效修改或边际收益不足。停止原因应明确记录，模型自评通过不等于正确；若有可靠 scorer，可保留历史最优版本防止回退。

12. 为什么不能把所有修订组件都默认打开？

    **面试回答：** 多轮调用增加串行延迟，长历史增加输入成本，工具增加基础设施和故障点。应按任务评估每个组件的边际质量收益，在相同预算下与直接生成、更多采样等方案比较。

## 14. 阅读地图与关联

| 论文 | 本讲对应内容 |
| --- | --- |
| [Deliberation Networks](https://papers.neurips.cc/paper_files/paper/2017/hash/c6036a69be21cb660499b75718a3ef24-Abstract.html) | 双遍解码与草稿潜变量 |
| [Learning to Model Editing Processes](https://arxiv.org/abs/2205.12374) | 多步编辑、操作与内容分解 |
| [Learning to Represent Edits](https://arxiv.org/abs/1810.13337) | 编辑向量、变化检索与迁移 |
| [Self-Refine](https://arxiv.org/abs/2303.17651) | 同一 LLM 生成、反馈、修订 |
| [Teaching LLMs to Self-Debug](https://arxiv.org/abs/2304.05128) | 代码执行、解释与程序修复 |
| [Reflexion](https://arxiv.org/abs/2303.11366) | 反思文字与 episodic memory |
| [CRITIC](https://arxiv.org/abs/2305.11738) | 检查者使用工具 |
| [LLMs Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798) | Intrinsic self-correction 的失败边界 |

- 课程索引：[CMU 11-763](CMU%2011-763.md)。
- 上一讲：[Lecture 07](CMU%2011-763%20-%20Lecture%2007%20-%20Chain%20of%20Thought%20and%20Intermediate%20Steps.md)。
- 下一讲：[Lecture 09](CMU%2011-763%20-%20Lecture%2009%20-%20Reasoning%20Models.md)。
- 主题入口：[LLM Inference](../../topics/inference/LLM%20Inference.md)。
