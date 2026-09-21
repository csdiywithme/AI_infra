---
type: course-note
status: developing
course: "[[CMU 11-763]]"
lecture: 11
lecture_date: 2025-09-30
area: inference
topics:
  - "[[LLM Inference]]"
aliases:
  - CMU 11-763 Lecture 11
  - Agents and Multi-Agent Communication
  - Agent 推理与多智能体通信
video_url: https://www.youtube.com/watch?v=ixLXrgF77ME
---

# Lecture 11：Agents and Multi-Agent Communication

> [!abstract] 本讲一句话
> Agent 是反复根据环境反馈选择行动的系统；其推理效率由动作粒度、观察表示、上下文保存方式与协作结构共同决定，不能只靠换一个更强的语言模型解决。

## 来源与范围

- 课程：CMU 11-664/763 — Inference Algorithms for Language Modeling，Fall 2025。
- 讲师：Graham Neubig；日期：2025-09-30。
- [课程视频](https://www.youtube.com/watch?v=ixLXrgF77ME)，约 1:00:43；正文覆盖完整英文自动字幕。
- [课程主页](https://www.phontron.com/class/lminference-fall2025/)与[日程](https://www.phontron.com/class/lminference-fall2025/schedule/)。
- 日程列有 HTML/PDF slides；整理时官网迁移后原课件文件未能取得，因此本讲按字幕与原论文校对，不声称逐页读取原 slides。
- 核对材料：[ReAct](https://arxiv.org/abs/2210.03629)、[CodeAct](https://arxiv.org/abs/2402.01030)、[SWE-Gym](https://arxiv.org/abs/2412.21139)。
- 讲师大量使用自己参与开发的 OpenHands 举例；产品名、模型速度、排行榜与服务数量均保留 2025-09 的历史语境，不作为当前产品比较。

> [!note] 本讲重点是 inference
> 讲师在开头明确：本讲不系统讲 agent 训练，也不全面比较架构。重点是 agent 特有的推理问题，包括 planning、environment representation、long context、evaluation、critic 与 delegation。文中公式和伪代码是帮助理解这些机制的工程整理。

## 视频时间索引

| 时间 | 内容 | 对应位置 |
| --- | --- | --- |
| [00:03](https://www.youtube.com/watch?v=ixLXrgF77ME&t=3s) | 作业发布与反馈安排 | 来源背景 |
| [01:59](https://www.youtube.com/watch?v=ixLXrgF77ME&t=119s) | Agent 定义与推理挑战 | 第 1 节 |
| [04:58](https://www.youtube.com/watch?v=ixLXrgF77ME&t=298s) | 应用类别 | 第 2 节 |
| [06:58](https://www.youtube.com/watch?v=ixLXrgF77ME&t=418s) | OpenHands 课堂演示 | 第 2.2 节 |
| [10:47](https://www.youtube.com/watch?v=ixLXrgF77ME&t=647s) | 通用 agent 与固定 workflow | 第 3 节 |
| [13:00](https://www.youtube.com/watch?v=ixLXrgF77ME&t=780s) | Planning 与 ReAct | 第 4 节 |
| [15:23](https://www.youtube.com/watch?v=ixLXrgF77ME&t=923s) | CodeAct 与动作粒度 | 第 5 节 |
| [17:49](https://www.youtube.com/watch?v=ixLXrgF77ME&t=1069s) | Task tracker 与持久化计划 | 第 6 节 |
| [20:31](https://www.youtube.com/watch?v=ixLXrgF77ME&t=1231s) | Thinking tool | 第 6.3 节 |
| [23:09](https://www.youtube.com/watch?v=ixLXrgF77ME&t=1389s) | 环境表示 | 第 7 节 |
| [25:54](https://www.youtube.com/watch?v=ixLXrgF77ME&t=1554s) | Accessibility tree | 第 7.2 节 |
| [26:41](https://www.youtube.com/watch?v=ixLXrgF77ME&t=1601s) | Screenshot 与 Set-of-Marks | 第 7.3 节 |
| [27:45](https://www.youtube.com/watch?v=ixLXrgF77ME&t=1665s) | 代码文件的有界读取 | 第 7.4 节 |
| [31:01](https://www.youtube.com/watch?v=ixLXrgF77ME&t=1861s) | 长上下文成本 | 第 8 节 |
| [32:17](https://www.youtube.com/watch?v=ixLXrgF77ME&t=1937s) | Prompt/KV caching | 第 9 节 |
| [33:46](https://www.youtube.com/watch?v=ixLXrgF77ME&t=2026s) | Context condensation | 第 10 节 |
| [35:25](https://www.youtube.com/watch?v=ixLXrgF77ME&t=2125s) | 忘记已创建 PR 的真实失败 | 第 10.2 节 |
| [36:16](https://www.youtube.com/watch?v=ixLXrgF77ME&t=2176s) | 网页历史压缩与元素筛选 | 第 10.3 节 |
| [39:57](https://www.youtube.com/watch?v=ixLXrgF77ME&t=2397s) | SWE-bench、WebArena、GAIA | 第 11 节 |
| [47:51](https://www.youtube.com/watch?v=ixLXrgF77ME&t=2871s) | Critic 与多轨迹 reranking | 第 12 节 |
| [52:24](https://www.youtube.com/watch?v=ixLXrgF77ME&t=3144s) | 为什么不直接生成测试 | 第 12.3 节 |
| [54:05](https://www.youtube.com/watch?v=ixLXrgF77ME&t=3245s) | Multi-agent 的动机与代价 | 第 13 节 |
| [58:30](https://www.youtube.com/watch?v=ixLXrgF77ME&t=3510s) | 为什么用文本而非隐藏状态通信 | 第 13.4 节 |
| [59:35](https://www.youtube.com/watch?v=ixLXrgF77ME&t=3575s) | AutoGen 等框架的定位 | 第 14 节 |

## 1. Agent 是闭环决策过程

本讲把 agent 定义为：为完成任务而迭代使用工具的系统。
一次工具调用可以是其中一个步骤，但 agent 的核心在于根据新观察继续做决策。
早期电话客服树、Siri/Cortana 等也有 agent 特征；agent 不等于 LLM 的同义词。

用本文的符号表示：

$$
h_t=(x,a_1,o_1,\ldots,a_{t-1},o_{t-1}),
$$

$$
a_t\sim\pi_\theta(\cdot\mid h_t),
\qquad (s_{t+1},o_t)=\operatorname{EnvStep}(s_t,a_t).
$$

$x$ 是用户目标，$s_t$ 是环境真实状态，$h_t$ 是模型看得到的历史。
这两个状态不相同：网页、文件、进程可能已经变化，而旧观察仍留在上下文中。
agent 必须用观察更新自己的工作判断。

一个任务成功，需要的不仅是最后一句回答合理，还包括它实际执行的动作是否达成目标。
例如“我已经修好了”是文本；代码差异和测试结果才是可核查的交付证据。

## 2. 应用与课堂演示

### 2.1 四类场景

| 场景 | 课堂例子 | 主要推理挑战 |
| --- | --- | --- |
| 软件开发 | 代码生成、调试、测试 | 阅读长代码、执行反馈、状态保存 |
| 网页自动化 | 提取信息、填表、测试 | 页面表示、元素定位、多步依赖 |
| 研究分析 | 信息搜集、报告生成 | 来源追踪、覆盖面、证据整合 |
| 交互环境 | 游戏、模拟、机器人 | 状态变化、规划、长轨迹 |

这些场景共享“动作—观察”接口，却不共享全部评价方式。
修复代码可以用测试，研究报告则需要检查事实与引用，机器人还涉及真实环境成功条件。

### 2.2 演示：寻找课程作业并估计耗时

讲师让 OpenHands 找到课程网站、下载 Homework 1，并估计完成作业的时间。
演示中能看到搜索、打开作业页面、克隆代码和阅读 PDF 等行为。
最后给出约 25–35 小时的估计，讲师计划与学生反馈对照。

这个数字不是经过验证的作业工时结论。
演示说明的是 agent 可以围绕一个开放任务临时选择工具序列。
学生还指出“是人完成，还是 agent 完成”会改变估计对象，体现任务描述可能缺失重要条件。

### 2.3 演示速度也是推理参数的结果

讲师临时切换模型，原因是原设置使用高 reasoning effort，课堂等待太久。
这是一组特定模型和配置的现场体验，不应扩展成当前模型速度排行榜。
可迁移的结论是：更长内部推理可能减少错误，也会增加每个动作前的等待。

## 3. 通用 agent 与固定 workflow

学生问为什么 OpenHands 没有直接使用 LangGraph/LangChain 一类框架。
讲师的设计理由是：软件工程任务路径差异很大，他们希望一个 agent 自己选择不同工具与行动顺序。
固定步骤之间的组织与多角色交接不是其最主要需求。

这段是架构取舍，不是说某个框架只能支持固定流程。
实际应先问：下一步由业务逻辑事先确定，还是必须根据环境动态决定？

例如：

```text
固定 workflow：读上传表格 → 校验字段 → 写数据库
动态 agent：定位报错 → 选文件 → 修改 → 运行测试 → 根据失败继续
```

前者可清楚定义每步输入输出；后者的探索深度和路径难以提前列举。
两者还可以组合：固定外层流程调用一个负责开放子任务的 agent。

## 4. ReAct：把推理与行动交织起来

### 4.1 Agent planning 与数学推导不同

数学题常把给定信息组合起来推导结论。
agent planning 更常问：为了整体目标，现在最值得做什么？
新动作可能获得此前没有的信息，因此计划必须随观察调整。

ReAct 的课堂结构是：

```text
Thought → Action → Observation → Thought → Action → ...
```

这里的 Thought 是模型生成的推理文本，不是一个单独执行的环境操作。
Action 改变或查询环境，Observation 提供实际反馈。
[ReAct 原论文](https://arxiv.org/abs/2210.03629)强调这两者相互补充：推理帮助更新计划，动作提供外部信息。

### 4.2 为什么不只生成动作

课堂给出两个理由：提高任务表现，以及让人理解系统当前在做什么。
例如“找到课程主页，接下来检查 assignments 页面”比只显示一串 click call 更易阅读。
可理解的进度说明能帮助使用者判断目标是否偏离。

但推理说明不是动作正确性的证据。
模型可能说“测试通过”，实际测试工具却没有成功执行。
工程上仍要保存实际调用与结果，避免让自然语言自述替代观察记录。

### 4.3 ReAct 的计划粒度

课堂称 ReAct 通常比较 reactive：一次推理主要决定下一步。
这容易对新反馈作出反应，但不天然维护完整、长期、可恢复的计划。
因此后面引入 task tracker，而不把一次 Thought 当作持久化项目状态。

## 5. CodeAct：把多个动作编成程序

### 5.1 课堂的十个商品价格例子

若要查询 10 个商品并找出最贵的一个，逐调用模式需要获取多个价格，再进行聚合。
CodeAct 允许生成程序，通过循环调用工具并计算最大值。

下面是本文的同类示意：

```python
prices = []
for item in items:
    result = get_price(item)
    prices.append((item, result["price"]))
most_expensive = max(prices, key=lambda pair: pair[1])
print(most_expensive)
```

这将动作空间统一为可执行程序，能直接利用变量、循环、条件与库函数。
程序还可以先过滤、聚合结果，再把小量信息交给 LM。
核心机制与 [CodeAct](https://arxiv.org/abs/2402.01030)一致。

### 5.2 一大步与许多小步之间的取舍

大程序减少 LM 往返，但会一次做出更多尚未验证的假设。
若前面一步返回的数据格式与预期不同，后面的所有操作可能一起失败。
小程序增加等待和 token 成本，却能更早得到反馈。

课堂明确建议把动作块大小作为 inference 取舍，而不是无条件认为程序越长越高效。
适合批量的情况包括：接口稳定、数据依赖清楚、失败可恢复。
适合分步的情况包括：新环境探索、返回格式未知、下一步取决于观察。

### 5.3 系统侧的简化成本模型

设每次模型往返开销为 $c_m$，原本需要 $k$ 次决策。
将这些决策合并成一段程序，可减少约 $(k-1)c_m$ 的往返开销。
但应扣除更长代码生成、失败重试和执行等待增加的成本。

这说明优化目标不是“最少工具调用数”，而是给定成功率下的总延迟与总成本。

## 6. 计划如何在长任务中保存

### 6.1 Task tracker 让计划成为结构化状态

课堂演示一个工具，维护任务项及其状态，例如 pending、in progress、done。
完成一步后再次调用工具更新状态，并保留其他任务项。
这样最近的上下文会重新包含整体计划。

计划还会写入磁盘上的文件，例如 `tasks.md`。
这使它不只依赖模型当前 context；即使历史被截断或压缩，也可以重新读取。

这里有两个独立收益：

- 近端上下文刷新，降低“忘记整体目标”的风险。
- 外部持久化，让状态能够跨上下文窗口恢复。

### 6.2 完成状态应与证据关联

由课堂机制得到的工程延伸是：不要让 `done` 只表示“我想过这一步”。
它应尽量关联文件、结果或测试证据。
否则 task tracker 只是把没有依据的自我评价结构化了。

一个可复用任务记录至少包含目标、状态、产物位置、关键结果和未解决问题。
任务变更时还要保留用户新约束，避免压缩过程把旧目标重新当成当前目标。

### 6.3 Thinking tool 为什么可以没有外部效果

课堂的 thinking tool 允许模型额外写一段长推理，再在下一轮采取行动。
它本身不查询或修改环境，价值是给没有专门训练长 reasoning 的模型增加推理空间。
因此它是一种 inference-time scaffold。

对已训练的长推理模型，讲师认为额外 thinking tool 可能不再必要。
但持久化 task planning 仍可能有价值，因为“会推理”与“跨窗口保存状态”是两件事。

## 7. Environment representation 决定模型能看到什么

### 7.1 纯文本与 Markdown

把网页转换为 Markdown 可以移除大量样式与脚本，留下文本、标题和部分链接。
这适合信息抽取和阅读，通常比原始 HTML 紧凑。

但普通 Markdown 不天然表达“这个输入框可以填写”“这个按钮对应哪个可点击元素”。
一个阅读接口与一个交互接口需要的信息不同。
若 agent 必须行动，就要保留可定位的交互元素。

### 7.2 Accessibility tree

Accessibility tree 原本服务于辅助技术，能提供语义角色、名称、层次和交互属性。
agent 系统可为元素分配 ID，让模型通过 ID 调用点击或输入工具。

```text
[17] textbox "Search"
[18] button "Submit"
[19] link "Assignments"
```

这只是本文示意，不是课堂页面原文。
与原始 DOM 相比，它把模型注意力放到交互相关语义上。
但每次页面变化后 ID 和元素状态可能改变，旧观察不能无限复用。

### 7.3 Screenshot 与 Set-of-Marks

截图保存空间布局、颜色、图像和视觉信息。
Set-of-Marks 在截图上的可交互元素周围画框并编号，同时提供文本对应信息。
模型因此可以用视觉理解位置，再通过标记选择动作。

课堂学生指出编号可能重叠，也可能让模型过分依赖文字。
讲师回答：效果依模型与训练而异，已见实验中 marks 的帮助超过干扰，但不是最终最优表示。
这应被理解成经验结果，不是任何页面都适用的保证。

### 7.4 文件读取也有表示问题

讲师补充：不要误读一个一千万行文件，把 context window 填满。
代码工具应支持指定行范围、搜索匹配、定位函数，再读取相关区域。
信息选择不仅是节省 token，也会影响模型能否看到真正相关的代码。

## 8. 长轨迹如何变成高成本

课堂提到 benchmark 轨迹可以达到约 100 次 action/observation，实际任务可能更长。
每一步都累积工具返回，尤其网页树、截图和文件内容可能很大。
上下文可容纳并不意味着经济上值得完整保留。

假设每轮新增 $m$ 个 token，共 $K$ 轮。
若每轮重新发送并重新计算全部历史，累计输入 token 数近似：

$$
\sum_{t=1}^{K}tm
=m\frac{K(K+1)}2.
$$

这是累计输入量的简化模型，不是 Transformer 总 FLOPs 的精确公式。
它说明：同样的早期内容可能反复计费或 prefill，轮数会放大成本。

优化路径有两个：保留内容但复用计算，或减少保留的内容。
下一节的 caching 与 condensation 分别对应这两个方向。

## 9. Prompt caching：复用未变化的前缀

### 9.1 为什么自回归模型可以复用历史表示

在 causal attention 下，早期 token 的表示不依赖未来新增 token。
若前缀完全相同，之前计算的 K/V 可被后续请求复用。
下一轮主要处理新 observation，并生成新的 action。

```text
第 1 轮：system + observation_1 → action_1
第 2 轮：[可复用前缀] + observation_2 → action_2
第 3 轮：[更长的可复用前缀] + observation_3 → action_3
```

课堂指出，具体服务未必缓存所有已生成 action；这属于实现差异。
不要把理论可缓存等同于某个 API 的当前计费与缓存策略。

### 9.2 Caching 不是消除长 context

复用前缀可以减少重复 prefill，但保留长 K/V 仍消耗显存或外部缓存容量。
新 token 也仍需与历史交互，长上下文不会完全免费。
缓存保存的是计算，不自动判断内容是否仍然有用。

在工程中，稳定 system prompt 和追加式历史通常有利于前缀复用。
修改早期消息、随机重排工具说明或替换旧 observation，可能使后续前缀缓存失效。

## 10. Context condensation：删掉内容也会删掉记忆

### 10.1 对历史做摘要

课堂介绍把较早的一部分步骤交给 LM 总结，以短摘要替代冗长原历史。
它可以保留目标、做过的尝试和关键结论，同时减少输入量。
讲师报告 OpenHands 实验中出现约两倍或更多成本降低，并能保持 SWE-bench 表现。

该结论依赖特定实验，不是任意摘要 prompt 的保证。
摘要模型可能遗漏以后需要的信息，尤其是当摘要时还不知道未来会问什么。

### 10.2 已创建 PR 却忘记的失败故事

课堂给出的真实失败是：agent 提交了 PR，压缩后忘记已经提交，于是再次创建 PR。
benchmark 中表现良好的摘要方法，放到真实产品后暴露了动作记忆缺失。

因此摘要不能只保存“讨论了什么”，还要保存“已经做了什么”。
尤其要保留有外部副作用的动作及其标识，例如 PR URL、提交 ID、已发送消息、已创建资源。
这些信息是避免重复执行的依据。

> [!tip] 工程延伸
> 可以把事实、计划与外部动作日志分开保存：摘要承载任务语义，结构化日志承载已执行操作。这样即使摘要漏掉一句话，执行器仍有条件识别重复动作。

### 10.3 只保留最新网页观察

一种常见办法是只保留最近的 accessibility tree 或 screenshot，删除更早的完整页面观察。
这样减少大量重复、过时网页内容。

代价有两个：

1. 旧页面上的证据可能丢失，未来要重新查询。
2. 删除历史中间内容改变 token 前缀，prompt caching 可能更不有效。

所以“越短越省”不是完整答案；还要计算缓存命中下降与重复检索的代价。

### 10.4 在进入 LM 前筛选网页元素

讲师补充 Mind2Web 的一个思路：用较小、较便宜模型挑出相关元素及其父节点，只把这些部分交给大模型。
它减少首次处理大页面的成本，而不仅是删除旧页面。
保留父节点是为了保留元素所在的结构语境。

该步骤本身可能漏掉关键元素，因此必须考虑筛选召回率。
若目标元素没进入大模型输入，后面的强模型也很难补救。

### 10.5 缓存与压缩的决策不能独立优化

以下是本文的简单决策框架：

$$
\text{压缩净收益}=
\text{未来输入节省}
-\text{摘要成本}
-\text{缓存失效成本}
-\text{信息丢失导致的重试成本}.
$$

不同工作负载下这些量不同。
一个两轮任务可能不值得摘要；一个上千步任务则必须有记忆管理。

## 11. Agent evaluation：要评价环境中的结果

### 11.1 SWE-bench

输入是 issue 描述与对应代码库状态，输出是修复补丁。
评价检查新增的故障相关测试，以及一部分原有测试。
目标既包括修复问题，也包括不破坏此前正常行为。

SWE-bench Verified 通过人工筛选减少题目描述不足或不可解的情况。
课堂强调：在 Verified 上高分不等于解决全部现实软件问题。
真实 issue 可能含糊，需要与用户澄清，而该能力在筛选后的集合中体现较少。

### 11.2 WebArena

WebArena 使用可控制的真实风格网站，例如购物、内容管理、论坛和 GitLab。
任务包含多步交互，例如统计购买记录或创建带特定 README 的仓库。
真实网站结构使操作更接近实际，但沙箱环境允许固定初始状态与检查最终结果。

### 11.3 GAIA

GAIA 是偏向研究与信息查找的通用助手评价。
任务可能需要网页检索、读文档、处理多模态信息，再组合得到答案。
它与代码 benchmark 测量的能力不同，不能用一个排行榜覆盖所有 agent 场景。

### 11.4 评价为什么昂贵

运行 agent 需要多次 LM 调用、工具等待、环境启动和可能的测试执行。
同一题还可能因为模型随机性或外部服务而有显著方差。
合理比较应固定任务、预算、初始环境和允许工具，并记录成功率与成本。

课堂也推荐通过环境集合发现研究任务；本文不把某个仓库当时包含的 benchmark 数量当作长期事实。

## 12. Critic：生成后再给第二次判断机会

### 12.1 从完整轨迹中选择

critic 的两个课堂用途是候选 reranking，以及安全或表现审计。
生成 $N$ 条轨迹后，使用 critic 给它们评分并选择：

$$
\tau_1,\ldots,\tau_N\sim\pi_\theta,
\qquad
\tau^*=\arg\max_{\tau_i}r_\phi(x,\tau_i).
$$

这里一条候选不只是最终答案，而可能包含数十次动作和观察。
因此 agent best-of-N 的生成成本远高于短文本 best-of-N。

课堂以 SWE-Gym 举例，展示 rollout 数增加时 resolve rate 上升，最高达到约 32%。
[论文摘要](https://arxiv.org/abs/2412.21139)对应 SWE-bench Verified 32.0% 的历史实验结果，不是当前系统能力保证。

### 12.2 Critic 如何训练

课堂的最简单流程：生成完整轨迹，用现有测试判断是否成功，训练模型预测成功与否。
最终结果标签训练的是 outcome reward model。
若希望评价每一个中间步骤，则需要 process reward model，以及更难收集的步骤监督。

完整轨迹打分可以读到行动后的证据，这有助于识别看似合理但实际上失败的过程。
但 critic 也可能被漂亮的解释误导，因此应检查它究竟使用了哪些证据。

### 12.3 为什么不直接让模型生成单元测试

学生提出：生成测试再运行，可能比“从轨迹预测成功”更独立于 agent 实现。
讲师认可这是值得研究的方向，但指出模型并不总能生成好测试，且研究类任务并不容易单测化。

还应区分训练时已有的 gold test 与部署时可获得的测试。
如果部署时确有可靠、完整的判定器，可以直接使用；训练集有判定器并不意味着新任务也有同样标签。
这正是学习 critic 的应用空间。

## 13. Multi-agent：收益主要来自什么

### 13.1 Specialization 的论证需要谨慎

常见类比是软件公司有产品、实现和 QA，所以 agent 也应分成多个专业角色。
讲师对仅凭这一类比的动机较怀疑，因为同一强 LM 可能已经掌握多种技能。
给相同模型换角色 prompt 不一定创造新的知识或独立判断能力。

专业化仍可能有意义，但应通过任务表现、工具权限或专用数据说明，而不能只靠组织结构类比。

### 13.2 并行化是更直接的理由

独立研究问题可分给多个 agent，同时搜索与阅读，再汇总结果。
这缩短墙钟时间，并可能扩大覆盖面。
软件开发也能并行，但共享代码依赖与修改冲突使任务拆分更困难。

若子任务耗时为 $T_1,\ldots,T_k$：

$$
T_{\rm serial}\approx\sum_iT_i,
\qquad
T_{\rm parallel}\approx\max_iT_i+T_{\rm coordination}+T_{\rm merge}.
$$

只有协调和整合成本不吞掉收益，并行才有优势。
总 token 成本通常不会因为墙钟时间下降而自动下降。

### 13.3 Robustness 与通信损失

第二个 agent 可以审查第一个 agent 的结果，类似 critic。
但同模型、同上下文、同提示可能产生相关错误，不能把“两个都同意”直接看成独立验证。

课堂最具体的失败是：独立浏览 agent 找到许多信息，返回时只给简短摘要，编码 agent 缺少完成任务需要的细节。
主 agent 不是没查过信息，而是交接时丢了信息。

因此 delegation contract 应明确：要回答什么、需要哪些原始证据、结果怎么引用、尚不确定什么。
只给一句“已经研究过，可以实现”几乎没有可复用价值。

### 13.4 为什么通常传文本

学生问能否直接传 logits 或更深层 hidden states，减少文本摘要损失。
讲师认为可能值得研究，但闭源模型 API 通常不暴露这样的接口。

进一步的工程问题是：不同模型的 hidden states 不一定具有兼容表示。
文本虽然可能丢信息，却便于跨模型、跨工具、跨人类协作，并能审计。
本文把后一点作为推导，没有将其写成课堂原话。

## 14. 框架不是任务分解的替代品

AutoGen 等框架可提供 agent 定义、对话模式、工具接入和人类交接等组织能力。
课堂认为它们对可明确表达的工作流很有帮助。
但框架不会自动决定某个复杂任务该怎么拆分，也不会保证传递的上下文足够。

实际选择时应先画出依赖关系：哪个子任务需要哪份输入、会修改哪些资源、输出由谁验证。
当多个任务必须不断读写同一状态时，增加 agent 数量可能放大同步和通信成本。

## 15. AI Infra 实现检查：几个关键接口

以下伪代码是本文的概念实现，不是某个 agent SDK 的当前 API：

```python
while not task_finished:
    history = memory.prepare_context(goal, plan, action_log)
    action = policy.next_action(history, allowed_tools)
    result = runtime.execute(action)
    action_log.append(action, result)
    memory.add_observation(result)
    plan.update_from_evidence(action, result)
    if memory.should_condense():
        memory.condense(preserve=action_log.external_effects())
```

这个循环中，每层应承担清楚的职责：

- Policy 选择下一步，但不伪造实际结果。
- Runtime 执行、限制资源、返回结构化成功或失败。
- Memory 控制输入大小，但保留影响后续行动的重要状态。
- Plan 表示目标进度；action log 表示已经发生的事实。

需要监控的不只是 token/s，还包括每任务轮数、tool latency、缓存命中、压缩次数、重复动作和恢复成功率。
agent 最慢的一部分可能是测试、网络或等待环境，不一定是 LM decode。

## 16. 自测与面试回答

### 1）Agent 与一次 function calling 的区别是什么？

**回答：** 一次 function calling 是单步动作接口；agent 反复根据观察选择下一步，直到完成目标或达到停止条件。因此需要维护历史、环境状态、计划与失败恢复。不能仅凭有工具字段就说明系统具有可靠的多步行动能力。

### 2）ReAct 为什么有帮助，又不能保证什么？

**回答：** 它交织推理、行动和观察，使模型能基于新信息更新计划，也能向使用者解释进度。但推理文本不一定忠实反映执行事实，正确性必须核查真实工具结果和最终环境状态。

### 3）CodeAct 何时比逐个函数调用更高效？

**回答：** 当多个调用具有稳定结构、可用程序循环或聚合，且不用在每一步等待 LM 决策时，它能减少模型往返与中间输出。探索未知环境时，大程序可能放大假设错误，所以动作粒度应按反馈需求选择。

### 4）Task tracker 与长 chain of thought 有什么区别？

**回答：** Chain of thought 是当前生成中的推理文本；task tracker 是明确的任务状态，可更新、持久化并重新读取。一个擅长长推理的模型仍可能在上下文压缩后丢失计划，外部状态管理解决的是不同问题。

### 5）为什么网页转 Markdown 不足以支持网页操作？

**回答：** Markdown 适合阅读，却不天然编码输入框、按钮状态和可定位元素。操作任务需要 accessibility tree 的语义与 ID，或截图及 Set-of-Marks 的视觉定位。表示应保留任务所需信息，而不是只追求文本最短。

### 6）Prompt caching 与 context condensation 分别优化什么？

**回答：** Caching 保留内容并复用未变化前缀的计算；condensation 通过删减或摘要减少内容。压缩可能破坏缓存，也可能丢失证据，因此应一起比较净成本与成功率，而不能单独最大化压缩率。

### 7）摘要为什么会让 agent 重复创建 PR？

**回答：** 摘要可能保存了“需要实现的任务”，却漏掉“已经执行过创建 PR”及其 URL。后续 agent 依据不完整状态再次操作。修复方向是明确保存外部副作用日志与资源标识，而不仅是提高摘要语言质量。

### 8）SWE-bench Verified 高分为什么不等于解决全部软件工程？

**回答：** 它测量特定代码库和筛选过的问题，主要通过测试验证补丁。现实工程还包括含糊需求、人类澄清、长期维护、部署与跨系统约束。评价集合覆盖的能力与真实任务整体必须分开。

### 9）Agent critic 为什么需要完整轨迹？

**回答：** 最终文字可能看不出工具是否失败、测试是否运行、修改是否覆盖需求；轨迹包含更多行动证据。代价是输入很长，也可能让 critic 被过程叙述干扰。应验证评分依据是否与实际成功相关。

### 10）Multi-agent 的并行收益如何判断？

**回答：** 看子任务是否可独立执行，以及协调、交接和合并成本。墙钟时间可以从耗时之和接近最长子任务耗时，但总 token 与工具调用可能增加。共享状态频繁冲突时，单 agent 或更少 agent 可能更有效。

### 11）多 agent 通信中最容易低估什么？

**回答：** 不是消息能否发送，而是结果是否保留了下游决策需要的证据。过短摘要会丢事实，完整历史又增加成本和噪声。任务交接应明确答案、证据位置、限制与不确定性，并允许按需追问细节。

### 12）为什么不能把同模型多个 agent 当作独立验证器？

**回答：** 它们可能共享训练偏差、提示和上下文，错误高度相关。多次同意只增加一致性，不必然增加真实正确性。更强的验证通常来自测试、独立来源、不同检查方法或可观察的环境结果。

## 与前后课程的联系

[[CMU 11-763 - Lecture 10 - Incorporating Tools|Lecture 10]]建立了工具接口与执行边界，本讲把它们扩展为长时间闭环。
[[CMU 11-763 - Lecture 12 - Reward Models and Best-of-N|Lecture 12]]继续形式化 critic 背后的 reward model，以及多候选生成和选择的概率与成本问题。
