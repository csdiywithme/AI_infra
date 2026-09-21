---
type: course-note
status: developing
course: "[[CMU 11-763]]"
lecture: 6
lecture_date: 2025-09-11
area: inference
topics:
  - "[[LLM Inference]]"
aliases:
  - CMU 11-763 Lecture 06
  - Other Controlled Generation Methods
  - Controlled Generation
video_url: https://www.youtube.com/watch?v=i4COjX4z1zY
---

# Lecture 06：Other Controlled Generation Methods

> [!abstract] 本讲一句话
> 形式约束可以把不合法 token 从支持集中删除；语义约束通常需要预测未来属性、奖励或模型之间的概率差。两者都能通过解码时改分数实现，但只有前者在满足完整假设时能提供形式合法性保证。

## 来源与范围

- 课程：CMU 11-664/763 — Inference Algorithms for Language Modeling，Fall 2025。
- 讲师：Amanda Bertsch；日期：2025-09-11。
- [课程视频](https://www.youtube.com/watch?v=i4COjX4z1zY)，时长 1:04:25；本文完整核对英文字幕。
- [官方 Slides，共 43 页](https://docs.google.com/presentation/d/1GG-sHP5KOClE3FnM7k7cwF4yjecTL4veMBgouWfyuxI/edit)；页码按官方 PDF，公式图与 FUDGE 数字例子另做视觉核对。
- [课程主页](https://www.phontron.com/class/lminference-fall2025/)。
- 主要论文：[FUDGE, Yang & Klein (2021)](https://arxiv.org/abs/2104.05218)、[Contrastive Decoding, Li et al. (ACL 2023)](https://aclanthology.org/2023.acl-long.687/)。
- 课堂简要关联：[RL with KL penalties is better viewed as Bayesian inference](https://arxiv.org/abs/2205.11275)、[Reward-Augmented Decoding, Deng & Raffel (2023)](https://arxiv.org/abs/2310.09520)。

> [!note] 阅读边界
> 主线与例子来自完整字幕和课件；“推导补充”“实现补充”用于澄清数学与工程条件。课堂展示的库、API 及模型输出按 2025 年背景理解，不是当前接口兼容性保证。RAD 的学生报告不在本录像里，不能把原论文的全部内容写成课堂已讲过。

> [!warning] 三个不能照抄的简化
> “不能逐 token 验证”可能只是当前局部判据不足，不代表无法构造带状态的约束器；“JSON 要什么自动机”取决于嵌套深度与 schema；“不能约束有效程序”也不代表不能用语法、符号表或编译器辅助约束代码。

## 视频时间索引

| 时间 | 内容 | 课件 |
| --- | --- | --- |
| [00:04](https://www.youtube.com/watch?v=i4COjX4z1zY&t=4s) | Syntactic 与 semantic constraints | 1–3 |
| [02:17](https://www.youtube.com/watch?v=i4COjX4z1zY&t=137s) | 完整结果可验证与逐 token 控制 | 4–6 |
| [05:32](https://www.youtube.com/watch?v=i4COjX4z1zY&t=332s) | 模板、前后缀、JSON | 7–10 |
| [10:21](https://www.youtube.com/watch?v=i4COjX4z1zY&t=621s) | Taylor Swift schema 与状态机 | 11–14 |
| [15:03](https://www.youtube.com/watch?v=i4COjX4z1zY&t=903s) | Token 边界与 token healing | 15–18 |
| [24:35](https://www.youtube.com/watch?v=i4COjX4z1zY&t=1475s) | llama.cpp、schema、框架接口 | 19–24 |
| [30:52](https://www.youtube.com/watch?v=i4COjX4z1zY&t=1852s) | 编程语言与自动机表达能力 | 25–27 |
| [39:11](https://www.youtube.com/watch?v=i4COjX4z1zY&t=2351s) | 合法性与 LM 概率如何结合 | 26–27、问答 |
| [41:13](https://www.youtube.com/watch?v=i4COjX4z1zY&t=2473s) | Logit mask 的实现 | 问答 |
| [45:50](https://www.youtube.com/watch?v=i4COjX4z1zY&t=2750s) | “不建议攀岩”的语义限制 | 28–32 |
| [49:27](https://www.youtube.com/watch?v=i4COjX4z1zY&t=2967s) | FUDGE 的 Bayesian 分解 | 33–36 |
| [55:39](https://www.youtube.com/watch?v=i4COjX4z1zY&t=3339s) | 与 RLHF、reward decoding 的关系 | 37–39 |
| [57:44](https://www.youtube.com/watch?v=i4COjX4z1zY&t=3464s) | Contrastive decoding | 40 |
| [59:41](https://www.youtube.com/watch?v=i4COjX4z1zY&t=3581s) | Safe / unsafe 模型对比 | 41–42 |
| [1:01:56](https://www.youtube.com/watch?v=i4COjX4z1zY&t=3716s) | 训练控制与推理控制的取舍 | 43 |
| [1:03:24](https://www.youtube.com/watch?v=i4COjX4z1zY&t=3804s) | 用 embedding 构造属性分数的问答 | 课后讨论 |

## 1. Controlled generation 到底在控制什么？

给定模型分布 $P_\theta(y\mid x)$，希望输出既符合模型，又满足额外属性 $a$。
属性可能是格式、词语、风格、内容禁忌或任务偏好。

本课已见过几种“改变原始最高概率目标”的方法：

- Length penalty：改变长短偏好。
- Diverse beam search：考虑候选之间的差异。
- Neurologic A*esque：用约束相关的未来估计安排搜索。

本讲进一步区分：能写成明确规则的约束，以及难以直接形式化的语义偏好。
这一区分决定了该用确定性检查器、预测器还是额外模型。

## 2. 完整可验证，不等于当前 token 已保证成功

### 2.1 三个问题要分开问

设完整输出的验证器为 $C(y)\in\{0,1\}$。

1. **最终验证**：给完整 $y$，能否判断 $C(y)=1$？
2. **前缀可行性**：给 $u$，是否仍存在合法完成 $z$，使 $C(uz)=1$？
3. **未来成功概率**：若按模型继续生成，有多大概率最终满足约束？

这三者不是同一个量。
一个 token 没有立即违规，不意味着模型以后一定会满足要求。
形式约束器通常负责第二个问题；FUDGE 更接近第三个问题。

### 2.2 课堂的 lexical constraint 例子

[03:41](https://www.youtube.com/watch?v=i4COjX4z1zY&t=221s) 要求输出包含 car、drive、snow。
在类似 “I drive my car during the ...” 的前缀后，模型可能更偏好 summer。
但 winter 可能更容易自然地引出 snow。

最终扫描是否含三个词很容易；当前一步概率却未必体现未来完成难度。
这解释了 lookahead 的价值。
“仍然可能出现 snow”和“接下来大概率自然出现 snow”是不同信号。

### 2.3 对“不可逐 token 验证”的精确补充

恰好生成 10 个 token 可以维护计数器，并控制何时允许 EOS。
固定词语覆盖也能用匹配自动机和已覆盖集合追踪，再结合剩余长度预算。
因此不能把课堂的局部直觉解释为这些约束理论上无法逐步实施。

不过，词语是否作为独立单词出现、大小写、tokenization、多词短语和长度预算都会影响规则。
没有把这些状态纳入检查器时，仅看当前 token 确实无法保证最终满足条件。

## 3. 模板约束：先问是否有必要训练

课件 7 页列出：固定问候开头、固定结束语、始终输出合法 JSON。
这些规律可以训练进模型，但若规则易验证且频繁变化，推理时控制通常更灵活。

### 3.1 静态前后缀与模型生成不同

如果产品只需要在 UI 末尾展示固定 footer，程序附加文本就足够。
如果希望后续正文与固定开头一致，则应把前缀纳入模型上下文。
把前缀隐藏地插到已经生成的正文前面，不能让模型追溯性地条件化于它。

同理，强制模型采出某段后缀，和输出完成后在外部拼接后缀，是两个接口契约。
要按任务决定哪些字符属于模型生成内容、哪些属于展示层。

### 3.2 Prompt 是要求，不是形式保证

[07:53](https://www.youtube.com/watch?v=i4COjX4z1zY&t=473s) 展示要求 JSON 后仍出现额外说明文字的例子。
补充更明确的 prompt 可能改善遵循，但不能由一次成功就得到“永远满足 schema”的保证。

还要区分：

- 能被 JSON parser 解析。
- 满足指定 key、类型、必填项及枚举值。
- 字段内容在事实和业务上正确。

三层正确性由不同机制负责。
有效 JSON 可以包含错误出生年份；schema 合法也不意味着答案真实。

## 4. 从 schema 到状态机

### 4.1 课堂例子

课件 11–14 页用 Taylor Swift 出生于 1989 年的信息，要求生成：

```json
{"name": "Taylor Swift", "birth year": 1989}
```

这是课堂给定信息的格式化例子，不是要求模型自行检索人物事实。
字段约定为 `name: string`、`birth year: int`。

状态机追踪当前位置：是否已生成 `{`、当前应生成哪个 key、是否处于字符串或整数中，以及何时可以闭合对象。
接受态表示完整文本满足规则；中间态允许继续，但不一定允许停止。

### 4.2 课堂找出的漏洞

[12:08](https://www.youtube.com/watch?v=i4COjX4z1zY&t=728s) 之后学生指出：

- 同一个 key 可能重复。
- 可选路径可能跳过必填字段。
- 简化字符串规则可能排除空格、转义等合法内容。
- 简单 FSA 不能表达任意深度嵌套。

要修复固定 schema，应追踪已输出 key、剩余必填项、字符串转义和合法数字规则。
不能把教学图中若干字母自环当成完整 JSON 实现。

### 4.3 无序字段会产生状态复杂度

若有 $K$ 个字段，可任意顺序出现且每个至多一次，“已出现字段集合”最坏有 $2^K$ 种组合。
这是理论状态规模，不要求实现立即展开所有状态。
可以动态维护 bitset、按需构造 parser 状态或采用固定字段顺序减少分支。

课堂提到几百个字段时，不需要手绘巨大图；schema 编译器负责生成或维护相应结构。
但自动生成规则，不等于运行代价恒定或所有 schema 特性都得到支持。

## 5. 字符规则如何映射到 token？

LM 词表的单位通常不是单个字符。
一个 token 可能包含多个字母、空格、引号、冒号或括号，也可能只对应部分字节序列。

因此合法性检查应把候选 token 的完整表示依次送入 parser 状态。
只有整个 token 消费完后仍处于可完成的状态，候选才合法。
不能只检查 token 的第一个字符，或只按表面字符串包含某个符号做过滤。

### 5.1 Grammar 状态与模型状态并行推进

实现上至少有：

- 模型前缀与 KV cache。
- 已输出文本／字节。
- Grammar 或 schema parser 状态。
- 接受／可停止标志。

每次选 token 后，两种状态必须同步更新。
EOS 通常只在接受状态允许；达到 token 上限而仍未接受，应返回失败或不完整状态，而不是宣称保证了有效输出。

## 6. Token healing：修复分词边界，不是反悔内容

### 6.1 为什么强制字符边界会降低质量？

[15:03](https://www.youtube.com/watch?v=i4COjX4z1zY&t=903s) 的 URL 例子涉及 `http`、`:`、`//`。
模型可能更习惯使用包含 `://` 的完整 token。
若外部前缀恰好在 `:` 结束，就可能把原本自然的 token 边界截断。

已给定的字符串不一定对应模型最习惯的分词延续。
这是 tokenizer 接口问题，不表示那段字符串语义错误。

### 6.2 Healing 的核心操作

回退最后一个、偶尔两个 token，然后要求新的延续必须先恢复已承诺的字符串前缀。
例如已承诺的结尾是 `:`，重采样的 token 可以以 `:` 开头并顺带包含 `//`。

关键不变量是：用户已经指定的文本保持不变。
改变的是 token 切分与后续展开机会。
它不是随意删去先前错误事实，也不是通用 tree-search backtracking。

### 6.3 什么时候尝试 healing？

课件 17 页列出启发式：

- 最后 token 很短。
- 边界没有明显空白或标点。
- 该 token 是另一个高概率 token 的精确前缀。
- 从外部给定文本继续生成。

这些是实现策略，不是单一完备判据。
多语言没有统一的空格分词习惯，不能把英语边界启发式直接当通用规则。

### 6.4 系统代价

回退意味着对应 KV 与 parser 状态也要恢复，并重新计算受影响位置。
此外要明确“保持原文”是在字符、Unicode code point 还是字节层面；可见字符相同不保证编码完全相同。
流式输出中，更不能撤回已经向用户承诺的内容而不说明。

[21:12](https://www.youtube.com/watch?v=i4COjX4z1zY&t=1272s) 的课堂澄清是：文本内容相同，tokenization 可以变化。
频繁 healing 会增加开销，因此不建议无条件每一步都做。

## 7. 课堂展示的工具接口意味着什么？

课件 18–24 页展示了 Hugging Face、llama.cpp、LangChain、服务端 structured outputs、Gemini schema 与 Outlines。
这里记录能力分类，不承诺 2025 年代码片段今天原样可运行。

| 能力 | 课堂例子 | 实现本质 |
| --- | --- | --- |
| Healing | `token_healing=True` | 回退分词边界并恢复前缀 |
| 自定义解码控制 | `LogitsProcessor` / 自写 forward loop | 根据状态改 token 分数 |
| Grammar 约束 | llama.cpp grammar | 用形式规则限制合法延续 |
| Schema 约束 | typed schema / structured output | 将结构要求编译成约束逻辑 |
| 高层包装 | LangChain、Outlines | 组织类型声明、模型接口和验证 |

高层 API 可能由不同底层方式实现：原生约束采样、prompt、工具调用、解析重试等。
不能仅凭函数名推断内部一定是同一种 FSA。
也不能把 JSON mode 与 schema-constrained output 当作完全同义的名称。

## 8. FSA、CFG、PDA 与程序语言的边界

### 8.1 任意嵌套为什么需要额外记忆？

普通有限状态机只有有限状态，没有任意增长的栈。
任意层数的括号配对需要记录尚未闭合的嵌套层级，标准做法是 pushdown automaton。

因此：

- 一般 JSON 的语法包含任意嵌套，属于 context-free 而非 regular。
- 固定有界嵌套的 schema，可以展开为有限状态约束。
- 支持递归的 schema，不应一概声称只需要 FSA。

课堂使用“CFG 加一个栈”的直觉讨论这一点；更精确地说，PDA 是识别 context-free 语言的机器模型。
实际库还会使用解析器、栈、缓存和其他程序结构，不一定显式生成一个图。

### 8.2 重复 key 的限制要看域是否有限

固定 schema 的有限 key 集合，可以用状态或 bitset 记录是否出现。
允许任意字符串 key 且禁止重复，需要保存并比较已出现字符串，不能仅靠一般 JSON 的上下文无关语法解决。

因此“禁止重复 key 超出 CFG”应理解为任意 key 的一般问题。
它不否定固定有限 schema 的可实现性。

### 8.3 “生成合法 C 程序”有几层含义？

[30:52](https://www.youtube.com/watch?v=i4COjX4z1zY&t=1852s) 与 [42:53](https://www.youtube.com/watch?v=i4COjX4z1zY&t=2573s) 的讨论涉及变量是否已声明等跨位置关系。
可以区分：

1. 表面语法正确，如括号和语句结构。
2. 静态语义正确，如类型和标识符作用域。
3. 运行行为符合用户需求。

语法约束有帮助，但不能单独保证后两层。
符号表、类型检查器、编译器和测试可提供进一步反馈。
不能把课堂的简化短答“no”误写成“程序代码无法做约束解码”。

## 9. 硬约束如何与模型概率结合？

设 $s$ 为当前 parser 状态，合法候选集合为 $A(s)$。
定义 mask：

$$
m_s(v)=
\begin{cases}
0,&v\in A(s),\\
-\infty,&v\notin A(s).
\end{cases}
$$

结合 logits $z(v)$：

$$
q(v\mid s)=\operatorname{softmax}(z(v)+m_s(v)).
$$

合法候选的相对概率保持不变；非法候选概率变为零。
这是 [41:13](https://www.youtube.com/watch?v=i4COjX4z1zY&t=2473s) 的实现核心。

### 9.1 三个数值与顺序问题

- 有限的“大负数”不在数学上等于零概率；严格 mask 应有明确语义。
- 所有候选都被屏蔽时，softmax 可能产生 NaN；要有死路检测和错误返回。
- 若先 top-k 再 grammar mask，top-k 可能恰好没有合法 token；若先 grammar mask 再 top-k，得到的是合法集合中的 top-k。两者不是等价顺序。

合法性判断还要包括 EOS、长度预算与 tokenizer 表示，不能只做文本后验检查。

### 9.2 自动机不是在无限语言上定义“均匀概率”

课堂 [39:11](https://www.youtube.com/watch?v=i4COjX4z1zY&t=2351s) 用均匀偏好的直觉说明规则不区分合法输出的质量。
严谨写法是布尔合法性指示函数，不是对无限多个合法字符串归一化的均匀分布。
概率偏好仍由 LM 给出。

### 9.3 局部屏蔽不等于全局条件采样

这是推导补充。
真正的全局条件分布是：

$$
P(y\mid C(y)=1)=\frac{P(y)\,\mathbf1[C(y)=1]}{P(C=1)}.
$$

局部 mask 只检查“是否还有合法完成”，没有乘上“以后成功的概率”。
例如第一步 A、B 各 0.5；A 的所有后续都合法，B 的后续仅有 0.1 概率合法。
二者初始都可完成，因此局部 mask 仍各给 0.5；但全局条件概率给 A：

$$
P(A\mid C=1)=\frac{0.5}{0.5+0.5\times0.1}\approx0.909.
$$

这说明“确保最终合法”和“按原模型条件分布精确采样”是不同承诺。
下一节 FUDGE 正是在估计类似的未来属性概率。

## 10. 语义约束：为何 ban 一个词不够？

课件 28–32 页要求推荐保持体形的爱好，但不要推荐攀岩。
把 `climbing` 的概率设为零，有三种典型问题：

1. 同义表述仍可能出现，如 bouldering。
2. 该词可能出现在允许的语境，如解释为什么不推荐某活动。
3. 模型先走到相关语境后再被拦截，可能接出不自然的句子。

语义限制作用于含义，不等于固定 token blacklist。
词表切分还会让一个词有多种 token 路径，单 id 屏蔽甚至不一定能保证字面禁词。

### 10.1 Sample then filter

先生成完整输出再分类或拒绝，往往更容易判断语义。
缺点是已经花掉完整生成成本，必要时还要重试。

若独立候选满足约束的概率为 $r$，单次成功前的期望样本数约为 $1/r$。
这是理想独立重试的成本估计，不保证实际分类器准确或重试会迅速成功。
当约束稀有时，纯后验拒绝容易浪费大量计算。

## 11. FUDGE：估计最终属性，逐 token 改分布

### 11.1 Bayes 分解

[49:27](https://www.youtube.com/watch?v=i4COjX4z1zY&t=2967s)、课件 33 页：令 $u$ 为前缀，$v$ 为候选，$a$ 为最终属性。

$$
P(v\mid u,a)
=\frac{P(a\mid uv)P(v\mid u)}{P(a\mid u)}
\propto P(v\mid u)P(a\mid uv).
$$

分母对本步所有候选相同，因此可先乘两个分数，再归一化。
基础 LM 提供局部语言概率；future discriminator 估计接上候选后最终满足属性的概率。

### 11.2 “Future” 不是当前片段分类

FUDGE 的预测目标不是“当前这几个词是否已经很正式”，而是“以此前缀继续，完整输出最终正式的可能性”。
课件 34 页用 formal / informal 数据说明：完整句子的属性标签可以用于训练它的多个截断前缀。

同一个短前缀如 “I” 可能出现在两种类别里；模型应学到不确定性。
较长的 “I would appreciate ...” 才可能提供更强的正式语体信号。
这正是 [FUDGE 论文](https://arxiv.org/abs/2104.05218) 区别于仅评价当前片段的方法之处。

### 11.3 课件数字例子

课件 35 页的前缀是 “Do you”。这里只展示部分候选：

| 下一词 | LM 概率 | 预测正式属性概率 | 乘积 |
| --- | ---: | ---: | ---: |
| want | 0.3 | 0.4 | 0.12 |
| prefer | 0.3 | 0.8 | 0.24 |
| thus | 0.1 | 0.9 | 0.09 |

`prefer` 胜出：它同时有较好的基础概率和属性概率。
只看属性会选 `thus`，只看 LM 则 `want` 与 `prefer` 并列。
表中的乘积未归一化，且候选未列全，不能把 0.24 当作最终采样概率。

### 11.4 Log-space 实现与控制强度

可写成：

$$
\tilde z(v)=\log P_\theta(v\mid u)+\beta\log D_\phi(a\mid uv).
$$

$\beta=1$ 对应直接代入 Bayesian 分解；其他值是调节控制强度的 tempered guidance。
对过小概率取 log 时应做数值保护，但 clipping 也会改变目标分布。

如果 discriminator 恰好估计同一基础模型的真实未来属性概率、候选支持完整且终止处理正确，Bayes 分解是精确的。
实际预测误差、训练分布偏移和候选预筛，会使实现成为近似。

### 11.5 成本与保证

课件 36 页明确三点：

- 每步要额外运行 future discriminator。
- 不能保证满足语义属性。
- 需要基础模型的 logits 或等价分数接口。

只对较小候选集评分、使用很小的模型可以省计算；课堂举了 top-200 一类实现选择，不是 FUDGE 的固定常数。
也可在末尾加 reject/filter，但额外验证的成本仍要计算。

## 12. 与 RLHF、reward-augmented decoding 的联系

[55:39](https://www.youtube.com/watch?v=i4COjX4z1zY&t=3339s) 将基础 LM 看作 prior，把偏好或奖励看作额外倾向。
这是一种理解训练控制与推理控制之间联系的视角，不表示两者工程上等价。

### 12.1 KL 正则目标的推导补充

给定参考分布 $p$、奖励 $r(y)$ 和 $\tau>0$：

$$
\max_q\;\mathbb E_{y\sim q}[r(y)]-\tau D_{KL}(q\Vert p).
$$

其理想无约束解为：

$$
q^*(y)=\frac{p(y)\exp(r(y)/\tau)}{Z}.
$$

这说明完整序列层面可把奖励理解为对参考分布的指数重加权。
训练可尝试让参数模型接近它；解码控制则尝试在推理时实现某种近似。
这一关联见 [Korbak et al.](https://arxiv.org/abs/2205.11275)。

### 12.2 不能随便把完整奖励塞进每步

若奖励在完整序列上定义，精确下一步条件通常涉及未来完成的 $\exp(r/\tau)$ 的条件期望。
它一般不等于对“当前前缀奖励”直接指数化，也不等于把未来奖励均值先指数化。
因此需要 future value / reward estimation 等近似方法。

课件 38 页仅简要介绍 RAD：用额外奖励模型在生成时重加权。
其 [原论文](https://arxiv.org/abs/2310.09520) 使用单向奖励模型复用历史激活以减轻开销；本录像未包含后续学生报告的详细实验与实现。

## 13. Contrastive decoding：利用 expert 与 amateur 的差异

[57:44](https://www.youtube.com/watch?v=i4COjX4z1zY&t=3464s) 的想法是：弱模型可能更明显地表现出重复、空泛等坏倾向。
选择 expert 认为相对更好、amateur 却没有那么偏好的 token。

基本对比得分：

$$
s(v)=\log p_E(v\mid u)-\lambda\log p_A(v\mid u).
$$

它比较的是相对概率，不是简单选择 expert 原本概率最大的 token。
相同词表且相同前缀下，减 logprob 与相应减 logits 的差别只是对所有候选共同的常数，因此排序一致。

### 13.1 为什么还需要 plausibility constraint？

若 expert 给某词 $10^{-6}$、amateur 给 $10^{-12}$，概率比很大，但该词仍可能完全不合上下文。
因此 [Li et al.](https://aclanthology.org/2023.acl-long.687/) 还限制候选在 expert 下足够可信，例如：

$$
A_\alpha(u)=\{v:p_E(v\mid u)\ge\alpha\max_w p_E(w\mid u)\}.
$$

只在该集合内做对比选择，避免极小概率比值主导结果。
这是原论文补充；课件的 `model1.forward - model2.forward` 只是概念伪代码。

### 13.2 实现约束

- 两个逐 token 分布需要对齐的词表和 token 语义；不能直接相减不同 tokenizer 的索引位置。
- Expert 与 amateur 都要更新各自上下文状态。
- 两模型大小不同，成本不一定恰好翻倍。
- 同一模型不同 prompt 的双路推理可共享权重，但上下文不同，KV 通常仍需分别维护。

这不是 contrastive search；相似名字对应不同算法，不能仅通过一个 generation 参数名互换。

## 14. Safety-oriented contrastive decoding

课件 41 页提到 Adversarial Decoding 与 ROSE 一类思路：用更偏向不希望内容的分布作为对照。
直觉是识别不希望的倾向有时比完整定义理想回答更容易。

这仍是语义偏好的软控制，不是形式安全证明。
“对照模型更不喜欢”不能直接推出事实正确、完全安全或不会违反业务规则。
实际应用仍需独立评测、检测器及最终输出验证。

[1:01:34](https://www.youtube.com/watch?v=i4COjX4z1zY&t=3694s) 的成本讨论强调需要多路 forward。
若两个分支同样大，额外计算可能接近另一遍模型；具体延迟取决于并行、缓存和 batch。

## 15. 一种统一的解码实现视角

下式是实现补充，概括不同组件，不表示所有方法应一起叠加：

$$
\tilde z_t(v)=z_t(v)+m_t(v)
+\beta\log D(a\mid y_{<t}v)
-\lambda\log p_A(v\mid y_{<t}).
$$

- $z_t$：基础模型 logits。
- $m_t$：硬规则 mask。
- $D$：未来属性分数。
- $p_A$：对比模型分布。

实际使用时应明确哪些项启用、各项归一化语义和执行顺序。
特别是 hard mask 的零支持不能被后面的运算意外恢复。

```python
state = grammar.initial_state()
prefix = initial_tokens
model_cache = None
finished = False

while not finished:
    logits, model_cache = model_step(prefix, model_cache)
    allowed = grammar.valid_next_tokens(state, remaining_budget)
    scores = logits.masked_fill(~allowed, -float("inf"))
    if not any_finite(scores):
        raise ConstraintDeadEnd()

    candidates = preselect(scores)  # optional; changes guidance support
    if use_future_discriminator:
        scores[candidates] += beta * log_future_attribute(prefix, candidates)
    if use_amateur:
        scores[candidates] -= contrast_weight * amateur_logprobs(prefix, candidates)

    token = choose(scores, candidates)
    state = grammar.consume_token(state, token)
    prefix.append(token)
    finished = is_eos(token) and grammar.is_accepting(state)
```

伪代码省略 cache rollback、完整候选管理及最后文本校验。
若 grammar 完全不用，应跳过对应 mask；若只用 hard grammar，不需要额外 neural scorer。

## 16. 与 AI Infra 的关联与评测清单

### 16.1 Grammar 编译与缓存

同一 schema 可复用编译结果，避免每请求重新构造规则。
但用户 schema、tokenizer、版本和配置要进入缓存键。
不同请求的 parser 当前状态不能混用。

### 16.2 每步 CPU–GPU 协调

合法 token 集合的计算、mask 搬运、候选排序和额外评分可能成为瓶颈。
只优化大模型 forward，而忽略 grammar 引擎及同步，会低估 structured output 的开销。
常用 schema 的预计算、批量更新与减少同步值得测量。

### 16.3 多模型缓存与 rollback

FUDGE、reward model 和 amateur 可能各有独立状态。
Token healing 回退时，不仅主模型，所有依赖前缀的状态都要一致回退。
漏回退任何一项，可能造成静默的分数错误。

### 16.4 需要分开报告的指标

- JSON 解析成功率、schema 通过率、完成率。
- 语义约束成功率及检测器误判率。
- 主任务质量、流畅性、长度与多样性。
- TTFT、逐 token 延迟、端到端延迟、吞吐。
- 额外模型调用、拒绝重试次数、峰值显存。

只在成功完成样本里统计 schema 成功率，会隐藏截断和死路。
只统计“禁词没出现”，会夸大语义控制效果。

## 17. 自测与面试回答

### Q1：最终可验证、前缀可行、未来成功概率有什么不同？

**回答：** 最终验证检查完整结果；前缀可行判断是否仍存在合法完成；未来概率估计按模型继续时成功的可能性。Grammar 多处理第二个问题，FUDGE 估计第三个，二者不能混为一谈。

### Q2：JSON 模式为何不等于答案正确？

**回答：** JSON parser 只检查语法，schema 再限制字段和类型，事实及业务约束还需其他验证。一个包含错误年份但格式完美的对象仍能通过前两层。

### Q3：约束 decoder 怎样判断一个 token 合法？

**回答：** 要把整个 token 的文本或字节表示从当前 parser 状态消费完，检查结果是否仍可完成，并正确处理 EOS 和长度预算。只看首字符或 token 名字是不够的。

### Q4：Token healing 与搜索回溯有什么区别？

**回答：** Healing 回退少量 token 以重新切分边界，但要求恢复已经给定的字符串；一般搜索回溯可以改内容。前者必须同步恢复 KV、parser 和其他 scorer 状态，并保证不破坏已承诺输出。

### Q5：为什么不能说所有 JSON schema 都只需一个小 FSA？

**回答：** 任意嵌套需要栈，递归 schema 不能一般化为固定小 FSA。有限字段、有限深度可有限状态化，但无序字段去重可能产生大量状态；实现通常用动态 parser 和缓存管理复杂度。

### Q6：Hard mask 是否精确采样 $P(y\mid C(y)=1)$？

**回答：** 通常不是。它只保留仍可能合法完成的 token，没有按每个分支未来合法完成概率重加权，所以能改变不同合法序列之间的全局概率。支持合法性与条件分布精确性是两种不同保证。

### Q7：FUDGE 的 future discriminator 预测什么？

**回答：** 它预测给定候选前缀后，完整输出最终拥有目标属性的概率，而不只是当前片段是否已显示该属性。利用 Bayes 分解把这个概率乘到 LM 下一 token 概率上，再归一化。

### Q8：为什么 FUDGE 没有语义硬保证？

**回答：** Discriminator 有预测误差、校准与分布偏移，候选预筛又会改变支持集。其控制是近似概率偏好，不是完备规则；必要时仍需末尾验证或拒绝重试，并计入额外成本。

### Q9：Contrastive decoding 为什么需要 plausibility constraint？

**回答：** 大概率比可能来自 expert 和 amateur 都几乎不可能的 token，直接最大化差值会选荒谬候选。先限制在 expert 认为可信的候选集，再比较相对概率，才能避免这种失败。

### Q10：同一个模型用两种 prompt 对比，成本能完全共享吗？

**回答：** 可以共享模型权重，但不同上下文通常需要独立 KV 与 forward 状态。总成本取决于共享前缀、batch 和并行，不能把权重只有一份理解为推理只需一遍。

### Q11：何时选训练控制，何时选推理控制？

**回答：** 稳定、广泛复用的偏好可以通过训练摊销；明确、易变的 schema 和规则适合推理时控制。语义要求常需两者结合，同时权衡训练成本、逐请求开销、质量和保证范围。

### Q12：用词向量到 formal cluster 的距离能替代 FUDGE 吗？

**回答：** 课堂最后认为可以把它作为属性分数尝试，但几何相似度不等于校准的最终属性概率。它可用于启发式 guidance，却不能不经验证就套用精确 Bayes 解释或宣称更可靠。

## 18. 本讲最重要的结论

1. 能写成确定规则的约束，优先明确规则与接受条件。
2. 不能靠局部禁词精确表达的语义要求，需要未来估计或完整结果验证。
3. Logit manipulation 是共同实现接口，不代表各种方法有相同保证。
4. 训练、grammar、future discriminator、reward model、contrastive model 可以互补，但每一层都增加自己的状态和成本。
5. “合法”“满足偏好”“分布精确”“任务正确”应分开陈述与评测。

## 关联内容

- [课程索引](CMU%2011-763.md)
- [上一讲：A Star and Best First Search](CMU%2011-763%20-%20Lecture%2005%20-%20A%20Star%20and%20Best%20First%20Search.md)
- [下一讲：Chain of Thought and Intermediate Steps](CMU%2011-763%20-%20Lecture%2007%20-%20Chain%20of%20Thought%20and%20Intermediate%20Steps.md)
- [LLM Inference](../../topics/inference/LLM%20Inference.md)
