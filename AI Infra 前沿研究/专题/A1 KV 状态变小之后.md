---
type: article
status: 完整（代表作）
verified: 2026-11-23
revised: 2026-11-23（第二轮：加入 DeepSeek-V4.1 证据）
threads: ["[[T1 状态的生命周期]]", "[[T2 代价搬家账本]]", "[[T3 结论反转条件]]"]
tags: [kv-cache, hybrid-attention, disaggregation, storage-tier]
---

# A1 KV 状态变小之后：从"显存里的缓存"到"可以跨机房搬运、可以落盘的数据"

> [!abstract] 一句话
> 过去一年最值得重写的判断不是"KV cache 很大所以要分层"，而是：**hybrid / 压缩 / 稀疏 attention 让每 token 的 KV 小了一个数量级，于是 KV 第一次可以走普通以太网、跨数据中心、落到 SSD——但代价从"容量"搬到了"状态种类变多、短请求反而更贵、缓存命中改变了值不值得搬"。**

阅读路线：先读"§1 workload 要什么"，再读"§2 为什么能成立"，§3 是代价搬到哪里，§4 是什么时候结论反转。每节末尾有"往哪走"链接。

---

## 0. 旧基线：2026 年 1 月时我们怎么理解 KV

HPCGame 讲座（2026-01-28，[视频 1:10:35](https://www.bilibili.com/video/BV1vmzXBBESY/?t=4235)）对大规模推理的描述是："prefill 和 decode 要求不同，就拆开，用 KV cache 的调度器和 transfer engine 把 KV 传走"，并以 Mooncake 为例说明"以 KV cache 为中心"构建 GPU 显存 / CPU 内存 / 磁盘 / 远端存储的整套系统。讲者自己评价"思想非常 trivial"。

这个基线里隐含了三个假设：

1. **KV 体积大**，所以 P→D 的 KV 传输必须走 RDMA 级网络（同一机房、同一 fabric）；
2. **KV 是同质的**：每层、每 token 一块，PagedAttention 的固定大小 block 就能管理；
3. **分层存储的瓶颈是容量**：放不下就往 CPU/SSD 推。

过去一年，三个假设都被动摇了。下面按 workload → 机制 → 代价 → 反转条件走一遍。

---

## 1. Workload 要什么：长上下文 + 长会话，但"新增的 token"很少

两个方向同时把 KV 的重要性推高：

- **百万 token 上下文成为旗舰默认**。DeepSeek-V4（Pro 1.6T / 49B 激活，Flash 284B / 13B 激活）两个模型都支持 1M context（[DeepSeek-V4 report, arXiv:2606.19348 v1, 2026-04-26, Abstract & §4.2.2](https://arxiv.org/abs/2606.19348)）；Kimi-K3 模型卡给出 2.8T / 104B 激活、93 层中 69 层 KDA + 24 层 Gated MLA、1,048,576 上下文（[HF 模型卡](https://huggingface.co/moonshotai/Kimi-K3)，本轮核对 2026-11-23）。
- **Agent 让一次"任务"变成几十轮请求**。UIUC 等对 Claude Code 类脚手架的测量：多数配置平均 12–62 轮/任务，SWE-bench Pro 上累计上下文均值 68.7K–80.1K token，最大 146K–166K；前缀缓存命中率 84.6–99.5%；LLM 时间中 decode 占 91.0–98.6%（[Agentic AI Workload Characteristics, arXiv:2605.26297 v1, 2026-05-25, §4–§5](https://arxiv.org/abs/2605.26297)；条件：vLLM v0.20.0、TP=2、2×H100 NVL、Qwen3.6-27B / Gemma4-31B；**作者测量**，理想化无容量驱逐，因此命中率是上界）。

把两者放一起，workload 的需求就清楚了：

| 需求 | 含义 |
|---|---|
| 状态要**长期保存** | 一个会话的 KV 要活过多轮、跨越工具调用的空档 |
| **新增 prefill 很少** | 命中率高时，每轮真正要算的只是最近的工具输出 |
| **偶尔有很长的未命中输入** | 新任务、新文档、长工具返回——这才是 prefill 的大头 |
| 状态**容量**远大于单卡 HBM | 1M context × 并发会话 |

> [!note] 这里已经能看到张力
> 对 agent，主要问题是"别丢掉状态"（见 [[A2 Agent 负载]]）；对长且未命中的输入，主要问题是"prefill 算力放哪、KV 怎么搬回来"。两者对 KV 的诉求不同，后面 §4 会用它来解释结论何时反转。

---

## 2. 为什么能成立：每 token KV 小了一个数量级

### 2.1 机制：只有少数层的 KV 随长度增长

Hybrid 模型把少量 full attention（或 MLA/GQA）层和大量**线性 / 有界状态**层交错（KDA、GDN、SWA、Lightning 等）。线性层的状态大小与序列长度无关；只有 full attention 层的 KV 随长度增长。DeepSeek-V4 走另一条路：不是线性层，而是**把 KV 本身压缩**——CSA 每 4 个 token 压成 1 个 KV 条目并用 Lightning Indexer 做 top-k 稀疏选择（Flash k=512，Pro k=1024），HCA 每 128 个 token 压成 1 个条目后做稠密 attention；再加 128 token 滑窗分支（[DeepSeek-V4 §2.3.1–2.3.3](https://arxiv.org/abs/2606.19348)）。KV 中 RoPE 维度存 BF16、其余维度存 FP8，"比纯 BF16 存储减少近一半"（§2.3.4）。

### 2.2 数字：放在同一把尺子上

**DeepSeek-V4 自报（作者估算，非实测）**：1M context 下，Pro 的 KV 是 V3.2 的 10%、单 token FLOPs 是 27%；Flash 分别是 7% 与 10%；相对 BF16 GQA8（head dim 128）基线约为 2%（Abstract、§1、§2.3.4、Figure 1）。

**Prefill-as-a-Service（Moonshot AI + 清华）** 给了一个更好用的系统指标——KV 吞吐 Φ_kv = KV 字节数 / prefill 时间，即 prefill 集群每秒"吐出"多少 KV（[arXiv:2604.15039 v1, Eq.1, Table 3](https://arxiv.org/abs/2604.15039)）。条件：8×H200，SGLang v0.5.9，**作者测量**：

| 模型（attention 结构） | 1K | 8K | 32K | 128K（Gbps） |
|---|---|---|---|---|
| MiniMax-M2.5（纯 GQA） | 4.94 | 32.87 | **59.93** | 47.82 |
| Qwen3-235B（纯 GQA） | 4.12 | 22.42 | 33.35 | 21.50 |
| Qwen3.5-397B（GDN+GQA 3:1） | 4.13 | 6.28 | 8.25 | 7.47 |
| MiMo-V2-Flash（SWA+GQA 5:1） | 0.82 | 2.85 | **4.66** | 4.71 |
| Kimi Linear 48B（KDA+MLA 3:1） | 1.19 | 2.29 | 3.87 | 4.88 |
| Ring-2.5-1T（Lightning+MLA 7:1） | 7.27 | 4.47 | 2.59 | 1.46 |

32K 时 MiMo-V2-Flash 与 MiniMax-M2.5 相差约 13×，Qwen3.5 对 Qwen3 约 4×（§2.2）。

### 2.3 一个可追踪的算例：从字节到带宽

论文唯一给出绝对 KV 大小的是内部 1T hybrid 模型（Kimi Linear 风格，3:1 KDA:MLA；Table 5，in-house vLLM，8×H200）：

| 输入长度 | KV 大小 | prefill 时间 | Φ_kv |
|---|---|---|---|
| 1K | 190.8 MiB | 0.44 s | 3.61 Gbps |
| 8K | 308.9 MiB | 0.72 s | 3.59 Gbps |
| 32K | 701.3 MiB | 1.84 s | 3.19 Gbps |
| 128K | 2316.3 MiB | 7.40 s | 2.62 Gbps |

**我的推算**（不是论文的说法）：(2316.3 − 701.3) MiB / 96K token ≈ **17 KiB/token 的边际增长**，外加约 **170 MiB 与长度无关的固定量**——这与"每请求一份线性层递归状态"的结构一致。

作为对照，一个纯 GQA 模型按"层数 × KV 头数 × head dim × 2（K 和 V）× 2 字节"估算：若取 94 层、4 个 KV 头、head dim 128、BF16（这是我按 Qwen3-235B 公开配置估算的数，本轮未重新打开 config.json 核对），每 token 约 188 KiB——比 17 KiB 大约 11×，与 Table 3 中 GQA 对 hybrid 的差距同一量级。

再到集群：论文 Eq.2 估算 512 张 H200 做 prefill、平均输入 32K 时，MiniMax-M2.5 需要约 3.8 Tbps 出口带宽，Ring-2.5-1T 约 170 Gbps（§2.3，**解析推算**）。前者只有 RDMA fabric 能承受，后者是普通数据中心间专线的量级。

> [!important] 改变了什么
> 旧基线中"KV 传输 ⇒ 必须同一 RDMA fabric"是**体积**决定的。体积降一个数量级后，P 和 D 可以放在**不同机房、不同代硬件**上：PrfaaS 的案例用 32 张 H200 做远端 prefill，64 张 H20（本地 PD，每节点 800 Gbps RDMA）做本地，经约 100 Gbps VPC 跨集群互联，平均出口仅约 13 Gbps；报告吞吐比同构 PD（96×H20）高 54%、P90 TTFT 低 64%（§4.1、Table 6）。
>
> **标注**：作者测量 + 解析模型——吞吐数字是把 profiling 数据代入 §3.4 的分析模型得出，不是端到端实跑；同成本对比时收益约 15%（§4.4）；单一内部模型、单一硬件对。第二轮读了 v2（2026-04-22）：论文仍只说"内部 1T 参数 hybrid 模型"，**没有声称已用于 Kimi 生产**；§4.4 还说 PrfaaS 集群"目前受算力限制、带宽余量充足"——也就是说，在这个案例里瓶颈已经从网络回到了 prefill 算力本身。

### 2.4 第二轮补充：V4.1 再压 4 倍，而且换了压法

DeepSeek-V4.1-Flash（[arXiv:2609.19969](https://arxiv.org/abs/2609.19969) v1，2026-09-17；HF 首发 2026-09-09）给出了本笔记里最小的每 token KV（§1、Fig. 1(b)，作者自述，未标明是测量还是估计）：

- **全局 KV 每 token 890 字节**，约为 V4-Flash 的 1/4；持久化 KV 约为 V4-Flash 的 1/8（§3.2.1）。
- 压法有三处变化：(1) **编码器–解码器结构（CED）**：20 层因果编码器 + 20 层解码器，解码器的全局 KV 由编码器第 L/2 层的隐藏状态投影得到，作者称 prefill 计算"近乎减半"（设计主张，无实测数字）；(2) 只用 CSA2、去掉 HCA，**主 KV、indexer K 与 top-k 索引在层间共享**；(3) 全局 KV 在训练中就用 FP4（E2M1，每 16 通道一个 E4M3 scale），作者称退化"轻微"；滑窗 KV 保持 FP8。

**我的推算**：按 890 B/token，一个 1M token 会话的全局 KV ≈ 0.83 GiB；按 §2.3 的 GQA 估算（约 188 KiB/token），同样长度约 179 GiB。**相差两个多数量级，意味着"每个会话一份 1M 上下文 KV 落盘长期保存"从不可想象变成每会话不到 1 GiB 的存储问题。**下一节的 72 小时 SSD 保留正是建立在这个量级上。

注意定义差异：890 B 是 V4.1 报告对"全局 KV"的口径（不含滑窗与 indexer 状态），与 PrfaaS Table 5 的"总 KV 大小"口径不同，两者只能比量级。

---

## 3. 代价搬到了哪里

KV 变小没有消灭代价，它把代价搬到了四个新位置。

### 3.1 状态种类变多：KV 不再同质

DeepSeek-V4 的推理框架需要**异构 KV 布局**（§3.5、Figure 6）：
- 一份"经典 KV cache"存 CSA/HCA 压缩条目，block 覆盖 lcm(4,128)=128 个原始 token（每 block 32 个 CSA 条目、1 个 HCA 条目）；
- 一份**每请求固定大小的 state cache**，存滑窗 KV 和尚未凑满一个压缩块的尾部 token。

报告直接写明 paged attention 与这种混合布局不兼容。PrfaaS 也一样：它把线性层状态和 full attention KV 作为同一 block pool 中的**不同组**管理（基于 vLLM hybrid KV manager，§3.2）。

这意味着讲座笔记里"未来引擎会同时管理多种状态"的判断已经变成事实，而且带来了具体问题：**前缀复用的粒度不再统一**。KV 条目可以按 block 复用，但线性层的递归状态只能在"恰好算到这个位置"时被快照——SGLang v0.5.19 才加入"SWA hybrid 的 decode 前缀复用"（[SGLang releases](https://github.com/sgl-project/sglang/releases)，2026-09-05，摘要级读取）。

### 3.2 存储层的新角色：落盘的是"压缩后的"状态

DeepSeek-V4 的 on-disk KV（§3.5.2）只存压缩后的 CSA/HCA 条目，用来消除共享前缀请求的重复 prefill；尾部不完整块重算。难点在滑窗状态：**SWA 体积约是压缩 CSA/HCA 条目的 8 倍**，所以给出三种策略——全量缓存 SWA、每 p 个 token 周期性 checkpoint、完全不存（命中时重算最后 n_win × L 个 token）。

这是一个很干净的"代价搬家"例子：压缩让主体 KV 便宜到能落盘，但**不可压缩的小窗口状态反而成了存储的大头**，于是在"存储字节"和"命中时重算"之间要做选择。

**V4.1 的回答是：给两种状态不同的寿命**（V4.1 §3.2.1–§3.2.2，逐字读到）：

| 状态 | 放哪 | 寿命 | 复用方式 |
|---|---|---|---|
| 全局 KV（压缩、FP4） | SSD | **保证至少 72 小时** | 前缀命中时复用完整全局 KV 前缀 |
| 滑窗 KV（FP8） | 从持久缓存中移出，放进**占每台机器主机 DRAM 10% 的池** | TTL 以分钟计，LRU 驱逐 | 只在"提示结束"和"输出结束"两处缓存 |
| 未缓存的滑窗状态 | — | — | "有界重放"：只重算最后 n_win 个 token；解码器侧重放是近似的，作者称质量影响"可忽略"（无数字） |

V4 报告曾提出"零 SWA 缓存"，命中时要对 L × n_win 个 token 做一次完整前向；V4.1 把它改成有界重放。SGLang 的 day-0 实现（[LMSYS，2026-09-10](https://www.lmsys.org/blog/2026-09-10-deepseek-v41)）【作者测量】：打开解码器侧有界重放后，8 条 8K 提示的 prefill 吞吐在 8×H200 上 1.56×、4×GB300 上 1.37×；AIME 2026 pass@1 开关前后都是 453/480。

这把 [[T1 状态的生命周期]] 的判据落成了一个具体设计：**长寿命、可压缩的放慢层并承诺保留期；短寿命、不可压缩的放快层并用 TTL；中间用有界重算兜底。**

厂商侧同步出现了专门的层。NVIDIA 在 GTC 2026（2026-03-16）提出 "G3.5"：在 G1 HBM / G2 主机内存 / G3 本地 SSD 与 G4 共享存储之间，加一层**以太网挂载、pod 级共享、专为 KV 优化的闪存层**，基于 BlueField-4，经 Dynamo 的 KV block manager 与 NIXL 搬运（[NVIDIA 技术博客](https://developer.nvidia.com/blog/introducing-nvidia-bluefield-4-powered-inference-context-memory-storage-platform-for-the-next-frontier-of-ai/)；产品名现为 [CMX](https://www.nvidia.com/en-us/data-center/ai-storage/cmx/)，页面修订 2026-08-04）。理由很有意思：KV 是"临时的、派生的、可重算的"数据，所以可以去掉副本、元数据等重存储服务。**厂商声明**：相对"传统存储"最高 5× TPS 与 5× 能效，未给出模型、上下文、集群或基线配置，不可复现；也没有给出供货日期。

开源侧：Mooncake v0.3.12/0.3.13（2026-07 至 08）加入 SSD offload 主节点重启后自动恢复、命中时 L2→L1 提升、NVLink/VRAM 段传输（[Mooncake releases](https://github.com/kvcache-ai/Mooncake/releases)）；llm-d v0.8.0（2026-06-24）加入 CPU→存储的多级 KV offload（[llm-d releases](https://github.com/llm-d/llm-d/releases)）；SGLang v0.5.20（2026-09-18）把 radix tree 与 Mooncake 等外部存储统一（摘要级）。

### 3.3 调度的新维度：带宽变成要被调度的资源

KV 能走普通网络不等于随便走。PrfaaS 明确说"仅靠 KV 变小不够"：请求突发、长度偏斜、前缀缓存分布不均、带宽波动都需要系统设计（§1、§2.3）。它的做法是：
- **按长度路由**：只有增量（未命中）长度超过阈值 t 的请求才送远端 prefill（案例中 t≈19.4K，49.6% 请求被卸载）；
- 逐层 prefill 流水、多连接 TCP、拥塞监测避免突发；
- 双时间尺度调度：短期做带宽/缓存感知路由，长期重平衡 P/D 节点角色（§3.3–§3.4）。

### 3.4 短请求反而更贵

看 §2.3 的算例：1K 输入的 KV 是 190.8 MiB，其中约 170 MiB 是固定的递归状态。**按每 token 平均，短请求的 KV 比长请求大得多**。Ring-2.5-1T 的 Φ_kv 随长度下降（1K 时 7.27 Gbps，128K 时 1.46 Gbps）也是同一现象：短请求 prefill 快、状态却不小。

---

## 4. 什么时候结论会反转

| 条件 | 结论 | 依据 |
|---|---|---|
| 模型是纯 GQA/MHA（无压缩、无线性层） | 跨机房 prefill **不可行**，回到 RDMA 同 fabric | Table 3：32K 时 33–60 Gbps/实例，512 卡约 2–3.8 Tbps |
| 请求短（几 K token） | 不值得远端 prefill：固定状态占主导，搬运不划算 | Table 5 固定量约 170 MiB；PrfaaS 只卸载 >t 的请求 |
| 前缀命中率很高（agent 多轮） | 增量 prefill 很小，瓶颈是**保留状态**而不是搬运或远端算力 | arXiv:2605.26297：命中 84.6–99.5%，decode 占 91–98.6% |
| 同成本比较，而非同卡数 | 收益从 54% 缩到约 15% | PrfaaS §4.4 |
| 必须缓存 SWA 等小窗口状态 | 存储字节可能被"不可压缩小状态"主导，需在落盘与重算之间权衡 | DeepSeek-V4 §3.5.2（SWA ≈ 8× 压缩条目） |
| 有严格的 RL 数值一致性要求 | 远端/异构 prefill 引入不同硬件与 kernel，logprob 可能不一致 | 见 [[A6 数值一致性]] |
| 主机 DRAM 被其他大状态占用（如 Engram 表 183–189 GiB） | SWA 池与 KV 卸载的可用 DRAM 变少；把表挪去 SSD 反而让 KV 受限负载吞吐 +36% | [[A11 条件记忆 Engram#4. 代价搬到哪里]] |
| 需要精确复现（滑窗状态不可近似） | 有界重放的"近似"不可接受，只能全量缓存 SWA 或全量重算 | V4.1 §3.2.2 |

> [!tip] 我会怎么向别人复述
> "KV 小了 10 倍，所以 P/D 可以不在一个机房"只对了一半。完整的句子是：**对于 hybrid/压缩 attention 模型、长且未命中的输入，prefill 可以放到更便宜或更远的算力上，KV 用普通网络带回；但短请求、高命中的 agent 轮次、纯 GQA 模型都不符合这个条件，而且引擎必须同时管理至少两种生命周期不同的状态。**

---

## 5. 未解决的问题（保持可见）

- [ ] PrfaaS 是否已在 Kimi 生产中使用？v2（2026-04-22）仍只说"内部模型"；跟踪 Mooncake 仓库与 Moonshot 后续发布。
- [ ] V4.1 的 890 B/token 是测量还是设计值？72 小时 SSD 保留在多大集群、多少会话下成立？报告 §5.2 以后与 §6 本轮未读到。
- [ ] CMX / G3.5 的 5× 数字在什么模型、上下文、命中率下成立？等待第三方或 MLPerf 类结果。
- [ ] 线性层递归状态的**前缀快照**粒度：每多少 token 存一次最划算？DeepSeek-V4 的"周期 checkpoint p"取值未在本轮读到的部分给出。
- [ ] 跨机房 KV 的一致性与安全边界（多租户、加密、故障时重算）无公开讨论。
- [ ] OSDI 2026 "KV Cache and Long Context" 会话的 Strata（分层上下文缓存）、ECHO（原生稀疏 attention 的 KV offload + 无损预取）、零拷贝 KV 卸载——第二轮已在 [USENIX 官方目录](https://www.usenix.org/sites/default/files/osdi26-contents.pdf) 核实标题，正文仍未读。

## 往哪走

- 状态为什么要"活过工具调用" → [[A2 Agent 负载]]
- P/D 之后还能怎么拆 → [[A3 分离的下一层]]
- 模型里还有另一种"可以放到 CPU 的大表"，而且和 KV 抢同一块主机内存 → [[A11 条件记忆 Engram]]
- 把所有"状态"放在一张图里 → [[T1 状态的生命周期]]
- 回到入口 → [[00 入口]]
