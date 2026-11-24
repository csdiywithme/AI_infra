---
type: article
status: 完整
verified: 2026-11-23
threads: ["[[T1 状态的生命周期]]", "[[T3 结论反转条件]]"]
---

# A2 Agent 负载：调度单位从"请求"变成"会话"

> [!abstract] 一句话
> Agent 任务是几十轮、输入远多于输出、但几乎全被前缀缓存命中的会话；真正贵的是 **decode** 和 **在工具调用空档里丢掉状态**。按请求调度、请求结束即驱逐 KV 的引擎默认行为，对这种负载是错的。

## 0. 旧基线

讲座把推理按场景分层（[视频 55:08](https://www.bilibili.com/video/BV1vmzXBBESY/?t=3308)），旧笔记在 2026-08 补充了 Dynamo 的 "agent priority / cache retention"。当时 agent 还是一个"未来方向"标签，缺少负载测量。现在有了。

## 1. Workload 要什么：测量数据

**Agentic AI Workload Characteristics**（UIUC / Gimlet Labs / Intel，[arXiv:2605.26297 v1](https://arxiv.org/abs/2605.26297)，2026-05-25）【作者测量】：

- 脚手架 Claude Code（经 Harbor），模型 Qwen3.6-27B、Gemma4-31B，vLLM v0.20.0 TP=2，2×H100 NVL；基准 ADE-Bench、DABStep、GAIA、SWE-bench Pro、Terminal-Bench 2.0。
- 轮数：多数配置 12–62 轮/任务；Gemma Instant 在 ADE 上可达 786 轮（大量失败重试：Edit 工具 2,757 次尝试失败 95.4%）。
- 上下文：SWE-bench Pro 累计均值 68.7K–80.1K token；失败任务的上下文均值最多大 1.8×。
- 前缀缓存命中 84.6–99.5%；LLM 时间里 decode 占 91.0–98.6%；输入/输出比 53.9×–559.8×【摘要级】。
- 工具时间占总运行时间 2–29%，LLM 占 71–98%。

**Continuum**（Berkeley / Stanford / Tensormesh / 清华，[arXiv:2511.02230](https://arxiv.org/abs/2511.02230)；第一轮读 v5 2026-05-11，第二轮核对到 v7 2026-09-08）用 GPT-5 轨迹测了工具延迟分布：重尾——最慢 10% 的 `fetch_url` 占总延迟 52.5%，最慢 10% 的 `cd` 占 94.1%（Fig.5）。

**读法**：高命中 + decode 主导 + 长尾工具空档 ⇒ 每轮的新增算力很少，但**状态必须熬过不可预测的空档**。

## 2. 被打破的假设与新机制

| 旧默认 | 为什么错 | 新机制 | 证据 |
|---|---|---|---|
| decode 完成即驱逐该请求 KV（Continuum §3.2 指出 vLLM/SGLang 均如此） | 下一轮几秒后就来，驱逐 = 重新 prefill 或重载 + 再排队 | **KV TTL**：工具调用期间按请求设定存活时间，超时释放；配合程序级 FCFS | Continuum §4–§5，实现于 vLLM |
| 静态保留策略 | 工具延迟重尾，静态保留会长期占显存甚至死锁 | TTL 按"显存×时间"成本与"miss 代价 + 乱序代价"收益求解 | Continuum §3.2、§4.1 |
| 请求级负载均衡 | 会话换实例就丢掉前缀缓存 | **会话亲和路由**：同一会话的各轮钉在同一引擎/DP rank | Miles v0.1 用 SGLang router 实现，报告 96% 前缀命中（见 [[A7 RL 后训练系统]]） |

Continuum 报告：延迟降低 1.12×–3.66×、吞吐提高 1.10×–3.22×（三组硬件/模型配置），某推理创业公司的真实 SWE-agent 负载上延迟最多降 8.18×（Intro）【作者测量】。v7 摘要给出的负载是 SWE-Bench、BFCL、OpenHands，模型 Llama-3.1 8B/70B、Gemma-3 12B、GLM-4.5 355B，主要对照是 InferCept；**§6 的逐配置条件两轮都没能读到**（抓取在 §6 之前截断）。引言还有一个重要的限定：打开 CPU 卸载（如 LMCache）后，重载代价变小，但**每轮重新排队的延迟仍在**——所以 TTL 解决的不只是"状态丢了"，还有"排队位置丢了"。

### 1.1 第二轮：agent 负载进入公开基准

- **MLPerf Endpoints v1.0** 的 Kimi K3 `agentic_combined` 场景（结果 2026-11-09 发布）把 agent 负载变成了按并发扫 Pareto 曲线的标准测量（见 [[A14 电力与成本#2.1 一组公开的、可以拆开的数字]]）。
- **InferenceX AgentX**：SemiAnalysis 开源基准里的长上下文多轮 agent 编码场景，用公开的轨迹语料回放，最短有效运行 900 s（[About](https://inferencex.semianalysis.com/about)）。
- 这两者的意义是：**"会话级"不再只是论文里的测量，开始成为采购比较的口径**——而一旦成为口径，TTL、会话亲和、KV 分层这些机制就直接影响每美元 token。

## 3. 代价搬到哪里

- **显存被"空闲会话"占用**：TTL 本质是用 HBM 容量换重算与排队；高并发时会挤压新请求。于是需要 [[A1 KV 状态变小之后]] 的下层存储（CPU、SSD、G3.5）承接"暂停中的会话"。
- **公平性**：程序级 FCFS 让已开始的会话优先，新会话 TTFT 变差。
- **预测问题**：TTL 收益项假设下一轮在 TTL 窗口内到达；Continuum 也承认排队估计只是用历史平均作代理（§4.1）。

## 4. 何时反转

| 条件 | 结论变化 |
|---|---|
| 工具调用很长（分钟级子 agent：GAIA 上 Agent 工具均值 321–916 s【摘要级】） | 保留在 HBM 不划算，应下沉到 CPU/SSD 或直接放弃重算；DeepSeek-V4.1 的做法是全局 KV 在 SSD 保留至少 72 小时、滑窗状态只用分钟级 TTL（[[A1 KV 状态变小之后#3.2 存储层的新角色：落盘的是"压缩后的"状态]]） |
| 命中率低（每轮大量新文档） | 回到 prefill 主导，[[A1 KV 状态变小之后]] 的远端 prefill 更相关 |
| 显存紧张、并发极高 | TTL 退化成 LRU；需要容量驱动的驱逐，2605.26297 的命中率上界不再成立 |
| 指标是单请求 TTFT 而不是任务完成时间 | 程序级调度看起来更差；**目标函数要换成会话/任务完成时间** |

## 5. 未解决

- [ ] 测量只覆盖一个脚手架、两种模型族、小样本（100 任务）；需要更多生产 trace。InferenceX AgentX 的公开轨迹语料是下一步可读的来源。
- [ ] 工具延迟分布能否被预测（工具类型、参数）以设定 TTL？
- [ ] SOSP 2026 的 *Scalable Context Orchestration for Serving LLMs Over Voice* 等是否给出另一类会话负载（只读到标题，[SOSP 2026 accepted](https://sigops.org/s/conferences/sosp/2026/accepted.html)）。

往哪走：[[A1 KV 状态变小之后]] · [[A7 RL 后训练系统]]（agent RL 的 rollout 就是 agent serving）· [[T1 状态的生命周期]] · [[00 入口]]
