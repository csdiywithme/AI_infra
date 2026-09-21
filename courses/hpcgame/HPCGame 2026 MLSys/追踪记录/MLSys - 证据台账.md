---
type: evidence-ledger
status: active
created: 2026-09-15
audit_scope: 历史定点复核与逐来源周度研究
last_verified: 2026-09-21
---

# MLSys：证据台账

> [!nav] 阅读入口
> [[HPCGame 2026 - 06 前沿追踪|当前判断与追踪入口]] · [[MLSys - 2026-08 至 09 旧版基线与日志|旧版原文归档]]

> [!important] 核验范围
> 9/15 初始化复核覆盖旧版 **2026-09-14 的六项记录**，建立四项勘误／口径澄清。9/21 增量见本页新证据索引与运行记录；早期基线和日志只补核了部分事件，仍未全部重新核验。未复核部分保留为历史线索，不能直接当作当前已核事实。这里核对公开一手文本、版本源码和归属，未在本地复现性能。

## 怎样读一张证据卡

一张卡对应一个可追溯的变化，ID 永久稳定。当前判断可以变，旧卡与旧日志不静默改写；通过“修正／取代”关系追加新证据。

| 字段 | 记录要求 |
| --- | --- |
| 发生日期与事件 | 标明模型发布、正式 tag、PR 合并、论文 v1/v2、issue 报告或修复的实际日期；分别记录，不能互相替代 |
| 最早已核日期 | 本轮能证明的最早公开节点；只看了 release 就写“该版已含”，不要声称机制首次提出 |
| 发现日期／核验日期 | 本任务何时看见、何时打开具体一手来源复核；发现旧事实不是新进展 |
| 精确身份 | 项目 + tag/commit/PR、模型 revision、论文版本或 issue；滚动文档需注明访问日期 |
| 可支持事实 | 来源明确支持的机制或结果；推论另列，避免将实现者的结论直接作为本笔记结论 |
| 成熟度／独立性 | 正式发布、实验功能、论文、roadmap/spec、用户 issue；作者／厂商自报、独立复测、维护者确认、仅文本核对分别标注 |
| 适用组合与限制 | 模型、硬件、dtype、并行方式、cache、工作负载、质量和延迟要求；未公开项明确写未知 |
| 关联判断／修正关系 | T1—T8 主线及受影响判断；旧记录定位；强化、演进、修正、撤回或待证 |
| 后续触发 | 哪个稳定版、修复 PR、独立复测、兼容组合或反例值得再次核查 |

主线：T1 数据、recipe 与训练；T2 后训练与计算预算；T3 serving；T4 模型结构、通信与内存；T5 集群、可靠性与隔离；T6 编译与 kernel；T7 新工作负载；T8 硬件与可移植性。一个事实可关联多条主线，但只建一张卡。

## 勘误索引

| 旧记录 | 处理 | 新依据 |
| --- | --- | --- |
| 2026-09-14 Transformers 条目把一组 cache/kernel 改动归给 v5.17.0 | ⚠️ 撤销“本周新增”；已在 8 月的 v5.15.0 记录 | [[#E-20260915-01]] |
| 2026-09-14 agent scheduling 称“第一个直接实例” | ⚠️ 撤回优先性断言；改为本次追踪新增的研究实例 | [[#E-20260915-02]] |
| 2026-09-14 DeepSeek 模型能力与 Dynamo 适配并列 | ⚠️ 补足支持组合；模型 FP4 KV／视觉能力不代表该 snapshot 支持 | [[#E-20260915-03]] |
| 2026-09-14 Φ-Bench 的百分比分数 | ⚠️ 澄清口径：综合 reward 分数，不是任务完成率；不撤销原数值 | [[#E-20260915-06]] |

以上在 2026-09-15 作为追加勘误生效；归档原文保持原样。

## E-20260915-01

**Transformers：版本与日期勘误。**

- **事件与身份**：正式 tag `v5.15.0`，2026-08-10 发布，release commit `5eddc12`；原错误引用 `v5.17.0` 是 2026-09-09 的另一版。
- **最早已核／发现／核验**：该组改动在 2026-08-10 正式版已含；具体 PR 首次公开及合并日期未回溯。旧任务 2026-09-14 记入，2026-09-15 核验归属。
- **可支持事实**：MLA cache 修复（#47761）、linear-attention 外部 kernels 改为显式 opt-in（#47630）、EncoderDecoder/OlmoHybrid assisted decoding（#47361）、sliding-window rollback（#47447）、continued forward 的 recurrent padding mask（#47087）均列在 v5.15.0。
- **成熟度／独立性**：正式发布的修复与行为变更；已核官方发布文本，未复现所有模型组合。
- **适用边界**：这些记录涉及不同模型与路径，不能外推为全部 hybrid 模型都已解决；不以“最新版本”替代具体版本的兼容性核验。
- **关联判断／修正**：T3、T4、T6。保留“cache rollback 与 continuation 是模型层 correctness 问题”的判断；撤销其作为 9 月 7—14 日新增进展的证据。勘误对象为旧版 2026-09-14 Transformers 整条。
- **后续触发**：这些 PR 的 regression、后续 release 的默认行为变化或真实部署复测。
- **一手来源**：[v5.15.0](https://github.com/huggingface/transformers/releases/tag/v5.15.0)；[原来误引的 v5.17.0](https://github.com/huggingface/transformers/releases/tag/v5.17.0)。PR 编号可从对应 release 逐项定位。

## E-20260915-02

**Agent workflow：撤回“第一个实例”的优先性断言。**

- **事件与身份**：论文 `arXiv:2609.10964v1`，2026-09-10 01:35 UTC 首次提交。
- **最早已核／发现／核验**：论文 v1 的公开日期 2026-09-10；旧任务发现于 2026-09-14；本次核验 2026-09-15。
- **可支持事实**：作者在 ready-turn release 层联合决定提交次序与未完成工作预算，使用 mean-CVaR 目标及在线工作量估计；在软件工程 agent 的真实执行 trace、多模型及不同到达率上，作者报告竞争负载下 workflow flow-time P95 最多改善 3.50×，轻载接近 eager release。
- **成熟度／独立性**：论文原型、作者评估；本次未见独立复测或生产部署证据。
- **适用边界**：最大改善值依赖 trace 与竞争程度；不表示每轮推理都快 3.50×，也不表示任务质量、费用或所有 workload 同比改善。
- **关联判断／修正**：T2、T3、T7。支持“端到端 workflow 目标可以改变提交策略”的推论。旧日志“第一个直接实例”缺少系统性优先性检索，正式撤回，替换为“本次追踪新增的 turn-release 研究实例”。
- **后续触发**：公开实现、真实环境部署、质量与成本共同控制的复测，以及和已有 workflow scheduler 的直接比较。
- **一手来源**：[论文 v1 与提交记录](https://arxiv.org/abs/2609.10964v1)。

## E-20260915-03

**DeepSeek-V4.1-Flash 与 Dynamo：模型能力和实现支持组合分开核验。**

- **事件与身份**：模型公告 2026-09-10；Dynamo `v1.6.0-deepseek-v4.1-flash-dev.1` 于 2026-09-12 发布，commit `7931147`。两者是不同事件。
- **最早已核／发现／核验**：以上公告／tag 是本次最早已核节点，未声称各子机制首次出现。旧任务发现 2026-09-14；本次核验 2026-09-15。
- **可支持事实：模型**：官方模型卡披露 20+20 层 Causal Encoder-Decoder、552B backbone、prefill/decode 激活 8B/16B；通过跨层 KV/索引复用、SWA bounded replay、FP4 main KV 降低状态开销，并包含 Engram、DSpark 与视觉输入。890 bytes/token 与相对 V4-Flash 的 HBM/SSD 缩减为作者自报，未独立复现。
- **可支持事实：实现**：Dynamo snapshot 建在 CUDA 13 的 SGLang `dev-dsv41` preview；其 8×GB200 recipes 使用 FP8 dense、FP4 experts、**FP8 KV**，且仅服务文本。aggregated profile 为两个 TP4/EP4 worker 并开 DSpark；1P1D 为每角色 TP4/EP4，**无 speculation**。
- **成熟度／独立性**：模型／API 正式发布；Dynamo 是非 QA-gated 实验快照，官方不建议生产，两个 profile 均未 benchmark，SGLang 适配尚未进入编号 release。两方资料仍均为实现者说明。
- **适用边界／修正**：T3、T4、T7、T8。旧日志没有声称 snapshot 已生产，但将多个能力列在一起容易误读；追加明确边界：FP4 KV、视觉、PD、DSpark 不能从模型卡拼成该 snapshot 的可用组合。“架构节省状态”与“开源 runtime 已完整实现”分别判断。
- **后续触发**：编号 SGLang/Dynamo 正式版、FP4 KV、视觉或 PD+speculation 的明确支持和对应 benchmark。
- **一手来源**：[官方公告](https://www.deepseek.com/en/news/deepseek-v4-1-flash/)；[模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)；[Dynamo 精确 tag 与支持矩阵](https://github.com/ai-dynamo/dynamo/releases/tag/v1.6.0-deepseek-v4.1-flash-dev.1)。

## E-20260915-04

**vLLM v0.29.0：默认 runner 与 RL 权重同步的正式版变化。**

- **事件与身份**：正式 tag `v0.29.0`，2026-09-09 发布，release commit `98dff2a`。
- **最早已核／发现／核验**：2026-09-09 正式版已含；机制首次 PR 日期未逐一回溯。旧任务发现 2026-09-14；本次核验 2026-09-15。
- **可支持事实**：MRV2 成为默认（#53183），MRV1 deprecated；`sharded_rdt` 让 worker 经 NIXL／Ray Direct Transport 获取自身 TP/EP 权重切片（#43375）。同版列有 Mamba prefix checkpoint、逐请求 speculation metrics、队列 admission control 与确定性分布式 prefix hash。
- **成熟度／独立性**：正式版发布说明已核；功能进入 release 不等于任意组合已经过生产验证。本卡不采用速度数字作独立性能结论。
- **适用边界**：sequence parallel、dual-batch overlap、elastic EP、自定义 logits 与部分 speculation 仍回退 MRV1；部分 ROCm 模型也保留旧路径。v0.32 移除 MRV1 是目标，不是已经发生的事实。
- **关联判断／修正**：T2、T3、T4、T8。旧条目的核心机制与日期获支持；“RL 与 serving 的联合调度面扩大”是本笔记推论，尚不证明收敛或训练成本获益。
- **后续触发**：回退组合补齐、MRV1 实际移除、RL 端到端质量／吞吐对照，以及默认切换引入的 correctness regression。
- **一手来源**：[vLLM v0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)，含 #53183、#43375 等原始变更链接。

## E-20260915-05

**cuDNN Frontend v1.29.0：按具体 engine 核验 attention 能力。**

- **事件与身份**：正式 tag `v1.29.0`，2026-09-13 发布，release commit `91dbf3e`，推荐配合 cuDNN 9.26+。
- **最早已核／发现／核验**：该正式版日期 2026-09-13；未将 release 中每条能力都称作首次实现。旧任务发现 2026-09-14；本次核验 2026-09-15。
- **可支持事实**：CuTe DSL DSA 加入 sparse forward，PyTorch SDPA 可经显式 provider 走 Python graph/router；FROST 加入 paged KV；Rubin SM107 增加独立的 prefill kernel lineage。
- **成熟度／独立性**：正式 release；其中 flex attention namespace 为 experimental，FROST 和 SM107 engine 显式 opt-in。本次仅核发布文本，没有跑硬件。
- **适用边界**：DSA forward 只覆盖列明的 SM100-family 与 head/prefill 形状，不含 decode、split-KV、常规 H128、SM90 或 FP8 cache。FROST paged KV 不接受 FP8/MXFP8 pool、sink、未对齐 page、d>256、缺 padding mask 或 packed block table。PyTorch provider 要求 2.13+ 并显式激活。
- **关联判断／修正**：T3、T4、T6、T8。强化“shape、dtype、cache layout 与架构决定 engine 路由”。补足旧条目遗漏的 DSA 支持边界，不能把“forward/backward API 已齐”读作训练推理全形状覆盖，也不以软件 target 推断硬件已经普遍生产部署。
- **后续触发**：支持矩阵扩展、opt-in 转默认、跨硬件实测及 engine 组合的正确性报告。
- **一手来源**：[cuDNN Frontend v1.29.0](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.29.0)，特别是 DSA、torch.sdpa、Paged KV、Rubin 各节。

## E-20260915-06

**Φ-Bench：百分比分数口径澄清。**

- **事件与身份**：论文 `arXiv:2609.10226v1`，2026-09-09 14:23 UTC 首次提交。
- **最早已核／发现／核验**：论文 v1 公开于 2026-09-09；旧任务发现 2026-09-14；本次核验 2026-09-15。
- **可支持事实**：85 项任务含 55 KFC、20 LHI、10 E2EO；性能任务先过 correctness gate，再作至少五组 AB-BA 配对测量。论文最高综合分 36.53，Hardware & Edge 最高 5.4，均以百分数展示。
- **口径澄清**：这是混合归一化连续性能 reward 与二值功能实施 reward 的综合分数。性能 reward 使用对参考解的对数归一；达到参考速度仍可能得到零分。**36.53% 不等于通过／完成了 36.53% 的任务**，也不能据此计算“失败任务比例”。旧记录的数值保留，追加此限制。
- **成熟度／独立性**：论文及作者 benchmark，未独立复测。每任务单 H20、8 CPU cores、32 GiB；KFC 单提交、LHI/E2EO 最多 16 个候选并取最好。GPT 使用 Codex，其他模型使用 Claude Code，scaffold 未统一。
- **关联判断**：T6、T8，兼及 T1—T5 系统任务。它支持“需测跨文件和整系统优化”的研究方向；不能仅凭低总分推导纯模型能力、通用成功率或实际工程人员替代率。
- **后续触发**：独立复测、同 scaffold／同预算比较、硬件迁移和作弊检测验证。
- **一手来源**：[v1 日期与摘要](https://arxiv.org/abs/2609.10226v1)；[v1 全文：Evaluation Metrics、Experimental Setup、Tables 2–3](https://arxiv.org/html/2609.10226v1)。

## 2026-09-21 证据索引

发现/核验日均为 2026-09-21；release 时间默认使用 UTC，必要时附北京时间。新版本纳入旧 PR 与机制首次公开分开记录。本轮为八主线来源扫描及重点深读，仍未完成历史和近 90 天的全部回填。

| ID | 事件 | 类型 / 主线 |
| --- | --- | --- |
| [[#E-20260921-01]] | verl 0.9.1：角色复用与权重切换协议 | 新版本 / T2、T3、T5 |
| [[#E-20260921-02]] | NVRx 0.7.0：异步快照一致性修复 | 新版本 / T1、T5 |
| [[#E-20260921-03]] | NCCL 2.32.3：诊断、GIN 与初步 Rubin 支持 | 新版本 / T4、T5、T8 |
| [[#E-20260921-04]] | SGLang-Omni 0.1.6：HTTP 准入与多阶段状态 | 新版本 / T3、T5、T7 |
| [[#E-20260921-05]] | cuTile Rust：共享 IR 辅助 AI 迁移验证 | 新报告 / T6 |
| [[#E-20260921-06]] | Shadow Engine Recovery：恢复的是容量，不是完整会话 | 8/25 补录 / T3、T5 |
| [[#E-20260921-07]] | Dynamo K-EXAONE：backend/PD 布局静默错误边界 | 实验快照 / T3、T4 |
| [[#E-20260921-08]] | FA4 b31：发布包与真实 paged-KV 组合验证 | 预发布 + 上游采用 / T2、T3、T6、T8 |
| [[#E-20260921-09]] | Megatron Core 0.19.2：FP4 与 1F1B overlap | 新版本 / T1、T4 |
| [[#E-20260921-10]] | SGLang 0.5.20：PD 初始化有界失败 | 新版本 / T3、T5 |
| [[#E-20260921-11]] | Kueue 0.19.5：预约释放与存活 Pod 对齐 | 新版本 / T5 |

## E-20260921-01

**verl：提高资源复用率，仍要证明有效学习成本下降。**

- **日期/身份**：v0.9.1 于 2026-09-20 发布，`1876b06`；#7373 为 8/12 公开、8/21 合并，#7511 为 8/21 公开、8/31 合并。本周新增的是版本事件。
- **新事实**：`separate_async` 可临时把闲置 trainer GPU 用于 rollout，默认关闭、不与 PD 组合；同步时在 verl 侧拦住新提交，修复 DP>1 drain 竞态。暂停引擎调度不等于关闭提交入口。
- **实验边界**：#7373 作者用 24 H100、Qwen3.5-35B-A3B、DAPO-Math-17k；Megatron TP2/PP2/CP2/EP8，vLLM TP4/n8，输入 2048/输出上限 32768；150 步从 18.80h 至 16.43h。训练 score 近似，但 mean staleness 0.522→0.556；没有独立质量等价评测。仅测有利的 2 trainer 节点 + 1 rollout 节点。
- **成熟度/影响**：正式版中的 opt-in 能力及修复，作者实验、未独立复现。🔄演进 T2/T3/T5；资源角色、请求门控、权重版本是同一协议，不能将固定步数提速直接写成 time-to-quality 提升。
- **后续**：不同资源比例和环境长尾下的独立质量—成本曲线；部分路径的组合限制。
- **来源**：[v0.9.1](https://github.com/verl-project/verl/releases/tag/v0.9.1) · [角色切换 #7373](https://github.com/verl-project/verl/pull/7373) · [同步门控 #7511](https://github.com/verl-project/verl/pull/7511)。

## E-20260921-02

**NVRx：异步写盘前，需要真正冻结 CPU 状态。**

- **日期/身份**：v0.7.0，2026-09-16 01:09:19 UTC，`ebf0352`。#376 于 7/21 创建、8/19 合并；#314 于 4/28 创建、8/8 合并，不是本周新发现的机制。
- **新事实**：persistent async checkpoint 在序列化前复制 CPU tensor，避免训练继续修改 optimizer state；CUDA storage 指针改变时刷新 IPC handle。版本源码含 CPU clone、SHM snapshot 和复用前 drain。
- **成熟度/边界**：修复随正式包发布；仅做源码/发布归属核对，无性能实验。该包 Attribution/Restart Agent 仍有 experimental 边界，不把整个 release 的全部组件视作成熟。
- **判断**：✅强化 T1/T5，并关联 T2。缩短暂停不能牺牲恢复点一致性；需要区分状态冻结、后台写入、持久化完成。
- **取证说明**：#376 网页缓存仍显示 Open，fresh API 显示已合并，且 tag 源码存在实现；本卡采用后两项，不沿用陈旧网页状态。
- **后续**：恢复后 optimizer/数据游标等价性、并发保存与重分配的回归。
- **来源**：[release](https://github.com/NVIDIA/nvidia-resiliency-ext/releases/tag/v0.7.0) · [#376](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/376) · [#314](https://github.com/NVIDIA/nvidia-resiliency-ext/pull/314) · [v0.7.0 固定源码](https://raw.githubusercontent.com/NVIDIA/nvidia-resiliency-ext/v0.7.0/src/nvidia_resiliency_ext/checkpointing/async_ckpt/filesystem_async.py)。

## E-20260921-03

**NCCL：通信能力、故障可观测性与硬件成熟度要同时验收。**

- **日期/身份**：`v2.32.3-1`，`12df1a1`，2026-09-17 21:09:42 UTC（北京时间 9/18）；API 与 release 正文一致。子机制首次提出日未逐项追溯。
- **新事实**：新增 socket GIN、CFT counted-write/wait；RAS 增加 GPU 进度计数及 watchdog DMA mirror，ATTN 暴露非致命 fallback/初始化问题。新增 Rubin SM107/CX9 等初步支持，但明确尚未做 Rubin performance-model tuning。
- **成熟度/边界**：正式 release；具体能力仍依赖平台与配置。没有本任务性能复测，也不能从 target 支持推出 Rubin 大规模生产可用；NetworkDirect 仍实验性。
- **判断**：✅强化 T4/T5/T8。通信验收从“能否传、带宽多少”扩展到“是否降级、为何停滞、何时失败”；硬件 enablement 与调优完成是两步。
- **后续**：真实 hang/降级诊断、Rubin 调优版、与并发计算共同测量的有效完成量。
- **来源**：[精确 release](https://github.com/NVIDIA/nccl/releases/tag/v2.32.3-1) · [发布日期 API](https://api.github.com/repos/NVIDIA/nccl/releases/tags/v2.32.3-1)。

## E-20260921-04

**SGLang-Omni：GPU 槽位限制不等于整条多模态流水线有界。**

- **日期/身份**：v0.1.6，2026-09-17 18:14 UTC，`7b49bc4`；#2163 于 9/14 合并，#1773 于 9/13 合并，#2115 于 9/12 合并。本周事件是版本纳入。
- **新事实**：ASR 在 HTTP 波形解码前做全局并发准入，超限返回 503；minimal PD 明确 KV page ownership，取消/释放需要安全调度边界。后者仍是最小实现，不能当作所有 speculative/projected-input 路径已支持。
- **实验陷阱**：#2163 作者用 RTX A6000 48GB、Qwen3-ASR-0.6B、64 段 24–109 秒英语音频、客户端并发 8；cap=8 的 WER 接近旧版，默认 cap=4 的评测却仅计入 5/64，其余因 503 跳过。已接纳样本上的指标不是全请求质量/产出。
- **成熟度/影响**：正式版本包含早期功能及修复；作者单环境测试，无独立复测。✅强化 T3/T5/T7：CPU 媒体内存、GPU 槽位、跨阶段状态和拒绝重试必须共同计费。
- **后续**：按全部提交请求计算成功率与 SLO goodput，包含 503、重试、取消和 host memory。
- **来源**：[v0.1.6](https://github.com/sgl-project/sglang-omni/releases/tag/v0.1.6) · [ASR #2163](https://github.com/sgl-project/sglang-omni/pull/2163) · [minimal PD #1773](https://github.com/sgl-project/sglang-omni/pull/1773) · [abort #2115](https://github.com/sgl-project/sglang-omni/pull/2115)。

## E-20260921-05

**cuTile Rust：AI 代码迁移可以由共享 IR 增加验证约束。**

- **日期/身份**：NVIDIA 技术报告 2026-09-16；迁移代码首次公开日及固定 commit 未核定，TileGym main 不能充当稳定版身份。
- **新事实**：cuTile Python、Triton-TileIR 与 Rust 共享 Tile IR/`tileiras`；工作流结合数值测试、IR 结构对照和性能测量，不只检查能否编译。
- **口径**：厂商报告 24 算子、347 配对配置，DGX B200，CUPTI device time，每配置取四次 CI 中最佳测量，Rust/Python 性能几何平均 0.995。不是端到端收益、跨硬件迁移或自动发现更优算法；部分操作仍用 unchecked API。
- **成熟度/影响**：公开实现/厂商技术报告与自测，未独立复现。🔄演进 T6：AI 优化可信度取决于参考语义、编译器中间结果和验收流程；共享 IR 不是自动正确性的证明。
- **后续**：锁定代码版本、跨 shape/极值/同步验证，报告 agent 与编译调优总成本。
- **来源**：[官方报告](https://developer.nvidia.com/blog/translating-cuda-tile-operations-from-python-to-rust-using-agentic-ai/) · [TileGym 目录（浮动 main）](https://github.com/NVIDIA/TileGym/tree/main/src/tilegym/ops/cutile_rs)。

## E-20260921-06

**补录：Shadow Engine Recovery 缩短容量恢复，不等于保全会话。**

- **日期/身份**：原始技术报告 2026-08-25；9/15 的 NVLink 6 文章再次引用不构成新 benchmark。本轮首次纳入基线，具体功能首次 commit 未追溯。
- **机制**：独立 GMS 保留共享权重，备用进程预建 communicator/graph；故障后接替服务。当前 preview 不继承 KV，且不覆盖硬件/节点故障。
- **实验口径**：厂商 2 个 B200 节点、每节点 TP8，GLM-5.2 NVFP4/FP8 KV，32k 输入/1k 输出、0.7 请求/秒；稳态 SIGKILL 一个 worker，观察 600 秒。第二 worker 恢复接流量由 283s 至 7.3s，不能解释成 GPU 故障或原请求无损恢复。
- **成熟度/边界**：preview，vLLM 主支持路径；Kubernetes ≥1.34 + DRA/对应驱动。未独立复测，额外上下文/graph/buffer 成本不为零。
- **判断**：✅强化 T3/T5，连接编译准备与 state lifetime。必须分开验收“进程恢复、容量恢复、原请求/会话恢复”。
- **后续**：KV 接续、整机故障域、长期 standby 成本及同 SLO 产出。
- **来源**：[8/25 原报告与完整实验](https://developer.nvidia.com/blog/restore-llm-inference-capacity-in-seconds-with-shadow-engine-recovery-in-nvidia-dynamo/) · [9/15 NVLink 6 分层恢复说明](https://developer.nvidia.com/blog/how-nvidia-nvlink-6-delivers-multi-layer-resiliency-for-ai-factories/)。

## E-20260921-07

**Dynamo K-EXAONE：自动 backend 选择和 PD 状态布局可能静默改变输出。**

- **日期/身份**：2026-09-17，`v1.4.1-k-exaone-2.0-750b-post.1`，`effab39`。问题首次发现日未披露；以该快照的公开 Known Issues 为证据。
- **组合/事实**：K-EXAONE-2.0-750B-A37B-NVFP4、B200、vLLM 0.28.0、FP8 KV；4 卡 TP4 aggregated 或 8 卡 1P1D。需明确选择 `FLASHINFER_CUTLASS`；auto 落到 `FLASHINFER_TRTLLM` 会在该 checkpoint 长输出中出现静默错误。PD 两端需一致 MTP 设置及 block-size 64。
- **成熟度**：分支实验快照，不走 stable QA、不建议生产；问题为维护者披露，未独立复现。不泛化为所有 FlashInfer、NVFP4 或 PD 都有问题。
- **判断**：⚠️修正 T3/T4 的验收粒度，承接 E-20260915-03 的组合原则，但不是同一个模型的修复。HTTP 成功与 load 成功均不足以证明模型语义正确。
- **后续**：修复 PR/稳定 release、长输出质量回归和 PD 两端配置校验。
- **来源**：[精确 tag 的 Known Issues](https://github.com/ai-dynamo/dynamo/releases/tag/v1.4.1-k-exaone-2.0-750b-post.1)。

## E-20260921-08

**FA4：上游合并、发布 wheel、被 runtime 实际选中，是三个验收点。**

- **日期/身份**：FA4 `fa4-v4.0.0.beta31` 于 2026-09-16 发布，`0dc2cb4`；#2810 8/19 公开、9/11 合并。TorchTitan #4737 于 9/16 创建、9/17 合并，`a3a819c`，提高文档最低版本要求。
- **新事实**：补齐 SM100 head_dim=256 forward 的 actual sequence length 与 paged-KV 路径；TorchTitan 维护者复现 b30 失败并验证发布 b31，而不是把 main 代码可用视为 wheel 可用。
- **验证边界**：GB200/SM100、BF16 forward、TP1；四类 GPU 用例对照 FP64 reference，并记录实际 FA4 调用。未使用 KV padding 必须为有限值，NaN poison 会污染输出；无性能重测、没有证明全新环境解析依赖可用。FP64 reference 不代表第三方独立复测。
- **成熟度/判断**：beta 包与已合并上游文档更新，✅强化 T2/T3/T6/T8 的精确组合验证。不能据此推定所有 vLLM/多卡/低精度路径自动启用。
- **后续**：runtime 默认选择、NaN/padding 约束、更多 TP/精度及端到端回归。
- **来源**：[b31](https://github.com/Dao-AILab/flash-attention/releases/tag/fa4-v4.0.0.beta31) · [#2810](https://github.com/Dao-AILab/flash-attention/pull/2810) · [TorchTitan #4737](https://github.com/pytorch/torchtitan/pull/4737)。

## E-20260921-09

**Megatron Core：FP4 能力要具体到层与并行调度路径。**

- **日期/身份**：`core_v0.19.2`，2026-09-18，`4b4acac`；#6135 于 7/29 公开、9/9 合并；发布说明通过 #7198 回移纳入本版。
- **事实**：版本将 Transformer Engine 依赖升至 2.18，包含 FP4 1F1B all-to-all overlap 支持；fine-grained forward 按层进入 FP8/FP4 context，MTP 层仍排除 FP4，数值验证未完。
- **成熟度/边界**：正式版本中的特定路径支持，发布文本/原 PR 核对；#7198 未单独深读，无本任务性能或收敛复现。
- **判断**：✅强化 T1/T4；低精度 × 层 × 调度的组合不能被“FP4 已支持”概括。
- **后续**：MTP 数值验证、模型质量与含通信的完整 step 时间。
- **来源**：[0.19.2](https://github.com/NVIDIA/Megatron-LM/releases/tag/core_v0.19.2) · [#6135](https://github.com/NVIDIA/Megatron-LM/pull/6135)。

## E-20260921-10

**SGLang：初始化 timeout 建立有界失败，但不是恢复机制。**

- **日期/身份**：v0.5.20，`94602c9`，2026-09-18 22:41:33 UTC（北京时间 9/19）；#37874 为 9/3 公开、9/4 合并。
- **事实**：PD transfer-engine 初始化为 NIXL/Mooncake/Ascend/Mori 设置共同 deadline，默认 60 秒。超时令启动失败并退出，不取消已挂调用，也不自动 retry。
- **成熟度/边界**：正式版纳入的修复；只核 release 和该 PR，不推断其他全部 PD 错误都被处理。不是 vLLM #55000/#55181 的修复证据。
- **判断**：✅强化 T3/T5；watchdog、终态和恢复策略是不同职责。
- **后续**：真实网络故障下的超时、进程回收与重试预算。
- **来源**：[v0.5.20](https://github.com/sgl-project/sglang/releases/tag/v0.5.20) · [#37874](https://github.com/sgl-project/sglang/pull/37874)。

## E-20260921-11

**Kueue：预约账本不能比实际 Pod 生命周期先走一步。**

- **日期/身份**：v0.19.5，2026-09-17 14:55:26 UTC，`8e60d76`；原改动首次公开/合并日未在本轮独立核定。
- **事实**：ElasticJobsViaWorkloadSlices 路径在存在 pending scale-up slice 时，等待被驱逐 Job 的 active Pods 停止后释放 reservation；ConcurrentAdmission 修补 QuotaReserved/PendingEvaluation 分支。
- **成熟度/边界**：正式 patch release 中的特定 feature 修复；两原 PR 单页访问失败，结论限 release 明示内容。未测调度吞吐或长期公平性，也不据此判全部弹性能力 GA。
- **判断**：✅强化 T5，资源账面可用与实际进程退出必须对齐；自定义 ComposableJob integration 另需注意 Load 签名变化。
- **后续**：复核原 PR、抢占/扩缩故障实验和预约释放时序。
- **来源**：[v0.19.5](https://github.com/kubernetes-sigs/kueue/releases/tag/v0.19.5)。

## 新卡模板

```markdown
## E-YYYYMMDD-NN

**一句话事实。**

- 事件与身份：
- 最早已核日期／发现日期／核验日期：
- 可支持事实：
- 成熟度／独立性：
- 适用组合／benchmark 口径／未知项：
- 关联 T 主线与当前判断：
- 修正／取代关系：
- 后续触发：
- 一手来源：
```

## 待复核范围

- 旧基线与 2026-08-31／2026-09-07 日志：保留原文；除运行记录明确列出的定点复核外，其余仍为未重新逐项核验的历史线索。再次用于当前结论时，先核对原始依据。
- 2026-09-15 的六张旧卡：只证明列明文本与归属，不表示已完成当日全领域增量扫描。9/21 的新增卡同样不意味着全部来源与历史均已核。
- 日期去重：论文 v1、v2、新代码、正式版、独立复测分别记事件。搜索重新发现旧论文、release 页面晚更新、转发或榜单重复不算新增。
