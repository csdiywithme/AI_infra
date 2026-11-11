---
type: evidence-ledger
status: archived
created: 2026-09-15
audit_scope: 历史定点复核与逐来源周度研究
last_verified: 2026-10-19
---

# MLSys：证据台账

> [!nav] 阅读入口
> [[领域地图与覆盖缺口|AI Infra 领域地图]] · [[MLSys - 2026-08 至 09 旧版基线与日志|旧版原文归档]]

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
| 2026-10-12 HiSparse 残余 P/D race 的风险描述未纳入后续反证 | ⚠️ 10/19补录10/5撤回提案及自然时序未复现；注入延迟hazard仍在，不宣称已修或普遍安全 | [[#E-20261019-05]] |
| 10/19 核查时将 vLLM #60374 记为 Open | 2026-11-11 回看官方 PR，确认 10/19 已以 `4bd75e9` 合入 `main`。发布包与端到端故障验证各自按对应证据判断。 | [[#E-20261019-04]] · [官方 PR](https://github.com/vllm-project/vllm/pull/60374) |

归档原文保留各自的核查时点。

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

## 2026-09-28 证据索引

以下均于 2026-09-28 发现并核验。新增 release、main 合入与旧结果补录分开；日期采用已取得的 API UTC 或原页面日历日期，不用搜索索引日期替代。除特别说明外，仅核原始文本和版本归属，无本任务 GPU 或独立性能复测。

| ID | 事件 | 类型 / 主线 |
| --- | --- | --- |
| [[#E-20260928-01]] | AReaL：部分组保留与失败归因进入学习目标 | main 合入 / T2、T5、T7 |
| [[#E-20260928-02]] | verl：权重 refit 的隐藏布局与 live storage | main 修复 / T2、T3、T4、T6 |
| [[#E-20260928-03]] | vLLM 0.30：HiSparse 分层 KV、Fast Start 与 runner 边界变化 | 正式版本 / T3、T4、T5 |
| [[#E-20260928-04]] | Dynamo 1.5：驻留策略与可达性索引分工 | 正式版本 / T3、T4、T5 |
| [[#E-20260928-05]] | Kueue 0.19.6：实际预约与 reclaim 计数一致 | 正式修复 / T5 |
| [[#E-20260928-06]] | FlashInfer 0.7：部署匹配的调优与跨进程复用 | 正式版本 / T3、T4、T6 |
| [[#E-20260928-07]] | cuDNN Frontend 1.30：编译制品身份与执行合约 | 正式版本 / T3、T5、T6 |
| [[#E-20260928-08]] | CUTLASS 4.8：正式包仍包含 preview 编译器/硬件路径 | 正式版本 / T6、T8 |
| [[#E-20260928-09]] | ROCm：多 GPU profiling 的节点级故障域 | 9/15 问题补录 / T5、T8 |
| [[#E-20260928-10]] | DeepSpeed Muon：loss 下降仍可能跑错 optimizer | 9/16 版本补录 / T1、T5 |
| [[#E-20260928-11]] | SGLang 分离式 diffusion 的控制面信任边界 | 新公开 CVE / T3、T5、T7 |

## E-20260928-01

**AReaL：哪些失败被保留、丢弃或计零分，会改变实际学习分布。**

- **日期/身份**：#1721 于 9/16 公开、9/23 合入 `46c4403`；#1750 于 9/22 公开、9/26 合入 `643b20b`。最新正式版仍为 8/25 的 v2.1.0，不能把新 main 行为归给该版。
- **新事实**：v2 **offline** rollout 默认保留不完整组中成功导出的成员，仅在可用成员间归一化 reward，同时清理完整 session 组；可用 `drop_incomplete_group` 恢复严格丢组，或限制最小可用组。online 行为未变。随后 #1750 提供 **v1-only** mean-only normalization，并尝试把模型失败和系统失败分开，模型失败可保留为零 reward 轨迹。
- **反证/边界**：#1750 明示审查风险尚未解决：invalid native receipt 仍可能回退到 legacy 分类，context overflow 可绕过 receipt 健康检查。缺 concat-leaf reward 不能由 scalar outcome 补造；不能写成“失败归因已解决”。
- **验证主体**：作者实现；#1721 未跑本地 pytest/多节点 GPU integration，CI 通过不能替代后者；#1750 报告 284 项定向 CPU tests，明确未验证组合后的真实模型 RL 学习正确性。
- **判断/承接**：🔄演进 T2/T5/T7，扩展 E-20260921-01 的状态协议判断：故障处理策略本身会选择训练样本、改变归一化分母。没有独立质量—成本改善结论。
- **后续**：receipt 隔离修复、正式版、按失败类别统计组保留率和独立质量。
- **来源**：[#1721](https://github.com/areal-project/AReaL/pull/1721) · [#1750](https://github.com/areal-project/AReaL/pull/1750) · [v2.1.0](https://github.com/areal-project/AReaL/releases/tag/v2.1.0)。

## E-20260928-02

**verl：传对数值还不够，refit 必须恢复 kernel 布局并保留使用中的存储。**

- **日期/身份**：#7986 于 9/22 公开并合入 `12ebe0c`；#7987 于 9/22 公开、9/24 合入 `6093e00`。作者实验发生于 9/21、9/23，不等于首次公开日。均晚于正式版 v0.9.1，属 main 修复。
- **机制**：FP8 MoE 重排可能替换 storage 却保持 shape/dtype，旧 refit 因此漏掉重排，向 kernel-layout 缓冲写入 checkpoint-layout 数据。修复追踪真实 storage replacement 并重做 stage/process。Blackwell BF16 的 4-D BlockMajorK 则暴露 shape 改变；新路径在 live storage 上建 checkpoint-shape view，写入后逐层 fold，保留参数对象/地址。
- **验证不能拼接**：FP8 主线 Hopper + block-FP8 `[128,128]` + EP>1 + FlashInfer CUTLASS 仅获源码/CPU 核查；实际 GPU probe 用的是另一未合并 MXFP8/TRT-LLM 分支。BF16 作者在 1×B200、随机两层八专家 Qwen3-MoE、TP1、Megatron→vLLM 0.24 验证旧版首次 sync 崩溃、新修订多轮 byte/layout/address 一致；vLLM 0.29 只有源码核查，真实 30B TP2/EP2 仍待测。
- **质量边界**：tiny-model GSM8K GRPO 的 reward 全为零；这是实现验证，不是有效学习或收敛证明。不采用被替代 PR 修订的 allocator 峰值作为真实大模型加速比。
- **判断/承接**：✅强化 T2/T3/T4/T6，承接 E-20260921-01/07 的组合验收。shape/dtype 相同不证明权重解释方式相同；优化链还要核 storage、layout preparation 和地址生命周期。
- **后续**：正式版归属、Hopper EP>1 GPU 验证、多卡真实模型反复 refit 与质量曲线。
- **来源**：[#7986](https://github.com/verl-project/verl/pull/7986) · [#7987](https://github.com/verl-project/verl/pull/7987) · [v0.9.1](https://github.com/verl-project/verl/releases/tag/v0.9.1)。

## E-20260928-03

**vLLM：KV 可复用、host 页可命中、权重可重载是不同生命周期。**

- **日期/身份**：正式 v0.30.0，9/22 05:20:54 UTC，`ced6857`。HiSparse #53781 为 8/25 公开、9/12 合入 `d43bb2f`；Fast Start #54921 为 9/2 公开、9/4 合入 `a69e75b`。本周新事件是 release 纳入。
- **机制**：HiSparse 在压力出现时将 KV spill 到 pinned host，稀疏 decode 仅搬入 top-k 缺失行到请求级 GPU hot buffer，indexer cache 保持 GPU 驻留。所有 worker 确认 copy 按 stream 顺序入队后才允许原 GPU block 复用；全部 copy 完成后 host 页才发布为 prefix hit。PD 中 NIXL 负责搬运，HiSparse 决定 decode 侧落点。
- **同版关系**：Fast Start 用 per-GPU daemon 保留量化后、TP 分片权重，经 CUDA IPC 供 engine 使用；不意味着保存了 KV/原会话或能抵抗整机故障。
- **成熟度/组合**：正式版本、维护者实现说明；HiSparse 原 PR 限 NVIDIA sparse-MLA 路径，不外推所有 cache 组/ROCm/模型。图表实验条件未完整核取，本卡不采速度数字。
- **演进而非旧卡勘误**：E-20260915-04 的 v0.29 回退边界保留为历史。v0.30 已列 MRV2 DBO eager/FULL CUDA graphs 与 PP 下 MTP、EAGLE3/DFlash/DSpark；当前判断不能继续照搬旧回退清单，也不能改成全组合成熟。
- **判断/后续**：🔄演进 T3/T4/T5。分别测搬运所有权、可见性、失败回收，以及固定质量/SLO 下稀疏性、host 带宽和并发的联合效果。
- **来源**：[v0.30.0](https://github.com/vllm-project/vllm/releases/tag/v0.30.0) · [HiSparse](https://github.com/vllm-project/vllm/pull/53781) · [Fast Start](https://github.com/vllm-project/vllm/pull/54921)。

## E-20260928-04

**Dynamo：engine 管驻留，router 管缓存是否完整且可达。**

- **日期/身份**：正式 v1.5.0，9/21 20:21:52 UTC（北京时间 9/22），`b83b1d9`，晚于上次运行。Mooncake 索引 #11239 于 7/6 公开、7/31 合入 `4c65c30`；Disk/locality #11571 于 7/11 公开、7/30 合入 `8a753c3`。
- **版本变化**：KVBM deprecated，但 1.5 仍带旧 wheel/examples；1.6 移除只是计划。官方推荐 engine-native host/disk offload；早期 KVCR 做跨节点共享，明确不是 KVBM 替代品。
- **索引合约**：Mooncake ordered events 维护 shared index；命中要求 TP/PP/layout 对应的物理对象完整，`group_id` 仅为校验后的 fast path。事件缺口/断流保守清空，尚无 snapshot/replay，晚订阅可能看不到既存对象，合同限 default tenant。另一 vLLM STORAGE 路径是 worker-local Disk，REMOTE/未知 locality 不进入该索引；不据此推定所有未知 media 均已 fail-closed。
- **不能拼接版本**：该版依赖 SGLang 0.5.18、TRT-LLM 1.3.0rc25、vLLM 0.28.0，各 backend 的 NIXL 版本也不同。不能自动获得本周 vLLM 0.30 功能，不能视为先前 V4.1/K-EXAONE 分支快照已通过正式 QA。GMS V1 等仍标 experimental。
- **判断/后续**：🔄演进 T3/T4/T5；“有副本”与“能安全路由为命中”分离。维护者实现/测试，无独立系统 benchmark；下一问为丢事件重建、多租户隔离与完整请求 goodput。
- **来源**：[v1.5.0](https://github.com/ai-dynamo/dynamo/releases/tag/v1.5.0) · [#11239](https://github.com/ai-dynamo/dynamo/pull/11239) · [#11571](https://github.com/ai-dynamo/dynamo/pull/11571)。

## E-20260928-05

**Kueue：回收的计数单位必须和实际预约一致。**

- **日期/身份**：v0.19.6 于 9/24 13:28:08 UTC 发布，`f0e95ce`；原 PR #16111 同日公开/合入，release backport #16132 同日合入 `ff16cab`。
- **事实**：`parallelism > completions` 时，实际只预约两者最小值，旧 ReclaimablePods 却按 parallelism 回收。作者例中 10/5 的 Job 仅完成一个 Pod，就错误释放仍运行的四个 Pod 的配额。修复 Indexed/NonIndexed 路径，统一按 `podsCount()`；同版也修复 StatefulSet/LeaderWorkerSet 已无 Pod 却仍持有 quota 的相反问题。
- **成熟度/边界**：正式 patch、维护者 PR 和测试说明；未做集群复现，没有可引用的利用率提升比例。不能把其余 DRA/TAS/故障功能一并视为 GA。
- **判断/承接**：✅强化 T5。承接 E-20260921-11，但本次是计数不一致，不是上周弹性 scale-up 的同一 bug；提前释放和滞留预约都破坏资源账本。
- **后续**：完成、驱逐、缩零和恢复路径的 quota invariant；原始剩余 bug 是否回归。
- **来源**：[v0.19.6](https://github.com/kubernetes-sigs/kueue/releases/tag/v0.19.6) · [#16111](https://github.com/kubernetes-sigs/kueue/pull/16111) · [#16132](https://github.com/kubernetes-sigs/kueue/pull/16132)。

## E-20260928-06

**FlashInfer：调优缓存要保存部署条件，而不只是一个最快 kernel 编号。**

- **日期/身份**：v0.7.0，页面发布日 9/22，`4d75a33`，正式版；Autotuner v2 原 PR #3861 最早可见 7/7、9/5 合入。不是把旧 rc 或 PR 重新写成首次机制发布。
- **事实**：新增 opt-in、文档仍标 experimental 的 `autotune_v2()`，按 eager/CUDA-graph 测量并把 policy 纳入环境身份；原子条目供多 rank 写入，同构 rank 可 reload 收敛。提供 tactic revalidation 合约，具体 runner 需接入。旧 API 保留；默认 auto 未全面改成 CUDA-graph，代表性 MoE probe、全部 runner 的校验接入仍是独立工作。
- **可引用的实验范围**：正式版引用作者 vLLM、Qwen3-8B-FP8、TP2、B200 同缓存重启：tuning window 128s→1s，**总启动** 410.7s→165.3s。两组数字不是同一计时范围，不是稳态吞吐；请求分布/并发/SLO/模型质量未在该摘要中给出，因此不推出 serving 加速倍数。无独立复测。
- **框架接入边界**：RFC 中 vLLM 迁移 #55671 仍为 draft，不代表已发布 vLLM 默认采用。异构 EP rank 的部分 warm-cache 情形仍需 collective 次数一致性保障，不能把同构 rank reload 测试外推到全部分布式组合。
- **成熟度不能整体提升**：同版统一 MoE API 转正式，但 PrimTS MoE 仍 experimental、B300 qualification 待完；experimental backend 默认不参加自动选择，需显式 opt-in。不能从发布版推断任意低精度 backend 可互换。
- **判断/后续**：🔄演进 T3/T4/T6。准备成本可跨进程摊销，但复用受硬件/软件/测量模式约束；下一问为真实输入分布、EP 路由形状及升级后失效/重调优。
- **来源**：[v0.7.0](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.0) · [#3861 原设计和边界](https://github.com/flashinfer-ai/flashinfer/pull/3861) · [RFC 与验证](https://github.com/flashinfer-ai/flashinfer/issues/3920)。

## E-20260928-07

**cuDNN Frontend：跨进程复用的是受环境约束的可执行制品。**

- **日期/身份**：正式 v1.30.0，页面发布日期 9/23，`42286a6`；不是上周的 1.30 nightly。#1031 为 9/12 公开、9/13 合入；#1146 为 9/19 公开并合入。API 补查 403，不对页面时间补造更高精度。
- **机制**：缓存导出的 tvm-ffi 对象而不只是 MLIR；身份包含前端源码摘要、DSL/tvm-ffi、driver/GPU 属性，未知 manifest/可疑条目按 miss 处理。同版修补 FlashInfer graph 的 layout/dtype 与绑定合约。另有 f16/bf16 SM100/107 路径 stride Int32→Int64 修复，特定大 stride 曾导致 abort/静默 alias；该问题不涵盖 FP8/MXFP8/THD。
- **实验边界**：#1031 作者在 SM100、cuDNN 9.25.1、DSL 4.7.1、tvm-ffi 0.1.11 的 121 项测试中报告 cold 44s→warm 12s，第二进程 bit-identical。是测试总时长/编译摊销，不是线上吞吐或会话恢复；没有独立复测。
- **判断/承接**：🔄演进 T3/T5/T6，补强 E-20260915-05 的按 engine 验收原则；形状合法还需布局/绑定/整数范围正确。正式包不意味着其中所有 engine 已成熟。
- **后续**：跨版本 cache invalidation、动态 B/H/layout 的回归，以及包含准备成本的真实 worker 周期收益。
- **来源**：[v1.30.0](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.30.0) · [#1031](https://github.com/NVIDIA/cudnn-frontend/pull/1031) · [#1146](https://github.com/NVIDIA/cudnn-frontend/pull/1146)。

## E-20260928-08

**CUTLASS：4.8 正式发布没有把所有 preview 路径变成成熟能力。**

- **日期/身份**：v4.8.0，页面发布日期 9/22，`098de2a`；相对 8/27 的 v4.8.0dev 是正式发布事件，不宣称其中机制均首次出现。
- **事实/边界**：`cute_ext` 编译路径仍需 `CUTE_DSL_USE_EXTENSION_COMPILER=1`，处于 preview、可能改变 PTX/SASS；默认切换属于未来计划。Operator API Rubin GEMM 仍 preliminary preview、待调优；发布说明要求尚待 CUDA 13.4 GA 配套提供的 R615 才能执行 SM107，Developer Preview R610 不足，不据此宣称工具链已可获得。nvMatmulHeuristics 当前限定 Blackwell。
- **兼容性证据**：release 列明经过测试的下游 commit 组合，不代表任意 FlashAttention/FlashInfer/cuDNN/PyTorch/TRT-LLM 版本均兼容。只核 release，无硬件或性能复现。
- **判断/后续**：✅强化 T6/T8 的“软件 target、工具链可得性、硬件可部署、性能调优”分阶段验收。更新 05 的当前版本边界，保留旧 dev 记录；下一问为默认切换和真实硬件/依赖矩阵。
- **来源**：[v4.8.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.8.0)。

## E-20260928-09

**补录：诊断作业本身也可能扩大故障域。**

- **日期/身份**：ROCm Core SDK 10.0.0 于 8/26 发布，TheRock tag `therock-10.0` / `16adc4d`；#8229 的最早已核公开日为 **9/15**。上周仅发现 release 入口、正文访问失败，本周补核版本文档与问题；不能把后补的 Known Issues 全部倒推为 8/26 已公开。
- **事实**：官方 Verified Issue 报告，单 MI300A 节点多 GPU 并发运行 `rocprofv3 --pmc` 或 `--kernel-trace` 可能触发整节点 hard reset，影响共驻作业，并需人工 power cycle。官方给出的 workaround 是含修复的 ROCm/rocm-systems develop 构建。
- **成熟度/限制**：维护者确认的问题与 workaround，不是泛化到全部 AMD/GPU profiler 的结论；未锁修复 commit，也未证实修复进入正式包。网页显示 Open，但 fresh API 本次 403，不能把缓存状态声称为即时状态。本任务不运行可能有破坏性的复现。
- **判断**：✅强化 T5/T8，收窄“观测只是被动开销”的假设；多租户隔离/故障注入应把诊断进程、节点恢复权限与故障范围一并纳入。
- **后续**：固定修复 PR/commit、正式版归属与多 GPU 隔离回归。ROCm 10.0 的其余支持矩阵补查见运行记录，不据正式版标题推定全平台等价。
- **来源**：[版本正文](https://rocm.docs.amd.com/en/docs-10.0.0/about/release-notes.html) · [#8229](https://github.com/ROCm/TheRock/issues/8229) · [release tag](https://github.com/ROCm/TheRock/releases/tag/therock-10.0)。

## E-20260928-10

**补录：loss 能下降，不证明实际执行的是所配置的 optimizer。**

- **日期/身份**：DeepSpeed v0.19.7，9/16 22:00:09 UTC，`f5af15c`；#8442 于 9/6 公开、9/8 合入 `cbd303e`；#8438 于 9/6 公开、9/10 合入 `0166cc8`。均为较早事件，此次补齐 T1 来源缺口。
- **事实**：stage 0 的 Muon 原先未调用 Newton–Schulz，矩阵更新退化成 raw-gradient SGD，loss 仍下降导致旧测试漏检。另一 ZeRO3 路径在 `zero.Init` 后依据一维占位 parameter 的 `ndim` 分类，错误转入 AdamW；修复读取原始 `ds_shape`。
- **验证/边界**：正式版纳入、作者测试；前者报告七项路径测试，后者用两层 Llama、ZeRO3、六步验证 Muon 更新计数 0→84。不是长期收敛、最终质量或性能优势证据；没有本任务 GPU 复测。
- **判断/后续**：✅强化 T1/T5。除了检查 loss，要验收 optimizer 分支和原始逻辑 shape；待真实 recipe 的收敛等价性、更多精度与恢复后路径验证。
- **来源**：[v0.19.7](https://github.com/deepspeedai/DeepSpeed/releases/tag/v0.19.7) · [#8442](https://github.com/deepspeedai/DeepSpeed/pull/8442) · [#8438](https://github.com/deepspeedai/DeepSpeed/pull/8438)。

## E-20260928-11

**SGLang：分离式多阶段服务的内部消息接口也是信任边界。**

- **日期/身份**：CERT/CC CNA 的 `CVE-2026-93088`，PUBLISHED，9/22 14:07:45 UTC 首次公开；9/17 reserved 不是公开披露日。发现/核验 9/28。
- **已确认范围**：CNA 明列 SGLang 0.5.11～0.5.14 的分离式 diffusion orchestrator：DiffusionServer 的无认证 ZeroMQ ROUTER 在验证前反序列化收到的 pickle 消息，构成无认证任意代码执行风险。该记录没有给出修复版本。
- **边界**：GitHub advisory 是 Unreviewed、版本 Unknown，不用它覆盖 CNA 的明确范围。v0.5.20 固定源码仍可见相应反序列化调用，但未完成 helper、网络策略和部署可达性审计；不能自行宣称 0.5.20 已确认受影响，也不能推定新 release 已修复。不推定用户部署或主机受影响。
- **成熟度/验证主体**：正式 CNA 风险记录 + 定点源码阅读，不是本任务漏洞复现。原发现者报告访问失败；后续需维护者公告、修复 PR、受影响/已修版本及防护验证闭环。
- **判断**：⚠️修正 T3/T5/T7 的验收范围：EPD/多阶段扩缩不只新增传输成本，还新增内部控制面和反序列化信任边界。HTTP 入口认证不能独自证明内部接口已隔离。
- **来源**：[CNA 原始记录](https://cveawg.mitre.org/api/cve/CVE-2026-93088) · [GitHub advisory](https://github.com/advisories/GHSA-xv76-mrpv-26pg) · [v0.5.20 固定源码](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/multimodal_gen/runtime/disaggregation/orchestrator.py)。

## 2026-10-12 证据索引

本轮为 9/28 之后的增量与候选补核，不补造 10/5 运行。除 E-08 首次发现为 9/28 外，下列卡片首次发现、最近核验均为 **2026-10-12**。日期按来源 UTC 记录；机制公开、合入和 release 分开。

| 证据 | 变化 | 状态 / 主线 |
| --- | --- | --- |
| [[#E-20261012-01]] | slime 持久样本与 trainer-only 恢复 | 正式版本、限定恢复 / T2、T5 |
| [[#E-20261012-02]] | score centering 与实现支持边界 | 论文补录、正式版/main 分开 / T2、T4 |
| [[#E-20261012-03]] | HiSparse draft 状态与跨 stream 修复 | 正式修复；PR披露残余PD race，后续修复未核 / T3、T4 |
| [[#E-20261012-04]] | SGLang 取消 drain 与权重更新 session | 正式版本、限定 backend / T2、T3、T5 |
| [[#E-20261012-05]] | Dynamo 校验地址与实际连接绑定 | 正式修复 / T3、T5、T7 |
| [[#E-20261012-06]] | MaxText overflow 的原状态 replay | 正式实现、定向 TPU 测试 / T1、T5、T8 |
| [[#E-20261012-07]] | KVTether：消息生命周期进入 KV 管理 | 新论文、作者回放 / T3、T7 |
| [[#E-20261012-08]] | SWE-Serve：服务全链路验收缺口 | 9/22 论文补录 / T5、T6 |
| [[#E-20261012-09]] | GPUNetIO 的共同底层与路径选择 | 新报告、既有机制补录 / T4、T8 |
| [[#E-20261012-10]] | FlashInfer 同 prepared weights 的不同计算路径 | 正式包、显式选择 / T4、T6、T8 |
| [[#E-20261012-11]] | Kueue 防止旧 placement 覆盖新 admission | 正式修复 / T5 |
| [[#E-20261012-12]] | V4.1 dev.2 扩展 PD speculation，但更换 backend | 非 QA preview / T3、T4 |

## E-20261012-01

**slime：持久样本、模型提交和进程恢复必须对齐。**

- **事件/日期**：新增正式 `v0.4.0`，10/9 03:04:12，`0b0c277`。Straw #2410 于9/24公开、9/28 04:39合入 `6857224`；trainer-only recovery #2444 于10/5公开、10/7合入 `4a7a509`。
- **变化**：Straw 持久化 prompt、partial rollout、样本及 batch，Ray 传控制/引用；detached ServingCluster 可保留健康 serving/router/队列并替换失败 trainer/manager。训练已执行不等于模型已 durable commit，重放数据须等提交边界再释放。
- **成熟度/边界**：版本已发布，默认仍 object-store/sync；当前共享 FS 目标为 JuiceFS。同一存活 Ray 集群内手动恢复，不提供自动任务重试/coordinator failover；冷启动不恢复 GPU KV、generation RNG。改 TP/PP 不保证 bitwise 一致，checkpoint 目录不包含全部池 payload。
- **验证**：作者 H100/JuiceFS 故障测试；真实 OOM 与 manager-loss 分别涉及小模型和 Qwen3-30B-A3B，后者用 stateless Adam，不能推成普通 Adam 全覆盖。未核独立长期质量、恢复 SLO 或成本优势。
- **判断/下一问**：✅强化 T2/T5，承接 E-20260921-02 与 E-20260928-01。恢复验收单位应是提交模型与被接受样本；后续核 Ray/owner/存储故障及重放的学习等价性。
- **来源**：[release](https://github.com/THUDM/slime/releases/tag/v0.4.0) · [#2410](https://github.com/THUDM/slime/pull/2410) · [#2444](https://github.com/THUDM/slime/pull/2444) · [固定恢复指南](https://github.com/THUDM/slime/blob/v0.4.0/docs/en/advanced/fault-tolerance.md) · [Straw 边界](https://github.com/THUDM/slime/blob/v0.4.0/docs/en/advanced/straw.md)。

## E-20261012-02

**Score centering：分布差异不仅是吞吐问题，也可能改变梯度目标。**

- **日期/增量**：论文 v1 为9/17 17:58:17，旧结果补录；slime #2405/#2406 于9/23合入 `4c4adbb`/`86c97a3`，10/9随0.4.0发布为新增版本事件。verl #8010 于9/23公开、10/1 14:52合入 `8718ca3`，仍为 main，未进入最新0.9.1正式版。
- **机制**：对 sampler/trainer 分布不同造成的 score 梯度漂移作修正，可与 importance sampling 组合；不是 reward/group centering，也不是使低精度引擎数值相同。
- **组合**：verl 默认关闭，仅 v1/TransferQueue、vLLM、FSDP、single-turn、bypass REINFORCE；支持列明 sync/async 路径，要求 top_p=1、top_k=-1、processed logprobs，禁 fused kernels。不能拼入 SGLang/Megatron/多轮；10/11提出的 #8196 truncated-support 仍 Open。slime 已发布的 top-p replay 是另一实现范围。
- **作者实验**：2×RTX PRO 6000、vLLM0.29/torch2.13、Qwen3-0.6B Countdown、FP8 W/A+KV sampler 对 BF16 trainer，400步/每臂一seed/64 held-out prompts；SC final held-out 0.56，对 TIS 0.31。另需 [#8064](https://github.com/verl-project/verl/pull/8064) 的 tied-embedding FP8 weight-sync 修复，不是#8010单独可复现的组合。长跑用旧 dict 提取，优化后仅三步复查；训练 reward 与 held-out 分开，无独立 time-to-quality 结论。
- **判断/下一问**：🔄演进 T2/T4；核修正开销、有效样本与独立质量，不能由固定步数分数推出学习更便宜。
- **来源**：[论文正文](https://arxiv.org/html/2609.20807v1) · [日期](https://arxiv.org/abs/2609.20807) · [slime #2405](https://github.com/THUDM/slime/pull/2405) · [#2406](https://github.com/THUDM/slime/pull/2406) · [verl #8010](https://github.com/verl-project/verl/pull/8010) · [尚未合入 #8196](https://github.com/verl-project/verl/pull/8196)。

## E-20261012-03

**vLLM：target forward 完成，不代表 draft 状态已经可镜像。**

- **日期/身份**：正式 `v0.31.0`，10/5 06:44:55，`db9527a`；#59309 于9/29 21:59公开、10/2 06:04合入 `4056c8a`。
- **修复**：HiSparse+MTP+FULL graphs 在GPU池饱和时，可能过早将尚未写入的 draft rows 镜像到host，copy stream 也可能读取上次 request IDs。新路径区分target/draft层，将draft mirror推迟到下一步开始，并在copy stream构造当前IDs。
- **验证主体/口径**：提交者H100测试及H200/B300指定配置。B300四卡、GLM-5.3 FP8 TP4、MTP3、FULL_DECODE_ONLY、8GiB GPU KV、40条约16k/512 code prompts，自报acceptance length 1.37→2.68；这不是吞吐比，无独立复测。
- **未闭合**：原PR仍承认 deferred host writes 与 NIXL P/D import 的 host-block reuse 缺排序；本次未核到后续修复。不能推出HiSparse+PD+MTP全组合正确。
- **判断/承接**：⚠️修正 E-20260928-03 的当前适用范围，不撤销其历史版本事实。T3/T4的可见性必须覆盖draft与跨stream生命周期。下次核剩余race、真实并发/SLO与版本回归。
- **来源**：[v0.31.0](https://github.com/vllm-project/vllm/releases/tag/v0.31.0) · [#59309](https://github.com/vllm-project/vllm/pull/59309)。

## E-20261012-04

**SGLang：取消后释放与多桶换权重，都需要完成协议。**

- **日期/身份**：正式 `v0.5.21`，10/2 01:09:04，`e00930c`。#41023 于9/24 03:15公开、14:35合入 `cd11037`；#40777 于9/22 21:09公开、9/24 03:04合入 `f4d9d2e`。新增是版本纳入/默认行为变化，机制PR早于上次运行。
- **PD变化**：Mooncake/NIXL 默认启用 deferred release；decode abort 后保留KV页与metadata slot，等所有已通知prefiller ranks的drain ACK或30秒超时，降低在途写污染下一请求的风险。Mori/fake仍立即释放；旧两帧ACK不计，新decode/旧prefill会走timeout。有限超时不等于任意网络故障下的安全证明。
- **RL变化**：tensor/distributed更新须置于begin/end session内：恢复可加载布局→多桶加载→一次kernel-layout finalize，枚举target/draft/all。本地执行的真实GPU验证仅1×B200 Marlin W4A16 MoE的A→B→A；Triton W4A16 MoE自动路径、MXFP4、compressed-tensors W4A16 linear及per-tensor FP8 linear未因此具备安全refit，DFlash/DSpark只有枚举测试。
- **判断/边界**：🔄演进 T2/T3/T5，承接 E-20260928-02；退出请求和传完bytes都不代表状态可复用。作者测试，非独立性能/生产证明。该release的SafeUnpickler改动也不能代替diffusion CVE闭环，见运行记录。
- **下一问/来源**：核混合版本ACK、超时回收与真实多卡反复refit。[release](https://github.com/sgl-project/sglang/releases/tag/v0.5.21) · [取消协议](https://github.com/sgl-project/sglang/pull/41023) · [更新协议与排除项](https://github.com/sgl-project/sglang/pull/40777)。

## E-20261012-05

**Dynamo：验证过的对象，必须就是随后实际使用的对象。**

- **日期/身份**：正式 `v1.5.1`，10/7 01:54:13，`2093662`；地址绑定 #14474 于9/8公开、9/19合入 `848206e`，较早机制随本次版本记录。
- **事实**：媒体下载的Python/Rust客户端连接已验证地址，保留原hostname的TLS校验，避免校验/连接二次解析的不一致。受目标策略保护的fetch默认拒绝代理；显式信任egress proxy是运维授权，不是自动安全。release另限制inline/remote媒体大小及本地目录，并为request-path overload hint设置1秒过期；monitor状态另算。
- **验证/限制**：PR作者报告AMD64/macOS ARM64及116 HTTP测试；非独立安全审计。依赖仍SGLang0.5.18/TRTLLM1.3rc25/vLLM0.28，不拼接新版功能。已列SGLang sidecar协议导致500、部分H.264/PyNvVideoCodec2.2.3导致worker崩溃；1.6修复目标不是已交付。
- **判断/下一问**：✅强化 T3/T5/T7，扩展 E-20260928-11 的信任边界：校验、连接、资源占用和过期状态必须一致。核重定向/代理策略、负载恢复和已知问题正式修复。
- **来源**：[v1.5.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.5.1) · [#14474](https://github.com/ai-dynamo/dynamo/pull/14474)。

## E-20261012-06

**MaxText：省内存的 overflow fallback，首先要保留未更新的训练状态。**

- **日期/身份**：#5321 于9/21 20:52公开、9/29 01:26合入 `0d5aac6`，随10/2 16:35:26正式 `maxtext-v0.2.5` / `493d592`发布；不是dropless概念首次出现。
- **变化**：step fallback 在图内遇overflow选择旧state（含RNG count）保留donation，再由host从返回的未更新state replay；overflow flag从逐层通信改全模型一次归约，避免长期保留额外训练状态。
- **边界**：Ring-of-Experts+ragged sort；默认None仍丢超限token。step不支持DiLoCo、预编译trainstep、optimizer/parameter host offload；layer局部分支不共享全部限制。Tunix RL扩展 #5570 截至10/12仍Open、目标分支atwigg/mlperf，依赖Tunix #2721，作者未跑真实RL，不能与预训练能力拼接。
- **作者验证**：TPU7x 4 chips/8 devices、deepseek3-test、EP4×FSDP2、synthetic30步强制overflow，与相同chunking dropless基线loss/grad norm逐步bit-identical。671B/tpu7x-1024仅AOT编译内存/HLO分析，不是实跑显存、吞吐或长期收敛。
- **判断/下一问**：✅强化 T1/T5/T8，并连接T2/T4。容量优化必须确认token与更新均未错误丢失；核真实数据、多步恢复和RL接入，而非只看能编译到多少设备。
- **来源**：[正式tag](https://github.com/AI-Hypercomputer/maxtext/releases/tag/maxtext-v0.2.5) · [#5321](https://github.com/AI-Hypercomputer/maxtext/pull/5321) · [固定配置边界](https://github.com/AI-Hypercomputer/maxtext/blob/maxtext-v0.2.5/docs/reference/core_concepts/moe_configuration.md) · [未合入RL扩展](https://github.com/AI-Hypercomputer/maxtext/pull/5570)。

## E-20261012-07

**KVTether：逻辑上已死亡、暂时不用和物理仍驻留的 KV，不能只靠 LRU 区分。**

- **日期/身份**：新论文 `2609.39819v1`，9/30 14:22:14 首次公开；正文 §5～8、表4。论文原型/作者实验，非 runtime 正式功能，未核固定实现 commit 或独立复测。
- **机制/变化**：harness 的消息替换、删除、fork 等映射到 KV 生命周期；先排除 temporary/expired，再决定 live-but-idle 的驻留。共享KV所有映射版本均为Discarded时，才按已死亡状态回收，并等待在途使用完成；live KV仍可按压力策略驱逐。承接 E-20260928-04 的物理完整/可达索引，增加上层逻辑存活条件。
- **实验边界**：3节点/12 GB200，GLM-4.7、SGLang0.5.8、1P2D/TP4，LMCache0.5.5/MoonCake0.3.8，每节点400GB host KV。三种轨迹各400任务、五档Poisson RPS；在真实硬件上用随机内容和固定长度回放，工具等待按记录模拟，不是在线任务质量实验。精度和可复现完整配置本轮未锁。
- **不采普适性能结论**：论文所谓 task pricing 是按外部 token 费率估算，并非这套集群 TCO。MRU 只是特定 suspended 重用分布的启发式，去掉生命周期跟踪后也可能不如 LRU。
- **判断/下一问**：🔄演进 T3/T7；“谁知道状态已经无用”成为跨层接口问题。下一步核事件丢失/乱序、多租户恶意 hint、真实内容与任务成功率，不把 trace 回放称生产部署。
- **来源**：[版本历史](https://arxiv.org/abs/2609.39819) · [机制与实验正文](https://arxiv.org/html/2609.39819v1)。

## E-20261012-08

**补录 SWE-Serve：局部测试通过，不能替代实际服务链路验收。**

- **日期**：论文 `2609.26777v1` 首公开9/22 17:54:59，官方报告9/23；9/28发现候选，10/12补核论文 §5.2/A.4.1 与公开仓库，不算本周新论文。
- **证据**：53个SGLang工程任务，其中19个有真实模型服务检查。相同627个补丁，完整 verifier 通过288个（45.9%），删除E2E检查后435个（69.4%）；无结果仍算失败。这是重评分，不是重新跑出另一组补丁。
- **敏感性**：其中4任务删除E2E后没有剩余测试；排除它们，剩15任务仍为191/495→304/495（38.6%→61.4%）。因此不能把总差直接解释成所有生产补丁的失败概率。
- **成熟度/边界**：公开 benchmark、作者测量；12 CPU/41单H100任务，closed-book mini-swe-agent，未涵盖多GPU/多节点或其他引擎。通过 verifier 不等于维护者认可、可合并或生产安全；本任务未执行测试。
- **判断/下一问**：✅强化 T5/T6，补足 E-20260921-08 的“源码—包—dispatch”链，向上延伸到请求—调度—cache—输出。下轮核升级回归与多节点失败场景，而非仅追加模型排行榜。
- **来源**：[论文日期](https://arxiv.org/abs/2609.26777) · [论文与敏感性](https://arxiv.org/html/2609.26777v1) · [官方报告](https://developer.nvidia.com/blog/how-swe-serve-exposes-the-gap-between-local-tests-and-live-serving/) · [公开制品](https://github.com/NVIDIA/swe-serve)。

## E-20261012-09

**GPUNetIO：共同底层减少重复实现，但库名不保证实际绕过 CPU。**

- **日期/身份**：10/6官方新报告；既有 GDA-KI、NCCL/NVSHMEM 集成机制为补录，未将报告日当首次实现。固定仓库 `b2b7a291ec70fdeeb1b27ba4b0ab8c8e260a729b`，10/5提交名为v5.0.0 rc1，不据此称新GA。
- **事实**：NCCL GIN 与 NVSHMEM GPUNetIO transport 共享底层 Verbs 实现。CPU 建对象/注册，GPU 发WQE/doorbell/查CQ；开源子集是 RDMA one-sided，不等于完整SDK的Ethernet/DMA/two-sided。配置SDK可经dlopen切换实际实现。
- **关键边界**：无直接GPU→NIC doorbell能力时使用CPU-assisted proxy；GPU-initiated不意味着所有机器都无CPU参与。低层API的并发同步仍由应用负责。只核文档和固定README，不采厂商带宽图为端到端优势。
- **判断/下一问**：✅强化 T4/T8；将“共同依赖、实际选路、progress责任”纳入兼容矩阵。后续核 transport/SDK/driver/NIC组合、争用与失败路径，而非按库名重复统计创新。
- **来源**：[10/6报告](https://developer.nvidia.com/blog/doca-gpunetio-gda-ki-unified-gpu-networking/) · [固定README](https://github.com/NVIDIA-DOCA/gpunetio/blob/b2b7a291ec70fdeeb1b27ba4b0ab8c8e260a729b/README.md)。后者网页cache miss，已用相同commit的raw正文核通。

## E-20261012-10

**FlashInfer：同一权重布局可以支持不同计算路径，但不等于默认动态精度。**

- **日期/身份**：#5735 于9/30 13:01公开、10/2 18:07合入 `0be3eb2`；正式v0.7.1于10/9 20:00:11发布，页面 `0b46115`。
- **机制**：SM120/121 CuTe DSL NVFP4 W4A16 routed-MoE直接读取cuTile W4A4同一prepared expert buffers：packed E2M1、128×4 swizzled E4M3 scales及per-expert FP32 scales；显式选择后可按token桶分派，不需第二份权重。未加默认backend list，默认dispatch不变。
- **作者口径**：DGX Spark/SM121，Qwen3.8-Flash-Next MoE层H2560/I640/E512/top10；完整层含routing/combine、CUDA Graph冷L2，不含preparation。1token对Marlin 1.36×，1024token反而落后；无整模型质量/请求SLO结论，未核下游vLLM接入已交付。
- **判断/下一问**：🔄演进 T4/T6/T8，承接 E-20260928-06 的部署匹配要求。物理布局的可共享性决定能否低成本选路；后续核真实token分布和质量—延迟收益。
- **来源**：[v0.7.1](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.1) · [#5735](https://github.com/flashinfer-ai/flashinfer/pull/5735)。

## E-20261012-11

**Kueue：旧的 placement 计算不能覆盖新的 admission 决定。**

- **日期/身份**：#15761 于9/17公开、10/7合入 `d1bd79b`；同日backport #16858 / `b13ee67`，正式v0.20.1于10/8 12:31发布，`237d77e`。
- **事实**：TAS第二次放置更新携带resourceVersion，拒绝旧pass覆盖新的admission/eviction；失败保留reservation并重读重试，但NotFound除外，回填不得覆盖缓存中较新更新或复活已删除对象。只改second-pass，不等于全部调度已事务化。
- **判断/边界**：✅强化 T5，承接 E-20260921-11 / E-20260928-05：不仅计数要对齐，还要确保释放/预约对应当前版本。维护者修复与测试，无独立集群利用率或性能测量。
- **后续/来源**：核并发驱逐/重试的reservation invariant与回归。[v0.20.1](https://github.com/kubernetes-sigs/kueue/releases/tag/v0.20.1) · [#15761](https://github.com/kubernetes-sigs/kueue/pull/15761) · [#16858](https://github.com/kubernetes-sigs/kueue/pull/16858)。

## E-20261012-12

**Dynamo V4.1：PD speculation 的支持扩大，但这是更换 backend 的新快照。**

- **日期/身份**：`v1.6.0-deepseek-v4.1-flash-dev.2`，10/8 23:42发布，`a201d5b`。
- **相对变化**：E-20260915-03 的dev.1是SGLang preview；dev.2改用patched vLLM0.31，aggregated/1P1D均提供DSpark-3，分别列B200/GB200/H200 recipes。不能把它解释成旧SGLang组合原地获得支持。
- **边界**：仍experimental、非QA、仅文本；NIXL protocol14不能与13混用。没有核完整benchmark制品，不转述性能，不能据此撤销FP4 KV/视觉未验证边界或称生产就绪。
- **判断/下一问**：🔄演进 T3/T4；旧“1P1D无speculation”只适用于旧快照，当前支持需同时绑定backend和协议版本。核正式QA与多组合回归。
- **来源**：[dev.2精确支持矩阵](https://github.com/ai-dynamo/dynamo/releases/tag/v1.6.0-deepseek-v4.1-flash-dev.2)。

## 2026-10-19 证据索引

以下卡片最近核验均为 **2026-10-19**。除特别注明外，首次发现也是10/19；Curator/NIXL/cuDNN版本及EfficientAgent线索在10/12已发现，本次补足机制。日期采用原始UTC；补录不是本周新发生。仅审计公开证据，未做本任务独立实验。

| 证据 | 变化 | 事件与状态 / 主线 |
| --- | --- | --- |
| [[#E-20261019-01]] | Midtraining以匹配后训练检验，而非只看基座proxy | 新论文、制品未齐 / T1、T2 |
| [[#E-20261019-02]] | 低精度去重会改变保留集合和代表关系 | 旧机制补录、已发布 / T1、T4、T6 |
| [[#E-20261019-03]] | Score centering依赖逐token采样支持receipt | 新合入main、未随稳定版 / T2、T4 |
| [[#E-20261019-04]] | P2P source、destination、peer注册须分别闭环 | 新合入main、0.32未包含 / T3、T4、T5 |
| [[#E-20261019-05]] | HiSparse后续实验收窄残余风险 | 10/5反证补录、未合入 / T3、T4 |
| [[#E-20261019-06]] | 扩容依据阶段工作量与部署标定 | 旧指南补录、experimental / T3、T5 |
| [[#E-20261019-07]] | Ascend sparse MQA复用的是metadata中的KV快照 | 新main实现 / T4、T6、T8 |
| [[#E-20261019-08]] | NIXL弹性连接准备与拓扑激活分开 | 旧版本机制补录 / T4、T5 |
| [[#E-20261019-09]] | QK norm属于数学合约，不只是kernel替换 | 旧版本机制补录 / T4、T6 |
| [[#E-20261019-10]] | Agent KV offload收益取决于重用工作集 | 9/27论文补录 / T3、T7、T8 |
| [[#E-20261019-11]] | 近似非前缀复用须测试持久缓存历史 | 10/5论文补录反证 / T3、T7 |
| [[#E-20261019-12]] | 流式Flow只提交完整因果chunk的KV | 新正式版纳入旧机制 / T3、T6、T7 |
| [[#E-20261019-13]] | DeepEP V2.5重新划分通信与框架责任 | 9/29实现补录、部分experimental / T4、T5、T8 |

## E-20261019-01

**Midtraining：recipe的终点应是后续可训练性，开放程度另验。**

- **日期/身份**：新论文 `2610.22019v1`，10/15 14:58:21公开；NVIDIA作者实验，未核版本化训练代码或独立复测。
- **机制**：Nemotron 3.5 Lightning（30B/3B active）与Super（120B/12B）在256K做软件工程数据midtraining；约10% agent traces、20%预训练replay，工具返回不计loss。基座proxy不足以可靠挑选recipe，改用固定短SFT检验后续能力。
- **结果边界**：GB300/BF16/Megatron，Lightning TP/CP/EP=2/4/8；500B额外midtraining后，两臂同100B-token full SFT，SWE Verified pass@1 60.7→64.5（OpenHands SDK1.17.0、500任务×3rollouts、temperature1/top-p0.95）。这是作者匹配后训练比较，不能省略500B前置成本称同成本获益。
- **反证/开放性**：Appendix A仍计划未来发布处理脚本和训练集metadata；D只有单seed/20次GRPO update，作者明确不足以测RL增益。E中Super在256K继续训练损害1M RULER，须额外长上下文阶段恢复，不能拼接所有阶段能力。
- **判断/下一问**：🧪探索 T1/T2；数据质量应在匹配SFT/harness下验收。等待manifest/脚本、完整成本、多seed长期RL，不因标题含Open就判可完整复现。
- **来源**：[日期](https://arxiv.org/abs/2610.22019) · [固定正文§3–5及附录](https://arxiv.org/html/2610.22019v1)。

## E-20261019-02

**Curator：低精度去重优化也会改变“训练哪些数据”。**

- **日期/身份**：#2388于9/10 17:35公开、21:40合入 `3fc330485c026a3e96bd0072fa001fe2a6e49477`；10/9 01:13发布v1.4.0 / `10fba1ea785863872ae6983b580ec6813e72cc79`。10/12已发现版本，本次为旧机制补录。
- **事实**：固定tag的pairwise compute和KMeans embedding storage默认float16，KMeans fitting/prediction/centroid-distance仍FP32；不能简称整个去重全FP16或FP32等价。
- **作者验证**：8H100、65,844,782文档、1000clusters，固定FP32 clusters/rankings逐ID比较、keep=hard、cosine阈值0.99：1437个保留/删除决策翻转（0.002182%）。双方均删除的ID，其nearest-neighbor parent一致率86.734%；当前输出只用removed IDs，代表关系不同不直接改变该输出，但会影响依赖映射的下游。
- **性能限制**：两规模workflow自报1.08×/1.71×来自不同EOS hosts且重跑KMeans，非受控同卡A/B；pairwise累计compute倍数不等于流水线倍数。无下游训练质量验证，近exact阈值需显式FP32。
- **判断/下一问**：✅强化 T1/T4/T6：验收保留集合与代表关系，不只验数值误差/吞吐；补同环境运行及最终质量，完整recipe缺口不因此关闭。
- **来源**：[release](https://github.com/NVIDIA-NeMo/Curator/releases/tag/v1.4.0) · [实验与限制](https://github.com/NVIDIA-NeMo/Curator/pull/2388) · [固定workflow](https://github.com/NVIDIA-NeMo/Curator/blob/v1.4.0/nemo_curator/stages/deduplication/semantic/workflow.py)。

## E-20261019-03

**verl：记录top-k，不等于完整知道实际sampler支持集。**

- **日期/身份**：#8196于10/11 03:30公开，10/18 04:36合入 `8ced5d2f4b148c2991217554e0c6f03028fa1470`；本次演进更新 E-20261012-02 的Open边界。最新正式版仍0.9.1，不能归入该版。
- **机制**：移除手动score_centering_top_p，由provider生成response_support_complete并运输为rollout_support_complete；仅有效逐token receipt选择support-normalized估计，记录预算不改变采样。完整性要穿过partial resume、AgentLoop/TQ和padding；k值或head概率质量接近1都不足以证明完整。
- **限制**：不完整/旧记录仍用top-k head＋trainer-tail近似，可能在未知真实支持集外分配tail mass，近零tail有epsilon误差，非精确恢复分布。#8199 Megatron与#8206优化仍Open，不拼接支持。
- **作者验证**：H100/Qwen3-0.6B provider定向GPU测试对应较早head；最终非Boolean receipt拒绝修订只跑本地19项schema tests/hooks，未重跑远端。较早feature head的Countdown/400step/单seed结果为SC held-out0.58对旧PG0.09，但15s/step对10s/step；RTX PRO6000、vLLM0.29、FP8 sampler/BF16 FSDP、top-p0.95/top-k128/记录128。不是最终merge公平性能或同质量成本证明。
- **判断/下一问**：🔄演进 T2/T4；概率修正依赖可审计采样metadata。核最终head完整回归、多seed和达到质量门槛的总耗时。
- **来源**：[#8196](https://github.com/verl-project/verl/pull/8196) · [合入](https://github.com/verl-project/verl/commit/8ced5d2f4b148c2991217554e0c6f03028fa1470) · [Megatron候选](https://github.com/verl-project/verl/pull/8199) · [优化候选](https://github.com/verl-project/verl/pull/8206)。

## E-20261019-04

**vLLM P2P：source pins、destination slots、peer注册是三个故障生命周期。**

- **日期/身份**：#59790于10/2公开、10/14 15:31合入 `e2aabc8b1521aad382b439e6be4e740d8e8211eb`；#60279于10/6公开、10/15 11:07合入 `8b27d274025e8dd1db3ca0405ac893492f8e7683`。新增main修复，承接10/12运行记录的Open候选。
- **修复**：producer保留source pins至WRITE达到terminal才发AbortAck/失败通知，不能假设release handle/remove peer返回就停止写入；descriptor释放失败仍尝试remove remote agent，避免陈旧注册阻止重连。
- **版本核验**：v0.32.0于10/18 06:50发布 / `5df63a7`，两个merge SHA与tag均diverged，固定tag nixl.py仍旧cancel/remove实现；**两项均未进入0.32.0**，发布日期更晚不表示包含main全部修复。
- **验证与剩余问题**：前项为NIXL1.4.1/UCX TCP/CPU DRAM probe；后项含作者mock及reviewer两节点NIXL1.5/IB rc_mlx5/DRAM WRITE checksum复查，非真实模型服务验证。producer可能无限持pins；10/14新issue #61532指出consumer abort-ack超时后释放slots，尚未复现E2E KV corruption。#55181/#60374仍Open，最新#60374验证未读完，不能沿用“仅mock”的旧判断。
- **判断/下一问**：✅强化 T3/T4/T5；某一侧修复不等于端到端故障闭环。核目标slot drain、有限资源回收、正式发布与真实故障回归。
- **来源**：[#59790](https://github.com/vllm-project/vllm/pull/59790) · [#60279](https://github.com/vllm-project/vllm/pull/60279) · [剩余#61532](https://github.com/vllm-project/vllm/issues/61532) · [#60374](https://github.com/vllm-project/vllm/pull/60374) · [正式版](https://github.com/vllm-project/vllm/releases/tag/v0.32.0) · [固定源码](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/v1/kv_offload/tiering/p2p/data/nixl.py)。

## E-20261019-05

**补录反证：HiSparse残余hazard，不等于已确认自然负载下错误输出。**

- **日期/身份**：后续#59633于10/1 15:57公开，10/5 21:17关闭且未合入；本次发现并读取最后Closing评论。修正 E-20261012-03 未纳入反证的当前风险强度，不改旧卡正文。
- **作者实验**：tiny DeepSeek-V3.2、MTP、P/D、1×H100、300requests/c16，自然时序NIXL post时没有pending deferred copy，copy约提前27ms；人为copy-stream延迟75ms后main有27次overwrite、提案0次。
- **边界收窄**：仅MTP draft-layer rows暴露；page spill持有host-block引用，不属原描述的同一风险。greedy输出与non-P/D reference一致，作者认为可能影响acceptance而非greedy token correctness；未证明所有sampling/模型安全。原“host近满才复用”亦被纠正：freed unhashed blocks进入free queue前部。
- **判断/下一问**：⚠️修正 T3/T4。应记录“注入延迟可触发、自然测试未复现、提案撤回未合入”，而非已确认生产corruption或已经修复。等自然复现、其他采样和真实负载反例。
- **来源**：[#59633及10/5 Closing评论](https://github.com/vllm-project/vllm/pull/59633)。

## E-20261019-06

**llm-d：扩容指标应接到阶段工作量和真实启动滞后。**

- **日期/身份**：v0.10.0于9/29 00:00发布；#2466于9/9公开、9/14合入。10/12已发现版本，本次补核固定指南；其中token-aware仍明确Experimental。
- **机制**：prefill以uncached-token backlog除本部署实测peakPrefillThroughput估算积压秒数；decode按pod KV occupancy。共置取两trigger需求较大值，PD分别扩角色；标定必须含实际P/D传输，不能借同卡他人的吞吐。单EPP副本限制来自process-local计数，0与query unknown须区分，Ready=True不证明伸缩真实生效。
- **验证/限制**：PR be80a09记录GPU-less Kind＋inference-sim＋KEDA2.20.2伸缩测试，不是GPU/SLO/成本证明。普通KEDA/EPP指南300秒双向stabilization防加载期间过度伸缩，也会延迟真正突发，需按cold-start重标定。
- **判断/下一问**：🧪探索 T3/T5；请求数不直接等于工作量，指标正确也不自动满足p99。核open-loop突发、metric生命周期和实际模型冷启动，不能把历史guide的未tag句当当前router状态。
- **来源**：[release](https://github.com/llm-d/llm-d/releases/tag/v0.10.0) · [固定token-aware指南](https://github.com/llm-d/llm-d/blob/v0.10.0/guides/workload-autoscaling/keda-epp-token-aware/README.md) · [普通指南](https://github.com/llm-d/llm-d/blob/v0.10.0/guides/workload-autoscaling/keda-epp/README.md) · [#2466](https://github.com/llm-d/llm-d/pull/2466)。

## E-20261019-07

**DeepGEMM-Ascend：metadata可能保存KV快照，而不只是索引。**

- **日期/身份**：项目9/30初始公开；#12于10/14公开并合入 `3a35326271b069b8b5bf1a754435acd6cc91575b`（09:22:30UTC提交）。PR正文空，事实来自patch和固定源码，不能只凭标题。
- **机制**：新增sparse/paged sparse MQA。paged FIRST将选定KV复制进metadata，REUSE读该快照而非当前raw KV；索引相同但KV内容变化也必须重建。阶段由host管理，没有device-visible phase。测试将raw KV清零、weights加倍仍读旧快照，重建才为零；这是已读测试设计，非本任务实跑。
- **组合**：固定README要求Ascend950/CANN9.20/torch_npu；Q/K FP8 E4M3或packed FP4、BF16 weights/output，head dim128、头数4～32且为4倍数、sparse block8/16、page64/128。有效block ID有序、同请求query连续和page mapping一致由caller负责，不全被runtime检查。UE8M0成对压int16/MN-major布局不同于CUDA。
- **判断/限制**：✅强化 T3/T4/T6/T8：prepared state身份需包含内容版本；不是整套attention/serving保证，既有峰值图不属于此次sparse新benchmark。后续核可变KV/错误metadata拒绝、完整模型接入和质量。
- **来源**：[#12](https://github.com/deepseek-ai/DeepGEMM-Ascend/pull/12) · [固定API](https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/3a35326271b069b8b5bf1a754435acd6cc91575b/csrc/apis/attention.hpp#L115) · [测试](https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/3a35326271b069b8b5bf1a754435acd6cc91575b/tests/test_attention.py#L948) · [组合](https://github.com/deepseek-ai/DeepGEMM-Ascend/blob/3a35326271b069b8b5bf1a754435acd6cc91575b/README.md)。

## E-20261019-08

**NIXL：准备新连接与让新拓扑生效，不是同一阶段。**

- **日期/身份**：#2138于8/22公开、8/25合入 `b0cbb237354d72b83500d5214d2b4484f9866fa3`；v1.5.0于9/30 07:31发布 / `1683cf3`。10/12发现版本，10/19补核旧机制，不声称首次随1.5发布。
- **机制/边界**：connect_ranks(activate=False)建立standby memory views，同时保留dispatch/combine的active views。激活先cudaDeviceSynchronize，再换views/context/P2P pointers，等待comm stream后释放旧views。固定Python合约限定低延迟模式且只保证与dispatch/combine并发；释放GIL不等于激活无停顿。
- **反证**：#2112小规模block-utilization优化被#2218于9/8回滚（9/6公开，`1e83d4a10e329cf3cdfc9c5e0821e4a9e932a739`）。维护者4GPU A/B、128tokens/top-k2/每rank2experts/hidden8192/30s timeout，在扩容与移除rank计划中出现receive timeout/assertion；未给硬件型号，不推性能。
- **判断/下一问**：✅强化 T4/T5：拓扑、在途工作和旧注册的切换需要屏障；不能推出任意弹性无停顿或长期goodput。fixed-capacity graph复用还可能保留不适合新拓扑的launch配置，需测真实变更回归。
- **来源**：[release](https://github.com/ai-dynamo/nixl/releases/tag/v1.5.0) · [#2138](https://github.com/ai-dynamo/nixl/pull/2138) · [固定合约](https://github.com/ai-dynamo/nixl/blob/v1.5.0/examples/device/ep/nixl_ep/buffer.py#L842) · [回滚反证](https://github.com/ai-dynamo/nixl/pull/2218)。

## E-20261019-09

**cuDNN：升级linear attention也要验数学合约，不能只对shape。**

- **日期/身份**：#1454于10/7 18:30公开、19:41合入 `db85ff42d6d8ce8ab05b1d94414620cfc451a0fa`；1.31于10/8发布，tag `51a3de7`，release明列shipped source `8ab41a3b8fd8f2eb0c6d4b0f4ff566c8cf1e4bed`。10/12已发现，本次补核。
- **事实**：FROST Q/K normalization由clamped norm改成 x×rsqrt(sum(x²)+1e-6)，小范数行数学结果会变。#1474说明SM90 CUDA KDA forward曾保留旧F.normalize，patch对齐additive epsilon；原SM90 CUDA backward拒绝use_qk_l2norm，不能因此说forward/backward全支持。
- **保留限制**：1.31 Known Issues仍列SM100/SM107 FROST forward负有效scale：masked negative attn_scale可NaN，per-tensor FP8负descale_q×descale_k可错误输出/nonfinite Stats。仅这些路径，不推广所有cuDNN plans；1.32 nightly不是已核修复证明。
- **判断/下一问**：⚠️修正 T4/T6/T8边界，承接E-20260928-07。核小范数、符号、实际backend及固定source一致性；无本次独立GPU或性能结果。
- **来源**：[1.31版本/限制/source](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.31.0) · [#1454](https://github.com/NVIDIA/cudnn-frontend/pull/1454) · [SM90差异](https://github.com/NVIDIA/cudnn-frontend/pull/1474) · [固定kernel](https://github.com/NVIDIA/cudnn-frontend/blob/8ab41a3b8fd8f2eb0c6d4b0f4ff566c8cf1e4bed/python/cudnn/linear_attention/frost/kernel/kda_prefill_f16.py#L121)。

## E-20261019-10

**EfficientAgent：单次搬运便宜，还不够证明KV offload划算。**

- **日期/身份**：`2609.33762v1`于9/27 17:01公开，10/12只有线索，本次补录正文及公开实现README；论文原型、作者实验，未锁实现commit或独立复跑。
- **机制**：收益还要求KV活到下次访问；并发agent全部上下文构成reuse working set。用stack-distance估容量，压力下保留小增量、拒绝大refill；容量足够则全写。host命中若只是替换GPU命中，并未减少计算。
- **口径**：8H20/TP8、BF16 Qwen3-Coder-30B-A3B、vLLM0.13/LMCache0.3.12、16active agents、固定343K GPU KV、1024token host chunks；4427调用/147.1M prompt的依赖保持回放，强制输出原token并保留工具等待，grading另算。host每rank5→20GiB使makespan208.91→128.05min，不能称同内存算法增益。
- **反例/限制**：40GiB时始终过滤写反而比全写慢；H800测试offload可变慢。模型/卡数/TP不同，非平台成本排名；固定轨迹性能不证明自由生成任务质量等价，无SLO/TCO结论。
- **判断/下一问**：✅强化 T3/T7/T8；串起并发、容量、重用距离与搬运收益。核动态任务池与真实质量，并区分总prompt和必须重算的cache-stable前缀之外token。
- **来源**：[日期](https://arxiv.org/abs/2609.33762) · [正文§3–5及附录A/B/E/G](https://arxiv.org/html/2609.33762v1) · [实现与依赖](https://github.com/KunmingSHAO/efficientagent_release)。

## E-20261019-11

**补录反证：近似非前缀复用的输入，还包括持久缓存的历史。**

- **日期/身份**：`2610.05833v1`，10/5 05:37公开，本次发现/补录。论文作者实验，未核公开实现固定commit和独立复测。
- **机制/实验**：Qwen3-8B、greedy/固定seed、LMCache0.5.1/vLLM0.23，关闭vLLM exact-prefix cache；7735个84文档滚动窗口、约7.2K～7.7Ktoken，确定字段抽取。相同prompt按五种请求次序逐个执行，组间清空cache、组内持续保存，比较5% token top-k与连续span修复。
- **结果**：跨次序答案差异率69.0%对26.1%，是成对答案不一致率，不是任务失败率。连续span仍非无损：逆序fidelity低至54.3%。清空每次cache的500请求对照无显著差别；消融支持连续性，未证明文档边界或偏差排序有额外显著收益。L4 median TTFT约5.7×仅该抽取负载，非系统吞吐/真实agent质量。
- **判断/边界**：⚠️修正 T3/T7的验收维度：近似cache需跨请求历史/重试/交错测试；不能泛化为精确prefix caching也会因该机制出错。连接E-20261012-07的生命周期，但两论文不是已验证组合。
- **下一问/来源**：固定实现、更多模型和非抽取任务、SLO及独立复测。[日期](https://arxiv.org/abs/2610.05833) · [正文§4与消融](https://arxiv.org/html/2610.05833v1)。

## E-20261019-12

**Omni流式Flow：已分配帧不等于已经可以跨hop复用的帧。**

- **日期/身份**：#2406于9/28公开、10/1合入 `257826d`；正式v0.1.8于10/12 18:22发布 / `870d774`，晚于上轮访问。新增是版本纳入，机制较早。
- **机制**：Fun-CosyVoice3对每个Euler step/DiT layer分别保存因果prefix KV及conv尾部，跨音频hop复用，不是在不同去噪步间近似共用。review发现chunk内帧还可看后续帧，非默认hop计划会误缓存未完成chunk；最终只提交完整50-frame chunks，余下重算，并区分committed/allocated frames。池不足回退whole-history路径，随stream释放。
- **作者验证/成本**：reviewer用真实权重和不同row/hop/reorder做bitwise对照；H10080GB/SeedTTS EN1088/c16，WER排除>50%样本后均1.18%、两臂均4个异常，speaker similarity70.30/70.42；性能为不同卡同时跑，非独立同卡受控结论。原H200自报还明确额外pool、约100秒compile warmup和首hop更慢，不以某hop节省推整服务收益。
- **判断/下一问**：✅强化 T3/T6/T7：精确cache先确定因果提交边界，吞吐收益须计容量与启动。核非默认chunk、取消/池压力回归及长音频质量；该CUDA路径不证明XPU/MLX等价。
- **来源**：[v0.1.8](https://github.com/sgl-project/sglang-omni/releases/tag/v0.1.8) · [#2406及review修正](https://github.com/sgl-project/sglang-omni/pull/2406)。

## E-20261019-13

**DeepEP V2.5：通信库提供原语，框架仍拥有路由与完成责任。**

- **日期/身份**：#763于9/23公开、9/29 05:22合入 `def865146e95aa9ea223e277c0a8a4c6b81e5ac7`；10/12只见入口，本次为锁定实现的补录。10/15去掉格式检查不是技术新增。
- **事实/组合**：V2.5拆分EPBuffer与Bucket/PP/Engram（后三者experimental），移除V1/NVSHMEM；固定文档要求NCCL2.32.3+、CUDA13.1+、Hopper+。不能沿用旧实现的依赖或按库名称零SM。
- **责任边界**：冗余expert提供NVLink weight/scale prefetch与FP32 grad reduction原语；副本规划、token rerouting和expert GEMM由框架实现。域内所有ranks参与，mapping/storage须存活到完成；复用前须恢复backward weights并完成gradient reduction。deferred epilogue的wait会执行epilogue并首次产生结果，不是纯ready查询。
- **判断/下一问**：🔄演进 T4/T5/T8；异步API要标谁触发工作、谁拥有buffer、谁提交结果。文档/源码核验，无整训练收益或生产可靠性证明；核框架集成、故障与跨版迁移。
- **来源**：[#763](https://github.com/deepseek-ai/DeepEP/pull/763) · [固定README](https://github.com/deepseek-ai/DeepEP/blob/def865146e95aa9ea223e277c0a8a4c6b81e5ac7/README.md)。

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
