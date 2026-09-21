---
type: research-protocol
status: active
protocol_version: 2
updated: 2026-09-21
timezone: Asia/Shanghai
automation_id: mlsys-follow-up
---

# MLSys：追踪规则与来源

> 阅读入口：[[HPCGame 2026 - 06 前沿追踪|当前判断]]。这页供执行任务时使用；日常阅读从当前判断、主题文章和本月周报进入。

## 1. 工作目标与阅读层次

持续回答：在给定质量、延迟、资源与可靠性要求下，哪些机制改变了完成训练或用户任务的时间、成本及可行规模？

| 层次 | 文件 | 更新方式 |
| --- | --- | --- |
| 全景关系 | `HPCGame 2026 - 00 MLSys 知识地图.md` | 月度整合或重大因果关系改变时更新 |
| 当前判断 | `HPCGame 2026 - 06 前沿追踪.md` | 保留 8 条主线、成立边界、反证与最新变化入口；目标约两屏至一篇短文 |
| 机制解释 | 同目录 `01`～`05`、`07`～`09` 主题文章 | 补充可复用解释与交叉链接；不追加新闻列表 |
| 可审计事实 | `追踪记录/MLSys - 证据台账.md` | 新事实新增证据 ID；纠错追加勘误，保留原记录 |
| 周度变化 | `追踪记录/MLSys - YYYY-MM 周度更新.md` | 按实际运行日追加小节，先关系变化后事实卡链接 |
| 月度重估 | `追踪记录/MLSys - 月度复盘.md` | 每个新自然月首次成功周任务复盘一次 |
| 覆盖与断点 | `追踪记录/MLSys - 运行记录.md` | 每次执行都记录已查来源、遗漏、截止时间、失败与下次待办 |
| 历史原文 | `追踪记录/MLSys - 2026-08 至 09 旧版基线与日志.md` | 冻结原文；使用前查看证据台账的勘误 |

文件名固定部分与本表一致。`YYYY-MM` 用 Asia/Shanghai 下的实际运行月份，例如 `MLSys - 2026-09 周度更新.md`。每月一个周报文件，首页不积累历史全文。

## 2. 八条问题主线

主线 ID 是持久标识；项目是观察入口，可以迁移、增减和交叉引用。同一事件只建一个证据卡，关联多个主线。

| ID | 主线与核心问题 | 概念入口 | 调查优先级 |
| --- | --- | --- | --- |
| T1 | 数据、recipe 与训练：达到同等质量需要做多少工作，如何高效执行？ | 01、02、07 | 每周；补数据管线、optimizer、activation/recompute 与完整复现条件 |
| T2 | 后训练与推理时预算：生成、环境、reward、更新如何形成有效反馈？ | 07、03 | 最高；重点异步、样本陈旧、数值一致性、质量—预算曲线 |
| T3 | 推理状态与服务调度：多阶段请求怎样在 SLO 内完成？ | 03、08、09 | 每周；跟踪完整 session、cache、speculation、PD/EPD 组合 |
| T4 | 模型结构、通信与内存：计算图改变后，状态和数据移动怎样改变？ | 02、04 | 每周扫描；轮换深入 MoE/attention/低精度/状态层级 |
| T5 | 集群效率与可靠性：排队、放置、故障、隔离损失了多少有效工作？ | 08、01、04 | 最高；将通信微基准接回长期训练和生产服务 |
| T6 | 编译、kernel 与自动优化：局部优化怎样转化为可维护的整图收益？ | 05 | 每周扫描；编译启动、动态 shape、正确性、AI 自动优化 |
| T7 | 多模态与新工作负载：新的时间结构和质量指标是否改变系统假设？ | 09、03、07 | 每周扫描；轮换深入 diffusion、实时音频、agent/environment |
| T8 | 硬件与可移植性：同一目标换平台后，性能、功能和成本如何变化？ | 05、04、08 | 每周扫描；轮换深入 NVIDIA/AMD/TPU/端侧及互连生态 |

每周覆盖所有主线的来源变化；深挖优先 T2、T5 和高影响事件。其他主线的专题深读按最久未深读优先，单条主线不应连续四次运行都只有标题扫描。核查不完整时保留缺口，不能以主线名称打勾代替实际来源检查。

## 3. 原有范围的归属

| 原有方向 | 新主线 |
| --- | --- |
| 开放训练数据、recipe、benchmark 污染 | T1；评测要求横跨全部主线 |
| FSDP/ZeRO/Megatron，TP/SP/PP/CP/EP | T1、T4；实际集群放置与恢复关联 T5 |
| MoE、DeepEP、DeepGEMM、UCCL | T4；训练/serving/编译影响关联 T1/T3/T6 |
| 精确、量化、稀疏、线性、hybrid attention | T4；异构 state 与质量关联 T1/T3/T7 |
| FP8/FP4、量化与低精度 recipe | T1/T4/T6/T8；必须说明 tensor、格式与质量条件 |
| vLLM/SGLang/KTransformers | T3；异构 offload 关联 T4/T8 |
| Speculative decoding、MTP、draft training | T3/T2；不混同所有 test-time compute 算法 |
| PD/EPD、KV pooling/tiering、multimodal/agent serving | T3/T4/T7；隔离、重试、autoscaling 关联 T5 |
| NCCL/NVSHMEM/NIXL/GIN/IBGDA | T4；生命周期与失败语义关联 T5 |
| UALink/NVLink/scale-up、scale-out 网络 | T4/T8；拥塞、拓扑碎片、放置关联 T5 |
| CUDA Tile/CuTe/Triton/TileLang、AI kernel generation | T6；跨平台收益关联 T8 |
| Rubin 及后续 GPU、C2C/DPU/storage/Engram/CXL | T8/T4；总成本与系统有效效率关联 T5 |

## 4. 来源登记

以下是发现入口，不是既定结论。先读 release/tag/报告正文，再按需要进入 PR、issue、版本化文档；`main`、`latest` 和 README 是可变页面，保存核验日与对应 commit/版本（能取得时）。链接失效时可寻找官方迁移地址，记录替换关系。

| 主线 | 每周优先检查入口 | 轮换、事件触发与反证来源 |
| --- | --- | --- |
| T1 | [Megatron Core](https://github.com/NVIDIA/Megatron-LM/releases)、[PyTorch](https://github.com/pytorch/pytorch/releases)、[TorchTitan](https://github.com/pytorch/torchtitan) | [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)、[NeMo Curator](https://github.com/NVIDIA-NeMo/Curator)、[Transformer Engine](https://github.com/NVIDIA/TransformerEngine/releases)；模型作者数据/recipe/训练报告、收敛与复现研究 |
| T2 | [verl](https://github.com/verl-project/verl)、[AReaL](https://github.com/areal-project/AReaL)、[slime](https://github.com/THUDM/slime) | 作者 RL 系统论文、环境与 reward 工具、推理预算/验证/蒸馏研究；[verl async 机制](https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md)是历史基线入口，0.9.1 已弃用旧 experimental trainer，后续核新 V1 文档，不以页面更新时间冒充机制首次出现 |
| T3 | [vLLM](https://github.com/vllm-project/vllm/releases)、[SGLang](https://github.com/sgl-project/sglang/releases)、[Dynamo](https://github.com/ai-dynamo/dynamo/releases) | [KTransformers](https://github.com/kvcache-ai/ktransformers)、[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/releases)、[LMCache](https://github.com/LMCache/LMCache)、[Mooncake](https://github.com/kvcache-ai/Mooncake)；旧 issue、复现脚本与兼容矩阵 |
| T4 | [DeepEP](https://github.com/deepseek-ai/DeepEP)、[NIXL](https://github.com/ai-dynamo/nixl/releases)、[NCCL](https://github.com/NVIDIA/nccl/releases) | [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[UCCL](https://github.com/uccl-project/uccl)、[NVSHMEM](https://docs.nvidia.com/nvshmem/release-notes-install-guide/index.html)、[FlashAttention](https://github.com/Dao-AILab/flash-attention)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer)；模型作者 attention/MoE/conditional memory 技术报告 |
| T5 | [Resiliency Extension](https://nvidia.github.io/nvidia-resiliency-ext/)、[llm-d](https://github.com/llm-d/llm-d)、[Kueue](https://github.com/kubernetes-sigs/kueue/releases) | [TorchFT](https://github.com/pytorch/torchft)、[vLLM security](https://docs.vllm.ai/en/latest/usage/security/)、维护者故障分析、生产 trace、checkpoint/straggler/failure-injection 论文 |
| T6 | [Triton](https://github.com/triton-lang/triton/releases)、[CUTLASS](https://github.com/NVIDIA/cutlass/releases)、[cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend/releases) | [torch.compile](https://docs.pytorch.org/docs/stable/torch.compiler.html)、[TileLang](https://github.com/tile-ai/tilelang)、[CUDA Tile](https://github.com/NVIDIA/cutile-python)、[Helion](https://github.com/pytorch/helion)；AI kernel/编译 agent 原论文、代码和独立执行评测 |
| T7 | [SGLang-Omni](https://github.com/sgl-project/sglang-omni)、[Diffusers](https://github.com/huggingface/diffusers/releases)、[SGLang](https://github.com/sgl-project/sglang) | 作者 diffusion/video/audio/diffusion-LM 技术报告；多阶段 streaming、跨步 cache、实时任务和环境执行的系统论文 |
| T8 | [NVIDIA 技术博客](https://developer.nvidia.com/blog/)、[ROCm TheRock](https://github.com/ROCm/TheRock/releases)、[AITER](https://github.com/ROCm/aiter) | 旧 ROCm 地址转至 [legacy-rocm-build](https://github.com/ROCm/legacy-rocm-build/releases)，不能仅盯旧仓库；[MaxText](https://github.com/AI-Hypercomputer/maxtext)、[SGL-JAX](https://github.com/sgl-project/sglang-jax)、[UALink](https://ualinkconsortium.org/)、[MLX](https://github.com/ml-explore/mlx)；TPU/NPU、CXL/C2C/DPU/HBM/供电散热资料须区分 spec、可获得硬件和部署 |

横切评测入口：[MLPerf Endpoints](https://mlcommons.org/benchmarks/endpoints/)、[MLPerf Inference](https://github.com/mlcommons/inference)。作者或厂商自测、同行审查提交、独立复测分别标注；同行审查不自动等于独立执行。

论文发现面应同时覆盖 MLSys、ML 编译、分布式/操作系统/网络方向及作者实验室新发布。搜索引擎、新闻和社交信息可以发现线索；最终事实必须落到原论文、官方实现或可审计的一手实验。每周至少做一次跨清单的问题检索，例如“达到同等 reward 的异步训练成本”或“多租户远程 KV 的失败恢复”，防止固定项目清单形成盲区。

## 5. 日期、版本与增量判定

每个事件至少分清四个时间：机制/结果首次公开日、所讨论版本/修订发布日、首次发现日、最近核验日。日期取来源正文、release metadata、arXiv version history 或 PR 时间；搜索页的 crawled/published 提示不能替代原始日期。

1. 先读运行记录中的逐来源/逐主线覆盖与缺口。旧版 `last_checked: 2026-09-14` 仅是历史报告声称的截止日，不能当作新体系全部已核。
2. 对成功检查过的来源，从其上次截止日前回看 14 天以覆盖晚索引和修订；按事件 ID、版本/PR 和核心机制去重。回看范围不决定某事实是否“新增”。
3. 初次纳入的主线先建立机制基线，并补查近 90 天的重要发布。旧机制新发现标 `补录`；未做完写入待办，下次继续。
4. 旧论文新增开源实现、独立复现、版本升级、修复或撤回，是新的证据事件；保留原论文日期和这次新事件日期。
5. 只有“新的机制/架构、重要实现或兼容变化、口径清楚的新实验、重大正确性/可靠性问题、原判断发生实质变化”进入周报。纯改名、无关小修、重复 day-0 宣传不计。
6. 无法核实的候选只进入运行记录待查队列，不能进入已确认事实或推动当前判断。404/访问失败不等于项目不存在，也不等于无更新。
7. 每条主线记录实际访问过的来源、实际覆盖截止和未查部分。只查部分来源标 `部分核查`；仅完成入口检查标 `扫描`；有针对性的机制/实验验证标 `深读`。这些状态不宣称穷尽领域。

## 6. 证据门槛

证据与判断分开记录：

- **研究判断**：✅强化、🔄演进、⚠️修正、🧪探索；它们说明对当前认识的影响，不代表软件成熟度。
- **实现状态**：spec/roadmap、论文原型、experimental/preview、已合入、已随某版本发布、默认启用；生产部署只能在有明确部署证据与条件时单独注明。
- **验证主体**：作者/厂商自报、维护者确认、独立复测、用户 issue（未确认/已确认/已修复/已回归）。
- **组合边界**：模型 × dtype/精度 × attention/cache 类型 × 并行/分离模式 × hardware × backend/driver。至少明确本条相关组合，未知写未知。
- **指标定义**：分数不自动等于成功率，acceptance length 不等于加速比，kernel speedup 不等于训练或请求完成速度。

性能事件必须尽量保存：对照版本、模型与质量/收敛门槛、硬件/卡数/互连、输入输出长度与到达分布、并发、batch/cache 命中、精度、SLO、p50/p95/p99、是否含启动/编译/重试/验证和权重同步、成本或能耗口径。RL 的训练 reward 与独立评测质量/任务成功率分别记录，不能只凭 reward 上升声称最终质量提高。关键维度缺失则降级为有条件的自报，不推出普适优势。

“首次”“已解决”“全面支持”“生产成熟”需额外证据；无法证明时用范围明确的描述。模型能力、runtime 能力、部署 recipe 能力分别确认，不能把三个来源的特性拼成一个已验证部署。

## 7. 周任务流程

1. 读取本规则、06 当前判断、运行记录、证据台账索引/勘误、当月周报和最近月度复盘。需要某概念时按 00 地图进入文章。
2. 确定实际日期、来源截止、待查队列与本月复盘状态；为本次建立运行条目。若同日有已完成条目，继续未完步骤或追加明确的二次检查，不重复写同一事件。
3. 按 T1～T8 批量检查一手入口，同时查未闭环的旧 issue、回归/撤回与跨清单问题。记录覆盖和失败。优先完成首次纳入的基线缺口。
4. 对候选核验正文和版本归属，比较原判断与既有证据。不能用上周生成的摘要代替原始来源。
5. 新增证据卡，ID 用 `E-YYYYMMDD-NN`。每卡以 `## ID` 为标题，方便跨页稳定链接。已有卡有错误，追加新勘误卡并在当前判断/周报引用它。
6. 按需追加当月周报的运行日小节：先 1～3 条“关系变化”，再最多 3～8 个最重要事件；不足 3 条不凑数。其余有效事实可留台账。补录、勘误与真正新增显式区分。新月首次有内容时创建对应月文件，并同步切换 00、06 的“本月变化”导航；先确认目标存在再改链接。新月尚无内容时保留上一期入口并标实际月份，不创建空周报。
7. 仅依据已核证据修改 06 的对应主线判断/边界/开放问题，保留反证；重大可复用解释可同步主题文章，修改处标核验日、来源和证据 ID，不堆周报。
8. 若到月度复盘时间，执行下一节。最后记录运行完成度和逐来源截止；完成核查的来源才推进日期，失败项保留原截止。
9. 检查新建文件、wiki 链接/标题、重复事件与证据日期一致性；失败时修正写入或保留明确未完成状态。

有实质变化：更新证据、周报与相关当前判断。没有实质变化：只维护运行覆盖记录，内容页不产生空更新。存在访问失败：明确“已核范围内未发现更新，另有待查”，不能报成全局无更新。

## 8. 月度复盘与范围调整

同一个每周任务承担月度复盘，不另建重复定时任务。实际运行月份不同于运行记录的 `last_monthly_review` 时触发；跳过的月份按真实日期补做并注明覆盖区间。2026-09-15 的初始化只建立起点，2026-10 首次周任务开始首轮周期复盘。

每次复盘覆盖从上次复盘到本次已核证据，回答：

1. 哪三条当前判断变强、变弱或改变适用范围？每条同时给支持与限制。
2. 哪条跨主题联系改变了？更新 00 地图或主题间链接，例如 RL 异步程度 ↔ 样本质量 ↔ 权重传输。
3. 有没有持续四周只见宣传、缺少代码/实验的热点？降低深读频率并说明恢复触发条件，保留入口扫描。
4. 哪个未在清单中的工作负载/机制值得纳入？至少给一个已核原始来源、与现有主线的关系和验收问题。
5. 哪些旧问题已经关闭、修复未发布、回滚或仍无法复现？逐个更新当前风险表。
6. 本轮还有哪些未核查的范围？月度复盘完成不等于把缺口清零。

输出追加到 `MLSys - 月度复盘.md`；凝练后的当前认识回写 06，必要时更新 00 和主题解释。仅在周期复盘实际完成后更新 `last_monthly_review`。

可在 T1～T8 既定研究范围内更新来源、优先级和子问题，并在月度记录理由；显著扩大到无关应用/行业，或引入付费服务、大规模下载/实验时需另行取得用户方向。常规任务只做公开资料研究与本地笔记维护。

## 9. 模板

### 证据卡

```text
## E-YYYYMMDD-NN
标题 / 主线 / 关联判断：
事件类型：新增 / 演进 / 补录 / 勘误 / 反证
首次公开日 / 本次事件日 / 首次发现日 / 最近核验日：
版本、tag/commit/PR、精确来源位置与链接：
新事实及相对已有证据的变化：
实现状态 / 验证主体：
适用组合、未覆盖组合及 benchmark 口径：
判断影响与不能推出的结论：
修正或承接哪些旧证据：
下一次应观察什么：
```

### 周报

```text
### YYYY-MM-DD
检查区间与覆盖：链接本次运行记录；列部分核查/失败。
关系变化：1～3 条，链接受影响的主题与证据。
重要事件：方向｜新事实｜相对上次变化｜判断｜为什么重要｜证据ID及一手链接。
补录与勘误：明确旧日期；不计当周新发现的技术进展。
当前判断影响：哪几条变了，为什么；下一项待验证条件。
```

## 10. 运行反馈

反馈优先说明改变认识的 1～3 个关系，再给最重要的来源与笔记入口，说明文件是否更新和覆盖缺口。不要为了格式凑出 3 条新闻。周期复盘有可行动的范围调整/认识变化时才作为重要更新呈现。

沿用任务所在对话的 heartbeat 通知规则；无实质变化或例行扫描按安静状态处理，勘误、重要新证据、完成重构或需要用户解决的执行失败应明确报告。通知偏好由自动化设置控制，不写入任务 prompt。
