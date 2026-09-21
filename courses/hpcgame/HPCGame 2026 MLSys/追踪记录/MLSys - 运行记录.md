---
type: research-run-state
protocol_version: 2
initialized: 2026-09-15
last_attempt: 2026-09-21
last_completed_weekly: null
last_successful_partial: 2026-09-21
last_run_status: partial-source-coverage
last_monthly_review: "2026-09"
last_monthly_review_kind: bootstrap
timezone: Asia/Shanghai
---

# MLSys：运行记录与核查覆盖

[[HPCGame 2026 - 06 前沿追踪|当前判断]] · [[MLSys - 追踪规则与来源|执行规则]] · [[MLSys - 证据台账|事实台账]]

## 1. 怎样理解这里的日期

- `last_attempt` 是最近执行日；`last_completed_weekly` 仅在当次计划的每周检查全部完成后推进，部分运行保留缺口。
- 每条来源的实际核查截止优先于全局日期。未完成项不会因其他来源成功而被跳过。
- `last_monthly_review` 用运行月份判断是否需要本月复盘；2026-09 是本次重构建立起点，不代表已经覆盖完整 9 月。
- 旧 tracker 的 2026-09-14 是历史报告截止，迁移时不冒充新体系完整核查。2026-09-15 是局部一手审计与基线初始化，没有开展全领域的新增检索。
- 下表“局部核验”表示核验了指定机制/事件；只有对来源时间窗口实际检查后，才填写该来源覆盖到哪一天。

## 2. 主线覆盖与断点

| 主线 | 最近局部核验 | 已查内容 | 完整来源窗口状态 / 下次补齐 |
| --- | --- | --- | --- |
| T1 | 2026-09-21 | Megatron 0.19.2 深读；PyTorch/Curator/TorchTitan 定向扫描 | 部分；数据/recipe、ZeRO/并行和 90 天基线未完 |
| T2 | 2026-09-21 | verl 0.9.1/相关 PR 深读；AReaL/slime/BRACE 原始来源 | 部分；main 缓存/429、近 90 天与独立质量成本仍有缺口 |
| T3 | 2026-09-21 | vLLM/SGLang/Dynamo releases；三旧 issue fresh API；FA4 组合 | 部分；其余 runtime/cache 项目与全部 main 未覆盖 |
| T4 | 2026-09-21 | NCCL 新版、NIXL 旧版复核；DeepEP/UCCL/DeepGEMM/NVSHMEM 扫描 | 部分；README 日期/commit 与旧性能结论未全面回溯 |
| T5 | 2026-09-21 | NVRx/Kueue 深读；llm-d/TorchFT 基线；Shadow 恢复边界 | 部分；长期有效 GPU 时间、全故障域和 90 天回填未完 |
| T6 | 2026-09-21 | cuTile Rust/FA4 深读；Triton/CUTLASS/cuDNN/TileLang/Helion 扫描 | 部分；TileGym commit 失败、未穷举全部代码变化 |
| T7 | 2026-09-21 | Omni 0.1.6 深读；Diffusers/H3/AuK 基线候选 | 部分；全部音视频/生成质量实验未查完 |
| T8 | 2026-09-21 | NCCL Rubin 边界、NVIDIA/ROCm TheRock/AITER/MLX/UALink 等入口 | 部分；ROCm 正文失败，平台等价实验和 C2C/DPU/storage 未完 |

## 3. 高优先级待办

1. **历史准确性补齐**：按 [[MLSys - 2026-08 至 09 旧版基线与日志]] 回访 8/24、8/31、9/7 的精确版本/日期。已完成的 9/14 六项不重复建卡；发现旧事件补录不计当周新增。
2. **新主线基线回填**：优先 T2、T5，再 T1/T7/T8；补查近 90 天版本、作者论文与原始实验，保留不能证实的部分。
3. **已报告故障闭环**：9/21 已用 fresh API 回访下列三项，均仍 Open；未核到修复发布。后续继续查维护者确认、关联 PR、修复版本、回滚与复现条件，不把状态核验当作故障修复。
   - [SGLang #36452](https://github.com/sgl-project/sglang/issues/36452)：启动临时显存与 KV 定容。
   - [vLLM #55000](https://github.com/vllm-project/vllm/issues/55000)：P2P secondary agent 启动挂起。
   - [vLLM #55181](https://github.com/vllm-project/vllm/issues/55181)：remote disconnect 与 recompute fallback。
4. **部署组合后续**：Dynamo V4.1 snapshot 之后是否发布 QA-gated 版本，是否补 FP4 KV/视觉/分离式 speculation。基准与日期见 E-20260915-03。
5. **轮换深读**：尚未完成深读的其他主线先于重复深挖热门项目；每周至少一个跨清单的问题检索。

## 4. 来源检查记录模板

每次运行只为实际访问来源记一行；可在同一行合并同项目的 release 与明确 PR，但不要把未访问的入口写入“已查”。初次检查与部分失败沿用这里的模板，完成后保留历史供审计。

| 来源/版本或查询 | 主线 | 本次实际查看的窗口/内容 | 结果及覆盖截止 | 未完成 / 下次动作 |
| --- | --- | --- | --- | --- |
| 待下一次每周运行填入 | T1～T8 | 按实际查询，不预填全局截止 | 扫描 / 深读 / 部分核查 / 失败 | 日期只对成功覆盖部分推进 |

## 5. 执行日志

### 2026-09-15

类型：用户授权的结构重建、机制基线初始化、历史证据抽样审计。完成范围不等同于一次完整周度扫描。

已完成的内容核验：

- 原 tracker 完整正文迁入历史归档，保留基线、三期日志与旧维护规则作为历史文本；当前流程由 v2 规则取代。
- 对 9/14 六项建立 E-20260915-01～06：Transformers 日期勘误、agent 优先性勘误、DeepSeek/Dynamo 组合边界、vLLM/cuDNN 核验、Φ-Bench 指标定义。
- 新增 07/08/09 三篇机制文章，补强 05 的整图编译与可移植性；各篇来源锚点已核验，但未以此声称已扫描相应领域全部新进展。
- 用八条问题主线组织 06；新增周报、月度复盘、来源规则与本运行覆盖表。

原始来源与精确审计日期见 [[MLSys - 证据台账]]；新机制来源见 07/08/09 及 05 新增章节。已核事实首次发生日与本次发现日分开保留。

下一次每周运行：先处理上述基线缺口和旧 issue，再按来源窗口继续增量核查。2026-10 首次成功周任务执行首轮周期复盘。

结构/链接与自动化验证：

- 15 份活跃研究文档中的 216 个内部文章/标题链接、证据 ID 唯一性、表格 alias 转义和代码围栏已通过静态检查；历史归档原文已与迁移前文本逐字核对。
- 独立结构复核确认 T1～T8 覆盖原范围、初始化与完整扫描状态区分一致、月度触发与跨月导航规则一致。
- 原自动化 `mlsys-follow-up` 已更新为“每周 MLSys 研究追踪与月度复盘”，保存后的 prompt 与 v2 流程一致；仍为原对话的 ACTIVE heartbeat，保留每周一 09:00 的日程（本地时区 Asia/Shanghai）。
- 本次完成的是研究组织、内容补强与自动化配置验收；未来无人值守执行是否成功仍需由后续真实运行确认，不将配置核对当作已成功执行了一次完整周任务。

### 2026-09-21

类型：每周增量研究，八主线均有入口扫描；T2/T5 优先深读，T6/T7 与精确部署组合有事件触发深读。已完成本次已核材料入库，**全来源、全部 main 提交及近 90 天回填仍未完成**，因此不推进 `last_completed_weekly`。这不是“全局无更新”，也不是全部历史已核。

计划回看窗口为 2026-09-01 至本次访问时，另处理 8/25 等基线与较早 PR。以下“至 9/21”只指列明入口在本次访问中的可见内容，不包括当天未来发布；网页缓存、提交历史不足和失败来源单独列出。

#### 实际来源覆盖

| 来源 | 主线 / 深度 | 实际窗口或事件与结果 | 截止及保留缺口 |
| --- | --- | --- | --- |
| [verl release](https://github.com/verl-project/verl/releases)、#7373/#7324/#7511 | T2/T3，深读 | 0.9.1（9/20）及 8 月 PR；0.9.0（8/14）基线；E-01 | release 至本次 9/21；未逐读全部 PR，固定版 async 文档与 #7781 单页失败 |
| [AReaL releases](https://github.com/areal-project/AReaL/releases)、README/async 入口 | T2，扫描/机制 | 最新可见 2.1.0（8/25），不计当周新发 | release 至 9/21；main commits 429，提交窗口未推进 |
| [slime releases](https://github.com/THUDM/slime/releases)、#2272/#2340 | T2，部分深读 | 0.3.2（8/28）；两 PR 9/3 合并，不归入该旧 release | release 至 9/21；commit 页仅有较旧 9/3 可见状态，之后待查 |
| [Megatron releases](https://github.com/NVIDIA/Megatron-LM/releases)、#6135 | T1/T4，深读 | Core 0.19.2（9/18），FP4 调度边界 E-09；旧三版仅扫日期 | release 至 9/21；#7198 未独立深读、完整 90 天回填未完 |
| [PyTorch 2.14 tag](https://github.com/pytorch/pytorch/releases/tag/v2.14.0) | T1/T6，旧基线复核 | 9/2 发布；distributed/编译/FSDP，nccl2 为 preview | 仅该 tag；不代表全部后续提交 |
| [TorchTitan](https://github.com/pytorch/torchtitan)、[#4737](https://github.com/pytorch/torchtitan/pull/4737) | T1/T2/T6，定点深读 | README；FA4 b31 的 9/17 合并，fresh API 原文 | PR 截至 9/21；commit 索引缓存只到 9/10，不推定后续无变化 |
| [Curator releases](https://github.com/NVIDIA-NeMo/Curator/releases) | T1，扫描 | 可见 1.3.0（7/27），数据管线基线 | release 至 9/21；main 缓存较旧、数据/recipe 全面调查未完 |
| [vLLM releases](https://github.com/vllm-project/vllm/releases) | T3，扫描 | fresh latest 仍 0.29.0（9/9） | 正式 release 至 9/21；不是 main 无改动 |
| [SGLang 0.5.20](https://github.com/sgl-project/sglang/releases/tag/v0.5.20)、#37874 | T3/T5，深读 | 9/18 UTC 发布；PD 启动 deadline，E-10 | release/已读 PR 至 9/21；#36612 单页失败，不展开其机制 |
| [Dynamo releases](https://github.com/ai-dynamo/dynamo/releases) 及四个精确 snapshot | T3/T4/T5，深读 | V4.1 9/12 限定复核；A.X-K2 9/15、Solar Open2 9/16、K-EXAONE 9/17 | 对列明 tag 核查至 9/21；E-07，不将分支 snapshot 当稳定升级 |
| [NVRx 0.7.0](https://github.com/NVIDIA/nvidia-resiliency-ext/releases/tag/v0.7.0)、#314/#376、tag 源码 | T5，深读 | 9/16 正式版，CPU/SHM 快照与 IPC；E-02 | release/版本源码至 9/21；fresh API 纠正 #376 旧网页 Open 缓存 |
| [Kueue releases](https://github.com/kubernetes-sigs/kueue/releases) | T5，深读/扫描 | 深读 0.19.5（9/17）；识别 0.18.9、0.20.0-rc.0，E-11 | 列表至 9/21；后两版及原修复 PR 未全部读完 |
| [llm-d 0.9.0](https://github.com/llm-d/llm-d/releases/tag/v0.9.0) | T5，基线扫描 | 8/17 release 与指南入口，不计当周 | tag/列表至 9/21；所有 versioned guide 和 90 天回填未完 |
| [TorchFT 0.2.0](https://github.com/meta-pytorch/torchft/releases/tag/v0.2.0) | T5，基线扫描 | fresh API 8/27，DDP/HSDP、streaming recovery 等入口 | 仅该版本；没有训练恢复/收敛实验复核 |
| [NCCL releases](https://github.com/NVIDIA/nccl/releases)、精确 tag/API | T4/T5/T8，深读 | 2.32.3-1（9/17 UTC），E-03 | release 可见窗口至 9/21；不含所有 main PR |
| [NIXL 1.4.1](https://github.com/ai-dynamo/nixl/releases/tag/v1.4.1) 及列表/API | T4，旧记录复核 | 9/1 23:51:45 UTC，778edd1；generation handle 与固定 expert-capacity 机制获支持 | release 至 9/21，旧 9/7 该条不重复建新卡 |
| [DeepEP](https://github.com/deepseek-ai/DeepEP)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[UCCL](https://github.com/uccl-project/uccl) | T4，入口扫描 | 当前 README：V2/GIN/实验路径、kernel/异构通信入口 | 9/21 页面快照；未锁 commit，不能赋予全部能力本周日期 |
| [NVSHMEM 3.7.2](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html) | T4，基线扫描 | 文档显示 7/22 更新，GPUNetIO 修复与限制 | 仅可见文档；页面更新日不是所列机制首次公开日 |
| [FA4 releases](https://github.com/Dao-AILab/flash-attention/releases)、b31/#2810 | T4/T6，深读 | b31 9/16、PR 9/11，连到 TorchTitan，E-08 | release/对应 PR 至 9/21；未全面核更早 beta |
| [FlashInfer releases](https://github.com/flashinfer-ai/flashinfer/releases) | T4/T6，扫描 | 0.7.0rc3（9/16）仅 trace-registry 修补；0.6.18.post1（9/5）与 0.6.18（8/29） | 不将 rc/nightly 计作稳定版；rc1/2 机制仍待补 |
| [Triton](https://github.com/triton-lang/triton/releases)、[CUTLASS](https://github.com/NVIDIA/cutlass/releases) | T6，历史定点复核 | 3.8.0 8/28/c01b677；4.8.0dev 8/27/cdcf8d8；旧 8/31 版本日期获支持 | release 至 9/21，未扫全部 PR；dev 不变成 GA |
| [cuDNN releases](https://github.com/NVIDIA/cudnn-frontend/releases) 与 1.29 tag | T6，定点深读 | 1.29 9/13/91dbf3e；1.30.0.dev68764132 为 9/19 unsupported nightly | release 至 9/21，不建立“1.30 正式发布”事实 |
| [TileLang releases](https://github.com/tile-ai/tilelang/releases)、[Helion releases](https://github.com/pytorch/helion/releases) | T6，扫描 | 0.1.14 9/2/7e3bbf3；1.4.0 7/29/5205859，均是较早事件 | release 至 9/21；机制回填未完 |
| [cuTile Python releases](https://github.com/NVIDIA/cutile-python/releases)、[Rust 报告](https://developer.nvidia.com/blog/translating-cuda-tile-operations-from-python-to-rust-using-agentic-ai/) | T6，深读 | Python 页面无 GitHub release；Rust 9/16 报告，E-05 | Python tags/PyPI 未查；TileGym commit history 失败 |
| [Omni release/README](https://github.com/sgl-project/sglang-omni/releases) 与 #2163/#1773/#2115 | T7/T5，深读 | 0.1.6（9/17）全链路准入/状态边界，E-04 | release/已读 PR 至 9/21；不把 main 全能力投射到该 tag |
| [Diffusers releases](https://github.com/huggingface/diffusers/releases) 与 0.40.0 | T7，扫描/基线 | 8/20/d035dcd；Modular/TP 属旧结果 | release 至 9/21，后续 commits 未遍历 |
| [MiniMax-H3 原始测量](https://www.lmsys.org/blog/2026-08-27-minimax-h3-h200)、[AuK 模型卡](https://huggingface.co/tencent/AuK)/cookbook | T7，基线候选 | H3 8/27 报告、8/18 实验；AuK 9/9 开源入口 | 暂不入重点证据，需对照 commit/质量/首次时间与旧日志去重；LMSYS 首页外壳不算全站已查 |
| [NVIDIA 技术博客](https://developer.nvidia.com/blog/)、NVLink 6/Shadow/LPX 具体正文 | T5/T8，深读/扫描 | 9/15 分层恢复报告回溯 8/25 Shadow，E-06；9/15 LPX 电源控制报告 | 截至 9/21 选定文章；未穷尽硬件/C2C/DPU/storage，LPX 泛化与部署证据待补 |
| [ROCm legacy](https://github.com/ROCm/legacy-rocm-build/releases)、[TheRock releases](https://github.com/ROCm/TheRock/releases) | T8，来源迁移扫描 | 旧 ROCm URL 重定向 legacy；新列表 10.0（8/26）、7.14.1（8/31） | release 列表至 9/21；10.0 release-notes 正文及 transition guide 访问失败，不据标题生成技术结论 |
| [AITER](https://github.com/ROCm/aiter)、[MaxText](https://github.com/AI-Hypercomputer/maxtext)、[SGL-JAX](https://github.com/sgl-project/sglang-jax) | T8，入口扫描 | 打开当前仓库入口；AITER 可见 7 月 Kimi 支持信息 | 未锁 commit/逐项比较；不是已完成跨平台质量成本核验 |
| [MLX releases](https://github.com/ml-explore/mlx/releases) | T8，扫描 | 最新可见 0.32.2（8/25/1f8e74e） | 列表至 9/21，无本周新增 release 证据；main 未遍历 |
| [UALink specifications](https://ualinkconsortium.org/specification/) | T8，基线扫描 | Common2.0/DLPL/Chiplet/Manageability 列表可见 | specification 入口，不是部署或性能证据，发布日期未重新追全 |
| BRACE [版本历史](https://arxiv.org/abs/2609.09783)/[v2 全文](https://arxiv.org/html/2609.09783v2) | T2，跨清单深读候选 | v1 9/9、v2 9/15；critic 陈旧修正与作者实验 | 补录候选，不采聚合页 9/16 日期；独立 time-to-quality 未成立 |
| OAK [原始摘要/历史](https://arxiv.org/abs/2609.19024)、[负载调节论文](https://arxiv.org/abs/2609.05406v1) | T5/T8，跨清单发现 | 检索恢复成本/功率/调度，打开原始页面；前者 ID 月份与页面 7/19 日期存在异常，后者 v1 为 9/4 | 仅候选：OAK 日期待交叉核验，后者正文/模型假设待读；未入已确认性能结论 |

#### 旧 issue 回访

fresh GitHub API 于本次访问确认以下三项仍 Open，没有核到对应修复发布：

- [SGLang #36452](https://github.com/sgl-project/sglang/issues/36452)：8/26 创建，9/18 更新；[新用户评论](https://github.com/sgl-project/sglang/issues/36452#issuecomment-5731957058) 涉及 v0.5.19/Qwen3.8-27B-NVFP4/NEXTN/FP8 KV，缺硬件与严控实验。不视为维护者确认，不视为对 0.5.20 的测试。
- [vLLM #55000](https://github.com/vllm-project/vllm/issues/55000)：9/2 创建、9/9 更新；secondary agent 初始化 hang。无已核修复 PR。
- [vLLM #55181](https://github.com/vllm-project/vllm/issues/55181)：9/3 创建、9/4 更新；remote disconnect 未达失败终态。新 release 或其他项目的 timeout 不构成该问题闭环。

完成的是“旧状态未回访”的缺口；故障本身并未关闭。对缓存陈旧的网页优先记录 API/固定版本依据，不沿用旧状态。

#### 待办与下次断点

1. **仍未全部复核的历史**：开放数据/recipe、FSDP/ZeRO/并行、模型/attention/低精度、UCCL/DeepEP/硬件的 8/24 精确性能结论，以及 8/31、9/7 尚未列入上述定点复核的条目。
2. **近 90 天基线仍不完整**：T2 的 AReaL/slime 与 critic/staleness 独立实验；T5 的 TorchFT/llm-d/Kueue 深层失败语义；T1 数据/optimizer；T7 生成质量与 T8 同条件成本。
3. **已查但仍需深挖的原始候选**：BRACE（旧论文补录）、slime 9/3 两 PR、Omni/H3/AuK；ROCm TheRock 10.0 版本正文；LPX 能耗控制的条件与可获得部署。候选不自动修改当前判断。
4. **本轮只做入口或未覆盖的来源**：DeepSpeed/Transformer Engine 独立 releases、KTransformers/TensorRT-LLM、LMCache/Mooncake、C2C/DPU/storage/Engram 独立新论文没有全面检查，保持原截止或未建立状态。
5. **新代码/修复触发**：FA4 NaN padding 与更多 TP/精度；Dynamo Known Issues 的修复版本；NVRx 恢复一致性；vLLM 三类生命周期问题。后续不得把“Open”当已确认根因或把“Closed”当已发布修复。
6. **执行与覆盖分开**：默认本地执行因新增 writable root 是 symlink 而在启动前失败；经已获批准的、明确限定到笔记/公开来源的 escalated read 和 apply_patch 完成维护。没有修改 sandbox 配置、符号链接或自动化日程，没有下载模型/视频、安装环境或执行 GPU 实验。

产物：新增 E-20260921-01～11，追加本月周报；更新 06 和 04/05/07/08/09 的可复用机制，来源登记修正迁移入口。历史归档原文冻结，9 月月度标记保持初始化；10 月首个成功周运行再做周期复盘。

验收：15 份活跃文档的 268 处可解析内部链接通过检查；17 张证据卡 ID 唯一（本次 11 张），周报重点事件恰好 8 项，表格转义/代码围栏/空白检查通过。历史归档 SHA-256 与执行前一致。另一代理只读复核了周报、台账与 06 的日期/成熟度/判断一致性，发现的两处措辞问题已修正；这是文档审查，不是独立 GPU 性能复测。
