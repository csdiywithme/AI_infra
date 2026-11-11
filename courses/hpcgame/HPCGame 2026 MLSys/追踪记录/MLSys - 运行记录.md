---
type: research-run-state
status: archived
protocol_version: 2
initialized: 2026-09-15
last_attempt: 2026-10-19
last_completed_weekly: null
last_successful_partial: 2026-10-19
last_run_status: partial-source-coverage
last_monthly_review: "2026-10"
last_monthly_review_kind: periodic
last_monthly_review_date: 2026-10-12
timezone: Asia/Shanghai
---

# MLSys：运行记录与核查覆盖

[[领域地图与覆盖缺口|AI Infra 领域地图]] · [[MLSys - 追踪规则与来源|旧来源登记]] · [[MLSys - 证据台账|事实台账]]

## 1. 怎样理解这里的日期

- `last_attempt` 是最近执行日；`last_completed_weekly` 仅在当次计划的每周检查全部完成后推进，部分运行保留缺口。
- 每条来源的实际核查截止优先于全局日期。未完成项不会因其他来源成功而被跳过。
- `last_monthly_review` 用运行月份判断是否需要本月复盘；2026-09 是本次重构建立起点，不代表已经覆盖完整 9 月。
- 旧 tracker 的 2026-09-14 是历史报告截止，迁移时不冒充新体系完整核查。2026-09-15 是局部一手审计与基线初始化，没有开展全领域的新增检索。
- 下表“局部核验”表示核验了指定机制/事件；只有对来源时间窗口实际检查后，才填写该来源覆盖到哪一天。

## 2. 主线覆盖与断点

| 主线 | 最近局部核验 | 已查内容 | 完整来源窗口状态 / 下次补齐 |
| --- | --- | --- | --- |
| T1 | 2026-10-19 | Midtraining新论文、Curator精度与数据选择深读 | 部分；开放处理制品、独立质量成本仍缺；PyTorch/Megatron/TE/TorchTitan未重查 |
| T2 | 2026-10-19 | verl完整性receipt合入；slime/AReaL索引与风险 | 部分；最终head完整验证/多seed成本、全main/90天未完 |
| T3 | 2026-10-19 | vLLM0.32与main差异、HiSparse反证、KV工作集/历史 | 部分；SGLang0.5.22正文未取得；LMCache/Mooncake/TRTLLM未重查 |
| T4 | 2026-10-19 | NIXL1.5、DeepEP2.5、Ascend snapshot深读 | 部分；DeepGEMM CUDA候选、UCCL全窗口、NVSHMEM独立文档未查 |
| T5 | 2026-10-19 | P2P生命周期、llm-d实验指南、旧风险回访 | 部分；正式故障/安全闭环和长期成本仍缺；Kueue/NVRx/TorchFT未重查 |
| T6 | 2026-10-19 | cuDNN1.31数学合约、TileLang Ascend定点核验 | 部分；AI kernel独立新实验、Helion/TileGym未查，全部main未核 |
| T7 | 2026-10-19 | Omni0.1.8 Flow缓存、两篇agent缓存论文补录 | 部分；Diffusers0.41仅发布正文，实时多模态全组合/质量未核 |
| T8 | 2026-10-19 | Ascend950固定组合、ROCm10.1补扫与旧风险 | 部分；平台等价成本、AITER/MLX/UALink/Rubin/C2C/DPU/storage/Engram专门窗口未核 |

## 3. 高优先级待办

1. **历史准确性补齐**：按 [[MLSys - 2026-08 至 09 旧版基线与日志]] 回访 8/24、8/31、9/7 的精确版本/日期。已完成的 9/14 六项不重复建卡；发现旧事件补录不计当周新增。
2. **新主线基线回填**：优先 T2、T5，再 T1/T7/T8；补查近 90 天版本、作者论文与原始实验，保留不能证实的部分。
3. **已报告故障闭环**：10/19 fresh API回访下列三项仍Open；#55181相关#59790/#60279已合入main但0.32.0未含，#60374仍Open且最新验证未读完，新增consumer边界#61532。不能把部分修复当总问题解决；当前依据见本次日志，月度表保留10/12历史时点。
   - [SGLang #36452](https://github.com/sgl-project/sglang/issues/36452)：启动临时显存与 KV 定容。
   - [vLLM #55000](https://github.com/vllm-project/vllm/issues/55000)：P2P secondary agent 启动挂起。
   - [vLLM #55181](https://github.com/vllm-project/vllm/issues/55181)：remote disconnect 与 recompute fallback。
4. **部署组合后续**：Dynamo V4.1 snapshot 之后是否发布 QA-gated 版本，是否补 FP4 KV/视觉/分离式 speculation。基准与日期见 E-20260915-03。
5. **轮换深读**：尚未完成深读的其他主线先于重复深挖热门项目；每周至少一个跨清单的问题检索。
6. **新增优先风险**：CVE-2026-93088 的维护者修复版本；ROCm #8229 的固定修复提交/正式包；AReaL #1750 未闭合 receipt 风险。不能凭 Open/Closed、版本更高或 develop workaround 判定已安全。
7. **10月优先级**：10/19完成T1 midtraining报告和数据选择深读，但处理制品仍待公开、独立质量成本仍缺；下一轮继续补这些证据，而不是恢复只追release。Ascend已补固定算子组合，未完成整服务成本。详见 [[MLSys - 月度复盘#2026-10]]；月度已完成，不重复复盘。

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

### 2026-09-28

类型：八主线部分来源扫描与重点机制深读，完成已核材料入库。按成功来源的 9/21 截止回看 14 天（9/7 起），并优先补查上轮访问失败和遗漏来源。下表“至 9/28”均指本次访问时实际可见入口，不含当天未来事件、未读分支或全部 PR。

本次有实质新增，新增 E-20260928-01～11（其中 ROCm、DeepSpeed 为补录），周报选择八个重点事件。部分历史/90 天回填、main 与平台窗口未覆盖，`last_completed_weekly` 继续留空，**不报告全局无更新或全部已核**。9 月已有初始化，`last_monthly_review` 不变；10 月首个成功周运行执行周期复盘，不能因持续有研究缺口而无限顺延复盘。

#### 实际来源覆盖

| 来源 / 原始入口 | 主线 / 深度 | 实际结果 | 截止 / 缺口 |
| --- | --- | --- | --- |
| [verl](https://github.com/verl-project/verl/releases)、#7986/#7987 | T2/T3/T4，深读 | 最新 release 仍 0.9.1；9/7～9/28 main API 索引 68 项，新 refit 修复 E-02 | 指定窗口索引/两 PR 至 9/28，非全部代码深读 |
| [AReaL](https://github.com/areal-project/AReaL/releases)、#1721/#1750 | T2/T5，深读 | release 仍 2.1.0；main API 49 项，部分组/失败归因 E-01 | 上次 main 429 缺口本轮补通；90 天与真实学习验证未完 |
| [slime](https://github.com/THUDM/slime/releases)、#2405/#2406 | T2，扫描 | release 仍 0.3.2；窗口 main 索引 8 项，最新 9/23；两 PR 描述为空 | 索引至 9/28；score centering/top_p replay 的 diff/quality 待查 |
| [DeepSpeed 0.19.7](https://github.com/deepspeedai/DeepSpeed/releases/tag/v0.19.7)、#8442/#8438 | T1/T5，深读补录 | 9/16 正式版及 Muon 两条错分支修复，E-10 | release/指定 PR 至 9/28；#8438 API 403 后 web 正文成功，main 未全查 |
| [Transformer Engine 2.19](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.19)、[#3475](https://github.com/NVIDIA/TransformerEngine/pull/3475) | T1/T4，补查 | 9/11 正式版；屏蔽 cuBLAS 13.7 在 B300/Rubin 的一个 grouped-GEMM 错误 algorithm | 正文已核；文件 diff 失败，不把 bot 的更窄 NVFP4/SM103/107 摘要当已核源码 |
| [Megatron](https://github.com/NVIDIA/Megatron-LM/releases)、[PyTorch](https://github.com/pytorch/pytorch/releases)、[Curator](https://github.com/NVIDIA-NeMo/Curator/releases) | T1/T6，扫描 | 可见最新仍 Core0.19.2 / PyTorch2.14.0 / Curator1.3.0 | 本次列表访问；Curator 缓存至 9/25，不声称其后窗口无更新；全部 main 未查 |
| [TorchTitan commits](https://github.com/pytorch/torchtitan/commits/main/) | T1/T2，受限扫描 | 页面仍只到 9/10 | 不推进 main 已核窗口，保留上次定点 #4737 核验 |
| [vLLM 0.30](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)、#53781/#54921 | T3/T4/T5，深读 | 正式 9/22，HiSparse/Fast Start/MRV2 组合，E-03 | release/指定 PR 至 9/28；全部 762 commits、B300 benchmark 条件未全核，#56472 页面失败 |
| [SGLang](https://github.com/sgl-project/sglang/releases)、三旧 issue | T3/T5，回访 | release 仍 0.5.20；fresh API 三项均 Open，更新时间不变 | release/指定 issue 至 9/28，main/所有回归未查；状态不等于根因或无关联修复 |
| [Dynamo 1.5](https://github.com/ai-dynamo/dynamo/releases/tag/v1.5.0)、#11239/#11571 | T3/T4/T5，深读 | 9/21 20:21 UTC 正式版晚于上次运行；tiered routing/KVBM deprecation，E-04 | Summary/Dependencies/Breaking Changes 与两 PR 至 9/28；旧 snapshot 修复状态未闭环 |
| [NVRx](https://github.com/NVIDIA/nvidia-resiliency-ext/releases)、[llm-d](https://github.com/llm-d/llm-d/releases)、[TorchFT](https://github.com/meta-pytorch/torchft/releases) | T5，扫描 | fresh releases 仍 0.7.0/0.9.0/0.2.0；TorchFT 短版说明已读 | release 至 9/28；main、子组件、versioned guides/训练恢复实验未全核 |
| [Kueue 0.19.6](https://github.com/kubernetes-sigs/kueue/releases/tag/v0.19.6)、#16111/#16132 | T5，深读 | 9/24 正式 quota 修复 E-05；同日0.20rc1只是预发布 | release/原 PR 至 9/28；rc1/DRA #15987未深读，后者 API403 |
| [LMCache 0.5.5](https://github.com/LMCache/LMCache/releases/tag/v0.5.5) 与 release 列表 | T3/T5，补扫 | 9/12 旧版 isolated IPC/协调器等；9/26 0.5.6rc1正文空，9/27nightly/platform wheels | 仅已读 release 部分；rc标签不因 API prerelease=false 而变正式，ABI/完整 PR 待核 |
| [Mooncake](https://github.com/kvcache-ai/Mooncake/releases) | T3/T4，补扫 | 0.3.14rc1 9/7正文空；0.3.13.post1 8/31 non-CUDA wheel 修补；0.3.13开头/变更表 | 列表至 9/28；大版正文截断，main/90 天基线未完成 |
| [TRT-LLM rc28](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.3.0rc28) | T3，部分深读 | 9/23 rc28、9/18 rc27、9/9 rc26；rc28前部 Known Issues/Features/Fixes | 均不当 stable；原问题首次日期、tag SHA 和跨版闭环待核 |
| [KTransformers 0.7.1](https://github.com/kvcache-ai/ktransformers/releases/tag/v0.7.1) | T3，补扫 | 9/15/0dce4c9，Qwen BF16 VLM LoRA、Kimi RAWINT4 text LoRA 入口 | 短 release 全文；模型/硬件/quality 指南未核，是旧补录候选 |
| [NCCL](https://github.com/NVIDIA/nccl/releases)、[NIXL](https://github.com/ai-dynamo/nixl/releases) | T4/T8，扫描 | 最新可见仍 2.32.3-1/1.4.1，不重复建卡 | release 列表至 9/28；main/更早机制未穷举 |
| [DeepEP](https://github.com/deepseek-ai/DeepEP)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[UCCL](https://github.com/uccl-project/uccl) | T4，入口 | 当前 README；DeepGEMM News 有9/10条目，其他能力无新事件日期锁定 | 仅9/28快照；未锁commit/全PR，不把滚动能力视为本周新增 |
| [FlashInfer 0.7](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.0)、#3861/#3920 | T4/T6，深读 | 9/22/4d75a33正式发布；部署模式、环境缓存与成熟度，E-06 | 原PR7/7～9/5及当前RFC；代表性probe/异构rank部分warm缓存仍有边界 |
| [CUTLASS 4.8](https://github.com/NVIDIA/cutlass/releases/tag/v4.8.0)、[cuDNN 1.30](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.30.0) | T6/T8，深读 | 9/22、9/23正式版本；cuDNN #1028/#1031/#1039/#1146正文；E-07/08 | 网页日期/commit已核，API403未补到精确秒；不混nightly/dev/preview |
| [Triton](https://github.com/triton-lang/triton/releases)、[TileLang](https://github.com/tile-ai/tilelang/releases)、[Helion](https://github.com/pytorch/helion/releases) | T6，扫描 | 最新可见仍3.8.0/0.1.14/1.4.0 | release至9/28，main未穷举 |
| [cuTile PyPI JSON](https://pypi.org/pypi/cuda-tile/json)、[TileGym目录](https://github.com/NVIDIA/TileGym/tree/main/src/tilegym/ops/cutile_rs) | T6，补查 | cuda-tile1.6.0首upload 9/10 00:03:56 UTC；GH tags缓存仅v1.5 | PyPI缺口补通；旧事件不计本周。TileGym commit history仍失败，原E-20260921-05限制不撤销 |
| [Omni](https://github.com/sgl-project/sglang-omni/releases)、[Diffusers](https://github.com/huggingface/diffusers/releases) | T7，扫描 | 最新可见仍0.1.6/0.40.0 | release至9/28；main网页缓存到9/11，不推进后续提交窗口 |
| [CVE-2026-93088 CNA](https://cveawg.mitre.org/api/cve/CVE-2026-93088)、[固定源码](https://github.com/sgl-project/sglang/blob/v0.5.20/python/sglang/multimodal_gen/runtime/disaggregation/orchestrator.py) | T3/T5/T7，深读 | 9/22公开记录及affected0.5.11～0.5.14，E-11 | CNA经public curl成功；webAPI失败/NVD正文空/原发现者页失败；修复版本未知 |
| [TheRock](https://github.com/ROCm/TheRock/releases)、[10.0正文](https://rocm.docs.amd.com/en/docs-10.0.0/about/release-notes.html)、[迁移指南](https://rocm.docs.amd.com/en/docs-10.0.0/about/transition-guide-TheRock.html)、#8229 | T8/T5，补录深读 | release仍10.0(8/26)/7.14.1(8/31)；正文缺口补通，E-09 | SDK/framework/ABI与Known Issues定点核查；7.14.1全文、修复commit/性能未全核。#8229网页缓存9/22，fresh API403 |
| [AITER](https://github.com/ROCm/aiter)、[MLX](https://github.com/ml-explore/mlx/releases)、[UALink](https://ualinkconsortium.org/specification/) | T8，入口 | AITER仍滚动README/7月新闻，MLX最新可见0.32.2，UALink spec列表 | 9/28入口，不是日期/生产/跨平台实验闭环；TPU/MaxText/SGL-JAX本轮未重查 |
| [NVIDIA博客](https://developer.nvidia.com/blog/)与SWE-Serve/Confidential Computing文章 | T5/T6/T8，发现/部分阅读 | 9/23、9/22新报告入口已打开 | 未深入代码/实验，保持候选；C2C/DPU/storage/Engram与MLPerf新硬件结果未完成 |
| 跨清单问题：工具进度、KV回收/隔离与checker可靠性 | T2/T3/T5/T6/T7，检索+定点正文 | [Ask the Tool](https://arxiv.org/abs/2609.18849) v1=9/16及正文；[KernelBench-M](https://arxiv.org/abs/2609.22220)原页/HTML/HF | 前者4×H100作者replay的post-tool指标，完整模型/SLO/commit未核；后者日期/计数有疑点，均不据候选改判断 |

#### 旧问题、补核与下一次断点

1. **三旧 issue**：[SGLang #36452](https://github.com/sgl-project/sglang/issues/36452) 最后更新9/18；[vLLM #55000](https://github.com/vllm-project/vllm/issues/55000) 9/9；[#55181](https://github.com/vllm-project/vllm/issues/55181) 9/4。9/28 fresh API均Open，无已核对应修复发布。CVE/ROCm新增待办与其分开跟踪。
2. **本轮实质补齐**：AReaL/slime main 指定窗口、DeepSpeed/TE release、LMCache/Mooncake/TRTLLM/KTransformers入口、ROCm10正文/迁移与cuTile PyPI。补通访问不表示全部机制/90天已回填。
3. **优先跟进**：AReaL receipt分类风险；verl refit真实多卡验证与正式版；CVE维护者修复；ROCm profiler正式修复包；Dynamo丢事件/晚订阅/多租户及旧snapshot回归；Kueue预约不变量。
4. **仍缺独立质量/成本证据**：开放数据recipe、异步RL同质量成本、长期恢复损失、跨平台同条件实验。BRACE原版记录本次仍v2 9/15，不重复上周论文候选。slime #2405/#2406、TRTLLM rc28 Known Issues、LMCache0.5.5机制留深读队列，不提升为已解决/生产。
5. **跨清单队列**：Ask the Tool 的工具进度—KV策略；KernelBench-M checker audit（原页v1显示9/2而ID为2609.22220，需交叉核日期及计数口径）；[SWE-Serve](https://developer.nvidia.com/blog/how-swe-serve-exposes-the-gap-between-local-tests-and-live-serving/)、[Confidential Computing](https://developer.nvidia.com/blog/enabling-private-high-performance-production-ai-inference-with-nvidia-confidential-computing/) 的公开制品与实验；LoRA共现2609.23321、streaming diffusion2609.22283尚仅线索。候选不等于正式证据。
6. **轮换覆盖**：下轮优先补T1开放数据/T8跨平台与C2C/DPU/storage，再回访T7生成质量。NVSHMEM、FA4、TPU来源本轮没有重核，不随主线表全局推进。OAK日期异常与旧8月/9月基线缺口仍保留。
7. **执行限制**：本地sandbox启动仍因另一个writable root为symlink而失败，经已批准、限定路径的escalated read/apply_patch维护。公开GitHub API部分请求403，改用原网页时记录缓存限制；没有修改配置/符号链接/日程，没有下载模型/视频、安装环境或执行GPU/安全复现实验。

产物：11张新证据卡、9/28周报、06当前判断和03/05/07/08/09机制解释。00本月导航已指向9月且仍有效，月度页与历史原文不修改。结构/日期/重复检查结果在本条末尾记录。

验收：15份活跃文档共321处可解析内部链接，无新增断链/标题错误；13个既有概念占位链接仍保留。28张证据卡ID唯一（本轮11张），周报8个重点事件。旧17张卡正文与旧周报逐字保持，冻结归档SHA-256仍为 `d712b9b8d167b68b790cefbfa924498ab7e7c4bba3d993f04a308c76054cc70f`。围栏、表格alias转义及 `git diff --check` 通过。两名研究代理分别复核了RL/optimizer、编译/CVE相关原始证据的转述，另检查内部日期/成熟度一致性；据此收窄了失败政策和未来驱动措辞。此为文档核查，不是独立GPU/漏洞复现。只修改本目录9份活跃文件，未动无关笔记、历史原文或自动化日程。

### 2026-10-12

类型：每周部分来源扫描＋重点深读＋首轮周期复盘。上次真实记录为9/28，10/5无完成记录；此次直接覆盖遗漏窗口，不造10/5日志。成功来源按各自截止回看14天（通常9/14起），未访问/失败来源保持旧截止。新增E-20261012-01～12，周报八个重点事件；10月复盘已完成，`last_completed_weekly`仍为空。

#### 实际来源覆盖

下表“至10/12”仅指本次访问时的指定API/正文/版本，不是全仓库全部提交，也不含当天未来事件。

| 来源 | 深度 / 本次可确认内容 | 截止与未完成范围 |
| --- | --- | --- |
| [slime0.4.0](https://github.com/THUDM/slime/releases/tag/v0.4.0)、#2410/#2444、固定恢复/Straw文档 | T2/T5深读；10/9正式发布，提交/重放边界E-01 | 列明文档与PR至10/12；非全部main/存储后端/长期质量 |
| [verl #8010](https://github.com/verl-project/verl/pull/8010)、[#8196](https://github.com/verl-project/verl/pull/8196)、[#8064](https://github.com/verl-project/verl/pull/8064)与[SC论文](https://arxiv.org/html/2609.20807v1) | T2深读；main合入与Open扩展分开；slime #2405/#2406补核，E-02 | latest仍0.9.1；main网页缓存旧，全main窗口不推进；refit #7986/#7987无新已核验证/版本事件 |
| [AReaL](https://github.com/areal-project/AReaL/releases)、[#1750](https://github.com/areal-project/AReaL/pull/1750) | release仍2.1.0；fresh API取9/28～10/12共10个main提交，最新ecda014919为10/12 02:46UTC；receipt风险正文未撤销 | 指定索引/风险至10/12；网页main失败已由API补通，但未深读全部提交/90天 |
| [Curator1.4.0](https://github.com/NVIDIA-NeMo/Curator/releases/tag/v1.4.0) | T1扫描，10/9 01:13UTC，10fba1e，release正文已开 | 机制/开放数据recipe/质量待深读，不能计数据质量进步 |
| [Megatron报告](https://developer.nvidia.com/blog/scale-bitwise-deterministic-pretraining-with-nvidia-megatron-core/)、[#7262](https://github.com/NVIDIA/Megatron-LM/pull/7262)及release列表 | 10/6确定性报告，原PR仍Draft/Open；26.09-alpha.rc1/rc2为候选，stable Core仍0.19.2 | 厂商2432GPU/800步数据缺完整复现组合，不入性能结论；不能把报告当正式功能 |
| [PyTorch2.14.1](https://github.com/pytorch/pytorch/releases/tag/v2.14.1)、[DeepSpeed](https://github.com/deepspeedai/DeepSpeed/releases) | 9/30 patch正文含CUDA/MPS correctness候选；DeepSpeed仍0.19.7 | CUDA原始说明/修复差分待核；TE/TorchTitan本轮未重查，保留9/28或更早截止 |
| [vLLM0.31](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)、[#59309](https://github.com/vllm-project/vllm/pull/59309) | T3/T4深读，10/5正式版与draft/stream修复E-03 | 指定内容至10/12；残余PD race后续修复未核，非全release所有PR |
| [SGLang0.5.21](https://github.com/sgl-project/sglang/releases/tag/v0.5.21)、#41023/#40777 | T2/T3/T5深读，10/2版的ACK默认与weight session，E-04 | 已读highlights/spec/PD/security/deps/breaking；其他779PR未穷举 |
| [Dynamo1.5.1](https://github.com/ai-dynamo/dynamo/releases/tag/v1.5.1)、[#14474](https://github.com/ai-dynamo/dynamo/pull/14474)、[V4.1 dev.2](https://github.com/ai-dynamo/dynamo/releases/tag/v1.6.0-deepseek-v4.1-flash-dev.2) | T3/T5/T7深读；10/7正式安全修补E-05，10/8非QA新snapshot E-12 | 版本和组合已核，snapshot完整benchmark未核；旧K-EXAONE未另做回归 |
| [Kueue0.20.1](https://github.com/kubernetes-sigs/kueue/releases/tag/v0.20.1)、#15761/#16858 | T5深读，10/8正式second-pass修复，E-11 | 仅指定事务路径，不代表全部TAS/DRA已核 |
| [CNA](https://cveawg.mitre.org/api/cve/CVE-2026-93088)、[0.5.21 orchestrator](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/multimodal_gen/runtime/disaggregation/orchestrator.py)、[#40259](https://github.com/sgl-project/sglang/pull/40259) | fresh CNA仍9/22，无fixed版；SafeUnpickler改srt/utils/common.py，不是已核diffusion闭环 | CNA web失败已用原站curl补通；未扩展affected范围、未执行漏洞 |
| 三旧issue与vLLM #60374/#60279 | fresh API回访；#36452/#55000无新更新时间，#55181有10/7讨论；两PR仍Open | 指定状态至10/12，mock tests不等真实RDMA验证，详见月度风险表 |
| [NVRx](https://github.com/NVIDIA/nvidia-resiliency-ext/releases)、[TorchFT](https://github.com/meta-pytorch/torchft/releases) | 列表最新可见0.7.0/0.2.0，未重复旧卡 | 网页分别缓存至10/11/上周；只确认可见版本，不声称缓存后无更新 |
| [llm-d](https://github.com/llm-d/llm-d/releases) | 9/29 v0.10.0/f22bbaf，读到镜像签名/仓库迁移及依赖矩阵 | 新版候选；签名范围仅llm-d仓库生成的server镜像，非router/sidecar，未深读全部guide |
| [MaxText](https://github.com/AI-Hypercomputer/maxtext/releases/tag/maxtext-v0.2.5)、#5321、固定配置、#5570 | T1/T5/T8深读，E-06；补TPU机制，RL扩展仍Open且非main目标 | 原v0.2.5短tag网页cache miss，正确maxtext-v0.2.5已核；不代表GPU/TPU等价成本 |
| [SGL-JAX](https://github.com/sgl-project/sglang-jax)、[ROCm #8229](https://github.com/ROCm/TheRock/issues/8229) | 前者README扫描；后者fresh API仍Open、created/updated均9/15，正文不变 | SGL-JAX未锁commit/新事件；ROCm仅develop workaround，未核固定修复包 |
| [FlashInfer0.7.1](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.7.1)、[#5735](https://github.com/flashinfer-ai/flashinfer/pull/5735) | T4/T6深读，10/9版共享prepared expert layout，E-10 | 非默认dispatch/整模型性能；autotune v2下游迁移未重查 |
| [cuDNN1.31](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.31.0)、[#1454](https://github.com/NVIDIA/cudnn-frontend/pull/1454) | 10/8正式版/51a3de73，release注明实际source tree 8ab41a3b8fd8f2eb0c6d4b0f4ff566c8cf1e4bed；Q/K norm与负scale等候选 | 未读diff/复现，保留候选，勿混tag与source tree |
| [Triton](https://github.com/triton-lang/triton/releases)、[TileLang](https://github.com/tile-ai/tilelang/releases)、[cuTile PyPI](https://pypi.org/project/cuda-tile/) | fresh API/PyPI：3.8仍旧，TileLang0.1.15于9/30新增，cuTile仍1.6 | 索引至10/12，TileLang正文只读摘要，机制待核；Helion/TileGym本轮未重查 |
| [CUTLASS](https://github.com/NVIDIA/cutlass/releases) | 网页最新仍4.8 | 仅列表，main/新工具链可得性未核 |
| [Omni](https://github.com/sgl-project/sglang-omni/releases)、[Diffusers](https://github.com/huggingface/diffusers/releases) | fresh API：Omni engine0.1.7为9/28 17:28UTC；voxt-v1.16.0为10/11独立macOS app；Diffusers0.41为10/6 | 新版本仅候选，质量/实现待核，不能把voxt当Omni engine升级 |
| [NCCL](https://github.com/NVIDIA/nccl/releases)、[NIXL](https://github.com/ai-dynamo/nixl/releases) | fresh API：C++仍2.32.3；nccl4py0.6为9/23旧事件；NIXL1.5为9/30新发布 | 索引至10/12，NIXL正文/PR与nccl4py机制待读，不自动推进实现覆盖 |
| [GPUNetIO](https://developer.nvidia.com/blog/doca-gpunetio-gda-ki-unified-gpu-networking/)与固定README | T4/T8深读，10/6报告、10/5 b2b7a29固定正文（raw补通），E-09 | NCCL/NVSHMEM既有集成补录，不把报告日当首次；NVSHMEM独立release文档本轮未重查 |
| [DeepEP](https://github.com/deepseek-ai/DeepEP)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[UCCL](https://github.com/uccl-project/uccl) | 当前README实读：DeepEP V2.5/Ascend入口，DeepGEMM9/30 Ascend/locality #462，UCCL异构入口 | 未锁commit/完整事件日期或性能，列高优先候选；不写“同API＝同性能” |
| [SWE-Serve](https://arxiv.org/html/2609.26777v1)、[仓库](https://github.com/NVIDIA/swe-serve)、manifest入口 | T5/T6补核9/28候选，论文正文/A.4.1、公开README，E-08 | 论文固定v1，代码main未锁commit，不是独立执行；manifest网页未逐任务审计 |
| 跨清单检索：agent KV failure/reuse/lifecycle | [KVTether](https://arxiv.org/html/2609.39819v1)9/30新论文深读E-07；[Ask the Tool](https://arxiv.org/abs/2609.18849)仍9/16v1；[Lachesis](https://arxiv.org/html/2610.08378v1)10/6正文§4/5与日期已开 | Ask the Tool完整质量/commit仍候选；Lachesis为HBF模拟，摘要/正文不同范围数值需厘清（device-years与daisy-chain），不采寿命结论 |
| [NVIDIA博客](https://developer.nvidia.com/blog/) | 最新可见10/8；选读GPUNetIO、Megatron及旧SWE-Serve | 不代表全部硬件；MLPerf/Rubin/UALink/C2C/DPU/storage/Engram与Confidential Computing旧候选仍有缺口 |

#### 断点与下一次优先事项

1. **安全/故障闭环**仍优先：CVE没有已核fixed版本；ROCm只有已核develop workaround；AReaL receipt；vLLM断连Open PR与HiSparse原PR披露的残余P/D排序（后续修复未核）。月度表保存当前证据，不因版本更高自动关闭。
2. **优先补T1数据/recipe与T2独立质量成本**：Curator1.4、SC多seed/真实模型与开销，避免再连续只深读框架correctness。MaxText的671B数字是AOT，不是运行。
3. **重要新版待查**：PyTorch2.14.1原始CUDA修复、cuDNN1.31、NIXL1.5、TileLang0.1.15、Omni0.1.7、Diffusers0.41、llm-d0.10及DeepEP/DeepGEMM新移植。版本已发现≠机制已验证，下一次从对应旧实际截止回看。
4. **未访问保留**：LMCache/Mooncake/TRTLLM/KTransformers、TE/TorchTitan、FA4、Helion/TileGym、AITER/MLX/UALink与专门硬件窗口，本轮没有重核，不随主线日期推进；原9/28及更早断点继续有效。
5. **旧历史/论文候选**：8/24、8/31、9/7精确基线与90天回填；BRACE、OAK日期异常、KernelBench-M日期/计数、LoRA/streaming diffusion、Confidential Computing/MLPerf候选均未清零。EfficientAgent2609.33762本轮仅检索发现，未开正文，不写事实。
6. **研究方法调整**已写入规则和月度复盘：harness生命周期信息加入T3/T5/T7，spec/roadmap保留扫描但不重复深挖；没有证据宣称某方向连续四周只有宣传。未改自动化日程，不另建任务。

产物：12张证据卡、10月周报、首次周期复盘、00/06本月导航与当前判断、03/04/05/07/09可复用解释及来源规则。公开资料研究，无模型/视频下载、环境安装、GPU/付费/漏洞实验；沿用限定路径的已批准escalated read/apply_patch，未改sandbox配置或符号链接。

验收：16份活跃文档共385处可解析内部链接，无新增断链/错误标题；13个既有概念占位保留。40张证据卡ID唯一，本轮12张，周报8项。此前28张卡正文、9月周报、冻结原文与9月初始化复盘逐字保持；归档SHA-256仍为d712b9b8d167b68b790cefbfa924498ab7e7c4bba3d993f04a308c76054cc70f。围栏、表格alias与git diff --check通过。三名只读研究代理复核各自来源卡，并对E2E/生命周期与月度摘要交叉复核；已补SC实验的额外weight-sync依赖、Kueue NotFound例外，区分死亡回收与live驱逐，以及未核修复和确认仍受影响。未修改历史原文、无关文件或自动化日程；没有把文档核查当GPU/安全独立实验。

### 2026-10-19

类型：每周部分来源核查、历史候选深读及反证补录。计划回看成功来源10/12之前14天（9/28起）；旧候选另回溯原始日期。10月周期复盘已完成，不重复。新增E-20261019-01～13，八项周报重点；`last_completed_weekly`仍为空，不将下面的定点成功扩展为全来源窗口完成。

#### 实际来源覆盖

“至10/19”只指本次访问时列明的来源与内容；未查部分不随主线日期推进。

| 来源 | 本次实际访问与深度 | 截止、失败与未完成 |
| --- | --- | --- |
| [Midtraining新论文](https://arxiv.org/html/2610.22019v1)及arXiv历史 | T1/T2深读§3–5、附录A～E；10/15报告、匹配SFT/开放制品/短RL边界，E-01 | 固定v1至10/19；没有已核训练脚本和完整数据manifest，非独立成本实验 |
| [Curator1.4](https://github.com/NVIDIA-NeMo/Curator/releases/tag/v1.4.0)、[#2388](https://github.com/NVIDIA-NeMo/Curator/pull/2388)、固定workflow、[#2138](https://github.com/NVIDIA-NeMo/Curator/pull/2138) | T1/T4深读FP16默认及paired选择实验E-02；CLIMB为7/8旧指南，不计新recipe | 列明版本至10/19；全部pipeline/下游训练质量未核。blob失败由同tag raw补通 |
| [verl #8196](https://github.com/verl-project/verl/pull/8196)、#8199/#8206、releases | T2深读receipt；10/18合入main，后两PR仍Open，E-03；正式仍0.9.1 | 指定内容至10/19，最终head GPU未重跑；全main、#7986/#7987本轮未回访 |
| [slime](https://github.com/THUDM/slime/releases)、[#2474](https://github.com/THUDM/slime/pull/2474) | release仍0.4；fresh main 9/28～10/19共30项，最新98e9ce5（10/19 02:14）；10/15合入2f4280e3的128K coding-agent example定点读 | #2474仅CPU测试等报告，真实RL质量未核；#2475只有标题/commit，留候选，非main全部深读 |
| [AReaL](https://github.com/areal-project/AReaL/releases)、[#1750](https://github.com/areal-project/AReaL/pull/1750) | fresh索引9/28起；10/12～19有14项含上轮已见一项，最新1683e8e（10/19 01:18）；release仍2.1 | #1750仍9/26receipt风险；#1791限流，generation-version标题不能作已核实现；无正式风险闭环 |
| [vLLM0.32](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)、#59790/#60279、tag nixl.py | T3/T5深读；10/18正式版与两main合入SHA/固定源码比较，确认未包含，E-04 | 只覆盖指定路径；814commits未穷举、其他lowprecision/security/compiler项待查 |
| [HiSparse #59633](https://github.com/vllm-project/vllm/pull/59633)、#59309与fresh comments | T3/T4补录10/5撤回提案与自然/注入实验反证，E-05 | #59309描述未变不代表最终风险强度；新证据覆盖旧认识，不改历史卡 |
| [SG #36452](https://github.com/sgl-project/sglang/issues/36452)、[vLLM #55000](https://github.com/vllm-project/vllm/issues/55000)、[#55181](https://github.com/vllm-project/vllm/issues/55181)、[#60374](https://github.com/vllm-project/vllm/pull/60374)、[#61532](https://github.com/vllm-project/vllm/issues/61532) | fresh API前三仍Open、updated9/18、9/9、10/7；#60374 updated10/18仍Open；#61532为10/14新consumer分析 | #60374最新validation未读完，不沿用仅mock旧描述；未证E2E corruption/整体修复 |
| [CVE CNA](https://cveawg.mitre.org/api/cve/CVE-2026-93088)、[SGLang releases](https://github.com/sgl-project/sglang/releases) | CNA仍9/22记录、affected0.5.11～0.5.14无fixed；fresh latest API确认0.5.22于10/19 02:55:19发布 | 后续API403，web仅tag baa9bef/10/15 tag时间，无完整release正文；**新版本已发现、机制待核**，不推CVE已修 |
| [Dynamo releases](https://github.com/ai-dynamo/dynamo/releases) | 索引/明确snapshot摘要：10/14 GLM5.3Flash dev.1、10/15 Nemotron3.5SuperVL dev.1均非QA；旧V4.1仍dev.2 | 未完整recipe/benchmark，候选，不称稳定1.6或旧组合升级 |
| [llm-d0.10](https://github.com/llm-d/llm-d/releases/tag/v0.10.0)、#2466及两固定KEDA指南 | T3/T5深读阶段工作量/启动标定、实验性/sim边界，E-06 | 9/29旧release补核；未读所有guide/实时badge/GPU实验，旧router未tag句不作当前结论 |
| [DeepGEMM-Ascend #12](https://github.com/deepseek-ai/DeepGEMM-Ascend/pull/12)、固定API/test/README | T4/T6/T8深读10/14实现、snapshot与caller合约，E-07 | 代码契约/测试设计核验，未运行，无整模型质量和平台排名 |
| [NIXL1.5](https://github.com/ai-dynamo/nixl/releases/tag/v1.5.0)、#2138/#2218及固定Python合约 | T4/T5补足连接/激活、同步与回滚E-08 | release仍1.5；main102后续提交未读，旧generation-handle/fixed-capacity不重复建卡 |
| [cuDNN1.31](https://github.com/NVIDIA/cudnn-frontend/releases/tag/v1.31.0)、#1454/#1474、shipped-source KDA | T4/T6补核数学合约与backend/负scale范围，E-09 | stable仍1.31；1.32 dev74371060（10/16）等仅扫描，不推已修Known Issues |
| [DeepEP #763](https://github.com/deepseek-ai/DeepEP/pull/763)及def8651固定README | T4深读V2.5拆分、移除旧依赖、框架责任和wait语义，E-13 | 9/29旧合入补录；10/15lint变化不入事实事件，未测吞吐/整训练 |
| [TileLang0.1.15](https://github.com/tile-ai/tilelang/releases/tag/v0.1.15)、[#3308](https://github.com/tile-ai/tilelang/pull/3308)、[固定Ascend指南](https://github.com/tile-ai/tilelang/blob/v0.1.15/tilelang/ascend/README.md)、#3059 | 定点机制已核：9/30 a35f8dd，Ascend950/dav-3510原生dialect不同于A2/A3社区backend；CUDA role-based warp specialization opt-in，tag用TL_ENABLE_AUTO_WARP_SPECIALIZATION | 不再标只读摘要；作者三算子/四shape图未逐数核，不用性能排名；main66后续提交未核 |
| [DeepGEMM CUDA #471](https://github.com/deepseek-ai/DeepGEMM/pull/471) | 10/14 1fc9c66、2.8.2 main，FP8CastWithClamp/clipped counter与paged stride候选，原PR正文已开 | 未完整diff/质量测量，不推训练收益或与Ascend组合 |
| [ROCm10.1](https://github.com/ROCm/TheRock/releases/tag/therock-10.1)、[固定notes](https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html)、[#8229](https://github.com/ROCm/TheRock/issues/8229) | 10/5 10:42正式发布为上轮漏扫补录；本次读resolved/known sections；fresh issue仍Open/updated9/15/无评论 | 未找到8229固定commit/正式修复归属；新页未列不等于修好。MI350 RCCL/MI200 memcpy等候选未深读 |
| [Triton](https://github.com/triton-lang/triton/releases)、[CUTLASS](https://github.com/NVIDIA/cutlass/releases)、[NCCL](https://github.com/NVIDIA/nccl/releases) | fresh releases仍3.8/4.8/C++2.32.3及nccl4py0.6，未重复旧卡 | 索引至10/19，不表示main无更新；cuTile PyPI本轮未查 |
| [Omni0.1.8](https://github.com/sgl-project/sglang-omni/releases/tag/v0.1.8)、[#2406](https://github.com/sgl-project/sglang-omni/pull/2406)、[#2293](https://github.com/sgl-project/sglang-omni/pull/2293) | T7深读10/12 18:22新release/Flow提交边界E-12；Qwen3-TTS延迟CI另核：最终只gate medians，p95/播放underrun打印而非gate | release/两PR至本次可见正文；非全部新模型/图优化/平台/安全路径。Voxt app仍与engine版本区分 |
| [Diffusers0.41](https://github.com/huggingface/diffusers/releases/tag/v0.41.0) | 正文highlights/compat读到QwenImage2.1、分片加载、SeaCache/混合FP8、ONNX弃用；网页latest仍0.41 | 已补发布正文，关联PR/具体质量与效率未核，不写模型组合完整支持 |
| 跨清单检索：agent缓存历史/质量与开放训练recipe | [EfficientAgent v1](https://arxiv.org/html/2609.33762v1)正文/附录与实现README深读E-10；[Request Order Matters v1](https://arxiv.org/html/2610.05833v1)方法/实验/消融深读E-11；发现midtraining交T1核验 | 两篇旧论文补录；实现commit/独立复测未完成。KV-Repurpose2610.13060、CacheScout2608.14624等仅发现，不据标题改判断 |

#### 下次断点与没有被推进的范围

1. **优先闭环**：SGLang0.5.22完整发布/修复路径；vLLM producer/consumer/注册三方与正式包；#60374最新validation；CVE fixed、ROCm #8229、AReaL receipt。HiSparse风险按E-05收窄，不能继续笼统称自然场景已确认损坏。
2. **T1/T2下一轮仍优先制品与质量—成本**：midtraining脚本/metadata和完整训练账本，verl最终head多seed/真实质量。slime128K example未证明学习收益；Curator控制了选择实验不等于控制了workflow速度。
3. **已读候选不夸大**：[OPEN-1B v1](https://arxiv.org/html/2609.17380v1)及[固定README](https://github.com/gensyn-ai/open-transformers/blob/d7b7b674cd851f7a16815229f060658180cb772b/README.md)已读，repop源码未公开、hash不含RNG/数据游标等，匹配不证明历史执行；仍待制品/audit。10/16 [SFC+KL作者实验](https://huggingface.co/blog/NormalUhr/sfc-kl)及[固定代码](https://github.com/NormalUhr/score-function-centering-trainer-generator-kl/tree/cbb64950aeaed2584b072f9b97602f2cc7a62d54)为单seed、best-heldout、不同eval间隔/部分seed不同，逐日志成本未核，不能当公平独立证明。
4. **旧来源保持旧截止**：PyTorch2.14.1 CUDA修复、Megatron/DeepSpeed/TE/TorchTitan，LMCache/Mooncake/TRTLLM/KTransformers，NVRx/TorchFT/Kueue，FlashInfer/FA4/Helion/TileGym/cuTile，UCCL/NVSHMEM独立文档，AITER/MLX/MaxText/SGL-JAX/UALink和Rubin/C2C/DPU/storage/Engram专门窗口本轮未重查。论文用旧LMCache/vLLM不表示当前项目release已核。
5. **90天/历史待查不清零**：8/24、8/31、9/7精确基线，BRACE、OAK日期异常、KernelBench-M、Ask the Tool、Lachesis数值范围、LoRA/streaming diffusion、Confidential Computing/MLPerf仍在队列。新release不能挤掉这些缺口；下轮轮换优先长期未深读的来源。
6. **访问与执行**：后段GitHub API多次403，改用正文/同SHA raw/patch的成功项已注明，未解决的正文缺口保留。sandbox仍因另一路径symlink启动失败，经限定授权目录的escalated read/apply_patch维护；未改配置/符号链接/自动化，未下载模型/视频、安装实验环境或执行GPU/付费/漏洞实验。

产物：13张证据卡、10/19周报八项、06当前判断及03/04/05/07/08/09机制解释。00本月导航仍有效；本轮不改月度复盘、冻结归档或历史周报段落。

验收：17份文档（含冻结基线）中448处内部链接可解析，无错误标题；13个既有概念占位保留，无新增断链。53张证据卡ID唯一，本轮13张；周报重点8项。此前40张证据卡正文哈希完全不变；10月周报10/19之前的正文哈希与运行前一致，9月周报、月度复盘、冻结基线和00/01/02/规则亦与运行前一致。冻结基线SHA-256仍为d712b9b8d167b68b790cefbfa924498ab7e7c4bba3d993f04a308c76054cc70f。围栏、表格alias与git diff --check通过。本次仅修改上述10份授权笔记。三名只读研究代理分别复核来源卡；另复核06、周报与运行记录的main/release日期、HiSparse反证边界和覆盖缺口，一致性通过。未执行独立GPU/安全/质量实验。
