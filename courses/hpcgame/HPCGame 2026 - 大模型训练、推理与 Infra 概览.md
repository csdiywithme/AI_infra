---
type: course-note
status: complete
course: HPCGame 2026 赛前讲座
lecture: 大模型训练、推理、Infra 概览
lecture_date: 2026-01-28
updated: 2026-08-24
area: systems
topics:
  - "[[Distributed Training]]"
  - "[[LLM Inference]]"
  - "[[GPU Kernel]]"
  - "[[Expert Parallelism]]"
  - "[[Context Parallelism]]"
  - "[[FlashAttention]]"
aliases:
  - HPCGame 2026 MLSys 概览
  - 大模型训练推理 Infra 概览
video_url: https://www.bilibili.com/video/BV1vmzXBBESY/
---

# HPCGame 2026：大模型训练、推理与 Infra 概览

> [!important] 已重组为可从任意问题进入的主题网络
> 这篇保留为完整的线性资料库；日常阅读请从 [[HPCGame 2026 - 00 MLSys 知识地图]] 进入。新结构拆为：[[HPCGame 2026 - 01 训练状态与并行|训练状态]]、[[HPCGame 2026 - 02 稀疏模型与数值格式|模型结构]]、[[HPCGame 2026 - 03 推理状态与调度|推理状态]]、[[HPCGame 2026 - 04 通信与内存层级|通信/内存]]、[[HPCGame 2026 - 05 Kernel、DSL 与硬件|Kernel/硬件]]、[[HPCGame 2026 - 06 前沿追踪|前沿追踪]]。

> [!info] 2026-09-15 追踪体系重构
> 持续研究从 [[HPCGame 2026 - 06 前沿追踪|当前判断]] 进入，并连接 [[HPCGame 2026 - 07 数据与后训练闭环]]、[[HPCGame 2026 - 08 集群效率与可靠性]]、[[HPCGame 2026 - 09 多模态与新工作负载]]。本长文是历史资料；已核勘误见 [[MLSys - 证据台账]]，运行范围见 [[MLSys - 运行记录]]。

> [!abstract] 一句话总纲
> 大模型系统不是一串孤立优化，而是 **数据与算法目标 → 并行分解 → 内存/通信代价 → 运行时调度 → 芯片与互连** 的连续 codesign；模型越稀疏、上下文越长、精度越低，理论 FLOPs 越不能代表真实成本，系统的主要矛盾越会转向数据搬运、状态管理和故障边界。

> [!summary] 我认为这场讲座最好的地方
> 它没有把 MLSys 讲成技术名词陈列，而是反复追问三个问题：**为什么需要这项设计、它把代价转移到了哪里、硬件/模型变化后结论还成立吗？** 这比记住“7D parallelism”“某库比某库快”更耐用。半年后的资料总体强化了这个框架，但也显示：库的内部机制、模型架构和硬件用途都可能迅速互换，应该学习 invariant，而不是给项目贴永久标签。

## 0. 来源、范围与阅读方法

- [讲座视频](https://www.bilibili.com/video/BV1vmzXBBESY/)：北京大学 Linux 俱乐部，发布于 2026-01-28，1:55:47。
- [公开 Slides（PKU 网盘）](https://disk.pku.edu.cn/link/AA79261A850FCC49EAAF63FC4EC4159211)。
- 正文依据用户提供的完整 SRT 字幕，并逐段与 41 页 Slides 交叉核对；字幕中的明显同音错误按上下文校正。
- “讲座笔记”忠实整理 2026-01-28 的内容；所有 **Follow-up** 均截至 **2026-08-24**，优先引用论文、官方文档、官方仓库或模型卡。
- 视频没有录入正式 Q&A，正文覆盖到 1:55:45。

更新标记：

- ✅ **强化**：半年后的实际发布/实现支持原判断；
- 🔄 **演进**：方向正确，但实现边界或主流形态已发生变化；
- ⚠️ **修正**：原说法需要增加重要条件；
- 🧪 **早期**：已有论文或实验实现，尚不足以视为稳定生产方案。

## 1. 全场逻辑地图

```text
能力来源
数据质量 / 配方 / 模型结构
        ↓ 决定计算图与训练稳定性
规模扩展
参数 × token × context length
        ↓ 单卡容量与时间不可接受
并行分解
DP/FSDP × TP/SP × PP × CP × EP
        ↓ 把“放不下”转化成通信、同步与调度问题
执行优化
FlashAttention / fused kernels / low precision / overlap
        ↓ 把瓶颈推向 HBM、网络、KV/activation state
推理运行时
batching / scheduling / prefix cache / speculative / P-D(-E) disaggregation
        ↓ 工作负载变成动态、长尾、故障敏感的分布式系统
基础设施
GPU + scale-up + scale-out + CPU/C2C + storage
        ↓
最终目标：在质量约束下优化 goodput、延迟、成本与可靠性，而非单点峰值 FLOPs
```

这条链上有三个反复出现的 invariant：

1. **优化不会消灭代价，只会移动代价。** MoE 减少每 token 激活参数，却引入路由、All-to-All 和负载均衡；量化减少字节和提升 tensor-core 峰值，却引入 scale、Q/DQ、layout 与收敛问题；PD 分离隔离阶段，却引入 KV 传输和跨池调度。
2. **内存是分层的，网络也是远端内存层级。** Register/SMEM/HBM、NVLink、RDMA、CPU DRAM、SSD 的共同问题是容量、带宽、延迟和可预测性不同。优秀设计是在正确时间把正确状态放到正确层。
3. **系统标签不是本质。** “训练库/推理库”“CPU-centric/GPU-centric”“vLLM/SGLang”“稠密/稀疏”都会融合；真正该比较的是 data path、control path、同步语义和支持的 workload。

## 2. 视频索引

| 时间 | 内容 | 对应笔记 |
| --- | --- | --- |
| [00:00](https://www.bilibili.com/video/BV1vmzXBBESY/?t=0) | MLSys 的边界、学习方法与 HPCGame 定位 | [[#3. 如何学习一个半年就会变样的领域]] |
| [04:14](https://www.bilibili.com/video/BV1vmzXBBESY/?t=254) | Data engineering 与公开训练复盘 | [[#4. 数据与配方：模型能力的第一层系统工程]] |
| [12:13](https://www.bilibili.com/video/BV1vmzXBBESY/?t=733) | Scaling 与 3D parallelism | [[#5. 并行训练：从约束推导组合，而不是数“几维”]] |
| [25:30](https://www.bilibili.com/video/BV1vmzXBBESY/?t=1530) | TP/SP、ZeRO/FSDP 与通信语义 | [[#5. 并行训练：从约束推导组合，而不是数“几维”]] |
| [36:05](https://www.bilibili.com/video/BV1vmzXBBESY/?t=2165) | MoE、EP 与 DeepEP/DeepGEMM | [[#6. MoE：省下稠密计算，换来动态通信]] |
| [43:26](https://www.bilibili.com/video/BV1vmzXBBESY/?t=2606) | 高效 Attention 的四条路线 | [[#7. Attention：精确算子优化正在让位于架构级混合]] |
| [48:25](https://www.bilibili.com/video/BV1vmzXBBESY/?t=2905) | 混合精度与模型技术报告 | [[#8. 低精度：峰值算力不是免费午餐]] |
| [55:08](https://www.bilibili.com/video/BV1vmzXBBESY/?t=3308) | 推理场景分层、边缘与异构 offload | [[#9. 推理系统：先辨认 workload，再选择优化]] |
| [1:03:03](https://www.bilibili.com/video/BV1vmzXBBESY/?t=3783) | vLLM/SGLang、continuous batching 与小规模 serving | [[#10. 单池推理：调度、缓存和 kernel 是一件事]] |
| [1:10:35](https://www.bilibili.com/video/BV1vmzXBBESY/?t=4235) | PD 分离、KV transfer 与大规模推理 | [[#11. 分离式推理：从 P/D 到 E/P/D 与分层 KV]] |
| [1:14:41](https://www.bilibili.com/video/BV1vmzXBBESY/?t=4481) | 硬件与网络总览 | [[#12. 通信：collective 正在变成可编程数据面]] |
| [1:22:40](https://www.bilibili.com/video/BV1vmzXBBESY/?t=4960) | NVSHMEM、IBRC、IBGDA | [[#12. 通信：collective 正在变成可编程数据面]] |
| [1:35:00](https://www.bilibili.com/video/BV1vmzXBBESY/?t=5700) | NCCL 新特性、NVLink 与 PCIe | [[#12. 通信：collective 正在变成可编程数据面]] |
| [1:41:07](https://www.bilibili.com/video/BV1vmzXBBESY/?t=6067) | Hopper/Blackwell/Rubin 与 GPU DSL | [[#13. GPU 与 kernel：优化对象从指令扩展到生成系统]] |
| [1:50:47](https://www.bilibili.com/video/BV1vmzXBBESY/?t=6647) | CPU、C2C、存储与 Engram | [[#14. CPU、存储与条件记忆：被重新纳入快路径]] |
| [1:54:26](https://www.bilibili.com/video/BV1vmzXBBESY/?t=6866) | 总结：codesign、memory hierarchy、持续学习 | [[#15. 讲座结论]] |

## 3. 如何学习一个半年就会变样的领域

讲者开场先做了一个很重要的认识论约束：这是个人视角下的 Infra 概览，不可能覆盖完整 MLSys。这个领域“入门容易、精通困难”，原因不是材料少，而是：

- 模型结构、硬件、通信库和 fused operator 都在并行迭代；
- 很多系统设计不是定理，而是对某个硬件、规模和 workload 的局部最优；
- 一项经典设计值得学的是它面对的矛盾和约束，而不是把实现当永久模板。

因此阅读系统论文/仓库时，应该固定问：

1. 它优化的目标是 latency、throughput、goodput、capacity、cost 还是可维护性？
2. 基线的硬件、shape、精度、batch 和网络是什么？
3. 它减少了 compute、memory traffic、communication volume，还是只做了 overlap？
4. 新增了什么状态、同步、故障域和调参维度？
5. 换一代 GPU、换模型架构或换集群拓扑后，结论是否反转？

> [!tip] 讲座对 HPCGame 的定位
> 竞赛价值不在“题越难越好”，而在用可控问题传递性能直觉：发现瓶颈、建立下界、理解硬件、验证优化。这个思路同样适合学习真实 MLSys。

## 4. 数据与配方：模型能力的第一层系统工程

### 4.1 为什么从 data engineering 开始

讲座没有从 GPU 开始，而是从训练数据开始，因为 Infra 的目标不是把 FLOPs 跑满，而是把训练预算转化为能力。参数、token、context length 都能 scaling，但质量取决于：

- 数据获取、清洗、去重、过滤和 domain mixing；
- 不同训练阶段的 curriculum 与 repetition；
- learning rate、context length、batch 等配方如何与数据共同变化；
- 评测是否泄漏或被针对性优化。

公开报告中最有价值的经常不是最终 recipe，而是 **失败尝试、消融和为什么失败**。不同团队得出冲突结论并不奇怪：数据分布、模型大小、学习率和上下文阶段不同，局部最优不能直接移植。

讲者推荐的端到端材料包括 Kaiyuan-2B、NVIDIA Nemotron 3、EvoLM 与 Smol Training Playbook。它们共同说明：高质量数据与 recipe 仍是核心壁垒，benchmark 分数不能只归因于模型结构或 Infra。

### 4.2 半年后：开放训练资产继续扩大，但“完全可复现”仍有层级

✅ **Kaiyuan-2B** 的 [论文](https://arxiv.org/abs/2512.07612) 继续提供 Quantile Data Benchmarking、Selective Repetition、Multi-Domain Curriculum 等完整方法与 Apache-2.0 资产；截至本次核查，论文仍是 2025-12 的 v1。

✅ **EvoLM** 的 [论文与资产说明](https://arxiv.org/abs/2506.16029) 已扩展为 v2，覆盖 100 余个 1B/4B 模型，以及 PT、CPT、SFT、RL 的数据与流水线。它的价值是把“一个最终 checkpoint”变成可分析的训练轨迹。

✅ Hugging Face 的 [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) 公布了 3B 模型约 11T token 的 staged mixture、4K→32K→64K 长上下文课程、失败实验与算力账本，是“公开负结果”这一主张的强例子。

✅/⚠️ Nemotron 3 的开放度进一步提高：[Super 120B-A12B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) 于 2026-03 发布，[Ultra 550B-A55B](https://research.nvidia.com/labs/nemotron/Nemotron-3-Ultra/) 于 2026-06 发布，并提供预训练/后训练数据与 NeMo Gym 环境线索。但“公开大量数据与配方”仍不等于复现原始组织内部所有数据、过滤版本、失败 runs 和基础设施细节。

> [!warning] 更新结论
> 开放训练的趋势比 1 月更强，但讲者对 **proprietary data/recipe 仍是 moat** 的判断没有失效。应该把 openness 分成：权重、代码、数据清单、实际数据、训练日志、失败实验、精确环境与可重跑脚本，而不是用一个“open”标签概括。

## 5. 并行训练：从约束推导组合，而不是数“几维”

### 5.1 Scaling 为什么必然进入分布式

训练总资源大致随三条轴增长：

```text
更多参数 → model states / 每 token compute 增长
更多 token → 总训练步数与数据管线压力增长
更长 context → activation / attention compute / KV-related state 增长
```

单卡放不下或跑不完后，并行不是目的，而是把计算图切到硬件拓扑上。经典 3D parallelism 是：

- **DP / ZeRO / FSDP**：沿 batch 与 model states 的 ownership 切；
- **TP + SP**：沿 hidden/heads 与 sequence activation 切；
- **PP**：沿 layers/depth 切；
- 现代稀疏和长上下文再加入 **EP** 与 **CP**。

把它叫“5D/7D”没有多少信息量。配置的本质是：

$$
N_{GPU}=N_{DP}\times N_{TP}\times N_{PP}\times N_{CP}\times N_{EP}\quad
\text{（实际 process group 可重叠或折叠）}
$$

关键约束是每个 rank 是否放得下、collective 是否能在对应拓扑跑动、local GEMM 是否仍足够大，以及 global batch 是否合理。

### 5.2 PP：沿深度切，支付 bubble 与 activation

GPipe 把 layers 切成 stages，再把 batch 切成 microbatches。理想流水必须解决：

- fill/drain bubble；
- stage imbalance；
- forward activation 的保存或 recompute；
- forward/backward 次序和权重版本一致性。

1F1B / PipeDream 类 schedule 通过交错 forward/backward 降低峰值 activation；DualPipe 等进一步尝试双向重叠。不要只看 bubble 公式：现实还受 kernel 粒度、跨机链路和 MoE stage 波动影响。

### 5.3 TP、SP 与 CP 不是同一种 sequence 切分

Tensor Parallel 把线性层权重沿 column/row 切分，利用等价的矩阵分块维持计算语义。它通常需要在层内频繁 collective，所以更适合 NVLink/NVSwitch 等高速域。

Megatron 的 **Sequence Parallel** 与 TP 配合，把原本 replicated 的 non-TP activations 沿 sequence 切开，并把某些 All-Reduce 改写为 Reduce-Scatter + All-Gather，主要目标是 activation memory。

**Context Parallel** 则在 sequence 维把整层 activation 和 attention context 分布到多个 rank，面向真正长上下文。它需要 attention 的 KV/partial result 通信，算法和拓扑选择更复杂。官方 [Megatron Context Parallel 文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) 也明确区分了二者。

### 5.4 ZeRO/FSDP：不改变 DP 数学，改变状态 ownership

ZeRO 的思想是把本来每卡 replicated 的 optimizer state、gradient、parameter 逐级 shard，并在需要时用 Reduce-Scatter / All-Gather 恢复局部视图。省下的是静态显存，代价是：

- 参数 materialization 的临时峰值；
- 更细碎、更频繁的通信；
- prefetch、overlap、reshard policy 与 allocator 碎片；
- checkpoint 和 state-dict 语义变复杂。

🔄 半年后 PyTorch 的 FSDP2 继续强化 per-parameter DTensor sharding；[`fully_shard`](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html) 已是组合式 API。PyTorch 2.13 又加入可选的独立 process group 来重叠 All-Gather 与 Reduce-Scatter（[官方 release blog](https://pytorch.org/blog/pytorch-2-13-release-blog/)）。这说明 FSDP 的重点从“能 shard”转向 **如何调度 materialization 与通信**。

### 5.5 Megatron 的地位与代价

讲者把 Megatron 视作工业训练的基线，同时提醒：深度融合的并行和优化会增加算法扩展与维护成本。这一判断半年后更明显。

✅ 当前 [Megatron MoE 指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html) 已组合 TP/SP/PP/CP/EP/DP，支持 MLA、MTP、DeepEP、HybridEP、Qwen3-Next 与 DeepSeek-V3.2，并用 **Parallel Folding** 将 attention 的 TP/CP/DP 与 expert 的 ETP/EP/EDP 解耦。

🔄 这不是“维度越多越先进”，而是不同子图拥有不同最优 mesh。Parallel Folding 正好验证讲座的核心方法：**从子图的通信与容量约束出发，再建立 process groups**。

## 6. MoE：省下稠密计算，换来动态通信

### 6.1 为什么 MoE 成为主流

Dense 模型每 token 激活全部 FFN 参数。MoE 用 router 只选择少数 experts，使总参数/容量增长快于每 token FLOPs。DeepSeek-V3 在 2024 年末显著加速了这一架构的普及。

但稀疏性把规则 GEMM 变成动态系统问题：

```text
tokens
  → routing / top-k
  → dispatch（All-to-All / All-to-Allv）
  → grouped expert GEMM
  → combine
```

核心难点包括负载均衡、shared experts、跨节点不均匀流量、padding/packing、通信与 GEMM overlap，以及 reduction 顺序带来的浮点差异。分布式变换必须尽量保持原算法语义，不能把“近似等价”默认为收敛等价。

### 6.2 DeepEP / DeepGEMM 的意义

DeepEP 与 DeepGEMM 展示了模型、GPU（H800）、Mellanox 网络与低精度 kernel 的极致 codesign。但这种性能具有硬件和 shape 假设，移植到 B300、EFA、AMD 或不同 hidden size，需要重新验证。

🔄 **最重要的半年更新**：[DeepEP V2](https://github.com/deepseek-ai/DeepEP) 已重构，并从 NVSHMEM 路径转向 **NCCL GIN**，增加 JIT、ElasticBuffer、EP2048、低 SM 占用与实验性的 0-SM Engram/PP/CP 通信。这不是原判断被否定，而是说明“GPU-centric”正在被 NCCL 吸收，库名不再等于固定 control/data path。

✅ [UCCL EP](https://github.com/uccl-project/uccl/blob/main/ep/README.md) 以 DeepEP 兼容接口扩展到 NVIDIA/AMD、EFA/Broadcom/ConnectX-7，说明异构移植已经有真实进展。

⚠️ 生产成熟度仍不能只看 benchmark：DeepEP 在 B300、JIT/NFS、特定 hidden size 上仍出现公开问题（例如 [B300 illegal access](https://github.com/deepseek-ai/DeepEP/issues/622)），UCCL 也有 [EFA EP16 间歇崩溃](https://github.com/uccl-project/uccl/issues/878) 报告。讲者所谓“工程地狱”并未消失，只是从“能否实现”转向“能否跨形状、硬件和故障稳定运行”。

## 7. Attention：精确算子优化正在让位于架构级混合

### 7.1 两个不同层次的问题

标准 attention 的序列长度复杂度为 $O(n^2)$。优化可以发生在两个层次：

1. **保持精确 attention 语义，优化 IO 与并行映射**：FlashAttention 系列；
2. **改变模型结构，减少必须计算/保存的状态**：GQA/MLA、稀疏 attention、线性/递归 attention、hybrid architecture。

FlashAttention 的核心不是近似，而是 tiling、online softmax 和 recomputation：减少 HBM 往返，把中间矩阵留在片上。它高度绑定 GPU 代际，因为可用的 asynchronous copy、tensor core、cluster 与 memory hierarchy 不断变化。

讲座把架构路线分成四类：

| 路线 | 代表 | 主要减少什么 | 新代价 |
| --- | --- | --- | --- |
| 量化 attention | SageAttention | QK/PV 的字节和低精度 compute | scale、精度与 kernel 适配 |
| compact KV | GQA、MLA | KV cache 容量/带宽 | 表达约束、特殊解码 kernel |
| sparse attention | DSA | 被计算的 token pairs | 稀疏索引、热度预测、负载不规则 |
| linear/recurrent | KDA、Mamba、RWKV | 避免完整 $n^2$ 与 KV | 新状态、训练稳定性、生态支持 |

讲者特别看好 composable/hybrid：不同 layer/token pattern 用不同 attention，而不是寻找一个万能替代。

### 7.2 半年后：hybrid 已从“前沿方向”进入旗舰模型

✅ [Kimi K3 模型卡](https://huggingface.co/moonshotai/Kimi-K3) 公布 2.8T total / 104B active、69 层 KDA + 24 层 Gated MLA、1M context 和原生视觉；[官方代码与报告](https://github.com/MoonshotAI/Kimi-K3) 随模型开放。这是“线性/递归 + full/compact attention”已经进入超大规模模型的直接证据。

✅ DeepSeek V4 已于 2026-04 发布；[官方透明度页面与 technical report](https://www.deepseek.com/en/transparency/) 提供模型信息。其公开 serving 支持显示 hybrid sparse attention 已成为运行时必须处理的现实，而非论文特例。

✅ 精确 attention kernel 也没有停止演进：[FlashAttention-4](https://arxiv.org/abs/2603.05451) 针对 Blackwell/B200 重写，在论文设定下报告最高相对 cuDNN 9.13 的 1.3×、相对 Triton 的 2.7×，并用 CuTe DSL 缩短开发/编译路径。数字是作者在特定 shape/硬件上的结果，不应跨代照搬；真正的结论是 **kernel 仍必须随 memory hierarchy 重写**。

🧪 稀疏 attention 的计算量下降不等于状态容量问题消失。[SGLang HiSparse](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/) 仍需 GPU hot buffer + host memory 的分层管理，说明 sparse pattern、KV placement 与 serving scheduler 必须共同设计。

> [!tip] 更新后的心智模型
> 未来引擎不只保存一种“KV cache”。它可能同时管理 MLA latent、recurrent state、稀疏索引、视觉/encoder state 和传统 KV。attention 优化正在从一个 kernel 问题变成 **多种状态的统一生命周期管理问题**。

## 8. 低精度：峰值算力不是免费午餐

### 8.1 理论峰值为什么不等于端到端收益

从 FP16/BF16 到 FP8，再到 FP4，tensor-core 峰值可能成倍增加。但端到端速度还取决于：

- quantize/dequantize 与 scale 读取是否被 fusion；
- block/tensor/channel scaling 与 tensor layout；
- accumulation dtype、stochastic rounding 和异常值处理；
- optimizer/gradient/activation/weight 各自用什么精度；
- 通信 payload 是否同时变小；
- 模型规模和训练 recipe 是否能保持收敛。

因此 mixed precision 不是 `dtype=` 一个开关，而是一张包含多种 dtype 与 scale tensor 的数据流图。

### 8.2 半年后：NVFP4 已有大规模证据，但通用 recipe 仍在形成

✅ NVIDIA [Transformer Engine NVFP4 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html) 给出 E2M1 value、16-value block FP8 scale、FP32 global scale、gradient stochastic rounding、Hadamard transform 和 2D weight scaling 等机制，支持 SM100+ 训练。

✅ Nemotron 3 Ultra 提供了大模型 NVFP4 预训练证据，不再只是小规模 simulation。[Ultra NVFP4 模型卡](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4) 说明其 20T token 预训练和 550B/55B active 配置。

⚠️ 厂商数字应先看统计口径：[Transformer Engine benchmark](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/speedups.html) 在 5B/B300 的 **每层 12 个 linear GEMM + 当步量化** 设置中，给出 NVFP4 autocast 相对 BF16 约 2.03×；预量化输入的 raw GEMM 则为 3.55×。这个对比直接量出了 Q/DQ 代价，但还不是包含 attention、optimizer、通信和 dataloader 的完整训练 step；真实端到端收益还会受更多 Amdahl 项限制。

🧪 [Full-stack FP4 Training](https://arxiv.org/abs/2607.04422) 等工作继续向 optimizer/gradient/communication 扩展，但发布时间很新，尚不能据此认为跨模型、跨硬件的 FP4 recipe 已收敛。

## 9. 推理系统：先辨认 workload，再选择优化

讲座把推理分成四个层级，这比“哪个引擎最快”更有用：

1. **端侧/资源受限**：隐私、功耗、内存最重要；常见是小模型、低 bit、CPU/NPU/GPU 异构；
2. **一两台机器 colocated**：prefill 和 decode 共用设备，重点是 continuous batching、cache、kernel 和单实例调度；
3. **数据中心 scale-out**：P/D 分离、大 EP、KV 池化、跨节点路由和可靠性成为主问题；
4. **多模态/agent**：encoder、tool calls、长寿命 session 与多轮复用引入新的状态和调度事件。

任何推理优化都应先写清四个量：

- **TTFT**：首 token 延迟，主要受 queueing + prefill 影响；
- **TPOT / ITL**：输出 token 间延迟，主要受 decode 批次和调度影响；
- **Throughput**：单位时间完成的 token/request；
- **Goodput**：满足 SLO 的有效吞吐，不能用大量违约请求堆出来。

### 9.1 端侧与异构 offload

llama.cpp / Ollama 代表模型量化与本地执行；KTransformers 则把大 MoE 模型的 experts 放入 CPU DRAM，只把 attention/热路径放 GPU，并利用 Intel AMX 等 CPU 指令。这不是“24GB 显卡装下完整模型”，而是把大容量冷参数留在主存，支付 CPU compute、PCIe/NUMA 与 RAM 容量。

✅ 半年后 [KTransformers](https://github.com/kvcache-ai/ktransformers) 已继续支持 DeepSeek V4 Flash、CPU-GPU expert scheduling、BF16/FP8、SGLang 集成以及 GPU/CPU/disk 三层 prefix cache，方向已从一次性 demo 发展为专门化运行时。

⚠️ 它仍不是轻量通用方案。官方 [安装说明](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/install.md) 给出的 DeepSeek-R1 Q4 例子约需 382GB DRAM 和 14GB VRAM。瓶颈只是从 GPU 容量移到了主存容量、带宽与平台调优。

## 10. 单池推理：调度、缓存和 kernel 是一件事

### 10.1 vLLM 与 SGLang 的起点

- vLLM 以 **PagedAttention** 起家：把 KV cache 按 blocks 管理，解决连续显存分配、碎片和共享问题；
- SGLang 以 **RadixAttention** 起家：用 radix tree 复用共享 prefix，服务结构化/多轮 program。

到 2026 年，两者都已经吸收 continuous batching、prefix cache、speculative decoding、quantization、chunked prefill、CUDA Graph、分布式 serving 等能力。历史标签仍有助于理解设计基因，但不能代替当前比较。

🔄 当前更准确的差异是：vLLM V1 用 hash-based KV blocks 做 automatic prefix caching，并形成广泛的平台/API/KV Connector 生态（[设计文档](https://docs.vllm.ai/en/latest/design/prefix_caching/)）；SGLang 继续围绕 Radix/HiCache/UnifiedRadix、day-0 model/kernel 集成和 P/D/EPD 构建系统。这个差异仍是快照，不是永久边界。

### 10.2 核心优化如何连接

**Continuous batching** 在 token step 边界插入/移除请求，提高 decode batch 利用率；代价是 scheduler overhead 和延迟干扰。

**Chunked prefill** 把长 prompt 分块，让 prefill 与 decode 交错，缓解 TTFT/TPOT 冲突；chunk 太小则 launch/scheduling 开销增大。

**Speculative decoding / MTP** 用 draft 或模型额外 heads 一次提出多个 tokens，再由 target 验证。收益依赖 acceptance rate、额外显存和 workload；vLLM 的 [官方文档](https://docs.vllm.ai/en/latest/features/spec_decode/) 也将主要适用场景定位在 medium/low-QPS 的 memory-bound workload，而不是所有高吞吐 serving。

**CUDA Graph / megakernel** 减少 host launch 与碎片化 kernel 开销，但要求 shape/control flow 更稳定，并可能增加编译和 graph cache 管理成本。

**Fine-grained overlap**（如 Flux/Comet 思路）把通信与计算切成细粒度流水；只有两者资源占用可兼容、依赖正确且 tail 可覆盖时，overlap 才会转化为 wall-clock 收益。

### 10.3 模型支持已经变成核心竞争力

截至 2026-08，[vLLM releases](https://github.com/vllm-project/vllm/releases) 已进入 0.27.x 并加入完整 Kimi K3 支持；[SGLang releases](https://github.com/sgl-project/sglang/releases) 已到 0.5.18。版本号本身不值得背，重要的是旗舰模型包含 hybrid attention、MoE、低精度与多模态后，运行时必须 day-0 支持新的 state type 和 kernel，而不再只是实现标准 Transformer decode。

## 11. 分离式推理：从 P/D 到 E/P/D 与分层 KV

### 11.1 为什么拆分 prefill 与 decode

Prefill 通常是大 GEMM、compute-heavy；decode 每步小、反复读权重/KV，更常 memory/latency-sensitive。放在同一 GPU 池会互相干扰，于是 DistServe 等工作把 P 与 D 放在不同 workers：

```text
request → prefill pool → KV transfer → decode pool → stream output
```

系统因此新增：KV transfer engine、P:D 容量比例、admission control、跨池 backpressure、KV-aware routing 和独立故障域。真正目标是用资源隔离改善 goodput/SLO，而不只是平均 token/s。

⚠️ **重要修正**：[vLLM disaggregated prefill 文档](https://docs.vllm.ai/en/latest/features/disagg_prefill/) 明确写明它本身 **不提升 throughput**；它主要用于独立调节 TTFT/ITL、隔离 tail latency。KV 传输和跨池排队完全可能抵消收益。讲座对 PD 价值的方向正确，但必须加上这条边界。

### 11.2 KV cache 变成独立数据系统

Mooncake、FlexKV、LMCache、HiCache 等把 KV 从单引擎私有显存提升为可跨层级、跨节点传输和复用的对象。此时需要考虑：

- block identity、hash/collision 与 tenant 隔离；
- GPU/CPU/remote storage 的 eviction、prefetch 与一致性；
- transfer 与 compute overlap；
- prefix 热度、session locality 与 cache-aware routing；
- worker 失败后 cache 元数据如何恢复。

✅ vLLM 当前文档列出 NIXL、Mooncake、LMCache、FlexKV 等多个 KV Connectors；[LMCache 分离示例](https://docs.vllm.ai/en/latest/examples/disaggregated/lmcache/) 展示了外置 connector 的接入方式。

✅ SGLang [HiCache 最佳实践](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_best_practices.mdx) 已覆盖 GPU/CPU/storage 三层与 HF3FS、Mooncake、NIXL、AIBrix 等后端，并能与 PD 组合。

⚠️ “有三层 cache”不等于生产可靠：公开问题中已有 [flat file directory 导致 ENOSPC](https://github.com/sgl-project/sglang/issues/28653) 以及 PD admission freeze 一类故障。这验证了讲者的判断：推理 infra 的难点会落在文件系统元数据、backpressure、版本组合和故障恢复等看似“不前沿”的位置。

### 11.3 从 PD 到 EPD、multimodal 与 agent-aware routing

✅ SGLang 已公开 [EPD disaggregation](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/epd_disaggregation.md)：encoder、prefill、decode 各自成为资源池。vLLM 也提供 [disaggregated encoder](https://docs.vllm.ai/en/stable/features/disagg_encoder/) 的 E→PD / E→P→D 拓扑。

✅ [NVIDIA Dynamo releases](https://github.com/ai-dynamo/dynamo/releases) 将分离式 serving、KV 管理、多模态/omni、agent priority/cache retention 等整合到生产导向运行时。这说明讲座预判的“未来是 multimodal/agent infra”已开始具象化。

> [!warning] 大规模推理的真实调参面
> P:D:E 比例、EP size、KV placement、prefix locality、失败重试、软件版本和网络拓扑彼此耦合。单个 kernel 快 20% 可能被一次 cache miss、一次跨池拥塞或一个冻结的 admission controller 完全抹掉。

## 12. 通信：collective 正在变成可编程数据面

### 12.1 网络同样是 memory hierarchy

讲座从 EFA / ConnectX 等网卡讲到 NCCL、NIXL、NVSHMEM、DeepEP/UCCL。理解它们不应从 API 名字开始，而应拆成：

- **transport**：PCIe、NVLink、InfiniBand/RoCE/EFA；
- **operation**：collective、P2P、one-sided put/get/atomic；
- **control path**：CPU 发起、GPU 发起，或 NIC/offload engine 参与；
- **data path**：数据从哪块 memory 到哪块 memory，是否 staging；
- **progress model**：谁推进 WQ、doorbell、completion 与重试。

传统 NCCL 被概括为 CPU-centric collective；NVSHMEM/IBGDA 则让 GPU 直接发起通信。这个分类帮助理解 latency 和 overlap，但不再是库的永久边界。

### 12.2 NVSHMEM、IBRC 与 IBGDA

NVSHMEM 提供 PGAS 风格的 symmetric heap、one-sided put/get/atomics 和 device-side synchronization，让 kernel 能在 GPU 内部发通信并与计算细粒度交错。

传统 IBRC 路径常需 CPU proxy 维护发送队列和 doorbell；IBGDA 把 WQ、DBR、completion 等控制对象暴露给 GPU，减少 CPU round trip。收益主要出现在小消息、动态 EP 或 kernel 内通信，但会增加 memory ordering、资源管理和兼容性复杂度。

讲者由此追问 DPU 的位置：如果 GPU 能直接驱动 NIC，DPU 是否仍必要？更准确的回答是：它不一定是 fast-path 发起所必需，但仍可承担虚拟化、隔离、安全、storage/network services、拥塞控制和 elastic orchestration。问题应是“哪些控制/数据面功能值得 offload”，而不是 DPU 有/无二选一。

### 12.3 NCCL 的演进：从固定 collective 库到更通用通信运行时

讲座列举了传统 NCCL 的若干限制：初始化/拓扑发现重、通信域偏静态、小消息延迟、SM 占用与同步模型受限；并跟进 2.27/2.28/2.29 的 SHARP、shrink/grow、symmetric memory、GIN、copy engine、Inspector、one-sided 和 NCCL4Py。

✅ 截至 2026-08，官方最新 release line 已到 NCCL 2.31.x。[2.31.2 release notes](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/release-notes/rel_2-31-2.html) 增加/强化了：

- **Compute Fabric Transport（CFT）** 的 host/device APIs，可注册 window memory 并从 device 发起 Put/Get/Red/NVLS；
- per-collective configuration 与 tuning；
- GIN 的 EFA GDA backend 与 device-side timeout；
- one-sided 多 context / multi-NIC；
- hierarchical 0-SM AllGather / AlltoAll；
- TMA cost model、CuTe DSL bindings 与更多 diagnostics。

这说明“GPU-initiated、低/零 SM、one-sided、可调 collective”正在进入 NCCL 主线。DeepEP V2 改用 NCCL GIN 是最强的交叉验证：原本看似对立的两条栈正在融合。

⚠️ release notes 同时列出 PAT 在 H100/B40 等组合上的 regression/known issues。讲者说“新功能可能有 bug，要动态评估”依然准确；通信库的 feature matrix 必须绑定版本、NIC、driver、CUDA 和 topology。

✅ [NVSHMEM 3.7.2 release notes](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html) 显示其已覆盖 Ampere/Hopper/Blackwell 与多代 CUDA，但依然保留兼容性、ABI 和平台限制。它不会因 NCCL 支持 GIN 就消失，更可能成为不同抽象层的互补。

### 12.4 Scale-up 标准：NVLink 不再是唯一叙事，但生态仍未收敛

NVLink/NVSwitch 解决机内/机架高速 scale-up；InfiniBand/RoCE/EFA 解决更大规模 scale-out。PCIe 与 NVLink 带宽/语义的鸿沟决定 TP、细粒度 EP 和 C2C offload 的可行域。

✅ [UALink 2.0 官方公告](https://ualinkconsortium.org/wp-content/uploads/2026/04/UALink-2.0-Specification-PR_FINAL.pdf) 于 2026-04 公布 Common 2.0、200G data/link layer 2.0、manageability 1.0 与 chiplet 1.0，并加入 in-network compute。标准进度比 1 月明显推进。

⚠️ specification 发布不等于大规模 interoperable 部署已经成熟。交换芯片、accelerator、线缆、管理栈、collective 库与认证计划仍要共同落地，所以讲座所说“scale-up 尚未收敛”只能改成 **标准正在收敛，生态与部署仍在形成**。

## 13. GPU 与 kernel：优化对象从指令扩展到生成系统

### 13.1 为什么每代 GPU 都会重写热点 kernel

讲座沿 Hopper→Blackwell→Rubin 观察到一个长期趋势：低精度 tensor-core compute 增长快于数据移动能力。于是瓶颈不断推向 register/SMEM/HBM/互连，硬件加入新的搬运和协作机制：

- Hopper：TMA、thread-block cluster/CGA、distributed shared memory、WGMMA；
- Blackwell：更低精度、tensor memory 与新 tensor-core data path；
- Rubin：更大 scale-up、C2C 与 activation sparsity 等平台级设计。

同一个数学算子在不同代 GPU 上的最佳 tile、pipeline stage、warp role 和 synchronization 可能不同。这正是 FlashAttention 一代代重写，而不是“一次优化永久通用”的原因。

### 13.2 GPU DSL 的真正 trade-off

讲者横向提到 CUTLASS/CuTe、Triton、TileLang、CuTe DSL、cuTile，以及 AMD/分布式方向的 DSL。比较它们不应只看代码行数：

| 维度 | 问题 |
| --- | --- |
| 表达能力 | 能否表达异步 pipeline、tensor-core layout、cluster、distributed op？ |
| 性能上限 | 是否能控制到达到目标硬件上限所需的细节？ |
| 开发速度 | 编译、autotune、debug、profile 的反馈周期多长？ |
| 可移植性 | 换 GPU vendor/代际后复用的是源码、抽象还是只有算法思想？ |
| 生态 | 能否进入 PyTorch/serving graph，处理动态 shape 和 fallback？ |

✅ CUDA 13.2 已正式扩展 [CUDA Tile](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/) 到 Ampere/Ada/Blackwell 并支持 pip 安装；CUDA 13.3 又发布 [C++ CUDA Tile](https://developer.nvidia.com/blog/develop-high-performance-gpu-kernels-in-c-with-nvidia-cuda-tile/)。tile abstraction 已从实验方向进入官方工具链。

✅ FlashAttention-4 使用 CuTe DSL 并取得接近硬件上限的结果，说明“更高层 DSL 必然牺牲顶级性能”不再成立；但它依然需要专家级 layout、pipeline 与硬件知识，DSL 只是缩短表达和编译路径。

### 13.3 2:4 structured sparsity：原批评需要细分用途

讲者不看好 2:4 稀疏的通用性：静态权重必须每 4 个元素恰有 2 个为零，模型训练、压缩和 kernel 都受约束，很多 workload 难以真正吃满宣传峰值。这对 **通用静态 weight sparsity** 仍成立。

⚠️ 但 Rubin 展示了新的使用方式。[NVIDIA Rubin 架构说明](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/) 把 2:4 用于 attention 的中间 activation，压缩 softmax 后数据，减少第二个 attention GEMM 及数据移动。也就是说，2:4 没有成为万能权重压缩，却可能作为硬件友好的 **动态中间表示** 获得新生命。

> [!tip] 修正后的结论
> 不要问“2:4 有没有用”，要问：稀疏对象是 weight、activation 还是 routing result？零值如何产生？是否支付 sorting/pruning/metadata？上下游是否能保持压缩格式？

### 13.4 AI 生成 kernel：进展巨大，但 generality 仍未解决

讲座在 1 月的判断是：AI kernel generation 很有潜力，但通用性与稳定性能尚不足。半年后的跟进呈现两面性：

✅ [CUDA Agent](https://arxiv.org/abs/2602.24286)、[DRTriton](https://arxiv.org/abs/2603.21465)、[A-TREX](https://github.com/alibaba/atrex-kernel-agent) 等已从单次代码生成推进到 profile-feedback、搜索/RL 和执行验证；研究问题从“能不能写 Triton”转向“能否持续逼近硬件上限”。

✅ [SOL-ExecBench](https://arxiv.org/abs/2603.19173) 尝试用硬件性能上限而不是普通 library baseline 评估生成 kernel，benchmark 本身开始更贴近真实优化目标。

⚠️ [UCCL CommBench](https://github.com/uccl-project/CommBench) 反而说明分布式通信 kernel 仍难：动态 world size、并发、网络故障和跨硬件语义远比单 GPU elementwise/GEMM 复杂。其公开榜单的 pass/good-performance 比例仍远未饱和。

因此原判断应更新为：**单 GPU、可执行验证、形状明确的 kernel 生成已非常有竞争力；跨 shape、跨硬件、跨通信语义并能长期维护的生产级 generality 仍是开放问题。**

## 14. CPU、存储与条件记忆：被重新纳入快路径

### 14.1 NVLink-C2C 改变 CPU/GPU 分工

传统 CPU offload 经 PCIe，带宽与延迟很难被隐藏；Grace Hopper / GB 系列以及 Vera Rubin 的 coherent C2C 提高 CPU-GPU 之间的可用带宽，使大容量 DRAM 不只是“最后兜底”，而可能参与 KV、expert、embedding、data preprocessing 与 checkpoint path。

✅ [Vera Rubin NVL72 官方页面](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) 将 72 Rubin GPU、36 Vera CPU、NVLink 6、ConnectX-9 与 BlueField-4 作为一个 rack-scale 平台，并给出每 superchip 1.8 TB/s C2C、每 GPU 3.6 TB/s scale-up 等厂商规格。

⚠️ 这些是 vendor platform claims / rollout 信息，不是任意 workload 的实测可用带宽。NUMA、coherence traffic、page placement、software support 和并发争用仍决定 offload 能否隐藏。

### 14.2 为什么 serving 的 storage 比 training 更难隐藏

训练数据访问通常可预测、顺序且可 prefetch；在线 serving 的 prefix/KV/session/embedding 访问由用户请求驱动，长尾和随机性更强。容量层扩展到 SSD/remote storage 后，必须面对：

- metadata 与小文件压力；
- 热度预测和 admission；
- prefetch 命中率；
- failure/replay 与多租户隔离；
- storage bandwidth 是否能被 compute window 覆盖。

因此“把状态放到更慢层”从来不是免费容量，只是在延迟预算允许时做 tiering。

### 14.3 Engram：条件记忆把 embedding table 重新带回模型

[DeepSeek Engram](https://arxiv.org/abs/2601.07372) 用条件记忆/大表查找补充神经网络计算，适合把部分知识容量放入可寻址 memory。其系统前提是 lookup、prefetch/offload 可以被主干 compute 隐藏，否则随机访存会成为新瓶颈。

🔄 1 月时它主要是论文方向；现在 DeepEP V2 已加入实验性的 **0-SM RDMA Engram communication**，说明网络/运行时正在为这种模型结构补数据面支持。

🧪 后续论文又探索 [CXL memory pooling](https://arxiv.org/abs/2603.10087)、[memory grafting](https://arxiv.org/abs/2605.20948) 与 [tokenizer-agnostic Engram](https://arxiv.org/abs/2607.29065)。这些支持“条件记忆会扩展 memory hierarchy”的研究趋势，但还不能视为大规模生产验证。

### 14.4 BlueField-4 的问题仍需开放看待

Rubin 平台继续配置 BlueField-4，说明 NVIDIA 并未认为 GPU-initiated networking 会消灭 DPU。更合理的分工是：

```text
GPU：极低延迟的数据面发起、kernel 内通信
NIC：RDMA / transport / congestion primitives
DPU：隔离、安全、虚拟化、storage/network services、弹性控制
CPU：全局调度、复杂控制面与异常路径
```

具体 workload 是否需要 DPU，仍应以端到端 profile 和运维边界回答，而不是从产品拓扑反推必然性。

## 15. 讲座结论

讲者最后收束为三点，半年后仍然成立：

1. **Extreme codesign**：模型架构、并行策略、kernel、通信库、GPU/CPU/NIC/存储必须一起优化；局部峰值不等于系统 goodput。
2. **Memory hierarchy 是主线**：attention、MoE、低精度、KV cache、C2C 和 Engram 看似不同，本质都在决定状态的容量、布局、搬运与复用。
3. **持续学习比技术站队重要**：硬件设计会演进，库会吸收竞争范式，经典方案会被重构。应保留问题分解和性能推理能力。

我的压缩版是：

> **MLSys 的核心工作，是在正确的语义与质量约束下，减少不可隐藏的数据移动；无法减少的部分，就用分层、并行、重叠和调度把它放到系统最能承受的位置。**

## 16. 截至 2026-08-24 的逐项核查总表

| 讲座方向 | 半年后的事实 | 判断 |
| --- | --- | --- |
| 开放数据与 recipe | Kaiyuan、EvoLM、Smol 继续公开训练轨迹；Nemotron 3 Super/Ultra 补充数据与环境资产 | ✅ 开放度增强，但完整 recipe moat 仍在 |
| Benchmark 与数据污染 | 更多训练报告公开数据组成，但 leaderboard 仍无法分离数据、recipe、prompt 与 Infra 贡献 | ✅ 原警告仍成立 |
| “几维并行” | Megatron 用 Parallel Folding 为 attention/expert 子图建立不同 mesh | ✅ 证明应从子图约束而非维度命名出发 |
| FSDP/ZeRO | FSDP2 per-parameter DTensor 与独立 process group overlap 继续成熟 | ✅ 重心转向 materialization/overlap 调度 |
| Context Parallel | Megatron 已把 CP 与 MoE、MLA、MTP 和多维并行组合 | ✅ 从前沿选项进入长上下文标准工具箱 |
| MoE / EP | DeepEP V2 支持 EP2048、ElasticBuffer、低/0-SM 路径；UCCL EP 扩到 EFA/AMD | ✅ 能力增强；⚠️ 跨硬件稳定性仍难 |
| CPU-centric vs GPU-centric | DeepEP 从 NVSHMEM 转向 NCCL GIN；NCCL 主线加入更多 device API | 🔄 概念仍有用，项目标签已失真 |
| 精确 attention | FlashAttention-4 针对 Blackwell 再次重写并使用 CuTe DSL | ✅ 每代硬件需要新映射 |
| Hybrid attention | Kimi K3、DeepSeek V4 把 KDA/MLA/sparse hybrid 带入旗舰模型 | ✅ 从研究方向进入部署现实 |
| Sparse attention | HiSparse 仍需 GPU/host 分层 cache | ⚠️ 少算 FLOPs 不等于状态管理简单 |
| FP4 training | Transformer Engine + Nemotron Ultra 提供大规模 NVFP4 证据；TE 的 linear-GEMM autocast 例中 2.03×，预量化为 3.55× | ✅ 已可用；⚠️ 完整训练 step 仍须实测 |
| CPU/GPU 大模型 offload | KTransformers 增加新模型、SGLang 和三层 cache 支持 | ✅ 专门化路径增强；⚠️ 仍需数百 GB DRAM |
| vLLM vs SGLang | 两者都覆盖 prefix cache、spec decode、PD；vLLM 偏 hash-block/connectors，SGLang 偏 radix/HiCache/day-0 | 🔄 起源差异保留，能力边界持续融合 |
| PD disaggregation | vLLM/SGLang/Dynamo 生态成熟，connector 增多 | ✅ 成为主流架构；⚠️ 官方明确“不自动提升吞吐” |
| KV pool / tiering | vLLM connectors、SGLang HiCache 已覆盖 GPU/CPU/storage 与多后端 | ✅ 进入工程实现；⚠️ metadata/backpressure 故障显性化 |
| Multimodal/agent infra | EPD、disaggregated encoder、agent hints/priority/cache retention 已出现 | ✅ 讲座预判开始落地 |
| NCCL 动态化 | 2.31 加入 CFT、per-collective tuning、GIN EFA GDA、0-SM、one-sided multi-NIC | ✅ NCCL 正从静态 collective 库扩展 |
| NVSHMEM / one-sided | 3.7.x 支持 Blackwell 与新 CUDA，但平台/ABI 限制仍多 | ✅ 持续发展，未被 GIN 替代 |
| UALink / scale-up | UALink 2.0 specification 已发布，加入 in-network compute | ✅ 标准推进；⚠️ 生态部署尚未收敛 |
| CUDA DSL | CUDA Tile 进入 Python/C++ 官方工具链；FA4 验证 CuTe DSL 性能上限 | ✅ 高层 tile abstraction 明显成熟 |
| 2:4 sparsity | Rubin 把 2:4 用于 attention activation，而不只是静态权重 | ⚠️ 对通用权重稀疏的批评仍对，用途已变化 |
| AI kernel generation | profile/RL agents 与更严格 benchmark 快速发展 | ✅ 单 GPU 进展大；🧪 通信/跨硬件 generality 未解 |
| CPU/C2C/DPU | Rubin 平台强化 C2C 并继续配置 BlueField-4 | ✅ CPU/DRAM/DPU 都回到系统 codesign，而非退出 |
| Engram / storage | DeepEP V2 已有实验 0-SM Engram path，CXL/memory grafting 论文增加 | 🧪 系统栈开始跟进，仍缺广泛生产证据 |

## 17. 六个最容易被误读的点

### 17.1 “PD 分离更快”

更准确：它允许 P/D 独立扩缩和 SLO 隔离，可能提高 **goodput**；KV transfer 与跨池排队意味着 raw throughput 可能不升反降。

### 17.2 “MoE 就是少算一点 FFN”

更准确：MoE 把稠密 compute 变成 routing + irregular communication + grouped GEMM + load balancing；成本结构改变了，不是简单缩小 dense model。

### 17.3 “低精度峰值翻倍，所以训练翻倍”

更准确：只有被加速的 GEMM 占比、Q/DQ fusion、内存/通信、recipe 都合适时才接近理论值。TE 的 B300 linear-layer benchmark 中，包含当步量化的 autocast 是约 2.03×，预量化 raw GEMM 是 3.55×；而完整训练 step 还有 attention、optimizer 和通信等额外限制。

### 17.4 “vLLM 是 PagedAttention，SGLang 是 RadixAttention”

这是历史起点，不是现状总结。当前应比较 scheduler、cache identity/eviction、model coverage、connector、PD/EPD、quant/kernel 与可靠性。

### 17.5 “NVSHMEM 是 GPU-centric，NCCL 是 CPU-centric”

这是理解传统控制路径的好近似，但 NCCL 2.31 + GIN/device APIs、DeepEP V2 已使边界融合。应直接追 data/control/progress path。

### 17.6 “2:4 sparsity 没前途 / 很有前途”

两句都太粗。静态权重 2:4 的模型约束确实强；Rubin 用它压缩 attention 中间 activation，则是不同的数据来源、生命周期和收益模型。

## 18. 我会如何沿着这场讲座继续学

### 18.1 从一个 optimization 写出完整账本

任选 FlashAttention、MoE EP、FSDP 或 PD 分离，必须能回答：

1. 数学语义是否改变？
2. FLOPs、HBM bytes、network bytes 各变化多少？
3. 新增了哪些 buffers、metadata 和 synchronization？
4. 哪段可以 overlap，critical path 还剩什么？
5. 哪个硬件/shape 假设一变，收益会消失？
6. 故障、长尾和多租户下会怎样？

### 18.2 一个可执行的实践顺序

1. 阅读/实现 `nano-vllm` 或 `mini-sglang` 一类最小引擎，画出 prefill/decode 与 KV 生命周期；
2. 在 vLLM/SGLang 测 TTFT、TPOT、throughput，逐项切换 continuous batching、chunked prefill、prefix cache、spec decode；
3. 用 [[Stanford CS336 - Lecture 08 - Parallelism II]] 的账本推导 DP/TP/PP/CP/EP 组合，再对照 Megatron process groups；
4. 用 [[Stanford CS336 - Lecture 06 - Kernels and Triton]] 的方法 profile 一个 attention/MoE kernel，记录 shape、roofline、occupancy 与 data movement；
5. 先做 NCCL/NVSHMEM microbenchmark，再讨论 EP overlap；没有链路基线，不应只看端到端 token/s；
6. 最后尝试 PD/HiCache，把 KV transfer、cache hit、排队和 failure injection 都纳入，而不是只跑无故障 happy path。

### 18.3 接下来值得持续跟踪的问题

- [ ] #question Hybrid attention 引擎能否用统一 state interface 管理 KV、MLA latent、KDA recurrent state、vision/encoder state？
- [ ] #question 0-SM communication 在 contention、multi-tenancy 和 failure 下的真实收益是多少？
- [ ] #question FP4 training 的跨模型 recipe 是否会收敛，还是继续高度依赖模型/数据/optimizer？
- [ ] #question EPD / agent-aware routing 的稳定目标函数是什么：TTFT、TPOT、tool latency、session completion 还是成本？
- [ ] #question UALink 2.0 何时形成跨厂商、可观测、可调优的 collective/software ecosystem？
- [ ] #question AI kernel agent 如何证明跨 shape 的正确性、性能稳定性与长期可维护性？
- [ ] #question Engram/CXL/remote KV 的 break-even 条件：可隐藏 compute window 至少需要多大？

## 19. 自测问题

1. 为什么“7D parallelism”不是一个足够的信息描述？给出两个子图需要不同 mesh 的例子。

    **面试回答：** “7D”没有说明各轴切什么、大小多少、哪些 process groups 重叠以及映射到哪种互联。例如同一批 GPU 上，attention 子图可用 TP/CP 切 heads 和 context，而 MoE 子图用 EP 放置 experts、重组 token；dense 参数与 expert 参数的 DP/分片组也可能不同，不能把所有维度机械相乘。

2. Megatron SP 与 CP 分别切了什么 activation，解决的主要瓶颈有何不同？

    **面试回答：** Megatron SP 与 TP 配合，把 LayerNorm、dropout 等原本在 TP ranks 复制的激活沿 sequence 分片，主要节省这部分 activation memory。CP 则把整层序列及 attention context 分布到多卡，支撑更长上下文，但需要交换 KV 或合并 attention partial，通信语义更重。

3. 为什么 MoE 的 FLOPs 下降可能让网络而不是 tensor core 成为瓶颈？

    **面试回答：** MoE 固定 Top-K 时只执行少数 experts，省下大量 dense FFN 乘加，却仍需把 token dispatch 到专家并 combine 回来。计算时间缩短后，All-to-All、热点 incast、打包和同步占比更高；每 expert token 太少还会降低 GEMM 效率，所以少 FLOPs 不等于低延迟。

4. FlashAttention 与 sparse/linear attention 分别改变了实现还是数学语义？

    **面试回答：** FlashAttention 主要用分块、online softmax 和重算减少中间矩阵读写，在浮点误差范围内实现同一个 dense softmax attention。Sparse attention 改变可读取的位置，linear attention 通常改变核函数或状态表示，属于模型计算语义变化；有益的近似不能直接称为等价 IO 优化。

5. FP4 tensor-core 峰值到端到端训练速度之间有哪些损失项？

    **面试回答：** 低精度峰值只覆盖支持该 dtype/shape 的计算部分，端到端还要支付量化/反量化、scale 搬运、累加精度处理、布局转换、访存、通信和调度。非 GEMM 算子、尾部和低利用率进一步限制收益；还需验证收敛与达到同等质量的训练步数，不能只比每步速度。

6. 什么 workload 下 speculative decoding 可能有效，为什么高 QPS 下不必然有效？

    **面试回答：** 在低并发、目标 decode 受权重带宽限制、draft 便宜且接受长度较高时，一次 target 验证可推进多个 token，容易获益。高 QPS 下 batch 已复用权重并占满算力，额外 draft/verify 会争用计算、KV 和调度资源，因此接受率高也未必提高吞吐。

7. PD 分离为什么更适合用 goodput 而非 raw throughput 评价？

    **面试回答：** PD 分离的主要价值是分别优化 TTFT、TPOT 并减少两阶段互扰，而 KV handoff 和双池排队可能增加总开销。Goodput 统计满足既定 SLO 的有效完成量，更能反映用户实际收益；raw throughput 可能把大量超时或长尾请求也算作成功产出。

8. 一次 KV cache miss 可能经过哪些 memory/network 层级？

    **面试回答：** 可先查本 GPU HBM 的 prefix/KV 状态，再查本机 DRAM、NVMe 或远端缓存池；远端取回可能经过 RNIC、网络、PCIe/NVLink 或 host staging，未命中则回源或重算。每一级都要加上查找、排队、布局转换和发布可读的成本，具体路径不是固定串行经过全部层级。

9. NCCL GIN 为什么使 CPU-centric/GPU-centric 的二分法失真？

    **面试回答：** NCCL GIN 允许在 GPU kernel 中发起网络操作，但底层既有 CPU proxy，也有 GPU 直接控制相关的后端。因此“发起者”和“推进通信的人”必须分开看，不能再按库名给 CPU-centric/GPU-centric 的永久标签；仍需核对版本、NIC 和 setup/progress 路径。[官方 device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)。

10. Rubin 对 2:4 sparsity 的新用法为什么不等于静态 weight 2:4 已经普适？

    **面试回答：** Rubin 介绍的是 attention 中间 activation 的动态 2:4 压缩，可减少后续 softmax/AV 路径的计算与搬运，稀疏对象和生命周期都不同于静态模型权重。它不证明任意模型剪成 2:4 权重仍能保质提速，还要计算选值、元数据、压缩和质量损失。[官方架构说明](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/)。

11. 为什么训练 storage I/O 往往比在线 serving 的 tiered state 更易预取？

    **面试回答：** 训练数据通常可按已知采样顺序提前读取，用较深队列把存储 I/O 与当前 step 重叠，访问也较容易批量化。在线 serving 的 KV 需求由请求到达、路由、命中和生成轨迹动态决定，且必须满足首 token/尾延迟预算，预测错还会浪费带宽或挤掉热状态。

12. 面对一个“AI 生成 kernel 比 baseline 快 2×”的结果，至少还要检查哪些条件？

    **面试回答：** 先核对同一硬件、shape、dtype、语义和精度容差，确认 baseline 已合理优化，再覆盖边界形状、随机输入和并发正确性。计时应区分编译/预热与执行、同步方式和重复波动，并检查是否靠特化或省略工作得利；最后看真实 workload 的端到端收益与适用范围。


## 20. 与现有主题笔记的连接

- 推理总图：[[LLM Inference]]
- 训练并行的详细账本：[[Stanford CS336 - Lecture 08 - Parallelism II]]
- GPU 与 kernel 性能方法：[[Stanford CS336 - Lecture 06 - Kernels and Triton]]
- GPU 架构背景：[[Stanford CS336 - Lecture 05 - GPUs and TPUs]]
- 数据与评测边界：[[Stanford CS336 - Lecture 13 - Data Sources and Datasets]]、[[Stanford CS336 - Lecture 12 - Evaluation]]

## 21. 资料索引（按主题）

### 训练、并行与模型

- [Kaiyuan-2B](https://arxiv.org/abs/2512.07612)；[EvoLM](https://arxiv.org/abs/2506.16029)；[Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
- [Megatron parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)；[MoE guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)；[Context Parallel](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [DeepEP](https://github.com/deepseek-ai/DeepEP)；[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)；[UCCL EP](https://github.com/uccl-project/uccl/blob/main/ep/README.md)
- [Kimi K3](https://github.com/MoonshotAI/Kimi-K3)；[DeepSeek V4 transparency/report](https://www.deepseek.com/en/transparency/)；[FlashAttention-4](https://arxiv.org/abs/2603.05451)
- [Transformer Engine NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)

### 推理与 KV

- [KTransformers](https://github.com/kvcache-ai/ktransformers)；[vLLM](https://github.com/vllm-project/vllm)；[SGLang](https://github.com/sgl-project/sglang)
- [vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/)；[SGLang PD disaggregation](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/pd_disaggregation.mdx)
- [SGLang HiCache](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_best_practices.mdx)；[Mooncake](https://github.com/kvcache-ai/Mooncake)；[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo)
- [DistServe](https://arxiv.org/abs/2401.09670)

### 通信、硬件与 kernel

- [NCCL 2.31.2 release notes](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2312/release-notes/rel_2-31-2.html)；[NVSHMEM 3.7.2](https://docs.nvidia.com/nvshmem/release-notes-install-guide/release-notes/release-3720.html)
- [NIXL](https://github.com/ai-dynamo/nixl)；[UCCL](https://github.com/uccl-project/uccl)；[UALink 2.0 announcement](https://ualinkconsortium.org/wp-content/uploads/2026/04/UALink-2.0-Specification-PR_FINAL.pdf)
- [Vera Rubin NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)；[Rubin GPU architecture](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/)
- [CUDA Tile Python](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/)；[CUDA Tile C++](https://developer.nvidia.com/blog/develop-high-performance-gpu-kernels-in-c-with-nvidia-cuda-tile/)
- [CUDA Agent](https://arxiv.org/abs/2602.24286)；[SOL-ExecBench](https://arxiv.org/abs/2603.19173)；[CommBench](https://github.com/uccl-project/CommBench)

## 22. 每周 Follow-up 日志

### 2026-08-31

- Hybrid attention / 开放权重与训练 recipe｜新事实：Qwen 发布开放权重的 `Qwen3.8-Flash-Next`，明确标为 Qwen4 架构的 experimental preview；语言模型为 125B/6B active，另有 51B n-gram embedding 与 4B MTP，按 `3 × Gated DeltaNet + 1 × Qwen Sparse Attention` 交错并在每层接 MoE，同时披露按参数类别混用 Muon/AdamW、取消 batch-size warmup 的 recipe｜相对上次的变化：原笔记主要用 Kimi K3/DeepSeek V4 说明 hybrid attention；这次新增了“线性 recurrent state + micro-block sparse attention + MoE + 可 offload 的 n-gram memory”同模组合及公开权重，但仍未开放训练数据或完整复现 recipe｜判断：🧪早期｜为什么重要：它进一步支持未来引擎需要统一管理 KV、recurrent state、MTP state 与外置参数/记忆，而不是只围绕 dense KV cache 设计；但官方另称正式版 Qwen3.8-Flash 才包含更多 production features，不能把预览权重等同于生产成熟｜一手来源：[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、[官方技术报告](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf)

- vLLM / speculative decoding / EPD / KV tiering｜新事实：vLLM `v0.28.0` 正式发布；Model Runner V2 加入 E/P/D disaggregation，分层 KV 支持磁盘 offload、可插拔 secondary-tier manager、partial load、metrics 与不依赖并行布局的 canonical CPU layout，同时加入 DFlash2、DSpark confidence-scheduled verification、draft model async scheduling，以及 Kimi K3/DeepSeek V4 的新 kernel 与 sparse-MLA 路径｜相对上次的变化：EPD 与多级 KV 从“生态正在探索/各 connector 拼接”推进为 vLLM 稳定版本中的成组能力，且 speculative decoding 开始和 hybrid/sparse 模型的 state/layout 一起演进｜判断：🔄演进｜为什么重要：调度器、模型 runner、KV connector、speculator 与并行布局正在成为同一个联合优化面；release note 中约 60% TTFT 改善、17 GiB/GPU 节省及若干 kernel speedup 均为项目自报，仍需按目标 workload 复测｜一手来源：[vLLM v0.28.0 release notes](https://github.com/vllm-project/vllm/releases/tag/v0.28.0)

- MoE 推理优化边界｜新事实：新论文在 OLMoE-1B-7B、DeepSeek-V2-Lite、Qwen3-30B-A3B 上报告：fused Triton kernel 隔离测试快 5.6–9.0×，但端到端仅 0.999×，瓶颈是每次 forward 约千次 kernel launch；去除全部 23 个 graph break 反而使模型慢 3×，INT4 路由漂移与质量损失也并不等价｜相对上次的变化：为原笔记“局部 kernel 加速不等于端到端收益、MoE 需看 launch/通信/调度”的判断增加了一组反例式实测，并修正“保持 router 路由一致即可保持质量”的过度简化｜判断：⚠️修正｜为什么重要：MoE 优化应先测 end-to-end ceiling、launch timeline 与 expert semantics，再决定融合、量化或编译；以上数字均是论文作者结果，尚非跨框架共识｜一手来源：[论文（arXiv:2608.26612）](https://arxiv.org/abs/2608.26612)

- Triton / kernel correctness / Rubin｜新事实：Triton `3.8.0` 正式发布，加入初始 Rubin SM107 支持和四 lane FP8/FP4 packed arithmetic，扩展 generic multi-CTA/TMA/multicast；同时新增 FpSan 符号浮点等价检查、扩大 ConSan 覆盖，并提供面向 symmetric memory 与 multi-node topology 的实验性 GSan data-race detector｜相对上次的变化：Rubin 的公开软件 enablement 已进入稳定 Triton release；同时 kernel DSL 开始内建数值语义、同步与分布式内存的验证工具，而不只追求代码生成与 autotune｜判断：✅强化｜为什么重要：这直接回应 AI 生成 kernel 和复杂 multi-CTA/通信 kernel 的 correctness 证明问题；但 release 中 Rubin 被称为 initial support，GSan 也明确是 experimental，不能据此推断生产完备｜一手来源：[Triton 3.8.0 release notes](https://github.com/triton-lang/triton/releases/tag/v3.8.0)

- CuTe/CUTLASS / FP4 / Rubin｜新事实：NVIDIA 发布 `CUTLASS 4.8.0dev`，CuTe DSL、Operator API 与 C++ building blocks 首次覆盖 Rubin SM107 的 dense、block-scaled FP8/FP4 GEMM，并暴露更大 SMEM/TMEM、B-collector reuse 与 mixed-precision 路径；`cute_ext` compiler pipeline 和 Operator API Rubin 支持均为 preview/preliminary｜相对上次的变化：原笔记对 Rubin FP4/稀疏能力的硬件判断出现了首批可阅读、可编译的 kernel building blocks，但执行 Rubin kernel 仍要求尚待 CUDA 13.4 GA 随附的 R615 driver，当前 Developer Preview 的 R610 不足｜判断：🧪早期｜为什么重要：可以提前研究 tile/layout 与低精度 kernel 形态，却不能把“repo 已有示例”误写为“Rubin 已可生产部署”｜一手来源：[CUTLASS 4.8.0dev release notes](https://github.com/NVIDIA/cutlass/releases/tag/v4.8.0dev)

- NIXL / PD connector correctness｜新事实：Dynamo `v1.4.2` 正式 patch 修复 Frontend 与 SGLang Runtime 镜像中的 NIXL loader path；此前 Rust binding 会静默落到 non-functional stubs，而同一解释器内 Python binding 正常，造成配置看似成功但数据路径并未真正工作｜相对上次的变化：把原笔记“connector 版本/路径/环境变量是工程雷区”的抽象警告升级为一个已确认且已修复的静默失效案例｜判断：⚠️修正｜为什么重要：PD/KV transfer 验收不能只看进程启动或 Python smoke test，必须观察真实传输、动态库解析、bytes/latency 与 fallback；否则 benchmark 可能测到错误路径｜一手来源：[Dynamo v1.4.2 release notes](https://github.com/ai-dynamo/dynamo/releases/tag/v1.4.2)

- SGLang / speculative decoding / KV 容量｜新事实：SGLang 新公开 issue 指出 `v0.5.18` 的 NEXTN/EAGLE draft worker 会先分配约 5.08 GiB 的临时 BF16 embedding/head，再在 KV pool 定容之后释放；报告者在单卡测试中测得 stock 的 `max_total_num_tokens` 为 149,806，修补初始化顺序后升至 305,070，截至核查时 issue 仍开放、无关联 PR｜相对上次的变化：speculative decoding 的成本从“额外 draft compute/acceptance rate”扩展到初始化时序对 KV admission capacity 的隐性影响｜判断：⚠️修正｜为什么重要：高并发服务应把启动期峰值、pool sizing 时点和最终空闲显存纳入容量审计；这是一份可复现的用户 issue/单环境 benchmark，不代表已被 SGLang maintainers 确认或修复｜一手来源：[SGLang issue #36452](https://github.com/sgl-project/sglang/issues/36452)

- AI kernel generation｜新事实：两项新工作把 benchmark 从常规 ML operator 推向更异构的任务：EMNLP 2026 接收的 DataKernelBench 评测 LLM 对 data-movement-heavy 数据库查询的 CUDA/Triton 优化，作者报告 H100 上 full-query CUDA 在 full pass rate 下相对 TorchPlan 为 2.11×；FABRICA 则用 49 个 CUDA→Cerebras CSL 配对任务测试跨架构重映射，作者报告 agentic workflow 将核心集成功率从 6/28 提至 26/28，并在 27 对 WSE-3 实机结果上取得 3.47× 几何平均加速｜相对上次的变化：原笔记的 CUDA Agent/SOL-ExecBench 视角被扩展到“非 ML dataflow”和“跨计算范式移植”，共同强调 target knowledge、执行反馈、failure-directed repair 与 correctness gate｜判断：🧪早期｜为什么重要：kernel agent 的下一道门槛不是继续刷熟悉算子，而是跨 workload/架构的语义重构与同目标硬件验证；两组性能均为论文作者 benchmark，尤其 FABRICA 仍是预印本，不宜外推到任意 shape/系统｜一手来源：[DataKernelBench](https://arxiv.org/abs/2608.25061)、[FABRICA](https://arxiv.org/abs/2608.25124)
