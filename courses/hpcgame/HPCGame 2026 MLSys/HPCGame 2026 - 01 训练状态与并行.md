---
type: topic-note
status: evergreen
course: HPCGame 2026 赛前讲座
area: distributed-training
topics:
  - "[[Distributed Training]]"
  - "[[Expert Parallelism]]"
  - "[[Context Parallelism]]"
---

# HPCGame 2026：训练状态与并行

> [!nav] 返回与扩散
> [[HPCGame 2026 - 00 MLSys 知识地图|知识地图]] · [[HPCGame 2026 - 02 稀疏模型与数值格式|模型结构]] · [[HPCGame 2026 - 04 通信与内存层级|通信]] · [[HPCGame 2026 - 06 前沿追踪|前沿追踪]]

> [!abstract] 核心问题
> 并行训练不是“用了几维”，而是：**哪些状态由谁拥有，在哪个子图上切分，何时 materialize，通信能否映射到合适拓扑。**

> [!nav] 接回完整训练周期
> 达到同等质量需要多少数据、更新与生成，见 [[HPCGame 2026 - 07 数据与后训练闭环]]；实际放置、排队、checkpoint 与故障恢复，见 [[HPCGame 2026 - 08 集群效率与可靠性]]。
>
> 本篇既有版本案例来自 2026-08/09 的历史材料；其当前状态按 [[MLSys - 证据台账]] 和 [[MLSys - 运行记录]] 核验，不将文章的更新日期当作全部事实核查日。

## 1. 先列状态，再选并行

| 状态 | 随什么增长 | 首要压力 | 常见处理 |
| --- | --- | --- | --- |
| parameter | 参数量 | 静态容量、读取带宽 | TP、PP、FSDP、offload |
| gradient | 参数量 | 容量、归约 | DP/EDP、Reduce-Scatter |
| optimizer state | 参数量 × 状态数 | 最大静态容量之一 | ZeRO-1/2/3、FSDP、CPU offload |
| activation | batch × sequence × hidden × layer | 峰值容量 | SP、CP、PP、recompute |
| attention context | sequence² 或 KV/latent state | 长上下文 | CP、MLA/hybrid、checkpoint |
| routing buffer | tokens × top-k × expert | 动态流量与不均衡 | EP、packing、A2A overlap |

训练规模的三个源头彼此不同：参数变大主要压 model states；token 变多主要压总训练时间与数据管线；context 变长主要压 activation、attention 与并行通信。把三者统称为“模型变大”会在第一步就选错方案。

## 2. 并行轴不是标签，而是 ownership 变换

$$
N_{GPU}=N_{DP}\times N_{TP}\times N_{PP}\times N_{CP}\times N_{EP}
$$

这个式子只是资源计数，不是设计答案。实际 process groups 可以重叠、折叠，并且不同子图可以拥有不同 mesh。

| 轴 | 切什么 | 主要收益 | 主要支付 |
| --- | --- | --- | --- |
| DP | batch | 算力扩展 | gradient synchronization、global batch |
| ZeRO/FSDP | parameter/gradient/optimizer ownership | 静态显存 | materialization、reshard、checkpoint 复杂度 |
| TP | hidden/head/linear weight | 单层放置与算力 | 高频层内 collective，要求高速域 |
| SP | non-TP activation 的 sequence 维 | activation memory | 与 TP 配套的 AG/RS |
| CP | 整层 activation/context 的 sequence 维 | 真正长上下文 | attention KV/partial-result 通信 |
| PP | layer/depth | 模型容量 | bubble、activation、stage imbalance |
| EP/ETP/EDP | expert 与 expert replica | 稀疏容量/算力 | 动态 dispatch/combine 与负载均衡 |

> [!warning] SP ≠ CP
> Megatron SP 主要切 TP 区域外的 activation；CP 把整层 context 分布出去并改变 attention 数据交换。两者都出现 sequence 维，但解决的瓶颈不同。

## 3. 三条常见因果链

### 3.1 FSDP：省复制 → 增加 materialization

```text
replicated parameter/gradient/optimizer
        ↓ shard ownership
静态显存下降
        ↓ forward/backward 按需恢复视图
All-Gather / Reduce-Scatter 增多
        ↓
prefetch、overlap、reshard、allocator、checkpoint 成为主要问题
```

ZeRO/FSDP 不改变数据并行的数学目标，改变的是状态 ownership。真正的性能问题不是“有没有 shard”，而是 materialization 的粒度、时点、临时峰值与 overlap。

截至基线核查，PyTorch FSDP2 已以 per-parameter DTensor `fully_shard` 为组合式 API；PyTorch 2.13 又加入可选独立 process group 来重叠 All-Gather 与 Reduce-Scatter。这一演进说明重点已从容量功能转向调度。

### 3.2 PP：省单卡深度 → 增加时间版本与 bubble

GPipe 把 layers 切成 stages，再把 batch 切成 microbatches。必须同时处理：

- fill/drain bubble；
- stage imbalance；
- activation 保存或 recompute；
- forward/backward 顺序与权重版本；
- MoE stage 的动态波动。

1F1B、interleaving、DualPipe 的意义都是重新安排时间，而不是消除依赖。只有被重叠的区域不争用同一关键资源，bubble 公式才会转化为 wall-clock 收益。

### 3.3 Parallel Folding：不同子图需要不同 mesh

现代模型里 attention 与 expert 子图可能有相反偏好：

```text
attention：更关心 head/context 切分与稳定 collective
expert：更关心容量、token routing 与 All-to-All
```

Megatron 的 Parallel Folding 将 attention 的 TP/CP/DP 与 expert 的 ETP/EP/EDP 解耦。这比“7D parallelism”更有信息：**先为每个子图找到局部 mesh，再处理它们的连接。**

## 4. 拓扑决定并行轴的可行域

| 拓扑层 | 适合承载 | 原因 |
| --- | --- | --- |
| 单 GPU | fused op、recompute | 无网络但受 HBM/SMEM/register 限制 |
| NVLink/NVSwitch 域 | TP、细粒度 CP/EP | 高频、小粒度 collective 更可承受 |
| RDMA scale-out | DP、较粗 EP/PP | 带宽可扩展但延迟和 tail 更明显 |
| CPU DRAM/C2C | offload、checkpoint staging | 容量大，是否能隐藏取决于可预测窗口 |
| storage | dataset/checkpoint、极冷状态 | 容量最大，随机在线访问最危险 |

因此 process group 不能脱离物理拓扑讨论。一个数学上合法的分解，可能把 local GEMM 切得过小，或把高频 collective 推到错误网络层级。

## 5. 与 MoE、低精度和 kernel 的连接

- MoE 让 EP 成为必要轴，但详见 [[HPCGame 2026 - 02 稀疏模型与数值格式#2. MoE：计算稀疏化，通信动态化]]；
- EP dispatch/combine 的 control/data path 详见 [[HPCGame 2026 - 04 通信与内存层级#3. Collective 正在变成可编程数据面]]；
- FP4 若只缩 GEMM 而不缩通信/optimizer state，训练 step 不会同比加速；
- TP/CP/EP 切得越细，kernel shape 越偏离单卡最优，详见 [[HPCGame 2026 - 05 Kernel、DSL 与硬件]]。

## 6. 实践账本

设计一个训练配置时，按顺序回答：

1. 每卡 parameter、gradient、optimizer、activation 各占多少？峰值发生在哪一时刻？
2. 每个子图的 local GEMM shape 是否仍足够大？
3. 每层/每步的 collective 类型、消息量和频率是什么？
4. 它们落在哪个 NVLink/RDMA/PCIe/C2C 层级？
5. 哪些通信真正被 overlap，哪些仍在 critical path？
6. checkpoint、failure recovery、elastic resize 是否保持相同 ownership 语义？

## 7. 易错判断

- “维度越多越先进” → 维度名不说明 subgraph mesh 与拓扑；
- “FSDP 只省显存” → 它重写状态生命周期与 checkpoint 语义；
- “PP bubble 越小越快” → stage imbalance、kernel 粒度和链路可能支配；
- “SP 与 CP 都切 sequence” → 被切的 activation 范围和通信语义不同；
- “MoE 只是再加一维 EP” → 它把规则计算变成输入相关的动态流量。

## 8. 一手资料

- [Megatron parallelism guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron MoE guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [Megatron Context Parallel](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [PyTorch FSDP2 `fully_shard`](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)
- [PyTorch 2.13 release blog](https://pytorch.org/blog/pytorch-2-13-release-blog/)
- [[Stanford CS336 - Lecture 08 - Parallelism II]]
