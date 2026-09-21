---
type: course-note
status: developing
source_status: transcript-and-slides-reviewed
course: Weiming HPC Training Camp × LCPU AI Infra Seminars
lecture: 6
official_session: Workshop 02
lecture_date: 2026-08-23
speaker: 孔昊然
area: systems
topics:
  - gpu-communication
  - rdma
  - collective
  - moe
  - kv-cache
  - topology
  - synchronization
  - fault-tolerance
aliases:
  - LCPU Session 06
  - LCPU Workshop 02
  - AI Communication Stack
video_url: https://www.bilibili.com/video/BV1FYhL6REgi/
slides_url: https://infra.seminars.lcpu.dev/slides/workshop02.pdf
---

# Session 06：AI Communication Stack

> [!abstract] 本讲核心
> 通信栈把跨设备 producer–consumer 依赖变成一套可检查、可推进、可恢复的交接条件。字节移动只是其中一部分；GPU 何时能安全消费，还取决于地址、完整性、可见性、顺序、buffer 生命周期和错误状态。

> [!info] 整理依据
> 已阅读完整字幕（2151 条，至 02:17:09），并对照官方 Workshop 02 的 116 页课件。B 站编号为第 6 讲，时长 02:17:11。实际录播重点是前两部分：消费条件与物理路径；后面各方案、workload、scaling 和新硬件多为概览。本文保留课件的进一步细节，PDF 页码含封面；原顺序见 [[LCPU AI Infra Seminars - Session 06 - AI Communication Stack - 精编字幕]]。

## 来源与导航

- [视频](https://www.bilibili.com/video/BV1FYhL6REgi/)
- [官方课件](https://infra.seminars.lcpu.dev/slides/workshop02.pdf)
- [课程日历](https://infra.seminars.lcpu.dev/schedule)

| 视频回看 | 实际讲述 |
|---|---|
| [00:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=0) | 从 Tensor Core、pipeline 引入跨设备数据供给 |
| [22:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=1320) | 五种语义与“状态、控制归谁”的主线 |
| [44:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=2640) | 延迟拆解与最慢 rank |
| [47:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=2820) | 可消费条件、ownership、generation |
| [60:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=3600) | 节点内 fabric、PCIe、host staging |
| [72:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=4320) | 跨节点、RC、拓扑、rail 与 scale-up/out |
| [90:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=5400) | RC / MRC / Falcon / UCCL / UET 概览 |
| [102:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=6120) | Collective、P2P、KV、MoE |
| [118:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=7080) | Overlap、counted writes、distributed kernel |
| [123:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=7380) | 大规模系统与硬件边界展望 |

> [!note] 讲者的限定
> 开场与约 29 分钟处均强调：这是带有个人视角的 high-level 梳理，通信高度依赖具体硬件和 workload，部分系统主张不像单卡 kernel 那样容易验证。下文比较用来建立提问框架，不是对所有部署的性能保证。录播末尾邀请提问，但没有实质问答。

| PDF 页 | 内容 |
|---|---|
| 3–25 | 从单 GPU pipeline 到跨设备依赖；通信层次与延迟拆分 |
| 26–34 | Delivery、Placement、Completion、Visibility、Ordering |
| 35–56 | Scale-up、PCIe P2P、staging、GPUDirect RDMA、拓扑 |
| 57–69 | Memory/transaction 与 verbs；RC、MRC、Falcon、UCCL、UET |
| 70–72 | Workload 与等待关系；collective / reduction placement |
| 73–82 | P2P 对象、KV、Mooncake、HiCache、DualPath、HiSparse |
| 83–87 | MoE dispatch/combine、DeepEP、NCCL EP、热点 expert |
| 88–96 | Locality、overlap、distributed kernel、MegaMoE |
| 97–106 | 扩展阅读：大规模网络、NCCLX/CTran、初始化与运维 |
| 107–116 | 扩展阅读：重划 compute/memory/interconnect 边界 |

## 1. 先确定“什么条件下可消费”

假设 GPU A 产生一个 32 KiB chunk，GPU B 的下一次计算依赖它。至少需要检查五类证据：

| 维度 | 要证明什么 | 单独成立仍不足以说明什么 |
|---|---|---|
| Delivery | 所需片段到达规定接收端点并满足该层完整性要求 | 未必已写进最终 GPU buffer |
| Placement | bytes 位于正确 address、offset 和 generation | 未必完整，也未必对 consumer 可见 |
| Completion | 明确的一层操作义务已结束 | 其他层的义务未必结束 |
| Visibility | consumer 在要求的 memory scope 能看到写入 | 未必获得 buffer 所有权 |
| Ordering | payload、signal、acquire 与执行之间有必要顺序 | 地址、内容和生命周期仍须正确 |

这不是五个一定按序执行的 wire stages，而是证明可消费所需的五类条件。

Buffer ownership/lifetime 横跨全部条件。源数据不能在 DMA 尚未读完时被覆盖；目标 slot 不能在消费者读完前被下一代复用。迟到的旧 signal 或旧 packet 也不能冒充当前 generation。

## 2. Completion 必须带“谁完成了什么”

同一个 chunk 可能有 source-buffer reuse、transport ACK、remote placement、runtime publication、consumer acquire 等多个完成点。

本地 CQE 只能按具体 verb/API 的规定解释。它不自动证明远端 GPU 已执行必要 acquire，也不表示远端应用已完成计算。WRITE_WITH_IMMEDIATE、flag 或 counter 仍需要与内存顺序和接收端协议配合。

```text
A producer 完成源数据
  → 正确发布 / issue
  → data mover / transport
  → B 正确 placement
  → publication + consumer ordering
  → B 消费
  → release / 下一代复用
```

Payload、control metadata、completion 是三类逻辑职责，物理上可能共用一条通路、在同一报文中捎带，或由同一 endpoint 产生。

## 3. 端到端延迟比链路时延包含更多内容

课件给出排查式分解：

$$
\begin{aligned}
T_{visible}\approx{}&T_{source-ready}+T_{launch/progress}+T_{staging}\\
&+T_{queue}+T_{serialization}+T_{propagation}\\
&+T_{placement}+T_{completion}+T_{consumer-wait}\\
&+T_{recovery}-T_{hidden-overlap}
\end{aligned}
$$

它帮助定位等待，不是各项永远可独立相加的精确排队模型。实际存在重叠、共享资源和依赖。

可按事件打点：producer ready → WQE post → first packet/DMA → last byte → ready signal → consumer start。小消息常被 launch/progress 或同步固定开销主导；大消息可能受序列化与带宽限制；高负载下 queue 和 recovery 更明显。

同步 step 经常取决于最慢必要依赖。平均带宽很好，但一个 rank 的 completion 晚，仍会形成全局等待。诊断先找到第一个偏离者，再向前追原因，避免把所有后续 timeout 都当成独立故障。

## 4. 四条常见物理路径

| 路径 | Payload 可能经过的资源 | 重点检查 |
|---|---|---|
| 专用 scale-up fabric | GPU A → NVLink/NVSwitch 等 → GPU B | fabric edge、HBM、endpoint、completion |
| 节点内 PCIe P2P | GPU A → PCIe switch/root complex → GPU B | peer access、IOMMU、NUMA 与实际拓扑 |
| Host staging | GPU → host buffer → 网络/本地搬运 → host buffer → GPU | 额外 DRAM/PCIe/copy 与多段完成 |
| GPUDirect RDMA | GPU HBM → RNIC → fabric → RNIC → GPU HBM | 注册与权限、GPU–NIC affinity、transport 和 visibility |

“同一台服务器”不保证两张 GPU 之间路径最短；“同一个 logical device”也不保证所有物理存储访问距离相同。逻辑 API 与物理距离必须分别核实。

### 一次 RC RDMA WRITE 的职责链

1. Setup：注册 memory region、建立 QP，部分开销可摊销。
2. Post：提交 WQE/doorbell。
3. Validate：检查本地访问、远端地址与权限。
4. Source DMA：读取数据。
5. Packetize / schedule：分片、排队、拥塞与路径控制。
6. Remote placement：校验并写入目标。
7. Reliability / retirement：ACK、retry、释放相应 outstanding 状态。
8. Publication / acquire：把传输结果接到实际 consumer 的同步协议。

不同实现可能重叠这些步骤；记录它们是为了弄清状态归谁，以及每个信号到底覆盖什么。

## 5. 拓扑决定共享边，mapping 决定谁使用这些边

Clos/Fat Tree 有多条等价路径，不代表单个 flow 自动吃到所有路径。Per-flow ECMP hash collision 可使某条 uplink 拥塞、另一条闲置。更多 channels/QPs 是否产生更多独立 flows，取决于实际映射，不能固定画等号。

Packet spraying 或 adaptive routing 能提高利用率，却可能增加乱序、gap tracking、replay 和 path-health 状态。交换机端减少的热点，有时会转变为端点的 SRAM 与恢复成本。

Torus/Mesh 的共享关系由坐标、相邻链路和 partition shape 决定；rail-optimized 关注 GPU–NIC–rail affinity、rank placement 与策略。三者不是必须互斥的标签。

验证时对齐 logical rank、GPU/NIC/NUMA 位置、实际 route、per-link bytes/stall 和 per-rank completion。厂商拓扑图只描述结构，无法单独证明 workload 的有效带宽。

## 6. 比较通信方案，要比较状态和控制放在哪里

课件对几条路线的比较可压缩为：

| 路线 | 主要控制位置 | 获得的能力 | 必须承担的成本/边界 |
|---|---|---|---|
| 传统 RC | RNIC/QP | 成熟的可靠 verbs 与硬件 progress | QP、序号、可靠性与路径耦合 |
| MRC | Endpoint / transport 的多路径控制 | Spraying、选择重传与部分路径故障恢复 | Path health、SACK/reorder/replay 状态；有限 verb 子集 |
| Falcon | 硬件辅助 transport + 软件 | 更快测量、拥塞反应和恢复 | 硬件/固件与 ULP mapping 复杂度 |
| UCCL-Tran | 可编程软件控制 + 既有 NIC primitives | 策略灵活、适配不同后端 | CPU/NUMA/polling/tail，以及后端相关数据重组 |
| UET | 标准规定 wire behavior | 互操作的传输契约 | 具体状态放在 NIC/host/DPU 由实现决定 |

UCCL 的 UC/UD、RC、SRD、AF_XDP 后端不能一概而论。例如 RC 保留 RNIC 的 packet reliability；裸 UD 需要更多软件可靠性/重组；AF_XDP fallback 不等价于 GPUDirect。

对任一方案都用同一个问题追踪：谁看到 loss/ECN/RTT 异常，谁保存重放源，谁限速或绕路，谁发布终态，谁恢复 runtime。无法局部恢复时仍要向上层交接，而不是无限等待。

## 7. Workload 同时生成流量和等待关系

相同 bytes 数量，不意味着相同关键路径：

| Workload | 数据/对象 | 典型等待 |
|---|---|---|
| AllReduce | 规则 partial tensor | Reduction 和结果分发的最慢必要边 |
| Pipeline parallelism | Microbatch activation | 下游 stage 等待形成 bubble |
| Prefill/Decode 分离 | 请求、layer、token range 对应的 KV blocks | Block ready → decode admission |
| MoE | Token + expert/source/offset metadata | Incast、热点 expert、GEMM 与 combine tail |
| Distributed kernel | Tile/task 级远端状态 | Credit、counter、phase、局部 progress |

Ring AllReduce 在常见 reduce-scatter + all-gather 模型下，每 rank 的发送量约为：

$$
2\frac{p-1}{p}S
$$

其中 $p$ 是 ranks 数，$S$ 是该 rank 输入张量大小。这个公式描述通信量，不直接包含 topology、startup、reduction compute 或 recovery 的时间。

把 reduction 下沉到 NIC/switch 可以减少某些 GPU 工作和重复传输，同时引入 reduction state、树配置、支持操作限制和故障处理。要同时测 network bytes、SM 时间、HBM traffic、setup 与结果发布。

现场还特别提醒浮点归约次序：改变 reduction placement 或树形结构可能改变加法顺序。浮点加法不满足实数意义的结合律，应检查误差与可复现性需求，而非假定 bitwise 结果不变。

## 8. KV / P2P 搬运的是有身份和寿命的对象

`ptr + bytes` 不能完整描述 KV。还需要 request、layer、token range、dtype/layout、version/generation、placement、ownership 和失败状态。

### Mooncake：传输层与对象管理层

Transfer Engine 执行已提供源/目标内存之间的数据移动。Store 负责 key、位置、副本、容量、pin/eviction 与可读状态。传输完成不必然表示对象已经提交为可读。

Zero-copy 只减少某些 payload copies。Lookup、注册、分配、metadata、queue、重试与 publication 仍可能处于关键路径。

### DualPath：把未利用的存储入口纳入调度

课件讨论的 DualPath 同时考虑 Prefill 与 Decode 节点的存储 NIC。Decode 侧可以先读命中 KV，再经 compute network relay 到 Prefill，配合 layer-wise overlap。

收益来自入口带宽聚合和调度；代价是额外 host buffer、PCIe/DRAM/CNIC traffic、layout 和完成状态。它并不使 KV 流量全程脱离模型通信网络，仍需 QoS 与资源预算。

### HiSparse：容量收益换按需搬运

完整 KV 留在 host，GPU 只持有 hot blocks。Top-k 选择之后的 miss 触发 lookup、slot 分配、swap-in 和 table publication。要测 hit rate、bytes/token、transfer tail 和 decode stall；hot pool 过小可能出现 thrashing。

Direct-to-host 让 P 的 KV 直接落到 D 的最终 host tier，可省去 D GPU staging，但 host memory、NUMA 和 publication 仍可能成为瓶颈。

## 9. MoE 通信要连同布局与计算偏斜一起分析

完整路径通常包括：

```text
router / top-k
  → count / quota / prefix
  → pack payload + metadata
  → dispatch
  → expert-major receive layout
  → grouped GEMM
  → combine / 权重归并
```

均匀 All-to-All benchmark 会漏掉 routing、layout、metadata、expert skew 和 combine。一个热门 expert 同时制造 ingress incast、计算队列和其他 ranks 的尾部等待，单靠 congestion control 不能消除计算偏斜。

DeepEP 与 NCCL EP 的 high-throughput / low-latency 路径体现不同消息形态的取舍：聚合和分层有利于大批量，较直接的路径减少小消息 startup。具体后端与资源占用以课件所讨论版本为准。

动态复制热点 experts 可分摊负载，却消耗权重容量、同步流量、route plan 和训练时梯度归并。评估要比较节省的 dispatch/GEMM/combine 等待，是否覆盖复制与同步成本。

## 10. Overlap 成立需要两项条件

第一，依赖允许。Consumer 能在 chunk ready 后开始，就可能流水；若始终等待整个 tensor，切块不会自动缩短关键路径。

第二，资源允许。通信和计算可能共同消耗 SM、HBM、L2、PCIe 和 fabric。Timeline 上重叠，不一定表示总时长缩短。

```text
producer:   chunk i+1
transfer:   chunk i
consumer:   chunk i-1
```

Streams/events 表达顺序，不能自动创造 bandwidth、保留 SM 或选择最佳 chunk size。过小的 chunks 增加 launch/signal/metadata 开销；过大则减少 overlap、放大尾部。

### 从 kernel 边界到 distributed kernel

Fused epilogue 可在 tile ready 后启动通信。Persistent distributed kernel 将 task queue、remote operations、signal/wait 与 compute 放入长期 progress loop，减少 host hand-off。

相应责任是对称/注册地址、credit、arrival counter、phase、buffer lifetime 和错误退出。Device-initiated 只说明谁发起，不必说明全部 progress 都由 GPU 执行；NIC 或 host proxy 仍可能参与。

MegaMoE 案例把 dispatch/remote pull、两个 GEMM、激活/重量化、remote combine write 和最终归并放进同一执行系统。其收益要以端到端依赖缩短和资源开销共同评估。

### Counted writes

将远端数据写入与完成 byte counter 关联，可以减少额外通知，但 consumer 仍要按规定等待和建立 ordering。Sender 的本地 completion 与 receiver 的目标计数具有不同语义；counter 复用仍需 phase/generation 保护。

## 11. 大规模下，初始化与恢复也是通信成本

课件扩展比较了三个层面的工作：MRC 关注 transport/path 故障，NCCLX/CTran 关注通信库和可编程 transport，100K+ GPU 通信栈工作关注初始化、资源、流控、诊断和测试全生命周期。

规模放大后的几组乘法：

- 每 rank 状态 × ranks × communicators → HBM/QP footprint。
- RTT 差异 × 带宽 → 不同路径所需的 outstanding window。
- 小概率局部失效 × 大量设备 → 经常发生的恢复事件。
- 局部 stall × 同步依赖 → 级联 timeout。

Lazy connection/channel 和 metadata 复用减少常驻成本；按 topology/BDP 调整 in-flight limits，避免既饿死远路径又在近路径制造 burst。代价是更多策略、reorder 状态和调试复杂度。

> [!note] 整理者推导
> 若每个独立端点的故障率近似为 $\lambda$，$N$ 个端点中至少一个出故障的事件率量级可接近 $N\lambda$。独立性与恒定失效率仅是简化假设，但足以解释为什么小系统里的罕见事件，在大集群里需要常态化处理。

目标指标应包含 useful goodput、restart 时间、tail latency 和 diagnosis 时间，而不仅是理想稳定阶段 GB/s。

## 12. 移动硬件边界时，也要追踪责任移到了哪里

课件把这些方向作为批判性思考案例：

| 方向 | 可能减少的开销 | 新约束 |
|---|---|---|
| Wafer-scale / Cerebras | 部分细粒度跨芯片边界 | 片上 SRAM 容量、off-wafer I/O |
| Groq 式静态调度 | 部分动态仲裁与调度抖动 | 动态请求、容量与外部故障仍需 runtime |
| Warm KV/context tier | HBM 容量压力、重复计算 | Metadata、预取、QoS、eviction、miss tail |
| HBF | 冷数据容量压力 | 页粒度、延迟、写寿命与写放大 |
| PNM / 3D stacking | Memory movement 与每 bit 能耗 | 散热、封装、mapping 与接口耦合 |
| NetDAM 类研究原型 | 部分近网络归约/搬运 | 操作身份、权限、幂等性与 partial-failure recovery |

不能只问“少了哪次 copy”，还要问新增多少 metadata、buffer、staleness、recovery 与 thermal cost。相关产品和研究方向的性能主张需要各自验证，课件介绍不等于部署建议。

## 13. AI Infra 视角与诊断清单

| 视角 | 关键观察 |
|---|---|
| Shape | Chunk/object 大小、fan-in/out、expert token 分布 |
| Compute | Communication kernels/packing/reduction 占用多少 SM |
| Memory | Staging、HBM traffic、KV pools、注册与buffer寿命 |
| Communication | Path、BDP、queue、ECN/retry、flow balance |
| Runtime/System | Post/poll、ready publication、communicator、恢复与诊断 |

实践中先记录端到端 producer-ready → consumer-start，再逐层对齐：

- [ ] Logical rank 与 GPU/NIC/NUMA/rail 映射明确。
- [ ] 实际路径与 fallback/staging 行为已确认。
- [ ] Payload、metadata 与 ready signal 的 generation 一致。
- [ ] CQE/ACK/counter 的语义与 memory scope 明确。
- [ ] 测到 per-rank tail，而不仅是平均吞吐。
- [ ] 区分发送端慢、网络慢、目标内存慢和 consumer scheduling 慢。
- [ ] 评估 overlap 时检查 compute duration 是否被资源争用拉长。
- [ ] Unknown result、重试、buffer 复用和 communicator recovery 有定义。

本讲将 [[LCPU AI Infra Seminars - Session 04 - Pipeline Ordering Data Orchestration]] 的异步依赖扩展到多设备，也将 [[LCPU AI Infra Seminars - Session 05 - Towards Modern Networking System]] 的流控、可靠性和状态位置落实到 AI workloads。

## 14. 自测题

1. 为什么 sender CQE 不能直接当作 remote GPU 的消费许可？

    **面试回答：** Sender CQE 只证明对应 verb/API 规定的本地操作完成条件，不能自动证明远端 GPU 已获得可见性和消费权。还要确认目标地址与 generation 正确、全部 payload 到达、信号按顺序发布，以及 consumer 执行所需 acquire/同步；接收完成与应用消费完成也不同。

2. Delivery 与 Placement 分别要检查什么？

    **面试回答：** Delivery 检查需要的片段是否到达规定端点并满足完整性要求；Placement 检查它们是否写入正确 buffer、offset 和 generation。包已到 RNIC 不代表已进入目标 GPU buffer，地址正确也不代表所有片段都齐全，两项都不能替代消费侧同步。

3. Buffer generation 怎样避免上一轮迟到信号造成错误？

    **面试回答：** 每次复用 slot 都关联一个 generation，完成通知必须同时匹配 slot 身份与当前代次，迟到的旧信号就不能解锁新一轮消费。但 generation 标签本身不能阻止旧 DMA 覆盖新数据，还必须保证旧传输退休、目标写入校验和安全复用。

4. PCIe P2P、GPUDirect RDMA 与 host staging 的路径有何区别？

    **面试回答：** PCIe P2P 让节点内设备经可用 PCIe 路径直接访问对方内存；GPUDirect RDMA 让 RNIC 直接 DMA GPU 内存并跨网络传输；host staging 则先落到 CPU DRAM 再搬到 GPU。三者的拓扑、注册、带宽与完成协议不同，支持统一地址也不保证走直接路径。

5. Clos 有多路径，为什么仍可能出现单 flow 热点？

    **面试回答：** 常见 per-flow ECMP 把一个 flow 固定到一条路径，多个 flow 的 hash 碰撞还会集中到同一 uplink，因此物理多路径不等于自动均匀利用。更多 flow、packet spraying 或 adaptive routing 可能改善，但要承担乱序、端点状态和恢复成本。

6. MRC、UCCL、UET 分别在哪个层面提供控制或规范？

    **面试回答：** 按本讲比较，MRC 在 endpoint/transport 层组织多路径、可靠性和故障恢复；UCCL 以可编程软件控制结合现有 NIC primitives 实现传输策略；UET 规定互操作的传输与 wire 行为。三者分别偏向传输机制、软件实现和规范，不能仅用“是否 GPU-centric”排名。

7. 为什么 Ring 的通信量公式不能直接给出实际 latency？

    **面试回答：** $2(P-1)S/P$ 只估计 ring all-reduce 每 rank 的发送字节量，没有计启动、逐轮依赖、有效带宽和拓扑争用。简化时间可写成 $2(P-1)\alpha+2(P-1)S/(PBW)$，真实值还受 reduction、分块流水和最慢 rank 影响。

8. TE 传输完成与对象 Store 发布可读状态有什么区别？

    **面试回答：** Transfer Engine 完成表示指定传输的义务结束；Store 发布可读还需确认对象所需片段齐全、版本和元数据一致，并建立消费者可见性与生命周期。对象可能分多次传输，即使某次 TE 完成，也不能提前把整个对象暴露给读者。

9. MoE expert skew 怎样同时制造网络和计算瓶颈？

    **面试回答：** 大量 token 路由到少数 experts 会造成对应设备的网络 incast 和排队，同时增大这些 expert 的 GEMM 工作量，其他设备却闲置。Dispatch、计算和 combine 的关键路径都被热点拉长，因此要联合看路由均衡、expert 放置及通信，不能只优化单个 GEMM。

10. 为什么 multi-stream 不保证通信计算有效重叠？

    **面试回答：** Multi-stream 只提供并发提交的可能，依赖、同步或设备资源不足仍会使执行串行。通信与计算还可能争用 SM、HBM、PCIe/NIC 带宽；只有独立工作可并行且关键路径实际缩短，才算有效重叠，需要 timeline 和端到端计时验证。

11. Device-initiated 与 GPU-only progress 是否等价？

    **面试回答：** 不等价。Device-initiated 表示请求由 GPU 发出，但后续可能仍由 CPU proxy、NIC 或其他 agent 推进；GPU-only progress 则额外要求数据面完成不依赖 CPU 持续参与。应分别追踪 setup、issue、progress、completion 和故障恢复由谁负责。

12. 大规模时，初始化、资源状态和诊断为什么会进入关键路径？

    **面试回答：** 规模扩大后，连接建立、拓扑发现、内存注册和同步启动成本可能显著增长，QP、窗口和映射状态也会受限。任何慢 rank 或故障都会拖住集体操作；若诊断无法定位停在哪个交接点，恢复时间也会进入训练或服务的端到端成本。


## 来源边界与进一步核验

- 录播约 01:30 后转为快速概览；本笔记的后端细分、详细状态链、公式与诊断清单还包含课件补充和明确标注的整理者推导。
- 录播对新硬件、协议版本和产品趋势的描述按讲座语境保留；不把宣传中的潜在收益写成已经实测的系统结论。
- 第 5 讲按用户要求不继续整理，既有课件草稿仅作可选背景，不属于本轮完整交付。
