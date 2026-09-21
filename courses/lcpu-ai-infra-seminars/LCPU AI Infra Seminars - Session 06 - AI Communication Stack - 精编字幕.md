---
type: transcript
status: polished
course: Weiming HPC Training Camp × LCPU AI Infra Seminars
session: 6
official_session: Workshop 02
speaker: 孔昊然
video_url: https://www.bilibili.com/video/BV1FYhL6REgi/
slides_url: https://infra.seminars.lcpu.dev/slides/workshop02.pdf
source_status: transcript-and-slides-reviewed
topics: [communication, rdma, topology, kv-cache, moe, synchronization]
---

# Session 06：AI Communication Stack｜精编字幕

> [!info] 整理说明
> 依据完整 SRT（2151 条，至 02:17:09）和官方 Workshop 02 课件整理。删除口头重复，校正专有名词，保留实际讲述顺序、主要论证及限制条件；不是逐字稿。时间采用真实字幕的分钟级回看锚点。结构化笔记见 [[LCPU AI Infra Seminars - Session 06 - AI Communication Stack]]。
>
> 本讲前两部分较细，后面为多项研究与系统工作的快速概览。这里不把课件中未逐项展开的细节补写成现场发言，也不把讲者对新系统的介绍当作独立验证的产品结论。

## 术语校正

| 原字幕误识别示例 | 校正 |
|---|---|
| nick / 尼克 / RNICK | NIC / RNIC |
| composition / COITION（完成语义） | completion |
| viability / AVISIBILITY | visibility |
| placement / replacement 混写 | placement，目标放置 |
| SKYUP / Scout / SCP | scale-up / scale-out，按语境区分 |
| M2C / MC | MRC |
| 优酷 / UCAL | UCCL |
| falon | Falcon |
| ultra internet / eastern | Ultra Ethernet / UET |
| NICO / NICOX / C船 | NCCL / NCCLX / CTran |
| monkey / do pass / high sparse | Mooncake / DualPath / HiSparse |
| content rights | counted writes |
| IBTA / IPTDA | IBGDA |
| real / torrent / fight tree | rail / torus / Fat Tree |
| celebrates / rock | Cerebras / Groq |

## [00:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=0) 为什么通信很难用一套固定套路讲清

通信是一个很大的主题，而且高度绑定硬件与 workload。具体场景下的问题往往非常具体，也常常来自人为设计的边界。本次不是一套逐步调优教程，而是从第一性原理出发建立通信栈的整体印象，帮助大家遇到问题时知道该查哪一层。

讲者也提醒，相关实验可能需要大量设备；许多系统判断不如单卡 kernel 容易验证。应带着批判性理解这些设计，再用自己的场景检查。

## [02:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=120) 从 Tensor Core 的数据供给问题开始

Tensor Core 算力不断提高，系统性能越来越取决于能否及时供应操作数，并把结果交给下一阶段。围绕计算单元的 data movement、synchronization 和 pipeline，已经成为关键问题。

从早期由线程经寄存器搬运，到 Ampere 的异步 global-to-shared 拷贝，再到 Hopper 的 TMA，部分数据搬运与地址生成职责逐步交给专用硬件。Blackwell 的 TMEM 又改变了计算与存储的组织边界。

这些演进不是凭空添加特性，而是在计算增长、数据供给和控制成本之间重新划分责任。

> [!note] 技术校正
> `cp.async` 的关键之一是支持异步 global-to-shared 搬运并避免 payload 经通用寄存器中转；不能把现场简略表述理解为它完全消除了线程地址计算。张量描述符式地址生成是后续 TMA 的重要特点。

## [07:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=420) 异步以后，“线程执行到这里”不再等于数据 ready

异步操作由后台代理执行。生产者线程已经提交指令，不代表数据搬运或计算已经结束。Consumer 需要知道后台操作是否完成，以及数据是否满足可见性和访问顺序要求。

因此同步不只是让线程集合到达同一位置，还包括数据与事件的同步。不同执行 proxy 之间，也不能无条件假定自动建立顺序。具体指令组合要使用其规定的同步机制。

## [10:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=600) Pipeline 改变的不只是代码排列

Pipeline 会改变 producer/consumer 关系、buffer 生命周期、同步发生位置和阶段重叠方式。源代码顺序也不能无条件等同于最终可见顺序；编译器和硬件的合法重排需要由内存模型约束。

一个逻辑上可并行的操作，仍可能与其他操作争用同一硬件资源。重叠既要看依赖，也要看资源。

## [12:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=720) 核心问题：谁决定执行，谁保存状态

前面几讲背后的共同问题是：谁决定工作在什么时候、什么资源上执行？谁维护依赖、推进执行，并判断阶段是否真正完成？

责任可以放在硬件、编译器、DSL、runtime 或用户代码中。越靠近硬件，可能获得更短反应时间，但增加芯片面积、固件复杂度与协议耦合；放在软件则更灵活，却付出调度、计算资源和维护成本。

细粒度重叠也一样：它增加潜在并行度，同时增加状态与同步开销。

## [15:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=900) 从单 GPU 扩展到跨设备

单 GPU 内，我们已有 warp scheduler、scoreboard、barrier、copy engine 和存储层次。扩展到多 GPU 后，producer 与 consumer 可能跨越 NVLink/NVSwitch、PCIe、RNIC、RDMA transport 与交换机。

原本的 readiness、ordering、completion、visibility，现在必须跨设备、地址空间、协议和故障域继续成立。不同服务器和组网的支持程度不同，不能只凭一个统一 API 假定底层完全一样。

本讲追踪两个问题：一个 tensor chunk 从 A 到 B 经过什么路径？B 凭什么知道它已经可以安全使用？

## [19:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=1140) 硬件图里的不同互联，不是同一层含义

讲者借 Grace/Hopper/Blackwell 等示意图区分 die-to-die、CPU–GPU 的 C2C、GPU–GPU 的 scale-up fabric，以及连接其他服务器的网络。

逻辑上的一个设备、一个节点或一个 scale-up 域，可能跨越多个物理部件。新的硬件机制还可能减少某些独立通知或计数开销，但不能只看峰值带宽数字，要看它改变了哪一段依赖和控制路径。

## [22:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=1320) 五个不同的问题：到达，不等于可消费

判断数据是否可用，至少要区分：

1. Delivery：所需片段是否到达约定端点。
2. Placement：是否写到正确目标位置。
3. Completion：某一层操作的义务是否完成。
4. Visibility：目标 consumer 在要求的 scope 下是否看得到。
5. Ordering：payload、signal 与消费操作之间是否满足必要顺序。

这五项是语义条件，不是固定串行的五段物理流程。一个机制可能同时满足多项，也可能需要多个协议共同完成。

本地发送完成、传输 ACK、远端可读与应用消费完成，是不同层面的事实，不能混成一个“通信结束”。

## [25:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=1500) 状态和控制不会消失，只会换位置

系统需要判断丢失、乱序和拥塞，推进传输，发布完成，保证可见性，并在无法恢复时报告错误。相关状态可以在 kernel、library、transport、NIC、DPU、switch 或集群控制面中。

硬件承担这些事也要占资源和功耗；软件承担也要占 CPU/GPU 时间。所谓 offload，应问原来的关键路径是否真的缩短，而不只是工作换了一个执行位置。

## [28:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=1680) 六个话题与讲述重点

课程依次讨论语义、硬件路径、状态归属、workload、大规模系统和硬件边界展望。重点是前两个：把“可消费”拆开，并看清实际路径。

后面的方案比较与案例带有讲者自己的分析视角。通信很依赖具体前提，某种 CPU 开销在一个 workload 中可以接受，在另一个低延迟场景中却可能不能接受。

## [31:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=1860) 从执行图到 consumer：一层调用隐藏了什么

Workload 的并行策略和执行图决定通信需求。上层可能调用 collective、P2P 或 EP library；库选择算法、chunk 和 channel schedule，再由 CPU/GPU 发起和推进。下层还包括 residency、staging、注册、DMA、可靠性、拥塞与路径控制。

最后，数据要正确 placement，并通过完成发布和 consumer 同步，才能接回执行图。对调用者透明，不等于这些成本不存在。

## [36:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=2160) Workload、scaling 与新硬件：先建立问题地图

讲者先预览后面内容：RC、MRC、Falcon、UCCL 和 UET 的责任边界不同；collective、MoE、KV movement 和 distributed kernel 的等待关系也不同。

大规模下，初始化、QP/连接状态、带宽时延积、流控、故障恢复和可观测性会成为主要成本。进一步改变硬件边界，也不能只问少了哪次跨芯片搬运，还要问新瓶颈变成了哪里。

## [43:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=2580) 通信栈是一套可检查的交接约定

通信栈把 producer 到 consumer 的依赖，变成一套能够检查、推进、恢复的 readiness contract。它保证数据在正确位置，满足所需顺序和可见性，再交给消费者。

不同系统的根本区别，是这套约定的状态和控制边界放在哪里，以及相应成本是否符合 workload。

## [44:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=2640) 不要只测空载带宽与平均延迟

端到端通信时间还包含 source ready、launch/progress、staging、排队、传输、目标放置、完成通知、consumer wait 和故障恢复。空载时看不到的队列与争用，在真实压力下可能占主导。

同步 workload 往往由最慢的 rank 或 channel 决定。平均值很好，不代表训练 step 或请求延迟很好；应关注 P99、P999 和最大必要依赖。ECMP 冲突、热点 expert、重传以及 CPU/HBM 争用都可能制造 straggler。

## [47:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=2820) 可消费还要求 buffer 的身份和寿命正确

即使传输与可见性满足要求，目标 buffer 已经被下一轮覆盖，消费仍然是错的。Source 何时允许复用，destination 何时允许回收，以及对象在哪个 generation，必须有明确协议。

Serving 可能希望 KV 长时间存活以便复用；训练则可能通过重计算换显存。生命周期策略本身就是 workload 的一部分。

Payload、control metadata 与 completion notification 要在逻辑上区分，即使它们物理上捎带在同一条消息中。

## [51:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=3060) Delivery 与 Placement

Delivery 的观察点可能是 RNIC 或 scale-up endpoint，不一定是最终 HBM 地址。允许乱序时，还需要知道哪些片段已经到达、哪些缺失。

Placement 关心 bytes 是否位于正确 address、offset 和 generation。允许分片或乱序写入，不意味着最终逻辑内容可以乱。中间 staging、reorder 与分片提交，都可能处在这个过程中。

## [54:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=3240) Completion 必须带具体作用域

发送 buffer 可复用、transport 完成、远端目标写入完成、额外 endpoint copy 完成，以及 consumer 获得可用信号，可能是多个不同事件。

因此必须读具体 API/verb 的契约：一个 CQE 或 ACK 究竟保证什么？不能把任何一层 completion 都当作远端应用已安全消费的证明。

## [56:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=3360) Visibility、Ordering 与旧信号

Visibility 相对于 consumer 和 memory scope 定义。源码中先写 payload、再写 flag，如果没有正确的跨代理、跨设备同步契约，就不能仅凭看到 flag 推断 payload 已经可用。

要区分 arrival order、placement order、publication/visibility order 和 execution order。接收端需要通过适当的 signal、acquire、fence 等机制衔接依赖；具体组合依平台和 API 而定，不是随便加一个 CPU fence 就能解决。

还要防止上一轮迟到的 packet 或 signal 被当成当前轮完成。Phase、epoch 或 generation 的设计，本质上是在给复用的存储和通知加身份。

## [60:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=3600) 节点内拓扑并不统一

传统 PCIe 服务器、DGX/HGX、Grace Hopper/Blackwell 和更大的 scale-up 系统有不同结构。即使都是八卡节点，专用互联、switch、CPU socket 和 NIC affinity 也可能不同。

拿到陌生机器，应先查看 topology，再判断理论上可用的最快路径。逻辑 rank 还需要映射到物理 GPU、NIC、NUMA 与 rail，不能只看 API 层的设备编号。

## [64:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=3840) 三种节点内路径：专用 fabric、PCIe P2P、host staging

专用 scale-up fabric 可在 GPU 间移动数据，不让 payload 经 host DRAM 或外部 NIC。没有专用互联时，若平台、IOMMU 与 peer-access 条件允许，可走 PCIe P2P。

条件不满足则可能 fallback 到 host staging，增加 GPU↔host 等搬运、DRAM/PCIe 使用与多段完成等待。这通常代价更高，但要先确认实际路径，不能只凭服务器名字猜测。

## [68:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=4080) 拿到一台服务器，要问哪些问题

哪些组件构成高带宽、低延迟域？域内带宽和延迟是否均匀？GPU 与 NIC 有怎样的 affinity？哪条链路存在 oversubscription？离开一个域以后，地址权限、ownership、completion、ordering 和故障边界发生了什么变化？

Oversubscription 是成本与容量设计的结果。即使静态端口比例没有收敛，特定流量矩阵仍可能出现瞬时热点。

双 die 封装还提醒我们：一个 logical GPU 不等于所有物理访问距离完全相同。但发现非均匀性与真正利用它获得收益，是两件事。

## [72:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=4320) 从 node 到 cluster：RNIC、switch 与端点职责

跨节点 GPUDirect RDMA 的典型路径，把 GPU 内存与 RNIC、网络和远端 GPU 内存连接起来。RNIC、switch、DPU 各自维护的状态不同；DPU offload 是否有效，要看它如何改变关键路径。

物理上跨机柜的 scale-up 系统，逻辑上仍可能提供接近节点内的访问方式。因此“节点内/节点间”不能只按机箱外观或线缆划分。

## [75:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=4500) 同一个 collective 可能同时使用两层网络

一次 RC RDMA WRITE 可拆成 setup、post、权限检查、source DMA、分片与调度、remote placement、ACK/retirement，以及发布给 consumer 的同步链。划分这些步骤，是为了知道每一层状态由谁保存。

分层 collective 或 DeepEP 的一些路径，会先在带宽更高的节点内整理数据，再使用较低带宽的跨节点域，降低昂贵路径上的流量。收益与实际拓扑、带宽比例和 workload 有关，不能把某个课堂比例当作通用常数。

## [77:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=4620) Fat Tree：路径多，不等于用得均匀

必须区分 path diversity、path selection 与 capacity balance。有多条等价路径，不代表一个 flow 自动用上全部路径；长流的 ECMP hash collision 可能让某条 uplink 拥塞。

Adaptive routing 和 packet spraying 可以改善利用率，却把乱序、gap tracking、replay 和恢复状态推到端点或 transport。优化没有让状态消失，只是重新分配了成本。

Non-blocking 的静态结构也不能保证所有瞬时 traffic matrix 都没有排队。Collective 又会把局部慢点放大全局尾部。

## [81:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=4860) Mesh、Torus 与 rail-optimized

TPU 的 mesh/torus 提供另一种互联组织：通信沿邻居边行进，partition shape 与 workload 决定路径长短和共享关系。

Rail-optimized 则强调 GPU–NIC–rail affinity 与 rank 映射，适合某些相对规则的流量。动态跨 rank 需求可能需要节点内转发或 PXN，不能保证任何 workload 都更快。

Rail 是映射与组织策略，Clos/Fat Tree、mesh/torus 是拓扑类别，不应把它们当作完全同一层且互斥的选择。

## [86:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=5160) Scale-up 与 scale-out 的真正边界

通常 scale-up 域内 RTT 更短、局部故障处理更紧密；scale-out 引入更多网络状态和恢复责任。但这不是仅由接口名字决定的硬分类。

真正要问的是：ownership 在哪里交接？Consumer 怎样证明数据已经可用？故障能否被限制在局部？不同新协议对边界的处理仍在演进。

## [90:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=5400) 从这里转入方案概览：memory semantics 与 verbs

前两部分建立了语义与架构基础，后面快速看几类设计。Memory/transaction 风格与 RDMA verbs 的表达方式、状态归属和错误处理并不相同。

Delivery 常由 link/transport 跟踪，placement 由 endpoint/DMA 执行，completion 由相应层发布；visibility 和 ordering 则必须与 memory system、consumer kernel 的同步契约衔接。

这些是逻辑职责，不意味着任何产品都恰好一层对应一个独立硬件模块。

## [95:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=5700) RC 与 IBGDA：减少 host 介入不等于没有协议

传统路径中的提交、progress、completion 处理可能有 host 参与。IBGDA 将部分相关控制和队列交互移到 GPU，降低一些 host round trip；DeepEP 等工作让这一方向受到更多关注。

但 device-initiated 描述的是发起和控制方式，不代表可靠性、NIC 状态和内存同步都消失。仍需逐项检查谁推进、谁等待、谁发布完成。

## [96:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=5760) MRC、Falcon、UCCL、UET：比较责任边界

MRC 关注多路径与路径故障恢复，希望某些网络故障不必直接中断训练。Falcon 是硬件辅助 transport，与可编程软件共同构成控制闭环。

UCCL 更强调软件层策略和不同后端支持，部分工作由 CPU 协助；这并不天然错误，关键是其 workload 是否能接受相关成本。

UET 则首先是互操作契约与规范，不能把规范当成一种唯一实现。状态究竟在 NIC、host 或其他处理器上，仍是实现者的选择。

## [100:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=6000) 用一次故障检查系统设计

谁观察异常？谁保存状态？谁限制新流量、绕路或重传？谁发布 completion/error？最终谁恢复 runtime？

把这些问题对同一种故障问一遍，就能看到方案真正的差异。反应速度、硬件面积、CPU 开销、可编程性与恢复复杂度之间，往往没有单向最优。

## [102:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=6120) Workload 不只产生 bytes，也产生等待

AllReduce、pipeline parallelism、MoE dispatch/combine、KV movement 与 distributed kernel，虽然都涉及搬运，但同步关系和尾部成本不同。

PP 下游 stage 必须及时拿到 activation，否则出现 bubble；KV 传输还要考虑对象在多级存储中的位置与可读状态；MoE 则带来动态 routing、incast 与计算偏斜。

## [104:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=6240) Collective：reduction 在哪里发生

Ring AllReduce 的常见模型中，每 rank 发送量约为输入大小的两倍（精确系数见笔记）。把 reduction 放到 switch/NIC 或其他专用单元，可以改变流量与 GPU 工作量。

但浮点加法不满足严格结合律。归约次序和位置变化后，要检查精度、误差容忍及可复现性，不能只比较通信速度。

## [106:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=6360) P2P 搬运的是有生命周期的对象

只说“把一段地址范围搬到另一处”，还不足以管理 activation 或 KV。对象何时产生、何时可读、由谁持有、何时失效，也必须纳入协议。

PP 与 PD 分离都涉及 P2P，但等待关系、可隐藏时间与优化目标不同。生命周期管理做错，可能读到旧数据；做得过于保守，又会浪费显存或吞吐。

## [107:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=6420) Mooncake、HiCache：把“怎么搬”和“放哪里”分开

多级存储需要高性能数据移动，也需要知道对象在哪个介质、哪个位置。Mooncake 等工作把传输能力与对象管理解耦；推理框架还可能有自己的 host/KV 管理层，例如 HiCache。

抽象隐藏了后端差异，不意味着 metadata、注册、分配、缓存策略和工程调优成本消失。

## [109:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=6540) DualPath：利用原本闲置的存储入口

讲者介绍 DualPath：如果只有 Prefill 一侧从存储网络读 KV，Decode 一侧的存储 NIC 可能闲置。把两侧入口都利用起来，有机会提高可用带宽。

但还需要考虑 relay、数据路径和 QoS，避免 KV movement 干扰更延迟敏感的通信。这也引出存储网络、compute network 与 scale-up fabric 能否协同设计的问题。

## [111:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=6660) HiSparse：把容量留在 host，把热点留在 HBM

稀疏 attention 不一定每步都读取全量 KV，因此可让较完整的 KV 留在 host，GPU 保存当前所需 hot blocks，miss 时再 swap in。

如果 top-k 变化剧烈、预取不准或 hot pool 太小，搬运延迟和 thrashing 会暴露出来。思想类似传统存储层次，真正困难在于正确且高效的工程实现。

Direct-to-host 还可让远端 KV 直接进入 Decode 侧 host tier，避免不必要的 GPU staging；具体哪段 buffer 被省掉，要结合实际路径判断。

## [113:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=6780) MoE：dispatch/combine 之外还有很多工作

EP 还包括 routing metadata、计数、packing、接收布局、grouped GEMM 与最终合并。热点 expert 会造成负载不均，通用而规则的 All-to-All API 未必能表达所有优化机会。

DeepEP 等工作展示了理解 workload 的通信库能做什么；high-throughput 与 low-latency 模式针对不同消息和批量条件。讲者也概览了后续后端变化，但具体实现应固定版本核实。

## [116:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=6960) 复制热点 expert：节省等待，也要付复制成本

一些设计动态或提前复制热点 expert，试图改善 token/compute 负载均衡。代价包括权重传输、额外显存、同步和训练中的梯度归并。

Replica placement 必须考虑 scale-up 域和可用带宽。负载均衡收益是否覆盖复制成本，要从完整 dispatch → GEMM → combine 的关键路径评估。

## [118:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=7080) 细粒度 overlap 不是免费收益

把本地 pipeline 的思想扩展到多 GPU，可以更早发布 chunk、重叠通信与计算。但粒度越细，状态、通知和固定开销也越多。

PDL 等机制可能减少部分 kernel 间等待；counted writes 等机制可能把数据写入与完成计数关联，减少独立消息。它们仍有具体契约，buffer/counter 复用也仍需防止旧一轮污染。

## [120:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=7200) Distributed kernel：把跨设备依赖带进执行系统

Distributed kernel 让多个设备围绕同一任务和数据流细粒度协作。它要明确 payload、completion、lifetime 与 progress 责任，不能只把几段 kernel 拼在一起。

如果为了隐藏通信而显著降低计算效率，端到端未必有收益。MegaMoE 等案例说明深度融合可能有效，但评估对象仍是总时间和资源，而不是 timeline 上是否出现重叠。

## [123:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=7380) 大规模：初始化、恢复与诊断进入关键路径

讲者快速比较 MRC、NCCLX/CTran 与大规模通信栈工作：分别关注路径故障、通信库/transport，以及初始化、资源、流控和运维。

高 BDP 或深队列本身不必然是问题；当它们导致资源耗尽、排队和尾部扩大时，才需要相应控制。控制面由 CPU 执行是否可接受，同样要按 workload 判断。

零拷贝可能把成本转移到用户 buffer 的分配与注册契约上，减少 copy 不等于没有其他工程代价。

## [127:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=7620) 扩大 scale-up 域也有成本和故障取舍

更大高速互联域能减少某些跨域成本，同时可能提高设备与交换结构成本，并改变故障影响范围。小规模时足够的状态资源，大规模时可能耗尽。

因此要同时看稳定阶段性能、恢复代价、可观测性与成本，而不是只看聚合峰值带宽。

## [129:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=7740) 推理 workload 推动新的硬件边界讨论

Decode 常面临数据供给压力，但是否 memory-bound 还取决于 batch size、序列长度、量化、并行策略与 kernel。低延迟、小批量、speculative decoding 等场景可能提出不同需求。

CPU、GPU、DPU 各承担什么，KV 放在哪一层，都可以重新讨论；这些是设计空间，不是单一正确答案。

## [131:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=7860) Wafer-scale、静态调度与 warm KV 层

Cerebras 把更多搬运留在 wafer 内，但仍受片上容量和 off-wafer I/O 约束。Groq 式静态调度减少部分动态控制，却增加编译器和对 workload 可预测性的要求。

Warm KV/context tier 可以缓解 HBM 容量压力，但集中式池与分布式池如何选择，还涉及 metadata、预取、失效和共享资源。讲者把这些作为待评估方向，而不是已经达成共识的结论。

## [133:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=7980) HBF、PNM、3D stacking 与低延迟互联

更大容量存储、近存/存内计算、垂直互联和低延迟链路，可以改变数据移动距离与能耗。但也引入粒度、延迟、寿命、热设计和接口约束。

NetDAM 等研究原型可作为重新划分通信与存储责任的参考。每次移动边界，都要继续追问：原来的状态真的消失了吗，还是由另一层承担了？

## [135:00](https://www.bilibili.com/video/BV1FYhL6REgi/?t=8100) 收尾：把框架用于读原文，而不是替代原文

大量工作无法在一次讲座中展开，感兴趣可继续阅读相关 technical report、协议、开源实现和硬件研究。通信基础设施本身也在快速演进，研究某个缺点前，应确认新版本是否已经改变了问题。

最后邀请提问，录播中未出现实质问答。本讲留下的主线是：沿着 producer → consumer 追踪数据、状态、顺序和生命周期，判断每一个系统边界承担了什么。

## 原始来源留档

- [视频](https://www.bilibili.com/video/BV1FYhL6REgi/)，02:17:11；[官方课件](https://infra.seminars.lcpu.dev/slides/workshop02.pdf)。
- 原始字幕文件名包含 `BV1FYhL6REgi_字幕.srt`，保留于 Downloads，未改写。
- SHA-256：`8257b8b287592a01b8c908af23fd23a2bb45a7756c26617cb973dbcefe06b05a`。
