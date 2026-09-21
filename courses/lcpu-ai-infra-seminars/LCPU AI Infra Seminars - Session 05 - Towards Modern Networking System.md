---
type: course-note
status: developing
source_status: slides-only-user-skipped
course: Weiming HPC Training Camp × LCPU AI Infra Seminars
lecture: 5
lecture_date: 2026-08-17
slides_date: 2026-08-15
speaker: 王志豪
area: systems
topics:
  - networking
  - flow-control
  - reliability
  - pcie
  - rdma
  - scale-up
  - scale-out
  - tile
aliases:
  - LCPU Session 05
  - Towards Modern Networking System
video_url: https://www.bilibili.com/video/BV1Xg836FE7G/
slides_url: https://infra.seminars.lcpu.dev/slides/session05.pdf
---

# Session 05：Towards Modern Networking System

> [!abstract] 核心问题
> 从两端通过导线交换数据，到多 GPU、跨机柜系统，网络始终在回答：接收端能否承受、失败由谁恢复、状态归谁管理、完成意味着什么。本讲用握手、credit、重放窗口和虚通道建立基础，再提出面向 tile 的网络分层与设备发起接口。

> [!info] 当前整理状态
> 2026-09-18：因插件无法提取字幕，用户已要求跳过第 5 讲，不再进行音频转写或后续整理。保留此前依据官方 79 页课件写成的草稿，但它不是完整录播笔记，也没有配套精编字幕，不属于本轮交付。视频时长 02:45:16。下文页码含封面；活动日历为 2026-08-17，课件封面为 2026-08-15，分别记录。

## 来源与导航

- [视频](https://www.bilibili.com/video/BV1Xg836FE7G/)
- [官方课件](https://infra.seminars.lcpu.dev/slides/session05.pdf)
- [课程日历](https://infra.seminars.lcpu.dev/schedule)
- 课件作者：Zhihao Wang。后半部包含讲者的架构提案与判断，下文按提案理解，不将其写成已普遍实现的行业标准。

| PDF 页 | 主题 |
|---|---|
| 4–5 | 全讲术语：domain、connection、transaction、tunnel、retirement |
| 12–23 | Wire、VALID/READY、长链路、credit、可靠性 |
| 24–26 | PCIe credit / replay / cut-through 案例 |
| 28–35 | Router、arbitration、HOL、virtual channels、packet/flit |
| 36–42 | 网络拓扑、因果、deadlock、Orderlock、InfiniBand |
| 44–52 | Domain、scale-up/out、状态位置、同步、SQ/CQ |
| 53–59 | Ethernet、IP、TCP 的分层职责 |
| 61–68 | 研究视角变化、现代负载与传统 QP 的耦合 |
| 69–71 | 提案：语义、连接、执行、事务、隧道、路径六层 |
| 73–78 | Stateful operations、tile descriptors、xPU issue、统一系统 |

## 1. 从导线开始：发送能力不等于接收能力

最小握手用三组信号表达数据与双方状态：DATA 保存值，VALID 表示发送方提供有效值，READY 表示接收方能够接收。真正交接发生在：

$$
transfer=VALID\land READY
$$

当下游不能接收时，上游要遵守保持数据和有效性的协议。不能仅凭自己“已经输出过”就认定交接成功。

线路变长后，反压需要时间返回。接收端发出“停”时，路径上还有在途数据；流控因而同时是一个容量和反馈延迟问题。

## 2. Credit 与重放窗口在两端保留不同的状态

| 机制 | 发出前需要什么 | 主要保留在哪里 | 解决什么 |
|---|---|---|---|
| Credit / admission | 可用容量的许可 | 接收缓冲与发送方 credit 账本 | 防止超过承诺接收容量 |
| ACK / retry | 可先发送，之后等待确认 | 未确认数据或可重放源状态 | 发现并恢复丢失/损坏 |

Credit 关注接收容量，retry 关注可靠交付。一个网络可以同时使用它们。Lossless 不意味着链路永不误码，也不意味着端点永不失效。

> [!note] 整理者推导：窗口为什么与 RTT 有关
> 若有效发送速率为 $B$ bytes/s，释放 credit 或确认所需往返时间为 $R$ 秒，要持续保持发送，相关在途窗口通常至少需要覆盖 $B\times R$ 的量级。举例：$50\,\mathrm{GB/s}\times10\,\mu s=500\,\mathrm{kB}$。这是说明量纲的假设例子，实际还要考虑包粒度、突发和协议预留。

窗口过小会让链路周期性等许可；盲目扩大窗口又会增加缓冲、排队和恢复状态。Little's Law 解释了为什么高带宽往往伴随更大的在途状态。

## 3. PCIe 案例：无损流控与可靠传输可以共存

课件讨论的 PCIe link-layer 模型中，credit 按流量类型和 VC 维护，用来限制发包；序号、CRC、ACK/NAK 与 retry buffer 负责链路错误后的恢复。两者职责不同。

Non-FLIT cut-through 场景说明了流水与验证之间的张力：交换机可能在入口包尾到达、CRC 最终确认之前，已经向下游发出部分内容。若随后发现错误，需要协议定义如何使已转发内容失效，以及在哪一跳重放。

这类机制不能仅凭“某处看到字节”认定完成。要明确检查的是发出、接收、校验、交付还是退休；也不能把某一代/模式的细节推广到所有 PCIe 版本。

## 4. Router 增加了共享资源与仲裁

将多条链路的接收端、FIFO、发送端连接到 crossbar，就产生路由器。问题从“能否传过去”扩展为多个输入同时争用一个输出时，谁先使用，以及公平性如何保证。

### HOL blocking

FIFO 队首的目标输出被阻塞，后面的包即使可以去空闲输出，也无法越过它。共享队列因此可能损失交换能力。

### Virtual channels

在一个物理通道上维护多个逻辑队列/状态，可以隔离某些阻塞关系；也可以将请求与响应放到不同资源类别，破坏死锁依赖环。但 VC 并不保证任意 routing 或 arbitration 都无死锁；仍要检查 channel dependency 和资源预留。

课件还介绍 DAMQ：缓冲空间可共享，各 VC 用 head/tail 等状态表达自身队列。因此“有多个 VC”不必意味着每个 VC 都拥有固定、完全独占的大块 SRAM。

## 5. 顺序是需要付费的协议属性

独立请求可能经过不同路径，耗时不同。源代码的先后发起，不自动给出远端观察到的完成顺序。若必须建立因果，可以等前一项返回后再发后一项，也可以通过序号、依赖标记和接收端提交规则表达。

强制串行通常减少并行性；允许乱序则增加跟踪缺口、重排和提交的状态。

| 问题 | 特征 | 分析入口 |
|---|---|---|
| Congestion | 需求暂时高于服务率 | 排队、背压、带宽分配 |
| Deadlock | 存在无法解除的资源等待环 | wait-for / channel dependency graph |
| Livelock | 持续活动却无法完成有用工作 | 重试、路由和调度的进展保证 |
| Orderlock | 排序条件与有界缓冲导致自阻塞 | 缺口、乱序持有、交付顺序与流控 |

课件借 Orderlock 工作说明：某些同时要求无损传输、保序递交、允许跨缺口持有乱序数据的组合，在有限缓冲与相应模型条件下会产生结构性问题。复习时应保留这些前提，不能简化为所有网络都通用的“三选二”定律。

## 6. 管理域与一致性域要分开看

Management domain 关注哪一范围内，系统有权定义资源、地址映射、权限和生命周期。一致性域关注哪些 agents 的访问遵循某种 coherence 契约。更大的管理域可以包含多个一致性域。

因此 rack-scale GPU fabric 不等于整个机柜运行单一 OS，也不等于所有内存访问都自动 cache coherent。具体的 peer access、映射、ordering 和故障规则仍需分别建立。

Scale-up / scale-out 也不只是距离单位。跨越边界后，地址表达、连接创建、故障半径、可靠性和完成语义可能改变。

## 7. SQ/CQ：把异步设备工作组织成有生命周期的对象

典型队列路径：

```text
Host 构造 SQ/WQ descriptor
  → MMIO doorbell
  → Device 读取任务并执行
  → 写入 completion record
  → Host polling 或 interrupt
  → 处理完成并释放相关资源
```

Doorbell 是通知机制，不是 payload 本身；CQ record 是某层工作完成的证据，其含义由 API 规定。中断与 polling 的优劣取决于负载、延迟要求、事件频率和 CPU 预算。

讲者用音频的稳定周期说明：若工作时间高度可预测，可用计时与批处理降低频繁中断成本。这个案例支持“根据工作规律选择通知方式”，而不是得出 polling 总优于 interrupt。

## 8. Ethernet、IP、TCP 各自承担一部分职责

| 层次 | 本讲强调的职责 | 不应默认得到的保证 |
|---|---|---|
| NIC / DMA / queues | 主机内存与网络端点连接 | 应用级对象已可消费 |
| Ethernet | 成帧、链路寻址、交换与校验 | 全路径可靠、应用顺序 |
| IP | 层次化地址与跨网络转发 | 必达、可靠、有序 |
| TCP | 端点可靠有序字节流与拥塞控制 | 应用消息边界、远端业务已执行 |

层次化命名让路由表可聚合；可靠性可以放在端点，避免每一层都维护完全相同的全局状态。不同 workload 的粒度、寿命和失败代价不同，因此也没有适合所有流量的单一最优协议。

## 9. 讲者提案：按生命周期拆开传统 QP 中耦合的状态

课件指出身份、序号、重传窗口、路径与异步队列具有不同生命周期。若强绑定在一个连接对象中，扩连接、多路径和故障恢复会互相牵制。

其提出的六层结构为：

| 层 | 主要责任 |
|---|---|
| 语义 | 对上提供 operation / task 的含义 |
| 连接 | 权限、映射、资源和生命周期 |
| 执行 | 将大 move / tile 操作拆成有界工作 |
| 事务 | 工作身份、准入、执行和退休 |
| 隧道 | 一段可靠传输与拥塞控制资源 |
| 路径 | 实际 packet forwarding / routing |

这里的 reliable tunnel 是课件定义的传输资源，不应直接理解成普通 overlay encapsulation。Transaction 也不是数据库 ACID 事务。

### Move、transaction、fragment、instance

- Move：完整的搬运任务，可以很大。
- Transaction：身份稳定、大小有界、可确定退休的一段工作。
- Fragment：transaction 内可独立定位和重放的片段。
- Instance：某个 fragment 的一次具体发送。

把 fragment 身份与发送路径分开，可让一次重传更换路径而不改变逻辑操作身份。这需要幂等放置、去重、权限与生命周期共同成立。

### 完成与未知结果

课件提案用 local retirement 表达发送方收齐确认、解除源数据保留义务。Terminal result 分为成功、确定拒绝和结果未知。超时只说明本地未及时获得充分证据，不能推出远端没有执行。

对非幂等操作，结果未知尤其重要：直接重试可能重复产生效果。协议需要稳定身份和重放语义，或者把恢复决策交给上层。

## 10. 面向 tile 的接口：CPU setup，xPU issue

Tensor Core、AMX/SME 和 tile DSL 都让 shape/stride 成为常见接口信息。讲者据此提出：通信也应更自然地表达 tile movement，而不必每次都绕回 CPU 队列管理。

设想中的分工是 CPU 建立连接、映射与权限，xPU 在数据 ready 后提交 tile/descriptor 操作，再用 barrier、commit/wait 或 atomic 协调依赖。

> [!note] 整理者理解
> 这与 Session 04 的 TMA 相呼应：保留更高层的 shape/layout 描述，可以减少重复地址生成和控制工作。但搬到远端后，还新增权限、网络故障、重传和 unknown result；单 GPU 的异步搬运契约不能直接原样覆盖跨节点通信。

课件最终把 compute cluster、system cache、DRAM、DMA、aggregation/reduction 和 NIC 放入一个整体设计视图。其价值是让计算、存储和互联共同设计，而不是只用更大的端口速率描述系统进步。

## 11. AI Infra 视角

| 视角 | 本讲对应问题 |
|---|---|
| Shape | Scalar/message/tile 粒度；如何分片和聚合 |
| Compute | 哪个设备执行地址生成、归约和协议工作 |
| Memory | RX buffer、replay window、NIC SRAM、映射表 |
| Communication | Flow control、retry、VC、routing、路径利用率 |
| Runtime/System | Domain、连接授权、retirement、超时与恢复 |

与 [[LCPU AI Infra Seminars - Session 04 - Pipeline Ordering Data Orchestration]] 的共同主线是足够在途工作、明确 ownership 和可验证 completion。[[LCPU AI Infra Seminars - Session 06 - AI Communication Stack]] 进一步追踪这些责任如何落在 GPU、runtime、RNIC 和 fabric 中。

## 12. 复习检查与自测

分析任一通信设计时，写清数据身份、源/目标地址、权限、buffer ownership、flow-control window、completion owner 和错误终态，然后沿一次真实数据交接验证。

1. 为什么长线上的 READY 不能立即阻止所有后续数据到达？

    **面试回答：** 反压信号传播需要时间，接收端撤销 READY 时，发送端可能还没看见，链路上也已有在途数据。因此接收端必须为反馈延迟内继续到达的数据留出容量，或使用 credit 预先限制发送量；高速长链路尤其受带宽时延积约束。

2. Credit 与 ACK 分别证明了什么？

    **面试回答：** Credit 证明接收端承诺了相应缓冲容量，解决“现在能不能发”；ACK 证明协议规定层面的数据已被确认，帮助退休发送状态或触发重传判断。ACK 不必表示远端应用已处理完成，credit 也不证明先前数据已可靠交付。

3. 为什么 lossless 与 reliable 是不同维度？

    **面试回答：** Lossless 通常指在指定流控和容量条件下避免拥塞丢包；reliable 关注发现丢失、损坏或重复并按协议恢复。无拥塞丢包仍可能发生链路误码和端点失败，而允许丢包的网络也能通过端到端重传提供可靠交付。

4. VC 怎样缓解 HOL？为什么 VC 数量增加不自动保证无死锁？

    **面试回答：** VC 把共享物理链路上的流量分到多个逻辑队列，让被阻塞流量不必卡住所有后续包，从而缓解队首阻塞。死锁取决于资源等待环，增加 VC 后仍需正确路由、类别划分和资源预留；仅增加队列数量不能证明无死锁。

5. 允许乱序为什么能改善并行，却增加端点状态？

    **面试回答：** 允许乱序后，独立请求可并行推进并利用不同路径，不必等最慢的前序包。代价是端点要维护序号、缺口、重组、去重和提交状态；若上层要求有序交付，还要承担重排缓冲与等待。

6. 管理域、地址可达域和一致性域有什么区别？

    **面试回答：** 管理域定义谁有权配置资源、映射、权限和生命周期；地址可达域定义哪些 agent 能通过规定路径访问哪些地址；一致性域定义哪些缓存和内存访问遵循共同的 coherence 规则。三者边界可以不同，因此“在一个机柜”或“地址可达”都不能推出自动缓存一致。

7. Doorbell、CQE 与远端应用完成是什么关系？

    **面试回答：** Doorbell 只是通知设备队列中有工作，不代表数据已发送。CQE 证明某项操作完成了 API 所定义的义务，可能允许复用源缓冲，但不自动证明远端应用已消费；远端完成还需其发布、acquire 和业务协议提供证据。

8. 为什么路径寿命不应简单等于连接身份寿命？

    **面试回答：** 连接身份承载权限、资源和逻辑操作连续性，通常应跨越某条路径的故障或拥塞继续存在。路径可被替换、负载均衡或重建；若强行绑定生命周期，每次换路都要重建连接与上层状态，放大恢复成本。

9. Fragment 与 instance 分开对多路径重传有什么帮助？

    **面试回答：** Fragment 标识需要交付的逻辑数据片段，instance 标识它在某次、某条路径上的具体发送。重传换路时保留 fragment 身份，接收端就能把多个 instance 归为同一片段并去重；前提是目标地址、generation、幂等放置和权限仍有效。

10. 超时后重试为什么可能产生重复效果？

    **面试回答：** 超时只说明发送端未及时取得结果，原操作可能已经执行，只是确认丢失或延迟。再次执行非幂等操作可能重复扣减、累加等效果；应使用稳定请求 ID、去重/幂等语义或结果查询，并把无法确认的情况保留为“结果未知”。

11. Tile 接口可能节省哪些地址生成/控制开销？新增哪些跨域契约？

    **面试回答：** Tile descriptor 可一次表达 shape、stride 和范围，减少逐片地址计算、CPU 提交与队列控制开销，也便于硬件批量搬运。跨设备后仍需定义地址映射、权限、布局、可见性、buffer 生命周期、重传和错误终态，不能把本地 TMA 契约原样套到网络。

12. 哪些课件内容是既有机制，哪些是讲者提出的设计方向？

    **面试回答：** VALID/READY、credit、ACK/retry、VC、SQ/CQ，以及 Ethernet/IP/TCP 分层是既有机制；具体保证仍取决于协议和版本。课件的语义—连接—执行—事务—隧道—路径六层重构，以及统一 tile/xPU 发起接口，是讲者提出的设计方向，应作为提案理解。


## 待字幕补齐

- 视频章节时间、讲者口头例子与问答。
- 录播对架构提案的限定条件与背景。
- 对课件中强概括性措辞的口头补充，尤其是可靠性、同步、管理域与协议适用范围。
