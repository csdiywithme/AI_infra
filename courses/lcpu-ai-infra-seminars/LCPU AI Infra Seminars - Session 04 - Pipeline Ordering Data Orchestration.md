---
type: course-note
status: developing
course: Weiming HPC Training Camp × LCPU AI Infra Seminars
lecture: 4
lecture_date: 2026-08-09
area: systems
topics:
  - gpu
  - cuda
  - tensor-core
  - latency-hiding
  - pipeline
  - warp-specialization
  - persistent-kernel
  - grouped-gemm
  - accelerator-architecture
aliases:
  - Pipeline Ordering
  - Data Orchestration
  - LCPU Session 04
video_url: https://www.bilibili.com/video/BV1NRuX6UEwJ/
slides_url: https://infra.seminars.lcpu.dev/slides/session04.pdf
---

# LCPU AI Infra Seminars - Session 04 - Pipeline Ordering: Data Orchestration

> [!abstract] 本讲一句话
> 当 Tensor Core 的 throughput 远快于单次数据访问 latency 的改善时，性能优化的核心不再是“让一次操作更快”，而是以 reuse、足够多的 in-flight work 和严格的 stage ownership，把 copy、compute、epilogue、取下一份工作组织成可重叠的 pipeline；从 Ampere 到 Hopper、Blackwell，硬件提供了更强的异步搬运与专用存储，也把更多 ordering 责任显式交给程序员和编译器。

## 来源与范围

- [讲座视频：Pipeline Ordering: Data Orchestration](https://www.bilibili.com/video/BV1NRuX6UEwJ/)
- 视频时长：01:33:01
- 主讲：卢怡霏
- [官方 Slides（57 页）](https://infra.seminars.lcpu.dev/slides/session04.pdf)
- [课程日历与讲座简介](https://infra.seminars.lcpu.dev/schedule)
- 精编字幕：[[LCPU AI Infra Seminars - Session 04 - Pipeline Ordering - 精编字幕]]

本文以用户提供的 SRT 为主，用官方课件校正 API、指令与架构术语。重点是单 GPU 内 data movement、synchronization、buffer lifetime 和 work scheduling 的统一思维，不展开各 API 的完整语法与逐架构微观实现。

## 视频索引

| 时间 | 内容 |
|---|---|
| [00:00](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=0) | 问题定义：movement、storage、synchronization、scheduling |
| [01:54](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=114) | GEMM tile 的物理旅程；reuse 与 overlap |
| [05:26](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=326) | Baseline copy–compute loop 与两个 barrier |
| [08:18](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=498) | Cooperative Groups 与 participant contract |
| [13:09](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=789) | Async copy 的 completion / reuse 正确性 |
| [15:52](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=952) | SM80 `cp.async`、alignment、warp entanglement |
| [19:28](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=1168) | `cuda::pipeline`、double buffering、stage ownership |
| [25:59](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=1559) | Pointer chasing、scoreboard 与 Nsight Compute |
| [33:45](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=2025) | Completion、visibility、ordering、ownership 四问 |
| [36:40](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=2200) | SM90 TMA、Tensor Map 与 `mbarrier` |
| [44:07](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=2647) | Generic/async proxy 与 proxy fence |
| [50:12](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=3012) | Warp specialization |
| [53:00](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=3180) | SM100 TMEM 与 `tcgen05` |
| [55:23](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=3323) | MoE 不规则 GEMM 与 tail imbalance |
| [57:51](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=3471) | Persistent kernel 与嵌套 pipeline |
| [66:12](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=3972) | Cluster Launch Control |
| [70:03](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=4203) | Grouped GEMM 与全局逻辑 tile 空间 |
| [78:33](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=4713) | TPU、IPU、TSP 的 orchestration 对比 |
| [90:15](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=5415) | 总结：Where does uncertainty live? |

## 1. 性能问题的根：吞吐在涨，单次 latency 没有同步下降

硬件的 compute throughput 和 memory bandwidth 持续增长，但频率与单操作 latency 的改善慢得多。现代高性能系统因此越来越依赖“容忍 latency”，而不是只追求“降低 latency”。

Little's Law：

$$
L=\lambda W
$$

- `W`：一项工作从 issue 到 completion 的平均 latency；
- `λ`：目标完成速率，即 throughput；
- `L`：平均 in-flight work 数量。

当 `W` 很大且仍想提高 `λ`，只能增大 `L`。GPU 可从四处找到在途工作：

| 来源 | 并行性 | 例子 |
|---|---|---|
| 同一 warp 内的独立指令链 | ILP | 两条互不依赖的 pointer chains |
| memory system 中并发请求 | MLP | 多个 outstanding loads、prefetch、多 stages |
| 同一 SM 上的 resident warps | TLP | 当前 warp stall 时切到另一 eligible warp |
| 跨 tile/stage 的显式重叠 | Pipeline | `copy(k+1)` 与 `compute(k)` 同时推进 |

> [!important]
> “有很多 active warps”只说明工作驻留着；真正能填 issue slots 的是 **eligible warps**。Latency hiding 的直接失败信号不是某个 warp stall，而是 scheduler 经常找不到 eligible work。

## 2. 两个基本杠杆：Reuse 与 Overlap

### 2.1 Reuse 减少远端搬运次数

在 GEMM 中先把 A/B tiles 放入 SMEM，再由多个 threads/warps 重复读取，可显著降低 HBM traffic。Reuse 处理的是“总共要搬多少”。

### 2.2 Overlap 隐藏剩余搬运 latency

即便 traffic 已减少，每次 HBM request 仍有长 latency。提前发 `copy(k+1)`，在它在途时执行 `compute(k)`，处理的是“等待时还能做什么”。

两者不能互相替代：只做 reuse 仍可能在 tile 边界停顿；只做 overlap 而不复用，会让过量 traffic 把 bandwidth 吃满。

## 3. 把 SMEM stage 看成有所有权的循环队列

Baseline 中一块 SMEM buffer 需要两个 barrier：

1. Producer 写完，consumer 才能读——保护 RAW dependency；
2. Consumer 读完，producer 才能覆盖——保护 buffer reuse。

多阶段 pipeline 只是把同一契约推广到 N 个循环复用的 slots：

```text
EMPTY --producer_acquire--> WRITING
      --producer_commit--> FULL
      --consumer_wait----> READING
      --consumer_release-> EMPTY
```

可以把它当作一个有明确 ownership 的 bounded FIFO：

- producer 只拥有 EMPTY stage 的写权；
- consumer 只拥有 ready/FULL stage 的读权；
- release 是把所有权交回 producer，不只是一个性能提示。

Double buffering 的执行分为 fill、steady state、drain。Prime 和 drain 是 pipeline 的固定成本；tile 数太少时，它们占比会变高。

## 4. `wait` 不是完整的正确性证明

讲座最有价值的框架，是把容易混在一起的同步问题拆成四问：

| 维度 | 问题 | 常见机制 | 漏掉后的典型错误 |
|---|---|---|---|
| Completion | 异步操作完成了吗？ | async-group wait、`mbarrier` transaction completion | 读取尚未到达的数据 |
| Visibility | 目标 consumer 能观察到数据吗？ | memory scope、barrier、proxy fence | Producer“完成”但 consumer 仍看不到 |
| Ordering | 不同 agent/proxy 的访问有先后关系吗？ | fence、phase/token、协议顺序 | 源码顺序被误当成跨代理顺序 |
| Ownership | Storage 现在归谁，可否覆盖？ | EMPTY/FULL state、consumer release | Stage 被上一轮 consumer 尚未读完就复用 |

此外还要验证 address/range、alignment、participant set 与 generation 是否正确。一个 completed copy 仍可能是错误地址、错误 stage 或错误 generation 的 copy。

### 4.1 Cooperative Groups 的意义

`__syncthreads()` 隐含“whole block participation”。Cooperative Groups 把 participant set、rank space 和 collective contract 写进类型与接口，便于发现 scope mismatch。但 contract 仍需程序遵守：使用 block group 却只让部分 threads 进入，照样可能 deadlock。

## 5. SM80：异步 copy 出现，但调度仍由软件组织

普通路径：

```text
GMEM --LDG--> register --STS--> SMEM
```

Ampere `cp.async` / `LDGSTS` 路径：

```text
GMEM --------cp.async--------> SMEM
```

它减少 payload 经过通用 register 的显式 transit 与相关指令，让 warp 在 copy 发出后继续 independent work。

但 programmer/compiler 仍需决定：

- 哪些 threads 搬哪些地址；
- copy/commit/wait 的 participant group；
- source/destination alignment 和 copy granularity；
- SMEM layout；
- stage 数量与 EMPTY/FULL ownership；
- SMEM/register 使用量与 occupancy 的平衡；
- prefetch 要提前多少，是否真的有 independent work 覆盖窗口。

### 5.1 Warp entanglement

Pipeline batch sequence 可能是 warp-shared。Divergent lanes 若 commit 不同数量 batches，会造成 sequence skew、over-wait 与额外 barrier updates。因此相关 protocol steps 应尽量在 converged warp 上执行。

## 6. 实验：为什么 Active 不等于 Eligible

单 pointer chain：

```text
p = next[p]
```

下一轮 load address 依赖本轮 load 返回的 `p`，形成 loop-carried RAW dependency。Scoreboard 阻止 consumer instruction 读取未 ready register，于是 warp active 但不 eligible；Nsight Compute 报告 Long Scoreboard。

讲座实验的关键信号：

- 单链约 199.85 cycles / issued instruction；
- 其中约 193.1 cycles、96.6% issue interval 是 Long Scoreboard；
- 每 scheduler 约 1.00 active warp，却仅约 0.01 eligible warp；
- 约 99.5% cycles 没有 eligible warp。

加入第二条独立 chain 后，两次 loads 可同时 outstanding。单链 2.10 ms 完成每 iteration 一个 load；双链 2.81 ms 完成两个 loads，归一化 load throughput 为：

$$
\frac{2\times 2.10}{2.81}\approx 1.49\times
$$

> [!tip]
> Profiling 时不要只问“哪个 warp 在 stall”，还要问“stall 期间 scheduler 是否有其他 eligible work”。也不要直接比较工作量不同实验的 raw runtime。

## 7. Pipeline depth 的资源方程

粗略直觉是：若单 stage compute window 为 `T_compute`，copy latency 为 `T_copy`，要完全覆盖 latency，stage 提前量至少需达到相近数量级：

$$
N_{stages}\gtrsim \left\lceil\frac{T_{copy}}{T_{compute}}\right\rceil+\text{protocol margin}
$$

这是我的近似推导，不是讲座给出的精确公式；实际还受 issue bandwidth、transfer size、cache、producer 能否足够早发出等影响。

Stage 增多同时提高 latency budget 和资源成本：

```text
more stages
  ├─ (+) earlier prefetch / more MLP
  ├─ (+) longer overlap window
  ├─ (-) more SMEM per CTA
  ├─ (-) possibly lower occupancy
  └─ (-) more phase / barrier / ownership state
```

因此目标不是最大 stages 或最大 occupancy，而是让关键 pipeline 有稳定供给、scheduler 很少出现 empty issue opportunity。

## 8. SM90：TMA 把地址生成和 bulk movement 交给硬件

### 8.1 Tensor Map 是 transfer contract

Hopper 从“许多 threads 发小 copies”转向“一个 elected issuer 发 bulk tensor transfer”。Tensor Map descriptor 编码 source geometry、tile geometry、SMEM placement 及 boundary/cache policy；kernel 只需给 descriptor、coordinates、destination 与 completion object。

这减少 producer 的 instruction/address-generation work，也让 OOB fill、L2 fetch policy、SMEM swizzle 成为 transfer 本身的一部分。

### 8.2 `mbarrier` 同时跟踪 threads 与 transaction bytes

`__syncthreads()` 只能回答 threads 是否到齐，不能回答后台 TMA transaction 是否完成。`mbarrier` phase 只有在 required arrivals 与 promised transaction bytes 都满足后才完成。

Barrier object 会循环复用，必须用 token/parity 区分 generation。一个全局 `ready` 布尔值既区分不了轮次，也表达不了 transaction completion。

### 8.3 Same address ≠ cross-proxy ordering

普通 CUDA `ld/st` 走 generic proxy；TMA、WGMMA、`tcgen05` 等走 async proxy。它们即使访问同一 SMEM address，也不能自动按 source-code order 互相可见。

TMA round trip 的三条边：

```text
GMEM --TMA--> SMEM
  ① mbarrier completion：data ready

threads modify SMEM
  ② fence.proxy.async + rendezvous：generic writes visible to async reader

SMEM --TMA--> GMEM
  ③ bulk async-group read completion：source stage reusable
```

Proxy fence 不是 rendezvous；只执行 fence 不表示其他 threads 已写完，通常仍需 block-level coordination。

### 8.4 Swizzle 服务于 consumer access pattern

TMA swizzle 重排 logical coordinates 到 physical SMEM banks 的映射，目标是让 WGMMA/consumer 访问减少 bank conflict。布局选择应从 consumer access pattern 反推，而不是机械保持 GMEM 顺序。

## 9. Warp Specialization：把执行资源变成长期 agents

TMA 只需少数 issuers 后，所有 warps 执行同一 control flow 变得浪费。Warp specialization 固定角色：

| 角色 | 主要职责 | 主要资源需求 |
|---|---|---|
| Producer warp group | acquire EMPTY、issue TMA、publish FULL | 地址/descriptor/control state |
| Consumer warp groups | wait FULL、WGMMA、accumulate、release | accumulator registers、math pipeline |
| Epilogue role（可选） | convert/store result | output fragment 与 store state |

收益是减少无关指令、按角色分配 register budget，并让 producer/consumer 按各自节奏前进。底层 ownership protocol 没变：`EMPTY → TMA → FULL → WGMMA → release → EMPTY`。

## 10. SM100：TMEM 缓解 register pressure，也增加 lifetime

Blackwell 将 Tensor Core accumulators 放入 TMEM：

```text
GMEM → TMA → SMEM operands
SMEM/TMEM → tcgen05 → TMEM accumulator
TMEM → LDTM → registers → epilogue → GMEM
```

好处是 consumer threads 不必让大 accumulator 长期挤占通用 registers；代价是新增 TMEM allocation/deallocation、MMA read/write ordering、LDTM→epilogue handoff，以及 2-CTA mode 的 pair lifetime、cooperation 与 termination。

硬件代际的规律不是“新机制替程序员管掉一切”，而是：更大粒度、更异步的机制减少了 data-path overhead，同时新增更显式、更细的协议状态。

## 11. 从 tile 内 pipeline 到 workload 调度

### 11.1 MoE 为什么暴露 tail imbalance

MoE routing 让每个 expert 收到不同 token 数，产生一组 M 不同的 GEMMs。即使每个 tile kernel 很快，不同 tile 的 boundary、K、epilogue 和 cache behavior 仍不同。平均分配 tile 数不等于平均分配时间。

### 11.2 Persistent kernel：复用 worker

Persistent worker 完成一 tile 后不退出，而是从 scheduler 领取下一 tile：

```text
initialize per-worker state once
while next_work(problem_id, coord):
    rebind per-tile pointers / strides / descriptors
    run inner K-loop pipeline
    finish epilogue
    release per-tile state
drain in-flight work and exit
```

这形成两层 pipeline：

- inner：一个 output tile 内，TMA 与 MMA stages 重叠；
- outer：多个 output tiles/problems 之间，work acquisition、compute、epilogue 与 state reuse。

必须区分 per-worker、per-stage、per-tile lifetime。Persistent kernel 的风险是资源长期驻留、coexistence 下降、termination/drain 复杂。

### 11.3 Persistent ping-pong

两个 consumer groups 交替执行：一组做 tile A epilogue 时，另一组做 tile B mainloop，从而 overlap MMA 与 epilogue。代价是更多 live state 与更复杂 termination。

### 11.4 Cluster Launch Control

CLC 让 resident block “steal” 尚未启动 CTA 的 coordinate：请求成功则接管，失败则退出。它试图同时获得 fixed-work-per-block 的 load balance/coexistence 与 persistent blocks 的 state reuse。

Work request 也是异步 pipeline：request 是 producer，产生 next-coordinate；其 latency 可被 current-tile compute 隐藏，同样需要 completion 与 result-buffer ownership。

### 11.5 Grouped GEMM

把多个 GEMM 的 tile ranges 拼成全局逻辑空间：

$$
\text{tiles}_g=\left\lceil\frac{M_g}{B_M}\right\rceil
\left\lceil\frac{N_g}{B_N}\right\rceil
$$

Static round robin 的优点是无需共享 queue，worker 仅凭 `program_id` 和 `NUM_SM` 推出 `tile_id += NUM_SM`。缺点是 tile cost 不等，仍有尾部。动态 queue/CLC 可缓解，但增加同步与状态开销。

跨 group boundary 时可保留 roles、barriers、stage protocol；必须更新 problem ID、A/B/C/D pointers、strides、descriptors、predicates 与 epilogue destination。

> [!quote]
> The pipeline survives the boundary; its data bindings do not.

## 12. 四种 accelerator：不确定性被放在哪里

| 架构 | Main orchestration | 谁吸收不确定性 | 优化焦点 / 代价 |
|---|---|---|---|
| GPU | Dynamic warp readiness + explicit pipeline waits | Hardware scheduler + programmer/compiler | buffering、eligible warps、occupancy；同步状态复杂 |
| TPU | Systolic-array spatial wave | 固化 data path + mapping | array mapping、fill/drain；shape rigidity |
| Graphcore IPU | Placement + BSP `Compute→Sync→Exchange` | Compiler placement + global phase | exchange volume、local-memory capacity、slowest tile |
| Groq TSP | Functional slices + static streams | Compiler static schedule | instruction/operand cycle 对齐；compiler burden，较少 dynamism |

所有架构都要回答 movement、storage、ordering、scheduling。区别只是 responsibility 在 programmer/compiler、runtime scheduler 与 hardware data path 之间如何分配。

## 我的理解与推导

### 1. Pipeline 的抽象对象不是“指令序列”，而是 token 的所有权流

表面上课程在讲 `cp.async`、TMA、mbarrier、WGMMA；更统一的抽象是：producer 生产一个带 generation 的 token，token 指向某份 storage 与 work，consumer 取得 token 后处理并归还 storage ownership。

这个抽象可跨尺度复用：

- tile pipeline 的 token 是“某 stage 中可用的 A/B data”；
- CLC pipeline 的 token 是“下一份 work coordinate”；
- persistent outer loop 的 token 是“某 problem 的 output tile”；
- 分布式系统中同样可把 ready/completion/visibility/ownership 扩到设备与网络。

### 2. 优化的目标不是消除 stall，而是消除无工作可发的窗口

单个 warp 的等待很正常。只要 scheduler 能找到别的 eligible warp/instruction，关键 pipeline 仍可保持吞吐。Profile 时应沿这条链推理：

```text
issue slots 空闲？
  → eligible warps 是否不足？
    → 因何不 eligible：scoreboard / barrier / dependency / throttle？
      → 能否从 ILP、MLP、TLP、跨-stage overlap 增加在途工作？
        → 新增并行性消耗多少 registers / SMEM / control state？
```

### 3. 每次增加异步性，都要补上一份显式状态

同步代码把 latency 暴露在控制流中；异步代码把 latency 隐藏起来，但需要 completion object、phase/generation、ownership 和 drain protocol。性能提升来自允许更多操作在途，复杂度也来自必须区分这些在途操作。

### 4. Persistent kernel 把 kernel 变成设备内 runtime

一旦 CTA 长期驻留、自己取 work、维护 roles/barriers/descriptors，并负责 termination，它已经不只是“算一个 tile 的函数”，而是一个小型 runtime：有 worker pool、scheduler、queue/token、资源生命周期与 shutdown protocol。这解释了其高性能与高复杂度为什么同时出现。

## AI Infra 视角

### Shape

- Tile shape 决定 reuse、SMEM footprint、Tensor Core utilization 与边界比例。
- MoE 中 M 动态变化，静态 tile-count balance 不能保证 runtime balance。
- TPU 对 shape 与 array mapping 更敏感；短任务的 fill/drain 成本尤其明显。

### Compute

- Tensor Core throughput 提升使供数更容易成为瓶颈。
- Warp specialization 把 control 与 math roles 分开。
- SM100 TMEM 改变 accumulator placement 与 epilogue handoff。

### Memory

- Reuse 先减 traffic，overlap 再藏 latency。
- Pipeline depth 在 latency budget 与 SMEM/occupancy 间权衡。
- Swizzle 必须服务于 consumer access pattern。
- Generic/async proxy 间需明确 visibility/order。

### Communication

- 单 GPU 内的 producer/consumer、completion、visibility、ordering、ownership，是理解 GPU-to-GPU communication 的缩影。
- TMA transaction、CLC work request 都可视为带 completion 的消息传递。

### Runtime/System

- Persistent kernel 把调度推进到 device side，减少 launch/setup overhead。
- Static round robin 成本低但无法感知 tile cost；dynamic work acquisition 更均衡但增加状态与同步。
- Coexistence 与 resource monopolization 是系统层而非单-kernel 指标。

## 实战检查清单

### 正确性

- [ ] Participants 与 collective scope 是否一致？
- [ ] 等待的是正确 operation、stage 与 generation 吗？
- [ ] Completion、visibility、ordering、ownership 四问是否分别有答案？
- [ ] Consumer release 前 producer 是否可能覆盖 storage？
- [ ] Generic/async proxy 间是否需要 fence？Fence 后是否还需要 rendezvous？
- [ ] Prime、drain、early-exit 与 termination 是否处理所有 in-flight work？
- [ ] Group boundary 后 descriptors/pointers/predicates 是否完整重绑？

### 性能

- [ ] 数据复用是否先把 HBM traffic 压下来？
- [ ] Copy issue 足够早吗？Compute window 足以覆盖 latency 吗？
- [ ] Active warps 多但 eligible warps 少吗？主要 stall reason 是什么？
- [ ] 能否增加 ILP、MLP、TLP 或跨 tile overlap？
- [ ] 增加 stages/unrolling 后 SMEM/register pressure 是否反而降低 occupancy？
- [ ] Swizzle 是否匹配 consumer layout，alignment/granularity 是否满足快路径？
- [ ] Persistent schedule 是否有 tail imbalance 或 resource monopolization？
- [ ] 比较实验时是否按工作量归一，而非只看 raw runtime？

## 本讲结论

1. 高吞吐系统靠足够的 in-flight work 隐藏长 latency；核心观测量是 eligible work，而不是 active work。
2. Reuse 减少 traffic，overlap 隐藏剩余 latency；两者都受 SMEM/register/occupancy 资源约束。
3. Pipeline stage 是带 EMPTY/FULL phase 与明确 ownership 的循环 buffer。
4. Completion、visibility、ordering、ownership 是四个独立问题；一个 wait 或 barrier 通常不能同时解决全部。
5. SM80→SM90→SM100 把搬运从 thread-level small copy 推向 TMA bulk transfer，再把 accumulator 移入 TMEM；每代都减少一类 data-path overhead，也引入新的 orchestration lifetime。
6. Persistent kernel、CLC 与 grouped GEMM 把 pipeline 从 tile 内部扩展到 device-side work scheduling，用更多状态换取 setup reuse 与负载均衡。
7. GPU、TPU、IPU、TSP 没有消灭 Data Orchestration，只是把不确定性和责任放在不同层。

## 自测问题

1. 为什么 100% occupancy 仍可能出现几乎没有 eligible warp 的情况？

    **面试回答：** Occupancy 衡量驻留 warp 比例，而 eligible 还要求操作数、依赖和执行资源都 ready。所有驻留 warp 可能同时等待长延迟 load 或同一 barrier，造成发射空泡；应查 scoreboard、同步和管线阻塞，再增加独立指令或跨 stage 工作。

2. Baseline copy–compute loop 中两个 block barrier 分别保护哪条依赖？

    **面试回答：** Copy 后的 barrier 保护 producer 写入到 consumer 读取之间的 RAW 依赖，确保完整 tile 已就绪。Compute 后的 barrier 保护读完到下一轮覆盖之间的 WAR 依赖，确保 storage 可以复用；两个屏障分别对应“可以读”和“可以重写”。

3. 为什么 async copy 完成不代表 stage 可以立刻被下一位 producer 覆盖？

    **面试回答：** Copy 完成只证明本轮数据已搬到目标，不证明使用这些数据的 consumer 已结束。Producer 必须等 consumer release，拿回该 stage 的写权，再覆盖下一代数据；否则会在计算过程中混入下一轮 tile。

4. Double buffering 在什么情况下仍无法隐藏 copy latency？

    **面试回答：** 只有两份 buffer 时，通常只能提前一个 compute 窗口发 copy；若 copy latency 大于这个窗口，消费者仍会等待。搬运带宽不足、issue 太晚、任务太短或共享资源争用也会阻止完全隐藏延迟；增加 stages 只能在容量与带宽允许时改善。

5. `mbarrier` 相比 `__syncthreads()` 多表达了什么？为什么需要 phase/token？

    **面试回答：** `mbarrier` 能拆开 arrive 和 wait，并在支持的操作中把预期事务字节数及异步完成纳入条件，不要求所有线程像 `__syncthreads()` 那样同时停住。Phase/token 标识循环复用中的具体一轮，避免上一轮完成被误认成本轮完成；参与者和计数仍必须正确。

6. 为什么同一 SMEM address 在 generic proxy 与 async proxy 间仍需 ordering 机制？

    **面试回答：** Generic 和 async proxy 是不同访问代理，同一个 SMEM 地址并不自动建立二者之间的先后及可见性。需要按 PTX 契约使用相应 proxy fence、完成等待和线程协调，使 producer 写入先于 async consumer 读取；单独的源码顺序或线程会合未必足够。

7. Warp specialization 改变了 ownership cycle，还是只改变了每一步的执行者？

    **面试回答：** Warp specialization 主要改变每个环节由谁执行，例如 producer warp 搬数据、consumer warp 做 MMA。EMPTY→WRITING→FULL→READING→EMPTY 的所有权循环仍然成立，只是要用更精确的分组同步连接不同角色，并处理退出和排空。

8. TMEM 缓解了什么资源压力，又新增了哪些 lifetime？

    **面试回答：** TMEM 减少 accumulator 对通用寄存器的占用，但增加 TMEM allocation、异步 MMA 写入、epilogue 读取和 deallocation 的生命周期。还需追踪 SMEM operand 何时可复用、TMEM 结果何时可覆盖；不同资源的 release 时刻不能混成一个“计算完成”。

9. Static grouped-GEMM round robin 为什么仍会产生 tail imbalance？

    **面试回答：** Round robin 平均分的是 tile 数，而 tile 的真实成本可能因 K、边界、epilogue、cache 和 expert shape 不同而变化。某个 worker 分到更多昂贵 tile 就会成为尾部；动态取任务可改善，但要权衡队列、同步与局部性开销。

10. Persistent kernel 为什么可以看作一个 device-side runtime？

    **面试回答：** Persistent kernel 内的 CTA 长期驻留，维护 worker 状态，自行领取任务、更新描述符、执行流水并判断终止。这已经包含 worker pool、调度、资源管理和 shutdown 等 runtime 职责，能摊薄 launch/setup 成本，也使并发共存与生命周期更复杂。

11. TPU、IPU、TSP 分别把哪类调度责任固化到 data path 或交给 compiler？

    **面试回答：** TPU 的 systolic array 把规则乘加和数据流固化到阵列，编译器负责分块映射与填充/排空；IPU 依靠编译期放置和 Compute–Sync–Exchange 阶段组织局部计算与交换。Groq TSP 更强调编译器按周期安排指令和数据流，以静态计划换取较少动态调度开销。


## 参考资料

- [NVIDIA CUDA Programming Guide: Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [NVIDIA CUDA Programming Guide: Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)
- [NVIDIA CUTLASS: Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [NVIDIA CUTLASS: `tcgen05` MMA Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [Triton Grouped GEMM Tutorial](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html)
- [Google Cloud TPU System Architecture](https://docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- [Graphcore IPU Programmer's Guide](https://docs.graphcore.ai/projects/ipu-programmers-guide/en/latest/about_ipu.html)
- Abts et al., “Think Fast: A Tensor Streaming Processor,” ISCA 2020.
