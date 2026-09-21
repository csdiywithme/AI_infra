---
type: transcript
status: polished
course: Weiming HPC Training Camp × LCPU AI Infra Seminars
session: 4
speaker: 卢怡霏
lecture_date: 2026-08-09
topics:
  - gpu
  - pipeline
  - latency-hiding
  - cuda
  - tensor-core
  - persistent-kernel
video_url: https://www.bilibili.com/video/BV1NRuX6UEwJ/
slides_url: https://infra.seminars.lcpu.dev/slides/session04.pdf
---

# LCPU AI Infra Seminars - Session 04 - Pipeline Ordering - 精编字幕

> [!info] 整理说明
> 本文依据用户提供的自动字幕，按官方课件校正术语、断句并删除口头重复。它是保留讲述顺序和主要例子的“精编字幕”，不是逐字稿；时间点对应原视频，可用于回看定位。

## 术语校正

| 自动字幕常见误识别 | 校正 |
|---|---|
| pipon / Python ordering | Pipeline Ordering |
| dan / deal orchestration | Data Orchestration |
| jam | GEMM |
| CCTA / blog / THBLOCK | CTA / block / thread block |
| GMM / GMAN | GMEM（global memory） |
| SMM / S man | SMEM（shared memory） |
| word / work group | warp / warp group |
| Harper | Hopper |
| Black quil / Black CD | Blackwell |
| ta / TAMA | TMA（Tensor Memory Accelerator） |
| team m / 7man | TMEM（Tensor Memory） |
| inside computer | Nsight Compute |
| proxy fans | proxy fence |
| group jam | grouped GEMM |
| epilot / app log | epilogue |

## [00:00](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=0) 为什么需要 Data Orchestration

上一讲介绍了 Tensor Core。Tensor Core 的优势是把计算吞吐拉得非常高，但这也制造了一个矛盾：当 Tensor Core 需要下一块数据时，数据未必已经到达正确的位置、完成传输并且可以安全使用。

所以本讲的 Data Orchestration 主要处理四类问题：

1. **Movement**：数据从哪里搬到哪里；
2. **Storage**：每个阶段的数据存在哪里；
3. **Synchronization / Ordering**：生产数据的人与消费数据的人如何确认数据已到达，或已被消费；
4. **Scheduling**：每个执行单元应在什么时间做哪一份工作。

这里会频繁使用两个角色：**producer** 为下一阶段准备数据或任务，**consumer** 使用这些数据或执行这些任务。CTA 在当前语境中基本可理解为 CUDA thread block。

## [01:54](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=114) 一个 GEMM tile 的物理旅程

进入 Tensor Core 计算前，GEMM 的 A、B tile 通常先从 HBM 搬到 SMEM；warp 或 warp group 再从 SMEM 取得 operands，交给 Tensor Core。累加结果在 Hopper 及以前通常长期占用 registers，在 Blackwell 上还可能进入专门的 TMEM；最后由 epilogue 把结果写回 GMEM。

这条路径有三个直接瓶颈：

- HBM 访问 latency 很长，一次 global-memory request 还会经历地址转换、cache 查询以及可能的下层 memory 访问；
- registers 和 SMEM 容量有限，不能无限预取。每增加一个 stage 都会多占 SMEM，多保留一条独立计算链则可能增加 register pressure；
- Tensor Core 太快，供数稍慢就会空转。

两个基本解法是 **reuse** 和 **overlap**。Reuse 让一个 A tile 被多个 N 方向输出复用、一个 B tile 被多个 M 方向输出复用，从而减少 HBM traffic；但搬运次数减少后，每次远端访问的 latency 仍然存在。Overlap 则是在等待数据时做别的工作，例如在搬运 tile `k+1` 时计算 tile `k`。

Little's Law 给出一个直观解释：

$$L=\lambda W$$

`W` 是一项工作从发出到完成的平均时间，`λ` 是希望维持的完成速率，`L` 是平均在途工作量。若 latency 很大而目标 throughput 也很高，就必须让足够多工作同时处于 in-flight 状态。GPU 中的 in-flight work 可以来自：同一 CTA 的多个 memory requests、多个 pipeline stages、同一 warp 中的独立 instruction chains，或同一 SM 上的多个 ready warps。

## [05:26](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=326) Baseline：串行 copy–compute loop

最朴素的实现是：一个 block 合作搬一个 tile，每个 thread 从 GMEM 读一个元素，再写入 SMEM。普通 load/store 会让数据显式经过 register：`LDG → register → STS`。

循环中通常需要两个 block-level barrier：

- 第一个保护 producer 到 consumer 的 **RAW（read-after-write）依赖**：consumer 读取 SMEM 前，必须确认 producer 已经写完；
- 第二个保护 buffer reuse：下一轮 producer 覆盖同一块 SMEM 前，必须确认本轮所有 consumer 已读完。

因此，一块可复用的 SMEM buffer 至少在两个状态间循环：producer 写完后由 `EMPTY → FULL`，consumer 读完后由 `FULL → EMPTY`。

这个写法正确但完全串行：copy 时不能 compute，compute 时下一次 copy 尚未发出。目标是尽早发起异步操作，让 `copy(k+1)` 与 `compute(k)` 重叠；发起方只在真正消费结果前执行 wait。

## [08:18](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=498) Cooperative Groups：把参与者契约写进接口

一个 tile 往往由一个 block 的许多 threads 合作搬运，因此必须回答：哪些 threads 参与 copy，哪些 threads 参与 wait？`__syncthreads()` 的函数签名没有显式写出“整个 block 都必须参与”。如果它藏在只由一部分 threads 进入的分支或 helper function 中，就可能 deadlock 或产生 undefined behavior。

Cooperative Groups 的价值是把 collective 的契约显式化：

- participant set：哪些 threads 属于这个 group；
- logical rank space：成员在 group 内的编号；
- collective contract：哪些成员必须共同参与 sync、communication 或 partition。

但它不会自动消灭错误。如果对象是 `thread_block`，仍然只有一部分 block 成员调用 `block.sync()`，程序依然不正确。若只想让 32 个 threads 做 collective，应先让整个 parent group 共同完成 `tiled_partition<32>`，再由目标 tile 的全部 32 个成员进入 warp-level collective。

常见 group scope 包括 warp-sized tile、CTA、thread-block cluster 与 cooperative grid。`grid.sync()` 需要 cooperative launch，而且 grid 必须满足同时驻留约束；否则一部分 CTA 在 barrier 等待，而剩余 CTA 因资源不足无法启动，会形成 deadlock。

## [13:09](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=789) Async copy 的三类接口与三层正确性

以 Cooperative Groups 为例，`memcpy_async(group, ...)` 负责发起 GMEM→SMEM 搬运，`wait(group)` 说明何时可消费。不同 namespace、group 参数和 completion object 共同定义不同契约：可能是 group collective，也可能由单个 calling thread 发起并由 pipeline 或 barrier 跟踪。

仅仅“调用过 wait”并不是完整的正确性证明，至少还要检查：

1. **Address/range**：源、目标指针有效，范围不越界且不发生错误重叠；
2. **Completion**：确实在跟踪这一次 copy 的 group、pipeline 或 barrier 上等待；
3. **Reuse/ownership**：上一位 consumer 释放 stage 前，下一位 producer 没有提前覆盖它。

一份 copy 即使完成，也可能是“完成了错误的那一份”，或写入了被过早复用的 stage。

## [15:52](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=952) Ampere / SM80：`cp.async`

SM80 将普通路径中的 `LDG → register → STS` 合并为 `LDGSTS / cp.async → SMEM`。数据 payload 不再必须显式经过通用 register，因此可以减少 register transit 和 instruction overhead；发起后 warp 也能继续执行 independent work。这为多阶段 GMEM→SMEM pipeline 提供了硬件基础。

高效路径依赖具体条件：source 与 destination 要满足所需 alignment，copy size 也要匹配 transfer granularity。4-byte alignment 是常见最低条件，16-byte alignment 更容易稳定进入高效异步路径。

还要注意 **warp entanglement**：某些 pipeline batch sequence 是 warp-shared 的。如果不同 lanes 在 divergent control flow 中 commit 不同数量的 async-copy batches，warp-wide sequence 会比某些 thread 自己理解的 sequence 前进得更远，导致 over-wait、额外 barrier 更新，吞掉本来希望获得的 overlap。实践上应让 commit 与 arrive-on 尽量在 converged warp 上执行。

## [19:28](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=1168) `cuda::pipeline`：N-stage 循环队列

只有一块 SMEM buffer 时，下一轮 copy 会覆盖当前 compute 所需数据，因此两者难以重叠。准备两个 stages 就得到 double buffering。

一个 stage 的 ownership cycle 是：

```text
producer_acquire(EMPTY)
    → issue copies for a future tile
    → producer_commit / publish FULL
    → consumer_wait(FULL)
    → compute / consume
    → consumer_release / return EMPTY
```

Producer 只能写自己 acquire 到的 EMPTY stage；consumer 只能读 wait 成功后的 FULL stage。`consumer_release` 与 baseline 中第二个 `__syncthreads()` 作用相似，都是保护 buffer reuse。

完整流水线有三个阶段：

- **prime / fill**：先搬入第一个 tile，此时没有旧 tile 可计算；
- **steady state**：每轮同时 issue future tile 的 copy，并 compute current tile；
- **drain**：最后一份 future copy 已发出，继续 wait、compute、release，清空仍在途的工作。

## [22:09](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=1329) Double buffering 为什么仍可能暴露 latency

Warp 可以区分为三种状态：

- **active**：已经 resident 在 SM 上且尚未退出；
- **eligible**：下一条 instruction 的 operands 和依赖已 ready，可以被选中发射；
- **issued**：本周期真正被 scheduler 选中并发出 instruction。

Active 不等于 eligible。一个 warp 可能正在等 global load 写回 register，或停在 barrier 上。Warp stall 本身是正常的；GPU 本来就会在一个 warp 等待时切换到另一个 eligible warp。真正损失 throughput 的是 scheduler 有 issue opportunity，却找不到 eligible warp 或 ready instruction。

Double buffering 常见的失败原因有三类：

1. Copy 发得不够早：current compute 时间短于 future-copy latency，走到 wait 时数据仍未到；
2. 所有 warps 同时走到同一个 wait：CTA 内即使有很多 active warps，也可能一起失去 eligibility；
3. 资源限制让可替换工作太少：更多 stages 占 SMEM，更多 unrolling/independent chains 占 registers，进而减少每个 SM 能 resident 的 CTA/warps。

Occupancy 是“实际 resident active warps / 架构最大 resident warps”的比例，受 threads/block、registers/thread、SMEM/block 及架构 block/warp 上限影响。但 occupancy 不等于 utilization，也不等于 ready-warp 比例：100% occupancy 时所有 warps 仍可能在等同一条长依赖；较低 occupancy 的 kernel 若 overlap 很好，也能跑满关键 pipeline。它更接近“可用于 latency hiding 的容量”，而不是最终性能目标。

## [25:59](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=1559) Pointer chasing 实验：看懂 scoreboard 与 eligibility

讲者在 RTX 5090 上构造了一个刻意的实验：每个 thread 维护一条或两条互不依赖的 pointer chains。单链中 `p = next[p]` 存在 loop-carried RAW dependency；本轮 load 返回新的 `p` 前，下一轮地址根本无法形成。

Nsight Compute 的 source view 可把等待定位到依赖链：`LDG.E` 负责产生 register，下一轮 `IMAD.WIDE` 需要读取该 register。Scoreboard 跟踪尚未完成的 register dependency；operand 未 ready 时，warp 就不能成为 eligible。该等待通常被报告为 **Long Scoreboard**。

单链实验中，平均每条 issued instruction 约对应 199.85 cycles，其中 193.1 cycles、约 96.6% 的 issue interval 属于 Long Scoreboard。这里的数值是按 issued instructions 归一后的 warp-state metric，不能机械地理解为“每次 global load 固定 latency 就是 193.1 cycles”。Scheduler statistics 更直接地显示：平均每 scheduler 约有 1 个 active warp，却只有 0.01 个 eligible warp，约 99.5% 周期没有 eligible warp。

## [30:47](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=1847) 两条独立链：ILP、MLP 与 TLP

两条 pointer chains `p0` 与 `p1` 互不依赖，使同一 warp 的 instruction stream 中多出 independent work：

- 多一条独立 instruction chain 是 **ILP（instruction-level parallelism）**；
- 两个 independent loads 能同时 outstanding，增加 **MLP（memory-level parallelism）**；
- 增加 resident warps，让一个 warp 等待时另一个能执行，是 **TLP（thread-level parallelism）**。

实验里单链每次迭代一个 load，运行约 2.10 ms；双链每次迭代两个 loads，运行约 2.81 ms。工作量翻倍但时间只增至约 1.34 倍，归一化 load throughput 约提升到 `1.49×`。这说明比较优化时不能只看 raw runtime，必须同时看完成的工作量。

对 kernel 调度而言，可以从多个方向寻找 missing eligible work：在 warp 内增加 ILP，由此增加 outstanding loads 和 MLP；提前发起 future-tile async copies，增加跨 tile 的 MLP；增加 resident warps 获得 TLP；增加 pipeline depth 给 copy 更长提前量。但最后一项会消耗更多 SMEM 和控制状态。

## [33:45](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=2025) Pipeline Ordering 的四个问题

到这里，同步问题可以拆成四个彼此不同的问题：

| 问题 | 要确认什么 |
|---|---|
| Completion | 异步操作是否真的完成？ |
| Visibility | Consumer 是否能观察到 producer 产生的数据？ |
| Ordering | 不同 execution agents / memory proxies 的访问是否建立了先后关系？ |
| Ownership | Consumer 释放前，其他 producer 是否可以复用这块 storage？ |

Stages 不是越多越好。更多 stages 能更早 prefetch、给 latency 更长隐藏窗口，但会增加 SMEM 占用、降低 occupancy，并增加 control state。

SM80 的结论是：`cp.async` 让小粒度 GMEM→SMEM 异步搬运更直接，但 programmer/compiler 仍需设计 participant group、thread-to-address assignment、alignment、SMEM layout、stage count、ownership，以及 SMEM/register/occupancy 之间的资源平衡。

## [36:40](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=2200) Hopper / SM90：TMA 与 Tensor Map

Ampere 的典型做法是许多 threads 各自计算地址、发起小 copies。Hopper 的 TMA 允许一个 elected thread 发起一块多维 tensor 的 bulk asynchronous transfer，硬件依据预编码 descriptor 完成地址生成与搬运。

Tensor Map 是 transfer contract，描述：

- source geometry：base pointer、rank、dimensions、byte strides；
- tile geometry：box dimensions、element strides；
- SMEM placement：interleave、swizzle；
- boundary/cache policy：OOB fill、L2 fetch granularity 等。

Host 侧用 driver API 构造 descriptor；kernel 发起 TMA 时主要提供 descriptor、当前 tensor coordinates、destination 与 completion object。L2 promotion 可以提示 memory system 以更大的邻近 segment 取数，方便后续复用；OOB fill 可让边缘 tile 越界位置由 TMA 填入定义值，常见为 zero fill，避免 threads 逐元素写 predicates。

## [39:50](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=2390) TMA completion：为什么需要 `mbarrier`

Tensor Map 主要回答“怎么搬”。TMA issue 后 async agent 在后台工作，CTA threads 继续执行，于是还需要回答“整块 transaction 何时真正到达 SMEM”。`__syncthreads()` 只能说明 threads 到达 barrier，不能自动包含一个独立 TMA transaction 已完成这一事实。

`mbarrier` 的 phase completion 同时考虑两类条件：

- 所需 thread arrivals 已到达；
- 绑定到该 phase 的 async transaction 已完成，例如承诺的 bytes 已到达。

两者都满足，barrier 才能翻转 phase，stage 才成为 FULL。由于同一个 barrier object 会在循环中反复使用，必须用 token 或 parity 区分 generation；否则可能把上一轮完成误当成本轮完成。高层 arrival token 绑定调用 `arrive()` 时的 phase；底层 PTX 常显式追踪一位 parity，0/1 交替。

## [44:07](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=2647) Generic proxy、async proxy 与三条不同的边

普通 CUDA `ld/st` 与 TMA/WGMMA 等 async hardware agent 并非同一执行主体，也不一定经过同一 memory access mechanism。CUDA memory model 用不同 proxy 表达这些访问路径。同一 SMEM address 可以被两条路径访问，但“地址相同”并不自动建立跨 proxy 的 ordering。

一个 TMA read-modify-write round trip 至少有三条边：

1. **GMEM→SMEM data ready**：TMA copy completion 绑定 mbarrier phase；barrier wait 通过后，waiting threads 才能安全读 SMEM；
2. **generic writes→async reads visibility**：threads 用普通 stores 修改 SMEM 后，要执行 `fence.proxy.async`，把 fence 前的 generic-proxy accesses 排到后续 async-proxy accesses 前；fence 不是 rendezvous，还需 block sync 确认所有 writers 都到达；
3. **TMA read completion→source reusable**：bulk async-group completion 确认 TMA 已读完 source SMEM，随后 stage 才能被覆盖。这条边保护的是 ownership。

因此，mbarrier completion、proxy fence 和 source-release wait 解决的是不同问题，不能相互替代。

## [48:18](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=2898) TMA swizzle：为 consumer 选择 layout

SMEM 被划为多个 banks，一个 warp 可以并行访问不同 banks。若许多 lanes 同时访问同一 bank 的不同地址，请求可能被拆成多次服务，形成 bank conflict。

Swizzle 改变 logical tensor coordinates 到 physical shared-memory addresses 的映射，使 consumer warp/warp group 的访问更均匀地落到各 banks。关键原则是：**GMEM 中的自然顺序不一定是 SMEM consumer 的最佳布局**。由于硬件按固定 pattern 重排地址，swizzle mode 会对 tensor box/row byte width、global base/stride alignment、SMEM base alignment 等提出限制，并以 16-byte units 为常见粒度。

## [50:12](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=3012) Warp Specialization：把 CTA 划成长期 agents

有了 TMA，一个或少数 elected threads 就能发起 bulk transfer，不再需要所有 warps 轮流执行相同 copy-control code。Warp specialization 把同一 CTA 的 warp groups 固定成长期不同的 agents：

- producer warp group：负责 TMA issue，以及 EMPTY/FULL stage 转换；
- consumer warp groups：等待 FULL stage、发起 WGMMA、维护 accumulators、消费后 release stage；
- 某些 schedule 还指定 consumer 或独立角色承担 epilogue。

收益包括：producer control flow 与 consumer math control flow 分离，减少每个 warp 的无关指令；不同角色可以使用不同 register budget；producer 可持续填 future stages，consumer 按 math pipeline 节奏消费。

底层 ownership cycle 没有改变，仍是：`acquire EMPTY → issue TMA → wait FULL → WGMMA → release → reuse`。变化的是每个动作由谁执行，以及 completion mechanism 更强。

## [53:00](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=3180) Blackwell / SM100：TMEM 与新的 lifetime

Hopper 及以前，accumulator 长期驻留 consumer-thread registers。输出 tile 或 K loop 变大时，accumulator 增多，register pressure 会限制 occupancy。Blackwell 引入 TMEM，为 Tensor Core accumulators 提供专门存储。

典型数据流变为：

```text
A/B: GMEM → TMA → SMEM
SMEM operands → tcgen05 → TMEM accumulator
TMEM → LDTM → registers → epilogue → GMEM D
```

`tcgen05` 由 elected thread 发起，A operand 可来自规定的 SMEM/TMEM source，B 通常来自 SMEM，accumulator destination 位于 TMEM；某些模式支持 2-CTA cooperation。

新能力同时带来新的 orchestration responsibility：SM80 主要管理 copy-stage lifetime；SM90 增加 bulk transfer、mbarrier phase 与 warp roles；SM100 又增加 TMEM allocation/deallocation lifetime、MMA 对 TMEM accumulator 的读写顺序、LDTM→epilogue handoff，以及 2-CTA mode 下 CTA-pair cooperation 与 termination。

## [55:23](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=3323) MoE：单个好 tile 不等于好 workload

MoE router 把 tokens 分配给少数 experts，每个 expert 对分到的 tokens 执行 GEMM，最后再按原 token 顺序合并。一个 batch 因而产生许多 `M×N×K` problems，尤其 M 随 routing 动态变化，tile 数量与时长不均。

即使每个 GEMM tile 已很好地使用 TMA/WGMMA，静态给每个 SM 相同数量的 tiles 仍可能产生 tail imbalance：不同 problem 的边界条件、K、epilogue、cache behavior 不同，先完成的 SM 空闲，少数 SM 继续处理昂贵 tile。

两个主要策略是：

- **Grouped GEMM**：把多个 GEMM problems 放进同一个 kernel；
- **Persistent kernel**：启动有界数量的 resident workers，让每个 worker 反复领取多个 tiles，而不是算完一个 tile 就退出。

## [57:51](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=3471) Persistent kernel：keep a worker alive

Conventional tiled GEMM 中，一个 output tile 对应一个 CTA；CTA 完成 K loop 与 epilogue 后退出。Persistent kernel 启动数量有界、按 occupancy policy 选择的 worker grid。Worker 完成一个 output tile 后向 scheduler 请求下一 tile，直到没有工作。

好处包括：

- 跨 tile 复用 role partition、descriptors、barriers 与 scheduler state 等 setup；
- 动态取 work 有机会缓冲 tail imbalance；
- 多个小 GEMMs 可合并进一个 launch，减少 launch overhead；
- 每个 worker 内可反复运行 TMA↔MMA 的 multistage inner pipeline。

可以把它理解为两个嵌套 pipeline：inner pipeline 在一个 output tile 的 K-loop stages 中重叠 data movement 与 compute；outer pipeline 在多个 output tiles/problems 之间领取、切换工作并复用资源。

Persistent outer loop 需要区分三种 lifetime：

- **per-worker state**：roles、scheduler state、barrier objects 等，跨许多 tiles 长期存在；
- **per-stage state**：SMEM slot、EMPTY/FULL phase、TMA completion token；
- **per-tile state**：problem ID、coordinates、base pointers、valid bounds、epilogue metadata，每次取新 work 都要重绑。

代价是 workers 长期持有 SMEM/registers，可能垄断资源、降低与其他 GPU workloads coexist 的机会；termination 也更复杂，必须正确 drain 所有 roles、stages 和 in-flight async operations。

## [64:34](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=3874) Persistent ping-pong 与 Cluster Launch Control

Persistent ping-pong 让两个 consumer warp groups 交替工作：一组做 tile A 的 epilogue 时，另一组开始 tile B 的 MMA，从而重叠 epilogue 与 mainloop，并复用 producer setup/scheduler state。代价是同时 live 的 tile/accumulator state 更多、termination 更难、资源占用更高。

Persistent 调度存在基本 trade-off：

- 每个 logical work item 启动一个 block，硬件自然调度，load balance 和 coexistence 较好，但 per-block setup 复用少；
- 只启动固定数量 persistent blocks，setup overhead 低，但 worker 数 launch 时已确定，运行期其他 kernel 的占用或启动节奏仍可能制造尾部不均衡，长期驻留也减少其他工作的调度机会。

Blackwell 的 **Cluster Launch Control（CLC）** 尝试结合两者：resident block 完成自己的 block coordinate 后，请求取消一个尚未启动的 logical block；若成功，就接管被取消 CTA 的 coordinate，算完后继续请求；失败则说明无 work 可偷，进入结束流程。

CLC request 本身也可看成 async producer：它生产的不是 A/B tile，而是下一份 work coordinate。把 request 提前到 current-tile compute 之前，可以用当前计算隐藏 work-acquisition latency。请求结果的 completion、decode 与复用也需要自己的 barrier/ownership protocol。

## [70:03](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=4203) Grouped GEMM：把不规则问题拼成逻辑 tile 空间

若每个小 GEMM 单独 launch，单个 shape 可能没有足够 output tiles 填满 GPU，重复 launch 也有开销。Grouped GEMM 把不同 M/N/K、base pointers、strides 和 epilogue metadata 的 problems 组成列表，交给同一个 kernel 与一组 persistent workers。

每个 GEMM 的 output matrix 在 M×N 平面被切为：

$$
\text{num\_tiles}[g]
=\left\lceil\frac{M_g}{BLOCK_M}\right\rceil
\left\lceil\frac{N_g}{BLOCK_N}\right\rceil
$$

各 group 的 tile ranges 首尾相接，形成一个 global logical tile-ID space。一个简单的 static persistent schedule 让 worker 从自己的 program ID 开始，按 worker-grid size 跨步，例如四个 workers 分别处理 `0,4,8,12...`、`1,5,9,13...`。它不需要访问共享 queue，只需初始 ID 与 stride。

但相同 tile 数不等于相同运行时间：boundary predicates、type conversion、epilogue 复杂度、cache locality 都会改变成本，因此 static round robin 仍可能产生 tail。可以用 queue-based work stealing、CLC 等 dynamic acquisition 缓解。

拿到 global tile ID 后，worker 先通过 group prefix ranges 找到 group ID，再换算 local tile ID 与 `(tile_m, tile_n)`。当 worker 跨越 group boundary 时，pipeline roles 和 stage protocol 可以继续存在，但 A/B/C/D base pointers、runtime strides、descriptors、predicates 与 epilogue destination 必须随新 group 重新绑定。**Pipeline 可以跨边界生存，数据绑定不可以。**

## [78:33](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=4713) 把视角扩展到其他 accelerator

前面的 GPU 设计并没有消灭 Data Orchestration，只是让 programmer/compiler、runtime scheduler 与 hardware data path 重新分担责任。为了看清这种责任划分，讲者比较了 TPU、Graphcore IPU 与 Groq TSP。

### [79:17](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=4757) TPU：把数据复用固化为空间波

TPU 的 MXU 使用 systolic array。数据从阵列边缘注入，每个 MAC cell 使用当前 operands 做 multiply-accumulate，再把仍需复用的数据直接传给相邻 cell。执行同样经历 fill、steady state、drain。

它把矩阵乘法中高频的数据 movement 和 reuse 固化成 spatial wave，避免每个 cell 各自从 HBM 取数。但结构过于规则时，小 shape、映射不佳或短 sequence 会让 fill/drain 占比过高，array 难以保持饱满。

### [82:39](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=4959) Graphcore IPU：local memory 与 BSP phases

IPU 由大量 tiles 组成，每个 tile 有 processing core 和就近 local memory。若下一 operation 位于另一个 tile，数据需通过 exchange fabric 移动。

典型执行模型是 `Compute → Sync → Exchange`：各 tile 先只对 local memory 计算；随后所有相关 tiles 到达 phase boundary，确保即将交换或覆盖的数据不再被使用；最后数据按预先规划的通信关系移动到下一阶段的 tile。

这种模型让 local work 与 communication phases 很清楚，但 placement 不好会增加 exchange volume；work partition 不均衡时，sync 必须等最慢 tile；单 tile local memory 容量也限制可放入的数据和 program。

### [85:31](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=5131) Groq TSP：functional slices 与静态 stream schedule

传统 multicore 在每个 core 内重复 instruction control、arithmetic、load/store 和 network interface 等多种功能；TSP 反过来将芯片组织为 functional slices：MEM 负责 SRAM read/write，SXM 做 stream permutation/routing，VXM 做 elementwise/vector operations，MXM 做 matrix operations，ICU 提供 instructions。

Instructions 沿 slice 的一个方向传播，data streams 横向经过不同 functional slices。Producer 与 consumer 通过显式 streams 连接。由于不像 GPU 那样依赖 warp scheduler 动态选择 ready work，compiler 必须安排：数据经过哪些 slices、指令在哪个 slice 执行、instruction 与 operand 在哪个 cycle 相遇，以及资源/stream 是否冲突。硬件少承担的动态不确定性，被转移给 compiler。

## [90:15](https://www.bilibili.com/video/BV1NRuX6UEwJ/?t=5415) 总结：Where does uncertainty live?

四种架构的核心差异是“不确定性放在哪里、谁负责吸收它”：

| 架构 | 主要 orchestration 模型 | 主要代价 / 优化关注点 |
|---|---|---|
| GPU | dynamic readiness + explicit waits | buffering、occupancy、eligible warps、同步状态 |
| TPU | systolic array 中的 spatial wave | shape rigidity、array mapping、fill/drain |
| Graphcore IPU | placement + BSP exchange phases | placement、exchange volume、global imbalance |
| Groq TSP | functional slices + static stream schedule | compiler burden、functional-slice utilization、较少 dynamism |

没有一种架构从源头消灭 movement、storage、ordering 和 scheduling；它们只是重新划分 programmer/compiler、runtime scheduler 与 data path 的责任。Data Orchestration 的本质，就是让数据在正确的时间到达正确的位置、让正确的执行单元使用它，并以足够多的在途工作维持系统吞吐。

## 参考资料

- [讲座视频：Pipeline Ordering: Data Orchestration](https://www.bilibili.com/video/BV1NRuX6UEwJ/)
- [官方课件 PDF](https://infra.seminars.lcpu.dev/slides/session04.pdf)
- [课程日历与讲座简介](https://infra.seminars.lcpu.dev/schedule)
