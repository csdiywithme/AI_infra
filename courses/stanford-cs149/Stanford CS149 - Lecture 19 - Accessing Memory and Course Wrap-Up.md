---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 19
lecture_date: 2023-12-07
area: systems
topics:
  - dram
  - memory-controller
  - row-buffer
  - bank-level-parallelism
  - memory-bandwidth
  - ddr
  - hbm
  - course-review
aliases:
  - CS149 Lecture 19
  - Accessing Memory
  - Course Wrap-Up
video_url: https://www.youtube.com/watch?v=J7v_ubArrno
---

# Stanford CS149 - Lecture 19 - Accessing Memory + Course Wrap-Up

> [!abstract]
> 课程最后一讲把前面一直抽象成“memory bandwidth”的黑盒拆开：DRAM cell 用电荷存 bit，访问先 precharge bitlines、activate 整行到 row buffer，再 column select/burst 到窄总线；因此 DRAM latency 取决于 row-buffer state，而 bandwidth 取决于 bulk transfer、bank-level parallelism、rank/channel width 和 request scheduling。Memory controller 将 physical addresses 映射为 channel/rank/bank/row/column，缓存并重排来自多核的大量 misses，在 row locality、bank parallelism、latency、fairness 和 energy 之间取舍。HBM 用 3D-stacked DRAM、through-silicon vias 和 interposer 把 memory 靠近 processor，以极宽接口换取更高 bandwidth、更低 energy/bit 和有限 capacity。全课最终归结为：找到 parallelism 往往不难，困难的是在 dependency 与 locality 约束下调度工作、把数据及时送到 compute；高层 abstraction 的价值，在于把这些复杂映射交给可复用的 compiler/runtime/system。

## 来源与范围

- [Lecture 19 视频：Accessing Memory + Course Wrap-Up](https://www.youtube.com/watch?v=J7v_ubArrno)
- [官方课件 PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/wrapup/18_wrapup.pdf)
- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)

课程网页/课件跳过 Midterm Review 编号，因此 PDF 标作 Lecture 18；本文按公开视频播放列表记作 Lecture 19。约 `00:44–41:39` 是 DRAM/HBM 技术内容，之后是全课总结、后续学习与研究建议；两部分均按视频保留。

## 视频索引

| 时间 | 内容 |
|---|---|
| [00:05](https://www.youtube.com/watch?v=J7v_ubArrno&t=5s) | 最后一讲安排：memory、课程总结、后续方向/AMA |
| [00:44](https://www.youtube.com/watch?v=J7v_ubArrno&t=44s) | CPU cache miss、last-level cache 与 memory controller |
| [02:36](https://www.youtube.com/watch?v=J7v_ubArrno&t=156s) | DRAM chip：二维 cell array、row buffer 与 data pins |
| [04:46](https://www.youtube.com/watch?v=J7v_ubArrno&t=286s) | Physical address/cache-line request 进入 DRAM |
| [06:01](https://www.youtube.com/watch?v=J7v_ubArrno&t=361s) | Precharge bitlines |
| [07:19](https://www.youtube.com/watch?v=J7v_ubArrno&t=439s) | Activate row；destructive read 到 row buffer |
| [07:59](https://www.youtube.com/watch?v=J7v_ubArrno&t=479s) | Column selection 与 data-bus transfer |
| [08:38](https://www.youtube.com/watch?v=J7v_ubArrno&t=518s) | Row-buffer hit：连续列访问更快 |
| [10:01](https://www.youtube.com/watch?v=J7v_ubArrno&t=601s) | 切行前 restore/write-back；row conflict 更慢 |
| [11:22](https://www.youtube.com/watch?v=J7v_ubArrno&t=682s) | 核心结论：DRAM latency 随 access pattern 变化 |
| [12:34](https://www.youtube.com/watch?v=J7v_ubArrno&t=754s) | RAS/CAS request stream 与 data-bus utilization |
| [14:00](https://www.youtube.com/watch?v=J7v_ubArrno&t=840s) | 两个 latency-hiding 手段：bulk transfer 与 pipeline |
| [15:14](https://www.youtube.com/watch?v=J7v_ubArrno&t=914s) | Cache line/contiguous traversal 与 DRAM burst efficiency |
| [16:20](https://www.youtube.com/watch?v=J7v_ubArrno&t=980s) | 多 banks 交错请求，实现 bank-level parallelism |
| [18:00](https://www.youtube.com/watch?v=J7v_ubArrno&t=1080s) | 多个 ×8 DRAM chips 组成 64-bit rank/DIMM |
| [20:33](https://www.youtube.com/watch?v=J7v_ubArrno&t=1233s) | Cache line 在 chips/banks/columns 间交错布局 |
| [24:10](https://www.youtube.com/watch?v=J7v_ubArrno&t=1450s) | Burst-mode transfer 与 command/data buses |
| [25:21](https://www.youtube.com/watch?v=J7v_ubArrno&t=1521s) | Channel 与 dual-channel memory |
| [25:58](https://www.youtube.com/watch?v=J7v_ubArrno&t=1558s) | Memory controller：address mapping、queueing、reordering |
| [27:47](https://www.youtube.com/watch?v=J7v_ubArrno&t=1667s) | Aggressive buffering 的 bandwidth–latency tradeoff |
| [30:27](https://www.youtube.com/watch?v=J7v_ubArrno&t=1827s) | DDR4-2400 bandwidth 算例 |
| [33:37](https://www.youtube.com/watch?v=J7v_ubArrno&t=2017s) | ECC memory：用额外 capacity/chip 提供 redundancy |
| [34:31](https://www.youtube.com/watch?v=J7v_ubArrno&t=2071s) | DRAM 小结：硬件重排 misses 以提高 row/bank locality |
| [35:18](https://www.youtube.com/watch?v=J7v_ubArrno&t=2118s) | 将 processor 与 memory 拉近：HBM/3D stacking |
| [37:18](https://www.youtube.com/watch?v=J7v_ubArrno&t=2238s) | TSV、silicon interposer 与超宽 memory interface |
| [38:16](https://www.youtube.com/watch?v=J7v_ubArrno&t=2296s) | 多 HBM stacks：高 bandwidth、有限 capacity |
| [40:03](https://www.youtube.com/watch?v=J7v_ubArrno&t=2403s) | HBM 同时改善 bandwidth、latency 与 energy/bit |
| [40:22](https://www.youtube.com/watch?v=J7v_ubArrno&t=2422s) | Memory lesson：把数据送到 processors 才是难点；compression 以算换带宽 |
| [41:53](https://www.youtube.com/watch?v=J7v_ubArrno&t=2513s) | 全课技术总结：heterogeneity、parallelism、specialization |
| [42:51](https://www.youtube.com/watch?v=J7v_ubArrno&t=2571s) | 现代软件与硬件 peak capability 的巨大差距 |
| [44:32](https://www.youtube.com/watch?v=J7v_ubArrno&t=2672s) | 三条主题：identify、schedule、abstract |
| [45:40](https://www.youtube.com/watch?v=J7v_ubArrno&t=2740s) | Stanford 后续课程方向 |
| [47:21](https://www.youtube.com/watch?v=J7v_ubArrno&t=2841s) | 通过 research/independent study 把实现能力用于真实项目 |
| [49:15](https://www.youtube.com/watch?v=J7v_ubArrno&t=2955s) | 大规模并行 RL/game simulation 项目案例 |
| [55:38](https://www.youtube.com/watch?v=J7v_ubArrno&t=3338s) | 从标准课程路径走向自主项目与方向选择 |
| [63:40](https://www.youtube.com/watch?v=J7v_ubArrno&t=3820s) | 加入研究项目的信号、准备与常见 pitfalls |

## 1. 从 load 到 DRAM：路径与角色

```text
load/store
  ↓ virtual→physical translation
L1 → L2 → LLC
             ↓ miss / cache-line request
        memory controller
             ↓ address mapping + scheduling
channel → DIMM/rank → chip → bank → row buffer → column/burst
```

CPU 发起的是一个 address，但 LLC miss 实际请求通常以 cache-line 为粒度。Memory controller 负责：

- 把 linear physical address 拆成 channel/rank/bank/row/column；
- 将 cache-line request 变成 DRAM commands；
- 满足 DRAM timing constraints；
- queue/reorder requests；
- 接收 burst data 并填回 cache；
- 在多 cores/requesters 间做 bandwidth、latency 与 fairness policy。

Software 通常不能直接选择某个 physical DRAM row；它只能通过 contiguous layout、stride、tiling、NUMA placement、huge pages 等间接影响 mapping 和 access pattern。

## 2. DRAM cell、row buffer 与一次访问

### 2.1 Cell 是模拟 charge storage

DRAM cell 以 capacitor 上的电荷表示 bit。Cell 很密集且便宜，但 charge 会泄漏，必须 refresh；read 也会扰动/破坏原值，需要 sense amplifier 读出后 restore。

二维 cell array 的每列连接 bitline，整行连接 wordline。Row buffer/sense amplifiers 同时读出一整行；外部 data pins 却很窄，所以内部一次 activate 的粒度远大于一次总线传输。

### 2.2 三个核心动作

1. **PRECHARGE**：把 bitlines 置到已知中间电位，为下一次 sensing 做准备；若旧行 open，先把 row-buffer state restore 回 cells；
2. **ACTIVATE / RAS**：打开目标 wordline，让 cell charge 扰动 bitline，sense amplifier 放大为 0/1，并把整行保存在 row buffer；
3. **READ/WRITE / CAS**：从 open row 选择 columns，经 data pins 进行 burst transfer。

课堂用约 `10 ns` 的阶段时间帮助建立数量级直觉；真实时序由 DDR generation、DIMM、frequency 与 timing parameters 决定。

### 2.3 Row state 导致 variable latency

| 当前 bank 状态 | 目标 | 需要的主要动作 | 相对代价 |
|---|---|---|---|
| 目标 row 已 open | 同 row | column read/write | row hit，最低 |
| bank idle/precharged | 任意 row | activate + column | row closed，居中 |
| 另一 row 已 open | 新 row | precharge/restore + activate + column | row conflict，最高 |

因此“DRAM access latency”不是一个固定数字。连续访问同一 row 可持续 burst；random pointer chasing 频繁换 row，会让 data bus 等待 precharge/activate，既高 latency 又低 effective bandwidth。

> [!important]
> Cache hit/miss 只是第一层不确定性。即使都 miss LLC，两个 requests 也可能因 row hit、bank overlap、queue position 不同而具有完全不同的 service time。

## 3. Bandwidth 的第一原则：让昂贵 pins 一直传数据

External data pins/board traces 是 scarce resource。若一次 request 的 command/activate 开销期间总线空闲，peak bandwidth 就没有被使用。

### 3.1 Bulk/burst transfer

一次 activate 后读取多个相邻 columns，把 setup cost 摊到更多 bytes：

$$
\text{effective BW}=\frac{\text{useful bytes}}{T_{precharge}+T_{activate}+T_{column/burst}}
$$

Burst 越长，固定前两项占比越小。这也是 contiguous traversal、cache-line transfer、coalescing/tiling 通常有效的物理原因之一：不仅用满了 cache line，也让 DRAM row/bus 更高效。

### 3.2 Bank-level parallelism

一个 bank 在 activate/precharge 时不能立即提供目标 data，但不同 banks 有独立 cell arrays/row state，可同时推进不同 requests，共享最终 data pins：

```text
time →
bank0: ACT ─ wait ─ DATA
bank1:     ACT ─ wait ─ DATA
bank2:         ACT ─ wait ─ DATA
bus:                 B0   B1   B2 ...
```

这与课程其他 latency-hiding 技术同构：

- CPU pipeline：不同 instructions 占据不同 stages；
- GPU multithreading：一个 warp 等 memory 时执行另一个 warp；
- DRAM banks：一个 bank 等 sensing 时，准备另一个 bank 的 request。

前提是地址能分散到不同 banks；大量 requests 冲向同一 bank 仍会 bank conflict/serialize。

## 4. Chip、rank/DIMM、bank、channel 的层级

这些名字容易混淆：

```text
memory controller
  ├─ channel 0 (独立 command/data interface)
  │    ├─ rank 0 / DIMM side
  │    │    ├─ ×8 chip 0: many banks
  │    │    ├─ ×8 chip 1: many banks
  │    │    └─ ... ×8 chip 7
  │    └─ optional rank 1
  └─ channel 1 ...
```

- **Bank**：chip 内可相对独立 activate 的 cell-array unit；
- **Chip width ×8**：单颗 chip 一次提供 8 bits；
- **Rank**：一组 chips 接收同一 command 并并行贡献 bits；8 颗 ×8 chips 合成 64-bit interface；
- **DIMM**：承载一个或多个 ranks 的 module；
- **Channel**：memory controller 到 DIMM(s) 的独立 command/data bus；dual channel 可并行发不同 commands，理论 bandwidth 近似翻倍。

同一 rank 内所有 chips 收到相同 bank/row/column command，但每颗保存不同 bit slice；合起来形成一个 bus word。

### 4.1 为什么 physical addresses 要 interleave

若整个 cache line 全落到一颗 ×8 chip，就浪费其他 data pins。实际 mapping 会把相邻 bytes/bit slices 分散到 rank 内 chips，再把后续 chunks 合理分散到 columns/banks/channels，使一个 64-byte cache line 可用 64-bit bus 的若干 bursts 取回，同时暴露 bank/channel parallelism。

Address mapping 没有一个普适公开公式：厂商会为 locality、parallelism、安全/row-hammer mitigation 等目标调整 bit placement。因此软件应依赖稳定的高层原则（contiguous/coalesced、足够 concurrency），而不是假定某几位恒等于 bank index。

## 5. DDR bandwidth 算例

课堂以 DDR4-2400、64-bit channel 为例：

- 实际 clock 约 `1.2 GHz`；
- Double Data Rate 在 rising/falling edges 各传一次，即 `2.4 GT/s`；
- 每 transfer 传 `64 bits = 8 bytes`。

$$
2.4\times10^9\ \text{transfers/s}\times 8\ \text{bytes}
=19.2\ \text{GB/s}
$$

Dual channel 理论峰值：

$$
2\times19.2=38.4\ \text{GB/s}
$$

注意：

- `DDR4-2400` 的 `2400` 是 mega-transfers/s 的营销命名，不是 `2.4 GHz` core clock；
- 这是 peak pin bandwidth，不包含 row conflicts、refresh、commands、turnaround、queueing 和 application utilization；
- CAS latency 只描述 open row 到 column data 的一段，不是 CPU load 的完整 end-to-end latency；还需算 cache hierarchy、NoC、controller、DRAM、return path。

## 6. Memory controller：在 request stream 上做动态调度

多核处理器持续生成 LLC misses。Controller 会 buffer 多个 requests，并寻找：

- open-row hits，避免 precharge/activate；
- 不同 banks/channels 的 independent requests，隐藏 latency；
- 足够长的 read/write batches，减少 bus direction turnaround；
- request age/QoS，避免某 core 永久 starvation；
- refresh/timing constraints 与 energy state。

一种经典直觉是 **FR-FCFS**（first-ready, first-come-first-serve）：优先 ready 的 row hits，再在同类中照 age 排。它提高吞吐，但可能让持续 row-hit stream 插队，伤害 latency/fairness。

### 6.1 Bandwidth–latency tradeoff

- 多 buffer、多 reorder：更容易凑 row hits/bank overlap，throughput 高；
- 立刻服务 oldest request：individual latency/predictability 较好，却可能频繁换 row；
- GPU 有大量 outstanding warps，偏向深 queue 与 bandwidth；
- real-time/latency-sensitive CPU workload 更关心 tail latency 与 isolation。

Memory controller 因而像另一台 out-of-order scheduler：过去大量复杂度用来从 instruction window 找 independent work，现在也用来从成千上万 memory requests 找最适合 DRAM state 的执行顺序。

### 6.2 Hardware reordering 与 software tiling 的对应

FlashAttention 由软件/algorithm 重排 computation 以命中 fast memory；memory controller 重排 cache misses 以命中 open rows/banks。两者都遵循：

```text
保持语义/依赖不变
        +
改变合法执行顺序
        ↓
提高 scarce resource utilization
```

硬件只能在已经到达 queue 的 requests 中选择。软件若只暴露一条 dependent linked-list chain，controller 没有足够 memory-level parallelism，再聪明也无事可调度。

## 7. Reliability：refresh 与 ECC

- **Refresh**：capacitor charge 会泄漏，DRAM 必须周期性读取/恢复 rows；refresh 会占用 bank/time，capacity 越大越显著；
- **ECC**：server memory 用额外 check bits/chip 存 redundancy，常见设计可纠正单 bit、检测多 bit errors；代价是额外 capacity、bandwidth、latency/energy。

视频用“8 颗 data chips 再加第 9 颗 redundancy chip”建立直觉；具体 ECC organization 会依 DIMM width/standard 变化。

## 8. HBM：用封装换一个更宽、更短的接口

传统 DIMM 通过 motherboard traces 连接 processor，长距离与 package pins 限制 interface width、frequency 和 energy。HBM（High Bandwidth Memory）采用：

- 多层 DRAM dies 垂直堆叠；
- **TSV（through-silicon via）** 穿过 dies 传递大量 signals；
- logic base die 管理 channels/banks；
- GPU/accelerator 与 HBM stacks 并排置于 silicon interposer/package；
- 极宽 interface（课堂示意每 stack 约 1024 bits），频率不必极高也可提供巨大 bandwidth。

缩短 wires、增加 pins 的结果：

- bandwidth 高；
- latency 通常更低；
- energy per transferred bit 更低；
- capacity 与封装成本受限，通常比 commodity DDR 小/贵。

所以系统可能形成更深 hierarchy：

```text
registers → SRAM/L1/L2/LLC → HBM → host DDR/CXL memory → storage/network
  fastest/smallest                                      slowest/largest
```

> [!important]
> HBM 本身仍是 DRAM，不是 cache。它可以由 software 当 device/global memory 使用，也可以在某些系统中作为另一层 cache/near memory；语义取决于 architecture。

### 8.1 与 FlashAttention 的准确对应

GPU attention 的关键层次通常是 on-chip SRAM/shared memory 与 off-chip HBM：naive implementation 在 HBM materialize $N\times N$ intermediates；FlashAttention 用 tiling/online softmax 将 live tiles 留在 SRAM/registers，减少 HBM traffic。视频口语中 “HBM vs. DRAM” 容易造成误解；准确说法是 **HBM 也是 DRAM，优化是在 HBM 与更小、更快的 on-chip memory 之间减少搬运**。

## 9. Compression：bandwidth-bound 时用 compute 换 bytes

若 execution units 因 memory 等待而 idle，可以额外执行 compression/decompression，换取更少 off-chip transfer：

$$
T_{compressed}=T_{compress}+\frac{B_{compressed}}{BW}+T_{decompress}
$$

只要它小于：

$$
T_{raw}=\frac{B_{raw}}{BW}
$$

就值得做。Graphics texture/framebuffer compression 是典型硬件例子；AI Infra 中的 quantization、KV-cache compression、activation compression、sparse encoding 也采用相同思想。

但 compression 不是无条件收益：random small blocks、poor compression ratio、latency-critical requests 或 compute 已饱和时，codec overhead 可能更糟。

## 10. 全课的三个统一主题

### 10.1 Identify parallelism

先画出 dependence：哪些 operations independent，哪些只能按 order。课程覆盖 data parallelism、SIMD/SIMT、task/graph parallelism、pipeline、distributed MapReduce/Spark 等。

### 10.2 Schedule parallelism

真正困难的是把已知 parallel work 映射到有限 resources：

- cores/warps/vector lanes；
- cache lines/banks/channels；
- queues, locks, atomic units；
- cluster workers/network/shuffles；
- accelerator stages/memories。

目标不只是 load balance，还要保持 locality、减少 communication/synchronization、隐藏 latency、控制 contention。

### 10.3 Raise the abstraction level

ISPC、CUDA、Halide、Spark、transactions、Spatial、PyTorch/SQL 等高层 abstraction 让 programmer 描述 intent/dependence，而由 compiler/runtime 在硬件细节上选择 schedule。

好的 abstraction 同时满足：

- 足以表达 correctness/algorithm；
- 给 implementation 留优化自由；
- 仍能让 performance programmer 理解 cost model；
- 需要时允许显式控制 locality/parallelism。

## 11. 课程知识地图

```text
为何并行
  └─ transistor/power constraints, throughput, latency hiding

如何找并行
  ├─ data parallel / SIMD / ISPC
  ├─ SIMT / CUDA / GPU
  ├─ graphs / work queues / task parallelism
  └─ MapReduce / Spark / distributed dataflow

如何让并行真的快
  ├─ workload balance + scheduling
  ├─ locality + cache / DRAM / HBM
  ├─ communication + synchronization
  ├─ roofline / arithmetic intensity
  └─ fusion / tiling / pipelining / streaming

如何保证正确
  ├─ coherence vs. consistency
  ├─ locks / atomics / lock-free
  └─ transactional memory

如何提升抽象又保留性能
  ├─ domain-specific languages
  └─ heterogeneous/specialized hardware
```

核心 performance question 可以浓缩成：

1. 做了多少 useful work？
2. 最稀缺的 resource 是什么？
3. 有没有足够 independent work 隐藏 latency？
4. 数据走了多远、走了几次？
5. schedule 是否让 dependency、locality 和 resource capacity 同时成立？

## 12. “软件很慢”时的数量级 sanity check

课程强调：现实软件距离硬件能力常有数量级差距。遇到“三小时任务”，先别默认必须加 cluster；先估算：

$$
T_{compute}\approx\frac{\#operations}{\text{achievable ops/s}},\qquad
T_{memory}\approx\frac{\#bytes moved}{\text{achievable BW}}
$$

再取主要瓶颈并加 synchronization/communication overhead。若估算是秒级而实测为小时级，优先检查：

- algorithmic complexity/重复工作；
- scalar vs. SIMD/tensor-core path；
- poor layout、cache/HBM thrashing；
- intermediate materialization；
- tiny kernels/tasks 与 launch/RPC overhead；
- serialization、lock contention、load imbalance；
- data format/conversion/compression；
- distributed shuffle/network。

Scale-out 会复制 inefficiency，也可能增加 coordination cost；但若单机 optimization 成本过高、working set 必须分布或 deadline 紧，cluster 仍是正确方案。关键是先有 cost model 再决定。

## 13. 视频中的大规模 simulation 案例

课堂展示为 RL agents 构建的专用 parallel game/simulation engine：传统方法启动许多 Unity/Unreal instances，重复 engine state/control，多个 copies 在同机 thrash；专用系统将成千上万 independent worlds lockstep/SoA 化，以类似 ISPC/CUDA 的方式跨 worlds 执行相同 physics、ray tracing 与 game logic。

核心 transformation：

```text
10,000 general-purpose engine processes
            ↓ batch/restructure
one data-parallel engine over 10,000 worlds
```

这把 instruction/control overhead 摊销，形成 coherent memory access，并适配 GPU wide parallelism。课堂报告若干 workload 达 `2–3` orders-of-magnitude speedup；它不是“GPU 自动让 Unity 快”，而是为目标 throughput 重写 engine/data layout/schedule。

这是全课思想的缩影：

- 找到 worlds 之间的 independent parallelism；
- 将 AoS/control-heavy execution 变成 lockstep data parallelism；
- 让 simulation 与 rendering 分别使用适合的 schedule；
- 用 specialization 让原本需要 cluster 的 trial generation 落到单 GPU。

## 14. 后续学习与项目建议

视频提到 Stanford 当时的 hardware programming/Spatial、CS229S、graphics/visual computing systems、EE hardware/OS 等课程。课程编号和开课时间会变，选课时应查当期 catalog；更持久的方向是：

- computer architecture / accelerator design；
- GPU programming、compiler/runtime、kernel DSL；
- distributed systems、databases、data processing；
- graphics/vision/ML systems；
- OS、memory systems、networking；
- performance engineering/research projects。

进入 research/independent study 的实用信号：已经学过基础课程、能指出具体感兴趣的 project/paper、能展示 assignment/side project 中超出要求的实现，并愿意先承担清晰的 engineering subproblem。项目匹配失败不一定是否定能力，也可能是 lab 当前没有稳定、可指导的切入口。

比“再上一门课”更进一步的练习是选一篇真实 systems paper：复现 baseline，建立 end-to-end profiler/cost model，只改变一个 schedule/layout/algorithmic decision，解释 speedup 来自 compute、bytes、parallelism 还是 synchronization。

## 15. 与 AI Infra 的连接

### 15.1 HBM bandwidth 是核心资源，但不是唯一数字

Training/serving kernel 需要区分 peak HBM BW、achieved BW、latency、capacity、bank/channel behavior。Batching 增加 outstanding requests 可隐藏 latency，但可能扩大 KV cache/queueing delay；controller/GPU scheduler 的目标也未必等于某个 request 的 tail latency。

### 15.2 Layout 影响到 DRAM 物理并行

Tensor contiguous/coalesced access 不仅减少 transactions，也更容易形成 burst、row locality 与 bank/channel utilization。Bad stride 可能让所有 warps 冲向少数 partitions/banks，即使总 bytes 不变仍降速。

### 15.3 Quantization 是 compute-for-bandwidth trade

FP16/BF16/INT8/FP8/weight-only quantization 减少 bytes 与提高 matrix-unit throughput；dequantization/fused scale 多做一些 arithmetic，但在 memory-bound decode 中经常净赚。必须以 end-to-end roofline 和 quality constraint 判断。

### 15.4 KV cache 体现 deep hierarchy

Hot KV blocks 可在 GPU HBM；更大 context/offloaded sessions 在 host DDR/CXL；cold state 甚至在 remote memory/storage。每层 capacity 更大但 bandwidth/latency 更差，调度器需要 prefetch、compression、eviction 与 admission control。

### 15.5 高层 framework 的价值取决于 cost transparency

PyTorch/SQL/Spark 让并行化与 fault handling 可复用，但 graph break、unexpected materialization、shuffle、host sync 会隐藏巨大成本。工程师既应使用 abstraction，也要能下钻到 execution graph、kernel、memory trace 与 hardware counter。

## 16. 易混点

1. **Row buffer 不是 CPU cache**：它是 DRAM bank 的 sense-amplifier state，按 row 开启且由 commands/timing 管理。
2. **DRAM latency 不是 CAS latency**：完整 miss 还含 precharge/activate、queue、interconnect、cache-fill 等。
3. **Bank 与 chip 不同**：一颗 chip 含多个 banks；rank 中多颗 chips 并行拼成 bus width。
4. **Rank 与 channel 不同**：同一 channel 可挂 ranks；不同 channels 可独立发 commands。
5. **DDR-2400 不是 2.4 GHz clock**：是约 2.4 GT/s，来自 1.2 GHz 的双边沿 transfer。
6. **Peak bandwidth 不等于 achieved bandwidth**：row conflicts、bank conflicts、refresh、read/write turnaround 与 insufficient concurrency 都会损失利用率。
7. **HBM 不是 SRAM/cache**：它是封装内的 stacked DRAM，仍比 on-chip SRAM 慢且耗能更高。
8. **Hardware request reordering 不能创造 MLP**：dependent pointer chain 一次只暴露一个 miss，controller 无法凭空并行化。
9. **Compression 不总是赢**：只有减少的 transfer time/energy 大于 codec cost 才划算。

## 17. 本讲结论

1. DRAM 用 row activation 将模拟 charge 批量放大到 row buffer，再以 column/burst 经窄总线输出。
2. Row hit、closed row、row conflict 的 command path 不同，所以同为 LLC miss，latency 也会变化。
3. Bulk transfer 摊销 command/activation 开销；多 banks 把内部 latency pipeline 化；多 chips/ranks 扩宽 word；多 channels 复制独立接口。
4. Memory controller 是高度动态的 scheduler，以 address mapping、queueing 与 reordering 提高 row/bank/channel utilization，同时权衡 latency、公平与能耗。
5. DDR 标称 bandwidth 由 transfer rate × bus width × channels 得到；它只是理论 pin rate。
6. HBM 通过 stacked DRAM、TSV 与 interposer 提供极宽且短的接口，以封装成本/capacity 换高 bandwidth、低 energy/bit。
7. Data placement、tiling、fusion、streaming、compression 的共同目标是少搬、近搬、批量搬，并暴露足够并行 requests。
8. CS149 的主线不是某个 API，而是从 dependencies 出发，找到 parallelism，再按 locality、communication 与 resource constraints 设计 schedule。
9. 高层 abstraction 让优化可复用；performance engineer 的职责是理解它隐藏的 execution/cost，并在必要时改变 algorithm、layout 或 mapping。

## 自测题

1. 一次 DRAM row conflict 为什么需要 precharge、activate、column 三步？Row hit 又省掉了什么？
2. 为什么 DRAM read 是 destructive 的？Row buffer/sense amplifier 在 restore 中扮演什么角色？
3. Cache line 为什么通常用 burst 传输，而不是逐 byte command？
4. Bank-level parallelism 与 GPU warp latency hiding 有什么共同结构？
5. 区分 bank、chip、rank、DIMM、channel，并说明 8 颗 ×8 chips 如何形成 64-bit rank。
6. 如何从 DDR4-2400、64-bit、dual-channel 推出 `38.4 GB/s`？为什么实测通常更低？
7. Memory controller 为什么要重排 requests？这种重排会伤害哪些 workload/metrics？
8. 为什么多个顺序扫描程序同时运行，也可能破坏彼此的 row locality？
9. HBM 的超宽接口为什么能在较低 pin frequency 下获得高 bandwidth 和较低 energy/bit？
10. 为什么说“把 attention matrix 放进 HBM”不是 FlashAttention 的核心优化？
11. 在什么条件下 compression/decompression 会提高 performance？
12. Controller 为什么无法加速严格 dependent linked-list traversal？Software 能如何改变这个局面？
13. 用 identify–schedule–abstract 三步重新解释一次 CUDA、Spark 或 Halide 课程案例。
14. 面对一个耗时三小时的 AI pipeline，你会如何用 compute/byte 下界判断先单机优化还是直接 scale out？

## 相关笔记

- [[Stanford CS149 - Lecture 03 - Multi-Core Architecture Part II and ISPC]]
- [[Stanford CS149 - Lecture 04 - Parallel Programming Basics]]
- [[Stanford CS149 - Lecture 07 - GPU Architecture and CUDA Programming]]
- [[Stanford CS149 - Lecture 09 - Distributed Data-Parallel Computing Using Spark]]
- [[Stanford CS149 - Lecture 10 - Efficiently Evaluating DNNs on GPUs]]
- [[Stanford CS149 - Lecture 11 - Cache Coherence]]
- [[Stanford CS149 - Lecture 12 - Memory Consistency]]
- [[Stanford CS149 - Lecture 15 - Domain-Specific Programming Languages]]
- [[Stanford CS149 - Lecture 18 - Hardware Specialization]]
- [[Stanford CS149]]
