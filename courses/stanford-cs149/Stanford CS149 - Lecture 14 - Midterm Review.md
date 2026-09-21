---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 14
lecture_date: 2023-11-14
area: systems
topics:
  - midterm-review
  - parallel-architecture
  - performance-optimization
  - mapreduce
  - cache-coherence
  - memory-consistency
  - data-parallel
  - cuda
aliases:
  - CS149 Lecture 14
  - Midterm Review
video_url: https://www.youtube.com/watch?v=nHPKVtLz5Ko
---

# Stanford CS149 - Lecture 14 - Midterm Review

> [!abstract]
> 这不是逐章重播，而是一堂“怎样把概念用于推理”的答疑课。核心方法是：先写清 abstraction/API 的 semantics，再把 logical work 映射到具体 execution resources，最后沿时间线追踪 work、data、permission 与 visibility。课堂依次复习 CAS/lock-free 的 speculation-and-validation、MapReduce 的 `map → shuffle/groupByKey → reduce`、MSI/MESI 的 per-line state machine、relaxed consistency 的 cross-thread visibility、segmented scan 的 primitive composition，以及 CUDA thread block/warp 的资源映射。考试不靠死背图，而靠能否逐步模拟系统。

## 来源与范围

- [Lecture 14 视频：Midterm Review](https://www.youtube.com/watch?v=nHPKVtLz5Ko)
- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)

本次 review 没有独立官方课件；教师现场打开前面讲次的 slides 并围绕学生提问展开。本文以视频为唯一主线，并把回答链接回对应的 Lecture 1–13 笔记。课程网页的课件编号跳过 review，所以下一份 DSL 课件写作 “Lecture 14”，但公开视频与本仓库按播放列表把它记作 Lecture 15。

## 视频索引

| 时间 | 内容 |
|---|---|
| [00:11](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=11s) | Midterm 的知识主线与题目风格 |
| [01:07](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=67s) | GPU/CUDA、DNN locality、data-parallel thinking |
| [02:27](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=147s) | Fine-grained locking、MSI、relaxed consistency |
| [04:17](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=257s) | CAS 与 lock-free 复习 |
| [06:09](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=369s) | 用 CAS 实现 `atomicMin` |
| [10:48](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=648s) | Lock-free：不 mutual-exclude，而是提交时验证 |
| [12:18](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=738s) | Lock-free 与 starvation/fairness 正交 |
| [13:36](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=816s) | CAS 的 coherence 成本 |
| [15:56](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=956s) | MapReduce 问答开始 |
| [17:59](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=1079s) | Map 产生 `(key,value)`，shuffle 按 key 重组 |
| [21:18](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=1278s) | Reduce 对每个 unique key 的 value list 运行 |
| [24:14](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=1454s) | Disk workload 更受 bandwidth 而非单请求 latency 限制 |
| [26:08](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=1568s) | Parallelism 不能弥补 locality/bandwidth 的灾难 |
| [28:15](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=1695s) | Scale-out/microservices 的 cost 反例 |
| [30:24](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=1824s) | MSI 问答开始 |
| [31:27](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=1887s) | Atomic CAS 从 `I` 直接按 write/BusRdX 处理 |
| [33:38](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=2018s) | 两位学生现场扮演 caches 走 MSI trace |
| [39:56](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=2396s) | 正确但不完全按 MSI：invalidate vs. 保留 shared copy |
| [43:41](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=2621s) | 每个地址/cache line 都有独立 coherence state |
| [45:35](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=2735s) | Modified owner、writeback 与 cache-to-cache transfer |
| [48:44](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=2924s) | MESI `E` state 的 read-then-write 优化 |
| [52:43](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=3163s) | Eviction 是普通 cache event，也会触发 coherence transition |
| [55:38](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=3338s) | Memory consistency：另一线程可见的顺序 |
| [58:29](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=3509s) | 单线程必须仍观察到 program semantics |
| [61:52](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=3712s) | Segmented scan 的定义与使用要求 |
| [63:58](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=3838s) | CUDA bulk launch 与 Assignment 2 task system 类比 |
| [65:25](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=3925s) | Threads/shared memory 共同限制 resident blocks |
| [67:32](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=4052s) | 为什么先把 blocks 分散到空闲 SMs |
| [69:42](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=4182s) | Warp：32 CUDA threads 的 SIMD execution group |
| [71:18](https://www.youtube.com/watch?v=nHPKVtLz5Ko&t=4278s) | CUDA threads 与 CPU SIMD/ISPC 的 abstraction 差异 |

## 1. Midterm 需要的不是清单，而是五种推理动作

### 1.1 从 abstraction semantics 开始

看到 ISPC、CUDA、MapReduce、Spark、CAS，先问 API 保证什么，而不是先猜实现：

- ISPC `programCount/programIndex` 描述 gang/lane；
- CUDA kernel launch 描述 blocks × threads；
- MapReduce `map` 输出 key-value pairs，system group/shuffle，`reduce` 接受 key + values；
- CAS 是单个 atomic read-compare-write；
- MSI state 描述本 cache 对某 line 的 permission。

### 1.2 画执行时间线

并发题要明确每个 thread 的 program order，再列可能 interleavings。Memory consistency 题还要区分：

- 本线程逻辑必须符合 sequential semantics；
- 其他 threads 何时观察到 stores，是 model 允许变化的部分。

### 1.3 追踪物理资源

```text
logical work
  → task / block / gang
  → hardware thread / warp / SIMD lane
  → core / SM
  → cache/shared memory/registers
```

算 resident blocks 时，thread slots、registers、shared memory、block limit 任一都可成为最小约束。

### 1.4 追踪 data 与 permission

每次 cache/MSI/CAS 操作都写：

```text
value 在哪里？
谁能读？谁能写？
谁持有最新 copy？
操作会发什么 transaction？
其他 caches 的 state 怎样变化？
```

### 1.5 最后才评价性能

用课程贯穿的四项：work distribution、locality、communication、contention。不要把“更多 parallelism”直接等同于“更快”。

## 2. 知识地图与最低掌握标准

| 模块 | 必须能做的事 | 常见误区 |
|---|---|---|
| Multi-core architecture | 区分 multi-core、SIMD、HW multithreading，分析 utilization | 把 concurrency 当 simultaneous execution |
| ISPC | 根据 gang width、mask、tasks 推导实际 work | 把 `programCount` 当 core count |
| Work distribution | 找 imbalance、overhead、critical path | 只算平均 work，不看 tail |
| Locality/communication | 改 loop order、tiling/fusion，估 traffic | 只数 FLOPs，不数 bytes |
| CUDA | 从 grid/block 到 SM/warp，算 resource limit | 把 warp 当 language-level guarantee |
| Data-parallel | 用 map/reduce/scan/segmented scan 重述算法 | 强行给每个 element 独立 thread，忽略 prefix/dependency |
| MapReduce/Spark | 区分 map、shuffle、reduce；narrow/wide dependency | 把 MapReduce `reduce` 等同普通 parallel reduction |
| Coherence | 逐步走 MSI/MESI trace | 混淆 lock ownership 与 cache-line ownership |
| Consistency | 分析 allowed reorderings 与 litmus outcome | 把同地址 coherence 当跨地址 ordering |
| Synchronization | 用 CAS 构造 atomic，找 race/deadlock | 认为 atomic variable 自动 order 所有普通 data |

## 3. Lock-free/CAS：考试要求到哪一层

课程明确：midterm 要理解 CAS 与“speculate → validate → retry”，能用 CAS 实现简单 atomic primitive；不会要求现场写完整 lock-free linked-list insert/delete。

### 3.1 CAS-based operation 的 invariant

```text
candidate result 只在 shared state 仍等于 initial observation 时才有效
```

这不是 mutual exclusion：两个 threads 可以同时 read/compute，只要不产生冲突，二者都可能成功；冲突时至少一个 CAS 失败重试。

### 3.2 Progress 与 fairness 分开

Lock-free 只保证 system-wide progress，不保证某 thread 成功；普通 spin lock 也未必公平。Fairness 可通过 ticket/queue、priority、backoff 等另行设计。

### 3.3 CAS 的成本

CAS 即使最终“不改值”，也必须当作可能 write 的 RMW：取得 `M`/exclusive ownership，在 compare/write 完成前防止 line 被另一 core 抢走。因此它不是“普通 load + 一个 if”，而是昂贵的 coherence serialization point。

## 4. MapReduce：名字相似，不等于 data-parallel primitive 完全相同

### 4.1 三阶段 semantics

```text
map(input record) -> zero or more (key, value)
shuffle/groupByKey -> key -> [value0, value1, ...]
reduce(key, values) -> output for that key
```

Map phase data-parallel 地作用于 records；shuffle 才是最重的 global communication；Reduce function 被“map 到每个 unique key”，每个 key 内再聚合 values。

### 4.2 为什么早期实现大量用 distributed file system

Google 已有可扩展、复制、容错的 distributed storage；用文件承接 stage boundary，fault tolerance 与 scheduling 简单。但这让每个 stage 读写海量 intermediate data，bandwidth 成为主导。

### 4.3 为什么 disk latency 在这里不是第一问题

Record-level work 丰富，可同时发许多 independent I/Os、prefetch 下一批，隐藏单次 latency；但总 bytes 必须真的通过 disk/network，aggregate bandwidth 不能被 hiding。故瓶颈是 throughput。

### 4.4 Scale-out 反例应怎样理解

若 data 能在一台大内存 server 内处理，跨 thousands of workers 的 serialization、shuffle、storage 与 fault-tolerance bookkeeping 可能让它慢于单机 streaming。Scale-out 的价值首先是 capacity/availability，而不是免费 speedup。

## 5. MSI/MESI：现场演示的五个高频坑

### 5.1 State 是 per cache line，不是全 cache 一个状态

同一 cache 可同时有 `X:S`、`Y:M`；`X` 的 transaction 不直接改变 `Y`。只有 capacity/conflict eviction 可能间接把 `Y` 驱逐。

### 5.2 State change 必须通知相关 participants

Cache controller 只知道 local metadata 与收到的 protocol messages。若一方从 shared 意图写而没有 invalidation，其他 cache 仍会依据旧 `S` permission 继续读 stale data。

### 5.3 “正确”不等于“协议规定的最优动作”

某 cache 在别人 read 后把自己的 valid copy 从 `M` 直接丢成 `I`，可能仍保持 coherence，但 MSI 规定保留为 `S`，因为将来 local read 可 hit。Protocol 不只维持 correctness，也在规定性能策略。

### 5.4 Modified owner 必须提供整条最新 line

写回的是 cache line，不只是变量 `X`。Line 内其他 bytes 也属于同一 coherence unit；实现若不跟踪 byte-level dirty，必须整体 transfer/writeback。

现代 MOESI-like protocol 可 cache-to-cache transfer，避免 requester 先等 memory；课程 MSI 简化成 owner 写回、requester 从更新后的 memory 获得。

### 5.5 MESI 的 `E` 优化 read-then-write

第一次 read 若无人持有，进入 clean exclusive `E`；后续 local write silent `E→M`，省掉 MSI 中 `S→M` 的 broadcast upgrade。若 snoop 到别人的 read，则 `E→S`。

## 6. Consistency：一定用“谁观察到什么”描述

Relaxed consistency 不是让单线程结果任意改变。Compiler/processor 可内部 OoO，但单线程 observable result 必须符合语言语义。

Relaxation发生在另一 processor 的 observation：

```text
T0 program order: W(X)=1; W(Y)=1
T1 may observe:   Y becomes 1 before X becomes 1
```

前提是 model 放松 `W→W`。这与 coherence 正交：对 `X` 的所有 writes 仍可 coherent，对 `Y` 也仍可 coherent；只是跨地址没有同一个 global order。

答 litmus 题的步骤：

1. 写 per-thread program-order edges；
2. 为期望 read result 添加 read-from / “必须在某 write 前后” edges；
3. 根据给定 memory model 保留或删除 ordering edges；
4. 查是否形成 cycle；
5. 若是 relaxed model，再检查 fence/acquire-release 是否补回 edges。

## 7. Segmented scan：复习重点是 primitive，而非背并行实现

Segmented scan 对一组变长 sequences 分别 scan：

```text
values:   [1 2 3 | 4 4 7 7 8]
exclusive sum:
          [0 1 3 | 0 4 8 15 22]
```

它可编码成 flat values + segment-start flags，并用 associative pair operator 在一次 parallel scan 中阻止前缀跨 segment 泄漏。

课程对 midterm 的要求：给出 segmented-scan primitive 后，能用它设计 higher-level data-parallel algorithm；不要求从零背出其并行 scan network。

## 8. CUDA：两级 bulk launch 与资源约束

Kernel launch 描述：

```text
grid = N blocks
each block = T CUDA threads + R registers/thread + S shared memory/block
```

Runtime 把 blocks 动态分配给 SM。一个 SM 的 resident blocks 上限：

$$
B_{resident}=\min\left(
B_{hw},
\left\lfloor\frac{T_{SM}}{T_{block}}\right\rfloor,
\left\lfloor\frac{S_{SM}}{S_{block}}\right\rfloor,
\left\lfloor\frac{R_{SM}}{R_{block}}\right\rfloor
\right)
$$

课堂例子中 thread slots 可容纳 3 blocks，但 shared memory 只容纳 2，因此实际 resident 2。Occupancy 是多个限制的最小值。

### 8.1 为什么先把 blocks 分散到空闲 SM

把两个 blocks 堆在一个 SM 会让它们 time-multiplex 同一组 ALUs，而另一个 SM idle；先占满不同 SM 才利用更多并行 compute resources。

### 8.2 Warp 是实现 grouping，不是 source abstraction

历史 NVIDIA 实现把同一 block 中连续 32 CUDA threads 组成 warp，在 32 lanes 上执行同一 instruction。CUDA source 写 scalar thread；hardware 可决定如何组成/调度 execution groups。

对比 ISPC/CPU SIMD：compiler 面向显式 ISA vector width 生成 vector instructions；若 ISA width 变化，常需重新编译/调整 gang。CUDA thread abstraction 给 vendor 更大实现自由度，但性能调优仍要理解当前 warp behavior、divergence 与 coalescing。

## 9. 一页答题模板

### 架构/映射题

```text
logical work:
execution grouping:
physical resources:
resident work limit:
idle resources and cause:
latency hiding mechanism:
```

### 性能优化题

```text
baseline work / bytes / synchronization:
bottleneck evidence:
transformation:
what traffic or idle time is removed:
new overhead / tradeoff:
```

### Coherence 题

```text
for each step:
local hit/miss → bus/message → data source → all cache states → memory freshness
```

### Consistency 题

```text
program-order edges + read-from constraints + model-preserved edges
→ cycle? → legal outcome?
```

### CAS/并发结构题

```text
observed state → candidate update → validation point → retry path
linearization point:
progress guarantee:
memory reclamation / ordering assumptions:
```

## 10. 与 AI Infra 的连接

- GPU kernel tuning 本质仍是 block/warp mapping、resident resources、locality 与 communication；模型名字不会改变这套分析。
- Distributed serving 使用微服务/函数化 scale-out 时，要把 serialization、network、state movement 与 per-request overhead 计入，不能只看 autoscaling convenience。
- Lock-free scheduler/allocator 既要证明 CAS logic，也要处理 memory ordering 与 reclamation；生产级难点通常在后两项。
- Model execution graph 的 fusion、Spark narrow-dependency fusion 与 FlashAttention 都在做相同事：让 producer/consumer 在高带宽本地层直接交接，避免 materialized intermediate。

## 11. 自测题

1. 为什么 CAS 即使条件失败，也通常需要按 write/RMW 获取 exclusive cache permission？
2. MapReduce 的 reduce 与普通 parallel reduce 有什么语义差异？
3. 为什么 massively parallel disk workload 可以隐藏 latency，却仍逃不过 bandwidth？
4. 对 `P0:R(X), W(X); P1:R(X)` 逐步写出 MSI transitions 与 data source。
5. “另一个 cache read 时，M owner 直接丢成 I”为什么可能正确但低效？
6. Relaxed consistency 为什么不会允许单线程 `x=1; print(x)` 打印旧值？
7. 用 segment flags 描述 segmented scan 应怎样阻止跨 segment 累积？
8. 给定 SM thread/shared-memory/register limits，计算 resident blocks。
9. CUDA threads 与 ISPC lanes 的 semantics/implementation 边界有何不同？
10. 面对一个未知并行程序，如何按 work、mapping、locality、communication、contention 五步定位瓶颈？

## 相关笔记

- [[Stanford CS149 - Lecture 07 - GPU Architecture and CUDA Programming]]
- [[Stanford CS149 - Lecture 08 - Data-Parallel Thinking]]
- [[Stanford CS149 - Lecture 09 - Distributed Data-Parallel Computing Using Spark]]
- [[Stanford CS149 - Lecture 11 - Cache Coherence]]
- [[Stanford CS149 - Lecture 12 - Memory Consistency]]
- [[Stanford CS149 - Lecture 13 - Fine-Grained Synchronization and Lock-Free Programming]]
- [[Stanford CS149 - Lecture 15 - Domain-Specific Programming Languages]]
