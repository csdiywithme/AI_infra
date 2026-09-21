---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 11
lecture_date: 2023-10-31
area: systems
topics:
  - spark
  - cache
  - cache-coherence
  - shared-memory
  - snooping
  - msi
aliases:
  - CS149 Lecture 11
  - Cache Coherence
video_url: https://www.youtube.com/watch?v=lrCfG2CPDEw
---

# Stanford CS149 - Lecture 11 - Cache Coherence

> [!abstract]
> 本讲前 27 分钟先收尾 Spark：narrow dependency 允许 fusion，lineage 用可重放的确定性变换实现 fault tolerance，而 scale-out 只有在单机装不下数据时才值得支付通信与运行时开销。之后进入 cache coherence：private write-back caches 会制造同一地址的多个、可能互相矛盾的副本；coherence 的任务是让单地址访问表现为一个尊重各线程 program order 的全局串行序列。实现抓手是 **single-writer/multiple-reader** 与 **data-value** 两个 invariant，经典 snooping 协议借助 bus 的 broadcast 与 serialization 属性维护它们，并引出 MSI 状态机。

## 来源与范围

- [Lecture 11 视频：Cache Coherence](https://www.youtube.com/watch?v=lrCfG2CPDEw)
- [官方课件 PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/cachecoherence/11_coherence.pdf)
- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)

本文按视频顺序整理。课程标题虽是 Cache Coherence，但视频先用约 27 分钟完成 Lecture 9 遗留的 Spark 内容；这一段与后半讲共享同一条主线：并行性能不只取决于并行度，还取决于 locality、communication 与 abstraction 的实现代价。

## 视频索引

| 时间 | 内容 |
|---|---|
| [00:55](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=55s) | 回到 Spark：为什么 intermediate data 不应反复写 HDFS |
| [03:42](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=222s) | RDD 如何实现，为什么朴素 materialization 太大 |
| [04:20](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=260s) | Fusion 与 tiling 都是在提高 locality/arithmetic intensity |
| [06:27](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=387s) | Narrow dependency 允许跨 transformations fusion |
| [07:29](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=449s) | Wide dependency、groupByKey 与跨节点 shuffle |
| [09:31](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=571s) | 相同 partitioner 如何把 join 变成 narrow dependency |
| [11:55](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=715s) | Lineage：用 transformation log 恢复丢失 partition |
| [18:30](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=1110s) | Spark 相对 Hadoop 的迭代算法收益 |
| [23:32](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=1412s) | 不要把 scalability 当 performance：数据装得下时单机可能更快 |
| [26:33](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=1593s) | Scale-up 与 scale-out |
| [27:30](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=1650s) | 正式进入 cache coherence |
| [29:19](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=1759s) | Cache line、cold miss 与 spatial locality 回顾 |
| [32:20](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=1940s) | Capacity miss；三 C miss model |
| [33:32](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=2012s) | Set associativity 与 conflict miss |
| [39:10](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=2350s) | Cache line metadata：tag 与 dirty bit |
| [41:01](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=2461s) | Write-through vs. write-back；write-allocate vs. no-write-allocate |
| [42:31](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=2551s) | Write-allocate + write-back miss 的完整动作 |
| [50:56](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=3056s) | Shared-memory abstraction 遇到 private caches |
| [53:24](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=3204s) | 三个处理器看到 `X=0/1/2`：incoherence 反例 |
| [55:49](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=3349s) | 为什么 lock 不能修复 cache coherence |
| [58:49](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=3529s) | “last write” 在并行系统中必须被精确定义 |
| [59:56](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=3596s) | Coherence 是单地址访问的 serialization |
| [61:45](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=3705s) | 两个 invariant：SWMR 与 data-value |
| [66:28](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=3988s) | 软件方案、snooping 与 directory 方案 |
| [70:35](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=4235s) | Snooping：每个 cache controller 监听 interconnect |
| [73:40](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=4420s) | Bus 的两项关键性质：broadcast + serialization |
| [78:26](https://www.youtube.com/watch?v=lrCfG2CPDEw&t=4706s) | MSI 状态与 BusRd / BusRdX / BusWB |

## 0. Spark 收尾：parallelism 之外必须计算 locality

Spark 针对的是在 cluster 上反复复用 intermediate data 的计算，例如 iterative ML、graph algorithms 与对同一大数据集的多次交互查询。MapReduce 把中间结果写入 replicated distributed file system，因此故障恢复容易，但每轮都支付 disk/network traffic。

RDD 的核心约束是：

- read-only、ordered collection of records；
- 由 persistent data 或已有 RDD 上的 deterministic、functional transformations 产生；
- transformation 不修改输入，action 才把结果返回应用；
- runtime 可以选择是否 materialize/persist，不必为每个逻辑 RDD 保留完整副本。

这让系统能够把“逻辑数据集”与“物理保存的一份大数组”分开。

### 0.1 Narrow dependency 允许 fusion

若 child RDD 的每个 partition 最多依赖一个 parent partition，就是 narrow dependency：

```text
load partition i
  → map partition i
  → filter partition i
  → local partial action
```

这些操作可在一条 streaming pipeline 中融合：一个 record 读进来后依次 map/filter，不物化中间 RDD，也无需跨节点通信。它与 CPU/GPU loop fusion 的目标相同：减少 intermediate traffic，提高 producer-consumer locality 与 arithmetic intensity。

### 0.2 Wide dependency 建立 stage boundary

`groupByKey`、未对齐 partitioning 的 `join` 或全局 `sort` 会让一个 child partition 依赖多个 parent partitions：

```text
parent partitions ── all-to-all shuffle ──> child partitions
```

wide dependency 意味着：

- 前一 stage 通常要整体完成，下一 stage 才能开始；
- 产生 network shuffle 与更大的 temporary state；
- 节点失败可能触发更广的 ancestor recomputation；
- fusion 无法跨越这条边界。

若两个 key-value RDD 使用同一个 `HashPartitioner`，相同 key 已共置到同一 partition，`join` 就可能只产生 narrow dependencies。这里 partitioning 是 physical layout contract，不只是 API 装饰。

### 0.3 Lineage：记录“怎样算”，而不是复制所有中间结果

RDD transformations 是 deterministic + functional 的，因此一个 partition 丢失时，可以从仍可靠存在的 input block 沿 lineage 重放：

```text
HDFS block
  → map/filter...
  → lost partition
```

lineage 记录的是粗粒度 bulk operations，而不是每条 record 的细粒度 mutation log，元数据小。它把 fault tolerance 从“每个 intermediate 都同步复制”改成“必要时重算”。

> [!important]
> Lineage 并不等于永不 materialize。最终 action、跨 stage 的 shuffle、用户显式 `persist()`、重复使用的昂贵 RDD 都可能需要保存；它只是让 runtime 拥有 storage/recompute 的选择权。

### 0.4 Scale-out 不等于快

视频引用的反例说明：若整个 graph/data set 已能装入一台大内存服务器，128 cores 的 Spark 可能仍慢于单线程程序。原因不是 parallelism 不存在，而是 distributed runtime、serialization、network 与 scheduling overhead 淹没了收益。

```text
scale-up  = 一台 shared-memory machine 内增加 cores/resources
scale-out = 用 network 连接多个独立 OS、独立 memory 的 nodes
```

选择 scale-out 的强理由是 **容量或容错需求确实越过单机边界**，而不是“节点更多看起来更 scalable”。

## 1. Cache 回顾：cache line 与三 C miss model

Cache line 把连续 bytes 一起搬运，既利用 spatial locality，也让 memory path 与 coherence metadata 的管理粒度可接受。

| Miss | 含义 | 加大总容量是否能消除 |
|---|---|---|
| Compulsory / cold | 第一次访问该 line | 不能根本消除，可 prefetch |
| Capacity | working set 超过 cache capacity | 可以 |
| Conflict | set associativity 限制了某地址可放的位置 | 提高 associativity 可减少 |

若 32 KiB L1、64 B line，则共有 `32 KiB / 64 B = 512` lines。Fully associative lookup 要比较 512 个 tags；8-way set associative 只需同时查 8 个 candidates，代价更低，但某个 set 满时会在其他 sets 尚空闲的情况下发生 conflict miss。

> [!warning]
> 视频纠正了早期课件示例：仅因 cache 太小而替换是 capacity miss；只有“总容量够但目标 set 冲突”才是 conflict miss。

## 2. Cache line 里不只有 data

典型 metadata 至少包括：

- `tag`：标识当前 line 对应哪个 memory block；不是单个变量的完整 byte address；
- `valid/state`：该 entry 是否可用，以及后续 coherence state；
- `dirty`：cache copy 是否比 memory 新；
- replacement/prefetch/ECC 等实现相关状态。

处理器给出地址后，index 选择 set，tag 与 set 中各 way 比较；line offset 再选择 line 内 byte/word。

### 2.1 两组不要混淆的写策略

| 维度 | 选项 | 关键区别 |
|---|---|---|
| 命中后的写去哪里 | write-through / write-back | 每次同时写 memory，或先只改 cache、eviction 时回写 |
| write miss 是否装入 line | write-allocate / no-write-allocate | 先 fetch 整行再局部写，或直接写下层不分配 |

Write-back + write-allocate 的 miss 路径：

1. 选择 victim；
2. victim dirty 则先 write back；
3. 从下层取回目标完整 cache line；
4. 修改其中目标 word；
5. 更新 tag/state，置 dirty。

必须取整行，是因为 store 通常只提供一个 word，line 里其他 bytes 仍需有正确值。

## 3. 为什么 private write-back caches 会破坏 shared memory

设 memory 中 `X=0`，三个处理器各有 private cache：

```text
P1 load X → P1 cache: 0
P2 load X → P2 cache: 0
P1 store X=1 → P1 cache: 1 (dirty), memory still 0
P3 load X → P3 cache: 0
P3 store X=2 → P3 cache: 2 (dirty)
P2 load X → still returns local 0
P1 evicts X → writes 1 to memory
```

此时同一地址同时存在 `memory=1`、`P2=0`、`P3=2`。这已不是“线程调度顺序不同”，而是 memory abstraction 本身没有定义。

### 3.1 为什么 lock 不能替代 coherence

即使用 lock 把 P1、P3 的 store 串行化，P1 仍可能只更新自己的 cache，P3 仍可能从 stale local copy 继续执行。Lock 决定“谁现在可以进入 critical section”；coherence 保证“每个 cache 中共享地址的副本符合共同的单地址顺序”。两者解决不同层的问题。

同理，单处理器也可能出现 coherence 问题：DMA device 更新 memory 中的 network buffer，而 CPU cache 仍保留旧 line。若频率低，可由 software flush/invalidate；多核共享内存的频率太高，通常需要 cache-line-granular hardware coherence。

## 4. Coherence 的精确定义

“读返回 last write”不够，因为并发 writes 没有天然的 wall-clock “最后”。更可操作的定义是：对每个地址 `X`，所有 processors 对 `X` 的 reads/writes 可以排成一个 total order，满足：

1. 每个 processor 自己对 `X` 的 operations 在该 order 中保持 program order；
2. 每次 read 返回该 order 中位于它之前的最近一次 write 的值。

```text
P0: W(X)=5 ────────────────
P1:          R(X)=5  W(X)=25
P2:             R(X)=5       R(X)=25

一种合法 serialization:
W0(5) < R1(5) < R2(5) < W1(25) < R2(25)
```

> [!note]
> Coherence 是 **per-address** property。不同地址之间的观察顺序是下一讲 memory consistency 的问题。

## 5. 两个 coherence invariants

### 5.1 Single-writer / multiple-reader（SWMR）

任一 cache line 在任一时刻只能处于两类 epoch：

- read-write epoch：恰好一个 cache 拥有可写权限；
- read-only epoch：零个或多个 caches 拥有只读副本。

它禁止“两个 private caches 同时悄悄写同一 line”。

### 5.2 Data-value invariant

进入新的 read-only epoch 后，所有 readers 必须看到前一个 read-write epoch 最后写出的值。SWMR 只管权限，data-value invariant 管最新数据如何随 ownership/epoch 转移。

两者缺一不可：

- 只有 SWMR 但交接旧数据：权限正确、值错误；
- 只有最新数据但允许多个 writers：马上又产生冲突副本。

## 6. 实现路线：software、shared cache、snooping、directory

### 6.1 Software/page-granular coherence

可以用 virtual-memory protection 与 OS trap 追踪 sharing，但 page 太大、trap 太慢，且 false sharing 极重，通常不适合频繁共享。

### 6.2 单一 shared cache

天然减少 private copies，但所有 processors 争同一 bandwidth；unshared data 也互相驱逐，产生 destructive interference。可作为较低层的 shared LLC，但很难作为大量 cores 的 L1。

### 6.3 Snooping

每个 cache controller 同时观察：

- local processor 的 read/write；
- interconnect 上其他 caches 的 coherence transactions。

Bus 很适合教学版 snooping，因为：

1. **broadcast**：一次 transaction 所有 caches 都听见；
2. **serialization**：一次只允许一个 transaction，天然给出全局次序。

代价是 bandwidth 与 electrical fan-out 不随 core count 良好扩展。

### 6.4 Directory

Directory 记录某 line 当前在哪些 caches 中，向 sharers/owner 定点发送请求与 invalidation，不必全局广播，也不必把不同 lines 的无关 transactions 全部串行化。详细机制在下一讲继续。

## 7. Write-through invalidation 为什么仍不够

一种最简单方案：每次 store 都更新 memory，并广播 `invalidate(X)`。它能避免 stale copies，但 **每个 store 都占 interconnect**。局部反复更新同一变量本应在 L1 中高速完成，却被迫产生全局通信。

因此实际目标是 write-back invalidation：获得 exclusive ownership 后，processor 可在 private cache 中多次 read/write，直到 ownership 转移或 eviction 才通信。

## 8. MSI：本讲建立 vocabulary，下一讲走状态机

| State | 含义 | Memory 是否一定最新 |
|---|---|---|
| `M` Modified | 只有本 cache 有有效副本，可读写，line dirty | 否 |
| `S` Shared | 一个或多个 caches 有 clean、read-only copy | 是 |
| `I` Invalid | 本 cache 没有可用副本 | 不适用 |

常见 bus transactions：

| Transaction | 请求者意图 |
|---|---|
| `BusRd` | 取 line 供读取 |
| `BusRdX` | 取 line 并获得 exclusive write permission |
| `BusWB` | 把 dirty line 的最新数据写回/提供给系统 |

关键不是背缩写，而是逐次问：

1. 本 cache 当前能否满足 processor request？
2. 是否要在 bus 上请求权限或数据？
3. 其他 snoopers 听到后怎样改 state？
4. 最新 data 现在由 memory 还是某个 owner 提供？
5. 两个 invariants 是否仍成立？

## 9. 与 AI Infra 的连接

- 多 socket CPU 上做 embedding preprocessing、parameter-server CPU logic 时，NUMA placement 与 shared queue 会把“算力足够”变成 coherence/remote-memory bottleneck。
- GPU programming 常强调 coalescing/HBM，但 CPU host 侧的 producer-consumer queues、reference counts、metrics counters 仍受 cache-line ownership 影响。
- Model serving 若把每个 request/thread 的 counters 紧密排列，逻辑独立的 counters 也可能因同一 cache line 发生 false sharing；下一讲会具体展开。
- Distributed training 中“scale-out 不等于快”的判断同样适用：模型/optimizer state 若能在单节点 GPU memory 中容纳，跨节点 collective 可能比增加 devices 带来更多成本。
- RDD lineage 与 modern dataflow/recomputation 的共同思想是：用确定、可重放的 computation graph 换掉昂贵的 eager materialization/replication。

## 10. 本讲结论

1. Spark 的高效来自 narrow-dependency fusion、partition-aware scheduling 与 lineage，而不是“集群天然更快”。
2. Cache line 同时是 locality 与 coherence 的搬运/跟踪粒度。
3. Private write-back caches 会让同一 shared address 出现多个不一致副本；lock 不能修复这个硬件抽象问题。
4. Coherence 为每个地址建立尊重 per-thread program order 的 serialization。
5. SWMR 管权限，data-value invariant 管最新值的交接。
6. Bus 的 broadcast + serialization 让 snooping 容易实现；directory 则为扩展性付出 metadata 与 protocol complexity。
7. MSI 用 `M/S/I` 表示 ownership 与可读写权限，下一讲完成状态机。

## 11. 自测题

1. 为什么相同 `HashPartitioner` 可能把 `join` 从 wide dependency 变成 narrow dependency？
2. Lineage 与 checkpoint/persist 分别在什么条件下更划算？
3. 32 KiB、64 B line 的 cache 有多少 lines？8-way 时有多少 sets？
4. Write-back + write-allocate 的 store miss 为什么要先读完整 line？
5. 两个 stores 已由 lock 串行化，为什么仍可能需要 coherence？
6. 用一句话分别定义 coherence、SWMR invariant 与 data-value invariant。
7. Bus 的哪两项性质帮助 snooping protocol 建立正确顺序？
8. 为什么 write-through invalidate 正确却可能很慢？
9. `M` line 上连续执行十次 store，理想情况下要多少次 bus transaction？为什么？

## 相关笔记

- [[Stanford CS149 - Lecture 09 - Distributed Data-Parallel Computing Using Spark]]
- [[Stanford CS149 - Lecture 10 - Efficiently Evaluating DNNs on GPUs]]
- [[Stanford CS149 - Lecture 12 - Memory Consistency]]
- [[Stanford CS149 - Lecture 13 - Fine-Grained Synchronization and Lock-Free Programming]]
