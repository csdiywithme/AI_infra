---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 9
lecture_date: 2023-10-24
area: systems
topics:
  - distributed-computing
  - spark
  - mapreduce
  - rdd
  - shuffle
  - fault-tolerance
  - data-locality
  - lineage
aliases:
  - CS149 Lecture 9
  - Distributed Data-Parallel Computing Using Spark
video_url: https://www.youtube.com/watch?v=jaMWmLq422U
---

# Stanford CS149 - Lecture 09 - Distributed Data-Parallel Computing Using Spark

> [!abstract]
> 本讲把 data-parallel primitives 从单颗 GPU 扩展到有独立 OS、独立内存和持久化存储的 cluster。MapReduce 用 map → group-by-key/shuffle → reduce 封装 task decomposition、data-local scheduling、failure recovery 和 straggler handling，但每阶段落盘使 iterative/interactive workload 代价很高。Spark 用 immutable、partitioned RDD 与 deterministic lineage 表达中间数据，让 runtime 能在内存中复用数据、融合 narrow transformations，并在故障时按 lineage 重算丢失 partitions；wide dependency 则形成 shuffle、通信与恢复成本的关键边界。

## 来源与范围

- [Lecture 9 视频：Distributed Data-Parallel Computing Using Spark](https://www.youtube.com/watch?v=jaMWmLq422U)
- [官方课件 PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/spark/09_spark.pdf)
- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)

视频在 narrow/wide dependency 刚引入时结束，讲师说明剩余 Spark 内容会在后续课程补完。本文前 14 节严格跟随本视频；第 15 节起明确标为“课件延伸”，整理同一份 Lecture 9 slides 中未在本视频展开的 partitioning、lineage recovery 与 performance caution，避免混淆视频覆盖范围。

## 视频索引

| 时间 | 内容 |
|---|---|
| [00:04](https://www.youtube.com/watch?v=jaMWmLq422U&t=4s) | 从 CPU/GPU data parallelism 扩展到 distributed computer |
| [01:23](https://www.youtube.com/watch?v=jaMWmLq422U&t=83s) | 10 万 cores、failure、memory efficiency 三个问题 |
| [02:30](https://www.youtube.com/watch?v=jaMWmLq422U&t=150s) | Cluster 的第一项收益：聚合 I/O bandwidth |
| [04:31](https://www.youtube.com/watch?v=jaMWmLq422U&t=271s) | Warehouse-scale computer |
| [08:06](https://www.youtube.com/watch?v=jaMWmLq422U&t=486s) | Rack、top-of-rack switch 与 server node |
| [11:16](https://www.youtube.com/watch?v=jaMWmLq422U&t=676s) | DRAM、SSD、network bandwidth 层级 |
| [14:33](https://www.youtube.com/watch?v=jaMWmLq422U&t=873s) | 独立地址空间与 message passing |
| [17:48](https://www.youtube.com/watch?v=jaMWmLq422U&t=1068s) | Distributed file system：GFS/HDFS |
| [20:07](https://www.youtube.com/watch?v=jaMWmLq422U&t=1207s) | File chunks、replication 与 NameNode metadata |
| [24:16](https://www.youtube.com/watch?v=jaMWmLq422U&t=1456s) | CS149 web log 分布式分析问题 |
| [26:22](https://www.youtube.com/watch?v=jaMWmLq422U&t=1582s) | 为什么不直接手写 MPI |
| [27:48](https://www.youtube.com/watch?v=jaMWmLq422U&t=1668s) | 复习 map/reduce 的 functional property |
| [30:35](https://www.youtube.com/watch?v=jaMWmLq422U&t=1835s) | MapReduce page-view counting API |
| [34:25](https://www.youtube.com/watch?v=jaMWmLq422U&t=2065s) | Word count dataflow |
| [37:22](https://www.youtube.com/watch?v=jaMWmLq422U&t=2242s) | MapReduce 实为 map → groupByKey/shuffle → reduce |
| [38:35](https://www.youtube.com/watch?v=jaMWmLq422U&t=2315s) | Map tasks 的 work queue 与 data-local scheduling |
| [41:15](https://www.youtube.com/watch?v=jaMWmLq422U&t=2475s) | Reducer assignment 与 hash partitioning |
| [45:07](https://www.youtube.com/watch?v=jaMWmLq422U&t=2707s) | Node failure、heterogeneity 与 scheduler |
| [46:43](https://www.youtube.com/watch?v=jaMWmLq422U&t=2803s) | Heartbeat、rerun 与 speculative execution |
| [50:34](https://www.youtube.com/watch?v=jaMWmLq422U&t=3034s) | MapReduce 的收益 |
| [51:52](https://www.youtube.com/watch?v=jaMWmLq422U&t=3112s) | MapReduce 的线性结构与 iterative workload 限制 |
| [54:38](https://www.youtube.com/watch?v=jaMWmLq422U&t=3278s) | 为什么要利用 aggregate cluster memory |
| [58:30](https://www.youtube.com/watch?v=jaMWmLq422U&t=3510s) | In-memory computation 的 fault-tolerance 挑战 |
| [61:20](https://www.youtube.com/watch?v=jaMWmLq422U&t=3680s) | Spark 与 RDD abstraction |
| [62:18](https://www.youtube.com/watch?v=jaMWmLq422U&t=3738s) | `textFile → filter → filter → count` lineage |
| [65:26](https://www.youtube.com/watch?v=jaMWmLq422U&t=3926s) | Transformations、actions 与 `persist` |
| [70:24](https://www.youtube.com/watch?v=jaMWmLq422U&t=4224s) | RDD partitions 怎样实现而不复制全部中间数组 |
| [73:16](https://www.youtube.com/watch?v=jaMWmLq422U&t=4396s) | 从 loop fusion/tiling 到 RDD fusion |
| [75:45](https://www.youtube.com/watch?v=jaMWmLq422U&t=4545s) | Narrow 与 wide dependencies |

## 1. 为什么需要 cluster：不仅是更多 compute

假设要顺序扫描 100 TB web logs，单节点 I/O throughput 为 50 MB/s：

$$
T_1\approx\frac{100\ \text{TB}}{50\ \text{MB/s}}\approx 23\ \text{days}
$$

若把数据均匀分到 1000 nodes，每个节点以同样速度读本地 shard，理想时间约：

$$
T_{1000}\approx 33\ \text{minutes}
$$

这里的 speedup 主要来自聚合 **storage I/O bandwidth**，不是某个 CPU core 更快。大数据系统把数据、SSD、DRAM、cores 一起横向扩展。

Cluster 也立刻引入三个新问题：

1. 怎样把一个 job 拆成 10 万 cores 可运行的 tasks？
2. 部分 node/rack/network 随时失败时，怎样不丢数据、不从头重算？
3. Network/SSD 比 DRAM 慢得多时，怎样减少数据移动？

### 1.1 Failure 由异常变成常态

若每台 server 的 failure rate 很小，`N` 台独立 servers 的 cluster-wide failure rate 仍近似随 `N` 增长：

$$
MTTF_{cluster}\approx\frac{MTTF_{node}}{N}
$$

所以系统不能把 failure 当作人工介入的罕见事故，scheduler 与 programming model 必须支持 automatic retry/recovery。

## 2. Warehouse-scale computer 的层级

```text
warehouse-scale computer
└── many racks
    ├── top-of-rack switch
    └── 20–40 servers/nodes
        ├── 1–2 CPU sockets, tens of cores
        ├── 128 GB–TB-scale DRAM
        ├── 10–30 TB SSD
        └── network interface
```

本讲给出的数量级示意：

| Path | 课堂数量级 | 直觉 |
|---|---:|---|
| CPU ↔ local DRAM | 100–200 GB/s | 最高带宽工作层 |
| CPU ↔ local SSD | 1–4 GB/s | persistent，但比 DRAM 慢约两个数量级 |
| Same-rack network | 1–2 GB/s | 经过 top-of-rack infrastructure |
| Cross-rack network | 0.1–2 GB/s | 早期尤其昂贵，现代 WSC 投资大量网络能力 |

数字随时代与硬件变化，真正需要记的是 hierarchy：

```text
local DRAM  >>  SSD ≈ modern network  >> early cross-rack network
```

当 network bandwidth 接近 local SSD bandwidth 时，“数据必须在本地磁盘”不再总是唯一选择；但访问 remote DRAM 仍远慢于 local DRAM，且消耗共享 fabric。

## 3. Distributed memory 与 message passing

每个 node 运行独立 OS，有独立 virtual address space：

```text
Node 0 address X ──send(payload, dst, tag)──▶ Node 1 receive buffer Y
```

Node 0 的 pointer 对 Node 1 没有直接意义。通信必须显式指定：

- recipient/sender；
- source/destination buffer；
- optional message tag。

### 3.1 Send/receive 本身带有同步语义

课堂问题是“message passing 是否还需要 synchronization？”准确理解是：

- 不一定再单独调用一个 shared-memory lock/barrier；
- receive completion 已表示匹配消息到达，形成 happens-before；
- 但程序仍会等待、可能 deadlock，并需要处理 ordering/completion。

因此“无需显式同步”不等于“没有同步”。这与 [[Stanford CS149 - Lecture 06 - Performance Optimization II Locality Communication and Contention#4.2 没有显式 barrier，不等于没有 barrier 语义|Lecture 6 的 message-passing 问题]] 是同一结论。

## 4. Distributed file system：先保证输入数据不会丢

GFS/HDFS 针对大文件、append/read-heavy workload：

1. 把超大 file 切成连续 chunks/blocks，典型为 64–256 MB；
2. 每个 block 保存 2–3 个 replicas；
3. Replicas 跨 racks 放置，避免 top-of-rack switch failure 同时丢失全部副本；
4. NameNode/master 保存 file namespace 与 block-location metadata；
5. Client 向 master 查询位置后，直接向 DataNode/chunk server 读写数据。

```text
client ──metadata request──▶ NameNode
client ◀──replica locations─ NameNode
client ──data transfer─────▶ selected DataNode
```

Master 不转发全部 file bytes，只在 control plane 管 metadata，避免成为 data bandwidth bottleneck。Master 自身仍要做 replication/failover 才能消除单点风险。

### 4.1 Storage fault tolerance ≠ computation fault tolerance

HDFS 确保原始 file block 还有副本，却不能自动恢复某 node DRAM 中尚未持久化的 intermediate result。执行框架仍需要知道：

- 哪些 tasks 完成；
- 哪些 intermediate data 因 node failure 丢失；
- 从哪些 persistent inputs 重跑哪些 tasks。

这正是 MapReduce/Spark scheduler 的职责。

## 5. MapReduce programming model

以访问日志统计 mobile user agents：

### 5.1 Mapper

每条 log record 独立解析，生成 `(key,value)`：

```text
input line
→ parse user_agent
→ if mobile: emit(user_agent, 1)
```

### 5.2 Group by key / shuffle

Runtime 必须让相同 key 的所有 values 到达同一 reducer partition：

```text
(Safari,1) from node 0 ┐
(Safari,1) from node 1 ├─ network shuffle ─▶ reducer for Safari
(Safari,1) from node 3 ┘
```

### 5.3 Reducer

每个 unique key 独立归约：

```text
reduce("Safari", [1,1,1,...]) → ("Safari", count)
```

所以更准确的名字是：

```text
Map → GroupByKey/Shuffle → Reduce
```

> [!important]
> Map 和 reduce 之间不是一条小边，而可能是全系统 all-to-all data redistribution。很多分布式 job 的真正主成本就在 shuffle。

### 5.4 关于 reduce 的 associativity

不同 keys 的 reducers 天然并行。同一 key 内若要先做 mapper-local combine、tree aggregation 或改变 grouping order，operator 需要 associative；否则必须保持严格输入顺序，削弱 parallelism 和优化空间。

Word count 的 integer addition 满足这个条件，因此可以先本地合并 `(word,1)` 再跨网络传 partial count，显著减少 shuffle bytes。

## 6. Map tasks：load balance 与 data locality 的权衡

每个 input block 对应一个或多个 map tasks。两种极端调度：

### 6.1 Central work queue

```text
idle worker → request next block → process it
```

优点是动态 load balance；缺点是 worker 可能远程读取 block，network 较慢时付出大量通信。

### 6.2 Move computation to data

优先在保存该 HDFS block replica 的 node 上运行 mapper：

```text
move a small function/task description
rather than move a 256 MB block
```

因为 file blocks 有多个 replicas，scheduler 往往能同时兼顾 locality 与 load balance：在持有任一 replica 的空闲 node 上运行 task。

这延续了课程的 producer-consumer locality 原则，只是“cache line 与 core”变成“file block 与 server”。

## 7. Reducer assignment 与 shuffle routing

Runtime 需要解决：

1. 哪个 node 运行哪个 reducer partition？
2. Mapper 怎样知道每个 key-value pair 发往哪里？

常见做法是 hash partitioning：

$$
partition(key)=hash(key)\bmod R
$$

所有 mappers 用同一 partitioner，因此同一 key 必然发往同一个 reducer partition。Scheduler 再把 reducer task 分配到可用 node。

若某个 node 已拥有某 key 大部分 intermediate values，可做 locality-aware reducer placement，减少移动；但当 network 足够强、data skew 明显或 reducer 数很大时，load balance 也同样重要。

### 7.1 Map/reduce phase boundary

Reducer 必须拿到一个 key 的所有 mapper outputs 才能给出最终结果。经典教学模型因此在 map 与 final reduce 之间形成 barrier-like boundary。

真实系统可能在 map 尚未全部结束时提前 fetch 已完成 partitions、sort/merge 或运行 partial combine，但最终 completeness 仍依赖所有相关 map outputs。

## 8. Scheduler 怎样处理 failures 与 stragglers

### 8.1 Failure detection

Workers 定期向 master/job scheduler 发送 heartbeat。超时后 scheduler 把 node 判定为 failed，并重新安排其未完成 tasks。

### 8.2 Rerun map task

Map 输入来自 replicated HDFS block；mapper 是 deterministic/functional transformation，不修改 input。因此可以在另一个持有 replica 的 node 上安全重跑：

```text
same persistent input + same mapper → same logical output
```

### 8.3 Rerun reducer

Reducer 中途失败，scheduler 可重新收集/读取 mapper intermediates 并重跑。若 final output 已可靠提交，则 task 完成后 node failure 不必重算该 output。

### 8.4 Speculative execution 处理 straggler

Cluster 中机器代际、负载、磁盘与网络状态不同，最后少数 slow tasks 会拖住整个 job。Scheduler 可复制尚未完成的 task：

```text
original task ─┐
duplicate task ├─ race → accept first valid completion, cancel the other
```

它用额外 work/资源换更低 tail latency。Functional task 使重复执行安全；若 task 有外部 side effect，就必须额外保证 idempotency/commit protocol。

## 9. MapReduce 的收益

MapReduce runtime 自动提供：

- input decomposition 与大量 tasks；
- mapper/reducer scheduling；
- data-local placement；
- shuffle routing；
- load balance；
- failure retry；
- speculative execution；
- distributed result persistence。

Programmer 只写 mapper/reducer，而不必直接管理 MPI sends、receives、failure detection 与 node membership。这种简单 abstraction 是它得以广泛使用的原因。

## 10. MapReduce 的结构性限制

### 10.1 Program structure 太线性

程序基本只能串成：

```text
map → reduce → map → reduce → ...
```

Complex multi-stage DAG、共享 intermediate、多个下游 consumers 表达笨重。DryadLINQ 等工作尝试推广为 general DAG。

### 10.2 Iterative algorithms 每轮读写 distributed filesystem

PageRank/iterative ML 每轮都要读上轮 output、计算、再把本轮 intermediate 写回 HDFS：

```text
iteration i:
HDFS read → map/shuffle/reduce → HDFS write
iteration i+1:
HDFS read → ...
```

即使 working set 能放进 aggregate DRAM，programming model 仍强制反复走低带宽、序列化与 checksum 较重的 storage path。

### 10.3 Interactive ad-hoc queries 重复加载同一数据

若同一 dataset 接连被许多 queries 使用，每个 query 都从 filesystem 开始，response latency 与 I/O overhead 很高。理想系统应把 hot working set 保留在 cluster memory。

## 11. Spark 的目标：in-memory + fault-tolerant

课程引用 2009 workload 数据说明，许多大数据 jobs 的 active working set 可以放进整个 cluster 的 aggregate memory。Spark 试图同时获得：

- MapReduce 的 high-level functional interface；
- DRAM reuse 的高 bandwidth；
- node failure 后的 automatic recovery；
- iterative 与 interactive computation 支持。

难点是 DRAM volatile。直接复制所有 intermediate 会减半有效 capacity/throughput；频繁 checkpoint 会回到 storage bottleneck；记录每次细粒度 update 又有很高 log overhead。

Spark 的答案是记录 **怎样重建数据**，而不默认复制每个 intermediate byte。

## 12. RDD：Resilient Distributed Dataset

RDD 的课堂定义：

> Read-only、ordered、partitioned collection of records；只能由 persistent data 或其他 RDD 经过 deterministic transformation 创建。

例如：

```scala
val lines       = spark.textFile("hdfs://cs149log.txt")
val mobileViews = lines.filter(x => isMobileClient(x))
val safariViews = mobileViews.filter(x => x.contains("Safari"))
val numViews    = safariViews.count()
```

```text
HDFS file → lines RDD → mobileViews RDD → safariViews RDD → scalar count
          textFile       filter             filter           action
```

### 12.1 Transformation

Transformation 接收 RDD，描述一个新 RDD：

- `map`
- `filter`
- `flatMap`
- `sample`
- `reduceByKey`
- `join`
- `sort`
- `partitionBy`

Transformation 不原地修改 parent RDD。

### 12.2 Action

Action 触发/请求结果返回 application 或 persistent sink：

- `count`
- `collect`
- `reduce`
- `lookup`
- `save`

`collect()` 会把所有 records 拉回 driver/host，dataset 大时可能造成 driver memory/network bottleneck；它不是普通的 distributed transformation。

### 12.3 Lineage

从 persistent input 到当前 RDD 的 transformation graph 称为 lineage：

```text
input partitions
→ map/filter partitions
→ shuffle/reduce partitions
→ output/action
```

Lineage 同时服务于 optimization、scheduling 和 fault recovery。

## 13. `persist`：把会复用的 RDD 留在 memory

若 `mobileViews` 被两个下游使用：

```scala
mobileViews.persist()

val chromeTimes = mobileViews
  .filter(_.contains("Chrome"))
  .map(extractTimestamp)
  .collect()

val safariCount = mobileViews
  .filter(_.contains("Safari"))
  .count()
```

`persist` 告诉 Spark runtime 缓存 materialized partitions，避免两个 actions 都从 ancestors 重算。

> [!note] 对视频口头解释的准确补充
> 未调用 `persist` 时，Spark 通常不会自动把每个 RDD intermediate 写入 HDFS；RDD 仍有 lineage，但 partition 在一个 action 后可能不保留，下次 action 需要从 lineage recompute。`persist` 的 storage level 可以是 memory、memory+disk 等。它是 reuse hint/contract，不保证所有 partitions 永不 eviction。

何时 persist 值得：

$$
\text{saved recomputation + saved upstream I/O}
>
\text{cache materialization + memory pressure + eviction cost}
$$

只被使用一次、可直接 fusion 的 RDD 通常不需要 materialize；反复使用或 lineage 昂贵的 RDD 更适合 persist/checkpoint。

## 14. RDD 不应被实现成每层完整复制的 arrays

若 `lines → lower → mobileViews → count` 的每个 RDD 都在每个 node 分配完整 partition array：

```text
lines partition
+ lowercased copy
+ filtered copy
```

内存占用可能比原始 file 更大，且每层都写/读 DRAM。

回顾 Lecture 6 的 loop fusion：

```cpp
int count = 0;
while (!input.eof()) {
    string line = input.readLine();
    string lower = toLower(line);
    if (isMobileClient(lower))
        count++;
}
```

每个 record 可在一个 streaming pipeline 中连续执行 `load → map → filter → local count`，不必 materialize `lines/lower/mobileViews` 三套 arrays。

### 14.1 Narrow dependency

准确的 Spark 定义是：**parent RDD 的一个 partition 最多被一个 child RDD partition 使用。**

Map/filter 通常是窄依赖：

```text
parent part 0 → child part 0
parent part 1 → child part 1
```

它允许：

- partition-local pipelining/fusion；
- producer/consumer 在同一 node 连续运行；
- 只重算丢失 child 所需的 parent partitions；
- 避免 cluster-wide shuffle。

> [!warning]
> 不要把 narrow 误记为“一个 child 只依赖一个 parent”。更关键的方向是：一个 parent partition 的 output 不会被拆散发送给多个 child partitions。

### 14.2 Wide dependency

GroupByKey/sort 等会让一个 parent partition 的 records 按 key 分散到多个 child partitions：

```text
parent part 0 ─┬→ child part 0
               ├→ child part 1
               └→ child part 2
```

这要求 shuffle，可能接近 all-to-all communication。Wide dependency 是以下成本的边界：

- network traffic；
- partitioning/serialization；
- phase/stage boundary；
- materialization；
- failure 后更大范围 recomputation。

视频在这里结束。

---

## 15. 课件延伸：partitioning 决定 join 是否需要 shuffle

> [!info] 覆盖范围
> 以下内容来自同一份 Lecture 9 官方课件的未讲完部分，不是本视频 77 分钟内已经展开的内容。

设两个 keyed RDD：

```text
A: (K,V)
B: (K,W)
join → (K,(V,W))
```

若 A、B 使用不同 partitioning，同一个 key 可能位于不同 nodes，join 必须 repartition/shuffle，形成 wide dependencies。

若二者预先用同一个 `HashPartitioner`/`RangePartitioner`：

```text
A partition i contains keys assigned to i
B partition i contains the same key range/hash bucket
→ join partition i locally
```

Join 就可变成 narrow/local operation。

这揭示一个分布式优化原则：不要孤立判断 operator 是否昂贵，成本取决于输入 layout/partitioning。

### 15.1 `partitionBy` 的成本是投资

预先 repartition 本身是一次 shuffle，但若 dataset 会被多次 join/group/query，持久化共同 partitioning 可摊销成本：

$$
\text{one-time repartition}
<
\sum \text{future avoided shuffles}
$$

这与单机上的 layout transformation、tensor layout prepacking 同构。

## 16. 课件延伸：lineage-based fault recovery

RDD transformations 是 bulk、deterministic、functional。Runtime 记录：

```text
RDD partition = transformation(parent partition(s))
```

Node 1 crash、丢失 `timestamps` partitions 2/3 时，不必恢复整个 cluster checkpoint：

1. 找出 lineage 中生成 partitions 2/3 的 transformations；
2. 从 replicated HDFS blocks 2/3 重新读取必要 input；
3. 在健康 node 上重跑对应 narrow pipeline；
4. 重新生成丢失 partitions。

### 16.1 为什么 lineage log 很小

数据库 update log 可能记录每条 record 的细粒度 mutation。RDD lineage 只记录 bulk operation：

```text
filter(predicate X)
map(function Y)
partitionBy(hash, 100)
```

一个 operation 描述百万 records，因此 logging overhead 与 data size 不成正比。

### 16.2 Lineage 不是万能的

- Long lineage 会使 recovery 重算过多；
- Wide dependencies 可能要求恢复多个 upstream partitions/shuffle outputs；
- Non-deterministic/user side-effect function 会破坏可重放性；
- 极昂贵 intermediate 可能值得可靠 checkpoint。

因此实践中结合：

- cache/persist；
- replication of selected data；
- checkpoint/durable storage；
- lineage recomputation。

## 17. 课件延伸：scale-out 不自动等于高 performance

Spark/Hadoop 解决了 resiliency、elastic cluster management 与 programmability，但 abstraction/runtime 仍带来：

- serialization/deserialization；
- object-heavy layout；
- scheduler overhead；
- multiple memory copies/checksums；
- unnecessary materialization/shuffle；
- poor SIMD utilization。

课件引用 COST 指标：**Configuration that Outperforms a Single Thread**。一个系统能在 100 nodes 上线性 scaling，不代表 absolute performance 好；它可能只是在并行化自身 overhead。

正确评估至少同时报告：

```text
best optimized single-thread/single-node baseline
scale-out speedup
absolute runtime
resource/cost/energy
```

Spark Project Tungsten、code generation、columnar/SoA layout、Weld、GPU acceleration 等努力，都是在保留高层分布式语义的同时降低单节点 overhead。

## 18. MapReduce、Spark 与手写 message passing 对照

| 维度 | MPI/message passing | MapReduce | Spark RDD |
|---|---|---|---|
| Control | Programmer 显式协议 | 固定 map/shuffle/reduce | Transformation DAG |
| Intermediate | Programmer 管理 | 常写 distributed FS | 默认由 lineage 表示，可 cache/persist |
| Fault recovery | Programmer/runtime 另行实现 | Task retry + persistent stage output | Partition recomputation + optional checkpoint |
| Locality | Programmer 显式 placement | Scheduler move compute to data | Partition-aware scheduler + pipelining |
| Iteration/reuse | 可高效但复杂 | 每轮落盘昂贵 | In-memory reuse |
| Flexibility | 最高 | 较低 | 较高，但受 RDD operators 限制 |
| Performance ceiling | 可精细优化 | I/O/runtime overhead 大 | 取决于 shuffle、serialization、layout 与 codegen |

没有一种模型在所有场景最好。High-level system 用可表达性换 automatic scheduling/resilience；MPI 类低层模型用编程复杂度换细粒度 control。

## 19. 分布式程序的系统化成本分析

对每条 operator edge 记录：

1. **Partitioning**：records 当前按什么 key/range 放置？
2. **Dependency width**：narrow 还是 wide？
3. **Bytes**：每个 record 多大，shuffle/input/output 总 bytes？
4. **Locality**：producer 与 consumer 能否 colocate？
5. **Materialization**：是否必须落 DRAM/SSD/HDFS？
6. **Reuse**：是否值得 persist/repartition？
7. **Skew**：某些 keys/partitions 是否显著更大？
8. **Recovery**：丢一个 node 需重算多大 lineage？

简化的 stage 时间常由最慢 partition 决定：

$$
T_{stage}\approx\max_i(T_{compute,i}+T_{I/O,i}+T_{network,i})+T_{coordination}
$$

所以平均负载很好仍不够：data skew、straggler 或单个 hot reducer 会决定 tail completion time。

## 20. 与 AI Infra 的连接

### 20.1 数据并行训练同样受 partition 与 collective 支配

Gradient computation 类似 local map，gradient aggregation 类似 reduce。不同之处是训练常用 all-reduce 而非把所有 data 发给一个 reducer，但问题仍是：

- bytes per step；
- network topology；
- overlap compute/communication；
- straggler rank；
- fault recovery/checkpoint。

### 20.2 Shuffle 与 MoE all-to-all

MapReduce 按 key 把 records 路由到 reducer，与 MoE 按 expert ID 把 tokens 路由到 expert ranks 同构：

```text
record/token
→ compute destination key
→ partition/count/pack
→ all-to-all shuffle
→ grouped local compute
```

Expert imbalance 就是 key skew；热门 expert 对应 hot reducer。Capacity factor、load-balancing loss、expert placement 都是在控制 partition size 与 network traffic。

### 20.3 Narrow/wide dependency 对应 fusion/collective boundary

LLM execution graph 中：

- 同一 GPU 上连续 elementwise ops 类似 narrow pipeline，可 fuse；
- tensor-parallel all-gather/reduce-scatter、MoE all-to-all 类似 wide edge，形成通信 stage boundary；
- 改变 tensor sharding/layout 可让后续 operator local，类似 `partitionBy` 后避免 join shuffle。

### 20.4 Data locality：move compute to data/model state

大模型 serving 中移动 weight/KV cache 往往比派发一个 request expensive。Scheduler 会倾向：

- 把 request 路由到已有 model weights/prefix KV 的 worker；
- 复用已加载 adapter；
- 避免在 nodes 间迁移 large cache state。

这正是 HDFS scheduler “move computation to data” 的现代版本。

### 20.5 Lineage 与 checkpoint 的取舍

训练不能只靠 lineage 从头重算数小时，因此会周期性 checkpoint model/optimizer/RNG/dataloader state。Checkpoint interval 仍是同一优化：

$$
\text{checkpoint overhead}
\quad\text{vs}\quad
\text{expected recomputation after failure}
$$

短而 deterministic 的 preprocessing DAG 可以重算；昂贵、长状态链需要 durable checkpoint。

### 20.6 Speculative execution 与 serving tail latency

MapReduce 复制 straggler task 与 hedged request 类似：对 tail-sensitive inference，可把慢请求副本发给另一 replica，取先返回结果。收益是降低 P99，代价是额外 GPU work 和对 overloaded cluster 的二次压力。

## 21. 本讲结论

1. Cluster 的重要收益之一是聚合 I/O bandwidth，不只是聚合 CPU cores。
2. 独立 OS/address spaces 要通过 message passing 通信；send/receive 本身建立同步关系。
3. Distributed filesystem 用分块、跨 rack replication 与 metadata master 保证 persistent input。
4. MapReduce 实质是 map → groupByKey/shuffle → reduce，shuffle 常是主要 communication cost。
5. Scheduler 同时优化 data locality、load balance、failure retry 与 straggler tail latency。
6. Functional、deterministic tasks 使安全重跑和 speculative execution 成为可能。
7. MapReduce 每阶段落盘，导致 iterative 与 interactive workload 低效。
8. Spark RDD 是 immutable、partitioned collection；transformations 形成 lineage，actions 请求结果。
9. `persist` 用 memory capacity 换复用，未 cache 的 RDD 通常从 lineage 重算而不是自动写 HDFS。
10. Narrow dependency 允许 partition-local fusion；wide dependency 意味着 shuffle、stage boundary 与更大 recovery domain。
11. Partitioning 是性能状态：预先 co-partition 可以把后续 join 从 wide 变 narrow。
12. Scale-out scalability 不能取代优化单节点 absolute performance。

## 自测题

1. 1000 nodes 扫描数据的 speedup 为什么主要来自 aggregate I/O bandwidth？
2. 为什么 cluster 规模扩大后必须把 node failure 当作常态？
3. HDFS 的 NameNode 为什么不直接转发所有 file data？
4. Distributed filesystem fault tolerance 为什么不能替代 computation fault tolerance？
5. 为什么 MapReduce 应更准确地叫 Map–GroupByKey–Reduce？
6. Hash partitioner 怎样保证同一 key 到同一 reducer？
7. Work queue 与 data-local mapper scheduling 的 trade-off 是什么？
8. Speculative execution 为什么依赖 task 可安全重复执行？
9. PageRank 在经典 MapReduce 上为什么代价高？
10. RDD 的 immutable/deterministic property 怎样支持 fault recovery？
11. Transformation 与 action 有什么不同？为什么 `collect()` 很危险？
12. 没有 `persist` 的 RDD 再次使用时通常发生什么？
13. Narrow dependency 的准确定义是什么？为什么不能只记“child 只依赖一个 parent”？
14. `groupByKey` 为什么产生 wide dependency？
15. 相同 hash partitioning 怎样把 join 变成 local/narrow operation？
16. Lineage recovery 与 checkpoint 各适合什么情况？
17. 为什么线性 scaling 仍可能有很差的 absolute performance？
18. MoE token routing 与 MapReduce shuffle 在结构上怎样对应？

## 参考资料

- [CS149 Fall 2023 Lecture 9 视频](https://www.youtube.com/watch?v=jaMWmLq422U)
- [Lecture 9 官方课件](https://gfxcourses.stanford.edu/cs149/fall23content/media/spark/09_spark.pdf)
- [Spark RDD Programming Guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html)
- [Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing](https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf)
- [The Datacenter as a Computer](https://www.morganclaypool.com/doi/abs/10.2200/S00516ED3V01Y201306CAC024)
- [[Stanford CS149 - Lecture 08 - Data-Parallel Thinking|Lecture 8：Data-Parallel Primitives]]
- [[Stanford CS149 - Lecture 06 - Performance Optimization II Locality Communication and Contention|Lecture 6：Message Passing、Locality 与 Communication]]
