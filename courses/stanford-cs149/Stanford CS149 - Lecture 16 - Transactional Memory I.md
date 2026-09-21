---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 16
lecture_date: 2023-11-28
area: systems
topics:
  - transactional-memory
  - synchronization
  - atomicity
  - isolation
  - serializability
  - eager-versioning
  - lazy-versioning
  - conflict-detection
aliases:
  - CS149 Lecture 16
  - Transactional Memory I
video_url: https://www.youtube.com/watch?v=rFFf3WIJ7BA
---

# Stanford CS149 - Lecture 16 - Transactional Memory I

> [!abstract]
> Transactional memory（TM）把同步从 imperative 的 “拿哪些锁、按什么顺序拿” 提升为 declarative 的 “这段 memory operations 必须 atomic”。系统动态追踪 transaction 的 read/write sets：无冲突 transactions 并发，有 read-write/write-write conflict 时才选择 stall、abort、restart 或 serialize。它希望提供 coarse lock 的易用性、fine-grained locking 的并发度，并额外带来 failure atomicity 与 composability。实现空间由两条正交轴组成：data versioning 决定未提交写放在哪里（eager + undo log，或 lazy + write buffer），conflict detection 决定何时发现冲突（pessimistic/encounter-time，或 optimistic/commit-time）。真正的性能与 correctness 取决于两条轴和 contention manager 的组合。

## 来源与范围

- [Lecture 16 视频：Transactional Memory 1](https://www.youtube.com/watch?v=rFFf3WIJ7BA)
- [官方课件 PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/transactions1/15_transactionalmem.pdf)
- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)

课程网页/课件跳过 Midterm Review 编号，因此 PDF 标作 Lecture 15；本文按公开播放列表记作 Lecture 16。

## 视频索引

| 时间 | 内容 |
|---|---|
| [00:49](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=49s) | 从 atomics、locks、barriers、lock-free 回顾同步层次 |
| [02:23](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=143s) | TM：提高 abstraction，同时争取 correctness + performance |
| [03:17](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=197s) | 实现 design space：versioning、detection、granularity |
| [04:28](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=268s) | Coarse vs. fine-grained locking 的根本权衡 |
| [06:13](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=373s) | Bank deposit：lock 与 `atomic {}` 对比 |
| [08:34](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=514s) | Declarative vs. imperative abstraction |
| [10:12](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=612s) | TM 的 atomicity、isolation、serializability |
| [12:39](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=759s) | Transaction 作为一个 bulk atomic memory operation |
| [15:10](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=910s) | Java HashMap：coarse lock 易用但不 scale |
| [17:01](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=1021s) | Per-bucket/fine-grained locks 与 complexity |
| [18:34](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=1114s) | Fine-grained lock overhead 可能吞掉小规模收益 |
| [21:27](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=1287s) | Tree update：相交 traversal path 不等于真实 data conflict |
| [22:57](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=1377s) | Read-set/write-set 与 conflict 判断 |
| [26:53](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=1613s) | Doubly-linked list 用一个 `atomic` region 表达 |
| [28:27](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=1707s) | Failure atomicity：abort 是整体 undo |
| [30:09](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=1809s) | Lock composition 破坏 modularity |
| [32:52](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=1972s) | Nested transactions 与 disjoint transfers |
| [36:01](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=2161s) | TM 三项主要收益总结 |
| [39:05](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=2345s) | Atomic region 不等于 lock；TM 也会被误用 |
| [40:04](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=2404s) | Transaction 内等待另一个 transaction 的 flag 会卡死 |
| [42:17](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=2537s) | Atomicity violation：region boundary 选错 |
| [43:50](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=2630s) | Implementation 必须同时保证 A/I/S 与 concurrency |
| [44:34](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=2674s) | 两条实现轴：data versioning、conflict detection |
| [46:20](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=2780s) | Eager versioning + undo log |
| [47:28](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=2848s) | Eager commit/abort |
| [51:24](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=3084s) | Lazy versioning + write buffer |
| [53:49](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=3229s) | Eager/lazy tradeoff |
| [56:23](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=3383s) | Conflict、read-set、write-set |
| [57:19](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=3439s) | Pessimistic detection：每次 access 时检查 |
| [60:39](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=3639s) | Early detect 后选择 stall 或 abort |
| [65:57](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=3957s) | Writer-wins policy 的 read-write cases |
| [69:04](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=4144s) | 两个 writers 反复互相 abort：livelock |
| [72:15](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=4335s) | Optimistic detection：commit-time validation |
| [74:09](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=4449s) | Doomed transaction 与 wasted work |
| [76:39](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=4599s) | Serialization order 解释 read-before-write |
| [77:07](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=4627s) | Optimistic detection 提供 forward progress 的例子 |
| [79:02](https://www.youtube.com/watch?v=rFFf3WIJ7BA&t=4742s) | Eager versioning 与 optimistic detection 不好搭配 |

## 1. Locks 的抽象层次问题

Locks 要程序员同时负责：

- 找到所有 shared-state invariants；
- 决定 lock granularity；
- 建立 global acquisition order；
- 覆盖 early return/exception；
- 在 correctness、parallelism、lock overhead 之间取舍。

Coarse lock 容易正确但 serializes unrelated work；fine-grained lock 暴露更多 concurrency，却让代码复杂、易 race/deadlock，并可能在每个 pointer hop 付出 lock cost。

TM 的语法意图：

```cpp
atomic {
    balance = account.get();
    account.put(balance + amount);
}
```

它声明 semantics，不指定 mechanism。Runtime 可用 locks、software instrumentation、cache coherence/hardware buffering 或混合方案实现。

## 2. TM 的三个核心性质

数据库 ACID 中，transactional **memory** 主要提供 A/I/S；通常不保证断电/崩溃后的 durability。

### 2.1 Atomicity：all or nothing

Transaction 的所有 writes 一起成为 committed state，或 abort 后一个也不生效。中途失败不能留下半条 linked-list update 或只扣款未入账。

### 2.2 Isolation：uncommitted state 不可被其他 transaction 观察

其他 transactions 不应读到当前 transaction 的 partial writes。Transaction 自己当然要 read-your-writes。

### 2.3 Serializability：结果等价于某个 transaction 顺序

实际执行可 overlap，但 committed result 必须等价于将完整 transactions 按某个 serial order 执行。Order 由 system 决定，不由 source 中 transaction 启动时刻预先指定。

```text
interleaved physical execution
        ↓ must be equivalent to
T2 → T0 → T1 → ... serial transaction order
```

把每个 transaction 看成 bulk atomic memory operation，就得到 transaction-granular sequential/serializable view。

## 3. Read-set、write-set 与 conflict

对 transaction `T_i`：

$$
R_i=\{\text{addresses read by }T_i\},\quad
W_i=\{\text{addresses written by }T_i\}
$$

两个并发 transactions 冲突，若至少有一方 write 同一 location：

$$
(W_i\cap R_j)\cup(R_i\cap W_j)\cup(W_i\cap W_j)\neq\varnothing
$$

`R_i ∩ R_j` 不构成 conflict。Tree 例子中两个 updates 都 read root/path nodes `1,2`，但分别只写 node `3` 与 `4`：read sets 重叠，却没有 read-write/write-write intersection，因此可以 overlap。

> [!important]
> Lock conflict 与 data conflict 不同。Hand-over-hand traversal 可能因共用路径 locks 而互斥，即使 transactions 最终写完全不相交的 leaves；TM 有机会推迟到真实 read/write conflict 才 serialize。

## 4. TM 的四项主要收益

### 4.1 易用性

Region boundary 表达 high-level invariant，不必把数据结构 implementation 细节泄漏到 synchronization policy。

### 4.2 动态 fine-grained concurrency

同一大 `atomic {}` region 的两个 instances 只在实际冲突时 serialize。Source region 大不等于执行时只能一个 transaction 在跑。

### 4.3 Failure atomicity

Exception/explicit abort 能恢复 transaction 开始前的 memory state；不必手工反向执行 partial updates、释放已拿 locks。

### 4.4 Composability

Lock-based `transfer(A,B)` 与 `transfer(B,A)` 可能因 acquisition order 相反 deadlock。要求所有 modules 遵守全局 lock order 会破坏 modularity。

Nested atomic regions 通常由 outer transaction subsume：

```cpp
atomic {
    withdraw(A); // internally atomic
    deposit(B);  // internally atomic
}
```

调用方可组合已有 atomic operations，得到一个更大 atomic invariant。若同时执行 `transfer(A,B)` 与 `transfer(C,D)`，data sets disjoint，仍可并发；`atomic` 不是 global lock。

## 5. TM 不是魔法：semantic misuse

### 5.1 Transaction 内 busy-wait 另一个 transaction

```cpp
atomic { while (!flag) { } }
atomic { flag = true; }
```

Isolation 使第一条 transaction 在第二条 commit 前看不到其 write；而第一条不结束，serialization/implementation又可能阻止第二条提供它等待的状态，导致无进展。Transaction 内通常不能执行依赖外部 concurrent action 才完成的 blocking operation。

### 5.2 Atomicity violation

若 logically indivisible invariant 被拆成多个 atomic regions，中间可插入别的 transaction：

```text
atomic { ptr = A; }
atomic { ptr = null; }
atomic { use(*ptr); }  // may dereference null
```

每个小 region 都 atomic，整体 algorithm 仍错误。正确 boundary 应覆盖 invariant 从检查到使用/更新的完整范围。

### 5.3 无法表达或 rollback 的 side effect

I/O、system call、blocking、irrevocable operation、无限 transaction、异常控制流等并不天然可 speculative rollback。实际 TM 会限制、abort/fallback 或把 transaction 变成 irrevocable。

### 5.4 Domain knowledge 仍可能胜过 generic TM

仅比较 read/write sets 可能产生 semantic false conflicts；专用 fine-grained algorithm 知道 operation commute、节点逻辑状态等，可能允许更高 concurrency。TM 追求“低认知成本下接近 fine-grained performance”，不承诺总是最优。

## 6. 实现目标与两条设计轴

实现必须：

1. 让 uncommitted writes 不泄漏（isolation）；
2. commit 看起来瞬时生效（atomicity）；
3. 选择合法 serial order（serializability）；
4. 尽可能 overlap non-conflicting transactions；
5. 冲突/容量/异常时可 abort 并恢复。

两条主要轴：

```text
data versioning:      eager  ↔ lazy
conflict detection:   pessimistic/early ↔ optimistic/late
```

再加 contention management 与 tracking granularity，构成完整 design space。

## 7. Data versioning

### 7.1 Eager versioning + undo log

第一次写 `X`：

```text
undo_log.append(X, old_value)
memory/cache[X] = new_value
```

- commit：丢弃 undo log；memory 已是新值，快；
- abort：反向应用 undo log，恢复 transaction 前 state；
- transaction 内普通 load 直接读最新 memory/cache；
- 同一 location 多次写通常只需记录第一次的 old value。

问题：abort 慢；恢复期间也要保持 isolation；其他 transactions 不能把 speculative memory 当 committed state；fault/overflow handling 更复杂。

### 7.2 Lazy versioning + write buffer（redo log）

Writes 留在 private speculative buffer：

```text
write_buffer[X] = new_value
committed_memory[X] unchanged
```

- transaction 内 read 先查 write buffer，再查 memory（read-your-writes）；
- abort：直接丢 buffer，快；
- commit：验证后把 entire write set 原子地 publish/写入 committed state；
- isolation 自然，因为 memory 没被 speculative write 改动。

问题：每次 read 多一个 lookup/forwarding path；large write set 需要容量；commit 慢且必须避免 partial publish 被看见。

### 7.3 对照

| | Eager | Lazy |
|---|---|---|
| Speculative write 位置 | memory/cache in-place | private buffer/log |
| Read-your-write | 直接读更新后 memory/cache | buffer forwarding |
| Commit | 快：discard undo log | 慢：publish redo/write buffer |
| Abort | 慢：restore undo log | 快：discard buffer |
| Isolation | 需要阻止别人看 speculative value | 较自然 |
| 适合 | abort 少、可早检测/排他 | optimistic validation、abort 相对可接受 |

## 8. Conflict detection

### 8.1 Pessimistic / encounter-time detection

每次 transactional read/write 前，用当前 active transactions 的 sets 检查 conflict。假定冲突可能发生，尽早处理。

优点：

- 早发现 doomed transaction，少浪费后续 work；
- 可在尚未执行 conflicting read 时 stall；
- 与 eager versioning/ownership acquisition 较自然。

代价：

- 每次 access 都有 metadata/check overhead；
- active conflicts 会长时间 stall；
- contention manager 不当会 deadlock/livelock/starvation。

### 8.2 Writer-wins 例子

课程假设 aggressive writer-wins：

- T0 已 `W(A)`，T1 即将 `R(A)`：T1 尚未读坏数据，可 stall 到 T0 commit，再重新读；也可 abort；
- T0 已 `R(A)`，T1 发起 `W(A)`：writer wins，T0 读过将被改的数据，必须 abort/restart；
- T0 与 T1 都想 `W(A)`：若双方每次同时 abort/restart，可能 livelock，需 age/priority/backoff 让一个完成。

若 stalled transaction 持有的其他 read set 后续又与对方 write 冲突，stall 仍可能升级为 abort；metadata 不能在 stall 时丢掉。

### 8.3 Optimistic / commit-time detection

Execution 阶段假设无冲突，commit 时 validation：把 committing transaction 的 write set 与其他 active transactions 的 relevant read/write sets 比较；冲突时通常让 committer 胜出，abort/restart others。

优点：

- ordinary transactional access path 更轻；
- low-contention workload 很合适；
- commit 建立明确 forward progress point。

代价：

- 可能让注定失败的 doomed transaction 跑很久；
- commit validation/publish 成为 burst；
- 需要 lazy/speculative versions，或复杂 isolation 机制。

> [!note]
> 课程例子按某个 serialization rule 主要比较 `W_committer` 与其他 active `R` sets。完整算法还要根据 versioning、write-write ownership 和 chosen serial order 处理其他 intersections；不要把课堂单向检查机械当成所有 TM 的统一公式。

## 9. Contention management 是 progress policy

检测到 conflict 之后还没结束，需要决定：

- abort requester 还是 owner；
- stall 还是立即 retry；
- younger/older transaction 谁优先；
- 是否随机/指数 backoff；
- 长 transaction 是否升级 priority；
- 如何避免 cyclic wait 与 repeated mutual abort。

Correct serial execution 可能有很多，contention manager 选择哪条 execution path；它影响 wasted work、latency fairness 与 throughput。

## 10. Tracking granularity 与 false conflict

Read/write sets 可按 byte/word/object/cache line/page 跟踪：

| Granularity | Metadata/check cost | False conflict |
|---|---:|---:|
| Fine（word/object） | 高 | 低 |
| Cache line | 中，能复用 coherence | 类似 false sharing |
| Page | 低/OS 友好 | 很高 |

两个 transactions 写同一 cache line 中不同 words，语义上不冲突，却可能被 line-granular TM 判冲突。与 cache coherence false sharing 同源：implementation granularity 大于 program sharing granularity。

## 11. 与 AI Infra 的连接

- Model-serving scheduler 更新多张关联表（request state、batch membership、KV allocation、quota）时，需要跨结构 invariant；transactional abstraction 比多锁顺序更易组合。
- GPU/accelerator runtime 的 command submission 含 device queues 与 host metadata，但 MMIO/I/O side effects 难 rollback，是 TM 边界的重要反例。
- Optimistic concurrency control（OCC）广泛用于 metadata store、distributed database、MVCC；read/write sets、validate、commit/abort 与本讲完全同构，只是 durability/replication 更复杂。
- Tensor/graph compiler pass 若更新 IR 的多个节点，可先在 private version 上变换，验证 invariant 后原子替换，概念上也是 lazy versioning。
- Cache-line-granular hardware TM 在 allocator/refcount 等热点 line 上易 capacity/conflict abort；layout 与 false sharing 仍决定性能。

## 12. 本讲结论

1. `atomic {}` 是 declarative semantics；lock 是一种 imperative mechanism，二者不等同。
2. TM 提供 atomicity、isolation、serializability，通常不提供 durability。
3. Read/write-set 真冲突而非 traversal/lock overlap 决定 transactions 是否必须 serialize。
4. TM 的主要优势是 coarse-lock 易用性、dynamic fine-grained concurrency、failure recovery 与 composition。
5. Atomic region boundary 选错、transaction 内等待/I/O 等仍会让程序失败。
6. Eager/lazy versioning 决定 speculative state 放在哪里；pessimistic/optimistic detection 决定何时发现 conflict。
7. Contention manager 与 tracking granularity 决定实际 progress、fairness、wasted work 与 false conflicts。

## 13. 自测题

1. Transactional memory 的 A/I/S 分别约束什么？为什么通常没有 D？
2. 两个 transactions 的 read sets 重叠是否冲突？写出完整 set-intersection 条件。
3. 为什么两个 tree updates 的 traversal paths 相交，TM 仍可能让它们并发？
4. Lock ordering 为什么伤害 composability，而 nested transaction 可以更自然组合？
5. 举一个每个小 atomic region 都正确、整体却 atomicity violation 的例子。
6. Eager versioning 为什么 commit 快、abort 慢？Lazy 正好相反的原因是什么？
7. Lazy transaction 内读取自己刚写的 `X` 应从哪里取？
8. Pessimistic detection 下，什么时候可以 stall，什么时候必须 abort？
9. Optimistic validation 为什么会产生 doomed work？它又为什么更适合低冲突 workload？
10. 两个 transactions 写同一 cache line 的不同 words，为什么 hardware TM 仍可能 abort？

## 相关笔记

- [[Stanford CS149 - Lecture 12 - Memory Consistency]]
- [[Stanford CS149 - Lecture 13 - Fine-Grained Synchronization and Lock-Free Programming]]
- [[Stanford CS149 - Lecture 17 - Transactional Memory II]]
