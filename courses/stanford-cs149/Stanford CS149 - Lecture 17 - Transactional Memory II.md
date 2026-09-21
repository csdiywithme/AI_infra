---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 17
lecture_date: 2023-11-30
area: systems
topics:
  - transactional-memory
  - software-transactional-memory
  - hardware-transactional-memory
  - version-clock
  - cache-coherence
  - conflict-detection
  - heterogeneity
aliases:
  - CS149 Lecture 17
  - Transactional Memory II
video_url: https://www.youtube.com/watch?v=Tbk1vnYLQqI
---

# Stanford CS149 - Lecture 17 - Transactional Memory II

> [!abstract]
> 本讲把 [[Stanford CS149 - Lecture 16 - Transactional Memory I|上一讲]]的 TM 设计空间落到两类实现。软件 TM 通过编译器插入 `STMRead/STMWrite` barriers，以 transaction descriptor 保存 per-transaction 状态、以 transaction record 保存 per-data lock/version；McRT 用 eager versioning、optimistic reads、pessimistic writes 和 global version clock 来检测 stale reads。代价是普通 load/store 被改写成 bookkeeping-heavy 的函数路径。硬件 TM 则复用 cache 保存 speculative versions，给 cache line 增加 transactional `R/W` bits，并借 coherence requests 检测冲突：更低开销，但 transaction 容量、cache eviction、abort 原因和硬件支持都成为限制。结尾进一步提出：通用 CPU/GPU 之外，specialization 与 heterogeneity 用更难的编程模型换 energy efficiency。

## 来源与范围

- [Lecture 17 视频：Transactional Memory 2](https://www.youtube.com/watch?v=Tbk1vnYLQqI)
- [官方课件 PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/transactions2/16_heterogeneity.pdf)
- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)

课程网页/课件跳过 Midterm Review 编号，因此 PDF 标作 Lecture 16；本文按公开播放列表记作 Lecture 17。视频约 `73:26` 起已进入下一讲 hardware specialization 的动机，本文一并记录，避免丢失视频内容。

## 视频索引

| 时间 | 内容 |
|---|---|
| [00:04](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=4s) | 回顾 eager/lazy versioning 与 pessimistic/optimistic detection |
| [02:47](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=167s) | 本讲路线：软件、硬件与 hybrid TM |
| [04:35](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=275s) | Software barriers：重写 transaction 内的每个 read/write |
| [06:42](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=402s) | Instrumented 与 uninstrumented function cloning |
| [07:31](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=451s) | Transaction descriptor 与 per-data transaction record |
| [09:28](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=568s) | Object/field/address/cache-line tracking granularity |
| [11:33](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=693s) | Coarse metadata 的 false conflicts |
| [14:01](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=841s) | Intel McRT STM 与 version-clock 设计 |
| [17:14](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=1034s) | `STMRead`：读后验证与 read-set insertion |
| [20:59](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=1259s) | `STMWrite`：取锁、undo log、in-place write |
| [22:12](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=1332s) | 全 read-set validation 与 timestamp extension |
| [24:06](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=1446s) | Commit：global clock `+2`、validate、publish versions |
| [25:10](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=1510s) | `foo → bar` atomic copy 完整执行例 |
| [32:43](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=1963s) | 被锁 reader 恢复后发现版本变化并 abort |
| [34:27](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=2067s) | STM 实现总结与 barrier 优化 |
| [36:11](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=2171s) | McRT 单核 overhead 数据及正确解读 |
| [40:56](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=2456s) | 未优化 STM 可有 `2–8×` per-thread overhead |
| [42:13](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=2533s) | 硬件化：cache 做 version buffer，coherence 做 conflict detection |
| [44:11](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=2651s) | Cache line 增加 transactional `R/W` bits |
| [47:18](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=2838s) | `load A; load B; store C=5` 的 HTM walkthrough |
| [49:40](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=2980s) | Commit 时通过 upgrade 公开 write set |
| [52:07](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=3127s) | 该例归类为 lazy versioning + optimistic detection |
| [55:53](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=3353s) | 多处理器 commit/abort 表格练习 |
| [62:33](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=3753s) | Bus 提供 serialization point，但不具 scalability |
| [67:08](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=4028s) | 商用 HTM 的限制、abort 与安全问题 |
| [68:21](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=4101s) | Software vs. hardware TM 总结 |
| [69:40](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=4180s) | 实际系统通常局部采用 TM 思想，而非全程序替换 locks |
| [73:26](https://www.youtube.com/watch?v=Tbk1vnYLQqI&t=4406s) | 过渡：hardware specialization 与 heterogeneity |

## 1. 从 design space 到 concrete implementation

上一讲的两条独立轴是：

```text
data versioning:      eager / lazy
conflict detection:   pessimistic / optimistic
```

并非四种组合都同样自然。例如 eager versioning 已把 speculative value 原地写入 memory；若又拖到 commit 才 optimistic detect conflict，失败时已经暴露/覆盖了大量状态，隔离和恢复很难处理。

本讲考察两点：

1. **Software TM（STM）**：一切 tracking/validation/logging 由插桩后的软件路径完成；
2. **Hardware TM（HTM）**：把 tracking 放进 cache/coherence fast path，只显式标记 transaction begin/end。

Hybrid TM 位于两者之间：只把软件最昂贵的 bookkeeping 部件硬件化，并保留 software fallback。

## 2. STM 为什么需要 software barriers

程序员写：

```cpp
atomic {
    x = *p;
    *q = x + 1;
}
```

编译器/runtime 实际需要类似：

```cpp
atomic {
    x = STMRead(tx, p);
    STMWrite(tx, q, x + 1);
}
```

这里的 **software barrier** 是进入 STM runtime 的函数/内联代码，不是线程 rendezvous barrier。它告诉 runtime：

- 哪个 address 被读/写；
- 是否需要拿 ownership/lock；
- 是否加入 read/write set；
- 是否记录 undo/redo information；
- 当前 transaction 是否仍然 valid。

若同一 function 既可能在 transaction 内调用，也可能在 transaction 外调用，常见做法是 **function cloning**：生成 instrumented 与 uninstrumented 两个版本；JVM/.NET runtime 也可动态完成 cloning。

> [!important]
> STM 的 abstraction 很简洁，但实现会把一次普通 load/store 扩成多个 metadata accesses、branches、atomic operations 与 log updates。源代码少，不代表 runtime work 少。

## 3. 两级 metadata

### 3.1 Transaction descriptor：per transaction/thread

记录一个 transaction 自身的状态：

- status：active / committed / aborted；
- local start timestamp；
- read set、write set；
- undo log 或 write buffer；
- contention-management/abort information。

### 3.2 Transaction record：per data unit

记录某份应用数据的共享状态，例如：

- 当前 unlocked + committed version；或
- locked + owning transaction pointer。

这与 cache coherence metadata 很像：数据可被共享读取，也可被某个 writer exclusive-own。

### 3.3 Tracking granularity 的 tradeoff

| 粒度 | 优点 | 代价 |
|---|---|---|
| object | metadata 少；同对象多次 access 可摊销 | 不同 fields 也会 false conflict |
| field/word | 更接近真实 data dependence，并发度高 | metadata、lookup 与 log 更大 |
| cache line | 便于硬件复用 cache/coherence | 与语言对象边界不一致；false sharing/conflict |
| mixed | object 用 object-level，array 用 element-level | runtime/compiler 更复杂 |

若 `T1` 写 `A.x/A.y`，`T2` 写 `A.z`，object granularity 判为 conflict；field granularity 则可并发。它正是 coherence false sharing 在 TM 中的对应物。

## 4. McRT：version clock + eager writes

课堂介绍的 McRT 组合是：

```text
versioning:       eager（in-place write + undo log）
reads:            optimistic
writes:           pessimistic（先拿 per-data lock）
```

### 4.1 Clock 与 transaction record

- `GVC`：global version clock；每次 successful write transaction commit 时递增；
- `tx.start`：transaction begin 时取得的 local snapshot/version；
- `record(x)`：data `x` 的一个 tagged word。

课堂用最低位区分 record 的两种解释：

```text
LSB = 1: unlocked，剩余 bits 表示 committed version
LSB = 0: locked，剩余 bits 表示 owner transaction pointer
```

因此 version 每次加 `2`，保留最低位作为 tag。概念上：unlocked/versioned 类似可共享状态；locked/owner 类似某个 writer 的 exclusive ownership。

### 4.2 `STMRead(x)`

简化后的逻辑：

```text
v = memory[x]
r = record(x)

if r is locked by another transaction:
    wait or abort according to contention policy

if version(r) > tx.start:
    validate(tx.read_set)
    if validation succeeds:
        tx.start = current GVC   // extend snapshot

add (x, observed-version) to read_set
return v
```

它称为 optimistic read，因为 reader 不先对 `x` 加 read lock；writer 可继续执行。Reader 先相信读到的 snapshot，发现版本推进或 commit 时再整体验证。

为什么 version 变新不必立即 abort？当前刚读的 `x` 可能恰好是最新 committed value；只要历史 read set 全部仍有效，就能把 transaction 的 logical snapshot 向前扩展。

### 4.3 `STMWrite(x, value)`

```text
check x is unlocked and its version is acceptable
atomically acquire record(x)
add x to write_set
undo_log.add(x, old memory[x])
memory[x] = value              // eager/in-place
```

Write 是 pessimistic：access-time 就拿 lock，冲突现在处理；但相对于既有 readers 又是 optimistic 的，因为 writer 不逐一阻塞 readers，后者会在 validation 时发现 stale observation 并 abort。

### 4.4 Validate read set

对每个 `(x, version_seen)`：

- 若 `x` 被别的 transaction 锁住，write 正在进行，当前 observation 不再可靠；
- 若 committed version 已晚于 transaction 的 snapshot/observed version，write 已经完成，当前 observation 已过期。

两者分别表示 **write in progress** 与 **write already committed**，所以两个检查都需要。

### 4.5 Commit

简化路径：

1. 原子地把 `GVC` 增加 `2`，取得新 commit version；
2. 再验证 read set，防止 validation window 内有人 commit；
3. 若失败，按 undo log rollback 并释放 locks；
4. 若成功，为 write set 各 record 写入 commit version并解锁；
5. 丢弃 undo log，transaction 完成。

正确实现必须仔细安排 clock increment、validation 与 lock release 的顺序；它们共同定义 transaction 的 serialization point。

## 5. 视频实例：atomic copy `foo → bar`

目标：`T1` 原子地把 `(foo.x, foo.y)` 复制到 `bar`，`T2` 原子地读 `bar`。若旧值 `(0,0)`、新值 `(9,7)`，`T2` 只能看到两者之一，不能看到 `(9,0)`。

执行要点：

1. `T1` 读 `foo.x`，将 `foo@v3` 放入 read set；
2. `T2` 读 `bar.x`，将 `bar@v5` 放入 read set；
3. `T1` lock `bar`，保存旧值到 undo log，再原地写 `bar.x`；
4. `T2` 读 `bar.y` 时发现 object-level `bar` 被锁，选择等待；
5. `T1` 读 `foo.y`、写 `bar.y`，validate `foo` 后 commit，使 `bar` version 从 `5` 变 `7`；
6. `T2` 恢复并读到新版 `bar.y`；commit validation 发现它早先的 `bar.x@v5` 已过期，于是整个 transaction abort/re-execute；
7. 重跑后 `T2` 得到一致的 `(9,7)`。

> [!note]
> 这也是 object granularity 的效果：`bar.x` 与 `bar.y` 共用一个 record。课堂在 write-set 画法上讨论过是否应有两个 entries；实际表示取决于 write set 是按 object 还是按 individual write 记录，但 correctness 依赖的是完整 undo/ownership information，而不是幻灯片方框数量。

## 6. STM overhead 与 compiler optimization

STM barrier 往往包含可被消去的重复工作：

- 同一 object 在 transaction 内重复 open/read/write；
- 第一次 write 后再次写仍重复 logging/checking；
- compiler 已知 alias/escape relation，却仍执行 generic lookup；
- read-your-own-write 本可直接 forward。

通过 inlining、barrier decomposition、redundant-check elimination、alias analysis、object coalescing 可显著降低 cost。视频数据的正确读法：

- 未优化 STM 的 barrier 路径可能造成约 `2–8×` per-thread overhead；
- McRT 示例中，无 compiler optimization 的单核 overhead 约 `70–80%`；优化后约降至相对 unsynchronized code 的 `40%`；
- coarse lock 在单核更便宜，不代表在多核 workload 上更好，因为它不 scale；
- 应与一个 **correct, thread-safe** baseline 比较，不能拿无 synchronization 的错误并行程序当最终基准。

## 7. HTM：把 cache/coherence 变成 transaction engine

软件最贵的是逐 load/store instrumentation。硬件已经拥有三样可复用机制：

1. **cache**：暂存 speculative writes，充当 write buffer/undo storage；
2. **coherence protocol**：广播/路由 shared 与 exclusive requests，可检测 data conflicts；
3. **processor state checkpoint**：保存 transaction begin 时 registers/PC，以便 abort restart。

每个 cache line 在 MESI state 外增加：

- `R=1`：本 transaction 读过该 line，属于 read set；
- `W=1`：本 transaction 写过该 line，属于 write set。

于是远端 coherence request 就能命中本地 transactional metadata：

| 本地状态 | 远端请求 | 意义 |
|---|---|---|
| `W=1` | read/shared request | 远端要读本地 speculative write：write-read conflict |
| `R=1` | exclusive/upgrade request | 远端要覆盖本地已读数据：read-write conflict |
| `W=1` | exclusive/upgrade request | 两个 writers：write-write conflict |
| line present, `R=W=0` | 任意 | 只在 cache 中，不属于 transaction，不因此冲突 |

### 7.1 `load A; load B; store C=5`

课堂示意采用 cache-buffered、commit-time publish：

```text
load A  => cache[A].R = 1
load B  => cache[B].R = 1
store C => cache[C].W = 1, speculative value = 5
```

事务执行期间，`C=5` 仅在私有 speculative cache state 中，其他 cores 看不到。Commit 时 transaction 请求 write-set lines 的 exclusive ownership/upgrade：

- 若其他 active transaction 的 `R/W` bits 显示 conflict，按 policy abort 某一方；
- 若全部取得 ownership，write set 一次成为 globally visible committed state；
- 清除 transactional bits，丢弃 register checkpoint。

该课堂方案是 **lazy versioning + optimistic conflict detection**：writes 在 cache buffer 中延迟公开，commit 时集中发送 upgrades/验证。

> [!important]
> 不要把“cache line 已在 cache”与“line 属于 transaction read set”混为一谈。只有 transaction 内 access 设置的 `R/W` bits 参与 conflict 判断。

## 8. Bus 练习揭示的 serialization point

视频用三个 processors 的 read/write sets 演练 commit。Lazy optimistic 的核心检查是：

$$
W_{committing}\cap R_{active}\neq\varnothing
$$

则 active reader 观察到的世界不能与 committing transaction 共存，必须 abort/restart。课堂简化为 bus-based system，一次只有一个 transaction commit，因此 bus 天然给出 global serialization point。

这使 reasoning 简单，却不 scale：现代多核使用 directory/NoC、多条并发 message path，不再有单一总线次序，需要 protocol 明确定义 distributed ordering、conflict arbitration 与 commit atomicity。

Write-write 是否立刻要求另一个 transaction abort，取决于具体版本/commit policy；课堂表格强调：若另一个 transaction 从未读旧值，后来的 write 覆盖前值仍可能对应合法 serial order。不能脱离 algorithm policy 只凭集合名字机械判断。

## 9. HTM 的现实边界

### 9.1 Capacity 与 associativity abort

若 read/write set 用 cache 表示：

- transactional line 被 evict；
- set-associative cache 某组装不下；
- transaction 跨越 context switch、interrupt 或 unsupported instruction；

硬件可能无法继续 tracking，只能 abort。这意味着 HTM 常是 **best effort**，不能保证一个无 data conflict 的 transaction 一定成功。

### 9.2 必须有 fallback 与 forward-progress strategy

典型结构：

```cpp
for (int retries = 0; retries < N; ++retries) {
    if (hardware_transaction_succeeds()) return;
}
lock(fallback_lock);
critical_section();
unlock(fallback_lock);
```

但 HTM path 必须订阅/检测 fallback lock，否则 transactional 与 lock path 可能同时进入 critical section。Large transaction、high contention、I/O/syscall 也通常直接走 fallback。

### 9.3 2023 视频中的 Intel 结论要按时间理解

课堂指出 Intel 的 transactional extensions 曾因较多 spurious abort、software ecosystem 不成熟与安全问题而受限/在部分产品中停用。这里应把它当作课程录制时对特定实现的观察，而不是“TM 指令从所有 ISA 永久消失”的一般定律。IBM 等厂商也实现过 HTM；学术与数据库系统中，software transaction ideas 仍有局部用途。

### 9.4 TM 更常作为局部工具

实际工程通常不是把整个 parallel application 包进 transactions，而是：

- 为少数很难用 locks 写对的数据结构提供 STM；
- 在 database/runtime 内部借用 versioning、validation、optimistic concurrency control；
- 用 HTM 做 lock elision，失败后退回成熟 lock path。

抽象的价值依旧成立，但 compiler、runtime、ISA、cache capacity 与 debugging ecosystem 必须一起成熟。

## 10. 视频结尾：从通用并行走向 specialization

前半课程讨论如何有效使用既有 CPU cores、SIMD units 和 GPUs；下一步问题是：若某个 algorithm/workload 很重要，是否能改变 hardware itself 来换取更高 energy efficiency？

真实 workload 同时包含：

- thread-level parallel regions；
- SIMD/data-parallel regions；
- predictable memory accesses，可 prefetch；
- irregular but cacheable accesses；
- 适合 fixed-function/specialized data paths 的 kernels。

现代 processor 已是 heterogeneous system：以课堂的 Skylake 图为例，除 CPU cores/caches 外，还有 integrated GPU、graphics/media engines 等。专用单元用更少的 programmability/control overhead 完成特定工作，能效更高；代价是每类 engine 需要不同 programming model、data movement 与 scheduling 策略。这是 [[Stanford CS149 - Lecture 18 - Hardware Specialization|第 18 讲]]的入口。

## 11. 与 AI Infra 的连接

### 11.1 Version clock 对应 optimistic concurrency control

Parameter server、metadata service、feature store 与 distributed database 常使用：读取某 version，写入前 validate version 未变；本质与 McRT snapshot + read-set validation 相同。区别只是 metadata 跨机器后还要处理网络、failure 与 replication。

### 11.2 Metadata granularity 决定 false conflict

以整张 tensor/object 加锁简单，但两个 workers 更新不同 shards 仍被串行化；以 cache line/chunk/row 追踪提高 concurrency，却增加 metadata 与 coordination cost。它与 TM object-vs-field granularity 是同一个系统设计问题。

### 11.3 Accelerator runtime 也需要 transaction-like publish

模型 checkpoint、KV-cache migration、serving configuration 切换经常先在 private/staging state 构建，再通过 pointer/version swap 一次发布。这个“lazy build + atomic publish”就是 transaction 思想的工程化变体。

### 11.4 Optimism 是否值得取决于冲突率

Low-contention、short critical sections 适合 optimistic execution；high-contention 或 long transaction 会浪费大量 speculative compute。在 GPU/cluster 上一次 abort 的成本可能包含 kernel/network work，因此要测 conflict distribution，而不是只看 uncontended latency。

## 12. 易混点

1. **Software barrier 不是 synchronization barrier**：前者是每次 memory access 的 STM instrumentation。
2. **Optimistic read 不等于不验证**：它是不先锁住所有 readers，但会在 version change/commit 时 validate。
3. **Eager write + pessimistic ownership 可以和 optimistic reads 共存**：不同 access type 可选不同 policy。
4. **Cache presence 不等于 transactional membership**：`R/W` bits 才说明 transaction access 过。
5. **HTM 不等于无限 transaction**：cache capacity/associativity 与 interrupts 都可触发 abort。
6. **总线例子的全序只是教学简化**：scalable interconnect 必须额外解决 distributed commit ordering。
7. **低单核 overhead 不代表高 scalability**：coarse lock 可能单线程最快，但多线程完全 serial。

## 13. 本讲结论

1. STM 依赖编译器/runtime 把普通 memory accesses 变成带 metadata 的 transactional accesses。
2. Transaction descriptor 描述“这个 transaction 做了什么”，transaction record 描述“这份 data 当前处于什么 transactional state”。
3. McRT 用 global clock 与 per-data version/lock，在 eager writes 下实现 optimistic readers 和 pessimistic writers。
4. STM abstraction 的主要性能成本来自 read/write barriers；compiler optimization 很重要，但难以把成本完全消除。
5. HTM 复用 cache 做 speculative version storage、复用 coherence 做 conflict detection，并 checkpoint registers 以便 restart。
6. HTM 更快但受 cache capacity、unsupported events、contention 与安全/生态问题限制，因而必须设计 fallback。
7. TM 更像 synchronization toolbox 中的局部高层工具，而不是对 locks 的无条件、全局替代。
8. 从 TM 的 cache/coherence 复用继续往前，就是为 workload 特征选择/设计专用 hardware 的异构计算问题。

## 自测题

1. 为什么 software transactional barrier 与线程 synchronization barrier 不是同一个概念？
2. Transaction descriptor 与 transaction record 分别保存什么？为什么两者不能简单合并？
3. Object-level tracking 如何产生 false conflict？它与 false sharing 有什么同异？
4. McRT 为什么把 global clock 加 `2` 而不是加 `1`？
5. `STMRead` 发现 data version 晚于 local timestamp 时，为什么可以先验证 read set，而不一定立即 abort？
6. 为什么 McRT 的 writes 是 pessimistic，而 reads 是 optimistic？
7. 在 `foo → bar` 例子中，`T2` 为什么可能先读到旧 `bar.x`、后读到新 `bar.y`，但最终仍不会 commit 这个混合结果？
8. HTM 中 `R/W` bits 如何借 coherence request 检测三类 conflict？
9. 为什么 cache eviction 即使不是 data conflict，也可能令 transaction abort？
10. 一个 HTM lock-elision fast path 为什么必须显式观察 fallback lock？
11. Bus 为什么让 commit reasoning 简单？为什么它又不能作为现代多核的 scalable 方案？
12. 哪类 AI Infra update 适合 optimistic transaction-like publish，哪类更适合 lock/serialization？

## 相关笔记

- [[Stanford CS149 - Lecture 11 - Cache Coherence]]
- [[Stanford CS149 - Lecture 12 - Memory Consistency]]
- [[Stanford CS149 - Lecture 13 - Fine-Grained Synchronization and Lock-Free Programming]]
- [[Stanford CS149 - Lecture 16 - Transactional Memory I]]
- [[Stanford CS149 - Lecture 18 - Hardware Specialization]]
- [[Stanford CS149]]
