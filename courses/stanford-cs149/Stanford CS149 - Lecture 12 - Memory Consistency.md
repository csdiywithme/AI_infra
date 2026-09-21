---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 12
lecture_date: 2023-11-02
area: systems
topics:
  - cache-coherence
  - msi
  - mesi
  - directory-protocol
  - false-sharing
  - numa
  - memory-consistency
  - sequential-consistency
  - memory-fence
aliases:
  - CS149 Lecture 12
  - Memory Consistency
video_url: https://www.youtube.com/watch?v=nFXWmo9MFiY
---

# Stanford CS149 - Lecture 12 - Memory Consistency

> [!abstract]
> 本讲先完成 cache coherence：逐条推导 MSI 状态机，用 MESI 的 clean-exclusive state 消除常见 read-then-write 的一次 upgrade transaction，再从 snooping bus 过渡到 directory、NUMA 与 false sharing。53 分钟后才进入 memory consistency：coherence 规定同一地址上的 writes 如何被一致观察，consistency 则规定不同地址上的 memory operations 允许怎样重排。Sequential consistency 最直观，却会阻止 store buffer、out-of-order overlap 等关键优化；真实硬件用 relaxed model 换性能，再由 fence、acquire/release、lock/barrier 与语言的 SC-for-DRF contract 恢复程序员需要的顺序。

## 来源与范围

- [Lecture 12 视频：Memory Consistency](https://www.youtube.com/watch?v=nFXWmo9MFiY)
- [官方课件 PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/locksconsistency/12_consistency.pdf)
- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)

视频前 53 分钟是 Lecture 11 的 cache coherence 续篇；本文不按标题删减。最后几页关于 language memory model 的内容在视频末尾只预告，本文用官方课件补全，并明确标记为课件内容。

## 视频索引

| 时间 | 内容 |
|---|---|
| [00:20](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=20s) | Coherence 定义与两个 invariant 回顾 |
| [03:49](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=229s) | Write-back invalidation protocol 要解决的两个问题 |
| [05:32](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=332s) | MSI 三个 states 与 memory 是否 up-to-date |
| [06:37](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=397s) | Processor actions 与 BusRd / BusRdX / BusWB |
| [08:42](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=522s) | 从 I 出发推导 processor-initiated transitions |
| [11:25](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=685s) | Snooper 对 bus transaction 的 transitions |
| [15:46](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=946s) | 三处理器 MSI trace |
| [21:10](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=1270s) | `M→S` 为什么是维护 SWMR 所必需 |
| [22:12](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=1332s) | Communication 改变 miss distribution 与 latency |
| [26:58](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=1618s) | MSI 的 read-then-write 双 transaction 问题 |
| [27:45](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=1665s) | MESI 的 Exclusive state |
| [30:29](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=1829s) | Bus/snooping 为什么不 scale |
| [31:23](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=1883s) | Directory：只联系 owner/sharers |
| [32:37](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=1957s) | Inclusive L3 与 per-line sharer vector |
| [35:21](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=2121s) | Coherence 对程序员：更多、更远、更慢的 misses |
| [36:09](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=2169s) | NUMA local vs. remote DRAM |
| [38:45](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=2325s) | Average memory access time 与 profiling tools |
| [40:43](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=2443s) | False sharing 的 per-thread counter 例子 |
| [44:10](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=2650s) | Line size 对 true/false sharing misses 的相反作用 |
| [49:49](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=2989s) | Cache coherence 总结 |
| [53:03](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=3183s) | 从 coherence 转向 memory consistency |
| [55:57](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=3357s) | 同地址 vs. 不同地址：两者的边界 |
| [58:09](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=3489s) | 四类 memory ordering |
| [59:58](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=3598s) | Store-buffering litmus test |
| [61:38](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=3698s) | Happens-before graph 与 cycle 判不可能 |
| [64:37](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=3877s) | Lamport sequential consistency |
| [67:24](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=4044s) | 为什么为了性能放松 ordering |
| [69:21](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=4161s) | Store buffer 怎样让 SC 禁止的 `00` 出现 |
| [71:50](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=4310s) | TSO/PSO：允许哪些 reorderings |
| [75:07](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=4507s) | 更激进 relaxed consistency |
| [76:00](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=4560s) | Fence 以性能换顺序 |
| [77:23](https://www.youtube.com/watch?v=nFXWmo9MFiY&t=4643s) | Data race、DRF 与同步库 |

## 1. MSI：把 permission 与 data movement 写成状态机

### 1.1 三个 states

| State | 本 cache 的权限 | 系统范围约束 | Memory |
|---|---|---|---|
| `I` Invalid | 不可读写，访问会 miss | 其他 cache 状态未知 | 未限定 |
| `S` Shared | 可读、不可静默写 | 一个或多个 clean copies | 最新 |
| `M` Modified | 可读写 | 恰好只有本 cache 有有效 copy | 可能陈旧 |

`M` 同时编码 dirty + exclusive ownership；`S` 表示 read-only sharing，而非“肯定有多个副本”。一个 cache 也可独自在 `S`。

### 1.2 两类输入与三类 bus transaction

状态机同时处理：

- local processor actions：`PrRd`、`PrWr`；
- snooped bus actions：其他 cache 发出的 `BusRd`、`BusRdX`、以及 data/writeback response。

`A / B` transition label 可读成：“收到/执行 A 后，cache controller 产生 B，并转移状态”。

### 1.3 Local processor 触发的主要 transitions

| 当前 | 操作 | Bus action | 新状态 | 原因 |
|---|---|---|---|---|
| `I` | `PrRd` | `BusRd` | `S` | 取 line 供读 |
| `I` | `PrWr` | `BusRdX` | `M` | 一次取得 data + exclusive permission |
| `S` | `PrRd` | 无 | `S` | read hit |
| `S` | `PrWr` | `BusRdX`/upgrade | `M` | invalidate 其他 sharers |
| `M` | `PrRd/PrWr` | 无 | `M` | owner 可本地反复读写 |

> [!important]
> `I + PrWr` 不必先 `BusRd` 再 `BusRdX`。`BusRdX` 本身同时表达“给我完整 line”和“我要写”；分两次会多一笔 transaction，且中间可能被别的 core 插入。

### 1.4 Snooped transitions

| 当前 | 观察到 | 动作 | 新状态 |
|---|---|---|---|
| `S` | `BusRd` | 无；memory 可供数据 | `S` |
| `S` | `BusRdX` | invalidate local copy | `I` |
| `M` | `BusRd` | owner 提供/写回最新 line | `S` |
| `M` | `BusRdX` | owner 提供最新 line 并失去 ownership | `I` |

从 `M` 离开时必须由 owner 交付 data，因为 memory 可能仍是旧值。`M→S` 既让 requester 读到最新值，也把原 owner 降为 reader，恢复 multiple-reader epoch。

## 2. 用 trace 检查协议，而不是死背箭头

假设 `X` 初始只在 memory：

| Step | Processor action | Bus | P1 | P2 | P3 | Data source |
|---|---|---|---|---|---|---|
| 1 | P1 `R(X)` | `BusRd` | S | I | I | memory |
| 2 | P3 `R(X)` | `BusRd` | S | I | S | memory |
| 3 | P3 `W(X)` | `BusRdX` | I | I | M | memory/local line |
| 4 | P1 `R(X)` | `BusRd` | S | I | S | P3 owner + writeback |
| 5 | P1 `R(X)` | none | S | I | S | P1 cache |
| 6 | P2 `W(X)` | `BusRdX` | I | M | I | memory（此时已最新） |

检查方法：

1. 当前是否 hit？
2. 若 miss，谁有最新值？
3. 请求者最终需要 read 还是 write permission？
4. 所有其他 copies 应留在 `S` 还是进 `I`？
5. 是否仍最多一个 `M`？若进入 `S`，memory 是否最新？

## 3. MESI：把 clean 与 ownership 解耦

MSI 中第一次 read 把 line 放进 `S`；随后同一 core write 还要发 upgrade，即使系统中根本没有其他 copy。MESI 增加：

| State | 含义 |
|---|---|
| `E` Exclusive | clean、只有本 cache 持有；memory 最新 |

若 `BusRd` 发现无人拥有/共享该 line，请求者进入 `E`；随后：

```text
E --local write, no bus--> M
E --snoop another BusRd--> S
```

`E→M` 是 silent upgrade，优化常见的 read-modify-write。它没有改变 coherence semantics，只减少了 communication。

> [!warning]
> `E` 不是 “dirty exclusive”；那是 `M`。`E` 的价值恰恰是 clean 但已确认唯一。

## 4. Snooping 到 directory：序列化应当是 per-line 的

Bus/snooping 的两个扩展性问题：

- 每个 transaction 被所有 caches 看见，即使绝大多数没有该 line；
- 不同 cache lines 的无关 transactions 也争同一 serialized bus。

Directory 为每个 line 维护 owner/sharer metadata：

```text
directory[X] = {state, owner or sharer-set}
```

请求 write 时只向实际 sharers 发 invalidation；请求 read dirty line 时直接找 owner。这样可使用 ring/mesh/point-to-point network，并只序列化同一 line 的冲突操作。

课上 Skylake-like 示例把 directory 放在 inclusive L3：若 L2 中存在 line，L3 必须有对应 entry，才能在 eviction/lookup 时准确知道 sharers。最直观 sharer vector 每 core 一 bit，随 core count 线性增大；现实会用 sparse pointer、limited directory、hierarchy 等压缩。

> [!note]
> MSI/MESI 是 permission state protocol；snooping/directory 是传播和定位这些状态变化的机制。两组概念正交，不是互斥替代品。

## 5. Coherence 的性能成本：AMAT、NUMA、false sharing

### 5.1 Communication 会改变 miss 分布

多线程程序不只增加“几次通信指令”。一旦别的 core 写入，原本会 L1 hit 的 line 被 invalidate，后续访问变成 coherence miss，可能要去 LLC、remote cache 或 DRAM。

$$
AMAT = \sum_i P(\text{access level }i)\cdot Latency_i
$$

并行版本可能让访问概率从 L1/L2 向 shared LLC、remote cache、DRAM 移动，因此即使 instruction count 类似，AMAT 也上升。

### 5.2 NUMA：地址空间统一，访问代价不统一

多 socket machine 中，每个 CPU 有 local DRAM；访问另一个 socket 的 DRAM 要跨 inter-socket link：

```text
CPU 0 ─ local DRAM 0   (high bandwidth / lower latency)
  │
  └── interconnect ── CPU 1 ─ local DRAM 1
```

First-touch allocation、thread affinity 与 partition placement 会决定主要访问是 local 还是 remote。正确性不受影响，性能可能完全不同。

### 5.3 False sharing

多个 threads 各自更新逻辑独立的 counters，但 counters 落在同一 cache line：

```cpp
int counter[num_threads];  // adjacent elements may share a line
```

每次 store 都需要整行 ownership，line 在 cores 间 ping-pong。课程 contrived example 在四核上从约 14.2 s 变为 padding 后 4.7 s。

```cpp
struct alignas(64) PerThreadCounter {
    int value;
    // padding / align to keep hot writers on different cache lines
};
```

| 增大 line size | True sharing | False sharing |
|---|---|---|
| 作用 | 更好利用 spatial locality，miss 可能下降 | 更多无关变量被捆绑，miss/invalidations 可能上升 |

Padding 不是免费：它浪费 capacity/bandwidth，可能破坏其他 phase 的 locality。先用 profiler/performance counters 定位，再调整 layout、partition 或 update aggregation。

## 6. Coherence 与 consistency 的边界

| 概念 | 地址范围 | 要回答的问题 |
|---|---|---|
| Coherence | 单个地址/cache line | 所有人是否以同一顺序观察对 `X` 的 writes？read 得到哪个 `X`？ |
| Consistency | 多个不同地址 | `W(X), R(Y), W(Z)...` 对其他 processors 可以以什么顺序显现？ |

没有 caches 仍需要 memory consistency model；有 private caches 才额外需要 coherence。Coherence 让系统“像没有多个 cache copies”，consistency 定义 shared-memory program 的合法行为。

## 7. 四类 memory ordering

对不同地址 `X != Y`，线程 program order 中可能有：

| 顺序 | 记号 | 例子 |
|---|---|---|
| Store → Load | `W(X) → R(Y)` | publish X 后读 Y |
| Load → Load | `R(X) → R(Y)` | 连续读取两个 locations |
| Load → Store | `R(X) → W(Y)` | 读取条件后写结果 |
| Store → Store | `W(X) → W(Y)` | 写 data 后写 ready flag |

Memory model 是 hardware/compiler 与 software 的 contract：哪些 program-order edges 对其他 threads 保证可见，哪些可被重排或重叠。

## 8. Sequential consistency（SC）

Lamport 的定义：

1. 所有 processors 的所有 memory operations 存在一个 total order；
2. 每个 processor 的 operations 在该 total order 中保持 program order；
3. reads 与该 order 中最近的 writes 一致。

“随机 switch”直觉：memory 每次挑一个 processor，完整执行它的下一次 memory operation，再挑另一个。

### 8.1 Store-buffering litmus test

初始 `A=B=0`：

```text
P0                  P1
A = 1               B = 1
r0 = B              r1 = A
```

在 SC 下：

- `(r0,r1)=(1,1)`：两次 store 都先发生；
- `(0,1)` 或 `(1,0)`：一种合法 interleaving；
- `(0,0)`：不可能。

若要同时读到 `0`，需满足：

```text
R0(B) < W1(B) < R1(A) < W0(A) < R0(B)
```

形成 happens-before cycle，意味着某事件必须先于自身，因此 SC 下不可能。

## 9. 为什么放松 SC：store buffer 与 overlap

Store miss 可能需要 ownership lookup、invalidations、network traversal，耗时数十到数百 cycles；随后对无关地址的 load 若会 hit，却被 SC 的 `W→R` 顺序阻塞。

Store buffer 让 processor：

1. 把 store 放入 buffer，先认为本线程已完成；
2. 继续执行独立 load；
3. 后台让 store 对其他 processors 变为 visible。

于是上一 litmus 中，两边的 stores 都还在各自 buffer，而 loads 先读 memory 的旧值，`(0,0)` 成为可能。这不是 incoherence：对每个地址仍可 coherent；只是不同地址的可见顺序不再等于源代码顺序。

## 10. Relaxed models 的层次

课程用“放开哪些 edges”建立直觉：

| 模型/方向 | 典型允许项 | 动机 |
|---|---|---|
| SC | 不允许四类重排 | 最直观 semantics |
| TSO-like | 允许较早 store 对别的 core 尚未 visible 时，后续不同地址 load 先完成（`W→R`） | store buffer / hide store latency |
| PSO | 还可让不同地址 stores 以不同顺序 visible（`W→W`） | 某 store miss、另一 store hit |
| Weak/Release consistency | 普通 data operations 之间更自由；在 sync edges 上恢复约束 | 最大化 overlap/OoO |

> [!warning]
> 课程口头讲解在 TSO/processor consistency/PSO 名称上当场做了更正。复习重点不是背一张不稳妥的名字表，而是对具体 architecture/language 明确问：它放松了哪条 ordering、是否 multi-copy atomic、用什么 fence/acquire/release 恢复顺序。

单线程 observable behavior 仍必须符合语言的 as-if rule；“reordering”只在其他 threads 的观察结果中暴露。若一个单线程程序因 OoO 得到不同结果，那是处理器/编译器错误，不是 relaxed consistency。

## 11. Synchronization 把必要的 order 加回来

### 11.1 Fence

Full memory fence 的概念是：fence 前的 memory operations 在 fence 后 operations 开始前完成/可见。不同架构还提供 load fence、store fence。它们昂贵，因为会限制 overlap、排空 buffers、约束 compiler/hardware schedule。

```text
ordinary reorderable operations
──────────── memory fence ────────────
ordinary reorderable operations
```

### 11.2 Acquire / release

比 full fence 更有针对性：

- release：其前面的 reads/writes 不得被移动到 release 之后；常用于 unlock/publish；
- acquire：其后的 reads/writes 不得被移动到 acquire 之前；常用于 lock/consume published state；
- 某线程的 release 被另一线程 acquire 观察到时，建立 synchronizes-with / happens-before edge。

```text
producer: write data → release(store ready=1)
consumer: acquire(load ready) → read data
```

没有这条 edge，consumer 可能先看见 flag，再看见旧 data。

### 11.3 Data race 与 SC-for-DRF（课件补充）

两次 accesses conflict，当且仅当：

- 来自不同 threads；
- 访问同一 memory location；
- 至少一次是 write。

若 conflicting accesses 未被 synchronization ordering，就存在 data race。C11/C++11/Java 等语言的关键 contract 是 **sequential consistency for data-race-free programs**：程序正确使用 atomic/lock/barrier/acquire-release 建立必要顺序，compiler/runtime 负责映射到硬件；racy code 则没有这个简单保证，C/C++ 中甚至可能是 undefined behavior。

> [!important]
> 应用开发者通常不应手工猜硬件 fence。优先使用语言级 `std::atomic`、mutex、barrier 与成熟并发库，让实现者处理 compiler reorder 与 architecture-specific instruction。

## 12. 与 AI Infra 的连接

- Lock-free request queue、GPU completion flag、CPU↔device ring buffer 的正确性都依赖 acquire/release，不是“变量已经 atomic”就自动 publish 了周围数据。
- Model server 的 per-core statistics、reference counts、work-stealing deque 容易出现 false sharing；cache-line layout 能比换更快 CPU 更有效。
- NUMA machine 上的 KV cache/embedding tables 若由远端 socket 高频访问，会同时承受 remote latency、limited interconnect bandwidth 与 coherence traffic。
- GPU kernel 内的 memory model、warp/block barriers 与 system-scope atomics也遵循同一思想：只在需要的 scope 上建立 ordering，避免全局 fence。
- “优化常见的无冲突访问，让同步只为冲突付费”贯穿 relaxed memory、lock-free 与接下来的 transactional memory。

## 13. 本讲结论

1. MSI 的 `M/S/I` 同时表示 data validity 与 read/write permission；离开 `M` 时 owner 必须交付最新 data。
2. MESI 的 `E` 让独占 clean line 可以 silent `E→M`，减少 read-then-write 的 transaction。
3. Directory 将全局 broadcast 变成对 sharers/owner 的定点通信；状态协议与传播机制是两层概念。
4. Coherence traffic 会增加 AMAT；NUMA 与 false sharing 是程序员必须能识别的具体表现。
5. Coherence 管同地址，consistency 管不同地址的 ordering。
6. SC 直观但限制 store buffer/OoO；relaxed models 用更多合法执行换性能。
7. Fence/acquire/release/locks 建立必要的 happens-before；语言用 SC-for-DRF 把硬件细节封装起来。

## 14. 自测题

1. 对 MSI 的 `I+PrWr`，为什么应发 `BusRdX` 而不是 `BusRd` 后再 upgrade？
2. `M` cache snoop 到 `BusRd` 与 `BusRdX` 时分别怎样转移？
3. `E` 与 `M` 都 exclusive，二者最关键差别是什么？
4. Directory 为什么常与 inclusive LLC 配合？它与 MESI 是替代关系吗？
5. 为什么更大的 cache line 同时可能降低 true-sharing miss、提高 false-sharing miss？
6. 给定 store-buffering litmus，画出 `(0,0)` 所需 happens-before cycle。
7. Store buffer 为什么改变 consistency，却不必破坏 coherence？
8. `data=42; ready=1;` 为什么在 relaxed machine 上需要 release/acquire？
9. Atomicity 与 ordering 有何不同？一个 atomic flag 是否足以自动保护普通 data？
10. “SC for DRF” 给应用程序员和同步库作者各提出什么要求？

## 相关笔记

- [[Stanford CS149 - Lecture 11 - Cache Coherence]]
- [[Stanford CS149 - Lecture 13 - Fine-Grained Synchronization and Lock-Free Programming]]
- [[Stanford CS149 - Lecture 16 - Transactional Memory I]]
- [[Stanford CS149 - Lecture 17 - Transactional Memory II]]
