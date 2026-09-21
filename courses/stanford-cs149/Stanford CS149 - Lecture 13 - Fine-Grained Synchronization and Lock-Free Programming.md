---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 13
lecture_date: 2023-11-09
area: systems
topics:
  - synchronization
  - locks
  - atomics
  - compare-and-swap
  - fine-grained-locking
  - lock-free
  - aba-problem
  - hazard-pointers
aliases:
  - CS149 Lecture 13
  - Fine-Grained Synchronization and Lock-Free Programming
video_url: https://www.youtube.com/watch?v=GA1ObImqaMo
---

# Stanford CS149 - Lecture 13 - Fine-Grained Synchronization and Lock-Free Programming

> [!abstract]
> 本讲从 progress failures 入手：deadlock 是无人前进，livelock 是不断动作却没有有效进展，starvation 是系统在前进但某个线程永远失败。随后把 lock 一直拆到 cache coherence：naive test-and-set 让 cache line 在等待者之间持续 ping-pong；test-and-test-and-set 把 critical section 内的 writes 变成 shared-state reads；ticket lock 进一步将每次 release 的争抢压成有序交接。CAS 则提供“先乐观计算、提交时验证”的通用原子原语，并由此进入 fine-grained locking 与 lock-free data structures。后者避免一个暂停线程永久阻塞全系统，却不自动消除 contention、starvation、ABA 与 safe memory reclamation 问题。

## 来源与范围

- [Lecture 13 视频：Fine-Grained Synchronization and Lock-Free Programming](https://www.youtube.com/watch?v=GA1ObImqaMo)
- [官方课件 PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/finegrained/13_lockfree.pdf)
- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)

视频详细讲到 lock 实现、CAS、fine-grained linked list，并在结尾引出 single-producer/single-consumer queue 与 lock-free stack。ABA、counter/DCAS、hazard pointer 和 lock-free linked-list deletion 的完整材料在官方课件后段，课堂因时间未逐页讲解；本文将其标为“课件补充”。

## 视频索引

| 时间 | 内容 |
|---|---|
| [00:48](https://www.youtube.com/watch?v=GA1ObImqaMo&t=48s) | Deadlock 的资源等待环 |
| [05:15](https://www.youtube.com/watch?v=GA1ObImqaMo&t=315s) | Deadlock 四个必要条件 |
| [06:50](https://www.youtube.com/watch?v=GA1ObImqaMo&t=410s) | Livelock：持续动作但无有效进展 |
| [08:29](https://www.youtube.com/watch?v=GA1ObImqaMo&t=509s) | Starvation：系统进展但某线程永远失败 |
| [09:45](https://www.youtube.com/watch?v=GA1ObImqaMo&t=585s) | MSI 与 cache coherence 快速复习 |
| [18:05](https://www.youtube.com/watch?v=GA1ObImqaMo&t=1085s) | `test-and-set` 原子指令 |
| [19:05](https://www.youtube.com/watch?v=GA1ObImqaMo&t=1145s) | 用 test-and-set 构造 spin lock |
| [21:20](https://www.youtube.com/watch?v=GA1ObImqaMo&t=1280s) | 等待者为何让 lock cache line 持续 bounce |
| [24:33](https://www.youtube.com/watch?v=GA1ObImqaMo&t=1473s) | 持有 lock 不等于持有 lock variable 的 cache line |
| [26:42](https://www.youtube.com/watch?v=GA1ObImqaMo&t=1602s) | Core 数增大时 lock/unlock cost 上升 |
| [28:35](https://www.youtube.com/watch?v=GA1ObImqaMo&t=1715s) | Test-and-test-and-set（TTAS） |
| [30:50](https://www.youtube.com/watch?v=GA1ObImqaMo&t=1850s) | 等待期间只读，release 时才出现 feeding frenzy |
| [33:24](https://www.youtube.com/watch?v=GA1ObImqaMo&t=2004s) | Ticket lock：FIFO fairness 与一次 release write |
| [36:28](https://www.youtube.com/watch?v=GA1ObImqaMo&t=2188s) | CUDA atomic operations 与 CAS |
| [38:18](https://www.youtube.com/watch?v=GA1ObImqaMo&t=2298s) | 用 CAS 实现 `atomicMin` |
| [42:52](https://www.youtube.com/watch?v=GA1ObImqaMo&t=2572s) | LL/SC 如何借助 coherence 检测 intervening access |
| [47:32](https://www.youtube.com/watch?v=GA1ObImqaMo&t=2852s) | CAS/LLSC 作为构造其他 atomics 的基础 |
| [48:06](https://www.youtube.com/watch?v=GA1ObImqaMo&t=2886s) | C++ `atomic<T>` 与 `is_lock_free()` |
| [49:17](https://www.youtube.com/watch?v=GA1ObImqaMo&t=2957s) | Atomicity 之外仍需要 memory ordering/fence |
| [52:53](https://www.youtube.com/watch?v=GA1ObImqaMo&t=3173s) | 并发 sorted linked list 的 races |
| [58:11](https://www.youtube.com/watch?v=GA1ObImqaMo&t=3491s) | Coarse lock：正确但把整个结构串行化 |
| [61:29](https://www.youtube.com/watch?v=GA1ObImqaMo&t=3689s) | Per-node lock 与 hand-over-hand locking |
| [65:02](https://www.youtube.com/watch?v=GA1ObImqaMo&t=3902s) | 删除节点为何同时锁 previous/current |
| [67:35](https://www.youtube.com/watch?v=GA1ObImqaMo&t=4055s) | Fine granularity 仍可能限制 overtaking |
| [69:42](https://www.youtube.com/watch?v=GA1ObImqaMo&t=4182s) | 粗锁、细锁与中等粒度的性能权衡 |
| [71:00](https://www.youtube.com/watch?v=GA1ObImqaMo&t=4260s) | Lock-free：乐观执行，提交时验证 |
| [72:49](https://www.youtube.com/watch?v=GA1ObImqaMo&t=4369s) | Blocking algorithm 中 lock holder 被换出的问题 |
| [73:45](https://www.youtube.com/watch?v=GA1ObImqaMo&t=4425s) | SPSC queue 与 lock-free stack 预告 |

## 1. 三类 progress failure

### 1.1 Deadlock

一组 threads 永远无法前进。经典 Coffman 条件同时成立时可能 deadlock：

1. mutual exclusion：resource 不能同时共享；
2. hold and wait：持有已有 resource，同时等待新的；
3. no preemption：不能强制夺回 resource；
4. circular wait：等待关系形成环。

```text
T0 holds A, waits B
T1 holds B, waits C
T2 holds C, waits A
```

破坏任一条件即可避免对应 deadlock，例如统一 lock ordering 破坏 circular wait，一次性申请全部 resources 破坏 hold-and-wait。

### 1.2 Livelock

Threads 不断响应彼此、回退、重试，但没有 operation commit。两个 transactions 每次冲突后同时 abort、同时 restart 是典型例子。随机或指数 backoff、priority/age-based contention management 可打破对称性。

### 1.3 Starvation

系统持续完成 operations，但某个 thread 长期或永久得不到服务。Naive spin lock 不保证公平；ticket lock 的 FIFO order 则提供更明确的 starvation protection。

| 状态 | 系统是否有动作 | 系统是否完成有用工作 | 某线程是否可能永不成功 |
|---|---:|---:|---:|
| Deadlock | 否 | 否 | 是，全部 |
| Livelock | 是 | 否 | 是，全部/多个 |
| Starvation | 是 | 是 | 是，个别 |

## 2. Atomic read-modify-write 是 lock 的地基

概念性 `test_and_set(addr)`：原子读取 `*addr`，并写入 `1`，返回旧值。

```cpp
void lock(int* l) {
    while (test_and_set(l) != 0) { }
}

void unlock(int* l) {
    *l = 0;
}
```

若返回 `0`，本线程把 free→held 的 transition 完成，因此获得 lock；返回 `1` 则继续 spin。

真实 ISA 可能提供 `compare_exchange`、`lock cmpxchg`、LL/SC 等。重点不是指令名字，而是 **read + condition + conditional write 作为不可分割的 linearization point**。

## 3. Naive test-and-set 为什么在 coherence 上很贵

Test-and-set 可能写，因此 cache 必须先取得 line 的 exclusive/modified permission。假设 P1 持有 lock，P2/P3 不断 test-and-set：

```text
P2 requests M → line moves to P2, CAS fails
P3 requests M → line moves to P3, CAS fails
P2 requests M → line moves back, CAS fails
...
```

关键反直觉：

> 持有 lock 是 abstract synchronization state；持有 lock variable 对应 cache line 是 physical cache state。P1 可拥有 lock，却在大部分 critical section 时间里根本没有该 line。

P1 最终 `unlock` 还要与所有 spinners 竞争 line ownership。等待者越多，bus/interconnect traffic 越大，critical section 本身的 memory accesses 也受干扰，lock/unlock latency 随 core count 上升。

## 4. Test-and-test-and-set：先本地读，再竞争写

```cpp
void lock(int* l) {
    while (true) {
        while (*l != 0) {          // ordinary read-spin
            cpu_relax();
        }
        if (test_and_set(l) == 0) // only compete when it looks free
            return;
    }
}
```

当 lock held 时，所有等待者可把 line 保留在 `S`，read-spin 都是 local cache hits，不产生连续 invalidation。Release 写 `0` 会 invalidate sharers；它们重新读到 `0` 后同时发 RMW，产生一次 **feeding frenzy / thundering herd**。

因此 TTAS 的收益是：

- critical section 期间 interconnect 安静；
- release 时仍有 `O(P)` contenders；
- 不保证公平；
- 比 naive TAS 多一次 read path，但通常值得。

Backoff 可进一步减少 release 后同时重试。

## 5. Ticket lock：把争抢变成 FIFO 排队

```cpp
struct TicketLock {
    atomic<int> next_ticket{0};
    atomic<int> now_serving{0};
};

void lock(TicketLock* l) {
    int mine = fetch_add(&l->next_ticket, 1);
    while (l->now_serving.load() != mine) cpu_relax();
}

void unlock(TicketLock* l) {
    l->now_serving.fetch_add(1);
}
```

性质：

- `fetch_add` 只在加入队列时执行；
- 等待是对 `now_serving` 的 reads；
- release 只需一次 increment；
- tickets 按 FIFO 服务，公平且避免某个 waiter 永久饥饿。

局限：所有 waiters 仍读同一 line，release 会 invalidate/更新所有 sharers；NUMA 大系统常用 MCS/CLH queue locks，让每个 waiter spin 在更局部的状态上。

## 6. CAS：验证“我依据的世界还没变”

概念定义：

```cpp
T compare_and_swap(T* addr, T expected, T desired) {
    T old = *addr;
    if (old == expected) *addr = desired;
    return old;
}
```

整个函数由 hardware atomic 地完成。CAS 的通用思想：

1. 读取旧 state；
2. 在 private/local state 中计算 candidate update；
3. CAS 检查 shared state 是否仍等于旧 state；
4. 未变则 commit，已变则重读并重算。

### 6.1 用 CAS 实现 `atomicMin`

```cpp
int atomic_min(atomic<int>* addr, int x) {
    int old = addr->load();
    while (x < old) {
        int desired = x;
        if (addr->compare_exchange_weak(old, desired))
            break;
        // failure updates old; recompute predicate
    }
    return old;
}
```

若线程 A 读到 `100` 想写 `90`，期间线程 B 已写 `80`，A 的 CAS 必须失败；否则 A 会把更小的 `80` 覆盖成 `90`。失败后 `old=80`，A 发现 `90` 不再改善 minimum，退出。

这并未把整个 min computation mutual-exclude；多个 threads 可并行计算，只在 commit point 竞争。

### 6.2 CAS 与 LL/SC

`load-linked(addr)` 读取并建立 reservation；`store-conditional(addr,val)` 仅在 reservation 未因相关 intervening access 失效时写入。Coherence protocol 很适合帮助实现 reservation invalidation。

| CAS | LL/SC |
|---|---|
| 比较 value 是否仍等于 expected | 检查从 LL 到 SC 期间 reservation 是否仍有效 |
| 容易遭遇 ABA | 对“中间被改过又改回来”更敏感 |
| x86 常见 | ARM/RISC 架构常见 |

## 7. Atomicity 不等于 memory ordering

即便 lock word 的 CAS/store 本身 atomic，critical-section data 仍可能被 compiler/hardware 重排：

```text
critical data write
unlock(lock = 0)
```

若 data write 被观察到晚于 unlock，另一个 thread acquire 后仍读到旧 data。正确 lock implementation 必须提供：

- acquire semantics on successful lock；
- release semantics on unlock；
- 必要时 architecture-specific fences；
- compiler barrier，禁止编译器跨边界移动 shared accesses。

`std::atomic<T>` 既提供 atomic operations，也让程序明确 memory order。`is_lock_free()` 表示该类型在实现中是否无需内部 lock；“类型名叫 atomic”不保证任何大小的对象都由单指令无锁实现。

## 8. Spin、sleep 与 hybrid mutex

Spin lock 适合：critical section 极短、holder 正在另一个 core 上运行、线程不会长时间 preempt。若等待很久，spin 浪费 CPU 与 memory bandwidth。

Heavyweight mutex 常采用 hybrid strategy：

1. 短暂 spin，赌 holder 很快 release；
2. 多次失败后进入 OS，sleep/park thread；
3. unlock 时唤醒 waiter。

选择不是“spin 或 block”二元对立，而是根据 expected wait、oversubscription 与 scheduler behavior 分层。

## 9. Concurrent linked list：先从 correctness 找保护范围

无同步的 sorted list 可能：

- 两个 inserts 都改同一个 `prev->next`，其中一个 node 丢失；
- insert 与 delete 交错，让新 node 指向已 free node；
- 两个 deletes double-free 或把 list 尾部断开。

### 9.1 Coarse-grained lock

给整个 list 一个 lock，包住完整 insert/delete。容易验证，但所有 operations 串行；即使访问不相交区域也不能并行。

实现还要防 early return 忘记 unlock。C++ 用 RAII (`std::lock_guard`) 把 unlock 与 scope 绑定，可消除这类 control-flow bug。

### 9.2 Per-node / hand-over-hand locking

Traversal 始终先锁 next，再释放 previous：

```text
lock(prev)
lock(curr)
unlock(prev)
advance
```

删除 `curr` 时必须同时持有 `prev` 与 `curr`：

- lock `prev`：防别人删除/插入并修改 `prev->next`；
- lock `curr`：防别人删除当前 node 或在其后修改相关链接。

永远不出现“手上没有任何 lock 却继续用刚读 pointer”的窗口，这就是 hand-over-hand / lock coupling。

### 9.3 Granularity 是性能变量

细粒度锁：

- 增加潜在并行度、减少单个 lock contention；
- 增加每 node storage、每步 acquire/release、代码复杂度和死锁风险；
- 链表入口处仍可能形成 convoy，后来的 operation 无法越过前方修改者。

粗锁有时更快，因为 pointer chasing 本来很便宜，逐 node locks 的 overhead 反而主导。中间方案可以每若干 nodes/partition/bucket 一把 lock。最终要 profile，而不是凭“更细一定更并行”下结论。

## 10. Blocking、lock-free、wait-free

### 10.1 Blocking

若某 thread 在不合时机暂停，可能无限期阻止其他 threads 完成 operation，就是 blocking。即使 lock 用 spin 实现，只要 holder 可阻塞全体，算法仍是 blocking。

### 10.2 Lock-free

Lock-free 保证 **system-wide progress**：无论哪个 thread 被 preempt，总有某个 thread 能在有限步骤内完成 operation。它不保证每个特定 thread 完成，因此仍可 starvation。

### 10.3 Wait-free

更强：每个 operation 都在有界步数内完成，即 per-thread progress。课程重点是 lock-free，但复习时不要把二者混用。

```text
wait-free ⇒ lock-free ⇒ non-blocking
```

Lock-free 基本套路：speculatively read/compute，最后用 CAS 验证并 linearize；失败就 retry。它把“进入前排他”改成“提交时检测冲突”。

## 11. SPSC queue：用 ownership 分离避免相互写同一 control word

固定容量 ring buffer：producer 只更新 `tail`，consumer 只更新 `head`：

```cpp
bool push(Queue* q, int v) {
    if (next(q->tail) == q->head) return false; // full
    q->data[q->tail] = v;
    q->tail = next(q->tail);                    // release-publish
    return true;
}

bool pop(Queue* q, int* out) {
    if (q->head == q->tail) return false;       // empty
    *out = q->data[q->head];
    q->head = next(q->head);
    return true;
}
```

单 producer/单 consumer 的限制至关重要；多 producer 会同时更新 `tail`，多 consumer 会同时更新 `head`。在 relaxed machine 上还需要 release/acquire，保证 data element 先于 tail publish、consumer 观察 tail 后能看到 data。

## 12. Lock-free stack（课件补充）

```cpp
void push(Stack* s, Node* n) {
    do {
        n->next = s->top;
    } while (!CAS(&s->top, n->next, n));
}

Node* pop(Stack* s) {
    while (true) {
        Node* old = s->top;
        if (!old) return nullptr;
        Node* next = old->next;
        if (CAS(&s->top, old, next)) return old;
    }
}
```

CAS 是 linearization point：只要 `top` 仍为读取时的 `old`，修改看起来就在这一瞬间发生。

### 12.1 ABA problem

仅比较 pointer value 不足以证明“世界没变”：

```text
T0 reads top=A, next=B; pauses
T1 pops A
T1 modifies/reuses A, pushes A back
T1 pushes D, stack becomes A→D→B→C
T0 sees top still A; CAS(A→B) succeeds
result: D is lost, structure corrupted
```

地址经历 `A→...→A`，CAS 只看到“仍是 A”，却不知道中间改变过。

### 12.2 Tagged pointer / version counter

把 top 与 monotonically increasing counter 一起比较：

```text
(top=A, version=7) ≠ (top=A, version=9)
```

需要 double-width CAS，或把 pointer+tag 打包进一个原子 word。Counter wraparound、alignment bits 与平台 CAS width 都是实现约束。

### 12.3 Safe memory reclamation 与 hazard pointer

即使 CAS 逻辑没有 ABA，T0 读取 `old_top` 后，T1 可能 pop 并 `delete old_top`；T0 再读 `old_top->next` 就 use-after-free。

Hazard pointer 思路：

1. 每个 thread publish 自己即将 dereference 的 node pointer；
2. 再次验证 pointer 仍是当前结构的一部分；
3. 被 remove 的 node 放入 retire list，不立即 free；
4. 周期扫描所有 hazard pointers，仅回收无人保护的 retired nodes。

替代方案包括 epoch-based reclamation、RCU、reference counting、GC。Lock-free algorithm 的难点往往不是 CAS，而是 object lifetime。

## 13. Lock-free 不等于没有 contention、也不一定更快

- 多 threads 更新同一 CAS location，仍会 cache-line bouncing；
- CAS 在 heavy contention 下反复失败，浪费 computation；
- Lock-free correctness、linearizability、ABA 与 reclamation 验证成本高；
- 机器只运行该 HPC/ML job、threads 很少被 preempt 时，良好 coarse/fine locks 可能同样快甚至更快；
- Web server/database 等有大量 threads、page faults、priority inversion、lock-holder preemption 时，non-blocking progress 更有价值。

> [!important]
> “没有 lock object”不是 lock-free 的定义；“用了 CAS”也不自动 lock-free。必须证明无论单个 thread 怎样暂停，系统范围总有 operation 能完成。

## 14. 与 AI Infra 的连接

- GPU runtime command queues、network completion queues、token scheduler ready queues 常采用 SPSC/MPSC ring buffer；producer/consumer 数量是选择算法的第一项前提。
- Model serving 的 global request queue 若用 naive spin lock，在几十 cores 上会被 coherence traffic 放大；per-shard queue、ticket/MCS lock 或 work stealing 更合适。
- CUDA atomics 也会在 L2/memory system 上 serialize；“原子指令一条”不代表便宜，热点 counter 仍需 warp/block aggregation。
- Lock-free KV-cache page freelist 会直接遇到 ABA 与 memory reclamation；tagged pointer 只能解决 identity 变化，不能单独解决 freed-memory dereference。
- C++ atomics 的 acquire/release 与 device-side memory scope 是系统正确性的 contract；只测试 x86 后认为代码正确，移到 ARM/accelerator 可能暴露 ordering bug。

## 15. 本讲结论

1. Deadlock、livelock、starvation 描述不同 progress failure，修复策略也不同。
2. Naive TAS lock 的主要扩展性问题来自 coherence，不只是循环指令数。
3. TTAS 把持锁期间的争抢变成 read hits；ticket lock增加公平性并把交接显式排队。
4. CAS/LLSC 是构造 atomics 的通用基础，但 atomicity 之外仍需 memory ordering。
5. Fine-grained locking 要先证明每个 pointer/link 的保护 invariant，再谈性能。
6. Lock-free 保证 system-wide progress，不保证 per-thread progress，也不消除 contention。
7. ABA 与 safe memory reclamation 是 lock-free pointer structure 的两道独立难题。

## 16. 自测题

1. 分别给出 deadlock、livelock、starvation 的最小例子；ticket lock 主要改善哪一个？
2. 为什么持有 lock 的 P1 可能没有 lock variable 的有效 cache line？
3. TTAS 为什么只把连续 traffic 压到 release 时刻，而没有完全消除 contention？
4. Ticket lock 相比 TTAS 增加了什么 fairness guarantee，代价是什么？
5. 用 CAS 写 `atomicMax`；CAS 失败后为什么必须重新计算 predicate？
6. Atomic CAS 正确为何不代表 critical-section data 的 publication 正确？
7. Hand-over-hand linked-list delete 为什么至少同时锁 `prev` 和 `curr`？
8. Lock-free 与 wait-free 分别保证谁前进？
9. 画出 stack 的 ABA interleaving，并说明 tagged pointer 检测的是什么。
10. Tagged pointer 解决 ABA 后，为什么还需要 hazard pointer/epoch reclamation？

## 相关笔记

- [[Stanford CS149 - Lecture 11 - Cache Coherence]]
- [[Stanford CS149 - Lecture 12 - Memory Consistency]]
- [[Stanford CS149 - Lecture 14 - Midterm Review]]
- [[Stanford CS149 - Lecture 16 - Transactional Memory I]]
