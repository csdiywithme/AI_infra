---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 4
lecture_date: 2023-10-05
area: systems
topics:
  - parallel-programming
  - ispc
  - amdahls-law
  - synchronization
  - barriers
  - gauss-seidel
aliases:
  - CS149 Lecture 4
  - Parallel Programming Basics
video_url: https://www.youtube.com/watch?v=0-ztm8SKq70
---

# Stanford CS149 - Lecture 04 - Parallel Programming Basics

> [!abstract]
> 本讲把前面接触过的 ISPC 细节提升成一个通用框架：并行程序需要处理 **decomposition（分解）、assignment（分配）、orchestration（协作）和 mapping（映射）**。课程用 ISPC、任务运行时和红黑 Gauss–Seidel 三组例子说明：先定义正确的并行语义，再考虑硬件如何执行；高性能往往来自改变算法、减少串行段和减少同步，而不只是“多开几个线程”。

## 来源与范围

- [Lecture 4 视频：Parallel Programming Basics](https://www.youtube.com/watch?v=0-ztm8SKq70)
- [官方课件 PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/progbasics/04_progbasics.pdf)
- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)

本文按视频讲解顺序整理，代码和图示再用课件校对。视频中关于实现选择、任务开销实验和 barrier 推理的口头解释也被保留下来。

## 视频索引

| 时间 | 内容 |
|---|---|
| [00:14](https://www.youtube.com/watch?v=0-ztm8SKq70&t=14s) | 真实处理器同时包含多核、SIMD、硬件多线程和超标量执行 |
| [04:33](https://www.youtube.com/watch?v=0-ztm8SKq70&t=273s) | ISPC gang 的抽象语义与具体实现 |
| [12:35](https://www.youtube.com/watch?v=0-ztm8SKq70&t=755s) | interleaved / blocked 分配与内存访问 |
| [15:16](https://www.youtube.com/watch?v=0-ztm8SKq70&t=916s) | `foreach` 的语义，而不是当前编译器恰好如何实现 |
| [21:03](https://www.youtube.com/watch?v=0-ztm8SKq70&t=1263s) | 含跨迭代依赖的 `foreach` 为什么错误 |
| [25:04](https://www.youtube.com/watch?v=0-ztm8SKq70&t=1504s) | per-instance partial sum 与 `reduce_add` |
| [27:59](https://www.youtube.com/watch?v=0-ztm8SKq70&t=1679s) | ISPC task、gang、线程池之间的关系 |
| [34:25](https://www.youtube.com/watch?v=0-ztm8SKq70&t=2065s) | 空任务实验：线程创建和调度不是免费的 |
| [41:10](https://www.youtube.com/watch?v=0-ztm8SKq70&t=2470s) | 并行编程的四项职责 |
| [43:03](https://www.youtube.com/watch?v=0-ztm8SKq70&t=2583s) | Amdahl 定律与图像平均值案例 |
| [53:23](https://www.youtube.com/watch?v=0-ztm8SKq70&t=3203s) | Gauss–Seidel 的数据依赖 |
| [57:56](https://www.youtube.com/watch?v=0-ztm8SKq70&t=3476s) | 用红黑迭代改变算法、暴露并行性 |
| [67:39](https://www.youtube.com/watch?v=0-ztm8SKq70&t=4059s) | `diff` 的数据竞争、锁与局部归约 |
| [72:32](https://www.youtube.com/watch?v=0-ztm8SKq70&t=4352s) | barrier 的语义以及为什么原实现需要三道 barrier |

## 1. 先分清抽象语义和实现

并行程序有两个容易混在一起的问题：

1. **程序承诺了什么语义？** 哪些实例、迭代或任务在逻辑上存在，允许什么执行顺序？
2. **机器如何实现这些语义？** 编译器是否生成 SIMD 指令，运行时用多少线程，操作系统把线程放到哪个核心？

ISPC 调用一个 `export` 函数时，语义上会创建一个 **gang**，其中有 `programCount` 个 program instances，每个 instance 有不同的 `programIndex`。这并不等于“创建 `programCount` 个线程”。

一种合法实现可以完全串行地逐个执行实例；常见高效实现则把 gang 映射成向量指令的各个 SIMD lane。前者很慢，却仍可能满足语言语义。这正是课程不断强调抽象与实现分离的原因：

> 正确性应该来自抽象语义；性能分析才依赖具体实现。

### 1.1 gang size 不等于硬件 SIMD 宽度

gang size 是编译目标的一部分。假设 gang 有 16 个实例、硬件一次处理 8 个 lane，编译器可以用两条向量指令完成一次 gang 操作。这两条指令之间没有数据依赖时，还可能给处理器提供额外的 instruction-level parallelism。

因此不要把下面几个数量机械地画等号：

- logical program instances；
- SIMD lanes；
- tasks；
- software threads；
- physical cores / hardware execution contexts。

它们属于不同抽象层，运行时和编译器负责其中一部分映射。

## 2. ISPC 的分配方式与内存访问

让一个 gang 处理长度为 `N` 的数组，常见分配方法有：

### Interleaved assignment

```text
instance 0: 0, 8, 16, ...
instance 1: 1, 9, 17, ...
...
instance 7: 7, 15, 23, ...
```

同一轮向量操作访问连续地址，通常可以合并成少量向量 load/store，也容易落在同一条 cache line 中。

### Blocked assignment

```text
instance 0: [0, N/8)
instance 1: [N/8, 2N/8)
...
```

每个实例自身访问连续，但同一条 SIMD 指令的 lane 可能跨越多个相距很远的 cache line，甚至多个页，带来更多 cache/TLB 访问。

这不是“interleaved 永远更好”。本讲后面的多核网格求解器反而适合 blocked rows，因为它减少处理器之间的边界通信。性能结论必须和具体硬件层级、数据布局与通信模式一起讨论。

## 3. `foreach` 表达的是独立迭代

`foreach (i = 0 ... N)` 的关键语义不是“第 `i % programCount` 个实例执行第 `i` 次迭代”，而是：

- gang 中的实例共同覆盖整个迭代域；
- 每个迭代恰好执行一次；
- 各次迭代必须可以独立执行；
- 编译器可以自由选择静态、交错、分块甚至动态分配。

当前实现常采用 interleaved assignment，但程序不应该依赖这个实现细节。

### 3.1 合法例子：逐元素变换

```c
foreach (i = 0 ... N) {
    output[i]     = abs(input[i]);
    output[i + N] = abs(input[i]);
}
```

每个迭代只读取自己的输入并写入独占位置，执行顺序不影响结果。

### 3.2 非法例子：跨迭代读写

```c
foreach (i = 1 ... N) {
    A[i] = A[i - 1] + 1;
}
```

迭代 `i` 读取迭代 `i-1` 写入的值，存在 loop-carried dependency。并行执行时可能先读取旧值，结果依赖调度顺序。

编译器通常无法完整发现这类错误：指针别名、索引和依赖可能由运行时数据决定。把循环标成独立迭代，是程序员给出的正确性承诺。

## 4. 归约：先局部累积，再合并

求和不能让所有实例直接更新一个 `uniform` 标量：

```c
uniform float sum = 0;
foreach (i = 0 ... N)
    sum += x[i];             // 多个实例竞争同一个值
```

正确思路是让每个 program instance 维护自己的 partial sum，最后进行横向归约：

```c
varying float partial = 0;
foreach (i = 0 ... N)
    partial += x[i];

uniform float sum = reduce_add(partial);
```

典型 lowering 是：

1. 用向量 load 读取一组元素；
2. 每个 lane 在自己的累加器中累积；
3. 末尾执行 horizontal reduction，把各 lane 的结果合成一个标量。

这种模式会在本讲的线程版求解器中再次出现：**高频更新留在局部，低频地合并全局结果**。

## 5. Task 不是 thread

> [!question]
> 视频里讲的这部分我没听懂，能展开讲下这里吗，包括这三种方式，以及讲的 OS context switch 与 hardware thread switch？

> [!answer] 先说结论
> **Task 是“要完成的一份逻辑工作”，thread 是“执行工作的工人”。** 高效系统通常创建少量、长期存在的 worker threads，再让它们从队列中执行大量 tasks，而不是给每个 task 创建一个 thread。OS context switch 是把一个 software thread 换出 hardware execution context、再换入另一个，涉及调度和状态保存/恢复；hardware-thread switch 则因为多个硬件上下文已经同时驻留在核心中，硬件只需选择另一个 ready context 发射指令，可以非常快。

### 5.1 先分清四层对象

| 对象 | 它是什么 | 谁创建/管理 | 例子 |
|---|---|---|---|
| Task | 一份逻辑工作及其参数 | 程序、编译器或 runtime | “计算第 17 个 tile” |
| Software thread | OS 可调度的执行流，拥有栈和寄存器状态 | 程序与 OS | `std::thread`、线程池 worker |
| Hardware thread / execution context | 核心中一套可驻留的架构状态 | 处理器硬件 | SMT 的 logical CPU、GPU warp context |
| SIMD lane | 一条向量指令中的一个数据通道 | 编译器与向量硬件 | AVX lane、ISPC program instance 的常见映射目标 |

它们不是一一对应的。例如可以有：

- 1,000 个 tasks；
- 8 个长期存在的 software worker threads；
- 8 个 hardware execution contexts；
- 每个 worker 执行 ISPC 代码时，一条向量指令再使用 8 个 SIMD lanes。

此时绝不是创建了 `1,000 × 8` 个线程。通常只有 8 个 software threads 同时从队列取任务，而每个线程在执行某个 task 时使用向量指令处理一个 ISPC gang。

ISPC 中一次 task launch 创建的是许多可调度工作，任务内部仍以 gang 为单位执行。常见 runtime 使用固定大小的线程池：

```mermaid
flowchart LR
    A["ISPC task launch"] --> B["多个逻辑 task / gang"]
    B --> C["运行时工作队列"]
    C --> D["固定数量 worker threads"]
    D --> E["OS 映射到硬件执行上下文"]
```

### 5.2 视频比较的三种执行方式

课堂从 [34:25](https://www.youtube.com/watch?v=0-ztm8SKq70&t=2065s) 开始，让三种实现反复执行许多“几乎什么都不做”的 tasks。故意使用空任务，是为了把管理开销放大到看得见。

#### 方式一：在调用线程中顺序执行

```cpp
for (int i = 0; i < numTasks; ++i)
    task(i);
```

- 始终只有调用者这一个 software thread；
- 没有真正的 task-level parallelism；
- 不需要创建线程、入队、唤醒、加锁或 `join`；
- 对空任务而言，成本接近一次普通函数调用和循环控制。

它的缺点是只能使用一个 hardware context；优点是固定开销最小。任务本身非常小时，省下来的管理成本可能比并行带来的收益更大。

#### 方式二：每个 task 创建一个 OS thread

```cpp
for (int i = 0; i < numTasks; ++i)
    threads.emplace_back(task, i);

for (auto &t : threads)
    t.join();
```

每个 task 都对应新的 `std::thread`，通常意味着一次 OS thread 创建和销毁。系统需要处理：

- 分配 thread control block 和栈；
- 进入内核创建可调度实体；
- 把 thread 放入 scheduler；
- 在少数 hardware contexts 上调度大量 runnable threads；
- 完成后的 `join`、清理和资源回收。

若机器只有 8 个 hardware contexts，却创建 10,000 个同时可运行的计算线程，这叫 **oversubscription**。多出来的线程不会产生额外 ALU，只会让 OS 反复切换它们，并扰动 cache/TLB。

这种方式语义上可行，但不适合大量细粒度任务。它更像反例，用来说明 **task 数量可以很大，software thread 数量却不应随 task 数量增长**。

#### 方式三：固定大小的 thread pool

```cpp
ThreadPool pool(numHardwareContexts);

for (int i = 0; i < numTasks; ++i)
    pool.enqueue(task, i);

pool.wait_until_done();
```

启动时创建少量 workers，此后一直复用：

```text
tasks:    t0 t1 t2 t3 t4 t5 ... t999
             ↓ 进入共享或分布式队列
workers:  w0 w1 w2 ... w7
             ↓ OS 映射
hardware contexts: h0 h1 h2 ... h7
```

每个 worker 完成一个 task 后再取下一个。与“每任务一个线程”相比：

- thread 创建/销毁成本只付一次；
- software threads 的数量与硬件资源相称；
- 仍能通过许多 tasks 做动态负载均衡；
- 每个 task 仍要付出入队、出队、同步和可能的 worker 唤醒成本。

所以 thread pool 位于两者之间：比直接调用多了调度开销，但能并行；比每任务创建线程便宜得多。

### 5.3 怎样理解空任务实验

视频测得：

- 直接顺序调用大约只需 1.6 ms；
- 对空工作而言，顺序调用比线程池快约 23 倍；
- 线程池又比“每任务创建一个 C++ 线程”快约 300 倍。

这些数字只代表课堂机器，重点是数量级关系：

```text
普通函数调用
    < 队列操作 + worker 调度
    << 创建、调度、销毁一个 OS thread
```

可用一个简化模型理解。设每个 task 的有效计算时间为 `w`，共有 `N` 个 tasks、`P` 个 workers：

$$
T_{sequential} \approx Nw
$$

$$
T_{pool} \approx \frac{Nw}{P}+Nq+T_{startup}
$$

这里 `q` 是每个 task 的队列/同步成本。空任务中 `w≈0`，顺序版几乎没有成本，thread pool 只剩 `Nq`，自然更慢。随着 `w` 增大，`Nw/P` 的并行收益逐渐超过 `Nq`，thread pool 才会胜出。

“每任务一线程”还要为每个 task 支付明显更大的创建/销毁成本 `c_thread`：

$$
T_{thread-per-task} \gtrsim \frac{Nw}{P}+N c_{thread}
$$

只有单个 task 非常重时，`c_thread` 占比才可能变得不显眼；即使如此，固定线程池通常仍是更可控的设计。

这就是 **task granularity（任务粒度）**：

- task 太小：调度成本淹没计算；
- task 太大：可调度工作太少，负载可能不均；
- 好的粒度：task 明显重于调度成本，同时 task 数量又明显多于 workers。

### 5.4 OS context switch 是什么

OS 把许多 software threads 映射到有限的 hardware execution contexts。若 runnable software threads 多于硬件上下文，scheduler 会做时间片切换：

```text
hardware context h0:
thread A 执行 → OS 换出 A → thread B 执行 → OS 换出 B → A ...
```

一次 OS context switch 通常涉及：

1. 因时钟中断、阻塞或主动让出而进入内核；
2. 保存旧 thread 的 program counter、stack pointer 和寄存器等状态；
3. scheduler 选择另一个 runnable thread；
4. 恢复新 thread 的状态并返回用户态；
5. 新 thread 重新建立 cache、TLB、branch predictor 等局部性。

最后一项不一定都在切换瞬间完成，却会在后续执行中形成间接代价。因此 OS switch 不只是“换一下寄存器”，还可能破坏工作集局部性。视频在 [39:06](https://www.youtube.com/watch?v=0-ztm8SKq70&t=2346s) 用“可能达到很高的 cycle 数”强调它与硬件切换之间的数量级差异；具体数值依机器和工作集而变，重要的是它远非免费。

### 5.5 Hardware-thread switch 为什么快

支持 hardware multithreading 的核心内部，同时保留多套 architecture state，例如每个 hardware thread 自己的：

- program counter；
- architectural registers；
- 部分控制状态。

但它们共享核心的 execution units 和部分 cache。因为 A、B 两套状态已经驻留在硬件中，不需要像 OS 那样先把 A 保存到内存、再恢复 B。若 A 在等待 cache miss，硬件可以直接从 B 选择 ready instructions：

```text
cycle 0: hardware thread A 的指令
cycle 1: A 等数据，选择 hardware thread B
cycle 2: 继续选择 ready instruction
```

细粒度 multithreading 可以按周期选择不同 context；SMT 甚至可能在同一周期从多个 hardware threads 发射指令。课堂把它概括为“约一个 cycle 的切换”，更准确的理解是：**没有一次完整的软件 save/restore，硬件调度器只是从已经驻留的 contexts 中选可执行指令。**

Hardware thread 也不是免费增加核心：多个 contexts 仍共享 ALU、load/store units、cache 和 memory bandwidth。它主要用于在一个 thread stall 时填补执行空槽，而不是把峰值资源凭空翻倍。

### 5.6 两种 switch 的直接对比

| | OS context switch | Hardware-thread selection / switch |
|---|---|---|
| 切换对象 | software threads | 核心中已驻留的 hardware contexts |
| 决策者 | OS scheduler | processor instruction scheduler |
| 状态位置 | 旧状态需保存，新状态需恢复 | 多套状态同时保留在硬件中 |
| 时间尺度 | 通常远高于单个 cycle | 可逐周期选择，SMT 可同周期混合发射 |
| 主要代价 | 内核调度、save/restore、局部性损失 | contexts 争用共享执行与存储资源 |
| 解决的问题 | 让大量进程/线程公平、隔离地共享 CPU | 用另一个 ready context 隐藏 pipeline/memory stall |

### 5.7 为什么纯计算程序通常不应过量创建 software threads

若程序是 CPU-bound，`P` 个 hardware contexts 已经能占满执行资源。继续增加 runnable software threads：

- 不增加计算单元；
- 增加 OS scheduling/context-switch 成本；
- 每个 thread 的栈和工作集占更多内存；
- 可能降低 cache/TLB locality。

因此常见策略是建立约等于可用 hardware contexts 数量的 worker pool，把更多并行度表示成 tasks。例外包括线程会长时间阻塞 I/O、runtime 了解阻塞情况，或应用故意保留少量 oversubscription 来隐藏某类软件等待；这时最佳数量需要测量。

一句话串起整段视频：

> 把丰富的并行性表示成大量轻量 tasks；用少量长期存在的 software workers 执行它们；让 OS 把 workers 映射到 hardware contexts；再让硬件用 SIMD 和 hardware multithreading 高效利用核心内部资源。

## 6. 并行程序的四项职责

| 职责 | 核心问题 | 本讲例子 | 通常由谁完成 |
|---|---|---|---|
| Decomposition | 如何把问题拆成可并行的工作？ | 像素、红格/黑格、数组迭代 | 主要由程序员/算法设计者 |
| Assignment | 工作分给哪个逻辑 worker？ | interleaved、blocked、task queue | 程序员、编译器或 runtime |
| Orchestration | 工作如何通信、同步、排序？ | lock、barrier、局部归约 | 程序与 runtime |
| Mapping | 逻辑 worker 放到哪些硬件资源？ | instance→lane、thread→core | 编译器、runtime、OS、硬件 |

四者不是完全独立，但这个框架能帮助定位问题。例如负载不均主要是 assignment；锁竞争属于 orchestration；线程被错误放置可能是 mapping。

### 6.1 Decomposition 往往最需要领域知识

编译器擅长执行你已经表达出来的并行性，却很难凭空发明另一个数值算法。后面的 Gauss–Seidel 例子中，真正的大突破不是调度器技巧，而是把原地更新算法改成 red-black ordering。

## 7. Amdahl 定律：先消灭串行瓶颈

设串行程序中不可并行的比例为 `S`，其余部分在 `P` 个处理器上理想并行：

$$
T_P = T_1\left(S + \frac{1-S}{P}\right)
$$

$$
\operatorname{Speedup}(P)=\frac{1}{S+(1-S)/P}
$$

当 `P → ∞`：

$$
\operatorname{Speedup}_{max}=\frac{1}{S}
$$

所以：

- 1% 串行比例，在 64 核上最多约 `1/(0.01+0.99/64) ≈ 39.3×`；
- 10% 串行比例，在 64 核上只有约 `8.8×`。

### 7.1 图像亮度与平均值

假设一个 `N×N` 图像要做两件事：

1. 调整每个像素亮度，工作量 `Θ(N²)`；
2. 求所有像素平均值，工作量 `Θ(N²)`。

若只并行第一步，总时间近似 `N²/P + N²`，即使无限处理器也只能接近 2 倍。

更好的做法是每个 worker 计算局部和，再归约 `P` 个 partial sums：

$$
T_P \approx \frac{2N^2}{P}+P
$$

当 `N²` 远大于 `P²` 时，归约成本很小，整体速度提升可以接近 `P`。

## 8. Gauss–Seidel：数据依赖限制了并行性

二维迭代求解器反复把每个网格点更新为邻居的某种平均值，并累积本轮变化 `diff`，直到收敛。

朴素 row-major 原地更新包含依赖：当前点会使用本轮刚刚更新过的左邻居和上一行。因此所有格点并不独立。

### 8.1 Wavefront 方案

沿对角线推进可以遵守依赖，同一条对角线上的点并行执行。但它有明显问题：

- 开头和结尾可并行点很少；
- 对角线之间频繁同步；
- 遍历顺序对行主序内存不友好。

它保留了原算法的精确更新顺序，却未必是高性能方案。

### 8.2 Red-black ordering

把网格染成棋盘格：

1. 所有红点只依赖黑点，可以同时更新；
2. 同步；
3. 所有黑点使用新的红点，可以同时更新；
4. 同步并检查收敛。

这改变了数值更新顺序，浮点中间值不会与 row-major 版本逐位一致，也可能需要更多迭代才能达到同样阈值。但它暴露出大量规则并行性，额外工作可能被更高并行度抵消。

> [!important]
> 并行优化有时需要接受“单步做法不同、最终满足同一收敛目标”，而不是强行保持串行执行顺序。

## 9. Assignment 会决定通信量

红格和黑格已经完成 decomposition，接下来还要把格点分给 processors：

- **interleaved rows**：每个 worker 拿每第 `P` 行；
- **blocked rows**：每个 worker 拿一段连续行。

在共享内存多核上，blocked rows 通常让大部分邻居仍由同一 worker 处理，只有块边界跨 worker。interleaved rows 则几乎每行都跨 worker，增加 cache coherence 和通信压力。

这里 blocked 优于 interleaved，与前面 SIMD 数组访问的判断相反。原因是优化目标不同：前者减少核心之间的边界通信，后者让同一条向量指令访问连续地址。

## 10. 两种表达方式

### 10.1 数据并行接口

```c
forall (red cells)
    update_red_cell();

forall (black cells)
    update_black_cell();
```

程序员声明迭代独立，系统负责实例数量、分配和调度。代码接近算法描述，通常更容易保证正确。

### 10.2 Shared-address-space threads

```c
worker(thread_id, num_threads) {
    rows = blocked_rows(thread_id, num_threads);
    while (!done) {
        update_red(rows);
        barrier();
        update_black(rows);
        barrier();
    }
}
```

线程直接共享地址空间，程序员显式决定行块、共享状态、锁和 barrier。控制力更强，但正确性责任也更多。

## 11. `x++` 不是原子操作

多个线程执行 `global_diff += delta` 时，底层通常是：

1. load `global_diff`；
2. add `delta`；
3. store 新值。

两个线程可能同时读到旧值，随后互相覆盖，丢失一次更新。这是 data race，而不是“小概率浮点误差”。

最直接的修复是在更新周围加锁，但把锁放进每个格点的内层循环会：

- 串行化高频路径；
- 产生严重 lock contention；
- 让 Amdahl 的串行部分变大。

更好的方式仍是局部归约：

```c
float my_diff = 0;

for (cell in my_cells)
    my_diff += update(cell);

lock(diff_lock);
global_diff += my_diff;
unlock(diff_lock);
```

每个 worker 从“每格点一次加锁”降为“每轮一次加锁”。

## 12. Barrier 到底保证了什么

barrier 是全体参与者的阶段边界：任何线程都不能越过，直到所有线程都到达。它不仅用于“大家等一下”，更是在建立跨线程的 happens-before 关系。

原共享内存求解器按程序执行顺序，每轮需要三道 barrier：

1. **重置完成屏障**：所有线程都完成本轮清零后，才能有人开始写入本轮贡献；否则迟到线程可能再次清零、覆盖早到线程已经写入的贡献；
2. **归约完成屏障**：所有线程都已经把 `myDiff` 合并进 `diff`，才能检查收敛；
3. **检查完成屏障**：所有线程都已经读取本轮 `diff`，最快线程才能开始下一轮并再次清零。

删除任意一道都会产生具体错误：

- 早到线程可能读取不完整的 diff；
- 某线程可能在别人检查前清零；
- 下一轮的贡献可能被迟到的清零覆盖。

### 12.1 用空间换同步：三份 `diff`

> [!question]
> 这里我希望你进一步拓展讲下：为什么需要三份 `diff`，它具体怎样把三道 barrier 降成一道？

> [!answer] 核心原因
> 算法真正需要的是“第 `k` 轮所有 partial sums 完成后，才能判断第 `k` 轮是否收敛”。另外两道 barrier 主要是在保护同一个变量名 `diff` 被连续多轮重复使用。给不同生命周期分配不同槽位后，上一轮的读取、本轮的累加和下一轮的清零可以访问三个不同地址，从而并发发生。

这个 one-barrier 版本在下一讲开头 [00:09](https://www.youtube.com/watch?v=mmO2Ri_dJkk&t=9s) 才完整揭晓。

#### 12.1.1 先看单个 `diff` 为什么需要三道 barrier

简化后的原始代码是：

```c
while (!done) {
    myDiff = 0.0f;

    diff = 0.0f;
    barrier();                 // B1：重置完成

    myDiff = computeMyRegion();

    lock(diffLock);
    diff += myDiff;
    unlock(diffLock);

    barrier();                 // B2：全体归约完成

    done = (diff / (n*n) < TOLERANCE);

    barrier();                 // B3：全体检查完成
}
```

三道 barrier 各自阻止一种具体的错误交错。

**没有 B1：迟到的清零会覆盖贡献。**

```text
thread 0: diff = 0 → 开始计算 → diff += 10
thread 1:                                      diff = 0
```

thread 0 的 `10` 被 thread 1 的迟到清零抹掉。

**没有 B2：早到线程会读取不完整的和。**

```text
thread 0: diff += 10 → 读取 diff，判断收敛
thread 1:                                      diff += 20
```

thread 0 检查的是 `10`，而不是完整结果 `30`。

**没有 B3：下一轮清零会破坏迟到线程的检查。**

```text
thread 0: 读取本轮 diff → 进入下一轮 → diff = 0
thread 1:                                         读取本轮 diff
```

thread 1 读到的是下一轮的零，而不是刚完成的本轮结果。

所以问题并非只有 `diff += myDiff` 是否原子。锁只保护多个线程同时执行 read–modify–write；它不保护 **跨阶段、跨迭代的生命周期**。

#### 12.1.2 哪些操作其实可以重叠

设聚合后的第 `k` 轮结果为 `D_k`。barrier 放行之后，不同线程速度不同，可能同时出现：

| 慢线程 | 快线程 |
|---|---|
| 仍在读取 `D_{k-1}`、判断上一轮是否收敛 | 已经开始计算并累加 `D_k` |
| 即将进入第 `k` 轮 | 甚至可以提前清零将来承载 `D_{k+1}` 的位置 |

这三件事之间没有真正的算法依赖：

1. 读取已经完成的 `D_{k-1}`；
2. 累加当前的 `D_k`；
3. 为下一轮准备一个值为零的 `D_{k+1}`。

原代码把它们都放在名为 `diff` 的同一个地址，制造了 storage reuse dependency。它们必须排队，不是因为数学上互相依赖，而是因为会覆盖同一块存储。

#### 12.1.3 三槽轮转代码

课程给出的核心伪代码可以写成：

```c
float diff[3];                    // 全局，初始全为 0

void solve() {
    int index = 0;                // 每个线程本地，但初值相同

    diff[0] = 0.0f;
    barrier();                    // 仅初始化时执行一次

    while (true) {
        float myDiff = computeMyRegion();

        lock(diffLock);
        diff[index] += myDiff;    // 累加本轮 D_k
        unlock(diffLock);

        diff[(index + 1) % 3] = 0.0f;  // 提前准备下一轮 D_{k+1}

        barrier();                // 每轮唯一的 barrier

        if (diff[index] / (n*n) < TOLERANCE)
            break;

        index = (index + 1) % 3;
    }
}
```

把第 `k` 轮的 `index` 记作 `k mod 3`，在快慢线程重叠的窗口中，三个槽位分别承担：

| 槽位 | 生命周期 | 谁在访问 |
|---|---|---|
| `diff[(k-1) % 3]` | 上一轮已完成结果 `D_{k-1}` | 慢线程仍可能读取、检查 |
| `diff[k % 3]` | 当前轮结果 `D_k` | 快线程正在加上自己的 `myDiff` |
| `diff[(k+1) % 3]` | 下一轮结果 `D_{k+1}` | 提前清零，等待下一轮使用 |

因为三个角色落在三个不同地址上，它们可以同时发生而不互相覆盖。

#### 12.1.4 唯一一道 barrier 同时完成两件事

每个线程到达 barrier 之前已经：

1. 把自己的 `myDiff` 加入 `diff[index]`；
2. 把下一槽 `diff[(index+1)%3]` 初始化为零。

因此 barrier 放行时同时保证：

- **本轮归约已完成**：之后读取 `diff[index]` 一定看到所有线程的贡献；
- **下一轮槽位已准备好**：之后把 index 向前轮转，可以安全向下一槽累加。

barrier 后各线程读取同一个完整的 `diff[index]`，所以会得到相同的收敛判断。快线程可以先进入下一轮，因为它写的是下一槽，不会覆盖慢线程仍在读取的本轮槽。

注意这不是“完全没有同步”：

- `diff[index] += myDiff` 仍需要 lock、atomic 或另一种 reduction 实现；
- 每轮仍有一道 barrier，保证归约完成；
- 消除的是围绕 reset 和 check 的另外两道 barrier。

#### 12.1.5 为什么两份不够

只用两个槽时：

$$
(k+1) \bmod 2 = (k-1) \bmod 2
$$

也就是说，“下一轮要清零的槽”恰好就是“上一轮慢线程可能仍在读取的槽”。可能发生：

```text
barrier k-1 放行

fast thread:  读完 D_{k-1} → 计算 D_k → 清零 D_{k+1}
slow thread:                                              还没读取 D_{k-1}
```

使用两个槽时，`D_{k+1}` 和 `D_{k-1}` 是同一地址。fast thread 的清零会让 slow thread 读错。使用三个槽后，它们才是不同地址。

一道 barrier 还限制了线程最多领先一个迭代阶段：fast thread 可以完成第 `k` 轮并在 barrier 等待，却不能在 slow thread 到达之前继续跨过 barrier。正因为最大重叠窗口中同时存在“上一轮、当前轮、下一轮”三个生命周期，三份 storage 正好够用。

#### 12.1.6 用前三轮走一遍

| 迭代 | 本轮累加 | 本轮提前清零 | barrier 后检查 |
|---:|---|---|---|
| 0 | `diff[0] += myDiff` | `diff[1] = 0` | 读取 `diff[0]` |
| 1 | `diff[1] += myDiff` | `diff[2] = 0` | 读取 `diff[1]` |
| 2 | `diff[2] += myDiff` | `diff[0] = 0` | 读取 `diff[2]` |
| 3 | `diff[0] += myDiff` | `diff[1] = 0` | 读取 `diff[0]` |

到第 2 轮清零 `diff[0]` 时，所有线程已经到达过第 1 轮 barrier。要到达那里，它们必须先读完第 0 轮的 `diff[0]` 并完成第 1 轮工作，所以旧的 `diff[0]` 已经无人需要，可以安全复用。

> [!note] 关于严格语言内存模型
> 课件是并行算法伪代码，写成所有线程都执行 `diff[next] = 0`。在严格 C/C++ 内存模型中，多个线程并发写同一非原子对象，即使都写零，也构成 data race。实际实现可以指定一个线程负责清零，然后用同一道 barrier 保证其他线程看见零；或使用符合 runtime 语义的原子/归约机制。这不改变三槽轮转的核心推理。

#### 12.1.7 这是多版本存储，而不只是一个小技巧

三份 `diff` 本质上是给不同迭代分配不同版本：

```text
单版本：上一轮读 ──等待── 本轮写 ──等待── 下一轮清零

三版本：上一轮读  diff[a]
        本轮写    diff[b]      同时进行
        下一轮清零 diff[c]
```

它和 double/triple buffering、ring buffer、epoch-based reclamation、流水线 stage buffers 属于同一类思想：

> 当同步仅仅来自同一存储位置被过早复用时，为不同生命周期提供独立版本，可以用少量空间换取更多并行和更少同步。

## 13. 对 AI Infra 的启发

### 13.1 Tensor program 的独立性契约

CUDA kernel、Triton program 和编译器生成的 fused kernel 都依赖同一原则：只有真正独立的工作才能自由重排。错误的 alias 或跨 tile 依赖，会让“看起来可向量化”的代码变成竞态。

### 13.2 局部累积再归约

loss reduction、gradient accumulation、histogram 和 attention 中的统计量，都适合先在 lane/warp/block/worker 内局部累积，再逐级合并。这样同时减少原子操作和全局通信。

### 13.3 Task granularity

推理服务中的 microbatch、数据加载任务和分布式执行图节点都存在同样权衡：任务太大导致负载不均，任务太小则 queue、RPC、线程调度和 kernel launch 开销占主导。

### 13.4 改变算法通常胜过微调调度器

red-black ordering 的核心不是线程 API，而是重构依赖图。类似地，AI 系统常通过算子融合、分块、流水线或复制只读/版本化状态来暴露并行性。

## 14. 本讲结论

1. 区分程序的并行语义和编译器/运行时的具体实现。
2. 并行程序要同时考虑 decomposition、assignment、orchestration 和 mapping。
3. `foreach` 承诺迭代独立；不要依赖当前 ISPC 的 lane 分配方式。
4. task 是逻辑工作，不等于 software thread；粒度太小会被调度开销吞没。
5. Amdahl 定律要求先处理串行段，高频全局锁尤其危险。
6. 数据依赖可能需要算法级改变，例如 red-black ordering。
7. 局部归约、blocked assignment 和状态版本化分别减少锁、通信与 barrier。

## 15. 自测题

1. 为什么“ISPC gang 有 8 个实例”不能推出“程序创建了 8 个线程”？

    **面试回答：** Gang 的 8 个实例是语言层的逻辑执行单位，不是 8 个 OS threads。ISPC 通常在一个软件线程内把它们编译成 SIMD 指令及掩码操作；要利用多个核心，还需要显式 task launch、线程池或其他多核运行时。

2. `foreach` 的正确性要求是什么？为什么 `A[i] = A[i-1] + 1` 不满足？

    **面试回答：** `foreach` 中各次迭代必须能按允许的分配和顺序执行，不能依赖相邻迭代先完成。`A[i] = A[i-1] + 1` 原地读取上一个元素的新值，形成循环携带依赖，任意并行会改变结果；可保留顺序或先改写成合适的 scan 等并行算法。

3. 在 SIMD 数组计算中 interleaved 常有利，而网格多核划分中 blocked rows 常有利，差异来自哪里？

    **面试回答：** 优化的是不同层次的通信：SIMD interleaved 让同一步各 lane 访问连续地址，便于 packed vector loads；网格多核 blocked rows 则让多数邻居留在同一 worker，减少跨核边界和一致性流量。不能脱离执行层次、数据布局与依赖关系判断哪种划分更好。

4. 若程序有 5% 串行部分，处理器数趋于无穷时最多加速多少？

    **面试回答：** 按 Amdahl 定律，固定问题规模且其余部分可理想并行时，$S(P)=1/(0.05+0.95/P)$，因此 $P\to\infty$ 时最多加速 $1/0.05=20$ 倍。实际通信、同步和负载不均会使结果更低。

5. 为什么把锁从格点循环内部移到局部归约之后能显著提速？

    **面试回答：** 把每格点的贡献先累加到线程私有变量，最后每个 worker 只加一次锁合并，可把每轮锁操作从格点数降到 worker 数。这样减少临界区次数、共享 cache line 迁移与锁竞争，让绝大多数计算并行执行；最终合并仍需正确同步。

6. 原求解器的三道 barrier 分别防止什么错误？

    **面试回答：** 第一道确保清零全部完成，避免迟到的清零抹掉早到线程的贡献；第二道确保所有局部和已合并，避免用不完整的 diff 判断收敛。第三道确保所有线程读完本轮结果，防止快线程下一轮清零覆盖慢线程还要读取的值。

7. 为什么复制三份轮转的 `diff` 能减少 barrier？它付出了什么代价？

    **面试回答：** 三槽分别保存上一轮待读结果、本轮累加值和下一轮待用的零，消除复用同一地址造成的读写冲突；每轮一道 barrier 可同时保证本轮归约完成和下一槽初始化完成。代价是额外存储和轮转管理，累加仍需锁或原子操作；严格 C/C++ 实现应由指定线程清零，避免并发普通写。


## 参考资料

- Stanford CS149 Fall 2023, *Lecture 4: Parallel Programming Basics*（视频与官方课件）
- Gene M. Amdahl, *Validity of the Single Processor Approach to Achieving Large-Scale Computing Capabilities*, 1967
