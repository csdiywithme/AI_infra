---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 3
lecture_date: 2023-10-03
area: systems
topics:
  - Memory Bandwidth
  - Arithmetic Intensity
  - ISPC
  - SPMD
aliases:
  - Stanford CS149 Lecture 03
video_url: https://www.youtube.com/watch?v=F4bVSyz_jxo
---

# Lecture 03：Multi-Core Architecture Part II + ISPC Programming Abstractions

> [!abstract] 本讲一句话
> 更多线程可以隐藏 memory latency，却不能突破 memory bandwidth；ISPC 则用 SPMD 编程抽象表达多个逻辑实例，再由编译器以 SIMD 实现，展示了“程序语义”和“硬件调度”必须分开理解。

## 来源与范围

- [课程视频](https://www.youtube.com/watch?v=F4bVSyz_jxo)，时长 1:16:19
- [官方 Slides](https://gfxcourses.stanford.edu/cs149/fall23content/media/multicore2-ispc/03_multicore2-ispc.pdf)
- 本讲覆盖：hardware multithreading 回顾、latency/throughput/bandwidth、pipelining、bandwidth-bound、arithmetic intensity、ISPC、SPMD、`programIndex`/`programCount`、`uniform`/varying、`foreach`、reduction
- 本讲不覆盖：ISPC tasks 的多核调度细节、系统化 Roofline 模型

## 视频索引

| 时间 | 视频内容 | 对应笔记 |
| --- | --- | --- |
| [00:05](https://www.youtube.com/watch?v=F4bVSyz_jxo&t=5s) | Hardware multithreading 回顾 | [1. 多线程到底解决什么](#1-多线程到底解决什么) |
| [10:07](https://www.youtube.com/watch?v=F4bVSyz_jxo&t=607s) | 3 compute + 12 stall 的利用率 | [1.2 利用率模型](#12-利用率模型) |
| [18:10](https://www.youtube.com/watch?v=F4bVSyz_jxo&t=1090s) | Multi-core × SIMD × threads 的组合 | [2. 三种吞吐技术的组合](#2-三种吞吐技术的组合) |
| [35:21](https://www.youtube.com/watch?v=F4bVSyz_jxo&t=2121s) | OS 与芯片分别负责哪层调度 | [3. 调度层次](#3-调度层次os-与硬件各做什么) |
| [39:12](https://www.youtube.com/watch?v=F4bVSyz_jxo&t=2352s) | 向量逐元素乘法 thought experiment | [4. Latency、Throughput 与 Bandwidth](#4-latencythroughput-与-bandwidth) |
| [47:57](https://www.youtube.com/watch?v=F4bVSyz_jxo&t=2877s) | 洗衣 pipeline 与瓶颈阶段 | [5. Pipelining](#5-pipelining提高吞吐不必降低延迟) |
| [53:36](https://www.youtube.com/watch?v=F4bVSyz_jxo&t=3216s) | 64-byte load 示例 | [6. Bandwidth-Bound Execution](#6-bandwidth-bound-execution) |
| [57:23](https://www.youtube.com/watch?v=F4bVSyz_jxo&t=3443s) | V100 所需带宽与实际带宽 | [6.2 向量乘法的算术强度](#62-向量乘法的算术强度) |
| [1:00:16](https://www.youtube.com/watch?v=F4bVSyz_jxo&t=3616s) | 为什么要提高计算/访存比 | [7. 如何突破带宽限制](#7-如何突破带宽限制) |
| [1:04:40](https://www.youtube.com/watch?v=F4bVSyz_jxo&t=3880s) | ISPC 与 SPMD | [9. ISPC 的 SPMD 抽象](#9-ispc-的-spmd-抽象) |
| [1:07:40](https://www.youtube.com/watch?v=F4bVSyz_jxo&t=4060s) | `programIndex` 与 `programCount` | [10. ISPC 的变量与执行模型](#10-ispc-的变量与执行模型) |
| [1:11:53](https://www.youtube.com/watch?v=F4bVSyz_jxo&t=4313s) | Interleaved vs blocked assignment | [11. 工作映射不是程序语义](#11-工作映射不是程序语义) |
| [1:15:04](https://www.youtube.com/watch?v=F4bVSyz_jxo&t=4504s) | `foreach` 提升抽象层次 | [12. foreach](#12-foreach把迭代集合交给编译器) |

## 1. 多线程到底解决什么

### 1.1 Latency hiding

Hardware multithreading 的目标不是让一次 memory request 更快，而是在 thread A 等待时执行 thread B，从而让 core 的 ALU 更少 idle。

它改善的是整体 throughput/utilization，而不是单个 thread 的 memory latency。

### 1.2 利用率模型

每个 thread 重复执行 $C$ cycles compute，然后等待 $L$ cycles：

$$
U_1=\frac{C}{C+L}
$$

理想交错且不存在其他瓶颈时，填满 execution unit 所需线程数近似：

$$
N\ge \left\lceil\frac{C+L}{C}\right\rceil
$$

但它有隐藏前提：

- threads 彼此独立并且 ready；
- context 数量足够；
- instruction throughput 足够；
- memory system 还有未使用的吞吐能力。

最后一条是本讲关键：若 memory 已经 100% 忙，再加线程只能增加排队，不能增加 bytes/s。

## 2. 三种吞吐技术的组合

假设芯片有：

- 16 cores；
- 每 core 8-wide SIMD；
- 每 core 4 hardware threads。

峰值同时执行的 ALU lanes 是：

$$
16\times 8=128
$$

为了给这些 lanes 提供足够的 latency-hiding work，可能需要：

$$
16\times 8\times 4=512
$$

个独立 data elements/tasks 在飞。

注意 512 不是 ALU 数量，而是为 128 个 lanes 准备的逻辑工作量。

## 3. 调度层次：OS 与硬件各做什么

现代系统至少有三层映射：

```text
application tasks / software threads
            ↓ runtime + OS
hardware execution contexts on cores
            ↓ core scheduler
execution units / SIMD lanes
```

- Runtime/OS：决定 software thread 映射到哪个 hardware context；
- Core：每周期从 ready instruction streams 中选择可执行指令；
- SIMD implementation：把一个 vector/SPMD group 映射到多个 lanes。

把这些层次混在一起，容易误以为“创建 512 threads 就有 512 个物理 ALU”。逻辑并行度与物理并行度不是一回事。

## 4. Latency、Throughput 与 Bandwidth

### 4.1 三个定义

| 指标 | 定义 | 单位示例 |
| --- | --- | --- |
| Latency | 单个操作从开始到完成的时间 | ns、cycles |
| Throughput | 系统单位时间完成的操作数 | requests/s、items/s |
| Bandwidth | 单位时间传输的数据量 | GB/s |

### 4.2 高速公路类比

从 San Francisco 到 Stanford 需要 0.5 小时，这是单辆车 latency。

- 提高车速：降低 latency，也可提高 throughput；
- 增加车道：不改变每辆车 latency，提高 throughput；
- 缩短车距形成 pipeline：不改变单车 latency，大幅提高道路 utilization 与 throughput。

因此 latency 与 throughput 可以解耦。

### 4.3 Memory bandwidth

Memory bandwidth 是 memory system 向 processor 提供数据的速率。即使一次访问要等待很久，只要内存能把许多请求 pipeline 起来，仍可得到较高总带宽。

反过来，低 latency 不保证高 bandwidth；大量并发请求可能在总传输速率上饱和。

## 5. Pipelining：提高吞吐不必降低延迟

一次洗衣：

```text
wash 45 min → dry 60 min → fold 15 min
```

单次 latency：

$$
45+60+15=120\text{ min}
$$

多批衣服 pipeline 后，steady-state throughput 由最慢阶段决定：

$$
\text{throughput}=\frac{1}{\max(45,60,15)}=1\text{ load/hour}
$$

单批仍需 2 小时，但系统每小时可完成一批。

同理，四级 instruction pipeline 可能有 4-cycle latency，却在流水线填满后每 cycle 完成一条指令。课程说“一周期一个 multiply”通常指 throughput，不是 multiply 的 latency。

## 6. Bandwidth-Bound Execution

### 6.1 64-byte load 示例

多个 threads 重复：

```text
X = load 64 bytes
Y = X + X
Z = X + Y
```

假设 core：

- 每 cycle 做 1 个 math operation；
- math 与 load 可 co-issue；
- memory 每 cycle 传 8 bytes；
- 有足够 threads 隐藏 latency。

一个 64-byte load 占用 memory transfer：

$$
\frac{64\text{ bytes}}{8\text{ bytes/cycle}}=8\text{ cycles}
$$

这 64 bytes 只支持 2 个 math instructions，所以 steady state 中 ALU 大量空闲。增加 outstanding loads 或 threads 可以覆盖起始 latency，却不能让 memory 每 cycle 传超过 8 bytes。

> [!important] Latency bound 与 bandwidth bound 的区别
> Latency bound 时，系统还有空闲传输能力，只是请求不够并发；bandwidth bound 时，传输资源持续满载，再增加并发只会排队。

### 6.2 向量乘法的算术强度

对 FP32：

$$
C[i]=A[i]\times B[i]
$$

每元素大约：

- 读 `A[i]`：4 bytes；
- 读 `B[i]`：4 bytes；
- 写 `C[i]`：4 bytes；
- 1 FLOP（按课程的 multiply 计数方式）。

Arithmetic intensity：

$$
I=\frac{1\text{ FLOP}}{12\text{ bytes}}\approx0.083\text{ FLOP/byte}
$$

V100 有约 5120 FP32 multipliers、1.6 GHz clock。若每 cycle 全部做乘法，所需数据带宽约为：

$$
5120\times1.6\times10^9\times12
\approx98\text{ TB/s}
$$

而课程示例中的 HBM bandwidth 约 900 GB/s，差两个数量级。因此 kernel 即使高度并行，也只能达到极低的 ALU efficiency。

### 6.3 简化性能上界

带宽给出的性能上界：

$$
P_{bandwidth}=B_{memory}\times I
$$

实际性能近似受以下较小者限制：

$$
P\le\min(P_{peak},\ B_{memory}\times I)
$$

这就是 Roofline 模型的核心关系。

## 7. 如何突破带宽限制

### 7.1 少从远端 memory 取数据

- Temporal reuse：已加载的数据在寄存器/cache/shared memory 中重复使用；
- Spatial locality：连续、合并访问完整利用 cache line 或 memory transaction；
- Inter-thread cooperation：一次加载后由多个 threads 使用；
- Fusion：避免中间结果写回 HBM 后再次读入。

### 7.2 提高 Arithmetic Intensity

让每个 byte 支持更多有用计算。有时重新计算比 store + reload 更便宜：

> 在现代 throughput processor 上，math 可能近似“免费”，数据移动才是关键资源。

这不是说 FLOPs 没成本，而是说在 bandwidth-bound 区域内增加未占满 ALU 的运算，可能不增加 wall time。

### 7.3 更高带宽的 memory

GPU 使用靠近 processor 的 HBM，通过更宽接口提供高带宽。但硬件带宽增长仍通常落后于算力增长，因此软件复用不可替代。

## 8. Abstraction vs. Implementation

本讲后半段的总主题：

### Abstraction / Semantics

- 程序操作“意味着什么”？
- 哪些结果必须与顺序语义一致？
- 程序员声明了哪些独立性或同步约束？

### Implementation / Scheduling

- 哪个 thread、core 或 SIMD lane 执行某个操作？
- 采用 blocked、interleaved 还是 dynamic assignment？
- 编译器、runtime 和硬件如何实现抽象？

同一种抽象可以有多个合法实现；不能从当前实现细节反推语言语义必然如此。

## 9. ISPC 的 SPMD 抽象

ISPC 是 Intel SPMD Program Compiler。调用一个 ISPC function 时，程序员的心智模型是：

```text
sequential C++
    ↓ call ISPC function
spawn a gang of program instances
instance 0, 1, 2, ... execute the same program on different data
    ↓ all complete
resume sequential C++
```

SPMD = Single Program, Multiple Data：

- 所有 instances 运行同一份程序；
- 每个 instance 有自己的 logical control flow 和局部变量；
- 每个 instance 根据自己的 index 处理不同数据；
- gang 完成后函数返回。

## 10. ISPC 的变量与执行模型

### 10.1 `programCount` 与 `programIndex`

- `programCount`：gang 中并行 program instances 的数量，对所有 instances 相同；
- `programIndex`：当前 instance 的编号，各 instance 不同。

手工 interleaved mapping：

```c
for (uniform int i = 0; i < N; i += programCount) {
    int idx = i + programIndex;
    result[idx] = work(x[idx]);
}
```

### 10.2 `uniform` 与 varying

- `uniform T`：所有 instances 共享同一个逻辑值；
- 未标记的普通 ISPC 值通常是 varying：每个 instance 可不同。

典型分类：

```text
uniform: N, terms, pointer base, loop counter shared by the gang
varying: idx, x[idx], partial result, per-instance branch condition
```

### 10.3 SPMD 抽象，SIMD 实现

ISPC compiler 通常：

- 将一个 gang 映射到硬件 SIMD width 或其小倍数；
- 生成 AVX2/AVX-512/NEON vector instructions；
- 用 vector mask 实现不同 instance 的条件控制流；
- 把 varying value 放在 vector register 中；
- 把 uniform value 作为 scalar/broadcast value 处理。

这正是课程强调的区别：程序员写的是多个逻辑 instruction streams 的 SPMD 模型，编译器生成的是单条 vector instruction 的 SIMD implementation。

## 11. 工作映射不是程序语义

### 11.1 Interleaved assignment

8 个 instances 处理：

```text
instance 0: 0, 8, 16, ...
instance 1: 1, 9, 17, ...
...
```

优点：同一轮访问连续地址，可用 packed vector load/store；通常利于 SIMD memory coalescing。

### 11.2 Blocked assignment

```text
instance 0: [0, count)
instance 1: [count, 2*count)
...
```

每个 instance 获得连续块，但同一时刻各 instances 的地址可能相距较远，vector gather/scatter 成本可能更高。

### 11.3 Dynamic assignment

工作成本不均时可以动态领取迭代，改善 balance，但增加调度/atomic overhead，也可能破坏地址连续性。

不存在永远最好的 mapping；它取决于：

- per-iteration cost；
- memory layout；
- SIMD load/store 能力；
- locality 与 balance 的折中。

## 12. `foreach`：把迭代集合交给编译器

```c
foreach (i = 0 ... N) {
    result[i] = work(x[i]);
}
```

其语义是：整个 gang 必须共同完成这些独立迭代。程序员不指定每个 instance 具体执行哪些 `i`。

因此编译器可以合法选择：

- interleaved；
- blocked；
- dynamic；
- 其他满足语义的 mapping。

抽象层次提高后，程序员更专注于“对每个元素独立做什么”，实现则可以随目标 ISA 和 workload 改变。

## 13. 跨 Instance 通信与 Reduction

错误的 sum 可能出现两种问题：

1. 每个 instance 有私有 `sum`，函数最后无法直接得到全局和；
2. 所有 instances 写同一个 `uniform sum`，产生 data race。

正确思路：

```c
float partial = 0.0f;
foreach (i = 0 ... N) {
    partial += x[i];
}
uniform float sum = reduce_add(partial);
```

先在每个 instance 内做 private accumulation，减少通信；再用 `reduce_add()` 汇总。ISPC 还提供 `reduce_min`、`broadcast`、`rotate/shift` 等 cross-instance primitives。

## 14. ISPC 的多核边界

一个 gang 通常由一个 CPU core 上的 SIMD instructions 实现。只调用一次普通 ISPC function 并不自动使用整颗多核 CPU。

ISPC 另有 task abstraction 用于跨 cores；Assignment 1 会把：

- task-level/multi-core parallelism；
- gang-level/SIMD parallelism

组合起来。

## 15. AI Infra 视角

### Memory Bandwidth

LLM decode 常反复读取大模型权重，batch 较小时 arithmetic intensity 低，容易 bandwidth-bound。增加请求可提高权重复用，直到 compute 或 memory bandwidth 饱和。

### Kernel Fusion

把 RMSNorm、projection、activation 等操作融合，可以避免中间 tensor materialization：

```text
unfused: HBM read → op A → HBM write → HBM read → op B → HBM write
fused:   HBM read → op A → on-chip reuse → op B → HBM write
```

FlashAttention 也是同一思想：增加 on-chip 计算和 tile 内复用，减少 HBM traffic。

### SPMD/SIMT

CUDA kernel 同样采用 SPMD 心智模型：许多 logical threads 运行同一 kernel；GPU 再把 warp 映射到底层 SIMD lanes。理解 ISPC 的 abstraction/implementation 分离，有助于避免把 CUDA thread 当成独立物理 core。

### Distributed Parallelism

Tensor Parallel 的抽象是多个 ranks 协同计算；实现可选择 ring/tree collectives、不同 chunking 与 overlap。语义与调度同样应分开分析。

## 本讲结论

1. Multithreading 隐藏 latency，但不能突破已经饱和的 bandwidth。
2. Latency 描述单个操作，throughput/bandwidth 描述单位时间处理能力；pipeline 可提高后者而不降低前者。
3. 低 arithmetic intensity kernel 即使高度并行，也可能只有极低 ALU efficiency。
4. 高性能程序应减少远端 memory traffic、复用数据并提高每 byte 的有用计算。
5. ISPC 提供 SPMD 抽象，并通常由 SIMD instructions 实现。
6. `foreach` 描述要完成的迭代集合，不固定具体 mapping；这是 abstraction 与 implementation 的典型分离。

## 自测问题

1. 为什么增加 hardware threads 能解决 latency-bound，却不能解决 bandwidth-bound？

    **面试回答：** Latency-bound 时往往是独立请求不足，更多 hardware threads 能在某个线程等待时提供计算和访存，把空闲时段填满。Bandwidth-bound 时传输通道已持续饱和，增加线程只会增加排队；此时需要减少传输量、提高数据复用或增加有效带宽。

2. 一个 pipeline 的单任务 latency 为 20 cycles，steady-state 每 cycle 完成一个任务，二者矛盾吗？

    **面试回答：** 不矛盾：20 cycles 是单个任务从进入到完成的延迟，每 cycle 一个是流水线填满后的吞吐率。若任务独立且各级可以重叠执行，就能同时容纳约 20 个在途任务；首个结果仍要等 20 cycles，之后结果可连续产生。

3. FP32 vector add `C=A+B` 的 arithmetic intensity 大约是多少？忽略 cache 与 write allocate。

    **面试回答：** 每个 FP32 元素读取 A、B 各 4 bytes，再写 C 4 bytes，总计 12 bytes，只做 1 次加法。因此算术强度约为 $I=1/12\approx0.083$ FLOP/byte；这是忽略 cache 复用和 write allocate 的流量口径。

4. ISPC program instance、SIMD lane 和 CPU core 是什么关系？

    **面试回答：** Program instance 是 ISPC 的逻辑 SPMD 执行实例，SIMD lane 是向量执行的数据通道，CPU core 是运行指令流的硬件核心。编译器通常将一个 gang 的实例映射为单核上的 SIMD 操作，实例数也可大于一次物理向量宽度；利用多个 CPU 核还需要任务或线程机制。

5. 为什么 interleaved mapping 常比 blocked mapping 更适合 packed vector loads？

    **面试回答：** Interleaved 分配下，同一步各 instance 访问相邻元素，例如第 k 轮访问 $kW+[0,W)$，容易生成一次 packed load/store。Blocked 分配虽然每个 instance 自己顺序访问，但同一时刻各 lane 的地址相隔一个块，往往需要成本更高的 gather/scatter。

6. 为什么用一个共享 `uniform sum` 直接累加会错？

    **面试回答：** `uniform sum` 只有一个逻辑值，不能隐式代表所有实例的归约结果；直接把 varying 贡献赋给它通常无法通过类型检查，绕过类型限制让实例写同一地址又会产生冲突。正确做法是各实例维护 varying partial sum，最后用 `reduce_add` 显式合并。 参见 [ISPC 类型规则](https://ispc.github.io/ispc.html#uniform-and-varying-qualifiers)。

7. `foreach` 允许编译器改变工作分配，却为什么不能改变程序结果？

    **面试回答：** `foreach` 的契约是整个 gang 完成给定迭代域，程序不能依赖某次迭代落在哪个 instance 或按什么次序执行。只要迭代独立、没有未同步的共享写入，合法分配就应满足同一计算语义；跨迭代依赖或依赖分配顺序的代码违反了这个前提。


## 参考资料

- [ISPC 官方文档](https://ispc.github.io/ispc.html)
- [[Stanford CS149 - Reading - The Story of ISPC|The Story of ISPC：详细阅读笔记]]（[原文系列](https://pharr.org/matt/blog/2018/04/30/ispc-all.html)），Matt Pharr
- [Roofline: An Insightful Visual Performance Model](https://doi.org/10.1145/1498765.1498785)
- [NVIDIA V100 Architecture Whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
