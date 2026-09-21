---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 2
lecture_date: 2023-09-28
area: systems
topics:
  - Multi-Core
  - SIMD
  - Hardware Multithreading
aliases:
  - Stanford CS149 Lecture 02
video_url: https://www.youtube.com/watch?v=CKmNpAO5rS4
---

# Lecture 02：A Modern Multi-Core Processor

> [!abstract] 本讲一句话
> 现代吞吐处理器把 multi-core、SIMD、superscalar 和 hardware multithreading 叠加使用：前三者提供不同维度的并行执行，硬件多线程则在某个线程停顿时用其他工作填补空槽。

## 来源与范围

- [课程视频](https://www.youtube.com/watch?v=CKmNpAO5rS4)，时长 1:16:14
- [官方 Slides](https://gfxcourses.stanford.edu/cs149/fall23content/media/multicore/02_basicarch_xX3ssOi.pdf)
- 本讲覆盖：multi-core、SIMD、divergence、coherent control flow、cache/prefetch、hardware multithreading、latency hiding、throughput computing
- 本讲不覆盖：memory bandwidth 的定量瓶颈与 ISPC 细节（下一讲）

> [!note] 时间索引说明
> 视频没有发布者章节，下表按主题转折提供近似切入点，适合回看定位；概念与数字已和官方 slides 核对。

## 视频索引

| 时间 | 视频内容 | 对应笔记 |
| --- | --- | --- |
| [00:00](https://www.youtube.com/watch?v=CKmNpAO5rS4&t=0s) | 回顾 instruction stream、superscalar 与 cache | [1. 四个容易混淆的概念](#1-四个容易混淆的概念) |
| [09:00](https://www.youtube.com/watch?v=CKmNpAO5rS4&t=540s) | `sinx` 数据并行示例 | [2. 贯穿本讲的 sinx](#2-贯穿本讲的-sinx) |
| [15:00](https://www.youtube.com/watch?v=CKmNpAO5rS4&t=900s) | 从单核走向 multi-core | [3. Multi-Core](#3-multi-core多个独立指令流) |
| [23:30](https://www.youtube.com/watch?v=CKmNpAO5rS4&t=1410s) | SIMD/vector execution | [4. SIMD](#4-simd一个指令控制多个-lane) |
| [31:30](https://www.youtube.com/watch?v=CKmNpAO5rS4&t=1890s) | 条件分支、mask 与 divergence | [5. SIMD 的关键限制](#5-simd-的关键限制coherent-execution) |
| [37:00](https://www.youtube.com/watch?v=CKmNpAO5rS4&t=2220s) | Superscalar、SIMD、multi-core 对比 | [6. 三类执行并行的对照](#6-三类执行并行的对照) |
| [43:00](https://www.youtube.com/watch?v=CKmNpAO5rS4&t=2580s) | Cache 与 prefetch | [7. 访存停顿](#7-访存停顿cache-与-prefetch) |
| [48:30](https://www.youtube.com/watch?v=CKmNpAO5rS4&t=2910s) | Hardware multithreading | [8. Hardware Multithreading](#8-hardware-multithreading用别的线程隐藏停顿) |
| [56:00](https://www.youtube.com/watch?v=CKmNpAO5rS4&t=3360s) | 吞吐/延迟折中与利用率练习 | [9. 多少线程才能填满 core](#9-多少线程才能填满-core) |
| [66:00](https://www.youtube.com/watch?v=CKmNpAO5rS4&t=3960s) | Skylake 与 V100 实例 | [10. 组合起来看真实处理器](#10-组合起来看真实处理器) |
| [73:00](https://www.youtube.com/watch?v=CKmNpAO5rS4&t=4380s) | 引出 latency 与 bandwidth | [11. 课程留下的问题](#11-课程留下的问题) |

## 1. 四个容易混淆的概念

| 机制                      | 并行对象             | 指令流数量 | 谁发现/表达并行性  | 主要目的               |
| ----------------------- | ---------------- | ----: | ---------- | ------------------ |
| Superscalar             | 同一指令流中的独立指令      |     1 | 硬件动态发现 ILP | 提高单线程 IPC          |
| SIMD                    | 同一指令作用于多个数据元素    |     1 | 编译器/程序语义   | 摊薄 fetch/decode 成本 |
| Multi-core              | 多个独立线程/任务        |    多个 | 软件创建并行工作   | 扩展总计算资源            |
| Hardware multithreading | 同一 core 上多个线程的指令 |    多个 | 硬件在线程间选择   | 隐藏 stall、提高利用率     |

这些机制可以同时存在。真实 CPU core 往往既 superscalar、又支持 SIMD、又有多个 hardware threads；芯片再由多个这样的 core 组成。

## 2. 贯穿本讲的 `sinx`

课程用 Taylor expansion 对数组每个元素近似计算：

$$
\sin(x)=x-\frac{x^3}{3!}+\frac{x^5}{5!}-\frac{x^7}{7!}+\cdots
$$

```cpp
void sinx(int N, int terms, float* x, float* y) {
    for (int i = 0; i < N; i++) {
        float value = x[i];
        float numer = x[i] * x[i] * x[i];
        int denom = 6;
        int sign = -1;
        for (int j = 1; j <= terms; j++) {
            value += sign * numer / denom;
            numer *= x[i] * x[i];
            denom *= (2*j + 2) * (2*j + 3);
            sign *= -1;
        }
        y[i] = value;
    }
}
```

不同 `i` 的循环迭代彼此独立，因此天然具有 data parallelism。但普通 C 的 `for` 只描述顺序语义；要用多核或 SIMD，程序员/编译器必须把“迭代独立”传达给执行系统。

## 3. Multi-Core：多个独立指令流

### 3.1 动机

在 pre-multicore 时代，大量晶体管用于让一个 instruction stream 更快：更宽的 out-of-order window、更复杂的投机、更大的 cache。

当单线程收益递减后，更合理的方向是复制 core：

```text
thread 0 → core 0 → x[0 ... k)
thread 1 → core 1 → x[k ... 2k)
...
```

每个 core 有独立 fetch/decode、execution context 与 ALU，因此能运行完全不同的 instruction stream 和控制流。

### 3.2 软件必须表达并行性

普通 `sinx()` 只生成一个 instruction stream，因此只使用一个 core。要利用多核，软件可以：

- 创建多个 C++/pthread/OpenMP threads；
- 使用 task runtime；
- 使用 data-parallel language construct；
- 由框架把 tensor operation 划分给多个 worker。

> [!important] 多核不是自动加速器
> 硬件提供多个 core，不代表单线程程序会自动拆成多个正确、均衡且低通信的线程。

## 4. SIMD：一个指令控制多个 lane

### 4.1 基本结构

SIMD（Single Instruction, Multiple Data）让一个 fetch/decode/control unit 控制多个 ALU lanes：

```text
vector instruction
       ↓
lane 0 lane 1 lane 2 ... lane 7
 x[0]   x[1]   x[2]       x[7]
```

相比复制完整 core，SIMD 复用控制逻辑，把更多晶体管和功耗用于算术单元。

### 4.2 显式 vectorization

以 AVX2 为例，256-bit vector 可以同时处理：

- 8 个 FP32；
- 4 个 FP64。

程序可以使用 intrinsics 显式写 vector load/mul/add，也可以由编译器根据循环独立性自动 vectorize。

### 4.3 CPU SIMD 与 GPU SIMT

- CPU 常见表现：binary 中真的包含 vector instructions；
- GPU 常见表现：程序看起来是多个 scalar threads，硬件把具有相同 PC 的 threads 成组，在 SIMD lanes 上共同执行；
- 两者底层都要求多个 lane 在同一时刻执行同一种操作。

## 5. SIMD 的关键限制：Coherent Execution

考虑每个元素根据正负走不同分支：

```cpp
if (x[i] > 0) {
    // path A: 3 operations
} else {
    // path B: 2 operations
}
```

若一个 8-wide SIMD group 中有些 lane 走 A、有些走 B，硬件通常：

1. 执行 A，mask 掉选择 B 的 lanes；
2. 执行 B，mask 掉选择 A 的 lanes；
3. 在分支汇合后恢复所有 lanes。

因此分支区域内不是所有 ALU 都做有用工作。极端情况下，每一步只有 1/8 lanes active，性能只有峰值的一小部分。

> [!question] 课堂问题
> 这里课上老师提了一个问题，怎么设计 condition 让我们的利用率掉到接近 1/8？我记得好像是通过分配 condition 以及每个 path 上的 operation 数目。

> [!answer] 让一个 lane 执行占主导的长路径
> 关键是让 8 个 lanes 中只有 1 个进入一条很长的路径，其余 7 个进入极短路径或不做任何工作。执行长路径时，只有这 1 个 lane active，其余 7 个都被 mask，因此该阶段的利用率是 $1/8$。只让 condition 按 1:7 分流还不够；长短路径的操作数必须高度不均衡。

设 SIMD 宽度为 $W=8$：

- $k$ 个 lanes 进入 true path，每个执行 $A$ 个操作；
- $W-k$ 个 lanes 进入 false path，每个执行 $B$ 个操作；
- 两条分歧路径被依次执行。

忽略分支和汇合开销时，分支区域的平均 lane utilization 为：

$$
U=\frac{kA+(W-k)B}{W(A+B)}
$$

令 $k=1$ 且 $A\gg B$：

$$
U\approx\frac{A}{8A}=\frac18
$$

例如，假设一个 8-wide SIMD group 对应连续的 8 个 `i`，其中只有一个元素执行很长的计算：

```cpp
forall (int i = 0; i < N; i++) {
    if (i % 8 == 0) {             // 每组只有 1 个 lane 为 true
        for (int j = 0; j < 1000; j++)
            x[i] = expensive(x[i]);
    } else {
        x[i] += 1;                // 极短路径
    }
}
```

若取 $A=1000$、$B=1$，则：

$$
U=\frac{1\times1000+7\times1}{8\times(1000+1)}
\approx12.57\%
$$

非常接近 $1/8=12.5\%$。如果完全省略 `else`，那么在条件体内部正好只有 1 个 lane 工作，利用率就是 $1/8$。

> [!important] 一个容易忽略的反例
> 如果 true 和 false 两条路径一样长，即 $A=B$，即使 condition 将 lanes 分成 1:7，平均利用率仍是
>
> $$
> U=\frac{1A+7A}{8(A+A)}=\frac12
> $$
>
> 而不是 $1/8$。所以最坏情况必须同时具备：少量 active lanes + 由它们执行的长路径占据绝大部分运行时间。

### Coherent control flow

若许多数据元素遵循相同 instruction sequence，就称执行具有 coherence。它是高效 SIMD 的必要条件，却不是多核并行的必要条件：不同 core 可以独立 fetch/decode 不同路径。

> [!tip] GPU 上的数据布局也影响控制流
> 把行为相近的数据分组，可使同一个 warp/wavefront 内的分支更一致，减少 divergence。

## 6. 三类执行并行的对照

### Superscalar

```text
同一个 thread：instruction A 与 B 独立
→ 同一周期派发到不同 execution units
```

- 不要求相同操作；
- 并行度较小；
- 由硬件动态发现。

### SIMD

```text
同一个 vector instruction
→ 多个 lane 对不同数据执行相同操作
```

- 并行度更宽；
- 控制开销低；
- 要求 coherent execution。

### Multi-core

```text
多个 threads/tasks
→ 多个 core 各自执行不同 instruction streams
```

- 控制流最自由；
- 复制的硬件和线程管理成本更高；
- 软件负责提供足够并行工作。

## 7. 访存停顿：Cache 与 Prefetch

### 7.1 Cache 缩短 latency

Cache 保存近期数据。命中较近层级可以显著缩短 load stall。

### 7.2 Hardware prefetching

Prefetcher 根据访问模式预测未来地址，提前把数据搬入 cache。它对连续、固定 stride 的访问尤其有效。

但它不总能奏效：

```cpp
int x = some_function();
int y = A[x];  // 地址依赖运行时结果，不容易提前预测
```

面对不可预测的 cache miss，处理器需要另一种办法让 ALU 不闲着：在等待期间执行其他线程。

## 8. Hardware Multithreading：用别的线程隐藏停顿

### 8.1 核心思想

一个 core 保存多个 execution contexts（PC、registers 等）。当 thread 0 因 load stall 时，core 可以从 thread 1 取指执行：

```text
time →
thread 0: compute |----- memory wait -----| compute
thread 1:         compute |----- wait -----|
thread 2:                 compute |----- wait -----|
core ALU:   useful useful useful useful ...
```

硬件多线程没有增加 ALU 数量，而是提高已有 ALU 的利用率。

### 8.2 Interleaved 与 simultaneous multithreading

- Interleaved multithreading：某个周期从一个 thread 选择指令；
- SMT：superscalar core 同一周期可从多个 threads 选择独立指令；
- Intel Hyper-Threading 是 SMT 的典型实现。

### 8.3 代价

- 每个 thread 需要 execution context storage；
- threads 争用 execution units、cache 和 bandwidth；
- 单个 thread 的 latency 可能增加；
- context 数量越多，每个 thread 可用的 register/cache 工作集可能越小。

Throughput-oriented system 愿意牺牲单任务 latency，以提高大量任务的总完成率。

## 9. 多少线程才能填满 Core

课程练习假设每个 thread：

- 连续做 3 cycles arithmetic；
- 随后等待一个 12-cycle memory operation；
- core 每周期只能执行一个 thread 的 arithmetic。

单线程利用率：

$$
U_1=\frac{3}{3+12}=20\%
$$

为了在某个 thread 等待时始终有其他 thread 提供计算，需要 5 个 threads 才能覆盖整个 15-cycle 周期：

$$
N_{threads}\ge \left\lceil\frac{C+L}{C}\right\rceil
=\left\lceil\frac{3+12}{3}\right\rceil=5
$$

其中 $C$ 是每次 stall 前的 compute cycles，$L$ 是 stall latency。

若每次 memory access 前可做 6 cycles arithmetic：

$$
N_{threads}\ge \left\lceil\frac{6+12}{6}\right\rceil=3
$$

这说明：

- 每次访存之间的计算越多，需要的 latency-hiding threads 越少；
- cache 降低有效 $L$，也减少所需线程数；
- 超过 100% utilization 后继续增加 threads 不会增加同一 execution unit 的吞吐，反而可能争用资源。

## 10. 组合起来看真实处理器

### Intel Skylake/Kaby Lake Core

一个 core 同时具有：

- 多路 fetch/decode 与 superscalar issue；
- scalar ALUs；
- 多个 8-wide AVX2 vector units；
- 2-way hardware multithreading；
- L1/L2 cache，并与其他 core 共享更远层级。

### NVIDIA V100 SM

V100 用极端吞吐设计：

- 一个 GPU 含 80 个 SM；
- 每个 SM 有大量 SIMD execution lanes；
- 每个 SM 保存许多 warp contexts；
- 满足 latency hiding 时，需要同时存在成千上万的独立 data elements。

> [!important] 峰值算力的隐藏前提
> “有 5120 个 ALU”并不等于任意程序都能使用它们。程序还需要足够的并行任务、coherent control flow、足够数据供应和可承受的 bandwidth。

## 11. 课程留下的问题

向量逐元素乘法高度并行、易 vectorize、几乎无分支，看起来非常适合 GPU：

$$
C[i]=A[i]\times B[i]
$$

但每个元素需要：

- load `A[i]`；
- load `B[i]`；
- 1 次 multiply；
- store `C[i]`。

它是否能真正使用 GPU 的峰值 ALU？答案取决于内存每秒能提供多少数据，而不仅是有多少并行元素。这引出下一讲的 latency、throughput 与 bandwidth。

## 12. AI Infra 视角

### Shape 与 Mapping

Tensor shape 决定可并行的元素数，kernel launch/grid 则把逻辑元素映射到 blocks、warps 和 lanes。

### SIMD/SIMT

- Tensor Core instruction 可以看作更专用、更高维的数据并行执行；
- padding、ragged sequence 与条件分支会造成无效 lane 工作；
- grouping/bucketing 可以提升 control-flow coherence。

### Latency Hiding

- GPU occupancy 依赖 resident warps；
- warp 等待 memory 时，scheduler 选择其他 ready warp；
- 但若所有 warps 同时受 bandwidth 限制，再多 warp 也不能增加 DRAM 吞吐。

### LLM Decode

Decode 中每个 request 有串行 token 依赖。Continuous batching 通过混合多个 requests，为 GPU 提供更多独立工作，本质上类似用其他线程/任务隐藏单个 request 的等待与低利用率。

## 本讲结论

1. Superscalar、SIMD、multi-core 与 hardware multithreading 是不同机制，可以叠加。
2. SIMD 通过共享控制降低成本，但要求执行路径一致；divergence 会浪费 lanes。
3. Multi-core 运行多个独立 instruction streams，软件必须暴露线程级并行。
4. Hardware multithreading 不增加 ALU，而是用更多 execution contexts 隐藏 stall。
5. 足够的并行工作只是必要条件；数据供应可能成为下一层瓶颈。

## 自测问题

1. 为什么“8 核 CPU”和“8-wide SIMD”都能一次处理 8 项数据，却不是同一种并行性？

    **面试回答：** 8 核可以运行 8 条独立指令流，各核能走不同分支；8-wide SIMD 则是一条向量指令驱动 8 个 lane 对不同数据执行相同操作。前者主要利用线程或任务并行，后者利用数据并行，遇到分支不一致时 SIMD 通常需要掩码分批执行。

2. 什么样的 `if` 会导致 8-wide SIMD 接近最差利用率？

    **面试回答：** 让每组 8 个 lane 中只有 1 个执行很长的分支，其余 7 个执行极短路径或不工作，长路径期间利用率就接近 $1/8$。若两条分支一样长，即使按 1:7 分流，分支区域平均利用率也约为 $1/2$，不能仅凭分流比例判断最坏情况。

3. Hardware thread 和 OS/software thread 分别是什么？

    **面试回答：** Hardware thread 是核心内驻留的一套执行上下文，如 PC 和寄存器状态，多个上下文共享执行资源。OS/software thread 是软件调度实体，拥有栈和软件执行状态，由 OS 映射到硬件上下文；硬件选择已驻留线程通常不需要完整的 OS 保存与恢复过程。

4. 3 cycles compute + 12 cycles stall 为什么需要 5 个 threads，而不是 4 个？

    **面试回答：** 每个线程的周期为 $3+12=15$ cycles，计算占比为 $3/15=20\%$，所以至少需要 $\lceil15/3\rceil=5$ 个独立线程。只有 4 个时，第一个线程等待期间其他 3 个只能提供 9 cycles 的计算，还缺 3 cycles；该结论假设切换无额外开销且带宽未饱和。

5. 为什么 occupancy 已经很高时，GPU kernel 仍可能很慢？

    **面试回答：** Occupancy 衡量驻留 warp 数，并不表示这些 warp 都 ready，也不等于 ALU 利用率。Kernel 仍可能受 DRAM 带宽、长依赖链、分支发散或访存不合并限制；足够隐藏延迟后再增加驻留线程，通常无法突破真正的吞吐瓶颈。


## 参考资料

- [ISPC Programmer's Guide](https://ispc.github.io/ispc.html)
- [Intel AVX2 Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [NVIDIA V100 Tensor Core GPU Architecture](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
