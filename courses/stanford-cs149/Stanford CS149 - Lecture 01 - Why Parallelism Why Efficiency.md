---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 1
lecture_date: 2023-09-26
area: systems
topics:
  - Parallel Computing
  - Processor Architecture
  - Memory Hierarchy
aliases:
  - Stanford CS149 Lecture 01
video_url: https://www.youtube.com/watch?v=V1tINV2-9p4
---

# Lecture 01：Why Parallelism? Why Efficiency?

> [!abstract] 本讲一句话
> 单线程性能已经不能再依靠频率和自动挖掘 ILP 持续增长；现代性能来自更多处理单元与专用硬件，而真正的难点是让它们在数据移动、负载均衡和功耗约束下保持高效。

## 来源与范围

- [课程视频](https://www.youtube.com/watch?v=V1tINV2-9p4)，时长 1:12:21
- [官方 Slides](https://gfxcourses.stanford.edu/cs149/fall23content/media/whyparallelism/01_whyparallelism_huXfOJ4.pdf)
- 本讲覆盖：speedup、parallel efficiency、ILP、superscalar、power wall、multi-core、specialization、memory latency 与 cache
- 本讲不深入：具体并行 API、SIMD 编程、cache 结构与一致性

## 视频索引

| 时间 | 视频内容 | 对应笔记 |
| --- | --- | --- |
| [04:56](https://www.youtube.com/watch?v=V1tINV2-9p4&t=296s) | 课程目标：parallelism 与 efficiency | [1. 两个目标](#1-两个目标performance-与-efficiency) |
| [07:49](https://www.youtube.com/watch?v=V1tINV2-9p4&t=469s) | 课堂并行求和实验 | [2. 并行程序的三个基本问题](#2-并行程序的三个基本问题) |
| [15:37](https://www.youtube.com/watch?v=V1tINV2-9p4&t=937s) | 负载不均衡限制加速 | [2.2 工作分配与负载均衡](#22-工作分配与负载均衡) |
| [28:50](https://www.youtube.com/watch?v=V1tINV2-9p4&t=1730s) | 通信和数据移动的代价 | [2.3 通信与同步](#23-通信与同步) |
| [31:19](https://www.youtube.com/watch?v=V1tINV2-9p4&t=1879s) | Fast 不等于 efficient | [1.2 Efficiency](#12-efficiency) |
| [42:33](https://www.youtube.com/watch?v=V1tINV2-9p4&t=2553s) | 单线程免费加速的终结 | [3. 为什么必须转向并行](#3-为什么必须转向并行) |
| [44:38](https://www.youtube.com/watch?v=V1tINV2-9p4&t=2678s) | 程序是处理器指令序列 | [4. 最小处理器心智模型](#4-最小处理器心智模型) |
| [52:37](https://www.youtube.com/watch?v=V1tINV2-9p4&t=3157s) | 指令级并行与依赖 | [5. ILP 与 Superscalar](#5-ilp-与-superscalar) |
| [56:59](https://www.youtube.com/watch?v=V1tINV2-9p4&t=3419s) | ILP 的收益递减 | [5.3 为什么 ILP 会触顶](#53-为什么-ilp-会触顶) |
| [58:37](https://www.youtube.com/watch?v=V1tINV2-9p4&t=3517s) | Frequency scaling 与 power wall | [6. Power Wall](#6-power-wall) |
| [1:02:59](https://www.youtube.com/watch?v=V1tINV2-9p4&t=3779s) | Multi-core 与异构专用单元 | [7. 现代处理器的回答](#7-现代处理器的回答并行与专用化) |
| [1:05:03](https://www.youtube.com/watch?v=V1tINV2-9p4&t=3903s) | Memory、load、latency 与 cache | [8. 数据访问决定效率](#8-数据访问决定效率) |

## 1. 两个目标：Performance 与 Efficiency

### 1.1 Speedup

给定同一个问题，使用 $P$ 个处理器时的加速比为：

$$
S(P)=\frac{T(1)}{T(P)}
$$

- $T(1)$：一个处理器的执行时间；
- $T(P)$：$P$ 个处理器的执行时间；
- 理想线性加速：$S(P)=P$。

加速比只说明“比单处理器快多少”，不说明投入的硬件是否被充分利用。

### 1.2 Efficiency

常用的并行效率：

$$
E(P)=\frac{S(P)}{P}=\frac{T(1)}{P\,T(P)}
$$

例如，10 个处理器得到 2 倍加速：

$$
E(10)=\frac{2}{10}=20\%
$$

它可能已经比单核快，但 80% 的并行资源没有转化成线性收益。

> [!important] Fast != Efficient
> “更快”是结果；“高效”要求性能提升与占用的处理器、芯片面积、功耗和数据移动成本相匹配。

## 2. 并行程序的三个基本问题

### 2.1 Decomposition：把工作拆开

首先识别哪些工作可以安全并行。若两个操作存在数据依赖，就不能任意同时执行或重排。

并行性可能来自：

- 不同数据元素执行同一种运算；
- 相互独立的任务；
- 一个任务内部相互独立的指令；
- pipeline 中不同阶段同时处理不同输入。

### 2.2 工作分配与负载均衡

将 $N$ 项工作静态切给 $P$ 个处理器时，如果每项成本不同，就可能出现：

```text
processor 0: ███████████████
processor 1: █████
processor 2: ███
processor 3: ███████
```

总完成时间由最慢的处理器决定。其余处理器提前完成后处于 idle，形成 load imbalance。

常见改进：

- 更均匀的静态划分；
- 把工作切成更多小任务后动态领取；
- work stealing；
- 让同一阶段的任务成本更接近。

但任务越细，调度与同步开销通常越高，因此粒度存在折中。

### 2.3 通信与同步

课堂求和实验说明：即使加法本身很便宜，传递 partial sum、等待别人完成、最终汇总都可能支配时间。

并行时间可以粗略拆成：

$$
T_P \approx T_{useful\ work}+T_{communication}+T_{synchronization}+T_{imbalance}+T_{overhead}
$$

增加处理器通常缩短第一项，却可能放大后四项。可扩展性不是“能并行运行”，而是增加资源后仍能持续获得收益。

## 3. 为什么必须转向并行

过去单线程程序可以从两类硬件进步中获得“免费加速”：

1. 更高的 CPU clock frequency；
2. 处理器自动利用 instruction-level parallelism（ILP）。

大约从 2000 年代中期开始，这两条路径都显著放缓：

- 频率提升受到功耗和散热限制；
- 通用程序中可自动挖掘的 ILP 有限；
- transistor density 继续增加，但这些晶体管更适合被用于更多 core、SIMD lane、cache 或专用单元。

于是软件若想显著变快，必须显式暴露更大规模的并行性。

## 4. 最小处理器心智模型

从处理器角度，程序是一个指令序列。一个简化 core 包含：

| 组件 | 作用 |
| --- | --- |
| Fetch/Decode | 取得并解析下一条指令 |
| Execution Unit / ALU | 执行算术、逻辑或 load/store |
| Execution Context | 保存 PC、寄存器和线程状态 |
| Cache | 保存部分近期数据，降低访问延迟 |

最简单的处理器每个 clock 执行一条指令。但真实程序中，一些相邻指令彼此独立，理论上可以同时使用多个 execution unit。

## 5. ILP 与 Superscalar

### 5.1 指令级并行

考虑：

$$
a=x^2+y^2+z^2
$$

三个乘法相互独立，可以同时执行；随后的加法必须等待乘法结果。依赖图比源码顺序更能表达哪些操作可以并发。

### 5.2 Superscalar execution

Superscalar 处理器会在同一 instruction stream 中寻找独立指令，并分派给多个 execution units。现代 CPU 还会使用 out-of-order execution：

- 只要依赖关系允许，后面的指令可以先执行；
- 最终必须呈现与程序语义一致的结果；
- 硬件承担依赖检查、调度和投机的复杂度。

### 5.3 为什么 ILP 会触顶

- 数据依赖限制可同时执行的指令；
- branch 和不可预测控制流限制可见窗口；
- 更宽的 issue width 需要更复杂、耗电的控制逻辑；
- 常见程序在每周期约 3～4 条指令附近就出现明显收益递减。

因此，继续堆叠单线程控制复杂度的性价比下降。

## 6. Power Wall

动态功耗近似为：

$$
P_{dynamic}\propto C\,V^2f
$$

- $C$：等效电容负载；
- $V$：电压；
- $f$：时钟频率。

提高频率通常还需要提高电压，因此实际功耗增长可能比 $f$ 更陡。芯片还有 leakage 导致的静态功耗。

高功耗意味着高热量，而封装、散热、电池与数据中心供电都给出硬上限。结果是：

- 不能让所有晶体管一直以最高频率工作；
- 多个较简单的处理单元往往比一个极复杂、极高频的 core 更节能；
- 专用硬件通过减少通用控制开销提升 performance per watt。

## 7. 现代处理器的回答：并行与专用化

现代系统把晶体管投入到不同形式的执行资源：

- Multi-core CPU：多个独立 instruction streams；
- GPU：大量 core 与宽 SIMD/SIMT 执行；
- Mobile SoC：大小核、GPU、ISP、NPU、media engine；
- TPU/AI accelerator：为矩阵乘法和张量数据流专门设计。

这形成异构系统。软件不仅要“并行”，还要把合适的工作映射到合适的处理单元。

## 8. 数据访问决定效率

### 8.1 Memory latency

Memory access latency 是从处理器发出请求到数据可用所需的时间。若下一条指令依赖该数据，core 会 stall。

课程给出的 Kaby Lake 示例：

| 数据位置 | 近似 latency |
| --- | ---: |
| L1 cache | 4 cycles |
| L2 cache | 12 cycles |
| L3 cache | 38 cycles |
| DRAM | ~248 cycles |

具体数字随架构而变，但层级之间的数量级差异长期存在。

### 8.2 Cache

Cache 是硬件实现细节：它不改变程序结果，只保存 memory 中一小部分数据的副本以改善性能。

它利用：

- Temporal locality：最近访问的数据可能再次访问；
- Spatial locality：附近地址可能很快被访问，因此以 cache line 搬运。

命中 cache 可以缩短 stall；miss 则需要向更远层级取数。

> [!tip] 本讲最重要的预告
> 高效处理几乎总会落到“如何高效访问数据”。后续 SIMD、GPU 和 kernel 优化都不能只数 ALU，还要数数据从哪里来、搬多少次。

## 9. AI Infra 视角

### Compute

- LLM kernel 的峰值 FLOPs 来自大量 core、SIMD/SIMT lanes 和专用 tensor units。
- 只有足够的独立工作才能接近峰值；小 batch、瘦矩阵和串行 decode 往往难以填满硬件。

### Memory

- 权重、activation 和 KV Cache 的搬运可能比算术更贵。
- Cache/local memory 中的数据复用决定实际 arithmetic intensity。

### Communication

- 多 GPU 并行中的 AllReduce、AllGather 与 All-to-All 对应本讲的通信项。
- 增加设备若同时增加同步等待，speedup 会快速偏离线性。

### Efficiency

除了 latency/throughput，还应报告：

$$
\text{MFU}=\frac{\text{measured useful FLOPs/s}}{\text{peak FLOPs/s}}
$$

它与并行效率思想一致：不仅问“跑多快”，还问“峰值资源有多少真正转化为有用工作”。

## 本讲结论

1. Speedup 与 efficiency 是两个不同指标，必须同时看。
2. 分解、工作分配和通信/同步决定并行程序是否 scalable。
3. 频率提升和自动 ILP 已经触顶，multi-core 与专用硬件成为主路径。
4. 功耗是架构设计的一等约束。
5. 数据访问和 cache 行为通常比算术操作本身更决定效率。

## 自测问题

1. 8 个处理器获得 4 倍加速时，并行效率是多少？还有哪些信息不足以判断优化是否值得？

    **面试回答：** 并行效率是 $E=S/P=4/8=50\%$。要判断优化是否值得，还要看单核基线是否已优化、绝对延迟与吞吐收益、硬件和能耗成本，以及增加的开发维护复杂度；仅凭加速比无法判断投入产出。

2. 为什么任务切得更细通常改善负载均衡，却可能降低整体性能？

    **面试回答：** 更细的任务让空闲 worker 更容易接走剩余工作，减少最后几个长任务造成的尾部等待。但任务数量增加也会放大入队、同步和调度成本，甚至破坏 cache locality；应让每块计算成本明显大于调度成本，同时保留足够多的块。

3. Superscalar 与 multi-core 各利用哪一种并行性？

    **面试回答：** Superscalar 利用同一线程内部独立指令的指令级并行，由硬件在一个周期向多个执行单元发射指令。Multi-core 利用不同线程或任务的并行性，各核具有独立指令流，软件需要暴露足够的独立工作；两者可以叠加。

4. 为什么晶体管数量继续增长，却不再自动带来相同比例的单线程加速？

    **面试回答：** 单线程提速受功耗、散热、数据依赖和有限 ILP 限制，更多晶体管不能直接换成等比例的更高频率或更宽有效发射。新增资源更多用于多核、SIMD、cache 和专用单元，程序需要相应的并行性与局部性才能利用它们。

5. Cache 为什么不改变程序语义，却会极大影响执行时间？

    **面试回答：** 在正确的内存语义下，cache 只是透明保存数据副本，读写结果仍遵守程序规定。它改变的是数据到达执行单元的代价：命中近端 cache 可大幅减少访存等待，缺失则要访问更远层级，因此相同指令数也可能有很不同的运行时间。


## 参考资料

- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)
- [The Future of Microprocessors](https://queue.acm.org/detail.cfm?id=1095411)，Kunle Olukotun、Lance Hammond
- [The Free Lunch Is Over](http://www.gotw.ca/publications/concurrency-ddj.htm)，Herb Sutter
