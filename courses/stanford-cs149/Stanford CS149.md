---
type: course
status: developing
area: systems
aliases:
  - CS149
  - Stanford Parallel Computing
---

# Stanford CS149：Parallel Computing（Fall 2023）

> [!abstract] 课程定位
> 从程序、处理器和内存三个视角理解现代并行计算：先识别并行性，再把工作映射到 multi-core、SIMD、硬件多线程和 GPU，最后围绕计算、访存、通信与同步优化效率。

## 课程信息

- 讲师：Kayvon Fatahalian、Kunle Olukotun
- 学期：Fall 2023
- [课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)
- [YouTube 播放列表](https://www.youtube.com/playlist?list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp)
- 先修建议：C/C++、数据结构与算法、基础计算机体系结构

## 学习主线

```text
为什么需要并行
→ 现代处理器怎样提供并行能力
→ 编程模型怎样表达并行性
→ 如何分配工作并控制通信
→ GPU、数据并行和分布式计算
→ 一致性、同步与专用硬件
```

## 课程笔记

1. [x] [Lecture 01：Why Parallelism? Why Efficiency?](Stanford%20CS149%20-%20Lecture%2001%20-%20Why%20Parallelism%20Why%20Efficiency.md)
2. [x] [Lecture 02：A Modern Multi-Core Processor](Stanford%20CS149%20-%20Lecture%2002%20-%20A%20Modern%20Multi-Core%20Processor.md)
3. [x] [Lecture 03：Multi-Core Architecture Part II + ISPC](Stanford%20CS149%20-%20Lecture%2003%20-%20Multi-Core%20Architecture%20Part%20II%20and%20ISPC.md)
4. [x] [Lecture 04：Parallel Programming Basics](Stanford%20CS149%20-%20Lecture%2004%20-%20Parallel%20Programming%20Basics.md)
5. [x] [Lecture 05：Performance Optimization I — Work Distribution and Scheduling](Stanford%20CS149%20-%20Lecture%2005%20-%20Performance%20Optimization%20I%20Work%20Distribution%20and%20Scheduling.md)
6. [x] [Lecture 06：Performance Optimization II — Locality, Communication, and Contention](Stanford%20CS149%20-%20Lecture%2006%20-%20Performance%20Optimization%20II%20Locality%20Communication%20and%20Contention.md)
7. [x] [Lecture 07：GPU Architecture and CUDA Programming](Stanford%20CS149%20-%20Lecture%2007%20-%20GPU%20Architecture%20and%20CUDA%20Programming.md)
8. [x] [Lecture 08：Data-Parallel Thinking](Stanford%20CS149%20-%20Lecture%2008%20-%20Data-Parallel%20Thinking.md)
9. [x] [Lecture 09：Distributed Data-Parallel Computing Using Spark](Stanford%20CS149%20-%20Lecture%2009%20-%20Distributed%20Data-Parallel%20Computing%20Using%20Spark.md)
10. [x] [Lecture 10：Efficiently Evaluating DNNs on GPUs](Stanford%20CS149%20-%20Lecture%2010%20-%20Efficiently%20Evaluating%20DNNs%20on%20GPUs.md)
11. [ ] Lecture 11：Cache Coherence
12. [ ] Lecture 12：Memory Consistency
13. [ ] Lecture 13：Fine-Grained Synchronization and Lock-Free Programming
14. [ ] Lecture 14：Midterm Review
15. [ ] Lecture 15：Domain-Specific Programming Languages
16. [ ] Lecture 16：Transactional Memory I
17. [ ] Lecture 17：Transactional Memory II
18. [ ] Lecture 18：Hardware Specialization
19. [ ] Lecture 19：Accessing Memory + Course Wrap-Up

## 延伸阅读

- [x] [[Stanford CS149 - Reading - The Story of ISPC|The Story of ISPC：ISPC 的起源、设计、实现与性能]]

## 用同一个框架复习每讲

| 层次 | 问题 |
| --- | --- |
| Work | 总工作量是多少，能否拆成独立任务？ |
| Parallelism | 是 ILP、SIMD、thread-level parallelism，还是它们的组合？ |
| Mapping | 谁负责把逻辑工作映射到 core、hardware thread 和 SIMD lane？ |
| Utilization | 哪些执行单元在空闲，为什么？ |
| Memory | 数据从哪里来，访问模式是否有局部性？ |
| Communication | 线程之间传什么，何时同步？ |
| Bottleneck | 当前受 compute、latency、bandwidth 还是 synchronization 限制？ |
| Efficiency | 加速是否与新增资源匹配？性能/功耗/成本是否合理？ |

## 与 AI Infra 的连接

- Transformer kernel 的 vectorization、warp execution 与分支分歧建立在 SIMD/SIMT 基础上。
- FlashAttention 的关键不是减少数学运算，而是减少 HBM 流量、提高数据复用。
- Tensor Parallel、Pipeline Parallel 和 Expert Parallel 都要同时分析工作划分、通信和负载均衡。
- GPU occupancy 可以看作利用大量 hardware contexts 隐藏延迟，但 occupancy 高不代表带宽或计算单元一定高效。
- Roofline/Arithmetic Intensity 延续了本课程最核心的问题：每搬运一个字节，能做多少有用计算？

## 关联主题

- [Transformer Block](../../topics/model-architecture/Transformer%20Block.md)
- [LLM Inference](../../topics/inference/LLM%20Inference.md)

## 建议实践

1. 完成 Assignment 1，用线程与 ISPC 对 Mandelbrot、SAXPY、K-means 做性能分析。
2. 每次优化同时记录 wall time、speedup、资源数和 efficiency，不只看“更快了多少”。
3. 对每个 kernel 估算 bytes moved、FLOPs 和 arithmetic intensity，再解释测量结果。
