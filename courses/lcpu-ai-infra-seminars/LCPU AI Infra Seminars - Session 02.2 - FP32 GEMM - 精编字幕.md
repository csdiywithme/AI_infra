---
type: transcript
status: polished
course: Weiming HPC Training Camp × LCPU AI Infra Seminars
session: "2.2"
speaker: 周宇轩
video_url: https://www.bilibili.com/video/BV1L8GA6YEAH/
source_status: transcript-reviewed
topics: [cuda, gemm, memory-hierarchy, roofline, tiling]
---

# Session 02.2：FP32 GEMM Quick Walkthrough｜精编字幕

> [!info] 整理说明
> 依据用户指定 Chrome 插件导出的完整 SRT 整理，共 644 条，覆盖至 30:02。保留讲述顺序与推导，删除口头重复、修正识别错误；不是逐字稿。以下时间为原字幕中的分钟级回看锚点，不表示每个改写句子的精确起点。未见独立问答环节。
>
> 官网 `session0202.pdf` 实际主要讲访存与 Reduce，不能直接当作本视频的完整课件。本稿以对应 BVID 的字幕为主，不用不匹配课件补写讲述内容。笔记见 [[LCPU AI Infra Seminars - Session 02.2 - FP32 GEMM Quick Walkthrough]]。

## 术语校正

| 字幕误识别示例 | 校正 |
|---|---|
| memory hook / 慢慢 hook | memory hierarchy，存储层次 |
| 占姆 / 詹姆 | GEMM |
| 扩大 / 库拉 | CUDA |
| 探索扣 / tenscall | Tensor Core |
| 计算器（保存累加值时） | register，寄存器 |
| RUFINE | Roofline |
| share 版本 / shin memory | shared memory / SMEM |
| class / cos（访存上下文） | coalescing，合并访存 |
| 现代化 | vectorization，向量化 |
| MO photo / MO store | MIO throttle / MIO stall |
| 库拉斯 | cuBLAS |
| CP A think / ta / EBERRY | `cp.async` / TMA / `mbarrier` |

## [00:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=0) 用一个 GEMM 建立存储层次的直觉

这部分演示如何用刚学到的 memory hierarchy 优化 FP32 GEMM。时间有限，不会覆盖所有后续优化；重点是建立直觉：面对一个计算任务，如何根据数据复用关系安排存储与执行。

GEMM 是通用矩阵乘法。这里考虑 `C=A×B`：左边是 A，右上方是 B，输出 C 的一个元素，对应 A 的一行和 B 的一列的点积。矩阵乘法在深度学习与科学计算中都很常见，值得专门优化。

## [01:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=60) 为什么从 FP32、CUDA Core 开始

性能目标是在给定时间完成尽可能多的浮点运算。一次乘法与一次加法分别算一个 FLOP，因此一个 FMA 对应两个 FLOP；完整矩阵乘法约有 `2MNK` 次浮点运算。

现代 GPU 为矩阵乘法提供 Tensor Core，但本例有意选择以 CUDA Core 为基础的 FP32 路径，便于把优化拆成手工可理解的步骤。深度学习常用 FP16 或更低精度输入，不必在所有位置都使用完整 FP32 输入精度。

> [!note] 精度边界
> 这里的教学选择不等于“任何 FP32 接口的矩阵乘法都不会使用 Tensor Core”。必须区分输入存储类型、实际乘法精度、累加精度，以及库是否允许 TF32 等路径。

## [03:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=180) 朴素实现：一个 thread 负责一个 C 元素

输出元素彼此独立，最直接的并行化方式就是把每个元素交给一个线程。CUDA 提供 grid → block → thread 的层级；一个 block 内的线程可以共享 shared memory，多个 block 组成 grid。

C 是 `M×N` 矩阵，可以先划分成二维 block tiles，再在 tile 内用二维 thread index 定位输出元素：block index 决定 tile 的起点，thread index 决定 tile 内偏移。每个线程用寄存器保存一个 accumulator，遍历 K，读 A 的一行和 B 的一列，做乘加，最后写回 C。

例如选择 `16×16` 的 block，grid 在 N、M 两个方向分别向上取整。边界线程需要判断是否落在有效矩阵范围内。此时实现非常容易理解，但还没有显式组织线程之间的数据复用。

## [07:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=420) 算法复用潜力很大，朴素实现却没利用起来

相邻输出元素会用到相同的 A 行或 B 列。例如一个 `2×2` 输出块，只涉及两行 A 和两列 B；如果四个线程各自读取，就会重复搬运。

理想情况下，A、B 各读一次，C 写一次，FP32 的计算强度为：

$$I_{ideal}=\frac{2MNK}{4(MK+KN+MN)}.$$

对于边长为 n 的方阵，就是 `n/6 FLOP/byte`。n 越大，数据潜在复用越高，因此大 GEMM 应有机会成为 compute-bound。

但朴素实现中，每个线程做 K 次乘加，读取 `2K` 个输入元素并写一个输出，按这种访问量估算：

$$I_{naive}=\frac{2K}{4(2K+1)}\rightarrow\frac14\ \text{FLOP/byte}.$$

它几乎是常数，没有利用矩阵规模变大带来的复用机会。这里统计的是程序所需访问量；实际 HBM 流量还会受到 cache 命中的影响，不能把两者完全等同。

## [09:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=540) 第一层 tiling：用 shared memory 显式共享输入

Cache 可以自动捕获局部性，但 GEMM 的访问模式足够规整，程序员事先就知道哪些数据会复用。于是可以把 shared memory 当作可编程的 scratchpad：让整个 CTA 协作搬入数据，再由各线程读取自己需要的元素。

一个 CTA 负责 `BM×BN` 的 C tile，沿 K 再切成 `BK` 大小的阶段。每阶段搬入 `BM×BK` 的 A tile 和 `BK×BN` 的 B tile，然后更新该 CTA 的输出。搬运任务如何分给线程，与每个线程最终计算哪个输出，可以分别设计，不必完全一致。

## [11:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=660) 每轮 K 循环为什么要同步两次

首先，线程协作把 A、B 搬入 shared memory。必须进行一次 block 同步，保证消费者读取前，其他线程负责的数据已经写好。随后各线程遍历当前 tile 的 K 维，从 shared memory 读数并累加。

计算结束后，还需要再同步一次。如果不等所有线程完成读取，跑得快的 warp 可能提前进入下一轮，覆盖 shared-memory buffer；跑得慢的 warp 就会把下一轮数据当作当前轮数据使用。

两个同步分别保护两种依赖：第一次是“写完才能读”；第二次是“读完才能覆盖”。

## [13:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=780) Tile 变大，复用增强，但不能只看公式

忽略最终 C 写回，单阶段从 global memory 搬运输入的计算强度为：

$$I_{CTA}\approx\frac{2BM\,BN\,BK}{4(BM\,BK+BK\,BN)}=\frac{BM\,BN}{2(BM+BN)}.$$

方形输出 tile 的边长为 T 时，约等于 `T/4`。T 为 32 时，可达约 `8 FLOP/byte`，远高于朴素实现。

这说明更大的输出 tile 有利于复用，但它只是方向，不是“越大越好”的结论：线程数、shared memory、寄存器和驻留资源都有上限。

## [14:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=840) 合并访存看的是 warp 的同一次请求

向 shared memory 搬运时，要让一个 warp 的线程访问连续且合适对齐的 global 地址。例如每个 lane 读取一个连续 FP32，整个 warp 覆盖 128 字节；硬件可以把它们组织成较少的内存事务。

容易误判的一点是：某个线程先读一个数，再读它旁边的数，并不自动意味着合并访存。Coalescing 看的是 warp 内各 lane 在同一条访存指令上发出的地址。若同一时刻线程之间相隔很远，即使每个线程自己的后续访问连续，当前这次访存仍然可能很差。

## [16:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=960) 把 thread 的连续方向对齐到数组的连续方向

对 C 风格的 row-major 二维数组，变化列下标的方向连续：`A[i][j]` 的线性地址是 `i*stride+j`。CUDA 的线程线性编号中，x 维变化最快。

因此，应尽量让相邻 `threadIdx.x` 对应连续列，而不是对应相邻行。若相邻 lane 分别访问不同行的相同列，地址间隔就会变成一整行的跨度，难以形成高效合并访问。

## [18:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=1080) 向量化搬运：一次搬四个 FP32

另一个方向是增大每条搬运指令的数据量。例如让一个线程搬运 `float4`，把四个连续 FP32 合在一次向量访问中。按连续列方向分组，有机会减少指令数量并提高在途数据量。

虽然有些写法编译器能自动向量化，但如果它无法证明对齐条件，就未必会生成期望的向量指令。因此教学中会显式表达 `float4` 搬运。实际实现还需要保证地址对齐、越界处理和正确的类型访问方式，不能仅靠强制转换期待提速。

## [19:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=1140) Occupancy 很高，为什么还是跑不满

Nsight Compute 中要区分几类 warp：active warp 已获得执行上下文与寄存器等资源，可以驻留；eligible warp 则在当前时刻具备发射条件，没有未满足的数据依赖，且目标执行通路可以接收指令。

讲者展示的 A100 实验中，每个 scheduler 约有 15.94 个 active warps，已经接近该配置的驻留上限，但很多 warp 仍不能发射。高 occupancy 不等于每个周期都有足够 eligible warps，更不等于计算单元满载。

## [21:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=1260) 真正的瓶颈转移到 shared-memory 指令

Profiling 指向 MIO throttle：相关指令队列压力很大，新的指令无法继续进入。结合这个 kernel 的指令统计，主要问题是 shared-memory load/store 指令太多。

把数据从 HBM 复用到 shared memory，只解决了第一层搬运；如果每个 FMA 仍需频繁向 shared memory 要数据，瓶颈可以继续出现在 SM 内部。因此下一步不是再堆 active warps，而是减少单位计算所需的 shared-memory 访问。

> [!note] 数值处理
> 这段字幕对 stall 周期和百分比的识别有明显混乱，未把不可靠的小数抄成测量结论。保留可确认的诊断链：高驻留 → eligible 不足 → MIO 队列压力 → 增强寄存器复用。

## [23:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=1380) 第二层 tiling：一个 thread 负责多个输出

让一个线程负责 `TM×TN` 的小输出 tile，用寄存器保存多个 accumulator。每个 K 步，读取 TM 个 A 值和 TN 个 B 值，通过外积更新 `TM×TN` 个输出。

最小例子是 `2×2`：读两个 A、两个 B，就能做四次 FMA；若四个输出独立计算，需要四次 A 读取和四次 B 读取。现在输入在寄存器里再次复用，shared-memory 指令压力下降。

一个线程计算更多输出，还有一个好处：同样的 CTA 输出 tile 不再需要一个输出对应一个线程。原来 `32×32` 的 C tile 就要 1024 个线程；现在可用较少线程覆盖更大的 tile，进一步提高 global-memory 复用。

## [25:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=1500) 更大的 tile 仍需调参

扩大 CTA tile、调整每线程工作量，会改变寄存器占用、occupancy、shared-memory bank 访问与指令行为。如果 work assignment 不好，可能产生 bank conflicts，使访问需要更多 wavefront 才能完成。

因此需要在资源允许范围内搜索 tile 大小和线程映射，而不是只优化理论计算强度。本次演示不展开 autotuning 的全部细节。

讲者报告，经过这些基础 tiling 优化，示例获得约五到六倍加速，并达到其对比配置下 cuBLAS FP32 性能的约 80%。这是课堂实验结果，不是跨 GPU、跨 shape 的性能保证，也不能与允许其他精度路径的库结果混比。

## [27:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=1620) 下一处瓶颈：load 仍在计算的关键路径上

即使 tiling 做得更好，当前程序仍是串行阶段：协作搬运 → 同步 → 计算 → 同步，再重复。搬运时计算单元不能持续工作，load 时间仍占据关键路径。

若希望接近 compute-bound，就要让计算尽可能持续进行。典型方法是多缓冲：一个 buffer 供当前计算使用，另一个 buffer 接收下一轮输入；下一轮交换角色，让搬运与计算重叠。

## [28:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=1680) 双缓冲与异步拷贝，连接后续课程

双缓冲把“下一块的 copy”和“当前块的 compute”并行起来，但前提是正确管理不同 buffer 的生命周期：计算不能读尚未完成的拷贝，拷贝不能覆盖尚未读完的数据。

硬件也逐步提供支持。Ampere 引入 `cp.async`；之后 TMA 进一步支持张量式搬运，减少部分地址计算负担，并与异步同步机制配合。这些内容会在 Tensor Core 与 Pipeline 课程中继续展开。

本讲的主线到此闭合：global memory 的数据在 CTA 内复用，shared memory 的数据在线程寄存器内复用，再用 pipeline 争取隐藏搬运时间。

## 原始来源留档

- 视频：[BV1L8GA6YEAH](https://www.bilibili.com/video/BV1L8GA6YEAH/)，时长 30:05。
- 原始 SRT：`北京大学未名超算队 × LCPU AI Infra Seminars 系列讲座：2.2-FP32 GEMM Quick Walkthrough_哔哩哔哩_bilibili_BV1L8GA6YEAH_字幕.srt`，保留于 Downloads，未改写。
- SHA-256：`477b367ec5b549fe1261781635d7e0ae22540ef4e0898a10ae86a7ac2db5e821`。
