---
type: course-note
status: developing
source_status: transcript-and-slides-reviewed
course: Weiming HPC Training Camp × LCPU AI Infra Seminars
lecture: W1
lecture_date: 2026-08-01
speaker: 孔昊然
area: systems
topics:
  - gpu
  - tilelang
  - compiler
  - layout
  - reduction
  - pipeline
  - persistent-kernel
aliases:
  - LCPU W1
  - Parallel Programming with TileLang
video_url: https://www.bilibili.com/video/BV1rhMD6YEKM/
slides_url: https://infra.seminars.lcpu.dev/slides/workshop01.pdf
---

# Workshop 01：Parallel Programming with TileLang

> [!abstract] 核心问题
> TileLang 把 tile 级操作映射到线程与硬件指令，但程序员仍要决定 workload 分块、存储层级、依赖结构与调参空间。理解 ownership、layout inference 和 lowering，才能判断简洁代码最后做了多少搬运、通信和同步。

> [!info] 整理依据
> 已阅读完整字幕（1658 条，至 01:26:30）并与官方 78 页课件对照。本文按知识结构重组；原讲述顺序与问答见 [[LCPU AI Infra Seminars - Workshop 01 - TileLang - 精编字幕]]。PDF 页码含封面、1 起计数；课件中的详细 reducer lowering 是补充材料，不能当作现场已经确认的结论。

## 来源与导航

- [视频](https://www.bilibili.com/video/BV1rhMD6YEKM/)
- [官方课件](https://infra.seminars.lcpu.dev/slides/workshop01.pdf)
- [课程日历](https://infra.seminars.lcpu.dev/schedule)
- 课件实验环境：H100 80GB、TileLang 0.1.12、TVM-FFI 0.1.11。实现细节按这一课件版本理解。

| 视频回看 | 内容 |
|---|---|
| [03:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=180) | Bank conflict 与同地址 broadcast 的澄清 |
| [14:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=840) | 四个编程问题与 DSL 的责任边界 |
| [18:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=1080) | JIT、lowering、TVM-FFI |
| [27:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=1620) | CTA、fragment 与 ownership |
| [35:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=2100) | Layout inference、transpose、同步 |
| [45:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=2700) | Packing annotation 与 reducer 现场问答 |
| [54:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=3240) | Occupancy、pipeline、persistent |
| [68:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=4080) | Profiling 与 FFI/fusion 的讨论 |

### 开场纠错：同 bank 不一定冲突

课堂特别澄清：多个线程访问同一 bank 中的**同一地址**时，可以广播；同 bank 的**不同地址**才是冲突分析的重要情形。16 个线程读一个地址、另 16 个读另一个 bank 中的地址，也可以用 multicast 服务，不要求全部 32 个线程都读同一个地址。具体宽度和实际事务还应结合目标硬件分析。

| PDF 页 | 主题 |
|---|---|
| 13–14 | TileLang 自动化边界与四个编程问题 |
| 15–25 | 最小 kernel、JIT、编译链、TVM-FFI、TIRx |
| 26–34 | CTA workload、fragment、`T.Parallel`、layout inference |
| 35–39 | Transpose：向量化、padding、线程映射和同步 |
| 40–44 | Tile primitives 与显式 layout annotation |
| 45–65 | Reducer、replication 与从 DSL 到 CUDA 的完整 lowering |
| 66–69 | Fragment replication、warp scheduler 与 occupancy |
| 70–71 | `T.Pipelined` 与 stage sweep |
| 72–76 | Persistent、MegaKernel、Engram backward |
| 77–78 | Profiling、SASS 与参考实现 |

## 1. 四个必须回答的问题

| 问题 | 用户决定 | 编译器在约束下完成 |
|---|---|---|
| 谁负责哪个元素？ | CTA tile、grid、threads、必要的显式 layout | tile 内 thread ownership 与线程局部索引 |
| 数据放在哪里？ | global/shared/fragment/local scope | 物理地址、register packing、部分 swizzle |
| 操作何时发生？ | producer/consumer 关系、循环与融合边界 | 合法 pipeline 展开、部分同步与异步 lowering |
| 配置多大？ | tile、线程数、stages、worker 数的候选空间 | 给定配置的代码生成；调优器测量候选 |

`T.Parallel` 的逻辑迭代次数可以远大于实际线程数。一次 logical iteration 由哪个 thread 的哪次 local iteration 完成，是布局映射问题。

## 2. JIT：哪些参数应该静态化

影响实现结构的参数通常适合 specialization，例如 tile shape、layout、memory scope 和部分控制结构。只改变工作总量的参数可考虑保留动态，减少编译产物数量。

代价关系是：更多静态信息给编译器更多优化机会，也增加编译时间、缓存占用和变体管理。并非任何动态 problem size 都能无条件保留动态；合法性取决于该 kernel 和编译器支持。

课件呈现的链条：

```text
Python DSL / kernel factory
  → 静态参数特化与前端解析
  → TIRx PrimFunc / IRModule + tile operations
  → pipeline planning / layout inference
  → LowerTileOp / 地址生成 / 同步与存储规划
  → host/device split
  → CUDA 或 CuTeDSL backend → cubin / SASS
  → JIT cache / runtime adapter / CUDA launch
```

TVM-FFI 处理跨语言 ABI、张量元数据、参数组织和生成的 host stub。对微秒级 kernel，Python 与 launch overhead 可能占据明显比例，因此 device kernel 时间与 end-to-end 调用时间要分别测量。

> [!important] 现场补充（01:09–01:11）
> 有参与者指出，其测量中 TVM-FFI 相比 PyTorch binding 的调用收益没有特别大，优势还包括减少编译/链接依赖；launch-bound 场景也应考虑 fusion。讲者认可需要在自己的 workload 上比较不同路径。不能把“使用 TVM-FFI”直接写成“必然解决 launch bottleneck”。Fusion 也可能增加寄存器与 SMEM 压力。

## 3. Fragment 是逻辑张量的分布式所有权

`T.alloc_fragment((M, N))` 声明一个逻辑 fragment，其映射可以写成：

$$
(i,j)\mapsto(\text{thread id},\text{thread-local slot})
$$

通常它由多个线程共同持有；不能把整个 $M\times N$ 都按每线程私有副本估算。出现 replication 时，才额外引入复制坐标：

$$
(i,j,r)\mapsto(\text{thread id},\text{local slot})
$$

需要区分三种布局：global tensor 的 shape/stride、shared memory 的物理排列，以及 fragment 的线程/register ownership。它们在 copy 或 GEMM 边界相互约束，但含义不同。

## 4. Layout inference：把局部约束传播到整条数据流

`T.gemm` 带来目标 Tensor Core 的 fragment/SMEM 约束；`T.copy` 连接两个 memory scopes；`T.reduce_sum` 约束 partial 与输出的 ownership；`T.Parallel` 连接循环和 buffer access。编译器沿这些使用关系传播约束、检查冲突，再为剩余自由维度选择布局。

显式 layout 是约束锚点。它可能减少重排，也可能与其他操作冲突，不能脱离 producer 和 consumer 单独选择。

### Transpose 案例（PDF 35–39）

直接转置容易导致一侧连续、另一侧 strided。课件实现通过线程内小块转置和 SMEM 中转，让 global input 与 output 两端都按连续方向 vectorize：

```text
连续 global read
  → 每线程 4×4 micro-transpose
  → shared memory 交换所有权
  → CTA barrier
  → 按输出 layout 连续读取、写回
```

SMEM 的额外 padding 用来打破不利的 bank 周期对齐，同时保持 vector access 所需对齐。显式 `loop_layout` 决定输出迭代由谁执行；它不提供同步。手写 shared stores 与另一个线程的 loads 之间仍需要 `T.sync_threads()` 等正确协调。

### 8 元素打包案例（PDF 44）

E5M6 每值 12 bits，8 个值正好打包成 3 个 `uint32`。若 packing helper 使用 thread-local buffer，就需要这 8 个值在同一 thread 中。显式 layout 将连续 8 元素交给同一 owner，避免为打包额外进行跨线程交换。

## 5. Tile primitives 为什么要保留到中间表示

若过早展开成逐线程 load/store，编译器就难以恢复“这是一整个 tile copy”或“这是一次矩阵操作”。保留高层 region、scope 与 shape，使后端有机会选择 scalar/vector load-store、cooperative copy、`cp.async` 或 TMA。

因此 `T.copy` 是数据运动语义，不承诺每次都生成 TMA。实际路径依赖 target、shape、alignment、scope 与 schedule，必须看生成代码。

## 6. Reducer：先明确 partial 分布，再选择 API

> [!note] 录播与课件的边界
> 46–53 分钟的现场讨论没有确定 `finalize_reducer` 的完整 lowering，曾混淆 global 与 CTA 内 SMEM/reducer 的作用域，最后明确留待查看生成代码。以下 replication 规则、128×64 示例与 pass 链来自官方课件的详细补充，不是录播中已经完成的推导；复现应固定课件版本并检查实际 lowering。

| Partial 的位置 | 典型处理 | 关键契约 |
|---|---|---|
| 同一线程 | serial/unroll 累加 | 线程局部依赖 |
| 一个完整 warp | `T.warp_reduce_*` | 规定 lanes 收敛执行；无效输入用单位元 |
| 分布式 fragment | `T.reduce_*` | 布局和 reduction dimension 决定通信 |
| `T.Parallel` 的声明式累计 | `T.alloc_reducer` + `T.finalize_reducer` | replication 决定 finalize 范围 |
| 多个 CTA | atomic 或多阶段 reduction | CTA 内 barrier 不足以完成 grid 归约 |

课件版本中：

- `replication="none"`：一个逻辑 reducer 元素有一个 owner，finalize 不负责收集其他线程的 partial；这些贡献必须已由 owner 正确累计。
- `replication="all"`：current thread range 内各线程拥有完整 logical shape 的 partial 副本，finalize 对每个逻辑元素执行全 range AllReduce。

真正产生贡献的线程数，可以少于 finalize 的通信线程数。编译器不保证按非零贡献自动缩小 collective。

### 课件实例：128 行、64 列、256 threads

布局为：

$$
t=(row\bmod16)\times16+\lfloor col/4\rfloor
$$

$$
s=\lfloor row/16\rfloor\times4+(col\bmod4)
$$

每行的 64 个输入分布在 16 个 threads，每线程持有连续 4 个值。但 fully replicated row reducer 让全部 256 threads 都有 128 个 row partial slots；没有贡献的槽位保存单位元。

finalize 为每个 row 执行 256-thread AllReduce。通信先通过 SMEM/barrier 跨 warp，再用 warp shuffle 完成 warp 内合并。

> [!note] 整理者推导
> 若每线程 reducer 有 128 个 FP32 slots，逻辑上对应 $128\times4=512$ bytes 的线程局部值；这只是活跃状态的量级，不是 ptxas 实际 register 数。编译优化、生命周期、复用与 spilling 会改变最终分配。这个案例提醒我们：简洁的 reducer 声明可能带来大量复制状态和通信。

### Lowering 中各阶段的职责

1. 前端生成 fragment buffer 与 reducer metadata。
2. `LayoutReducer` 根据 replication 建立 reducer layout。
3. `LayoutInference` 主要确定贡献值和 parallel iterations 的 ownership。
4. `LowerTileOp` 按 `ReplicateExtent` 生成 no-op 或全 range AllReduce。
5. CUDA backend 选择具体 shuffle、shared workspace 和 barrier 模板。

排查问题时应定位是哪一阶段做出了不符合预期的选择，而不是只读最终 CUDA 中的循环。

## 7. `T.Pipelined`：用更多在途工作隐藏延迟

编译器把逻辑循环改写成 fill、steady state、drain，对存活区间重叠的 buffer 创建多个版本，再加上相应 commit/wait/barrier。

增加 stages 可能提高 copy 与 compute 的重叠，也可能增加 shared memory、register live ranges 和填充/排空成本，从而降低 residency。实际 buffer versions 由依赖和生命周期分析决定，不必等于 `num_stages`。

课件在 H100、4096³ GEMM、128×128×32 tile、128 threads 的特定实验中，stage 5/6 较优。这个结果不能直接搬到其他 shape 或 GPU。

## 8. Persistent 决定任务归谁，pipeline 决定时间顺序

Persistent kernel 使用有限的 worker CTAs 循环处理更大的逻辑任务域。worker 数等于 SM 数只是常见配置；代码即使没有直接调用 `T.Persistent`，也可能具有相同调度语义。

课件的 Engram backward 把 `grad_w_local` 放在 worker 的任务循环外。每个 worker 持续累计多个 token 的 partial，结束后写出一次，再由另一个 reduction kernel 合并。它用稳定 worker identity 与长期局部状态，减少逐 token 的 global atomic 竞争。

代价包括局部状态占用、尾部负载不均、状态重置与队列争用。Persistent 与 pipeline 可组合：前者定义 tile/task 到 worker 的映射，后者重叠 worker 内不同 iterations。

## 9. 证据驱动的调优顺序

1. Correctness：边界、dtype、layout 和 reduction 单位元。
2. Lowering trace：哪次 pass 改变 ownership、copy path 或同步。
3. Generated CUDA / SASS：实际使用的 instructions、buffer versions、barrier。
4. Resource report：registers、SMEM、spill 与 residency。
5. Nsight Compute：memory throughput、stall 与执行管线。
6. Nsight Systems：launch gap、跨 kernel 依赖和 CPU/GPU overlap。

录播另介绍 in-kernel profiling（IKet）作为细粒度阶段观测工具，并说明所用 TileLang 0.1.12 release 尚未包含相应支持。它补充 NCU/Nsight Systems，并非替代。是否可用须按自己的工具版本核实。

SASS 可以证明实现不同，却不能单独证明性能更好；性能归因需要计时、资源和 counters 一起支持。

## 10. AI Infra 视角与前后讲关联

| 视角 | 本讲关注 |
|---|---|
| Shape | CTA tiles、逻辑 iteration domain、reduction axis |
| Compute | Tensor Core lowering、线程局部 work、vectorization |
| Memory | Fragment ownership、SMEM layout、replication footprint |
| Communication | Shuffle、shared exchange、reducer AllReduce |
| Runtime/System | JIT variants、host ABI、launch overhead、persistent workers |

[[LCPU AI Infra Seminars - Session 03 - Tensor Core 从 mma.sync 到 tcgen05]] 解释硬件要求怎样的 operand layout；本讲解释编译器怎样满足这些约束。[[LCPU AI Infra Seminars - Session 04 - Pipeline Ordering Data Orchestration]] 则展开 pipeline、buffer lifetime 与异步正确性。

## 11. 自测与复习检查

1. `T.Parallel(1024)` 和启动 1024 threads 有什么区别？

    **面试回答：** `T.Parallel(1024)` 定义 1024 次逻辑迭代，不决定物理线程数。实际 CTA 可以只有 128 或 256 个线程，编译器根据 layout 将多次迭代映射到同一线程的不同局部位置；线程数由 kernel 配置决定。

2. Global stride、SMEM swizzle、fragment ownership 分别描述什么？

    **面试回答：** Global stride 描述逻辑坐标如何换算成全局内存地址；SMEM swizzle 描述共享内存中的物理排列，以匹配 bank 访问；fragment ownership 描述每个逻辑元素由哪个线程、哪个局部寄存器槽持有。它们在 copy/GEMM 边界必须兼容，但属于三个不同层次。

3. 为什么 `T.copy` 不能直接等同于 TMA？

    **面试回答：** `T.copy` 表达 tile 的数据搬运语义，具体指令由目标硬件、内存作用域、shape、对齐和 schedule 决定。它可能生成标量/向量访存、`cp.async` 或 TMA，因此要检查 lowering 和生成代码才能确认实际路径。

4. 显式 layout 怎样帮助 8-wide packing？何时反而导致冲突？

    **面试回答：** 若打包函数需要线程局部的连续 8 个值，显式 layout 可把它们分配给同一线程，省去跨线程交换。例如课件中 8 个 12-bit 值可打成 3 个 `uint32`；但该布局可能与 GEMM/reduction 的 ownership 或 SMEM bank 分布冲突，导致重排、低效访存或编译失败。

5. 为什么只有 16 threads 有贡献，仍可能执行 256-thread AllReduce？

    **面试回答：** 归约范围由 reducer 的复制布局和 finalize 契约决定，不由本轮非零贡献线程数决定。课件的 fully replicated reducer 为全部 256 个线程保存 partial，没贡献的槽位放单位元，因此仍执行 256-thread AllReduce，编译器不会自动缩成 16-thread collective。

6. `replication="none"` 为什么不能用来自动跳过需要的跨线程合并？

    **面试回答：** 按课件版本，`replication="none"` 表示一个逻辑元素只有一个 owner，finalize 不再收集其他线程的 partial。只有所有贡献已经正确归并到该 owner 时才成立；直接改这个参数会漏算，不能把它当作去除必要通信的优化开关。

7. Pipeline stages 为什么与 buffer versions 不必相等？

    **面试回答：** Stages 描述逻辑操作在时间上的交错，buffer versions 由各 buffer 的读写依赖和存活区间决定。某些 buffer 需要多个版本避免覆盖，另一些可及时复用，因此不能把 `num_stages` 直接当作每个 buffer 的副本数；应看实际 lowering。

8. Persistent 为什么可以减少 atomic，却增加 register pressure？

    **面试回答：** 稳定的 persistent worker 可以跨任务保留局部 partial，先累加多个贡献再写回或做第二级归约，从而减少逐 token 的 global atomics。但长期存活的 accumulator 和调度状态占用寄存器，可能降低 occupancy、引发 spill；persistent 本身并不自动减少 atomic。

9. 如何区分 host launch bottleneck 与 kernel memory bottleneck？

    **面试回答：** 先比较端到端调用时间与 CUDA events 测得的设备执行时间，再用 Nsight Systems 看 CPU 提交和 kernel 间空隙。若主要时间在 kernel 内，则用 Nsight Compute 查实际访存量、带宽和 stall；CUDA Graph 或批量 launch 是否有效也能辅助验证 host 开销假设。

10. 查看 lowering trace、SASS 和 profiler，各自能证明什么？

    **面试回答：** Lowering trace 解释编译器在哪一步决定布局、copy 路径和同步；SASS 证明最终执行哪些指令、是否向量化或 spill；profiler 反映实际耗时、资源利用和阻塞。三者结合才能把源码设计、生成实现与性能结果连起来，单看 SASS 不能证明更快。


复现时保存 target、TileLang/FFI 版本、shape、dtype、tile、threads、stages、correctness tolerance 和计时方式，再记录生成代码与资源变化。

## 参考资料

- [官方 Workshop 01 课件](https://infra.seminars.lcpu.dev/slides/workshop01.pdf)
- [TileLang 文档](https://tilelang.com/)
- [Lowering Trace 工具说明](https://tilelang.com/tools/lower_trace.html)
- [DeepSeek TileKernels](https://github.com/deepseek-ai/TileKernels)
- [TileLang RMSNorm 示例](https://github.com/tile-ai/tilelang/blob/main/examples/norm/rms_norm.py)
