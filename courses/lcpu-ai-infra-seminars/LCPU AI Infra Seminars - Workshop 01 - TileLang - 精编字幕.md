---
type: transcript
status: polished
course: Weiming HPC Training Camp × LCPU AI Infra Seminars
session: W1
speaker: 孔昊然
video_url: https://www.bilibili.com/video/BV1rhMD6YEKM/
slides_url: https://infra.seminars.lcpu.dev/slides/workshop01.pdf
source_status: transcript-and-slides-reviewed
topics: [tilelang, layout, compiler, pipeline, persistent-kernel]
---

# Workshop 01：Parallel Programming with TileLang｜精编字幕

> [!info] 整理说明
> 依据完整 SRT（1658 条，至 01:26:30）与官方课件校正术语、断句和口头重复，保留实际讲述顺序及关键现场讨论，不是逐字稿。时间为分钟级回看锚点。结构化笔记见 [[LCPU AI Infra Seminars - Workshop 01 - Parallel Programming with TileLang]]。
>
> DSL 变化很快，以下是讲座所用版本的观点与行为，不承诺适用于未来版本。现场未解决的问题仍标为未确认；课件中的后续详细分析另放笔记，不倒填成讲者现场说过的话。

## 术语校正

| 原字幕常见误识别 | 校正 |
|---|---|
| 太浪 / 海浪 / 台浪 / 拍浪 | TileLang |
| Q DSL / cut dsl | CuTe DSL |
| git / 吉他（编译上下文） | JIT |
| layout influence | layout inference |
| TBMFI / TPPMFFI | TVM-FFI |
| TIRF / TLX | TIRx |
| prime funk | PrimFunc |
| ban conflict / broadcast 混写 | bank conflict / broadcast，按语义区分 |
| alloc producer / final producer | `alloc_reducer` / `finalize_reducer` |
| INSTGRAM / ian grand | Engram |
| SARS / SAS | SASS |
| NICO / NCS | Nsight Compute / Nsight Systems |

## [00:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=0) 从三页 GEMM 示例回到数据流

先回顾第一次介绍 TileLang 时的简单 GEMM：为输入分配 shared memory，为输出累加值分配 fragment，通过 pipeline、copy、GEMM 表达主要流程。

数据先从 global memory 搬到 shared memory，再用矩阵操作更新寄存器中的 fragment，最后写回 global memory。不要只记 API 名称，要看清每一步数据在哪里、怎样移动。

硬件计算能力增长很快，喂满计算单元却需要复杂的数据搬运组织。TileLang 的价值，是让我们用较少代码表达算法和数据流，把部分硬件相关细节交给编译器。

## [03:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=180) 对上一讲的补充：broadcast 的关键是相同地址

不能把 shared-memory broadcast 理解成“必须 32 个线程一起访问一个 bank 才发生”。一个 bank 可以视作独立的小型 SRAM，多个 bank 能并行服务。重要的是：同一个 bank 在一次服务中不能任意响应多个不同地址，但多个线程需要同一地址时，可以把读出的值广播给它们。

例如 16 个线程读 `shared[0]`，另外 16 个读另一个 bank 上的 `shared[1]`，也可以通过 multicast 完成，不必产生 bank conflict。每两个线程读同一地址的例子也能据此分析。判断冲突，应把 bank 与 address 分开，而不是只数线程。

## [05:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=300) TileLang 的定位：高层语义与可下探能力

讲者展示了 CuTe DSL 相关分享中的 TileLang 示例。上层是相对独立的 IR 与 tile 操作，下层通过不同 pass 和 lowering 逐步变成贴近目标硬件的实现。

TileLang 允许显式指定 memory allocation 与 movement，也允许专家使用更细粒度的原语。但如果处处都手工控制，DSL 减少心智负担的价值就会下降。因此本讲主要站在 developer 视角，把它作为 tile-level library 使用，理解其提供的便利与边界。

## [08:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=480) 相同 copy/GEMM 可以落到不同后端

高层代码相似，不同 target 的底层代码可以不同。完成目标支持后，同一个 copy 或 GEMM 可以选择适合 SM80、SM90 或其他架构的路径，例如异步搬运或 TMA。

这也是支持不同厂商硬件的基础，但“语言理论上可扩展”不等于“每种硬件已经有成熟实现”。本讲引用的部分示例来自较早分享，具体支持要以当前所用源码和编译结果为准。

## [11:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=660) 今天不只讲 API，而是建立可迁移的心智模型

TileLang 隐藏了一部分工作，但不会替程序员做完所有决定。今天围绕 JIT/TIR lowering/TVM-FFI、fragment 与 layout inference、tile library primitives、pipeline 与 persistent，以及 profiling 展开。

最重要的是弄清：哪些决策交给编译器，哪些仍要自己承担；对不确定的行为，怎样通过 lowering trace、生成代码和 profiling 寻找证据。

## [14:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=840) 写 kernel 要回答四个问题

第一，CTA、warp、thread 分别负责哪些数据。第二，数据位于 global、shared 还是 registers。第三，load、compute、store 如何排序与重叠。第四，tile、threads、stages、CTA 数等参数如何选择。

TileLang 的一个重要特征是：程序员决定 CTA workload，编译器可推导 tile 内很多线程映射。Memory scope 仍需表达；复杂依赖与同步，也不能完全期待编译器自动解决。

数据搬运不只发生在单卡，分布式环境中的 remote data 也有类似 producer/consumer 问题，后面的通信课会继续展开。

## [17:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=1020) 最小 kernel 与 JIT 的取舍

`PrimFunc` 描述函数，`T.Kernel` 声明 grid、CTA 与线程数，`T.Parallel` 表达逻辑并行域。逻辑迭代如何映射到线程，则受到 layout inference 约束。

JIT 在首次遇到静态配置时生成 artifact，相同配置可复用缓存。更多静态信息能帮助推导、降低运行时开销，但也会增加编译变体、首次编译成本与产物大小。讲者展示的主要是 lazy JIT 写法，也提醒语法与推荐形式正在演进。

## [20:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=1200) 编译链不是黑盒：逐步查看 lowering

Python DSL 前端构造 TIRx，经历 layout inference、目标相关优化、host/device 分离和后端编译。越往下，表示越接近具体硬件。

Lowering trace 能输出各 pass 前后的 IR，帮助判断某次变换是否符合预期。与其仅凭高层源码猜编译器做了什么，不如直接查看从逻辑操作到最终实现的过程。

## [23:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=1380) TVM-FFI：降低 host 侧封装成本

TVM-FFI 把生成的 host/device 产物封装成可调用对象，支持张量互操作，并把一部分参数和调用处理移到更低开销的路径中。对执行时间很短的 kernel，微秒级 host overhead 就值得关注。

这里先介绍它的设计目的与案例；后面现场讨论进一步指出，实际调用收益要测量，不能直接假设换一种 binding 就会显著提速。

## [26:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=1560) CTA tile 的大小仍由程序员选择

`T.Kernel` 决定逻辑 CTA 数、每个 CTA 的 tile 与线程数。Tile 太小，复用和 instruction-level parallelism 不足；太大，又会增加寄存器和 shared-memory 压力，降低驻留数量。

因此 TileLang 自动化的是许多 tile 内映射，不是任意 workload decomposition。Occupancy 是观察资源使用的一个角度，但不是性能本身。

## [28:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=1680) Fragment：把 CTA 共同持有的寄存器视作逻辑 tile

线程级编程直接管理各线程的私有寄存器。TileLang 中的 fragment，则让程序员描述一个逻辑张量，再由布局决定各元素由哪个线程、哪个寄存器槽位持有。

不能把整个 fragment 理解成“每个线程各有一整份”。它通常是分布在 CTA 各线程中的逻辑 tile；复制行为需要额外区分。

## [29:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=1740) 回顾 Hopper 的资源，理解抽象最终落在哪里

讲者从整卡、SM、subpartition 逐层介绍 HBM、L2、register file、shared memory/L1、warp scheduler、CUDA Core、Tensor Core、load/store 单元及 TMA。

这些资源有物理上限。高层 DSL 不能取消容量、带宽与执行管线的限制；它最终仍要把逻辑任务映射到硬件上。课堂的容量与 SM 数针对展示的 Hopper 配置，不应泛化到所有 GPU。

## [35:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=2100) `T.Parallel` 与 layout inference

程序员声明逻辑迭代域，编译器结合输入输出、循环和 tile operators 的约束，推导 thread ownership 与 thread-local iteration。

需要区分 global tensor 的 shape/stride、shared-memory 物理布局、fragment 的线程/register 分布。它们相互约束，但不是同一种 layout。Layout inference 的价值，是把大量原本需要手工处理的映射一致性问题放进编译期求解。

## [37:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=2220) Transpose：分块、padding 与同步

Transpose 很适合展示访存约束：输入连续的方向与输出连续的方向不同。分块和 shared-memory 中转能让 global 访问两端尽量连续。

Shared-memory padding 则用来打破不利的 bank 周期。例如一整行跨度与 bank 数对齐时，按列访问可能落在同一个 bank；加入适当 padding，可使同列不同元素分布到不同 bank。后续课还会讲 swizzle。

显式 loop layout 可以决定计算由哪些线程执行，但它不会自动替代同步。存在跨线程 shared-memory 读写依赖时，程序员仍需表达同步边界；编译通过不代表没有 race。

## [41:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=2460) Tile primitives 为什么好用

Copy、reduce、GEMM 等原语既减少代码量，也保留 tile region、scope、shape 等高层信息，让它们与 parallel loops 一同参与 layout inference。

`T.copy` 不等于固定使用 TMA。后端需要根据 scope、shape、alignment、target、schedule 选择实现。大块且对齐的数据可能适合某种搬运方式，小块或不满足约束的数据则未必。若自动选择不理想，应检查生成代码，必要时补充 annotation 或改进后端。

## [45:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=2700) 显式 annotation：让打包所需数据属于同一线程

例子把 8 个 E5M6 值打包成 3 个 `uint32`：每值 12 bits，合计 96 bits。如果这 8 个值散落在不同线程，打包就需要额外交换。

通过显式 layout，让同一线程持有所需的连续 8 个值，可匹配 thread-local packing helper。Annotation 是对映射施加约束，不只是提示“希望这样做”。

## [46:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=2760) Reduce 要先问：partial 分布在哪里

少量值如果都在同一线程内，可以直接局部求和；跨线程的 partial 则需要通信。Reducer API 的意义，是用声明式形式包装 partial 累计和最终合并，减少手工处理存储层级与同步的负担。

Replication 则描述逻辑数据如何复制给不同线程，不只与 reduction 有关。选择 API 时，要先确定 ownership、作用域和最终谁需要结果。

## [49:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=2940) 现场问答：`finalize_reducer` 到底做了什么

参与者追问：finalize 是只同步，还是也执行计算？临时 shared-memory workspace 怎样分配？本例是向 global memory 汇总，还是 CTA 内线程之间归约？

讨论中一度使用“加到 global”来描述最终结果，随后有人指出，这个例子的 reducer 需要按 CTA 内 register/shared-memory 的作用域理解，不能直接等同于跨 CTA 的 global reduction。

讲者明确说，自己尚未逐步检查该例的 lowering，现场不能确定完整实现，后续会整理生成代码与实验。因此本段应保留为开放问题，而不是从讨论中任选一句作为 API 的准确契约。

> [!note] 后续学习入口
> 官方课件包含更详细的 reducer/layout/lowering 分析，已另整理到笔记第 6 节。它可以帮助回答这里的问题，但不属于现场已经完成的推导。

## [54:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=3240) Warp scheduler：用其他就绪 warp 隐藏等待

驻留 warp 的上下文保留在片上，因此硬件可以低成本选择另一个 ready warp 发射，不需要像操作系统进程切换那样保存和恢复完整上下文。

更多 occupancy 给 scheduler 更多可选工作，但不代表它们都 ready，也不代表目标执行单元没有瓶颈。寄存器、shared memory、block size 等都可能限制驻留数量。

## [58:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=3480) `T.Pipelined`：显式构造跨迭代重叠

Pipeline 从逻辑迭代的 producer/consumer 关系生成启动、稳态和排空阶段，通过滚动 buffer 等机制增加在途工作。

增大 `num_stages` 可能提前发起更多搬运，帮助隐藏延迟；但也增加 shared memory、register live ranges 和启动/排空开销，甚至降低 occupancy。因此 stages 是 trade-off，不是越大越好。

## [60:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=3600) Stage sweep：先排除编译，再比较稳态时间

讲者展示一个小实验，枚举不同 stages，观察 latency。JIT 冷启动会引入编译成本，应与编译后的运行测量区分。

示例曲线先改善、后趋缓或变差，说明存在与硬件、shape 和资源有关的最优区域。现场对拐点的口头表述有调整，不把某个数字写成普遍最优；具体实验配置与课件结果可在笔记中查看。

## [62:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=3720) Persistent：把任务调度交给长期运行的 workers

Persistent kernel 使用有限 CTA workers 循环领取或处理多个逻辑任务，而不是每个任务都对应一个新 CTA。它提供更直接的任务顺序、跨 task 状态和 producer/consumer 阶段控制。

这并非免费性能：工作队列、尾部任务、负载均衡、长期状态生命周期，都要自己处理。规则的任务流可能用 pipeline 就足够，不必为了使用一个“更高级”的特性而引入 persistent。

## [66:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=3960) Engram backward：看调度语义，不只看 API 名称

讲者选用 TileKernels 中的 Engram backward。它不一定直接调用 `T.Persistent`，但有限 CTA 循环处理多个任务、保留跨任务状态，具有 persistent 的调度思想。

当任务需要自定义次序、长期 partial state 或复杂依赖，persistent 更有表达力。Pipeline 与 persistent 并不互斥：一个强调迭代间重叠，一个强调任务到 worker 的映射与状态生命周期。

MegaKernel 也应从 workload、lifetime 与 dependency 出发评估，而不是只凭名字判断更快。

## [68:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=4080) NCU、Nsight Systems 与 in-kernel profiling

NCU 提供 occupancy、throughput、stall 等指标；Nsight Systems 帮助观察 host launch overhead、kernel timeline、依赖和 overlap。

当一个 kernel 内部阶段很多，聚合计数器可能不够定位问题，细粒度 in-kernel profiling 就有价值。但这些工具是互补的，不是新工具替代旧工具。

## [69:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=4140) 现场问答：祖传 CUDA 能否用 FFI“救一下”

有人问，CuTe DSL 或已有 CUDA kernel 能否利用 TVM-FFI。讨论认为可以从封装路径上考虑，但另一位参与者分享：其测试中 TVM-FFI 的调用性能并没有比 PyTorch binding 快特别多，减少编译和链接依赖也是一个重要优点。

对 launch-bound kernel，应考虑 fusion 等手段，而不是期待换一种 binding 就解决问题。讲者回应：可以在自己的 case 上比较不同路径；fusion 也可能因寄存器或 shared-memory 用量增加而降低 occupancy，因此仍需整体测量。

## [72:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=4320) 用生成代码验证，不让模型分析代替证据

讲者介绍 IKet 一类 in-kernel profiling，并指出所使用的 TileLang 0.1.12 release 尚未包含相应支持，需关注具体版本。

另一个思路是检查 CUDA、PTX 或 SASS，必要时让大模型辅助分析。但从 PTX 改写到期望的 SASS 并不总是简单，也没有“模型读完就能优化”的保证。

若只是确认某个 pass 是否改变实现，往往先看生成 CUDA 或 lowering trace 就足够；性能是否真的改善，还必须实测。

## [75:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=4500) Lowering trace 可以成为实验入口

现场补充，工具不仅能展示 pass 前后的变化，也支持在中间表示或生成代码上编辑、重新编译和测量。这让我们可以直接测试某个编译变换，而不必完全接受讲者对编译器的口头解释。

复杂编译链不应被当作不可观察的黑盒；高层到低层之间的证据越完整，性能归因越可靠。

## [77:00](https://www.bilibili.com/video/BV1rhMD6YEKM/?t=4620) 版本、参考实现与最后回顾

讲者推荐 TileLang examples、TileKernels 等实现，提醒部分示例使用 lazy JIT，语法未必最新。硬件目标与软件版本都会改变 lowering 和最佳参数。

最后再次总结：程序员选择 CTA workload、memory scope 与依赖；fragment/layout inference 减少线程映射负担；tile primitives 保留高层语义；pipeline 与 persistent 处理不同维度的组织问题；最终用 lowering trace、生成代码与 profiler 验证。

关于 reducer 的具体 lowering，仍作为课后确认事项；不要因为 API 简洁，就假定其复制状态、通信范围和同步成本也简单。

## 原始来源留档

- [视频](https://www.bilibili.com/video/BV1rhMD6YEKM/)，01:26:31；[官方课件](https://infra.seminars.lcpu.dev/slides/workshop01.pdf)。
- 原始字幕文件名包含 `BV1rhMD6YEKM_字幕.srt`，保留于 Downloads，未改写。
- SHA-256：`998721a53cc5a64a7a969730a7dfc24362325f8dd0a5575aa4deca72e3d8adaf`。
