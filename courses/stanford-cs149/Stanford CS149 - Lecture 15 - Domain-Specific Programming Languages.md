---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 15
lecture_date: 2023-11-16
area: systems
topics:
  - domain-specific-languages
  - halide
  - liszt
  - scheduling
  - tiling
  - fusion
  - auto-scheduling
  - graph-coloring
aliases:
  - CS149 Lecture 15
  - Domain-Specific Programming Languages
video_url: https://www.youtube.com/watch?v=sRuyBNxCkGQ
---

# Stanford CS149 - Lecture 15 - Domain-Specific Programming Languages

> [!abstract]
> Domain-specific language（DSL）主动放弃“能表达任意程序”的 generality，以换取 productivity 与 performance：用户用领域中自然的 primitives 描述意图，system 因为掌握更多 semantics，能够选择 data representation、parallelization、locality optimization、synchronization 乃至 specialized hardware。Halide 的关键不是少写几个 loop，而是把 image-processing algorithm 与 machine-specific schedule 分离：改 schedule 可探索 tile、vectorize、parallel、compute-at/fusion，却不改变结果。Liszt 则限制程序只能通过 mesh topology/fields 访问数据，从而让 compiler 自动生成 cluster ghost exchange，或把 GPU write conflicts 转成 graph coloring 的多个无原子 parallel phases。

## 来源与范围

- [Lecture 15 视频：Domain-Specific Programming Languages](https://www.youtube.com/watch?v=sRuyBNxCkGQ)
- [官方课件 PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/dsl/14_dsl.pdf)
- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)

课程网页/课件没有把期中复习计入 lecture number，因此 PDF 标作 Lecture 14；公开视频播放列表把 Midterm Review 编为 Lecture 14，本笔记按视频编号记作 Lecture 15。

## 视频索引

| 时间 | 内容 |
|---|---|
| [01:05](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=65s) | 从 C++/ISPC/CUDA 低层实现转向高层 abstractions |
| [02:14](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=134s) | 两个 case studies：Halide 与 Liszt |
| [03:41](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=221s) | Productivity、performance、generality 三轴 |
| [06:09](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=369s) | DSL 主动牺牲 generality |
| [09:13](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=553s) | System 利用 primitive semantics 选择 implementation |
| [11:01](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=661s) | Halide 与 image-processing workload |
| [13:58](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=838s) | 3×3 blur 的 work 分析 |
| [15:14](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=914s) | Separable 2D convolution → 两次 1D convolution |
| [17:13](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=1033s) | 少算术却多 footprint/traffic |
| [20:18](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=1218s) | 只保留三行 temporary 的 fused 方案 |
| [23:46](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=1426s) | 小 buffer 换来的 recomputation |
| [28:29](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=1709s) | Chunking：halo overhead 与 cache capacity 的平衡 |
| [30:41](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=1841s) | Fusion/tiling 可有意识用少量 recomputation 换 locality |
| [31:24](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=1884s) | 怎样再加入 parallelism 与 2D tiling |
| [34:09](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=2049s) | Halide functional definition |
| [37:50](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=2270s) | Bright/gather/realize 与 delayed evaluation |
| [39:54](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=2394s) | Functions 组成 dependency DAG |
| [42:35](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=2555s) | 朴素 compiler：每个 function materialize 成完整 array |
| [44:12](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=2652s) | 真正困难的是快速探索 optimization design space |
| [45:54](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=2754s) | Algorithm 与 schedule 分离 |
| [48:19](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=2899s) | `tile` 对 loop nest 的变换 |
| [50:27](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=3027s) | `vectorize`、`parallel` |
| [51:12](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=3072s) | `compute_root` vs. `compute_at`：producer-consumer fusion |
| [53:47](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=3227s) | Programmer/DSL/compiler 的责任分工 |
| [54:56](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=3296s) | 少量 DSL code 超过 hand-tuned assembly 的原因 |
| [56:47](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=3407s) | Structured schedule space 帮助 auto-scheduling |
| [58:11](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=3491s) | Human vs. auto-scheduler 搜索曲线 |
| [61:16](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=3676s) | 从 DSL 直接生成 FPGA circuit |
| [63:31](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=3811s) | Liszt：面向 mesh simulation 的 DSL |
| [65:49](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=3949s) | Mesh topology 与 fields abstraction |
| [68:13](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=4093s) | 受限访问让 compiler 推导 parallelism/locality/sync |
| [69:52](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=4192s) | Cluster mapping 与 ghost regions/messages |
| [71:28](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=4288s) | GPU mapping：one thread per edge 的 write conflicts |
| [72:36](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=4356s) | Conflict graph coloring 替代 atomics |
| [75:18](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=4518s) | DSL primitives 必须贴合 domain expert 的思维 |
| [76:38](https://www.youtube.com/watch?v=sRuyBNxCkGQ&t=4598s) | 少量、可组合 primitives 才能长久 |

## 1. 为什么需要 DSL

课程前半要求程序员亲自决定：decomposition、loop order、tasks、SIMD lanes、CUDA blocks、shared-memory tiling、synchronization。这样能理解实现，但世界上既懂领域又善于底层性能调优的人很少。

DSL 的价值主张：

```text
domain expert expresses what in natural domain primitives
system/compiler chooses how for a target machine
```

例如：

- SQL：relation/filter/join，system 选择 index、join order、parallel execution；
- PyTorch：tensor/op graph，runtime 选择 CPU/GPU/TPU kernel、layout、fusion；
- Halide：pixel functions/pipeline，schedule 选择 loop transformations；
- Liszt：mesh elements/fields，compiler 选择 graph representation、partition 与 synchronization。

## 2. 三角权衡：productivity、performance、generality

通用语言允许任意 pointer/alias/control flow，因此编译器常无法证明强 transformation 安全；低层高性能语言又把大量 mapping decisions 留给程序员。

DSL 主动限制 expressiveness：

```text
give up some generality
  → expose stronger semantics/invariants
  → compiler can validate, transform, parallelize, specialize
  → productivity + performance
```

> [!important]
> “Domain-specific” 不只是 syntax 更短。真正资产是 representation：它排除了难分析的 programs，并让 system 精确知道每个 primitive 的语义与 data dependencies。

## 3. Case study 1：从 blur 推导 performance design space

### 3.1 直接 2D convolution

3×3 box blur：

$$
out(x,y)=\frac{1}{9}\sum_{i=-1}^{1}\sum_{j=-1}^{1}in(x+i,y+j)
$$

对 `N×N` filter、`W×H` image，work 为：

$$
\Theta(N^2WH)
$$

若 rows 能留在 cache，每个 input 从 DRAM 近似只读一次，原始版本的 traffic 未必像 source-level 9 loads/pixel 那么糟。

### 3.2 利用 separability 少做算术

Box/Gaussian-like separable filter 可拆成 horizontal + vertical：

$$
tmp(x,y)=\frac{1}{N}\sum_i in(x+i,y)
$$

$$
out(x,y)=\frac{1}{N}\sum_j tmp(x,y+j)
$$

Work 从 `N²WH` 降为 `2NWH`。3×3 仅 9→6；100×100 则 10000→200，算法级收益巨大。

### 3.3 但少 FLOPs 不等于快

若完整 materialize `tmp`：

- 多一个 `W×H` allocation；
- input read、tmp write、tmp read、output write；
- memory footprint 增约 50%（相对 input+output）或按课程口径增加一整张 intermediate；
- arithmetic intensity 可能约减半。

Compute-bound 时少算术有利；bandwidth-bound 时多 intermediate traffic 可能更慢。

### 3.4 极小 temporary：locality 好，但重复计算

为了输出一行，只生成所需的 `N` 行 horizontal temporary，然后 vertical reduce。Temporary 可留 cache，但下一输出行又重新生成大部分重叠 rows。

3×3 时每输出行：

```text
3 horizontal rows × 3 ops + 3 vertical ops = 12 ops/pixel
```

比最初 9 ops 更糟。Rolling buffer 能复用 rows，却引入 copy/indirection，并在 row 方向制造 sequential dependency，可能影响 parallelization。

### 3.5 Chunk/tile：在 recomputation 与 cache capacity 之间取中间点

一次产生 `C` 行 output，需要 `C+N-1` 行 temporary。相对 halo overhead：

$$
\frac{N-1}{C}
$$

`C` 越大，recomputation 越少；但 temporary 超过 cache 后，producer-consumer locality 消失。故 tile size 是由 cache capacity、parallelism 与 halo overhead 联合决定，不是越大越好。

二维 tiling 还能在 X/Y 两轴创造 parallel chunks，同时在每块周围支付 halo。

> [!note]
> 本例首次明确展示：最优性能可能故意 **多算一点**，以避免更昂贵的 memory traffic。Work efficiency 与 machine efficiency 不总是一致。

## 4. Halide 的 algorithm：声明“每个点是什么”

Halide 用纯函数定义 stages：

```cpp
Func blurx, out;
Var x, y;

blurx(x, y) = (in(x-1,y) + in(x,y) + in(x+1,y)) / 3;
out(x, y)   = (blurx(x,y-1) + blurx(x,y) + blurx(x,y+1)) / 3;
```

特点：

- 无显式 loops/allocations；
- 每个 `Func` 定义坐标到值的 mapping，不等同于已经 materialize 的 array；
- functions 构成 dependency DAG；
- `realize(width,height)` 才要求计算某个 domain；
- boundary condition 可作为高层 primitive，compiler 可生成 interior fast path + boundary code；
- functional semantics 限制 alias/mutation，使 loop movement 更容易证明安全。

朴素 compiler 可以每个 `Func` 分配完整 buffer、按 DAG 顺序各跑一个 loop nest，语义正确但可能性能差。

## 5. Halide 的 schedule：声明“怎样执行”

Algorithm 与 schedule 分离：

```text
algorithm changes → output semantics may change
schedule changes  → output semantics must not change
```

### 5.1 `tile`

```cpp
out.tile(x, y, xo, yo, xi, yi, 256, 32);
```

把原来的 `(x,y)` loop nest 变为 outer tile loops `(xo,yo)` + inner loops `(xi,yi)`，tile 256×32。Schedule 为生成的 loops 命名，供后续 directives 引用。

### 5.2 `vectorize` 与 `parallel`

```cpp
out.vectorize(xi, 8);
out.parallel(yo);
```

- inner X 以 8-wide SIMD 生成；
- outer Y tiles 分给 thread pool；
- programmer 指定 high-level mapping，compiler 负责 target ISA、tail/boundary、index arithmetic。

### 5.3 `compute_root` vs. `compute_at`

```cpp
blurx.compute_root();       // complete producer buffer first
blurx.compute_at(out, xo);  // produce only data needed by each output tile
```

`compute_at` 将 producer loop nest 嵌入 consumer 的某层，控制：

- intermediate allocation size；
- producer-consumer locality；
- recomputation/halo；
- available parallelism。

放得越内层，temporary 越小、locality 越强，但重复计算越多；放得越外层则相反。这正是前面手工 blur 推导的 design space。

## 6. 为什么高层 schedule 可能胜过手写 assembly

单个 inner loop 的 hand-written assembly 可能优于 compiler 10–30%，但 expert 在低层代码中尝试一个全新 tiling/fusion/layout 往往需要一天；Halide schedule 改几个参数即可重新生成完整、正确的版本。

因此总体结果可能是：

```text
slightly worse local code generation
+ much broader/faster global schedule exploration
= better end-to-end performance
```

Production value还包括 portability：同一 algorithm 可为 x86、ARM、GPU 等维护不同 schedules，而不是复制整套 algorithm code。

## 7. Structured design space 与 auto-scheduling

Schedule language 把 optimization 表示成结构化 choices：loop order、tile sizes、compute location、vector width、parallel loop。它不仅帮助人，也给 automated search 清晰的 search space。

视频展示人类专家在一小时中不断改 schedule、profile、看 `objdump`，performance 有升有降；auto-scheduler 用 tree search/cost model 很快给出强 baseline，三个应用中短时对比赢了两个。要点不是“AI 永远胜专家”，而是：

- good representation 将无穷 code space 压成有意义的 schedule space；
- correctness 由 compiler 保证，search 只需优化 performance；
- domain/target cost model 可复用到许多 pipelines。

## 8. 从 DSL 直接生成 specialized hardware

若 compiler 已掌握 algorithm DAG 与 memory access semantics，为什么一定要落到 general-purpose ISA？可直接生成 FPGA spatial pipeline/circuit：

```text
DSL graph → hardware pipeline/dataflow → FPGA
```

ISA 是 software/hardware 的通用接口，也带来 fetch/decode/control overhead；domain-specific representation 可以绕过它，执行更专用的 data path。代价是 compilation、hardware resources、debugging 与 deployment complexity。

## 9. Case study 2：Liszt 的 mesh abstraction

Scientific simulation 以 mesh 为自然对象。Liszt 不暴露 pointer-based graph representation，而提供：

- iterate vertices/edges/cells；
- 从 element 查询 topology neighbors；
- 在 mesh elements 上定义 fields（temperature、position、flux 等）；
- 读写 fields。

程序员表达：

```text
for each edge e:
    read fields on adjacent vertices
    compute flux
    update fields on adjacent vertices
```

不能写任意 data-dependent pointer arithmetic，这个限制让 compiler 知道每个 iteration 的 read/write footprint。

## 10. 同一 Liszt program 映射到两类机器

### 10.1 Distributed cluster

Compiler/runtime：

1. partition 大 mesh，尽量 balance elements；
2. 根据 topology 找跨 partition dependencies；
3. 为边界建立 ghost/halo copies；
4. 每 time step 生成 message exchange；
5. 对 local elements 并行执行。

Domain scientist 不写 MPI pack/send/receive/unpack，但 DSL 的语义足以自动产生。

### 10.2 GPU：one thread per edge 遇到 multi-writer

不同 edges 可能更新同一 vertex，直接 parallel 会 race。简单方案是 atomic update；若 atomics 太贵，compiler 可建立 conflict graph：

```text
one node = one mesh edge iteration
conflict edge = two iterations write same vertex
```

对 conflict graph coloring，使相邻 nodes 不同色；按颜色依次执行多个 parallel loops：

```text
parallel all blue edges   // pairwise non-conflicting
barrier
parallel all orange edges
...
```

这用 preprocessing + 多 phases 换掉每 update atomic。颜色数增加 serial phases，coloring/metadata 也有成本；但对重复 time steps，预处理可 amortize。

## 11. 设计 DSL 的原则

### 11.1 Representation 必须贴合用户心智模型

- 图像工程师想 pixels/pipelines；
- 物理学家想 mesh/fields；
- ML 工程师想 tensors/operators；
- 数据工程师想 tables/relations。

若 abstraction 不自然，productivity 目标失败；若 abstraction 遮蔽最佳 implementation 所需信息，performance 目标失败。

### 11.2 从最佳实现反推 front-end 需要暴露什么

设计流程：

1. 写 domain expert 最自然的 program；
2. 写目标硬件上理想的实现；
3. 问 compiler 从前者到后者缺什么 information；
4. 只增加能声明这些 information 的最小 abstractions。

### 11.3 少量、可组合 primitives

每加一个 primitive 都扩大 compiler/optimizer 的组合状态、correctness surface 与 backend 数量。好系统倾向少量、精确定义、可组合 primitives；真正的力量来自 composition，用户最终会发现设计者没预料到的用法。

## 12. 与 AI Infra 的连接

- PyTorch/JAX/Triton/TVM/MLIR 都延续“高层 tensor semantics + target-specific schedule/lowering”的路线。
- TorchInductor/XLA 的 fusion 与 Halide `compute_at` 同构：控制 intermediate materialization、tile lifetime 与 producer-consumer locality。
- Triton 把 schedule space 缩成 program IDs、block shapes、layouts；auto-tuner 在结构化 config space 中搜索，与 Halide auto-scheduler 思路一致。
- LLM compiler 的难点不是把 Python 翻成 CUDA，而是选择 representation，使 shape、layout、reduction、mask、precision 与 side effects 都足够明确。
- Graph/mesh coloring 展示了 compiler 可把 data dependence 转为 execution schedule；Mixture-of-Experts routing、sparse kernels、serving admission control 也常做相似预处理。
- DSL 直接生成 FPGA/ASIC dataflow 是 hardware-software co-design 的入口，下一讲 hardware specialization 会继续。

## 13. 本讲结论

1. DSL 用较少 generality 换更高 productivity 与 performance；收益来自 semantics，不只是简短语法。
2. Separable blur 表明 FLOPs、footprint、traffic、recomputation 与 parallelism 必须一起优化。
3. Halide 把 algorithm 与 schedule 分离，schedule changes 不改变结果。
4. `tile/vectorize/parallel/compute_at` 是对 loop nest 与 producer-consumer placement 的高层控制。
5. Structured schedule space 同时帮助 human iteration 与 automated search。
6. Liszt 通过受限 mesh API 获得依赖信息，可自动生成 cluster communication 或 GPU conflict-free phases。
7. 好 DSL 的 primitives 少、自然、可组合，并保留通向已知最佳实现的路径。

## 14. 自测题

1. 为什么 separable convolution 减少 FLOPs 后仍可能更慢？
2. 对 tile height `C`、filter size `N`，halo recomputation overhead 如何随 `C` 变化？
3. Halide `Func` 与已经 materialize 的 2D array 有什么区别？
4. `compute_root` 与 `compute_at` 分别如何改变 footprint、locality、recomputation？
5. 为什么 schedule language 能让 auto-scheduler 比直接搜索 C++ code 更可行？
6. 一个 schedule change 为什么理论上不应改变 image output？这一保证依赖什么 language restrictions？
7. Liszt compiler 为什么能推导 ghost cells，而普通 C++ compiler 通常不能？
8. GPU mesh update 中 graph coloring 与 atomics 的 tradeoff 是什么？
9. 为什么“primitive 越多越方便”常会破坏 DSL 的长期可优化性？
10. 任选 PyTorch/JAX/Triton，分别指出它的 domain objects、restricted operations 与 system 可利用的 semantics。

## 相关笔记

- [[Stanford CS149 - Lecture 06 - Performance Optimization II Locality Communication and Contention]]
- [[Stanford CS149 - Lecture 08 - Data-Parallel Thinking]]
- [[Stanford CS149 - Lecture 10 - Efficiently Evaluating DNNs on GPUs]]
- [[Stanford CS149 - Lecture 14 - Midterm Review]]
- [[Stanford CS149 - Lecture 18 - Hardware Specialization]]
