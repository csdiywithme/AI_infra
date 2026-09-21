---
type: course-note
status: developing
course: Weiming HPC Training Camp × LCPU AI Infra Seminars
lecture: 3
lecture_date: 2026-08-02
area: systems
topics:
  - gpu
  - cuda
  - tensor-core
  - mma
  - wgmma
  - tcgen05
  - shared-memory
  - tensor-memory
  - low-precision
aliases:
  - Tensor Core 从 mma.sync 到 tcgen05
  - LCPU Session 03
video_url: https://www.bilibili.com/video/BV1LxM96eE43/
slides_url: https://infra.seminars.lcpu.dev/slides/session03.pdf
---

# LCPU AI Infra Seminars - Session 03 - Tensor Core：从 `mma.sync` 到 `tcgen05`

> [!abstract] 本讲一句话
> Tensor Core 的核心不是“一条更快的矩阵乘指令”，而是一份不断演进的软硬件数据契约：Ampere 用 warp 寄存器 fragment 驱动 `mma.sync`，Hopper 用 warpgroup、SMEM descriptor 和异步 `wgmma` 扩大计算块，Blackwell 再引入 TMEM 与 `tcgen05`，把 accumulator、scale factor 和跨 CTA 协作显式纳入数据通路。

## 来源与范围

- [讲座视频：Tensor Core 从 `mma.sync` 到 `tcgen05`](https://www.bilibili.com/video/BV1LxM96eE43/)
- 视频时长：02:40:12
- 主讲：孙远航
- [官方 Slides（136 页）](https://infra.seminars.lcpu.dev/slides/session03.pdf)
- [课程日历与讲座简介](https://infra.seminars.lcpu.dev/schedule)
- [本讲 Wiki](https://infra.seminars.lcpu.dev/wiki/QsOcw3uHpifm10k5ouCcK6Uonee)
- 精编字幕：[[LCPU AI Infra Seminars - Session 03 - Tensor Core - 精编字幕]]

本文以用户提供的 SRT 为主，用官方课件校正指令、布局和架构术语。重点是 Tensor Core 的 programming model 与数据通路；具体 PTX 语法、支持 shape 和架构限定仍应以对应版本 PTX ISA 为准。

## 视频索引

| 时间 | 内容 |
|---|---|
| [00:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=0) | 两个核心问题：Tensor Core 是什么，怎样为它准备数据 |
| [04:30](https://www.bilibili.com/video/BV1LxM96eE43/?t=270) | 从计算强度理解专用矩阵单元的价值 |
| [09:50](https://www.bilibili.com/video/BV1LxM96eE43/?t=590) | 为什么 Tensor Core 不直接接收完整 tensor |
| [11:30](https://www.bilibili.com/video/BV1LxM96eE43/?t=690) | 读懂 `mma.sync.aligned.m16n8k16` |
| [21:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=1260) | 单条 MMA 的计算强度推导 |
| [25:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=1500) | A100 Tensor Core 峰值推导 |
| [29:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=1740) | 三代演进：瓶颈逐渐从算术转向数据供给 |
| [42:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=2520) | Fragment：逻辑矩阵到分布式寄存器片段 |
| [57:15](https://www.bilibili.com/video/BV1LxM96eE43/?t=3435) | 为什么 fragment layout 是固定的 |
| [65:47](https://www.bilibili.com/video/BV1LxM96eE43/?t=3947) | SM80 手写路径与 `ldmatrix` |
| [78:45](https://www.bilibili.com/video/BV1LxM96eE43/?t=4725) | SM80 小结：GMEM → SMEM → registers → MMA |
| [80:08](https://www.bilibili.com/video/BV1LxM96eE43/?t=4808) | SM90 为什么引入 `wgmma` |
| [87:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=5220) | 异步执行、async proxy 与可见性规则 |
| [91:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=5460) | `commit_group` / `wait_group` |
| [98:26](https://www.bilibili.com/video/BV1LxM96eE43/?t=5906) | Matrix descriptor 与 canonical layout |
| [109:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=6540) | SMEM bank conflict 与 swizzle |
| [119:30](https://www.bilibili.com/video/BV1LxM96eE43/?t=7170) | SM100：TMEM 改写 accumulator 数据通路 |
| [124:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=7440) | `tcgen05.mma`、instruction descriptor 与 `mbarrier` |
| [137:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=8220) | 2-CTA MMA 与跨 CTA 数据复用 |
| [140:24](https://www.bilibili.com/video/BV1LxM96eE43/?t=8424) | FP8 / FP4 与 scale factor |
| [144:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=8640) | DeepSeek-V3 的细粒度缩放思路 |
| [147:50](https://www.bilibili.com/video/BV1LxM96eE43/?t=8870) | Blackwell 硬件 block scaling |
| [149:50](https://www.bilibili.com/video/BV1LxM96eE43/?t=8990) | 存储量化与计算量化的区别 |
| [152:46](https://www.bilibili.com/video/BV1LxM96eE43/?t=9166) | 总结：数据位置、描述方式和同步责任的演进 |

## 1. Tensor Core 的价值：提高每次数据搬运产生的计算量

### 1.1 朴素逐元素 FMA 的算术强度很低

若每次只读两个 FP32 操作数并做一次 FMA，可粗略写成：

$$
I=\frac{2\ \mathrm{FLOP}}{8\ \mathrm{B}}=0.25\ \mathrm{FLOP/B}
$$

这不是完整 GEMM kernel 的 roofline 模型，而是说明“读两个数只用一次”的代价。GEMM 真正有价值的性质是：计算量为 $O(n^3)$，输入/输出数据量为 $O(n^2)$。只要让一个 tile 在片上反复参与乘加，算术强度就会随着复用增大。

Tensor Core 把固定小块 $D=A\times B+C$ 做成专用硬件，但仍嵌在 SM、warp、register 和 shared-memory 层级中。CUDA Core 继续处理地址生成、循环、边界、缩放与后处理。它因此兼具专用计算密度和 GPU 的可编程性。

### 1.2 单条 `m16n8k16` 的计算强度

对 `mma.sync.aligned.m16n8k16`：

$$
16\times 8\times 16=2048\ \mathrm{MAC}=4096\ \mathrm{FLOP}
$$

若 A、B 为 FP16，D 为 FP32，按本次 MMA 的输入和结果 payload 计算：

$$
\begin{aligned}
\mathrm{bytes}
&=16\times16\times2+16\times8\times2+16\times8\times4\\
&=1280\ \mathrm{B}
\end{aligned}
$$

所以：

$$
I_{mma}=\frac{4096}{1280}=3.2\ \mathrm{FLOP/B}
$$

这个数只描述一条指令边界上的 operand payload；完整 kernel 还会通过寄存器 accumulator 复用、SMEM tile 复用和多级 cache 进一步改变实际 HBM 算术强度。

### 1.3 A100 峰值的数量级推导

讲座用“每 SM 有多少 Tensor Core、每个 Tensor Core 每周期完成多少乘加、GPU 有多少 SM、频率多少”来恢复 A100 FP16 Tensor Core 的理论峰值，得到约：

$$
311.9\ \mathrm{TFLOP/s}
$$

若 HBM 带宽粗略按 $2\ \mathrm{TB/s}$ 估算，达到峰值所需算术强度约为：

$$
\frac{311.9\ \mathrm{TFLOP/s}}{2\ \mathrm{TB/s}}
\approx156\ \mathrm{FLOP/B}
$$

这解释了为什么“会调用 Tensor Core”远远不等于“能接近 Tensor Core 峰值”：必须用 tiling 和 reuse 把有效算术强度从个位数提高两个数量级，并同时让片上数据供给跟上计算吞吐。

## 2. `mma.sync`：warp 共同持有一组分布式 fragment

### 2.1 读懂指令签名

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

| 字段 | 含义 |
|---|---|
| `mma` | Matrix multiply-accumulate |
| `sync` | 指令在 warp 粒度同步执行 |
| `aligned` | warp 中参与 threads 必须共同、收敛地执行 |
| `m16n8k16` | $A:16\times16$，$B:16\times8$，$D:16\times8$ |
| `row.col` | A row-major，B column-major；描述逻辑布局 |
| `f32.f16.f16.f32` | D/A/B/C 的 element types |

“一条 warp-level 指令”不代表每个 thread 都拥有完整矩阵。32 个 lanes 各自在 registers 中持有一小部分元素，这些片段合起来才构成 A、B、C、D。

### 2.2 Fragment 是分布式 ABI

可以把 fragment 理解成 Tensor Core 指令与 warp registers 之间的 ABI：

```text
logical tile
    ↓ fixed lane/register mapping
32 per-thread register fragments
    ↓ one converged mma.sync
logical output tile
```

固定 mapping 有三个作用：

- 硬件能在不携带任意 gather/scatter 网络的前提下取到所需元素；
- 编译器和库能预先安排 register packing；
- 数据装载指令可以直接生产 MMA 所需的 register 形态。

这也意味着手写 PTX 时不能只保证“数值都在寄存器里”，还必须保证每个元素位于规定的 lane 和 register slot。

### 2.3 为什么 A/B 沿 K 维连续

矩阵乘中 reduction 发生在 K 维。让 A 的 row 和 B 的 column 沿 K 连续，能让一个较小的连续块直接对应本次 dot-product 所需的数据。逻辑 layout、SMEM layout 和 fragment layout 是三个不同层次，不能只用“row-major/column-major”概括全部物理排列。

## 3. SM80 数据路径：`ldmatrix` 把 SMEM tile 变成 fragment

Ampere 的典型手写路径是：

```text
GMEM ──cooperative copy──> SMEM
SMEM ──ldmatrix──────────> A/B register fragments
register C ──────────────> accumulator fragment
all fragments ──mma.sync─> register D
```

### 3.1 `ldmatrix` 解决的不是普通 load 问题

普通 scalar/vector load 按线程地址取数；`ldmatrix` 按一组 lanes 的协作协议，从 SMEM 中取若干 $8\times8$ 小矩阵并转置/打包成 Tensor Core 所需的 register fragment。其价值包括：

- 减少手工 lane shuffle 和 register permutation；
- 让 memory transaction 与 fragment layout 对齐；
- 为 A/B 的不同逻辑布局提供对应变体。

`ldmatrix` 仍然受 SMEM bank conflict 影响。地址正确只能保证 correctness，不能保证一次 warp-level load 以理想 transaction 数完成。

### 3.2 SM80 的主要压力落在通用 registers

A、B、C、D 都要在 registers 中表达。扩大 tile 或增加 pipeline stages 会快速增加 register footprint，进而影响 occupancy；同时每条 MMA 的发起与数据准备仍由 warp 完成。这正是下一代把 operand 读取交给 SMEM descriptor、再把 accumulator 移到 TMEM 的背景。

## 4. SM90：`wgmma` 把发起粒度扩大到 warpgroup

Hopper 的 `wgmma` 由 4 个 warps，即 128 threads 组成的 warpgroup 协作。常见数据位置是：

- A：register 或 SMEM；
- B：SMEM；
- C/D accumulator：register；
- SMEM operands：由 64-bit matrix descriptor 描述。

与 SM80 相比，A/B 不必全部先复制进每个 lane 的 fragment registers，Tensor Core 数据通路可按 descriptor 直接读取 SMEM。更大的 M/N shape 也提高了一次指令的工作量。

### 4.1 异步不是“指令返回就能读结果”

`wgmma` 通过 async proxy 执行。程序需要组织：

```text
wgmma.mma_async ...
wgmma.commit_group
... independent work ...
wgmma.wait_group N
```

- `commit_group` 把之前发出的 MMA 归入异步组；
- `wait_group N` 确保最多只留下 N 组未完成；
- 在正确 wait 之前，不能把 accumulator 当作已完成结果使用；
- generic proxy 对 SMEM 的写入若要被 async proxy 读取，需要满足 PTX 规定的 proxy ordering / fence 契约。

因此 source-code 顺序、warp convergence、操作 completion 和跨 proxy visibility 是不同问题。

### 4.2 Matrix descriptor 是地址与布局契约

Descriptor 不携带整个矩阵，而是把 Tensor Core 找到 SMEM tile 所需的关键信息压进 64 bits，例如：

- start address；
- leading-byte offset（LBO）；
- stride-byte offset（SBO）；
- base offset / layout information；
- swizzle mode。

部分地址字段以 16-byte 为单位编码，因此构造时会表现为右移 4 bits；这不是任意位运算，而是利用强制 alignment 扩大有限字段的可寻址范围。

### 4.3 Canonical layout 与 128-byte core matrix

课件用 $8$ 行 × 每行 $16$ bytes 的 128-byte core matrix 解释 descriptor 寻址。更大的 operand tile 可由多个 core matrices 沿 K、M/N 方向拼接。LBO/SBO 描述的是这些块之间的物理步长，而不是简单等同于高级语言二维数组的 leading dimension。

验证 descriptor 时，最好先画出：

1. logical coordinates；
2. canonical core-matrix 分块；
3. 每块的 byte address；
4. swizzle 后的 physical bank-group mapping。

### 4.4 Swizzle 是为 consumer 消除 bank conflict

SMEM 有 32 个 banks，每个 bank 每周期服务一个 4-byte word；可把连续 4 banks 看成 16-byte bank group。若多个请求落到同一 bank，访问会被拆分。

Swizzle 不改变 logical tensor 的值，而是用行坐标中的若干 bits 对 bank-group bits 做 XOR 类映射：

$$
g_{physical}=g_{logical}\oplus f(row)
$$

它的目标是把 WGMMA 同一拍要读的地址分散到不同 banks。布局应从 consumer 的访问模式反推；“GMEM 中连续，所以 SMEM 中也原样连续”并不总是性能最优。

## 5. SM100：TMEM 与 `tcgen05` 重新定义 accumulator 生命周期

Blackwell 每个 SM 引入约 256 KB Tensor Memory（TMEM）。课件把它画成 128 lanes × 512 columns 的结构，每个 cell 为 32 bits。TMEM 由 Tensor Core 数据路径高带宽访问，减少大 accumulator 对通用 registers 的占用和搬运压力。

### 5.1 TMEM 是显式管理的片上资源

程序需要：

1. 分配连续 columns，数量为不小于 32 的 2 次幂；
2. 在 `tcgen05.mma` 中把结果写入 TMEM；
3. 用 `tcgen05.ld` 把需要的结果读回 registers，或用 `tcgen05.cp/st` 走其他路径；
4. 在所有使用者完成后显式 deallocate。

例如 `m128n256` 的 FP32 accumulator 有：

$$
128\times256\times4=128\ \mathrm{KiB}
$$

即占用约一半 TMEM。TMEM 容量因而直接限制 CTA 内同时存活的 accumulator tiles 与 pipeline depth。

### 5.2 `tcgen05.mma` 把动态信息和静态配置分开

与每个 lane 持有 operand fragment 不同，`tcgen05.mma` 可由单个 thread 发起。指令使用：

- TMEM address 指向 accumulator / destination；
- SMEM descriptors 指向 A/B；
- instruction descriptor（idesc）编码 shape、data type、transpose、scale 等控制信息。

这种接口更像“提交一个矩阵计算 transaction”，但它仍依附 CTA/cluster 的内存、同步和资源生命周期，并不是独立 kernel launch。

### 5.3 `mbarrier` 承担完成通知

`tcgen05.mma` 是异步的。程序通常让它在完成时 arrive 一个 `mbarrier`，consumer 等待对应 phase 后再读取 TMEM。正确性仍需同时证明：

- A/B 的生产者已经完成写入；
- async proxy 可观察到这些 writes；
- MMA 已完成并通知正确的 barrier generation；
- TMEM columns 在 consumer 完成前没有被覆盖或释放。

### 5.4 2-CTA MMA 是空间上的 B 复用

2-CTA 模式让 cluster 中两个 CTAs 协作完成更大的 M 方向工作，并复用同一个 B tile。它减少 B 的重复搬运，但增加了：

- cluster placement 和 launch constraints；
- 两个 CTA 共同进入协议的要求；
- shared/TMEM 地址和 barrier ownership 的复杂度；
- 尾部、负载不均与资源占用风险。

因此 2-CTA 不是无条件更快，而是在 B traffic 足够关键、两侧工作足够对称时，用额外协作换取空间复用。

## 6. 三代 Tensor Core 的统一对比

| 维度 | SM80 `mma.sync` | SM90 `wgmma` | SM100 `tcgen05` |
|---|---|---|---|
| 主要发起者 | 1 warp / 32 threads | 1 warpgroup / 128 threads | 可由 1 thread 发起，CTA/cluster 共同遵守协议 |
| A/B 位置 | registers | A 可在 registers/SMEM，B 通常在 SMEM | SMEM，由 descriptors 描述 |
| Accumulator | registers | registers | TMEM |
| 数据布局接口 | 固定 per-lane fragments | 64-bit matrix descriptors | matrix descriptors + instruction descriptor |
| 典型 shape 粒度 | 较小，如 `m16n8k16` | 更大 M/N shape | 更大 shape，并支持 1-CTA / 2-CTA |
| 完成模型 | 同步 warp-level instruction | async group，commit/wait | async + `mbarrier` completion |
| 程序员核心责任 | 正确装载 fragment、warp convergence | descriptor、swizzle、proxy ordering、warpgroup 协议 | TMEM 生命周期、idesc、barrier phase、CTA/cluster 协议 |
| 主要资源压力 | register capacity/bandwidth | SMEM bandwidth、register accumulators | SMEM + TMEM capacity/bandwidth、cluster coordination |

> [!note] 我的理解：演进的重点不是矩阵乘公式改变了
> 三代硬件都在做小块 $A\times B+C$。真正持续变化的是数据放在哪里、谁发起、怎样描述布局、怎样报告完成。A/B 逐步减少对通用 register transit 的依赖，accumulator 最终被搬到 TMEM；同时 issue cardinality 从 warp 到 warpgroup，再到单线程提交异步 transaction。通用寄存器与发起指令的压力下降了，但 descriptor、proxy、barrier 和资源生命周期的责任变得更显式。

## 7. 低精度：scale factor 是隐藏的第三类输入

### 7.1 FP8 格式在范围与精度之间交换

课件比较两类常见 FP8：

| 格式 | Exponent | Mantissa | 课件给出的最大有限值 | 特点 |
|---|---:|---:|---:|---|
| E4M3 | 4 bits | 3 bits | 448 | 精度更好，范围较小 |
| E5M2 | 5 bits | 2 bits | 57344 | 范围更大，精度更低 |

FP4 的表示范围和有效精度更紧。直接把一整张 tensor 用同一个 scale 压入低精度，少数 outliers 会把 scale 拉大，使大量普通值挤在很少的可表示 levels 中。

### 7.2 从 per-tensor 到 block scaling

量化可概括为：

$$
x_q=Q\left(\frac{x}{s}\right),\qquad \hat{x}=s\cdot x_q
$$

Per-tensor scaling 只保存一个 $s$，元数据少，但容易被 outlier 支配。细粒度 block scaling 为小块保存独立 scale，能贴合局部分布；代价是更多 scale metadata、更多寻址和更复杂的数据布局。

对矩阵乘，scale block 不能随意切。若希望 Tensor Core 在 K 方向先累加一段再应用 scale，scale 必须在对应 K segment 内保持一致，否则：

$$
\sum_k (s_{A,k}A_{q,k})(s_{B,k}B_{q,k})
$$

无法化为一次统一缩放。硬件 block scaling 因此把支持的 block shape、K granularity 与 scale-factor layout 写进指令契约。

### 7.3 累加精度与 promotion

输入是 FP8/FP4 不等于内部所有步骤都用同样低的精度。Hopper 的低精度 MMA 有自己的乘积与 accumulator 数据通路；当长 K accumulation 的舍入误差不可接受时，可把部分和定期 promotion 到更高精度 accumulator。

以讲座介绍的 DeepGEMM 思路为例，可每若干个 `k32` 小段把部分和提升到 FP32；promotion 越频繁，数值误差越小，但搬运与额外指令开销越大。到了计算吞吐更高的 Blackwell，相同 promotion 工作可能占据更高的相对时间，因此硬件原生 block scaling 与 TMEM 数据路径更重要。

> [!note] 我的理解：低精度 MMA 实际上有三类输入
> 除 A、B 之外，scale factors 决定量化整数/浮点编码怎样映射回实际数值。它们有自己的精度、布局、带宽、cache/TMEM placement 和同步需求。只计算 A/B bytes 而忽略 scales，会低估 block-scaled kernel 的数据供给成本。

### 7.4 存储量化不等于计算量化

- **存储量化**：权重在 HBM 中压缩，加载后反量化到 FP16/BF16，再用较高精度 MMA。主要收益是减少容量和 memory traffic，不改变最终 MMA 的精度类别与对应理论峰值。
- **计算量化**：Tensor Core 直接接受 FP8/FP4 operands，并在硬件路径中应用 scales。它既减少 operand traffic，也使用更高吞吐的低精度计算单元，但数值和 scale orchestration 更复杂。

两者可以组合，但评估性能时必须区分瓶颈来自 HBM bytes、反量化指令，还是 Tensor Core throughput。

## 8. 把三代接口看成越来越声明式的数据契约

> [!note] 我的推论
> `fragment → matrix descriptor → instruction descriptor` 可以看成逐步声明式的接口演进：SM80 让每个 lane 亲自把元素放入指定 register slot；SM90 让软件描述 SMEM 中的数据位置与布局；SM100 再把运算 shape/type/scale 和 TMEM destination 描述为 transaction。硬件获得更大自由去调度数据通路，软件则必须更精确地构造 descriptor 和同步协议。

这也说明优化 Tensor Core 的本质是 data orchestration：

```text
GMEM tile
  ↓ transfer / reuse
SMEM physical layout
  ↓ fragment or descriptor
Tensor Core issue
  ↓ asynchronous completion
register or TMEM accumulator
  ↓ epilogue / store
GMEM result
```

算术只在中间一格。其余大部分工程工作都在决定 tile 的位置、布局、所有权和时间顺序。下一讲 [[LCPU AI Infra Seminars - Session 04 - Pipeline Ordering Data Orchestration]] 正是把这些 ordering 问题系统化。

## 9. 从 AI Infra 五个视角重新理解本讲

| 视角 | 本讲问题 | 典型设计量 |
|---|---|---|
| Shape | 一次 MMA 覆盖多大的 M/N/K？ | instruction shape、CTA tile、K granularity |
| Compute | 每周期能完成多少 MAC？ | datatype、Tensor Core 数量、issue rate、promotion cost |
| Memory | Operands/accumulators/scales 放在哪里？ | GMEM、SMEM、register、TMEM、swizzle、reuse |
| Communication | 哪些 agents 共享和传递数据？ | warp、warpgroup、CTA、2-CTA cluster、proxy |
| Runtime/System | 如何选择并调度正确 kernel？ | compute capability、shape specialization、workspace、tail policy |

一个 Tensor Core kernel 的性能问题通常不是单独属于某一行。例如增加 CTA tile 能提升 reuse，却会同时增加 SMEM/TMEM footprint；选择 2-CTA 能复用 B，却会改变 cluster scheduling 与尾部行为。

## 10. 手写或审查 Tensor Core kernel 的检查清单

### 10.1 Correctness

- [ ] 当前 compute capability 是否支持目标 MMA shape、datatype 与 qualifier？
- [ ] 所有 required lanes/warps/CTAs 是否以规定方式共同执行？
- [ ] A/B/C/D 的 logical shape、leading dimension、transpose 与 instruction signature 是否一致？
- [ ] SM80 fragment 中每个元素是否进入正确 lane/register slot？
- [ ] SM90/SM100 descriptor 的 base、LBO、SBO、alignment 与 swizzle bits 是否正确？
- [ ] Generic writes 对 async proxy 是否按 PTX 规则可见？
- [ ] `commit_group` / `wait_group` 或 `mbarrier` phase 是否覆盖真正的数据依赖？
- [ ] TMEM allocation 大小、地址、生命周期与 deallocation 是否匹配？
- [ ] Scale-factor block shape、K granularity、datatype 和 placement 是否与 MMA mode 一致？
- [ ] 边界 tile、非整除 K 和 2-CTA 尾部是否有明确定义？

### 10.2 Performance

- [ ] HBM 算术强度是否足以接近目标 Tensor Core roofline？
- [ ] GMEM → SMEM copy 是否 coalesced，并有足够 reuse？
- [ ] SMEM layout 是否服务于 `ldmatrix` / `wgmma` / `tcgen05` 的实际访问，而非只追求视觉连续？
- [ ] Bank conflicts 是否通过 padding/swizzle 消除？
- [ ] Register、SMEM、TMEM footprint 是否让 occupancy 或并发 stages 过低？
- [ ] Async MMA 前后是否有足够 independent work，还是发出后立刻 wait？
- [ ] Descriptor/address-generation 工作是否落在关键路径？
- [ ] Promotion 与 scale loading 是否成为低精度 kernel 的新瓶颈？
- [ ] 2-CTA 模式节省的 B traffic 是否大于 cluster coordination 成本？
- [ ] Nsight Compute 中的 Tensor Core utilization、SMEM transactions、eligible warps 和 stall reasons 是否与性能假设一致？

## 11. 自测题

1. 为什么 Tensor Core 能提高计算密度，却不能自动让一个朴素 GEMM 达到峰值？

    **面试回答：** Tensor Core 提高的是矩阵乘计算吞吐，朴素实现却可能反复搬运输入、缺少复用，使计算单元一直等数据。要靠 tiling、合适的 fragment/SMEM 布局和异步流水保证供数，并兼顾寄存器、同步和尾部浪费，才能接近峰值。

2. `mma.sync.aligned.m16n8k16` 的 M/N/K 分别决定 A、B、D 的什么形状？

    **面试回答：** 这条指令做 $D=A B+C$：$M=16,N=8,K=16$，所以 A 为 $16\times16$，B 为 $16\times8$，C、D 为 $16\times8$。这些是整个 warp 协作完成的逻辑矩阵形状，不是每个线程各自持有的形状。

3. 为什么 fragment 不能只理解为“线程私有的一小块矩阵”？

    **面试回答：** Fragment 是逻辑 tile 到多个 lane/register slots 的分布式映射，构成 MMA 指令的数据接口。每线程确实持有局部片段，但局部片段不能任意安排；必须让所有元素出现在指令规定的 lane 和槽位，且参与线程共同执行。

4. `ldmatrix` 相比普通 vector load 额外承担了什么布局转换责任？

    **面试回答：** `ldmatrix` 按 warp 协作协议把 SMEM 小矩阵读入规定的 lane/register 布局，相关变体还能转置，直接准备 MMA 所需 fragment。普通 vector load 只扩展单线程读取宽度，不自动完成这种跨线程分发；`ldmatrix` 仍需正确对齐并避免 bank conflict。

5. 为什么 SM80 增大 tile 或 pipeline depth 容易碰到 register pressure？

    **面试回答：** SM80 的 MMA operands 和 accumulator 都要占通用寄存器，更大 tile 会增加 accumulator，更多在途操作可能延长 operand 和地址状态的存活期。寄存器超预算会降低 occupancy 或 spill；若流水缓冲主要在 SMEM，首先增加的也可能是 SMEM 压力，要看生成代码确认。

6. `wgmma` 已经发出后，为什么源码中下一行也不能立刻安全读取 accumulator？

    **面试回答：** `wgmma` 是异步发起，源码执行到下一行不代表 Tensor Core 已写好结果。需要 commit 相应异步组，再等待覆盖该 accumulator 的组完成后才能读取；operand 的可见性和跨 proxy ordering 还要单独满足，不能用普通源码顺序替代。

7. Matrix descriptor 中 LBO/SBO 与普通二维数组 stride 有什么区别？

    **面试回答：** LBO/SBO 是 descriptor 所规定的 canonical 分块布局中的字节偏移，含义还受 major mode 和 swizzle 影响，并非直接填二维数组的行跨度。构造时要把逻辑坐标展开到硬件规定的 core-matrix 布局，并满足字段编码单位和对齐要求。

8. Swizzle 为什么应由 consumer access pattern 决定？

    **面试回答：** Bank conflict 取决于同一次 consumer 访问中哪些地址同时被读取，而不是数据在源码里看起来是否连续。Swizzle 应把这些同时访问映射到不同 banks，并让 producer 以相同约定写入；只优化写入端可能反而拖慢 Tensor Core 的读取。

9. TMEM 解决了什么瓶颈，又引入了哪些必须显式管理的资源问题？

    **面试回答：** TMEM 把大 accumulator 从通用寄存器搬到专用存储，缓解寄存器容量和数据通路压力。代价是显式管理 allocation/columns、异步 MMA 完成、读回及释放；容量、并发 tile 数和跨 CTA 协作都要计入，消费者结束前不能覆盖或释放。

10. 2-CTA MMA 复用了什么数据，什么情况下收益可能被抵消？

    **面试回答：** 按本讲沿 M 扩展的 2-CTA 配置，两个 CTA 协作复用 B tile、计算不同输出行，降低 B 的重复供给。若两侧负载不均、尾部多，或 cluster 协调和资源占用限制并发，节省的流量可能不足以抵消开销；具体复用方式还应核对目标指令模式。

11. 为什么 per-tensor scale 容易在存在 outlier 时损失普通值精度？

    **面试回答：** 单一 scale 要容纳最大的 outlier，往往把量化步长拉大，使大多数普通值只能落在少数表示档位，误差增大。细粒度 block scale 可适应局部分布，但增加 scale 元数据、加载和布局成本；也可用 clipping，在异常值误差与普通值精度间取舍。

12. 存储量化和计算量化分别改善哪条性能路径？

    **面试回答：** 存储量化先压缩 HBM 中的权重，加载后可反量化到 BF16/FP16，主要节省容量和访存。计算量化让 Tensor Core 直接执行低精度乘法，还可能提高计算吞吐；两者都需计入 scale 和转换成本，低位宽存储并不保证用了低位宽计算。


## 12. 进一步阅读

- [LCPU AI Infra Seminars Session 03 Slides](https://infra.seminars.lcpu.dev/slides/session03.pdf)
- [LCPU AI Infra Seminars 课程日历](https://infra.seminars.lcpu.dev/schedule)
- [NVIDIA PTX ISA：Warp Level Matrix Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions)
- [NVIDIA PTX ISA：Asynchronous Warpgroup Level Matrix Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions)
- [NVIDIA PTX ISA：Tensor Memory](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

## 最后总结

1. Tensor Core 用固定小块矩阵乘和片上复用提高算术强度，但峰值能否兑现取决于数据供给。
2. SM80 的核心抽象是 register fragment；SM90 是 SMEM descriptor + async warpgroup；SM100 是 descriptor + TMEM + asynchronous transaction。
3. 三代演进持续把 payload 从通用 registers 移向专用数据通路，同时把 layout、visibility、completion 和 lifetime 契约显式化。
4. 低精度性能不只由 FP8/FP4 throughput 决定，scale factor 的粒度、布局、读取和 accumulation/promotion 同样关键。
5. 写高性能 Tensor Core kernel，本质上是在正确性约束下同时编排 shape、movement、layout、reuse、synchronization 与资源生命周期。
