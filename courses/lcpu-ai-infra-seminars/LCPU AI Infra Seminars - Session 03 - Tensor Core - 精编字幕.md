---
type: transcript
status: polished
course: Weiming HPC Training Camp × LCPU AI Infra Seminars
session: 3
speaker: 孙远航
lecture_date: 2026-08-02
topics:
  - gpu
  - tensor-core
  - cuda
  - mma
  - wgmma
  - tcgen05
  - low-precision
video_url: https://www.bilibili.com/video/BV1LxM96eE43/
slides_url: https://infra.seminars.lcpu.dev/slides/session03.pdf
---

# LCPU AI Infra Seminars - Session 03 - Tensor Core - 精编字幕

> [!info] 整理说明
> 本文依据用户提供的自动字幕，结合官方 136 页课件校正术语、断句并删除口头重复。它保留讲述顺序、现场推导和关键问答，但不是逐字稿；时间点可用于回看原视频。

## 术语校正

| 自动字幕常见误识别 | 校正 |
|---|---|
| 探测扣 / TGO / TENSGO | Tensor Core |
| 库拉库 / 扩大 Core | CUDA Core |
| 计算器 / 竞技器 | register |
| 销售版本 / 下版本 | shared memory / SMEM |
| PDX | PTX |
| Fragment / FRA 码 | fragment |
| LD Matrix | `ldmatrix` |
| W 减 MMA / WGM 没 | `wgmma` |
| descriptor / description | matrix descriptor |
| 随走 / Swiss | swizzle |
| TMM / TVM | TMEM（Tensor Memory） |
| TCG05 / TCT05 | `tcgen05` |
| ember / M barrier | `mbarrier` |
| skill / skate factor | scale factor |
| IP8 / IP4 | FP8 / FP4 |
| Hover / H 版 | Hopper / H100 |
| B 版 | Blackwell / B200 |

## [00:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=0) 开场：会不会写高性能 kernel，看两件事

讲者用一句很直白的话开场：判断一个人会不会写 kernel，可以先看他是否会用 shared memory，以及是否会用 Tensor Core。本讲会把前面学过的 CUDA programming model、SIMT 与 memory hierarchy 串起来，回答两个核心问题：

1. Tensor Core 到底是什么？
2. 怎样为 Tensor Core 准备数据，并让数据通路符合硬件要求？

课程按三代架构展开：Ampere/A100 的 `mma.sync`、Hopper/H100 的 `wgmma`、Blackwell/B200 的 `tcgen05`，最后讨论低精度与 block scaling。本讲专注 Tensor Core 本身；tiling、pipeline 与 warp specialization 留到后续课程。

## [04:30](https://www.bilibili.com/video/BV1LxM96eE43/?t=270) 为什么要用专用矩阵乘单元

朴素 FP32 GEMM 每次迭代读 A、B，做乘加，再维护 accumulator，计算强度很低，约为 `0.25 FLOP/byte`。让 memory access 本身显著变快很困难，更现实的方向是让每次读入的数据参与更多计算。

矩阵乘特别适合硬件加速有两个原因：它是深度学习中最重要的计算形态之一；更关键的是，它有 `O(n³)` 的计算量，却只有 `O(n²)` 的数据量。规模增大时，潜在计算密度随之提高。

Tensor Core 成功的原因之一，是它没有把 GPU 变成只能执行整个 tensor operation 的僵硬 NPU，而是把一个小块 `D=A×B+C` 能力嵌进现有 SM、warp、register 和 shared-memory 体系，让 CUDA Core 仍能处理地址计算、控制与其他算子。

## [09:50](https://www.bilibili.com/video/BV1LxM96eE43/?t=590) 为什么硬件不直接接收完整 tensor

如果硬件一次接收完整 tensor、算完再返回，中间很难插入其他处理，CUDA Core 与已有访存层级也会闲置，灵活性很差。Tensor Core 因此只负责固定 shape 的小块矩阵乘加。程序员把大 GEMM 拆成 tiles，为小块准备数据，并在周围组织数据搬运、复用和后处理。

这个设计选择决定了本讲后面的主线：Tensor Core 的算术本身不复杂，困难的是怎样把 tile 变成硬件要求的 fragment/layout，并以足够带宽喂给它。

## [11:30](https://www.bilibili.com/video/BV1LxM96eE43/?t=690) 读懂 `mma.sync.aligned.m16n8k16`

以 Ampere PTX 指令为例：

```text
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

- `mma`：matrix multiply-accumulate；
- `sync`：执行前隐含 warp synchronization；
- `aligned`：warp 内 32 个 threads 必须共同执行，否则行为未定义；
- `m16n8k16`：A 为 `16×16`，B 为 `16×8`，输出为 `16×8`；
- `row.col`：A row-major、B column-major，本质上都让 reduction/K 维连续；
- `f32.f16.f16.f32`：D/A/B/C 的类型。A、B 用 FP16，C、D 用 FP32 累加。

低精度相乘后用 FP32 accumulator，是为了避免 FP16 的范围与精度不足。要知道某种 shape/type 是否受支持，应直接查 PTX ISA 的 Warp Level Matrix Instructions 表，而不是只凭二手资料。

不同代 GPU 用 compute capability 区分可用特性，例如 SM80 对应 Ampere 数据中心架构、SM90 对应 Hopper、SM100 对应 Blackwell。硬件没有完整 FP32 Tensor Core 数据通路，常用的是 TF32，以减少电路面积和功耗。

## [21:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=1260) 现场推导一：单条 MMA 的计算强度

对 `m16n8k16`，MAC 数为：

$$16\times 8\times 16=2048\ \text{MAC}=4096\ \text{FLOP}$$

FP16 A、B 与 FP32 D 的字节量：

$$16\times16\times2+16\times8\times2+16\times8\times4=1280\ \text{B}$$

因此单条 MMA 的计算强度约为：

$$4096/1280=3.2\ \text{FLOP/byte}$$

它已经比朴素 CUDA Core 实现高很多，但离整卡 machine balance 仍很远。继续增大 M/N/K 可以提高计算强度，却也扩大 operand tile、数据输入压力和临时存储需求，这正是后续硬件代际变化的直接动因。

## [25:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=1500) 现场推导二：A100 FP16 Tensor Core 峰值

已知一条 `m16n8k16` 完成 4096 FLOP，HMMA latency 为 8 cycles；GA100 每个 SM 有 4 个 subpartitions，各有一组 Tensor Core；A100 有 108 个 SM，boost 频率约 1.41 GHz：

$$
\frac{4096}{8}
\times4
\times108
\times1.41\text{GHz}
\approx311.9\text{ TFLOPS}
$$

A100 约 `312 TFLOPS / 2.0 TB/s ≈ 156 FLOP/byte`，而单条 MMA 只有 3.2 FLOP/byte，相差约 50 倍。单条硬件指令很快并不代表 kernel 能跑满；还需要 tiling、reuse 和 pipeline 把同一份数据反复利用。

## [29:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=1740) 三代演进的根因：计算翻倍，数据通路跟不上

A100 已有约 312 TFLOPS，H100 接近 1 PFLOPS 的 FP16 Tensor Core 峰值，单靠扩大 register read/write bandwidth 无法跟上。于是三代硬件依次把不同数据搬离 registers：

- SM80：A/B/C/D 全在 registers；
- SM90：A/B 留在 SMEM，Tensor Core 通过 descriptor 直接取数，D 仍在 registers；
- SM100：D accumulator 进一步移入 TMEM，并把 MMA 发射收缩到一个 thread。

计算阵列仍然在做 `D=A×B+C`，真正快速变化的是 operand placement、发射粒度、同步机制与数据准备方式。

## [32:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=1920) 第一轮问答：Tensor Core 与 CUDA Core 的关系

数据放在 register 还是 SMEM 是架构与指令接口决定的，不是 kernel 可以任意选择。旧的 SM80 `mma.sync` 路径在新卡上仍可用，但想利用新一代峰值就必须适配新数据通路。

一个 SM 有多个 subpartitions；warp 在 subpartition 上执行，可使用 CUDA Core，也可发起 Tensor Core 操作。两者通过 register file、SMEM 和显式同步协作。Tensor Core 只做矩阵乘加，CUDA Core 仍负责地址、控制、标量/向量运算与 epilogue。所谓“为 Tensor Core 准备数据”，就是让对应 lanes 的 registers 或 descriptors 恰好表示硬件要求的 operands。

## [42:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=2520) 真正的编程难点：如何准备数据

`mma.sync` 在 PTX 层由每个 thread 执行，但语义上是整个 warp 协作完成。以 `m16n8k16` 为例，整个 warp 共同承载：

- D：128 个 FP32 elements；
- A：256 个 FP16 elements；
- B：128 个 FP16 elements；
- C：128 个 FP32 elements。

分摊到每个 thread，表现为 4 个 D registers、4 个打包 A registers、2 个打包 B registers、4 个 C registers。由于 CUDA register 是 32 bit，两个 FP16 会打包进一个 register。

Tensor Core 编程要掌握两件事：读懂 fragment 图，以及用好 `ldmatrix`、descriptor、swizzle 等脚手架。

## [44:50](https://www.bilibili.com/video/BV1LxM96eE43/?t=2690) Fragment：矩阵元素如何分摊到 lanes

Fragment 图告诉我们：矩阵某个 `(row,col)` 元素归哪个 lane、放进该 lane 的哪个 operand register。它不是任意调度，而是硬件固定连接方式；只有按这个布局准备数据，MMA 结果才正确。

以 A fragment 为例，`16×16` tile 被分为四个象限。每个标签跨两个 FP16 columns，便于打包进一个 32-bit register。某个 lane 会从四个区域各取得两个 FP16，共 8 个 elements，组成 4 个 registers。

B 为 `8×16`，共 128 个 FP16 elements，因此每个 lane 只需 2 个打包 registers。D/C 的 fragment 又有不同 row/column 映射。使用时要能回答两个方向的问题：给定矩阵坐标，它属于哪个 lane/register；给定 lane/register，它对应矩阵哪些位置。

## [57:15](https://www.bilibili.com/video/BV1LxM96eE43/?t=3435) Fragment 图不是“为了好写”，而是硬件契约

现场问答强调：fragment 分配方法是固定的，不能人为改成自己喜欢的布局。程序员的任务不是让代码写起来方便，而是让数据符合 Tensor Core 接口。

Fragment 本身位于 registers，不产生 SMEM bank conflict；冲突发生在从 SMEM 为 fragment 取数的过程中。手工实现会根据 `lane_id` 推导 group/local index，计算每个 thread 应从 SMEM load 的 A/B elements，打包后填入 PTX operands；MMA 完成后再按 D fragment 把结果写回正确位置。

这套手工推导有助于理解硬件，但生产代码通常依赖 `ldmatrix` 或更高层库。

## [65:47](https://www.bilibili.com/video/BV1LxM96eE43/?t=3947) 手工 SM80 MMA 的数据路径

最简数据路径是：

```text
GMEM → SMEM
SMEM → per-lane A/B registers
mma.sync → per-lane D registers
D fragment → SMEM / GMEM
```

每个 thread 在运行时依据 `lane_id` 算出自己的 A/B positions。A100 上常通过 padding 改变 SMEM row stride，例如把逻辑 `16×16` 存为带额外列的布局，避免多个 lanes 落到相同 banks。

发射 `mma.sync` 后，硬件按固定 fragment 解释 A/B/C registers，并把属于该 lane 的 D fragment 写回 registers。D 的坐标映射与 A 不同，写回也必须按 D layout 计算。

## [72:46](https://www.bilibili.com/video/BV1LxM96eE43/?t=4366) `ldmatrix`：让硬件完成 SMEM→fragment 搬运

手工计算每个 lane 的 element addresses 繁琐且容易出错，`ldmatrix` 把 SMEM 中若干 `8×8` 小矩阵装入 warp registers，并按 MMA fragment 自动分发。

它有三个容易误解的点：

1. 每个 lane 提供的是某个小矩阵行的首地址，不是该 lane 最终得到的 element address；
2. 小矩阵在 SMEM 中的排列顺序决定输出 fragment 的顺序；
3. `ldmatrix` 只负责搬运和重排，不会自动消除 bank conflict。

一次 `ldmatrix.x4` 搬 `32 lanes × 16 B = 512 B`。SMEM 每 cycle 理论上提供 128 B，理想下限约 4 cycles。Row stride 若让 lanes 均匀落在 bank groups 上，可达到下限；若 stride 为 128 B 的倍数，多个 rows 可能反复落到同一组 banks，形成严重 conflict。

## [78:45](https://www.bilibili.com/video/BV1LxM96eE43/?t=4725) SM80 小结

SM80 的关键特征：

- 发射者：一个 warp，32 threads 共同执行；
- A/B/C/D：全部位于 registers；
- 同步：`mma.sync` 隐含 warp sync；
- 数据准备：fragment 图、`ldmatrix`、padding/swizzle；
- 主要压力：per-thread address calculation、register movement 与 SMEM bank conflict。

## [80:08](https://www.bilibili.com/video/BV1LxM96eE43/?t=4808) Hopper / SM90：为什么要有 `wgmma`

若 Tensor Core 性能翻倍，SM80 路径要求 register file read/write bandwidth 同步翻倍，而 register file 是芯片上最昂贵、最难继续扩展的资源。Hopper 采用三个方向：

1. 增大 tile，提高每次读取的复用率；
2. 让 A/B 不再经过 registers，直接从 SMEM 送入 Tensor Core；
3. 把 MMA 改为异步，让 Tensor Core 计算时 CUDA Core 能处理其他工作。

## [82:33](https://www.bilibili.com/video/BV1LxM96eE43/?t=4953) `wgmma`：从 warp 扩展到 warpgroup

四个 warps 组成一个 128-thread warpgroup，正好对应 SM 的四个 subpartitions。一条 `wgmma.mma_async` 调动整个 SM 的四组 Tensor Core。

- M 固定为 64，因为四个 warps 各负责 16 rows；
- K 的字节宽度固定为 32 B，元素数随数据类型变化；
- N 从 8 开始、步长为 8，最大可到 256；
- A/B 不再是 register fragments，而是指向 SMEM 的 64-bit matrix descriptors；
- D accumulator 仍在 registers；`m64n256` 时每 thread 可需要 128 个 FP32 registers。

`wgmma` 只在 `sm_90a` architecture-specific target 上可用，并不向 Blackwell 前向兼容。

## [87:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=5220) 异步带来的两个正确性问题

`wgmma` 发射后立即返回，因此必须回答：什么时候能安全读取 accumulator D？什么时候能覆盖 A/B 所在的 SMEM？

CUDA 普通 `ld/st` 与 CUDA Core 运算走 generic proxy；TMA、`wgmma` 等走 async proxy。两条访问路径即使指向同一块 SMEM，也不能仅凭源码顺序假设互相可见。

若 A/B 由普通 stores 写入，应在 `__syncthreads()` 后执行 `fence.proxy.async.shared::cta`，让 async proxy 看到 generic writes。若由 TMA 搬入，数据已经走 async proxy，只需用相应 completion mechanism 确认搬运完成。

## [91:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=5460) `wgmma` 生命周期：commit / wait

`wgmma` 与 `cp.async` 使用相似模式：先执行必要 fence，发若干 MMA，调用 `commit_group` 将它们分组，再用 `wait_group<N>` 保证最多只剩 N 组 in flight。

`wait_group<0>` 等所有 groups 完成；`wait_group<1>` 允许最新一组继续在途，同时处理更早结果。Commit 与 wait 之间，warpgroup 的 CUDA Core 可以做上一 tile 的 epilogue、FP8 promotion，或采用 producer/consumer warpgroup 分工。

五类规则尤其重要：

- wait 完成前不能读写对应 D registers；
- 不能过早覆盖 `wgmma` 正在读取的 A/B SMEM；
- generic writes 必须通过 proxy fence 对 async reader 可见；
- CUDA Core 改写 D 后，下一次 `wgmma` 前要重新建立 ordering；
- 编译器若怀疑 D 在异步窗口内被访问，可能保守插入 `wait_group<0>`，让整个 overlap 消失。

## [98:26](https://www.bilibili.com/video/BV1LxM96eE43/?t=5906) Matrix descriptor 与 canonical layout

Descriptor 告诉硬件：A/B 已按它能识别的 canonical layout 摆在 SMEM。布局的基础不是单个 element，而是 **core matrix**：8 rows × 16 bytes，共 128 B。无论数据类型如何，先把 elements 打包成 16-B units：FP16 每组 8 elements，FP8 每组 16 elements。

大 tile 是 core matrices 的二维平铺，因此需要两个步长：LBO（leading dimension byte offset）与 SBO（stride dimension byte offset），再加 SMEM start address、swizzle mode 和 base offset，共同编码进 64-bit descriptor。相关地址/offset 以 16 B 对齐，编码时低四位省略。

Canonical layout 可按 M 后 K 或 K 后 M 平铺。程序员不再为每个 thread 算 element address，而是声明整块 tile 的几何与布局。

## [109:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=6540) Swizzle：用地址位重排消除 bank conflict

SMEM 有 32 banks，每 bank 4 B；一个 16-B unit 横跨 4 banks，可视作一个 bank group。若 row stride 为 128 B，core matrix 的 8 rows 可能都从同一 bank group 开始，形成 8-way conflict。

Swizzle 把表示 row 的地址位与 bank-group bits 做 XOR，让不同行映射到不同 bank groups。重排后，同一次 core-matrix access 能覆盖全部 bank groups，而每列仍是一一映射，不破坏可逆性。

Descriptor 必须告诉 Tensor Core使用哪种 swizzle，否则硬件无法解释物理地址。生产代码通常让 TMA 按目标 SMEM layout 落盘；手写教学例子可以暂时关闭 swizzle，便于观察实际排布。

## [115:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=6900) 手工 `wgmma` 骨架与 SM90 小结

一个最简、尚未做 overlap 的版本依次执行：

```text
write A/B into SMEM
__syncthreads()
fence.proxy.async.shared::cta
warpgroup_fence
issue wgmma with A/B descriptors
commit_group
wait_group<0>
```

三代共同保留 GMEM→SMEM 段，变化的是 Tensor Core 从哪里读 operands、accumulator 放在哪里。SM90 相比 SM80：发射者从一个 warp 变为一个 warpgroup；A/B 从 registers 移到 SMEM；completion 从隐式 warp sync 变为 commit/wait；最大单指令 tile 从 `m16n8k16` 扩到 `m64n256k16`。

## [119:30](https://www.bilibili.com/video/BV1LxM96eE43/?t=7170) Blackwell / SM100：为什么需要 TMEM

`wgmma` 仍有两个问题：大 accumulator 占用大量 general-purpose registers，CUDA Core 对这些 registers 的访问还可能迫使异步 MMA 串行化；同时每次发射仍要求 128 threads 一起执行，尽管 Tensor Core 计算本身不需要这些 threads 参与。

SM100 引入每 SM 256 KB 的 TMEM，按 `128 lanes × 512 columns` 组织，每格 32 bit。TMEM 不能由普通 CUDA Core load/store 直接访问，只能通过 `tcgen05.ld/st/cp`。每个 warp 只能访问自己对应的 32 lanes，完整读出 accumulator 仍需四个 warps 协作。

`m128n256` accumulator 恰好占 `128×256`，即半块 TMEM，很适合 double buffering：Tensor Core 可在另一半继续累加，而上一结果被搬出做 epilogue。

## [122:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=7320) TMEM allocation / deallocation

TMEM 按 columns 分配：分配一列就获得全部 128 lanes。列数必须是 2 的幂且至少 32。Allocation 是 warp-level instruction，需要完整 warp 执行；返回的 TMEM base address 被写到某个 SMEM address，使其他 warps 可读取。使用结束后必须显式 deallocate，否则会阻塞后续 CTA 获得 TMEM。

这使 TMEM 更像由 kernel 显式管理的片上专用资源，而不是编译器自动分配的普通 registers。

## [124:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=7440) `tcgen05.mma`：单线程发射

`tcgen05.mma` 不再带 `.sync.aligned`；一个 elected thread 就能发起 MMA。`cta_group::1/2` 表示一或两个 CTAs 共同计算。Kind 只定义粗粒度类型类，精确的 A/B/D 类型、M/N 与 transpose/negate 等由 32-bit instruction descriptor（idesc）在运行时确定。

- D accumulator 用 TMEM address 表示；
- B 一定来自 SMEM descriptor，A 可来自 SMEM descriptor 或 TMEM；
- predicate 决定是覆盖 D 还是累加到已有 D；
- MX 类 kind 可让硬件读取 scale factors 并反量化，不需要 CUDA Core 参与；
- weight-stationary 变体可让 B 驻留复用。

## [127:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=7620) `mbarrier`：统一异步 completion

SM100 不再用 `wgmma.wait_group`，而把 completion 交给 Hopper 已引入的 `mbarrier`。它是 SMEM 中一个 64-bit object，维护：

- phase/parity：每一代完成后翻转；
- arrival count：还缺多少次 thread arrival；
- transaction count：还缺多少 bytes 的异步传输。

`tcgen05.commit` 把此前单线程发起的 MMA completion 绑定到 mbarrier；任意能访问该 barrier 的 threads 都可 `try_wait`。Acquire wait 保证 barrier 前的相关结果在返回后可见；cluster scope 还能跨同一 cluster 内的 CTAs 同步。

## [132:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=7920) SM100 pipeline、`tcgen05.ld` 与同步规则

典型 pipeline 是 TMA 把 operands 搬入 SMEM，elected thread 连续发 `tcgen05.mma`，再 commit 到 mbarrier；参与 epilogue 的 warps 等 phase 翻转后从 TMEM load accumulators。

`tcgen05.ld.sync.aligned` 的 `.sync.aligned` 重新出现：MMA 只需单线程发射，但把 TMEM 数据搬到每个 thread 的 registers，需要各 threads 亲自参与。`tcgen05.ld` 本身也异步，读取目标 registers 前还需 `tcgen05.wait`。

Generic/async proxy 规则仍存在：普通 `st.shared` 写 A/B 后，要用 `fence.proxy.async.shared::cta` 让 `tcgen05.mma` 可见。若 thread A 发 MMA、其他 warps 做 epilogue，在线程同步建立顺序后、各线程执行自己的 `tcgen05` 指令前，还需 `tcgen05.fence::after_thread_sync`。`tcgen05.cp` 用于 SMEM→TMEM 异步 copy，主要搬 scale-factor matrices；同一 issuer 的 `cp` 与 `mma` 保序，无需中间额外同步。

## [137:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=8220) 2-CTA MMA：用跨 CTA 复用换带宽

更大的 M/N/K 提高计算强度，但也提高 input pressure。若相邻两个 output tiles 沿 M 方向拼接，它们使用同一份 B。2-CTA MMA 让相邻 CTAs 共同计算：每个 CTA 只存半份 B，Tensor Core 通过硬件通路读取 peer CTA 的部分数据，SMEM 容量与 B 搬运带宽压力都降低，M 可扩大到 256。

Completion 可通过 cluster-scoped mbarrier 通知两边。它是硬件支持的协作模式，不是用普通跨 CTA 通信临时拼出来的软件技巧。

## [140:24](https://www.bilibili.com/video/BV1LxM96eE43/?t=8424) 低精度的动机：算力仍然不够

每代 Tensor Core 峰值快速增长，但模型需求增长更快。降低位宽可以减少 HBM 容量与带宽需求，并提高每条 MMA 的计算强度。Hopper 是首代原生支持 FP8 Tensor Core 计算的 NVIDIA GPU，但 FP8 的表示范围和精度太有限，必须与 scale factor 配合。

FP8 有两种常用格式：E4M3 最大有限值约 448，精度相对好；E5M2 动态范围可到 57344，但只有 2-bit mantissa。单一 FP8 格式无法直接覆盖训练中 tensor 的真实分布。

## [142:20](https://www.bilibili.com/video/BV1LxM96eE43/?t=8540) Per-tensor scale 为什么会被 outlier 毁掉

Per-tensor scale 常取：

$$s=\mathrm{amax}/448,\qquad x_q=\mathrm{round}(x/s)$$

反量化时再乘回 `s`。若绝大多数值在 `[-1,1]`，却有一个 3000 的 outlier，`s` 会被 outlier 决定，小值量化后大量变为 0，信息丢失。

因此需要更细粒度 scaling。但 scale 不能在 reduction/K 维逐元素任意变化，因为：

$$D[m,n]=\sum_k A[m,k]B[k,n]$$

若 scale 随每个 k 任意变化，就不能提出求和号，原问题不再是一条普通 GEMM。可行方案是 per-row，或让 scale 沿 K 分段常数的 per-block scaling。

## [144:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=8640) DeepSeek-V3 的 fine-grained scaling

DeepSeek-V3 对 activation 使用 per-token、沿 K 分块的 scale，对 weights 使用更细的二维 block scale。每个 scale 只控制局部数值范围，使 outlier 不再拖累整个 tensor，同时保持分块内可表达为 GEMM 加少量 scale 运算。

Fine-grained scale 解决了动态范围问题，但 Hopper FP8 Tensor Core 还有 accumulator precision 问题：乘积按共享指数对齐后只保留较少有效 mantissa bits，长 K reduction 会产生明显误差。

## [146:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=8760) Promotion：用 CUDA Core 修补 FP8 累加精度

课程给出的估计是：Hopper FP8 accumulation 只保留约 14-bit 有效尾数，而完整 FP32 mantissa 应有 24 bits；K=4096 时误差可到约 2%。DeepGEMM 使用两个 accumulators：`part` 是 `wgmma` 临时 accumulator，`acc` 是 CUDA Core 持有的真正 FP32 accumulator。每 `4×k32=128` 的 reduction 区间做一次 promotion，把 partial result 全精度累加到 `acc`。

这会占用 CUDA Core。Hopper 上 promotion 时间占比约 23%；Blackwell Tensor Core 峰值增长远快于 FP32 CUDA Core，照搬该方案可能让 promotion 占比上升到约 44%，成为新瓶颈。

## [147:50](https://www.bilibili.com/video/BV1LxM96eE43/?t=8870) Blackwell 硬件 block scaling

Blackwell 让硬件按 K 维每 32 或 16 个 elements 读取一个 scale factor，直接在 Tensor Core 路径中反量化，不再由 CUDA Core promotion/scale 运算兜底。MX 前缀数据格式采用硬件 block scaling；scale factor 也作为 MMA operands 存入 TMEM，`tcgen05.cp` 负责把其矩阵搬入。

FP4 E2M1 只有极少可表示格点；FP6 的 E3M2/E2M3 分别在范围与精度间取舍。UE8M0、UE4M3 是 scale-factor formats，不是普通数据格式。MXFP 属于 OCP Microscaling 标准，NVFP 则采用 NVIDIA 自身的打包/scaling 方案。

## [149:50](https://www.bilibili.com/video/BV1LxM96eE43/?t=8990) 存储量化与计算量化不是一回事

常见 W4A16/GPTQ/AWQ 配合 Marlin 一类 kernel，通常是 **存储量化**：weights 在 HBM 以 4 bit 保存，进入 Tensor Core 前由 kernel 解量化回 BF16，真正的 MMA 仍是 BF16×BF16。它主要减少容量和 memory traffic。

本讲讨论的 FP8/FP4 Tensor Core 是 **计算量化**：低精度 operands 真正进入 MMA，峰值计算吞吐也随位宽下降而增加。

讨论量化时应明确三个维度：量化 weight、activation、KV cache、gradient 还是 optimizer state；粒度是 per-tensor、per-channel、per-block 还是 per-group，scale 用何种格式；时机是 PTQ、QAT，还是原生低精度训练。

## [152:46](https://www.bilibili.com/video/BV1LxM96eE43/?t=9166) 总结：三代 Tensor Core 是数据通路的演进

```text
SM80 / mma.sync
  1 warp 发射
  A/B/C/D 全在 registers
  fragment + ldmatrix + padding/swizzle

SM90 / wgmma
  1 warpgroup 发射
  A/B 在 SMEM，D 在 registers
  descriptor + async proxy + commit/wait

SM100 / tcgen05
  1 elected thread 发射
  A/B 在 SMEM，D 在 TMEM
  idesc + mbarrier + TMEM lifetime + 2-CTA + hardware scaling
```

Tensor Core 逐代从 CUDA Core 控制流中剥离，越来越像独立 accelerator。相应地，程序员从“为每个 lane 准备 register fragment”走向“描述 tile/layout、管理异步 completion 与专用存储 lifetime”。后续 Pipeline Ordering 课程会继续解决：即使单次 MMA 很强，怎样通过 tiling、pipeline 和 warp specialization 持续喂饱它。

## [154:00](https://www.bilibili.com/video/BV1LxM96eE43/?t=9240) 课后问答：单线程发射与 2-CTA 的意义

单线程发射的直接好处是无需让整个 warpgroup 为发射动作同步，也便于给 issuer 分配很少资源。其余 threads 若不执行任务，并不存在“白白浪费计算”的简单结论；真正占用取决于 warp/CTA 的整体资源配置和角色分工。

2-CTA MMA 是硬件设计。两个 CTAs 作为 cluster 被放到相邻 SM，通过专用通路共同驱动 Tensor Core。相较于分别做两个 `M=128` GEMM，`M=256` 的联合计算能让 B 只搬/存一份并跨两边复用，从而降低 SMEM 容量与读取压力。课程最后强调：收益本质仍然是 data reuse，而不是“两个 CTA 神奇地让算术本身更快”。

## 参考资料

- [讲座视频：Tensor Core 从 mma.sync 到 tcgen05](https://www.bilibili.com/video/BV1LxM96eE43/)
- [官方课件 PDF](https://infra.seminars.lcpu.dev/slides/session03.pdf)
- [课程 Wiki：Session 3 Tensor Core](https://infra.seminars.lcpu.dev/wiki/QsOcw3uHpifm10k5ouCcK6Uonee)
