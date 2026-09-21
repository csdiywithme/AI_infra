---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 18
lecture_date: 2023-12-05
area: systems
topics:
  - hardware-specialization
  - energy-efficiency
  - heterogeneous-computing
  - accelerators
  - fpga
  - spatial
  - dataflow
  - streaming
  - attention
aliases:
  - CS149 Lecture 18
  - Hardware Specialization
video_url: https://www.youtube.com/watch?v=2tAb3EgyjNw
---

# Stanford CS149 - Lecture 18 - Hardware Specialization

> [!abstract]
> Dennard scaling 结束后，power budget 不再随 transistor scaling 自动改善；在固定功率下提高 performance，必须降低每个 operation 的 energy。通用 CPU 的大部分能量耗在 instruction fetch/decode/scheduling、register/cache data movement 与 clock/control，而不是 arithmetic 本身。Specialization 通过删掉不服务于目标 workload 的通用控制、定制 datapath 和 memory hierarchy，获得数量级的 energy/area efficiency，但越专用越难编程、越昂贵且越不灵活。本讲用 CPU→GPU/DSP→domain-specific accelerator→FPGA→ASIC 的连续谱解释这项权衡，并以 Spatial DSL 展示如何显式表达 independent parallelism、pipeline parallelism、on-chip memory 与 data movement。Inner product 与 attention 案例揭示核心原则：accelerator performance 不是“多放 ALU”，而是把 parallelism、locality、tiling、streaming 和 buffering 共同映射成硬件。

## 来源与范围

- [Lecture 18 视频：Hardware Specialization](https://www.youtube.com/watch?v=2tAb3EgyjNw)
- [官方课件 PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/hwaccel/17_heterogeneity_Spatial_wWfLWLq.pdf)
- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)

课程网页/课件跳过 Midterm Review 编号，因此 PDF 标作 Lecture 17；本文按公开视频播放列表记作 Lecture 18。开头承接 [[Stanford CS149 - Lecture 17 - Transactional Memory II#10. 视频结尾：从通用并行走向 specialization|上一视频最后的 heterogeneity 引入]]。

## 视频索引

| 时间 | 内容 |
|---|---|
| [00:05](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=5s) | 从 heterogeneous computing 走向 algorithm-specific specialization |
| [01:15](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=75s) | Dennard scaling 结束与 energy constraint |
| [02:02](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=122s) | `Energy = Power × Time`；固定 power 下性能依赖 energy/op |
| [02:53](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=173s) | Supercomputer、datacenter、mobile 都受 energy/thermal 限制 |
| [05:48](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=348s) | CPU 为什么低效：arithmetic 只占 instruction energy 的小部分 |
| [07:35](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=455s) | SIMD 摊销 control overhead，但 width 受 utilization 限制 |
| [09:48](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=588s) | FFT 的 CPU/ASIC area 与 energy efficiency 对比 |
| [11:41](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=701s) | ASIC 的代价：single purpose、设计周期与成本 |
| [12:11](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=731s) | DSP：special instructions/addressing 换效率，牺牲 programmability |
| [13:35](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=815s) | Anton molecular-dynamics accelerator |
| [15:11](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=911s) | TPU 与 ML domain-specific acceleration |
| [17:03](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=1023s) | FPGA：CLB/LUT + registers + programmable interconnect |
| [19:01](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=1141s) | FPGA hard blocks：BRAM 与 DSP/multiplier units |
| [21:05](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=1265s) | Programmability–energy efficiency continuum |
| [23:17](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=1397s) | GPU/TPU 边界变化与 Tensor Cores |
| [24:52](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=1492s) | Accelerator 可定制 memory system 与 compute resources |
| [26:13](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=1573s) | RTL 与 C-based HLS 的抽象问题 |
| [27:52](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=1672s) | Spatial：面向 performance programmers 的 accelerator DSL |
| [29:21](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=1761s) | Independent parallelism 与 dependent/pipeline parallelism |
| [32:32](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=1952s) | Spatial memory templates：DRAM、SRAM、FIFO、line/shift buffer |
| [34:05](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=2045s) | 显式 load/store、gather/scatter 与 stream |
| [35:59](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=2159s) | Control templates：`Accel`、`Foreach`、`Reduce` |
| [37:49](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=2269s) | Parallelization/schedule/size 参数与 design-space exploration |
| [39:20](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=2360s) | Inner-product accelerator walkthrough |
| [41:53](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=2513s) | Tiling：批量搬运 DRAM 数据到 on-chip SRAM |
| [43:28](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=2608s) | Load tile、intra-tile reduction、inter-tile accumulation |
| [45:03](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=2703s) | 三类 parallelism：stage pipeline、parallel reduce、SIMD-like datapath |
| [48:18](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=2898s) | Pipeline schedule 与 double buffering |
| [50:20](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=3020s) | Programmer 与 compiler 的责任边界 |
| [52:28](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=3148s) | Attention：fuse/tiling 避免 materialize attention matrix |
| [55:38](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=3338s) | Softmax 三阶段与 kernel-by-kernel data movement |
| [57:35](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=3455s) | FIFO producer–consumer streaming softmax |
| [60:34](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=3634s) | `N×N` materialization 变成小 FIFO / row buffer |
| [63:12](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=3792s) | Long sequence 下与 FlashAttention 的互补关系 |
| [64:13](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=3853s) | Streaming vs. kernel-by-kernel execution |
| [69:36](https://www.youtube.com/watch?v=2tAb3EgyjNw&t=4176s) | Accelerator 总结：application insight 决定资源与 memory hierarchy |

## 1. 为什么性能问题变成 energy 问题

### 1.1 Dennard scaling 结束后的约束

过去制程缩小带来更多 transistors，同时 voltage/power density 下降，芯片能在近似相同 power 下提高 frequency/功能密度。Dennard scaling 结束后，更多 transistors 不再意味着都能同时以最高频运行；power delivery 与 heat removal 成为硬约束。

基本关系：

$$
E=P\times t
$$

若 chip/rack/mobile thermal envelope 给定最大功率 $P_{max}$，吞吐为：

$$
\text{Throughput}\lesssim \frac{P_{max}}{E_{op}}
$$

所以想在同一 power budget 下做更多工作，就要降低 energy per useful operation，而不能只堆 cores/transistors。

### 1.2 约束存在于所有尺度

- **Supercomputer/HPC**：成千上万 nodes 的供电与 cooling 是系统设计上限；
- **Datacenter**：服务器生命周期内的 power/cooling operating cost 可与甚至超过采购成本；
- **Mobile/edge**：battery capacity、无风扇 passive cooling 与 skin temperature 限制 sustained performance。

Performance-per-watt 不是附加指标，而是能否持续交付 performance 的前提。

## 2. 通用 CPU 的能量花在哪里

执行一个 multiply-add 不只发生 arithmetic：

```text
instruction fetch → decode → dependency checking → scheduling
→ operand register/cache access → arithmetic → result movement/retirement
                 + clock/control distribution
```

课堂引用的示意中，真正 arithmetic 只占约 `6%` instruction energy；精确比例随工艺与微架构变化，但结论稳定：**programmability、dynamic scheduling 与 data movement 很贵**。

CPU 为未知未来 workload 提供：

- arbitrary control flow；
- precise exceptions/speculation；
- large register files/caches；
- dynamic dependency/resource scheduling；
- coherent shared-memory abstraction。

即使某 kernel 不需要这些能力，CPU 仍为它们付 silicon/energy cost。Specialized accelerator 则把已知的 schedule、dependency 和 movement 固化/显式化，删去 general-purpose tax。

## 3. SIMD 为什么有用、为什么还不够

SIMD 用一条 instruction 控制 $W$ 个 data lanes，把 fetch/decode/control cost 摊薄：

$$
\text{control cost per element}\approx \frac{C_{control}}{W}
$$

但 width 越宽，越需要足够 data parallelism 和规则 control flow。Divergence、tail、irregular accesses 或 dependency 使 lanes idle：peak throughput 上升，average utilization 未必上升。

即使 workload 非常 SIMD-friendly，register/cache accesses、clock tree、instruction infrastructure 等仍消耗显著能量。因此 GPU/SIMD 是 specialization continuum 的一步，不是终点。

## 4. Programmability–efficiency continuum

```text
更易编程 / 更通用                                      更高能效 / 更专用
CPU ── GPU ── DSP ── domain accelerator ── FPGA ── ASIC
```

| 平台 | 专用化方式 | 优点 | 主要代价 |
|---|---|---|---|
| CPU | 通用 scalar/SIMD + caches | 生态成熟、迭代快 | control/data-movement overhead 大 |
| GPU | throughput/SIMT、宽 SIMD | data parallel workload 高效 | divergence、host/device movement、fixed hierarchy |
| DSP | MAC、special addressing 等 | signal processing 高效 | instructions 难编译，常需低层优化 |
| Domain accelerator | 固定 domain primitive | framework 可隐藏硬件，能效高 | domain 演化会降低利用率/寿命 |
| FPGA | 可重构 LUT/interconnect + hard blocks | 上线后可重配，低延迟 dataflow | area/frequency/energy 不如 ASIC，编程难 |
| ASIC | 固化 datapath/memory/control | 最佳 area/energy/throughput | NRE、周期、风险高，算法变化难适配 |

课堂引用的旧制程 FFT 研究展示 CPU 与 ASIC 之间可有约 `100×` energy efficiency、`1000×` area efficiency 量级差距；应理解为 specialization potential 的案例，不是适用于所有 workload/工艺的固定倍率。

## 5. 三类 specialization 实例

### 5.1 DSP：domain-tuned ISA

DSP 为 filtering/FFT 等提供 MAC、circular/bit-reversed addressing 等复杂 primitives，减少 instruction/control/data movement。但复杂 ISA 不容易由普通 compiler 自动利用，常需 assembly/intrinsics 与 domain expertise。

### 5.2 Anton：algorithm–hardware co-design

D. E. Shaw Research 的 Anton 为 molecular dynamics/n-body interactions 定制 compute/network/memory，使关键 force calculation 与 particle movement 高效执行。它的成功来自对 algorithm communication/locality 的整体设计，而非只替换一颗更快 ALU。

### 5.3 TPU：domain-specific but still programmable

早期 TPU 围绕大规模 integer matrix multiply array，后续版本支持较小 array 和 low-precision floating point。ML accelerator 不能完全固定为单个 network：model/operator/precision 快速变化，因此一般在 dense/sparse tensor primitives 上专用，同时保留一定 programmability。

GPU 加入 Tensor Cores 说明 continuum 会移动：高价值 workload 足以让 general-purpose product 吸收 domain-specific units。

## 6. FPGA：ASIC 与 processor 之间的 middle ground

### 6.1 基本组成

- **Configurable Logic Block / LUT**：truth table 实现任意小型 Boolean function；
- **flip-flop/register**：在逻辑间保存 state，构成 pipeline；
- **programmable interconnect**：连接 LUT/register，组成任意 datapath；
- **hard macros**：BRAM/URAM、DSP/multiplier、SerDes/PCIe 等。

纯 LUT 实现最灵活，却在 area、routing 与 energy 上昂贵；常用 arithmetic/memory 做成 hard blocks 可提高密度和频率。FPGA 可连接 DDR、CPU/host、其他 FPGAs，也可通过云服务获得。

### 6.2 FPGA 不是“可编程 CPU”

CPU 配置的是随时间执行的 instruction stream；FPGA 配置的是空间上的 circuit/data path。一个 cycle 可让多个不同 pipeline stages 同时工作，吞吐来自 spatial replication 与 pipelining。

## 7. 为什么 C-based HLS 仍然困难

RTL（Verilog/VHDL）精确但低层。High-Level Synthesis 想把 C/C++ 变成 circuit，却面临 abstraction mismatch：

- C 描述 sequential machine 的 operations，不直接表达 hardware structure；
- memory port 数、pipeline initiation interval、buffering 与 resource sharing 都是隐含的；
- compiler 必须从 pointer/loop 推断 parallelism 和 locality；
- 为得到好结果，程序员又加入大量 unroll/pipeline/partition pragmas。

于是“不会硬件也能写 C 生成高效 accelerator”的目标，常变成“用 C 语法做 hardware design”。问题不在语法高级，而在语言是否让重要 design choices 成为一等概念。

## 8. Spatial DSL 的抽象

Spatial 面向已懂 algorithm、parallelism、locality 的 performance programmer。它不隐藏 accelerator 的资源 tradeoff，而是用 domain constructs 表达它们。

### 8.1 两类 parallelism

1. **Independent/spatial parallelism**：复制多个 processing elements，同时执行 map/unrolled loop；
2. **Dependent/pipeline parallelism**：不同 stages 处理不同 items，像 assembly line；stage 之间有真实 dependence，整体仍可每 cycle 接收新输入。

```text
cycle 1: item0[S1]
cycle 2: item1[S1] → item0[S2]
cycle 3: item2[S1] → item1[S2] → item0[S3]
```

前者花更多 compute units，后者通过 overlapped stages 提高 throughput；两者可嵌套。

### 8.2 显式 memory hierarchy

Spatial 让 programmer 声明：

- off-chip `DRAM`；
- on-chip `SRAM`；
- scalar/register/accumulator；
- FIFO；
- line buffer、shift register。

在 CPU 上 cache controller 自动搬数据，程序员只能写 cache-friendly access；在 accelerator DSL 中，programmer 显式 `load/store`、`gather/scatter` 或 stream，决定何时把数据从 DRAM 搬进 on-chip storage。

### 8.3 Parallel patterns 与 parameters

- `Foreach` 对应 map/data-parallel iteration；
- `Reduce` 表达 associative aggregation/reduction tree；
- `Accel`/`Accel(*)` 标记一次或持续运行的 accelerator region；
- `par`、tile size、pipeline/stream schedule 等作为 design parameters；
- compiler 根据 parallel factor 做 SRAM banking/duplication、buffer insertion 和 target mapping。

程序员指定“需要多少 logical parallelism、数据住哪里、何时移动”；compiler 保证 physical ports/banks/buffers 支撑它，并报告 performance/resource estimates。

## 9. Inner-product accelerator：完整 mapping

目标：

$$
result=\sum_{i=0}^{N-1}v_1[i]\cdot v_2[i]
$$

### 9.1 Tiling 与 memory movement

两个 vectors 在 DRAM，accelerator 内建两个 tile-sized SRAM：

```text
for each tile t:
    DMA vec1[t] → tile1 SRAM
    DMA vec2[t] → tile2 SRAM
    partial[t] = Reduce_i(tile1[i] * tile2[i])
result = Reduce_t(partial[t])
```

逐 element 访问 DRAM 会浪费 interface latency/bandwidth。一次搬一 tile 类似把 working set 带回 pantry：大笔 transfer 摊薄 setup cost，并在 on-chip SRAM 上复用/并行访问。

### 9.2 三种优化维度

1. **Parallel reduce**：一次做 $P$ 个 multiplies，再用 reduction tree；$P$ 越大，吞吐高但 multiplier/adder/tree/ports 越多；
2. **Tile size**：决定 DMA burst、SRAM capacity、reuse 与 tail behavior；
3. **Pipeline schedule**：重叠下一 tile load、当前 tile compute、上一 tile accumulation。

```text
time →
load:       T0   T1   T2   T3
compute:         T0   T1   T2   T3
accumulate:           T0   T1   T2   T3
```

Pipeline 需要 double buffering：load 不能覆盖 compute 尚在读的 tile。若三 stages 平衡，理想 throughput 可接近 `3×` sequential stages，但受最慢 stage、fill/drain 与 bandwidth 限制；额外 buffers 也占资源。

### 9.3 责任边界

Spatial programmer 负责：

- 用 `Foreach/Reduce` 表达 algorithm；
- 设计 memory hierarchy 和 explicit movement；
- 选 tile/parallel factors；
- 决定 pipeline/stream schedule。

Compiler 负责：

- memory banking/replication；
- double buffers/FIFOs 与 handshaking；
- 对具体 FPGA/accelerator target 生成低层实现；
- 估算 resource usage、latency/throughput，辅助 design-space search。

## 10. Attention：kernel boundary 也是 architecture choice

普通 kernel-by-kernel attention 会将中间 $S=QK^T$ 或 softmax matrix 写回 accelerator memory，再由下一 kernel 读回。中间矩阵占 $O(N^2)$ capacity/bandwidth，常让 arithmetic units 等 memory。

### 10.1 Softmax 的 dependence

对每行：

$$
p_{ij}=\frac{e^{s_{ij}}}{\sum_k e^{s_{ik}}}
$$

可拆成：

1. elementwise `exp`；
2. row-wise reduction 得 denominator；
3. elementwise division。

若每阶段是独立 kernel，就需要 materialize stage outputs。

### 10.2 FIFO streaming producer–consumer

Spatial 可把 stages 同时运行：

```text
exp producer ──FIFO──> row-reduce consumer ──FIFO──> normalize consumer
```

Producer enqueue，consumer dequeue；中间 state 只保留 pipeline 正在使用的 elements，而不是整个 $N\times N$ matrix。小 FIFO 起 decoupling/double-buffering 作用，允许 stages 有短暂速率差。

收益：

- kernel-level pipeline parallelism；
- 避免 off-chip/on-chip large-memory round trips；
- compiler 可从 composition 生成 fused dataflow，不必手写 monolithic fused kernel；
- 每个 stage 仍保持 modular expression。

### 10.3 Streaming 不会消除 algorithmic lower bound

Naive softmax 仍需整行 denominator，可能要 buffer 一行；sequence 很长时 row FIFO/SRAM 也过大。FlashAttention 的 online max/sum/rescaling 重排算法，将 state 限为 tile/running statistics。两者是互补层次：

```text
streaming model：消除 stage 间不必要 materialization
FlashAttention：改变算法，减少单个 row/tile 必须保留的 state
```

Hardware-friendly execution model 能自动获得一部分 fusion/streaming benefit，但在 capacity boundary 处仍需要 algorithm insight。

## 11. 统一性能模型：compute、movement 与 resources

Specialization 的 design loop：

1. 找 workload hot path；
2. 判断 bottleneck：compute、bandwidth、latency、capacity 或 synchronization；
3. 找 parallelism：data、pipeline、task；
4. 找 locality/reuse，设计 tile 与 on-chip storage；
5. 定义 data movement 和 producer–consumer topology；
6. 选 parallel factor，检查 memory ports/banks；
7. 评估 area、energy、frequency、utilization；
8. 若 algorithm/shape 改变，重新平衡设计。

常见误区是只增加 MAC 数：

$$
\text{Achieved throughput}
=\min(\text{compute peak},\ \text{memory supply},\ \text{pipeline bottleneck})
$$

如果 DRAM 或某 pipeline stage 供不上，额外 MAC 只是 dark silicon。

## 12. 与 AI Infra 的连接

### 12.1 Tensor Core 是可控的 specialization boundary

GPU 保留 SIMT flexibility，同时把 matrix multiply 作为 special instruction/data path。Framework/compiler 必须完成 layout、tile、precision 与 fusion 映射；hardware speedup 只有在 software stack 能稳定喂满单元时才兑现。

### 12.2 Operator fusion 的本质是缩短数据旅程

FlashAttention、fused MLP、quantized GEMM 的共同收益往往不是少做 arithmetic，而是中间结果留在 registers/SRAM/FIFO，避免 HBM round trip。能效优化首先是 data-movement optimization。

### 12.3 Serving shapes 决定 accelerator utilization

ASIC/systolic array 对大、规则 matrix 高效；decode 阶段 small batch、variable sequence、KV-cache-heavy workload 可能 memory-bound 且 array utilization 低。选 accelerator 要按 end-to-end shape distribution，不应只看 peak TOPS。

### 12.4 Compiler 是 accelerator 产品的一部分

硬件能效若要求每位用户手写 RTL 就难以普及。Graph compiler、kernel DSL、auto-tiling、cost model 与 profiler 决定可用性；Spatial 把 parallelism/locality choices 暴露给 performance programmer，正是硬件与 framework 之间的接口设计。

### 12.5 Disaggregation 会重新引入 movement cost

把 model、memory、accelerator 分到不同 devices/racks，可提高 resource pooling，却把 on-chip FIFO 变成 PCIe/CXL/network transfer。Specialization 的收益必须扣除 host orchestration 与 interconnect energy/latency。

## 13. 易混点

1. **Specialization 不只是新 instruction**：memory hierarchy、movement、control 与 interconnect 往往比 ALU 更关键。
2. **FPGA 不是总比 GPU/ASIC 好**：它在 flexibility、latency 和 development cycle 上折中，绝对能效通常仍低于同任务 ASIC。
3. **Pipeline parallelism 不要求 stages 独立**：相邻 stages 有 dependence，但处理不同 items 时可 overlap。
4. **Pipeline 并非免费**：需要 buffers、control、fill/drain，throughput 受最慢 stage 限制。
5. **Streaming 不等于零 storage**：必须容纳 rate mismatch 与 algorithmic live state，可能是一行甚至更多。
6. **HLS 的难点不只是 compiler 不够聪明**：sequential C 没有直接表达 ports、space、time 与 physical resources。
7. **峰值能效不是 application 能效**：utilization、shape、bandwidth、host overhead 都会改变 end-to-end 结果。

## 14. 本讲结论

1. 固定 power/thermal budget 下，performance growth 要依靠更低 energy per operation。
2. 通用 CPU 为 programmability 支付大量 instruction/control/data-movement energy；specialization 通过去掉这些通用开销提高效率。
3. CPU、GPU、DSP、domain accelerator、FPGA、ASIC 构成可编程性与能效的连续 tradeoff，而非简单优劣排名。
4. Accelerator design 的核心是 application insight：识别 parallelism、locality、movement 与 bottleneck。
5. Spatial 把 on/off-chip memory、data movement、parallel patterns 与 pipeline schedule 作为语言一等概念。
6. Spatial parallelism 复制计算资源；pipeline parallelism 让 dependent stages 对不同数据 overlap，两者都消耗 physical resources。
7. Tiling、parallel reduction、double buffering 共同决定 inner-product accelerator 的吞吐与资源占用。
8. FIFO streaming 能跨 kernel 消除中间 materialization；遇到 row/state capacity 限制时，仍需 FlashAttention 一类 algorithm transformation。
9. 真正的 accelerator 是 hardware + compiler + runtime/framework 的共同产物。

## 自测题

1. 从 $E=P\times t$ 推导为什么固定 power budget 下 performance 取决于 energy/op。
2. 为什么 arithmetic 只占通用 CPU instruction energy 的一小部分？列出至少四项 overhead。
3. SIMD width 加倍为什么不保证 application energy efficiency 加倍？
4. CPU、DSP、FPGA、ASIC 分别在 programmability–efficiency continuum 的什么位置？
5. FPGA 为什么同时需要 LUT 和 hard DSP/BRAM blocks？
6. C-based HLS 为什么常靠大量 pragmas 才能得到高性能？
7. Independent parallelism 与 pipeline parallelism 的硬件资源需求有何不同？
8. Inner product 为什么需要两级 reduction？Tile size 如何影响 DRAM efficiency 与 SRAM capacity？
9. 三 stage pipeline 为什么需要 double buffering？理想 `3×` speedup 在什么条件下成立？
10. Programmer 与 Spatial compiler 各自负责哪些决策？
11. FIFO streaming 如何减少 attention 中间矩阵的 materialization？
12. 为什么 streaming softmax 仍可能需要一整行 buffer？FlashAttention 又改变了什么？
13. 为 LLM decode 选择 accelerator 时，为什么 peak TOPS 不够？

## 相关笔记

- [[Stanford CS149 - Lecture 03 - Multi-Core Architecture Part II and ISPC]]
- [[Stanford CS149 - Lecture 07 - GPU Architecture and CUDA Programming]]
- [[Stanford CS149 - Lecture 10 - Efficiently Evaluating DNNs on GPUs]]
- [[Stanford CS149 - Lecture 15 - Domain-Specific Programming Languages]]
- [[Stanford CS149 - Lecture 17 - Transactional Memory II]]
- [[Stanford CS149 - Lecture 19 - Accessing Memory and Course Wrap-Up]]
- [[Stanford CS149]]
