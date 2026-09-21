---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 10
lecture_date: 2023-10-26
area: systems
topics:
  - deep-learning-systems
  - convolution
  - gemm
  - tiling
  - kernel-fusion
  - flash-attention
  - tensor-cores
  - mixed-precision
aliases:
  - CS149 Lecture 10
  - Efficiently Evaluating DNNs on GPUs
video_url: https://www.youtube.com/watch?v=qbKtU0X6-WU
---

# Stanford CS149 - Lecture 10 - Efficiently Evaluating DNNs on GPUs

> [!abstract]
> 本讲用 DNN 把前九讲的原则串成一条完整优化链：先把 convolution 映射成已有高性能实现的 GEMM，再通过 cache/shared-memory blocking 把 naive `O(1)` arithmetic intensity 恢复到 `O(B)`；随后发现 explicit im2col、逐算子 kernel 和 `N²` attention matrix 又制造了巨量 off-chip traffic，于是使用 implicit GEMM、operator fusion 和 online tiled softmax 消除 materialization。最后通过低精度与 Tensor Core 说明：高效 DNN evaluation 是 model topology、algorithm、schedule/compiler 和 specialized hardware 的共同设计。

## 来源与范围

- [Lecture 10 视频：Efficiently Evaluating DNNs](https://www.youtube.com/watch?v=qbKtU0X6-WU)
- [官方课件 PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/dnneval/10_dnneval.pdf)
- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)

本文以视频实际讲解为主，并用课件校正 tensor dimensions、Winograd operation count、online softmax 公式和 A100 历史规格。课程发表于 2023 年，具体 library API、GPU 峰值和主流模型会变化；本文重点是仍然稳定的 mapping、locality、fusion 与 specialization 原理。

## 视频索引

| 时间 | 内容 |
|---|---|
| [01:12](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=72s) | Assignment 3 renderer 的透明 circle ordering dependency |
| [04:32](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=272s) | 只有 overlap circles 才真正有顺序依赖 |
| [06:04](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=364s) | 本讲主题：DNN 映射到 CPU/GPU |
| [08:20](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=500s) | Neuron 是 dot product + bias + nonlinearity |
| [10:34](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=634s) | Fully connected 与 convolutional connectivity |
| [12:30](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=750s) | 3×3 image convolution：blur 与 gradient filter |
| [15:25](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=925s) | 多 filters、多 channels 与 activation tensors |
| [18:57](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=1137s) | 三类优化入口之一：改进 model topology |
| [22:40](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=1360s) | 算法/模型设计可能带来远大于硬件代际的收益 |
| [25:53](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=1553s) | 七层 loop nest 的 direct convolution |
| [27:46](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=1666s) | Convolution → explicit GEMM / im2col |
| [31:16](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=1876s) | 多 filters/channels 怎样扩大 GEMM dimensions |
| [34:20](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=2060s) | 从三重循环实现 dense GEMM |
| [35:25](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=2125s) | Naive GEMM 为什么退化成 bandwidth-bound |
| [37:57](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=2277s) | Blocked GEMM 与 `O(B)` arithmetic intensity |
| [41:32](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=2492s) | CPU cache 与 GPU shared-memory scratchpad 的区别 |
| [44:22](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=2662s) | 多级 blocking、SIMD microkernel 与 transpose choices |
| [47:59](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=2879s) | Explicit im2col 的 duplication 与 memory footprint |
| [49:02](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=2942s) | Implicit GEMM：按需从 tensor 取 tile |
| [52:14](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=3134s) | CUTLASS 的 GEMM/tensor iterators |
| [53:10](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=3190s) | Batch/shape 太小时 GPU 填不满 |
| [55:02](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=3302s) | Direct convolution 与 compiler scheduling |
| [56:04](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=3364s) | Winograd/FFT：改变 arithmetic mix |
| [57:48](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=3468s) | cuDNN 算法选择：implicit GEMM、direct、FFT、Winograd |
| [60:32](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=3632s) | 跨 DNN ops 的 activation memory traffic |
| [61:56](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=3716s) | Conv + scale/bias + max-pool fusion |
| [64:00](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=3840s) | Transformer attention 的 fusion 问题 |
| [67:13](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=4033s) | Naive attention 物化 `N×N` matrix |
| [69:00](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=4140s) | 分块/online softmax 的代数分解 |
| [70:58](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=4258s) | Fused attention：Q/K/V tiles 直接累积 O |
| [72:27](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=4347s) | 手工 fused APIs 与 compiler-generated fusion |
| [74:09](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=4449s) | Low precision、算法、software、hardware 四层共同优化 |
| [75:13](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=4513s) | GPU 为什么适合 DNN |
| [76:05](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=4565s) | General-purpose GPU 又为什么可能不是最优 |
| [77:11](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=4631s) | 用更大矩阵指令摊薄 control overhead |
| [78:43](https://www.youtube.com/watch?v=qbKtU0X6-WU&t=4723s) | A100 Tensor Core 与 mixed-precision throughput |

## 0. 开场例子：先找 dependency，再选择并行轴

Assignment 3 要按给定 back-to-front 顺序渲染透明 circles。若直接 “one CUDA thread per circle”，多个 circles 同时更新同一 pixel：

- 出现 write race；
- 更重要的是 alpha blending 不满足任意重排，顺序改变会改变颜色。

但 dependency 只存在于 **overlap 的 circles 对同一 pixel 的更新**。不相交 circles 可自由并行。若能为每个 pixel 建立按原顺序排列的 candidate circles list，就可以：

```text
parallel over pixels
    sequentially blend only circles overlapping this pixel
```

这个开场与 DNN 主体共享方法论：不要因为表面 loop 有 dependency 就认定整个问题串行；找出 dependency 的最小作用域，再改变 decomposition/data structure。

## 1. 从 neuron 到 tensor operation

一个简化 neuron：

$$
y=f(w^Tx+b)
$$

- `x`：input vector；
- `w`：learned weights；
- `b`：bias；
- `f`：nonlinearity，如 ReLU `max(0,x)`。

一层有多个 neurons，把 weight vectors 堆成 matrix：

$$
y=f(Wx+b)
$$

若同时处理 batch，把多条 `x` 堆起来，就从 matrix-vector multiply 变成 matrix-matrix multiply。系统角度不需要先理解学习理论；workload 的主体是 large dot products、reductions 与 elementwise transforms。

## 2. Convolution：局部连接、共享 weights

对 batch `N`、input height/width `H,W`、input channels `C`、filter spatial size `R,S`、output filters/channels `K`，输出空间为 `P,Q`：

$$
Y[n,p,q,k]=b[k]+
\sum_{r=0}^{R-1}\sum_{s=0}^{S-1}\sum_{c=0}^{C-1}
X[n,p+r,q+s,c]\,W[k,r,s,c]
$$

Direct implementation 有七层循环：

```text
for n in batch
  for p,q in output spatial positions
    for k in output filters
      for r,s in filter window
        for c in input channels
          accumulate X * W
```

同一 filter weights 在所有 spatial locations 复用；相邻 outputs 的 input windows 重叠；不同 batch/output positions/filters 提供大量 parallelism。

### 2.1 Filter 的含义不改变 systems 分析

- 全 `1/9` 的 3×3 filter 做 blur；
- 有正有负的 filter 可检测 horizontal/vertical gradient；
- DNN 只是让这些 weights 由 training 学得，并堆叠许多 filters/layers。

对于性能，关键 dimensions 是 `N,P,Q,K,R,S,C`，不是 filter 在语义上检测“边缘”还是“纹理”。

## 3. 四个互补的优化层次

### 3.1 Model/topology innovation

改变 depth、width、kernel size、connectivity、skip connection、depthwise separable convolution 等，可同时减少 parameters、FLOPs 和 activations。

课程展示早期 CNN 到后续网络在相近 accuracy 下，数年内 weights/compute 有约 25× 的算法级差距。Hardware 四年很难自动提供同等数量级的通用 speedup。

### 3.2 Software scheduling

在 topology 固定后，选择：

- loop order；
- tiling/blocking；
- SIMD/vectorization；
- layout/packing；
- operator fusion；
- kernel/library variant。

这是本讲主体。

### 3.3 Approximation/compression

- FP16/BF16/INT8/更低 bit precision；
- sparsity/pruning；
- quantization/low-rank 等。

它们可能减少 storage、bandwidth 和计算能耗，但要管理 accuracy、range 与 hardware support。

### 3.4 Hardware specialization

用 Tensor Core/TPU/NPU 等 larger-granularity matrix operation 摊薄 instruction/control overhead，提升 operations per joule。

> [!important]
> 四层不是替代关系。只做快 kernel，可能被更好的 model topology 淘汰；只压低 FLOPs，若 shape 与 hardware 不匹配也未必快；只有 Tensor Core 而没有 tiling/fusion，仍会饿死在 memory traffic 上。

## 4. Convolution 映射到 explicit GEMM

每个 output position 需要一个 `R×S×C` input patch。把 patch flatten 成一行：

```text
A / im2col matrix: (N·P·Q) × (R·S·C)
B / weight matrix: (R·S·C) × K
C / output matrix: (N·P·Q) × K
```

于是：

$$
C=A\times B
$$

### 4.1 为什么这很诱人

- 多 filters 自然成为 B 的多列；
- 多 channels 自然扩大 reduction dimension；
- cuBLAS/BLAS 等已有极度优化的 GEMM；
- 问题从七层 irregular indexing 规约成标准三层 matrix multiplication。

这延续了 [[Stanford CS149 - Lecture 08 - Data-Parallel Thinking|Lecture 8]]：把复杂算法表达为已有高性能 primitive。

### 4.2 Explicit im2col 的新成本

相邻 patches 重叠，同一 activation 会被复制到许多 rows。对 `R×S` filter，课件给出的近似是 input DRAM traffic/materialized storage 放大约 `R·S` 倍。

例如 3×3 convolution 的 im2col matrix 可能约为原 input 的 9 倍；batch、channels 与 backprop saved state 再扩大 footprint，容易耗尽 GPU memory。

> [!warning]
> 把算法“规约到 GEMM”只解决 compute kernel；若为使用 GEMM 先在 HBM 中构造巨大重复矩阵，可能把收益重新输给 memory traffic。

## 5. Naive GEMM 为什么没有发挥理论 arithmetic intensity

Square GEMM：

```cpp
for (int i = 0; i < N; ++i)
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < N; ++k)
      C[i][j] += A[i][k] * B[k][j];
```

数学上：

- work `Θ(N³)`；
- unique data `Θ(N²)`；
- 潜在 arithmetic intensity `Θ(N)`。

但 row-major layout 下：

- A row 连续，spatial locality 好；
- B column 跨行跳跃，几乎每次触碰不同 cache line；
- 若矩阵远大于 cache，A/B data 在不同 `(i,j)` 间反复从 DRAM 读；
- 内层每读 A、B 两个值只做一次 multiply-add。

实际 traffic 接近随 work 增长，achieved arithmetic intensity 退化为 `Θ(1)`，于是 bandwidth-bound。

### 5.1 “Unique bytes 少”不等于“实际 bytes 少”

Lecture 6 区分 inherent 与 artifactual communication。在 GEMM 里：

- `3N²` matrix data 是 logical unique footprint；
- 坏 loop order/cache capacity 造成的 `Θ(N³)` reload 是 artifactual traffic。

必须用 blocking 改变 execution order，才能接近潜在复用。

## 6. Blocked GEMM：把 reuse 放进 cache/shared memory

把 A、B、C 切成 `B×B` tiles：

```text
C_tile += A_tile × B_tile
```

一个 tile multiply 近似：

- 搬 `Θ(B²)` elements（A/B/C tiles）；
- 做 `Θ(B³)` multiply-add work。

所以：

$$
AI_{tile}=\Theta\left(\frac{B^3}{B^2}\right)=\Theta(B)
$$

Block 越大，reuse 越多；但 working set 必须容纳于目标 storage：

$$
bytes(A_{tile})+bytes(B_{tile})+bytes(C_{tile})
\lesssim capacity_{target}
$$

### 6.1 CPU cache 与 GPU scratchpad

| CPU cache | GPU shared memory |
|---|---|
| 同一地址空间的透明 hardware-managed copy | 显式 address space / software-managed scratchpad |
| 程序发普通 loads，cache 自动放置 lines | CUDA threads 合作把 global tiles copy 进 `__shared__` |
| 可能有 conflict/capacity misses | 受 shared-memory capacity/bank access 限制 |
| Replacement 由 hardware 决定 | Tile lifetime/overwrite 由 kernel 决定 |

两者目标相同：把 producer/consumer 在时间上靠近，使一个 off-chip load 支持许多 FMAs。

### 6.2 多级 blocking

现代 CPU/GPU 有 register、L1/shared、L2、LLC 等层级。高性能 GEMM 常嵌套 tiles：

```text
LLC-sized macro tile
→ L2/L1 tile
→ register micro-tile
→ SIMD/Tensor Core instruction tile
```

每层都在回答：当前层从下一层取得一次数据，能在本层复用多少次？

## 7. SIMD microkernel 与 layout choices

在小 tile 内，仍要决定怎样 vectorize：

### 7.1 Broadcast/splat A，vector-load B

```text
bvec = load contiguous B[k, j:j+W]
aval = splat A[i,k]
cvec += aval * bvec
```

一次更新 W 个 C outputs，B access 连续。代价包括 splat，并要维持多个 C accumulators 形成 ILP。

### 7.2 Transpose/pack B 后做 vector dot

若要沿 reduction dimension 向量化，可先把 B tile transpose/pack，使原本 column 成为 contiguous row，再做 SIMD dot product。

### 7.3 Register block 同时计算多个 outputs

高性能 microkernel 会在 registers 中保留一个 `Mr×Nr` C tile，加载 A/B fragments 后用 outer-product-like FMAs 更新，直到 reduction 完成再 store C。

### 7.4 没有一个 schedule 适合所有 layers

DNN 每层的 GEMM shape 可能是：

- tall-skinny；
- short-wide；
- small batch；
- channel/filter dimensions 非 SIMD/tile multiples。

因此 cuBLAS/cuDNN/autotuner 会按 shape、dtype、layout 和 workspace 选择不同 kernels。只实现一个 square-matrix-optimized GEMM，常在真实网络某些 layers 上很差。

## 8. Implicit GEMM：只在 on-chip tile 中“展开” convolution

Explicit GEMM 先把完整 im2col matrix 写到 HBM。Implicit GEMM 保留同样的 GEMM loop/tile structure，但 A matrix 的元素不来自 materialized array，而由 index mapping 直接从原 activation tensor 取：

```text
logical GEMM coordinate (row, reduction index)
→ decode batch/output position/filter offset/channel
→ load original X[n,h,w,c]
→ place only current fragment into shared memory/registers
```

### 8.1 关键区别

```text
explicit im2col:
X tensor → write full duplicated A to HBM → read A tiles → GEMM

implicit GEMM:
X tensor → gather current A tile directly into shared memory → GEMM
```

只在 unavoidable 的 DRAM→on-chip copy 时完成重排，不先在 DRAM 中创造另一个巨型副本。

### 8.2 Trade-off

收益：

- 不需要 `O(N·P·Q·R·S·C)` auxiliary HBM storage；
- 不增加一轮完整 im2col write/read；
- 保留成熟 GEMM microkernel。

代价：

- inner tile load 要做复杂 tensor address arithmetic；
- source addresses 可能不规则，coalescing/locality 需精心设计；
- padding/stride/dilation/boundary 增加 predicate；
- 有时需预计算 iterator/lookup metadata，用少量 memory 换地址计算。

CUTLASS 正是位于 raw CUDA assembly 与 high-level framework 之间的 building-block library：提供 warp/block GEMM、tensor iterators、shared-memory movement 和 reductions。

## 9. GPU 需要足够大的工作量

V100/A100 级 GPU 有许多 SM、warps 和 matrix pipelines。小 problem/batch 可能没有足够 blocks/tiles：

- SMs 空闲；
- latency hiding 不足；
- launch/setup overhead 占比高；
- matrix dimensions 无法高效填满 instruction tile。

视频展示 batch `N`、output `P×Q`、filter `R×S` 增大后 achieved TFLOP/s 上升。不是“多做工作本身让算法更优”，而是 fixed hardware 需要足够 independent tiles 才能接近 peak throughput。

> [!note]
> Training 大 batch 常有高 GPU utilization；latency-sensitive inference 的 batch=1 更难。后来 continuous batching 正是在不显著增加单请求等待的前提下，把多个 requests 聚合成可填满 GPU 的 work。

## 10. Direct convolution、FFT 与 Winograd

Conv 不必一定转 GEMM。

### 10.1 Direct convolution

直接对七层 loop nest 做 tiling/vectorization，避免 im2col。Compiler projects 如 XLA、Triton 等尝试从 tensor program 自动生成 schedule；最内层高性能 kernels 仍常依赖专家调优/模板搜索。

### 10.2 FFT convolution

Convolution theorem：

```text
input/filter FFT
→ frequency-domain pointwise multiply
→ inverse FFT
```

对 large filters 可能减少 asymptotic work，但有 transform、complex arithmetic、workspace 与 shape overhead；small 3×3 filters 未必值得。

### 10.3 Winograd convolution

通过代数重写/common subexpressions，用更多 additions 换更少 multiplications。课件的 1D 例子：两个 outputs、3-tap filter：

```text
direct:   6 multiplies + 4 adds
Winograd: 4 multiplies + 8 adds
```

在 2D 3×3 filter、2×2 output tile 上可显著减少 multiplications。是否更快取决于：

- multiply vs add cost；
- transform overhead；
- numerical stability/precision；
- tile size与memory traffic；
- batch/filter count 能否 amortize weight transforms。

## 11. Vendor library 的本质是多算法、多 schedule 选择器

cuDNN convolution API 不只收 X/W/Y pointers，还要 tensor descriptors、convolution parameters、algorithm/workspace preferences。候选包括：

- explicit GEMM/im2col；
- implicit GEMM；
- direct convolution；
- FFT variants；
- Winograd variants。

没有单个算法支配所有 shapes。Library/runtime 通常依据：

```text
N,H,W,C,K,R,S
stride/padding/dilation/groups
dtype/layout
available workspace
determinism constraints
device architecture
```

进行 heuristic 或 benchmark/autotune 选择。

> [!important]
> “调用 vendor library”不是放弃系统思考。你仍需要给它好的 layout、shape、batch、dtype、workspace，并理解为什么某层落入 memory-bound 或低-utilization regime。

## 12. Inter-op materialization：单个 GEMM 快还不够

一层常是：

```text
Conv/GEMM → scale+bias → activation → pooling → next layer
```

若每个 op 是单独 kernel：

1. Conv 把巨大 activation tensor 写 HBM；
2. scale+bias 读回、做极少 arithmetic、再写；
3. pooling 再读回、reduce、再写。

Scale/bias 与 pooling 的 arithmetic intensity 很低，严重 bandwidth-bound。中间 tensor 可能 1 GB，单独 materialization 的 bytes 远大于 elementwise math。

### 12.1 Producer-consumer fusion

若 blocked conv 正在 registers/shared/cache 中生成 output tile：

- 立即应用 scale+bias；
- tile 覆盖 2×2 region 后立即 max-pool；
- 只把最终 pooled output 写 HBM。

不仅省去中间 store+load，2×2 max-pool 的 final output 还只有 conv activation 的 1/4 spatial elements。

$$
AI_{fused}=\frac{work_{conv}+work_{epilogue}}{bytes_{input}+bytes_{weights}+bytes_{final\ output}}
$$

相比各 op 单独执行，分母删除了 intermediate reads/writes。

## 13. 为什么 fusion 在 framework 中难

早期 library 会为常见 pattern 手写 API：

```text
Conv2D + Bias + BatchNorm + ReLU
Conv2D + Resize + Pad
```

组合数量很快爆炸。更现代的路线是：

1. Framework 保留 tensor graph 和 operation semantics；
2. Compiler 分析 producer-consumer dependence；
3. 选择 “compute at”/tile boundary；
4. 生成一个 fused kernel，不经 HBM 交流 intermediates。

困难不只是拼 source code：

- 两个 ops 的最佳 tile/layout 可能冲突；
- fusion 增加 registers/shared memory，降低 occupancy；
- reduction/softmax 有跨元素 dependence；
- dynamic shapes/predicates 增加 variant space；
- 随机性、aliasing、numerical order 限制重排。

因此高性能系统需要两部分：excellent core GEMM/conv implementation + 让周边 bandwidth-bound ops 不拖慢它的 compiler/runtime intelligence。

## 14. Naive attention：计算不只是 FLOPs，最大问题是 `N²` materialization

忽略 scaling/mask，多头中的一个 attention 可写成：

$$
S=QK^T,\qquad P=softmax_{row}(S),\qquad O=PV
$$

其中：

```text
Q: N×d
Kᵀ: d×N
S/P: N×N
V: N×d
O: N×d
```

Naive operator-by-operator execution：

1. GEMM 生成 `S`，写 `N²` elements 到 HBM；
2. Softmax 多次读/写每行（max、exp/sum、normalize）；
3. 第二个 GEMM 读 `P` 和 `V` 生成 O。

当 `N=10,000`，单个 FP32 `N×N` matrix 约 400 MB；考虑 heads、batch、training intermediates，很快达到 GB 级。即使两个 GEMMs 各自高度优化，整个 pipeline 仍被 `S/P` memory traffic 和 capacity 支配。

## 15. Online softmax：把全行 reduction 变成可合并的状态

稳定 softmax 对 row vector `x`：

$$
m(x)=\max_i x_i
$$

$$
\ell(x)=\sum_i e^{x_i-m(x)}
$$

$$
softmax(x)_i=\frac{e^{x_i-m(x)}}{\ell(x)}
$$

看似必须先 materialize 整行，才能知道 global max/sum。把 `x` 分为 chunks `x^{(1)},x^{(2)}`，分别维护 `(m_1,\ell_1)` 与 `(m_2,\ell_2)`：

$$
m=\max(m_1,m_2)
$$

$$
\ell=e^{m_1-m}\ell_1+e^{m_2-m}\ell_2
$$

因此每个 tile 的 local max/sum 可以用常数大小 state 合并，既数值稳定又无需保存整行 scores。

### 15.1 为什么旧 partial output 需要 rescale

当新 tile 带来更大 max `m_{new}`，此前所有 exponentials 原本按 `m_{old}` 缩放。必须乘：

$$
e^{m_{old}-m_{new}}
$$

把旧 accumulator 转换到新 scale，再与新 tile contribution 合并。Fused algorithm 会做一些额外 rescaling arithmetic；这是用廉价 compute 换昂贵 HBM traffic。

## 16. Fused tiled attention / FlashAttention

对 Q row block `Q_i` 和 K/V column blocks `K_j,V_j`：

```text
keep Q_i and output accumulator O_i on chip

for each K_j, V_j tile:
    load K_j, V_j
    S_ij = Q_i K_jᵀ
    update per-row running max m_i and normalizer l_i
    rescale old accumulator if max changed
    accumulate exp(S_ij - m_i) V_j into O_i

normalize/store final O_i
```

### 16.1 节省了什么

- 从不在 HBM materialize 完整 `S` 或 `P`；
- Temporary score storage 从 `Θ(N²)` 降到 on-chip tile `Θ(B_rB_c)`；
- Q/K/V tiles 每次 load 后完成两段 matrix math 与 row reductions；
- O tile 留在 registers/shared/cache 中累积；
- Arithmetic intensity 显著提高。

完整 inputs/outputs 仍需 `Θ(Nd)` storage；消失的是 `Θ(N²)` intermediate。

### 16.2 它不是 approximation

在舍入误差范围内，online/tiled softmax 计算与普通 stable softmax 相同的数学结果。它改变的是 operation order 与 data movement，不是 attention definition。

### 16.3 为什么 compiler 自动发现它很难

普通 local fusion 看到 softmax 的 row reduction，会认为必须先完成整行，形成 global materialization boundary。要跨越它，需要识别 max/sum 的可分块代数结构并推导正确 rescaling。

一旦数学分解已知，底层实现又回到 CS149 熟悉的内容：loop reorder、tiling、producer-consumer locality、on-chip storage 与额外 compute/communication trade-off。

> [!note]
> 视频把该 Stanford 工作描述为把可处理 sequence length 从约 8K 推到约 32K，并获得小常数倍 speedup。具体数字依 hardware/model 而变；稳定结论是避免 `N²` HBM materialization 同时降低 memory footprint 与 I/O。

## 17. Fusion 的收益与代价要一起算

Fusion 常见收益：

- 删除 intermediate HBM store/load；
- 删除 kernel launch/synchronization boundary；
- 在 registers/shared memory 复用 producer output；
- 更早 reduce/compact，减少 final bytes。

Fusion 也可能带来：

- register pressure 与 spill；
- shared-memory pressure 与 occupancy 下降；
- 大 kernel instruction footprint；
- 重复计算/online rescaling；
- producer/consumer tile shape 冲突；
- 并行度下降。

评估应比较：

$$
\Delta T\approx
-\frac{bytes_{eliminated}}{BW}
-launch/sync_{eliminated}
+compute_{extra}
+spill/occupancy_{penalty}
$$

不能把 “fused” 当作自动更快的标签。

## 18. Low precision：同时改变 compute、capacity 与 bandwidth

把 FP32 换成 FP16/BF16/INT8：

- 同样 HBM bandwidth 每秒可搬更多 elements；
- cache/shared memory 容纳更大 tiles；
- register file 保存更多 values；
- SIMD/Tensor Core 一条 instruction 做更多 operations；
- model/activation footprint 下降。

但还要处理：

- dynamic range/overflow；
- accumulation precision；
- scaling/zero point；
- conversion/dequantization overhead；
- accuracy 与 calibration/training stability。

课程强调低精度和 sparsity 属于 approximation/compression 层，不能只以 nominal TOPS 判断端到端收益。

## 19. 为什么 GPU 适合 DNN

DNN 的高性能核心通常具有：

- 大量 independent output tiles；
- dense regular matrix operations；
- 良好 blocking 后的高 arithmetic intensity；
- 相同 instruction 作用于许多 elements，适合 SIMD/SIMT；
- 大 model/activation 需要高 memory bandwidth；
- 成熟的 cuBLAS/cuDNN/CUTLASS ecosystem。

GPU 原本为 graphics 建造的大量 ALUs、SIMD 与 latency hiding，恰好能承载这些 workloads。

## 20. 为什么 general-purpose GPU 又可能不是最优

普通 scalar/vector processor 每条 instruction 都要付出：

- fetch/decode/control；
- register/addressing；
- scheduling；
- generality 支持。

若 workload 大量重复 matrix multiply，可把 instruction granularity 从 FMA 扩大：

```text
scalar FMA
→ vector dot product
→ small matrix multiply-accumulate (MMA)
```

一条 4×4 或更大 MMA instruction 表达几十/上百 arithmetic ops，把 control overhead 摊到更多 math 上。

课件引用的历史能耗估计中，相对固定功能电路，half-precision FMA、4-component dot 与 4×4 MMA 的 programmability overhead 依次显著下降；具体百分比只是设计直觉，不是跨工艺恒定常数。

## 21. Tensor Core：受限但吞吐量极高的矩阵指令单元

课程以 A100 为例：每个 SM 除 FP32/INT32 ALUs 外还有 Tensor Cores，可执行小矩阵 multiply-accumulate。课件给出的全芯片历史峰值：

```text
FP32 CUDA cores: about 19.5 TFLOP/s
mixed FP16 input / FP32 accumulate Tensor Cores: about 312 TFLOP/s
```

FMA 按 2 FLOPs 计。数字体现 order-of-magnitude 差异，但只有当 computation 能：

- 表达为支持的 MMA tile；
- 使用支持的 dtype；
- 满足 alignment/layout；
- 提供足够 tiles；
- 持续供给 operands；

才能接近 Tensor Core peak。

Tensor Core 不是“自动加速所有 CUDA code”，而是一个 domain-specific instruction path。

## 22. Prefill 与 decode：用本讲框架理解 LLM inference（扩展）

> [!info] 课程概念延伸
> 视频讨论 transformer attention，但未展开 autoregressive serving 的 prefill/decode 区别。下面用本讲模型连接现代 AI Infra。

### 22.1 Prefill

一次处理长 prompt 的多个 tokens，Q/K/V 和 MLP 常是较大的 GEMMs：

- parallelism 大；
- weight reuse 较好；
- 更容易使用 Tensor Cores；
- attention 适合 FlashAttention tiling。

### 22.2 Decode

每 step 每 request 只增加一个 token，许多 linear layers 接近 matrix-vector/small-M GEMM：

- batch 小时 GPU tiles 填不满；
- 每 token 要读大量 weights/KV；
- arithmetic intensity 低，常 memory-bandwidth-bound；
- launch/dispatch overhead 更明显。

这正对应视频“batch size 1 性能下降”：GPU 峰值 FLOPs 很高，不代表低并行、小 shape 的 decode 能用上。

### 22.3 Continuous batching

Serving runtime 把不同 requests 的 decode steps 动态合批，提高 GEMM M dimension 和 resident work；代价是 scheduling complexity 与可能的 queueing latency。

## 23. 与 AI Infra 的更多连接

### 23.1 Epilogue fusion

现代 GEMM 常把 bias、activation、residual add、quantize 写进 epilogue，在 accumulator 仍在 registers 时完成。它就是本讲 `Conv + scale/bias` 的直接延伸。

### 23.2 Transformer fusion

常见 fusion targets：

- RMSNorm/LayerNorm + projection；
- bias + activation（如 SwiGLU）；
- residual add + norm；
- RoPE + Q/K layout transform；
- dequantization + GEMM；
- attention score/mask/softmax/value aggregation。

需要逐个比较 saved bytes 与 register/occupancy cost。

### 23.3 Quantized weight-only GEMM

Decode 常受 weight bandwidth 限制。INT8/INT4 weights 可减少每 token 读取 bytes；kernel 在 on-chip path dequantize，再用 higher-precision accumulate。即使增加 dequant math，端到端仍可能更快：

$$
\text{extra arithmetic} < \text{saved HBM transfer time}
$$

### 23.4 Shape-aware kernel dispatch

一个模型中 prefill、decode、QKV projection、MLP、vocab projection shapes 不同。高性能 runtime 会按：

```text
M,N,K + dtype + layout + GPU arch
```

选择/编译/autotune kernel，而非用一套 tile 参数覆盖全部 layers。

### 23.5 Activation memory 与 training

Training 要保存 activations 用于 backward。Explicit im2col/未融合 intermediates 会与 saved activations 共同放大 footprint。Gradient checkpointing 则反向使用同一个 trade-off：丢弃部分 activations，backward 时重算 forward，用 extra compute 换 memory capacity。

## 24. DNN kernel 的系统化分析清单

对每个 layer/op chain：

1. **Math**：FLOPs/MACs 是多少？Reduction dimensions 是什么？
2. **Logical data**：weights、inputs、outputs 各多少 bytes？
3. **Actual traffic**：是否重复 reload/materialize/transpose？
4. **Arithmetic intensity**：在 HBM、L2、shared-memory 各层分别是多少？
5. **Parallelism**：有多少 output tiles/warps？Batch/shape 足以填满 GPU 吗？
6. **SIMD/MMA fit**：dimensions 是否适合 vector/Tensor Core tile？
7. **Locality**：A/B/activation 的 reuse 是否在目标 storage lifetime 内发生？
8. **Fusion**：哪些 producer-consumer intermediates 可不落 HBM？
9. **Resources**：register/shared memory 是否限制 occupancy 或造成 spill？
10. **Algorithm variant**：direct、implicit GEMM、FFT、Winograd、Flash-style tiling 哪个适合该 shape？
11. **Precision**：降低 bytes/op 后，accuracy 和 conversion cost 是否可接受？
12. **End-to-end**：最快单 kernel 是否真的缩短模型 latency/提高 throughput？

## 25. 本讲结论

1. Neuron、fully connected 与 convolution 最终都可视为 dot products、reductions 和 tensor transforms。
2. 更好的 model topology 可能带来远大于单代 hardware 的效率提升。
3. Explicit im2col 把 convolution 变成 GEMM，却以 `R·S` 级数据复制和 auxiliary storage 为代价。
4. Naive GEMM 虽有 `Θ(N)` 潜在 arithmetic intensity，坏 schedule 会因重复 DRAM traffic 退化成 `Θ(1)`。
5. Blocking 让 `B×B` tiles 在 on-chip storage 复用，把 arithmetic intensity 提升到 `Θ(B)`。
6. CPU cache 是透明的，GPU shared memory 是 software-managed scratchpad；目标都是 producer-consumer locality。
7. Implicit GEMM 只在 shared memory/register tile 中展开 convolution matrix，避免 HBM materialization。
8. 不同 DNN layers/shapes 需要不同 schedule，library/autotuning 不可少。
9. 单个 GEMM 很快仍不够；elementwise/pooling intermediates 若反复落 HBM，会让整网 bandwidth-bound。
10. Online softmax 使 attention 可以 tile/fuse，消除 `N²` score/probability matrices。
11. Fusion 可能增加 compute 与 resource pressure，必须用 saved traffic 与 occupancy 一起评估。
12. Low precision 与 Tensor Core 同时利用更小数据和更大 instruction granularity，但要求 shape/layout/dtype 匹配。
13. DNN efficiency 是 model、algorithm、software schedule/compiler 与 hardware 的 cross-layer co-design。

## 自测题

1. 透明 circle renderer 为什么不能直接 one-thread-per-circle？改变并行轴后怎样保证正确？

    **面试回答：** 多个 circle 可能同时写同一 pixel，既有写竞争，又会打乱透明 alpha blending 所要求的前后顺序；普通原子写也不能自动恢复这个顺序。可改为并行处理 pixels，每个 pixel 由一个线程拥有，并按原 circle 顺序依次混合覆盖它的候选圆，候选列表可先分桶筛选。

2. 写出 `N,P,Q,K,R,S,C` convolution 的七层 loop/reduction 结构。

    **面试回答：** 外层遍历 n∈[0,N)、p∈[0,P)、q∈[0,Q)、k∈[0,K)，先置 `Y[n,p,q,k]=b[k]`；内层遍历 r∈[0,R)、s∈[0,S)、c∈[0,C)，累加 `X[n,p+r,q+s,c]*W[k,r,s,c]`。四个输出轴可并行，r、s、c 为归约轴；此处假设 stride=1，边界 padding 按约定处理。

3. Conv→GEMM 时 A、B、C 三个 matrices 的 dimensions 分别是什么？

    **面试回答：** 按本文把每个输入 patch 展成一行的约定，A 的维度是 $(NPQ)\times(RSC)$，权重 B 是 $(RSC)\times K$，输出矩阵 C 是 $(NPQ)\times K$，满足 $C=AB$。再把输出 reshape 回 $N\times P\times Q\times K$；其他约定可能整体转置，但归约维必须一致。

4. 3×3 explicit im2col 为什么会近似放大 9 倍 input representation？

    **面试回答：** Explicit im2col 为每个输出位置复制一份 $3\times3\times C$ patch，相邻 patch 大量重叠。展开元素数为 $NPQ\cdot9C$，原输入为 $NHWC$，比值为 $9PQ/(HW)$；只有 stride=1 且输出空间与输入近似相等时才约为 9 倍，边界和 stride 会改变比例。

5. Square GEMM 的潜在 arithmetic intensity 为何是 `Θ(N)`，naive loop 又为何可能只有 `Θ(1)`？

    **面试回答：** Square GEMM 做 $\Theta(N^3)$ 运算，而唯一输入输出只占 $\Theta(N^2)$ 元素，理想充分复用下强度为 $\Theta(N)$。Naive loop 在矩阵远大于 cache 时可能不断重载 A/B，实际流量也接近 $\Theta(N^3)$，于是强度退化为 $\Theta(1)$；唯一数据量不等于实际传输量。

6. `B×B` blocked GEMM 为什么有 `Θ(B)` arithmetic intensity？Block size 为什么不能无限增大？

    **面试回答：** 一次 tile 乘法用 $\Theta(B^2)$ 的 A/B/C 数据完成 $\Theta(B^3)$ 运算，因此在 tile 能被复用的前提下，算术强度为 $\Theta(B)$。B 太大会超过 cache/shared memory 或寄存器容量，产生重载、spill 或降低驻留并行度，还会增加边界浪费，所以不能无限增大。

7. CPU cache 与 CUDA shared memory 在管理方式上有什么本质区别？

    **面试回答：** CPU cache 由硬件自动缓存普通地址空间的 cache lines，替换和缺失通常对程序透明。CUDA shared memory 是 block 内显式使用的 scratchpad，程序负责协作加载、数据布局、生命周期和同步；两者都用于复用数据，但控制责任与容量约束的表达不同。

8. GEMM microkernel 为什么要同时保留多个 C accumulators？

    **面试回答：** 多个 C accumulators 提供相互独立的 FMA 依赖链，帮助覆盖一次乘加的延迟、提高指令级并行。同时一份加载的 A/B fragment 能更新多个输出，提高寄存器内复用；accumulators 太多又会增加寄存器压力，可能导致 spill 或降低 occupancy。

9. Implicit GEMM 相比 explicit im2col 节省什么，又增加什么？

    **面试回答：** Implicit GEMM 不在 HBM 中物化完整 im2col 矩阵，而是在加载当前 tile 时从原张量按需取数，省去大辅助存储及展开矩阵的一轮写读。代价是更复杂的地址计算、边界谓词和访存组织，需要精心设计 iterator、布局和合并访问。

10. Batch size 1 为什么可能远低于 GPU peak throughput？

    **面试回答：** Batch=1 可能让矩阵的某个输出维度很小，独立 tiles 不足、权重复用差，launch 开销和尾部 mask 占比也更高，难以填满 GPU。它不是必然低效：卷积仍可通过大量空间位置产生并行度，因此要看完整 shape、算术强度和 tile 数。

11. Direct、FFT、Winograd、implicit GEMM 各适合什么条件？

    **面试回答：** Direct 常适合小规模、分组或特殊形状，避免展开开销；FFT 对较大卷积核可能用变换成本换更少计算。Winograd 常用于小型、stride=1 的卷积核，但受变换和数值精度限制；implicit GEMM 适合能高效映射矩阵 tiles 的密集卷积，最终需结合 shape、dtype 和 workspace 测量选择。

12. 为什么 Conv 后单独执行 scale+bias 很容易 bandwidth-bound？

    **面试回答：** Scale+bias 每元素只有乘法和加法，却要把卷积输出从 HBM 读回再写出；若用 FP32 并忽略可复用参数，约为 2 FLOPs/8 bytes，强度很低。将它融合进卷积或 GEMM 的 epilogue，可在 accumulator 仍驻留片上时完成，省掉这次额外读写。

13. Fusion 可能降低 performance 的三个原因是什么？

    **面试回答：** 第一，寄存器和 shared-memory 占用上升，导致 spill 或不足的驻留 warps；第二，生产者与消费者的最佳 tile/layout 不兼容，降低计算效率或并行度。第三，融合引入额外重计算、同步、控制与更大的指令开销；节省的 HBM 和 launch 时间必须大于这些代价。

14. Naive attention 的 `N²` intermediates 在哪些步骤被读写？

    **面试回答：** 第一个 GEMM 把 $S=QK^T$ 的 $N\times N$ 分数矩阵写到 HBM，softmax 再读取 S 并生成同样大小的 P；若分成多个子步骤，还会额外读写 max/exp/normalize 的中间量。第二个 GEMM 又从 HBM 读取 P 来计算 PV，核心问题是两次矩阵乘之间物化了平方级张量。

15. 怎样用 `(m,ℓ)` 合并两个 softmax chunks？旧 accumulator 为什么要 rescale？

    **面试回答：** 令 $m=\max(m_1,m_2)$，则 $\ell=e^{m_1-m}\ell_1+e^{m_2-m}\ell_2$。若 $a_j=\sum_i e^{x_i-m_j}v_i$ 是未归一化输出累加器，同样合并为 $a=e^{m_1-m}a_1+e^{m_2-m}a_2$，最后输出 $a/\ell$；rescale 是把旧贡献换到共同的最大值尺度。

16. FlashAttention 为什么是 exact reordering，而不是 attention approximation？

    **面试回答：** FlashAttention 仍计算相同的点积、mask、softmax 和对 V 的加权求和，只通过分块与 online softmax 改变运算顺序和数据驻留位置，避免完整 S/P 写入 HBM。它没有用稀疏化或低秩替代 attention 定义；浮点舍入可不同，因此 exact 不代表与朴素实现逐位一致。

17. Tensor Core 的高 peak 为什么不自动转化为任意 DNN op 的高 performance？

    **面试回答：** Tensor Core 的峰值只适用于受支持的矩阵乘加形状、dtype 和布局，并要求足够多的 tiles 与持续供数。DNN 还包含归约、激活、索引和通信等操作，小形状、padding 或带宽瓶颈也会使 MMA 单元闲置；端到端性能取决于整个算子链而非某类单元峰值。

18. 为什么 LLM prefill 更像大 GEMM，而 decode 更容易 memory-bound？

    **面试回答：** Prefill 一次处理多个 prompt tokens，线性层有较大的 M 维，权重能在更多 token 间复用，通常更像高强度 GEMM。小 batch decode 每请求每步只处理一个新 token，线性层接近矩阵向量乘，还需读取 KV cache，因此更易受内存带宽限制；增大有效 batch 会改变这一判断。

19. INT4 weight-only quantization 为什么可能“多做 dequant math 却更快”？

    **面试回答：** INT4 权重相对 FP16 可把纯权重存储压到约四分之一，再加 scale 等元数据；在权重带宽主导的 decode 中，省下的 HBM 传输可能超过解包和反量化开销。前提是融合反量化路径高效、量化精度可接受，且 kernel 没转而受计算、元数据或小形状限制。

20. 对一个 fused Transformer kernel，怎样同时检查 HBM bytes、register pressure、occupancy 与 shape fit？

    **面试回答：** 固定同一 shape 与精度先比较端到端耗时，统计实际 HBM 读写及中间张量是否消失；再查每线程寄存器、spill 和 shared-memory 用量。结合 resident/ready warps 判断延迟隐藏，检查矩阵维度、布局、尾部 mask 与 MMA tile 是否匹配，通过 tile 或融合范围对照实验定位收益和代价。


## 参考资料

- [CS149 Fall 2023 Lecture 10 视频](https://www.youtube.com/watch?v=qbKtU0X6-WU)
- [Lecture 10 官方课件](https://gfxcourses.stanford.edu/cs149/fall23content/media/dnneval/10_dnneval.pdf)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/)
- [NVIDIA cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [[Stanford CS149 - Lecture 07 - GPU Architecture and CUDA Programming|Lecture 7：CUDA、Warp、SM 与 Shared Memory]]
- [[Stanford CS149 - Lecture 08 - Data-Parallel Thinking|Lecture 8：Primitives、Scan 与 Fusion]]
- [[Stanford CS149 - Lecture 06 - Performance Optimization II Locality Communication and Contention|Lecture 6：Arithmetic Intensity、Locality 与 Roofline]]
- [LLM Inference](../../topics/inference/LLM%20Inference.md)
