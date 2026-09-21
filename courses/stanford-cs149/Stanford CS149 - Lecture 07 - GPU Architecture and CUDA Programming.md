---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 7
lecture_date: 2023-10-17
area: systems
topics:
  - gpu
  - cuda
  - spmd
  - simt
  - warp
  - shared-memory
  - latency-hiding
  - occupancy
aliases:
  - CS149 Lecture 7
  - GPU Architecture and CUDA Programming
video_url: https://www.youtube.com/watch?v=qQTDF0CBoxE
---

# Stanford CS149 - Lecture 07 - GPU Architecture and CUDA Programming

> [!abstract]
> 本讲把前几讲的 multicore、SIMD、multithreading 和 locality 放到 GPU 上重新组合。CUDA 提供 bulk-synchronous 的 SPMD 编程模型：程序员启动大量 CUDA threads，并用 thread blocks 表达局部协作；GPU 再把 blocks 调度到 SM，把相邻 threads 组成 warp 做隐式 SIMD，并保留大量 resident warps 隐藏延迟。理解 GPU 的关键不是背“几千个 cores”，而是分清逻辑 thread、warp、block、SM 四个层次，以及 shared memory、barrier 和资源占用怎样限制调度。

## 来源与范围

- [Lecture 7 视频：GPU Architecture and CUDA Programming](https://www.youtube.com/watch?v=qQTDF0CBoxE)
- [官方课件 PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/gpucuda/07_gpuarch.pdf)
- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)

本文按视频顺序组织，代码、数字与架构图再用课件校正。课堂以 Volta V100 的 SM 为具体例子；后续 NVIDIA 架构会改变单元数量和部分调度细节，但 CUDA 的 block/warp/SM 分层仍是理解现代 GPU 的基础。

## 视频索引

| 时间 | 内容 |
|---|---|
| [00:18](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=18s) | 本讲路线：历史、CUDA、GPU architecture |
| [01:30](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=90s) | GPU 没有引入新基本概念：仍是 multicore、SIMD、multithreading |
| [02:37](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=157s) | 图形渲染与 per-pixel shader 的数据并行性 |
| [08:06](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=486s) | 通用计算如何从 GPU 的并行增长中获益 |
| [09:17](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=557s) | 用两个全屏三角形“伪装”通用计算的早期 GPGPU |
| [10:24](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=624s) | Brook 与 stream/data-parallel programming |
| [13:44](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=824s) | 从 graphics pipeline 接口转向 compute mode |
| [15:05](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=905s) | CUDA：给定 kernel，一次启动 N 个 SPMD 实例 |
| [17:31](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=1051s) | CUDA thread 与 ISPC program instance 的对应 |
| [18:40](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=1120s) | Grid、thread block 与多维索引 |
| [24:10](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=1450s) | Host CPU 与 device GPU 两个执行世界 |
| [26:01](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=1561s) | 独立地址空间、`cudaMalloc` 与 `cudaMemcpy` |
| [29:14](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=1754s) | 边界 block 会产生多余 threads，kernel 内必须检查范围 |
| [32:46](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=1966s) | Thread-local、block-shared 与 device-global 地址空间 |
| [34:07](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=2047s) | 一维 convolution 示例 |
| [36:47](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=2207s) | Cooperative loading 与 shared memory tiling |
| [39:11](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=2351s) | `__syncthreads()` 为什么不可省 |
| [45:25](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=2725s) | CUDA 的同步手段与 programming model 小结 |
| [46:21](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=2781s) | 一百万 CUDA threads 不等于一百万物理执行单元 |
| [48:06](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=2886s) | Kernel metadata 与硬件 work scheduler |
| [50:26](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=3026s) | V100 SM sub-core、执行上下文与 SIMD ALUs |
| [53:01](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=3181s) | Implicit SIMD：相同 PC 的 CUDA threads 一起执行 |
| [54:53](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=3293s) | Warp 是 32 个 CUDA threads 的硬件执行分组 |
| [60:25](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=3625s) | 完整 SM：warp schedulers、shared memory 与 latency hiding |
| [66:54](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=4014s) | Block 的资源分配与 V100 全芯片规模 |
| [70:19](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=4219s) | Blocks 怎样驻留、结束并补充到 SM |
| [73:00](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=4380s) | Barrier 决定 block 必须整体 resident |
| [76:22](https://www.youtube.com/watch?v=qQTDF0CBoxE&t=4582s) | 跨 block 原子操作合法，但不能假设 block 执行顺序 |

## 1. GPU 的起点：极大规模、结构一致的数据并行

传统图形管线需要为一帧中的大量顶点和像素执行相似程序。以 fragment/pixel shader 为例：

```text
每个 pixel 获得位置、法线、纹理坐标等输入
→ 独立运行 material/shading program
→ 输出 RGBA color
```

4K 图像有数百万像素，还要每秒刷新几十次。工作之间高度独立、控制流相似，因此增加 core 和 SIMD ALU 比继续追求单线程延迟更自然。

这正好避开 2000 年代 CPU 面临的两个瓶颈：

- 提高时钟频率会使功耗迅速上升；
- 单线程中可提取的 ILP 已经有限。

GPU 很早就把新增晶体管投入并行吞吐量。于是研究者发现：即使最初接口只允许“画三角形”，底层芯片已经是一台很强的数据并行处理器。

## 2. 从 GPGPU hack 到 CUDA

### 2.1 全屏三角形的 hack

早期通用 GPU 计算需要伪装成渲染：

1. 画两个覆盖整个屏幕的三角形；
2. 让图形管线为每个 pixel 调用 shader；
3. shader 不再计算颜色，而是更新时间步、粒子位置或科学计算数据；
4. 把 RGBA 四个通道解释成普通数值。

若目标是生成 `512×512` 个函数调用，就渲染覆盖 `512×512` pixels 的表面。这证明了 GPU 的计算能力，却把计算问题强行塞进 graphics API。

### 2.2 Brook：先暴露数据并行，再翻译回图形接口

Stanford 的 Brook 研究项目提供 stream/data-parallel abstraction：函数看似作用于 scalar，调用时却映射到 collection 的所有元素。编译器再 source-to-source 生成底层 graphics program。

它改善了用户接口，但实现仍绕过图形管线。真正需要的是 GPU 自己提供 compute interface。

### 2.3 CUDA compute mode

2007 年 NVIDIA 把接口改成：

```text
给 GPU 一段 kernel program
+ 要执行的逻辑实例数量
→ GPU 自己决定怎样把这些实例映射到硬件
```

这就是 CUDA 的核心。它既不让用户逐个创建 OS-style threads，也不要求继续画三角形；它通过一次 bulk launch 表达大量同构 SPMD work。

> [!important]
> ISPC 在历史上可以看作对 CUDA programming style 的回应：既然 CUDA 能用 SPMD 风格编程 GPU，能否用相似抽象高效编程 CPU SIMD？两者语法和逻辑实例很像，硬件映射却不同。

## 3. CUDA programming model 的四层结构

```text
kernel launch
└── grid：一次 launch 中的所有 blocks
    ├── block (blockIdx)
    │   ├── CUDA thread (threadIdx)
    │   ├── CUDA thread
    │   └── ...
    └── block
```

### 3.1 CUDA thread 是逻辑实例

CUDA kernel 是 single program，系统为不同 `threadIdx` / `blockIdx` 运行许多实例：

```cpp
__global__ void matrix_add(float* C, const float* A, const float* B,
                           int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height)
        C[j * width + i] = A[j * width + i] + B[j * width + i];
}
```

这里的 “thread” 是 CUDA 定义的 program instance，不应自动脑补成 CPU 上拥有完整独立 core 的 OS thread。它最终可能只是 warp 中的一条 lane。

### 3.2 Block 是程序员声明的协作组

Threads 被显式组织成 blocks。一个 block 中的 threads：

- 被放置在同一个 SM 上；
- 能访问该 block 的 shared memory；
- 能执行 block-wide barrier；
- 在生命周期中必须能够同时保持 resident。

不同 blocks 原则上是独立工作，调度器可以按任意顺序、在任意 SM 上执行它们。

### 3.3 多维索引是便利，不是本质

CUDA 支持 1D/2D/3D 的 grid 和 block，便于 image、matrix 与 tensor 索引。例如：

```cpp
dim3 block_dim(4, 3);                     // 每 block 12 threads
dim3 grid_dim(ceil_div(width, 4),
              ceil_div(height, 3));
matrix_add<<<grid_dim, block_dim>>>(...);
```

二维 ID 能直接得到 `(i,j)`，省去从线性 ID 反算多维坐标时的一些 divide/modulo。但同一算法完全可以用一维 ID 表达，多维不是 CUDA 能并行的原因。

### 3.4 与 ISPC 的近似映射

| ISPC | CUDA | 含义 |
|---|---|---|
| program instance | CUDA thread | 一个 SPMD 逻辑实例 |
| `programIndex` | `threadIdx` | 协作组内实例 ID |
| task | thread block（近似） | 可独立调度的一组实例 |
| task index | `blockIdx` | 组 ID |
| gang | warp（实现层近似） | 一起执行 SIMD 的实例集合 |

最后一行只是帮助建立直觉：ISPC gang 宽度由编译目标确定，CUDA warp 则是 NVIDIA 的硬件执行分组，不是 source-level block。

## 4. Launch rounding：创建的是 threads，不是“对每个元素自动调用”

假设矩阵是 `11×5`，block 是 `4×3`：

```text
grid.x = ceil(11/4) = 3
grid.y = ceil(5/3)  = 2
total CUDA threads = 3×2×4×3 = 72
valid elements = 55
```

最后一列和最后一行的部分 threads 没有对应元素，所以 kernel 内必须做 bounds check。

> [!warning]
> CUDA 语义不是“数组每个元素自动对应一个 thread”。程序员只声明 grid/block 尺寸；每个实例再根据 ID 决定要做什么。创建多余 threads 本身没错，越界 load/store 才是错误。

这也意味着一个 thread 不必只处理一个元素。为了减少 launch/index overhead 或提高复用，也可以让每个 thread 循环处理多个元素。

## 5. Host 与 device：两套执行环境和地址空间

本讲先采用最清晰的离散 GPU 模型：

```text
CPU / host address space       GPU / device address space
host pointer A                 device pointer d_A
host DRAM                      device DRAM (global memory)
          \______ PCIe / interconnect ______/
```

典型执行流程：

```cpp
float* d_A;
cudaMalloc(&d_A, bytes);
cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

kernel<<<grid, block>>>(d_A);

cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost);
cudaFree(d_A);
```

在这个模型里：

- `malloc` 得到 host allocation；
- `cudaMalloc` 得到 device allocation；
- CPU 不能直接解引用 device pointer；
- GPU kernel 不能直接把普通 host pointer 当作本地 DRAM pointer；
- `cudaMemcpy` 是两个地址空间之间的显式数据移动。

现代系统支持 unified virtual addressing、managed memory、CPU-accessible GPU memory 等能力，但地址看起来统一不代表代价统一。离散 GPU 从 host memory 取数据仍可能跨 PCIe，远慢于访问本地 HBM/GDDR。

### 5.1 数据传输也是 message passing

从系统视角看，host-to-device `cudaMemcpy` 很像一条大消息：

- 有传输 latency；
- 受 interconnect bandwidth 限制；
- 可用 asynchronous copy 与计算 overlap；
- buffer 在 operation 完成前有生命周期约束。

因此端到端 GPU 加速必须把数据传输算进去。一个 kernel 即使比 CPU 快很多，频繁的小批量 H2D/D2H copy 也可能吞掉全部收益。

## 6. CUDA device memory scopes

| Scope | CUDA 表达 | 谁能访问 | 典型物理位置/用途 |
|---|---|---|---|
| Per-thread | 默认局部变量 | 当前 thread | registers；溢出时可能进入 local memory |
| Per-block | `__shared__` | 同一 block 的 threads | SM 上低延迟 scratchpad |
| Device-global | `cudaMalloc` allocation | 所有 CUDA threads | GPU DRAM，经 cache hierarchy |

Shared memory 的“shared”是 **block 内共享**，不是整张 GPU 共享。它由程序员显式管理，常用于：

- 把可复用的 global data 先 tile 到 SM；
- 在线程间交换 partial result；
- reduction、scan、matrix tile 等局部协作。

Block 的存在因此不是单纯的命名层次，而是在告诉 GPU：这组 threads 有 locality 和 synchronization 关系，应共同放到一个 SM。

## 7. 一维 convolution：用协作加载创造跨 thread 复用

设输出为三点卷积：

$$
y[i] = w_0x[i] + w_1x[i+1] + w_2x[i+2]
$$

### 7.1 直接版本

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
y[i] = w0*x[i] + w1*x[i+1] + w2*x[i+2];
```

每个 thread 发出 3 次 input load。相邻输出窗口高度重叠，但任何单个 thread 都没有 reuse；复用发生在相邻 threads 之间。Cache 可能自动命中，却没有在程序中明确保证数据只从 global memory 取一次。
>[!note] 这里学生有提问怎样组织matrix在cuda 内存中的排列顺序也会对性能有影响，老师说他指定的行优先排序
### 7.2 Shared-memory tiled 版本

若一个 block 计算 128 个输出，需要输入窗口共 `128+2=130` 个元素：

```cpp
__shared__ float support[130];

int local = threadIdx.x;
int base  = blockIdx.x * blockDim.x;

support[local] = x[base + local];
if (local < 2)
    support[128 + local] = x[base + 128 + local];

__syncthreads();

y[base + local] =
    w0*support[local] +
    w1*support[local+1] +
    w2*support[local+2];
```

工作分为三阶段：

1. 128 个 threads 各加载一个元素；thread 0、1 再加载 halo 的两个元素；
2. 所有 threads 在 barrier 等待 block-local tile 完整；
3. 每个 thread 从 shared memory 读取三点并输出。

若只按 kernel 发出的标量 load 次数估算：

```text
直接版本：128 outputs × 3 = 384 global loads
tiled 版本：130 global loads + shared-memory accesses
```

global-load 指令量约减少到原来的 `130/384 ≈ 34%`。真实 DRAM transaction 还会受 cache、coalescing 和 cache-line granularity 影响，不能直接把 384 与 130 当作 HBM transaction 数。

### 7.3 `__syncthreads()` 为什么必要

所有 threads 是并发逻辑实例，GPU 没承诺 thread 0 加载完后才运行 thread 1 的计算。没有 barrier，某个 thread 可能读取尚未由邻居写入的 `support[local+1]`。

```text
cooperative stores to shared memory
                  ↓
             __syncthreads()
                  ↓
all prior shared writes visible to block
                  ↓
parallel convolution reads
```

Barrier 建立 phase boundary：到达之后，才可假设 block 内所有前序 shared-memory writes 已完成。

> [!warning]
> `__syncthreads()` 必须由 block 中所有相关 threads 以一致方式到达。把它放入只有部分 threads 执行的分支，可能导致其余 threads 永远等不到同伴。

## 8. CUDA 的同步边界

本讲介绍三类常见同步：

- `__syncthreads()`：同一 block 内的 barrier；
- atomics：对 shared 或 global memory 做不可分割更新；
- host/device completion：host 等待 kernel 或 copy 完成。

其中最重要的结构性限制是：**普通 kernel 内没有可供任意 blocks 使用的全局 barrier。** Kernel launch 的 blocks 必须能被独立地以任意顺序调度。

### 8.1 跨 block 访问同一 global memory 可以合法

Histogram 中，不同 blocks 可以：

```cpp
atomicAdd(&bins[value], 1);
```

这不依赖 block 顺序。并发时会产生 atomic contention，性能可能差，但语义正确。

### 8.2 跨 block 等待对方可能死锁

下面的模式危险：

```text
block 1: while (!flag) { }   // 等 block 0
block 0: flag = 1
```

若 GPU 同时只能驻留一个 block，而调度器先运行 block 1，它会一直占着资源等待；block 0 永远无法被调度。

因此 blocks 可以通过 global memory 交互，但 kernel 不能假设：

- 哪个 block 先执行；
- 所有 blocks 同时 resident；
- `blockIdx` 小的 block 先完成。

需要全局阶段边界时，最清晰的做法通常是结束当前 kernel，再启动下一个 kernel。

## 9. Bulk launch 如何落到有限硬件上

一个 kernel 可以声明约一百万 CUDA threads、约八千个 blocks。GPU 不会寻找一百万套物理 context；它把待执行 blocks 维护成 work pool：

```text
grid 中的 independent blocks
        ↓ hardware scheduler
可容纳资源的 SM 获得一个或多个 blocks
        ↓
block 完成并释放 registers/shared memory/thread contexts
        ↓
调度下一个 block
```

编译后的 kernel 除了 instruction stream，还携带资源需求，例如：

- threads per block；
- registers per thread；
- shared memory bytes per block。

调度器只有在一个 SM 的剩余资源都足够时，才能让新 block resident。这与 Assignment 2 的 task scheduler 思想相同，只是调度逻辑直接实现于 GPU hardware。

## 10. Warp：CUDA SPMD 怎样获得 SIMD 吞吐量

### 10.1 CPU 显式 SIMD 与 GPU 隐式 SIMD

CPU SIMD 常由 compiler 在 binary 中生成明确的 vector instruction，vector register 本身包含多条 lanes。

GPU 侧每个 CUDA thread 从抽象上拥有自己的 scalar registers 和 program counter。硬件查看一组相邻 threads；如果它们下一条 instruction 相同，就用 SIMD ALUs 一起执行。

```text
32 CUDA thread contexts
  each: scalar registers + PC
            ↓ same instruction?
         one warp instruction
            ↓
       SIMD execution lanes
```

这种实现常称 SIMT（single instruction, multiple threads）或 implicit SIMD。课堂强调它在性能直觉上仍然就是 SIMD，只是向程序员暴露的抽象不同。

### 10.2 Warp 是硬件分组，不是 block

NVIDIA 把 32 个连续 CUDA threads 组成一个 warp：

- warp 是调度和 SIMD execution 的基本单位；
- block 是编程模型中的协作与共享内存单位；
- 一个 128-thread block 通常对应 4 warps；
- 程序员声明 block size，但不会用 CUDA 语法“创建一个 warp”。

即使某代硬件只有 16 条相关 ALUs，一个 32-thread warp 也可分两个 cycles 执行。Warp width 与同类 ALU 数不是必须一一相等。

### 10.3 Divergence 等价于 SIMD masking

若 warp 内 threads 走不同分支：

```cpp
if (threadIdx.x % 2 == 0)
    path_A();
else
    path_B();
```

硬件必须分别推进不同 PC 的 threads，并屏蔽另一组 lanes。若 A、B 路径各占一半且工作量相当，许多 lane 在每个阶段都闲置。

CUDA thread 拥有独立 PC 给硬件留下了更灵活的实现空间，但对大多数性能分析，仍可把 warp 当作 32-wide masked SIMD。连续 thread IDs 控制流越一致，SIMD utilization 通常越高。

## 11. SM：multithreading、SIMD 与多发射的组合

课堂用 V100 说明一个 SM 的思想。简化看，一个 SM 包含：

- 多组 warp execution contexts；
- 多个 warp schedulers / fetch-decode 单元；
- FP、integer、load/store、transcendental 等多类 execution pipelines；
- shared memory / L1；
- 大量 register storage。

一个 V100 SM 可保存约 64 warps，也就是 `64×32=2048` 个 live CUDA thread contexts。每个 cycle，多个 scheduler 从可运行 warps 中挑选少数 warp 发射：

```text
resident warps: W0 W1 W2 ... W63
                    ↓ choose ready warps
warp schedulers: issue up to several warp instructions
                    ↓
SIMD execution pipelines
```

### 11.1 大量 resident warps 用来隐藏延迟

如果 warp A 的 global load 尚未返回，它暂时不可运行。SM 不必让 ALU 空等，可以发射 warp B、C、D 的 ready instructions。

这就是 hardware multithreading 的 GPU 版本：

$$
\text{latency hiding capacity}
\uparrow
\quad\text{as runnable resident warps}\uparrow
$$

但它只能隐藏 latency，不能创造 bandwidth。如果所有 warps 都持续发出内存请求，最终仍受 HBM bandwidth 限制。

### 11.2 V100 课堂数字的意义

视频以 80 个 SM、约 1.2 GHz、每 SM 多组 FP SIMD lanes 估算约 `12.7 TFLOP/s` FP32 峰值，并指出全芯片可有约 16.3 万个 live CUDA thread contexts。

这些数字不是要求死记的规格表。它们说明：

- 峰值来自 `SM 数 × 每 SM ALUs × clock × 每指令 operations`；
- “同时 live”不等于所有 CUDA threads 每 cycle 都执行；
- GPU 用非常多 contexts 供 scheduler 选择，以保持有限的 ALUs 忙碌。

## 12. Residency、资源约束与 occupancy

假设一个 block 需要：

```text
128 threads = 4 warps
520 B shared memory（`130 × sizeof(float)`；视频后面的简化调度例子按约 512 B 讲解）
R registers per thread
```

一个 SM 可同时放几个 blocks，取决于多重上限：

$$
B_{resident}=\min\left(
B_{arch},
\left\lfloor\frac{T_{SM}}{T_{block}}\right\rfloor,
\left\lfloor\frac{S_{SM}}{S_{block}}\right\rfloor,
\left\lfloor\frac{R_{SM}}{R_{block}}\right\rfloor
\right)
$$

其中：

- `T` 是 thread/warp context 限制；
- `S` 是 shared memory；
- `R` 是 registers；
- `B_arch` 是架构规定的 resident-block 上限。

视频的简化例子中，SM 仍有 thread contexts，却因 shared memory 已满而无法接纳下一个 block。这说明 occupancy 不是只看 threads/block。

### 12.1 Occupancy 的准确含义

通常可把 occupancy 理解为：

$$
\text{occupancy}=\frac{\text{resident active warps}}{\text{architecture max resident warps}}
$$

更高 occupancy 往往提供更多 latency-hiding opportunities，但不是性能目标本身：

- kernel 若 compute-bound，较低 occupancy 也可能已能填满 ALUs；
- 减少 registers 追求 occupancy 可能导致 spill，反而增加 memory traffic；
- 增加 shared memory tiling 会降低 occupancy，却可能大幅减少 HBM traffic；
- divergence 或不合并访存不会被高 occupancy 自动修复。

正确问题是：当前是否有足够的 ready warps 隐藏关键 latency，并使瓶颈资源饱和？

## 13. 为什么一个 block 必须整体装得进一个 SM

设 GPU core 只能容纳 128 thread contexts，但程序声明一个 256-thread block。为什么不能先跑前 128 个，再跑后 128 个？

考虑 block 中的 barrier：

```text
first 128 threads run → reach barrier → retain their contexts and wait
remaining 128 threads cannot become resident
→ nobody can complete barrier
→ deadlock
```

除非硬件做昂贵的 preemption/context swapping，否则无法保持语义。CUDA 面向高吞吐执行，标准模型要求一个 block 的所有 threads 都能同时 resident，因此：

- block size 不能超过 device per-block limit；
- block 的总 registers/shared memory 必须能被一个 SM 容纳；
- 写死极端 block size 会降低跨代架构可移植性。

这里的“同时”指 contexts 同时 live，并不表示所有 threads 每 cycle 都在 ALU 上执行。

## 14. Block 是 locality、synchronization 与 scheduling contract

把本讲最核心的 block 抽象总结成三句话：

| 维度 | 程序员的声明 | 硬件承诺/限制 |
|---|---|---|
| Locality | 这些 threads 共享 `__shared__` data | 放在同一个 SM，可访问同一 scratchpad |
| Synchronization | 它们可能执行 block barrier | 全部 contexts 必须同时 resident |
| Scheduling | 不同 blocks 不依赖执行顺序 | blocks 可任意次序分派到任意 SM |

因此选择 block size 不只是“多少个 threads 一组”的语法问题。它同时影响：

- warp 是否被完整填满；
- 每个 SM 能 resident 几个 blocks/warps；
- shared-memory tile 的形状；
- barrier 的参与范围；
- global-memory 访问是否连续。

## 15. CUDA kernel 的系统化性能分析

### 15.1 先画映射

```text
problem elements
→ grid / block coordinates
→ CUDA threads
→ warps of 32 threads
→ resident blocks on SMs
→ memory addresses touched by adjacent lanes
```

任何一层不清楚，都容易出现“逻辑并行很多，但硬件利用率很低”。

### 15.2 再问五个问题

1. **Parallelism**：grid 是否有足够多 independent blocks 填满所有 SM？
2. **SIMD utilization**：warp 内分支是否一致，block size 是否产生大量不完整 warp？
3. **Memory**：相邻 threads 是否访问连续地址，数据是否值得 tile 到 shared memory？
4. **Latency hiding**：register/shared-memory 使用是否允许足够 resident warps？
5. **Synchronization**：barrier/atomic 是否必要，是否出现 contention 或不合法的跨 block 顺序假设？

### 15.3 最后分清三个“多”

- launch 的 CUDA threads 多：表示逻辑工作量大；
- resident thread contexts 多：表示能保存许多进行中的 threads；
- 每 cycle active lanes 多：才表示 SIMD ALUs 真正在做有用工作。

三个数字相关，但绝不相同。

## 16. 与 AI Infra 的连接

### 16.1 Tensor kernel 的 tiling

GEMM、convolution、attention 都在重复同一个模式：

```text
HBM/global memory 中的大 tensor
→ block 协作加载 tile
→ shared memory / registers 中复用
→ barrier 保证 tile ready
→ warp-level compute
```

本讲三点 1D convolution 是现代 tensor kernel 的最小原型。矩阵乘只是把 `130-element support` 换成 A/B tiles，并在 tile 上进行更多 multiply-accumulate。

### 16.2 Warp divergence

MoE token routing、sampling、ragged batch、稀疏索引等 workload 容易让 warp 内不同 threads 走不同路径。逻辑上 32 个 tokens 并行，不代表一个 warp 能以 32 lanes 的效率推进。

优化常见方向是按 length/expert/branch outcome 分桶，使同一 warp 处理形状和控制流相近的数据。

### 16.3 Host-device transfer 与 serving latency

模型权重通常长期驻留 GPU，就是为了 amortize host-device transfer。推理系统中的 CPU preprocessing、token copy、KV cache 管理若频繁跨 PCIe，会产生：

- 固定 launch/copy latency；
- interconnect bandwidth 压力；
- host/device synchronization bubbles。

Pinned memory、asynchronous copy、CUDA streams 和 batching 都是在这条基本成本模型上继续发展。

### 16.4 Occupancy 与 fused kernels

Fused attention/kernel 可能使用更多 registers 和 shared memory，降低 occupancy，却减少中间 tensor 的 HBM materialization。只看 occupancy 会误判；应同时比较：

$$
\text{extra on-chip resource cost}
\quad\text{vs}\quad
\text{saved HBM bytes and kernel boundaries}
$$

这与 [[Stanford CS149 - Lecture 06 - Performance Optimization II Locality Communication and Contention|Lecture 6]] 的 arithmetic intensity/communication 分析完全衔接。

## 17. 本讲结论

1. GPU 仍由 multicore、SIMD 和 multithreading 构成，只是规模和资源取舍不同。
2. CUDA 是 SPMD programming model：一次 bulk launch 创建 grid，grid 由 blocks 和 CUDA threads 组成。
3. CUDA thread 是逻辑实例；warp 才是 NVIDIA 隐式 SIMD 的执行分组。
4. Thread block 是 shared-memory locality、barrier synchronization 与 placement 的共同边界。
5. Blocks 必须独立于执行顺序；跨 block atomic 可以正确，但 spin-wait 依赖调度顺序可能死锁。
6. GPU 通过大量 resident warps 切换来隐藏 latency，但隐藏不了 bandwidth bottleneck。
7. Registers、shared memory、threads/warps 共同限制 residency；occupancy 高不是最终目标。
8. GPU 优化的核心仍是让有限执行单元做更多有用工作，并减少昂贵的数据移动与等待。

## 自测题

1. CUDA thread、warp、thread block、SM 分别属于哪一抽象层？

    **面试回答：** CUDA thread 是 SPMD 逻辑实例，thread block 是程序员定义的协作、共享内存和同步组；warp 是 NVIDIA 把线程组织起来执行和调度的分组，通常含 32 个线程。SM 是实际硬件计算单元，可驻留并调度多个 blocks 和 warps，不能把逻辑线程数当作物理 ALU 数。

2. 为什么 `11×5` matrix 配 `4×3` block 会启动 72 个 threads？多余 threads 应怎样处理？

    **面试回答：** Grid 需向上取整为 $\lceil11/4\rceil\times\lceil5/3\rceil=3\times2$ 个 blocks，每块 12 threads，因此共启动 72 个线程。只有 55 个对应有效元素，应按坐标检查 `x<11 && y<5` 再访问；若有 block barrier，多余线程也要按一致控制流参与所需同步。

3. 为什么 shared-memory convolution 在读取 tile 后需要 `__syncthreads()`？

    **面试回答：** Tile 由不同线程协作加载，而一个线程的卷积会读取其他线程写入的 shared-memory 元素。`__syncthreads()` 保证所有参与线程到齐并使前序写入可见，防止读取未完成的数据；不能仅凭 warp 或线程编号猜测谁先执行。

4. 128-thread block 在 NVIDIA GPU 上通常包含几个 warps？这是否表示它只能在 4 cycles 内完成？

    **面试回答：** 通常是 $128/32=4$ 个 warps。它只说明执行分组数量，整个 kernel 还包含多条指令、依赖和访存等待；单条 warp 指令的吞吐也取决于架构及执行单元，所以不能推出整个 block 在 4 cycles 内完成。

5. GPU 拥有 16 万 live thread contexts，为什么不等于每 cycle 有 16 万条标量指令同时执行？

    **面试回答：** Live contexts 表示寄存器、PC 等状态驻留，供调度器在等待时切换到其他 ready work。每周期真正能发射和执行的指令受 scheduler、各类流水线和 ALU 数量限制，很多线程此时正在等待数据或依赖；上下文容量的主要作用是隐藏延迟。

6. 为什么 block 内 barrier 合法，而在普通 kernel 内用两个 blocks 自制 barrier 可能死锁？

    **面试回答：** Block 的资源必须能整体驻留在一个 SM，合法的 block barrier 因而能等待该组线程推进。普通 kernel 的不同 blocks 不保证同时驻留，自制跨 block barrier 可能让已驻留 blocks 占住资源等待尚未调度者；全局阶段通常用有序的多次 kernel launch 表达。

7. 一个 kernel occupancy 从 50% 提升到 100%，性能为何可能完全不变甚至下降？

    **面试回答：** 50% occupancy 可能已提供足够 ready warps，让计算单元或带宽饱和，继续增加便没有收益。若为提升 occupancy 缩减寄存器或 tile，还可能引起 spill、减少数据复用并增加同步；应优化实际耗时和瓶颈资源，而非把 occupancy 数字本身最大化。

8. Shared-memory tiling 降低 global-load 指令量时，为什么还不能直接断言 DRAM bytes 按同样比例下降？

    **面试回答：** Global-load 指令不等于 DRAM transaction：原版本的重复 load 可能已命中 L1/L2，连续请求也可能合并。Tiling 减少的是显式加载次数，还引入 shared-memory 访问和同步；要判断实际 HBM 节省，应测量 cache 行为、事务量和传输字节。

9. Host 与 device 使用统一虚拟地址后，为什么仍必须关心 data placement？

    **面试回答：** 统一虚拟地址统一的是指针命名，不保证数据位于同一物理存储或具有相同访问成本。GPU 访问 host DRAM 可能跨 PCIe，managed memory 也可能发生迁移和缺页；仍应规划权重与工作集驻留位置、传输批次及计算重叠。

10. 对一个 Transformer kernel，怎样依次检查 divergence、memory access、tiling 和 residency？

    **面试回答：** 先映射 token、head 和 tile 到线程，检查 warp 内分支与尾部 mask；再看相邻 lane 地址、合并访问和 HBM 字节。随后评估 shared/register tiling 的复用与同步成本，最后检查寄存器、shared memory、spill 和驻留 warps 是否足以隐藏延迟，并用实际耗时验证。


## 参考资料

- [CS149 Fall 2023 Lecture 7 视频](https://www.youtube.com/watch?v=qQTDF0CBoxE)
- [Lecture 7 官方课件](https://gfxcourses.stanford.edu/cs149/fall23content/media/gpucuda/07_gpuarch.pdf)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [[Stanford CS149 - Lecture 03 - Multi-Core Architecture Part II and ISPC|Lecture 3：ISPC、SIMD 与硬件多线程]]
- [[Stanford CS149 - Lecture 06 - Performance Optimization II Locality Communication and Contention|Lecture 6：Locality、Communication 与 Roofline]]
