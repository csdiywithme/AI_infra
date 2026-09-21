---
type: topic-note
status: evergreen
course: HPCGame 2026 赛前讲座
area: gpu-kernels
topics:
  - "[[GPU Kernel]]"
  - "[[Triton]]"
  - "[[GPU Architecture]]"
---

# HPCGame 2026：Kernel、DSL 与硬件

> [!nav] 返回与扩散
> [[HPCGame 2026 - 00 MLSys 知识地图|知识地图]] · [[HPCGame 2026 - 02 稀疏模型与数值格式|模型/精度]] · [[HPCGame 2026 - 04 通信与内存层级|通信/层级]] · [[HPCGame 2026 - 08 集群效率与可靠性|集群与冷启动]] · [[HPCGame 2026 - 09 多模态与新工作负载|新工作负载]] · [[HPCGame 2026 - 06 前沿追踪|前沿追踪]]

> [!abstract] 核心问题
> Kernel 优化不是把同一份代码编译到新 GPU，而是针对新的 **compute/memory/synchronization ratio** 重新映射数据。DSL 和 AI agent 改变开发方式，但没有取消硬件知识与正确性验证。

> [!nav] 按困惑进入
> 局部 kernel 与正确性：第 1～8 节；整图编译、启动与动态 shape：[[#10. 整图编译：把局部收益变成可重复的运行时行为|第 10 节]]；跨平台功能和性能：[[#11. 非 NVIDIA 平台：分别验证三种可移植性|第 11 节]]。
>
> 第 1～9 节的具体版本/规格/benchmark 是原 2026-08/09 案例，未因本次增补而全部重新核验。当前 cuDNN 边界与 Φ-Bench 口径分别见 [[MLSys - 证据台账#E-20260915-05]]、[[MLSys - 证据台账#E-20260915-06]]。

## 1. 为什么每代 GPU 都重写热点 kernel

低精度 tensor-core compute 的增长通常快于数据移动，因此瓶颈不断推向 register、SMEM/TMEM、HBM 与互连。

| 代际 | 关键变化 | 对 kernel 的含义 |
| --- | --- | --- |
| Hopper | TMA、cluster/CGA、distributed SMEM、WGMMA | producer/consumer warp、异步搬运与 cluster 协作 |
| Blackwell | 更低精度、TMEM、新 tensor-core path | scale/layout 与 accumulator 管理更复杂 |
| Rubin | FP8/FP4、更多 SMEM/TMEM、B collector、activation sparsity、平台级 scale-up | tile、buffer reuse、mixed precision 与数据移动再次重排 |

同一个 attention/GEMM 在不同代际的最佳 tile、pipeline stage、warp role 和 synchronization 都可能不同。FlashAttention 连续重写，证明可迁移的是算法思想，不是固定实现。

## 2. 先用性能账本定位，不先写代码

对任意 kernel，先记录：

```text
shape / dtype / layout
FLOPs 与理论 tensor-core ceiling
HBM bytes 与 arithmetic intensity
register / SMEM / TMEM footprint
occupancy 与 active CTAs
launch 数与 graph breaks
同步、barrier、collective
端到端调用占比
```

只有 kernel 在 critical path 且接近可优化下界时，局部重写才值得。MoE 新论文中 fused Triton 隔离快 5.6–9×、端到端仅 0.999×，就是没有先看 launch-bound ceiling 的典型反例。

## 3. DSL 的比较轴

| 维度 | 需要回答 |
| --- | --- |
| 表达力 | 能否表达 async pipeline、tensor layout、cluster、distributed op？ |
| 性能上限 | 能否控制达到目标硬件上限所需细节？ |
| 反馈周期 | compile、autotune、debug、profile 多久？ |
| 可移植性 | 复用的是源码、抽象还是只有算法思想？ |
| 生态 | 能否进入 PyTorch/serving graph，处理动态 shape 与 fallback？ |
| 可验证性 | 数值、同步、竞态和跨 shape 如何证明？ |

### 几个层次

- CUTLASS/CuTe：贴近 tensor layout 与硬件 primitive，控制强、专家门槛高；
- Triton：以 block program 与编译器 lowering 提供较快开发/自动调优；
- TileLang/CUDA Tile：进一步把 tile abstraction 变成可组合工具链；
- CuTe DSL：说明 Python/DSL 并不必然牺牲顶级性能，但仍要求专家理解 layout/pipeline。

CUDA 13.2/13.3 已把 CUDA Tile 扩到官方 Python/C++ 工具链；FlashAttention-4 用 CuTe DSL 达到 Blackwell 上限附近。更高层 abstraction 缩短表达路径，不会自动找到正确映射。

## 4. Rubin 的软件 enablement：什么已经有，什么还没有

### Triton 3.8.0：稳定 release 中的初始支持

Triton `3.8.0` 正式发布并加入：

- 初始 Rubin SM107 MMA/multicast barrier/Gluon 支持；
- 四 lane packed FP8/FP4 arithmetic；
- generic multi-CTA 扩展到 layout conversion、reduction、gather/scatter、TMA 与 multicast；
- FpSan 符号浮点等价检查；
- 更广 ConSan；
- 面向 symmetric memory/multi-node 的实验性 GSan data-race detector。

稳定 release 不代表所有子功能稳定：Rubin 是 initial support，GSan 明确为 experimental。

### CUTLASS 4.8.0dev：可读可编译的预览 building blocks

CUTLASS `4.8.0dev` 首次提供 Rubin SM107 的 dense/block-scaled FP8/FP4 GEMM、B collector reuse、更大 SMEM/TMEM、mixed precision，以及 CuTe DSL/C++ building blocks。

但 `cute_ext` compiler pipeline 与 Operator API Rubin 支持均是 preview/preliminary；执行 Rubin kernel 还要求未来 CUDA 13.4 GA 随附的 R615 driver，当前 Developer Preview 的 R610 不足。正确表述是“可以提前研究 kernel 形态”，不是“Rubin 已可生产部署”。

## 5. 2:4 sparsity 要先问稀疏对象

对通用静态 weight 2:4，训练/压缩约束强，很多 workload 难以吃满宣传峰值，这一批评仍成立。

Rubin 把 2:4 用于 attention 的中间 activation，压缩 softmax 后数据并减少第二个 attention GEMM/数据移动。这是不同的来源和生命周期。

> [!tip] 正确问题
> 稀疏对象是 weight、activation 还是 routing result？零如何产生？sorting/pruning/metadata 的成本是多少？上下游能否保持压缩格式？

## 6. AI kernel generation 的真实进展

研究已经从“一次生成 Triton”推进到 profile feedback、执行验证、搜索/RL 与 failure-directed repair：CUDA Agent、DRTriton、A-TREX 等都体现这一变化。

Benchmark 也在扩张：

- SOL-ExecBench 用硬件上限而不是普通 library baseline 评估；
- DataKernelBench 把任务扩展到 data-movement-heavy 数据库查询，作者报告 H100 full-query CUDA 在 full pass rate 下相对 TorchPlan 2.11×；
- FABRICA 用 49 个 CUDA→Cerebras CSL 任务测试跨架构重映射，作者报告核心集成功率 6/28→26/28、WSE-3 实机 3.47× 几何平均加速；
- UCCL CommBench 显示通信 kernel 仍受动态 world size、并发、网络语义与故障制约。

当前边界可以写成：

```text
单 GPU + shape 明确 + 可执行验证：已很有竞争力
跨 shape / 硬件 / 通信语义 + 长期维护：仍是开放问题
```

论文数字均需视作作者 benchmark，不能外推到任意 shape、模型或系统。

## 7. 正确性不只是输出 `allclose`

| 风险 | 需要的证据 |
| --- | --- |
| 数值重排 | 多 dtype、极端值、误差传播、symbolic equivalence |
| 并发/同步 | race、barrier、async copy、memory ordering |
| shape generality | 边界 shape、非对齐、动态 batch/context |
| 分布式 | world size、tail、timeout、partial failure |
| 性能稳定 | compile/autotune cost、p50/p99、fallback |
| 可维护性 | 新 GPU/driver/compiler 下的回归测试 |

Triton FpSan/ConSan/GSan 的出现很关键：kernel DSL 开始把数值语义、同步与分布式 memory correctness 变成一等工具，而不只追求生成速度。

### 验证链还要穿过编译产物与安装包

2026-09-21 增量核验：cuTile Rust 的迁移报告提供“参考实现 → 共享 IR 结构 → 数值/性能检查”的实例。中间表示能够增加检查约束，但不代替极值、同步与跨 shape 测试，也不证明 AI 自动找到了更好的算法。[[MLSys - 证据台账#E-20260921-05]] · [官方报告](https://developer.nvidia.com/blog/translating-cuda-tile-operations-from-python-to-rust-using-agentic-ai/)

另一个边界是发布制品：FA4 main 的修复要进入 wheel，再由 runtime 实际调用才形成可用组合。TorchTitan 对 b31 的验证还显式记录实际 FA4 调用，避免“配置已激活，实际走了其他 engine”。这让 source、wheel、依赖环境、dispatch 和边界输入成为连续验收链。[[MLSys - 证据台账#E-20260921-08]] · [上游验证 #4737](https://github.com/pytorch/torchtitan/pull/4737)

## 8. 审核“快 2×”的清单

1. baseline 是否是合理、同版本、同 precision 的强实现？
2. 输入 shape/dtype/layout 是否覆盖真实分布？
3. 是否包含 compile、autotune、Q/DQ、layout conversion？
4. correctness 是否跨 shape 与随机种子验证？
5. GPU clocks、warmup、graph、stream 是否一致？
6. 局部 kernel 在端到端占比多少？
7. 换 GPU、batch、context 或并发后是否反转？
8. failure、长尾和多租户下是否仍成立？

## 9. 一手资料

- [Rubin architecture](https://developer.nvidia.com/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/)
- [Triton 3.8.0](https://github.com/triton-lang/triton/releases/tag/v3.8.0)
- [CUTLASS 4.8.0dev](https://github.com/NVIDIA/cutlass/releases/tag/v4.8.0dev)
- [CUDA Tile Python](https://developer.nvidia.com/blog/cuda-13-2-introduces-enhanced-cuda-tile-support-and-new-python-features/) · [CUDA Tile C++](https://developer.nvidia.com/blog/develop-high-performance-gpu-kernels-in-c-with-nvidia-cuda-tile/)
- [FlashAttention-4](https://arxiv.org/abs/2603.05451)
- [CUDA Agent](https://arxiv.org/abs/2602.24286) · [SOL-ExecBench](https://arxiv.org/abs/2603.19173) · [CommBench](https://github.com/uccl-project/CommBench)
- [DataKernelBench](https://arxiv.org/abs/2608.25061) · [FABRICA](https://arxiv.org/abs/2608.25124)

## 10. 整图编译：把局部收益变成可重复的运行时行为

> [!abstract] 2026-09-15 补充的观察维度
> Kernel 决定一段计算怎样执行；graph compiler 决定哪些段可以融合、哪些 buffer 可以复用、哪些输入能复用编译结果。两层共同决定启动时间、稳态速度和尾延迟。

```text
Python / model graph
  → capture 与 guards
  → 分区、融合、内存规划
  → kernel lowering 与 autotune
  → 编译产物缓存
  → runtime dispatch / replay / fallback
```

这里有三种不同的“没能一直走快路径”：

| 现象 | 发生什么 | 应观察什么 |
| --- | --- | --- |
| graph break | 一段程序无法放入同一捕获图，回到 Python/eager 或分成多个图 | 原因、跨边界 fusion/launch 成本、未编译时间占比 |
| guard failure / recompilation | shape、stride、Python 值等假设失效，需要另一个编译版本 | 触发输入、编译次数、cache 数量、请求尾延迟 |
| cache miss / cold start | 进程或环境未能复用编译、autotune、capture 结果 | key、版本/硬件兼容、加载耗时、暖机覆盖率 |

PyTorch 官方文档说明 `dynamic=None` 会在 shape 变化触发重编译后尝试动态形状；`dynamic=True` 尽可能提前动态化，但不是对任意控制流、layout、Python 值都不重编译的保证。`TORCH_LOGS=recompiles` 和 `tlparse` 可追溯原因。[重编译文档](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.recompilation.html)

动态 shape 与分桶各有代价：更通用的图减少特化次数；固定 bucket 可能提高 kernel/graph 复用，却增加 padding 和 bucket 管理。刻意在小而不稳定的子图前切开，有时能保住大图缓存；不能把 graph break 数量本身当作最终目标。[PyTorch troubleshooting](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html)

**编译账本**需要同时记录首次请求、同进程稳态、跨进程暖启动、输入分布改变后的请求；还要核对 fusion 后的临时内存与 allocator 峰值。内存下降可以换来更大 batch/KV pool，这种收益会传递到 [[HPCGame 2026 - 03 推理状态与调度|admission 与调度]]。

对 diffusion，重复 denoising block 给编译产物较多复用机会，但分辨率、帧数、batch、LoRA rank 和控制分支会改变形状/guards。PyTorch 官方实践文档包含编译缓存、shape 变化和 LoRA 热切换的具体条件；这是实现与作者实验依据，不是任意流水线的速度承诺。[torch.compile 与 Diffusers](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)

持续追问：**一次更强的编译优化，需要多少次调用才能摊薄准备成本？** 在短命 worker、突发流量和多模型切换中，应把答案连到 [[HPCGame 2026 - 08 集群效率与可靠性|扩缩与冷启动]]；在训练中，应连到 [[HPCGame 2026 - 07 数据与后训练闭环|总实验周期]]。

## 11. 非 NVIDIA 平台：分别验证三种可移植性

| 层次 | 能复用什么 | 还需证明什么 |
| --- | --- | --- |
| 语义/功能 | checkpoint、算子定义、请求 API | 模型/精度/状态和错误处理是否一致 |
| 实现/工具链 | DSL 源码、框架 integration、benchmark | 后端覆盖、layout、编译、collective 与调试支持 |
| 性能/运维 | 目标质量和 SLO 的实验设计 | 每种硬件的并行布局、尾延迟、成本和恢复表现 |

AMD 与 TPU 提供两个不同的观察窗口：

- **AMD/ROCm：已开源实现。** AITER 是 AMD 的算子库，README 描述 Triton、FlyDSL、HIP/Opus 等实现路径与 ROCm 依赖；JAX-AITER 另通过 XLA FFI 接入 AITER 的 CK/ASM 等手调 kernel。多条专用路径并存，说明统一调用接口后面仍需要架构调优。[AITER](https://github.com/ROCm/aiter) · [JAX-AITER](https://github.com/ROCm/jax-aiter)
- **TPU/JAX：已开源引擎。** SGL-JAX 独立实现 JAX/TPU 的 continuous batching、Radix KV 管理、TP 和模型执行；可借它观察同类 serving 思想如何映射到 TPU。README 的功能清单只证明列出的支持，不代表与 SGLang GPU 后端在所有模型/量化/PD 组合上等价。[SGL-JAX](https://github.com/sgl-project/sglang-jax)
- **Pallas：活跃开发中的 kernel 工具。** 官方文档提供 GPU/TPU 入口，共享 Ref、BlockSpec、pipeline 等思想，同时明确两者具有硬件专用 API。它是 XLA 自动优化不足时的手动控制层，不能据此推断同一份 kernel 在各平台自动最优。[Pallas 官方文档](https://docs.jax.dev/en/latest/pallas/index.html)

跨平台比较先固定模型语义、质量、输入分布和 SLO；再分别寻找合理的 backend、精度、并行与 cache 配置。单 kernel 的峰值、硬件规格和厂商模型 benchmark 是三类证据，不能互相替代。

**当前判断**：可移植的算法与调度思想越来越多，达到相近的端到端效率仍依赖硬件专门化。值得跟踪的是新增功能是否补齐实际部署缺口、同等质量成本是否改善，以及迁移后调试/维护工作量是否下降。

本节链接当前官方文档和仓库，用于建立比较维度；未经固定 release/commit 与事件日期核查，不把 `main` 文档里的能力写成当周新发布。增量证据进入 [[HPCGame 2026 - 06 前沿追踪]]。
