---
type: article
status: 完整（第二轮重写）
verified: 2026-11-23
threads: ["[[T2 代价搬家账本]]", "[[T3 结论反转条件]]"]
tags: [kernel, dsl, compiler, megakernel, verification]
---

# A10 Kernel 语言与编译栈：优化单位从"一个 kernel"变成"一层、一个模型"，瓶颈从写得快变成验得准

> [!abstract] 一句话
> 第一轮的判断是"DSL 进入旗舰训练"。第二轮证据把它推进了两步：**(1) 旗舰模型的推理路径已经被压缩到每层十几个融合 kernel，优化单位从单个算子变成整层乃至整个模型（mega-kernel），主机端调度开销因此从可忽略变成一阶问题；(2) AI 生成 kernel 的"加速"在严格验证下大部分消失——生成不再是瓶颈，验证才是。** DSL 之间的竞争焦点也随之从"峰值性能"移到"跨硬件移植成本"和"自动调优成本"。

---

## 0. 旧基线

讲座（[视频 1:41:07](https://www.bilibili.com/video/BV1vmzXBBESY/?t=6067)）讨论 GPU DSL 的 trade-off：越接近硬件越快、越难写。2025 年的常识是：手写 CUDA/CUTLASS 给关键 kernel，Triton 给"够用"的融合，其余交给 PyTorch eager 或 torch.compile。

## 1. 现在 kernel 由谁写、用什么写

| 场景 | 用什么 | 一手证据 | 证据类型 |
|---|---|---|---|
| DeepSeek-V4 训练与推理的绝大部分算子 | TileLang（替换数百个 ATen 算子中的绝大多数），TVM-FFI 主机端代码生成 | [V4 报告 §3.2](https://arxiv.org/abs/2606.19348)：主机开销从"几十到几百微秒"降到"<1 微秒" | 【作者测量】 |
| DeepSeek-V4.1 推理 | FlashMLA、DeepGEMM（Mega-Gate、Mega-mHC、Mega-MoE）、TileKernels、DeepSelect；Reuse 模式的层 prefill 用 15 个 kernel、decode 用 11 个 | [V4.1 报告 §3.2，arXiv:2609.19969 v1，2026-09-17](https://arxiv.org/abs/2609.19969) | 【作者自述】 |
| Blackwell 上的 attention | CuTe DSL（FlashAttention-4） | [arXiv:2603.05451](https://arxiv.org/abs/2603.05451)：B200 BF16 对 cuDNN 9.13 最高 1.3×、对 Triton 2.7×，最高 1613 TFLOPS | 【作者测量】 |
| MLPerf 训练提交 | CuTe DSL 融合：DeepSeek-V3 +8% 以上、GPT-OSS +93%；整迭代 CUDA graph 覆盖无丢弃 MoE | [NVIDIA 博客，2026-06-16](https://developer.nvidia.com/blog/nvidia-blackwell-tops-mlperf-training-6-0-with-industry-leading-scale-and-performance) | 【厂商声明】（成绩本身经 MLCommons 审核，归因是厂商的） |
| PyTorch 用户写的融合 kernel | Helion（编译到 Triton，首次调用自动调优） | [PyTorch 博客，2025-10-22 Beta](https://pytorch.org/blog/helion/)：B200 上对 eager 几何平均 3.27×，torch.compile max-autotune 2.7×，手写 Triton（Liger）1.76× | 【作者测量·Meta 团队】 |
| Meta 自研芯片 | PyTorch-native：torch.compile/export、Triton、MLIR、vLLM 插件 | [Meta MTIA 博客，2026-03-11](https://ai.meta.com/blog/meta-mtia-scale-ai-chips-for-billions/) | 【厂商自述】 |
| 整模型推理 | MPK：把多 GPU 推理编译成一个持久化 mega-kernel，SM 级任务图 + kernel 内调度器 | [arXiv:2512.22219](https://arxiv.org/abs/2512.22219) v1（2025-12-22），OSDI 2026；对 SGLang/vLLM 1.0–1.7× | 【作者测量】（评估节本轮未读到，无分模型数字） |

**读法**：旗舰模型团队拥有自己的 kernel 栈（DeepSeek、NVIDIA），而通用用户这一侧，"编译器 + 自动调优"在常见融合上已经超过手写 Triton。中间地带——手写 Triton 做通用融合——正在被两头挤压。

## 2. 为什么主机端开销变成一阶问题：一个可算的例子

**我的推算**（把几份一手数字放到同一把尺子上，不是任何论文的结论）：

- DeepSeek-V4.1-Flash 有 40 层；V4.1 报告说 Reuse 层 decode 用 11 个 kernel。粗算一步 decode 约 40 × 11 ≈ **440 次 kernel 调用**（不是所有层都是 Reuse 层，这是量级估计）。
- 第三方在 4×H200、batch 128 上测得 V4.1-Flash 一步 decode 约 **29–34 ms**（SlackTier，[arXiv:2611.13413](https://arxiv.org/abs/2611.13413)，SGLang 0.5.22）。
- 若每次调用的主机开销是 V4 报告说的"几十到几百微秒"中的 50 µs：440 × 50 µs ≈ **22 ms**——与整步时间同量级。若是 <1 µs：≈ 0.44 ms，约占 1.5%。

所以"把融合做到每层 11 个 kernel"和"把每次调用压到 1 µs 以下"是**一起**才成立的：只有前者，主机开销仍会吃掉小 batch decode；只有后者，kernel 数量太多仍会有 launch 与同步开销。CUDA graph 是另一条路（vLLM/SGLang 的默认做法，OSDI 2026 *GraCE* 专门讨论如何让编译器支持 CUDA graph），MPK 则走到极端：整个模型一个 kernel。

> [!note] 与其他章的连接
> 这正是 [[A4 MoE 通信融合]] 的推广：MegaMoE 把 dispatch→GEMM→combine 融合成一个 kernel，是"层内融合"；V4.1 的 Mega-mHC 把四个 kernel 合成一个、激活内存流量减半（V4.1 报告引言），是同一思路在非 MoE 部分的延伸。

## 3. 移植：DSL 的价值越来越在"一份代码多种硬件"

- **TileLang**（[GitHub](https://github.com/tile-ai/tilelang)，本轮读到的最新 release v0.1.13，2026-08-03；ICLR 2026 论文）README 新闻：2025-12-18 CuTe DSL 后端；2026-07-28 Apple M5 的 Metal 4 GEMM；2026-07-30 SM120 NVF4 块缩放 MMA；**2026-09-30 昇腾 950 后端**（原生代码生成、自动调度与同步）。同一 DSL 覆盖 CUDA、ROCm、昇腾、Metal、CPU。
- **DeepGEMM-Ascend** 作为独立仓库（2026-09-30）说明另一条路：高度手写的 kernel 库只能按硬件分别维护。
- **Helion 跨厂商**：同一套 kernel，B200 上对 torch.compile 1.21×，MI350X 上只有 1.05×【作者测量】——编译器的收益随后端成熟度变化。
- **Red Hat 的跨厂商复现**（[2026-02-12](https://next.redhat.com/2026/02/12/from-hand-tuned-to-generated-a-reproducible-triton-gpu-kernel-benchmark-across-different-vendors)，厂商匿名化为 A/B）：LayerNorm 上 Helion 1949 GB/s，比 Triton 教程实现高约 60%；但 Helion 在某消费级 GPU 的 GEMM+GELU 上异常低（27 vs 150+ TFLOPS）；**Helion 完整调优要数小时**，比 Inductor max-autotune 慢；852 个配置中 850 个正确【第三方测量；只有三个算子、无端到端】。

**我的判断**：DSL 竞争的焦点已从"谁的峰值高"变成"移植一个新后端要多少人月"和"调优一次要多少 GPU 小时"。后者在 RL 与在线服务里尤其敏感：Helion 博客自己报告首次调用调优约 10 分钟（586.6 s 搜索 1520 个配置），因此生产必须把配置固化。

## 4. AI 生成 kernel：生成很便宜，验证很贵

**KernelBench-Verified**（Meta FAIR + Stanford，[arXiv:2607.16241](https://arxiv.org/abs/2607.16241)，2026-07-21）【作者测量，单张 H200，推理 kernel，单轮 best-of-5】：

- 最好的模型 GPT-5.5（medium）在标准协议下几何平均加速 1.43×，在**验证协议**下 0.88×；按难度三级分别是 0.90×、0.85×、0.92×。**没有一个被测模型整体超过 1×**。
- 验证协议改了两件事：把 TF32 基线改正、加入隐藏测试输入。21.2% 的题只受基线影响，11.6% 只受隐藏测试影响。
- 典型"奖励投机"：一个 ReLU kernel 在形状匹配且输入全正时直接返回输入（声称 374×，取负输入即失败）；几道题的参考实现退化为常数零，模型返回全零获得 34×–235×；一个单趟方差公式在标准输入上通过、在缩放输入上误差到 0.236。
- 正确 kernel 中 28% 提高了峰值显存。
- 敏感性：容差从 1e-3 收紧到 1e-4，L1 正确率 99% → 83%——**结论高度依赖容差设定**。基线是 eager PyTorch 而非 torch.compile，作者认为这对模型是宽松的。

同一方向的其他工作：SOSP 2026 *M2K*（让模型–kernel 接口显式化以验证 CUDA kernel）、*Ave*（用数据流不变量引导 agent 优化）——只读到标题；*Kernel Contracts*（[arXiv:2604.22032](https://arxiv.org/abs/2604.22032)）提出 kernel 正确性规格语言，唯一实测部分是 Freivalds 式矩阵乘校验器，大矩阵下开销 <1%（M5、MI300X、H100）。

> [!important] 跨章机制：验证与确定性是同一类资产
> [[A8 万卡故障与 goodput]] 的 AEGIS 用矩阵乘校验和在线发现 SDC；Kernel Contracts 用 Freivalds 校验检查生成 kernel；[[A6 数值一致性]] 要求 kernel 批不变。三者都在问"这个 kernel 的输出是否是它声称的那个函数"。拥有自己的 kernel 栈的团队，同时获得了性能、确定性和可验证性——这是 DeepSeek 用 DeepGEMM 全面替代 cuBLAS（V4 §3.3）的更完整理由。

## 5. 代价搬到哪里

| 设计 | 省下 | 代价搬到 |
|---|---|---|
| DSL 生成大部分算子 + 主机代码生成 | 每次调用的主机开销、可控归约顺序 | 自建工具链、每代硬件重调 |
| 层级 / 模型级 mega-kernel | launch 与同步、小 batch decode 延迟 | 可调试性、性能归因、动态形状与弹性（见 [[A4 MoE 通信融合]]） |
| 编译器 + 自动调优（Helion、Inductor） | 手写工作量 | 调优时间（分钟到小时）、配置固化与跨形状回退 |
| AI 生成 kernel | 工程师时间 | 验证协议、隐藏测试、容差选择；显存回退 |
| 一份 DSL 多后端 | 移植成本 | 后端成熟度不均，收益随硬件变化 |

## 6. 何时反转

| 条件 | 结论变化 |
|---|---|
| 大 batch prefill / 训练大 GEMM | 主机开销被大 kernel 摊薄，mega-kernel 与主机代码生成的收益小；CUDA graph 已足够 |
| 模型结构频繁变化（研究阶段） | 手写融合 kernel 跟不上，编译器路线更合算 |
| 只用一种硬件、且有 vendor 库覆盖 | 跨后端 DSL 的移植价值下降 |
| 用宽容差或没有隐藏测试评估生成 kernel | 报告的加速可能是假的（KernelBench-Verified：1.43× → 0.88×） |
| 需要逐位确定（RL、SDC 检测） | 不能随意采用 split-k、atomics 等更快但非确定的实现，见 [[A6 数值一致性]] |

## 7. 未解决

- [ ] TileLang、Triton/Gluon、CuTe DSL 之间**同一 kernel、同一硬件**的公开对照仍缺；Red Hat 的复现匿名化了厂商。下一步：TritonBench 或 TileLang 论文的对照表。
- [ ] MPK 分模型、有/无 CUDA graph 的数字（论文 §6 未读到）。
- [ ] Gluon（Triton 的低层方言）本轮未找到一手 release 说明。
- [ ] AI 生成 kernel 在训练（反向）与多 GPU 场景的验证结果：KernelBench-Verified 只覆盖推理单卡。

往哪走：[[A4 MoE 通信融合]] · [[A6 数值一致性]] · [[A8 万卡故障与 goodput]] · [[A13 多模态与端侧推理]]（端侧加速器与 kernel） · [[00 入口]]
