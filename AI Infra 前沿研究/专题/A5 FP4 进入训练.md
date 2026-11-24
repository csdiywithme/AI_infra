---
type: article
status: 完整
verified: 2026-11-23
threads: ["[[T2 代价搬家账本]]", "[[T3 结论反转条件]]"]
---

# A5 FP4 进入训练：从"推理量化"到"预训练 recipe + QAT + rollout 精度"

> [!abstract] 一句话
> 一年前 FP4 主要是推理权重格式；现在有了 10T token 级的 NVFP4 预训练证据、一个 550B 模型卡声明用 NVFP4 预训练，以及 DeepSeek-V4 用 FP4 存路由专家并在后训练做 FP4 QAT。但所有 recipe 都还**保留部分 BF16**，且不同论文的差距数字不可横比。

## 0. 旧基线

讲座（[视频 48:25](https://www.bilibili.com/video/BV1vmzXBBESY/?t=2905)）："峰值算力不是免费午餐"——端到端收益取决于 Q/DQ 融合、scale 布局、累加精度、收敛。这个框架仍然适用，下面是新证据如何填它。

## 1. 三种 recipe，对照着看

| | NVIDIA NVFP4 预训练 | Quartet II | Full-Stack FP4 |
|---|---|---|---|
| 来源 | [arXiv:2509.25149](https://arxiv.org/abs/2509.25149) v1 2025-09-29，v2 2026-03-04（可读部分未见内容变化） | [arXiv:2601.22813](https://arxiv.org/abs/2601.22813) v2 2026-06-01，ICML | [arXiv:2607.04422](https://arxiv.org/abs/2607.04422) v1 |
| 格式 | E2M1，16 元素块，E4M3 块 scale，FP32 张量 scale | 同 NVFP4 + "4/6" scale 选择 | 同 NVFP4 + 4/6 |
| FP4 范围 | 所有线性层 Fprop/Dgrad/Wgrad；attention、embedding、head 留 BF16 | 前向 RTN；两个反向 GEMM 都 NVFP4【摘要级】 | 线性层 + 优化器状态 + 7 个 attention matmul 中的 4 个 |
| 保留高精度 | 12B 模型前 2 + 后 8 块（约 16% 线性层）BF16；衰减阶段切回 BF16 可收窄差距 | 未读到 | "不保留 BF16 块"，但 PV、dO·Vᵀ 等路径 BF16；另有 BF16 低秩支路 |
| 关键技巧 | 16×16 Hadamard（仅 Wgrad）、权重 2D scale、梯度随机舍入 | MS-EDEN：128 元素块 Hadamard + RTN，仅对 FP8 scale 做随机舍入 | LoRA-SVD 支路、量化 AdamW、NVFP4 Newton–Schulz |
| 规模 | 12B 混合 Mamba-Transformer，10T token | 最大 1.9B / 38B token | 3B / 64B token |
| 硬件 | Blackwell | RTX 5090、B200 真 kernel | 8×A800 **模拟量化** |
| 差距 | 损失相对误差稳定期 <1%，衰减期略高于 1.5%；MMLU-Pro 62.58 vs FP8 62.62，代码 56.67 vs 59.52 | 对 BF16 的损失差距比已有 NVFP4 方案再降 ≥20%【摘要级】 | 训练损失对 BF16 差 1.47% |

【均为作者测量】基线不同（FP8 vs BF16）、模型与 token 数不同、优化器不同——**不要横比这三个数字**。共同信号是：Wgrad 路径最敏感，随机舍入/Hadamard 都集中在反向。

## 2. 落到生产模型

- **Nemotron 3 Ultra**（550B / 55B 激活，约 20T token）：模型卡写明"使用 NVFP4 recipe 预训练"，大多数线性层的权重、激活、梯度用 NVFP4，部分层保持 BF16 或 MXFP8（[HF 模型卡](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4)，2026-06-04）【已发布·厂商自述】。
- **DeepSeek-V4**：路由专家参数 FP4；后训练对专家权重与 indexer QK 路径做 FP4 QAT；indexer attention 以 FP4 运行；作者称当前硬件 FP4×FP8 峰值等于 FP8×FP8，"未来硬件理论上高效 1/3"（[arXiv:2606.19348 §1、§2.3.4](https://arxiv.org/abs/2606.19348)）。注意：这是**存储与带宽**收益为主，当前不是算力收益。
- **推理 KV**：vLLM v0.32.0 支持 NVFP4 KV cache（SM8x/SM12x/SM90）【摘要级】。
- **DeepSeek-V4.1**：**全局 KV 在训练中就以 FP4 存储**（E2M1，每 16 通道一个 E4M3 scale，无第二级 scale，在 RoPE 之后量化），作者称退化"轻微"；滑窗 KV 仍为 FP8；同节又说为了硬件覆盖面选择 OCP MXFP4——两种格式表述并存，报告原文如此（[arXiv:2609.19969 §2.4.4](https://arxiv.org/abs/2609.19969)）。这是 FP4 第一次在本笔记读到的旗舰报告里用于**激活状态（KV）**而不只是权重。
- **RL rollout**：Miles v0.1 支持 NVFP4/MXFP4/MXFP8/FP8 rollout，并称低精度 reward 曲线接近 BF16（无数字）；权重增量同步时 FP4 rollout 每次只有约 0.5% 参数变化（BF16 约 2%）（[Miles v0.1](https://www.lmsys.org/blog/2026-08-18-miles-v0-1/)）。

### 2.1 公开基准：FP4 训练进了 MLPerf，但"谁用了哪种 FP4"不清楚

- MLPerf Training v6.0（2026-06-16）：MLCommons 新闻稿只说"提交者使用了多种不同的 FP4 精度 recipe"，没说谁用了哪种（[新闻稿](https://www.globenewswire.com/news-release/2026/06/16/3312914/0/en/correction-mlcommons-releases-mlperf-training-v6-0-results.html)）。
- NVIDIA 的 v6.0 技术博客**没有提 NVFP4**，描述的是 DeepSeek-V3 的 MXFP8 attention recipe 和 MXFP8 logit 投影；只有提交者补充说明里笼统列了"NVFP4 precision"（[博客](https://developer.nvidia.com/blog/nvidia-blackwell-tops-mlperf-training-6-0-with-industry-leading-scale-and-performance)）。所以**不能**用 MLPerf v6.0 作为"NVFP4 预训练已在公开基准中验证"的证据。
- AMD：Llama 2 70B LoRA 从 v5.0 的 MI300X MXFP8 到 v6.0 的 MI355X MXFP4 快 3.5×（提交者自述，硬件与精度同时变化）；v6.1（2026-11-16）FLUX.1 用 MXFP6/MXFP4 与 MXFP8/MXFP4 两种 recipe，AMD 称是"第一个公开、可复现的 FP6 训练演示"（[AMD 博客](https://www.amd.com/en/blogs/2026/amd-advances-ai-training-across-more-workloads-in-mlperf-training.html)，【厂商声明·非官方分数】；"第一"的范围仅限 AMD 所知的公开 MLPerf 提交）。

**读法**：低精度训练正在进入审计过的基准，但目前只能确认"FP4 参与了"，不能确认"哪种 FP4、覆盖哪些层"。

## 3. 代价搬到哪里

- **精度敏感层仍需高精度** ⇒ 混合精度数据流图更复杂（同一模型里 FP4/FP8/MXFP8/BF16/FP32 并存）。
- **Q/DQ 开销**：Quartet II 指出小矩阵上量化开销在 B200 上占主导【摘要级】；B200 上从约 3B 参数开始才有加速。
- **格式碎片化**：NVFP4 vs MXFP4（NVIDIA 报告 8B/1T token 时 MXFP4 损失差约 2.5% vs NVFP4 约 1.5%，需多约 36% token 才追平）；SGLang v0.5.18 提供加载时 NVFP4→MXFP4 重量化，以便在 AMD 上跑。
- **训推不一致**：rollout 用 FP4、训练用 BF16，logprob 不再一致，需要 [[A6 数值一致性]] 的修正手段。

## 4. 何时反转

| 条件 | 结论 |
|---|---|
| 非 Blackwell / 无原生 FP4 tensor core | 只剩存储收益，算力可能更慢（Full-Stack 仅模拟） |
| 小模型、小矩阵 | 量化开销吃掉收益 |
| 代码类下游能力优先 | NVIDIA 报告代码分差最大（56.67 vs 59.52），需要衰减期切回高精度 |
| MoE 大模型、长 horizon | NVIDIA 论文自己列为未测试；Nemotron 3 Ultra 是厂商自述，无对照实验公开 |

## 5. 未解决

- [x] TE NVFP4 页面第二轮已打开（[TE 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)）：**页面上没有任何加速数字**，也没有 B300；只写明训练支持 SM 10.0 / 10.3、权重默认 2D scale、激活与梯度 1D、随机舍入只用于梯度、RHT 只支持 BF16 输入。旧笔记的 2.03× / 3.55× 在现有官方页面上**找不到出处**，本笔记不采用。
- [ ] Nemotron 3 Ultra 的 NVFP4 预训练是否有同规模 BF16/FP8 对照？
- [ ] FP4 通信（梯度 all-reduce、EP dispatch）尚无一手大规模结果。

往哪走：[[A6 数值一致性]] · [[A4 MoE 通信融合]] · [[T3 结论反转条件]] · [[00 入口]]
