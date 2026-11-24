---
type: article
status: 完整
verified: 2026-11-23
threads: ["[[T2 代价搬家账本]]"]
---

# A4 MoE 通信融合：从"重叠 All-to-All"到"把通信写进同一个 kernel"

> [!abstract] 一句话
> 2025 年的做法是让 DeepEP 的 dispatch/combine 与 GEMM 在时间上重叠；2026 年的旗舰做法是把 dispatch → Linear1 → SwiGLU → Linear2 → combine **融合成一个 mega-kernel**，按"波次"流水，在 NVLink 对称内存上边算边传。收益最大的恰恰是 RL rollout 这类小 batch、长尾场景。

## 0. 旧基线

讲座（[视频 36:05](https://www.bilibili.com/video/BV1vmzXBBESY/?t=2165)）：MoE 把稠密计算换成路由 + 不规则通信 + grouped GEMM + 负载均衡；DeepEP/DeepGEMM 的意义在于把 EP 通信与专家 GEMM 做成可重叠的专用库。

## 1. 机制：wave 化的融合 EP

DeepSeek-V4 §3.1（[arXiv:2606.19348](https://arxiv.org/abs/2606.19348)）：
- 把 Dispatch、Linear-1、SwiGLU+FP8 转换、Linear-2、Combine 融合，按专家"波次"调度——一部分专家的 token 到了就开始算，不等全部 dispatch 完成。
- Dispatch 用 FP8，Combine 用 BF16；路由专家权重 FP4。
- 【作者测量】相对非融合基线：一般推理 1.50–1.73×，延迟敏感场景（如 RL rollout）最高 1.96×；理论分析（Flash 配置）1.92× vs Comet 1.42×。在 NVIDIA GPU 与华为昇腾 NPU 上验证。
- 为什么 RL rollout 收益最大：rollout 尾部只剩少量长序列，batch 很小，传统"先全量 dispatch 再算"的同步屏障占比最高。

开源实现：DeepGEMM 的 **MegaMoE**（README 新闻 2026-04-16 首次出现）——单 kernel 融合 EP dispatch、FP8×FP4 或 FP8×FP8 两层线性、SwiGLU、EP combine，在 NVLink 通信与 tensor core 计算间重叠；需要多进程对称内存、PyTorch ≥2.9（[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)，README 新闻截至 2026-09-30；仓库无 release tag，本轮未取到最新提交日期）。

## 2. 下游：引擎如何吸收

| 引擎/版本 | 日期 | 内容 | 标签 |
|---|---|---|---|
| SGLang v0.5.20 | 2026-09-18 | FlashInfer MegaMoE 后端 `--moe-runner-backend flashinfer_megamoe` | 【已发布】 |
| SGLang v0.5.21 | 2026-10-02 | MegaMoE 接入 Qwen MoE（MXFP4/NVFP4 专家）；DeepEP v2 带批不变推理；elastic EP 扩容后 CUDA graph 重捕获 | 【已发布】 |
| vLLM v0.32.0 | 2026-10-18 | Model Runner V2 支持 Elastic EP【摘要级】 | 【已发布】 |
| FastAFD | 2026-07-06 | FFN 侧 MegaMoE 风格融合：8 节点 step 延迟比分离的 DeepEP+DeepGEMM 降 44%（Qwen）/ 42%（MiniMax）【摘要级】 | 【原型】 |
| DeepGEMM-Ascend | 2026-09-30 | 单独仓库 | 【已发布】（未读其内容） |
| DeepEP V2 | README 无日期 | "Expert Parallelism 的完整重构"：统一的 ElasticBuffer API，**用 NCCL Gin 后端替代 NVSHMEM**；对 V1 峰值最高 1.3×、SM 占用最多省 4×；实验特性"0 SM Engram（RDMA）""0 SM PP（RDMA）""0 SM CP（Copy Engine）"；"0 SM RDMA 低延迟 EP 不再支持"；要求 NCCL ≥2.30.4、PyTorch ≥2.10 | 【已发布】作者测量 |
| DeepSeek-V4.1 推理 | 2026-09-17 报告 | 继续用 DeepGEMM Mega-MoE，并加入 Mega-Gate、Mega-mHC（见 [[A10 Kernel 语言与编译栈]]） | 【作者自述】 |
| vLLM v0.34.0 | 2026-11 | DP+EP 缩容经 `POST /v1/fault_tolerance/apply`（#46370） | 【已发布】 |

来源：[SGLang releases](https://github.com/sgl-project/sglang/releases)、[vLLM releases](https://github.com/vllm-project/vllm/releases)、[FastAFD](https://haoailab.com/blogs/fastafd/)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[DeepEP](https://github.com/deepseek-ai/DeepEP)。

**DeepEP V2 的一个信号**：通信库从 NVSHMEM 换到 NCCL 的新后端，并把"不占 SM"的通信从 EP 推广到 Engram、PP、CP。这说明"把通信从计算 SM 上拿走"已经是一个通用设计目标，而不只是 MoE 的技巧。可移植性则是另一个方向：OSDI 2026 *UEP: Portable Expert-Parallel Communication*、*UCCL-Tran*（可扩展的 GPU 网络传输层）只读到标题（[OSDI '26 目录](https://www.usenix.org/sites/default/files/osdi26-contents.pdf)）。

**训练侧的公开基准**：MLPerf Training v6.0（2026-06-16）首次加入 DeepSeek-V3 671B MoE 预训练，Google 称其为该套件中"第一个大规模 MoE 负载"（提交者自述）；扩展效率见 [[A12 训练并行与数据管线#1. 规模扩展本身：公开可审计的数字]]。

## 3. 代价搬到哪里

- **硬件绑定更深**：融合 kernel 依赖对称内存与 NVLink 域内的 load/store 语义；跨节点 RDMA 不能直接套用。
- **可调试性下降**：一个 kernel 里同时有通信与计算，性能问题难以拆账；数值问题也更难定位（DeepSeek-V4 §3.3 为此用 DeepGEMM 全面替代 cuBLAS、放弃 split-k，以保证批不变）。
- **弹性**：EP 规模变化（Elastic EP）需要重建对称缓冲区与 graph。
- **可诊断性**：SDCHunter（OSDI 2026）指出，融合的 MoE EP kernel 会**掩盖通信签名**，使逐层比对定位 SDC 更难（见 [[A8 万卡故障与 goodput#2.2 SDC：发现本身成了主要成本]]）；InplaceMoE 在其首次检出 SDC 的 kernel 中占 17.3%，与 GEMM 并列第一。
- **掉队者**：SOSP 2026 的 *StreamEP: Straggler-Tolerant MoE Decoding without Communication Barriers*（只读到标题，[SOSP 2026](https://sigops.org/s/conferences/sosp/2026/accepted.html)）说明"屏障 + 掉队者"仍是开放问题。

## 4. 何时反转

- **大 batch prefill**：通信量与计算都大，传统重叠已足够，融合的边际收益小（DeepSeek-V4 一般推理 1.5× vs 小 batch 1.96× 的差异即体现）。
- **跨节点 EP**：mega-kernel 的 NVLink 假设不成立，回到 RDMA + 代理/IBGDA 路线。
- **AFD 部署**：专家在独立 GPU 上，融合发生在 FFN 侧，attention 侧不再需要 EP 通信（见 [[A3 分离的下一层]]）。

往哪走：[[A3 分离的下一层]] · [[A10 Kernel 语言与编译栈]] · [[A6 数值一致性]] · [[00 入口]]
