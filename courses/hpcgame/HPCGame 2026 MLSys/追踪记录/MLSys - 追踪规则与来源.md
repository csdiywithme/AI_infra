---
type: research-protocol
status: archived
protocol_version: 2
updated: 2026-10-12
timezone: Asia/Shanghai
automation_id: mlsys-follow-up
---

# MLSys：旧追踪规则入口与来源索引

现行流程见 [[AI Infra 系统主线（2026-10 重构）/追踪记录/追踪规则|系统主线追踪规则]]，分来源覆盖记录见 [[研究记录]]。HPCGame 历史运行与证据见 [[MLSys - 运行记录]]、[[MLSys - 证据台账]]。

## 来源登记

下表来自 2026 年 10 月的来源登记，用于回查旧笔记的研究入口。

以下是发现入口，不是既定结论。先读 release/tag/报告正文，再按需要进入 PR、issue、版本化文档；`main`、`latest` 和 README 是可变页面，保存核验日与对应 commit/版本（能取得时）。链接失效时可寻找官方迁移地址，记录替换关系。

| 主线 | 旧流程常用入口 | 轮换、事件触发与反证来源 |
| --- | --- | --- |
| T1 | [Megatron Core](https://github.com/NVIDIA/Megatron-LM/releases)、[PyTorch](https://github.com/pytorch/pytorch/releases)、[TorchTitan](https://github.com/pytorch/torchtitan) | [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)、[NeMo Curator](https://github.com/NVIDIA-NeMo/Curator)、[Transformer Engine](https://github.com/NVIDIA/TransformerEngine/releases)；模型作者数据/recipe/训练报告、收敛与复现研究 |
| T2 | [verl](https://github.com/verl-project/verl)、[AReaL](https://github.com/areal-project/AReaL)、[slime](https://github.com/THUDM/slime) | 作者 RL 系统论文、环境与 reward 工具、推理预算/验证/蒸馏研究；[verl async 机制](https://github.com/verl-project/verl/blob/main/docs/advance/fully_async.md)是历史基线入口，0.9.1 已弃用旧 experimental trainer，后续核新 V1 文档，不以页面更新时间冒充机制首次出现 |
| T3 | [vLLM](https://github.com/vllm-project/vllm/releases)、[SGLang](https://github.com/sgl-project/sglang/releases)、[Dynamo](https://github.com/ai-dynamo/dynamo/releases) | [KTransformers](https://github.com/kvcache-ai/ktransformers)、[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/releases)、[LMCache](https://github.com/LMCache/LMCache)、[Mooncake](https://github.com/kvcache-ai/Mooncake)；旧 issue、复现脚本与兼容矩阵 |
| T4 | [DeepEP](https://github.com/deepseek-ai/DeepEP)、[NIXL](https://github.com/ai-dynamo/nixl/releases)、[NCCL](https://github.com/NVIDIA/nccl/releases) | [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[UCCL](https://github.com/uccl-project/uccl)、[NVSHMEM](https://docs.nvidia.com/nvshmem/release-notes-install-guide/index.html)、[FlashAttention](https://github.com/Dao-AILab/flash-attention)、[FlashInfer](https://github.com/flashinfer-ai/flashinfer)；模型作者 attention/MoE/conditional memory 技术报告 |
| T5 | [Resiliency Extension](https://nvidia.github.io/nvidia-resiliency-ext/)、[llm-d](https://github.com/llm-d/llm-d)、[Kueue](https://github.com/kubernetes-sigs/kueue/releases) | [TorchFT](https://github.com/pytorch/torchft)、[vLLM security](https://docs.vllm.ai/en/latest/usage/security/)、维护者故障分析、生产 trace、checkpoint/straggler/failure-injection 论文 |
| T6 | [Triton](https://github.com/triton-lang/triton/releases)、[CUTLASS](https://github.com/NVIDIA/cutlass/releases)、[cuDNN Frontend](https://github.com/NVIDIA/cudnn-frontend/releases) | [torch.compile](https://docs.pytorch.org/docs/stable/torch.compiler.html)、[TileLang](https://github.com/tile-ai/tilelang)、[CUDA Tile](https://github.com/NVIDIA/cutile-python)、[Helion](https://github.com/pytorch/helion)；AI kernel/编译 agent 原论文、代码和独立执行评测 |
| T7 | [SGLang-Omni](https://github.com/sgl-project/sglang-omni)、[Diffusers](https://github.com/huggingface/diffusers/releases)、[SGLang](https://github.com/sgl-project/sglang) | 作者 diffusion/video/audio/diffusion-LM 技术报告；多阶段 streaming、跨步 cache、实时任务和环境执行的系统论文 |
| T8 | [NVIDIA 技术博客](https://developer.nvidia.com/blog/)、[ROCm TheRock](https://github.com/ROCm/TheRock/releases)、[AITER](https://github.com/ROCm/aiter) | 旧 ROCm 地址转至 [legacy-rocm-build](https://github.com/ROCm/legacy-rocm-build/releases)，不能仅盯旧仓库；[MaxText](https://github.com/AI-Hypercomputer/maxtext)、[SGL-JAX](https://github.com/sgl-project/sglang-jax)、[UALink](https://ualinkconsortium.org/)、[MLX](https://github.com/ml-explore/mlx)；TPU/NPU、CXL/C2C/DPU/HBM/供电散热资料须区分 spec、可获得硬件和部署 |

横切评测入口：[MLPerf Endpoints](https://mlcommons.org/benchmarks/endpoints/)、[MLPerf Inference](https://github.com/mlcommons/inference)。作者或厂商自测、同行审查提交、独立复测分别标注；同行审查不自动等于独立执行。
