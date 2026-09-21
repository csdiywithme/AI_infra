# AI Infra 学习笔记

以 Stanford CS336 为基础框架，结合 CMU 11-763，系统学习大语言模型与 AI Infra，重点关注 LLM 推理优化，并沉淀课程笔记、主题知识、实验记录和求职准备材料。

本仓库同时是一个 Obsidian Vault。Markdown 正文保持可移植；Obsidian 的 Properties、Bases、Backlinks、Templates 和 Callouts 用于组织、检索与预览。

## 入口

- Obsidian 动态首页：[Home](Home.md)
- 学习路线：[AI Infra Learning Roadmap](roadmaps/AI%20Infra%20Learning%20Roadmap.md)
- 实验索引：[Experiments](labs/Experiments.md)

## 课程主线

- [Stanford CS336 — Language Modeling from Scratch](courses/stanford-cs336/Stanford%20CS336.md)：建立从模型、数据、训练到系统实现的完整基础。
- [Stanford CS149 — Parallel Computing](courses/stanford-cs149/Stanford%20CS149.md)：建立 multi-core、SIMD/SIMT、memory hierarchy、GPU 与并行性能优化基础。
- [CMU 11-763 — Inference Algorithms for Language Modeling](courses/cmu-11-763/CMU%2011-763.md)：学习生成算法、质量与延迟权衡、可控生成及 inference-time scaling。
- [LCPU AI Infra Seminars](courses/lcpu-ai-infra-seminars/LCPU%20AI%20Infra%20Seminars%20-%20整理进度与来源.md)：围绕 GEMM、Tensor Core、流水线、TileLang、通信与 LLM Serving 整理课程笔记和精编字幕。
- [HPCGame 2026 — MLSys 知识地图](courses/hpcgame/HPCGame%202026%20MLSys/HPCGame%202026%20-%2000%20MLSys%20知识地图.md)：串联训练、推理、通信、Kernel、后训练与集群效率，并保留前沿追踪和证据记录。

## 最新进展（2026-09-21）

以下按资料与笔记整理进度记录，学习掌握程度与实验结果以各笔记中的记录为准。

- **CS149**：新增第 11–19 讲，现已覆盖第 1–19 讲及 ISPC 阅读笔记；补充自测参考回答与课程编号说明。
- **CS336**：扩充第 1–18 讲的自测参考回答，覆盖资源核算、模型架构、GPU、并行、推理、数据与后训练。
- **CMU 11-763**：新增 15 篇，累计整理 17 讲；覆盖采样与搜索、推理链、工具与智能体、推理时扩展、KV Cache 和稀疏/线性注意力。课程索引标注各讲资料来源及仍缺授课材料的讲次。
- **LCPU**：纳入 FP32 GEMM、Tensor Core、Pipeline、TileLang、AI Communication Stack、Inference & LLM Serving 六部分的笔记与精编字幕；第 5 讲保留课件草稿状态。
- **HPCGame 与主题笔记**：新增 MLSys 总览、知识地图、九个专题和追踪记录；补充 LLM Inference 的课程入口与 Transformer Block 自测参考回答。

## 主题笔记

- [Transformer Architecture](topics/model-architecture/Transformer%20Architecture.md)
- [Transformer Block](topics/model-architecture/Transformer%20Block.md)
- [LLM Inference](topics/inference/LLM%20Inference.md)

## 知识沉淀流程

```text
课程学习
→ 课程笔记保留来源与讲次上下文
→ 原地记录扩展问题
→ 跨课程结论沉淀到主题笔记
→ 实现或实验验证系统判断
→ 更新主题结论与掌握程度
```

## 目录约定

```text
courses/    按课程组织的来源笔记
topics/     可持续更新的主题知识
labs/       实现、profiling 与 benchmark
roadmaps/   学习顺序和阶段目标
templates/  Obsidian 原生笔记模板
assets/     图片及小型附件
```

附件先进入 `assets/_inbox/`，整理笔记时再决定转写为 Markdown、删除，或按
`assets/courses/<course>/lecture-XX/` 归档。长期保留的图片使用语义化文件名、相对
Markdown 链接和来源时间戳；完整约定见 [assets/README](assets/README.md)。

Properties 中的 `status` 使用：

```text
seed → developing → stable
```

主题掌握程度使用：

```text
explain → derive → implement → benchmark
```

未解决问题写在产生位置：

```markdown
- [ ] #question 问题内容 → 预计沉淀到相关主题
```

来自视频的问题应同时记录时间戳：

```markdown
- [ ] #question [视频 51:28](https://www.youtube.com/watch?v=VIDEO_ID&t=3088s)——问题内容
```

在 Obsidian 中打开仓库根目录后，建议首先打开并收藏 [Home](Home.md)。
