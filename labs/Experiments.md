---
type: lab-index
area: ai-infra
aliases:
  - AI Infra Experiments
---

# Experiments

> [!abstract] 实验目标
> 用可复现的实现、profiling 和 benchmark 验证主题笔记中的系统判断，把“能解释”逐步推进到“能测量并说明原因”。

## 实验列表

- [B300 GEMM 工作流](b300-codex-workflow/README.md) · 2026-11-04 首次 smoke 在 CUDA 编译阶段结束，记录与后续构建修订见工作流说明。

![[AI Infra Catalog.base#实验]]

## 最小实验要求

每个实验至少记录：

1. 问题与假设；
2. 模型、硬件、软件版本和关键配置；
3. Workload，包括 batch、sequence length 与数据类型；
4. 指标及测量方法；
5. 原始结果与必要图表；
6. 解释、局限和异常；
7. 应更新的主题笔记。

新实验从 [Experiment Note Template](../templates/Experiment%20Note%20Template.md) 创建。
