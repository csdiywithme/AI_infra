---
type: course-index
status: developing
course: Weiming HPC Training Camp × LCPU AI Infra Seminars
updated: 2026-09-18
---

# LCPU AI Infra Seminars：整理进度与来源

本轮 **2.2、W.1、6、7 已全部完成**：每讲一份精编字幕与一份结构化笔记，共 8 份，沿用 Session 03/04 的标准。第 5 讲因无法提取字幕，已按用户要求跳过。

## 交付入口

| 讲次 | 时长 | 笔记 | 精编字幕 |
|---|---|---|---|
| 2.2 FP32 GEMM | 00:30:05 | [[LCPU AI Infra Seminars - Session 02.2 - FP32 GEMM Quick Walkthrough\|课程笔记]] | [[LCPU AI Infra Seminars - Session 02.2 - FP32 GEMM - 精编字幕\|精编字幕]] |
| W.1 TileLang | 01:26:31 | [[LCPU AI Infra Seminars - Workshop 01 - Parallel Programming with TileLang\|课程笔记]] | [[LCPU AI Infra Seminars - Workshop 01 - TileLang - 精编字幕\|精编字幕]] |
| 6 AI Communication Stack | 02:17:11 | [[LCPU AI Infra Seminars - Session 06 - AI Communication Stack\|课程笔记]] | [[LCPU AI Infra Seminars - Session 06 - AI Communication Stack - 精编字幕\|精编字幕]] |
| 7 Inference & LLM Serving | 03:25:12 | [[LCPU AI Infra Seminars - Session 07 - Inference & LLM Serving\|课程笔记]] | [[LCPU AI Infra Seminars - Session 07 - Inference & LLM Serving - 精编字幕\|精编字幕]] |

“已完成”指本次整理完成，不代表个人已掌握全部内容。课程笔记沿用 Vault 的 `status: developing`，精编字幕为 `status: polished`。

## 标准与来源边界

- 精编字幕保留原讲顺序、推导及重要问答，校正术语，删除重复口头语；不是逐字稿。
- 回看时间来自原 SRT，按分钟级位置组织；没有从课件页码猜时间。
- 笔记按概念重组，包含公式、比较、AI Infra 视角、实践检查和自测题。
- 区分录播、课件补充、整理者推导及未确认观点，不把未来硬件方向当作已实测结论。
- 原始 SRT 保留在 Downloads，未修改。第 7 讲首次错误副本也未删除或覆盖。

## 特别核验记录

### 2.2：课件文件名不等于内容匹配

[官网 `session0202.pdf`](https://infra.seminars.lcpu.dev/slides/session0202.pdf) 主要讲访存与 Reduce，与 [FP32 GEMM 视频](https://www.bilibili.com/video/BV1L8GA6YEAH/) 不完全对应。本轮以对应 BVID 的完整字幕为主，不将该 PDF 冒充本视频完整课件。

### W1：保留现场不确定性

约 49–53 分钟的 reducer 讨论未确定完整 lowering，约 69–71 分钟对 TVM-FFI 的收益有不同实测反馈。字幕保留这些边界；官方课件中较详细的 reducer 分析单独纳入笔记，避免倒填为现场已确定结论。

### 第 6 讲：编号与覆盖范围

B 站第 6 讲对应官网 **Workshop 02**。录播重点是可消费语义和物理路径；后半部为方案、workload、scaling 与硬件方向的概览。笔记中更细分内容另参考课件，不全部归为现场展开。

### 第 7 讲：已排除误命名副本

首次名称含 `BV1gY4d6GEwR_P字幕.srt` 的文件，SHA-256、2151 条字幕及约 02:17:09 结束时间均与第 6 讲完全相同，因此没有采用。用户重新导出的 `BV1gY4d6GEwR_字幕.srt` 为 3602 条，覆盖至 03:25:11，内容已完整阅读并匹配本讲。

目前未取得可核实的第 7 讲官方课件，因此不编造页码。对带宽下界、长上下文 FLOPs、RC/IBGDA 分类、graph 约束及 speculative decoding 的“无损”条件另加澄清；新增论文/实现核验链接放在笔记相应位置。

### 第 5 讲：按用户要求退出范围

插件提示未检测到可提取字幕，用户要求跳过，已停止该讲转写与整理。此前的 [[LCPU AI Infra Seminars - Session 05 - Towards Modern Networking System]] 仅保留为 `slides-only-user-skipped` 草稿，不计入交付。

## 原字幕指纹

| 讲次 | 条数 | 最后字幕结束 | SHA-256 |
|---|---:|---|---|
| 2.2 | 644 | 00:30:02.160 | `477b367ec5b549fe1261781635d7e0ae22540ef4e0898a10ae86a7ac2db5e821` |
| W1 | 1658 | 01:26:30.660 | `998721a53cc5a64a7a969730a7dfc24362325f8dd0a5575aa4deca72e3d8adaf` |
| 6 | 2151 | 02:17:08.879 | `8257b8b287592a01b8c908af23fd23a2bb45a7756c26617cb973dbcefe06b05a` |
| 7 | 3602 | 03:25:11.300 | `1155361fa657812777f26dbe8ed57afe02f0b1f4f98e9b0cf98aa67282adcb33` |

## 视频与官方材料

- [合集](https://space.bilibili.com/3461562830424779/lists?sid=8682435) / [课程日历](https://infra.seminars.lcpu.dev/schedule)
- [2.2 视频](https://www.bilibili.com/video/BV1L8GA6YEAH/)
- [W1 视频](https://www.bilibili.com/video/BV1rhMD6YEKM/) / [Workshop 01 课件](https://infra.seminars.lcpu.dev/slides/workshop01.pdf)
- [第 6 讲视频](https://www.bilibili.com/video/BV1FYhL6REgi/) / [Workshop 02 课件](https://infra.seminars.lcpu.dev/slides/workshop02.pdf)
- [第 7 讲视频](https://www.bilibili.com/video/BV1gY4d6GEwR/)

视频元数据由 B 站公开 view 接口的合集条目核实。字幕通过用户指定的 Chrome「BiliBili 字幕提取器」获得，不依赖未登录接口的空字幕列表。
