---
type: article
status: 完整（第二轮重写；修正了第一轮一个错误推断）
verified: 2026-11-23
threads: ["[[T1 状态的生命周期]]", "[[T2 代价搬家账本]]", "[[T3 结论反转条件]]"]
tags: [engram, conditional-memory, embedding-offload, memory-tier]
---

# A11 条件记忆 Engram：一种按"截止时间"而不是按"带宽"放置的模型参数

> [!abstract] 一句话
> Engram 用确定性哈希的 N-gram 查表补充 MoE：查哪一行**只取决于输入 token**，所以可以在前向开始前发起读取。2026 年 9 月它进入了发布的旗舰模型（DeepSeek-V4.1-Flash：552B 主干 + 196B Engram 参数），随后第二家实验室的模型（Qwen3.8-Flash-Next）也带上了同类 N-gram 表。第二轮研究改变了我对"能放多远"的理解：**决定放置的不是 PCIe 带宽，而是"表在第几层被用到"给出的截止时间；而主机内存一旦放了 180 GiB 的表，就开始和 KV 卸载抢同一块 DRAM。**

> [!warning] 对第一轮的更正
> 第一轮我根据 vLLM v0.33.0 的一条说明（"GPU Engram tables read from peer HBM"）推断"生产部署因延迟预算更紧而放在对端 HBM"。第二轮读到完整条目与官方部署配方后，这个推断**不成立**：那一条是修复 GB200 上 **GPU 常驻模式**下 DEP decode 卡死的问题（#62091）；vLLM 的 V4.1 配方默认用 `--engram-config '{"cpu_offload":true}'` 把表放在**锁页主机内存**、经 UVA 读取（[vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)）。下文 §4 的反转条件已按新证据重写。

---

## 0. 旧基线

讲座在 CPU / C2C / 存储一节提到 Engram（[视频 1:50:47](https://www.bilibili.com/video/BV1vmzXBBESY/?t=6647)），当时它只是论文（[arXiv:2601.07372](https://arxiv.org/abs/2601.07372) v1，2026-01-12；本轮读到 v2，2026-07-12）。第一轮笔记只读了论文前半，§6.4 的系统条件没读到。

## 1. 机制：为什么 Engram 的行可以提前取

论文（v2，§2.2–§2.5，逐字读到）：

- token ID 先规范化（NFKC + 小写），128k 词表的有效规模降 23%；
- 每个 N-gram 阶有 K 个哈希头（乘法–异或哈希），各索引一张素数大小的表；取回的向量以当前隐藏状态为 query 做门控后融合；
- **训练**：表按行分片到多张 GPU，前向用 All-to-All 收集活跃行、反向把梯度发回；
- **推理**：从主机内存经 PCIe 异步预取，与前面的层重叠；可以再做多级缓存——高频 N-gram 放 HBM 或 DRAM，长尾放 NVMe（Zipf 分布）；
- **放在早层**：建模上"早介入"更好（替代底层的局部模式重建）——这一点后面会变成系统约束。

对比两种"大参数"：

| | MoE 专家 | Engram 表 |
|---|---|---|
| 选哪部分 | 路由器根据**隐藏状态**决定，到该层才知道 | **输入 token** 决定，前向开始时就知道 |
| 能否提前取 | 不能（只能投机） | 能 |
| 每 token 访问量 | 一整个专家 FFN 的权重 | 每层每头一行 |

### 1.1 稀疏预算怎么分

论文 §3.1（逐字）：在等总参数、等激活参数下，把稀疏预算的 **20–25%** 从专家挪给 Engram 最好（ρ≈75–80%，损失从 1.7248 降到 1.7109）。**我的推算**：V4.1-Flash 的 196B / (552B + 196B) ≈ 26%，Qwen3.8-Flash-Next 的 51B / (125B + 51B) ≈ 29%——两款发布模型都落在论文给出的区间附近（两者训练细节不同，这只是一致性检查，不是因果证据）。

## 2. 发布与部署：从论文到两个模型

| 项 | 内容 | 来源 / 证据类型 |
|---|---|---|
| DeepSeek-V4.1-Flash | HF 首发 2026-09-09（FP8 "0909"），权重更新 2026-10-17（"1017"）；40 层，384 路由专家 + 1 共享、每 token 选 6；**Engram 在第 1 与第 14 层**（从 0 计），N-gram 阶 {2,3,4}，8 个哈希头，FP8；196B 参数；prefill 每 token 激活 8B、decode 16B；1M 上下文 | [V4.1 报告 §2.1、§2.4.2，arXiv:2609.19969 v1，2026-09-17](https://arxiv.org/abs/2609.19969)；[vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)；[NeMo AutoModel 文档](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/deepseek-ai/DeepSeek-V4.1-Flash) 【已发布】 |
| V4.1 推理设计 | 嵌入放主机内存，后台 RDMA 预取；第一模块的预取与第一个 Transformer block 重叠；**报告未给开销数字** | V4.1 报告 §2.4.2【作者自述】 |
| V4.1 训练 | 表按行分到 engram-parallel 组，优化器状态在表副本间分片；每个流水线 stage 开始前预取；嵌入学习用动量 + Sinkhorn 平衡，学习率 5× | V4.1 报告 §2.5、§3.1.3、§4.2.2 |
| V4.1 RL | **rollout 时表留在 GPU 显存**，以避免主机内存碎片导致的 OOM | V4.1 报告 §3.1.3 |
| Qwen3.8-Flash-Next | 125B 主干 + 51B N-gram 嵌入（检查点里叫 `ple`），每 token 激活 6B，3:1 GDN 与 Qwen Sparse Attention；只用二元与三元组、各 8 头；**不做 token 规范化**（与 DeepSeek 实现不同）；vLLM 默认 CPU 卸载，且"卸载目前只在 NVIDIA 设备上运行"；昇腾 910B 上表放 /dev/shm | [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)、[NeMo Engram 组件](https://docs.nvidia.com/nemo/automodel/latest/nemo-automodel/nemo_automodel/components/models/qwen3_8_flash_next/engram.md) 【已发布】；发布日期两页都未给出 |

**措辞注意**：我不说"第一个使用 N-gram 嵌入的模型"——论文自己的基线里就有 OverEncoding，Gemma 系列有逐层嵌入；能确认的只是"本轮读到的、带 Engram 式条件记忆的发布模型有两个，DeepSeek 的在先"。

### 2.1 引擎侧的演进（按版本）

| 版本 | 日期 | 与 Engram 相关的项 |
|---|---|---|
| vLLM v0.30.0 / v0.31.0 | — | V4.1 架构首次进入 v0.30.0；配方要求 ≥0.31.0 |
| SGLang（day-0，版本号博客未写） | 2026-09-10 | 两种布局："host-sharded"（每 rank 一份分片，保留 all-reduce）与"shared host"（一份完整主机副本，去掉 all-reduce）；GB300 容器的共享映射不支持大页，所以评测用了 host-sharded（[LMSYS 博客](https://www.lmsys.org/blog/2026-09-10-deepseek-v41)） |
| vLLM v0.33.0 | 11-02 | GPU 常驻表从对端 HBM 读，修复 GB200 DEP decode 卡死（#62091）；PP 下的 Engram 修复；Engram all-to-all 卡死修复 |
| vLLM v0.34.0 | 2026-11（具体日期未核实） | 主机表在 DP rank 间共享（#62359）；Qwen3.8 的 PLE 可**磁盘卸载**（`disk_offload`，#59958，DGX Spark 上少用 27 GiB） |
| DeepEP V2 | README 无日期 | 实验特性"0 SM Engram（RDMA）"：不占 SM 的 Engram 通信 |

## 3. 一个可算的例子：带宽不是瓶颈，截止时间才是

**我的推算**，数字来自 SlackTier（Microsoft 等，[arXiv:2611.13413](https://arxiv.org/abs/2611.13413)，2026-11，实测于 4×H200、TP4/EP4、SGLang 0.5.22）：

- V4.1 每层每 token 查 3 阶 × 8 头 = **24 行**，每行 264 B ⇒ 每 token 每层约 6.2 KiB，两层约 12.4 KiB（SemiAnalysis 独立给出"约 12.4 KiB/token 位置"，一致）。
- decode、batch 128：每层每步约 3,000 行 ≈ **0.8 MB**（论文实测）；256 个 decode token 中 95–99.6% 的行互不相同——缓存不太帮得上 decode。
- 0.8 MB 走 PCIe Gen5（按约 50 GB/s 有效带宽算）只要约 16 µs。**带宽完全够。**
- 但**截止时间**很紧：表在第 1 层被读，而第 1 层之前只有约 **0.27–0.3 ms** 的计算可以遮掩；第 14 层之前约 **9.3 ms**（同一模型实测，相差约 30×）。NVMe 读的固定截距约 0.30 ms，几乎等于第 1 层的整个窗口。

结论：**同一张表，放在第 14 层可以从 SSD 读；放在第 1 层就只能放 DRAM 或加 HBM 热缓存。** 论文的经验规则是"第 k 层的表大约有 k 层的计算可以遮掩"。这把论文的建模选择（早层更好）直接变成了系统约束（早层更难卸载）——一个模型设计与存储层级互相牵制的干净例子。

实测（Table 2，batch 128，agent / chat 两种 trace，相对"表放主机 DRAM"的步长 29.41 / 33.71 ms）：

| 配置 | 步长变化 |
|---|---|
| 两张表放 1 块盘，无缓存 | +6.4% / +7.2% |
| 放 4 块盘，无缓存 | +2.6% / +2.6% |
| 4 块盘 + 100 万行分层缓存 | +0.7% / +1.7% |
| 对端机器 DRAM 经 400 Gb/s IB | −0.7% ~ 0.0% |

主机内存占用从 223 GiB 降到 34 GiB；把省下的主机内存给 KV 复用后，KV 受限的 agent 负载中位吞吐 +36%、p90 TTFT 38 s → 24 s【作者测量；补丁未公开】。

## 4. 代价搬到哪里

- **主机 DRAM 被双重占用**：V4.1 的 Engram 表 183–189 GiB（不同格式统计不同），而 V4.1 推理还把 SWA KV 池设为每台机器主机 DRAM 的 10%（[[A1 KV 状态变小之后]]）。SGLang 的 host 卸载在 4×GB300 上把 KV cache 容量提高 36%、decode 吞吐与 TTFT"相当"【作者测量，未给百分比】——表挪出 HBM 换来的是 KV 空间；SlackTier 又把表挪出 DRAM 换回 KV 复用。**Engram 放置和 KV 分层是同一个容量预算问题。**
- **布局与操作系统细节**：大页是否可用决定了用"共享主机副本"还是"分片 + all-reduce"（SGLang）；RL rollout 因主机内存碎片而把表放回 GPU（V4.1）。
- **训练通信**：表分片 + All-to-All，反向梯度回到所有者；NeMo 实现中所有者行存的是各 rank 贡献的均值。
- **移植**：Qwen3.8 的卸载目前只在 NVIDIA 上运行；昇腾用 /dev/shm 放表。

## 5. 何时反转

| 条件 | 结论 | 依据 |
|---|---|---|
| 表在**很早的层**被读（第 1 层） | 不能放 SSD；需 DRAM 或 HBM 热缓存 | SlackTier：W1≈0.3 ms vs NVMe 截距 0.30 ms |
| 表在**较深的层**（第 14 层） | 单块 NVMe 无缓存也在 ±0.5% 内（8–256 会话） | 同上 |
| **SSD 实现方式** | mmap、非截止时间感知的 SSD 卸载不划算：B200、约 125 tok/s/user 时 DRAM 1.21 亿 token/美元 vs SSD 0.52 亿（SemiAnalysis，2026-09-18，【第三方测量·付费墙部分未读】）；截止时间感知 + 多盘 + 热缓存则只多 0.7–1.7%（SlackTier）。**两组数字不矛盾，差在实现、盘数和缓存** | 两者硬件、并发不同，不可直接横比 |
| **KV 受限**的负载 | 表留在 DRAM 的机会成本是 KV 复用；挪到 SSD 反而提升吞吐 | SlackTier +36% |
| **RL rollout** | 表回到 GPU | V4.1 §3.1.3 |
| 多 GPU 推理、表放 HBM | 在 B300 上把表移回 HBM"没有超出运行间波动"的收益；而放 DRAM 让 TP4 变 TP2，Pareto 最多提升 1.6× | SemiAnalysis【第三方测量】 |
| 非 NVIDIA 硬件 | 卸载路径尚未通用 | Qwen3.8 配方 |
| 想在推理时去掉 Engram 省内存 | 不可行：去掉后 CRUXEval 答案损失 0.2848 → 0.3093 bits/token | SemiAnalysis |

## 6. 论文自己的系统测量（§6.4，【摘要级】）

H800；为避免专家并行的通信干扰，用稠密 4B 与 8B 主干；100B 参数的 Engram 层整表放主机 DRAM，插在第 2 个 block；512 条序列、长度在 100–1024 均匀分布。吞吐：4B 9,031.62 → 8,858.28 tok/s，8B 6,315.52 → 6,140.02 tok/s（**我的算术**：约 1.9% 与 2.8%）。这就是引言里"<3%"的条件：**稠密小主干、短序列、表在第 2 层**——与 V4.1 的 MoE、1M 上下文、第 1 层部署相差很远。

## 7. 未解决

- [ ] V4.1 报告 §5.2 以后与 §6 局限本轮仍未读到（PDF/HTML 均截断）。
- [ ] Qwen3.8-Flash-Next 的发布日期与技术报告。
- [ ] 非拉丁语言下哈希冲突与覆盖率：论文无定量讨论（§6.5 只有定性），Qwen3.8 不做规范化——两种选择对多语言的影响无公开对照。
- [ ] DeepEP V2 "0 SM Engram" 的性能数字。

往哪走：[[T1 状态的生命周期]] · [[A1 KV 状态变小之后]]（同一块主机内存）· [[A9 硬件分工化]]（CPU 内存与 C2C）· [[00 入口]]
