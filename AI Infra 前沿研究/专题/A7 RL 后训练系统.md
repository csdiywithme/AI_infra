---
type: article
status: 完整
verified: 2026-11-23
threads: ["[[T1 状态的生命周期]]", "[[T2 代价搬家账本]]"]
---

# A7 RL 后训练系统：rollout 是一个推理服务，训练器是它的消费者

> [!abstract] 一句话
> RL 后训练的系统瓶颈不在梯度计算，而在 rollout 的长尾、权重同步、沙箱环境和训推一致性。2026 年的系统把 rollout 做成独立服务（异步、多版本并存、会话亲和），把权重同步从"全量广播"变成"P2P RDMA / 增量"，并把环境沙箱当作一等资源。

## 0. 旧基线

讲座没有专门讲 RL 系统。2025 年主流（verl、slime、OpenRLHF）已经用推理引擎做 rollout、用 Megatron/FSDP 训练，但同步 on-policy、全量权重广播是默认。

## 1. Workload 要什么

- **长尾**：Meituan 的 DORA 报告 rollout 占每步 50–80% 时间，生成长度高度偏斜（[arXiv:2604.26256](https://arxiv.org/abs/2604.26256)，2026-04-29）。
- **agent 环境**：每个 episode 一个新沙箱；工具执行时间、失败和容器启动都进入关键路径（ProRL Agent，[arXiv:2603.18815](https://arxiv.org/abs/2603.18815)，2026-03-19）。
- **超大模型的权重同步**：1T 级模型每步把新权重推到所有 rollout 引擎。
- **数值一致**：见 [[A6 数值一致性]]。

## 2. 三个系统，三种切法

| | Miles v0.1（RadixArk 等，基于 slime） | DORA（美团） | ProRL Agent（NVIDIA NeMo Gym） |
|---|---|---|---|
| 来源 | [LMSYS 博客](https://www.lmsys.org/blog/2026-08-18-miles-v0-1/) 2026-08-18；github.com/radixark/miles | [arXiv:2604.26256](https://arxiv.org/abs/2604.26256) | [arXiv:2603.18815](https://arxiv.org/abs/2603.18815)；github.com/NVIDIA-NeMo/ProRL-Agent-Server |
| 主攻 | 全异步 + 生产化全家桶 | rollout 长尾 + 有界滞后 | 沙箱化 rollout 服务 |
| 异步策略 | 有界缓冲；过期样本可保留/重试/丢弃 | 多版本流式训练：rollout 实例同时持有多个策略版本，K 步滞后上限；同版本实例间零重 prefill 迁移 KV | 三阶段（INIT/RUN/EVAL）独立队列，过期任务取消 |
| 权重同步 | P2P RDMA；磁盘增量 | P2P 权重与请求迁移 | 不涉及 |
| 一致性 | R3 路由回放、TITO（token 进 token 出）、Zero-KL | — | token 进 token 出，避免重分词漂移 |
| 关键数字【作者测量】 | Kimi-K2 1T 权重更新 53.3 s → 7.2 s；GLM-4.7-Flash 每次更新 62.4 GB → 0.69–0.83 GB；GLM-5.2 744B-A40B 终端 RL，64×GB300（32 rollout + 32 训练），每步约 4.5 min，rollout 权重平均滞后 1.7 步，前缀命中 96% | 端到端吞吐最高 2.12×、rollout 阶段最高 8.2×（对同步）；工业部署数千加速卡 rollout 6.2× | Qwen3-14B 消融（8×H100）：全系统 GPU 利用 78%、0.37 实例/s；去掉负载均衡 42%、0.25/s |

## 3. 机制要点（为什么这样设计有效）

1. **权重增量很稀疏**：Miles 测得 BF16 rollout 每次更新约 2% 参数变化、FP4 约 0.5%——低精度量化让小更新"量化后不变"，增量更小。这让"磁盘增量 + 3–5 s 暂停"可行。
2. **会话亲和 ⇒ 前缀命中**：agent RL 的每条轨迹就是一个多轮会话；把各轮钉在同一引擎得到 96% 命中——与 [[A2 Agent 负载]] 的 serving 测量一致。
3. **多版本并存**：DORA 不等所有 rollout 结束就切版本，而是让新旧版本在不同实例上并行生成，再按待处理量重分 DP 组——把长尾从"整体等待"变成"局部滞后"。
4. **沙箱工程决定 GPU 利用**：ProRL 用 ptyprocess 代替 tmux、进程内 IPython、Unix socket 代替 TCP loopback，单项消融就能把 GPU 利用从 68% 提到 78%。

上游：SOSP 2026 *TensorHub: Scalable and Elastic Weight Transfer for LLM RL Training*（只读到标题）。下游：DeepSeek-V4 §5.2 小节标题包括"可抢占、容错的 rollout 服务"（p.34）、"百万上下文 RL 框架"（p.35）、"agent 沙箱基础设施"（p.35）——第二轮再次尝试，PDF 与 HTML 都在 §5.2 之前截断，正文仍未读到。

### 3.1 第二轮：DeepSeek-V4.1 披露的沙箱规模

V4.1 报告 §5.1（[arXiv:2609.19969](https://arxiv.org/abs/2609.19969)，逐字读到 §5.1.4）【作者自述】：

- 流程是 SFT → RL → 在线蒸馏，作者说"没有算法创新"，变化在数据管线：任务组织成（问题，环境，验证器）三元组，异步 RL 跨多种 agent 脚手架运行。
- 沙箱平台 DSec 支撑"数百万个并发沙箱实例"；单节点容器密度从约 1,000 提高到 2,500 以上"才出现可测的端到端退化"（曲线在图 7–8，未读数值）；用自研放置引擎，**不用 Kubernetes**。
- 训练器被抢占时，rollout **连同全部状态挂起并卸载**，而不是丢弃重来——这是 [[A8 万卡故障与 goodput]] 的"恢复粒度"思路在 RL 里的版本：故障单位是角色，而不是副本。
- Engram 表在 rollout 时留在 GPU（避免主机内存碎片 OOM）；Miles 的 day-0 支持则把冻结的 FP8 Engram 表放主机、**不参与权重同步**（[LMSYS](https://www.lmsys.org/blog/2026-09-10-deepseek-v41)）——两家对同一张表做了相反的放置选择，原因都写在各自约束里（见 [[A11 条件记忆 Engram]]）。

**我的推算**：若单节点 2,500 个容器、总量"数百万"，则需要上千台沙箱节点——**沙箱机群的规模与 GPU 机群同量级**。ProRL Agent 的消融（沙箱工程单项把 GPU 利用从 68% 提到 78%）在这个规模下意味着沙箱不是附属设施，而是 RL 吞吐的一阶因素。

### 3.2 第二轮：RL 系统成为顶会专门议题与公开基准

- OSDI 2026 设"RL Training at Scale"会话：*Weave*（分离式 RL 后训练的协同调度）、*RLinf*、*DynaRL*、*RollArt*（分离式多任务 agent RL）、*Seer*（快速同步 RL）；可靠性会话中有 *RobustRL*（RL 后训练的按角色容错）——均只读到标题（[OSDI '26 目录](https://www.usenix.org/sites/default/files/osdi26-contents.pdf)）。
- MLPerf Training v6.1（2026-11-16）加入 RL 后训练基准（据 NVIDIA 博客）；MLCommons 的说明页本轮无法打开，测什么（GRPO？rollout 是否计时？）未核实。
- vLLM v0.34.0 的 `VLLM_RELOAD_MODE=direct` 把检查点张量直接写入运行中的模型，是权重同步路径上的引擎侧支持。

## 4. 代价搬到哪里

- **off-policy 偏差**：异步换吞吐，代价是策略滞后；需要重要性采样或过期丢弃，并与 [[A6 数值一致性]] 的数值偏差叠加。
- **系统面变宽**：rollout 引擎版本、路由回放、权重格式（FP4/FP8）、沙箱镜像都必须对齐。
- **资源划分**：Miles 旗舰案例 rollout:训练 = 1:1；最优比例随轨迹长度变化，又是一个要调的比例。

## 5. 何时反转

- **短轨迹、单轮数学题**：长尾不显著，同步 on-policy 更简单、偏差更小。
- **小模型**：权重同步不是瓶颈，P2P/增量的复杂度不值得。
- **严格 on-policy 算法**：多版本流式不适用，只能靠更快的推理与更好的负载均衡。

往哪走：[[A6 数值一致性]] · [[A2 Agent 负载]] · [[A8 万卡故障与 goodput]] · [[00 入口]]
