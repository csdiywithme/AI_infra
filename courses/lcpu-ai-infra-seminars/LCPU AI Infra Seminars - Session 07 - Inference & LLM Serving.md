---
type: course-note
status: developing
course: Weiming HPC Training Camp × LCPU AI Infra Seminars
session: 7
speaker: 孔昊然
video_url: https://www.bilibili.com/video/BV1gY4d6GEwR/
source_status: transcript-reviewed-with-primary-source-checks
topics: [llm-inference, kv-cache, serving, scheduling, disaggregation, moe, speculative-decoding]
aliases: [LCPU Session 07, Inference & LLM Serving]
---

# Session 07：Inference & LLM Serving

> [!abstract] 一句话主线
> 推理优化不是给引擎打开一组“高级开关”，而是在模型、请求分布、缓存状态、硬件和延迟要求给定后，安排权重与 KV 的复用、计算与搬运、批处理与投机，最大化满足 SLO 的有效吞吐。

## 来源与使用说明

- [视频](https://www.bilibili.com/video/BV1gY4d6GEwR/)，03:25:12；讲者孔昊然。
- [[LCPU AI Infra Seminars - Session 07 - Inference & LLM Serving - 精编字幕]]：保留原顺序、分钟级时间与现场补充。
- 已阅读正确导出的 3602 条字幕，覆盖至 03:25:11。第一次名为第 7 讲的字幕实际重复第 6 讲，已排除。
- 当前未取得可核实的第 7 讲官方课件；本文不编造页码或课件内容。公式是依据讲述展开的整理者推导，注明假设；部分易混淆术语另核对原论文/官方实现。
- 讲者关于公司部署、引擎成熟度、未来硬件及宣传倍数的评价，是讲座时的观点，不作为当前产品选型结论。

## 回看地图

| 时间 | 主题 |
|---|---|
| [04:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=240) | 自回归推理、Prefill/Decode、KV 合法性 |
| [14:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=840) | KV 容量、计算强度、带宽下界、batch |
| [27:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=1620) | SLO、goodput、真实 trace 与 benchmark |
| [44:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=2640) | PagedAttention、prefix cache、分层 KV、MLA |
| [65:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=3900) | 长上下文、CP、稀疏 attention |
| [83:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=4980) | Continuous batching、chunked prefill、scheduler |
| [99:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=5940) | PD 分离、KV transfer、全局路由 |
| [117:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=7020) | TP/PP/EP/CP、DP attention 与负载均衡 |
| [141:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=8460) | Attention/GEMM/通信 kernel、graph、MegaMoE |
| [168:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=10080) | Speculative decoding 与 batching 的取舍 |
| [183:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=10980) | 引擎分层、RL rollout、总结与适用边界 |

## 1. 为什么可以缓存 KV，而不必缓存所有历史 Q

对确定模型、前缀、位置编码及推理配置，causal attention 使历史 token 的表示不依赖未来追加 token。历史每层 K/V 因此可以保存，供新 token 的 Q 查询。

新一步的 attention 读取历史 K/V；历史 Q 已经完成其查询工作，普通自回归 decode 不再需要它。双向 encoder 等允许未来输入改变历史表示的结构，不能直接套用同一种缓存合法性论证。

KV 也不只是 token ID 的函数：它依赖完整前缀、模型权重、层、位置与配置。相同的一段后缀如果前文不同，不能仅凭 token ID 相同就共享 KV。

丢弃缓存后可以重计算，也可以从另一层存储取回；哪个划算是系统决策，而不是“缓存永远比算快”。

## 2. KV 容量：先把模型结构代入，再谈优化

对普通 MHA/GQA/MQA，设层数 $n_l$、KV heads 数 $h_{kv}$、每 head 维度 $d_h$、每元素字节数 $s$。单请求、单 token 的全模型 KV 大小为：

$$\kappa=2n_lh_{kv}d_hs.$$

前面的 2 表示 K 与 V。长度 S、batch B 的逻辑 KV 总量近似为：

$$M_{KV}=B S\kappa.$$

这还未计页尾碎片、元数据、复制、工作区、量化 scale 和额外 indexer 状态；经过 TP/CP 后每卡实际占用要按真实分片重新算。

课堂 70B GQA 示例使用 80 层、8 个 KV heads、head dimension 128、BF16：

$$\kappa=2\times80\times8\times128\times2=327680\ \text{bytes}=320\ \text{KiB/token}.$$

128 Ki tokens 的单请求逻辑 KV 为 40 GiB。这里统一使用二进制单位，不能把 KiB/GiB 和十进制带宽混算。

MLA 缓存 latent 与解耦位置分量，不能继续机械套 `2×heads×head_dim`；DSA 还可能有 indexer cache。模型结构直接改变容量、可承载 batch、传输量与调度空间。

## 3. Prefill 与 Decode：常见判断是条件结论

| | Prefill | 普通 Decode |
|---|---|---|
| 一次处理 | 许多未缓存 prompt tokens | 每请求一个新 token |
| 线性层形态 | 较大的 GEMM | 小 batch 时接近 GEMV；batch 大时变为 GEMM |
| 主要复用 | 同一权重被多个 prompt tokens 使用 | 同一权重被 batch 中多个请求使用 |
| 常见压力 | 算力、长上下文 attention、activation | 权重带宽、KV 带宽、launch/scheduling |
| 例外 | 高 prefix hit 后待算后缀很短 | 大 batch/speculation 后算力或通信成为瓶颈 |

“Prefill compute-bound，Decode memory-bound”是有用的起点，不是模型或阶段名字自带的性质。应该先算实际 shape、算术强度与流量，再查 profiler。

## 4. Decode 的简化 Roofline 推导

以下先忽略 attention 的上下文相关 FLOPs，适用于权重相关线性层主导的粗估。设参数量 P、权重字节数 $s_w$、batch B、每请求上下文 L：

$$F_{step}\approx2BP,\qquad Q_{step}\approx s_wP+B\kappa L.$$

$$I_{decode}\approx\frac{2BP}{s_wP+B\kappa L}.$$

于是：

- 权重读取主导时，$I\approx2B/s_w$，batch 增大可摊薄权重读取。
- KV 读取主导时，$I\approx2P/(\kappa L)$，在这个简化模型里增大 batch 的收益趋于饱和。
- 长上下文下还要把 attention FLOPs 加回分子；真实访存也会因稀疏、cache、融合和布局而变化。

对 MoE，不能把总参数量 P 直接当作每步实际读取的权重。应考虑被激活 experts 的并集、每 expert 的 token 数、共享层和真实权重复用。

时间下界可粗写成：

$$T_{step}\gtrsim\max\left(\frac{F_{step}}{C_{eff}},\frac{Q_{step}}{BW_{eff}}\right).$$

端到端还包含通信、CPU 调度、排队、kernel launch 与不能隐藏的同步；上述式子不是完整 serving 模型。

### 课堂的“42 ms / 24 tokens/s”该怎样读

70B 参数、2 bytes/weight，约有 140 GB 权重；按 3.35 TB/s 的理想带宽串行读一次：

$$140\ \text{GB}/3350\ \text{GB/s}\approx41.8\ \text{ms}.$$

若每次只生成一个 token，则约 24 tokens/s。**这不是一张 H100 80GB 能直接运行 140GB 权重的部署方案**；它只是忽略容量与并行的带宽算术。真实系统必须量化、切分或 offload，再用实际每设备流量、聚合带宽和通信成本重算，不能直接复用这个数字。

## 5. 长上下文：把省略的 attention 成本加回来

对标准 dense causal attention，设隐藏维度 d，单步 decode 的 QK 与 AV 主计算约为：

$$F_{attn,decode}\approx4n_lLd.$$

加上线性层：

$$F_{decode/token}\approx2P+4n_lLd.$$

以课堂 70B、80 层、d=8192 粗估，两项相当的位置约为：

$$L_*\approx\frac{P}{2n_ld}\approx53{,}400\ \text{tokens}.$$

> [!important] 不能混用“每步成本”和“整段 Prefill 总成本”
> 若有效利用 causal 三角区域，整个长度 L 的 Prefill attention 主计算约为 $2n_ldL(L+1)$，加上线性层约 $2PL$。常数依实现和统计口径不同，但总量随 L² 增长。上面的 53K 是单步/单位置的简化交叉估计，不是任意 Prefill 总耗时的统一临界点。

FlashAttention 主要减少中间矩阵的 HBM 读写，不会把 dense attention 的平方级算术自动改成线性级。稀疏、压缩或线性 attention 才是在改变访问/模型结构，准确性与训练要求需要另外分析。

## 6. Serving 指标：先定义 SLO，再谈吞吐

| 指标 | 衡量什么 | 常见误读 |
|---|---|---|
| TTFT | 请求到首个输出 token 的时间 | 不只是 Prefill kernel 时间，还含排队、路由与 KV 获取 |
| TPOT | 通常是首 token 后的平均每输出 token 时间 | 平均值可能掩盖停顿；单 token 输出要特殊处理 |
| ITL | 输出 token 间隔/流式间隔的分布 | Speculative decoding 成批输出时，要说明 token 与 chunk 时间戳口径 |
| E2E latency | 请求开始到最后输出完成 | 受输出长度影响，不能不控制长度直接比 |
| Throughput | 单位时间完成的请求/tokens | 要注明 input/output、缓存命中和是否包含违约请求 |
| Goodput | 满足指定 SLO 的有效工作量/时间 | 不存在脱离 SLO 与 workload 的唯一数值 |

一个常见但须明确口径的定义是：

$$Goodput=\frac{\sum_{r\in\mathcal R}\mathbf1[r\text{满足SLO}]\,work(r)}{\Delta t}.$$

`work(r)` 可按成功请求数或输出 token 数定义。需要同时报告 offered load、完成量、拒绝/失败量和违约比例，不能靠丢弃慢请求制造“更好”的结果。

Online 更关心延迟约束下的 goodput；offline 可以接受更长等待以提高吞吐。Agent/coding 的多轮与前缀复用，又会改变缓存和调度需求。

## 7. PagedAttention、prefix cache 与分层存储：三种不同问题

| 机制 | 解决的问题 | 不自动解决的事 |
|---|---|---|
| PagedAttention | KV 物理分配与碎片；逻辑块→物理块 | 跨请求前缀匹配、跨机传输 |
| Prefix caching | 复用相同前缀产生的状态 | 请求一定路由到有缓存的实例 |
| KV tiering/offload | 扩大可保存状态的容量 | 取回一定比重算快，或一定满足 TTFT |

按最大序列长度预留连续空间会浪费 HBM，进而限制 batch。分页将尾部浪费控制到页粒度，但增加 block-table 寻址与管理；kernel 必须支持该布局。

Page size 大有利于部分批量搬运/寻址，小则减少碎片并改善管理粒度。最佳大小受 kernel、对齐、请求长度和传输方式约束，不能仅凭一个 TMA 阈值决定。

### Prefix hit 的正确性条件

除 token 前缀外，还应匹配模型/权重版本、adapter、位置和影响状态的配置；多模态输入也需要相应身份。共享块被追加或修改时要处理引用计数、copy-on-write 和所有权。活跃请求仍依赖的状态不能直接回收。

vLLM 的 block-hash 路线与 SGLang 的 radix-tree 路线是课堂的历史设计对比，不代表今日功能边界互斥。最终要在同一 workload 上测试。

稳定模板放在随机 session ID、时间戳之前，有利于前缀共享；反过来，故意改变最前面的 token 可以构造 cold-cache 压测。但 benchmark 必须报告这种构造，不能与 warm-cache 结果混比。

## 8. Cache-aware routing：命中最多，不等于完成最快

把请求发往已有前缀的实例，可以节省重算或传输，但也可能加重热门实例队列。路由要比较：

$$T_{queue}+T_{lookup/fetch}+T_{uncached\ prefill}+T_{handoff}.$$

缓存命中率是手段，不是最终目标。应检查长队列、跨实例传输、KV 寿命与 goodput；同一个 session 永远绑定某实例，也不保证负载均衡。

## 9. KV 取回还是重算：用剩余延迟预算判断

简化传输时间：

$$T_{fetch}\approx T_{lookup}+T_{registration/queue}+\frac{M_{KV}}{BW_{eff}}+T_{layout}+T_{publish}.$$

与对应前缀重算的时间和资源占用比较，并扣除能隐藏的 overlap。传输更快未必更便宜；重算更快也可能挤占其他请求算力。

层次可以包括 HBM、本地 DRAM、NVMe 和远端池。离计算更远往往容量更大，但有效带宽、共享流量、NUMA 与命中延迟决定是否可用。

MLA/稀疏读取缩小传输量，会改变 offload 的可行性。Mooncake、LMCache、NIXL 等分别提供有关组件，但“传输完成”仍要与对象可读状态和生命周期衔接。关联 [[LCPU AI Infra Seminars - Session 06 - AI Communication Stack]]。

## 10. MLA、稀疏 attention 与系统连锁反应

### MLA：压缩状态，不是免费减少所有计算

MLA 保存低维 latent 及位置相关分量。推理可以重建 K/V 后计算，也可通过吸收投影在 latent 空间组织计算；选择取决于阶段和 kernel。更小 KV 同时改善容量、decode 访存和 PD 传输，但可能增加投影计算或布局复杂度。

### 录播介绍的演进路线

- NSA：结合块压缩、重要块选择与局部窗口，强调可训练且硬件友好。
- DSA：轻量 Lightning Indexer 打分、top-k、选中位置的稀疏主 attention；indexer 本身仍有长度相关成本。
- 更长上下文的 hybrid attention：继续引入压缩与局部/全局信息通路，避免“便宜但仍扫描全长”的部分重新主导。

官方模型卡描述 V4 的 CSA 与 HCA 组合，vLLM 官方实现说明另有局部滑动窗口路径。这里用它校正术语，不扩展抄录新版本细节。[官方模型卡](https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf)、[vLLM 实现说明](https://vllm-project.github.io/2026/04/24/deepseek-v4.html)。

### 系统代价

新增 indexer 意味着更多状态。恢复一个 prefix 时，需要其所需状态集合一致可用，不能只恢复 latent、忘掉 indexer 或 generation。它们可以采用不同物理管理策略，但逻辑状态必须一致；不应把课堂“一起命中、一起逐出”理解成任何引擎都必须逐字节同时搬运。

HiSparse 类方案把完整 KV 放在 host，HBM 只保留 hot/selected blocks；风险是选择变化、miss tail 与 thrashing。稀疏 attention 改变了模型的信息访问，不能等同于 FlashAttention 这种数学等价的 IO 优化。

## 11. CP、chunked prefill、PD 分离：不要混为一谈

| 手段 | 主要切分对象 | 主要目标 | 代价 |
|---|---|---|---|
| Context parallelism | 单请求的上下文/attention 工作 | 容量、长请求 TTFT | KV/partial 通信与负载均衡 |
| Chunked prefill | 时间上的 Prefill 调度粒度 | 缓解 Prefill 对 Decode 的干扰 | 单请求 Prefill 完成更晚、调度成本 |
| PD 分离 | 资源池与服务阶段 | 分别调 TTFT、TPOT、硬件与并行策略 | KV handoff、两侧排队、资源配比 |

Ring Attention 让 KV blocks 在设备间流动，以 online softmax 合并结果；causal 三角形可能使简单等长分块负载不均，需要更合适的划分。CP 没有消除 dense attention 的总算术。

录播常说 Decode 一般不开 CP，是其主要部署语境；DCP 确实存在，不能写成算法禁止。是否有收益取决于长 KV、batch、带宽与通信成本。

## 12. Continuous batching 与 scheduler

Static batching 在“攒齐一批”和“全批完成”形成等待。Continuous batching 把调度粒度降到迭代：完成者退出，满足资源预算的新请求进入。

线性层可以将不同请求的 token 拼成大矩阵乘；attention 仍必须保持请求边界和各自 KV，不能跨请求混合注意力。

新请求的长 Prefill 会干扰正在 Decode 的请求。Chunked prefill 常先照顾 Decode，再用剩余 token budget 插入 Prefill chunks，但具体优先策略是引擎可选设计。

Scheduler 同时管理：

- waiting/running 请求与 token budget；
- KV block 容量、引用和可回收集合；
- Preemption 时重算还是 swap；
- 用户/租户公平、优先级、deadline 与 admission control；
- Graph bucket 与 CPU/GPU 异步提交；
- 分布式状态发布、取消与错误回收。

不能为了更高 batch 永远延后低优先级请求，也不能让一个长请求耗尽所有资源。公平与 SLO 是调度的一部分，不是 kernel 之外可以忽略的指标。

## 13. PD 分离什么时候值

适合的信号：共置时 TTFT 与 TPOT 冲突明显；Prefill/Decode 希望使用不同并行或硬件配置；负载波动需要独立扩缩；互联能在预算内传 KV。

不适合的信号：请求很短、原本已满足 SLO、跨节点带宽差、KV 巨大或新增 handoff/queue 主导。

课堂用 400 Gb/s ≈ 50 GB/s 估算：搬 40 GB 的理想时间约 0.8 s。若严格使用前面的 40 GiB，则约为 0.86 s，且还未扣协议效率、竞争和 layout 转换。

可以逐层计算、逐层传输，以第 i 层的 transfer 重叠第 i+1 层 compute。但 one-sided RDMA 不等于没有资源影响：仍使用 HBM、PCIe/NIC、网络与完成协议。

两池采用不同 TP/CP/EP 时，KV shard/layout 可能不同，需要显式转换。传输库支持到什么程度须按版本确认，不能默认自动处理任意重排。

## 14. 并行方式：切分什么，就引入什么依赖

| 方式 | 切分对象 | 主要通信/代价 | 选择依据 |
|---|---|---|---|
| TP | 层内权重/张量维度 | 频繁 collective；实现可用不同组合 | 低延迟高带宽域、shape 与 heads 支持 |
| PP | 层序列 | Activation P2P、pipeline bubble | 容量、跨节点资源与可流水程度 |
| EP | 完整 experts | Token dispatch/combine、可变 All-to-All | 每 expert batch、路由偏斜、互联 |
| CP | Context / attention 工作 | KV 或 partial 交换 | 长上下文容量与 TTFT/Decode 带宽 |
| DP attention | 请求/attention 计算 | 与后续 EP 的衔接与各 rank 协调 | KV locality、局部 attention、全局 expert 利用 |

TP 能否按 KV heads 均匀切分是重要实现条件，但不能说 head 数不整除就绝对无法 TP：某些实现会复制 KV heads 或采用其他映射，代价需要重算。

TP 切 MoE 时，每卡可持有各 expert 的分片；EP 则让完整 expert 分布在不同卡。后者只在命中 expert 时读取其权重，但 dispatch、expert skew 和跨 rank tail 必须一起优化。

大 EP 的收益并非“卡越多越快”：它可能减少每卡权重驻留、释放 KV 容量并承载更大 batch；也可能因每 expert tokens 太少、路由不均与通信而变慢。

EPLB/冗余 expert 可以分摊热点，代价是权重移动、额外容量、路由更新与状态同步。重排是否必须暂停服务，是实现选择，不能把录播的一般情形写成绝对规则。

## 15. 算子、graph 和融合分别优化什么

| 技术 | 主攻问题 | 仍需关注 |
|---|---|---|
| FlashAttention | 避免显式大 attention 中间矩阵、优化 IO | 目标架构、shape、精度与总算术 |
| Split-KV | 小 Q 情况并行度不足 | Partial 输出与 softmax 统计的正确合并 |
| Grouped GEMM | 多 expert 小 GEMM 的 launch/调度开销 | Dynamic shape、layout、每 expert token 分布 |
| CUDA Graph | 多次 host launch/调度开销 | Capture/replay 约束、bucket padding、内存地址稳定 |
| DeepEP 类通信库 | EP 不规则通信和拓扑利用 | 版本、吞吐/延迟模式、SM/NIC 资源 |
| MegaMoE / 大融合 kernel | 算子边界与粗粒度等待 | Tile 依赖、remote state、寄存器/SMEM/功耗 |

Contiguous packing 让 experts 的 token 紧凑排列，减少空槽；masked fixed-capacity layout 较容易适配某些静态 graph 路径，但会浪费容量/计算。不能说 CUDA Graph 从原理上只支持 masked layout：这是讲座所讨论实现的取舍，其他 graph update 或静态上限设计可能不同。

CUDA Graph 通常减少 host 开销，不消除真实数据依赖，也不保证消除全部 launch 成本。MegaKernel 通过 tile-level 状态与 warp specialization 重组依赖，但资源压力和复杂度更高。

> [!important] RC 与 IBGDA 不是二选一协议
> RC 是 RDMA transport 类型，IBGDA 是 GPU 发起/控制相关机制。它们属于不同维度，GPU 发起可以使用可靠传输；不能把“CPU 发起=RC，GPU 发起=不是 RC”作为分类。

## 16. Speculative decoding：打破“一轮目标模型只产一个 token”

Draft 提出多个候选，target 批量验证。若目标模型原本受权重带宽限制，验证多个候选可以复用一次权重读取；但 verification 绝非总是免费。

设基线每 token 时间 $T_0$，每轮最终推进的 token 数为随机变量 A：

$$Speedup\approx\frac{\mathbb E[A]\,T_0}{T_{draft}+T_{verify}+T_{bookkeeping}}.$$

当 $T_{verify}\approx T_0$ 且忽略管理成本，才得到课堂的简化形式：$\mathbb E[A]/(1+T_{draft}/T_0)$。

应测 acceptance length，而不只测单个 token acceptance rate；树状分支、bonus token、拒绝位置和回滚都影响每轮真实推进量。

### “无损”的严格含义

标准 speculative sampling 可以通过正确的接受/拒绝与纠偏保持 target 的分布；不等于任何 draft 后接任意验证都无损，也不等于随机种子相同就逐 token 完全相同。Greedy 与 sampling 路径需要分别验证，数值差异也要考虑。[原始论文](https://arxiv.org/abs/2211.17192)。

### 讲座介绍的 draft 路线

- 独立小模型：简单直接，但有独立模型成本。
- Medusa、EAGLE、MTP：不同方式利用 target 信息或额外预测能力；MTP 可随主模型训练提供，EAGLE 类 draft 也需要匹配与训练。
- 链式与树式：候选拓扑影响覆盖率、验证规模、mask 和缓存管理。
- DFlash：轻量 block diffusion 并行起草，减少逐 token draft 的串行依赖；收益仍依赖接受长度与执行开销。[作者仓库](https://github.com/z-lab/dflash)。
- DSpark：以置信度调度的半自回归生成探索速度/接受质量的折中。论文结果不等于任何负载上的全局 Pareto 最优保证。[论文](https://arxiv.org/abs/2607.05147)、[官方 DeepSpec](https://github.com/deepseek-ai/DeepSpec)。

### 为什么大 batch 可能让 speculation 收益下降

大 batch 已经复用权重并占用更多计算；draft/verify 的额外 FLOPs 和 KV 状态更可能抢占目标模型资源。投机深度、候选数、并发与 batch 应联合调整，而不是把各自最大值同时打开。

Draft 和 target 有各自状态；拒绝候选时，未提交 KV 必须回滚或隔离，不能污染后续正式生成与 prefix cache。

## 17. 引擎栈与 RL rollout

讲座将系统拆成：上层 routing/编排与 KV 管理，中层引擎/scheduler，下层 attention、GEMM、通信库和目标硬件。vLLM、SGLang、TensorRT-LLM、Dynamo 等只是理解分层的例子，不构成固定功能排名。

RL rollout 的目标可能更接近 offline throughput/整轮完成时间，但仍有 straggler、权重更新、KV 失效与训练/推理数值一致性问题。这里的“一致性”是要明确概率、logprob、dtype 和 kernel 语义契约，不能不加条件地要求所有系统 bitwise 相同。

Agent 还有工具调用、环境与 sandbox，使“一个请求从头连续占 GPU 到结束”的假设失效。暂停、恢复、跨轮前缀与环境状态，都会进入系统设计。

## 18. 用什么顺序诊断一个慢请求

1. 定义问题：TTFT、TPOT/ITL、E2E、goodput 还是成本不达标？
2. 固定模型与流量：长度、到达过程、prefix hit、sampling、输出长度分布。
3. 看排队与路由：慢在等资源，还是执行本身？缓存亲和是否制造热点？
4. 量化状态：权重、KV、碎片、workspace、draft/indexer 状态占多少？
5. 观察关键路径：lookup/fetch、Prefill、handoff、Decode、collective、CPU gaps。
6. 找匹配手段：batch、chunk、PD、CP、kernel、graph、spec；每次先验证一个主要假设。
7. 检查反作用：长尾、拒绝率、内存压力、精度、通信与功耗是否变差？
8. 用真实 trace、固定 SLO 与可重复配置重新比较。

## 19. Benchmark 记录清单

- [ ] GPU/CPU、节点数、互联、软件版本/commit、模型与量化方式。
- [ ] TP/PP/EP/CP/DP、PD 配比、KV dtype/page size、memory budget。
- [ ] Input/output 分布、到达率与 burst、并发、prefix hit、warm/cold 状态。
- [ ] Graph buckets、padding、spec draft/深度、sampling 与输出终止条件。
- [ ] 预热与编译分开；每次测试缓存如何重置或保留有记录。
- [ ] TTFT/TPOT/ITL/E2E 的定义、分位数与 SLO；offered load、goodput、失败/拒绝率。
- [ ] 同时记录每卡吞吐与总集群吞吐，避免用更多设备“证明”软件更高效。
- [ ] Correctness 与质量评测独立完成；不把误差、少生成或丢请求误当提速。

## 20. 自测与跨讲连接

1. 为什么相同后缀不能无条件共享 KV？模型版本变化后呢？

    **面试回答：** KV 依赖完整前缀及其逐层表示，不只是当前 token ID；相同后缀配上不同前文，通常会产生不同 KV。共享应匹配前缀、位置、模型/权重版本、adapter 及其他相关配置；模型更新后旧 KV 一般应失效或按版本隔离。

2. 算出一种 GQA 模型每 token 的 KV，指出分页与分片会怎样改变实际占用。

    **面试回答：** 以 80 层、8 个 KV heads、head dim 128、BF16 为例，每 token 为 $2\times80\times8\times128\times2=327680$ bytes，即 320 KiB。分页把分配量上取整到整页并增加元数据；TP/CP 若均匀切分可减少每卡占用，但复制 KV heads、页尾和工作区会使实际值偏离简单除法。

3. 为什么 batch 增大会摊薄权重流量，却不一定继续摊薄每请求 KV 流量？

    **面试回答：** 同一层权重可被 batch 内多个请求共同使用，一次读取对应更多 token 的计算，因此每请求权重流量降低。但各请求通常有独立前缀和 KV，每加一个请求也增加一份历史 KV 读取；除有效共享前缀等情形外，这部分不会因 batch 增大自动摊薄。

4. 42 ms 的课堂估算为什么不能直接作为单张 H100 的部署性能？

    **面试回答：** 42 ms 来自约 140 GB BF16 权重除以 3.35 TB/s 的理想带宽，只是忽略其他开销的搬运估算。单张 H100 80GB 放不下这些权重，更别说 KV；实际需量化、分片或 offload，再按每设备流量、有效带宽和通信重算。

5. PagedAttention、prefix cache、offload 分别管理哪个问题？

    **面试回答：** PagedAttention 管理 KV 的分页分配和地址映射，减少连续分配与碎片问题；prefix cache 复用相同合法前缀的已算状态；offload 把状态放到 DRAM、NVMe 或远端，扩展容量。三者可组合，但分别要付出寻址、命中管理和取回延迟。

6. Cache-aware routing 为什么有时应放弃最高命中率？

    **面试回答：** 最高命中实例可能已经排长队，或取回/转换缓存的成本很高。路由应比较排队、KV 获取、未缓存部分 prefill 和 handoff 的总时间，并兼顾负载与 SLO；宁可多重算一点，也可能更早返回首 token。

7. CP、chunked prefill 与 PD 分离分别切分什么？

    **面试回答：** CP 把单请求的上下文及 attention 工作分到多设备，换取容量和并行；chunked prefill 把 prefill 在调度时间上切成小块，与 decode 交错；PD 分离把 prefill、decode 放入独立资源池。它们分别切空间工作、调度粒度和服务阶段，代价分别是通信、调度和 KV handoff。

8. 为什么 PD 分离可能使短请求变慢？

    **面试回答：** 短请求原有 prefill 和 decode 很快，分离后却额外增加跨池排队、KV 传输、布局转换和交接同步。若原本没有明显的阶段干扰，这些固定成本可能大于隔离收益；要在相同硬件预算和 SLO 下比较端到端延迟。

9. Attention DP + MoE EP 减少了什么，又增加了什么同步？

    **面试回答：** Attention DP 让每个 rank 处理自己的请求和 KV，可减少 attention 部分的 TP 通信并保持 KV locality；MoE EP 让完整 experts 分布在设备间。代价是 token dispatch/combine 的跨 rank 交换，以及动态 batch、负载和执行进度的协调；热点 expert 仍会拉长尾部。

10. CUDA Graph 与 MegaKernel 分别能消除哪些开销，不能消除哪些？

    **面试回答：** CUDA Graph 主要摊薄重复 host launch 和调度开销，通常不消除 kernel 的真实计算、访存和依赖。MegaKernel 将多个操作融合到设备内，减少 launch、部分中间写回及粗粒度等待，但增加寄存器和同步复杂度；两者都不能消除必要的数据传输和依赖。

11. “Speculation 无损”需要什么验证契约？接受率高为何仍可能变慢？

    **面试回答：** Greedy 路径需验证与目标 greedy 决策一致；sampling 路径需用正确接受/拒绝和纠偏保持目标分布，并正确处理 mask、终止和 KV 回滚。“无损”不代表同 seed 逐 token 一样。即便接受率高，draft、验证和管理的耗时若超过所节省的目标轮次，仍会变慢。

12. 在固定 SLO 下，你如何证明一次优化提升了 goodput，而不是只改善漂亮的平均数？

    **面试回答：** 固定模型、质量、硬件预算、输入输出分布、到达率及缓存条件，对比相同 TTFT/TPOT 等 SLO 下的有效完成量/秒。除 goodput 外还要看尾部分位数、拒绝/失败率及 offered load，反复测量；不能靠少生成、丢慢请求或只报平均延迟制造收益。


- [[LCPU AI Infra Seminars - Session 02.2 - FP32 GEMM Quick Walkthrough]]：分层复用与 Roofline。
- [[LCPU AI Infra Seminars - Workshop 01 - Parallel Programming with TileLang]]：布局、编译、pipeline 与 persistent。
- [[LCPU AI Infra Seminars - Session 03 - Tensor Core 从 mma.sync 到 tcgen05]]：计算单元与 operand 供给。
- [[LCPU AI Infra Seminars - Session 04 - Pipeline Ordering Data Orchestration]]：异步依赖、buffer 生命周期。
- [[LCPU AI Infra Seminars - Session 06 - AI Communication Stack]]：KV handoff、EP 与可消费语义。

最终复述：Serving 把前几讲的计算、存储与通信问题放进带 SLO 的在线系统；真正要优化的是正确请求完成的关键路径，而不是某一个孤立组件的峰值。
