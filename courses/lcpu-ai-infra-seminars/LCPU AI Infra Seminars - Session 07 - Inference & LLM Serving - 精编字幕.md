---
type: transcript
status: polished
course: Weiming HPC Training Camp × LCPU AI Infra Seminars
session: 7
speaker: 孔昊然
video_url: https://www.bilibili.com/video/BV1gY4d6GEwR/
source_status: transcript-reviewed-with-primary-source-checks
topics: [inference, kv-cache, serving, batching, moe, speculative-decoding]
---

# Session 07：Inference & LLM Serving｜精编字幕

> [!info] 整理说明
> 依据正确导出的完整 SRT（3602 条，至 03:25:11）整理，保留原讲述顺序、主要推导和现场补充，删除口头重复、会议控麦及无关停顿；不是逐字稿。时间为原字幕分钟级回看锚点。当前未取得可核实的本讲官方课件，因此不编造页码、图表和未出现的代码。
>
> 本讲大量结论有模型、硬件与 workload 前提。下文保留这些限定，并用单独说明区分整理者校正。详细公式与复习题见 [[LCPU AI Infra Seminars - Session 07 - Inference & LLM Serving]]。

## 术语校正

| 常见误识别 | 校正 |
|---|---|
| PREFL / profile / PREFAIL | Prefill |
| 抵扣 / decode 混写 | Decode |
| k b cash / 可以开始 | KV cache |
| RUFly / ROOFLINE | Roofline |
| 败吃 / 百事赛 | batch / batch size |
| VRM / SG 浪 | vLLM / SGLang |
| 配置 attention / PATTENTION | PagedAttention |
| redis tree | radix tree |
| 显示复制 | 写时复制，copy-on-write |
| 服从/谷歌吞吐等误识别 | goodput（按上下文） |
| contest / contact parallel | context parallelism，CP |
| sports attention | sparse attention |
| contain / continue batching | continuous batching |
| group jam / deep jam | grouped GEMM / DeepGEMM |
| flash info / flash m ra | FlashInfer / FlashMLA |
| 逮 NEMO / 尼克松 | Dynamo / NIXL |
| ego / D flash / D spark | EAGLE / DFlash / DSpark |
| ma carl / 买个 MOE | MegaKernel / MegaMoE |
| EPRB / LPLB | EPLB，expert parallel load balancing |

## [00:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=0) 从硬件、pipeline、通信，走到 workload

前面的 Tensor Core 讲座介绍了 A100、H100、B200 等架构与编程抽象；Pipeline Ordering 讨论如何组织单卡数据搬运与计算；通信讲座又把范围扩展到多 GPU 与 RDMA 集群。

但上一次对 workload 讲得比较快。今天将解释推理中的 KV cache、EP 通信及整套 serving 组织，看看这些硬件和系统机制究竟服务于什么需求。

## [03:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=180) 所有“最佳方案”都有约束条件

推理发展很快。今天会介绍一些经典工作，但原本限制一旦变化，结论可能也变化。如果可以重新设计硬件，而不是只能使用现有 GPU，系统就可能出现另一种解法。

本讲主要讨论自回归模型，从 KV cache 视角分析 Prefill 与 Decode；引入 speculative decoding 后，还会把 Decode 中的 draft 与 verify 分开看。

## [06:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=360) 一次请求的两个主要阶段

Tokenization 后，Prefill 处理 prompt，随后自回归 Decode 逐步生成。Prefill 一次有很多 tokens，线性层通常表现为 GEMM；普通 Decode 每请求每步只有一个新 token，小 batch 时更接近 GEMV。

这种 shape 差异造成权重复用和算术强度的差异，也让两个阶段对硬件与调度的需求不同。后面所有优化都围绕这些特点展开。

## [08:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=480) KV cache 的三个问题

为什么缓存合法？因为 causal mask 下，历史位置的表示不依赖后续追加 token；给定相同前缀与模型配置，历史 K/V 不会因新增 token 而改变。双向 encoder 等结构不能直接套用这条结论。

为什么存 K/V，不存历史 Q？新 token 的 Q 要读取历史 K/V，历史 Q 已完成本步查询，后续普通 Decode 不再使用它。

丢掉 KV 后怎么办？不能只凭 token ID 直接查回所有层的状态，通常需要重算，或从其他存储位置取回。重算与取回谁划算，是系统要分析的取舍。

## [12:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=720) Attention 的演进，也是在改变状态成本

从 MHA，到 MQA/GQA，再到 MLA、稀疏和线性 attention，各种设计改变了计算、KV 容量和信息保留方式。本讲主要以 DeepSeek 的开放实现为线索，但不据此断言某种架构对所有模型都最好。

模型效果、训练资源和系统成本之间存在取舍。KV cache 可以理解为自回归 Transformer 为后续计算保存的状态；不同模型保存的状态形式并不相同。

## [14:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=840) 算一遍 KV 大小

对常见 GQA，KV 字节数由层数、KV heads、head dimension、元素精度以及 token 数相乘得到，K 与 V 分别保存时还要乘 2。

课堂以 70B GQA 配置举例：80 层、8 个 KV heads、head dimension 128、BF16，约为每 token 320 KiB。原字幕把层数识别得很混乱，这里按能与公式一致的 80 层校正。

MLA 不再独立保存完整 K/V，而保存 latent 与位置分量，所以不能直接套同一个公式。精度、层数与状态维度改变后，缓存容量和搬运成本可以显著变化。

## [17:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=1020) 简化计算量：每 token 约两倍参数量

先忽略长上下文 attention 的额外计算，可以把每 token 的主要线性层计算量近似成 `2P FLOPs`。Decode 为产生一个 token，却可能需要读取大量权重和历史 KV，因此小 batch 时算术强度很低。

Roofline 说明：算力再高，如果每搬一个 byte 只做很少运算，性能仍受带宽限制。不同硬件的峰值算力与带宽比不同，同样 workload 的瓶颈也可能不同。

## [19:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=1140) 140 GB 权重与 42 ms 的课堂估算

70B 参数、每参数 2 bytes，大约是 140 GB。若按 3.35 TB/s 的理想带宽读一次，就要约 42 ms；若这一轮只生成一个 token，约相当于 24 tokens/s。

讲者用这个非常简化的算术强调：只优化少量计算，很难越过权重读取下界。可以尝试提高有效带宽、压缩权重、增大 batch 或使用投机解码。

> [!important] 整理者校正
> 这不是“单张 H100 80GB 可直接装下 140GB 权重”的部署测量。容量、分片、聚合带宽、通信和 KV 都被暂时忽略。真实系统必须按实际部署重算该下界。

## [22:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=1320) Batch 复用权重，却不能无限提高计算强度

如果同时处理 B 个请求，一次权重读取可以服务 B 个新 tokens，而各请求的 KV 仍然不同。这让增大 Decode batch 成为重要优化。

但 batch 越大，KV 总流量与容量也越大；当 KV 主导时，收益不会继续按 B 线性增长。Attention 状态压缩、减少显存浪费和更好的缓存布局，会间接帮助承载更大 batch。

## [24:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=1440) 两个阶段为什么可能值得分开

Prefill 通常有更高计算强度，关注合适并行与高性能 attention；Decode 则常关注 batch、KV/权重量化、带宽和 speculative decoding。

这些性质差异让 PD 分离成为自然方向。但它是一个带条件的系统选择，后面还要分析 KV 传输、队列和资源池配比，不能此时就认定分离必然更好。

MFU 适合观察有效计算相对峰值算力，memory-bound 场景还需看带宽利用等指标。低 MFU 不一定意味着 kernel 写得差，也可能是算术强度本来不高。

## [27:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=1620) 吞吐、延迟与成本

用户感受到的是请求延迟和流式输出是否顺畅，服务提供者还要考虑单位资源能产出多少有效 tokens。Batch 有利于吞吐，却可能增加等待或单步延迟。

Online serving 需要在延迟约束下优化吞吐；offline 的约束较宽，可以采用不同策略。满足 SLO 的有效吞吐才是这里关心的 goodput，而不是不计超时地追求最大 tokens/s。

## [30:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=1800) Agent/coding 的前缀复用会改变 Prefill

多轮 agent 和 coding 请求可能复用大量历史前缀。如果大部分 prompt 的 KV 已命中，Prefill 只计算较短的新后缀，剩下的关键工作可能是缓存查找、取回与调度。

因此“Prefill 必然 compute-bound”会失效。讲者引用了若干线上命中率观察来说明趋势；这些比例与业务、实现和统计口径有关，不应当成所有 agent workload 的常数。

## [33:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=1980) TTFT、TPOT、ITL 与 E2E

TTFT 是到首 token 的时间，包含路由、排队、Prefill，以及必要的 KV 获取。TPOT 常表示首 token 后的平均每 token 时间，主要反映 Decode 阶段。E2E latency 则看整个请求何时完成。

现场补充，TPOT 的平均值与 ITL 分布应区分。投机解码可能一次输出一段 tokens，更要说明观测的是 token 间隔还是流式 chunk 间隔，不能把平均数当成用户没有感受到停顿。

生产环境还要报告分位数，区分 input/output、缓存命中/未命中与请求类型。

## [37:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=2220) 用真实请求分布验证引擎

聊天、长文总结、agent coding 的输入输出长度、到达过程、工具调用与缓存复用不同。为了证明优化有效，需要真实 trace 或有代表性的压测，而不是只看一个固定 shape。

可以使用引擎提供的 benchmark 工具，也可以记录生产 trace 再 replay。关键是 workload 是否对应真正要服务的场景。

## [40:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=2400) Benchmark 的常见陷阱

Cold start、JIT、graph capture 与第一次初始化不能无说明地混进稳态测量；缓存预热或污染也会改变结果。只报告吞吐或平均延迟，可能隐藏严重长尾。

另外，某个引擎“很慢”可能只是配置不合适，而不是架构差。比较前要确认模型、并行、量化、batch、内存预算与版本，记录完整测试条件。

## [43:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=2580) 接下来从四个角度看 KV

KV 怎样分配与保存？怎样跨请求复用？怎样在多级存储间流动？模型架构又怎样减少状态大小？

这些问题分别对应分页、prefix caching、tiering/offload 和 attention 设计。它们相关，但不能互相替代。

## [44:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=2640) PagedAttention：不要为每个请求预留最大长度

早期按最大序列长度预留空间，会产生很大浪费。显存装满了未使用的预留区，batch 就开不大，Decode 权重复用也上不去。

PagedAttention 借鉴虚拟内存分页，让逻辑 KV blocks 映射到物理 block pool，只按需要增长，减少碎片。Kernel 则通过 block table 找到物理位置，必须显式支持这种布局。

## [46:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=2760) Page size 与 copy-on-write

大 page 可能更利于批量搬运、地址组织和部分硬件路径，小 page 则减少尾部浪费。某些 kernel 对 page size 有支持限制，最终要同时考虑布局、对齐、搬运与碎片。

共享前缀还可以结合 copy-on-write：没有修改时共享，确需写入时再复制。它减少不必要拷贝，但也引入首次写入成本、引用管理和同步责任。

## [48:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=2880) Prefix cache：复用完全一致的前缀状态

给定相同模型与相关配置，完全一致的 token 前缀可以复用对应 KV。实际管理常按 block 粒度，而不是任意单 token 随意共享。

课堂比较 vLLM 的 block-hash 思想和 SGLang 的 radix-tree 思想。它们是理解历史设计的入口；引擎演进很快，具体 workload 谁表现更好应直接测量。

缓存还需要 eviction 策略，但活跃请求正在依赖的状态不能直接回收。

## [50:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=3000) 一个实用细节：随机前缀会破坏后面的共享机会

若把随机 session ID 或时间戳放在最前面，前缀一开始就不同，后面相同模板也很难作为连续前缀命中。稳定模板放在前面，通常更利于共享。

反过来，这也能用于构造 cold requests，但压测必须说明缓存条件，不能拿冷缓存与热缓存结果直接比较。

## [51:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=3060) 多实例以后，缓存命中成为路由问题

一个大集群通常有多个 serving 实例。KV 若只在某个实例的 HBM 中，请求要路由到那里，才能直接命中。

Session/cache-aware routing 能提高复用，却可能制造负载不均。最大化命中率不等于最小化请求完成时间：还要看排队、取回、重算与不同实例的负载。

## [53:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=3180) HBM 太贵，KV 可以进入更深的层次

KV 可以位于 HBM、host DRAM、本地 NVMe 或远端存储池，也可能有新的中间层。容量变大通常伴随不同带宽和延迟。

需要比较取回与重算：搬回来是否更快？是否能满足剩余 TTFT 预算？重算是否占用了本可服务其他请求的算力？答案依模型 KV 大小、硬件有效带宽和负载而变。

## [55:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=3300) 缩小 KV 与重叠传输，扩大 offload 的可行空间

KV 越小，传输通常越容易划算；如果能与计算流水，暴露在关键路径上的时间还可减少。

Mooncake、LMCache、NIXL 等组件在不同层面帮助管理或传输 KV，屏蔽部分底层介质与网络差异。但 metadata、完成通知与生命周期依然存在，只是由库承担了部分实现。

## [57:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=3420) MLA：不再独立保存完整 K 和 V

MLA 保存降维后的 latent，再加位置相关分量。推理时可重建完整 K/V 按多头方式算，也可通过投影组织直接在 latent 空间计算。

这在 Decode 的访存受限场景有潜在收益，但也需要相匹配的高性能 kernel。不能只看缓存缩小，还要看额外计算、布局与硬件利用率。

FlashMLA 等实现体现了模型状态设计与 kernel 协同的重要性。具体精度、支持 shape 和版本要单独检查，不把课堂例子当作所有实现的固定要求。

## [60:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=3600) 第一轮 KV 小结

分页减少碎片，prefix caching 减少重复计算，routing 决定能否利用已有状态，分层存储扩大容量，MLA 等架构减少状态本身。

但常数优化没有自动改变 dense attention 随上下文长度增长的算法阶。长上下文仍需要单独分析。

## [65:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=3900) 长上下文使 `2P FLOPs/token` 不再够用

前面的粗估省略了 attention 中与历史长度有关的 QK、AV。对标准结构，单步 Decode 的相关计算约随 `4×层数×长度×隐藏维度` 增长；长到一定程度，它就不能忽略。

Prefill 的 dense attention 总计算随长度呈平方级增长；Decode 还需要读取越来越长的 KV。上下文扩大时，容量与延迟可能同时越过可接受范围。

> [!note] 公式口径
> 课堂提到的约 53K 交叉点是特定参数下的简化比较。笔记区分了单步 Decode 和完整 causal Prefill 的计数，不将它泛化为所有模型的统一阈值。

## [68:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=4080) Context parallelism 与 Ring Attention

长请求可能一张卡放不下，也可能 TTFT 太长。CP 把上下文相关工作分给多个设备；Ring Attention 让 KV blocks 在 GPU 间流动，计算当前块时接收下一块，再用 online softmax 等方式合并结果。

Causal attention 的三角形工作量会导致简单划分不均，头尾配对等策略可改善平衡，但更复杂的划分也增加实现成本。

CP 解决单请求容量与并行问题；chunked prefill 则解决调度干扰，两者不是同一个技术。

## [71:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=4260) 从引擎并行回到模型结构

长上下文 Prefill 可以使用 CP，并与 PD 分离配合。Decode 的并行策略则按其瓶颈另选；课堂常见例子不启用 CP，但后文也明确存在 Decode CP。

模型侧还可以采用 sparse attention。讲者强调，训练参与的稀疏化和训练外直接删减历史信息，效果风险不同；不能仅凭某些 benchmark 就宣称所有任务无损。

## [73:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=4380) NSA：可训练、块粒度、硬件友好的稀疏

课堂介绍 NSA 的组合思路：压缩相邻 token blocks，选择重要块，并用滑动窗口保留局部细节。块粒度组织更容易对应硬件访问。

核心不是任意少读一些数据，而是让模型训练适应信息选择方式，并让选择与实际 kernel 相匹配。

## [75:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=4500) DSA：轻量 indexer 加稀疏主 attention

DSA 增加 Lightning Indexer，用较低成本给历史位置打分，top-k 后再执行主 attention。这样把昂贵计算集中在选中位置。

但 indexer 自身仍有随长度增长的工作。当上下文进一步扩大，原本很小的常数项也可能成为主导。讲者随后用 hybrid/compressed attention 说明继续压缩和多通路设计的动机。

## [78:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=4680) 新 attention 意味着新的状态管理

除了 latent KV，DSA 还需要 indexer 相关缓存。Prefix 的有效命中必须恢复所需的整套一致状态，不能只命中其中一部分就当作请求已准备好。

稀疏访问也改变 offload：主 attention 可能只需取回选中的 blocks，于是完整 KV 可以放在更低层，HBM 只保留 hot blocks。如何让 miss 不进入关键路径，是 HiSparse 一类工作的工程难点。

Top-k 本身也有算法和 kernel 成本；跨层选择相似等观察只能在相应模型与场景中验证，不能无条件利用。

## [81:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=4860) Attention 与 serving 必须一起演进

状态大小、层次、读取位置与算子结构变化后，KV 管理、传输、prefetch 和 kernel 都要调整。不能只替换数学公式，期待旧引擎自动获得全部收益。

Decode 增强权重复用仍是重要方向，但 batch 怎样组成，又回到 scheduler 的问题。

## [83:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=4980) 从 static batching 到 continuous batching

Static batching 等一批请求一起开始，又等整批结束。输入输出长度不一致时，短请求已经结束，资源仍可能被整批同步拖住。

Continuous batching 在迭代边界让完成请求退出、新请求加入，前提是 token 和 KV 预算允许。线性层可以拼接 tokens 形成大 GEMM；attention 仍要保持每个请求的独立状态。

## [86:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=5160) 新请求的 Prefill 会干扰正在 Decode 的请求

如果突然加入一个超长 Prefill，大 GEMM 可能占据计算资源，正在生成的请求出现明显 ITL 抖动或违反 TPOT 要求。

Chunked prefill 把长 prompt 的工作拆成 chunks，按预算逐步插入，在保护 Decode 的同时推进新请求。代价是长请求的 Prefill 完成可能更晚，TTFT 与 Decode 平滑性需要权衡。

## [89:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=5340) Scheduler 管的不只是一个 waiting queue

系统要管理 waiting 与 running 集合、每步 token budget、KV block pool，以及请求增长后的容量。Block 不够时必须 preempt 或释放资源：是丢弃后重算，还是 swap 到 host/更深层？

分布式、多级 KV 还要处理对象状态的一致性、引用、完成发布与回收。对用户来说只关心 SLO，框架却必须在后台把这些状态管正确。

## [92:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=5520) 公平、优先级与 admission control

不能让一个长请求长期占满所有预算，也不能为了短期吞吐让低优先级请求永久饥饿。对预计无法满足 SLO 的请求，需要按产品策略拒绝、降级或调度到合适资源。

这些不是附属问题，而是在线系统真正能否提供服务的一部分。

## [93:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=5580) CUDA Graph 与异步调度

Decode 有许多执行很短的 kernels，逐个由 CPU launch 的开销可能显著。预先捕获 graph 后重放，可以降低部分 host 开销。

常见实现只准备若干 shape buckets，没有恰好命中时要 padding，浪费部分计算与内存。因此也要比较 graph 带来的收益与 bucket 代价。

另一条路线是融合 kernel；CPU 调度与 GPU 执行也可以异步重叠。所有手段都要检查是否真的减少关键路径，而不是只是多开一个线程或 stream。

## [96:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=5760) 调度这一章的小结

Continuous batching 改善整批等待，chunked prefill 缓和阶段干扰，token/KV 预算控制容量，swap/recompute 管理被抢占状态，graph/异步调度减少 host gaps。

这些工作没有完全消除 Prefill 与 Decode 的需求差异，因此下一步考虑资源池分离。

## [99:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=5940) PD 分离：把两种压力交给不同资源池

Prefill 与 Decode 可以使用不同硬件、并行方式和调度策略。Prefill 常关注 TTFT 与长请求处理，Decode 常关注 batch、带宽和输出间隔。

分离让两侧可以分别调参和扩缩，但增加了显式 KV handoff。甚至采用异构硬件，也只是扩大设计空间，不自动保证工程兼容与收益。

## [103:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=6180) 先算 KV transfer，别只画两个池子

以 400 Gb/s 链路的理想 50 GB/s 算，传 40 GB KV 至少约 0.8 s，还假设链路独占、有效带宽打满。这样的成本是否能接受，要看请求的 TTFT 预算。

KV 更小的模型更容易受益。模型状态设计与网络成本直接相关；如果互联不够好，PD 分离可能从根本上不划算。

讲者还举自定义硬件作为反例方向：如果可以重新设计计算与带宽比例，共置也可能成为合理方案。具体公司与芯片性能未在本稿独立核验，仅保留这个设计启示。

## [106:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=6360) 逐层传输与传输库

Prefill 完成第 i 层后，可以开始传该层 KV，同时计算第 i+1 层，尝试隐藏传输。非阻塞、one-sided 操作可减少某些对端软件参与。

NIXL 等传输组件通过注册的内存与描述信息组织寻址，选择受支持后端，屏蔽部分硬件差异。它们不意味着通信不占 HBM、PCIe 或网络，也不消除完成与可见性要求。

## [108:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=6480) 不同并行策略意味着 KV layout 可能不同

Prefill 可能使用 CP，Decode 使用不同 TP/EP 配置；KV 在内存中的分片和排列就可能不同。迁移时需要处理 layout/shard 转换。

转换可发生在不同阶段，某些库能在特定受支持路径中融合处理，但不能默认任意布局都能自动重排。

## [109:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=6540) PD 比例、缓存与两次排队

P/D 的数量应按实际请求分布、命中率和 SLO 调整。动态转换实例角色还涉及其持有的 KV 和已接收请求，不能简单把它看成开关。

短请求尤其可能因两次调度、传输和 handoff 变慢。全局 Prefill 队列要同时考虑负载均衡、cache-aware routing 与取回/重算成本；Decode 侧还有自己的 admission 与 batch 组织。

## [112:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=6720) Mooncake 与 KV-centric 视角

Mooncake 等工作将 KV 作为系统中心对象：命中前缀时只算增量，在需要的阶段取回状态，并让传输层处理具体路径。

PD 分离适合 TTFT/TPOT 冲突显著、互联足够、两侧确实需要独立调节的场景。短请求、较差互联、原本已满足需求时，共置或 chunked prefill 可能更合适。

## [115:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=6900) 所有加速倍数都要回到同一 SLO

论文宣传的吞吐提升可能建立在某种特殊长度分布、并发和 SLO 上。比较 serving 策略，应固定有效吞吐口径，再看是否对自己的流量成立。

PD 分离与 chunked prefill 是对相关问题的不同解法，不是前者出现后后者就没有价值。

## [117:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=7020) 推理并行：没有 backward，却多了长寿命 KV

推理也使用 TP、PP、EP、CP 等划分，但没有训练时的 backward 与大梯度归约，同时要维护跨步 KV，并满足更敏感的在线延迟。

TP 常引入 collective，PP 需要 activation P2P，EP 需要按专家路由的 dispatch/combine，CP 则交换上下文相关状态。第 6 讲的通信原语在这里对应具体 workload。

## [119:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=7140) 并行要贴合互联域

节点内专用互联和跨节点 RDMA 的有效带宽、延迟和故障行为不同。频繁、延迟敏感的 TP 通常更适合放在高速域内，PP 则有机会流水，但能否隐藏通信仍要算。

现场预告了大 NVLink 域上的 EP 实践：更大的 scale-up 域与 C2C 会改变传统假设，也需要针对 CPU/GPU 和拓扑重新优化。峰值带宽比只是起点，不代表真实 collective 的速度比。

## [123:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=7380) TP 与 AllReduce

TP 的通信可通过 AllReduce，或等价阶段组合组织；不同消息大小、硬件和低延迟/高吞吐目标会选择不同实现。

通信能否与计算重叠、归约发生在 GPU 还是网络单元，以及引擎是否使用 custom collective，都会影响实际延迟。不能只知道一个原语名称就推断成本。

## [125:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=7500) PP：直观，但要填 bubble

PP 把层切成阶段，每个阶段持有自己的权重与 KV，阶段间传 activation。代价是流水 bubble 和额外依赖。

是否使用 PP，与容量、节点间通信和请求组织有关。讲者描述了其常见部署判断，但并不是说 PP 在推理中一概无用。

## [127:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=7620) EP 与 TP 切 MoE 的区别

TP 可以把每个 expert 的权重都切开，每卡持有各 expert 的一部分；EP 则把完整 experts 分布到不同卡，把 token 路由到对应位置，再 combine 回来。

MoE 的路由是稀疏的，但被激活 expert 内部依然是 dense GEMM。真实通信常是不均匀的 All-to-All-v，而非每对设备都发送相同大小。

## [129:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=7740) 为什么 Decode 可能使用很大的 EP

讲者以 DeepSeek 报告的部署方案说明：把 experts 分布开，可以降低每卡权重驻留，给 KV 腾出空间，并用更大的整体 batch 提高每个 expert 的复用。

这也带来热点问题。某个 expert 同时接收更多 token、执行更多计算，可能成为整个系统的 straggler。大 EP 的收益依赖 token 分配和通信实现，不只是设备数量。

## [131:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=7860) 冗余 experts 与 EPLB

热点 expert 可复制到多张卡，再把命中 token 分流；也可以周期性重新放置 experts。它们降低负载不均，但付出权重迁移、容量、路由更新和同步成本。

某些实现重排时可能影响服务，需要调度窗口；能否在线不停服，取决于更细致的版本和状态管理。

## [132:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=7920) Attention DP + MoE EP

Attention 保持请求及 KV 的局部性，各 rank 计算自己请求的 attention；进入 MoE 层时，再通过 EP 组织 tokens 与 experts。

这样减少某些 KV 复制或通信，但各 DP ranks 的 batch 与 EP 阶段仍要协调。负载不均不会因换一个并行名字而消失。

CP 与其他方式可以组合，Decode CP 也存在；不同层之间的布局转换要由实现正确衔接。

## [137:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=8220) 并行策略最终由哪些条件决定

模型结构、KV heads、状态大小、请求长度与到达分布、缓存命中、SLO、scale-up 域大小和跨节点互联，共同缩小可行组合。

确定大方向后，内部还有大量 shape、batch、chunk、placement 和 kernel 参数可调。因此优化空间很大，但不能脱离前提盲目枚举。

## [141:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=8460) Kernel 不只有计算，也包括通信

推理需要打满 Tensor Core 的计算 kernel，也需要高效访问 HBM 和互联的通信 kernel。Memory hierarchy、TMA、warp specialization、persistent 和细粒度依赖，都是前面课程在推理场景中的应用。

硬件特性不是越多越好，而是必须匹配实际 shape、布局与数据流。只有使用了某个 API，不能证明已经接近硬件上限。

## [144:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=8640) FlashAttention：数学等价的 IO 优化

朴素 attention 显式写出 QK 中间矩阵，再做 softmax 和 AV，会产生大量 HBM 读写。Online softmax 允许按块更新统计量与输出，避免完整中间矩阵落到 HBM。

不同代实现进一步调整并行与 pipeline，以适配目标架构。它首先是在保持 attention 语义的前提下优化存储访问，不能与改变模型访问模式的 sparse attention 混为一谈。

## [146:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=8760) Split-KV 与专用 attention 库

Decode 的 Q 很短，沿 Q 维可能没有足够并行度。可以切分 KV，让多个工作单元处理不同历史段，再正确合并输出和 softmax 统计。

这增加了并行度，也增加 partial state 与合并成本。FlashInfer 等库为 paged KV、不同精度和 shape 提供多种路径；应先查现成支持，再决定是否自写。

## [148:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=8880) 模型专用 kernel 的收益与假设

FlashMLA、DeepGEMM 中的相关实现，体现了为特定 attention、精度和硬件做深度优化。若改变 KV dtype、shape 或架构假设，就可能需要重新适配，不能保证性能保持。

DSA 的 indexer、top-k、稀疏主 attention 也形成一条新的 kernel 链；模型演进会直接改变实现。

## [150:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=9000) Grouped GEMM 与两类 token layout

单卡多个 experts 若各自 launch 一个小 GEMM，会产生较大开销。Grouped GEMM 在一个 kernel 中调度多个矩阵任务，常结合 persistent tile workers。

紧凑的 contiguous packing 减少空槽，适合某些高吞吐路径；固定容量的 masked layout 更容易满足某些静态 graph 路径，但会有容量和计算浪费。

它们是具体实现的权衡，不应推广成“所有 CUDA Graph 都只能使用 masked layout”。

## [152:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=9120) DeepGEMM、低精度与硬件共同设计

DeepGEMM 包含不同精度的 GEMM、grouped GEMM 及模型相关算子，是观察 JIT 特化与硬件优化的学习入口。

FP8/更低精度不只是把 dtype 改小。Scaling 的粒度、累加方式、数据排列和目标 Tensor Core 支持，都影响精度与性能。相关细节可回看 Tensor Core 讲座。

## [154:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=9240) DeepEP：吞吐模式与低延迟模式

本段主要介绍 V1：一种路径先利用节点内高速互联整理、去重或转发，再走 RDMA，降低昂贵跨节点流量；低延迟路径则更重视直接发起和较短关键路径。

讲者提醒 V2 与 V1 已有较大变化，应阅读对应版本，不把两代后端、buffer 与 SM 使用策略混在一起。

> [!important] 分类校正
> 录播用传统 host 路径与 IBGDA 对照说明 GPU 发起。严格说，RC 是 transport 类型，IBGDA 是发起/控制机制，不属于互斥分类；GPU 发起不意味着不再使用可靠传输。

## [157:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=9420) Graph 减少 launch，却不消灭真实依赖

把多次 launch 捕获成 graph，可以减少 host 提交负担；常用路径需要管理地址、shape 与 buckets。Replay 后 kernel 之间仍有正确性依赖。

如果希望 tile 一 ready 就被下游消费，而不是等待整个 tensor 或 kernel 边界，就要进一步设计更细的生产消费协议。

## [159:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=9540) MegaMoE：把通信和计算融合到 tile 级

讲者展示将 dispatch、两次 GEMM、中间激活处理和 combine 放在同一执行系统中的思路。Symmetric memory 为远端访问提供约定的布局和映射，warp specialization 则分工处理通信、MMA 与 epilogue。

这些机制让已 ready 的 tiles 更早推进，但要求提前分配与管理状态、buffer、依赖和远端访问语义。它不是“取消所有同步”，而是把同步从粗边界改成细粒度协议。

## [162:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=9720) 融合 kernel 的适用边界

课堂案例有特定硬件与精度前提，迁移到其他设备需要重新适配和测量。深度融合还可能遇到寄存器、SMEM、功耗与热限制。

对于 launch-bound、memory-bound 工作，融合有较大组织空间；若计算单元已经是主要瓶颈，单纯进一步融合未必提高吞吐，要寻找其他关键路径开销。

## [165:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=9900) 算子部分小结

Attention kernel 优化 IO 和并行度；grouped GEMM 合并 expert 任务；EP 库理解不规则通信；CUDA Graph 减少 host 开销；MegaKernel 用 tile-level 编排减少粗粒度等待。

这些优化仍隐含一个重要前提：目标模型每轮通常只为每个请求推进一个 token。下面从另一个维度改变这个前提。

## [168:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=10080) Speculative decoding：一轮验证多个候选

Draft 先生成若干候选，target 一次验证它们。如果 target 原本 memory-bound，多做一些计算未必等比例增加时间，因为权重读取可以被多个候选复用。

所谓“verify 几乎免费”，只能在特定资源余量下近似成立。并发大、上下文长或验证规模过大时，额外计算和 KV 状态都可能进入关键路径。

## [170:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=10200) 用推进长度除以整轮成本

收益取决于每轮最终接受并推进多少 tokens，以及 draft 和 verify 花多少时间。课堂把它简化成：平均推进长度除以 `1 + 相对 draft 开销`，前提是 verify 近似一次普通 target step。

更准确的评估还要包含验证增长、回滚、缓存管理与调度成本。接受率高是有利条件，但不等于最终一定加速。

> [!note] “无损”的边界
> 标准 speculative sampling 通过正确的接受/拒绝和纠偏可以保持 target 分布；不是任意草稿加验证都自动无损。录播未展开证明，笔记附了原始论文入口，并区分分布等价与逐次随机输出相同。

## [171:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=10260) Draft 从小模型到 EAGLE 与 MTP

早期可用独立小模型起草；随后有额外 heads、利用 target 信息的方案，以及模型训练自带的 MTP。

MTP 让开放模型可以直接携带一个起草能力；EAGLE 类方法则可以针对目标模型和负载训练。Draft 越匹配真实任务，越可能获得较好的接受长度，但训练也有成本。

链式还是树式候选，是引擎配置与算法设计的一部分。树增加覆盖，也增加验证、mask 和状态管理，不能仅凭分支更多判断收益。

## [174:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=10440) DFlash：让 draft 本身也并行

自回归 draft 仍需要逐步产生候选，形成串行成本。DFlash 的 block diffusion 思路，是并行预测一个候选块，再交给 target 验证。

它打开了并行起草方向，但候选质量差时仍可能成为纯开销。Draft 与 target 的 KV 状态也需要分别管理，拒绝的候选不能污染正式上下文。

## [177:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=10620) 大 batch 与 speculation 可能竞争同一份算力

小 batch 时 target 留有较多计算余量，draft/verify 更容易利用；大 batch 已提高算术强度，再加入大量验证工作，就可能挤占有效吞吐。

因此 batch、draft 深度、候选数量与接受质量之间存在折中。课堂介绍 DSpark 作为探索这一速度—质量平衡的方案，但不应把“Pareto”宣传理解成任意系统都能无代价达到全局最优。

## [179:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=10740) Serving 中到处都有 scheduler

请求要分配给多实例，实例又分 P/D；Decode 内还有 draft/verify；每种模型有不同 KV 状态，状态又位于不同存储层。

每层都在决定何时执行、数据放哪、何时可读与何时回收。活跃请求依赖的块不能被释放，不同 generation 不能混淆，多个请求不能错误写入同一地址。

这就是为什么一个看似简单的“生成下一个 token”，会需要复杂系统。

## [183:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=10980) 用分层理解引擎，而不是找永远独占的功能

讲者回顾 vLLM 的分页路线、SGLang 的 radix 前缀管理、TensorRT-LLM 的硬件相关优化，以及 Dynamo 这类更上层编排组件。

功能会迅速互相吸收，不能永久地用某个早期卖点划分引擎。真正差异可能是对具体模型、场景、硬件的投入和工程成熟度；需要同条件测试，而不是根据名称选赢家。

## [188:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=11280) 从算子库到编排层，形成完整方案

引擎下面有 attention、GEMM、通信库与 custom kernels，上面有路由、资源池与 KV 管理。每层都可以替换或针对负载调整。

演进主线是更细的调度、更精确的资源管理，以及模型与硬件共同设计。单实例优化之外，跨实例、缓存路由和部署策略还有大量空间。

## [191:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=11460) 模型效果相近时，serving 成本变得重要

当模型能力差距很大，用户可能优先选择更强模型；当任务效果相近，成本和速度就更影响选择。

Serving 工程的目标，是在既定模型上，把前面所有复杂性组织成满足 SLO 的低成本、高 goodput 方案。模型设计时提前考虑推理，会减少后续工程的困难。

## [192:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=11520) RL rollout 与 agent 带来新的约束

RL rollout 包含推理，但目标可能更接近 offline throughput 与整轮完成时间，同时要处理 stragglers、权重更新和训练/推理切换。

Agent 又引入工具、环境和 sandbox；请求会暂停、恢复，状态不只在 GPU 上。相关基础设施会成为推理系统的一部分。

训练与 rollout 的 kernel、dtype 和数值行为不同，还可能引入 train–inference mismatch。需要定义和检查一致性契约，不能只因训练指标变好，就假定实际 rollout 完全对应。

## [195:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=11700) 总结：同一套约束下，联合优化多个层次

提高 Decode 计算强度，可以增加 batch、改善缓存容量和引入 speculation；减少 KV 和碎片，可以承载更多请求；利用存储层次，可以扩大状态保留空间；graph 与融合可以减少控制开销；PD 与并行策略可以分别满足阶段需求。

但这些手段不是简单相加。更大 batch 会改变 speculation 收益，更复杂 attention 会改变缓存与传输，更多并行会引入通信，更深融合会增加资源压力。

## [200:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=12000) 收尾补充：别把传统 GPU 的约束当成自然定律

现场没有展开独立的长问答，讲者继续强调：PD 分离等设计是在一组假设下成立的。如果可以自定义计算/带宽比例，甚至重做硬件与引擎，可能得到另一种合理方案。

本稿不将现场展示的公司专有芯片性能作为已验证事实；保留的结论是，系统最优方案必须连同约束一起理解。

## [202:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=12120) 特定硬件优化与宣传倍数的警惕

讲者提到针对 SM12x 等硬件的推理优化分享，说明除了模型和部署设计，具体架构仍有大量实现空间。相关材料可作为进一步阅读，而不能直接移植其性能结论。

最后再次提醒：很大的加速倍数可能来自特定甚至挑选过的场景。应在自己的 workload、SLO 和完整配置下验证，分清是在延迟、吞吐、设备成本还是质量上做了交换。

## [205:00](https://www.bilibili.com/video/BV1gY4d6GEwR/?t=12300) 结束

课程鼓励通过作业与源码阅读动手验证。整讲不是一套固定配置答案，而是一组分析维度：给定模型、流量和硬件后，沿状态与关键路径选择合适实现。

## 原始来源留档

- [视频](https://www.bilibili.com/video/BV1gY4d6GEwR/)，03:25:12。
- 有效文件：`北京大学未名超算队 × LCPU AI Infra Seminars 系列讲座：7-Inference & LLM Serving_哔哩哔哩_bilibili_BV1gY4d6GEwR_字幕.srt`，保留于 Downloads，未改写。
- 有效 SRT SHA-256：`1155361fa657812777f26dbe8ed57afe02f0b1f4f98e9b0cf98aa67282adcb33`。
- 名称含 `_P字幕.srt` 的首次文件实际是第 6 讲副本，未用作本稿来源。
