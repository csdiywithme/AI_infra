---
type: topic-note
status: evergreen
course: HPCGame 2026 赛前讲座
area: inference-systems
topics:
  - "[[LLM Inference]]"
  - "[[KV Cache]]"
  - "[[Disaggregated Inference]]"
---

# HPCGame 2026：推理状态与调度

> [!nav] 返回与扩散
> [[HPCGame 2026 - 00 MLSys 知识地图|知识地图]] · [[HPCGame 2026 - 02 稀疏模型与数值格式|模型状态]] · [[HPCGame 2026 - 04 通信与内存层级|通信/存储]] · [[领域地图与覆盖缺口|AI Infra 领域地图]]

> [!abstract] 核心问题
> 推理引擎不是“执行模型”的薄封装，而是一个动态状态系统：**请求产生哪些状态、状态能否复用、在哪一层保存、何时搬运、怎样在 SLO 内调度。**

> [!nav] 三个扩散方向
> 推理作为训练的数据生产者，见 [[HPCGame 2026 - 07 数据与后训练闭环]]；副本扩缩、租户隔离和故障恢复，见 [[HPCGame 2026 - 08 集群效率与可靠性]]；音视频与多阶段生成，见 [[HPCGame 2026 - 09 多模态与新工作负载]]。
>
> 下面的版本和 issue 是历史案例，核查范围见 [[MLSys - 运行记录]]。模型支持、runner 支持与部署 recipe 的具体组合见 [[MLSys - 证据台账#E-20260915-03]] 和 [[MLSys - 证据台账#E-20260915-04]]。

## 1. 先固定 workload 和目标函数

| 场景 | 主要约束 | 典型系统问题 |
| --- | --- | --- |
| 端侧/资源受限 | 功耗、内存、隐私 | 低 bit、CPU/NPU/GPU 异构、offload |
| 单机/小池 colocated | 单实例利用率 | continuous batching、prefix cache、kernel |
| 数据中心 scale-out | 独立扩缩与故障 | P/D/EPD、KV transfer、跨节点 routing |
| 多模态/agent | 长寿命、多阶段、外部事件 | encoder state、tool latency、session retention |

四个指标必须同时出现：

- **TTFT**：排队 + prefill 的首 token 延迟；
- **TPOT / ITL**：decode step 间延迟；
- **Throughput**：单位时间处理的 token/request；
- **Goodput**：满足 SLO 的有效完成量。

平均 throughput 可以靠堆积违约请求提高，因此大规模 serving 的目标更常是 goodput，而不是 token/s 最大化。

## 2. 请求不是 token 流，而是状态生命周期

```mermaid
flowchart LR
    A[request/session] --> B[encoder / prefill]
    B --> C[KV / latent / recurrent state]
    C --> D[decode / speculative verify]
    D --> E[stream / tool call]
    E -. resume .-> A
    C --> F[GPU hot tier]
    F <--> G[CPU / remote tier]
    G <--> H[storage]
```

现代模型可能同时产生：传统 KV、MLA latent、recurrent state、MTP/draft state、sparse index、vision/encoder embedding、prefix identity 与 session metadata。运行时若仍假设只有同构 KV blocks，很快会在 hybrid/multimodal 模型上失效。

## 3. 单池优化彼此耦合

| 机制                    | 直接收益                   | 新代价/耦合                          |
| --------------------- | ---------------------- | ------------------------------- |
| continuous batching   | 提高 decode batch 利用率    | scheduler overhead、延迟干扰         |
| chunked prefill       | 让长 prefill 与 decode 交错 | chunk/launch 开销、TTFT/TPOT 权衡    |
| prefix cache          | 复用共享前缀                 | identity、eviction、tenant 隔离     |
| speculative/MTP       | 一次提出多个 token           | acceptance、draft 显存、verify cost |
| CUDA Graph/megakernel | 降低 host launch         | shape 稳定性、编译和 graph cache       |
| fine-grained overlap  | 隐藏通信                   | 资源争用、依赖与 tail 覆盖                |

不能逐个打开优化开关再把收益相加：chunked prefill 改变批次形态，prefix cache 改变 prefill 到达率，speculative decoding 改变每 step token 数，CUDA Graph 又依赖 shape 分布。

### Speculative decoding 的边界

收益取决于 acceptance rate、draft/target 成本和 workload。vLLM 官方把它主要定位于 medium/low-QPS、memory-bound 场景；高 QPS 下额外 compute/显存未必换来更多 goodput。

SGLang `v0.5.18` 的公开 issue 还显示：NEXTN/EAGLE 初始化时的临时 embedding/head 若在 KV pool 定容后才释放，可让单卡 `max_total_num_tokens` 从修补后的 305,070 降到 149,806。这个单卡报告指出了容易漏算的一项：**启动期状态的释放时序会影响稳态准入容量。** [[MLSys - 2026-08 至 09 旧版基线与日志#2026-08-31|记录与来源]]

## 4. vLLM 与 SGLang 的设计起点

- vLLM 由 PagedAttention 和块式 KV 管理起家。
- SGLang 由 RadixAttention 和程序级复用起家。

按目标版本和工作负载比较以下实现。近期的调度与缓存机制见 [[D05 推理服务与调度]]、[[D06 KV 与模型状态系统]]。

```text
scheduler policy
cache identity / eviction / tiering
hybrid + multimodal state coverage
speculator 与 model runner 的组合
connector compatibility
kernel/quant backend
failure 与 observability
```

vLLM `0.28.0` 的稳定 release 已把 Model Runner V2 的 E/P/D、磁盘 KV offload、可插拔 secondary tier、partial load、metrics 与 canonical CPU layout 放进同一版本，说明这些能力正在从外围 connector 变成 runtime 核心。

## 5. PD/EPD：隔离资源，不保证 raw throughput

```text
request → prefill pool → state transfer → decode pool → stream
                ↑
       optional encoder pool
```

拆分后得到 P:D:E 独立扩缩和 SLO 隔离，同时新增：

- KV/encoder transfer engine；
- admission control 与跨池 backpressure；
- KV-aware routing 和 prefix locality；
- 独立故障域、重试与状态 ownership；
- pool ratio 与网络拓扑的联合规划。

> [!warning] 关键修正
> vLLM 官方文档明确说明 disaggregated prefill 本身不提升 throughput；它主要用于独立调节 TTFT/ITL 和隔离 tail。KV transfer 与跨池排队完全可能抵消收益。

SGLang 已公开 EPD；vLLM 提供 disaggregated encoder 的 E→PD/E→P→D 拓扑。Dynamo 又把多模态、agent priority 与 cache retention 接到分离式 runtime。演进方向不是“更多池一定更快”，而是让不同状态/阶段拥有不同资源模型。

## 6. KV cache 已是独立数据系统

| 问题 | 不能省略的系统语义 |
| --- | --- |
| identity | block hash、collision、model/version、tenant |
| placement | GPU、CPU、remote memory、SSD |
| lifecycle | allocate、reuse、evict、prefetch、invalidate |
| routing | prefix/session locality、热度与成本 |
| consistency | transfer completion、ownership、partial load |
| failure | worker 消失后 metadata 与重放 |

vLLM KV Connectors 已覆盖 NIXL、Mooncake、LMCache、FlexKV 等；SGLang HiCache 覆盖 GPU/CPU/storage 和 HF3FS、Mooncake、NIXL、AIBrix 等后端。

“有三层 cache”并不代表可靠：flat file directory 导致 ENOSPC、PD admission freeze、connector 版本组合都已在公开问题中出现。Dynamo `1.4.2` 甚至修复过 Rust NIXL binding 静默落到 non-functional stubs、而 Python binding 正常的 loader-path 问题。验收必须观察真实 bytes/latency/fallback，不能只看进程启动。

### 让驻留、可见性和路由分别拥有合约

2026-09-28 增量核验带来两个互补实例。HiSparse 中，copy **已按 stream 顺序入队**与**已经完成**对应不同权限：前者满足约束后可复用源 GPU block，后者才允许 host 页成为可命中的前缀。Dynamo 的分层索引则要求副本完整、布局匹配且可达；看到一个对象事件还不够。

因此把链条写成：**拥有数据 → 传输/有序依赖 → 发布可见 → 可路由 → 释放/失效**。engine 决定哪些状态驻留，router 决定在哪个 worker/tier 复用；断流后保守丢索引会降低命中率，却不能靠猜测补回正确性。[[MLSys - 证据台账#E-20260928-03]] · [[MLSys - 证据台账#E-20260928-04]] · [HiSparse PR](https://github.com/vllm-project/vllm/pull/53781) · [Dynamo 索引 PR](https://github.com/ai-dynamo/dynamo/pull/11239)

此处是两个项目机制的联系，不是已验证的组合 recipe：Dynamo 1.5 依赖 vLLM 0.28，不自动拥有 0.30 的 HiSparse/MRV2 新能力。Fast Start 延长权重存活也不是原 KV/会话恢复；连接 [[HPCGame 2026 - 08 集群效率与可靠性]]。

### 取消、forward结束与物理完成不是同一个事件

2026-10-12核验：SGLang的PD取消协议说明，停止请求后远端prefiller仍可能写目标页；HiSparse修复则说明target forward结束后draft rows还可能未写完。由此应分别画出**停止新工作 → 收束在途访问 → 发布可见状态 → 允许复用**，而不是用一个finished布尔值代表全部。

具体实现仍有限制：ACK只覆盖指定backend，timeout不是任意延迟下的安全证明；HiSparse原PR披露的P/D host-block排序缺口后续修复未核。[[MLSys - 证据台账#E-20261012-03]] · [[MLSys - 证据台账#E-20261012-04]] · [取消协议](https://github.com/sgl-project/sglang/pull/41023)。上层还需回答状态是否值得保留，见 [[HPCGame 2026 - 09 多模态与新工作负载#从消息生命周期到KV生命周期]]。

**2026-10-19追加修正**：HiSparse后续提案关闭未合入；其自然时序测试未复现，仅人工75ms延迟触发draft-row overwrite，greedy输出未变。因此上段“未核后续”不能再读作自然负载已确认错误，当前强度以 [[MLSys - 证据台账#E-20261019-05]] 为准。[Closing反证](https://github.com/vllm-project/vllm/pull/59633)

vLLM另一P2P路径则说明source pins、目标slots和peer注册有各自生命周期：source drain和清理注册的main修复，不自动解决consumer超时回收；正式0.32也未含这两项。验收要在**双方停止访问、注册已清理、buffer可复用**之间画依赖，不能用释放handle返回代表全部完成。[[MLSys - 证据台账#E-20261019-04]] · [producer drain](https://github.com/vllm-project/vllm/pull/59790)

### 缓存正确、能留住、值得留，是三个问题

2026-10-19补核带来三层检查：**内容版本与因果边界**决定能不能复用，**整个任务池的重用距离与容量**决定能否留到下次，**搬运与重算代价**决定是否划算。Ascend prepared metadata可能含旧KV快照；EfficientAgent的exact-prefix回放则显示，host层太小会反复写入后又提前驱逐，层足够大时总是过滤写反而有害。不能由“单次load比prefill便宜”推出offload整体更快。[[MLSys - 证据台账#E-20261019-07]] · [[MLSys - 证据台账#E-20261019-10]] · [工作集论文](https://arxiv.org/html/2609.33762v1)

若采用近似非前缀复用，还要将持久cache历史纳入输入：相同prompt可能因先前请求不同而读到不同近似状态。逐次清cache测试会漏掉这类积累；应保留重试/分支/交错对照。这里的论文反例不适用于所有exact-prefix机制。[[MLSys - 证据台账#E-20261019-11]] · [请求历史实验](https://arxiv.org/html/2610.05833v1)

## 7. CPU-GPU offload 是另一个专门化 runtime

KTransformers 把大 MoE experts 放入 CPU DRAM，只让 attention/热路径留在 GPU，并利用 AMX 等 CPU 指令。它已支持新模型、CPU-GPU expert scheduling、SGLang 与 GPU/CPU/disk 三层 prefix cache。

但这不是“消费级显卡轻松装下大模型”：官方 DeepSeek-R1 Q4 示例仍约需 382GB DRAM 和 14GB VRAM。瓶颈从 GPU 容量移动到主存容量、NUMA、PCIe/C2C 与 CPU compute。

## 8. Agent serving 改变调度事件

传统请求近似 `prefill → decode → finish`；agent session 更像：

```text
prefill → decode → tool wait → external IO → resume
        ↘ image/audio encoder ↗
```

调度目标因此可能从单次 TTFT/TPOT 变成 tool latency、session completion、cache-retention value 与端到端 cost。长寿命 session 还会把 cache eviction 与 admission 变成经济决策，而非纯 LRU。

## 9. 一次完整的 serving 实验

1. 固定 arrival process、prompt/output 分布、并发和 SLO；
2. 同时记录 TTFT、TPOT、throughput、goodput 与 tail；
3. 画出 KV/state 从 allocate 到 free 的时间线；
4. 分别记录 compute、queue、transfer、cache hit/miss；
5. 注入 worker/connector/storage 故障；
6. 再切换 batching、chunk、prefix、spec、PD/EPD，而不是只测 happy path。

## 10. 一手资料

- [vLLM prefix caching](https://docs.vllm.ai/en/latest/design/prefix_caching/) · [speculative decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) · [disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [vLLM 0.28.0](https://github.com/vllm-project/vllm/releases/tag/v0.28.0)
- [SGLang HiCache](https://github.com/sgl-project/sglang/blob/main/docs_new/docs/advanced_features/hicache_best_practices.mdx) · [EPD](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/epd_disaggregation.md)
- [SGLang EAGLE/KV capacity issue](https://github.com/sgl-project/sglang/issues/36452)
- [Dynamo 1.4.2 NIXL fix](https://github.com/ai-dynamo/dynamo/releases/tag/v1.4.2)
- [KTransformers](https://github.com/kvcache-ai/ktransformers)
