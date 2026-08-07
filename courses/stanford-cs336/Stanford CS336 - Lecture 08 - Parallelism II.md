---
type: course-note
status: developing
course: "[[Stanford CS336]]"
lecture: 8
lecture_date: 2026-04-22
area: systems
topics:
  - "[[Distributed Training]]"
  - "[[ZeRO]]"
  - "[[FSDP]]"
  - "[[3D Parallelism]]"
  - "[[Expert Parallelism]]"
  - "[[Context Parallelism]]"
aliases:
  - Stanford CS336 Lecture 08
  - CS336 Parallelism II
video_url: https://www.youtube.com/watch?v=6-cXp-aOmdg
---
# Lecture 08：Parallelism II

> [!abstract] 本讲一句话
> 大模型训练没有一种万能并行法：ZeRO/FSDP 分片训练状态，tensor/expert parallel 利用高速域切 width，pipeline parallel 跨慢链路切 depth，sequence/context parallel 切 length，最后用 data parallel 填满剩余设备；组合的依据是“每 rank 放得下、collective 跑得动、local kernel 仍高效”。
## 来源与范围

- 课程：Stanford CS336 — Language Modeling from Scratch, Spring 2026
- 讲师：Tatsunori Hashimoto
- 上课日期：2026-04-22
- [课程视频](https://www.youtube.com/watch?v=6-cXp-aOmdg)，时长 1:20:11
- [课程主页](https://cs336.stanford.edu/)
- [官方 Lecture 8 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf)
- 本讲覆盖：accelerator network topology、DDP、ZeRO-1/2/3、FSDP、pipeline/tensor/sequence/context/expert parallel、activation memory、3D/4D parallelism 和训练配置思路
- 本讲不替代：具体框架版本的 FSDP/Megatron 配置文档、目标集群 NCCL benchmark、模型发布方的正式 technical report

> [!warning] 来源边界
> 正文按公开视频完整英文字幕与 2026 官方 PDF 交叉核对。时间点可以直接跳转；课件中的显存数字和公开模型配置是特定 dtype、硬件与训练阶段的快照，不能把同名模型的一次配置复制为通用答案。
## 视频时间索引

| 时间 | 课堂内容 | 对应笔记 |
| --- | --- | --- |
| [00:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=0s) | TPU mesh、GPU switched topology 与 collectives | [[#1. 把 datacenter 视为一台计算机\|1–2]] |
| [11:25](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=685s) | Data parallel 与 model-state 账本 | [[#3. DDP 的计算扩展与内存瓶颈\|3]] |
| [16:48](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1008s) | ZeRO-1/2/3 的逐级分片 | [[#4. ZeRO：逐步分片 replicated state\|4]] |
| [21:05](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1265s) | ZeRO-3 = FSDP | [[#4.3 ZeRO-3 / FSDP：再 shard parameters\|4.3]] |
| [22:49](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1369s) | FSDP incremental request/free 与 overlap | [[#5.1 Overlap\|5.1]] |
| [33:20](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2000s) | Pipeline bubble 与 schedule | [[#6. Pipeline parallel：沿 depth 切模型\|6]] |
| [42:07](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2527s) | Tensor parallel | [[#7. Tensor parallel：沿 width 切 model\|7]] |
| [49:30](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2970s) | Activation memory 与 sequence parallel | [[#8. Activation memory 不会自动随 TP 线性下降\|8–9]] |
| [57:33](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3453s) | Expert parallel 与 process groups | [[#10. Expert parallel：沿 experts 切稀疏 MLP\|10]] |
| [1:13:23](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4403s) | 公开模型的组合并行配置 | [[#11.3 课堂公开配置案例\|11.3]] |
## 1. 把 datacenter 视为一台计算机

单 GPU 扩展有两个硬上限：
- **Compute**：训练需要的总 FLOPs 远超单芯片可在合理时间完成的量；
- **Memory**：model states 与 activations 超过单卡 HBM。
理想多 GPU scaling 希望同时得到：
$$\text{aggregate compute} \propto P$$
以及：
$$\text{aggregate memory capacity} \propto P$$
但只有在通信没有压倒计算时，聚合资源才是有效资源。
### 1.1 两类 accelerator network
课件用两类极端帮助理解：
- **Mesh/torus**：每个 device 只连近邻，成本可控、结构规则，适合映射规则 tensor sharding；
- **Switched/fat-tree/all-to-all-like domain**：任意端点通信更灵活，适合 collectives 与不规则 expert routing，但交换网络昂贵。
真实系统是分层的：

```text
GPU local HBM
↕
high-bandwidth scale-up domain
↕
scale-out fabric
↕
multiple pods / datacenter network
```
“为什么不把所有 GPU 全互连”答案是 port count、cabling、switch silicon、power、cost、fault domain 和 routing complexity 都随 domain size 增长。
## 2. Collective 的带宽下界

设 $P$ ranks，每 rank 的输入/最终完整结果为 $M$ bytes。
高带宽 ring 算法中：
$$\operatorname{allreduce} = \operatorname{reduce\_scatter} + \operatorname{all\_gather}$$
每 rank 通信量：
$$V_{\text{allreduce}} \approx 2\frac{P-1}{P}M$$
这是 bandwidth-limited regime 中重要的基线：如果一个方案声称精确 all-reduce 却远少于这个必要 data movement，需要检查是否改变了语义、精度、replication 或统计口径。
但 bandwidth-optimal 不等于 latency-optimal：
$$T_{\text{collective}} \approx n_{\text{steps}}\alpha + \frac{V}{B_{\text{effective}}}$$
小 message、跨大 world size 时，step 数和软件调度可能占主导。
## 3. DDP 的计算扩展与内存瓶颈

Naive data parallel 把 batch $B$ 分给 $P$ ranks：
$$B_{\text{local}} = \frac{B}{P}$$
每 rank 保存完整 parameters 和 optimizer state，独立 forward/backward，再 all-reduce gradients。
### 3.1 Model-state 显存账本
设参数量为 $N$，不同 state 的 bytes/parameter：
- model parameter：$b_p$；
- gradient：$b_g$；
- FP32 master weight（若有）：$b_m$；
- optimizer states：$b_o$。
DDP 每 rank：
$$M_{\text{state,DDP}} = N(b_p+b_g+b_m+b_o)$$
一个常见 mixed-precision AdamW 口径：

| State | bytes/parameter |
| --- | ---: |
| BF16 parameter | 2 |
| BF16 gradient | 2 |
| FP32 master weight | 4 |
| FP32 first moment | 4 |
| FP32 second moment | 4 |
| 合计 | 16 |
若使用 pure BF16 update、低精度 optimizer state 或没有 FP32 master copy，数字会不同。不能死记“每参数 16 bytes”，应把实际框架 state dict 和 allocator 实测纳入。

课堂为了比较 ZeRO stages，另采用了 **12 bytes/parameter、8×A100 80GB** 的统一口径。由此得到的理论容量校准值是：

| 方法 | 最大参数量（课堂理想估算） |
| --- | ---: |
| Baseline DDP | 6.66B |
| ZeRO-1 | 16B |
| ZeRO-2 | 24.62B |
| ZeRO-3 / FSDP | 53.33B |

它们只比较 model states，没有扣除 activations、temporary all-gather、通信 buffers 和碎片，因此不是可直接部署的显存上限。
完整峰值还包括：
$$M_{\text{peak}} = M_{\text{state}} + M_{\text{activations}} + M_{\text{temp/buffers}} + M_{\text{fragmentation/runtime}}$$
DDP 只让 local-batch activation 下降，不让 model-state memory 随 $P$ 下降。
### 3.2 DDP 通信
令 gradient payload：
$$M_g = Nb_g$$
Ring all-reduce 每 rank 每 step：
$$V_{\text{DDP}} \approx 2\frac{P-1}{P}M_g$$
Global batch 必须足够大，让 local compute 能覆盖这部分 communication；否则 strong scaling 很快饱和。
## 4. ZeRO：逐步分片 replicated state

ZeRO 的核心不是改变 data-parallel 数学，而是：
1. 把无需始终 replicated 的状态分片；
2. 用 reduce-scatter/all-gather 在正确时机恢复所需视图；
3. 尽快释放 full materialization；
4. 用 prefetch/overlap 隐藏通信。
定义：
$$S_p=Nb_p,\quad S_g=Nb_g,\quad S_o=N(b_m+b_o)$$
### 4.1 ZeRO-1：shard optimizer states
每 rank 保留：
- 完整 parameters；
- 完整 gradients；
- $1/P$ optimizer states。
每 rank 静态 state：
$$M_{\text{Z1}} \approx S_p+S_g+\frac{S_o}{P}$$
逻辑流程：
1. 每 rank 在 local batch 计算完整 gradients；
2. reduce-scatter，让每 rank 得到负责 parameter shard 的 reduced gradient；
3. 各 rank 用本地 optimizer shard 更新自己的 parameter shard；
4. all-gather 更新后的 parameters。
通信仍约为一个 reduce-scatter 加一个 all-gather；当 gradient/parameter dtype 相同，大 message bandwidth regime 下与 DDP all-reduce 同量级。
### 4.2 ZeRO-2：再 shard gradients
每 rank：
$$M_{\text{Z2}} \approx S_p+\frac{S_g+S_o}{P}$$
Backward 时，某 layer gradient ready 后立刻：
1. reduce-scatter；
2. 只保留本 rank shard；
3. 释放 full gradient buffer。
难点是不能在整个 backward 结束后才处理，否则峰值时仍实例化了完整 gradients，达不到预期节省。
### 4.3 ZeRO-3 / FSDP：再 shard parameters
Steady-state 每 rank：
$$M_{\text{Z3}} \approx \frac{S_p+S_g+S_o}{P}$$
但执行某个 FSDP unit 时，需要临时 all-gather full parameters：

```text
prefetch parameter shard(s)
→ all-gather full unit parameters
→ forward/backward compute
→ reshard/free full parameters
→ reduce-scatter gradients
```
因此峰值不是简单总 state 除以 $P$：
$$M_{\text{peak,Z3}} \approx \frac{S_p+S_g+S_o}{P} + M_{\text{largest all-gather unit}} + M_{\text{prefetch}} + M_{\text{activations}} + M_{\text{buffers}}$$
Wrap unit 太大，temporary full parameter 峰值高；太小，collective 数量多、latency 和调度 overhead 高。
## 5. ZeRO/FSDP 的通信与边界

课件按“大 $P$、相同 parameter/gradient payload、忽略 latency”的归一化估算：

| 方法 | 近似 payload/step | 主要通信 |
| --- | ---: | --- |
| DDP | $2S$ | gradient all-reduce |
| ZeRO-1 | $2S$ | gradient reduce-scatter + parameter all-gather |
| ZeRO-2 | $2S$ | incremental gradient reduce-scatter + parameter all-gather |
| ZeRO-3/FSDP | $3S$ | forward/backward parameter all-gather + gradient reduce-scatter |
这里每个 collective 更精确地还要乘：
$$\frac{P-1}{P}$$
并按各自 dtype 替换 $S$。
为什么 ZeRO-3 约为 $3S$：
1. Forward 前 all-gather parameters：约 $S_p$；
2. Backward 前/中再次 all-gather parameters：约 $S_p$；
3. Backward reduce-scatter gradients：约 $S_g$。
实现可通过 caching、reshard policy 和 recomputation 改变次数/峰值。
### 5.1 Overlap
理想 prefetch：

```text
compute unit i
       ├──────────────┤
all-gather unit i+1
          ├──────┤
```
可见通信时间近似：
$$T_{\text{visible comm}} \approx \max(0,T_{\text{comm}}-T_{\text{overlappable compute}})$$
Overlap 需要：
- 足够大的 compute unit；
- separate streams 与正确 dependency；
- 预留 next-unit buffer；
- 稳定 collective order；
- 网络没有被其他 groups 饱和。
### 5.2 FSDP 不解决什么
- 不自动减少 activation memory；
- 不消除 parameter communication；
- 不保证通信可完全 overlap；
- 不保证 checkpoint/optimizer migration 简单；
- 不替代 tensor/context/expert parallel；
- 不同框架所称 “FSDP” 和 ZeRO stage 的细节可能不同。

> [!important] ZeRO-1 “几乎免费”有条件
> 课件的结论针对大 message、bandwidth-limited、overhead 可忽略的理想区间。小 modules、大 world size、跨层碎片化 collectives 或网络拥塞会让 latency 显著。
## 6. Pipeline parallel：沿 depth 切模型

把 layers 分给 $P_{\text{PP}}$ stages。相比 FSDP 传 parameters，PP 主要在 stage boundary 传 activations。
Microbatch activation payload：
$$M_A = B_\mu S H b_a$$
它与 parameter count 无直接线性关系，所以当 model parameters 很大而 boundary activation 相对小时，PP 的通信性质很好，适合跨较慢 inter-node link。
### 6.1 Bubble
Naive layer-wise model parallel 同一时刻只有一个 stage 工作，利用率约 $1/P_{\text{PP}}$。
将 minibatch 切成 $m$ microbatches 后，简单 schedule 的 bubble-to-useful ratio 近似：
$$\frac{P_{\text{PP}}-1}{m}$$
相应 bubble fraction：
$$f_{\text{bubble}} \approx \frac{P_{\text{PP}}-1} {m+P_{\text{PP}}-1}$$
因此 $m$ 要远大于 stage 数；但 microbatch 过小又会让 local matmul 效率下降。
### 6.2 Schedule 与 memory
- GPipe：全部 forward 后全部 backward，bubble 直观但 activation stash 大；
- 1F1B：warmup 后交替 forward/backward，减少 in-flight activations；
- Interleaved：每个 rank 多个 virtual stages，降低 bubble，增加通信/调度；
- Zero-bubble：把 backward 拆成 input-gradient 与 weight-gradient，利用 weight-gradient 的调度自由填空洞。
Stage memory 不只是 parameters，还包括：
$$M_{\text{stage}} = M_{\text{local state}} + n_{\text{in-flight}} M_{\text{activation/microbatch}}$$
## 7. Tensor parallel：沿 width 切 model

Transformer block 的常见 sharding：

| Component | 常见切分 |
| --- | --- |
| QKV projection | Column parallel |
| Attention output projection | Row parallel |
| MLP up/gate projection | Column parallel |
| MLP down projection | Row parallel |
| Norm/router/small ops | Replicated 或 sequence-sharded |
### 7.1 Column/row pairing
第一层：
$$Y^{(r)} = XW_{\text{col}}^{(r)} \in \mathbb{R}^{BS\times H_{\text{ff}}/P_{\text{TP}}}$$
中间 activation 在 shard 上本地计算。第二层每 rank 得到 partial output：
$$Z^{(r)} = Y^{(r)}W_{\text{row}}^{(r)} \in \mathbb{R}^{BS\times H}$$
最终：
$$Z = \sum_r Z^{(r)}$$
通过 all-reduce 或 reduce-scatter 完成。
### 7.2 与 pipeline 的对比
- TP 没有 pipeline bubble；
- 不要求很大 batch；
- 每个 Transformer block 都有 blocking activation collectives；
- 通信量与 $BSH$ 和 layers 成正比；
- 适合低 latency、高 bandwidth 的 scale-up domain。

![FSDP 参数 all-gather、计算、reduce-scatter 与释放时间线](../../assets/courses/stanford-cs336/lecture-08/l08-23m10s-fsdp-timeline.png)

> 视频关键帧：[23:10](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1390s)。课堂图的重点不是“所有 all-gather 一次发完”，而是 unit 级 request、compute、reduce-scatter、free，并尽量让下一 unit 的通信藏在当前计算后面。
课件给出每层 TP activation communication 的近似形式：
$$V_{\text{TP,layer}} \propto BSH \frac{P_{\text{TP}}-1}{P_{\text{TP}}}$$
具体常数取决于 forward/backward collectives、dtype、是否使用 sequence parallel 和统计 send/receive 的方式，不能脱离实现死记。
TP degree 过大时：
- local GEMM 变窄，Tensor Core 利用率下降；
- collective latency 增长；
- 跨 node 后 bandwidth 急剧变差。
这就是实践中 TP 经常限制在单个高速互连 domain 的原因，而不是“TP 理论上最多只能 8”。
## 8. Activation memory 不会自动随 TP 线性下降

Parameters 被 tensor-shard 后，一部分 matmul activations 也被分片；但 block 中还有 replicated activations：
- LayerNorm input/output/statistics；
- Dropout masks；
- residual streams；
- attention/MLP input；
- 某些 attention intermediates。
可以写成：
$$M_{\text{act/layer}} = M_{\text{TP-sharded}} + M_{\text{replicated pointwise}} + M_{\text{attention quadratic}}$$
即使 $P_{\text{TP}}$ 增大：
$$M_{\text{TP-sharded}}\rightarrow \frac{1}{P_{\text{TP}}}M_{\text{TP-sharded}}$$
replicated 项不会下降。长序列时 quadratic attention 项还可能占主导；FlashAttention/recomputation 可避免保存部分 $S^2$ intermediates。

课件进一步给出一组统一口径的 activation memory 公式。令 $s$ 为 sequence length、$b$ 为 microbatch、$h$ 为 hidden size、$a$ 为 attention heads、$t$ 为 TP degree：

| 配置 | 每层 activation 近似 |
| --- | --- |
| 无并行 | $sbh(34+5as/h)$ |
| TP | $sbh(10+24/t+5as/(ht))$ |
| TP + SP | $sbh(34/t+5as/(ht))$ |
| TP + selective recompute | $sbh(10+24/t)$ |
| TP + SP + selective recompute | $34sbh/t$ |

其中 replicated 的 $10sbh$ 来自 LayerNorm 的 $4sbh$、dropout 的 $2sbh$、attention/MLP inputs 的 $4sbh$。Sequence parallel 的价值正是把这些逐 token pointwise 项也沿 sequence 切开。

![Sequence parallel 把 LayerNorm/dropout 等逐 token 激活沿序列切分](../../assets/courses/stanford-cs336/lecture-08/l08-49m30s-activation-memory-formulas.png)

> 视频关键帧：[49:30](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2970s)。图中的 `g/ĝ` 在 forward/backward 分别对应 all-gather 与 reduce-scatter 的互换。
## 9. Sequence parallel 与 context parallel

两者都沿 sequence 切，但解决的范围不同。
### 9.1 Sequence parallel
目标主要是把 TP block 中原本 replicated 的 pointwise activations 分片：
$$X^{(r)} \in \mathbb{R}^{B\times(S/P_{\text{SP}})\times H}$$
LayerNorm、dropout、residual 等逐 token operations 可本地执行。
进入需要 tensor-parallel layout 的 linear 前后，用 all-gather / reduce-scatter 在 sequence-sharded 与 tensor-sharded layout 间转换。
收益：
$$M_{\text{pointwise act/rank}} \approx \frac{1}{P_{\text{SP}}} M_{\text{pointwise act}}$$
实践中 SP 常与 TP 使用同一个 group/degree。
### 9.2 Context parallel
Context parallel 面向超长 sequence 的 attention 与 KV/activation：
$$Q^{(r)},K^{(r)},V^{(r)} \in \mathbb{R}^{B\times H_{\text{heads}}\times(S/P_{\text{CP}})\times d}$$
但每个 query 仍依赖其他 shards 的 keys/values。Ring attention 类算法让 K/V blocks 在 ranks 间轮转，每 rank：
1. 计算 local $Q$ 对当前 K/V block 的 partial attention；
2. 用 online softmax 合并 max、normalizer 和 output；
3. 把 K/V block 传给下一个 rank。
KV/activation memory 可近似降到 $1/P_{\text{CP}}$，代价是与 sequence-sized K/V 相关的通信和更复杂的 causal load balance。

> [!note] SP 与 CP 不只是命名差异
> SP 常指 TP 周围的 pointwise activation sharding；CP 覆盖 attention context dependency。框架术语可能不同，判断时应看实际 tensor layout 和 collective。
## 10. Expert parallel：沿 experts 切稀疏 MLP

MoE 有 $E$ 个 experts，分到 $P_{\text{EP}}$ ranks。Router 为 $T=BS$ 个 tokens 选择 top-$k$ experts。
Dispatch：

```text
tokens on data ranks
→ pack by destination expert/rank
→ all-to-all
→ local expert GEMMs
→ reverse all-to-all
→ restore token order and combine
```
理想均衡的 token-expert assignments/rank：
$$\frac{kT}{P_{\text{EP}}}$$
Expert parameter memory：
$$M_{\text{expert params/rank}} \approx \frac{1}{P_{\text{EP}}} M_{\text{all expert params}}$$
### 10.1 为什么 EP 可能优于对 expert MLP 做 TP
- 每个 expert GEMM 可以保持较宽，减少 matmul fragmentation；
- 每 rank 只保存一部分 experts；
- 计算只激活 top-$k$ experts。
代价：
- 两次 all-to-all；
- token imbalance 导致 straggler；
- capacity padding/dropping；
- token 数不足时每个 expert batch 太小；
- 路由与 packing kernel overhead；
- fabric bisection bandwidth 压力。
### 10.2 Attention 与 MoE MLP 的并行需求不同
MoE 只替换 MLP，attention 仍是 dense。可能出现：
- Attention 希望较高 TP/CP；
- Expert MLP 希望较高 EP、较低 expert tensor parallel；
- 两部分使用不同 process-group factorization。
因此现代系统可能区分：

```text
attention: DP × TP × CP
experts:   EDP × ETP × EP
```
组合不再只是一个全模型统一 TP degree。
## 11. 组合并行：从约束出发，而不是从名词出发

总 GPU 数通常分解为：
$$P_{\text{total}} = P_{\text{DP}} \times P_{\text{TP}} \times P_{\text{PP}} \times P_{\text{CP}} \times P_{\text{EP}}$$
但某些 dimensions 可能共享/嵌套 process groups，尤其 MoE 中不能盲目相乘；应以实际 rank mesh 为准。
### 11.1 实用求解顺序
1. **单 rank memory ledger**
   - parameters、gradients、optimizer；
   - activations/KV；
   - temporary all-gather、collective buffers。
2. **先让模型放得下**
   - 高速域内使用 TP/EP；
   - 跨 node 用 PP；
   - 或在 bandwidth 足够时用 FSDP/ZeRO-3。
3. **检查 local kernel shape**
   - local hidden/head/expert tokens 是否仍足够大；
   - microbatch 是否能饱和 GEMM。
4. **检查 communication**
   - 每种 collective 的 bytes、次数、group、link；
   - 是否可 overlap。
5. **用 DP 使用剩余设备**
   - 但受 global batch 上限约束。
6. **若 batch 不够大**
   - gradient accumulation 提高有效 batch/通信效率；
   - 它不会减少每 token 总 compute。
### 11.2 Topology-aware mapping
常见启发式：

```text
within fastest scale-up domain:
    TP / EP
across nodes:
    PP or FSDP/DP depending on payload and bandwidth
remaining replicas:
    DP
long context:
    add CP
```
这是启发式，不是定理。若 node 内没有全带宽互连、跨节点 fabric 很强，或模型/shape 不同，结论会变化。

### 11.3 课堂公开配置案例

视频最后没有停留在抽象名词，而是对照公开资料展示组合配置：

- DeepSeek V3：示例包含 PP16、EP64、ZeRO-1；attention 与 expert MLP 使用不同并行需求；
- Llama 3 405B：示例为 DP128、TP8、PP16；长上下文阶段提高 CP、降低 DP；
- 还比较了 Gemma 2、Mixtral 8×22B、Nemotron 3、Qwen 3。

> [!warning] 读配置表的方式
> 这些数字与 checkpoint 阶段、context length、global batch、cluster topology、框架版本绑定。应学习的是“约束改变时如何重新分解 rank mesh”，而不是背某个模型名对应的一组 degrees。
## 12. 并行策略对比

| 方法 | 主要切分 | Param state/rank | Activation/KV | 主要通信 | 主要限制 |
| --- | --- | ---: | ---: | --- | --- |
| DDP | Batch | 不下降 | local batch 下降 | 每 step gradient AR | global batch、state capacity |
| ZeRO-1 | Optimizer | 部分下降 | 不变 | RS grad + AG param | parameters/grad replicated |
| ZeRO-2 | Optimizer+grad | 更多下降 | 不变 | incremental RS + AG | parameters replicated |
| ZeRO-3/FSDP | 全部 state | 约 $1/P$ | 不变 | param AG + grad RS | latency、temp full params |
| PP | Layers/depth | 约 $1/P$ | in-flight buffers | boundary P2P | bubble、stage balance |
| TP | Hidden/heads | sharded part 约 $1/P$ | 部分下降 | 每 block collectives | 需要高速低延迟链路 |
| SP | Sequence pointwise | 通常不变 | 约 $1/P$ 对相关项 | AG/RS layout change | 与 TP layout 耦合 |
| CP | Context | 通常不变 | sequence/KV 约 $1/P$ | K/V 或 partial attention | 长序列通信 |
| EP | Experts | expert params 约 $1/P$ | routing buffers | token all-to-all | load balance、fabric |
## 13. AI Infra 视角

### Shape
- 每个并行维度都改变 local tensor shape；
- divisibility、head/expert count、sequence shard 和 microbatch 决定 padding；
- local shape 太小会让通信赢了、kernel 输了。
### Compute
- DP 分 samples，TP 分 layer matmul，PP 分 layers，EP 分 active experts；
- bubble、routing imbalance、narrow GEMM 和 checkpoint recompute 都改变 useful FLOPs utilization；
- 只看 aggregate peak 会严重高估。
### Memory
- 区分 steady-state shard memory 和 temporary full materialization；
- FSDP prefetch、pipeline in-flight activations、EP routing buffers 会抬高峰值；
- ZeRO 不自动减少 activation，TP 也不自动分掉全部 activation。
### Communication
对每种 collective 做五元组：

```text
(payload bytes, frequency, process group, topology, overlap window)
```
缺任何一项，都无法预测性能。
### Runtime
- Process-group construction 与 collective order 必须一致；
- Async streams、prefetch depth、bucket/wrap size 是性能参数；
- Sharded checkpoint、failure recovery 和 elastic restart 是大规模训练必要组成。
## 14. 自己的推导与易错点

### 14.1 FSDP 是否值得：一个 overlap 条件
对一个 unit：
$$T_{\text{AG}} \approx \alpha_{\text{AG}} + \frac{P-1}{P} \frac{S_{\text{unit}}}{B}$$
若下一 unit compute：
$$T_{\text{compute,next}} \ge T_{\text{AG}}$$
则理想上可隐藏 all-gather；unit 太小会被 $\alpha$ 主导，太大又增加峰值 full parameters。这说明 wrap granularity 存在中间最优点。
### 14.2 PP 与 TP 的链路选择
PP 每 boundary payload 约：
$$B_\mu S H b$$
TP 每 block 多次 activation collective，且乘 layers。即使单次 payload 相似，TP 的频率更高、通常 blocking，因此对 latency/bandwidth 更敏感。这是 PP 更适合跨节点的主要原因。
### 14.3 EP 的最低 token 密度
理想每 expert assignments：
$$\frac{kBS}{E}$$
若这个值太小，expert GEMM 的 batch dimension 太小，算力利用率下降。增加 EP degree 虽节省 expert parameter memory，却不会凭空增加 tokens；需要增大 batch/sequence、减少 experts/group，或做 expert grouping。
### 14.4 常见误区
1. **ZeRO stage 越高一定越快**：更省 memory，通常也增加/碎片化通信。
2. **ZeRO-3 稳态除 $P$ 就是峰值**：还要加 all-gather unit 和 prefetch buffers。
3. **FSDP 等于 TP**：FSDP 在计算前恢复 parameters；TP 直接对 sharded math 求值。
4. **PP 不需要大 batch**：需要足够 microbatches 隐藏 bubble。
5. **TP degree 越大，单卡越省且越快**：local GEMM 变窄且 collective 增加。
6. **SP 与 CP 完全相同**：一个常分 pointwise activations，一个解决 attention context dependency。
7. **EP 只减少 parameters**：还引入 all-to-all、routing buffer 和 imbalance。
8. **DP×TP×PP×CP×EP 总能直接相乘**：groups 可能嵌套、共享或只应用于部分 block。
9. **框架报告 reserved memory 就是 tensor memory**：allocator cache、fragmentation 和 communication workspace 也在其中。
10. **公开模型的并行配置可直接复用**：训练阶段、硬件和 context 不同会改变最优点。
## 15. 本讲结论

1. Datacenter 是新的计算单元，network topology 是模型架构约束的一部分。
2. DDP 提供 compute scaling，但复制全部 model state。
3. ZeRO-1/2/3 依次 shard optimizer、gradients、parameters；FSDP/ZeRO-3 用临时 all-gather 换线性 state capacity。
4. FSDP 的真实峰值包含 full unit、prefetch、activation 与 buffers，通信约为两次 parameter all-gather 加一次 gradient reduce-scatter。
5. Pipeline parallel 切 depth，适合跨较慢链路，但需要 microbatches 与精细 schedule 控制 bubble。
6. Tensor parallel 切 width，没有 pipeline bubble，却每层依赖高速 collective。
7. Sequence/context parallel 分别处理 replicated pointwise activation 与长上下文 attention/KV。
8. Expert parallel 分片 experts，并用 all-to-all route tokens；load balance 是核心。
9. 3D/4D 并行是 topology-aware 的约束求解，不是把所有 degree 任意相乘。
10. 最终配置必须同时通过 memory ledger、communication model、local-kernel benchmark 和 end-to-end profile。
## 16. 自测问题

1. 为什么 accelerator domain 不能无限做成全互连？
2. DDP 的 parameter、gradient、master weight 和 Adam state 各占多少显存？
3. ZeRO-1/2/3 分别 shard 什么？
4. 为什么 ZeRO-1 的带宽通信量可以和 DDP 同量级？
5. 为什么 ZeRO-3/FSDP 的 normalized payload 常近似 $3S$？
6. FSDP steady-state memory 与 peak memory 有何不同？
7. FSDP wrap unit 太大和太小分别有什么问题？
8. Pipeline bubble ratio 如何随 stage 和 microbatch 变化？
9. 为什么 PP 经常比 TP 更适合跨节点？
10. Column/row TP 如何组合完成一个 MLP？
11. 为什么 TP 无法自动线性降低全部 activation memory？
12. Sequence parallel 和 context parallel 的语义差异是什么？
13. Ring attention 如何合并不同 K/V shards 的 softmax？
14. Expert parallel 为什么需要两次 all-to-all？
15. 哪些因素会造成 expert load imbalance？
16. 如何将总 GPU 数映射到 DP/TP/PP/CP/EP process groups？
17. 在选择并行度时，为什么必须检查 local GEMM shape？
18. 如何判断某个 collective 能否被 compute overlap？
## 参考资料

- [Stanford CS336 Spring 2026](https://cs336.stanford.edu/)
- [CS336 Lecture 8 PDF](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf)
- [Lecture 8 video](https://www.youtube.com/watch?v=6-cXp-aOmdg)
- [ZeRO](https://arxiv.org/abs/1910.02054)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
- [PyTorch FSDP tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)
- [Megatron-LM](https://arxiv.org/abs/1909.08053)
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
- [GPipe](https://arxiv.org/abs/1811.06965)
- [Megatron Core MoE documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
