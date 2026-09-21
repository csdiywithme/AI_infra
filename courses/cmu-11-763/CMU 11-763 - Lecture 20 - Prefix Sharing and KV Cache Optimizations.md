---
type: course-note
status: developing
course: "[[CMU 11-763]]"
lecture: 20
lecture_date: 2025-11-11
area: inference
source_mode: slides
topics:
  - "[[LLM Inference]]"
  - "[[Transformer Block]]"
aliases:
  - CMU 11763 L20
  - KV Cache Optimizations
---

# Lecture 20：Prefix Sharing and KV Cache Optimizations

> [!abstract] 核心主线
> KV cache 用显存换掉历史 token 的重复计算，却随序列长度、并发数和模型层数增长。本讲按“少存哪些维度”组织优化：跨 query head 共享、压成 latent、删 token、跨层共享、降低位宽。另一条正交路线是复用相同前缀已有的计算。

## 0. 来源与阅读方式

- 主来源：[Amanda Bertsch，2025-11-11 官方课件](https://docs.google.com/presentation/d/1GU18uYRbdggLKmr1j62cZ9iB9vlCVHB34rI8FwmK9r4/edit)，35 页，已读取全页文字并核对关键公式、表格与示意图。
- 讲次按[官方课程日程](https://www.phontron.com/class/lminference-fall2025/schedule/)的实际授课顺序编号，不计假期与海报展示。
- **本篇是课件整理版**：本次未在用户给出的公开播放列表找到这一讲视频，因此不编造时间戳和课堂问答。文末问答是复习题，非录音转述。
- 公式、数值算例、伪代码和系统实验设计中的教学展开均在文中标出；不把经验性建议写成所有架构通用的定理。

### 课件导航

| 课件页 | 主题 | 需要回答的问题 |
| --- | --- | --- |
| 4–8 | Cache 内容与大小 | 为什么存 K/V、不存历史 Q？ |
| 9–15 | MHA/MQA/GQA/MLA | 究竟减少了哪个维度？ |
| 17–25 | Token pruning、attention sink、块级选择 | 删除的信息以后还能访问吗？ |
| 27–30 | 跨层共享 | 需要改训练，还是可直接用于已有模型？ |
| 32–35 | Quantization | 更小的缓存何时才真的更快？ |

## 1. KV cache 缓存的到底是什么？

来源：课件 4–6 页。

对某一层，当前 token 的表示经过投影得到 $q_t,k_t,v_t$，再计算：

$$
o_t=\operatorname{softmax}\left(\frac{q_tK_{\le t}^{\top}}{\sqrt{d_h}}\right)V_{\le t}.
$$

历史位置 $j<t$ 的 K/V 在因果模型中不会因为右侧追加 token 而改变，所以保存后可重复读取。

新 token 只需要自己的 query，不需要历史 query：未来的 attention 不会拿 $q_j$ 作为被查询对象。

> [!important] 课件勘误
> 第 7–8 页把缓存中的两个向量写成 key/query。这里应为 **key/value**。数值公式里的系数 2 不变，但数据依赖的含义必须改正。

### 1.1 Cache 省掉了什么，没省掉什么？

省掉历史 token 的逐层投影和 hidden-state 重计算；没有省掉当前 query 对历史 K/V 的读取和 attention 运算。

因此，decode 仍随已有上下文变长而变慢。固定头维时，一步 dense attention 对上下文长度 $T$ 是线性的，不是常数时间。

更长推理链不仅多生成 token，也让后续每一步读更多缓存。

### 1.2 位置编码也属于缓存语义

课件以 RoPE 为背景：通常 query/key 带位置信息，缓存保存旋转后的 key，value 不做同样的旋转。

这不是“缓存一串 token 对应的固定 embedding”。深层 K/V 已经依赖此前所有输入、模型参数、位置设置和 attention 结构。

## 2. 什么时候可以复用前缀？

来源：课件第 6 页；以下 cache-key 检查表是系统实现展开。

两个请求如果在某个长度 $P$ 之前完全相同，且执行条件相同，那么这段前缀的 K/V 可以共享。

典型例子：相同 system prompt、相同 few-shot 示例、多候选生成的共同输入、agent 对话的已有历史。

但“相同”需要比较模型实际看到的输入，而非肉眼看到的文字：

- 相同 token IDs 和顺序；
- 相同模型权重、adapter 与相关模型配置；
- 相同位置索引、RoPE 配置及 attention mask 语义；
- 多模态输入时，相同的实际输入表示与预处理语义。

改变 sampling temperature 不必使已计算 K/V 失效，因为它作用在输出分布上；改变模型、在最前面插入新提示词，则通常使旧缓存失效。

### 2.1 为什么不能随便缓存公共后缀？

`A + 公共文本` 与 `B + 公共文本` 中，公共文本在深层的表示受前面 A/B 影响，不是相同状态。

“相同前缀可复用”来自因果依赖，不能推广成“任意重复文本都能复用”。

### 2.2 共享不等于复制

教学伪代码：

```python
def fork_candidates(prefix, count, cache_pool):
    shared = cache_pool.lookup_or_prefill(prefix)
    # prefix blocks are shared; each branch owns its new suffix.
    return [Branch(prefix_blocks=shared.acquire(), suffix_blocks=[])
            for _ in range(count)]
```

真正的实现需要引用计数、内存块生命周期和分支写入隔离。一个分支继续生成时，不能改写其他分支正在复用的前缀。

## 3. 先算清楚显存账

来源：课件 7–11 页；下面统一单位并补全算例。

设：

- $B$：独立序列数；
- $T$：每条序列已缓存 token 数；
- $L$：层数；
- $H_{kv}$：KV head 数；
- $d_h$：每个 head 的维度；
- $b$：每个数的字节数。

若各层配置相同，且未做其他压缩：

$$
M_{KV}=2BTLH_{kv}d_hb.
$$

不同长度的请求应使用 $\sum_iT_i$，而非机械套最大长度乘 batch；分页分配仍有尾块浪费与元数据成本。

### 3.1 课件里的 128k 示例

取 $L=32,d_h=128$、FP16，即 $b=2$，单请求、$T=128000$：

| 架构 | $H_{kv}$ | 每 token、全层缓存 | 128000 token 缓存 |
| --- | ---: | ---: | ---: |
| MHA | 32 | 512 KiB | 62.5 GiB |
| GQA | 8 | 128 KiB | 15.625 GiB |
| MQA | 1 | 16 KiB | 1.953125 GiB |

这里 **128k 指 128000，不是 131072**。如果按 128 Ki token 计算，MHA 是 64 GiB。不要把十进制 token 数、MB、MiB 混在一起比较。

这些数值只算 KV，不包括模型参数、激活、临时 workspace 和碎片。

### 3.2 共享前缀的节省量

$N$ 个候选共享长度 $P$ 的输入，各有长度 $S$ 的输出。忽略尾块等开销：

$$
M_{naive}=\mu N(P+S),\qquad M_{shared}=\mu(P+NS),
$$

其中 $\mu=2LH_{kv}d_hb$。

节省量为 $\mu(N-1)P$。输出越长、分支越分散，公共前缀在总显存中的占比越低。

## 4. MHA、MQA、GQA：减少 KV head 数

来源：课件 9–11 页。

- MHA：每个 query head 有自己的 K/V head。
- MQA：多个 query head 共享一组 K/V。
- GQA：把 query heads 分组，每组共享 K/V。

例如 32 个 query heads、8 个 KV heads，每个 KV head 服务 4 个 query heads。

收益包括缓存容量、K/V 投影规模和可能的内存流量下降；不意味着所有 attention FLOPs 都按同样比例下降，因为 query heads 仍然存在。

### 4.1 能不能直接把现有 MHA 改成 MQA？

不能把任意训练好的 MHA 的多组 K/V 随手合并，并承诺保持质量。

这类架构本身需要训练支持；转换已有模型也需要有依据的权重转换和适配过程。它与“推理时删几个缓存 token”不是同一层面的优化。

## 5. MLA：不显式缓存展开后的所有 K/V

来源：课件 12–15 页；原始架构见 [DeepSeek-V2](https://arxiv.org/abs/2405.04434)。

核心是把每个 token 的内容 K/V 联合编码到较小的 latent：

$$
c_t^{KV}=W^{DKV}h_t.
$$

逻辑上可通过上投影得到各 head 的 K/V，但实现不一定要把它们全部展开、再写入显存。

### 5.1 为什么矩阵吸收有用？

教学展开：用列向量记号，若 $k_j=W^{UK}c_j$，则：

$$
q_t^\top k_j=q_t^\top W^{UK}c_j
=((W^{UK})^\top q_t)^\top c_j.
$$

因此可变换 query，再直接和低维 latent 做点积。与 value 有关的线性变换也可在合适的位置结合后续输出投影处理。

重点是调整计算顺序，使缓存保持压缩形态；不是每步都把整个历史 K/V 解压到 HBM。

### 5.2 RoPE 为什么打断简单吸收？

如果 key/query 之间插入随位置变化的旋转，点积中的变换包含位置 $t,j$，不再能全部合成一张与位置无关的固定矩阵。

课件给出的处理是分离内容和位置部分：内容走压缩 latent，另外保留小的 RoPE key 分量。于是缓存约为每 token、每层 $d_c+d_R$ 个数。

这解释了为什么 MLA 缓存不是“只有一个 latent”，还要算位置分支。

## 6. Pruning：少保留历史 token

来源：课件 17–25 页。

架构不变时，也可以选择只让未来 query 访问一部分历史 K/V。

常见依据包括最近窗口、起始 token、历史 attention 重要性、块级检索，以及动态的保留/淘汰规则。

### 6.1 Attention sinks

课件观察到部分模型的开头几个 token 长期吸收 attention mass。即使它们没有明显任务语义，删掉也可能破坏模型行为。

因此流式场景的一种设计是：保留少量开头 sink tokens，再保留固定长度的最近窗口。

这解决的是有限状态下持续运行的稳定性，不是恢复无限历史的精确信息。淘汰掉的一段关键事实不能因为“有 sink”就被重新读取。

### 6.2 位置处理需要和缓存策略一起设计

课件第 19 页展示滚动窗口的位置重映射：保存未做 RoPE 的 key，再按当前保留窗口的位置安排施加变换。

这是特定 streaming 方案的设计，不能当作所有模型都可随意重编号且保持输出不变的结论。

### 6.3 从删掉到查回来

课件依次提到 LM-Infinite、InfLLM，以及 TurboRAG/DBSA 的块级选择。

它们提示两种不同资源取舍：

1. 真正丢弃不重要状态，省存储但失去访问能力；
2. 把部分状态留在其他位置，按需要检索，增加索引与传输开销。

“GPU 上 cache 小”并不自动等于“整个系统没有存这些信息”。

## 7. StarAttention：并行处理长上下文

来源：课件 22–23 页。

课件先把输入分给多个 host，并在局部编码中保留 sink/anchor；decode 时广播 query，各 host 处理自己的 K/V，然后聚合。

需要区分：局部编码策略可能引入近似，而**给定各处 K/V 后的 softmax 聚合**可以用正确归一化实现。

教学推导：对 shard $j$ 的 logits $s_{ji}$，记录

$$
m_j=\max_i s_{ji},\quad
\ell_j=\sum_i e^{s_{ji}-m_j},\quad
u_j=\sum_i e^{s_{ji}-m_j}v_{ji}.
$$

令 $m=\max_jm_j$，则全局输出：

$$
o=\frac{\sum_je^{m_j-m}u_j}{\sum_je^{m_j-m}\ell_j}.
$$

不能直接平均各 host 已归一化的 attention output，因为各处的总概率质量不同。

## 8. 跨层共享：减少层这一维的存储

来源：课件 27–30 页。

| 方法 | 课件中的关键设计 | 应如何理解 |
| --- | --- | --- |
| Cross-layer attention | 相邻若干层共用 K/V | 预训练时让模型适应共享 |
| YOCO | 后半部分层 cross-attend 到共享缓存 | 架构改变，不是任意层直接复用 |
| MiniCache | 利用已有模型深层缓存相似性合并 | 推理期压缩近似，需要质量检查 |

不同层的 hidden state 和投影不同，所以缓存通常不相同。“相似”只支持近似压缩的动机，不支持逐元素相等的假设。

## 9. Quantization：减少每个数占的字节

来源：课件 32–34 页。

简化的均匀量化写成：

$$
q=\operatorname{clip}(\operatorname{round}(x/s)+z,q_{min},q_{max}),
\qquad \hat{x}=s(q-z).
$$

除了低位数值，还需要 scale、可能的 zero-point、分组信息、残差高精度区域等。因此 FP16 到 4-bit 的实际总显存收益不能只看 $16/4$。

### 9.1 Key 和 value 的误差路径不同

Key 误差改变 attention logits，从而可能改变关注哪个 token；value 误差改变读出内容。

课件建议在只能保留一者高精度时优先保护 key。应把它看作提示敏感性的经验法则，不是统一最优策略。

补充证据：[KIVI](https://arxiv.org/abs/2402.02750)专门区分 key 的 per-channel 与 value 的 per-token 量化。两者需要不同分组策略，说明只问“几 bit”还不够。

### 9.2 Cache 小了，为什么反而可能更慢？

以下是实现分析，不是课件中的 benchmark：

- 反量化和格式转换可能引入额外 kernel；
- 短上下文节省的读取量很少；
- kernel 不支持压缩格式直接计算，导致中间展开；
- 小 batch 下硬件利用率仍然不足。

评测应记录容量、TTFT、每 token 延迟、吞吐和质量，不能只截一张显存图。

## 10. 本讲优化地图

| 优化 | 主要减少 | 是否通常改变原模型计算/输出 | 核心代价 |
| --- | --- | --- | --- |
| 精确 prefix sharing | 重复前缀的拷贝与计算 | 不改变数学模型 | 状态管理与命中率 |
| MQA/GQA | KV heads | 需要相应架构/训练 | 表达能力与训练取舍 |
| MLA | 每 token 的缓存宽度 | 架构改变 | 低秩设计与 kernel |
| Token pruning | 历史 token 数 | 通常近似 | 遗忘、检索与选择成本 |
| 跨层共享 | 缓存层数 | 架构改变或近似 | 层间差异 |
| Quantization | 每元素位宽 | 数值近似 | 量化误差与解码开销 |

Paged allocation 与上述多数方法正交：它主要改善内存管理，并不自动减少每个有效 KV 元素的信息量。

## 11. 面向 AI Infra 的最小实验

教学设计：固定同一个支持 GQA 的模型与硬件，不把更换模型的质量差异混入缓存优化对比。

1. 输入长度取短、中、长三档；明确实际 token 数。
2. 分别运行单请求、离线 batch、在线请求流。
3. prefix sharing 测无共享、固定 system prompt、长共享上下文三个场景。
4. 量化对比使用相同输入和生成上限，记录实际输出长度。
5. pruning 增加长距离事实检索任务，避免只测短文本流畅度。
6. 记录预热、峰值显存、TTFT、TPOT、完成请求吞吐和失败率。

计算准确性检查：精确缓存路径先比较 logits 的数值误差；近似方法再比较任务质量，不能要求所有近似优化逐 token 完全一致。

## 12. 自测与参考回答

### Q1. 为什么不存历史 Q？

未来 token 只用自己的 query 查询历史 K/V；旧 query 不参与这一步的数据依赖。

### Q2. 32 层、8 KV heads、128 head dim、FP16 的每 token 缓存多大？

$2\times32\times8\times128\times2=131072$ bytes，即 128 KiB；单序列 10000 tokens 约 1.22 GiB。

### Q3. GQA 把 32 个 KV heads 改为 8 个，attention 计算就减少到四分之一吗？

缓存容量按这个比例缩小，但 query heads 仍为 32，不能对全部 FLOPs 和延迟套同一比例。

### Q4. 给原 prompt 最前面加一句话，后面文本没变，缓存能复用吗？

通常不能；后续 hidden state 依赖新增前缀，位置也可能变化。需要重新计算受影响部分。

### Q5. 只改变 temperature 呢？

若模型输入和执行配置未变，已有 K/V 仍可有效；temperature 改的是 logits 后的选择分布。

### Q6. MLA 为什么还要存位置分支？

RoPE 的位置相关变换阻碍把全部投影吸收到固定矩阵；分离的小位置分支保留位置敏感性。

### Q7. Attention sinks 等于长程记忆吗？

不是。它帮助维持流式 attention 的行为，但不保存所有被淘汰内容的可检索细节。

### Q8. 多卡 attention 能直接平均每张卡的输出吗？

不能。必须按各 shard 的 softmax 分母及数值稳定缩放聚合；局部质量不一定相同。

### Q9. MiniCache 和 YOCO 是一回事吗？

不是。课件将前者作为已有模型缓存相似性上的推理期近似，后者是训练支持的 decoder-decoder 架构。

### Q10. 4-bit KV 能保证端到端快 4 倍吗？

不能。位宽只改变部分内存流量；还有权重、计算、量化元数据、反量化与调度成本。

### Q11. Prefix sharing 为什么对长 system prompt 的 BoN 特别有用？

多分支重复输入可只存/算一次，节省量随 $(N-1)P$ 增大；但独立输出后缀仍需各自状态。

### Q12. 怎么判断一个 KV 优化是否值得上线？

先确定是容量不足、带宽受限还是命中重复前缀不足；在真实长度/并发分布下测质量和服务指标。仅看理论 cache bytes 不够。

## 13. 关联与待研究问题

- [[CMU 11-763]]：课程索引与材料覆盖情况。
- [[Transformer Block]]：attention、GQA、RoPE 与 prefill/decode 的基础。
- [[LLM Inference]]：状态管理、搜索分支和服务性能的统一入口。
- [[CMU 11-763 - Lecture 22 - Linearizing Attention and Sparse Models]]：从压缩 KV 转向固定大小 recurrent state。
- [ ] #question 同一显存预算下，量化 KV、缩短 context 和减小 batch 对请求尾延迟分别有什么影响？
- [ ] #question 多候选搜索中，怎样把候选剪枝、cache 引用计数和 block 回收统一调度？

## 14. 来源清单

1. [官方 35 页课件](https://docs.google.com/presentation/d/1GU18uYRbdggLKmr1j62cZ9iB9vlCVHB34rI8FwmK9r4/edit)：正文各节均标页码。
2. [DeepSeek-V2 原论文](https://arxiv.org/abs/2405.04434)：MLA 架构来源；不把论文总体吞吐收益单独归因于某个缓存公式。
3. [KIVI 原论文](https://arxiv.org/abs/2402.02750)：用于补充 key/value 量化策略的非对称性。
