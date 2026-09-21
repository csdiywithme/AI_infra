---
type: course-note
status: developing
course: "[[CMU 11-763]]"
lecture: 22
lecture_date: 2025-11-18
area: inference
source_mode: slides
topics:
  - "[[LLM Inference]]"
  - "[[Transformer Block]]"
aliases:
  - CMU 11763 L22
  - Linear and Sparse Attention Variants
---

# Lecture 22：Linearizing Attention and Sparse Models

> [!abstract] 核心主线
> Dense attention 可以直接查询不断增长的历史，但需要保存不断增长的 KV cache。Sparse attention 减少查询范围；linear attention/SSM 把历史压进固定大小状态；hybrid 在部分层保留精细检索能力。关键不只是把 $O(T^2)$ 改成 $O(T)$，而是同时理解表达能力、prefill 并行化、decode 状态与硬件实现。

## 0. 来源与范围

- 主来源：[Amanda Bertsch，2025-11-18 官方课件](https://docs.google.com/presentation/d/19jJynlmA8wMDHOOv66ZCF7UbiG8rwzb2tl6quSdegyc/edit)，43 页；已读取全页文本，核对关键公式与图示。
- 讲次按[课程日程](https://www.phontron.com/class/lminference-fall2025/schedule/)中的实际授课顺序编号。
- **本篇是课件整理版**。未在给定公开播放列表找到该讲视频，故使用页码定位，没有虚构字幕时间或现场问答。
- 为避免照抄课件简写造成错误，以下统一使用列向量。推导、伪代码、复杂度表和自测为教学展开。

### 课件导航

| 页码 | 主题 |
| --- | --- |
| 3–8 | Sparse Transformer、Longformer、Reformer、local/global layers |
| 9–17 | RNN 固定状态与非线性递归的并行化问题 |
| 18–24 | Linear attention 的 recurrent、parallel、chunkwise 形式 |
| 25–29 | 衰减、门控与硬件效率 |
| 30–36 | Mamba、Mamba-2、Gated DeltaNet |
| 37–40 | 状态容量限制与 hybrid models |
| 41–43 | 对 decoding 与 prefix caching 的影响 |

## 1. 从“删 KV”到“让模型按稀疏方式工作”

来源：课件 3–8 页。

上一讲讨论对已存在的 cache 进行 pruning。本讲先换一个角度：训练时就规定 attention 只连接一部分位置，让模型适应这种信息流。

### 1.1 典型稀疏模式

课件以 Sparse Transformer、Longformer、Reformer 为例说明不同设计：

- 局部窗口适合近邻依赖；
- 全局连接为远距离信息提供通路；
- 固定模式或内容相关分组，以较少连接替代所有位置两两比较。

它们不共享一个唯一的复杂度公式。窗口宽度、全局 token 数、稀疏布局和实现方式都影响计算量。

### 1.2 Local + global layers

另一种思路是在部分层用固定窗口，在少数层保留全局 attention。

课件引用当时的 Llama-4、Command-R 作为架构例子。这里学习的是组合机制，不把课件里的模型快照当作今后所有版本的规格。

局部层只需维护最近 $w$ 个 token 的 K/V，全局层仍保留整个长度 $T$ 的 cache。

设共有 $L$ 层，其中 $L_g$ 层全局、其余局部，其他配置一致：

$$
M_{KV}=2BH_{kv}d_hb\left[L_gT+(L-L_g)\min(T,w)\right].
$$

这是容量估算，不含元数据与其他激活。只要有全局层，整体缓存仍可能随 $T$ 增长，不能称为完全固定内存。

## 2. RNN：固定状态很诱人，为什么曾经难以训练加速？

来源：课件 9–17 页。

传统递归可写成：

$$
h_t=\sigma(W_hh_{t-1}+W_xx_t),\qquad y_t=W_oh_t.
$$

无论历史多长，都只需携带 $h_t$；decode 的历史状态不随序列增长。

问题是即使训练时整条输入已知，$h_t$ 也依赖非线性变换后的 $h_{t-1}$。不能简单把所有 token 的递归转成一轮普通矩阵乘法。

Transformer 则可以在已知输入的一个层内并行构造 Q/K/V，再用 masked attention 并行计算各位置；这不等于所有层都能同时计算。

### 2.1 本讲想同时要的三个性质

1. Training/prefill 能有效并行；
2. Decode 单步不需要扫描全部历史；
3. 历史状态大小不随 $T$ 增长。

线性递归让我们有机会同时接近这些目标，但还需要控制状态容量、数值稳定性和实际 kernel 性能。

## 3. 先纠正复杂度的参照对象

课件第 14 页把“每个新增 token 的 dense attention”写成对输入长度二次。对于使用 KV cache 的标准 decode，这一表述不准确。

固定模型维度时：

| 形式 | 长度 $T$ 的 prefill/整段 mixing 工作量 | decode 第 $t$ 步 | 持久历史状态 |
| --- | --- | --- | --- |
| Dense causal attention | $O(T^2d)$ | $O(td)$ | $O(Td)$ |
| 固定窗口 attention | $O(Twd)$ | $O(wd)$ | $O(wd)$ |
| 简单 linear attention | $O(Td_kd_v)$，合适实现 | $O(d_kd_v)$ | $O(d_kd_v)$ |

表中忽略投影、FFN、层数和 batch，仅比较一层的 token mixing。Linear attention 的 $O(T)$ 是**关于序列长度**；状态关于 head 维度可能是二次的。

整段自回归生成累加 $1+2+\cdots+T$，dense attention 总工作量仍是二次。不要把整段总量写成单步成本。

## 4. 移除 softmax 后为什么可以递归？

来源：课件 18–22 页。先考虑未归一化点积 attention。

设 $q_t,k_t\in\mathbb R^{d_k}$，$v_t\in\mathbb R^{d_v}$：

$$
o_t=\sum_{j\le t}v_j(k_j^\top q_t).
$$

利用矩阵乘法结合律：

$$
S_t=\sum_{j\le t}v_jk_j^\top\in\mathbb R^{d_v\times d_k},
\qquad o_t=S_tq_t.
$$

因此：

$$
\boxed{S_t=S_{t-1}+v_tk_t^\top,\qquad o_t=S_tq_t.}
$$

过去的信息被累加进一张固定大小的矩阵，而不是保存 $t$ 组独立 K/V。

### 4.1 不是“把模型中的 softmax 删掉就免费提速”

这个变换对**新的未归一化点积算子**成立，不等价于原来的 softmax attention。

标准 softmax 权重包含依赖当前 query 的归一化和指数项，不能用上述简单矩阵和精确表示。

因此更换算子通常意味着架构与训练变化，不是对任意现有模型的无损编译优化。

### 4.2 一个小算例

教学例子：$d_k=d_v=2$，令 $k_1=(1,0)^\top,v_1=(2,0)^\top$，$k_2=(0,1)^\top,v_2=(0,3)^\top$。

$$
S_2=\begin{pmatrix}2&0\\0&3\end{pmatrix}.
$$

查询 $q=(1,0)^\top$ 得到 $(2,0)^\top$；查询 $(1,1)^\top$ 得到 $(2,3)^\top$。

如果很多 keys 相似，它们的写入会叠加。固定状态通过压缩获得效率，同时也可能把本应分开的记忆混在一起。

## 5. Kernelized linear attention：不要漏掉分母

来源：课件第 20 页；以下补全公式。

若相似度可写成 $\phi(q)^\top\phi(k)$，则归一化输出为：

$$
o_t=
\frac{\left(\sum_{j\le t}v_j\phi(k_j)^\top\right)\phi(q_t)}
{\left(\sum_{j\le t}\phi(k_j)\right)^\top\phi(q_t)}.
$$

除矩阵状态 $S_t$ 外，还需维护向量 $z_t=\sum_{j\le t}\phi(k_j)$。实现需处理分母接近零与数值稳定性。

> [!important] 记号修正
> 课件用“phi=1”来描述后文简化。对这里保留 $q^\top k$ 的形式，准确表述应为使用恒等特征映射 $\phi(x)=x$，并另行去掉归一化分母；不是令所有特征都等于常数 1。

恒等映射也不保证相似度非负，因此这个简化形式不能自动解释为概率加权平均。

## 6. Recurrent、parallel、chunkwise 是三种计算组织方式

来源：课件 22–24 页。

### 6.1 Recurrent：适合一个 token 一个 token 地 decode

```python
# Teaching pseudocode; one head, column-vector convention.
def step(state, query, key, value):
    state = state + outer(value, key)
    output = state @ query
    return output, state
```

每步更新固定大小状态；生成未来 token 的依赖仍然存在，不能因为 state 固定就把未知输出同时生成。

### 6.2 全并行表达式：可并行，不代表低工作量

对整段输入，仍能写成：

$$
O=((QK^\top)\odot M)V,
$$

此处改用行向量矩阵记号，$M_{ij}=1[j\le i]$。若直接生成 $T\times T$ 矩阵，工作量和中间存储又变成二次。

课件第 23 页同时展示 softmax 对照式。对于 softmax，屏蔽未来位置需要在 logits 上加 $-\infty$ mask；不能把未来 logits 乘零后再 softmax，否则零 logits 仍有正概率。

### 6.3 Chunkwise：块内矩阵乘，块间状态传递

把序列分成长度 $C$ 的块。一个块的输出拆成：

1. 当前 query 读取进入本块前的历史状态；
2. 当前 query 读取本块内更早的 token。

块结束时汇总本块的 $v_jk_j^\top$，得到下一块状态。

简单无门控情形下，块更新是加法，可用 prefix sum 聚合。对线性仿射递归，也可构造关联的组合操作。

$$
(A_2,b_2)\circ(A_1,b_1)
=(A_2A_1,A_2b_1+b_2).
$$

这说明并行 scan 为什么可能；但只有选择了可高效组合的矩阵结构，算术量和内存流量才划算。

固定块长下，块内二次代价累加约为 $O(TC)$，另加状态相关计算。块越小不一定越快：矩阵乘尺寸、kernel launch、状态读写都会改变。

## 7. 固定状态不能无限叠加：遗忘与写入门控

来源：课件 25–29 页。

简单累加状态会受到两个问题影响：旧内容累积过多，当前 token 也不一定值得写入。

一种示意形式是：

$$
S_t=\alpha_tS_{t-1}+\beta_tv_tk_t^\top.
$$

- $\alpha_t$ 控制保留历史多少；
- $\beta_t$ 控制新信息写入多少。

更一般的线性递归是 $S_t=A_tS_{t-1}+B_tg(x_t)$。它可以对状态保持线性，但让系数依赖输入，从而具有选择性。

“Linear RNN”不等于整个网络没有非线性：输入投影、门控、激活、输出模块仍可非线性。

### 7.1 为什么数值问题很重要？

反复应用不稳定的状态转移会放大误差，衰减过快又会遗忘。低精度下更要注意累加误差和动态范围。

课件强调不能把转移矩阵当任意随机矩阵使用：需要结合稳定性、记忆长度和可并行实现来设计。

## 8. Mamba 与 Mamba-2 的课程视角

来源：课件 30–34 页。

Mamba 使用 selective state space model：部分状态更新参数随 token 输入变化，让模型选择如何写入、保留和读出信息。

课件将 $\Delta$ 解释为离散化连续系统时的步长。直觉上它影响状态演化速度，不是额外生成了若干个可见 reasoning tokens。

Mamba-2 进一步利用 structured state space duality，调整状态转移结构与计算布局，让较大的状态更适合高效矩阵运算。参见作者的 [Mamba-2 模型说明](https://goombalab.github.io/blog/2024/mamba2-part1-model/)。

这里的共同目标是：既有递归形式适合 decode，又有块状/并行形式适合训练和 prefill。

## 9. Gated DeltaNet：写“纠正量”，而非只做累加

来源：课件 35–36 页；公式展开对照 [Gated Delta Networks](https://arxiv.org/abs/2412.06464)。

设状态表示 key 到 value 的映射，当前状态对 $k_t$ 的预测为 $S_{t-1}k_t$。Delta rule 用预测误差更新：

$$
S_t=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^\top.
$$

带遗忘门的一种对应写法为：

$$
\bar S_t=\alpha_tS_{t-1},\qquad
S_t=\bar S_t+\beta_t(v_t-\bar S_tk_t)k_t^\top.
$$

若新 key 和已存 key 相似，单纯外积相加会不断堆叠；delta update 会参考当前已知内容，只写入需要纠正的部分。

这不保证没有干扰，也不是对无限字典的精确存储；仍受有限状态、key 相关性和训练目标约束。

## 10. Hybrid：让固定状态和精确检索各自发挥作用

来源：课件 37–40 页。

课件指出，某些固定状态模型在复制、in-context learning、associative recall 等任务上更困难；而且短上下文时，一张大的 recurrent state 可能比 KV cache 更占空间。

因此可以在多数层用 linear/SSM block，在少数层使用全局 attention。

课件快照包括 Jamba 的 7:1 Mamba/Transformer 配置，以及 Qwen-3-Next 的 3:1 Gated DeltaNet/Transformer 配置。它们是具体架构的例子，不是 hybrid 的固定定义。

### 10.1 一个状态大小交叉点

教学估算：同一 head 维度 $d$，dense KV 有约 $2Td$ 个数，简单矩阵 state 有 $d^2$ 个数。

两者相等时：

$$
T\approx d/2.
$$

当 $d=128$，交叉点约为 64 tokens。真实模型还要计 head 数、状态维度、GQA、额外卷积状态和实现元数据，不能用这个例子预测整模型显存。

## 11. 换了架构，采样、搜索和缓存哪些变？

来源：课件 41–42 页。

只要模型仍输出下一 token 的分布，temperature、top-p 等分布操作仍可定义，beam/BoN 等序列选择思想也仍成立。

但实现中的“模型状态”已经不同：

- Transformer 分支带一串可共享的 KV blocks；
- recurrent 分支带当前固定状态；
- hybrid 分支同时带两者。

### 11.1 Prefix caching 不是不可能，而是缓存策略变了

对完全相同的前缀，recurrent 模型也能复用该前缀结束时的状态快照。

困难在于：只有最终状态，通常不能从中恢复任意更短前缀的状态。若请求在历史中间某处分叉，就需要保存对应 checkpoint，或从更早 checkpoint 重算。

KV block 结构较自然地支持不同长度的公共前缀；固定状态模型需要主动决定保存哪些边界。

### 11.2 Beam search 与 speculative rollback

教学展开：每个 beam 要维护自己的 recurrent state；分叉时复制状态，剪枝时释放状态。

投机生成若需要撤回若干步，必须能恢复目标边界的状态。不能假设删掉末尾 KV block 的逻辑可以原封不动搬过来。

所以“解码数学不变”不等于“serving 引擎无须改变”。

## 12. 硬件实现：渐进复杂度不是 benchmark

来源：课件 27–29 页；以下是性能分析展开。

线性复杂度的逐 token 小算子可能 GPU 利用率很差；块状矩阵运算即使多做一些算术，也可能更快。

至少区分：

1. 总 FLOPs；
2. HBM 读写量；
3. 状态更新的依赖深度；
4. 张量核友好程度；
5. kernel launch、通信和调度开销。

评测需分开 prefill 与 decode，并覆盖短/长上下文、小/大 batch。不能用训练吞吐推断单请求 decode 延迟。

## 13. 最小正确性与性能实验

教学练习，适合衔接 kernels 学习：

1. 用小张量实现未归一化的 dense masked attention。
2. 实现 $S_t=S_{t-1}+v_tk_t^\top$ 的递归形式。
3. 在相同 Q/K/V 上比较逐 token 输出，应在数值误差内一致。
4. 实现 chunkwise 形式，检查块边界、当前 token 是否纳入、mask 方向。
5. 把 reference 换成 softmax attention，观察不再相等，解释这是算子不同而非 bug。
6. 逐步扩大 $T$，比较时间和峰值显存；不要把 Python 循环版本当最终 kernel 性能。
7. 做分支与 rollback 单测，验证状态复制没有别名写入。

## 14. 自测与参考回答

### Q1. Dense attention 的单步 decode 是 $O(T^2)$ 吗？

有 KV cache 时，对历史长度是 $O(T)$ 的 attention 扫描；整段 prefill 或逐步生成的累计 attention 工作量才可能是二次。

### Q2. Linear attention 的 linear 指什么？

通常指合适实现下总 mixing 工作量随序列长度线性增长；不是说对 hidden dimension 也线性，更不是整个网络只有线性函数。

### Q3. 为什么 softmax 不能用一个简单 $S=\sum vk^\top$ 精确替代？

指数和 query 相关的归一化使它不是普通点积的直接累加形式；移除它改变了 attention 算子。

### Q4. 去掉 softmax 后直接计算 $QK^\top$，是不是已经线性了？

没有。显式 $T\times T$ 矩阵仍是二次；需要重排为 recurrent、scan 或 chunkwise 实现。

### Q5. 为什么 normalized kernel attention 需要第二个状态？

矩阵状态累计加权 value 的分子，向量状态累计特征用于分母，保证按同一相似度归一化。

### Q6. Chunkwise 是否意味着所有块完全互不依赖？

不是。块内工作可并行，块间仍有状态传递；可用可结合的运算/scan 组织这种依赖。

### Q7. Linear RNN 可以有输入相关门控吗？

可以。对前一状态的更新保持线性/仿射，并不禁止参数由输入通过非线性网络产生。

### Q8. Delta rule 比单纯写入外积多考虑什么？

它先查询当前状态已知的 value，再写预测误差，避免只对相关 key 反复累加新值。

### Q9. 为什么 short context 下 recurrent state 未必省显存？

固定矩阵 state 本身可能很大；短序列的 KV 数量很少，需要比较实际维度和 head 配置。

### Q10. Hybrid 的全部缓存都变成常数了吗？

没有。保留的 global attention 层仍随上下文增长，常数部分只是其他 recurrent/local layers。

### Q11. 固定状态模型完全不能做 prefix caching 吗？

能复用匹配前缀的状态快照；难点是任意前缀边界的恢复，需要保存 checkpoint 或重算。

### Q12. 用 top-p 的数学接口没变，为什么引擎还要改？

因为分支、回滚、缓存复用和状态生命周期已经不同。概率分布后处理与模型执行状态是两层接口。

## 15. 关联与继续研究

- [[CMU 11-763]]：全课程索引。
- [[CMU 11-763 - Lecture 20 - Prefix Sharing and KV Cache Optimizations]]：KV 压缩与复用。
- [[Transformer Block]]：标准 attention 参照实现。
- [[LLM Inference]]：推理算法、架构和 serving 的交界。
- [ ] #question 固定显存预算下，hybrid 应分配多少层给全局 attention？如何按任务的 associative recall 需求测量？
- [ ] #question Prefix cache 命中收益能否覆盖 recurrent checkpoints 的额外存储？需要怎样的请求长度与分叉分布？

## 16. 来源清单

1. [官方 43 页课件](https://docs.google.com/presentation/d/19jJynlmA8wMDHOOv66ZCF7UbiG8rwzb2tl6quSdegyc/edit)：稀疏、线性、SSM 与 hybrid 主线。
2. [Mamba-2 作者说明](https://goombalab.github.io/blog/2024/mamba2-part1-model/)与[原论文](https://arxiv.org/abs/2405.21060)：SSM/attention 对偶及架构背景。
3. [Gated Delta Networks 原论文](https://arxiv.org/abs/2412.06464)：门控与 delta update 的来源。
