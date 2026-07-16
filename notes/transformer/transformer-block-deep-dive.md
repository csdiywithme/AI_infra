# Transformer Block 深入理解：Attention、FFN、Norm 与 Residual

## 1. 核心心智模型

Transformer block 反复完成两件事：

> **Attention 在 token 之间传递信息；FFN 在单个 token 内加工信息。**

Norm 负责稳定模块输入的数值尺度，Residual 则保存并累积每个模块产生的更新。

可以把 residual stream 想象成一块贯穿所有层的共享黑板：

```text
Residual Stream：[B, L, d]
        │
        ├─ Norm：整理数值尺度
        ├─ Attention：从其他 token 读取信息
        └─ Residual Add：把读取结果写回主干
        │
        ├─ Norm：再次整理数值尺度
        ├─ FFN：每个 token 独立加工特征
        └─ Residual Add：把加工结果写回主干
```

现代 Pre-Norm block 可以写成：

\[
X_1=X+\operatorname{Attention}(\operatorname{Norm}(X))
\]

\[
X_2=X_1+\operatorname{FFN}(\operatorname{Norm}(X_1))
\]

整个 block 的主干形状始终保持为：

\[
[B,L,d]
\]

Attention 的 output projection 和 FFN 的 down projection 最终都必须恢复到 \(d\) 维，才能与 residual stream 相加。

## 2. 符号与输入形状

- \(B\)：batch size；
- \(L\)：sequence length；
- \(d\)：model hidden size；
- \(H_q\)：Query attention heads 数；
- \(H_{kv}\)：Key/Value heads 数；
- \(d_h=d/H_q\)：每个 Query head 的维度；
- \(d_{ff}\)：FFN 中间层维度；
- \(V\)：词表大小。

输入 Transformer block：

\[
X\in\mathbb R^{B\times L\times d}
\]

它表示 \(B\) 个序列，每个序列有 \(L\) 个 token，每个 token 使用 \(d\) 维向量表示。

实际实现可能使用 `[B, L, H, d_h]` 或 `[B, H, L, d_h]` 等不同内存布局，但逻辑含义相同。

## 3. Attention 与 FFN 的本质区别

### 3.1 Attention 沿 sequence 维度混合

Attention 让当前位置从其他 token 读取信息：

```text
token 1 ─┐
token 2 ─┼─→ token 4 的上下文化表示
token 3 ─┤
token 4 ─┘
```

其核心矩阵乘法之一是：

\[
A_{L\times L}V_{L\times d_h}
\]

左侧 \(L\times L\) 矩阵决定不同 token 如何混合，因此 Attention 主要沿 \(L\) 轴交换信息。

### 3.2 FFN 沿 hidden 维度混合

FFN 对每个 token 独立执行：

```text
token 1: [d] → FFN → [d]
token 2: [d] → FFN → [d]
token 3: [d] → FFN → [d]
```

其核心矩阵乘法是：

\[
X_{L\times d}W_{d\times d_{ff}}
\]

每一行独立乘以同一组参数，因此 FFN 不交换 token 之间的信息，主要沿 \(d\) 轴加工特征。

可以简化为：

\[
\boxed{\text{Attention mixes tokens; FFN mixes features}}
\]

## 4. Q、K、V 的作用

Attention 需要解决两个问题：

1. 当前 token 应该从哪里读取信息？
2. 匹配成功后，应该传递什么信息？

输入 \(X\) 经过三个可学习的 projection：

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]

### 4.1 Query：当前位置需要什么

Query 是当前位置发出的查询，可以理解为：

> 我现在需要寻找什么信息？

例如代词 `she` 的 Query 可能需要寻找前文中的人、女性实体和合适的语法主体。

### 4.2 Key：每个 token 如何被匹配

Key 是历史 token 暴露出来的索引或标签，可以理解为：

> 我具有什么特征，应该如何被其他 token 找到？

例如人名 `Alice` 的 Key 可以编码“人、女性实体、句子主体”等匹配特征。

### 4.3 Value：真正传递的内容

Value 是匹配成功后传递的 payload，可以包含 token 当前的语义和上下文状态。

因此：

\[
\boxed{QK^T\text{ 决定信息路由，}V\text{ 携带信息内容}}
\]

Q/K 产生的 score 不是最终输出，它只决定如何加权读取 V。

### 4.4 为什么不能直接令 Q=K=V=X

独立的 \(W_Q,W_K,W_V\) 允许模型分别学习：

- 如何表达当前位置的查询需求；
- 如何表达历史 token 的可匹配特征；
- 如何表达真正需要传递的内容。

Q 和 K 使用不同 projection 还允许 attention 关系不对称：A 想读取 B，并不意味着 B 也同样想读取 A。

## 5. Single-Head Attention 的数据流

先忽略多头，令：

\[
Q,K,V\in\mathbb R^{B\times L\times d}
\]

### 5.1 计算匹配分数

\[
S=\frac{QK^T}{\sqrt d}
\]

形状变化：

\[
[B,L,d]\times[B,d,L]\rightarrow[B,L,L]
\]

其中 \(S_{ij}\) 表示第 \(i\) 个 Query token 对第 \(j\) 个 Key token 的匹配程度。

除以 \(\sqrt d\) 是为了避免 hidden dimension 增大时点积尺度过大，使 softmax 过度尖锐。

### 5.2 Causal Mask

Decoder-only 模型不能看到未来 token，因此使用：

\[
M_{ij}=
\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases}
\]

然后：

\[
A=\operatorname{softmax}(S+M)
\]

由于 \(e^{-\infty}=0\)，未来位置经过 softmax 后权重为 0。

Mask 没有可学习参数，也不改变 tensor shape，只限制信息流动方向。

### 5.3 Softmax 维度

Softmax 在 Key 维度，即最后一个维度执行：

\[
\sum_j A_{ij}=1
\]

含义是：对于每一个 Query token，把分配给所有可见 Key token 的读取权重归一化。

### 5.4 读取 Value

\[
O=AV
\]

形状为：

\[
[B,L,L]\times[B,L,d]\rightarrow[B,L,d]
\]

对第 \(i\) 个 token：

\[
o_i=\sum_jA_{ij}v_j
\]

因此完整逻辑是：

```text
Q：我需要什么？
K：谁适合被读取？
QKᵀ：每个历史 token 与我有多相关？
Softmax：把相关性变成读取权重
V：每个 token 真正携带的内容
AV：把相关信息聚合到当前位置
```

## 6. Multi-Head Attention

单个 head 只有一种匹配空间。Multi-head attention 将 hidden dimension 拆成多个子空间：

\[
d=H_qd_h
\]

例如：

\[
d=4096,\qquad H_q=32,\qquad d_h=128
\]

逻辑形状：

\[
Q,K,V:[B,H_q,L,d_h]
\]

每个 head 独立计算：

\[
S_h=Q_hK_h^T
\]

所有 attention scores：

\[
S:[B,H_q,L,L]
\]

每个 head 输出 `[B, L, d_h]`，拼接后恢复：

\[
\operatorname{Concat}(O_1,\ldots,O_H):[B,L,d]
\]

最后经过 output projection：

\[
O=\operatorname{Concat}(O_1,\ldots,O_H)W_O
\]

其中 \(W_O:[d,d]\)，最终输出仍为 `[B, L, d]`，从而可以与 residual stream 相加。

虽然有 \(H_q\) 个 heads，但每个 head 只有 \(d_h=d/H_q\) 维，因此总宽度仍是：

\[
H_qd_h=d
\]

实现时通常用一次大矩阵乘法得到 Q/K/V，再 reshape 为多个 heads，而不是执行 \(H_q\) 次完整的 \(d\times d\) projection。

## 7. RoPE 与 Q/K 的关系

RoPE 的作用是让 attention routing 感知位置关系。

Q/K 决定当前位置应该读取哪个历史位置，所以位置信息直接影响 Q/K 的匹配；V 负责携带内容，不负责决定地址，因此 RoPE 通常只作用在 Q 和 K 上。

可以理解为：

- Q/K：包含地址与相对位置信息；
- V：匹配地址后读取的 payload。

RoPE 不改变 shape：

\[
[B,H,L,d_h]\rightarrow[B,H,L,d_h]
\]

## 8. MHA、MQA 与 GQA

### 8.1 MHA

\[
H_q=H_{kv}=H
\]

每个 Query head 有独立的 K/V head，表达能力强，但 KV Cache 最大。

### 8.2 MQA

\[
H_{kv}=1
\]

所有 Query heads 共享一组 K/V，KV Cache 最小，但共享程度最高。

### 8.3 GQA

\[
1<H_{kv}<H_q
\]

例如：

\[
H_q=32,\qquad H_{kv}=8
\]

每 4 个 Query heads 共享一个 K/V head。

逻辑形状：

\[
Q:[B,H_q,L,d_h]
\]

\[
K,V:[B,H_{kv},L,d_h]
\]

GQA 主要减少：

- K/V projection；
- KV Cache 容量；
- Decode 阶段的 HBM 读取量；
- 单请求显存占用。

## 9. FFN 的作用

Attention 完成信息读取后，FFN 对每个 token 的特征进行非线性加工。

可以粗略类比为：

```text
Attention：查找并读取相关资料
FFN：拿到资料后在当前位置加工特征
```

这只是便于理解的心智模型，并不代表 FFN 存在人类式思考。

### 9.1 为什么 FFN 对每个 token 独立

经典 FFN：

\[
\operatorname{FFN}(x)=\sigma(xW_1)W_2
\]

其中：

\[
W_1:[d,d_{ff}],\qquad W_2:[d_{ff},d]
\]

整体形状：

\[
[B,L,d]\rightarrow[B,L,d_{ff}]\rightarrow[B,L,d]
\]

矩阵乘法只作用在最后一个维度，因此所有 token 共享同一组参数，但彼此之间不在 FFN 中交换信息。

### 9.2 为什么先扩维再缩维

通常：

\[
d_{ff}>d
\]

扩维允许模型在更高维空间中构造更多非线性特征，经过激活与门控后再压回 residual stream 的宽度 \(d\)。最终恢复到 \(d\) 是为了执行 residual add。

### 9.3 SwiGLU 为什么有三个矩阵

现代 Llama-style FFN 常使用：

\[
\operatorname{SwiGLU}(x)=
\left[
\operatorname{SiLU}(xW_{gate})\odot xW_{up}
\right]W_{down}
\]

形状变化：

```text
Gate: [B,L,d] × [d,d_ff] → [B,L,d_ff]
Up:   [B,L,d] × [d,d_ff] → [B,L,d_ff]
SiLU(Gate) ⊙ Up           → [B,L,d_ff]
Down: [B,L,d_ff] × [d_ff,d] → [B,L,d]
```

Gate 控制哪些特征应该通过以及通过多少，Up 产生候选特征，Down 将结果恢复到 residual stream 的宽度。

### 9.4 Hidden size 与 MLP size

`hidden_size`（\(d\)）是 residual stream 的宽度。每个 token 在 Transformer 主干中始终使用 \(d\) 维表示：

\[
X:[B,L,d]
\]

`mlp_size` 通常也叫 `intermediate_size` 或 \(d_{ff}\)，是 token 进入 FFN 后临时扩展到的中间维度：

\[
[B,L,d]
\rightarrow
[B,L,d_{ff}]
\rightarrow
[B,L,d]
\]

因此，MLP size 不是模型参数量，也不是贯穿所有 block 的主干宽度。Down projection 必须把中间状态恢复为 \(d\)，才能与 residual stream 相加。

### 9.5 MLP size 对容量和系统成本的影响

如 9.2 所述，扩维为每个 token 提供更大的非线性特征空间。具体选择多大的 \(d_{ff}\)，则是模型容量与系统成本之间的权衡。

增大 \(d_{ff}\) 通常会增强单层的逐 token 计算容量，但同时线性增加：

- FFN 参数量；
- FFN FLOPs；
- 权重内存与内存流量；
- 并行切分与 kernel shape 的设计压力。

因此，\(d_{ff}/d\) 是模型质量与计算预算之间的架构超参数，不存在必须等于某个常数的数学定理。

### 9.6 SwiGLU 的 \(8d/3\) 从哪里来

经典 ReLU/GELU FFN 常使用 \(d_{ff}=4d\)，包含两个矩阵：

\[
W_1:[d,4d],\qquad W_2:[4d,d]
\]

忽略 bias，其参数量约为：

\[
d(4d)+(4d)d=8d^2
\]

SwiGLU 包含 Gate、Up、Down 三个矩阵，参数量约为：

\[
3dd_{ff}
\]

若希望 SwiGLU 与经典 \(4d\) FFN 的参数量和主要计算量大致相同，可以令：

\[
3dd_{ff}\approx8d^2
\]

得到：

\[
\boxed{d_{ff}\approx\frac83d\approx2.67d}
\]

这就是早期 LLaMA 使用“大约三倍 hidden size”的经典来源。实际实现还会将中间维度对齐到 256 等硬件友好的整数倍。

例如 \(d=4096\) 时：

\[
\frac83\times4096\approx10922.7
\]

对齐后可取 \(d_{ff}=11008\)，实际比例约为 2.6875。

### 9.7 Llama 3 的比例为什么不同

\(8d/3\) 只是保持经典 FFN 参数预算的基线，不是架构约束。Llama 3 系列根据总体参数预算和训练实验选择了更宽的 FFN。

| 模型 | Hidden size \(d\) | MLP size \(d_{ff}\) | 比例 |
| --- | ---: | ---: | ---: |
| Llama 3.1 8B | 4,096 | 14,336 | 3.50 |
| Llama 3.1 70B | 8,192 | 28,672 | 3.50 |
| Llama 3.1 405B | 16,384 | 53,248 | 3.25 |

因此“MLP size 约为 hidden size 三倍”只是经验性描述。具体比例还受到模型深度、总参数量、训练质量和硬件对齐要求影响。

以 8B 配置为例，单层 SwiGLU MLP 的参数量约为：

\[
3\times4096\times14336\approx176\text{M}
\]

32 层仅 MLP 权重就约为：

\[
176\text{M}\times32\approx5.64\text{B}
\]

这说明 dense LLM 的大部分参数和短上下文主要计算通常位于 MLP。需要注意，MLP 的中间 activation 不进入 KV Cache；KV Cache 只保存各 attention 层的历史 K/V。

## 10. Norm 的作用

随着层数增加，hidden state 的尺度可能不断变化。Norm 在模块读取 residual stream 之前，将每个 token 的特征调整到相对稳定的数值范围。

Norm 对每个 token 的 hidden dimension 独立执行：

\[
[B,L,d]\rightarrow[B,L,d]
\]

它不会混合不同 token，也不会改变序列长度或 hidden size。

### 10.1 LayerNorm

对于一个 token \(x\in\mathbb R^d\)：

\[
\mu=\frac1d\sum_i x_i
\]

\[
\sigma^2=\frac1d\sum_i(x_i-\mu)^2
\]

\[
\operatorname{LayerNorm}(x)
=
\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
\]

### 10.2 RMSNorm

RMSNorm 不减均值，只控制均方根尺度：

\[
\operatorname{RMS}(x)
=
\sqrt{\frac1d\sum_i x_i^2+\epsilon}
\]

\[
\operatorname{RMSNorm}(x)
=
\gamma\odot\frac{x}{\operatorname{RMS}(x)}
\]

其计算复杂度约为 \(O(BLd)\)，远低于大型矩阵乘法。但实际 GPU 上仍可能受到 HBM 读写和 kernel launch 影响，因此 FLOPs 少不代表 latency 可以完全忽略。

## 11. Residual 的作用

Attention 和 FFN 更适合理解成对主干状态的增量更新：

\[
\Delta X_{attn}
=
\operatorname{Attention}(\operatorname{Norm}(X))
\]

\[
X_1=X+\Delta X_{attn}
\]

\[
\Delta X_{ffn}
=
\operatorname{FFN}(\operatorname{Norm}(X_1))
\]

\[
X_2=X_1+\Delta X_{ffn}
\]

Residual connection 的作用包括：

- 保留原始信息；
- 让每层只需学习增量更新；
- 改善深层网络中的梯度传播；
- 提供固定宽度 \(d\) 的主干表示。

## 12. 模块关系总表

| 模块 | 主要作用 | 混合 token | 混合 feature | 最终 shape |
| --- | --- | ---: | ---: | --- |
| Norm | 稳定输入尺度 | 否 | 对 feature 做统计和缩放 | 不变 |
| Attention | 从其他 token 读取信息 | 是 | 是 | `[B,L,d]` |
| FFN | 每个 token 内加工特征 | 否 | 是 | `[B,L,d]` |
| Residual | 保存并累积模块更新 | 否 | 否 | 不变 |
| RoPE | 为 attention routing 注入位置关系 | 不直接混合 | 旋转 Q/K 特征 | 不变 |

## 13. 从 Shape 推导运算量

不需要背诵所有公式，只需要记住矩阵乘法规则：

\[
[m,k]\times[k,n]\rightarrow[m,n]
\]

参数量：

\[
kn
\]

FLOPs 约为：

\[
2mkn
\]

因为每个输出元素需要大约 \(k\) 次乘法和 \(k\) 次加法。

下面均保留 batch size \(B\)。

### 13.1 MHA Projection FLOPs

Q/K/V projection 均为：

\[
[BL,d]\times[d,d]\rightarrow[BL,d]
\]

每个成本：

\[
2BLd^2
\]

Output projection 同样是：

\[
2BLd^2
\]

所以标准 MHA 的 Q/K/V/O projections 合计：

\[
\boxed{8BLd^2}
\]

### 13.2 Attention Matrix FLOPs

每个 head 的 \(QK^T\)：

\[
[L,d_h]\times[d_h,L]\rightarrow[L,L]
\]

全部 heads：

\[
2BH_qL^2d_h=2BL^2d
\]

\(AV\) 同样需要：

\[
2BL^2d
\]

合计：

\[
\boxed{4BL^2d}
\]

\(L^2\) 来自 \(QK^T\) 与 \(AV\) 两个矩阵乘法，不是需要单独死记的结论。

### 13.3 GQA Projection FLOPs

令：

\[
d_{kv}=H_{kv}d_h
\]

Q 和 O projection 仍各需要：

\[
2BLd^2
\]

K/V projection 各需要：

\[
2BLd\,d_{kv}
\]

因此 GQA 主要减少 K/V projection 和 KV Cache，不会把所有 attention arithmetic 都按 \(H_q/H_{kv}\) 同比例缩小。

### 13.4 SwiGLU FLOPs

Gate、Up、Down 三个矩阵乘法各约为：

\[
2BLdd_{ff}
\]

合计：

\[
\boxed{6BLdd_{ff}}
\]

### 13.5 Norm

Norm 约为：

\[
O(BLd)
\]

理论 FLOPs 较少，但常包含多次 element-wise 操作与 reduction，实际性能需要同时考虑内存流量和 kernel fusion。

## 14. Prefill 与 Decode

### 14.1 Prefill

一次处理整个 prompt：

\[
Q:[B,H_q,L,d_h]
\]

\[
K,V:[B,H_{kv},L,d_h]
\]

Attention score：

\[
[B,H_q,L,L]
\]

因此完整序列 attention 包含 \(L^2\) 计算。

### 14.2 Decode

使用 KV Cache 后，每一步只有一个新 Query：

\[
Q_{new}:[B,H_q,1,d_h]
\]

历史缓存：

\[
K_{cache},V_{cache}:[B,H_{kv},L,d_h]
\]

Attention score：

\[
[B,H_q,1,L]
\]

所以：

- Prefill 总 attention 计算具有 \(L^2\) 特征；
- 单步 Decode attention 随历史长度 \(L\) 线性增长；
- Decode 需要反复读取模型权重和 KV Cache；
- Decode 更容易受到 memory bandwidth 限制。

## 15. 完整数据流

一个 token 的表示穿过 Transformer block 时：

1. Residual stream 保存当前位置的表示；
2. Norm 稳定数值尺度；
3. Q 表达当前位置需要什么；
4. K 表达历史 token 如何被匹配；
5. \(QK^T\) 决定读取权重；
6. V 携带真正需要传递的内容；
7. Attention 聚合历史信息；
8. Output projection 恢复到 \(d\) 维；
9. Residual add 将信息写回主干；
10. Norm 再次稳定尺度；
11. FFN 在每个 token 内进行非线性特征加工；
12. Down projection 恢复到 \(d\) 维；
13. Residual add 再次写回主干；
14. 下一层继续读取和加工。

可以最终压缩为：

\[
\boxed{\text{Norm：稳定模块输入}}
\]

\[
\boxed{\text{Q/K：决定从哪里读取}}
\]

\[
\boxed{\text{V：决定读取什么内容}}
\]

\[
\boxed{\text{Attention：跨 token 通信}}
\]

\[
\boxed{\text{FFN：单 token 内计算}}
\]

\[
\boxed{\text{Residual：保存并累积更新}}
\]

## 16. 自测问题

1. 为什么 \(QK^T\) 的结果是 \(L\times L\)？
2. 为什么 softmax 在 Key 维度执行？
3. 为什么 Q/K 决定信息路由，而 V 携带内容？
4. 为什么 RoPE 通常作用在 Q/K 而不是 V？
5. Attention 与 FFN 分别沿哪个维度混合信息？
6. 为什么 Attention 和 FFN 最终都要恢复到 \(d\) 维？
7. 为什么 SwiGLU 有 Gate、Up、Down 三个 projection？
8. 为什么 MHA 拆成多个 heads 后 projection 总宽度仍然是 \(d\)？
9. GQA 主要减少哪些计算和内存成本？
10. 为什么 Prefill 和 Decode 的 attention shape 不同？
11. 为什么 Decode 更容易 memory-bound？
12. 如何从矩阵 shape 现场推导参数量和 FLOPs？
13. Hidden size 与 MLP/intermediate size 分别表示什么？
14. 为什么经典 SwiGLU 中间维度约为 \(8d/3\)？
15. 为什么实际 Llama 配置不必严格遵循 \(8d/3\)？

## 17. 参考资料

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [Meta Llama 3 官方实现](https://github.com/meta-llama/llama3/blob/main/llama/model.py)
- [CMU 11-763 Lecture 01 讲义](https://www.phontron.com/class/lminference-fall2025/assets/slides/2025-08-26-lm-intro/index.html)
