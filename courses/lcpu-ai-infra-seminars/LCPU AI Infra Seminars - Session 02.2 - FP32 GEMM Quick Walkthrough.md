---
type: course-note
status: developing
course: Weiming HPC Training Camp × LCPU AI Infra Seminars
session: "2.2"
speaker: 周宇轩
video_url: https://www.bilibili.com/video/BV1L8GA6YEAH/
source_status: transcript-reviewed
topics: [cuda, gemm, roofline, tiling, profiling]
---

# Session 02.2：FP32 GEMM Quick Walkthrough

> 核心问题不是“怎样让每个线程做得更快”，而是“同一份数据进入某一层存储后，能否多算一些，再离开”。

## 来源与阅读方式

- [视频](https://www.bilibili.com/video/BV1L8GA6YEAH/)，30:05；讲者周宇轩。
- [[LCPU AI Infra Seminars - Session 02.2 - FP32 GEMM - 精编字幕]]：按时间顺序回看。
- 本文依据完整字幕整理。公式展开、伪代码与实验清单是整理者为复习补充的推导，并非原幻灯片逐字复刻。
- 官网名为 `session0202.pdf` 的材料主要涉及访存/Reduce，与本视频不完全对应，未用它冒充本讲课件。

## 一、整讲的因果链

| 阶段 | 发现的问题 | 优化手段 | 代价或新瓶颈 |
|---|---|---|---|
| 一个线程算一个 C 元素 | 输入重复读取 | CTA 级 SMEM tiling | shared-memory 访问与同步 |
| CTA 共享 A/B tile | 每次 FMA 仍频繁读 SMEM | 每线程算 `TM×TN`，寄存器复用 | 寄存器占用、映射与 bank conflict |
| 多层 tiling | load 和 compute 串行 | 多缓冲、异步搬运 | buffer 生命周期与同步正确性 |

它对应两类优化：减少实际搬运量，以及让不可避免的搬运与计算重叠。二者不同，不能互相替代。

## 二、从数学工作量到 Roofline

设 `A[M,K]`、`B[K,N]`、`C[M,N]`，本讲考虑 `C=AB`，不是完整的 `αAB+βC` 接口。

$$C_{mn}=\sum_{k=0}^{K-1}A_{mk}B_{kn},\qquad F\approx2MNK.$$

一个 FMA 是一条融合乘加操作，但按常用性能统计计两个 FLOP。指令数与 FLOP 数不是同一概念。

Roofline 给出：

$$P\le\min(P_{compute},\ BW\times I).$$

这里的带宽与计算强度必须属于同一层级：讨论 HBM，就使用 HBM 实际流量；讨论 shared memory，就估计 SMEM 层的流量。不能把 L1 命中的 load 全算作 HBM 传输，再拿 HBM 带宽解释性能。

### 2.1 算法层的理想复用

忽略额外拷贝，A/B 只读一次，C 只写一次：

$$Q_{ideal}=4(MK+KN+MN),\quad I_{ideal}=\frac{2MNK}{4(MK+KN+MN)}.$$

方阵边长 n 时，`I=n/6`，随规模增长。但这个下界假设充分复用；有限片上容量、具体 tiling 和边界都会增加真实流量。

### 2.2 朴素线程实现

每个输出读 `2K` 个 FP32、写一个 FP32：

$$I_{naive}=\frac{2K}{8K+4}\approx0.25\ \text{FLOP/byte}.$$

这解释了为何同一个算法可以有很高的理论计算密度，却被一个低复用实现做成带宽瓶颈。Cache 可能挽救部分重复访问，但不应代替显式的复用设计。

## 三、CTA tiling：global → shared

令 CTA 计算 `BM×BN` 输出。每一轮 K 分块：

- 搬入 `BM×BK` 的 A 和 `BK×BN` 的 B；
- 产生 `2BM·BN·BK` FLOP；
- FP32 输入流量为 `4BK(BM+BN)` 字节。

于是，忽略 C 写回：

$$I_{CTA}=\frac{BM\,BN}{2(BM+BN)}.$$

如果 `BM=BN=T`，则 `I=T/4`。`T=32` 时约为 8 FLOP/byte，是朴素模型的 32 倍计算强度，**不等于实际性能必然提高 32 倍**。

注意 BK 在这个近似中约掉了：增大 BK 并不会自动提高这里的输入复用比，但会影响同步次数、SMEM 容量、pipeline 与边界开销。

### 两个 barrier 对应两种数据危险

```text
for k_tile:
    cooperative_load(A_tile, B_tile)
    barrier()       # 等所有生产者写完
    accumulate_from_shared()
    barrier()       # 等所有消费者读完，才能复用存储
```

第一处防止读未完成数据；第二处防止提前覆盖。双缓冲能改变组织方式，但不会消灭这两种依赖。

## 四、Coalescing 与 vectorization 不要混淆

| 维度 | Coalescing | Vectorization |
|---|---|---|
| 观察对象 | 同一条指令中，不同 lane 的地址 | 一个线程单条指令的数据宽度 |
| 核心收益 | 减少不必要的内存事务 | 减少搬运指令，增大单指令数据量 |
| 典型设计 | 相邻 lane 对应连续列 | 每线程搬 `float4` |
| 常见误区 | 每线程自己的访问连续就够了 | 强制转换成 `float4*` 必定更快且正确 |

行主序矩阵地址为 `base+(row*stride+col)*sizeof(float)`。CUDA 线程线性化中 x 维最快，通常让它对应连续的 col 更自然。实际 warp 跨行与否还取决于 block shape，不能只看变量名字。

向量化需要检查起始地址、行跨度、尾部元素及编译器实际生成的指令；它不能自动修复一个跨 lane 地址分散的设计。

## 五、Register tiling：shared → registers

一个线程保存 `TM×TN` 个 accumulator。每个 k 取 `TM` 个 A、`TN` 个 B，然后外积更新：

```text
for k:
    a[0:TM] = load_shared_A(k)
    b[0:TN] = load_shared_B(k)
    for i in 0:TM:
        for j in 0:TN:
            acc[i,j] += a[i] * b[j]
```

每步输入 `TM+TN` 个数，做 `TM×TN` 次 FMA：

$$I_{SMEM}\approx\frac{2TM\,TN}{4(TM+TN)}.$$

`TM=TN=2` 时，四次 FMA 只需要四个输入值，而四个独立输出共需要八个输入读取。更大的每线程 tile 还降低“输出数量对应线程数量”的约束，让较少线程覆盖较大的 CTA 输出 tile。

但 accumulator 数按 `TM×TN` 增长，会吃寄存器。过大的 thread tile 可能降低 occupancy，甚至造成 spill。理论复用变好，不保证整体更快。

## 六、读 profiling：从驻留数量追到阻塞原因

[19:00–23:00](https://www.bilibili.com/video/BV1L8GA6YEAH/?t=1140) 是本讲非常值得反复看的部分。

1. Active：已分配上下文、可驻留，不代表当前能发射。
2. Eligible：数据依赖与执行通路条件满足，可以参与发射选择。
3. Issued：最终被 scheduler 选择并发射。

课堂中 active warps 已接近该 A100 配置的上限，仍存在大量 stall。MIO throttle 与指令统计共同指向 SMEM 指令压力，因此 register tiling 才是对应的下一步。

> 不要只看到一个 stall 名称就机械下结论。需要同时看 kernel 访问模式、相关指令数量、资源限制和优化前后的变化。MIO 并不是任何场景下都只服务 shared memory。

## 七、Pipeline：从相加的时间走向重叠的时间

无重叠时，稳态每个 tile 的耗时近似：

$$T_{serial}\approx T_{load}+T_{compute}.$$

理想且资源可并行时，多缓冲可使稳态接近：

$$T_{overlap}\gtrsim\max(T_{load},T_{compute}).$$

这是整理者的简化模型；真实时间还有 pipeline 启动/排空、同步、共享资源争用和带宽约束。增大缓冲深度也会消耗 SMEM，不会无限提升吞吐。

异步提交不等于完成。正确实现必须确保：数据 ready 后才消费，消费结束后才覆盖。后续关联：

- [[LCPU AI Infra Seminars - Session 03 - Tensor Core 从 mma.sync 到 tcgen05]]
- [[LCPU AI Infra Seminars - Session 04 - Pipeline Ordering Data Orchestration]]

## 八、实验复现清单

- [ ] 固定 GPU、M/N/K、dtype、布局、时钟/运行环境；注明 cuBLAS 是否允许 TF32。
- [ ] 从朴素版本开始做 correctness test，包括非 tile 整数倍的边界。
- [ ] 分别加入 coalescing、SMEM tiling、vector load、register tiling；一次只改一个主要因素。
- [ ] 每个版本记录耗时、有效 TFLOP/s、寄存器/SMEM 用量、occupancy、eligible/issued 与 stall。
- [ ] 检查实际 load 指令与 bank-conflict 指标，不用源码外观替代生成代码。
- [ ] 再测试双缓冲，观察是否真的把 load 从关键路径移走。
- [ ] 至少覆盖方阵、小矩阵和长瘦矩阵；预热，多次测量，检查误差。

课堂提到的约 5–6 倍加速、约 80% cuBLAS 性能，仅保留为该实验报告；字幕不足以恢复所有 benchmark 参数，不应把它当作可直接复现的通用标尺。

## 九、自测

1. 为什么理想方阵 GEMM 的 AI 随 n 增长，而朴素线程版本接近常数？

    **面试回答：** 理想实现中，方阵 GEMM 做约 $2n^3$ FLOPs，只读 A、B 并写 C 共 $12n^2$ bytes，所以 FP32 算术强度为 $n/6$。朴素实现每算一个输出都重读一行 A 和一列 B，按未命中缓存的流量估算为 $2n/(8n+4)\approx0.25$ FLOP/byte；差别来自跨输出的数据复用。

2. CTA tiling 的公式中 BK 为什么约掉？它为什么仍然值得调参？

    **面试回答：** 每轮计算量和输入流量都正比于 BK，二者相除后得到 $I=BM\,BN/[2(BM+BN)]$。但 BK 决定 K-loop 次数、同步频率、SMEM 占用、流水窗口和边界浪费，因此仍会明显影响实际性能。

3. 两个 barrier 分别保护什么？删掉第二个会出现哪类错误？

    **面试回答：** 第一个 barrier 保证所有线程把 tile 写入 SMEM 后才开始读取，保护写后读依赖。第二个保证所有消费者读完后才能装入下一轮，保护读后写依赖；删掉它可能让快线程覆盖慢线程仍在读取的数据，得到混合了两轮 tile 的错误结果。

4. 为什么 occupancy 高仍然会出现发射不足？

    **面试回答：** Occupancy 只表示有多少 warp 驻留，不代表它们当前能发射。所有 warp 都可能在等数据、barrier 或执行管线资源；要结合 eligible warps、issued instructions 和 stall 原因判断，针对性增加独立工作或减少 SMEM 指令压力。

5. `float4` 和合并访存解决的是不是同一件事？

    **面试回答：** 不是同一件事：`float4` 扩大单线程单条访存的数据宽度，主要减少指令数量；合并访存关注同一指令中不同 lane 的地址能否合成少量事务。向量化还要求对齐和正确的尾部处理，不能自动修复跨 lane 的离散访问。

6. 为什么 register tiling 同时影响 SMEM 压力与 CTA 可覆盖的输出大小？

    **面试回答：** 每线程维护 $TM\times TN$ 个输出时，每个 k 只需读 $TM+TN$ 个输入就能做 $TM\times TN$ 次 FMA，降低单位计算的 SMEM 读取量。同样线程数也能覆盖更大的 CTA 输出 tile，但 accumulator 寄存器随 $TM\times TN$ 增长，过大可能降低驻留数或引发 spill。

7. 什么条件下双缓冲几乎没有收益，甚至会变慢？

    **面试回答：** 若没有足够独立计算可与搬运重叠、tile 数太少，或原本已受持续带宽限制，双缓冲收益就很有限。它还增加 SMEM、同步和填充/排空开销，可能降低 occupancy；应验证真实 overlap，而不是仅凭有两个 buffer 判断提速。


一句话复述：先通过 global → shared → registers 的分层复用减少搬运，再以受控的异步流水隐藏剩余搬运；每一步都要由 profiling 验证。
