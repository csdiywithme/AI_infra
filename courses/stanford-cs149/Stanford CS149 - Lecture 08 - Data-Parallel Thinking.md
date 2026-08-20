---
type: course-note
status: developing
course: "[[Stanford CS149]]"
lecture: 8
lecture_date: 2023-10-19
area: systems
topics:
  - data-parallelism
  - map-reduce
  - scan
  - segmented-scan
  - gather-scatter
  - sparse-computing
  - algorithmic-complexity
aliases:
  - CS149 Lecture 8
  - Data-Parallel Thinking
video_url: https://www.youtube.com/watch?v=Ba3TqxSgnTk
---

# Stanford CS149 - Lecture 08 - Data-Parallel Thinking

> [!abstract]
> 本讲把思考单位从“每个 worker/thread 做什么”切换为“对整个 sequence 做什么”。Map、fold/reduce、scan、segmented scan、gather/scatter、sort 与 groupBy 把依赖结构封装进少数并行 primitives；算法只要组合这些 primitives，就能复用经过优化的 CPU/GPU/cluster 实现。课程通过 parallel scan、CSR sparse matrix-vector multiply 和 particle binning 说明：理论 work/span 只是第一层，真正高效的实现还必须贴合 SIMD utilization、memory hierarchy、communication 与机器规模。

## 来源与范围

- [Lecture 8 视频：Data-Parallel Thinking](https://www.youtube.com/watch?v=Ba3TqxSgnTk)
- [官方课件 PDF](https://gfxcourses.stanford.edu/cs149/fall23content/media/dataparallel/08_dataparallel.pdf)
- [CS149 Fall 2023 课程主页](https://gfxcourses.stanford.edu/cs149/fall23/)

本文以视频实际讲解为主。课件尾部还有 histogram 等参考例子，视频在约 75 分钟处明确说未展开；本文只把它作为 particle binning 的自然延伸，不把课件附页当作课堂主线。

## 视频索引

| 时间 | 内容 |
|---|---|
| [00:42](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=42s) | 从 thread-centric 转向 operations-on-sequences |
| [01:45](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=105s) | GPU 要求十万量级可并行工作 |
| [02:44](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=164s) | 依赖图与“调用高并行 primitive”的思想 |
| [04:56](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=296s) | Sequence：只能通过受限操作访问的有序集合 |
| [06:43](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=403s) | `map` 的定义、类型与 side-effect-free 条件 |
| [10:33](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=633s) | `map` 为什么天然可并行 |
| [13:18](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=798s) | `fold` / reduction |
| [15:04](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=904s) | Parallel fold 为什么要求 associativity |
| [18:36](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=1116s) | Map + fold fusion 与 JIT transformation |
| [19:59](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=1199s) | Scan、inclusive 与 exclusive |
| [25:23](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=1523s) | `O(N log N)` work、`O(log N)` span 的 scan |
| [28:24](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=1704s) | Work-efficient upsweep/downsweep scan |
| [31:17](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=1877s) | 理论 work-efficient 不等于最快实现 |
| [32:19](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=1939s) | 两核 CPU 上的分块顺序 scan |
| [34:04](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=2044s) | Warp/SIMD scan 的五步实现 |
| [37:22](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=2242s) | 为什么 `N log N` work 在 32-wide SIMD 上反而更快 |
| [40:45](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=2445s) | 128/1024 元素的 hierarchical scan |
| [45:05](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=2705s) | Segmented scan 与 irregular nested sequences |
| [50:04](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=3004s) | CSR sparse matrix-vector multiply |
| [57:02](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=3422s) | Gather/scatter 与不规则内存访问 |
| [60:12](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=3612s) | 用 sort + map + segmented scan 实现 scatter-op |
| [63:00](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=3780s) | Filter、groupBy 等更多序列算子 |
| [64:03](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=3843s) | 粒子 uniform-grid 构建问题 |
| [66:00](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=3960s) | Lock、per-cell lock、replication 等初始方案 |
| [69:35](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=4175s) | 换并行轴仍会造成 `cells × particles` 总工作 |
| [70:49](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=4249s) | Map + sort + map 的无锁数据并行方案 |
| [76:07](https://www.youtube.com/watch?v=Ba3TqxSgnTk&t=4567s) | 总结：把 irregular parallelism 归约到 regular primitives |

## 1. 为什么要换一种方式描述并行算法

前几讲常从 worker 出发：

```text
thread 0 处理哪一段？
thread 1 处理哪一段？
锁放在哪里？
怎样把任务发给 workers？
```

本讲改成从 collection 出发：

```text
把函数作用到所有元素（map）
把一列元素归约成一个值（fold/reduce）
求所有前缀结果（scan）
按 key 重排并分组（sort/groupBy）
```

假设系统已经有这些 primitives 的高质量并行实现，那么算法只需组合它们，便能获得大规模并行性，而不必在每个应用里重新手写 thread scheduling 和 synchronization。

这正是 NumPy、tensor compiler、GPU primitives library 和后续 Spark 的共同思想。

## 2. Sequence：用受限访问换取可推理性

本讲定义 sequence 为有序元素集合：

$$
S=[s_0,s_1,\ldots,s_{N-1}]
$$

它与普通 array 的关键差别不在存储布局，而在接口：程序不能随意写 `S[i]`，只能通过规定的 sequence operations 操作元素。

为什么这种限制有价值？

- 任意 load/store 可能引入难以发现的 cross-iteration dependency；
- operation 的契约能明确哪些元素独立、哪些结果必须组合；
- runtime 可在不改变语义的情况下 reorder、partition、vectorize、distribute；
- compiler 能识别 operation chain，做 fusion、eliminate materialization 等变换。

> [!important]
> Data-parallel abstraction 不是声称“所有算法没有依赖”，而是把依赖限制在少数具有已知结构的 primitives 内。Map 没有元素间依赖；scan 有前缀依赖，但库作者知道如何并行实现它。

## 3. Primitive 总览

| Primitive | 输入 → 输出 | 主要语义 | 常见用途 |
|---|---|---|---|
| `map` | sequence → sequence | 对每个元素独立应用函数 | elementwise op、projection |
| `filter` | sequence → shorter sequence | 保留满足 predicate 的元素 | compaction、selection |
| `fold/reduce` | sequence → scalar | 用 binary op 合并全部元素 | sum、max、norm |
| `scan` | sequence → sequence | 输出每个前缀的 reduction | offset、rank、allocation |
| segmented scan | nested/flagged sequence → sequence | 每个 segment 独立 scan | ragged data、CSR |
| `gather` | indices + source → dense result | 按 index 取值 | embedding lookup、indirection |
| `scatter` | indices + values → destination | 按 index 写值 | histogram、routing |
| `sort` | sequence → ordered sequence | 按 key 排列 | grouping、canonicalization |
| `groupBy` | `(key,value)` → key + values | 相同 key 聚集 | aggregation、inverted index |
| `join` | keyed sequences → matched records | 按 key 关联 | database/data analytics |

高层表达简洁不代表一定快。课件在主题页就加了限定：若整个 operation chain 被 memory bandwidth 限制，暴露再多 parallelism 也不能越过 bandwidth roof。

## 4. Map：最直接的数据并行算子

类型签名：

$$
map : (a\rightarrow b)\rightarrow Seq\ a\rightarrow Seq\ b
$$

例如：

```text
input = [3, 8, 4, 6]
f(x) = x + 10
map(f, input) = [13, 18, 14, 16]
```

只要 `f` 是 side-effect free，`f(s_i)`：

- 只读当前输入元素；
- 不修改共享外部状态；
- 不依赖其他 invocation 的执行顺序。

于是 map 可任意 partition：

```text
split S into S0...SP-1
parallel for each partition i:
    Oi = map(f, Si)
concatenate O0...OP-1 preserving sequence order
```

### 4.1 为什么 purity 很重要

若 `f` 偷偷执行：

```cpp
counter++;
output = x + counter;
```

不同 elements 之间就通过 `counter` 产生 dependency，reorder 后结果改变。Map 的并行保证来自契约，不来自函数名字。

### 4.2 Map 可以改变元素类型

`f` 不必是 `int→int`：

```text
int → string
particle → cell_id
token → (expert_id, token_id)
```

输入输出 sequence 长度相同，但元素类型可以不同。

## 5. Fold / reduce：何时可以并行重组

Fold left 的概念是：

```text
acc0 = init
acc1 = f(acc0, s0)
acc2 = f(acc1, s1)
...
```

一般 `foldLeft` 强制顺序依赖，不能任意变成 reduction tree。若 binary operator `⊕` 满足结合律：

$$
(a\oplus b)\oplus c=a\oplus(b\oplus c)
$$

则可以分块归约：

```text
S = S0 || S1 || ... || SP-1
ri = reduce(⊕, Si)       // partitions parallel
result = reduce(⊕, [r0, r1, ..., rP-1])
```

### 5.1 Associative，不一定要 commutative

结合律允许改变括号；交换律允许改变元素顺序。

若 implementation 保持 partition 顺序，只改变 parenthesization，`⊕` 可以不满足 commutativity。字符串拼接就是典型例子：

```text
("a" + "b") + "c" = "a" + ("b" + "c")
但 "a" + "b" ≠ "b" + "a"
```

所以 parallel fold 的精确条件不是“op 一定可交换”，而是重组方式必须符合 op 的代数性质与 API 顺序语义。

### 5.2 浮点加法的现实问题

数学实数加法结合，但 IEEE floating-point 加法不严格结合：

$$
(a+b)+c\ne a+(b+c)
$$

Parallel reduction tree 可能得到与 sequential fold 不同的末位结果。它通常仍可接受，但需要区分：

- 数值上近似正确；
- bitwise deterministic；
- 跨 device/thread-count reproducible。

训练中的 gradient reduction、指标聚合都可能遇到这一点。

## 6. Fusion：组合 primitives 不应必然产生中间数组

假设先 map 再 reduce：

```text
tmp = map(x → 10*x, S)
result = reduce(+, tmp)
```

逐字执行会：

1. 读 `S`；
2. 写完整 `tmp`；
3. 再读 `tmp`；
4. 求和。

若 compiler 知道 map/reduce 语义，可变换成：

```text
result = reduce((acc, x) → acc + 10*x, S)
```

这消除一个 intermediate sequence 和一次额外 traversal。PyTorch/XLA/other JIT systems 的 kernel fusion 就建立在类似 algebraic knowledge 上。

> [!note]
> 高层 primitives 提供的是优化机会，不自动保证消除 materialization。Eager runtime 若逐算子启动 kernel，仍可能把每个中间 tensor 写到 HBM。

## 7. Scan：输出所有前缀归约

给定 associative operator `⊕`。

Inclusive scan：

$$
y_i=x_0\oplus x_1\oplus\cdots\oplus x_i
$$

Exclusive scan：

$$
y_i=e\oplus x_0\oplus\cdots\oplus x_{i-1}
$$

其中 `e` 是 identity。例如加法的 exclusive scan：

```text
input:     [3, 8, 4, 6]
exclusive: [0, 3, 11, 15]
inclusive: [3, 11, 15, 21]
```

Scan 与 fold 的关系：inclusive scan 的最后一个元素等于整个 sequence 的 fold，但 scan 保留每一个 partial result。

### 7.1 Scan 为什么重要

Prefix sum 可把局部 decision 变成全局位置：

```text
predicate per element
→ 0/1 flags
→ exclusive scan
→ 每个保留元素的唯一 compacted output offset
```

因此 scan 是 stream compaction、parallel allocation、radix sort、histogram offsets、ragged tensor offsets 和 routing 的基础 building block。

## 8. 用 work/span 分析 parallel scan

定义：

- Work `W(N)`：所有 processors 总共执行的操作数；
- Span `S(N)`：无限 processors 下仍不可避免的最长依赖链；
- 平均可用 parallelism 约为 `W/S`；
- 在 `P` processors 上，理想时间至少满足：

$$
T_P\ge \max\left(\frac{W}{P},S\right)
$$

### 8.1 Hillis–Steele 风格：work-inefficient，但 span 小

每轮把距离 `1,2,4,...` 的前缀合入当前元素：

```text
step 0: y[i] += y[i-1]
step 1: y[i] += y[i-2]
step 2: y[i] += y[i-4]
...
```

共 `log₂N` rounds，每轮约 `N` 个 active-element operations：

$$
W(N)=\Theta(N\log N),\qquad S(N)=\Theta(\log N)
$$

Sequential scan 只有 `Θ(N)` work，所以该算法为换取大规模 parallelism 做了渐近更多工作。

### 8.2 Blelloch 风格：upsweep + downsweep

Work-efficient scan 分两阶段：

1. **Upsweep/reduce tree**：逐层计算区间 totals；
2. **Downsweep**：把每个区间的 base prefix 向子区间传播。

每层工作量形成几何级数：

$$
\frac N2+\frac N4+\cdots< N
$$

两个阶段总 work 仍为 `Θ(N)`，span 约 `2log₂N`：

$$
W(N)=\Theta(N),\qquad S(N)=\Theta(\log N)
$$

从 asymptotic analysis 看它更优：与 sequential algorithm 同阶 work，同时保持 logarithmic span。

## 9. 为什么 work-efficient scan 不一定实际更快

理论只告诉我们 operations 与 dependency depth，没有告诉我们 operations 怎样映射到执行资源。

### 9.1 两核 CPU：分块顺序 scan 更自然

对两个 cores，可以：

1. 两核分别顺序 scan 左右一半；
2. 取左半最后的 total 作为右半 base；
3. 并行把 base 加到右半所有输出。

它有：

- 近似 `1.5N` work；
- 连续 streaming access；
- 很少的 cross-core communication；
- perfect first-phase load balance。

相对复杂 tree algorithm，它更符合 cache locality 和少核机器的资源规模。

### 9.2 32-wide SIMD：`N log N` 反而可能更快

对一个 32-lane warp，Hillis–Steele scan 只需 5 条加法阶段：

```text
offset 1
offset 2
offset 4
offset 8
offset 16
```

虽然总 scalar work 是 `NlogN`，但每个阶段恰好映射为一个 wide SIMD instruction。近似执行时间是 5 steps。

Work-efficient upsweep/downsweep 需要约 10 levels，而且越靠近 tree root，active lanes 越少：

```text
32 active → 16 → 8 → 4 → 2 → 1
```

它做的 scalar work 更少，却需要更多 instructions，且 SIMD lanes 长时间被 mask。于是：

```text
work-inefficient SIMD scan: 5 wide steps, lanes mostly useful
work-efficient tree scan:  ~10 steps, increasingly sparse lane use
```

> [!important]
> “少做工作”只有在释放的资源能让这一阶段更快、或让其他工作使用时才转化为更短 wall time。若硬件无论如何都发射一个 wide instruction，屏蔽 31 lanes 不会让该 instruction 变成 1/32 的时间。

## 10. Hierarchical scan：让算法匹配机器层级

对 128 elements、4 warps：

1. 每个 warp 并行 scan 自己的 32 elements；
2. 每个 warp 产生一个 total；
3. 一个 warp scan 这 4 个 totals，得到各 warp bases；
4. 所有 warps 并行把 base 加回本地结果。

```text
four local 32-element scans
        ↓ four totals
scan totals
        ↓ four bases
parallel base add
```

对 1024 elements，同样先做 32 个 warp scans，再 scan 32 个 totals，再 add bases。视频按理想 SIMD steps 给出直觉：约 `5 + 5 + 1 = 11` steps 完成 1024-element block scan。

更大 sequence 再扩展到 blocks：

```text
kernel 1: each block scans its tile and emits block total
kernel 2: scan block totals
kernel 3: add block bases to tiles
```

这体现 heterogeneous algorithm：

- warp 内用 SIMD-friendly `NlogN` scan；
- block 内组合 warp partials；
- blocks 间用多次 kernel launch 形成 global phase boundaries；
- CPU 少核上又可能使用顺序分块版本。

同一个 primitive 的语义不变，实现随机器层级改变。

## 11. Segmented scan：规则地处理不规则嵌套数据

许多数据是 sequence of sequences：

- graph：每个 vertex 对应一列 edges；
- particle simulation：每个 particle 对应可变长 neighbor list；
- text：每个 document 对应可变数量 words；
- sparse matrix：每一 row 有不同数量 nonzeros。

只并行 outer sequence 可能不够；而 inner lengths 不同又会造成 load imbalance。Segmented scan 把所有 inner sequences flatten，再用 segment-start flags 标边界：

```text
nested: [[1,2], [6], [1,2,3,4]]
data:   [ 1,2,   6,   1,2,3,4 ]
flags:  [ 1,0,   1,   1,0,0,0 ]
```

Exclusive segmented sum：

```text
[[0,1], [0], [0,1,3,6]]
```

实现仍可沿用 scan tree，只需随 partial value 一起传播 segment information；组合跨越 start flag 时阻断前一 segment 的 prefix。

这样 parallelism 与 flatten 后的总元素数成比例，而不是只与 outer sequence 长度成比例。

## 12. CSR sparse matrix-vector multiply

设 `y=Ax`，`A` 大部分为零。Compressed Sparse Row 存储：

```text
values:     所有 nonzero values，按 row 连续排列
cols:       每个 nonzero 对应的 column index
row_starts: 每一 row 在 values/cols 中的起始 offset
```

例如：

```text
values = [3,1,  2,  4,  2,6,8]
cols   = [0,2,  1,  2,  1,2,3]
rows   = [0,2,3,4,...]
```

传统 “one worker per row” 会受 row length 不均衡影响。Data-parallel formulation 把 parallelism 展开到 nonzeros：

### 12.1 第一步：gather 与 map

对每个 nonzero `k`：

$$
products[k]=values[k]\times x[cols[k]]
$$

`x[cols[k]]` 是一次 gather，整个乘法是 map over all nonzeros。

### 12.2 第二步：构造 row flags

由 `row_starts` 得到 flatten sequence 中每一行的 start flag。

### 12.3 第三步：segmented sum

对 `products` 做 inclusive segmented scan。每个 segment 的最后一个元素就是对应 row 的 dot product。

### 12.4 第四步：取 segment ends

Gather 每个 row 的最后 partial，得到 `y`。

于是可用 parallelism 约为 `nnz`，而不是 row count：

$$
P_{available}\propto nnz(A)
$$

代价是 gather 可能非常不规则，而且若 primitives 分开执行，会物化 `gathered_x`、`products`、flags 等中间数组。实际 sparse kernel 常融合这些步骤。

## 13. Gather 与 scatter：数据并行中的重排原语

定义：

$$
gather(index,source)[i]=source[index[i]]
$$

$$
scatter(index,input)[index[i]]=input[i]
$$

### 13.1 Gather 的硬件成本

一个 SIMD gather 中，各 lanes 可能访问：

- 不同 cache lines；
- 不同 pages；
- 完全 data-dependent 的地址。

最坏时每 lane 都产生独立 cache miss，远贵于从连续地址执行一次 vector load。GPU 支持 gather-like accesses，CPU 从 AVX2 起也有 gather instruction，但“有一条指令”不表示它与连续 load 同价。

这正是 [[Stanford CS149 - Lecture 03 - Multi-Core Architecture Part II and ISPC|ISPC]] 中 uniform/contiguous access 与 varying indexed access 性能差异的根源。

### 13.2 Scatter 的冲突语义

若所有 `index[i]` 唯一，scatter 是 permutation，不存在 write conflict。若多个 inputs 指向同一 destination：

```text
index = [1, 1, 0, 2, 0, 0]
```

普通 store 的结果依赖执行顺序。应用通常真正需要的是：

$$
output[index[i]]=atomicOp(output[index[i]],input[i])
$$

直接 atomic 可正确，但热点 bin 会 contention。

## 14. 用 sort + segmented scan 替代高争用 scatter-op

对非唯一 indices：

1. 按 `index` 排序 `(index,input)` pairs；
2. 用 map 比较相邻 keys，标出每组 start；
3. 每个相同 index group 做 segmented scan/reduce；
4. 每组只把一个 aggregate 写回 destination。

```text
arbitrary destinations
→ sort by destination
→ equal destinations become contiguous segments
→ segmented reduction
→ one write per destination
```

这个变换把随机、竞争性的 updates 转成规则的 sort 和 segmented primitive。它不一定总比 atomic 快：

- 冲突很少时，sort overhead 可能更大；
- destination 热点很多、atomic 严重串行时，先 grouping 更有利；
- 若后续本来就需要 key order，sort 成本可被摊销。

核心不是“永远用 sort”，而是知道 contention 可以通过 **reorganize data before update** 转成批量归约。

## 15. Particle binning：从锁思维切到 sequence 思维

问题：有约一百万 particles 和一个 uniform spatial grid；构建：

```text
cell 0 → [particle ids...]
cell 1 → [particle ids...]
...
```

该结构让 N-body/simulation 只遍历相邻 cells 中的 particles。

### 15.1 直接 append 的几种方案

| 方案 | 优点 | 主要问题 |
|---|---|---|
| 一个 global lock | 最容易写对 | 所有 particles 串行化 |
| 每 cell 一个 lock | 不同 cells 可并行 | 10 万 threads 争少数热点 cells |
| 每 worker 一套 cell lists | 消除写冲突 | storage 随 workers 增长，还需 merge |
| 每 cell 一个 worker，扫描全部 particles | 无 contention | parallelism 受 cell count 限制，总 work 为 `cells×particles` |

这些方案在少量 CPU threads 上可能合理，却不自然扩展到十万 CUDA threads。

### 15.2 Data-parallel solution

第一步，对每个 particle 独立计算 cell ID：

```text
particle_ids = [0,1,2,...]
cell_ids = map(position → containing_cell, positions)
```

第二步，以 `cell_id` 为 key 对 `(cell_id,particle_id)` sort：

```text
before: (4,p0), (1,p1), (4,p2), (7,p3), ...
after:  (1,p1), (4,p0), (4,p2), (7,p3), ...
```

第三步，用 map 比较相邻 cell IDs，找每一 bin 的 start/end offsets。

最终无需建立许多链表：

```text
sorted_particle_ids: flat particle sequence grouped by cell
cell_start[c], cell_end[c]: cell c 对应的 contiguous range
```

整个过程是：

```text
map → sort/groupBy → map(boundaries)
```

Parallelism 与 particle count 成比例，且没有并发 append/lock。后续访问同一 cell 的 particles 还是连续内存，locality 也更好。

> [!note]
> Empty cells 没有出现在 sorted keys 中，需要把对应 `start/end` 初始化成空 range。视频尾部提示 histogram/empty bins 也要处理这个 special case。

## 16. “高层 primitive”与“高性能实现”的分工

应用作者关注：

- 能否把 irregular problem 变成 sequence transformations；
- 哪些 algebraic properties 成立；
- 是否创造了不必要 intermediate data；
- operation chain 的总 bytes/work 是多少。

Library/runtime 作者关注：

- CPU vs GPU vs cluster 的 partition strategy；
- SIMD width 与 lane utilization；
- 多级 scan/reduction；
- cache/shared-memory locality；
- synchronization、communication 与 load balance。

CUDA 上的 Thrust/CUB、C++ parallel algorithms、tensor compiler 和 Spark RDD APIs 都体现这个分工。调用 `scan` 的人只使用语义；实现者可随架构重写底层算法。

## 17. 与 AI Infra 的连接

### 17.1 Tensor programs 本质上是 sequence operator graphs

Elementwise activation 是 map，loss/normalization 中有 reductions，cumulative operation 是 scan，top-k/sort 和 indexing 是 gather/scatter。Compiler 看见 operator graph 后可进行：

- operator/kernel fusion；
- layout transformation；
- tiling；
- algebraic simplification；
- parallel scheduling。

若用户直接写不可分析的任意 pointer mutation，系统就更难安全重排。

### 17.2 Ragged batches 与 prefix sum

Variable-length requests 常以 flat tokens + offsets 表示：

```text
lengths per request
→ exclusive scan
→ starting offset per request
→ flat token/KV buffer
```

这与 segmented sequence 完全相同。Continuous batching、packed sequences 和 paged KV metadata 都大量依赖 prefix sums/offset calculations。

### 17.3 Embedding、KV cache 与 gather

Embedding lookup 是 `weights[token_ids]` 的 gather；paged KV cache 按 block table 读取也是 data-dependent gather。性能取决于：

- indices 是否局部；
- 多个 lanes 是否落在相同/相邻 cache lines；
- 能否 sort/bucket requests 改善访问；
- 是否被 memory bandwidth/latency 限制。

### 17.4 MoE routing 是 particle binning 的同构问题

把 particles 分到 cells 与把 tokens 分到 experts 结构相同：

```text
token → expert_id                 (map)
sort/group tokens by expert      (sort/groupBy)
compute expert offsets/counts    (scan/reduce)
move tokens to packed buffers    (gather/scatter)
```

直接让所有 tokens atomic-append 到 expert queues 会在热门 expert 上 contention；sort/prefix-sum 路线用规则 data parallelism 生成 packed batches。

### 17.5 Work-efficient 与 throughput-efficient

GPU kernel 的目标不是只最小化 FLOPs。FlashAttention、fused MoE 和 quantized kernels 都可能做少量额外 arithmetic，换取：

- 更少 HBM traffic；
- 更高 tensor-core/SIMD utilization；
- 更少 kernel launches；
- 更规则的 data access。

这与 warp scan 的结论相同：应优化映射后的 critical resource time，而非只优化抽象 scalar work。

## 18. 本讲结论

1. Sequence abstraction 限制任意访问，让 dependency 和可重排性更容易推理。
2. Map 的元素调用天然独立；parallel reduce 需要 operator 的代数性质，核心是 associativity。
3. Scan 把局部 decision 转成全局 offset，是 compaction、routing 与 ragged representation 的基础。
4. Work/span 揭示理论 parallelism，但不能代替对 SIMD utilization、locality 和 communication 的分析。
5. `O(N)` work 的 scan 可能在 warp 内慢于 `O(NlogN)` work 的五步 SIMD scan。
6. 高性能 scan 是 hierarchical/heterogeneous 的：warp、block、device/CPU 层用不同策略。
7. Segmented scan 把不规则 sequence-of-sequences flatten 成规则 data-parallel computation。
8. Gather/scatter 表达不规则数据移动；它们虽并行，却可能是 memory bottleneck 或 contention source。
9. Sort/groupBy 可把随机冲突更新转换成连续 segments 和批量 reduction。
10. Data-parallel thinking 的价值是复用少数高质量 primitives，并给 compiler/runtime 留下全局优化空间。

## 自测题

1. Sequence 与普通 array 的关键区别是什么？这种限制为什么有助于并行化？
2. `map(f,S)` 要能任意并行执行，`f` 应满足什么条件？
3. Parallel reduction 为什么需要 associativity？为什么不一定需要 commutativity？
4. 浮点加法 reduction 为什么可能无法 bitwise 重现 sequential 结果？
5. Inclusive scan 与 exclusive scan 的定义有何区别？如何相互转换？
6. Hillis–Steele scan 与 Blelloch scan 的 work/span 分别是什么？
7. 为什么 work-efficient scan 在 32-wide SIMD 内可能更慢？
8. 怎样从 32-element warp scan 构造 1024-element block scan？
9. Segmented scan 怎样用 flags 表示 nested sequences？
10. CSR SpMV 如何分解为 gather、map、segmented scan 和 gather-ends？
11. Gather instruction 为什么可能比 contiguous vector load 贵很多？
12. 多个 scatter inputs 指向同一 index 时，语义上需要解决什么问题？
13. 用 sort + segmented reduction 替代 atomics 的收益和代价分别是什么？
14. Particle binning 为什么不适合让每个 cell 扫描全部 particles？
15. MoE routing 与 particle binning 在 primitive graph 上怎样对应？

## 参考资料

- [CS149 Fall 2023 Lecture 8 视频](https://www.youtube.com/watch?v=Ba3TqxSgnTk)
- [Lecture 8 官方课件](https://gfxcourses.stanford.edu/cs149/fall23content/media/dataparallel/08_dataparallel.pdf)
- [NVIDIA Thrust Documentation](https://nvidia.github.io/cccl/thrust/)
- [NVIDIA CUB](https://nvidia.github.io/cccl/cub/)
- [[Stanford CS149 - Lecture 07 - GPU Architecture and CUDA Programming|Lecture 7：CUDA、Warp 与 SM]]
- [[Stanford CS149 - Lecture 06 - Performance Optimization II Locality Communication and Contention|Lecture 6：Communication、Locality 与 Contention]]
