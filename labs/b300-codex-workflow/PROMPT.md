# 给 Codex 的首次任务

在当前目录按 AGENTS.md 和 README.md 执行。

首个任务是现有 B300 04l GEMM 的工作流 smoke：baseline 与 candidate 先保持完全一致；只检查封存、远程构建、数值矩阵、同一次GPU调用的ABBA和本地验收是否闭合。预期 KEEP_BASELINE，不寻找或声称优化收益。

先完成无需云计算的 doctor、prepare、seal 和CPU检查，报告快照路径、source hash与待执行命令。没有用户此次明确的源码上传和GPU次数/时限授权，不调用 submit --allow-cloud。

首轮实机通过之后，再接受单个优化假设。只编辑 work/candidate/candidate.cu，先跑check，再按需profile，最后对冻结的最终候选跑bench和gate。profile失败不要自动扩大采集范围或重启收费任务。性能不达标就保留baseline并解释失败证据。
