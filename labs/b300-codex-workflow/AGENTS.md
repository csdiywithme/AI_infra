# Codex 工作约定

本目录是用户现有 04l B300 GEMM 的实验基础设施。采用 KDA 的任务契约、假设、验证和证据记录方式。

1. 先读 README.md、contract.json、work/origin.json 与上一候选的 verdict.json。首轮先用完全相同的 baseline/candidate 做闭环 smoke，预期 KEEP_BASELINE。
2. 优化时仅编辑 work/candidate/candidate.cu，每次只验证一个具体假设。先写 hypothesis.md：瓶颈证据、改什么、期待哪个指标变化、何时回退。
3. fixed/、快照、evaluator、contract、policy、已保存结果不是候选编辑面。若要修改 ABI、测试或测量协议，应单独提出新的 task revision，不能用来放宽当前验收。
4. 不修改原课程仓库，不补写题目完整答案。本工作流只测试用户已经存在的算子和其修改；依原仓库 AGENTS.md 做解释、review 和基础设施支持。
5. seal 后才可 submit。每个 phase 用新 attempt 目录；同一 attempt 只提交一次。存在 call_id 用 poll；SUBMISSION_UNKNOWN 先检查 Modal app，禁止猜测后重交。
6. 远端上传与收费执行需要用户授权。没有此次明确授权，不运行 submit --allow-cloud。得到一次任务的授权后，在其 mode、次数和时间范围内执行，不逐条重复确认。
7. 性能只从 bench 的 verdict.json 判定。profile 是诊断，check 是数值门槛，CPU 测试只是基础设施检查。禁止把 synthetic fixtures、历史不同allocation数值或NCU replay时延当新结果。
8. ACCEPT_SCOPED 只支持合同声明的 shapes/精度/ABI，不代表任意 shape、同步形式证明或服务端收益。脚本不自动替换原 baseline。
9. 计时失败、数值错误、NCU LaunchFailed、容器终止都保留为失败记录。禁止删除不利结果、复测到碰巧获胜，或静默扩大付费预算。
10. 完成后写 experiment.md：source/contract hash、hypothesis、实际检查、原始结果链接、判断、适用范围。保留正确但更慢的候选及回退原因。
