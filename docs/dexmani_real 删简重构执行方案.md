# DexMani Real 删简重构执行记录

原分阶段执行方案已完成并由当前源码、配置和测试取代；本文件不再作为待执行工作流。
运行行为的权威顺序为源码、schema/config、面向使用者的文档与仓库地图。

## 已完成范围

- 收敛 public policy runtime 与 Real lifecycle 的边界，保留 policy 侧的模型/训练所有权。
- 统一 coupled arm/hand command publication、generation、SafetyGate、worker progress 与 shutdown
  的 fail-closed 语义。
- 明确 teleop smoothness、learned-policy jump rejection 与 worker SDK slew protection 三类限制。
  learned-policy arm 使用 20° reject-only guard；hand 使用独立的
  `policy.hand_max_action_jump_rad=1.0` reject-only guard。首条 policy action 参考 measured
  feedback，之后参考上一条成功发布的 target。
- 完成 observation、Prediction IPC、recording/replay provenance、配置所有权与数据路径的审计；
  没有改变持久化 schema 或 IPC wire contract。
- 删除已被上述边界替代的临时/兼容代码；当前清理继续以可证明的无调用或重复逻辑为限。

## 验收记录

- 离线完整回归：`conda run -n real_robot pytest -q`，`162 passed, 102 subtests`。
- 集中真机回归：CUDA policy check、shadow no-publication、低风险 run H→B→S、同进程重复
  H→B→S、RUNNING 中 S 与 ESC/e-stop。未进行 schema、训练、采样器或 IPC 的兼容迁移。

## 未覆盖的受控实验

- 人为注入超过 policy hand jump 阈值的模型动作。
- SDK CRC/拒绝、持续 sensor failure，以及 contact-induced feedback lag。
- 真实相机/point-cloud 的端到端时延测量与 physical replay。

这些项目如需执行，必须先定义独立、受控的实验验收标准；不能恢复本文件原先的阶段工作流。
