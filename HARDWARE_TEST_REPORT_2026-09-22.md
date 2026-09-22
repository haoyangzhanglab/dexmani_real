# 真机测试总结（2026-09-22）

已完成 xArm7 与 XHand 的短时读写、固定目标发送及宿主时序测量。两台执行器分开测试：高层发布 16 Hz，现有 worker 30 Hz，每次发送阶段 10 秒。模型测试按用户要求暂缓。

## 授权与实际操作

- 用户确认现场看护和急停就绪，授权当前位置保持，不做 HOME 或策略动作。
- xArm 使用最新实测关节位置作为固定目标；标准连接流程使能电机、应用运行参数并切换 Mode 6。
- XHand 初始实测姿态不满足工作限位。用户随后授权“直接发送一个合法 qpos”，测试才继续。
- XHand 使用现有 `project_hand_command` 将最新实测姿态投影到既有工作限位，形成一个固定绝对目标。SDK 顺序第 6/8/10/12 关节目标设为 5°，相对实测值分别增加约 6.17°/5.08°/4.83°/4.67°；其余目标保持实测值。此项涉及实际关节移动。
- 未执行 HOME、策略动作、物理 replay、VR teleop、相机采集或持久标定写入；未修改限位。

## 测量结果

以下均为宿主时钟测量；发布至 SDK 入口的延迟不代表物理响应或到达时间。

| 测量 | 样本数 | mean (ms) | p95 (ms) | max (ms) |
|---|---:|---:|---:|---:|
| XHand `get_state()`，发送阶段 | 306 | 11.454 | 12.032 | 12.516 |
| XHand `send_action()` | 160 | 10.153 | 10.583 | 10.623 |
| xArm 宿主发布 → SDK 入口 | 160 | 18.502 | 33.093 | 33.169 |
| xArm SDK 发送调用耗时 | 160 | 0.245 | 0.289 | 0.320 |
| xArm 成功宿主发布间隔 | 159 | 62.500 | 62.545 | 62.614 |
| XHand 成功宿主发布间隔 | 159 | 62.501 | 62.534 | 62.622 |

实际有效宿主发布频率：xArm **15.999912 Hz**，XHand **15.999863 Hz**。这是 `execute=True` 固定目标测试，不是带模型的 PolicyRunner action-step 测量，也不是 `execute=False` dry-run 结果。

### xArm

- 160 次宿主发布、160 次 SDK 发送调用；SDK 返回码全部为 0。
- 最大实测保持偏移约 `1.94e-6 rad`。
- 退出后只读复核：Mode 6、State 4（停止）、error/warn 均为 0、关节速度为 0。

### XHand

- 初始复核中，第 6/8/10/12 关节反馈约为 −1.08°/−0.08°/0.17°/0.58°，低于配置工作下限 5°；前两项反馈数值也低于配置机械下限 0°。未直接发送这些超限反馈值，未据此调整限位或判定机械结构越限。
- 合法目标测试：160 次宿主发布、160 次 `send_action()`，全部返回 `ACCEPTED`，无 `CRC_UNCONFIRMED` 或 `REJECTED`；发送阶段 306 次状态读取全部可用。
- 结束时最大关节目标误差 `0.0189046 rad`，约 **1.08°**。SDK 接受不等于精确到位。
- 未回 HOME 或恢复初始姿态；最后目标未自动撤回。软件 `DISARMED` 和 SDK 断开不代表 XHand 电机失能。

### 观测与退出

- 只读阶段 xArm、XHand 各 300/300 次读取成功，XHand 另有 60/60 次复核成功；未观测到 XHand board errors。
- 只读阶段成功读取完成间隔均未超过 133.33 ms；发送阶段消费到的 arm/hand 观测年龄最大约 33.15/30.92 ms。上述结果不证明完整传感器链路的物理采样延迟。
- XHand 只读聚合/稠密触觉结构有效；测时包装跳过 `calibrate_tactile`，未验证零偏标定或触觉精度。
- 两个 worker 均正常退出，`exitcode=0`；确认子进程退出后释放共享内存，SDK 连接关闭，软件状态为 `DISARMED`。

## 方法、进度与剩余问题

临时脚本复用现有 `arm_loop` / `hand_loop`，仅包装 SDK 调用采集时间。生产源码、worker 频率、XHand 先读反馈后处理命令的顺序、latest-target、run_id 和 raw/IPC schema 均未因测时修改。内部 ring sequence 仅用于临时统计对齐，没有新增公开命令身份或 ACK 协议。

`codex_task.md` 的五项 commissioning 测量中，XHand 读取耗时、XHand 发送耗时、arm 发布至 SDK 延迟已完成；成功发布间隔已在固定目标条件下测量。**策略推理耗时及带模型的真实有效 action Hz 尚未测量**，原因是用户明确要求暂不测模型。

仍需后续授权实验验证：

- arm + hand + camera + GPU 并发负载下的观测新鲜度、推理耗时和实际 action-step 间隔。
- XHand 小角度反馈与配置下限不一致的原因，以及目标跟踪精度；本次未作标定或限位修正。
- 物理急停按钮、故障注入和长时间运行；本次没有执行这些测试。

当前证据仅覆盖分执行器、短时、固定目标条件，不能作为 latest-target 逐条交付保证，也不足以决定提高 worker 频率、调整 XHand 读写顺序或引入异步推理。

前一轮离线复查：`python -m compileall -q dexmani_real examples` 退出码 0；`git diff --check` 退出码 0；Ruff 不可用（退出码 127），未安装或升级依赖。离线结果与以上真机测量分别记录。

任务原文要求 commissioning 仅报告、不执行；后续用户明确授权了上述真机测试，因此在该授权范围内执行。模型、HOME 等未授权项目保持未执行。本次整理仅新增此报告，保留已有 24 个文件的改动，不提交或推送，也不重复运行硬件测试。

## 原始记录

关键统计已保存在本文件；以下 `/tmp` 文件为临时证据，系统清理后可能消失：

- `/tmp/dexmani_hand_readonly.json`
- `/tmp/dexmani_hand_recheck.json`
- `/tmp/dexmani_arm_readonly.json`
- `/tmp/dexmani_arm_hold.json`
- `/tmp/dexmani_arm_postcheck.json`
- `/tmp/dexmani_hand_target.json`

测量脚本：`/tmp/dexmani_commission_readonly.py`、`/tmp/dexmani_arm_hold_measurement.py`、`/tmp/dexmani_hand_target_measurement.py`。这些脚本包含真实硬件操作，不可作为离线检查运行。
