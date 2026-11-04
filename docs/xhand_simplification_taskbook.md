# XHand 收敛任务书

> xArm 的 HOME、准备、轨迹接口和取消契约以 [xArm 轻量化任务书](xarm_simplification_taskbook.md) 为准；下文冲突约定仅为历史记录。
状态：两轮审查后的修复已落实，离线验证结果见下文；尚未进行真机验收。

本轮从 `main` / `51afdfac4a840f35fd4a2e61725b5df9715c5667` 的干净工作区开始，在当前实现上修复，不回退历史代码。本文替代原来的逐项任务、实施流水账和过细约定，以当前源码及最新任务要求为准。项目保持个人机器人研究所需的轻量实现，不新增状态系统、自动恢复、配置模式或证据框架。

## 首帧与开始准备

Teleop 和 Policy 接纳 B 时保留现有取消上下文，按以下顺序准备：

1. 确认设备停止，在原初值等待预算内取得新鲜关节反馈和消费者所需输入；录制另要求 aggregate、dense 都已有真实初值。
2. 保持停止，完成 writer 等阻塞初始化；无触觉初值不创建 writer。
3. writer ready 后仍保持停止，复用采样及等待预算，等待 ready 之后的新关节反馈和有效控制输入。一次读 CRC 可在后续正常采样机会恢复；触觉保持值不必是新帧。
4. 重新检查取消，准备 streaming，再取当前观测完成最终接纳和初始化。writer 之前的快照不能成为首次动作锚点。

准备期没有可用初值只结束本次准备，健康会话可再次 B；不延长运行期 TTL、伪造来源时间或吞掉设备故障。准入复用控制与录制的消费者检查，不再额外扫描同一份内部来源；等待不到 writer 后新关节反馈时明确报告原因。停止、退出或急停不会被迟到的 writer ready 覆盖。相机 worker 或 writer 失败按现有异常边界传播并收尾，零行不发布；收尾失败保留实际 owner。

重定向 reset 不裁剪实测姿态。范围内实测值的独立副本建立首次参考，后续仅在提交尝试正常返回后，以实际 `RobotCommand` 推进参考。求解候选不推进参考，也不与输入、参考共享可变数组。真实首帧间隔决定限速额度，不假设已等满名义周期；10 ms、4 rad/s 时最终每轴最多变化 0.04 rad。mapper 使用相同操作范围，不能在后续 clip 中增加跳变。

范围外起点在任何本段目标发送前明确拒绝，列出关节、实测角度和要求范围；只结束本段，就位后可重新 B。不新增自动入域或操作前 HOME。

默认 distal 5°/10° 下限保持：历史提交 `0b0e223` 将其明确描述为额定机械范围的保守操作子集，不能认定为无实际作用的预弯偏好。默认 HOME 的 distal 及近零 proximal 目标已向范围内移动，使 ±5° 到位区间都位于默认操作范围。用户自定义限位或 HOME 不会自动修改，须相应留出到位余量；HOME 成功不豁免实际起点范围检查。

## 固定全模态与触觉

启用录制始终保留完整 Raw：arm/hand 实测关节、动作提交尝试、RGB、depth、aggregate、dense、独立来源时间及现有元数据，Teleop 另保留 VR 调试字段。不按策略切换采集模式。

Canonical 始终导出以下 13 字段，形状、单位和 dtype 沿用 `data/export.py`：`joint_state`、`action`、`action_ee`、`finger_tactile_force`、`taxel_tactile_force`、`fingertip_xyz`、`eef_pose`、`camera_intrinsic`、`camera_extrinsic`、`rgb`、`depth`、`point_cloud`、`hand_pointcloud`。保持原行数、顺序、episode 边界和名义 dt，不筛行或伪造数据。

- 开始录制前，两触觉通道都须有 SDK 边界校验过的真实样本，来源时间为正且非未来；无需先清零，也不要求同帧。
- 录制中逐通道保持最近完整有效数组及原来源时间，恢复后自然替换。不填假零、不刷新旧时间、不因触觉超龄丢整行。
- SDK 原始载荷继续检查结构和有限性。`XHandFeedback` 保存固定结构的内部快照，消费者不再逐 tick 重扫触觉 shape/finite；仍检查真实来源时间，策略实际使用触觉时继续检查其输入 TTL。
- 录制的相机进程、writer I/O、队列满和其他真实失败处理保留。Raw/导出文件仍验证结构、数值、因果时间和 RGB 计数。
- 元数据保持 `hand_tactile_hold_policy="last_valid_until_recovery"`；触觉最大年龄超过 0.5 s 只告警并保留年龄统计，不作为导出拒绝门槛。

三类手部来源时间分别表示关节、aggregate、dense 的真实新样本，不能合并或刷新。`source_camera_sequence` 保持同源对齐用途；`source_id` 等已有元数据不单独迁移格式。

## 通信、HOME 与清零

- 读 CRC 丢弃该帧，不更新任何来源时间。普通绝对位置写 CRC 未确认时继续下一周期最新目标，连续 5 次才按通信故障停止；允许的非 CRC 命令返回清零连续计数，读取成功不清零。普通写 CRC 正常返回仍推进提交尝试参考，不重构提交协议。
- 活动期间及发送副作用边界，手关节反馈 TTL 仍为默认 0.5 s。持续失联、未知 SDK/设备错误和非法目标继续报错并停止。过流 `1501035` / 板码 `110` 保持限频告警，不一律升级为会话故障。
- 停止仍使用原 `4/30 s` 的实测 hold 新鲜度和 0.25 s 有限 CRC 重试，过旧则 passive；退出请求 passive。软件期限不能中断挂起的 native SDK。
- 启动不自动 HOME、清零。显式 hand HOME 先停止 arm，保持其停止；首次 CRC 可在下个反馈机会补发一次，至多两次，并由操作期限内提交后的连续新反馈判断到位。
- 显式 T：Robot 停止、检查取消、清空触觉缓存后，驱动向 `0x11`–`0x15` 各请求一次，保存已完成请求的实际返回码。删除 `on_start` 反向回调；关节缓存保留，清缓存后恰好取消可在下次 B 重新等待初值，无须恢复旧缓存。驱动保留原始结果字典，仅在既有 Raw 元数据边界编码 JSON，不维护第二套结果或格式。
- HOME 取消遵循 [xArm 任务书](xarm_simplification_taskbook.md)：无在途请求、停止确认且资源收尾成功时可回待命；Owner 已 closing 或在途结果不明时结束设备会话，不能仅凭手部停止成功复用。

`HandConfig` 构造时将 NumPy 整数标量规范为 Python `int`，设备参数、CLI JSON、Raw 配置归档及 Policy YAML 共用这一表示，不新增序列化适配层。布尔和浮点不转换，继续由整数校验拒绝；长度、数值范围及额定机械边界检查保留。

## SDK 反馈读取

本地 `real_robot` 环境为 `xhand_controller==1.5.2`，实现是 native 扩展。现有绑定的状态没有可靠采样时间或序号，`read_state` 说明也不保证发送后的缓存更新；旧 1.1.8 文档不足以证明当前 SDK 的缓存来源。因此继续 `read_state(True)`，没有启用未经验证的反馈缓存读取，也不宣称减少总线时延。

## 验证与限制

使用现有 `/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python`，不安装新依赖。先运行相关定向测试，再运行受影响回归；pytest 禁用 cacheprovider，Ruff 使用 `--no-cache`，Mypy 如需运行则禁用增量和缓存。

本轮测试覆盖实际 FK/NLopt → VRController → ActionMapper → RobotCommand 的首帧范围与限速、范围外拒绝及再次开始、HOME ±5°；Teleop/Policy 的慢 writer、一次 CRC 恢复、短 writer 后新关节反馈、取消、writer/相机失败和持续无初值；配置整数的 CLI/JSON/YAML/SDK 使用、清零取消和实际结果写入 Raw 属性。复用真实 recorder → Raw → Zarr 测试，验证全部 13 字段、行数、保持值和原来源时间，触觉策略仍执行新鲜度检查。

实际验证（2026-11-04）：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/zhanghaoyang/miniconda3/envs/real_robot/bin/python -m pytest -p no:cacheprovider tests -q
/home/zhanghaoyang/miniconda3/envs/real_robot/bin/python -B -m ruff check --no-cache --select F dexmani_real scripts tests
```

全测试 **716 passed in 4.13s**，无跳过；Ruff F 通过。Mypy 扩展检查本轮相关 10 个实现模块，使用 `--check-untyped-defs --no-incremental --cache-dir=/dev/null --follow-imports=silent --ignore-missing-imports`；其中 Raw 有 7 项既有资源类型标注诊断，其余 9 个模块无诊断。以 HEAD 的 Raw 通过 `--shadow-file` 对照，同样 7 项，仅行号因新增编码移动，未新增诊断；扩展类型检查未全部通过，未用额外判空、Any 或 ignore 掩盖。

Teleop、Policy 和独立 XHand 工具的三个真实 CLI 均通过 NumPy 配置打印测试，未连接设备。将当前包源码和 pyproject 复制到 `/tmp`，使用现有 setuptools 构建 wheel 成功；81 个 Python 模块与当前源码一致，标定 JSON 已打包。构建不证明外部 SDK、模型和仓库资产可脱离实验环境运行。`git diff --check` 通过，仓库未生成构建产物或禁用的三类工具缓存。

未进行真实设备连接、运动、HOME、清零、串口/EtherCAT 或 checkpoint 推理。默认 HOME 的新目标、控制跟随、CRC/触觉异常和物理停止时延仍须真机核实；离线求解与 Fake SDK 不证明实际物理执行或到位。
