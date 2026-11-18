# XHand 手部重定向

将 VR/HTS 的人体右手关键点转换为 XHand 关节目标。腕部运动由 VR 控制器另行映射为机械臂末端目标，两者组成完整遥操作动作。

## 数据流

**21 个关键点 → 掌系与骨长标定 → XHand 目标几何 → 有界优化 → 平滑关节目标**。

请求几何表达人体动作意图，可以超出机器人可达范围。求解器在关节范围和变化率约束内逼近目标，通过时间阻尼与平滑保持连续；对指目标描述指尖间距，不表示真实接触。

## 核心模块

| 模块 | 职责 |
| --- | --- |
| [landmarks.py](../dexmani_real/teleop/retargeting/landmarks.py) | 建立掌系、标定骨长，按 XHand 机构生成目标几何。 |
| [objective.py](../dexmani_real/teleop/retargeting/objective.py) | 组合位置、方向、对指和阻尼目标，计算目标值与解析梯度。 |
| [solver.py](../dexmani_real/teleop/retargeting/solver.py) | 用 NLopt SLSQP 求解，管理平滑与提交参考。 |
| [HandFK](../dexmani_real/geometry/kinematics.py) | 基于 Pinocchio 与 URDF 提供正运动学和雅可比，也供导出与策略观测使用。 |

## 与控制链的关系

[VR 接收端](../dexmani_real/sensor/vr.py) 提供关键点，[VRController](../dexmani_real/teleop/vr.py) 组合腕部映射与 `HandRetargeter` 输出，再由共享动作映射和遥操作循环提交、录制。重定向模块不直接发送设备命令；求解参考从实测姿态建立，仅在命令提交正常返回后更新。

## 关节顺序与限位

SDK、遥操作输出、Policy 和 Replay 共用 [SDK 关节表](../dexmani_real/robot/model.py) 的 12 列弧度向量。顺序为拇指 3 轴、食指 3 轴、中指 2 轴、无名指 2 轴和小指 2 轴。`HandFK` 按名称映射到 Pinocchio 的内部列，SDK 与 URDF 的关节值同号，旋转方向由 URDF 的带符号局部轴决定。HOME 配置的 `_deg` 字段先转换为弧度。

额定限位来自仓库的 [右手 URDF](../assets/robots/xhand/xhand_right.urdf)，启动时与 [组合 URDF](../assets/robots/xhand/xarm7_xhand_right.urdf) 对照。实验操作限位来自 [HandConfig](../dexmani_real/config/xhand.py) 的 `qpos_min_rad` 和 `qpos_max_rad`，须在额定范围内。`Robot.validate_command()` 在发送前拒绝越界的普通目标，并报告关节名称。暂停时的实测位置保持按额定限位检查。

## 提交语义

现场求解的时间间隔从上次命令提交返回算到本周期开始计算目标的时刻。`retarget` 不推进提交参考。拒绝或发送抛错也不推进。成功重发旧命令时，参考位置保持，提交时间更新。零时间间隔返回独立的参考副本，时间倒退报错。

返回[项目架构](../README.md)。
