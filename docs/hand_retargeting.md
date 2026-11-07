# XHand 手部重定向

手部重定向将 VR/HTS 的人体手部关键点转换为 XHand 关节目标，是遥操作中的手部动作生成模块。腕部运动由 VR 控制器单独映射为机械臂末端目标，再与手部目标组成完整动作。

## 数据流

**手部关键点 → 掌系与骨长标定 → XHand 目标几何 → 有界优化 → 平滑关节目标**。

输入为人体右手的 21 个关键点。模块建立掌部坐标系，利用标定骨长和机器人机构几何，将人体骨链映射为 XHand 的请求构型，再通过正运动学生成一致的位置、方向和对指距离目标。

请求几何表达人体动作意图，可以超出机器人的可达范围。求解器在关节范围和变化率约束下，联合拟合骨链位置、手指方向与对指距离，并通过时间阻尼和平滑保持动作连续。几何对指目标描述指尖间距，不代表真实接触。

## 核心模块

以下三个模块位于 [teleop/retargeting](../dexmani_real/teleop/retargeting/) 下。

| 模块 | 职责 |
| --- | --- |
| [landmarks.py](../dexmani_real/teleop/retargeting/landmarks.py) | 建立掌系、标定骨长，并按 XHand 机构生成目标几何。 |
| [objective.py](../dexmani_real/teleop/retargeting/objective.py) | 组合位置、方向、对指和时间阻尼目标，计算目标值与解析梯度。 |
| [solver.py](../dexmani_real/teleop/retargeting/solver.py) | 使用 NLopt SLSQP 完成有界优化，管理输出平滑与提交参考。 |

[HandFK](../dexmani_real/geometry/kinematics.py) 基于 Pinocchio 和机器人 URDF 提供正运动学与雅可比，也供数据导出和策略观测使用。

## 与控制链的关系

[sensor/vr.py](../dexmani_real/sensor/vr.py) 接收人体手部数据，[VRController](../dexmani_real/teleop/vr.py) 组合腕部映射与 `HandRetargeter` 的输出。共享动作映射生成机器人关节目标，遥操作控制循环负责提交与录制；重定向模块本身不发送设备命令。

重定向参考由实测初始姿态建立，并在命令提交正常返回后更新，使后续求解与已提交目标保持一致。

返回[项目架构](../README.md)。
