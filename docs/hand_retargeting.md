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

## 离线预览

已有 Raw 可直接运行算法，无需连接机器人或启动传感器：

```bash
python scripts/preview_hand_retargeting.py /data/episodes/pick/episode_001
python scripts/preview_hand_retargeting.py /data/episodes/pick/episode_001 --config experiments/hand.py
python scripts/preview_hand_retargeting.py /data/episodes/pick/episode_001 --max-frames 300 --no-view
```

入口只读取所选前缀的手部实测关节、VR landmarks、时间行及段边界，不读取 RGB、深度或 HOI，不写回数据。路径支持 `~`，相对路径以当前工作目录为准。默认从 `effective_parameters` 恢复完整 `hand_retargeting`、`hand_retargeting_calibration` 与手部关节限位；缺少必要字段时直接报错，不估算标定或补当前默认参数。

`--config` 使用导出 `CONFIG: RealConfig` 的 Python 文件，仅替换 `CONFIG.hand_retargeting`，此时不消费记录中被替换的重定向参数。标定和关节限位始终来自该 Raw，配置文件中的设备与限位设置不会替换它们。

计算时间使用每行记录的 `observation_ns`。每段首行按实测关节和该行时间 `reset`，只展示参考；后续有效候选在同一记录时间 `commit`，表示离线接受候选。拒绝行保留上一参考位置与时间，显示原因并使候选曲线断开。段间不继承平滑参考或累计时间。

Rerun 在手部局部坐标系对照实测、离线候选与已接受参考的指尖，并提供各关节曲线、段边界及拒绝标记；使用其时间轴浏览即可。`--no-view` 不初始化 viewer，只打印帧数、拒绝及原因、求解耗时统计；`--max-frames` 限制读取和计算范围。耗时仅统计实际 `retarget` 调用，段首行不计入。

离线参考推进不证明真实硬件曾成功提交。Raw 中动作字段只记录提交尝试，生产控制器的提交时序保持原样；预览也不追求真实执行的逐位复现。

返回[项目架构](../README.md)。
