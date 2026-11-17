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

使用已有 Raw 独立运行手部算法：

```bash
python scripts/preview_hand_retargeting.py /data/episodes/pick/episode_001
python scripts/preview_hand_retargeting.py /data/episodes/pick/episode_001 --config experiments/hand.py
python scripts/preview_hand_retargeting.py /data/episodes/pick/episode_001 --max-frames 300 --no-view
```

入口只读取所选前缀的实测手关节、VR landmarks、时间行及段边界，不读取图像或 HOI，不连接设备、不写回数据。路径支持 `~`，相对路径以 cwd 为准。

预览与在线重定向共用 NLopt 求解器；依赖安装见 [CPU 离线环境](offline_cpu.md)。无窗口预览不需要 Ruckig、Policy 或 Rerun。关键点每次最多读取 256 行，计算块之间延续同一个求解参考，只有录制段界会重置。返回的关节曲线仍按所选帧数保留。

默认从 `effective_parameters` 恢复完整的 `hand_retargeting`、`hand_retargeting_calibration` 及手部限位。`--config` 加载导出 `CONFIG: RealConfig` 的 Python 文件，仅替换 `hand_retargeting`；标定和限位始终取自 Raw。缺少所需记录或配置错误直接报错。

计算使用记录的 `observation_ns`：段首按实测关节和该行时间 `reset`，只展示参考；后续有效候选以同一行时间 `commit`。拒绝行保留上一参考位置与时间，显示原因并断开候选曲线；段间重置平滑状态。这里的 `commit` 表示离线接受候选，不能证明真实硬件执行成功。

Rerun 在手部局部坐标系对照实测与有效候选的指尖；段首和拒绝行显示保留参考。关节曲线同时保留实测、候选与参考，标出段界及拒绝行，使用时间轴浏览。`--no-view` 不初始化 viewer，仅打印帧数、拒绝原因和求解耗时；耗时只包含 `retarget` 调用。`--max-frames` 限制读取和计算的前缀。

## 数值验证与提交语义

`tests/test_hand_retargeting.py` 使用真实 Pinocchio，按 SDK 关节顺序有限差分检查位置雅可比、位置/近段方向/末段方向/对指/阻尼及总目标。覆盖关节上下限邻域、内部姿态、对指关闭/部分/完全启用、8/62.5/200 ms 间隔及 Huber 转折；梯度验收容差为 `atol=1e-7, rtol=1e-5`。另检查掌系的刚体变换不变性、最终命令变化率、平滑、reset、退化几何和优化器坏输出。

`retarget` 不推进提交参考；拒绝或发送抛错也不推进。在线若成功重发旧命令，位置保持、提交时间更新；这是一次实际提交，与离线拒绝行不调用 `commit` 的行为不同。零时间间隔返回独立的参考副本，时间倒退报错。

## 回归样本

`tests/fixtures/hand_retargeting/data.h5` 保留一段真实录制的 397 帧时间、VR 关键点、实测关节和提交目标，不含图像。样本采用当前 Raw 格式，供数值测试、离线预览和数据导出共用；来源名称保存在元数据中。段界、拒绝和坏输入由合成测试覆盖。

返回[项目架构](../README.md)。
