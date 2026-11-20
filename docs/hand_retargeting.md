# XHand 手部重定向

将 VR/HTS 的人体右手关键点转换为 XHand 关节目标。腕部运动由 VR 控制器另行映射为机械臂末端目标，两者组成完整遥操作动作。

## 数据流

**21 个关键点 → 掌系与骨长检查 → SLSQP 或 GeoRT → 有界关节目标**。

请求几何表达人体动作意图，可以超出机器人可达范围。求解器在关节范围和变化率约束内逼近目标，通过时间阻尼与平滑保持连续；对指目标描述指尖间距，不表示真实接触。

## 核心模块

| 模块 | 职责 |
| --- | --- |
| [landmarks.py](../dexmani_real/teleop/retargeting/landmarks.py) | 建立掌系、标定骨长，按 XHand 机构生成目标几何。 |
| [objective.py](../dexmani_real/teleop/retargeting/objective.py) | 组合位置、方向、对指和阻尼目标，计算目标值与解析梯度。 |
| [solver.py](../dexmani_real/teleop/retargeting/solver.py) | 用 NLopt SLSQP 拟合 XHand 目标几何。 |
| [geort.py](../dexmani_real/teleop/retargeting/geort.py) | 用 GeoRT 从五个指尖预测关节角，按操作限位和变化率生成目标。 |
| [base.py](../dexmani_real/teleop/retargeting/base.py) | 共享骨长标定、平滑系数和已确认的关节参考。 |
| [HandFK](../dexmani_real/geometry/kinematics.py) | 基于 Pinocchio 与 URDF 提供正运动学和雅可比，也供导出与策略观测使用。 |

## 与控制链的关系

VR 接收端提供关键点，`VRController` 组合腕部映射与所选后端的输出。共享动作映射生成机器人目标，由遥操作循环提交、录制。 [VR 接收端](../dexmani_real/sensor/vr.py) · [VRController](../dexmani_real/teleop/vr.py)

## GeoRT 实验配置

`HandRetargetingConfig` 默认使用 `slsqp`。通过 `collect_teleop.py --config` 加载实验配置来选择 GeoRT。`geort_checkpoint_dir` 指向含 `config.json` 和 `last.pth` 的目录。启动时加载 CPU 模型并校验关节映射。录制的 `effective_parameters` 包含后端、骨长标定与两个 checkpoint 文件的 SHA-256。

```python
from dexmani_real.config.real import RealConfig
from dexmani_real.config.teleop import HandRetargetingConfig

CONFIG = RealConfig(
    hand_retargeting=HandRetargetingConfig(
        backend="geort",
        geort_checkpoint_dir="/path/to/checkpoint",
    ),
)
```

在 `real_robot` 环境使用 PyTorch 和固定版本的 GeoRT 源码。推理会读取 checkpoint 配置以还原训练区间，再按本项目的操作限位出目标。GeoRT 掌系以腕点为原点，用中指 MCP 定义前向、食指和无名指 MCP 定义侧向。骨长标定用于检查关键点质量。参考 `xhand_v1` 适合验证推理接口。本项目的离线样本显示其对指欠佳，采集应使用经过本项目留出数据与实机验收的模型。 [参考权重与配置](https://github.com/lmz-rbs/rbs_manus_xhand_teleop/tree/be12054595fbe0967cd0c323cadc16ece8e65ae6/checkpoints/xhand_right_2026-04-09_01-20-23_xhand_v1) · [GeoRT 许可证](https://github.com/facebookresearch/GeoRT/blob/fb2075e2a3272be372016580c0b6ebd6120228f0/LICENSE)

```bash
git clone https://github.com/facebookresearch/GeoRT.git /path/to/GeoRT
git -C /path/to/GeoRT checkout fb2075e2a3272be372016580c0b6ebd6120228f0
export PYTHONPATH="/path/to/GeoRT${PYTHONPATH:+:$PYTHONPATH}"
```

## 离线训练与对照

采集张手、握拳、独立屈指、拇指与其余四指的对指，以及任务中的抓取动作。按 episode 留出验收段。在具有 PyTorch CUDA、Pinocchio、Open3D 和 SAPIEN 的 `sim` 环境运行训练，`--config` 可指定训练采用的 XHand 操作限位。训练使用本仓 URDF 的指尖正运动学，输出模型与训练记录。多段数据可重复传入各自的 `--train-episode` 或 `--validation-episode`。

```bash
python scripts/tools/train_geort.py --geort-source /path/to/GeoRT \
  --train-episode /path/to/train_episode \
  --validation-episode /path/to/validation_episode \
  --output-dir /path/to/new_checkpoint
```

用留出段对比预测值。工具读取 Raw 的关键点、起点实测关节和操作限位，用同一观测时间模拟计算与确认提交，报告手部计算时延、限位和限速指标、拒绝次数及 FK 对指距离。

```bash
python scripts/tools/evaluate_hand_retargeting.py \
  --checkpoint-dir /path/to/new_checkpoint /path/to/validation_episode
```

比较单个 episode 时，可加 `--trajectory-output comparison.rrd`，再用 `rerun comparison.rrd` 并排查看两个后端的手部点云。

以 SLSQP 的指尖间距、动作连续性与任务成功率为基线。先检查离线轨迹的手形与干涉，再做受控实机对指、抓取和整拍时延测试，通过后为采集选择 GeoRT。FK 指尖间距是几何指标，手内干涉和实际接触需单独检查。 [GeoRT 碰撞损失实现](https://github.com/facebookresearch/GeoRT/blob/fb2075e2a3272be372016580c0b6ebd6120228f0/geort/trainer.py#L319-L335)

## 关节顺序与限位

SDK、遥操作输出、Policy 和 Replay 共用 [SDK 关节表](../dexmani_real/robot/model.py) 的 12 列弧度向量。顺序为拇指 3 轴、食指 3 轴、中指 2 轴、无名指 2 轴和小指 2 轴。`HandFK` 按名称映射到 Pinocchio 的内部列，SDK 与 URDF 的关节值同号，旋转方向由 URDF 的带符号局部轴决定。HOME 配置的 `_deg` 字段先转换为弧度。

设备启动检查与 `HandFK` 构建都会核对 12 个 SDK 关节的父子连杆、带符号转轴、`origin xyz/rpy` 和额定位置限位。`xyz` 使用米，`rpy` 使用弧度。[右手 URDF](../assets/robots/xhand/xhand_right.urdf) · [组合 URDF](../assets/robots/xhand/xarm7_xhand_right.urdf)

实验操作限位由 `qpos_min_rad` 和 `qpos_max_rad` 设置，须在额定范围内。`Robot.validate_command()` 在发送前拒绝越界的普通目标，并报告关节名称。暂停时的实测位置保持按额定限位检查。[HandConfig](../dexmani_real/config/xhand.py)

## 提交语义

首个控制目标使用实测手部姿态和起始观测完成时间作为求解参考，后续由已确认的命令更新。现场求解的时间间隔从参考时间算到本周期开始计算目标的时刻。求解拒绝时，可在原租约内重发上一条已确认命令。重发成功后更新参考时间并保留参考位置。

读 CRC 待恢复时跳过控制拍，目标租约和来源时效继续生效。普通写入在截止前仍未确认时结束本段，请求停止，并用新鲜手部反馈保持当前位置。

返回[项目架构](../README.md)。
