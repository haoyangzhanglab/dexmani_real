# XHand 手部重定向

将 VR/HTS 的人体右手关键点转换为 XHand 关节目标。腕部运动由 VR 控制器另行映射为机械臂末端目标，两者组成完整遥操作动作。

## DexPilot 手部求解

**21 个关键点 → 掌系与骨长检查 → 15 个 DexPilot 参考向量 → 有界关节目标**。

使用 `control` 安装选项固定的 `dex-retargeting==0.4.6`，通过 `DexPilotOptimizer` 和本仓 XHand 右手 URDF 拟合 10 个指尖间向量及 5 个腕到指尖向量。通过 `collect_teleop.py --config` 可以配置标定容差、平滑和关节速度。录制的 `effective_parameters` 保存实际求解参数、骨长标定和已安装的包版本。 [DexPilot 源码](https://github.com/dexsuite/dex-retargeting/blob/8632b2cab32e1b51ce379940c414a0f78332ff6b/dex_retargeting/optimizer.py)

- 常规向量由 HTS 骨段方向投影到 XHand 机构后生成，用来保持张手、屈指和握拳的形状。掌系建立、骨长检查和形态投影见 [landmarks.py](../dexmani_real/teleop/retargeting/landmarks.py)。
- 对指权重随原始 HTS 指尖间距在 `pinch_start_m` 与 `pinch_full_m` 之间平滑增加，默认是 30 mm 和 15 mm。食指、中指和小指比无名指更接近拇指时逐步获得补偿，领先 5 mm 时竞争系数升到 1。拇指对指向量沿投影方向缩短，缩短时应用 `pinch_full_m` 下限。同时接近的非拇指对拟合原始掌系相对向量。主、次向量的权重上限为 16 和 6。 [求解与提交参考](../dexmani_real/teleop/retargeting/solver.py)

VR 接收端提供关键点，`VRController` 组合腕部映射与手部关节目标，再由遥操作循环提交和录制。 [VR 接收端](../dexmani_real/sensor/vr.py) · [VRController](../dexmani_real/teleop/vr.py)

启动时的骨长标定统计当前连接在截止前收到的独立、新鲜关键点包。重连后重新计数，收满样本后立即复核取消状态和连接，再计算标定。

## 时序诊断

用 `collect_teleop.py --timing-output` 录制的时序中，`summary.hand_retarget` 汇总保留窗口内的手部求解时延分位数和失败次数。逐拍原因记录在 `hand_retarget` 事件中。也可用 `python -m dexmani_real.runtime.teleop_timing TRACE.json` 查看汇总。

## 关节顺序与限位

SDK、遥操作输出、Policy 和 Replay 共用 [SDK 关节表](../dexmani_real/robot/model.py) 的 12 列弧度向量。顺序为拇指 3 轴、食指 3 轴、中指 2 轴、无名指 2 轴和小指 2 轴。`HandFK` 按名称映射到 Pinocchio 的内部列，SDK 与 URDF 的关节值同号，旋转方向由 URDF 的带符号局部轴决定。HOME 配置的 `_deg` 字段先转换为弧度。

设备启动检查与 `HandFK` 构建都会核对 12 个 SDK 关节的父子连杆、带符号转轴、`origin xyz/rpy` 和额定位置限位。`xyz` 使用米，`rpy` 使用弧度。[右手 URDF](../assets/robots/xhand/xhand_right.urdf) · [组合 URDF](../assets/robots/xhand/xarm7_xhand_right.urdf)

实验操作限位由 `qpos_min_rad` 和 `qpos_max_rad` 设置，须在额定范围内。`Robot.validate_command()` 在发送前拒绝越界的普通目标，并报告关节名称。暂停时，实测位置保持将目标裁剪到额定范围。[HandConfig](../dexmani_real/config/xhand.py)

## 提交语义

首个控制目标使用实测手部姿态和起始观测完成时间作为求解参考，后续由已确认的命令更新。现场求解的时间间隔从参考时间算到本周期开始计算目标的时刻。求解拒绝时，可在原租约内重发上一条已确认命令。重发成功后更新参考时间并保留参考位置。

读 CRC 待恢复时跳过控制拍，目标租约和来源时效继续生效。普通写入在截止前仍未确认时结束本段，请求停止，并用新鲜手部反馈保持当前位置。

返回[项目架构](../README.md)。
