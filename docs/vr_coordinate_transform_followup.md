# VR 腕部姿态到 EEF 坐标换算：审查记录与后续验证

## 1. 状态与范围

本记录源于以下现场报告：操作者认为手腕绕机器人世界坐标系的 `+X` 旋转时，机械臂 EEF
没有绕机器人世界 `+X` 旋转。

截至本记录，VR wrist 到 EEF 的旋转公式已完成源码审查和离线几何验证，但尚未取得对应的
真实 HTS wrist 四元数样本。因此，**没有足够证据安全地替换当前的姿态乘法顺序或手工交换
坐标轴**。当前工作状态如下：

- 已修复：位置和姿态共用同一份固定 `T_vr_to_robot` 标定；不再在遥操作重锚定时动态改写
  heading。
- 已修复：腕部旋转的限幅/接受状态以“已接受姿态”为基线，tracking 毛刺不会通过下一帧绕过
  per-frame 限幅。
- 已验证：在“HTS wrist 四元数是 wrist-local 到 VR-world 的姿态”这一合同下，当前公式会把
  机器人世界 `+X` 的空间旋转映射为 EEF 世界 `+X` 的空间旋转。
- 待验证：真实 HTS 数据是否满足上述姿态合同，以及操作者的物理动作是否确实是机器人世界轴
  的外在（spatial）旋转而非手腕自身轴的内在（body）旋转。

本文是待办和证据记录，不是新的标定格式或运行时行为规范。标定的当前数值仍以
[`config/vr_transform.json`](../dexmani_real/config/vr_transform.json) 为唯一来源。

## 2. 坐标系和姿态约定

| 符号 | 含义 | 当前代码中的来源/用途 |
| --- | --- | --- |
| `V` | VR tracking 世界坐标系，已由 Unity left-handed 转为 FLU | `sensor/vr_worker.py` 发布 `wrist_quat_wxyz` |
| `H` | wrist 的局部坐标系 | HTS wrist pose 的局部基底 |
| `B` | 机器人 base 坐标系 | 固定 VR 标定的目标坐标系 |
| `W` | 机器人 world 坐标系 | 当前 VR teleop 中与 `B` 相同（`base_to_world_rot = I`） |
| `E` | EEF 局部坐标系 | IK target 的姿态基底 |

记 `{}^A R_B` 为把 `B` 中表示的向量转换到 `A` 中表示的正交旋转矩阵。当前实现所依赖的
输入合同是：

```text
R_wrist(t) = {}^V R_H(t)
```

即 wrist 四元数描述的是“wrist 局部轴在 VR 世界中的朝向”，而不是一个已经相对某个初始帧或
头显帧表达的增量。`vr_worker.py` 负责 Unity → FLU 的坐标基变换与 `xyzw → wxyz` 排列；
`arm_mapper.py` 不应再次做 Unity 轴交换。

固定标定为：

```text
T_vr_to_robot = {}^B R_V
```

当前标定程序只从朝向估计一个绕 `Z` 的旋转，约定由
`R_z(-theta) maps VR FLU forward → robot base +X` 定义。它足以表达已完成 Unity → FLU
基变换后的纯 heading 对齐；它**不能**代表任意未知的三维安装误差。

## 3. 当前实现的数据流与公式

运行路径为：

```text
HTS/Unity wrist pose
  -> unity_left_to_flu_rotation + WXYZ 规范化
  -> RuntimeChannels.vr_ring.wrist_quat_wxyz
  -> ArmWristMapper.reset() 记录 wrist/EFF anchor
  -> ArmWristMapper.map() 生成 EEF world target
  -> IK / safety gate / arm worker
```

在一次 reset 的时刻 `0`，mapper 保存 `R_wrist(0)` 和 EEF anchor。每一帧首先计算：

```text
Δp_V = p_wrist(t) - p_wrist(0)
ΔR_V = R_wrist(t) R_wrist(0)^T
```

若 `R_wrist` 确实为 `{}^V R_H`，上式的 `ΔR_V` 是在 **VR 世界轴** 中表达的空间旋转。
映射到机器人后，当前代码等价于：

```text
Δp_B = position_scale · {}^B R_V · Δp_V
ΔR_B = {}^B R_V · scale_and_limit(ΔR_V) · ({}^B R_V)^T

Δp_W = {}^W R_B · Δp_B
ΔR_W = {}^W R_B · ΔR_B · ({}^W R_B)^T

p_E^W(t) = {}^W R_B · p_E^B(0) + Δp_W
R_E^W(t) = ΔR_W · {}^W R_E(0)
```

最后一行的**左乘**是关键：它把空间/world-frame 旋转施加到 EEF 初始姿态上。举例来说，
若 `ΔR_W = R_x(α)`，无论 EEF 初始朝向为何，目标都是
`R_x(α) · R_E^W(0)`，因此旋转轴是机器人世界 `+X`。

这与 body/local-frame 旋转不同。若数据或操作者动作的语义是“绕当前手腕自身的轴旋转”，相对
旋转及 EEF 合成方式都不能仅靠互换一处左/右乘来猜测，必须先确认输入合同。

## 4. 已确认的事实与离线证据

| 结论 | 证据 | 边界 |
| --- | --- | --- |
| 位置与旋转现在使用同一固定标定 | `teleop/loop.py` 只把 `vr_calibration.transform` 传为 `vr_to_robot_rot`；`arm_mapper.py` 对位置用左乘、对旋转用共轭变换 | 不证明标定本身正确 |
| 空间旋转在当前公式下保持机器人世界轴 | `tests/test_arm_wrist_mapper.py::test_robot_world_x_rotation_remains_world_x_with_nonidentity_anchors` 使用非单位 wrist/EFF anchor 检查 `R_x` 的预乘结果 | 是合成几何测试，不含真实 tracker |
| 固定标定对位置和旋转一致 | `tests/test_arm_wrist_mapper.py::test_fixed_calibration_maps_position_and_rotation_with_one_transform` | 不验证真实标定采集姿势 |
| Unity→FLU 与四元数排列在单一入口完成 | `sensor/vr_worker.py` 对 wrist 使用 `unity_left_to_flu_rotation` 和 `xyzw_to_wxyz`，之后发布规范化四元数 | 仍需确认 HTS 的 `wrist` pose 是世界姿态 |
| IK/FK 不会天然把世界 `X` 换成其他轴 | 临时离线 MPlib target→IK→FK 检查中，`R_x(+5°) · R_E(0)` 的 FK 姿态残差约为 `9.12e-6 rad`；显示成 `-X, -5°` 是等价轴角表达 | 不是实机 SDK 运动验证 |
| tracking 毛刺不会写回原始姿态作为下一帧基线 | `ArmWristMapper` 保存 `_accepted_wrist_rot`，并用它计算下一帧的 per-frame delta | 该机制会限幅旋转幅度，不会修复坐标轴语义 |

离线测试在最近一次审查中全部通过；这些证据支持“公式在声明的数学合同下正确”，但不能证明现场的
输入和物理动作满足该合同。

## 5. 仍未闭合的根因假设

| 假设 | 为什么可能造成现象 | 如何证伪/确认 |
| --- | --- | --- |
| HTS wrist 四元数不是 `{}^V R_H` 世界姿态 | `R(t)R(0)^T` 会被解释为错误的空间 delta | 保存 q0/q1 与 source metadata，在离线分析中计算旋转轴；结合 HTS provider 合同复核 |
| 操作者做的是 wrist-local/body 旋转 | wrist 当前朝向非单位时，local `X` 通常不等于机器人世界 `X` | 明确以机器人 base 轴为参照做外在旋转，并重复不同 wrist anchor 的试验 |
| heading-only 标定遗漏了三维坐标基差异 | `R_z(-theta)` 无法吸收 roll/pitch 或上游 frame 定义错误 | 分别测量 robot `X/Y/Z` 的输入轴；若呈固定三维旋转偏差，设计 full-SO(3) 标定而非手工调 yaw |
| 原始 VR 轴正确，但后续 target/IK/执行链路改变了结果 | 真实目标可能因 total/per-frame limit、IK reject、collision gate 或控制器状态而保持/偏离 | 在不命令硬件的 harness 中记录 mapper target；随后受控实机记录 target、IK/FK、SDK accepted target 和 feedback |
| 旋转被限幅或 reset/re-anchor 打断 | 表现为“不转”或幅度不足，容易被误判为轴错误 | 对照 mapper 的 clamp/re-anchor 日志与 q0/q1 的角度 |

当前没有证据支持直接把 `ΔR_V` 改为 `R(0)^T R(t)`、把 EEF 合成改为右乘，或交换 `X/Y/Z`。
这些改动会使已经通过的空间旋转合同失效，并可能把一个输入语义问题变成隐蔽的姿态错误。

## 6. 首要证据：取得真实 VR 样本

当前没有专用的 VR-only 采集入口。后续需要验证时，应在不命令机器人的受控采集流程中保存两段
静止 wrist 四元数及其 source metadata，再离线计算：

- `q0`、`q1`：Unity→FLU 后的 WXYZ wrist 四元数；
- `VR axis`：`q1 q0⁻¹` 的空间轴，仍在 `V`；
- `predicted robot axis`：`{}^B R_V` 变换后的轴；
- `robot ±X axis error`：忽略正负方向后与机器人 X 轴的夹角；
- `robot +X direction error`：保留正负方向后的夹角。

采样动作必须以物理机器人 base 为参照，做约 10–20° 的机器人世界 `+X` **外在旋转**；不要把
“扭转手腕自身的 X 轴”当作同一个动作。建议至少重复三次，并保留每次完整原始样本与采样条件。

结果解释：

| 输出模式 | 下一步 |
| --- | --- |
| `predicted robot axis` 稳定接近 `[+1, 0, 0]`，且 `±X axis error` 很小 | 原始 VR + 固定标定满足轴不变性；继续检查 mapper target、IK/FK 和实际执行，不改坐标公式 |
| 稳定接近 `[-1, 0, 0]` | 轴本身仍是 X；`+X direction error` 约 180° 表示动作方向相反，应先核对操作者动作符号和 source pose 语义 |
| 稳定接近 `Y`、`Z` 或其它固定斜轴 | 输入 frame/标定不一致；先确认 Unity→FLU 和 HTS pose 语义，再设计标定修复 |
| 多次结果漂移大或角度很小 | tracking、手部未静止或动作参照不明确；先改善采样，不能据此改算法 |

## 7. 后续修正决策顺序

1. 保存至少三组 raw q0/q1/离线轴分析结果，并记录 wrist 初始朝向和物理动作说明。
2. 若原始诊断不满足预期，先修复或重新定义 **输入边界**：HTS pose 合同、Unity→FLU 转换或
   `T_vr_to_robot` 标定。若需要全三维标定，应新增有质量门禁、版本化 schema 和迁移/拒绝策略，
   不能覆写当前 yaw 数值以“试一试”。
3. 若原始诊断正确，增加一个不命令硬件的 mapper target trace，比较 q0/q1 推导的
   `ΔR_W`、EEF target 和预期 `R_x(α) · R_E(0)`。
4. 仅在 mapper trace 也正确后，进行受控、低速的实机检查；同时记录 mapper target、IK target、
   FK、SDK accepted target、实际 joint/EEF feedback，以及 limit/re-anchor/gate 日志。
5. 根据第一处不满足合同的边界修复，并为该边界加入可重复的纯函数回归测试；禁止通过删除
   limit、freshness、collision 或 SDK-boundary validation 来掩盖问题。

## 8. 安全约束

- 在真实数据尚未确认前，不运行完整 VR 遥操作来“碰运气”验证轴；需要采样时，先为该目的设计
  最小、无机器人命令的受控采集流程。
- 不手工编辑 `vr_transform.json` 来交换轴或反转四元数；该文件是 runtime 的单一标定来源，修改
  必须来自可复现的标定程序和质量证据。
- 若后续需要实机验证，先按既有 arm safety/lifecycle 边界执行，并从低速、无障碍、可随时撤权的
  条件开始；离线几何正确不等同于硬件安全。

## 9. 相关实现与测试

- [`sensor/vr_worker.py`](../dexmani_real/sensor/vr_worker.py)：HTS wrist/head 数据的 Unity→FLU
  变换与 VR ring 发布。
- [`teleop/vr_transform.py`](../dexmani_real/teleop/vr_transform.py)：固定标定 schema、SO(3) 校验与
  heading-only 合同。
- [`teleop/arm_mapper.py`](../dexmani_real/teleop/arm_mapper.py)：reset-relative wrist→EEF 映射、
  accepted pose 和旋转限幅。
- [`teleop/loop.py`](../dexmani_real/teleop/loop.py)：加载固定标定并构造 mapper。
- [`tests/test_arm_wrist_mapper.py`](../tests/test_arm_wrist_mapper.py)：固定标定、world-X 旋转及
  accepted-pose 行为的离线合同。
