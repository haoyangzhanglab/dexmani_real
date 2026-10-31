# XHand 手部重定向

使用 Pinocchio FK、解析雅可比和一次有界 NLopt SLSQP 求解，联合拟合近端/末段方向、骨链位置和有意对指距离。目标是按 XHand 的实际自由度匹配整手姿态。

## 模块与调用链

| 模块 | 职责 |
| --- | --- |
| `teleop/retargeting/landmarks.py` | HTS 输入校验、掌系、固定骨长标定及 XHand 目标对应 |
| `teleop/retargeting/objective.py` | 位置、方向、对指、阻尼的函数值与解析梯度 |
| `teleop/retargeting/adaptive.py` | 求解限位、滤波、诊断及提交状态 |

在线入口为 `sensor/vr.py` → `teleop/vr.py::VRController` → `AdaptiveHandRetargeter`。候选求解不推进状态；只有动作提交成功并在授权锁内复核后，`commit` 才从实际 `RobotCommand` 更新 warm start、阻尼中心、滤波中心和时间戳。重复读取同一 VR 缓存帧（由 `ring_sequence` 识别）时复用手部候选，每个新段从新实测关节重置，会话内骨长标定保留。无效输入或求解失败明确终止当前段，不静默保持旧目标冒充成功。

失败或异常控制步会将诊断保存到 `termination_details` 的 `stage="control_timing"` 条目；`hand_retargeting` 字段来自本步重定向调用已生成的诊断。成功控制步不逐帧保存这些诊断；重复读取同一 VR 缓存帧时不附带上次求解的诊断。

## 输入与机构对应

右手输入为米制、HTS 原始 21 点：wrist=0；拇指 CMC/MCP/IP/TIP=1/2/3/4；四指 MCP/PIP/DIP/TIP=5–8、9–12、13–16、17–20。接收端将 Unity 左手 XYZ 转为 FLU `(z,-x,y)`。掌系由 wrist→middle MCP 与 pinky MCP→index MCP 确定：+X 掌侧、+Y 拇指侧、+Z 中指侧，不逐指配准。

启动采集 25 个新鲜且来源时间递增的样本，固定 15 段骨长和掌宽的中位数；不要求平手，不从弯曲前缀估计零关节角。骨长范围 2–180 mm、相对 MAD≤10%、相对骨长变化≤35%。新录制的 `meta.hand_retargeting_calibration_json` 保存 `sample_count`、`human_bone_lengths_m`（5×3，米）及 `human_palm_width_m`（米）。

每个候选帧先完成掌系与骨长校验，无效时继续采样；通过后只保留已算出的 16 项测量，最终直接求 median/MAD，不再次准备原始帧。URDF 的中立骨段、掌宽、拇指基点与圆轨迹半径在重定向对象构造时固定，逐帧映射复用这些几何量。

| 手指 | 人体方向 | 机器人骨段 | 限制 |
| --- | --- | --- | --- |
| 拇指 | MCP→IP、IP→TIP | thumb_rota_link1→link2→tip | 三轴不能表达完整人体拇指自由度 |
| 食指 | MCP→PIP、DIP→TIP | index_rota_link1→link2→tip | 约 ±10° 侧摆，没有独立 DIP |
| 中/无名/小指 | MCP→PIP、DIP→TIP | 对应 link1→link2→tip | 两轴屈曲，没有侧摆和独立 DIP |

四指位置目标使用固定 MCP 基点及骨长比例，合并人手 PIP→DIP→TIP；方向目标独立保留原始 DIP→TIP。两者不一定能同时满足，不能把人体 DIP 硬映射到不存在的关节。拇指 MCP 目标按掌骨 XY 方位投影到 URDF 的固定高度圆轨迹；投影退化时关闭不可观测的位置项，仍保留指骨方向与对指目标。

重定向、数据导出和策略观测共用组合 URDF 的手部子树。SDK/URDF 顺序按关节名对应，雅可比在堆叠后统一转换为 SDK 列顺序；运行限位由 `hand.qpos_min_rad/qpos_max_rad` 指定，并检查其包含于 URDF 限位。目标映射复用骨段长度进行标定容差检查和方向归一化。

## 目标、参数与运行约束

位置残差按固定 0.10 m 归一化，使用 0.02 m 阈值的径向 Huber；归一化尺度不作为配置项，以免与骨链、指尖及捏合权重重复调节目标强度。方向使用 `Huber(u_robot-u_human, 0.3)/0.3`，阈值约对应 17.3°。方向梯度包含完整归一化导数 `(I-uuᵀ)J/length`。阻尼系数为 `damping_s/dt`。

| 参数 | 当前值 |
| --- | ---: |
| shape / tip 权重 | 0.25 / 2.0 |
| proximal_direction / direction 权重 | 2.0 / 1.0 |
| direction_huber_delta | 0.3 |
| pinch 权重 | 8.0 |
| pinch 开始 / 完全激活距离 | 30 / 15 mm |
| damping_s / smoothing_tau_s | 0.002 / 0.02 s |
| 最终命令变化率上限 | 4 rad/s |
| 求解预算 | 50 次 / 25 ms |

次数预算限制求解工作量，时间预算限制计算耗时。`ftol_abs` 同时作为 NLopt 的绝对 loss 收敛容差，以及接受判据 `final_loss <= seed_loss + ftol_abs` 的容差；两处使用同一 loss 单位，默认值与接受集合保持原状。阻尼在优化中惩罚相对已提交命令的位移，EMA 在求解后平滑输出，两者作用不同。

捏合是辅助目标：掌宽归一化距离门控乘以末段夹角 60→90° 的门控，使用观察到的缩放距离，不强制固定间隙，也不削弱姿态项。它只是几何意图代理，数据中没有接触或意图真值。

求解变量是滤波前候选。最终命令为 `previous + alpha*(candidate-previous)`，`alpha=-expm1(-dt/tau)`；`tau=0` 时 `alpha=1`。`dt` 来自本次控制观测与上次成功提交观测的主机时间戳差，不使用 VR 来源帧间隔；准备阶段同一观测的 reset 与预计算使用名义控制周期。搜索步长除以 alpha，保证滤波不额外压低最终变化率上限。返回前按限位裁剪浮点舍入误差，提交时严格校验关节范围。运行链路在提交前检查反馈时效。

## 验证边界与限制

XHand 无法独立表达人体 DIP 屈曲及中/无名/小指侧摆，拇指也只有三个驱动轴。方向与位置目标可能无法同时满足；指尖接近不能证明整手姿态一致。强屈曲和部分拇指对掌姿态仍需进一步验证，离线求解结果不代表机器人已经执行或达到目标，当前实现尚未经真机验收。

实现参考 [Sharpa](https://github.com/sharpa-robotics/sharpa-vision-retargeting-sdk/blob/73ac80e980b980b11d8853b874f8efea79095add/supplements/teleop_hand.py) 的固定形态思想及 [Wuji](https://github.com/wuji-technology/wuji-retargeting/blob/531f6ed4250b475d2e9231f54e988fc9b1c5b4ea/wuji_retargeting/opt/adaptive_analytical.py) 的形态/方向联合拟合，没有照搬高自由度手对应或增加通用优化框架。
