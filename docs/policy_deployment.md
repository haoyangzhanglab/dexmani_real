# 策略部署

`dexmani_policy` 提供模型及观测／动作描述，本仓库负责读取观测、调度预测、执行控制和录制评估数据。

## 数据流

**传感器与机器人反馈 → 观测历史 → 模型推理 → 动作序列调度 → 动作映射 → 机器人**。

观测按模型需求组合图像、关节、触觉和点云。绝对关节动作仅按运行关节限位裁剪，保留输入的角度分支，不根据实测位置自动换成 2π 等价角。末端动作经共享局部 IK 转为关节目标；IK 允许保留跟踪残差。Raw 保存观测与指令提交尝试，不代表两设备原子执行或物理到位。

## 核心模块

| 模块 | 职责 |
| --- | --- |
| [session.py](../dexmani_real/deployment/session.py) | 装配模型、传感器和机器人，组织多段评估与结果汇总。 |
| [observation.py](../dexmani_real/deployment/observation.py) | 将观测历史编码为模型输入。 |
| [inference.py](../dexmani_real/deployment/inference.py) | 加载、预热、预测与关闭模型。 |
| [schedule.py](../dexmani_real/deployment/schedule.py) / [rollout.py](../dexmani_real/deployment/rollout.py) | 接续动作序列，推进单段观测、推理、控制与录制。 |

设备交互、运动约束和 Raw 存储复用 `robot`、`control`、`data` 模块。

## 执行模式

| 模式 | 预测与动作接续 |
| --- | --- |
| sync | 等待预测，执行一段动作后再请求下一次预测。 |
| async | 执行当前序列时提前预测，按观测时间接续仍有效的新动作。 |
| rtc | 支持 RTC 的模型接收当前动作前缀，按已提交进度接续预测。 |

## ODE 的动作承接

加载模型后，`PolicyInfo.requires_prev_action` 标识 ODE。它只支持 `sync` 且 `execution.action_steps == n_action_steps`；默认从模型读取步数，异步、RTC 和缩短 chunk 会在连接设备前报错。其他策略保持原有调度与预测参数。

每个 Rollout 从 `prev_action=None` 开始。每条 `Robot.send_action()` 正常返回后，将裁剪后的 Arm 7 维、Hand 12 维目标合并为新的物理单位数组；完整执行 H 步后，把最后一次提交的目标传入下一次预测。不会提前使用预测末帧，也不会用实测关节位置覆盖目标。提交异常不推进承接值或动作游标，现有错误流程结束本段；中途停止立即终止并清空承接，下一段重新初始化。预热不发送指令，不改变承接值。

这里的提交目标与 Raw 的指令字段语义一致；正常返回不证明两设备原子执行、SDK 回执或物理到位。Arm 的 Ruckig 中间参考仍由驱动管理，不作为 ODE 起点。裁剪生效时，下一 chunk 从裁剪目标重新起步，这是执行约束适配，不能视为原始无约束积分曲线的连续延伸。

不连接设备的回归测试：

```bash
python -m unittest discover -s tests -p 'test_policy_execution.py' -v
```

实机闭环、SDK 提交与伺服跟踪仍需单独验收。

`--inference-steps` 对 ODE 设置每个控制周期的积分子步数，不是整个 chunk 的场计算次数；它不改变 H 或控制周期。

## 来源时间与时效

观测采用 latest-value 语义：有效样本可重复读取，读取不刷新来源时间。年龄按主机 `time.monotonic_ns()` 计算；Arm／Hand 使用 SDK 查询开始时间，Camera 使用新帧首次接收时间，VR 使用完整腕部／关键点观测中较早的接收时间。这些均非同步硬件采样时间，Hand SDK 也没有可靠的硬件采样序号或时间戳。

| 用途 | 默认预算 | 约定 |
| --- | --- | --- |
| Arm 反馈 | 160 ms（`4 / 25` s） | 默认 25 Hz 查询，查询周期须小于 TTL。 |
| Hand／模型必需触觉 | 200 ms | 共用主机查询时效预算。 |
| Camera／普通 Point Cloud | 约 133 ms（`4 / 30` s） | 点云继承相机来源时间与序列，处理消耗同一预算。 |
| HOI | 250 ms | 独立的 `execution.hoi_max_age_s`，仍按来源相机时间计龄。 |
| VR 来源 | 500 ms | 暂定掉线容忍度，不能由控制频率推导。 |
| VR／Keyboard 目标租约 | 250 ms | 与来源 TTL 独立，由控制循环续租。 |
| 策略命令租约 | 2 s | 命令到期后停止执行。 |

配置见 [config](../dexmani_real/config/)。HOI 预算来自已记录回放，依据与验证范围见[性能结论](object_reconstruction.md#当前实现与性能结论)；普通点云延迟和 VR 真实生产频率仍待测量。

`ObservationNeeds` 只检查消费者实际需要的模态。模型输入失效时清空历史并暂停新预测，恢复首帧重新填满 `n_obs_steps`。普通策略的已有有限动作序列和在途预测按原模式继续；HOI 缺失、过期或位姿失效立即停止当前计划。关键机器人反馈始终须有效。

## Raw 录制与连续区间

- 模型所需 RGB／Depth 与 HOI 或普通点云同源；Raw 独立读取控制采样时最新的 Camera RGB-D，按普通 Camera TTL 检查并保留实际 `camera_ns`、`camera_sequence`。
- 录制要求两个触觉通道各出现过一次有效值，之后可 Hold 最近有效值及原时间戳；显式 reset 使旧值失效。模型不需要的相机／触觉只在启用录制时成为依赖。导出仅诊断来源时间非正、未来或回退，不修补样本。
- 相机暂不可用或队列满时丢弃整行，后续有效行继续接收；仅供录制的 Camera Worker 退出不终止控制，模型所需服务和设备 Owner 故障仍须处理。
- 已知文件／编码 I/O 失败停止录制、保留 staging 并报告，控制继续；初始化失败则不启动要求录制的实验。Writer 实际退出后才完成收尾，发布与失败以 `Recorder.result()` 为准，`poll_finish()` 仅表示收尾完成。程序与设备错误继续传播。
- RGB／Depth／状态逐行对应，MP4 PTS 相对首行控制时间。质量拒绝、队列满、暂停或调度跳槽后，在下一条已保存行记录 `/timing/segment_starts`；非空录制从 0 开始、严格递增，`recording_truncated_reason` 仅作诊断。
- 导出与回放必须有合法 `segment_starts`，不从时间间隔猜测断点。导出保留有效行，每个连续区间形成一个 Zarr `meta/episode_ends` 边界，训练窗口不跨断点；`raw_episode_count` 统计文件，`episode_count` 统计区间。区间按标称频率使用，周期抖动只告警、不重采样；控制时间倒退、非法数组和所选字段非有限值仍拒绝。

多区间回放须显式选择，如 `python scripts/replay_episode.py EPISODE --segment 0`。编号从 0 开始，非法选择在连接设备前报错；回放以所选区间首行检查起始姿态，不跨段或补过渡动作。批量导出可用 `--exclude` 排除不符合契约的 Episode；失败 staging 不发布。

## 暂停、结束与 HOME

| 场景 | 行为 |
| --- | --- |
| Teleop 输入／反馈暂不可用或目标过期 | 确认两设备停止后软暂停；恢复先完成 Arm 准备，清空旧邮箱与伺服参考，再用准备后的新反馈和有效 VR 重新锚定。暂停期间不提交、不续租，Episode 时长继续计时。 |
| Policy 反馈暂不可用 | 结束当前 Episode，不在同段自动恢复；设备、录制和在途预测收尾后才可开始下一轮。 |
| HOME 的 `FeedbackUnavailable` 或手部 `HomeTimeout` | 本次 HOME 未完成；停止收尾成功且无硬故障后保留空闲会话，可重新按 H 或按原流程 B。手部失败时不继续 Arm HOME，不自动重试。 |

Teleop 暂停／恢复期间，S/C/D 结束本段，Q 保存退出，Esc 急停；HOME 期间 S/Q/Esc 也优先。来源不明的停止不自动恢复；停止未确认、真实 SDK 故障、程序错误及底层普通 `TimeoutError` 继续退出或报错。Keyboard Jog 保持其退出边界。

Teleop 自动恢复尚未完成实机验收。SDK 接受停止请求不等于物理停稳；Arm 准备检查实测停稳，Hand 无等价证明。仍需验收输入中断、重新锚定、反馈异常与硬故障区分，以及暂停／准备期间的操作指令优先级。

返回[项目架构](../README.md)。
