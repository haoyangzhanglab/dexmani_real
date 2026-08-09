# DexMani Real 全量深度代码审查总报告

> 审查对象：DexMani Real `167f15a5f76b798ea5a90e44fe3e478eecc266d2`  
> 审查日期：2026-08-09  
> 报告范围：本次聊天中发现和复核的全部 xArm7、XHand、retarget、policy 部署、进程生命周期、多模态同步、录制、质量分析与 replay 问题  
> 安全边界：只做静态检查、mock、纯数学验证和已有 HDF5 只读分析；未连接、探测或控制真实硬件

## 1. 文档目的与完整性声明

本报告是本次聊天的统一交付，不再只覆盖最后一次 XHand SDK/retarget 专项。它完整收录：

- xArm7 闭环审查的 16 个正式 finding；
- policy 部署、进程管理与数据同步审查的 20 个正式 finding；
- XHand SDK、触觉与 retarget 审查的 10 个正式 finding；
- 前序审查明确点名、但未进入上述 finding 表的 4 个录制与 camera 数据链缺陷。

因此本文保留 **50 条问题记录**。其中 46 条沿用原报告编号，4 条补充编号为 `DATA-01` 至 `DATA-04`。部分问题从不同层面描述同一根因，例如 `F-13` 与 `PD-10`、`PD-08` 与 `DATA-02`；本文不会因存在关联而删除原 finding，也不会把 50 条简单理解为 50 个互不相关的修复任务。

三份专项报告仍保留为证据附录：

- [xArm7 完整闭环深度审查](xarm7_closed_loop_deep_review_167f15a.md)
- [策略部署、进程管理与数据同步深度审查](policy_deployment_process_sync_deep_review.md)
- [XHand SDK 使用与 Retarget 专项深度审查](xhand_sdk_retarget_special_review_167f15a.md)

本总报告可以独立阅读：每条问题均给出状态、主要源码位置、触发与影响、最小修复方向和验证要求；专项报告提供更长的逐行调用链、数值输出和对照项目证据。

## 2. 版本、方法和严重度

### 2.1 固定版本

| 项目 | Commit | 用途 |
|---|---|---|
| DexMani Real | `167f15a` | 被审查实现 |
| LeFranX | `a39906e` | Quest、DexPilot、坐标变换和 XHand 使用对照 |
| DexUMI | `acddb8f` | XHand 状态、增益、触觉和记录对照 |
| Dexora | `6f2869d` | 独立硬件进程、节流和单位边界对照 |
| TAG | `2f5b1ab` | 两阶段 NLopt、Pinocchio Jacobian、pinch 和映射对照 |

参考项目只能证明某种做法或契约存在；“实现不同”本身不构成缺陷。xArm SDK 语义还结合了本机安装的 `xarm-python-sdk 1.18.4` 源码和原专项中记录的官方文档。

### 2.2 证据标准

- **离线复现**：纯函数、mock、共享内存、临时 HDF5 或进程模型可稳定复现；
- **静态确认**：源码调用链足以确定软件行为，固件或调度的最终物理幅度仍可能需要实机；
- **条件性风险**：现有 VR policy 不一定触发，但公开接口允许未来 policy/replay 触发；
- **待实机确认**：源码只能确认变量未受控或观测不足，不能断言固件实际运动后果。

只有具备可达调用链、数值不变量、确定性复现或录制数据证据的问题才列入 finding。厂商单位、真实接触力、制动距离和绝对精度等无法离线裁决的内容单独列为待验证项。

### 2.3 严重度

- **P0**：可能把未经统一安全验证的动作送入硬件，或执行危险旧轨迹；修复前必须阻断相应路径；
- **P1**：可能导致错误动作、故障漏报、停止不可确认、跨执行器失协、严重时序错配或数据语义损坏；
- **P2**：稳定造成性能、精度、恢复、可观测性、可复现性或架构边界问题，但直接物理风险已有明显限制；
- **P3**：主要影响非默认模式、诊断一致性或开发体验。

### 2.4 数量统计

| 领域 | P0 | P1 | P2 | P3 | 合计 |
|---|---:|---:|---:|---:|---:|
| xArm7 闭环 `F-*` | 1 | 7 | 8 | 0 | 16 |
| Policy/进程/同步 `PD-*` | 1 | 12 | 7 | 0 | 20 |
| XHand/Retarget `XH-*` | 0 | 4 | 5 | 1 | 10 |
| 数据链补充 `DATA-*` | 0 | 4 | 0 | 0 | 4 |
| **总计** | **2** | **27** | **20** | **1** | **50** |

## 3. 全部问题索引

### 3.1 xArm7 闭环

| ID | 等级 | 状态 | 问题 | 核心影响 |
|---|---:|---|---|---|
| F-01 | P0 | confirmed | live replay 绕过碰撞与 workspace 检查 | 端点合法但中间碰撞的路径可进入 Mode 6 |
| F-02 | P1 | confirmed | Mode 6 进入/恢复顺序及后置条件错误 | 可能在 state/mode 未就绪时发布 ready |
| F-03 | P1 | confirmed | `DISARMED` 未映射到控制器 state 4 | 软件安全状态与真实控制器状态失真 |
| F-04 | P1 | confirmed | 状态读取失败仍刷新 timestamp | 陈旧反馈被伪装为新鲜状态 |
| F-05 | P1 | confirmed | 阻塞队列、退出优先级和软件急停竞态 | 故障可误报正常退出，停止不可确认 |
| F-06 | P1 | confirmed | C24 自动恢复重发原故障目标 | 恢复未消除诱因并可能反复触发 C24 |
| F-07 | P1 | confirmed | canonical `--no-hand` readiness 依赖错误 | policy heartbeat 约 1 秒后触发 FAULT |
| F-08 | P2 | confirmed | sent/accepted/executed 录制语义混淆 | replay 和质量分析无法还原命令生命周期 |
| F-09 | P2 | confirmed | 回零收敛忽略 qvel | 运动中穿过容差带即可提前 ACK |
| F-10 | P2 | confirmed | JSON 动力学配置被 CLI 默认覆盖且绕过校验 | 配置意图、运行值和 provenance 不一致 |
| F-11 | P1 | confirmed | 姿态 EMA 在 ±π 分支走长旋转路径 | 2° 连续输入可生成 89.5° 中间目标 |
| F-12 | P2 | 部分待实机 | arm worker 30 Hz 重发旧目标 | 可能反复触发 Mode 6 重规划 |
| F-13 | P2 | confirmed | 平滑绑定名义 16 Hz，动作无 TTL | jitter/backlog 改变带宽并执行旧折返点 |
| F-14 | P2 | confirmed/效果待实机 | jerk 未受控，replay 未复现完整动力学 | 平滑和复现实验条件不可比较 |
| F-15 | P2 | confirmed | tracking/replay 指标未做命令—反馈时齐 | latency 被错误计为空间跟踪误差 |
| F-16 | P2 | confirmed | FK 使用通用 URDF，无机身专属校准 | 内部 FK 不能证明绝对笛卡尔精度 |

### 3.2 Policy 部署、进程管理与同步

| ID | 等级 | 状态 | 问题 | 核心影响 |
|---|---:|---|---|---|
| PD-01 | P0 | 条件性 confirmed | 缺少所有 policy 必经的统一动作安全网关 | 新模型可绕过 workspace/collision gate |
| PD-02 | P1 | confirmed | 默认 `fork` 与 GPU/线程运行库不兼容 | 启动可能死锁或继承无效 CUDA/线程状态 |
| PD-03 | P1 | confirmed | daemon policy 禁止创建子进程 | DataLoader、多进程预处理和模型服务不可用 |
| PD-04 | P1 | confirmed | 没有模型级 policy ready/能力握手 | 模型未 warmup 即 ARMED，依赖图错误 |
| PD-05 | P1 | confirmed | 推理、heartbeat、急停输入和录制共线程 | 慢推理同时冻结安全协调和记录时钟 |
| PD-06 | P1 | confirmed | policy 异常会被报告为正常 Q 退出 | OOM/shape/CUDA 故障失去真实事故语义 |
| PD-07 | P1 | confirmed | 卡死 policy 未确认退出即关闭 IPC | 残留进程可访问已 unlink 的共享内存 |
| PD-08 | P1 | confirmed | 没有因果一致的多模态快照 | 同一 observation 混入不同物理时刻数据 |
| PD-09 | P1 | confirmed | camera 时间戳不在 host monotonic 域 | 无法与 arm/hand 直接做可靠时间差 |
| PD-10 | P1 | confirmed | action 无 observation 因果链和 TTL | 旧推理结果和 queue backlog 仍可执行 |
| PD-11 | P1 | confirmed | arm 与 hand 不是可追踪的联合动作 | arm 执行旧版本而 hand 已执行新版本 |
| PD-12 | P1 | 条件性 confirmed | action chunk 没有安全调度边界 | chunk backlog、重叠和跨执行器错配 |
| PD-13 | P1 | 条件性 confirmed | policy 热重启无 epoch/排空协议 | 新实例继承旧 queue、ring 和 recurrent state |
| PD-14 | P2 | confirmed | 缺少通用 Policy/Observation/Action 接口 | backend 与 SharedStorage、VR 实现紧耦合 |
| PD-15 | P2 | confirmed | ring 容量不足且无跨模态历史对齐 | 不能支持常见 0.5–2 秒 observation horizon |
| PD-16 | P2 | confirmed | camera read 总复制 RGB/depth/pointcloud | 无谓内存带宽、分配和 seqlock 竞争 |
| PD-17 | P2 | confirmed | 缺少模型资源治理和实时准入 | 无法用 p99、显存和 deadline 判断可部署性 |
| PD-18 | P2 | confirmed | 训练—部署闭环缺少模型与输入语义 | episode 无法复现真实模型输入和动作阶段 |
| PD-19 | P2 | confirmed | 配置解析不适合插件和可复现实例 | singleton 被原地污染，非法值可绕过校验 |
| PD-20 | P2 | confirmed | 测试未覆盖策略部署故障模型 | 现有全绿无法证明 hang/TTL/epoch 等安全性 |

### 3.3 XHand SDK、触觉与 Retarget

| ID | 等级 | 状态 | 问题 | 核心影响 |
|---|---:|---|---|---|
| XH-SDK-01 | P1 | confirmed | post-open 异常绕过统一清理 | 后续可能需要 24 V 断电才能重连 |
| XH-SDK-02 | P1 | confirmed | send 失败不锁存全局 fault | 手不动作但系统仍可保持 RUNNING |
| XH-SDK-03 | P1 | confirmed | 静止 qpos 被误判 stale 并自锁 | 正常保持约 0.5 秒后 retarget 可永久冻结 |
| XH-RT-01 | P1 | confirmed | 退化 landmarks 被当作有效手姿 | 全零输入仍产生大幅闭合目标 |
| XH-TACT-01 | P2 | confirmed | release 后 raw tactile 无限复用 | `contact=False` 可配上旧接触 taxel |
| XH-TACT-02 | P2 | confirmed | 启动 bias 未执行无接触前置条件 | 真实启动载荷被吸收为零点 |
| XH-RT-02 | P2 | confirmed | pinky scaling 混用新旧父节点 | 中间骨段方向可反转 |
| XH-RT-03 | P2 | confirmed | DexPilot reset 使用错误逆映射 | warm-start 最大错位 75.66° |
| XH-ARCH-01 | P2 | confirmed | TAG/policy 连带导入厂商 XHand SDK | 破坏“仅 hand worker 导入 SDK”的进程边界 |
| XH-SDK-04 | P3 | confirmed | 隐式 stub 的连接和反馈语义矛盾 | `connect=True` 但 `is_connected=False` |

### 3.4 录制与 camera 数据链补充

| ID | 等级 | 状态 | 问题 | 核心影响 |
|---|---:|---|---|---|
| DATA-01 | P1 | confirmed | missing grid slot 被下一条未来样本回填 | 无效槽携带 future observation/action，易形成训练泄漏 |
| DATA-02 | P1 | confirmed | camera freshness 未在 policy/recorder 执行 | 相机停流时旧帧仍被当作当前帧记录 |
| DATA-03 | P1 | confirmed | RGB MP4 内部缺帧未按 grid span 补齐 | 缺帧后的 RGB 整体前移，与 state/action 错位 |
| DATA-04 | P1 | confirmed | quality filter 过滤 data.h5 但原样复制 sidecar | 输出的 depth/RGB 与过滤后的时间轴不一致 |

## 4. xArm7 闭环详细复核

### F-01 — live replay 绕过碰撞与工作空间检查

- **证据**：`examples/real/replay_traj.py:179-296,823-837,917-1001` 只验证 dataset、finite、connected 和 error；起点对齐失败后使用 `np.linspace`。`dexmani_real/robot/arm_loop.py:383-408` 仅检查 finite、等价带和关节限位。
- **触发与影响**：轨迹端点在限位内，但中间发生 self collision、arm-hand collision、穿桌或越 workspace 时，未检查的中间目标仍会进入 Mode 6。C22/C31 是固件 backstop，不能替代项目环境模型。
- **现有保护不足**：安全的 EEF approach 只覆盖部分起点对齐；正式 replay 和 joint-linear fallback 没有复用 `check_path_collisions()`、workspace/table 检查和 `plan_joint_qpos_path()`。
- **最小修复**：移除未检查的 linear fallback；加载时拒绝 invalid/disconnected/grid-fill/safety-reject；对每个相邻段密集做 arm-arm、arm-hand、workspace 和 table preflight，并用轨迹 hash 绑定验证结果。
- **回归**：构造“两端安全、中点碰撞”轨迹，断言零命令入队；覆盖 NaN、越界、错误 shape 和被修改后的 hash。

### F-02 — Mode 6 进入/恢复顺序和后置条件不完整

- **证据**：`dexmani_real/robot/arm_loop.py:153-174,426-451,540-544` 在启动和恢复中存在 mode/state 顺序不一致；`813-816` 的 homing 恢复反而使用另一顺序。
- **影响**：`set_mode(6)` 后控制器可能处于 state 5；代码使用后台 cached `arm.state`，可能在 live mode/state 未满足时设置 `arm_ready`。
- **修复**：唯一 lifecycle helper 执行 `motion_enable → set_mode(target) → set_state(0) → settle → live get_state/get_err_warn_code → verify mode/state`；任何 setter 或后置条件失败都 sticky fault。
- **回归**：mock cached state=2、live state=5；setter 返回 0 但 live error 非零；覆盖 Mode 0↔6 双向转换。

### F-03 — `DISARMED` 与控制器 state 4 不一致

- **证据**：`dexmani_real/robot/safety.py:26` 承诺 DISARMED 对应 state 4；`arm_loop.py:155-164,207-211,586-591` 却在主进程仍 DISARMED 时 enable、state 0、Mode 6 并 ready。
- **影响**：软件禁止消费动作，但真实控制器已处于可运动状态；VR 最长等待期间以及 early-return cleanup 都不满足安全状态契约。
- **修复**：ready 前保持并 live-confirm state 4；只在 ARMED 边沿进入 Mode 6/state 0；FAULT/DISARMED 回 state 4；main 等 controller ACK 后才能 ARMED。
- **回归**：测试完整 `DISARMED→ARMED→RUNNING→FAULT→DISARMED` SDK 调用序列，配置各阶段失败都必须 stop 后 disconnect。

### F-04 — arm feedback 读取失败仍产生“新鲜”状态

- **证据**：`arm_loop.py:482-500,521-527,553-572` 在读取失败时 forward-fill `last_qpos`、清零 qvel/tau，却写当前 monotonic timestamp；`shared_storage.py:466-493` 的 helper 丢弃源 timestamp/mode。
- **影响**：policy、calibration 和 replay 把旧 qpos 用于 FK、tracking 和安全判断；heartbeat 只能证明 worker 活着，不能证明设备反馈新鲜。
- **修复**：分离 worker publish time 和 device source time；只有 shape=7、finite、SDK success 才推进 source timestamp；连续失败锁存 fault，所有消费者使用统一 age/connected gate。
- **回归**：异常、短数组、NaN/Inf 都不得推进 source timestamp；达到阈值后 FAULT，恢复必须发布新 source timestamp。

### F-05 — 阻塞 producer、退出优先级和软件急停未闭环

- **证据**：bounded `arm_action_q` 定义于 `shared_storage.py:56,256`；keyboard、calibration、replay 多处使用阻塞 `put()`；policy timeout 同时影响 `is_running`；supervisor 和 arm loop 对 estop/shutdown 的判断顺序见 `shared_storage.py:879-905`、`arm_loop.py:289-301,577-582`。
- **影响**：consumer 卡死时 producer 无法轮询 ESC；policy fault 可被解释为正常 Q；`is_running=False` 可能使 arm loop 在执行 emergency stop 前退出；强制 terminate 后 cleanup 不执行。
- **修复**：统一有界 put helper；结构化 sticky fault；supervisor 按 estop/fault→worker death→heartbeat→explicit quit 排序；worker 无条件 loop 顶部优先处理 estop，stop 必须 live-confirm state 4。
- **回归**：填满 queue、同时设置 estop 和 shutdown、第一次 state 4 失败、worker 忽略 SIGTERM 等场景均需有确定结果。

### F-06 — C24 恢复后重发原故障目标

- **证据**：`arm_loop.py:420-458,529-546` 清 error/warn 后保留 `last_target`，下一 tick 以同样 speed/acc 再发。
- **影响**：恢复没有消除目标、速度、奇异点或跟踪误差诱因，可能约 1 秒内重复 C24；恢复 setter 返回码也未完整检查。
- **修复**：首次 C24 读取 fresh measured qpos，发送一次 measured hold；正确恢复并验证 Mode 6，降低动力学或要求新命令；恢复失败直接 sticky fault。
- **回归**：mock C24 后下一命令不得是原目标；覆盖 clean/mode/state 失败和 cached error 延迟。

### F-07 — canonical `--no-hand` 无法正常启动

- **证据**：main 在 `examples/real/vr_teleop_hand_record.py:67,121-154` 不创建 hand worker，但 `vr_teleop_policy.py:471-489` 仍无条件等待 `hand_ready`。
- **影响**：main 已 ARMED 并 seed heartbeat，policy 尚在等待 hand；约一个 heartbeat threshold 后系统进入 FAULT。
- **修复**：ready/heartbeat/process 依赖统一由 enabled-component graph 生成；新增真实 `policy_ready`，main 必须等待它。
- **回归**：`hand_enabled=False` 不创建 hand event/worker也能持续运行；启用 hand 时仍 fail-closed 等待。

### F-08 — sent、accepted、executed 语义混淆

- **证据**：`episode_recorder.py:120-123,458-465`、`vr_teleop_policy.py:964-984,1353-1400,1509-1817` 和 `timestamp_buffer.py:188-204` 将 policy intent、queue put、最近 SDK success 和同 grid feedback 混入一行。
- **影响**：`action_arm_joint_sent` 不能证明 SDK accepted；held tick 可能没有 queue write；`arm_last_cmd_seq` 不一定对应本行 action；replay/quality 无法重建命令生命周期。
- **修复**：兼容增加 queued/action seq、accepted seq/qpos、feedback source time/age、mode/error；旧 sent 字段只解释为 queued/last intent。
- **回归**：queue success+SDK reject、SDK success+反馈未到、held 未入队三种状态必须可区分并按 sequence 关联。

### F-09 — homing 只看位置误差、不看速度

- **证据**：`arm_loop.py:718-764,780-783` 已读取 qvel，但 milestone 仅要求两个 qpos 样本进入容差。
- **影响**：机械臂高速穿越容差带时可提前进入下一 milestone；最终 ACK 可能发生在仍运动时。
- **修复**：同时限制 max qvel，并要求 200–500 ms 稳定驻留；最终 live-check error/state/mode，恢复 Mode 6 后再次确认 READY。
- **回归**：容差内高 qvel、穿越、振荡、读失败和 Mode 6 恢复失败均不得成功。

### F-10 — JSON、CLI 和 dataclass 校验不一致

- **证据**：`vr_teleop_hand_record.py:63-75,113-128` 的 argparse 默认 speed/acc 会覆盖 JSON；`defaults.py:532-574` 通过 `setattr` 修改 singleton，不重跑 `__post_init__`。
- **影响**：用户配置、打印配置、实际 ArmLoopConfig 和 HDF5 metadata 可能不一致；非法负值可绕过初始化校验。
- **修复**：CLI 默认 `None`；固定 `explicit CLI > JSON > defaults`；解析到新 dataclass，完整校验后原子发布并记录 resolved hash。
- **回归**：JSON-only、CLI-only、冲突和非法值矩阵；runtime、policy 和 metadata 必须一致。

### F-11 — rotation-vector EMA 在 ±π 处选择长弧

- **证据**：`signal_utils.py:10-78` 对绝对 rotvec 线性混合；`arm_mapper.py:243-247` 已维持 quaternion 连续性，但转换时又按 `w>=0` 翻转。`+179°→-179°` 离线复现输入短弧 2°、第一输出离前态 89.5°。
- **影响**：腕部小幅连续动作可变成长路径回摆；joint delta、collision 和 workspace gate 只限制每帧，不恢复正确意图。
- **修复**：在相对旋转上做 `exp(alpha*log(q_target*q_prev^-1))*q_prev` 或 shortest-arc SLERP，按相邻 quaternion dot 选符号。
- **回归**：任意轴 ±π、`q/-q` 等价和 property test；输出步长不得超过输入短弧。

### F-12 — 无新 action 时仍以 30 Hz 重发旧 endpoint

- **证据**：`arm_loop.py:288-315,381-418` 只有新 action 才更新 `last_target`，但每 tick 都调用 `set_servo_angle(last_target)`；policy 默认 16 Hz。
- **影响**：重复发送是确定事实；固件是否对完全相同目标去重未知。若每次按官方 Mode 6 语义触发重规划，可能造成 settling 拖尾、qvel ripple 或额外 C24。
- **修复**：motion setter 仅在收到新 action 时调用；TTL 到期只发送一次 measured hold，之后等待恢复，不在 arm worker 插值。
- **回归**：queue 空 100 tick 时 setter=0 次；单 action=1 次；实机 A/B 比较重发和 new-action-only。

### F-13 — 固定周期平滑与无 TTL 的 FIFO backlog

- **证据**：`defaults.py:83-90` 和 `vr_teleop_policy.py:350-352` 按固定 16 Hz 调 alpha/每帧 step；`arm_loop.py:64-85,309-315` 解析 action time 却不拒绝 old action。
- **影响**：调度 jitter 改变真实目标速度与滤波带宽；SDK 阻塞后 FIFO 仍执行操作员已经反向之前的旧折返点。
- **修复**：从 observation anchor 定义 TTL；EMA 用 `alpha_eff=1-exp(-dt/tau)`；delta limiter 使用受限真实 dt；记录 queue age 和 inter-arrival p99。
- **回归**：注入 jitter、50–250 ms 阻塞和方向反转 backlog，过期目标不得发送。

### F-14 — jerk/provenance 未受控，replay 动力学不闭合

- **证据**：`arm_loop.py:31-33,197-206` 设置 maxacc 但不设置/记录 joint jerk；`replay_traj.py:120-140,229-237,703-704,1196-1253` 未完整恢复 speed/acc/jerk/loop rate。
- **影响**：不同控制箱配置和重启状态下的平滑性不可比较；`--speed` 改 cadence 但不形成明确的动力学实验语义。
- **修复**：明确 jerk 是 managed 还是 unmanaged；记录固件、serial、resolved speed/acc/jerk；区分路径复现和时间缩放模式；不自动持久化控制箱配置。
- **回归**：legacy fallback 必须显式；provenance 不同禁止标为可直接比较；setter 失败不得 ready。

### F-15 — tracking/replay 指标没有完成时齐

- **证据**：`arm_loop.py:484-519` 以最新 endpoint 减当前 feedback；`episode_quality.py:447-460` 同索引 action-state；`replay_traj.py:514-603` 虽估 lag，却未用 lag 重算 MAE/RMSE。
- **影响**：传输/规划 latency 被算成空间误差；快轨迹天然比慢轨迹“更不准”；无法区分 lag、稳态误差、overshoot 和模型偏差。
- **修复**：分开报告 generated→accepted latency、lag-compensated residual、低 qvel settled error、overshoot/settling/qvel ripple，并 mask stale/filled/rejected frame。
- **回归**：固定 3 帧 delay、零幅值误差时，补偿后 RMSE 应接近 0。

### F-16 — nominal URDF FK 不是绝对 TCP ground truth

- **证据**：`arm_loop.py:133-137`、`planning/kinematics.py:18-44`、policy planner 和 replay 均使用同一通用 `xarm7_xhand_collision.urdf`；没有 serial→kinematics suffix/calibration 加载路径。
- **影响**：内部 original/replay FK 可因共享 bias 看起来一致，但不能证明真实 TCP 绝对位置；camera、法兰、桌面和安装误差还会叠加。
- **修复**：获取 per-robot calibrated kinematics artifact；记录 robot serial、URDF/kinematics/TCP hash；外部测量分别评估 repeatability 和 absolute accuracy。
- **回归**：nominal/calibrated FK 差异、跨 model hash 拒绝直接比较、多姿态外部测量。

## 5. Policy 部署、进程管理与同步详细复核

### PD-01 — 缺少不可旁路的 ActionSafetyGate

- **状态**：P0 条件性风险。当前 canonical VR 路径有 delta、limit、workspace 和 collision 检查，但接口不强制未来策略复用。
- **证据**：`vr_teleop_policy.py:1275-1303,1353-1357` 内部做 safety；`shared_storage.py:175,362-386` 仍公开 raw queue/action；`arm_loop.py:381-405` 只做 finite/limit。
- **影响**：新 backend、keyboard、replay 或 plugin 可提交关节合法但路径碰撞、越 workspace 或 arm-hand 组合不安全的动作。
- **修复**：强制 `PolicyOutput→ActionAdapter→ActionSafetyGate→Scheduler→IPC`；backend 不获得 SharedStorage 写权限，所有 producer 迁移到同一发布边界。
- **回归**：NaN、shape、limit、collision、workspace、gate exception 都不得写 queue/ring；producer 清单测试禁止旁路。

### PD-02 — 默认 fork 不适合 GPU/线程模型运行时

- **证据**：`vr_teleop_hand_record.py:17-18,130-140` 使用默认 multiprocessing；Linux 离线结果为 `configured_start=None, effective_start=fork`。
- **影响**：父进程若初始化 CUDA/OpenMP/MKL/Torch 后 fork，子进程可能继承不可复用 context、锁和线程池，表现为偶发 hang。
- **修复**：创建任何 Queue/Value/Event 前统一选 `spawn` context；模型和 CUDA 只在 policy 子进程初始化；只传可重建配置/IPC 描述。
- **回归**：spawn 下完整离线启动；父进程预初始化 Torch CPU/CUDA 后子进程仍可确定启动/失败。

### PD-03 — daemon policy 阻断子进程能力

- **证据**：canonical 所有 worker 在 `vr_teleop_hand_record.py:131-138` 设置 `daemon=True`；daemon dummy 创建子进程稳定抛 `AssertionError`。
- **影响**：DataLoader `num_workers>0`、多进程预处理、模型 server 和部分异步组件不可用。
- **修复**：policy 非 daemon，由 main 显式监督和回收；初期要求 `num_workers=0`，需要强隔离时增加独立 inference process。
- **回归**：policy 子进程创建受控 child、正常关闭、child crash 和 parent shutdown。

### PD-04 — 缺少 model-level ready 和能力握手

- **证据**：`shared_storage.py:193-196` 只有设备 ready；main 在 `vr_teleop_hand_record.py:148-176` 随即 ARMED；policy 又硬编码等待四设备。
- **影响**：模型加载、compile、warmup 尚未完成时系统已 ARMED；可选传感器和 `--no-hand` 依赖错误；checkpoint/normalizer 错误过晚暴露。
- **修复**：新增 loaded/warmed/ready、manifest hash、required sensor mask；只有 hash、dummy inference、shape/finite、fresh sensors、SafetyGate/scheduler 均通过才 ready。
- **回归**：慢加载、错误 checkpoint、缺 optional/required sensor 和 warmup OOM。

### PD-05 — 慢推理会冻结实时协调、ESC 和 recorder

- **证据**：`vr_teleop_policy.py:568-1401` 单线程串行 heartbeat、keyboard、ring read、推理/IK/collision、IPC 和 recorder。
- **影响**：一次 C/CUDA hang 同时停止 heartbeat、freshness 检查、hold 发布、急停键处理和 16 Hz 记录时钟。
- **修复**：PolicyCoordinator 保持 deadline、IPC、安全、heartbeat 和 recorder；inference 通过单槽 mailbox 接受 immutable snapshot。无法取消的 native hang 必须隔离到进程。
- **回归**：inference 20/60/200 ms、永久 hang、OOM；coordinator 仍按时 heartbeat/hold/fault。

### PD-06 — policy 异常被误标正常退出

- **证据**：`vr_teleop_policy.py:1403-1408` 的 finally 无条件 `is_running=False`；`shared_storage.py:879-885` 先把它解释为 Q key normal exit。
- **影响**：OOM、tensor shape、CUDA 和预处理异常进入正常退出统计，自动化验收和事故追踪失真。
- **修复**：分离 main-owned shutdown、user quit、estop 和 sticky fault，并保存 owner/code/message；supervisor 按 fault 优先级判定。
- **回归**：正常 Q、Python exception、SIGKILL、heartbeat timeout 和 estop 的 session summary 必须互不混淆。

### PD-07 — 未确认 policy 退出即关闭共享 IPC

- **证据**：`shared_storage.py:516-533` join 5 秒→terminate→join 1 秒后不 recheck，直接 close/unlink。忽略 SIGTERM 的 dummy 在函数返回后仍 alive。
- **影响**：旧 policy 继续访问已 unlink IPC；父进程错误宣布停机；新 session 可能和旧模型实例并存。
- **修复**：cooperative→SIGTERM→recheck→SIGKILL→final join/exitcode，确认所有进程退出后才 close/unlink；失败时阻止复用 prefix。
- **回归**：忽略 SIGTERM、卡在 native call、正常慢退出和 writer flush。

### PD-08 — 缺少 causal ObservationSnapshot

- **证据**：`vr_teleop_policy.py:879-918` 顺序读取 arm→VR→camera→hand→tactile 的各自 latest；多个 helper 丢弃 seq/write time。各模态 freshness 规则互不一致。
- **影响**：视觉、arm、hand、tactile 可能来自不同 transition；同一 HDF5 行的 cross-modal skew 不可恢复，window 还可能使用 anchor 之后的 future sample。
- **修复**：以 host monotonic anchor 做 causal `timestamp<=anchor` 选帧；每模态保存 source/publish/seq/age/valid/generation，并限制 max skew。
- **回归**：受控多速率序列、missing/future/stale frame、clock reset 和 required sensor invalid。

### PD-09 — camera clock 域不可直接对齐

- **证据**：`sensor/realsense.py:481-523` 同时有 device timestamp 和 wall-clock host time；camera header `ring_buffer.py:559-574` 只保存 device timestamp，policy 又丢弃 ring monotonic write time。
- **影响**：camera timestamp 不能直接减 arm/hand monotonic time；设备重启、timestamp reset 和系统时钟调整破坏对齐。
- **修复**：同时发布 device capture、host receive monotonic、ring publish monotonic、frame number 和 camera generation；在线对齐只用 host monotonic。
- **回归**：模拟 device clock reset/wrap 和 host wall-clock jump，monotonic 对齐必须保持。

### PD-10 — action 缺少 observation provenance 和期限

- **证据**：`shared_storage.py:362-386` action 只有 qpos、seq、queue-created time 和 hold；`arm_loop.py:64-85` 不检查最大 age。60 秒前 action 离线仍被 parser 接受。
- **影响**：慢推理、queue backlog、旧 epoch 和 chunk 迟到目标仍合法；无法测 observation-to-action latency。
- **修复**：envelope 增加 epoch/action/observation ID、anchor/inference/target/valid-until time、representation/frame/units 和 paired arm/hand target；worker 拒绝 expired/old epoch/out-of-order。
- **回归**：过期、epoch mismatch、ID 倒退/重复和 frame/unit mismatch 全部 fail closed。

### PD-11 — arm/hand command 没有共同版本

- **证据**：arm 使用 FIFO queue，hand 使用 latest-wins ring，hand payload `shared_storage.py:142-146` 只有 12-D qpos。连续发布两组时 arm 仍取第 1 组、hand 已是第 2 组。
- **影响**：实际 arm-hand 组合不再是 policy 做过 collision check 的组合；无法测 apply skew或关联 ACK。
- **修复**：共同 `action_id/policy_epoch/target_ns`；两 worker 回显 received/accepted/applied seq；超 skew coordinated hold。
- **回归**：arm backlog、C24、hand 正常和一侧 SDK reject 时，另一侧不得被记作成功联合动作。

### PD-12 — action chunk 缺安全 scheduler

- **状态**：未来 chunk policy 的条件性 P1；不能通过扩大 arm queue 或在 arm worker 插值解决。
- **影响**：chunk 变成隐藏执行 backlog；新 observation 到来后旧 prefix 继续执行；arm-hand chunk index 错位；与 Mode 6 双重插值。
- **修复**：chunk 仅保留在 policy-side scheduler；丢弃 expired prefix、处理 overlap，每个控制 tick 依据最新 measured state 重新验证并发布一对 target。
- **回归**：chunk overlap、推理迟到、方向反转、缺一侧 target 和 expired whole chunk。

### PD-13 — policy hot restart/hot swap 无 generation 协议

- **证据**：ready event 不表示 generation；arm queue、hand ring、command seq 和 recurrent/chunk state 均没有 policy identity；当前 supervisor 只支持整体 fail-stop。
- **影响**：只重启 policy 时，新实例可继承旧动作、旧 hand target 和旧 observation。
- **修复**：修复前禁止 live hot restart；正式协议必须 stop-confirm→DISARMED→epoch++→drain→measured hold→reset state→fresh generations→warmup/ready→explicit ARMED。
- **回归**：restart 前 queue/ring 预填旧数据，新 epoch 不得消费；FAULT 不允许自动恢复运动。

### PD-14 — 没有 backend-neutral Policy/Observation/Action API

- **证据**：`policy/__init__.py` 只导出 VR `PolicyConfig/policy_loop`；`vr_teleop_policy.py:1679-1818` 的 state/action 主要服务 recorder，timestamp 也不是 observation anchor。
- **影响**：学习策略容易复制 VR loop、直接访问 SharedStorage 并再次旁路安全、同步和生命周期语义。
- **修复**：定义 `PolicyBackend.load/warmup/reset/infer/close`、`ObservationSpec/Snapshot`、`PolicyOutput/SafeJointAction/CommandAck`；backend 只接 immutable snapshot。
- **回归**：至少用 dummy、slow、invalid-output 和 stateful backend 验证统一接口。

### PD-15 — history ring 不支持常见 observation horizon

- **证据**：`shared_storage.py:45-50` 中 camera=5、arm/hand/VR=8；按 30 Hz 仅约 167/267 ms，camera 没有历史读取 API。
- **影响**：0.5–2 秒模型窗口无法构造；各流不能按共同 anchor causal 取样；tactile 又是稀疏 ring。
- **修复**：由 ObservationSpec 计算容量并 startup 校验；camera 增加 seqlock history；每个窗口位置保存 valid/age/seq，padding 策略显式。
- **回归**：不同 source rate、jitter、窗口不足、ring wrap 和 torn read。

### PD-16 — camera read 无条件复制所有大模态

- **证据**：`ring_buffer.py:416-519` 的 `CameraRingBuffer.read_latest()` 总是复制 RGB、depth 和固定 pointcloud，再做最终 seqlock recheck。
- **影响**：RGB-only policy 和 recording-off 仍承担大块复制；竞争失败前已经消耗内存带宽和分配成本。
- **修复**：ObservationSpec 驱动选择性复制，并在复制前后共享同一 seqlock 验证；不向模型暴露可被覆盖的 SHM view。
- **回归**：RGB-only 不复制 depth/PC；writer 竞争下仍拒绝 torn frame；测量 copy/preprocess/H2D p99。

### PD-17 — 无模型资源预算和实时准入

- **证据**：当前无统一 CPU affinity/thread 数、CUDA owner/显存、inference 分位数、deadline miss、expired action 和 OOM/warmup 指标。
- **影响**：模型平均推理时间看似合格仍可能因 p99、显存峰值或 camera pointcloud 竞争破坏 16 Hz 控制。
- **修复**：manifest 声明 device/dtype/memory/control rate/max inference/horizon/sensors/thread policy；录制开启和背景负载下用 p99/max 放行。
- **回归**：CPU/GPU、录制 on/off、RGB/RGB-D/PC、warm/cold 和受控负载矩阵。

### PD-18 — episode 缺少精确模型输入和动作阶段语义

- **证据**：schema v13 保存基础 camera/teleop metadata，但缺 policy/checkpoint/normalizer hash、preprocess spec、observation seq/age/skew、raw output、safe action 和 arm/hand ACK。
- **影响**：无法确定训练样本与部署模型实际看见的 tensor 是否相同，也无法区分 raw model output、SafetyGate 输出和硬件 accepted command。
- **修复**：additive schema 分层 `observation_* / policy_output_* / action_safe_* / command_* / arm_ack_* / hand_ack_*`，保存可复现 preprocess manifest；不改变旧字段含义。
- **回归**：snapshot→preprocess→record→read→preprocess hash 一致；old schema 可读；queue success+SDK reject 可区分。

### PD-19 — mutable singleton 配置不适合作为 policy plugin 边界

- **证据**：JSON 原地修改 module singleton、不重新跑 dataclass 校验、不支持 nested config，CLI 默认还可覆盖；离线可写入 `control_hz=-1`。
- **影响**：多次实例/测试互相污染；plugin 从隐式全局读取参数；resolved config 无唯一 hash。
- **修复**：新建不可变配置对象、完整 schema validation、固定优先级并记录 manifest/hash；plugin 只接显式 config。
- **回归**：同进程连续加载两个配置无污染；非法 nested/数值失败；序列化 round-trip 稳定。

### PD-20 — 自动化测试未覆盖部署故障模型

- **证据**：当前 tests 覆盖 IK/collision/homing/timestamp/recording 等，但没有 spawn 五进程、daemon child、model warmup/OOM/hang、SIGKILL、causal snapshot、TTL/epoch、paired action/chunk 等测试。
- **影响**：90 tests 全绿仍无法证明真实策略部署生命周期和故障降级安全。
- **修复**：建立无硬件 multiprocessing integration suite，加 fake workers/SDK/clock 和 deterministic fault injection。
- **回归**：本文第 10 节测试矩阵应成为策略放行门槛。

## 6. XHand SDK、触觉与 Retarget 详细复核

### XH-SDK-01 — open 成功后的异常路径没有统一清理

- **证据**：`robot/xhand.py:195-250` open 后无 finally 执行 verify/make/init；`225-236` ID mismatch 只 close；`hand_process.py:74-105` init exception 不 disconnect。mock 得到 `connected_flag=True, close_calls=0`；ID mismatch 无 INIT。
- **影响**：EtherCAT slave 可能未回 INIT/watchdog 状态，下次连接出现 SDO 错误并需要 24 V power-cycle。
- **修复**：只要 native open 成功，所有退出路径进入幂等 `_abort_open_device()`：best-effort stop/INIT、close、watchdog wait、清 handle/flag；worker 再做第二层 disconnect。
- **回归**：在 list ID、make command、initial read、tactile reset/bias 各点注入异常，cleanup 至多一次且原始错误不被覆盖。

### XH-SDK-02 — send fault 被健康 read 覆盖且不升级 supervisor

- **证据**：`xhand.py:824-872` send failure 设置 error；`807-821` 下一次 board read 用寄存器状态重写 error；`hand_process.py:220-248,334-355` send threshold 只调用本地 clear。mock 显示 send 后 true、健康 read 后 false。
- **影响**：读通写断时手不执行，但 connected/RUNNING 可继续，arm 仍运动且抓物可能失稳。
- **修复**：拆分 read transport、send transport、board error；send threshold 直接 sticky `shared.error_state`；健康 read 不得清 send fault，记录 last success/seq/code。
- **回归**：持续 send fail+read success 必须阈值 FAULT；单次恢复可清 counter；不同错误域互不覆盖。

### XH-SDK-03 — 静止反馈被误判 stale 并形成 hold 自锁

- **证据**：`defaults.py:74-80` 使用 15 帧和 `1e-4 rad`；`hand_process.py:313-327` 仅看 qpos 是否变化；`vr_teleop_policy.py:1218-1224` stale 后 hold。
- **影响**：正常保持约 0.5 秒即 stale；hold 使 qpos 更不变化，状态只能靠编码器抖动或外力恢复。
- **修复**：分开 source freshness 和 tracking stall；成功 force-read/seq/timestamp 决定 fresh；只有“命令要求运动且误差长期不收敛”才 stalled，已收敛命令清 counter。
- **回归**：100 帧静止且命令已收敛不 stale；变化命令+固定反馈触发 stall；恢复帧可解除。

### XH-RT-01 — finite 但退化的 VR landmarks 未 fail closed

- **证据**：`vr_receiver_process.py:107-135` 只做类型/reshape；`hand_retarget.py:53-109` 退化 palm frame 返回 identity；DexPilot/TAG 只查 shape+finite。全零输入仍得到 TAG 最大约 81.6°、DexPilot 最大约 110° 的有限目标。
- **影响**：新鲜坏帧会逐步闭手；0.20 rad/frame、EMA 和 limits 只减速，不验证意图；hand-hand collision 本来就关闭。
- **修复**：共享 `validate_hand_landmarks()` 检查掌宽、骨长、三角面积/条件数、chirality、尺度/速度；palm frame 退化返回失败，保持上一有效命令且不更新 temporal state。
- **回归**：全零、共点、共线、零骨长、突跳拒绝；刚体变换和合理尺度允许。

### XH-TACT-01 — raw tactile release 后无限 forward-fill

- **证据**：`hand_process.py:372-379` 仅 contact 时写 sparse raw；policy `908-918` 缓存上次数据；recorder 将当前 contact 和旧 raw 写同一行。现有 427 帧 episode 中 313 个 no-contact 帧有 71 帧 nonzero raw，release 后相同 raw 最长持续 132/59 个 policy sample。
- **影响**：训练和质量分析把旧接触纹理标到 release 段；未来 tactile pinch gate 会被 stale contact 误导。
- **修复**：falling edge 写 release frame；dtype 增加 timestamp/seq/valid/contact mask；policy/recorder 保存 age/freshness，旧 schema 缺字段时为 unknown。
- **回归**：contact→release→contact、多指分别 release、read error 与 release 区分、schema round-trip。

### XH-TACT-02 — tactile zeroing 可把真实载荷吸收到 bias

- **证据**：`xhand.py:397-403,440-506` connect 时自动 reset 并 software bias；`515-550` 只在 docstring 要求无接触，代码不验证。
- **影响**：启动时稳定接触 `F` 会进入 bias，持续接触随后约为 0，contact threshold 和 HDF5 全部低估。
- **修复**：reset 前检查载荷；校零变为显式可确认阶段，发布 calibrated flag/bias/time；无法确认无接触时拒绝绝对接触判定。
- **回归**：offset-only 可归零、稳定 5 N 不得接受、read failure 不得产生 NaN bias。

### XH-RT-02 — pinky progressive scaling 破坏骨段方向

- **证据**：`hand_retarget.py:112-157` 覆盖 PIP 后再用 `old DIP-new PIP` 计算下一段。共线 `[0,.03,.06,.10]`、scale=2.2 产生 `[0,.066,.0528,.15664]`，PIP→DIP 反向；真实 episode 297/427 帧反转。
- **影响**：DexPilot 差向量和 TAG pinky tip target 被非单调几何污染，导致 pinky 触达误差、过屈或抖动。
- **修复**：写入前保存所有原始骨段向量，再逐段乘 scale；或统一从 MCP 对原坐标缩放，不能混用新旧父节点。
- **回归**：scale=1 identity；新旧对应骨段点积非负；长度按预期缩放；episode 反转数归零。

### XH-RT-03 — DexPilot reset 对排列使用两次正向 mapping

- **证据**：`hand_retarget.py:213-216,294-295` 的 mapping 正确用于 internal→SDK；`333-340` reset 从 SDK→internal 时再次使用同 mapping。home warm-start 最大误差 75.66°。
- **影响**：常规输出顺序正确，但 B-press/resume/audio gate 后首帧优化从错误姿态开始，增加迭代、时延和 temporal bias。
- **修复**：使用 inverse `np.argsort(mapping)` 并调用 `SeqRetargeting.set_qpos(full_internal)`；同时 reset LPFilter/EMA/projected state。
- **回归**：唯一值、home、midpoint、随机 bounds 的 SDK↔internal round-trip 和首帧状态清理。

### XH-ARCH-01 — policy 为限位导入 native XHand SDK

- **证据**：`vr_teleop_policy.py:424-441` 初始化 TAG；`hand_retarget.py:443-458` 导入 `robot.xhand.XHandConfig`；`xhand.py:13-19` 顶层加载 vendor native module。
- **影响**：违反“仅 hand worker 导入厂商 SDK”；policy 启动被 `.so` ABI、依赖和副作用耦合。
- **修复**：TAG 直接从 `defaults.hand.qpos_min_rad/max_rad` 读取 source of truth，不导入 driver module。
- **回归**：明确阻止 `xhand_controller` 导入时 TAG 仍成功，policy `sys.modules` 中不出现 vendor SDK。

### XH-SDK-04 — SDK missing stub 不是一致的仿真设备

- **证据**：`xhand.py:13-19,195-212` ImportError 后 connect=True；`725-741` state 固定零且 control=None 令 `is_connected=False`；非零 home 最大误差 1.4078 rad，worker 必然 homing timeout。
- **影响**：诊断先称连接成功，随后又失败；绕过 homing 的调用者得到 command 与 feedback 不一致的假数据。
- **修复**：hardware worker 默认 `allow_stub=False`；显式 stub 时反馈跟随 joint-limited command，连接语义一致并在 metadata 标明 stub。
- **回归**：默认缺 SDK fail closed；显式 stub send→read round-trip、home 收敛和 episode 标志。

## 7. 录制与 camera 数据链补充问题

### DATA-01 — 下一条样本向过去 grid slot 做 future-fill

- **证据**：`recording/timestamp_buffer.py:28-58,128-204` 对延迟到来的 timestamp，把同一个 source sample 重复写到从 `next_global_idx` 到其自身 slot 的所有位置；例如 t=10.0 值 A、t=10.2 值 B、dt=0.1，输出值为 `[A,B,B]`，`flag_sample_valid=[True,False,True]`。
- **调用链**：policy 某 tick/recording overrun→下一次 `EpisodeRecorder.add_frame()`→`TimestampAlignedBuffer.add()`→B 写入本应属于 10.1 的过去位置→synthetic timestamp 仍写 10.1。
- **影响**：无效槽显式携带 future observation、action、VR 和 health 状态。虽然 v13 有 validity flag，reader、replay 和多数直接 HDF5 消费者没有强制 mask；训练若忽略 flag 会发生未来信息泄漏并伪造控制时序。
- **与其他 finding 的关系**：这是 PD-08 causal snapshot 和 PD-15 padding 语义在 recorder 内的具体实现缺陷；保留独立编号是因为它已经写入持久化数据。
- **修复**：缺槽应使用 past-only hold-last 或显式 NaN/invalid，不允许未来样本回写过去；所有默认 reader/training/replay API 强制暴露/应用 validity mask，raw HDF5 访问需在文档中标为低级接口。
- **回归**：延迟一格/多格、episode 起点、capacity 截断和连续 stall；任意 slot 的 source timestamp 必须 `<= grid timestamp`。

### DATA-02 — camera freshness 检查只有注释，没有执行

- **证据**：camera worker `sensor/camera_process.py:132-221` read 失败时只继续 heartbeat，不更新 ring；policy `_read_camera_frame()` 在 `vr_teleop_policy.py:1472-1480` 丢弃 ring sequence/write time并重复返回 latest；`episode_recorder.py:389-397` 写有 freshness 注释，却没有比较 frame number，`camera_health` 还从不存在的 top-level key读取而默认 0。
- **触发**：RealSense 停流、USB read 连续失败或 camera writer停止，但进程仍 alive；camera ring 保留最后一帧。
- **影响**：policy 每 tick 把同一旧 RGB/depth/PC 当作当前 observation；recorder继续写健康状态 0，heartbeat和 HDF5 都不能区分“重复采样”与“相机停流”。
- **与其他 finding 的关系**：PD-08 描述跨模态 snapshot，PD-09 描述时钟域；本项确认了现有 recorder 的 frame-number freshness 逻辑实际缺失。
- **修复**：policy 保留 ring seq、slot publish monotonic 和 header frame number；按 source age/generation 判断 valid；camera heartbeat 与 successful-frame progress 分离；recorder保存 camera seq/age/fresh/health。
- **回归**：ring frame number 固定而 heartbeat前进时必须 stale；新 frame恢复后 fresh；camera process crash/read failure/低帧率语义分开。

### DATA-03 — RGB MP4 对内部缺帧的补齐位置错误

- **证据**：`episode_recorder.py:490-504` 只有 `camera_frame is not None and k>0` 才向 MP4 写一帧；camera 缺失时只延长 `_cam_grid_end[-1]`，这一 span 仅用于 depth/pointcloud 的 `_forward_fill_cameras()`。`episode_reader.py:237-253` 只在 MP4 总帧数短于 grid 时在**尾部**重复最后帧。
- **确定性时序**：grid 0 有 A、grid 1 缺失、grid 2 有 B；正确 RGB 应为 `[A,A,B]`，MP4 实际为 `[A,B]`，reader 尾填后得到 `[A,B,B]`。B 被提前一格，与 grid 1 的 arm/action 对齐。
- **影响**：任一内部 camera gap 会使其后的 RGB 相对 state/action 整体左移，直到尾部补齐；depth/pointcloud 使用 span fill，三种 camera modality 自身也互相错位。
- **与其他 finding 的关系**：属于 PD-08/PD-18 的持久化表现，但不是单纯 timestamp metadata 缺失，而是可确定的索引错误。
- **修复**：录制时按每个 grid slot 实际写 RGB（重复上一 causal frame并标 fresh=false），或保存 `rgb_frame_index/grid_to_rgb_index` side table，由 reader按索引展开；禁止只做 tail padding。
- **回归**：begin/middle/end 缺帧、连续缺帧和第一帧缺失；RGB/depth/PC 使用同一 grid mapping，并保存 freshness。

### DATA-04 — filter 后 data.h5 与 depth/RGB sidecar 不同时间轴

- **证据**：`tools/episode_quality.py:826-960` 遍历 `EpisodeReader.h5f` 的 merged view，对 shape[0]==T 的 dataset应用 mask并写入新的 `data.h5`，其中也包括从旧 `depth.h5` 路由出来的 depth；随后又对目录格式直接 `shutil.copy2(depth.h5, rgb.mp4)`。固定 HEAD 的 `episode_reader.py:54-93` 会优先从 sidecar 读取 depth，因此未过滤的复制版 `depth.h5` 反而遮蔽了写入新 `data.h5` 的已过滤 depth；RGB则从未应用mask。
- **触发**：调用 `EpisodeQuality.filter(output_dir=...)` 且 mask 删除任意非尾部帧。
- **影响**：输出 data.h5 的 robot/action 是 kept timeline，reader实际看到的 depth.h5/rgb.mp4 仍是 original timeline；同索引 camera 对应错误 robot/action，长度也可与 metadata 不一致。新 data.h5 内还可能残留一个不会被标准reader使用的 filtered depth副本。该产物不应进入训练或 replay。
- **与其他 finding 的关系**：PD-18 要求训练—部署可复现，本项是现有质量工具主动生成语义损坏 episode 的具体路径。
- **修复**：以同一 mask 重写 depth 和 RGB；保留 original index/source timestamp/freshness；临时目录原子完成所有 sidecar 后再发布；校验所有 T 和 metadata一致。
- **回归**：mask 删除首/中/尾、多段；读取输出后逐模态 marker必须对应 original kept index；encoder failure不得留下看似完整目录。

## 8. 关联、去重与根因

### 8.1 主要关联簇

| 根因簇 | 相关 finding | 统一修复边界 |
|---|---|---|
| 动作安全旁路 | F-01、PD-01 | 所有 producer 必经 SafetyGate 和完整 path/transition validation |
| 生命周期与停止 | F-02、F-03、F-05、PD-04、PD-06、PD-07、PD-13、XH-SDK-01 | generation-aware lifecycle、stop ACK、退出原因和幂等 cleanup |
| feedback/freshness | F-04、PD-08、PD-09、XH-SDK-03、XH-TACT-01、DATA-01/02/03 | source timestamp/seq/age/valid，严格 causal 对齐 |
| 动作时间与联合版本 | F-08、F-13、PD-10、PD-11、PD-12 | JointActionEnvelope、TTL、epoch、paired ACK、policy scheduler |
| 数据语义 | F-08、F-15、PD-18、XH-TACT-01、DATA-01/03/04 | 分层记录 raw/safe/queued/accepted/feedback 和 modality index |
| 配置与复现 | F-10、F-14、F-16、PD-17、PD-19 | immutable resolved manifest、设备/model/provenance hash |
| retarget 输入/几何 | F-11、XH-RT-01/02/03 | manifold-aware rotation、landmark validity、几何与 mapping invariant |

### 8.2 五个深层根因

1. **安全检查属于具体调用者，而不是公共动作边界。** VR policy 和 safe homing 有完整检查，但 replay、新 policy 和 raw queue 可以旁路。
2. **process health、device health、model health 与 data freshness 混用。** heartbeat、connected、timestamp 和 error bool 无法表达独立故障域。
3. **“同一 loop/grid 行”被误当作“同一物理时刻”。** latest reads、future-fill、RGB tail fill 和 arm-hand 独立 IPC 都破坏因果关系。
4. **动作只有目标数组，没有生命周期。** 缺 observation ID、epoch、TTL、共同 action ID 和 ACK，无法安全处理 backlog、chunk、recovery 和重启。
5. **持久化格式保存了数值，却没有保存数值的来源阶段和有效性。** raw model output、safe command、queued/accepted、feedback、camera mapping 和 calibration provenance 需要分层。

## 9. 已反向检查、未判定为问题的部分

- xArm C22/C31 当前按 immediate sticky fault 处理，没有继续自动 clean/retry；
- canonical VR IK 的 workspace/collision gate、home path 的密集碰撞检查和 HomeRequest/HomeResult 关联基本正确；
- seqlock 基本实现、arm queue `maxsize=2` 和 hand latest-wins 的单流语义合理；问题是缺共同 action identity，不应通过扩大 queue 修复；
- 16 Hz grid-aligned recording 的目标正确；问题是 missing-slot 和 camera mapping 的实现语义；
- 不在 arm worker 加插值、由 Mode 6 固件平滑的总体选择正确；
- XHand state 按 finger item ID 落入 SDK slot、force-update read、position rad 单位和双层 finite/limit/step guard 基本正确；
- TAG model↔SDK mapping round-trip 精确，TAG reset mapping在 bounds内正确；
- Pinocchio Jacobian 中心差分最大绝对误差约 `4.37e-14`、relative L2约 `1.13e-10`；
- 当前 Quest→MANO→URDF identity target rotation 在已有 episode 上明显优于直接移植 TAG glove root rotation；不判定为坐标系 bug；
- TAG Stage 1失败 hold、Stage 2失败回退 Stage 1，policy 对 non-finite/bounds和 arm-hand transition有兜底。

## 10. 推荐修复顺序

### Batch 0 — 立即限制运行路径

1. 修 F-01：修复前禁止 live replay；
2. 修 PD-01：新增 policy 不得直接写 raw queue/ring；
3. 对所有由 quality filter 生成的目录执行 DATA-04 一致性检查，现有不一致产物不得训练；
4. 在 DATA-02/03 修复前，camera stale/gap episode不得默认进入训练。

### Batch 1 — 控制器和进程 fail-closed

1. F-02/F-03/F-05/F-06：统一 xArm lifecycle、state 4、C24 measured hold和 stop ACK；
2. XH-SDK-01/02/03：XHand cleanup、send sticky fault和 freshness/stall 分离；
3. PD-02/03/04/05/06/07：spawn、非 daemon policy、model ready、coordinator隔离、结构化退出和 SIGKILL确认；
4. F-07：enabled dependency graph。

### Batch 2 — 因果 observation 与数据新鲜度

1. F-04、PD-08/09、DATA-02：统一 source/publish/receive timestamp、seq、age、generation和valid；
2. XH-TACT-01/02：触觉 release/freshness和显式 calibration；
3. DATA-01：禁止 future-fill，缺槽使用 causal hold-last或invalid；
4. DATA-03：统一 grid→camera modality mapping。

### Batch 3 — 动作协议和联合调度

1. PD-10/11/12/13：ActionEnvelope、TTL、epoch、paired arm-hand ID、ACK和chunk scheduler；
2. F-08/F-13：命令阶段语义、真实 dt 和 stale action拒绝；
3. F-12：new-action-only SDK send；
4. PD-14：backend-neutral API。

### Batch 4 — Retarget、几何和模型质量

1. XH-RT-01：退化 landmarks fail closed；
2. XH-RT-02/03：pinky 原始骨段向量和 DexPilot inverse mapping；
3. F-11：SO(3) shortest-arc EMA；
4. XH-ARCH-01/SDK-04：恢复 SDK进程边界和显式 stub。

### Batch 5 — 复现、指标和工程准入

1. F-09/10/14/15/16：homing稳定性、resolved dynamics、时齐指标和 calibrated model provenance；
2. PD-15/16/17/18/19：history、selective camera copy、资源准入、schema和immutable config；
3. DATA-04：全模态 filter事务；
4. PD-20：把全部故障模型加入 CI。

## 11. 自动化验证矩阵

### 11.1 生命周期和进程

| 场景 | 必须满足 |
|---|---|
| xArm setter 返回非零或 live postcondition失败 | 不 ready，stop-confirm 后 disconnect |
| DISARMED等待 120秒 | live controller保持 state 4 |
| XHand open 后任意 init异常 | INIT/close幂等执行，handle清空 |
| policy正常Q、异常、OOM、SIGKILL | exit reason互不混淆 |
| worker忽略SIGTERM | SIGKILL后确认exitcode才unlink IPC |
| spawn + model warmup | parent不初始化CUDA，policy ready前保持DISARMED |

### 11.2 时间、快照和动作

| 场景 | 必须满足 |
|---|---|
| arm/hand/camera/VR不同速率 | snapshot只选 `source<=anchor` 且报告skew |
| camera heartbeat前进但frame number固定 | camera invalid/stale |
| action年龄超过TTL或epoch旧 | arm/hand都拒绝 |
| arm backlog、hand latest更新 | action ID不匹配时coordinated hold |
| chunk迟到或重叠 | expired prefix丢弃，每tick仅发布一对target |
| policy jitter | EMA时间常数和速度界按真实dt稳定 |

### 11.3 数据与录制

| 场景 | 必须满足 |
|---|---|
| timestamp grid缺槽 | 不使用future sample；invalid/source index可恢复 |
| camera内部缺一帧 | RGB/depth/PC都映射到正确grid |
| contact→release→contact | raw、contact、timestamp、fresh一致 |
| filter任意非连续mask | data/depth/RGB按同一original index重写 |
| queue success、SDK reject | queued与accepted不同 |
| old schema episode | 新reader兼容读取，缺字段为unknown而非伪造fresh |

### 11.4 数学和 Retarget

| 场景 | 必须满足 |
|---|---|
| quaternion跨 ±π | 沿2°短弧，不出现89.5°跳变 |
| landmarks全零/共点/共线 | TAG/DexPilot返回hold且不污染temporal state |
| pinky scale 1/随机scale | identity/骨段方向与长度不变量 |
| SDK/internal唯一值 | 双向round-trip精确 |
| homing qpos在容差但qvel高 | 不ACK |

## 12. 本次验证记录

### 12.1 已运行

- 固定 HEAD：`167f15a5f76b798ea5a90e44fe3e478eecc266d2`；
- 早期全仓离线基线：compileall通过，mypy所审23个source files通过；
- 本轮重新运行：

```text
conda run -n real_robot pytest -q tests
90 passed in 1.63s
```

- xArm mock/数学：Mode/state调用链、旧反馈timestamp、queue/退出分析、`+179°→-179°` 输出89.5°；
- policy进程模型：effective start=`fork`、daemon child assertion、60秒旧action仍接受、arm/hand版本分离、忽略SIGTERM进程仍alive；
- XHand mock：post-open异常、ID mismatch、send失败+健康read、implicit stub；
- retarget数值：退化输入、pinky反转、DexPilot warm-start、TAG mapping、Pinocchio有限差分；
- HDF5只读：`episodes/episode_20260808_230538/data.h5`，427帧schema v12，用于触觉、pinky、TAG FK和坐标候选复核；
- DATA-01/02/03/04：重新检查 timestamp buffer、camera read/recorder、MP4 reader和filter sidecar可达调用链。

文档汇总结束前发现工作区的 camera/recording 源码出现了与本报告无关的并行未提交修改。为避免把移动中的工作树当成审查证据，四个 `DATA-*` finding 已再次使用 `git show HEAD:<path>` 对固定 `167f15a` 内容核对；这些并行源码修改既未由本报告产生，也未计入“已修复”。

### 12.2 未运行

- 任何 `examples/real/` 遥操作、replay、homing、calibration或hardware diagnostic；
- xArm `192.168.1.111` 网络探测、Mode/state setter、C24/C22/C31制造；
- XHand EtherCAT/RS485 enumerate/read/send/reset_sensor；
- RealSense/Quest实时采集；
- 实物抓取、pinch、触觉标定、温升、制动距离和碰撞测试；
- GPU模型、真实checkpoint、CUDA OOM或长时间性能压力测试。

现有 90 个测试通过与本文 finding 不矛盾：测试尚未覆盖完整 controller lifecycle、process kill escalation、causal multimodal snapshot、TTL/epoch、camera内部缺帧、sidecar filter、XHand send-only fault和退化landmarks。

## 13. 仍需厂商资料或真机裁决的候选风险

以下不计入 50 条 confirmed/conditional finding：

1. XHand tactile `raw_force/calc_force * 0.1` 是否确为 N；需要厂商规格或砝码标定；
2. EtherCAT `slave_position=1` 在多slave拓扑中的正确性；
3. TAG Stage 2零距离pinch、权重2000且无tactile gate是否造成过载；
4. TAG绝对fingertip error、pinky bounds饱和是否需要多用户scale/base offset标定；
5. XHand temperature是否需要 `&0xFF`；当前未发布、不参与控制；
6. xArm对完全相同Mode 6 endpoint是否在固件内去重；
7. joint jerk对当前固件Mode 6的具体作用；
8. nominal与per-robot calibrated kinematics的真实TCP差异；
9. 真实SIGKILL、网络断连和SDK native hang下能否在要求时间内确认stop；
10. camera/pointcloud与真实GPU policy共同运行时的p99 deadline和显存竞争。

## 14. 最小真机验收清单

真机阶段必须单独授权、低速空载、清空工作区并有人守物理急停：

1. 连续20次xArm/XHand connect→DISARMED→ARMED→RUNNING→stop→disconnect，无需power-cycle；
2. 各初始化阶段注入失败，控制器/hand都回到可确认安全状态；
3. read-only与write-only链路分别中断，独立health domain在阈值内FAULT；
4. arm/hand静止10秒不误stale，变化命令+固定反馈必须stall；
5. 遮挡Quest、camera停流和frame freeze，系统hold且episode正确标invalid；
6. arm/hand action ID、SDK ACK和feedback seq可计算apply skew；
7. 低速短弧跨quaternion ±π，无非意图腕部回摆；
8. pinky全行程骨段方向正常，DexPilot/TAG reset首帧无joint-order跳变；
9. tactile无接触显式校零，带载启动拒绝；release时raw/contact/fresh一致；
10. replay只接受preflight hash匹配的安全轨迹，故意构造中点碰撞时零命令发送；
11. camera中间丢帧、episode filter后，RGB/depth/PC与state/action marker逐帧一致；
12. 记录开启、RGB-D/PC和目标模型共同运行10–30分钟，控制/推理/action age报告p50/p95/p99/max和deadline miss。

## 15. 最终判定

当前系统已经具备五进程隔离、seqlock、bounded arm queue、latest-wins hand ring、VR canonical safety gate、home path planning、双层XHand限位和TAG优化器等良好基础，但这些正确局部尚未形成统一的端到端安全契约。

在 P0 的 replay/SafetyGate 边界、P1 的 controller/worker生命周期、feedback/camera freshness、action TTL/epoch/联合版本、退化retarget输入以及四个持久化数据语义问题完成修复前：

- 不应开放未经逐轨迹preflight的live replay；
- 不应接入可直接写SharedStorage的新学习策略；
- 不应把camera gap、future-fill或filter sidecar未对齐的episode直接用于训练；
- 不应宣称现有HDF5能够证明SDK accepted/executed动作、绝对tracking精度或跨模态因果一致性；
- 系统只适合有人值守、低速、受控的实验室遥操作与诊断。

本报告只新增文档，没有修改运行源码、公共API、SharedStorage dtype、HDF5 schema、配置或校准数据。
