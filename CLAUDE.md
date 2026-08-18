# DexMani Real：实现导航

`AGENTS.md` 是本仓库的绑定工作合同；本文件只提供快速定位和修改闭环。README 负责用户入口，`docs/` 负责较稳定的领域/数据合同。遇到冲突时以代码、schema 和 `AGENTS.md` 为准。

## 60 秒定位

| 任务 | 起点 | 必须继续检查 |
|---|---|---|
| 默认值/运行时覆盖 | `config/defaults.py` | `config/runtime.py`、派生容量/超时/metadata |
| 跨进程字段 | `utils/schema.py` | `shm/shared_storage.py`、所有 producer/consumer、recording |
| ring/queue/flag/event | `shm/shared_storage.py` | 分配、读写、ready/heartbeat、close/unlink、故障路径 |
| VR 遥操作 | `teleop/loop.py` | snapshot → mapper/retarget → planning → safety → samples |
| 手部 retarget | `teleop/hand_control.py` | `hand_retarget.py`、`docs/hand_retargeting.md`、worker |
| arm/hand action 或安全 | `policy/safety.py` | `send_command`、arm/hand worker、supervisor、home/e-stop |
| arm/hand 伺服环 | `robot/arm_loop.py` / `robot/hand_process.py` | `arm_sdk.py`（SDK 面 + 启动轨道）、`homing.py`、`_LoopState`+块函数、safety gate、HOME/e-stop |
| FK/IK/碰撞/路径 | `planning/` | teleop hold/fallback、delta clamp、replay dense preflight |
| episode schema/质量 | `recording/episode_schema.py` | writer → reader → processing/visualization/replay |
| processed HDF5/Policy Zarr | `data_processing/pipeline.py`、`data_processing/zarr_export.py` | 单 episode 压紧、provenance、Policy Dataset 合同 |
| 回放 | `examples/replay_episode.py` | provenance、dense preflight、live safety path |
| learned policy | `deployment/coordinator.py` | worker、config、contracts、lifecycle、integration adapter |
| CLI/生命周期 | 对应 `examples/*.py` | domain lifecycle、readiness、supervision、shutdown |

## 所有权

```text
main/lifecycle: config → SharedStorage → spawn → readiness → supervise → shutdown
sensor workers: device input → shared state
teleop: observation selection → mapping/IK/retarget → action → fixed-grid sample
deployment worker: observation → model → policy_plan_ring
deployment coordinator: plan → shared SafetyGate → robot action
robot workers: shared command → SDK → measured feedback
recorder: received sample → validate/write/publish
```

- main 不映射 VR、不生产 action、不选择 recording sample。
- hardware SDK 不跨进程传递；拥有 SDK 的 worker 不做策略决策。
- `deployment/` 不依赖 `integrations/`；适配器只能反向依赖 deployment contracts。
- `recording/` 不改变采样时机和业务语义；RecorderIO 只序列化、校验并事务式发布固定网格样本。

## 不可破坏的协议

### 数据与时钟

- `utils/schema.py` 是共享 NumPy payload 的单一来源；边界检查 shape、dtype、finite 和单位。
- ring 使用 seqlock；`get_last_k(k)` 返回已验证、oldest-first 的结果，可能少于 `k`，且拒绝 `k > maxlen`。
- arm queue 是有序 `maxsize=2`；hand command ring 是 latest-wins，`policy_plan_ring` 也是 latest-wins 且 `maxlen=3`。
- ARM_STATE 的 `mode` 是缓存 report 属性（必要非充分）；`accepts_motion_commands` 是 arm worker 写入的「当前接受伺服命令」单一事实（DISARMED/homing/非阻塞 Mode-6 转换窗口内为 0，仅确认 Mode-6 movable 的那帧置 1）；producer 就绪门是 `mode==6 AND accepts==1`，不能只看 `mode`（re-arm 时缓存 mode 仍是 6）。
- `is_running`、`is_recording`、`error_state`、`estop_request` 是简单 flag；只有 `safety_state` 存 `SafetyState`。
- heartbeat 使用 `time.monotonic()`；`error_state` sticky。
- recording 以 `1 / control_hz` 固定网格采样，不按传感器到达时间采样。

### 生命周期与命令

- main 管理 `DISARMED ↔ ARMED`、`→ FAULT` 和 shutdown；policy 管理 `ARMED ↔ RUNNING`。
- `run_generation` 标记控制时代；BEGIN、pause、home、feedback fault 使旧命令失效。camera stall/写盘错误丢弃 episode，但等待下一次显式 BEGIN 才推进 generation。
- worker 在共享命令跨越 SDK 边界前再次检查 generation、有效期、状态和 payload。
- 普通 pause 是 command quiescence：不发送 measured hold，不调用 State 6；让固件完成已接受的 endpoint。
- `KeyboardHandler` 控制键按下—释放边沿触发；长按/自动连发不得重复触发 BEGIN、pause、HOME 等状态变更。HOME 是一次性阻塞操作，返回后必须丢弃积压的 HOME 事件；ESC 仍独立即时锁存。
- `AudioFeedback` 只有一个串行播放 worker：`play()` 抢占当前提示并清空队列，`queue()` 必须在当前提示完成后才播放，提示音不得重叠。
- `AudioFeedback` 必须记录播放器启动、返回码和截断 stderr；退出提示使用有界 idle wait，不能依靠固定 sleep 猜测播放完成。
- XHand 状态读取始终发起实时 SDK 事务，不提供缓存读取开关。单帧读取失败保留真实连接状态，并发布 `state_valid=0`、`read_healthy=0`、`qpos_stale=1`。固件过流**警告**（`read_state` 错误码 `1501035`）是可恢复信号而非读失败：保留 last-known 电流、不置 `state_valid`/`read_healthy` 失效，仅累计 `overcurrent_error_count` 作观测（episode 质量统计），不暂停、不锁存 fault、无需手动恢复；电流兜底由固件 `tor_max` 负责。板级 `jointboard_err` 的电流保护（`ERROR_CURRENT_PROCTED`）是更严重的硬保护，仍走 `error_state`→FAULT 路径，不做降级。
- 手部反馈单帧 unhealthy/不可用（`HAND_FEEDBACK_UNHEALTHY`/`HAND_FEEDBACK_UNAVAILABLE`）在发布路径是 **hold 而非 fault**：原地保持 + 继续，唯一 fault 路径是 loop 的 `hand_disconnect_timeout_s` 去抖；不要把手部反馈瞬态异常当成立即 FAULT。
- `RateManager` 的运行实例带 loop label；共享环历史读取先复制最易被覆盖的最旧槽。
- xArm Mode 6 已负责 arm trajectory smoothing；不要加入应用侧插值。
- ARMED/RUNNING 的 Mode-6 进入是**非阻塞 issue+confirm**（`issue_mode_enter` 发 `set_mode(6)+set_state(0)`，之后每 tick 用 `mode_enter_ready` 非阻塞确认），loop 不阻塞、每 tick 发布 `accepts_motion_commands=0` 直到确认；超 1.0s 走 `error_state`+`stop_controller`+EXIT，转换窗口内瞬时 controller error 当作「未就绪」而非 fault。home-restore 的 `enter_mode6` 仍同步。
- SafetyGate 是 action publish 的统一边界；固件仍是最后 backstop。

### Episode 与策略

- runtime episode 只接受 schema v17；目录包含 `data.h5`、`depth.h5`、`rgb.mp4`，失败不原子发布。
- 采集按 `episodes/<task_name>/episode_*` 发布，目录 task 与 `/meta.task_label` 同源；任务失败由操作者在清洗前删除完整 episode。一个 raw 只产生一个压紧后的 processed HDF5；禁止 `__segNNN`。
- 正式数据层级均位于仓库根目录：raw 为 `episodes/<task_name>/`，清洗输出为 `episodes_processed/<task_name>/`，训练容器为 `dataset/<task_name>.zarr`；不增加 `inputs` staging 层。
- 所有 processed 模态共用一个 keep mask；默认 quality policy 只审计时序异常，`strict` 才删除高置信度 impulse/stall。删行压紧形成不连续或超阈值动作 bridge 时默认拒绝整条 raw，不允许通过切段规避。
- processed HDF5 v3 的 `provenance/` 保存 source row/time/drop reason；Real-native core 为 `joint_state/action/action_ee/contact_force/fingertip_points`，RGB-D profile 另含 `rgb/depth/camera_intrinsic/camera_extrinsic`，点云 profile 另含 `point_cloud`。
- Zarr 只导出 `data/*` 和 `meta/episode_ends`，不导出 provenance、manifest 或 `task_success`；批次中任一未显式排除的 raw 被拒绝时，不发布部分数据集。
- `recording/episode_schema.py` 同时约束 writer、reader 和 finalizer；不要维护第二份 dataset 列表。
- point cloud 不作为独立 sidecar 保存，由 depth 和 metadata 在消费边界确定性派生。
- RecorderIO 是唯一 recorder backend；START/STOP、status 与 sample 均走固定共享 dtype 控制路径。
- learned-policy inference worker 只写 `policy_plan_ring`；coordinator 是唯一 robot-action producer。
- 模型输出是 proposal，不是 command；必须经过共享 SafetyGate/发送边界。

## 修改配方

### 新增共享字段或 ring

1. 在 `utils/schema.py` 定义 shape/dtype、单位和 invalid value。
2. 更新 `SharedStorage` 的创建、清理和 readiness。
3. 更新唯一 producer、所有 consumer 和 failure/shutdown 路径。
4. 若要持久化，同步 `episode_schema`、writer、reader、processing、visualization、replay。
5. 用 fake/offline payload 覆盖有限值、shape、旧 generation 和关闭路径。

### 修改控制、IK 或碰撞

1. 先改纯 helper，并保留明确的 frame/unit/shape 校验。
2. 追踪候选拒绝、hold/fallback、delta clamp、worker recheck 和 replay preflight。
3. 不用应用插值替代固件能力，不为通过离线检查而削弱安全门。
4. 至少验证 unreachable、collision、stale input、feedback fault 和正常恢复。

### 修改 recording/replay

1. 先读 `episode_schema.py` 和 `episode_reader.py`，确认是不是 schema 变更。
2. START/STOP/status/sample 保持固定 dtype，并经 RecorderIO 共享控制路径传递。
3. 写入仍需临时目录、校验、fsync 和原子发布；任何 stream/codec/overflow 错误 fail closed。
4. 同步 reader、质量规则、可视化、数据处理和 replay；不要“只改 writer”。

### 新增 worker/CLI

1. CLI 只解析参数并调用 domain lifecycle。
2. lifecycle 统一负责 config、storage、spawn、readiness、supervision、worker death 和 shutdown。
3. 通过共享数据面通信，不传 SDK、可变对象图或跨进程业务回调。

## 开发与交付检查

```bash
git status --short
rg -n "<symbol-or-config-key>" dexmani_real examples
conda run -n real_robot python -m compileall -q dexmani_real examples
git diff --check
git diff --stat
```

只运行与改动匹配的离线检查；不要启动硬件入口代替测试。交付时说明：改了什么、运行了什么、哪些硬件验证未运行。若改变用户入口或实现导航，再同步 README/本文件；若改变数据字段，更新 `docs/dataset/` 对应合同。
