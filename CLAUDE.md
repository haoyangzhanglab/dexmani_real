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
| FK/IK/碰撞/路径 | `planning/` | teleop hold/fallback、delta clamp、replay dense preflight |
| episode schema/质量 | `recording/episode_schema.py` | writer → reader → processing/visualization/replay |
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
- `recording/` 不改变采样时机和业务语义；direct 与 RecorderIO 共用 `EpisodeRecorder`。

## 不可破坏的协议

### 数据与时钟

- `utils/schema.py` 是共享 NumPy payload 的单一来源；边界检查 shape、dtype、finite 和单位。
- ring 使用 seqlock；`get_last_k(k)` 返回已验证、oldest-first 的结果，可能少于 `k`，且拒绝 `k > maxlen`。
- arm queue 是有序 `maxsize=2`；hand command ring 是 latest-wins，`policy_plan_ring` 也是 latest-wins 且 `maxlen=3`。
- `is_running`、`is_recording`、`error_state`、`estop_request` 是简单 flag；只有 `safety_state` 存 `SafetyState`。
- heartbeat 使用 `time.monotonic()`；`error_state` sticky。
- recording 以 `1 / control_hz` 固定网格采样，不按传感器到达时间采样。

### 生命周期与命令

- main 管理 `DISARMED ↔ ARMED`、`→ FAULT` 和 shutdown；policy 管理 `ARMED ↔ RUNNING`。
- `run_generation` 标记控制时代；BEGIN、pause、home、feedback fault 使旧命令失效。camera stall/写盘错误丢弃 episode，但等待下一次显式 BEGIN 才推进 generation。
- worker 在共享命令跨越 SDK 边界前再次检查 generation、有效期、状态和 payload。
- 普通 pause 是 command quiescence：不发送 measured hold，不调用 State 6；让固件完成已接受的 endpoint。
- xArm Mode 6 已负责 arm trajectory smoothing；不要加入应用侧插值。
- SafetyGate 是 action publish 的统一边界；固件仍是最后 backstop。

### Episode 与策略

- runtime episode 只接受 schema v17；目录包含 `data.h5`、`depth.h5`、`rgb.mp4`，失败不原子发布。
- `recording/episode_schema.py` 同时约束 writer、reader 和 finalizer；不要维护第二份 dataset 列表。
- point cloud 不作为独立 sidecar 保存，由 depth 和 metadata 在消费边界确定性派生。
- direct 是默认 recorder backend；`v17` 只改变 RecorderIO transport，不改变 episode 字段语义。
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
2. direct 与 RecorderIO 必须 field-for-field 一致；START/STOP/status/sample 保持固定 dtype。
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
