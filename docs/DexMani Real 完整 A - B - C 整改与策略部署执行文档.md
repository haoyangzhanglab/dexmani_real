# DexMani Real 完整 A / B / C 整改与策略部署执行文档

**适用对象：** Claude Code / Codex / 人工开发者  
**执行假设：** 当前待修改仓库尚未执行本文任何 A/B/C 改动  
**执行原则：** A、B、C 从头执行，不利用“公开 main 已经改过”作为跳过依据  
**目标系统：** xArm7 + XHand + RealSense + Quest VR + 数据采集/回放 + Learned Policy Deployment

---

# 0. 文档目的

本文统一此前以下工作：

- xArm7 使用方式审查；
- XHand 使用方式审查；
- VR 使用方式审查；
- DexMani Real 项目审查；
- 进程隔离审查；
- SharedStorage / shared-memory 审查；
- 数据录制审查；
- ManiUniCon 架构分析；
- DexMani Policy 推理/部署分析；
- 策略部署方案；
- C 阶段二次架构审查。

最终执行路线为：

```text
未整改 DexMani Real
        │
        ▼
Phase A
软件正确性 + 硬件 SDK contract
+ IPC/lifecycle + perception/recording
        │
        ▼
Phase B
XHand lifecycle 真实硬件验证
+ 架构减法
        │
        ▼
A/B Runtime Freeze
        │
        ▼
Phase C
通用 Learned-Policy Deployment Runtime
        │
        ▼
FakeBackend
        │
        ▼
DexMani Policy Adapter
        │
        ▼
第二 Backend Swap 验证
        │
        ▼
真实机器人分级部署
```

---

# 1. 最高优先级原则

## 1.1 不允许跳阶段

执行顺序固定：

```text
A0
↓
A1-A10
↓
A Gate
↓
B0
↓
B1
↓
B2
↓
B3-B5
↓
B Gate
↓
A/B Freeze
↓
C0-C11
```

除明确写为：

```text
AUDIT → PASS 后无需代码修改
```

的项目外，不得自行认定：

```text
“这个应该已经修了”
“main 上似乎存在”
“历史 commit 做过”
```

然后跳过。

对于所有 audit 项，即使最终是：

```text
NO CODE CHANGE
```

也必须留下：

```text
检查文件
检查符号
检查结果
证据
```

---

# 2. 基线记录

任何修改前执行：

```bash
git status --short
git rev-parse HEAD
git log -5 --oneline
git diff --stat
```

记录：

```text
BASE_SHA=
BRANCH=
WORKTREE=
PYTHON=
CONDA_ENV=
DATE=
```

如果 worktree 非 clean：

禁止：

```text
git reset --hard
git checkout .
git restore .
git stash
```

除非用户明确要求。

---

# 3. 首轮必读文件

代码代理必须首先阅读：

```text
AGENTS.md
README.md
CLAUDE.md
```

然后阅读：

```text
dexmani_real/config/defaults.py
dexmani_real/config/runtime.py

dexmani_real/utils/schema.py

dexmani_real/shm/shared_storage.py
dexmani_real/shm/ring_buffer.py
dexmani_real/shm/camera_ring.py

dexmani_real/robot/arm_loop.py
dexmani_real/robot/xhand.py
dexmani_real/robot/hand_process.py
dexmani_real/robot/types.py
dexmani_real/robot/safety.py

dexmani_real/policy/runtime.py
dexmani_real/policy/safety.py
dexmani_real/policy/loop_timing.py

dexmani_real/teleop/loop.py
dexmani_real/teleop/snapshot.py
dexmani_real/teleop/hand_control.py

dexmani_real/sensor/camera_process.py
dexmani_real/sensor/pointcloud_processor.py

dexmani_real/recording/timestamp_buffer.py
dexmani_real/recording/episode_recorder.py
dexmani_real/recording/episode_reader.py
dexmani_real/recording/transaction.py

examples/collect_teleop.py
examples/replay_episode.py
examples/calibrate_camera.py
examples/calibrate_vr_heading.py
```

目标架构必须保持：

- `SharedStorage` 是唯一跨进程数据面；
- 高频 IPC 使用固定 shape NumPy payload；
- xArm/XHand/RealSense SDK 归各自 worker 所有；
- arm transport 为有序、短、有 backpressure 的 queue；
- hand transport 为 latest-wins；
- RecorderIO 不负责控制决策；
- 控制路径使用 monotonic clock；
- 不跨进程传活 SDK 对象。  
这些都是 DexMani 当前架构的核心边界。

---

# 4. 离线与硬件执行边界

代码代理可以自动运行：

```text
compileall
纯 Python deterministic checks
dtype checks
fake SDK tests
fake lifecycle tests
synthetic SharedStorage tests
reader/writer tests
scheduler tests
```

未经明确授权不得运行：

```text
真实 xArm motion
真实 XHand finger motion
homing
collect_teleop live
keyboard live control
replay --live
RealSense calibration
XHand lifecycle soak
SIGTERM/SIGKILL hardware lifecycle test
```

---

# 5. Phase A 总目标

Phase A 解决：

```text
SDK contract correctness
控制决策正确性
worker cleanup
shared-memory publication correctness
action ACK semantics
VR recovery correctness
3D perception configuration correctness
calibration durability
recording semantics correctness
```

Phase A **不是检查阶段**。

只要当前未整改仓库存在原审查问题，就必须实际修改。

---

# 6. Phase A0 — 建立 Offline Regression Harness

新增：

```text
checks/
└── offline/
```

至少建立：

```text
check_arm_c24_recovery.py
check_arm_live_error.py
check_worker_cleanup.py
check_ring_commit.py
check_ring_history.py
check_coupled_ack.py
check_vr_rotation_recovery.py
check_perception_contract.py
check_atomic_calibration.py
check_recording_send_semantics.py
```

原则：

```text
Python + assert
```

优先于大型 pytest framework。

不要为了“有测试”首先引入：

```text
mock framework
test manager
simulation framework
dependency injection framework
```

---

# 7. Phase A1 — xArm C24 Recovery

## 问题

C24 recovery 不能只：

```text
clean error
→ clean warn
→ mode 6
→ state 0
→ measured hold
```

必须遵守控制器重新使能 contract。

原审查明确定位到 C24 recovery 缺失 `motion_enable(True)`。

---

## 修改

文件：

```text
dexmani_real/robot/arm_loop.py
```

目标流程：

```text
optional C24 diagnostics
        ↓
clean_error()
        ↓
clean_warn()
        ↓
motion_enable(True)
        ↓
进入统一 Mode-6-ready helper
        ↓
验证 live state/mode
        ↓
读取 fresh measured qpos
        ↓
exactly one measured-hold endpoint
```

禁止重新手写第二套：

```text
set_mode(6)
set_state(0)
wait mode/state
```

若已有 `_enter_mode6_ready()`：

直接复用。

---

## 保持不变

不能改：

```text
短时间第二次 C24 → sticky fault
max consecutive recovery
measured hold 只发一次
speed
mvacc
nearest-equivalent wrapping
```

---

## Offline Test

FakeArm 记录调用顺序。

必须验证：

```text
clean_error
<
clean_warn
<
motion_enable
<
mode/state ready
<
get measured qpos
<
servo hold
```

如果：

```text
motion_enable != 0
```

不得继续发送 servo target。

---

# 8. Phase A2 — xArm 控制决策改用 Live Controller Error

## 问题

控制 API 返回 non-zero 后，使用：

```python
arm.error_code
```

这样的本地缓存字段做故障分类是不可靠的。

---

## 修改

增加极窄 helper：

```python
def _read_live_error_code(arm) -> int:
    ...
```

基于同步 controller query，例如：

```text
get_err_warn_code()
```

语义：

```text
成功
→ 返回 controller live error

读取失败
→ fail closed
→ 不 fallback cached error
```

---

## 适用场景

至少：

```text
set_servo_angle failure
homing restore Mode-6 decision
关键 recovery/postcondition
```

---

## 不适用场景

普通 telemetry：

```text
status display
periodic diagnostic logging
```

可以继续使用 cached telemetry。

不要把整个 arm loop 都变成：

```text
每 tick 同步 query controller error
```

---

# 9. Phase A3 — Homing Restore Fail-Closed

文件：

```text
dexmani_real/robot/arm_loop.py
```

在 planned homing 结束，准备恢复实时 Mode 6 时：

不能依据 cached：

```python
arm.error_code
```

决定：

```text
restore servo mode
```

应：

```text
live error query
        ↓
error == 0
        ↓
允许恢复

error != 0
        ↓
不恢复

query failed
        ↓
不恢复
```

诊断失败不是：

```text
assume healthy
```

而是：

```text
fail closed
```

---

# 10. Phase A4 — 澄清 `motion_enabled` 命名

如果代码中存在类似局部变量：

```python
motion_enabled
```

但实际语义只是：

```text
controller currently accepts Mode-6 command
```

改成：

```python
accepts_motion_commands
```

或：

```python
controller_motion_ready
```

只做局部语义澄清。

禁止因此：

```text
新增 SafetyState
新增 motion state machine
调用 motion_enable(False)
```

---

# 11. Phase A5 — TCP Load 配置化

## 问题

机械臂负载参数不应硬编码在：

```text
arm_loop.py
```

---

## 修改

在：

```text
config/defaults.py
```

的 arm runtime 参数中增加：

```python
tcp_load_mass_kg: float
tcp_load_cog_mm: tuple[float, float, float]
```

提供当前硬件默认值。

---

## Validation

检查：

```text
mass finite
mass >= 0
COG shape == 3
COG all finite
```

---

## ArmLoopConfig

让：

```text
ArmLoopConfig.from_runtime(...)
```

携带 resolved 值。

设备初始化：

```text
set_tcp_load(
    weight=cfg.tcp_load_mass_kg,
    center_of_gravity=cfg.tcp_load_cog_mm,
)
```

---

## 禁止

不要新建：

```text
TcpLoadManager
TcpCalibrationRegistry
TcpLoadParamsHierarchy
```

已有 frozen config 足够。

---

# 12. Phase A6 — XHand `clear_error()` 语义修复

## 问题

如果：

```python
XHand.clear_error()
```

只清：

```text
local Python error latch
last error code
last error string
```

而没有发送硬件 clear command，则命名错误。

---

## 修改

改为：

```python
clear_local_error()
```

同步所有 callsite。

日志明确：

```text
reset local driver error latch
```

而不是：

```text
clear XHand hardware error
```

---

## 不要动

```text
retry count
board watchdog
shared.error_state sticky semantics
真实 SDK error behavior
```

---

# 13. Phase A7 — 删除误导性的 `XHand.stop()`

如果 repo 中：

```python
XHand.stop()
```

满足：

```text
没有真实 callsite
同时会发送 mode=0 / torque=0 等新硬件 command
```

则直接删除。

原审查明确把该 API 判为无真实调用且具有误导性的 dead API。

不要：

```text
deprecated alias
empty stop()
future-use comment
```

XHand shutdown contract 应明确为：

```text
command producer already quiescent
        ↓
worker exits
        ↓
disconnect
```

而不是：

```text
shutdown
→ 临时制造一个新 motion command
```

---

# 14. Phase A8 — Actuator Worker Exception-Safe Cleanup

这是 Phase A 的 P0 项。

## Arm

`arm_loop()` 的主要运行区必须：

```python
try:
    run_loop()
finally:
    cleanup()
```

cleanup：

```text
best effort state=4
→ bounded verification
→ disconnect
```

如果已有：

```python
_disconnect_arm(...)
```

复用它。

---

## Hand

`hand_process()`：

```python
try:
    run_loop()
finally:
    hand.disconnect()
```

注意：

**Hand finally 只 disconnect。**

禁止自动增加：

```text
stop()
unforce()
mode0
zero torque command
home
```

---

## 必测

Fake device：

```text
normal loop exit
exception in control loop
exception in state publication
exception during command processing
```

全部都应进入 finally。

---

# 15. Phase A9 — Shared-Memory Ring Commit Contract

这是 A 阶段另一个 P0。

## 核心规则

一个逻辑序列号只有在：

```text
payload 完整写入
+
seqlock commit 完成
```

之后才可以对 reader 可见。

原审查发现的典型错误是 sequence 在 payload 之前发布。

---

## 推荐写入顺序

```text
choose slot
        ↓
mark slot odd / writing
        ↓
write payload
        ↓
write authoritative publish timestamp
        ↓
mark slot even / committed
        ↓
publish logical global sequence LAST
        ↓
publish write index
```

具体顺序需适应现有 ring 实现，但核心不变量：

> reader 看到新的 logical sequence 时，该 slot 已经是完整 committed frame。

---

## Publish timestamp ownership

生产者可以提供：

```text
source timestamp
receive timestamp
```

但真正：

```text
publish_monotonic_ns
```

应由 ring 在 commit 临界点附近写入。

对于 camera 大数组：

```text
RGB copy
depth copy
pointcloud copy
```

必须先完成，之后再采 publish timestamp。

否则 timestamp 会虚假提前。

---

# 16. Phase A10 — `get_last_k()` 历史恢复

当 reader 遍历最近 k 个序列时：

```text
某 slot 已经被覆盖
```

不能：

```text
立即 return
```

否则较旧但仍有效的历史被全部丢失。

正确：

```text
seq mismatch
→ mark dropped
→ continue search older logical sequence
```

返回：

```text
oldest-first
```

且：

```text
实际有效数量允许 < k
```

---

# 17. Phase A11 — Coupled Arm + Hand ACK

## Action Transport 语义

arm：

```text
ordered bounded queue
```

hand：

```text
latest-wins ring
```

两者不是 transaction。

禁止新增：

```text
PREPARE
COMMIT
ROLLBACK
```

---

## `wait_applied=True`

### Arm-only

成功条件：

```text
arm.last_cmd_seq >= action_id
```

### Arm + Hand

成功条件：

```text
arm.last_cmd_seq >= action_id
AND
hand.last_cmd_seq == action_id
AND
hand healthy
```

如果：

```text
hand.last_cmd_seq > action_id
```

表示 hand endpoint 已经被 supersede：

```text
立即返回失败
```

不要继续等 timeout。

当前公开实现中的 coupled ACK 语义正是这种形式，可作为目标实现参考。

---

## 不要改变实时路径

普通：

```text
16 Hz / realtime action
```

不应该每次阻塞等待 ACK。

只有明确：

```python
wait_applied=True
```

的调用才执行同步确认。

---

# 18. Phase A12 — Hand Delta Preflight

原审查中的 W3A 不应被无声删除。

必须检查：

```text
所有能产生 coupled arm+hand action 的路径
```

是否在 arm endpoint 入队之前已经完成：

```text
whole hand command
+
mechanical bound
+
operational bound
+
command-to-command delta
```

验证。

如果多个路径重复：

可以抽：

```python
validate_hand_command_delta(...)
```

如果只有一个清晰路径：

保持内联。

原则：

```text
违反 delta
→ reject whole hand candidate
```

禁止：

```text
np.clip
部分关节执行
```

---

# 19. Phase A13 — VR Rotation Spike Recovery Audit

此项即使历史审查曾判定已修，也必须在用户的未整改 repo 重新核查。

检查：

```text
teleop arm mapper
```

是否：

```text
raw VR pose
→ update baseline/history
→ gating/rotation validation
```

还是错误地：

```text
gated pose
→ 反过来成为下一帧 raw baseline
```

目标：

异常 rotation frame 被拒绝之后：

```text
下一帧比较基线仍来自真实 raw tracking sequence
```

避免一个 spike 永久污染 reference。

如果当前实现已满足：

```text
AUDIT PASS / NO CODE CHANGE
```

如果不满足：

实际修复。

---

# 20. Phase A14 — 3D Perception Runtime Contract

文件：

```text
sensor/camera_process.py
sensor/pointcloud_processor.py
config/runtime.py
```

---

## 20.1 Desk Plane Single Source of Truth

禁止生产点云同时存在：

```text
runtime resolved environment.table.plane_abcd
```

和：

```text
pointcloud_processor 自动读取 desk_plane.json
```

两个独立事实源。

正确：

```text
YAML/default
        ↓
runtime resolver
        ↓
resolved plane_abcd
        ↓
camera process
        ↓
pointcloud processor
```

当：

```text
table.enabled == false
```

则：

```text
desk_plane = None
```

不得 fallback 到旧 JSON。

---

## 20.2 Alignment Fail Early

生产 point cloud 若依赖：

```text
depth_to_color
```

则启动时明确检查：

```text
align_mode == depth_to_color
```

不满足：

```text
startup failure
```

不要运行到 point cloud 几十帧之后才隐式产生错误几何。

---

# 21. Phase A15 — Point Cloud Metadata Audit

检查已有 schema 是否完整记录：

```text
pointcloud_valid
number of points
valid depth ratio
camera health
filter/settings needed to reproduce output
```

特别确认：

```text
has_pointcloud
```

语义必须基于：

```text
至少存在一个 valid pointcloud frame
```

而不是：

```text
只要 camera 有 frame
```

如果已经满足：

```text
AUDIT PASS
```

不要建立新的 metadata service。

---

# 22. Phase A16 — Calibration Atomic Write

适用于：

```text
save desk plane
camera calibration
VR heading calibration
```

禁止：

```python
open(path, "w")
json.dump(...)
```

直接覆盖正式文件。

---

## Atomic Write

```text
build complete JSON
        ↓
write temp in same directory
        ↓
flush
        ↓
fsync(temp)
        ↓
os.replace(temp, final)
        ↓
fsync(parent directory)
```

如果要保留 backup：

```text
copy old file
```

而不是：

```text
rename old file away
→ 中途 crash 导致正式 path 消失
```

如果 ≥2 个 caller：

可以建立极小：

```python
atomic_json_dump(...)
```

不要建立通用 transaction framework。

---

# 23. Phase A17 — Fixed-Grid Recording：Hold ≠ Send Event

这是数据语义 P0。

固定控制网格中的：

```text
CAUSAL_HOLD_LAST
```

表示：

```text
为了网格连续性沿用 effective target
```

不表示：

```text
机器人在这个 grid tick 又收到了一次 command
```

---

## Synthetic Grid Hold

可以继承：

```text
effective arm target
effective hand target
```

但必须清零/置 sentinel：

```text
flag_action_queued
action_id
action_created_monotonic_ns
command send timestamp
其它 send-event provenance
```

---

## Replay

如果：

```text
send_mask == false
```

则：

```text
不得重新发送该 synthetic grid hold
```

否则原始数据中的：

```text
“没有发命令”
```

会被 replay 错误转换为：

```text
“发了一次重复命令”
```

---

# 24. Phase A18 — Recording Schema Gate

优先：

```text
不 bump schema
```

如果现有 v16 sentinel 已能表达：

```text
grid valid
但没有 send event
```

就只修 producer/reader/replay 语义。

只有确实无法兼容表达时才提出：

```text
schema migration proposal
```

代码代理不得自行升级格式。

---

# 25. Phase A19 — SafetyGate 保持窄职责

Phase A 不得借整改之机重新引入：

```text
generic path collision geometry
dense transition collision check
软件速度轨迹发生器
软件加速度轨迹发生器
```

实时 SafetyGate 应维持：

```text
well-formed
+
joint limits
+
workspace
+
必要的当前状态安全条件
```

固件：

```text
velocity
acceleration
low-level safety
```

仍是最后执行边界。

显式：

```text
homing
replay dense preflight
```

可以拥有更重几何检查。

---

# 26. Phase A20 — Mode 6 语义

正常 xArm servo：

```text
Mode 6
+
set_servo_angle(wait=False)
```

不要加入 application-side interpolation。

机器人应该接收离散合法 endpoint，由 xArm Mode 6 处理平滑。

---

# 27. Phase A Commit 建议

建议拆分：

```text
A0  test: add minimal offline regression checks

A1  fix(xarm): follow SDK contract during C24 recovery

A2  fix(xarm): use live controller errors for control decisions

A3  refactor(robot): clarify arm readiness and XHand local error semantics

A4  refactor(xarm): move TCP load into runtime configuration

A5  fix(runtime): make actuator worker cleanup exception-safe

A6  fix(shm): publish ring sequence only after committed payload

A7  fix(shm): preserve older verified history across overwritten slots

A8  fix(policy): require both actuator acknowledgements for coupled sync action

A9  fix(teleop): preserve raw VR recovery baseline

A10 fix(perception): use resolved production calibration contract

A11 fix(calibration): publish calibration JSON atomically

A12 fix(recording): separate synthetic grid hold from send-event provenance
```

如果某 audit 已经 pass：

该 commit 不需要产生。

但 audit 报告仍需记录。

---

# 28. Phase A Offline Gate

必须完成：

```bash
python -m compileall -q dexmani_real examples checks
```

以及所有：

```text
checks/offline/*
```

必须确认：

```text
[ ] C24 call order PASS
[ ] live error fail-close PASS
[ ] homing restore live error PASS
[ ] actuator finally cleanup PASS
[ ] ring commit PASS
[ ] get_last_k history PASS
[ ] coupled ACK PASS
[ ] hand supersede PASS
[ ] hand delta reject PASS
[ ] VR recovery PASS
[ ] perception config PASS
[ ] atomic JSON PASS
[ ] synthetic hold/send semantics PASS
[ ] replay send-mask PASS
```

---

# 29. Phase A Definition of Done

Phase A 完成时：

```text
xArm controller decision 不依赖 stale cached error

C24 recovery:
clean
→ enable
→ mode6-ready
→ one measured hold

XHand:
local error API 不冒充 hardware clear
无危险 dead stop API

所有 actuator unexpected Python exception:
→ finally cleanup

Shared-memory sequence:
只代表 committed frame

Fixed-grid recording:
grid continuity != command send event

point cloud:
只消费 resolved calibration

calibration:
atomic

coupled ACK:
arm + hand 全部确认
```

Phase A 完成后才进入 B。

---

# 30. Phase B 总目标

Phase B 专门处理：

```text
XHand EtherCAT lifecycle
```

这是不能仅靠静态代码推导的硬件问题。

B 阶段必须：

```text
patch
→ offline fake
→ 人工 hardware A/B
→ 根据结果 merge/revert
```

而不是：

```text
看起来更干净
→ 直接重构
```

原审查明确将 B1 和 B2 标记为硬件门控实验。

---

# 31. Phase B0 — 固定实验环境

每次实验记录：

```text
Git SHA
patch SHA
XHand SDK version
native library version
serial number
left/right hand
EtherCAT interface
OS
kernel
Python version
multiprocessing start method
XHand firmware if available
power-cycle status
```

不能比较：

```text
不同 SDK
不同 hand
不同 NIC
不同 kernel
```

却声称是代码变量导致。

---

# 32. Phase B0.1 — Baseline Soak

在改 B1 前先运行原始 lifecycle：

每轮：

```text
construct driver
→ connect
→ one fresh read
→ disconnect
→ destroy
→ short interval
→ next iteration
```

不得在未授权情况下发 finger motion。

推荐：

```text
100 cycles minimum
500 cycles if practical
```

记录：

```text
connect success
connect failure
open error
vendor stderr/stdout
SDO failure
connect latency
disconnect latency
need power-cycle?
next reconnect successful?
```

原审查给出了同样的 reconnect soak 核心测量方式。

---

# 33. Phase B1 — Single Controller Discovery → Open

## 原问题

旧实现如果：

```text
temporary XHandControl
→ EtherCAT discovery
→ close

new XHandControl
→ actual open
```

会让 discovery/open 跨两个 controller/native-resource lifecycle。

---

## 实验 patch

改成：

```text
XHandControl
      ↓
enumerate/discover
      ↓
same controller
      ↓
open
      ↓
success
      ↓
becomes self.control
```

每个 retry attempt 可以重新创建 controller。

但同一个 attempt：

```text
discover + open
```

必须使用同一个 controller。

---

## B1 禁止同时改

保持：

```text
retry count
retry delay
disconnect behavior
INIT logic
watchdog wait
command loop
read loop
```

B1 一次只改：

```text
controller ownership during discovery/open
```

---

# 34. Phase B1 Fake Test

Fake XHandControl：

验证：

```text
enumerate controller identity
==
open controller identity
```

失败 retry：

```text
attempt 1 object != attempt 2 object
```

成功后：

```text
self.control
```

必须就是成功 attempt 的 controller。

---

# 35. Phase B1 Hardware Gate

人工执行与 baseline 相同 soak。

通过条件：

```text
failure rate <= baseline
无 reproducible write sdo failed regression
无新增 reconnect hang
无新增 power-cycle
无明显 FD/native resource leak
失败 diagnostics 仍完整
```

如果成功：

```text
B1 = MERGE
```

如果失败：

保留完整 logs。

只有出现可重复证据：

```text
single-controller fail
AND
isolated-discovery 在完全相同 setup 成功
```

才能考虑极窄 compatibility switch。

禁止预先加入：

```text
use_legacy_xhand_discovery: bool
```

---

# 36. Phase B2 — Close-Only Disconnect

B2 只有 B1 的 implementation decision 已固定之后开始。

---

## B2 旧行为

典型旧 shutdown：

```text
_request_slave_init()
→ close_device()
→ watchdog sleep
→ release controller
```

---

## B2 实验 patch

只改变为：

```text
close_device()
→ connected = False
→ self.control = None
```

前置条件：

```text
command loop 已退出
read loop 已退出
不会再 send/read
```

---

## B2 一次只改这个变量

不同时改：

```text
discovery
retry
timeout
command mode
threading
watchdog configuration
```

---

# 37. Phase B2 Hardware Test Matrix

必须至少测试：

### Normal exit

```text
connect
→ read
→ close
→ reconnect
```

### SIGTERM

验证：

```text
finally cleanup
→ next process reconnect
```

### Python exception

主动 fake/trigger Python exception：

```text
loop exception
→ finally
→ disconnect
→ reconnect
```

### Supervisor recovery

worker unexpected exit：

```text
supervisor detects
→ global shutdown
→ subsequent clean session reconnect
```

### SIGKILL

只记录：

```text
OS/native recovery behavior
```

不能要求：

```text
Python finally
```

在 SIGKILL 下执行。

---

# 38. Phase B2 判定

重点不是：

```text
close() 返回 0
```

而是：

```text
close 后下一轮是否可靠 reconnect
```

观察：

```text
stale slave state
SDO failure
open retry
power-cycle requirement
watchdog dependency
```

---

## PASS

只有 close-only 在 soak 中与 baseline/B1：

```text
同等或更稳定
```

才 merge，并删除：

```text
INIT helper
EC state constant
post-disconnect watchdog
对应 obsolete comments
```

---

## FAIL

如果出现可重复：

```text
stale OP
SDO failure
reconnect degradation
```

则：

```text
REVERT B2
```

保留：

```text
INIT transition
+
watchdog wait
+
close
```

Phase B2 仍然是：

```text
EXECUTED
```

只是最终工程决策：

```text
NO-GO
```

这不属于“跳过”。

---

# 39. Phase B3 — XHand Lifecycle 最终清理

根据 B1/B2 结果，清理：

```text
obsolete comments
temporary experimental logging
dead constants
unused discovery helpers
```

但只删除已经由实验结果证明不再需要的东西。

---

# 40. Phase B4 — Runtime Topology 结构减法

硬件行为稳定之后才能开始。

目标不是重新设计 runtime，而是消除重复 topology/lifecycle glue。

可以引入一个极小：

```python
@dataclass(frozen=True)
class WorkerSpec:
    name: str
    target: Callable
    args: tuple
    ready_name: str | None
```

用于收敛：

```text
spawn
start
readiness wait
join
```

重复代码。

---

## 禁止

```text
ProcessManager framework
worker registry
plugin registry
service container
dependency injection container
actor framework
```

---

# 41. Phase B5 — Dead Config / Dead State / Comment Cleanup

每删除符号前：

```bash
rg "<symbol>" dexmani_real examples checks
```

必须确认：

```text
producer
consumer
recording schema
reader
replay
docs
dynamic getattr
```

不能删除：

```text
持久化 v16 reserved field
```

仅因为：

```text
当前 runtime 总是填 0
```

Schema cleanup 必须独立 migration。

---

# 42. Phase B6 — 大文件拆分

仅在一个大文件确实阻碍维护时执行。

允许：

```text
arm_loop.py
→ homing-specific logic
→ robot/homing.py
```

条件：

```text
单一 domain
无 behavior change
无 circular import
offline equivalence
```

禁止：

```text
顺手把 arm_loop 拆成十几个 manager
```

---

# 43. Phase B Gate

Phase B 完成必须产出：

```text
B1 decision:
PASS/FAIL
hardware evidence

B2 decision:
PASS/FAIL
hardware evidence

final XHand connect lifecycle:
...

final XHand disconnect lifecycle:
...
```

并再次执行 A 全部 regression。

---

# 44. A/B Freeze Report

进入 C 之前生成：

```text
docs/ab_runtime_freeze_report.md
```

至少记录：

```text
A base SHA
A final SHA

B base SHA
B final SHA

xArm Mode
xArm recovery contract

XHand connect contract
XHand disconnect contract

arm queue semantics
hand ring semantics

SharedStorage ring commit semantics

SafetyGate contract

run_generation semantics

recording grid semantics

episode schema version
```

C 阶段必须把这些看作：

```text
FROZEN RUNTIME CONTRACT
```

---

# 45. C 阶段目标

C 阶段不是：

```text
把 dexmani_policy 代码复制进 dexmani_real
```

而是：

> 建立一个能够接入不同模型仓库，同时完全复用 A/B 后机器人 runtime、安全、IPC 和生命周期机制的策略部署层。

最终：

```text
DexMani Policy
π0
ACT
Diffusion Policy
RDT
其它模型
```

理论上都通过 adapter 接入。

---

# 46. ManiUniCon 的正确参考方式

ManiUniCon 的 `TorchModelPolicy` 显式拆分：

```text
model
obs_wrapper
act_wrapper
```

并具有 observation horizon、多步 action 与 timestamp/chunking 概念，这些适合作为 C 的设计参考。

但不要复制其所有 ownership。

特别是不要让一个大型 policy object 同时拥有：

```text
SharedStorage
model
keyboard listener
recording
action publication
system stop
```

C 阶段要利用 A/B 已建立的 DexMani runtime 边界。

---

# 47. C 最终架构

```text
           ┌─────────────────────────────┐
           │       Sensor Workers        │
           │ camera / arm / hand state   │
           └──────────────┬──────────────┘
                          │
                          ▼
                   SharedStorage
                          │
                          ▼
             ┌──────────────────────┐
             │   Inference Worker   │
             │                      │
             │ causal observation   │
             │       ↓              │
             │ ObservationAdapter   │
             │       ↓              │
             │ PolicyBackend        │
             │       ↓              │
             │ ActionAdapter        │
             └──────────┬───────────┘
                        │
                        ▼
                 policy_plan_ring
                   latest-wins
                        │
                        ▼
             ┌──────────────────────┐
             │Deployment Coordinator│
             │                      │
             │ generation           │
             │ plan freshness       │
             │ scheduler            │
             │ ActionCandidate      │
             │ SafetyGate           │
             └──────────┬───────────┘
                        │
                 command transport
                   │           │
                   ▼           ▼
              arm queue     hand ring
                   │           │
                   ▼           ▼
              arm worker   hand worker
```

---

# 48. C 的最高原则

## Inference Worker 禁止直接：

```text
arm_action_q.put()
hand_cmd_ring.write()
xArm SDK
XHand SDK
SafetyState mutation
RecorderIO command
run_generation advance
```

模型输出只是：

```text
proposal
```

不是：

```text
robot command
```

---

# 49. Phase C0 — 冻结 Deployment Contracts

新建：

```text
dexmani_real/deployment/
```

建议：

```text
contracts.py
config.py
loader.py
observation.py
worker.py
coordinator.py
lifecycle.py
metrics.py
```

不要一开始创建：

```text
manager/
plugins/
registry/
services/
backend_registry/
```

---

# 50. Phase C1 — 公共 Causal Reader

A 阶段已经验证 teleop causal observation。

C 不允许复制第二套。

从：

```text
teleop/snapshot.py
```

抽取纯 shared-memory causal reader：

```text
shm/causal_reader.py
```

---

## Contract

对任一 observation anchor：

```text
source_monotonic_ns <= anchor
receive_monotonic_ns <= anchor
publish_monotonic_ns <= anchor
```

不能使用未来 frame。

---

## Consumer

```text
teleop/snapshot.py
        ↓
causal_reader

deployment/observation.py
        ↓
causal_reader
```

---

## 要求

抽取前后 teleop：

```text
same synthetic input
→ same selected frame
```

不要同时改变：

```text
age threshold
VR threshold
camera threshold
source precedence
```

---

# 51. Phase C2 — 公共 Candidate Publication Boundary

当前所有 control source 最终都应该汇入：

```text
ActionCandidate
```

公开实现已经将其建模为包含 observation、generation、monotonic timing、arm/hand target 等字段的 backend-neutral contract，可作为最终目标。

建议最终公共函数：

```python
def validate_and_send_candidate(
    shared,
    candidate,
    *,
    gate,
    prepare_timeout_s,
) -> ActionCandidate | None:
    ...
```

---

## 职责

```text
check current run_generation
        ↓
read fresh actuator feedback
        ↓
allocate global monotonic action_id
        ↓
SafetyGate.validate
        ↓
send_command
        ↓
return actual sent candidate
```

---

## 不应承担

```text
VR mapping
policy inference
action chunking
recording sampling
keyboard behavior
homing
```

---

# 52. Phase C3 — Deployment Protocols

```python
class PolicyBackend(Protocol):
    def load(self) -> None: ...
    def reset(self, *, run_generation: int) -> None: ...
    def infer(self, model_input: Any) -> Any: ...
    def close(self) -> None: ...
```

---

```python
class ObservationAdapter(Protocol):
    def encode(
        self,
        observation: ObservationBatch,
    ) -> Any: ...
```

---

```python
class ActionAdapter(Protocol):
    def decode(
        self,
        raw_output: Any,
        *,
        context: InferenceContext,
    ) -> JointActionChunk: ...
```

---

# 53. PolicyBackend Ownership

PolicyBackend 可以：

```text
import torch
load checkpoint
allocate CUDA
run model
maintain recurrent/model-local state
```

不能：

```text
hold SharedStorage
hold robot SDK
modify safety state
allocate global action_id
send command
record episode
```

---

# 54. ObservationBatch

使用 process-local immutable object：

```python
@dataclass(frozen=True)
class ObservationBatch:
    observation_id: int
    run_generation: int
    anchor_monotonic_ns: int

    arm_history: ...
    hand_history: ...
    tactile_history: ...
    camera_history: ...

    source_sequence: ...
    source_monotonic_ns: ...
    publish_monotonic_ns: ...
    valid_mask: ...
```

它：

```text
不进入 SharedStorage
```

因此无需建立 IPC dtype。

---

# 55. Observation Horizon Validation

启动时：

```text
requested arm history <= arm ring capacity
requested hand history <= hand ring capacity
requested camera history <= camera ring capacity
```

不足：

```text
fail startup
```

不要默默：

```text
duplicate latest frame
```

也不要为了一个模型将所有 ring 放大几十倍。

---

# 56. JointActionChunk

Phase C 首版 runtime canonical action：

```python
@dataclass(frozen=True)
class JointActionChunk:
    arm_qpos: np.ndarray            # [N, 7]
    hand_qpos: np.ndarray | None    # [N, 12]
    target_monotonic_ns: np.ndarray # [N]
    valid_mask: np.ndarray          # [N]
```

固定：

```text
representation = joint_position
units = rad
frame = robot_joint
```

---

# 57. 首版明确不支持

Runtime core 不直接支持：

```text
Cartesian EE target
delta pose
velocity
torque
impedance
latent hand action
FAAS action
```

模型 adapter 可以做：

```text
model representation
→ native joint target
```

但进入 coordinator 之前必须已经成为：

```text
7D arm + optional 12D hand joint target
```

---

# 58. Phase C4 — Lazy Backend Loader

配置：

```yaml
deployment:
  backend_target: "package.module:build_backend"
  observation_adapter_target: "package.module:ObservationAdapter"
  action_adapter_target: "package.module:ActionAdapter"

  model_config_path: "..."
  checkpoint: "..."
  device: "cuda:0"
```

Loader 只做：

```text
split module:symbol
import module
get symbol
instantiate
validate protocol
```

---

## 禁止

```python
if policy == "dp3":
elif policy == "maniflow":
elif policy == "pi0":
```

在 deployment core 中出现。

---

# 59. CUDA / Torch Import Boundary

Main：

```text
不 import torch
不 load checkpoint
不 initialize CUDA
```

正确：

```text
Main
→ spawn inference worker
→ inference child lazy import
→ backend.load()
```

这样避免：

```text
parent CUDA context
跨 spawn model object
硬件 runtime 被 PyTorch dependency 污染
```

---

# 60. Phase C5 — POLICY_PLAN_DTYPE

Inference 与 Coordinator 之间是跨进程 IPC。

因此增加固定 shape dtype：

```text
POLICY_PLAN_DTYPE
```

到：

```text
utils/schema.py
```

建议：

```text
plan_id
run_generation
observation_id

observation_anchor_monotonic_ns

inference_started_monotonic_ns
inference_finished_monotonic_ns

num_steps
arm_present
hand_present

target_monotonic_ns[MAX_POLICY_CHUNK_STEPS]

arm_qpos[MAX_POLICY_CHUNK_STEPS, 7]
hand_qpos[MAX_POLICY_CHUNK_STEPS, 12]

valid_mask[MAX_POLICY_CHUNK_STEPS]
```

---

# 61. MAX_POLICY_CHUNK_STEPS

这是：

```text
runtime transport capacity
```

不是：

```text
某一个模型 horizon
```

可选择保守固定值，例如：

```text
32 或 64
```

但必须：

```text
配置/adapter requested N <= MAX
```

否则 fail。

禁止 silently truncate。

---

# 62. DexMani Policy 与 Runtime 解耦

DexMani Policy 当前 joint action 定义包含 7 维 arm + 12 维 XHand；其标准训练/推理配置还使用 horizon、observation steps 和 action steps 等模型侧序列参数。

这些值只能属于：

```text
DexManiPolicyAdapter
```

不得成为：

```text
DexMani Real deployment runtime hard constant
```

因此：

```text
runtime MAX chunk
≠
policy n_action_steps
```

---

# 63. Phase C5.1 — policy_plan_ring

在：

```text
SharedStorage
```

新增：

```text
policy_plan_ring
```

语义：

```text
latest plan wins
```

推荐短 ring：

```text
2~4 slots
```

---

## 为什么不是 Queue

旧 observation 生成的 model plan：

```text
通常不应积压等待未来执行
```

新 plan：

```text
应该 supersede old plan
```

因此不能：

```text
FIFO all inference outputs
```

---

# 64. SharedStorage 增量

同步修改：

```text
utils/schema.py
shm/shared_storage.py
resource name list
allocation
cleanup
config
```

增加：

```text
inference ready
inference heartbeat
```

但不创建第二个 process-health system。

现有 SharedStorage 的角色就是集中拥有 rings、queues、events/flags，而非承载业务逻辑。

---

# 65. Phase C6 — Fake Backend

真实模型之前必须先建立：

```text
FakeObservationAdapter
FakePolicyBackend
FakeActionAdapter
```

FakeBackend：

```text
CPU only
deterministic
no torch
no hardware
```

例如：

```text
current arm qpos
+
tiny predefined deterministic offset sequence
```

输出：

```text
N × 7
```

hand 可选：

```text
copy current qpos
```

---

# 66. Fake Backend 是架构门

C6 阶段禁止：

```python
import dexmani_policy
```

如果没有 DexMani Policy 就无法跑：

```text
observation
→ backend
→ action chunk
→ plan ring
```

说明 core 抽象失败。

---

# 67. Phase C7 — Inference Worker

入口：

```python
def inference_loop(
    shared: SharedStorage,
    config: DeploymentConfig,
) -> None:
    ...
```

不是：

```python
class PolicyProcess(mp.Process)
```

生命周期继续由 A/B runtime 统一 supervision。

---

# 68. Inference Startup

顺序：

```text
heartbeat early
        ↓
lazy import
        ↓
instantiate adapters/backend
        ↓
backend.load()
        ↓
validate observation horizon
        ↓
validate output contract
        ↓
ready(inference)=true
```

load 失败：

```text
明确异常
→ process failure
→ supervisor
```

不得继续进入：

```text
dummy safe mode
```

而让系统误以为模型 ready。

---

# 69. Inference Loop

```text
read current generation
        ↓
create causal ObservationBatch
        ↓
ObservationAdapter.encode
        ↓
record inference_start
        ↓
PolicyBackend.infer
        ↓
record inference_end
        ↓
ActionAdapter.decode
        ↓
validate JointActionChunk
        ↓
re-read current generation
        ↓
same generation?
      /       \
    no         yes
   DROP        publish plan
```

---

# 70. In-Flight Generation Cancellation

例如：

```text
generation = 12
capture observation
start CUDA inference
        ↓
operator pause
generation = 13
        ↓
old inference returns
```

必须：

```text
DROP
```

禁止：

```text
plan.run_generation = current_generation
```

重新贴标签。

---

# 71. Backend Reset

发现 generation 改变：

```python
backend.reset(run_generation=new_generation)
```

用于清：

```text
RNN state
temporal action history
diffusion conditioning cache
previous action state
model-local chunk state
```

Backend 只能响应 generation。

不能自己：

```text
increment generation
```

---

# 72. Phase C8 — Deployment Coordinator

Coordinator 是 learned-policy 唯一 robot-action producer。

负责：

```text
plan selection
plan freshness
generation check
endpoint scheduler
ActionCandidate creation
SafetyGate
command publication
policy semantic watchdog
RUNNING ↔ ARMED control-source state
```

---

## 不负责

```text
checkpoint
torch
model preprocessing
pointcloud neural encoding
robot SDK
HDF5 write
```

---

# 73. Chunk 永远不能直接进入 Arm Queue

错误：

```python
for action in actions:
    arm_action_q.put(action)
```

这是硬性禁止。

arm queue 应继续保持：

```text
ordered
short
backpressure
```

而不是 policy trajectory buffer。

---

# 74. Hand Chunk 同样不能直接 dump

禁止：

```python
for hand_action in chunk:
    hand_cmd_ring.write(...)
```

否则：

```text
latest-wins ring
```

会在一个 inference tick 中把前 N-1 个动作立刻覆盖。

---

# 75. Coordinator Active Plan

本地维护：

```text
active_plan
consumed step indices
```

读到新 plan：

```text
generation current?
observation newer?
plan fresh?
shape valid?
```

全部通过：

```text
replace active_plan
```

否则：

```text
drop
```

---

# 76. Scheduler：Model Faster Than Control

假设当前 tick 时：

```text
step 2
step 3
step 4
```

都已经 due。

不要：

```text
连续快速发 3 条
```

应：

```text
coalesce
→ 选择 latest due endpoint
```

旧的 overdue intermediate target 已失去实时控制意义。

---

# 77. Scheduler：Model Slower Than Control

如果当前 control tick：

```text
没有新 endpoint due
```

则：

```text
不发新命令
```

禁止自动：

```text
duplicate last command
generate new action_id
hold measured position
interpolate
```

机器人继续完成固件最后已接受 endpoint。

---

# 78. 禁止 Application-Side Arm Interpolation

不能：

```text
model step A
model step B

→ numpy interpolation
→ 生成 8 个 intermediate arm targets
```

xArm Mode 6 平滑仍由 firmware 承担。

---

# 79. Endpoint → ActionCandidate

真实 due endpoint 转成：

```text
ActionCandidate
```

字段：

```text
observation_id
run_generation
created_monotonic_ns
target_monotonic_ns
valid_until_monotonic_ns

arm_qpos
hand_qpos

joint_position
rad
robot_joint

is_hold=False
```

然后才能：

```text
common candidate publisher
→ SafetyGate
→ transport
```

---

# 80. Phase C9 — Policy Failure Semantics

必须区分：

```text
bad model result
```

和：

```text
hardware/process failure
```

---

## 80.1 Drop Only

以下可仅 drop 单个 plan：

```text
generation mismatch
old observation
superseded plan
expired plan
```

同时计 metrics。

---

## 80.2 Abort Policy Run

以下属于 policy semantic failure：

```text
NaN
Inf
wrong action shape
invalid timestamp ordering
unsupported representation
SafetyGate rejection
repeated no-valid-action timeout
```

推荐：

```text
advance generation
RUNNING → ARMED
stop new commands
require explicit restart
```

不一定直接：

```text
FAULT
```

因为机器人硬件未必有故障。

---

# 81. Inference Process Failure

以下：

```text
worker crash
CUDA fatal error
backend unhandled exception
heartbeat timeout
```

交给：

```text
existing supervisor
```

处理。

不要建立：

```text
DeploymentWatchdogProcess
```

第二套 health infrastructure。

---

# 82. Command Silence Watchdog

Coordinator 记录：

```text
last_valid_policy_command_monotonic_ns
```

当：

```text
RUNNING
```

但长期没有任何有效 policy endpoint：

超过：

```text
max_command_silence_s
```

则：

```text
advance generation
RUNNING → ARMED
```

不能让 UI/状态保持：

```text
RUNNING
```

但 policy 已死锁。

---

# 83. Phase C10 — Lifecycle

新增：

```text
deployment/lifecycle.py
```

负责：

```text
resolve configuration
create SharedStorage
spawn arm
spawn hand optional
spawn camera
spawn inference
spawn coordinator
spawn recorder optional

readiness
ARMED
supervise
verified shutdown
```

---

# 84. Thin Policy CLI

新增：

```text
examples/run_policy.py
```

只包含：

```text
argparse
config override
resolve
run lifecycle
return exit code
```

禁止写：

```text
model code
CUDA code
scheduler
SafetyGate
SharedStorage business logic
```

---

# 85. Policy Workflow 默认不启动 VR

Learned policy 若只需要：

```text
camera
arm state
hand state
tactile
```

则不启动 Quest worker。

VR 不是所有 runtime workflow 的强依赖。

只有 adapter 明确声明需要 VR 时才启用。

---

# 86. Phase C11 — DexMani Policy Adapter

Fake Backend 完整通过后才能开始。

建议：

```text
dexmani_policy/
└── integrations/
    └── dexmani_real.py
```

如果不能修改 model repo：

```text
dexmani_real/
└── integrations/
    └── dexmani_policy.py
```

也可接受。

但：

```text
deployment/*
```

不得直接 import 该 integration。

---

# 87. DexMani Policy Backend

职责：

```text
load Hydra config
instantiate Agent
load checkpoint
load EMA if required
load normalizer
run predict_action
return model-native output
```

---

# 88. DexMani Observation Adapter

负责：

```text
ObservationBatch
        ↓
policy expected observation dictionary/tensors
```

可处理：

```text
RGB
point cloud
joint state
history stacking
normalization input format
batch dimension
device transfer
```

不同 DexMani Policy agent 的感知模态并不完全相同，例如 RGB+Joint 与 PointCloud+Joint 都存在，因此这一逻辑必须位于 adapter 而不是 deployment core。

---

# 89. DexMani Action Adapter

负责：

```text
model output
→ denormalize
→ native robot representation
→ arm [N,7]
→ optional hand [N,12]
→ JointActionChunk
```

---

# 90. Joint Policy 首版

Phase C 第一版只允许：

```text
native joint action
```

如果 checkpoint 是：

```text
EE action
```

且不存在已经验证的：

```text
EE → joint
```

转换：

```text
startup reject
```

不要把：

```text
Cartesian policy support
IK design
```

顺带塞进 C。

---

# 91. FAAS 等模型内部表示

如果 DexMani Policy 内部使用：

```text
FAAS
latent hand
其它 expanded representation
```

必须由模型仓库/adapter：

```text
convert back to native 12D XHand
```

Deployment core 永远只看到：

```text
native canonical joint action
```

---

# 92. Deployment Config 边界

DexMani Real 可以拥有：

```text
backend_target
observation_adapter_target
action_adapter_target

checkpoint
model_config_path
device

inference_hz

observation_horizon
max_observation_age_s
max_plan_age_s

max_command_silence_s
action_validity_s

hand_enabled
```

---

# 93. 不进入 Runtime Config

模型内部：

```text
transformer depth
hidden dimension
diffusion schedule
flow steps
point encoder
tokenizer
optimizer
training batch
EMA schedule
```

全部保留在 model repository。

---

# 94. Metrics

至少：

```text
observations_built

observation_age_ms
observation_skew_ms

inference_ms
inference_failures

plans_created
plans_superseded
plans_stale
plans_generation_dropped

plan_age_ms

endpoints_due
endpoints_coalesced
endpoints_published

safety_rejections
policy_aborts

command_silence_abort
```

先用：

```text
普通 counters + structured logging
```

不要引入 Prometheus/OpenTelemetry。

---

# 95. Recording 与 C

C 初始实施：

```text
不修改 HDF5 episode schema
```

先把 live deployment 做正确。

---

# 96. Deployment Provenance

至少日志记录：

```text
DexMani Real commit
model repo commit if known

backend target
observation adapter target
action adapter target

checkpoint path
checkpoint hash if available

resolved runtime config hash
model config hash
```

不要把完整 JSON config：

```text
塞进 SharedStorage 高频 payload
```

---

# 97. 如果以后需要 Policy Recording

单独开：

```text
C-Recording Migration
```

先判断：

```text
现有 action/sample 字段是否已经足够
```

如果仅缺：

```text
model provenance
```

优先增加 episode metadata。

如果真的有不兼容数据表示：

才 schema bump。

---

# 98. C Offline Tests

新增：

```text
check_deployment_contracts.py
check_policy_plan_dtype.py
check_causal_observation.py
check_fake_backend.py
check_inference_generation.py
check_plan_scheduler.py
check_candidate_publication.py
check_policy_failure_semantics.py
check_backend_swap.py
```

---

# 99. 必测故障矩阵

### Model

```text
backend import failure
backend load failure
checkpoint missing
inference exception
NaN
Inf
empty chunk
wrong dimension
unsupported output
```

### Observation

```text
camera missing
camera stale
arm stale
hand stale
history shorter than requested
future frame
causal mismatch
```

### Generation

```text
generation changes before inference
during inference
after plan publication
before endpoint due
```

### Scheduler

```text
fast model
slow model
many overdue steps
new plan supersedes old
old plan arrives late
```

### Runtime

```text
inference worker crash
coordinator crash
heartbeat timeout
arm queue pressure
hand disabled
SafetyGate rejection
```

全部：

```text
fail closed
```

---

# 100. Backend Swap 验收

至少：

```text
FakeBackend
+
DexManiPolicyBackend
```

或者：

```text
FakeBackendA
+
FakeBackendB
+
DexManiPolicyBackend
```

切换只允许修改：

```text
config
adapter implementation
checkpoint
environment
```

---

## 不允许修改

```text
robot/*
sensor/*
SafetyGate
SharedStorage fundamental semantics
deployment/coordinator.py
```

如果换模型需要：

```python
if backend == "dexmani_policy":
```

写进 coordinator：

Phase C 未完成。

---

# 101. C Hardware Gate H0 — No Command

真实硬件部署第一阶段：

```text
camera/state active
inference active
coordinator active
command publication disabled
```

观察：

```text
observation
model input
inference latency
plan
scheduled endpoint
candidate
SafetyGate result
```

不运动。

---

# 102. H1 — Connected Dry Run

连接：

```text
arm
hand
```

但：

```text
candidate publication dry-run
```

确认：

```text
planned command
SafetyGate command
expected transport
```

一致。

---

# 103. H2 — Arm Only Restricted

条件：

```text
hand disabled
small motion
restricted workspace
operator e-stop ready
```

首次真实 policy motion 只开 arm。

---

# 104. H3 — Arm + Hand

H2 稳定后启用 XHand。

检查：

```text
共同 action_id
hand latest-wins
无 command backlog
无 silent clip
```

---

# 105. H4 — Pause During Inference

必须真实测试：

```text
inference running
        ↓
pause
        ↓
generation++
        ↓
old CUDA inference returns
```

确认：

```text
old plan never executes
```

---

# 106. H5 — Fault / Worker Death

人为安全方式测试：

```text
terminate inference process
```

确认：

```text
supervisor
→ fault path
→ robot verified shutdown
```

不要用真实机器人制造危险 actuator fault 来测试。

---

# 107. H6 — Soak

最后才运行：

```text
长时 inference
plan replacement
pause/restart
arm+hand
recording optional
```

观察：

```text
memory
shared-memory leak
queue pressure
plan lag
CUDA memory growth
heartbeat
command silence
```

---

# 108. C Commit 建议

```text
C0  docs: freeze A/B runtime contracts

C1  refactor(shm): extract causal observation reader

C2  refactor(policy): add reusable candidate publication boundary

C3  feat(deployment): add backend/adapter contracts

C4  feat(deployment): add lazy backend loader and configuration

C5  feat(shm): add policy plan dtype and ring

C6  feat(deployment): add deterministic fake backend

C7  feat(deployment): add inference worker

C8  feat(deployment): add coordinator and endpoint scheduler

C9  feat(deployment): add lifecycle and thin CLI

C10 test(deployment): add failure and generation regression checks

C11 feat(integration): add DexMani Policy adapter

C12 test(deployment): verify backend replacement

C13 docs: document deployment runtime and hardware gates
```

---

# 109. C 架构 Reject List

Review 发现以下任一项：

```text
Inference Worker writes arm queue

Inference Worker writes hand ring

Inference Worker owns SafetyState

Inference Worker imports robot SDK

model adapter owns SharedStorage

model-specific branch in coordinator

Torch imported by core package at import time

whole action chunk dumped into robot transport

software arm interpolation

parallel process watchdog

parallel recording framework

new global plugin registry

JSON/object dtype in high-frequency IPC

XHand lifecycle modified by policy deployment
```

直接：

```text
REJECT
```

---

# 110. 完整执行顺序

严格按：

```text
A0  Offline harness

A1  C24
A2  live error
A3  homing live error
A4  naming
A5  TCP load
A6  XHand local-error naming
A7  remove XHand.stop

A8  actuator finally cleanup

A9  ring commit
A10 ring history

A11 coupled ACK
A12 hand delta audit

A13 VR recovery audit

A14 perception resolved calibration
A15 pointcloud metadata audit
A16 calibration atomic publication

A17 recording hold/send semantics
A18 schema compatibility
A19 SafetyGate regression
A20 Mode-6 regression

──────── A GATE ────────

B0  hardware baseline

B1  single-controller discovery/open
    → patch
    → fake
    → human soak
    → decision

B2  close-only disconnect
    → patch
    → fake
    → human soak
    → decision

B3  lifecycle cleanup
B4  topology reduction
B5  dead config cleanup
B6  optional domain split

──────── B GATE ────────

A/B FREEZE

C0  deployment contract freeze
C1  causal reader
C2  candidate publisher
C3  backend/adapter protocols
C4  loader/config
C5  plan dtype/ring
C6  FakeBackend
C7  inference worker
C8  coordinator/scheduler
C9  failure semantics
C10 lifecycle/CLI
C11 DexMani Policy adapter
C12 backend swap
C13 metrics/provenance

──────── C OFFLINE GATE ────────

H0 no command
H1 connected dry run
H2 arm-only
H3 arm+hand
H4 pause/in-flight inference
H5 worker death
H6 soak
```

---

# 111. 每个 Subphase 强制报告格式

代码代理完成一个 subphase 后必须输出：

```text
## Phase
A9

## Goal
修复 ring commit publication contract

## Base SHA
...

## Files read
...

## Files changed
...

## Exact behavior before
...

## Exact behavior after
...

## Contract preserved
...

## Offline commands executed
...

## Results
PASS/FAIL

## Hardware test
NOT RUN / MANUAL PASS / MANUAL FAIL

## Risks
...

## Deferred
...

## Diff audit
是否存在 unrelated diff
```

---

# 112. 禁止“一次性完成 A”

不要提示 Claude Code：

```text
“把 Phase A 全部做掉”
```

然后接受一个巨大 diff。

正确：

```text
A1
→ review
→ commit

A2/A3
→ review
→ commit

...
```

特别是：

```text
ring
recording
robot lifecycle
```

不能混进同一个巨大 commit。

---

# 113. 禁止自行声称 B 完成

B1/B2 必须经过真实硬件。

Claude Code 最多可以说：

```text
B1 patch prepared
offline checks PASS
awaiting hardware result
```

不能说：

```text
B1 complete
```

除非真人硬件 checklist 有结果。

---

# 114. A/B/C 全局不变量

整个整改期间必须持续保持：

```text
SharedStorage
=
唯一跨进程数据平面

SDK object
=
device-owner local

arm action
=
ordered bounded transport

hand action
=
latest-wins transport

cross-process payload
=
fixed NumPy dtype

recording
=
fixed control grid

control clock
=
monotonic

SafetyGate
=
单一机器人 action safety boundary

run_generation
=
旧 action / plan cancellation token

Mode 6
=
arm realtime smoothing owner
```

---

# 115. 最终系统边界

最终：

```text
                 Sensors
                    │
                    ▼
              SharedStorage
                    │
         ┌──────────┴──────────┐
         │                     │
         ▼                     ▼
      Teleop               Inference
         │                     │
         │                 Policy Plan
         │                     │
         │                     ▼
         │                Coordinator
         │                     │
         └──────────┬──────────┘
                    ▼
              ActionCandidate
                    ▼
                SafetyGate
                    ▼
             Command Transport
               │           │
               ▼           ▼
             xArm        XHand


           fixed-grid sample
                    │
                    ▼
               RecorderIO
                    │
                    ▼
                  HDF5
```

---

# 116. 最终 Definition of Done — Phase A

```text
[ ] A0 offline harness 完成

[ ] C24 recovery SDK contract 正确
[ ] live controller error 用于控制决策
[ ] homing restore fail-closed
[ ] arm readiness naming 清晰
[ ] TCP load runtime-configured

[ ] XHand local-error API 语义正确
[ ] dead/dangerous XHand stop API 处理完成

[ ] arm cleanup exception-safe
[ ] hand cleanup exception-safe

[ ] ring sequence 只代表 committed frame
[ ] history reader 不因一个 overwritten slot 丢全部历史
[ ] publish timestamp ownership 正确

[ ] coupled arm+hand ACK 正确
[ ] hand supersede fail-fast
[ ] hand command delta whole-command reject

[ ] VR spike recovery 正确

[ ] perception 只有一个 resolved table-plane source
[ ] required alignment fail-fast
[ ] pointcloud metadata 正确
[ ] calibration atomic

[ ] synthetic grid hold 不伪造 send event
[ ] replay 不重发 synthetic no-send slot

[ ] schema 未发生无理由 bump
[ ] SafetyGate 未膨胀
[ ] Mode 6 smoothing contract 未改变
```

---

# 117. 最终 Definition of Done — Phase B

```text
[ ] baseline reconnect soak 有记录

[ ] B1 patch 已真实硬件 A/B
[ ] B1 merge/revert 有实验依据

[ ] B2 patch 已真实硬件 A/B
[ ] B2 merge/revert 有实验依据

[ ] 最终 connect lifecycle 被文档化
[ ] 最终 disconnect lifecycle 被文档化

[ ] 无未经实验支持的 XHand workaround

[ ] topology reduction 行为不变
[ ] dead config cleanup 有 rg evidence
[ ] 没有大型 manager/registry/framework
```

---

# 118. 最终 Definition of Done — Phase C

```text
[ ] inference 是独立 worker

[ ] model / observation / action adapter 分离

[ ] inference 不直接写机器人 command

[ ] model core 不持有 SharedStorage

[ ] model lazy load in child

[ ] deployment core 无特定 model branch

[ ] policy plan 是 fixed IPC dtype

[ ] policy plan latest-wins

[ ] generation 能取消 in-flight inference

[ ] coordinator 做 endpoint scheduling

[ ] chunk 不进入 arm queue

[ ] hand chunk 不 dump 到 latest-wins ring

[ ] 无 application-side interpolation

[ ] 每个真实 endpoint → ActionCandidate

[ ] 每个实际动作 → SafetyGate

[ ] invalid policy output fail closed

[ ] inference process failure → existing supervisor

[ ] FakeBackend 完整端到端 PASS

[ ] DexManiPolicyAdapter PASS

[ ] backend swap 不修改 runtime core

[ ] hardware H0-H6 分级完成
```

---

# 119. 整个项目最终验收

只有：

```text
A PASS
+
B PASS
+
C PASS
```

才能声称此次整改完成。

最终不是追求：

```text
“代码更统一”
```

而是获得五个稳定边界：

```text
1. Hardware Boundary
2. Process / IPC Boundary
3. Safety Boundary
4. Recording Boundary
5. Learned-Policy Boundary
```

其关系为：

```text
Model 可替换

Control Source 可替换

Robot Runtime 不随模型变化

Safety Runtime 不随模型变化

Device Worker 不随模型变化

Recording Core 不随模型变化
```

这才是 DexMani Real 从研究脚本演化为稳定具身智能真实机器人 runtime 的最终目标。