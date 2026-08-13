# DexMani Real — 真机验证流程（简化重构后）

> **终止门文档**。本文件是 `simplify-phase1` 分支（Phase 1 + Phase 2 零行为变化重构）
> 可被信任前的真机关卡，是 `docs/execution-plan.md` §1「8 项检查清单」的**逐步可执行版**。
>
> **硬件授权门**：CLAUDE.md 规定「Do not run examples/ without explicit hardware
> authorization」。未获显式授权前，本文档任何一步都**不得执行**；这是纯计划。

## 0. 范围与顺序

顺序设计为：先证明两个回归提交（`7ce4813` 实测保持、`7c908f6` homing 返回 + 手部校验），
再走 Phase 1 广度，最后用一次完整 collect → visualize → replay 往返覆盖 Phase 2
共享内存面（seqlock / 心跳 ready 合并）。

| 步 | execution-plan §1 检查项 | 提交 / 阶段 |
|---|---|---|
| 1 | #1 实测保持 apply + 重建锚 | `7ce4813` |
| 2 | #5 homing 失败路径 | `7c908f6` |
| 3 | #7 手部命令校验 | `7c908f6` |
| 4 | #2 Mode-6 跟踪 | Phase 1 |
| 5 | #3 碰撞恢复 | Phase 1 |
| 6 | #4 homing 收敛 | Phase 1 |
| 7 | #6 手部接触/回程/稳态 + hand-home ACK | Phase 1 |
| 8–10 | #8 录制往返 | Phase 2 |

## 0.1 前提与通用中止

- **前提**：操作者已获硬件授权；xArm7 + XHand 上电并连接（arm IP 见
  `dexmani_real/config/defaults.py`）；Quest VR 经 TCP 8000 可达（USB 时
  `adb reverse tcp:8000 tcp:8000`）；`camera.json` / `vr_transform.json` 存在且有效。
  所有命令从仓库根目录、conda `real_robot`、`PYTHONPATH=.` 运行。两名操作者：
  一人在机器人旁（可触物理急停），一人在键盘/头显。
- **通用中止**：物理急停 / ESC 是每步的兜底——任何非预期运动、任何非刻意触发的
  `safety → FAULT(3)`、任何非预期的 `arm_loop` 错误、任何 worker 死亡 → 按 ESC，
  若臂未停则物理断电。中止后必须重新 homing 才能继续。
- **日志字符串是「预期签名」而非精确正则**：下方列出的日志行是「要看到什么」的
  提示，实际操作以真实日志为准；关键判断点是**是否出现 `delivery timed out`、
  `TypeError`、`safety → FAULT`、episode 被丢弃**这几类失败签名。

---

## Step 0 — 预检 + 已知良好基线

**(a) 运行**
```bash
conda run -n real_robot python -m compileall -q dexmani_real examples
conda run -n real_robot python -m pytest tests/ -q          # 预期 20 passed，无硬件
```

**(b) 前置**：`simplify-phase1` 工作树 clean；臂周围无遮挡；XHand 张开/中立；Quest 未戴。

**(c) 通过**：`20 passed`，零编译错误。

**(d) 中止**：任何 pytest 失败 → 停，不上真机；对照 `tests/README.md` harness 映射排查。

**(e) 去风险**：离线回归门（Phase 1 + Phase 2 seqlock/harness 全部测试）。

随后建立已知良好基线（后续每步的前置）：

**(a) 运行**
```bash
conda run -n real_robot python examples/keyboard_teleop.py
```

**(b) 前置**：手已装；无需录制/相机/VR。

**(c) 通过**：启动日志 `arm: ready`、`hand: ready`；`_read_initial_arm` 成功；按 **R**
臂经 Mode 0 回 home（预期 `arm_loop: homing entering Mode 0 …`、`arm_loop: HOME
complete …`、`arm: home reached`、`arm_loop: homing restored Mode 6`，随后
`hand: home command accepted (action_id=…, milestones=…)`）。按 **Q** 干净退出
（`exit_code 0`，`safety … clean=True`）。

**(d) 中止**：静止时任何 `C22/C24/C31`、任何 homing 失败、任何 `safety → FAULT`。

**(e) 去风险**：确认「重构前」基线健康，后续失败才能归因于重构而非硬件。

---

## Step 1 — 实测保持 apply + 重建锚（`7ce4813`）

唯一的净行为变更。原 bug：`hold_sent_at_s` 从未写入，`ControlHold.observe_delivery`
永不报 `applied`，所有实测保持（PAUSE/STOP/QUIT/vr_stale/hand_recovered）在 0.75 s
apply 超时处 fault，`_complete_reanchor` 永不触发。

**(a) 运行**
```bash
conda run -n real_robot python examples/collect_teleop.py --no-record
```

**(b) 前置**：Step 0 已 homing；Quest 戴上、VR 已连（打印 `VR connected`）；`--no-record`
跳过相机/RecorderIO，循环只含 teleop + arm + hand + vr。

**(c) 通过**（按序日志）：
1. 按 **B** → `B: …`、`safety: ARMED(1) → RUNNING(2)`。
2. 摆动手腕，臂跟随。
3. 按 **C**（暂停）→ `pause hold published` 后 **`pause hold applied (action_id=…)`**
   （**不是** `pause hold delivery timed out`），`safety: RUNNING(2) → ARMED(1)`。
4. 再按 **C**（恢复）→ `completed resume re-anchor`（或 `released … after fresh
   re-anchor`），`safety: ARMED(1) → RUNNING(2)`，臂再次跟随。
5. 重复暂停→恢复 3 次；再测 **vr_stale**：RUNNING 中摘头显 → `vr_stale hold
   published` 后 `vr_stale hold applied`；重新戴回 → `released vr_stale hold after
   fresh re-anchor`，运动恢复。
6. 按 **Q** → `quit hold published` → `quit hold applied`；再按 **Q** 干净退出。

**(d) 中止**：任何保持出现 `delivery timed out`；`safety → FAULT(3)`；暂停后臂仍动。
任一 = `7ce4813` 回归 → 停，检查 `teleop/loop.py::_enter_measured_hold_impl` +
`TeleopLoopState.hold_sent_at_s`。

**(e) 去风险**：`7ce4813`（实测保持 apply + 重建锚）——整个重构唯一的净行为变更。

---

## Step 2 — homing 失败路径（多里程碑，`7c908f6`）

原 bug：`_execute_mode0_milestones_impl` 失败返回裸 `HomeResult`、成功返回
`(HomeResult, ndarray)` 元组，任何多里程碑失败都抛 `TypeError` 杀 arm worker。

**(a) 运行**
```bash
conda run -n real_robot python examples/keyboard_teleop.py
```

**(b) 前置**：臂**不在** home——先用 WASD/方向键移开，使 return-home 需多个 Mode-0
里程碑（更长的路径，最好触发两段 wrapped+band-alignment 回退，即多里程碑路径）。手已 homing。

**(c) 通过**：
1. 按 **R** 开始 homing；观察 `arm: home path selected=… milestones=N` 且 `N ≥ 2`，
   以及 `arm_loop: homing entering Mode 0 MoveJoint (N motion milestones, …)`。
2. 运动中按 **ESC**（刻意、安全）。预期 worker 存活：**`arm_loop: HOME failed —
   e-stop requested`**（或 `… shutdown requested`），控制器打印 `arm: home failed —
   e-stop requested`（或 `arm: home wait aborted — e-stop requested by operator`），
   **arm worker 不崩溃**——`arm_loop` 继续发布状态/心跳；无 `TypeError: cannot
   unpack …` traceback。
3. 按文档恢复流程清除 estop/fault，随后重跑并完成一次正常 home（Step 6）确认 worker 仍可用。

替代（若运动中 ESC 不便）：用物理急停按钮，走同一 `_shared_abort_reason_impl →
e-stop requested` 失败返回。

**(d) 中止**：`arm_loop` 任何 `TypeError` traceback、keyboard teleop 报 `worker
exited`、或 ESC 后臂未停。`TypeError` = `7c908f6` 回归。

**(e) 去风险**：`7c908f6` homing 返回类型统一；真实 homing 失败下的 worker 抗崩溃。

---

## Step 3 — 控制器侧手部限位 + 命令间增量优雅 hold（`7c908f6`）

原 bug：Phase 1.2 误删了 `_sanitize_hand_command` 的控制器侧手部关节限位与
命令间增量检查（`7c908f6` 恢复）。可观察：越界手部命令应**优雅 hold**（臂+手同时）
而非粘滞 fault 或臂手失步。

**(a) 运行**（用「收窄」配置覆盖确定性触发——运行时只许收窄、不许放宽包络）：
```bash
conda run -n real_robot python examples/collect_teleop.py --no-record --config /tmp/hand_tight.yaml
```
其中 `/tmp/hand_tight.yaml` 设 `hand.max_delta_rad: 0.02`（默认 `0.20`）。

**(b) 前置**：臂+手已 homing；VR 已连；手部 retargeting 激活。极小的增量界保证正常
手部动作必然超界。

**(c) 通过**：
1. 按 **B**，teleop RUNNING。
2. 快速张合手。预期节流日志 `invalid hand command — holding: hand command
   violates command-to-command delta limit`（或 `… joint limits`）。
3. 观察那些帧内臂**不动**且**无 `safety → FAULT`**——循环臂+手同时 hold
   （`retarget_ok=False`、`hand_cmd_valid=False`），保持 RUNNING，放慢手速后恢复
   正常（受限）手部运动。
4. 确认 `arm_action_q` 深度有界（状态行 `q=…`），无端点洪泛。

同会话稍后（Step 7）跑一次默认配置的自然检查，确认该检查对合法运动**不误报**。

**(d) 中止**：手部限位违规触发任何 `safety → FAULT(3)`（这正是 `7c908f6` 要回归掉
的「SafetyGate 粘滞 fault」行为），或手部命令被拒时臂仍在动（「失步」行为）。

**(e) 去风险**：`7c908f6` 控制器侧手部检查；优雅 hold vs 粘滞 fault 语义。

---

## Step 4 — Mode-6 跟踪（Phase 1 广度）

**(a) 运行**
```bash
conda run -n real_robot python examples/keyboard_teleop.py
```
（仅臂即可，手可选。）

**(b) 前置**：臂已 homing；工作区无遮挡；碰撞灵敏度为配置档。

**(c) 通过**：快速腕旋（yaw/roll，J/K/L 与 ←/→ 键）约 30 s。臂跟踪**无**
`arm_loop: tracking_err=…_rad threshold=…_rad` 告警尖峰（告警需连续 3 帧超阈且
5 s 节流，孤立单帧噪声不算）。无 C22/C24/C31。固件仍是唯一速度/加速度兜底。

**(d) 中止**：反复 `tracking_err` 告警（IK 命令与实测发散）；任何非刻意的
C24/C22/C31。

**(e) 去风险**：Phase 1 — Mode-6 伺服路径未变；验证删除速度包络未重新引入跟踪发散。

---

## Step 5 — 碰撞恢复（Phase 1 广度）

**(a) 运行**
```bash
conda run -n real_robot python examples/keyboard_teleop.py
```

**(b) 前置**：臂已 homing；慢速移动中在腕部路径放柔软轻物（顺性泡沫/操作者护好的
前臂）。紧握急停。

**5a — C24（可恢复）**
**(c) 通过**：慢速移动中接触诱发 **C24**。观察：不锁 fault，`error_code` 回 0，臂做
有界实测保持后继续；`_recover_c24_measured_hold` 成功路径静默运行。除非 2 s 内重复
接触，否则**不得**出现 `arm_loop: second C24 inside 2s — latching fault`。
**(d) 中止**：任何非刻意快速重复接触导致的 `second C24 inside 2s`（锁存）；臂未停稳。

**5b — C22/C31（碰撞致命）**
**(c) 通过**：更强碰撞诱发 C22 或 C31。预期 **`arm_loop: collision fault C31
detected; …`**（或 C22），随后 `shared.error_state.value = True` →
`safety: … → FAULT(3)`，臂减速安全停。worker 存活；Main 监督粘滞 fault。按协议
断电/重 homing 恢复。
**(d) 中止**：fault 已触发但臂仍驱动（固件兜底失效）→ 立即物理急停。

**(e) 去风险**：Phase 1 — C24 有界恢复与 C22/C31 粘滞 fault 语义；确认
`recoverable_errors={24}` / `collision_fault_errors={22,31}` 路由在重构后仍在。

---

## Step 6 — homing 收敛（Phase 1 广度）

**(a) 运行**
```bash
conda run -n real_robot python examples/keyboard_teleop.py
```

**(b) 前置**：臂移到非 home 位姿（含一个强制 wrapped+band-alignment 两段路径的位姿，
如 J7 相对 band 转 >180°）。

**(c) 通过**：按 **R**。按序预期：`arm_loop: homing entering Mode 0 MoveJoint (N
motion milestones, speed=…)` → 逐里程碑推进（位置+速度静置）→ `arm_loop: HOME
complete …` → `arm: home reached` → **`arm_loop: homing restored Mode 6`**。臂在
规范 home 静止（位置与速度分别在 `arm.homing.convergence_rad`（默认 ~0.15°）与
`arm.homing.velocity_convergence_rad_s`（默认 0.03 rad/s）内）。若走 wrapped 路径：
`arm: canonical home path rejected — falling back to wrapped+alignment …` 后
`arm: band-alignment appended (… milestones, …)`、`arm: home reached`。

**(d) 中止**：`arm: home failed — …`（收敛/整体超时）、`arm: no validated home-path
candidate — holding`、或 homing 未恢复 Mode 6。

**(e) 去风险**：Phase 1 — Mode-0 里程碑 homing、静置收敛、Mode-6 恢复；验证
`plan_joint_home_path` / `plan_band_alignment_path` 碰撞安全路径仍落到规范 home。

---

## Step 7 — 手部接触/回程/稳态 + hand-home ACK（Phase 1 广度）

**(a) 运行**
```bash
conda run -n real_robot python examples/collect_teleop.py --no-record
```

**(b) 前置**：手已装；VR 已连；进入 teleop（**B**）。

**(c) 通过**：
1. **稳态/回程**：闭手握物或全闭；`qpos` 不变或未收敛到请求角度应作为**合法执行
   结果**、而非 freshness fault——确认手健康
   （`connected/state_valid/send_healthy/read_healthy` 全真）时**无**
   `hand feedback unhealthy` / `… error_state` 触发。v16 `qpos_stale` 位保持
   reserved-false。
2. **接触**：刻意接触不产生手部 freshness fault；仅当 VR/手反馈**真的** stale 时
   teleop 才继续或优雅 hold。
3. **触觉复位**：确认接触时触觉值非零，且上电 `_reset_tactile_sensors` 未留下
   全零 ring/little 列（对应 memory 记录）。
4. **Hand-home ACK**：按 **H**（或退出后 H）→ 臂 home 前先见 **`hand: home command
   accepted (action_id=…, milestones=…)`**；随后臂 home 完成。若 hand-home 被拒，
   预期 `Return-home cancelled: hand-home command was not accepted` 且臂**不动**。

**(d) 中止**：健康接触/稳态期间任何手部 freshness fault（正是「手部反馈边界检查移除」
要消除的 bug 类）；任何触觉解析出原始 ADC 量级值（缺 ÷10）。

**(e) 去风险**：Phase 1 — 手部接触/回程/稳态语义与 hand-home 里程碑 ACK
（`publish_hand_home_and_wait_applied`）。

---

## Step 8 — 完整录制往返（Phase 2 共享内存面）

同时跨所有 worker 锻炼 Phase 2.1 `SeqlockSlot`（arm_state_ring、hand_state_ring、
camera_ring、vr_ring、控制 ring）与 Phase 2.2 心跳/ready 结构化数组。

**(a) 运行**
```bash
conda run -n real_robot python examples/collect_teleop.py --task phase2_smoke --operator op
```

**(b) 前置**：全能力——arm + hand + VR + camera + RecorderIO 全开（无 `--no-hand`、
无 `--no-record`）。相机视野清楚。

**(c) 通过**：
1. 启动：**所有** readiness 触发（`arm: ready`、`hand: ready`、`VR connected`、
   `camera: ready`、`recorder: ready`），`All subsystems ready — safety=ARMED(1)`。
   合并后的心跳数组**无**假 `heartbeat is missing or stale`。
2. 按 **B** → teleop+record RUNNING；执行 ~20–30 s 代表性任务（平移+旋转+手抓）
   且 `录制` 状态活跃。
3. 按 **S**（停止/保存）→ `已保存` / `录制已保存: episodes/<episode> (<N> 帧)`
   （N ≈ 16 Hz × 时长）。会话 `clean=True` 结束。
4. 记录 episode 路径与帧数，供 Step 9–10。

**(d) 中止**：任何 `camera frame is unhealthy or stale`、recorder
`stream mismatch/overflow/ENOSPC`（episode 被丢弃而非部分发布）、任何 worker
心跳超时 fault、或 `safety → FAULT`。episode 被丢弃 = Phase-2 seqlock/心跳回归待查。

**(e) 去风险**：Phase 2 — seqlock 去重（teleop 循环读所有 ring）与心跳/ready 合并
（supervisor 看 7→1 结构化心跳、7→1 ready 标志）在全活负载下。

---

## Step 9 — 可视化录制 episode

**(a) 运行**
```bash
conda run -n real_robot python examples/visualize_episode.py episodes/<episode> --info
conda run -n real_robot python examples/visualize_episode.py episodes/<episode>
```

**(b) 前置**：离线（无硬件）；episode 路径来自 Step 8。

**(c) 通过**：`--info` 打印合法 v16 结构（`/meta` 有 depth_scale，`num_frames` 与
Step 8 一致）；Rerun 窗口加载并回放臂轨迹、手位姿、相机点云/深度，无 NaN 空洞或
shape 错误。确认 camera_ring 与 arm_state_ring 载荷经 Phase-2 重构后完好序列化。

**(d) 中止**：`schema-v16 episode is missing /meta depth_scale` 或任何数据集 shape 不匹配。

**(e) 去风险**：Phase 2 — camera ring（CameraRingBuffer / SeqlockSlot）与 arm-state
ring 完整性（持久化到 HDF5 v16）。

---

## Step 10 — 重放 dry-run（默认）与 live 重放

**(a) 运行**
```bash
# 离线 dry-run（无硬件）
conda run -n real_robot python examples/replay_episode.py episodes/<episode> --dry-run
# 精确提交流 dry-run
conda run -n real_robot python examples/replay_episode.py episodes/<episode> --source sent --dry-run

# 可选 live 重放（硬件）——密集预检后真正重新驱动
conda run -n real_robot python examples/replay_episode.py episodes/<episode> --source sent --live \
    --max-frames 200 --output replay_results/phase2_smoke/
```

**(b) 前置（dry-run）**：无（离线）。**(live)**：臂+手物理处于录制首帧位姿 5° 内
（`START_POSE_TOLERANCE_DEG`），手在位（episode `hand_available=true`）或仅在物理手
已固定时用 `--no-hand`；运行时配置匹配录制 provenance（`joint_max_acc`、
`joint_max_speed`、`arm_loop_hz`、模型 SHA-256 不变）。

**(c) 通过**：
- Dry-run：`Dry-run complete: validated <N> frames at nominal … Hz` 且无 shape/有限性错误。
- Live：密集预检通过（`Replay safety gate ready`），起始位姿对齐 5° 内，随后
  `Replay: 200 frames @ … Hz` 跑到 `Replay completed.`，打印一致性指标（`Arm joint
  MAE/RMSE`、`EEF pos error`、`Tracking lag`），且 `Replay data saved` / `Metrics
  saved`。重放后提示处按 **H** 使臂回 home（`arm: home reached`）。

**(d) 中止**：任何预检拒绝（fail-closed provenance/geometry）、重放中任何 `arm
controller error C22/C31`、跟踪发散、或 `Q`/`ESC` 未能建立实测保持。live 重放可选
——若前面任一步失败，勿跑 live。

**(e) 去风险**：Phase 2 — arm-state ring 读路径（`read_arm_state_dict`、
`arm_state_ring.read_latest`）与 hand ring 经第二个完整生产者/消费者循环；同时再次
锻炼整个重构都触碰过的 SafetyGate + `send_command` → worker apply 路径。

---

## 完成门

Steps 1–10 全过（Step 5b 与 Step 10-live 条件必做，其余强制）才算 `simplify-phase1`
可信任。任一失败映射到首要怀疑对象：

- `hold delivery timed out` 或重建锚失败 → **`7ce4813`**（`teleop/loop.py`
  `_enter_measured_hold_impl` / `TeleopLoopState.hold_sent_at_s`）。
- homing 失败 `TypeError` / `HOME failed` 且 worker 死亡 → **`7c908f6`** homing 返回
  类型（`robot/arm_loop.py::_execute_mode0_milestones_impl`）。
- 手部限位/增量粘滞 fault 或臂手失步 → **`7c908f6`** `_sanitize_hand_command`
  （`teleop/hand_control.py`）。
- episode 被丢弃 / 心跳超时 fault / ring 读 shape 错误 → Phase-2 提取
  （`shm/ring_buffer.py::SeqlockSlot`、`shm/shared_storage.py` 心跳/ready 合并）。

**明确未纳入本流程**：`examples/deploy_policy.py`（学习策略推理 ring）是独立实验
能力，未受 Phase-1/Phase-2 简化锻炼；应在策略部署前用
`hardware_deployable: true` 的 PolicySpec 单独验证。
