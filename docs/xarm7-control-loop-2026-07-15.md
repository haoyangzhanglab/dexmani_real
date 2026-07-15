# xArm7 控制回路 — 完整架构文档

> **用途**：本文档自包含地描述 DexMani Real 项目中 xArm7（7-DOF 机械臂）的全部控制路径，
> 供没有代码库访问权的读者（人类或 LLM）理解系统行为。所有结论基于 2026-07-15 的代码实测核查
> （6 个子系统深读 + 4 项对抗式 fact-check），并已包含当日落地的 5 个 bug 修复。
> **行号仅供参考**（标"约"处已因当日修改发生偏移）；以函数/方法名为准。

**系统背景**：xArm7（UFACTORY 7 自由度机械臂）+ XHand（12 自由度灵巧手）做 VR 遥操作数据采集。
Python 3.10，主控制频率 50Hz。机械臂通过 UFACTORY Python SDK（`xarm.wrapper.XArmAPI`，TCP 连接
控制器）驱动。本文档只覆盖 **arm** 的控制回路；hand 仅在数据流交汇处提及。

---

## 1. 总览：一台机械臂、两条连接、三种控制路径

同一台 xArm 控制器上存在**两条独立的 XArmAPI TCP 连接**，服务互斥的控制路径：

| 路径 | 连接归属 | 固件模式 | 用途 | 文件 |
|---|---|---|---|---|
| ① 遥操作实时环 | `ArmInnerLoop` 线程自建 | **Mode 6**（关节在线轨迹规划） | 50Hz VR 遥操作 | `dexmani_real/robot/inner_loop.py` |
| ② 阻塞式移动 | `XArm7` 驱动自建 | **Mode 1**（servo-j 逐点）/ **Mode 0**（阻塞插值） | return-to-home、规划路径执行 | `dexmani_real/robot/xarm7/xarm7.py` |
| ③ 急停 | `XArm7` 连接 | `set_state(4)` | EMERGENCY_STOP | 同上 |

**关键约束**：固件的 mode/state 是**控制器全局状态**，与哪条 TCP 连接发指令无关。两条连接同时活动
会互相切模式导致对方指令失败。代码层**没有互锁**，靠调用时序保证互斥：

- 归位（H 键）前必须先停内环（`TeleopController._do_home`），释放 Mode 6；
- 归位后内环**不自动重启**，只在下一次按 B 时经 `_ensure_inner_running` 惰性重建；
- `XArm7.clear_error()` **刻意不改控制模式**，避免在内环运行时从下面切走 Mode 6。

### 1.1 端到端架构图（遥操作主路径）

```
VR 头显 (Quest) ──TCP──> VRReceiverProcess（独立进程）──SharedMemory──┐
                                                                    ▼
┌─ 外环: TeleopController 主线程 @50Hz ──────────────────────────────────────────┐
│  (teleop/core/controller.py, RateManager 混合忙等限速, 误差 <1ms)               │
│                                                                              │
│  ① 读 VR 帧(SHM) → 陈旧检查(>0.5s→hold; 累计≥3s→停录回IDLE)                     │
│  ② arm_qpos, err, ts = _arm_inner.get_state()   err=True → 升级急停            │
│  ③ state = robot.get_state(arm_qpos=内环值)      注意: 此路径 arm_qvel/tau=NaN  │
│  ④ pipeline.compute_action():                                                │
│       VR wrist → ArmWristMapper(相对增量映射)                                  │
│       → workspace clamp → 笛卡尔 EMA(IK之前、arm唯一平滑级)                     │
│       → solve_teleop_ik(两级: 差分IK → MPlib回退)                              │
│       任一环节失败 → held: 返回上一条成功命令                                    │
│  ⑤ validate_action() 预发送门(8项)  error/断连→急停; 其余失败→hold替换          │
│  ⑥ 分流下发:  arm  → _arm_inner.set_target(arm_qpos_cmd)                      │
│              hand → robot.send_action(action)  (仅手部!)                      │
│  ⑦ 录制已验证的实发帧 (flag_held = !ik_ok ∨ !retarget_ok ∨ validate_failed)    │
└──────────────────────────────────────────────────────────────────────────────┘
                    │ set_target: Lock 保护的 latest-wins 单槽信箱（无队列）
                    ▼
┌─ 内环: ArmInnerLoop daemon 线程 @50Hz (robot/inner_loop.py) ───────────────────┐
│  1. RateManager 限频 (loop_period=0.02s)                                      │
│  2. 超时守卫: target 缺失/超龄 0.2s → 复位软启动斜坡 + hold(读当前位置回发)       │
│  3. NaN 守卫: 非有限目标 → 回发 last_valid_qpos                                │
│  4. get_joint_states 读实测 + 双重错误检查(code!=0 ∨ error_code!=0 → 停环)      │
│  5. 被动监视: A4 跟踪误差>0.35rad 告警 / A5 mode≠6 漂移告警(独立节流)            │
│  6. _send_target: 每步 L∞ Δclamp ±0.3rad → 软启动斜坡(前20帧)                  │
│       → arm.set_servo_angle(wait=False, speed, mvacc)   ← 纯透传               │
└──────── 固件 Mode 6 在线轨迹重规划(限速 90°/s, 限加速 500°/s²) ── xArm7 硬件 ────┘
```

**Mode 6 语义**（核心设计）：固件自己做在线轨迹重规划并**强制遵守** speed/acc 限制。内环不做
插值、不做 PID——只是"透传 + 安全守卫"。这实际上消解了传统内外环的区别（历史上曾有 Mode 4
速度伺服 + 250Hz PID 方案，注释残留见 §9）。Mode 6 需固件 ≥ 1.10.0（启动时校验，低版本仅警告）。

---

## 2. 关键组件与文件地图

| 组件 | 文件 | 职责 |
|---|---|---|
| `TeleopController` | `teleop/core/controller.py` | 外环状态机 + 50Hz 主循环 + 录制生命周期 + 安全升级 |
| `TeleopPipeline` | `teleop/core/pipeline.py` | 单帧命令生成：映射 → clamp → EMA → IK |
| `ArmWristMapper` | `teleop/vr/arm_mapper.py` | VR 腕部 → 目标 EEF 位姿（reset-relative 增量映射） |
| `TeleopIKSolver` | `planning/ik.py` | 两级 IK（差分 DLS 主路径 + MPlib 回退） |
| `ArmInnerLoop` | `robot/inner_loop.py` | 50Hz 内环线程，独占一条 XArmAPI，Mode 6 透传 |
| `RobotInterface` | `robot/interface.py` | 硬件统一入口：手部下发、阻塞归位、状态聚合、急停 |
| `validate_action` | `robot/validate.py` | 预发送安全门（**模块级自由函数**，非方法） |
| `XArm7` | `robot/xarm7/xarm7.py` | SDK 薄封装：连接、阻塞移动、限位、错误锁存 |
| `XArm7MotionPlanner` | `planning/planner.py` | 离线规划（screw → 多 RRT，9 项验证链） |
| `RateManager` | `utils/rate_manager.py` | 高精度限速（sleep + 忙等混合，<1ms 误差；含 `reset()`） |
| 入口脚本 | `examples/real/vr_teleop_shm.py` | 组装全部组件 + preflight + VR 首帧等待(120s) |

**持有关系**（易误解处）：`ArmInnerLoop` 由 **TeleopController** 创建并持有，`RobotInterface`
不引用它。`RobotInterface.send_action()` **只发手部命令**（docstring 已注明 arm 由内环处理）。

---

## 3. 外环：TeleopController

### 3.1 状态机

```
IDLE ──B(开始遥操+录制)──→ TELEOP ⇄ C(暂停/恢复) ⇄ PAUSED
  ↑                          │
  └─── S(停止+保存) / H(归位) ┘
  ESC / 内环错误 / validate致命失败 → EMERGENCY_STOP（仅 Q 退出或 H 恢复）
```

- 状态只有 4 个：`IDLE / TELEOP / PAUSED / EMERGENCY_STOP`。**recording 是 bool 标志不是状态**，
  随 B 开始、S 保存、Q/H/VR 超时丢弃。
- **注意**：模块头注释与 CLAUDE.md 写 "VR-disconnect timeout → EMERGENCY_STOP"，**实现不是**——
  VR 累计断连 ≥3.0s 时停录（不保存）并回 IDLE，不进急停（见 §9 差异表 #3）。

### 3.2 每 tick 流程（50Hz，单线程）

主循环：`while running: _handle_keyboard() → _tick() → limiter.wait()`。tick 内部：

1. **状态短路**：EMERGENCY_STOP 直接 return；PAUSED → `_arm_inner.set_target(None)`（hold 哨兵）
   后 return；IDLE return。
2. **VR 陈旧检查**：单帧年龄 >0.5s → `set_target(None)` 保持 + 计数；累计丢帧折算 ≥3.0s →
   `_stop_recording(save=False)` + 回 IDLE。计时按帧数/target_hz 折算——主循环 overrun 时
   真实 wall-clock 会略长于 3s。
3. **读内环状态**：`arm_qpos, error_state, ts = _arm_inner.get_state()`；`error_state=True` →
   `_escalate_to_emergency`（内环出错到外环发现最多滞后一个周期 20ms）。
4. **状态聚合**：`robot.get_state(arm_qpos=内环值)` —— 传入 arm_qpos 时跳过 arm SDK 读取，
   **arm_qvel/arm_tau 置 NaN**（重要副作用：遥操作模式下录制的 qvel/tau 全是 NaN，且
   validate 的力矩门因 NaN 静默跳过）。
5. **命令计算**：`pipeline.compute_action()`（见 §4）。仅当 `ik_ok` 时更新 `_last_arm_cmd`。
6. **预发送验证**：`validate_action(robot, action, actual_arm_qpos, actual_arm_tau)`。
   error/断连 → 急停；其余失败（力矩/温度/碰撞谓词命中）→ 用 `_hold_action()`（`_last_good_arm`）
   替换后**继续下发**。
7. **分流下发**：arm → `_arm_inner.set_target(action.arm_qpos_cmd)`；hand → `robot.send_action(action)`。
   可选 sync 握手（`policy_ready`/`robot_ready`，策略推理用，遥操默认关闭）。
8. **录制**：验证后录制（记录的是**实际发出**的内容）。`flag_held = (not ik_ok) ∨ (not retarget_ok)
   ∨ validate_failed` —— 注意手部重定向失败也会把整帧标 held。
9. 每 2s 打印状态（含内环跟踪误差）；tick 超 1.5× 周期告警。

### 3.3 按键语义

| 键 | 条件 | 行为 |
|---|---|---|
| B | IDLE | `_ensure_inner_running`（惰性重启内环）→ `_reset_mapper`（VR/EEF 锚定标定 + 清 EMA）→ 开始录制 → TELEOP |
| C | TELEOP↔PAUSED | 暂停=发 None 哨兵；恢复**必须** `_reset_mapper` 成功（有 VR 帧）才回 TELEOP |
| S | TELEOP/PAUSED | 停止 + **保存** episode → IDLE |
| Q | 任意 | **丢弃**录制并退出 |
| H | 非 EMERGENCY_STOP | 归位（§6）。**无条件丢弃进行中的录制**——想保数据先按 S |
| ESC | 任意 | `_escalate_to_emergency` |

---

## 4. 命令生成链（VR → arm_qpos_cmd）

### 4.1 ArmWristMapper（映射层）

reset-relative 增量映射：B/C 键时把当前 VR 腕部位姿与当前 EEF 位姿锚定为零点，之后
`delta = scale × (当前腕部 − 锚点)`，经 `vr_to_base_rot`/`base_to_world_rot` 坐标变换加到 EEF 锚点上。

安全裁剪：位置 delta 裁到 `eef_delta_bounds`；**每帧旋转 delta 上限 1.0 rad (~57°)**（拦截 VR
跟踪毛刺）；四元数符号连续性处理。

### 4.2 workspace clamp + 笛卡尔 EMA

- workspace clamp 在 EMA **之前**（边界处不会产生越界的平滑中间值），clamp 回调来自 `RobotInterface`
  （默认工作空间 x[0.28,0.72] y[−0.45,0.45] z[0.05,0.5] m）。
- 笛卡尔 EMA 是 **arm 唯一的平滑级**，作用在 IK 之前的笛卡尔目标上（不是关节空间）：位置 R³ 标准
  EMA，旋转 quat→rotvec so(3) EMA。**alpha 加权在新目标上**（alpha=1.0 为直通，越小滤波越强，与
  常见约定相反）。默认 `ema_alpha_pos=0.8 / ema_alpha_rot=0.4`；**主入口 `vr_teleop_shm.py` 刻意
  设为 1.0/1.0（直通）**。
- 平滑后的目标同时记入 `action.target_eef_pos/rot6d`（HDF5 的 `action_arm_ee`）。

### 4.3 solve_teleop_ik（两级 IK）

**Tier 1 — 差分 IK（主路径，确定性）**：DLS（阻尼最小二乘）从**实测 qpos** 迭代，最多 100 次，
step=0.05，λ²=1e-5，收敛阈值 1e-3；解出后验证世界系位姿误差 ≤ 8mm / 0.08rad，超限降级 Tier 2。
注释声称 <1ms/帧。

**Tier 2 — MPlib position IK（随机回退）**：seed 顺序 prev_cmd(n=3) → current_qpos(n=2)；候选
过滤链：位姿误差 → **每关节 90° jump-limit** → **肘部分支一致性**（J4 跨越 [−5°, +15°] 阈值带且
|ΔJ4|>40° 判翻转拒绝）→ 加权归一化 L2 距离排序（关节权重 [3.0, 1.2, 1.0, 0.5, 0.5, 0.8, 0.3]）；
prev_cmd seed 且 L∞≤15° 快速接受。

**注意**：90° jump-limit 与肘部一致性**只在 Tier 2 回退路径生效**。Tier 1 无逐帧关节跳变门，
依赖确定性迭代 + canonicalize 分支对齐 + 内环 Δclamp 兜底（CLAUDE.md 的表述覆盖面被夸大，
见 §9 #5）。

**命令组装（两级共用）**：canonicalize（以 prev_cmd 为分支参照）→ 可选 nullspace 关节限位斥力
（15° margin，尽力而为，失败静默跳过）→ self/env 碰撞门（FCL；env 门有"向上恢复豁免"：目标 Z
高于当前 EEF Z+1mm 时放行，防低位卡死）。

**held 语义**：任何环节失败（wrist NaN、mapper 未锚定、workspace 越界无 clamp、IK 全失败、碰撞
门命中）都返回**上一条成功命令**。`_last_arm_cmd` 只在 ik_ok 时推进，连续 hold 帧全部锚定在最后
一次成功 IK 上，且**每 tick 仍主动下发**（held ≠ 停发）。

### 4.4 validate_action（预发送门）

模块级自由函数 `robot/validate.py`（**不是** RobotInterface 方法——CLAUDE.md 过时，见 §9 #1）。
8 项检查，fail-fast 顺序：

| # | 检查 | 遥操作实际状态 |
|---|---|---|
| 1 | robot.is_error() 错误门 | 生效；失败 → 急停 |
| 2 | arm 连接门 | 生效；失败 → 急停 |
| 3 | 每关节力矩门（限值 [50,50,30,30,30,20,20] Nm） | **静默 no-op**：遥操路径 tau=NaN，`isfinite` 前提不满足 |
| 4 | 温度门（70°C） | 未接线（调用点不传 temps） |
| 5 | env 碰撞谓词 | 未接线（调用点不传 predicate） |
| 6 | workspace clamp（非致命，原地写回） | 生效 |
| 7 | arm 关节限位 clip（**软限位**，原地写回，见 §7.1） | 生效 |
| 8 | hand 关节限位 clip（原地写回） | 生效 |

**副作用**：clip 类检查**原地修改** action 数组——下发与录制的都是改写后的值。

---

## 5. 内环：ArmInnerLoop（Mode 6）

### 5.1 线程模型与通信

- 单 daemon 线程 `arm_inner_loop`，**在线程内自建** XArmAPI 连接（与 XArm7 驱动的连接独立）。
- 跨线程通信：单把 `threading.Lock` 保护的共享变量。`set_target()` 是 **latest-wins 单槽信箱**
  （无队列，新目标覆盖旧目标）；`set_target(None)` 是显式的"释放/保持"哨兵，且**也会刷新时间戳**。
- `get_state()` 返回 `(arm_qpos, error_state, target_ts)` 快照。

### 5.2 启动序列

```
XArmAPI(ip) → clean_error/clean_warn/motion_enable(True)
→ _init_mode: 固件版本校验(≥1.10.0, 仅警告) → set_mode(0)+state(0) → sleep(0.05)
  → set_mode(6)+state(0) → sleep(0.05) → state(0) → set_joint_maxacc(8.73 rad/s²)
→ mode==6 校验，失败重试一次，再失败 → error_state=True 拒绝进入主循环
→ set_collision_sensitivity(1) → 读初始 qpos → set _ready_event
```

### 5.3 主循环（每帧）

```python
while not stop_event:
    limiter.wait()                                # 50Hz
    target, target_ts = 锁下快照
    # 超时守卫: 曾收到过目标 且 (target is None 或 年龄>0.2s)
    if 超时:
        self._ramp_step = 0                       # ← 2026-07-15 新增: 复位软启动斜坡
        _hold_position(arm)                       # 读当前位置原样回发
        continue
    if NaN目标: 回发 last_valid_qpos; continue     # NaN 守卫
    code, states = arm.get_joint_states()         # 读实测
    if code != 0 or arm.error_code != 0:          # 双重错误检查(覆盖 C31 碰撞)
        error_state = True; break                 # ← 线程永久退出，无自动重启
    _monitor(arm, current_qpos)                   # A4/A5 被动监视，只告警不改命令
    _send_target(arm, target)                     # ↓
```

`_send_target`：

```python
# 1. 每步 L∞ delta 钳位 ±0.3 rad（相对 _last_sent_target；正常步长 ~0.03 rad，10 倍余量）
# 2. 软启动斜坡: 前 20 帧(0.4s) 速度从 0.2 → 1.5708 rad/s 线性爬升
# 3. arm.set_servo_angle(angle, is_radian=True, speed, mvacc, wait=False)  ← 透传给固件
# 4. 成功才更新 _last_sent_target / _ramp_step；code!=0 → error_state（下一帧 break）
```

### 5.4 错误处理语义（不对称，刻意设计）

| 情况 | 处理 | 可恢复性 |
|---|---|---|
| `get_joint_states` 抛异常 | error_state + **continue** | 可自愈（下帧状态有效则复位 False） |
| 返回 code≠0 / `arm.error_code`≠0 | error_state + **break** | 线程永久退出，需外部重建（B 键 `_ensure_inner_running`） |
| `set_servo_angle` 失败 | error_state，不 break | 下一帧经 error_code 检查退出（1 帧延迟） |
| 线程 fatal 异常 | logger.exception + error_state + finally | 同上 |
| finally | `_signal_ready_only()`（唤醒 sync 阻塞的 controller）+ disconnect | Mode 6 固件断连后保持位置，无需显式停止 |

### 5.5 被动监视（A4/A5）

- **A4 跟踪误差**：`L∞|last_sent_target − current_qpos| > 0.35 rad` 告警（捕捉固件错误码漏掉的
  软饱和/跟随误差）。
- **A5 mode 漂移**：每帧复查 `arm.mode == 6`（SDK 缓存属性）。
- 两者节流**相互独立**（各 50 帧 ≈1s 冷却；commit 869f4e9 修复了共用节流互相压制的问题）。
- 只告警，从不改命令或 error_state。hold/NaN 路径 continue 跳过监视。

---

## 6. 阻塞路径：return_to_home（H 键）

`TeleopController._do_home` → **先停内环**（释放 Mode 6，避免双连接冲突）→
`RobotInterface.return_to_home()`：

```
预处理: 连续关节 2π 等价吸附 → 已在家(1°内)短路 → hand.reset + 碰撞模型手部姿态同步
Tier 1: planner.plan_path(home_eef)  — screw → 多RRT，9 项验证链，path_score 排序
        → _execute_waypoints: 按 1° 稠密线性插值，逐点 is_error() 检查
          → arm.send_action()（懒切 Mode 1, set_servo_angle_j）→ sleep(dt)
          开环下发（无到位确认）；中途失败有 warning 日志（2026-07-15 新增）
Tier 2: (Tier1 失败时) 关节直线插值 + 自碰/环境碰/桌面FK 复查；
        直线有碰撞 → 两段式绕行（先 J0-J2 抬臂归位，后 J3-J6）
Tier 3: (Tier2 也失败) _reset_blocking — SDK 阻塞移动，无碰撞规避
Phase 2: 关节空间精调到精确 home（同样过碰撞检查 + _execute_waypoints）
Finalize: arm.reset() — Mode 0 + set_servo_angle(wait=True)，20°/s 阻塞收敛到 init_qpos
```

- 每级之间用 `not arm.is_error()` 门兜底：任何一级执行中锁存错误后，后续级全部跳过并返回 False。
- 归位后模式停在 Mode 0，内环已死；controller 回 IDLE。下一次 B 键才重建内环重新进 Mode 6。
- waypoint 节奏：dt 默认 0.02s、1° 步长 ≈50°/s；示例传 `home_dt=0.04` 降到 ~25°/s。
- 最终精度**完全依赖**末尾阻塞 reset（Phase 1/2 是开环的）。

---

## 7. 安全层级总表（自上而下）

| 层 | 机制 | 关键数值 |
|---|---|---|
| 映射层 | 旋转 delta/帧上限；位置 delta 边界裁剪 | 1.0 rad/帧 |
| 目标层 | workspace clamp（EMA 之前） | x[0.28,0.72] y[±0.45] z[0.05,0.5] m |
| IK 层 | 位姿误差验证；jump-limit + 肘部一致性（仅 Tier2）；self/env 碰撞门；nullspace 限位斥力 | 8mm/0.08rad；90°/关节；J4 带 [−5°,+15°] |
| 预发送门 | validate_action 8 项（错误/连接/力矩/温度/env/workspace/arm clip/hand clip） | 力矩 [50,50,30,30,30,20,20] Nm；温度 70°C |
| 内环 | 0.2s 超时 hold；NaN 守卫；Δclamp；软启动斜坡；A4/A5 监视；错误即停 | ±0.3 rad/步；20 帧斜坡；0.35 rad 告警 |
| 固件(Mode 6) | 在线轨迹规划限速限加速；碰撞检测(C31) | 90°/s；500°/s²；灵敏度 1（最低） |
| 驱动层 | 软件软限位 clip；固件 reduced-mode 限位；TCP 负载配置 | 见 §7.1；1.2kg@[0,0,80]mm |
| 急停 | 先停内环 → arm.set_state(4) + hand 零增益；error 锁存到 clear_error | — |

### 7.1 关节限位三层嵌套（2026-07-15 修复后）

严格包含关系 **软限位 ⊂ 固件缩减限位 ⊂ config(=出厂)限位**，保证贴边命令永不触发固件错误：

| 关节 | config（=出厂） | 固件 reduced（内缩 2°） | 软限位（内缩 2.5°，所有软件 clip 用这层） |
|---|---|---|---|
| J1/J3/J5/J7 | ±360° | ±360°（全旋转关节不内缩） | ±360° |
| J2 | [−118°, 120°] | [−116°, 118°] | [−115.5°, 117.5°] |
| J4 | [−11°, 225°] | [−9°, 223°] | [−8.5°, 222.5°] |
| J6 | [−97°, 180°] | [−95°, 178°] | [−94.5°, 177.5°] |

- 单一事实来源：`xarm7._inset_joint_limits()`；常量 `_FIRMWARE_LIMIT_MARGIN_RAD=2°`、
  `_SOFT_LIMIT_MARGIN_RAD=2.5°`。软件 clip 点：`validate_action` 第 7 项（遥操路径）与
  `XArm7._limit_joint_range`（Mode 1 路径），均用 `XArm7.qpos_min_soft/qpos_max_soft`。
- J4 下限即肘翻转边界，低位/近身抓取会主动逼近——这是修复前最易触发固件停机（推断为 C23
  "Joints Angle Exceed Limit"，中等置信度，未实机验证）的场景。
- `preflight.py` 的位置范围检查仍用完整 config 限位（检查的是实测状态而非命令，暂未对齐）。

---

## 8. 错误与恢复语义

- **锁存（sticky）错误**：任何 SDK 失败、`stop()`、模式切换后检失败都置 `error_state=True`，
  **只有显式 `clear_error()` 能清除**。`stop()` 即使成功也无条件锁存——停止是单向闸门，不是可逆暂停。
- `XArm7.is_error()` 四条判据：`arm is None ∨ !connected_flag ∨ error_state ∨ arm.error_code≠0`。
- `clear_error()`：clean_error/clean_warn/motion_enable(True)/set_state(0) + 后检；**刻意不改
  控制模式**（防止打断内环的 Mode 6）。C31/C32 碰撞后由其中的 motion_enable 重新使能。
- **急停链**：ESC/内环错误/validate 致命失败 → `_escalate_to_emergency`：先 `set_target(None)`
  + `stop()` 停内环（防 SDK 报错刷屏），再 `robot.emergency_stop()` = `arm.set_state(4)` +
  hand 零增益命令。恢复：H（内部 clear_error + 归位）或 Q 退出。
- **VR 断连**：单帧 >0.5s → hold；累计 ≥3.0s → 停录（不保存）回 IDLE（**不是急停**）。

---

## 9. 已知陷阱与"文档-代码差异"表（防误导，重要）

以下条目是**代码库内注释/CLAUDE.md 与实际行为的已确认差异**。阅读本项目其他文档或代码注释时
遇到这些说法，以本表为准：

| # | 过时/错误的说法（出处） | 实际行为 |
|---|---|---|
| 1 | "RobotInterface.validate_action()，4 operations，torque 未接线"（CLAUDE.md） | 是 `robot/validate.py` 的**模块级自由函数**，定义 8 项；力矩门已接线但因遥操路径 tau=NaN 而**静默 no-op** |
| 2 | "Path execution: torque monitoring per waypoint"（CLAUDE.md） | 不存在。逐 waypoint 只查 `is_error()`；力矩保护实际靠固件碰撞检测 |
| 3 | "VR-disconnect timeout → EMERGENCY_STOP"（CLAUDE.md + controller 头注释） | 实现是停录回 IDLE，不进急停 |
| 4 | 笛卡尔 EMA 默认 0.8/0.4（CLAUDE.md 数据流图） | 主入口 `vr_teleop_shm.py` 设 **1.0/1.0 直通**；且 EMA 在 IK 之前的笛卡尔空间，不是关节空间 |
| 5 | "arm IK anomaly jump-limit (default 90°)"（CLAUDE.md） | 只在 Tier 2 position-IK 回退路径生效；Tier 1 差分 IK 无逐帧跳变门 |
| 6 | "solve_teleop_ik … desk safety preferred"（CLAUDE.md） | teleop 热路径的桌面安全走 CollisionModel env 碰撞门；`FingertipDeskSafety`(FK 指尖 Z) 只用于离线 plan_path |
| 7 | `ik.py` 注释引用 `_tick_mode4()` @250Hz | 旧架构残留；现为 Mode 6 @50Hz 透传，无 mode 4 |
| 8 | controller 注释 "PID process" / "Read arm state from PID process" | ArmInnerLoop 是**同进程线程**，无 PID 无插值 |
| 9 | `ArmInnerLoopConfig.synchronized` 字段 | **死配置**：从未被读取；sync 握手唯一门控是构造时是否传入 sync 对象 |
| 10 | `ArmInnerLoop.__init__` 的 `dt=1/125` 参数 | 已废弃（API 兼容保留）；周期来自 `cfg.loop_period` |
| 11 | xarm7 注释 "continuous-rotation joint" | J1/3/5/7 是 ±360° **硬限位**关节，非无限旋转（对限位逻辑无影响） |

**其他行为陷阱**（代码正确但易误解）：

- `robot.send_action()` **只发手部**；arm 命令走 `_arm_inner.set_target()`。
- 遥操作录制的 `arm_qvel/arm_tau` 全是 NaN（get_state 传入内环 qpos 的快捷路径所致）。
- VR 陈旧期间（0.5–3s）tick 提前 return，**该帧完全不录制**（不是记 held=True）；时间网格靠
  `TimestampAlignedBuffer` 前向填充补齐。
- hold 路径不更新 `_last_sent_target`：长时间 hold 后恢复，Δclamp 基准是 hold 前的目标（arm 在
  hold 中位置基本不动，基准近似有效）。
- `XArm7.get_state()` 未连接时直接解引用 `self.arm` 会 AttributeError（调用方须保证已连接）。
- hand 限位被 clip 两次（validate 第 8 项 + XHand.send_action 内部）；hand 的 E3 delta clip
  (0.3 rad) 与 E2 EMA 在驱动层。
- `_reset_blocking()` 只受固件错误门保护：仅 Python 侧锁存错误（如 stop() 后）且固件干净时，
  它会真的驱动无避障运动。
- 硬件序列号 1305-8499 段（XARM7_X4_1305 型）出厂限位比 config 更紧（如 J4 下限 −6°）——若为
  该型号，SDK 客户端会先以 OUT_OF_RANGE 拒绝贴边命令（待实机确认）。

---

## 10. 2026-07-15 修复记录（本文档已按修复后状态描述）

| # | 缺陷 | 核查结论 | 修复 |
|---|---|---|---|
| 1 | `_do_home` 末尾调 `limiter.reset()` 但 RateManager 无此方法 → 按 H 必抛 AttributeError 穿透主循环 | CONFIRMED | `RateManager` 补 `reset()`（重锚 wake 时间 + 清告警节流），语义同旧 RateLimiter |
| 2 | 内环重启丢 sync：`_ensure_inner_running` 重建不传 `sync`；且 sync 模式下 stop() 关不掉阻塞在握手中的线程、错误退出不唤醒 controller —— 三处合成"归位/急停恢复 + B 必死锁"（潜伏，当前无人启用 sync） | CONFIRMED | ① 重建传 `sync=self._sync` + 清陈旧 policy_ready；② `stop()` 置 stop_event 后 set policy_ready 解除阻塞；③ finally 补 `_signal_ready_only()` 唤醒 controller |
| 3 | 软启动斜坡 `_ramp_step` 只增不减，PAUSED/hold 后恢复无斜坡 | PARTIAL（机制属实；C/B 恢复强制重锚定 + 固件限速兜底，后果比预想小；真正窗口是 VR 短失联恢复） | hold 分支进入时 `_ramp_step = 0`，统一覆盖 PAUSED/IDLE/VR 失联/主线程卡顿 |
| 4 | Tier1 `_execute_waypoints` 返回值被忽略 | PARTIAL（字面成立，但"应回退 Tier2 却没回退"被反驳——错误锁存不变量使下游门全部正确兜底；实际缺口仅是无日志） | Phase 1/2 中途失败各补含 `last_error_message` 的 warning（log-only，不改控制流） |
| 5 | 软件限位裁到全量程、固件 reduced 内缩 2° → J2/J4/J6 各有 2° 宽的"合法却触发固件停机"区间 | CONFIRMED（medium；唯一实质缺陷） | 见 §7.1：`_inset_joint_limits` 单一事实来源，软限位 2.5° 内缩严格嵌入固件门 |

---

## 11. 参数速查表

| 参数 | 值 | 位置 |
|---|---|---|
| 外环/内环频率 | 50 Hz（0.02s） | `TeleopControllerConfig.target_hz` / `ArmInnerLoopConfig.loop_period` |
| Mode 6 限速/限加速 | 1.5708 rad/s (90°/s) / 8.7266 rad/s² (500°/s²) | `ArmInnerLoopConfig` |
| 内环目标超时 | 0.2 s（10 帧）→ hold | `target_timeout_s` |
| 每步关节 Δclamp | 0.3 rad（L∞，≈10 倍正常步长余量） | `max_joint_delta` |
| 软启动斜坡 | 20 帧（0.4s），0.2 → 1.5708 rad/s | `speed_ramp_frames/min` |
| 跟踪误差告警 | 0.35 rad，节流 50 帧 | `tracking_error_warn_rad` |
| VR 单帧陈旧阈值 / 累计断连 | 0.5 s / 3.0 s | controller |
| 笛卡尔 EMA | 默认 pos 0.8 / rot 0.4；主入口 1.0/1.0 | pipeline / `vr_teleop_shm.py` |
| 映射旋转 delta 上限 | 1.0 rad/帧 | arm_mapper |
| IK 位姿误差门 | 8 mm / 0.08 rad | `TeleopProfile` |
| IK jump-limit / fast-accept | 90°/关节 / 15° L∞（仅 Tier2） | `TeleopProfile` |
| 限位 margin | 固件 2° / 软件 2.5°（J2/J4/J6） | `xarm7.py` 模块常量 |
| 归位 waypoint | 1° 插值 + sleep(0.02s) ≈50°/s；最终 reset 20°/s | interface / XArm7Config |
| 力矩/温度限 | [50,50,30,30,30,20,20] Nm / 70°C | `robot/types.py` |
| 模式切换开销 | ≈0.1s（两次 sleep 0.05） | `_set_mode` / `_init_mode` |
| Mode 6 固件要求 | ≥ 1.10.0 | `_init_mode` 版本校验 |
| 碰撞灵敏度 / TCP 负载 | 1（最低，防误报 C31）/ 1.2 kg @ [0,0,80] mm | XArm7Config |

---

*生成于 2026-07-15，基于 branch `feat/collection-hardening-r1-o1-i1-a4-r3` 当日工作区状态
（含未提交的 5 项 bug 修复）。核查方法：6 个并行子系统深读 + 4 项对抗式 fact-check，
全部结论落到具体函数级代码证据。*
