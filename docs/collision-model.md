# 碰撞检测机制

xArm7 + XHand 碰撞检测系统的架构、层级、调用路径与使用指南。

---

## 1. 概述

碰撞检测由 `CollisionModel`（`planning/collision_model.py`）统一管理，基于 Pinocchio + FCL（Flexible Collision Library）实现。系统独立于 MPlib，支持自碰撞（机器人自身关节间碰撞）和环境碰撞（机器人与桌面/障碍物碰撞）两种检测类型。

**核心约束**：碰撞 URDF 中的手部采用固定张开姿态（home/open），是一个保守假设 — 实际手部闭合时会更接近物体，但碰撞模型使用的是最安全的（张开）姿态进行检测。

### 文件职责

| 文件 | 职责 |
|------|------|
| `collision_model.py` | 碰撞模型核心：FK 更新、自碰撞/环境碰撞检测、障碍物管理 |
| `planner.py` | 路径规划入口，持有 `CollisionModel` 和 `IKCandidateManager` |
| `ik_candidates.py` | IK 候选管理 + 碰撞过滤 wrapper（代理到 `CollisionModel`） |
| `ik.py` | 遥操作 IK 求解器 + 碰撞安全门（`_check_teleop_collision_gate`） |

---

## 2. 双模式架构

`CollisionModel` 支持两种 DOF 模式，共用同一个 SRDF（`xarm7_xhand.srdf`）：

| 模式 | DOF | URDF | 用途 |
|------|-----|------|------|
| `hand_dof=False` (7-DOF) | 7 (仅臂) | `xarm7_xhand_collision.urdf` | MPlib 路径规划 |
| `hand_dof=True` (19-DOF) | 19 (7 臂 + 12 手) | `xarm7_xhand_right.urdf` | 遥操作碰撞检测 / IK 验证 |

19-DOF 模式中，碰撞检查接受 `(7,)` 形状的臂部 qpos，并通过 `set_hand_qpos()` 设置的缓冲区自动扩展为 `(19,)` 进行 FK。

### 碰撞对配置

```
URDF → 添加所有 N×(N-1)/2 候选对 → SRDF 删除 Adjacent + Never → 选择性重新启用高风险对
```

- 7-DOF: 34²/2 = 561 → SRDF 过滤后 ~0-141 对
- 19-DOF: 40²/2 = 780 → SRDF 过滤后 255 对
  - 额外启用：拇指指尖 ↔ 食指指尖（最常见的 pinch 碰撞风险）

---

## 3. 双层环境碰撞架构 (Tier 1 + Tier 2)

环境碰撞检测采用两级策略，平衡速度与精度：

```
check_env_collision(qpos)
  │
  ├─ Tier 1: Z-min 预筛选 (~17 μs, 零 FCL 调用)
  │   计算所有机器人几何体的最低 Z 坐标
  │   if z_min > obs_z_max + 0.05m: return False  ← 安全，跳过 Tier 2
  │   if z_min <= obs_z_max + 0.05m: 进入 Tier 2
  │
  └─ Tier 2: Z-过滤 FCL 网格碰撞 (2-8 ms, 19-DOF 模式)
      对每个障碍物，遍历机器人几何体：
        if z_geom > obs_z_max + 0.25m: skip  ← 臂基座/肩部/上臂跳过
        否则：FCL mesh-mesh collide() → 精确检测
```

### Tier 1 参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `_Z_TIER1_MARGIN` | 0.05 m | 指节网格半高度 (~4cm) + 1cm 安全余量 |
| `_Z_TIER2_MARGIN` | 0.25 m | 距障碍物超过 25cm 的几何体跳过 FCL |

### 关键设计决策：Tier 1 用于 HOLD，Tier 2 用于 REJECT

- **遥操作 HOLD**（`check_env_collision_fast` / `check_teleop_collision`）：使用 Tier 1
  - 保守策略：宁可误停（HOLD），不冒险继续
  - 操作员保持手部可见地高于桌面 → Tier 1 几乎总是通过
  
- **IK 拒绝决策**（`check_env_collision` → `has_env_collision`）：使用 Tier 2
  - 精确策略：只拒绝真正碰撞的 IK 解
  - Tier 1 过于保守，会拒绝接近（但未接触）桌面的有效 IK 解

此设计原则在三个调用点统一执行：
1. `ik_candidates.py` — IK 候选过滤
2. `planner.py` — 路径规划验证
3. `ik.py` — 遥操作 IK 安全门

---

## 4. API 速查

### CollisionModel（核心）

```python
from dexmani_real.planning.collision_model import CollisionModel

cm = CollisionModel(hand_dof=True, collision_config=config)

# 手部姿态（19-DOF 模式必须调用）
cm.set_hand_qpos(hand_qpos)               # 每帧遥操作前调用

# 自碰撞
cm.check_self_collision(qpos)             # → bool（stop_at_first=True）
cm.check_self_collision_details(qpos)     # → CollisionInfo（完整碰撞对信息）

# 环境碰撞
cm.check_env_collision(qpos)              # → bool（Tier1 + Tier2，路径规划用）
cm.check_env_collision_fast(qpos)         # → bool（仅 Tier1，遥操作热路径）
cm.check_teleop_collision(qpos)           # → (has_self, has_env)，单次 FK

# 段/路径碰撞
cm.check_segment_collision_free(s, e, step)        # 线段自碰撞检查
cm.check_segment_env_collision_free(s, e, step)     # 线段环境碰撞检查

# 障碍物管理
cm.add_table(height, x_center, half_x, half_y, half_z)
cm.add_box_obstacle(name, half_extents, position, rotation=None)
cm.remove_obstacle(name)
cm.clear_obstacles()
```

### 通过 Planner 代理调用

```python
planner = XArm7MotionPlanner(config, planning_profile, teleop_profile)
# set_hand_qpos 会同时更新 planner.collision_model
planner.set_hand_qpos(hand_qpos)

# 碰撞查询（通过 __getattr__ 代理到 collision_model）
planner.has_self_collision(qpos)    # → bool
planner.has_env_collision(qpos)     # → bool（Tier2）
planner.check_self_collision(qpos)  # → CollisionInfo
```

---

## 5. 调用路径全景

### 5.1 遥操作热路径 (50 Hz)

```
TeleopController
  ├─ TeleopPipeline.compute_action()
  │   └─ planner.solve_teleop_ik(target, current, prev)
  │       └─ ik.py: _check_teleop_collision_gate(qpos_cmd)
  │           ├─ ik_mgr.has_self_collision(qpos)    # stop_at_first, ~35μs
  │           └─ ik_mgr.has_env_collision(qpos)     # Tier1+Tier2
  │
  └─ RobotInterface.validate_action()
      └─ collision_model.check_env_collision_fast()  # Tier1 only, ~17μs
```

### 5.2 路径规划 (按需)

```
planner.plan_path(target, current)
  ├─ collect_ik_candidates()
  │   └─ ik_candidates.py: filter_ik_candidate()
  │       ├─ cm.check_self_collision(q)              # 自碰撞
  │       └─ cm.check_env_collision(q)               # 环境碰撞 (Tier2)
  │
  ├─ 路径验证 (_validate_path)
  │   ├─ _check_self_collision(path[i])              # 每 waypoint
  │   ├─ _check_env_collision(path[i])               # 每 waypoint (Tier2)
  │   └─ _check_desk_safety(path[i])                 # FK 指尖 Z 检测
  │
  └─ _is_shortcut_valid(prev, nxt)
      ├─ cm.check_self_collision(mid)
      ├─ cm.check_self_collision(q1) + cm.check_self_collision(q3)
      └─ cm.check_env_collision(q1) + cm.check_env_collision(q3)
```

### 5.3 归位 (Return-to-Home)

```
planner.plan_path(home_eef, current)
  └─ 同上路径规划流程
```

---

## 6. 性能特征

实测环境: Intel i9-13900K

| 操作 | 7-DOF | 19-DOF | 说明 |
|------|-------|--------|------|
| 自碰撞 (`stop_at_first`) | ~30 μs | ~35 μs | 单次 FK + computeCollisions |
| 环境碰撞 Tier 1 (安全) | ~17 μs | ~17 μs | Z-min 比较，零 FCL 调用 |
| 环境碰撞 Tier 2 (近桌面) | — | 2-8 ms | FCL mesh-mesh，~15-25 几何体 |
| 线段检查 (Δ=0.5 rad) | — | ~870 μs | 25 采样点 × ~35 μs |
| 单次 FK + 几何体更新 | ~8 μs | ~12 μs | forwardKinematics + updateGeometryPlacements |

### 遥操作每帧碰撞开销（常见路径）

```
无碰撞帧:  17 μs (Tier1 only, check_env_collision_fast)
            或 35 μs (check_teleop_collision: 自碰撞 + Tier1 环境)
碰撞帧:    ~35 μs (自碰撞 bool gate) + 2-8 ms (Tier2 FCL，仅手部靠近桌面时)
```

---

## 7. 手部 qpos 初始化

19-DOF 模式下，`CollisionModel` 的手部姿态缓冲区默认为零（张开手），必须显式初始化：

```python
# 仿真中
planner.collision_model.set_hand_qpos(sim.get_full_qpos()[7:])

# 真机中
planner.set_hand_qpos(robot.get_state()["hand_qpos"])
```

未初始化时会发出一次 warning（通过标志位去重），碰撞检查退化为使用零姿态（张开手）。这是保守的安全假设 — 张开手比闭合手占据更多空间，因此不会漏报碰撞，但可能产生误报。

---

## 8. 设计原则

1. **单一 FK 复用**：`check_teleop_collision` 执行一次 FK，同时完成自碰撞（FCL）和环境碰撞（Tier1 Z-min），避免两处分别 FK。

2. **障碍物碰撞对隔离**：环境碰撞使用直接的 `fcl.collide()` 调用，障碍物的碰撞对**不注册**到主模型的 `computeCollisions()` 中。这保证自碰撞检测速度不受障碍物数量影响。

3. **Tier 1 预筛选优先**：在常见遥操作场景中（手部明显高于桌面），Tier 1 返回 False（安全），跳过昂贵的 FCL mesh-mesh 检测。Tier 2 仅在手部靠近桌面时触发。

4. **保守失败策略**：NaN/非有限 FK 结果、FCL 异常 → 返回 True（假设碰撞），确保安全优先。

5. **SRDF 单一来源**：7-DOF 和 19-DOF 模型共用同一个 SRDF，避免碰撞对配置不一致。

---

## 9. 相关文件索引

| 文件路径 | 关键内容 |
|----------|---------|
| `planning/collision_model.py` | `CollisionModel` 类，所有碰撞检测逻辑 |
| `planning/ik_candidates.py:342-417` | IK 候选的碰撞过滤 wrappers |
| `planning/planner.py:36-49` | `XArm7MotionPlanner` 碰撞代理 |
| `planning/planner.py:355-400` | 路径验证链（自碰撞+环境碰撞+桌面安全） |
| `planning/ik.py:352-385` | 遥操作 IK 碰撞安全门 |
| `planning/collision_config.py` | `CollisionConfig`（Tier 参数可配置化） |
| `planning/desk_safety.py` | `FingertipDeskSafety`（FK 指尖 Z 补充检测） |
| `assets/robots/xhand/xarm7_xhand.srdf` | 统一碰撞对定义 |
| `assets/robots/xhand/xarm7_xhand_collision.urdf` | 7-DOF 碰撞 URDF |
| `assets/robots/xhand/xarm7_xhand_right.urdf` | 19-DOF 完整 URDF |
