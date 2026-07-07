# DexMani 坐标系变换链

> VR 遥操作中从 Quest 原始位姿数据到机械臂末端运动的完整坐标变换链路。
> 覆盖 Unity 左手系 → FLU 右手系 → 机器人世界系的每一步变换，
> 以及 B 键按下时的 HeadPose 朝向标定机制。

---

## 目录

1. [坐标帧总览](#1-坐标帧总览)
2. [变换链路 (数据流)](#2-变换链路-数据流)
3. [FLU 坐标帧](#3-flu-坐标帧)
4. [ArmWristMapper 差分映射](#4-armwristmapper-差分映射)
5. [Heading Calibration (朝向标定)](#5-heading-calibration-朝向标定)
6. [完整变换链 (数学)](#6-完整变换链-数学)
7. [数据生命周期](#7-数据生命周期)
8. [关键参数速查](#8-关键参数速查)
9. [文件索引](#9-文件索引)

---

## 1. 坐标帧总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                         坐标帧层级                                   │
│                                                                     │
│  Quest IMU (Unity 左手系, Y-up)                                     │
│    │  pos=(x→右, y→上, z→前)   quat=(qx,qy,qz,qw)                  │
│    │                                                                │
│    ▼ unity_left_to_flu_position / unity_left_to_flu_rotation        │
│                                                                     │
│  FLU 右手系 (X→前, Y→左, Z→上)                                      │
│    │  pos=(x→前, y→左, z→上)   quat=(qw,qx,qy,qz)                  │
│    │                                                                │
│    │  ★ 原点在地面, +X = 建 Guardian 时面朝方向 (固定!)              │
│    │                                                                │
│    ▼ ArmWristMapper.map()                                           │
│    │  delta = (current - reference)                                 │
│    │  ── vr_to_base_rot ──→ robot_base_frame                       │
│    │  ── base_to_world_rot ──→ world_frame                         │
│    │                                                                │
│  Robot World (右手系, Z-up)                                          │
│    │  pos=(x→前, y→左, z→上)   quat=(qw,qx,qy,qz)                  │
│    │                                                                │
│    ▼ IK + ArmInnerLoop                                              │
│                                                                     │
│  xArm7 EEF Pose                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 各帧定义

| 帧 | 坐标系 | X | Y | Z | 原点 | 生命周期 |
|----|--------|---|---|---|------|---------|
| **Unity Left** | 左手 | 右 | 上 | 前 | Quest 内部 | HTS SDK 内部 |
| **FLU** | 右手 | 前 | 左 | 上 | 地板, Guardian 原点 | VR 数据全生命周期 |
| **Robot Base** | 右手 | 前 | 左 | 上 | 机械臂底座 | IK 求解器内部 |
| **World** | 右手 | 前 | 左 | 上 | 机械臂底座 (base=world 时) | Planner + Mapper |

---

## 2. 变换链路 (数据流)

```
┌──────────────────┐
│ Meta Quest HTS   │  120-240Hz IMU + 相机
│ (Unity 左手系)    │
└────────┬─────────┘
         │ TCP (adb reverse tcp:8000 tcp:8000)
         ▼
┌──────────────────────────────────────────────────────────┐
│ HTSClient.iter_events()                                  │
│                                                          │
│  ParsedPacket (Unity 左手系, raw CSV)                    │
│    ├─ WristPosePacket   → HandFrameAssembler             │
│    ├─ LandmarksPacket   → HandFrameAssembler             │
│    └─ HeadPosePacket    → HandFrameAssembler             │
│         │                                                │
│         ▼                                                │
│  AssembledFrame                                          │
│    ├─ HandFrame(wrist, landmarks)  @ hand side filter    │
│    └─ HeadFrame(head)              @ hand side filter    │
│                                                          │
│  ★ hand_filter="both" 时才放行 HEAD 包                   │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ VRReceiverProcess._run()  (子进程)                        │
│                                                          │
│  HandFrame:                                              │
│    wrist=(x,y,z,qx,qy,qz,qw)  ← Unity 左手系            │
│    landmarks=21×(x,y,z)       ← Unity 左手系            │
│                                                          │
│  HeadFrame:                                              │
│    head=(x,y,z,qx,qy,qz,qw)    ← Unity 左手系            │
│    → 缓存到 _latest_head_pos / _latest_head_quat_wxyz    │
│                                                          │
│  ── unity_left_to_flu_position/rotation ──→              │
│    ★ (z, -x, y) 位置变换                                 │
│    ★ 对应的四元数变换                                     │
│    ★ xyzw → wxyz (transforms3d 惯例)                     │
│                                                          │
│  frame_dict:                                             │
│    wrist_pos(3), wrist_quat_wxyz(4)   ← FLU             │
│    landmarks(21,3)                     ← FLU             │
│    head_pos(3), head_quat_wxyz(4)     ← FLU (缓存)      │
│                                                          │
│  → shm.write_vr_frame(frame_dict)                        │
└──────────────────────┬───────────────────────────────────┘
                       │ SharedMemory RingBuffer (FILO, ~3 slots)
                       ▼
┌──────────────────────────────────────────────────────────┐
│ 主循环 (50Hz)                                            │
│                                                          │
│  vr_receiver.read_latest() → frame dict                  │
│                                                          │
│  ┌─────────────────────────────────────────┐             │
│  │ B 键按下:                                │             │
│  │   arm_mapper.set_heading(head_quat)      │             │
│  │   arm_mapper.reset(wrist, eef)          │             │
│  └─────────────────────────────────────────┘             │
│                                                          │
│  每帧: arm_mapper.map(wrist_pos, wrist_quat)             │
│    → target_pos, target_quat (World)                     │
│    → Workspace clamp                                     │
│    → IK solve (DLS, Pinocchio)                           │
│    → EMA smooth (joint space, α=0.6)                     │
│    → ArmInnerLoop.set_target(qpos)                       │
└──────────────────────┬───────────────────────────────────┘
                       │ threading.Lock + numpy array
                       ▼
┌──────────────────────────────────────────────────────────┐
│ ArmInnerLoop (250Hz, daemon thread)                      │
│                                                          │
│  PID(error) → velocity → clip → accel limit → jerk limit │
│  → vc_set_joint_velocity(qvel)  (mode 4)                 │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
                  ┌─────────┐
                  │  xArm7  │
                  └─────────┘
```

---

## 3. FLU 坐标帧

### 3.1 定义

**FLU** = Forward-Left-Up 右手坐标系：

| 轴 | 方向 |
|----|------|
| +X | 前 (Forward) |
| +Y | 左 (Left) |
| +Z | 上 (Up) |

### 3.2 从 Unity 左手系的推导

HTS SDK 源码 (`hand_tracking_sdk`) 中的变换函数：

```python
# Step 1: Unity 左手系 → Unity 右手系 (Y 轴反转)
def unity_left_to_right_position(x, y, z):
    return (x, -y, z)

# Step 2: Unity 右手系 → FLU (轴重映射)
# Unity右手: X=右,  Y=上, Z=后
# FLU:       X=前,  Y=左, Z=上
# 映射: FLU_X = Unity_Z, FLU_Y = -Unity_X, FLU_Z = Unity_Y
def unity_right_to_flu_position(x, y, z):
    return (z, -x, y)  # 注意: 这里用的是转换后的右手系y (已反转)

# 合成:
# unity_left_to_flu_position(x, y, z) = (z, -x, y)
```

**四元数**：另有对应的 `unity_left_to_flu_rotation(qx, qy, qz, qw)` 做等效变换。

### 3.3 FLU 原点与方向

| 属性 | 决定因素 | 可变性 |
|------|---------|--------|
| **原点** | Quest Guardian 建边界时站立位置 (地板投影) | 每次建边界可能不同 |
| **+X 方向** | Guardian 建边界时面朝方向 | 每次建边界可能不同 |
| **+Z 方向** | 重力反方向 (始终向上) | 固定 |

**关键限制**：FLU 的 +X 不跟随用户身体旋转。用户转身后，"伸手向前"在 FLU 中不再沿 +X 方向。
这就是为什么需要 [Heading Calibration](#5-heading-calibration-朝向标定)。

### 3.4 FLU 水平面投影

Heading calibration 中，头部前向在 FLU 水平面 (X-Y 平面) 的投影：

```
forward_flu = head_rot @ [1, 0, 0]     ← 头部在 FLU 中的前向 (3D)
forward_2d  = forward_flu[:2]           ← 投影到 X-Y 平面
forward_2d /= ||forward_2d||            ← 归一化 (单位向量)
θ = atan2(forward_2d[1], forward_2d[0]) ← 与 FLU +X 的偏航角
```

---

## 4. ArmWristMapper 差分映射

### 4.1 原理

`ArmWristMapper` 采用**重置相对差分映射**：B 键按下时记录手腕参考位姿 (`wrist_pos0`, `wrist_rot0`) 和机械臂 EEF 参考位姿 (`eef_pos0`, `eef_rot0`)。之后每帧计算手腕相对于参考的增量，通过旋转矩阵变换后叠加到机械臂参考位姿上。

### 4.2 reset() — 建立映射参考

```python
# arm_mapper.py: reset()
self.wrist_pos0 = wrist_pos              # FLU 空间手腕位置
self.wrist_rot0 = quat2mat(wrist_quat)   # FLU 空间手腕朝向
self.eef_pos0   = eef_pos                # World 空间 EEF 位置
self.eef_rot0   = quat2mat(eef_quat)     # World 空间 EEF 朝向
```

### 4.3 map() — 每帧计算目标位姿

```python
# arm_mapper.py: map()

# === 位置 ===
delta_pos_vr    = wrist_pos - wrist_pos0                        # (1) VR 空间位移
delta_pos_base  = pos_scale * (vr_to_base_rot @ delta_pos_vr)   # (2) VR→Base 旋转
delta_pos_world = base_to_world_rot @ delta_pos_base            # (3) Base→World 旋转
target_pos      = eef_pos0 + delta_pos_world                     # (4) 叠加到参考位姿

# === 朝向 ===
delta_rot_vr    = wrist_rot @ wrist_rot0.T                      # (1) VR 空间旋转增量
delta_rot_vr    = scale_rot(delta_rot_vr)                       # (2) 旋转缩放
delta_rot_base  = vr_to_base_rot @ delta_rot_vr @ vr_to_base_rot.T  # (3) 相似变换
delta_rot_world = base_to_world_rot @ delta_rot_base @ base_to_world_rot.T
target_rot      = delta_rot_world @ eef_rot0                     # (4) 叠加
```

### 4.4 旋转矩阵的角色

| 矩阵 | 默认值 | 作用 | 设置时机 |
|------|--------|------|---------|
| `vr_to_base_rot` | `I` (3×3) | VR (FLU) 增量 → Base 增量的旋转 | `set_heading()` / 构造时 |
| `base_to_world_rot` | `I` (3×3) | Base 增量 → World 增量的旋转 | 构造时 (当 base_pose_world ≠ I) |

**位置**使用普通矩阵乘法 (`R @ v`)，**朝向**使用相似变换 (`R @ M @ R.T`)，保证旋转增量的语义在不同帧之间正确传递。

---

## 5. Heading Calibration (朝向标定)

### 5.1 动机

FLU 的 +X 方向是建 Guardian 时固定的，不跟随用户身体。当用户转身后，直觉上的"前"与 FLU 的 +X 不再一致。

**解决方案**：B 键按下时，从 HeadPose 提取用户的面朝方向，计算偏航旋转矩阵 `R_heading`，写入 `vr_to_base_rot`。之后用户的"前"始终映射为机器人的 World +X。

### 5.2 算法

```
输入: head_quat_wxyz (FLU 空间, wxyz 惯例)

1. head_rot = quat2mat(head_quat)              ← 头部朝向旋转矩阵

2. forward_flu = head_rot @ [1, 0, 0]         ← 头部前向在 FLU 中的 3D 向量

3. forward_2d = forward_flu[:2]                ← 投影到水平面
   forward_2d /= ||forward_2d||                ← 归一化

4. θ = atan2(forward_2d[1], forward_2d[0])    ← 与 FLU+X 的偏航角

5. R_heading = R_z(-θ)                         ← 绕 Z 轴旋转 -θ
     ┌                ┐
     │  cosθ  sinθ  0 │
   = │ -sinθ  cosθ  0 │
     │   0     0    1 │
     └                ┘

6. vr_to_base_rot = R_heading
```

### 5.3 验证

设用户在 FLU 水平面的面朝方向为 `[cosθ, sinθ]`。

| 用户动作 | FLU 位移向量 | 经 R_heading 后 | 机器人运动 |
|---------|-------------|-----------------|-----------|
| 手向前伸 | `α·[cosθ, sinθ, 0]` | `α·[1, 0, 0]` | **World +X** (前) |
| 手向左伸 | `α·[-sinθ, cosθ, 0]` | `α·[0, 1, 0]` | **World +Y** (左) |
| 手向上抬 | `α·[0, 0, 1]` | `α·[0, 0, 1]` | **World +Z** (上) |

### 5.4 实现 (`set_heading()`)

```python
# arm_mapper.py: set_heading()
def set_heading(self, head_quat_wxyz: np.ndarray) -> None:
    head_q = np.asarray(head_quat_wxyz, dtype=np.float64)

    # Guard: NaN / zero quaternion → keep current heading
    if not np.all(np.isfinite(head_q)):
        return
    if np.linalg.norm(head_q) < 1e-12:
        return
    head_q = head_q / np.linalg.norm(head_q)

    head_rot = quat2mat(head_q)
    forward_flu = head_rot @ np.array([1.0, 0.0, 0.0])
    forward_2d = forward_flu[:2]
    norm_2d = np.linalg.norm(forward_2d)

    # Guard: head looking nearly vertical → keep current heading
    if norm_2d < 1e-6:
        return

    forward_2d /= norm_2d
    theta = np.arctan2(forward_2d[1], forward_2d[0])
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    self.vr_to_base_rot = np.array([
        [cos_t,  sin_t, 0.0],
        [-sin_t, cos_t, 0.0],
        [0.0,    0.0,   1.0],
    ])
```

### 5.5 触发时机

| 事件 | `set_heading()` | `reset()` | `vr_to_base_rot` |
|------|:---:|:---:|------|
| **B 键** (开始遥操作) | ✅ | ✅ | 从头朝向重新计算 |
| **C 键** (暂停→恢复) | ❌ | ✅ | 保持之前的标定 |
| **H 键** (归位) | ❌ | ❌ (`clear()`) | 保持 (下次 B 覆盖) |
| **构造时** | ❌ | ❌ | `I` (默认) |

### 5.6 边界情况

| 情况 | 检测方式 | 降级行为 |
|------|---------|---------|
| HeadFrame 尚未到达 | `head_pos = [0,0,0]` | 跳过 `set_heading()`，保持默认 `I` |
| Head quaternion NaN | `np.all(np.isfinite)` → False | `logger.warning` + 保持当前 heading |
| 低头看地 (前向≈垂直) | `norm_2d < 1e-6` | `logger.warning` + 保持当前 heading |
| Head quaternion 为零向量 | `norm < 1e-12` | `logger.warning` + 保持当前 heading |

### 5.7 HeadPose 数据获取

HeadPose 通过 HTS SDK 的 `HeadFrame` 事件获取：

```
HTS SDK:
  HeadPosePacket (Unity 左手系, side=HEAD)
    → hand_filter="both" 才放行 (hand_filter="right" 会过滤掉)
    → HandFrameAssembler (include_head_frames=True)
    → HeadFrame(head=HeadPose(x,y,z,qx,qy,qz,qw))

VRReceiverProcess:
  isinstance(event, HeadFrame) → 缓存到局部变量
    head_flu_pos = unity_left_to_flu_position(x, y, z)
    head_flu_quat = unity_left_to_flu_rotation(qx, qy, qz, qw)
    → _latest_head_quat_wxyz = xyzw_to_wxyz(*head_flu_quat)
```

HeadPose 缓存在子进程局部变量中，每当 `HandFrame` 写入 SHM 时附带最新的缓存值。

---

## 6. 完整变换链 (数学)

### 6.1 符号约定

| 符号 | 含义 |
|------|------|
| `p_wrist(t)` | t 时刻手腕在 FLU 中的位置 |
| `R_wrist(t)` | t 时刻手腕在 FLU 中的朝向 (旋转矩阵) |
| `p_wrist0` | B 键按下时 (reset 时刻) 的手腕位置 |
| `R_wrist0` | B 键按下时 (reset 时刻) 的手腕朝向 |
| `p_eef0` | B 键按下时的 EEF World 位置 |
| `R_eef0` | B 键按下时的 EEF World 朝向 |
| `R_heading` | 朝向标定旋转矩阵 (`vr_to_base_rot`) |
| `R_base2world` | Base→World 旋转 (`base_to_world_rot`, 默认 I) |
| `s_pos` | 位置缩放因子 (`pos_scale`, 默认 1.0) |
| `s_rot` | 旋转缩放因子 (`rot_scale`, 默认 1.0) |

### 6.2 Forward Mapping (每帧)

```
位置:
  Δp_vr     = p_wrist(t) - p_wrist0                         ... (1) FLU 空间位移
  Δp_base   = s_pos · R_heading · Δp_vr                     ... (2) 旋转+缩放到 Base
  Δp_world  = R_base2world · Δp_base                        ... (3) Base→World
  p_target  = p_eef0 + Δp_world                             ... (4) 叠加到参考

朝向:
  ΔR_vr     = R_wrist(t) · R_wrist0^T                        ... (1) FLU 空间旋转增量
  ΔR_vr     = scale_rot(ΔR_vr, s_rot)                        ... (2) 旋转缩放
  ΔR_base   = R_heading · ΔR_vr · R_heading^T                ... (3) 相似变换到 Base
  ΔR_world  = R_base2world · ΔR_base · R_base2world^T       ... (3') Base→World
  R_target  = ΔR_world · R_eef0                              ... (4) 叠加到参考
```

### 6.3 特例：Heading Calibration 后的行为

当 `R_heading = R_z(-θ)` 且 `R_base2world = I` 时：

```
位置简化:
  Δp_world = s_pos · R_z(-θ) · (p_wrist(t) - p_wrist0)
  p_target = p_eef0 + Δp_world

朝向简化:
  ΔR_world = R_z(-θ) · ΔR_vr · R_z(-θ)^T
  R_target = ΔR_world · R_eef0
```

### 6.4 逆映射 (用于理解/调试)

从目标 EEF 位姿反推对应的 VR 手腕位姿（在 `vr_to_base_rot` 非奇异时）：

```
p_wrist(t) = p_wrist0 + (1/s_pos) · R_heading^T · R_base2world^T · (p_target - p_eef0)
```

---

## 7. 数据生命周期

### 7.1 VR 帧数据流

```
HTS SDK (Unity Left)
  │
  │ unity_left_to_flu_*
  ▼
FLU (dict, in child process)
  │
  │ shm.write_vr_frame()
  ▼
SharedMemory (numpy structured array, VR_FRAME_DTYPE)
  │
  │ shm.read_latest_vr() → array_to_vr_frame()
  ▼
FLU (dict, in main process)
  │
  │ arm_mapper.map()
  ▼
World (target pose, numpy arrays)
  │
  │ IK + EMA
  ▼
Joint Space (arm_cmd, 7 floats)
```

### 7.2 VR_FRAME_DTYPE 布局

```
┌─────────────────────┬──────┬──────────────────────────────────┐
│ 字段                 │ 大小  │ 说明                             │
├─────────────────────┼──────┼──────────────────────────────────┤
│ wrist_pos           │ 24B  │ FLU 手腕位置 (3×f8)              │
│ wrist_quat_wxyz     │ 32B  │ FLU 手腕朝向 (4×f8, wxyz)       │
│ landmarks           │ 504B │ FLU 手部关键点 (21×3×f8)        │
│ head_pos            │ 24B  │ FLU 头部位置 (3×f8) ★新增       │
│ head_quat_wxyz      │ 32B  │ FLU 头部朝向 (4×f8, wxyz) ★新增 │
│ recv_ts_ns          │ 8B   │ HTS 接收时间戳 (u64 ns)          │
│ source_ts_ns        │ 8B   │ HTS 源时间戳 (u64 ns)            │
│ sequence_id         │ 8B   │ 帧序号 (u64)                     │
│ source_frame_seq    │ 8B   │ 源帧序号 (u64)                   │
│ local_recv_ns       │ 8B   │ 本地接收时间 (u64 ns)            │
│ side                │ 4B   │ 手侧 (i32: 0=right, 1=left)     │
├─────────────────────┼──────┼──────────────────────────────────┤
│ 合计                 │ ~660B│ (align=True, 含 padding)         │
└─────────────────────┴──────┴──────────────────────────────────┘
```

---

## 8. 关键参数速查

### 8.1 VR 映射

| 参数 | 值 | 位置 |
|------|-----|------|
| `pos_scale` | 1.0 (1:1) | `vr_teleop_arm_only.py` |
| `rot_scale` | 1.0 (1:1) | `vr_teleop_arm_only.py` |
| `max_delta_rot_rad` | 1.0 (~57°) | `arm_mapper.py` |
| `vr_to_base_rot` (默认) | `I` (3×3) | `arm_mapper.py:__init__` |
| `base_to_world_rot` (默认) | `I` (3×3) | `arm_mapper.py:__init__` |

### 8.2 工作空间

| 轴 | 范围 | 位置 |
|----|------|------|
| X | [0.28, 0.72] m | `vr_teleop_arm_only.py` |
| Y | [-0.45, 0.45] m | `vr_teleop_arm_only.py` |
| Z | [0.05, 0.50] m | `vr_teleop_arm_only.py` |

### 8.3 平滑

| 参数 | 值 | 位置 |
|------|-----|------|
| EMA (关节空间) | α=0.6 | `vr_teleop_arm_only.py` |
| 平滑方式 | 关节空间 (非笛卡尔) | 对标 sim `TeleopPipeline` |

### 8.4 时序

| 参数 | 值 | 位置 |
|------|-----|------|
| 外环频率 | 50 Hz | `CTRL_DT = 0.02` |
| VR staleness 阈值 | 0.5 s | `VR_STALE_THRESHOLD_S` |
| 内环频率 | 250 Hz | `ArmInnerLoop` |

---

## 9. 文件索引

| 文件 | 角色 |
|------|------|
| `sensor/vr_receiver_process.py` | VR 子进程: HTS SDK → FLU 转换 → SHM 写入 |
| `shm/layouts.py` | `VR_FRAME_DTYPE` 定义 + 序列化/反序列化 |
| `shm/frame_manager.py` | SharedMemory 读写封装 |
| `teleop/vr/arm_mapper.py` | `ArmWristMapper`: 差分映射 + `set_heading()` |
| `examples/real/vr_teleop_arm_only.py` | 主遥操作脚本: B 键 wiring + 主循环 |
| `planning/ik.py` | DLS 迭代 IK (位置→关节角) |
| `planning/pose_utils.py` | 四元数工具 (wxyz↔xyzw, 归一化) |
| `robot/inner_loop.py` | Arm 内环 PID (250Hz mode 4) |
| `utils/signal_utils.py` | `ema_smooth()` 关节空间平滑 |

---

## 附录 A: Quaternion 惯例

整个系统中四元数使用两种惯例：

| 惯例 | 顺序 | 使用位置 |
|------|------|---------|
| **wxyz** | `(w, x, y, z)` | `transforms3d`, `ArmWristMapper`, SHM, VR frame dict |
| **xyzw** | `(x, y, z, w)` | `scipy.spatial.transform.Rotation` (默认), HTS SDK raw |

转换：`wxyz → xyzw`: `np.roll(q, -1)`  /  `xyzw → wxyz`: `np.roll(q, 1)`

## 附录 B: HTS SDK HandFilter 行为

| `hand_filter` | LEFT 手 | RIGHT 手 | HEAD | 用途 |
|:---|:---:|:---:|:---:|------|
| `"left"` | ✅ | ❌ | ❌ | 仅左手 |
| `"right"` | ❌ | ✅ | ❌ | 仅右手 (旧默认) |
| `"both"` | ✅ | ✅ | ✅ | 双手+头部 (新默认, heading calibration 需要) |

`VRReceiverProcess` 在 `hand_filter="both"` 时会自动过滤 LEFT 手帧，只保留 RIGHT + HEAD。

---

*最后更新: 2025-07-07*
