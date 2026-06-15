# 观测模态与动作模态 — 设计讨论文档

> **定位**: 整理 RobotState / RobotAction / HDF5 的字段变更，逐项对比「当前 CLAUDE.md 定义」vs「拟新增」，标注待讨论的设计决策。
>
> **前提**: 不涉及代码实现细节，只讨论「数据是什么、形状是什么、从哪里来」。

---

## 1. 变更总览

| 维度 | 现状 (CLAUDE.md) | 拟变更 | 影响范围 |
|------|------------------|--------|---------|
| **旋转表示** | `eef_quat_wxyz(4)` 一种 | 新增 `eef_rot6d(6)` 双存 | RobotState, HDF5 /obs |
| **指尖位置** | 无 | 新增 `fingertip_pos(5,3)` | RobotState, HDF5 /obs, kinematics |
| **触觉 raw** | `full=True` 才返回 | `get_state()` 始终返回 raw | XHand.get_state(), RobotState |
| **RobotAction 目标位姿** | 仅 `arm_qpos_cmd` + `hand_qpos_cmd` | 新增 `target_eef_pos(3)` + `target_eef_rot6d(6)` | RobotAction, HDF5 /action |
| **HDF5 /obs** | 无 `eef_rot6d`, `fingertip_pos`, `hand_tactile_raw` | 三个字段全部加入 | EpisodeRecorder, convert_data |
| **HDF5 /action** | 仅 `arm_qpos(7)`, `hand_qpos(12)` | 新增 `eef_pos(3)`, `eef_rot6d(6)` | EpisodeRecorder |
| **策略动作空间** | 仅 joint space (19D) | joint(19D) + eef(21D) 双模式 | ActionParser, ObservationBuilder |
| **QualityFlags** | 未实现 | 11-bit uint16 | 新增 recording/quality_flags.py |

---

## 2. RGB / Depth / 点云 — 当前处理流程

> 本节梳理现有代码中相机数据的完整处理链路，以及 HDF5 录制方案。

### 2.1 当前代码资产

| 文件 | 角色 |
|------|------|
| `sensor/realsense.py` | RealSense 驱动：`start()→read()→CameraFrame→stop()` |
| `utils/pointcloud_utils.py` | 纯函数：`rgbd_to_pointcloud()` 及采样/过滤/变换 |
| `utils/camera_calib.py` | `CameraCalib`：从 `config/calib/cameras.json` 加载内参+外参 |
| `config/calib/cameras.json` | 标定文件 (eye_to_hand 或 eye_in_hand) |

### 2.2 RealSense 数据流（现状）

```
RealSense hardware
  │
  └─ pipeline.wait_for_frames()
       ├─ depth_frame (z16) → depth_raw (uint16)
       │     ├─ depth = depth_raw * depth_scale → float32 (meters)
       │     └─ hole_filling_filter (可选)
       ├─ color_frame (bgr8) → bgr → rgb (uint8, BGR→RGB 已转换)
       └─ align: depth_to_color (默认) → 深度图 warped 到 RGB 视角

CameraFrame:
  .rgb          (H, W, 3)    uint8    RGB
  .depth        (H, W)       float32  meters
  .depth_raw    (H, W)       uint16   raw z16
  .K            (3, 3)       float64  内参矩阵
  .intr         (4,)         float32  [fx, fy, cx, cy]
  .timestamp    scalar       float64  seconds (硬件时间戳)
  .host_time    scalar       float64  host 读取时刻
  .frame_id     int                   帧序号
  .depth_scale  scalar       float64  raw→meters 比例因子
  .align_mode   str                   "depth_to_color" 等
  .frame_name   str                   "camera_color_optical" 等
```

### 2.3 点云生成链路（`rgbd_to_pointcloud`）

```
depth(H,W) + K(3,3) + rgb(H,W,3) + config
  │
  ├─ 1. depth_to_meters()
  ├─ 2. make_rays(K) → rays(H,W,3)
  ├─ 3. depth_to_xyz: rays * depth → (N, 3) 相机坐标系
  ├─ 4. image_to_colors: rgb / 255 → (N, 3) [0,1]
  ├─ 5. filter_points_by_depth(min_depth, max_depth)
  ├─ 6. transform_points(T_out_camera) → 变换到目标坐标系
  ├─ 7. crop_points(workspace) → 裁剪
  ├─ 8. sample_points(npoints, "random"/"fps"/"first")
  └─ 9. pack_xyzrgb → (N, 6) float32
```

**关键参数 (PointCloudConfig)**:
- `npoints`: 采样点数 (默认 1024)
- `min_depth / max_depth`: 深度有效范围 (0.05m ~ 1.5m)
- `sampling`: 采样策略 (random / fps / first / none)
- `workspace`: 3D 裁剪空间 [x_min, y_min, z_min, x_max, y_max, z_max]
- `T_out_camera`: 可选的外参变换（将点云转到 base 或其他坐标系）

### 2.4 CameraCalib 标定管理

```python
calib = CameraCalib("config/calib/cameras.json")

# eye_to_hand (静态相机):
K = calib.get_K("camera_0")                          # (3,3)
T = calib.get_extrinsics("camera_0")                 # T_base_camera (4,4) 静态
meta = calib.to_meta_dict("camera_0")
# → {"camera_serial", "camera_type", "camera_K", "camera_T_base_camera"}

# eye_in_hand (腕载相机):
T = calib.get_extrinsics("camera_wrist", T_base_eef) # T_base_eef @ T_eef_camera 逐帧计算
meta = calib.to_meta_dict("camera_wrist")
# → {"camera_serial", "camera_type", "camera_K", "camera_T_eef_camera"}
```

### 2.5 HDF5 相机存储方案

```
/camera/rgb          (T, H, W, 3)   uint8    逐帧原始 RGB，不压缩
/camera/depth        (T, H, W)      float32  逐帧深度 (meters)
/camera/timestamps   (T,)           float64  硬件时间戳
/camera/K            (3, 3)         float64  内参矩阵
/camera/extrinsics   (T, 4, 4)      float64  逐帧 T_base_camera
```

**存储原则**:
- 不压缩 (JPEG/PNG/H.264 均不使用)，保证训练 dataloader 随机读取效率
- `rgb` 使用 `uint8`（每像素 1 字节，640×480×3 ≈ 0.9MB/帧）
- `depth` 使用 `float32`（每像素 4 字节）
- `extrinsics` 逐帧存储:
  - eye-to-hand: 每帧重复相同值（保证 episode 自包含）
  - eye-in-hand: 逐帧计算 `FK(arm_qpos) @ T_eef_camera`

### 2.6 Episode 自包含原则

- Episode 文件不依赖外部标定文件
- `/meta/camera_K` 提供快速读取的内参标量值 `[fx, fy, cx, cy]`
- `/camera/K` 提供 3x3 矩阵格式（方便预处理脚本直接使用）
- `/camera/extrinsics` 逐帧存储完整外参

### 2.7 点云的定位 — 派生数据，不存储

点云是 RGB + Depth + K + extrinsics 的派生数据。HDF5 只存原始图像+内外参，训练 dataloader 按需生成点云。

**理由**:
- 点云 (N, 6) 中 N 取决于采样策略，格式不固定
- 从原始 rgb+depth 恢复点云无损，不存在信息丢失
- 不同的策略可能需要不同的点云参数 (npoints, workspace, sampling)

### 2.8 参考项目分辨率/点云参数对比

| 参数 | LeFranX (P1) | ManiUniCon (P1) | Open-Teach (P2) | DexUMI (P2) | **本项目** |
|------|:---:|:---:|:---:|:---:|:---:|
| **RGB 分辨率** | 320×240 | 1280×720 或 640×480 | 1280×720 | 640×480 (RS), 1280×800 (OAK) | **640×480** |
| **Depth 分辨率** | N/A | 与 RGB 相同 | 与 RGB 相同 | 与 RGB 相同 | **与 RGB 相同** |
| **FPS** | 30 | 15–30 | 30 | 30 (RS), 60 (OAK) | **30** |
| **相机型号** | USB 摄像头 | D435/D455 / L515 | 未指定 | 未指定 / OAK | **L515** |
| **对齐模式** | N/A | depth_to_color | depth_to_color | depth_to_color | **depth_to_color** |
| **点云 npoints** | N/A | 1200 / 2048 | N/A | 全分辨率 | **1024** |
| **点云 workspace** | N/A | [0.2,1.03,-1.2,1.2,-0.3,0.7] | N/A | N/A | **待定** |
| **采样策略** | N/A | FPS / uniform | N/A | 不采样 | **random** |
| **标定格式** | N/A | YAML (fx,fy,cx,cy+pose) | Runtime API | Runtime API | **JSON (cameras.json)** |

**关键发现**:
- **所有 RealSense 项目统一用 `depth_to_color`** 对齐模式 — 本项目保持一致
- **ManiUniCon 是本项目点云 pipeline 的主要参考** — workspace bounds、npoints、FPS 采样都来自它
- **分辨率选择**: 640×480 是安全底线（ManiUniCon 低配、DexUMI 默认），1280×720 需要 2.25 倍存储和 dataloader 吞吐。当前本项目的 `RealSenseConfig` 已默认 640×480
- **点云参数**: 当前 `PointCloudConfig` 默认 `npoints=1024`、`min_depth=0.05`、`max_depth=1.5`、`sampling="random"`，处于参考项目参数范围的中位
- **BunnyVisionPro 无相机**，纯 VR 项目，已从对比中移除

### 🟡 待讨论: 相机相关问题

1. **RealSense 接口命名**: 当前使用 `start()/stop()`，CLAUDE.md 要求 `connect()/disconnect()`。是否需要统一？
2. **depth_raw (z16) 存储**: 当前 CameraFrame 有 depth_raw 字段。HDF5 是否需要存储？恢复 float32 depth 时可能丢失精度（但差异极小）
3. **多相机支持**: 当前 CameraCalib 支持多相机配置，但 RealSense 一次只能连一个。EpisodeRecorder 需要支持多相机源
4. **分辨率选择**: 当前默认 640×480@30fps，是否需要 1280×720？ManiUniCon 两种都支持（通过不同 YAML profile 切换）

---

## 3. RobotState 字段逐项讨论

### 3.1 现状 (CLAUDE.md Section 2.8)

```
arm_qpos(7)  arm_qvel(7)  arm_tau(7)
eef_pos(3)   eef_quat_wxyz(4)
hand_qpos(12)  hand_current(12)  hand_tactile_sum(5,3)  hand_temperature(12)
arm_connected  hand_connected  hand_error  timestamp
```

**共计 13 个字段。**

### 3.2 拟新增字段

#### (A) `eef_rot6d` — EEF 旋转的 rot6d 表示

| 属性 | 值 |
|------|-----|
| Shape | `(6,)` float64 |
| 来源 | `quat_wxyz_to_rot6d(eef_quat)` — 函数已在 `pose_utils.py` 实现 |
| 动机 | rot6d 是策略模型的标准旋转输入（连续、无四元数双覆盖歧义），避免训练时逐帧转换 |

**讨论**:
- 同时存 quat + rot6d 增加约 10 个 float64/帧 (80 字节)，对于 HDF5 来说可以忽略
- quat 用于几何运算（IK/坐标变换），rot6d 直接喂策略 — 两个消费者的需求不同
- **待确认**: 是否需要在 RobotState 中做实时转换？还是只在 HDF5 存储层做？

#### (B) `fingertip_pos` — 5 指尖世界坐标

| 属性 | 值 |
|------|-----|
| Shape | `(5, 3)` float64 |
| 来源 | 链式 FK: `T_world_eef @ T_eef_handbase @ T_handbase_fingertip(hand_qpos)` |
| 动机 | 策略需要指尖空间感知（精细操作任务的关键先验） |

**链式 FK 路径**:

```
arm_base
  └─ [arm_qpos + MPlib URDF] → EEF frame (T_world_eef)
       └─ [T_eef_handbase — 静态标定] → hand_base
            └─ [hand_qpos + Hand URDF] → fingertip_i (T_handbase_fingertip_i)
```

**⚠️ 需要讨论的关键问题**:
1. **Hand URDF 能否用于真实 XHand**？当前 `assets/robots/xhand/xhand_right.urdf` 存在，但需要验证:
   - URDF 中 joint 顺序是否与 XHand SDK 的 12 个关节位置一致？
   - SAPIEN 模型中 joint 名称为 `right_hand_thumb_rota_joint1` 等，与 XHand SDK 的 `thumb_joint1` 映射关系需要确认
2. **`T_eef_handbase` 标定值从哪来**？目前还是手工测量/标定，需要有配置入口
3. **计算开销**: Pinocchio FK 对 12-DOF hand 模型约 <0.1ms，50Hz 控制循环可接受
4. **容错**: 如果手部 URDF 不可用/未提供，`fingertip_pos` 返回 NaN

#### (C) `hand_tactile_raw` — 原始触觉阵列

| 属性 | 值 |
|------|-----|
| Shape | `(5, 120, 3)` float64 |
| 来源 | XHand SDK `sensor_data[i].raw_force` — 每次 `read_state()` 同时返回 |

**现状**: 当前 XHand `get_state(full=False)` 不返回触觉数据，`full=True` 才返回。但 SDK 的 `read_state()` 调用是一次性返回所有数据（关节 + 触觉），筛选不减少 SDK 开销。

**建议**: `RobotInterface.get_state()` 始终读取 hand `full=True`（即触觉 raw 始终可用），不做筛选。

**⚠️ 讨论**: `(5,120,3)` = 1800 个 float64 = 14.4 KB/帧。在 50Hz 遥操作录制中，100s episode = 5000 帧 × 14.4KB = 72MB。这个大小是否可接受？还是只在 certain 场景（精密操作）才录？

---

## 4. RobotAction 字段变更

### 4.1 现状

```python
RobotAction:
    arm_qpos_cmd: np.ndarray   # (7,)  rad  — 发给硬件的最终关节命令
    hand_qpos_cmd: np.ndarray  # (12,) rad  — 发给硬件的最终关节命令
```

### 4.2 拟变更为

```python
RobotAction:
    arm_qpos_cmd: np.ndarray             # (7,)  rad
    hand_qpos_cmd: np.ndarray            # (12,) rad
    target_eef_pos: np.ndarray | None    # (3,)  m    — IK 前输入的 EEF 位置
    target_eef_rot6d: np.ndarray | None  # (6,)       — IK 前输入的 EEF 旋转
```

**语义**:
- `arm_qpos_cmd` / `hand_qpos_cmd`: 经过 joint limit + delta limit 的最终硬件命令（始终存在）
- `target_eef_pos` / `target_eef_rot6d`: IK 求解前的目标位姿，来源:
  - 遥操作时 → `ArmWristMapper.map()` 的输出（VR wrist → EEF）
  - 部署 eef 模式 → 策略模型输出
  - 部署 joint 模式 → `None`（策略直接输出关节位置）

**讨论**:
- `ArmWristMapper.map()` 当前返回 `{"pos", "quat_wxyz"}`，需要同样的 rot6d 转换
- **待确认**: 为什么用 rot6d 而非 quat_wxyz 存 target？A: 与策略输出格式一致（策略输出 rot6d）
- **或者**: 存 quat_wxyz 更自然（ArmWristMapper 直接产出四元数），由 ActionParser 转换

---

## 5. HDF5 Schema 变更

### 5.1 /obs 分组 — 新增 3 个数据集

| 数据集 | Shape | 说明 |
|--------|-------|------|
| `eef_rot6d` **(新)** | `(T, 6)` float64 | 与 eef_quat_wxyz 双存 |
| `fingertip_pos` **(新)** | `(T, 5, 3)` float64 | 链式 FK 计算 |
| `hand_tactile_raw` **(新)** | `(T, 5, 120, 3)` float64 | 原始触觉阵列 |

### 5.2 /action 分组 — 新增 2 个数据集

| 数据集 | Shape | 说明 |
|--------|-------|------|
| `eef_pos` **(新)** | `(T, 3)` float64 | EEF 目标位置（IK 输入） |
| `eef_rot6d` **(新)** | `(T, 6)` float64 | EEF 目标旋转（IK 输入） |

### 5.3 /meta 属性 — 新增

| 属性 | 说明 |
|------|------|
| `action_space` | `"joint"` 或 `"eef"`，标记本 episode 的监督类型 |

### 5.4 相机存储策略

| 字段 | 存储格式 | 说明 |
|------|---------|------|
| `rgb` | 逐帧 uint8 | 不压缩，保证随机读取效率 |
| `depth` | 逐帧 float32 | — |
| `extrinsics` | 逐帧 `(4,4)` float64 | eye-to-hand 重复相同值，eye-in-hand 逐帧 FK |

---

## 6. 新增模块概览

### 6.1 已部分实现

| 文件 | 状态 | 内容 |
|------|------|------|
| `robot/robot_interface.py` | ⚠️ 草稿 | RobotState, RobotAction, RobotInterface, HandKinematics |
| `recording/quality_flags.py` | ✅ 完成 | 11-bit QualityFlags |

### 6.2 待实现

| 文件 | 依赖 | 职责 |
|------|------|------|
| `recording/episode_recorder.py` | robot_interface, quality_flags | HDF5 录制：start_episode → add_frame → stop_episode |
| `data/episode_reader.py` | h5py | 懒加载读取，iter_frames(skip_rejected) |
| `data/data_validator.py` | episode_reader | 检查 nan/inf/shape/时间戳/关节范围/电流 |
| `deploy/observation_builder.py` | robot_interface | RobotState+CameraFrame → 归一化 obs dict |
| `deploy/action_parser.py` | robot_interface, pose_utils | 策略输出 → RobotAction (joint/eef 双模式) |
| `deploy/safety_monitor.py` | robot_interface | workspace/限位/力矩/电流/温度/通信 检查 |
| `deploy/policy_loader.py` | — | 从 checkpoint 目录加载模型+norm_stats+config |
| `deploy/policy_runner.py` | 以上所有 | 部署主循环：obs→model→action→smooth→safety→send |
| `controller/teleop_controller.py` | robot_interface, teleop/* | 遥操作主循环（_tick + 状态机） |

---

## 7. 数据流对比

### 7.1 遥操作录制（现状 vs 拟变更）

```
现状:
  VR wrist → ArmWristMapper → target_pose → IK → arm_qpos_cmd
  VR landmarks → XHandRetargeter → hand_qpos_cmd
  RobotAction(arm_qpos_cmd, hand_qpos_cmd) → send_action
  RobotState(arm_qpos, eef_quat, hand_qpos, ...) → Recorder

拟变更:
  VR wrist → ArmWristMapper → target_pos + rot6d ─┐
                                                   ├→ IK → arm_qpos_cmd
  VR landmarks → XHandRetargeter → hand_qpos_cmd ─┘
  RobotAction(arm_qpos_cmd, hand_qpos_cmd, target_pos, target_rot6d)
  RobotState(arm_qpos, eef_quat, eef_rot6d, fingertip_pos, hand_qpos, tactile_raw, ...)
```

**关键差异**: 
- RobotAction 携带 EEF 目标位姿信息（供 EEF-space 策略训练）
- RobotState 增加 rot6d + fingertip_pos + tactile_raw（供策略输入）

### 7.2 策略部署

```
RobotState → ObservationBuilder.build() → obs dict → model.predict(obs)
  → ActionParser.parse(raw_output, state, mode) → RobotAction
  → EMA smooth(alpha=0.5) → SafetyMonitor.check() → send_action
```

---

## 8. 待讨论决策点

### 🔴 决策 1: 相机相关问题

见 [Section 2.7 🟡 待讨论](#-待讨论-相机相关问题):
1. RealSense 接口命名统一 (`start/stop` → `connect/disconnect`)
2. `depth_raw` (z16) 是否需要存入 HDF5
3. 多相机支持方案

### 🔴 决策 2: fingertip_pos 的可行性

- Hand URDF 的 joint 名/顺序是否与真实 XHand SDK 返回的 12 DOF 顺序一致？
- 手部 URDF 的 base link 坐标轴朝向是否与 EEF 法兰一致？
- **建议**: 先做一个 smoke test — 加载 `xhand_right.urdf`，输入一组 hand_qpos，验证手指末端位置是否在合理范围（5-15cm 量级）

### 🔴 决策 3: hand_tactile_raw 的存储成本

- `(5,120,3)` float64 = 14.4KB/帧
- 100s episode @50Hz = 5000 帧 = 72MB（仅触觉 raw）
- 是否需要压缩？是否需要可选存储（默认不录 raw，通过配置开关）？

### 🔴 决策 4: RobotAction 中 target_eef 用什么旋转格式？

- **方案 A**: `target_eef_rot6d` — 与策略输出一致
- **方案 B**: `target_eef_quat_wxyz` — ArmWristMapper 直接产出四元数，无需转换
- **方案 C**: 同时存两个（和 RobotState 一样）— 冗余但灵活
- **建议**: 方案 B — 遥操作场景用 quat，ActionParser 内部转 rot6d 供训练

### 🟡 决策 5: XHand.get_state() 的 full 参数还需要吗？

- 目前 `RobotInterface.get_state()` 总是调 `hand.get_state(full=True)`
- 如果 RobotInterface 是唯一调用方，XHand 内部的 full 参数可以简化

### 🟡 决策 6: 手部 URDF 路径配置

- 放在哪里？`config/calib/` 还是 `RobotInterfaceConfig` 里？
- `T_eef_handbase` 的标定方式？需要写一个标定工具吗？

---

## 9. 建议实施顺序

```
Phase 1: 类型定义（无硬件依赖）
  1. recording/quality_flags.py        ✅ 已完成
  2. robot/robot_interface.py          ⚠️ 草稿，待讨论后修改
     - RobotState (新增 eef_rot6d, fingertip_pos, hand_tactile_raw)
     - RobotAction (新增 target_eef_pos, target_eef_quat)

Phase 2: 数据层（纯文件 I/O，无硬件依赖）
  3. data/episode_reader.py
  4. data/data_validator.py
  5. recording/episode_recorder.py

Phase 3: 部署层（依赖 RobotInterface 接口，可独立测试）
  6. deploy/observation_builder.py
  7. deploy/action_parser.py
  8. deploy/safety_monitor.py
  9. deploy/policy_loader.py
  10. deploy/policy_runner.py

Phase 4: 控制器（依赖真实硬件 + teleop 模块）
  11. controller/teleop_controller.py
```

---

## 10. 当前代码现状速查

| 文件 | 关键接口 | 变更需求 |
|------|---------|---------|
| `robot/xarm7.py` | `get_state()→dict`, `send_action(np.ndarray)→bool` | 无需改动（RobotInterface 内部调用） |
| `robot/xhand.py` | `get_state(full)→dict`, `send_action(np.ndarray)→bool` | 无需改动 |
| `planner/kinematics.py` | `compute_eef_pose_world()→Pose` | 无需改动（FK 已完备） |
| `planner/pose_utils.py` | `rot6d_to_quat_wxyz()`, `quat_wxyz_to_rot6d()` | ✅ 已实现，无需改动 |
| `planner/workspace_safety.py` | `check()`, `clamp()` | ✅ 已实现，无需改动 |
| `teleop/arm_wrist_mapper.py` | `map()→{"pos","quat_wxyz"}` | 可能需要新增 rot6d 输出 |
| `teleop/hand_retarget.py` | `retarget(landmarks)→np.ndarray` | 无需改动 |
