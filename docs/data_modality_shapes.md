# 数据模态形状 — 最终规范

> 综合参考项目实践（LeFranX / ManiUniCon / DexUMI）、社区惯例（ImageNet 预训练、PointNet++）和本项目硬件约束后确定。
> 本文档是最终答案，CLAUDE.md 和 obs_action_design.md 中与此冲突的条目以此文档为准。

---

## 1. 相机数据

### 1.1 RGB

| 阶段 | 形状 | dtype | 范围 | 说明 |
|------|------|-------|------|------|
| **硬件输出** | `(480, 640, 3)` | uint8 | [0, 255] | RealSense L515 @30fps, BGR→RGB 在 CameraFrame 中已转换 |
| **HDF5 存储** | `(T, 480, 640, 3)` | uint8 | [0, 255] | 逐帧原始存储，不压缩 |
| **模型输入** | `(3, 224, 224)` | float32 | [0, 1] 或 ImageNet 归一化 | 推理时 dataloader 做 resize + normalize |

**Resize 策略**: `cv2.resize(..., (224, 224), interpolation=cv2.INTER_LINEAR)`

**选 224 的理由**:
- ManiUniCon 所有 7 个策略模型统一使用 224×224
- DexUMI 使用 240→224 center crop
- ImageNet 预训练模型（ResNet/ViT）标准输入是 224×224
- LeFranX 的 128×128 是针对 RL 的低配方案，不适合模仿学习

**为什么不存储时就 resize 到 224**:
- 存储原始 640×480 保留最大信息，离线分析可用全部细节
- 不同策略可能需要不同分辨率/裁剪策略
- DexUMI 的做法（240→224 crop）在 dataloader 层做，灵活可配
- 640×480@uint8 = 0.9MB/帧，可控

### 1.2 Depth

| 阶段 | 形状 | dtype | 范围 | 说明 |
|------|------|-------|------|------|
| **硬件输出** | `(480, 640)` | float32 | meters | raw z16 × depth_scale |
| **HDF5 存储** | `(T, 480, 640)` | float32 | meters | 逐帧存储 |
| **模型输入** | `(1, 224, 224)` | float32 | meters (clip 后) | resize 用 INTER_NEAREST |

**Resize 策略**: `cv2.resize(..., (224, 224), interpolation=cv2.INTER_NEAREST)`

**RGB vs Depth 关键差异**:
| | RGB | Depth |
|--|-----|-------|
| 插值 | INTER_LINEAR | **INTER_NEAREST** — 防止深度边缘插值产生 ghost 值 |
| 通道 | 3 | 1 |
| 归一化 | /255 或 ImageNet | clip [0.01, 6.0] 后保持原始 meters |
| 预处理 | 无 | clip + 可能 1/depth 变换（参考 ManiUniCon CDM） |

---

## 2. 点云

### 2.1 生成参数

```python
PointCloudConfig(
    npoints=1024,           # 每帧采样点数（PointNet++ 标准：1024 或 2048）
    min_depth=0.05,          # m，过滤太近的点
    max_depth=1.5,           # m，过滤太远的点（与 workspace 匹配）
    sampling="random",       # 实时控制用 random（快），离线训练可改 fps
    workspace=None,          # 3D 裁剪区域，与 workspace_safety 共用
    device="cpu",            # 实时控制用 cpu，训练 dataloader 可上 GPU
    return_tensor=True,      # torch.Tensor 便于后续操作
)
```

**npoints 选 1024 的理由**:
- ManiUniCon FalconPCD 使用 2048，但通用策略用 1200
- PointNet++ 论文标准是 1024
- 1024 点 × (3+3) = 6144 floats，模型输入可控
- 从 640×480 = 307K 像素中采样 1024 点，采样率约 0.3%，足够覆盖工作空间

### 2.2 点云在 HDF5 中不存储

点云是 RGB + Depth + K + extrinsics 的派生数据。HDF5 只存原始图像+内外参，训练 dataloader 按需生成。

---

## 3. RobotState — 完整定义

```python
@dataclass
class RobotState:
    # ── Arm 关节传感器 (XArm7 SDK) ──
    arm_qpos: np.ndarray          # (7,)    float64  rad      arm.angles
    arm_qvel: np.ndarray          # (7,)    float64  rad/s    arm.get_state() velocities
    arm_tau: np.ndarray           # (7,)    float64  N·m      实为电机电流

    # ── EEF 位姿 (Pinocchio FK, 双表示) ──
    eef_pos: np.ndarray           # (3,)    float64  m        FK 计算
    eef_quat_wxyz: np.ndarray     # (4,)    float64           FK 计算, 几何运算用
    eef_rot6d: np.ndarray         # (6,)    float64           策略模型输入

    # ── Hand 关节传感器 (XHand SDK) ──
    hand_qpos: np.ndarray         # (12,)   float64  rad      finger_state[i].position
    hand_current: np.ndarray      # (12,)   float64  mA       finger_state[i].torque

    # ── 触觉 (XHand SDK, 始终返回) ──
    hand_tactile_sum: np.ndarray  # (5,3)   float64  N        指尖 calc_force (fx,fy,fz)
    hand_tactile_raw: np.ndarray  # (5,120,3) float64         原始 ADC, 每 taxel 3 轴
    hand_temperature: np.ndarray  # (12,)   float64  °C

    # ── 派生: 指尖世界坐标 (链式 FK) ──
    fingertip_pos: np.ndarray     # (5,3)   float64  m        world frame

    # ── 状态标志 ──
    arm_connected: bool
    hand_connected: bool
    hand_error: bool
    timestamp: float              # time.perf_counter()

# 本体感知总维度 (不含触觉 raw): 7+7+7+3+4+6+12+12+(5*3)+(5*3)+5*3 = 101
# 含触觉 raw: 101 + 1800 = 1901
```

**设计决策**:
- `eef_rot6d` 和 `eef_quat_wxyz` 双存：quat 供 IK/几何运算，rot6d 直接喂策略（参考 Zhou et al. 2019，DexUMI 未使用 rot6d 但社区主流做法如此）
- `hand_tactile_raw` 始终返回，因为 SDK 一次读取已包含全部数据，筛选不减少开销
- `fingertip_pos` 通过 Pinocchio hand FK 计算，需要 hand URDF + `T_eef_handbase` 标定

---

## 4. RobotAction — 完整定义

```python
@dataclass
class RobotAction:
    # 硬件命令 (始终存在, 经过 joint limit + delta limit)
    arm_qpos_cmd: np.ndarray             # (7,)    float64  rad
    hand_qpos_cmd: np.ndarray            # (12,)   float64  rad

    # EEF 目标 (可选, IK 前输入)
    target_eef_pos: np.ndarray | None    # (3,)    float64  m
    target_eef_quat_wxyz: np.ndarray | None  # (4,) float64
```

**为什么用 `target_eef_quat_wxyz` 而非 `target_eef_rot6d`**:
- 遥操作时 ArmWristMapper 直接产出四元数，无需转换
- 部署 eef 模式时策略输出 rot6d → ActionParser 内部转 quat → IK
- HDF5 中存储 quat：离线几何验证更自然
- 训练 EEF-space 策略时 dataloader 把 quat → rot6d

---

## 5. HDF5 录制 Schema — 最终版

```
episode_XXX.h5
│
├─ /meta/ attributes:
│   task_label, operator, tags, duration, fps,
│   num_frames, num_valid_frames, success,
│   camera_serial, camera_type,           # "eye_to_hand" | "eye_in_hand"
│   camera_K: (4,) float64,              # [fx, fy, cx, cy]
│   camera_T_base_camera: (16,) float64, # flat 4x4 (eye-to-hand)
│   或 camera_T_eef_camera: (16,),       # flat 4x4 (eye-in-hand)
│   action_space: "joint" | "eef",
│   retargeting_config: str,             # JSON
│   pipeline_snapshot: str,              # JSON
│
├─ /obs/
│   arm_qpos            (T, 7)         float64  rad
│   arm_qvel            (T, 7)         float64  rad/s
│   arm_tau             (T, 7)         float64  N·m
│   eef_pos             (T, 3)         float64  m
│   eef_quat_wxyz       (T, 4)         float64
│   eef_rot6d           (T, 6)         float64           ← 新增
│   hand_qpos           (T, 12)        float64  rad
│   hand_current        (T, 12)        float64  mA
│   hand_tactile_sum    (T, 5, 3)      float64  N
│   hand_tactile_raw    (T, 5, 120, 3) float64           ← 新增
│   hand_temperature    (T, 12)        float64  °C
│   fingertip_pos       (T, 5, 3)      float64  m        ← 新增
│
├─ /action/
│   arm_qpos            (T, 7)         float64  rad
│   hand_qpos           (T, 12)        float64  rad
│   eef_pos             (T, 3)         float64  m        ← 新增
│   eef_quat_wxyz       (T, 4)         float64           ← 新增
│
├─ /vr/
│   wrist_pos           (T, 3)         float64  m
│   wrist_quat_wxyz     (T, 4)         float64
│   landmarks           (T, 21, 3)     float64
│
├─ /camera/
│   rgb                 (T, 480, 640, 3)  uint8           ← 480p 原始
│   depth               (T, 480, 640)     float32  m      ← 480p 原始
│   timestamps          (T,)              float64
│   K                   (3, 3)            float64         内参矩阵
│   extrinsics          (T, 4, 4)         float64         逐帧 T_base_camera
│
└─ /quality_flags       (T,)           uint16  11-bit
```

---

## 6. 策略层数据形状

### 6.1 观测 (ObservationBuilder 输出)

```python
# 归一化后的 dict，策略模型 forward 按需取用
obs = {
    # ── 本体感知 (均归一化到 zero-mean unit-variance) ──
    "arm_qpos":       np.ndarray,  # (7,)    float32
    "eef_pos":        np.ndarray,  # (3,)    float32
    "eef_rot6d":      np.ndarray,  # (6,)    float32
    "hand_qpos":      np.ndarray,  # (12,)   float32
    "fingertip_pos":  np.ndarray,  # (5,3)   float32  (可 flatten 到 15)
    "hand_tactile_sum": np.ndarray, # (5,3)   float32  (可 flatten 到 15)

    # ── 视觉 (归一化后) ──
    "rgb":            torch.Tensor, # (3, 224, 224)  float32  [0,1] 或 ImageNet norm
    "depth":          torch.Tensor, # (1, 224, 224)  float32  meters (clip 后)

    # ── 点云 (从 rgb+depth 实时生成) ──
    "pointcloud":     torch.Tensor, # (1024, 6)  float32  xyz+rgb, 在 workspace 内
}
```

### 6.2 动作 (策略输出 → ActionParser → RobotAction)

**Joint space** (action_space="joint", 19D):
```
[arm_qpos(7), hand_qpos(12)]  → 反归一化 → RobotAction
```

**EEF space** (action_space="eef", 21D):
```
[eef_pos(3), eef_rot6d(6), hand_qpos(12)]  → 反归一化 → rot6d→quat → IK(eef_pose, current_qpos) → arm_qpos(7) → RobotAction
```

---

## 7. 形状汇总表

| 数据 | 存储形状 | 模型输入形状 | dtype |
|------|---------|------------|-------|
| RGB | `(T, 480, 640, 3)` | `(3, 224, 224)` | uint8 → float32 |
| Depth | `(T, 480, 640)` | `(1, 224, 224)` | float32 |
| Point cloud | 不存储 | `(1024, 6)` | float32 |
| arm_qpos | `(T, 7)` | `(7,)` | float64 → float32 |
| arm_qvel | `(T, 7)` | 训练可选 | float64 |
| arm_tau | `(T, 7)` | 训练可选 | float64 |
| eef_pos | `(T, 3)` | `(3,)` | float64 → float32 |
| eef_quat_wxyz | `(T, 4)` | 不直接喂策略 | float64 |
| eef_rot6d | `(T, 6)` | `(6,)` | float64 → float32 |
| hand_qpos | `(T, 12)` | `(12,)` | float64 → float32 |
| hand_current | `(T, 12)` | 训练可选 | float64 |
| hand_tactile_sum | `(T, 5, 3)` | `(5,3)` 或 flatten `(15,)` | float64 → float32 |
| hand_tactile_raw | `(T, 5, 120, 3)` | `(5, 120, 3)` 专用编码器 | float64 → float32 |
| hand_temperature | `(T, 12)` | 训练可选 | float64 |
| fingertip_pos | `(T, 5, 3)` | `(5,3)` 或 flatten `(15,)` | float64 → float32 |
| action arm_qpos | `(T, 7)` | — | float64 |
| action hand_qpos | `(T, 12)` | — | float64 |
| action eef_pos | `(T, 3)` | — | float64 |
| action eef_quat_wxyz | `(T, 4)` | — | float64 |
| quality_flags | `(T,)` uint16 | — | uint16 |

---

## 8. 关键设计决策汇总

| 决策 | 结论 | 参考依据 |
|------|------|---------|
| RGB 模型输入尺寸 | **224×224** | ManiUniCon 全策略统一, ImageNet 预训练标准 |
| Depth 模型输入尺寸 | **224×224** | 与 RGB 对齐，INTER_NEAREST 插值 |
| RGB 存储尺寸 | **480×640** (原始) | 保留完整信息，dataloader 灵活 resize |
| Depth 存储尺寸 | **480×640** (原始) | 同上 |
| 点云不存 HDF5 | ✅ | 派生数据，dataloader 实时生成 |
| 点云 npoints | **1024** | PointNet++ 标准，ManiUniCon 1200-2048 之间 |
| 旋转双表示 | quat(几何) + rot6d(策略) | Zhou et al. 2019，策略训练更稳定 |
| Action 存 quat 不存 rot6d | ✅ | 遥操作为原生 quat，dataloader 按需转 rot6d |
| hand_tactile_raw 始终返回 | ✅ | SDK 无额外开销 |
| depth 存储 float32 不存 raw z16 | ✅ | 精度差异可忽略 |
