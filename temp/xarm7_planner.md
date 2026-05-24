# xArm7 ArmMotionPlanner 使用说明

本文档说明 `ArmMotionPlanner` 的安装、接口定义、典型使用方式和部署注意事项。该 planner 面向 xArm7 这类 7 自由度冗余机械臂，在 MPlib 基础上增加了：

- world/base frame 位姿转换
- 多 seed IK 搜索
- 等效关节 `2π` 分支选择
- 近距离 screw motion 优先尝试
- IK + joint-space RRT fallback
- 路径 unwrap 后重新 TOPP 时间参数化
- 起点碰撞检查、终点位姿校验、路径质量筛选

---

## 1. 安装

### 1.1 Python 依赖

建议在虚拟环境中安装：

```bash
pip install mplib numpy transforms3d
```

MPlib 是独立于 ROS 的 Python 运动规划库，可用于碰撞自由路径规划、逆运动学和点云环境建模。当前 planner 依赖：

```python
import mplib as mp
import numpy as np
from transforms3d.quaternions import qconjugate, qmult, qnorm, rotate_vector
```

### 1.2 文件放置

推荐文件结构：

```text
your_project/
├── arm_motion_planner.py
├── planner_utils.py
├── assets/
│   ├── xarm7.urdf
│   └── xarm7.srdf
└── examples/
    └── plan_xarm7.py
```

如果当前文件仍命名为：

```text
arm_motion_planner_clean.py
planner_utils_clean.py
```

建议部署前重命名为：

```text
arm_motion_planner.py
planner_utils.py
```

否则需要同步修改：

```python
from planner_utils import ...
```

---

## 2. 核心类

```python
class ArmMotionPlanner:
    ...
```

该类封装了 MPlib `Planner`，对 xArm7 的 7 维 planning joints 做了专门处理。初始化时会检查 planning joint 数量是否为 7。

### 2.1 构造函数

```python
def __init__(
    self,
    urdf_path: str | Path,
    srdf_path: str | Path,
    move_group: str,
    joint_vel_limits: np.ndarray | None = None,
    joint_acc_limits: np.ndarray | None = None,
    base_pose: mp.Pose | None = None,
    use_convex: bool = False,
    equivalent_joint_indices: list[int] | None = None,
):
```

参数说明：

| 参数 | 类型 | 说明 |
|---|---|---|
| `urdf_path` | `str | Path` | 机械臂 URDF 文件路径 |
| `srdf_path` | `str | Path` | 机械臂 SRDF 文件路径 |
| `move_group` | `str` | MPlib move group 名称，通常是末端执行器 link 对应的 group |
| `joint_vel_limits` | `np.ndarray | None` | 关节速度限制，shape 为 `(7,)`；传给 MPlib TOPPRA 时间参数化 |
| `joint_acc_limits` | `np.ndarray | None` | 关节加速度限制，shape 为 `(7,)`；传给 MPlib TOPPRA 时间参数化 |
| `base_pose` | `mp.Pose | None` | 机械臂 base 在 world frame 下的位姿；默认位于世界原点 |
| `use_convex` | `bool` | 是否使用 convex collision geometry |
| `equivalent_joint_indices` | `list[int] | None` | 允许 `2π` 等效分支处理的关节索引；默认自动选择 joint range 大于 `2π` 的关节 |

示例：

```python
from pathlib import Path
import mplib as mp
import numpy as np

from arm_motion_planner import ArmMotionPlanner

planner = ArmMotionPlanner(
    urdf_path=Path("assets/xarm7.urdf"),
    srdf_path=Path("assets/xarm7.srdf"),
    move_group="link7",
    joint_vel_limits=np.ones(7) * 1.0,
    joint_acc_limits=np.ones(7) * 2.0,
)
```

---

## 3. 函数定义

### 3.1 位姿与状态函数

#### `set_base_pose`

```python
def set_base_pose(self, base_pose: mp.Pose):
```

设置机械臂 base 在 world frame 下的位姿，并同步到内部 MPlib planner。

```python
planner.set_base_pose(mp.Pose(
    p=np.array([0.0, 0.0, 0.0]),
    q=np.array([1.0, 0.0, 0.0, 0.0]),
))
```

#### `canonicalize_qpos`

```python
def canonicalize_qpos(self, qpos: np.ndarray, strict: bool = True) -> np.ndarray:
```

将输入 qpos 转成合法 joint representation，并调用 MPlib 的 joint limit wrapping。该函数会 copy 输入，避免原地修改调用方数组。

#### `validate_qpos_limits`

```python
def validate_qpos_limits(self, qpos: np.ndarray, tolerance: float = 1e-6) -> bool:
```

检查 qpos 是否在 joint limit 内。

#### `fk_world`

```python
def fk_world(self, qpos: np.ndarray) -> mp.Pose:
```

计算当前 qpos 下末端执行器在 world frame 下的位姿。

```python
current_pose = planner.fk_world(current_qpos)
```

#### `validate_final_pose`

```python
def validate_final_pose(
    self,
    final_qpos: np.ndarray,
    goal_pose_world: mp.Pose,
    pos_threshold: float,
    rot_threshold: float,
) -> tuple[bool, float, float]:
```

检查最终关节角对应的 FK 是否真正到达目标 world pose。

返回：

```python
(valid, pos_err, rot_err)
```

其中 `rot_err` 单位为弧度。

#### `check_current_state`

```python
def check_current_state(self, current_qpos: np.ndarray) -> dict[str, Any] | None:
```

检查当前状态是否满足：

- joint limit
- self collision free
- environment collision free

若状态合法，返回 `None`；否则返回 planning failure dict。

---

### 3.2 IK 函数

#### `ik_world`

```python
def ik_world(
    self,
    goal_pose_world: mp.Pose,
    current_qpos: np.ndarray,
    threshold: float = 1e-3,
    pos_threshold: float | None = None,
    rot_threshold: float = float(np.deg2rad(0.5)),
    random_seed_count: int = 16,
    max_candidates: int = 8,
    score_mode: str = "nearest",
    verbose: bool = False,
) -> list[np.ndarray]:
```

在 world frame 下求目标位姿的多个 IK 候选。

主要逻辑：

1. 将 world goal pose 转到 base frame。
2. 使用 current qpos、last goal qpos、joint center、等效 `2π` 分支和随机 seed 做 IK。
3. 对 IK 解做 self collision / env collision 过滤。
4. 使用 FK 验证 position / rotation error。
5. 对候选解排序并去重。

`score_mode`：

| 模式 | 说明 |
|---|---|
| `nearest` | 优先接近当前 qpos |
| `continuous` | 优先接近上一目标分支，适合连续 servo/task sequence |
| `margin` | 优先远离 joint limit |

#### `ik_best_world`

```python
def ik_best_world(
    self,
    goal_pose_world: mp.Pose,
    current_qpos: np.ndarray,
    threshold: float = 1e-3,
    pos_threshold: float | None = None,
    rot_threshold: float = float(np.deg2rad(0.5)),
    random_seed_count: int = 64,
    score_mode: str = "nearest",
    verbose: bool = False,
) -> np.ndarray | None:
```

返回单个最优 IK 解；如果没有解，返回 `None`。

---

### 3.3 障碍物与场景函数

#### `update_point_cloud`

```python
def update_point_cloud(
    self,
    points: np.ndarray,
    resolution: float = 1e-3,
    name: str = "scene_pcd",
):
```

将 world frame 下的点云加入 planning world。

```python
points = np.random.randn(1000, 3) * 0.1
planner.update_point_cloud(points, resolution=0.005)
```

#### `remove_point_cloud`

```python
def remove_point_cloud(self, name: str = "scene_pcd") -> bool:
```

移除点云障碍物。

#### `add_obstacle_box`

```python
def add_obstacle_box(
    self,
    size: np.ndarray,
    pose: mp.Pose,
    name: str = "obstacle_box",
    replace: bool = True,
):
```

添加 box 障碍物。

```python
planner.add_obstacle_box(
    size=np.array([0.3, 0.2, 0.1]),
    pose=mp.Pose(
        p=np.array([0.5, 0.0, 0.3]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
    ),
    name="table_box",
)
```

#### `remove_obstacle_box`

```python
def remove_obstacle_box(self, name: str = "obstacle_box") -> bool:
```

移除指定 box 障碍物。

---

### 3.4 轨迹构造函数

#### `build_timed_path`

```python
def build_timed_path(
    self,
    plan: dict[str, Any],
    current_qpos: np.ndarray,
    time_step: float,
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
```

将 MPlib 返回的 raw plan 转成最终 timed path。

处理逻辑：

1. 先对 `plan["position"]` 做 `unwrap_path_to_reference()`，消除等效关节 `2π` 表示跳变。
2. 如果 unwrap 只是整条路径的常数 offset，则保留 MPlib 原始 `velocity / acceleration / time`。
3. 如果 unwrap offset 在路径中途发生变化，则对 unwrap 后 path 调用 `self.planner.TOPP()` 重新时间参数化。

返回：

```python
(
    {
        "qpos": qpos,
        "qvel": qvel,
        "qacc": qacc,
        "time": time,
        "duration": duration,
    },
    debug,
)
```

常见 debug 字段：

| 字段 | 说明 |
|---|---|
| `unwrap_changed` | unwrap 是否改变了 raw path 表示 |
| `unwrap_offset_is_constant` | unwrap offset 是否沿时间恒定 |
| `max_unwrap_delta_deg` | 最大 unwrap 偏移，单位 deg |
| `max_unwrap_offset_step_deg` | 相邻 waypoint unwrap offset 最大变化量，单位 deg |
| `timing_source` | `mplib_original` 或 `topp_after_unwrap` |

---

### 3.5 主规划函数

#### `plan_near_pose_world`

```python
def plan_near_pose_world(
    self,
    goal_pose_world: mp.Pose,
    current_qpos: np.ndarray,
    pos_err: float,
    rot_err: float,
    constraint: dict[str, Any] | None = None,
    time_step: float = 0.05,
    screw_qpos_step: float = 0.1,
    final_pos_threshold: float = 1e-3,
    final_rot_threshold: float = float(np.deg2rad(0.5)),
    verbose: bool = False,
) -> dict[str, Any] | None:
```

近距离规划分支。

行为：

1. 如果当前末端已经到目标位姿附近，直接返回 shortcut。
2. 如果没有 constraint 且目标距离较近，尝试 MPlib `plan_screw()`。
3. screw 成功后调用 `build_timed_path()`。
4. 做 path quality 检查和 final pose 检查。
5. 成功返回轨迹；不适用或失败则返回 `None`，由 qpos planner fallback。

#### `plan_qpos_candidates_world`

```python
def plan_qpos_candidates_world(
    self,
    goal_pose_world: mp.Pose,
    current_qpos: np.ndarray,
    constraint: dict[str, Any] | None = None,
    time_step: float = 0.05,
    planning_time: float = 1.0,
    rrt_range: float = 0.1,
    simplify: bool = True,
    ik_threshold: float = 1e-3,
    ik_pos_threshold: float | None = None,
    ik_rot_threshold: float = float(np.deg2rad(0.5)),
    ik_random_seed_count: int = 16,
    max_goal_candidates: int = 8,
    verbose: bool = False,
) -> dict[str, Any]:
```

全局 fallback 规划分支。

行为：

1. 调用 `ik_world()` 生成多个 IK goal candidates。
2. 对每个 candidate 调 MPlib `plan_qpos()`。
3. 调用 `build_timed_path()` 生成最终 timed trajectory。
4. 检查 path quality 和 final pose。
5. 按 `path_score` 和 `ik_score` 选最优轨迹。

如果没有 IK 解：

```python
{"status": "IK Failed", "mode_used": "ik", "debug": {...}}
```

如果所有 qpos planning 都失败：

```python
{"status": "Planning Failed", "mode_used": "qpos", "debug": {...}}
```

#### `plan_pose_world`

```python
def plan_pose_world(
    self,
    goal_pose_world: mp.Pose,
    current_qpos: np.ndarray,
    constraint: dict[str, Any] | None = None,
    time_step: float = 0.05,
    planning_time: float = 1.0,
    rrt_range: float = 0.1,
    simplify: bool = True,
    ik_threshold: float = 1e-3,
    ik_pos_threshold: float | None = None,
    ik_rot_threshold: float = float(np.deg2rad(0.5)),
    ik_random_seed_count: int = 16,
    max_goal_candidates: int = 8,
    screw_qpos_step: float = 0.1,
    verbose: bool = False,
) -> dict[str, Any]:
```

主入口。一般使用这个函数即可。

调度逻辑：

```text
canonicalize current_qpos
        ↓
check current state
        ↓
compute current FK error to goal
        ↓
try shortcut / screw motion
        ↓
fall back to IK + qpos planning
```

---

## 4. 返回结果格式

成功时返回：

```python
{
    "status": "Success",
    "mode_used": "shortcut" | "screw" | "qpos",
    "qpos": np.ndarray,       # shape: (T, 7)
    "qvel": np.ndarray,       # shape: (T, 7)
    "qacc": np.ndarray,       # shape: (T, 7)
    "time": np.ndarray,       # shape: (T,)
    "duration": float,
    "goal_qpos": np.ndarray,  # shape: (7,)
    "ik_score": tuple | None,
    "path_score": tuple | None,
    "debug": dict,
}
```

失败时返回：

```python
{
    "status": "IK Failed" | "Planning Failed",
    "mode_used": "precheck" | "ik" | "qpos",
    "debug": dict,
}
```

常见 `debug["reason"]`：

| reason | 含义 |
|---|---|
| `current_joint_limit` | 当前 qpos 不在 joint limit 内 |
| `current_self_collision` | 当前状态自碰撞 |
| `current_env_collision` | 当前状态与环境碰撞 |
| `pose_shortcut` | 当前 pose 已经到达目标 |
| `step_limit` | 路径相邻 waypoint 跳变过大 |
| `excursion_limit` | 相对起点偏移过大 |
| `total_motion_limit` | 累积运动量过大 |
| `final_pose_error` | 最终 FK 没有达到目标位姿 |
| `goal_joint_limit` | IK goal 不满足 joint limit |

---

## 5. 基本使用示例

```python
from pathlib import Path

import mplib as mp
import numpy as np

from arm_motion_planner import ArmMotionPlanner

planner = ArmMotionPlanner(
    urdf_path=Path("assets/xarm7.urdf"),
    srdf_path=Path("assets/xarm7.srdf"),
    move_group="link7",
    joint_vel_limits=np.ones(7) * 1.0,
    joint_acc_limits=np.ones(7) * 2.0,
)

current_qpos = np.array([0.0, -0.5, 0.0, 1.0, 0.0, 0.8, 0.0])

goal_pose = mp.Pose(
    p=np.array([0.45, 0.10, 0.35]),
    q=np.array([1.0, 0.0, 0.0, 0.0]),
)

result = planner.plan_pose_world(
    goal_pose_world=goal_pose,
    current_qpos=current_qpos,
    time_step=0.05,
    planning_time=1.0,
    rrt_range=0.1,
    ik_random_seed_count=32,
    max_goal_candidates=8,
)

if result["status"] == "Success":
    qpos = result["qpos"]
    qvel = result["qvel"]
    qacc = result["qacc"]
    time = result["time"]
    print("mode:", result["mode_used"])
    print("duration:", result["duration"])
else:
    print("planning failed:", result)
```

---

## 6. 点云和障碍物示例

### 6.1 使用点云作为环境障碍

```python
scene_points = np.load("scene_points.npy")  # shape: (N, 3), world frame
planner.update_point_cloud(scene_points, resolution=0.005)

result = planner.plan_pose_world(goal_pose, current_qpos)

planner.remove_point_cloud()
```

### 6.2 添加 box 障碍物

```python
planner.add_obstacle_box(
    size=np.array([0.4, 0.4, 0.05]),
    pose=mp.Pose(
        p=np.array([0.45, 0.0, 0.2]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
    ),
    name="table",
)

result = planner.plan_pose_world(goal_pose, current_qpos)

planner.remove_obstacle_box("table")
```

---

## 7. 约束规划示例

`constraint` 需要包含：

```python
{
    "function": constraint_function,
    "jacobian": constraint_jacobian,
    "tolerance": 1e-3,
}
```

示例结构：

```python
def constraint_function(qpos: np.ndarray, out: np.ndarray):
    # out[:] = 0 表示满足约束
    out[0] = 0.0


def constraint_jacobian(qpos: np.ndarray, out: np.ndarray):
    # out 写入 constraint 对 qpos 的 jacobian
    out[:] = 0.0

constraint = {
    "function": constraint_function,
    "jacobian": constraint_jacobian,
    "tolerance": 1e-3,
}

result = planner.plan_pose_world(
    goal_pose_world=goal_pose,
    current_qpos=current_qpos,
    constraint=constraint,
    simplify=False,
)
```

注意：有 constraint 时，内部会关闭 path simplification，因为 constrained planning 不支持 path simplification。

---

## 8. 部署注意事项

### 8.1 坐标系约定

- `goal_pose_world` 是 world frame 下的末端目标位姿。
- `base_pose` 是机器人 base 在 world frame 下的位姿。
- `fk_world()` 返回 world frame 下的末端位姿。
- 点云和 box obstacle 默认也按 world frame 理解。

如果机器人 base 不在世界原点，必须先调用：

```python
planner.set_base_pose(base_pose)
```

### 8.2 qpos 维度

当前实现专门面向 xArm7，期望 planning joint 数量为 7。输入 `current_qpos` 必须是 7 维 move group qpos。

### 8.3 等效关节分支

planner 会对 joint range 大于 `2π` 的关节启用等效分支处理：

```python
equivalent_joint_indices = np.where(joint_range > 2π + 1e-6)
```

路径生成后会通过 `unwrap_path_to_reference()` 消除 `±2π` 表示跳变。若 unwrap 只是常数 offset，则保留 MPlib 原始速度/加速度；若 unwrap offset 在路径中途变化，则重新调用 `TOPP()` 对 unwrap 后路径做时间参数化。

### 8.4 `last_goal_qpos`

`last_goal_qpos` 用于连续任务中的 IK 分支连续性偏置。以下情况建议手动清空或重新创建 planner：

- 机械臂被人工拖动
- 任务序列切换
- base pose 大幅变化
- planning scene 大幅变化

当前类未提供 `reset_history()`，可以直接：

```python
planner.last_goal_qpos = None
```

### 8.5 screw 与 qpos fallback

`plan_pose_world()` 会先尝试：

1. 当前 pose 已到目标：shortcut
2. 近距离目标：screw motion
3. 否则或 screw 失败：IK + qpos planning

screw 失败不一定代表任务失败，因为它是局部方法；qpos planning 是 fallback。

### 8.6 轨迹执行

执行前建议至少检查：

```python
assert result["status"] == "Success"
assert result["qpos"].shape[1] == 7
assert result["qvel"].shape == result["qpos"].shape
assert result["qacc"].shape == result["qpos"].shape
assert result["time"].shape[0] == result["qpos"].shape[0]
```

若控制器只接受 position trajectory，可以只使用：

```python
result["time"], result["qpos"]
```

若控制器支持速度和加速度前馈，可以同时使用：

```python
result["qpos"], result["qvel"], result["qacc"], result["time"]
```

### 8.7 障碍物生命周期

同名点云或 box 会覆盖/替换旧对象时，建议显式 remove：

```python
planner.remove_point_cloud("scene_pcd")
planner.remove_obstacle_box("table")
```

避免旧障碍物残留影响后续任务。

### 8.8 真机部署建议

部署到真实 xArm7 前建议：

1. 先在仿真中验证 URDF/SRDF、move group、base pose。
2. 检查 `debug["timing_source"]`，确认是否频繁触发 `topp_after_unwrap`。
3. 检查 `debug["max_unwrap_offset_step_deg"]`，如果经常很大，说明路径存在明显分支跳变。
4. 保守设置 `joint_vel_limits` 和 `joint_acc_limits`。
5. 每次更新 scene 后重新规划，不复用旧轨迹。
6. 执行失败或外部干预后，清空 `last_goal_qpos`。

---

## 9. 工具函数说明

`planner_utils.py` 中包含：

| 函数 | 说明 |
|---|---|
| `transform_pose(parent, child)` | 计算 `parent * child` 位姿组合 |
| `relative_pose(frame_world, pose_world)` | 计算 world pose 相对某个 frame 的局部位姿 |
| `pose_error(pose1, pose2)` | 返回 position error 和 quaternion shortest rotation error |
| `wrap_to_reference(qpos, ref_qpos, joint_limits, equivalent_joint_indices)` | 将 qpos 中的等效关节分支选到离 ref 最近的位置 |
| `unwrap_path_to_reference(path, ref_qpos, joint_limits, equivalent_joint_indices)` | 沿路径逐点选择连续的等效关节分支 |
| `evaluate_path(path, start_qpos, goal_qpos, last_goal_qpos)` | 基于 step、excursion、total motion 对路径做质量筛选和打分 |

---

## 10. 最小 smoke test

```python
import numpy as np
import mplib as mp

from arm_motion_planner import ArmMotionPlanner

planner = ArmMotionPlanner(
    urdf_path="assets/xarm7.urdf",
    srdf_path="assets/xarm7.srdf",
    move_group="link7",
)

current_qpos = np.zeros(7)
current_pose = planner.fk_world(current_qpos)

result = planner.plan_pose_world(current_pose, current_qpos)

assert result["status"] == "Success"
assert result["mode_used"] == "shortcut"
assert result["qpos"].shape == (1, 7)
```

该测试用于确认：

- URDF/SRDF 能加载
- move group 名称正确
- FK 能正常计算
- shortcut 分支能正常返回

