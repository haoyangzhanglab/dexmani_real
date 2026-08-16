# DexMani Sim HDF5/Zarr 数据格式完整说明

> **审计对象：** `/home/zhanghaoyang/Desktop/dexmani_sim`  
> **审计日期：** 2026-08-15  
> **结论来源：** 写入器、传感器生产链、转换器、回放器、可视化器的静态审计，以及
> 现存文件的只读元数据普查。  
> **执行边界：** 未启动 SAPIEN 场景、GPU 渲染、交互程序或机器人程序。

## 0. 阅读导航

| 目标 | 重点章节 |
|---|---|
| 理解 observation/action/done 对齐 | 3 |
| 查 HDF5 dataset、shape、dtype、单位 | 4 |
| 查 HDF5 根 attributes | 5 |
| 区分 success、failed、done、truncated | 6 |
| 查 Zarr array、chunk 和 episode 边界 | 7 |
| 查现存数据规模和已定位问题 | 8、10 |
| 实现防御性 reader | 11 |
| 按 Sim/Policy 命名理解 Real v16 字段 | [Real→Sim/Policy 标签映射表](real_to_sim_mapping.md) |

本独立版正文与统一数据字典的
[`DexMani Sim` 第 11 节](hdf5_episode.md#11-dexmani-sim-hdf5zarr-数据格式)
保持同步。

---

## 1. 结论摘要

DexMani Sim 当前存在两层数据容器：

1. **HDF5 episode**：一个文件对应一个 episode，保存 13 个顶层逐帧 dataset 和 episode 级 HDF5 attributes。
2. **Zarr task dataset**：一个目录对应一个 task；默认只合并该 task 的成功 HDF5，将各 episode 沿第 0 维拼接，并用 `meta/episode_ends` 保存边界。

现存数据的实测结果为：

- 1113 个 HDF5 文件，合计 237777 帧；1000 个成功 episode、113 个失败 episode。
- 8 个 Zarr v2 store，各含 125 个成功 episode，合计 210083 帧。
- 1113 个 HDF5 的 13 个 dataset 键、单帧 shape、dtype 和 gzip 压缩级别完全一致。
- 所有 HDF5 内部 dataset 的首维长度一致，并与 `episode_steps` 一致。
- 8 个 Zarr 的 `episode_ends` 均与当前成功 HDF5 按文件名字典序累加得到的边界完全一致；每个 store 的首、末样本在全部 13 个 dataset 上与源 HDF5 精确相等。
- 当前实际图像分辨率是 **320×240（W×H）**，dataset shape 是 `(N,240,320,...)`。
  README 和 CLAUDE.md 中的 `480×640` 是过时说明。
- 当前全部 1113 个 HDF5 的 `action_space` 都是 `joint`；代码还支持 `ee`，但本次没有实际 `ee` 文件可用于样本级验证。

HDF5 和 Zarr 均没有显式的 schema 名称或 schema version。下文所称“标准格式”是当前生产代码和现存文件共同体现的事实，而不是由文件内版本标记保证的正式版本协议。

## 2. 数据生成和转换链路

```text
env.reset(seed) -> 初始观测 s0
                    |
                    v
MimicGen joint path -> EnvRecorder -> episode_<seed>.h5
                                         |
                                         | 仅 succeeded_episode，按文件名字典序
                                         v
                              <task>.zarr/data/*
                              <task>.zarr/meta/episode_ends
```

关键实现：

| 环节 | 源文件 | 作用 |
|---|---|---|
| 环境观测 | `dexmani_sim/envs/base_env.py` | 组合相机、机器人、接触力和手部几何数据，完成点云与分割预处理 |
| RGB-D/分割/相机矩阵 | `dexmani_sim/sensors/camera.py` | 生成 `rgb`、`depth`、actor segmentation、相机内外参和原始世界点云 |
| 接触力 | `dexmani_sim/sensors/contact.py` | 将每个物理子步的接触 impulse 换算为 force，并对一个控制步求平均 |
| 理想手部点云 | `dexmani_sim/sensors/imagine_point_cloud.py` | 从 11 个手部 render mesh 采样固定 512 点并随 link pose 变换到世界系 |
| HDF5 写入 | `dexmani_sim/mimic_gen/utils/env_recorder.py` | 对齐观测、动作、done，写 dataset 和 attributes |
| episode 级扩展属性 | `dexmani_sim/mimic_gen/base_generator.py`、`dexmani_sim/gen_episode_data.py` | 写轨迹质量、抓取验证和场景随机化信息 |
| HDF5→Zarr | `dexmani_sim/data_process/hdf5_to_zarr.py` | 拼接顶层 dataset，写累计 episode 结束下标 |
| HDF5 派生工具 | `dexmani_sim/data_process/hdf5_custom_modify.py` | keep/delete/rename/merge/derive，可生成非标准 HDF5 |
| 回放 | `dexmani_sim/replay_episode.py` | 根据 `action_space` 选择 `action` 或 `action_ee` |
| 可视化 | `dexmani_sim/data_process/hdf5_visualize.py` | 读取标准 dataset 并在 Rerun 中显示 |

标准路径：

```text
mimic_data/episodes/<task>/succeeded_episode/episode_<seed:04d>.h5
mimic_data/episodes/<task>/failed_episode/episode_<seed:04d>.h5
mimic_data/dataset/<task>.zarr/
```

`<seed:04d>` 表示至少补足 4 位；超过 9999 的 seed 不会截断。

## 3. 时间轴与行对齐语义

这是读取该格式时最重要的约定。

`BaseGenerator.reset()` 先保存初始观测 `s0`。随后每次 `env.step(action)` 返回动作执行后的下一状态。执行 N 个动作后，内存里有 N+1 个观测，但保存时丢弃最后一个观测，仅保留前 N 个。因此第 `t` 行的关系是：

```text
obs[t] = s_t
action[t] = a_t
done[t] = step(a_t) 返回的终止标志，即对应 s_(t+1)

s_t --a_t--> s_(t+1)
```

具体含义：

- `rgb/depth/.../joint_state` 的第 `t` 行是 **执行 `action[t]` 之前** 的观测。
- `done[t]` 是 **执行 `action[t]` 之后** 的结果，不与 `obs[t]` 处于同一个状态时刻。
- 最终 next observation `s_N` 不在文件中。
- 文件没有逐帧 timestamp。只能在固定步长假设下用 `time[t] = t * dt` 构造名义时间。
- `dt = frame_skip * physics_dt`。现存文件统一为 `frame_skip=15`、`physics_dt≈1/240 s`、`dt≈0.0625 s`，名义控制频率约 16 Hz。
- `obs_alignment="obs[t]_before_action[t]"` 只说明 observation/action 关系，没有单独说明 `done` 的 post-action 语义。

## 4. HDF5 容器结构（13 个 datasets）

### 4.1 根结构

现存 1113 个文件的根层均只有 dataset，没有 group：

```text
/
├── action
├── action_ee
├── camera_extrinsic
├── camera_intrinsic
├── contact_force
├── depth
├── done
├── fingertip_points
├── imagine_point_cloud
├── joint_state
├── point_cloud
├── rgb
└── segmentation
```

所有 dataset：

- 使用 HDF5 gzip，`compression_opts=4`。
- 因启用压缩而采用 chunked layout；写入器未指定 chunks，实际 chunk shape 由 h5py 自动选择，会随 episode 长度变化，不能视为 schema 常量。
- 未启用 shuffle 或 Fletcher32 checksum。
- dataset 本身没有标准 attributes；episode 元数据位于文件根 attributes。

### 4.2 Dataset 总表

下表中的 N 为本 episode 的动作数，也等于每个 dataset 的第 0 维和 `episode_steps`。

| Dataset | HDF5 shape | dtype | 单位 | 坐标系/范围 | 物理意义 |
|---|---:|---|---|---|---|
| `rgb` | `(N,240,320,3)` | `uint8` | 8-bit intensity | RGB，通常 `[0,255]` | SAPIEN `Color` render target 的前三通道，先乘 255、clip，再转 `uint8` |
| `depth` | `(N,240,320)` | `uint16` | mm | 相机光轴深度；0=无效/背景 | OpenGL camera-space `Position[...,2]` 取负后乘 1000；超出视锥或 actor id 为 0 的像素置 0 |
| `segmentation` | `(N,240,320)` | `uint8` | 无 | actor 级语义 id | 0=背景，1=桌面，2=机械臂，3=手；其余为按 scene actor id 偏移得到的物体 id |
| `point_cloud` | `(N,1024,6)` | `float32` | 前 3 列 m；后 3 列无量纲 | SAPIEN 世界系；RGB `[0,1]` | 相机可见前景点云，经世界系变换、工作空间 crop、2.5 mm voxel 聚合及 FPS/补采样 |
| `camera_intrinsic` | `(N,9)` | `float32` | 焦距/主点为 pixel | OpenCV intrinsic，row-major `3×3` | 每帧重复保存的相机 K 矩阵 |
| `camera_extrinsic` | `(N,12)` | `float32` | 平移为 m | OpenCV RDF camera；row-major `3×4` world→camera | `[R_cw \| t_cw]`；可视化器用 `R_wc=R_cw.T`、`t_wc=-R_wc@t_cw` 求 camera pose |
| `joint_state` | `(N,19)` | `float32` | rad | 自定义关节顺序 | 动作执行前的实际仿真 qpos：arm 7 + hand 12 |
| `contact_force` | `(N,15)` | `float32` | N | SAPIEN 世界系 xyz | 5 个手指 `link2` 的净接触力；每控制步对 15 个物理子步的 force 求平均 |
| `fingertip_points` | `(N,15)` | `float32` | m | SAPIEN 世界系 xyz | 5 个 fingertip link 原点的位置，flatten 为 5×3 |
| `imagine_point_cloud` | `(N,512,6)` | `float32` | 前 3 列 m；后 3 列无量纲 | SAPIEN 世界系；RGB `[0,1]` | 手部 render mesh 的完整“想象点云”；不受相机遮挡、视锥或工作空间 crop 限制 |
| `action` | `(N,19)` | `float32` | rad | 自定义关节顺序 | 关节目标：arm 7 + hand 12；是命令目标，不是反馈 qpos |
| `action_ee` | `(N,21)` | `float32` | pos 为 m；rot6d 无量纲；hand 为 rad | `custom_eef_link` 世界位姿 | `[eef_pos(3),eef_rot6d(6),hand_qpos(12)]` |
| `done` | `(N,)` | `bool` | 无 | `False/True` | `env.step(action[t])` 返回的 `success OR failed`；是 post-action transition 标签 |

SAPIEN 官方说明相机 `Position` texture 使用 OpenGL camera space（x 向右、y 向上、z 向后），model matrix 将其变换到世界系；官方 API 将 intrinsic/extrinsic 称为 OpenCV 格式。参见 [SAPIEN Camera 文档](https://sapien-sim.github.io/docs/user_guide/rendering/camera.html) 和 [CameraEntity API](https://sapien.ucsd.edu/docs/latest/apidoc/sapien.core.html)。

dataset 没有保存 unit、coordinate frame、列名或有效值规则等 attributes；上表语义来自生产代码，而不是文件内的机器可读描述。

坐标约定汇总：

- SAPIEN 世界使用机器人学右手系，可按 x 向前、y 向左、z 向上理解。
- 机器人 root 位于世界原点，并绕世界 z 轴 yaw `+π/6`；robot FK 输出再乘 root pose，所以 `action_ee`、fingertip 和两种点云都已在世界系，不是 robot-root local frame。
- 相机 `Position` texture 使用 OpenGL/Blender camera frame：x 向右、y 向上、z 向后；`depth=-z`，所以它是光轴深度而不是到相机中心的欧氏距离。
- `camera_extrinsic` 使用 OpenCV camera frame（RDF：x 向右、y 向下、z 向前），表示 world→camera。
- 相机可能在 reset 时按 episode 随机平移/重新朝向，但不会在标准 episode 内逐帧移动；K/extrinsic 仍被重复写入 N 行。

### 4.3 关节和手指展开顺序

`joint_state` 与 `action` 的 19 维顺序完全相同：

| 索引 | 名称 | 部位 |
|---:|---|---|
| 0–6 | `joint1` … `joint7` | xArm7 |
| 7 | `right_hand_thumb_bend_joint` | thumb |
| 8 | `right_hand_thumb_rota_joint1` | thumb |
| 9 | `right_hand_thumb_rota_joint2` | thumb |
| 10 | `right_hand_index_bend_joint` | index |
| 11 | `right_hand_index_joint1` | index |
| 12 | `right_hand_index_joint2` | index |
| 13 | `right_hand_mid_joint1` | middle |
| 14 | `right_hand_mid_joint2` | middle |
| 15 | `right_hand_ring_joint1` | ring |
| 16 | `right_hand_ring_joint2` | ring |
| 17 | `right_hand_pinky_joint1` | pinky |
| 18 | `right_hand_pinky_joint2` | pinky |

`contact_force.reshape(5,3)` 和 `fingertip_points.reshape(5,3)` 的手指顺序均为：

```text
thumb, index, middle, ring, pinky
```

每个手指内部顺序为 `(x, y, z)`。接触力取各手指第二指节 `link2`，而 fingertip position 取各手指 tip link；二者不是同一个 link 原点。

### 4.4 点云处理细节

#### `point_cloud`

原始输入来自相机每个有效前景像素：

```text
[x_world_m, y_world_m, z_world_m, r, g, b]
```

处理顺序：

1. 仅保留 `Position[...,3] < 1` 且 actor segmentation `> 0` 的像素。
2. 用相机 model matrix 从 OpenGL camera space 变换到 SAPIEN 世界系。
3. 进行闭区间 crop：`x∈[0.15,0.85] m`、`y∈[-0.5,0.5] m`、`z∈[table_surface_z+0.003,1.0] m`。
4. 用 `voxel_size=0.0025 m` 聚合，同 voxel 的 xyz 和 RGB 都取平均。
5. 若点数不少于 1024，对 xyz 做 farthest-point sampling，并携带对应 RGB。
6. 若点数不足 1024，用带 replacement 的随机索引复制已有点；随机源由 episode seed 设置。
7. 如果 crop 后没有任何点，整帧输出 1024×6 的 0。

代码注释写着“5 mm”，但实际调用值是 **2.5 mm**；格式解释应以执行代码为准。

#### `imagine_point_cloud`

它不是由 depth 反投影得到，而是对完整手部 render mesh 做表面采样：

- `right_hand_link`：192 点。
- 5 个手指的 `link1` 和 `link2`：10×32 点。
- 总数：`192 + 10×32 = 512`。
- 采样点保存在 link local frame，读取时随每个 link 的世界 pose 变换。
- RGB 来自 render mesh vertex/material colors。
- 该点云不包含被操作物体，也不表达可见性。

### 4.5 `action` 与 `action_ee`

`action_ee` 的旋转 6D 表达不是 Euler angle，也不是 rotation vector：

```text
rot6d = concat(R[:, 0], R[:, 1])
```

即旋转矩阵前两列，列优先拼接为 6 个数。解码时使用 Gram–Schmidt 正交化并以叉乘恢复第三列。

两种 `action_space` 下文件仍同时保存两个动作 dataset：

| `action_space` | 实际送入 `env.step` | `action` | `action_ee` |
|---|---|---|---|
| `joint` | 19D joint target | 原始 joint target | 对该 joint target 做 FK 得到的 EEF+hand 表达 |
| `ee` | 21D EEF+hand target | normal motion 中为生成该目标所依据的 joint path | 从 joint path 计算并实际送入环境的 EEF+hand target |

注意：

- `action_dim` 始终取 `action.shape[1]`，因此即使 `action_space="ee"`，其值仍为 19，而不是 21。
- `action` 是 recorder 收到的目标值。`robot.apply_action()` 在真正下发 PD target 前还会按 joint limits clip，因此极端情况下文件中的 `action` 可能与最终应用值不同。
- EE `hold_phase` 在执行 EEF action **之后**重新调用 IK 来填充同一行 `action`，所以 hold 行的 19D 值不保证等于环境在该步执行前求得的 joint target。
- 现存 1113 个文件全部是 `action_space="joint"`；`ee` 行为是代码契约，未由现存样本覆盖。

### 4.6 标准文件没有记录的信息

标准 HDF5 不包含：

- 逐帧 timestamp 或 wall-clock time。
- reward。
- 独立的 per-step `success`、`failed`、`truncated`、`success_condition` 或完整 `info`；`done` 不能区分成功与失败。
- terminal next observation `s_N`。
- joint velocity、acceleration、effort/torque 或实际 applied/clipped command。
- EEF feedback pose；`action_ee` 是由命令 joint target 做 FK 得到的动作表示。
- 被操作物体逐帧 pose、velocity、contact graph 或 segmentation label map。
- 相机 near/far、FOV、畸变系数或颜色空间标识；当前代码常量为 near=0.1 m、far=5.0 m、vertical FOV=58°，但文件只保存 K/extrinsic。
- 点云每点的 segmentation/object id、normal 或 visibility confidence。
- schema version、producer commit、依赖版本、生成时间或内容 checksum。

视频不嵌入 HDF5。启用 `save_video()` 时，它另存为：

```text
mimic_data/episodes/<task>/videos/episode_<seed:04d>_<succeeded|failed>.mp4
```

## 5. HDF5 根 Attributes 数据字典（46 个实测键）

### 5.1 固定基础属性

| Attribute | 实测 dtype | 单位/格式 | 物理或逻辑意义 |
|---|---|---|---|
| `task_name` | UTF-8 string | snake_case | task 名称 |
| `seed` | `int64` | 无 | episode seed；与文件名编号一致 |
| `success` | `int64` | 0/1 | 最终用于分类和目录选择的 `wrapper.task_done`；可能受轨迹质量拒绝影响 |
| `episode_steps` | `int64` | frame | N，即动作数和每个 dataset 的首维 |
| `truncated` | `int64` | 0/1 | recorder 最后一次在 normal interaction phase 看到的时间上限标志 |
| `dt` | `float64` | s | 一个控制步的仿真时长，`frame_skip * physics_dt` |
| `physics_dt` | `float64` | s | 单个 PhysX step 时长 |
| `frame_skip` | `int64` | physics step/control step | 每个 action 执行的物理子步数 |
| `action_dim` | `int64` | dimension | `action` 的末维；当前恒为 19 |
| `action_space` | UTF-8 string | `joint`/`ee` | 环境实际采用的控制空间；回放器据此选动作源 |
| `obs_alignment` | UTF-8 string | 固定文本 | 当前为 `obs[t]_before_action[t]` |
| `success_once` | `int64` | 0/1 | recorder 是否曾观察到稳定成功 `info["success"]` |
| `success_at_end` | `int64` | 0/1 | 保存时 `env.success_condition` 的瞬时值，不等同于稳定保持后的 done |
| `success_frame_idx` | `int64` | action index | 首次观察到稳定成功的 action/done 行；从 0 开始，未成功为 -1 |
| `failed` | `int64` | 0/1 | finalize 时是否命中 task 的显式失败条件 |
| `failed_reason` | UTF-8 string | 文本 | 设计上用于失败原因；现存 1113 个文件全部为空字符串 |

这些逻辑量为了兼容写入都没有统一使用 HDF5 bool；除 `traj_quality_is_low_quality` 和 dataset `done` 外，多数布尔语义属性实际存为 `int64` 0/1。

### 5.2 抓取、fallback 与轨迹质量属性

以下属性由 MimicGen 生成器通过 `extra_attrs` 写入。现存文件除 `fallback_reason` 外全部存在。

| Attribute | dtype | 单位/格式 | 定义 |
|---|---|---|---|
| `fallback_count` | `int64` | count | 规划完全失败后使用 2 帧 hold fallback 的次数 |
| `fallback_reason` | UTF-8 string | 文本，可缺省 | 最后一次 fallback 的原因；仅当存在原因时写入，不是全部原因列表 |
| `grasp_verified` | `int64` | 0/1 | 所有抓取验证是否都通过；没有验证项时按真处理 |
| `grasp_verified_list` | `int64[G]` | 每项 0/1 | 每次 grasp 的验证结果，按发生顺序；现存 shape 为 `(1,)` 或 `(2,)` |
| `traj_quality_is_low_quality` | HDF5 `bool` | True/False | 任一 primitive 被判为低质量 |
| `traj_quality_primitive_count` | `int64` | count | 参与质量日志的 primitive 数量 |
| `traj_quality_low_quality_count` | `int64` | count | 低质量 primitive 数量 |
| `traj_quality_min_efficiency` | `float64` | ratio | 有至少 4 个 waypoint 的 primitive 中，`straight_line/path_length` 的最小值 |
| `traj_quality_mean_efficiency` | `float64` | ratio | 上述效率的均值 |
| `traj_quality_total_large_turns` | `int64` | count | 全 episode 大转角数量之和；大转角阈值为 30° |
| `traj_quality_max_turn_deg` | `float64` | degree | 所有 primitive 的最大相邻路径方向转角 |

质量计算只使用 arm path 的 EEF 世界位置；至少 5 个 waypoint 时先做一次 3 点均值平滑。长度小于等于 1 mm 的 segment 不参与方向转角。低质量判定为：

```text
efficiency < 0.7
OR num_large_turns > 3
OR max_turn_deg > 60°
```

当直线位移小于 0.03 m 时，不使用 `max_turn_deg > 60°` 这一项。某些 task 可关闭 `enforce_path_quality`，因此 `traj_quality_is_low_quality=True` 并不必然意味着 `success=0`。现存数据中有 111 个成功 episode 同时标记为低质量。

### 5.3 场景属性

| Attribute | dtype | 条件 | 定义 |
|---|---|---|---|
| `scene_info_json` | UTF-8 JSON string | `scene_info` 非空；当前始终存在 | 场景随机化的主要机器可读记录 |
| `scene_seed` | UTF-8 string | 当前始终存在 | `scene_info.seed` 的字符串副本；数值语义应优先用根 `seed` |
| `scene_table_height_offset` | UTF-8 string | 启用桌高随机化 | 桌高偏移的字符串副本，单位 m |
| `scene_skybox` | UTF-8 string | 启用 per-episode skybox | skybox 文件名或 `clean`；当前样本未出现 |
| `scene_objects_<key>` | UTF-8 string | 每个 manipulated/goal object | 对象信息 dict 的 Python `str()`；不是严格 JSON |

`scene_info_json` 的结构为：

```json
{
  "seed": 0,
  "table_height_offset": 0.009108850619643628,
  "skybox": "optional.exr",
  "objects": {
    "object_key": {
      "model_id": 3,
      "user_scale": 1.25
    }
  }
}
```

没有 `model_name` 的原生 SAPIEN entity 使用 `{"type": "ClassName"}`。`skybox` 和 `table_height_offset` 仅在对应随机化开关启用时存在。场景 JSON 不保存每个物体的初始 pose、质量、摩擦、灯光参数、材质参数、相机随机偏移或 clutter 的完整配置。

现存文件观察到的动态对象 attribute key 为：

| Task | `scene_objects_*` key（并集） |
|---|---|
| `multi_grasp` | `scene_objects_027_table-tennis`, `scene_objects_111_callbell` |
| `open_box` | `scene_objects_flip_box` |
| `peg_insertion` | `scene_objects_peg` |
| `pick_apple_messy` | `scene_objects_003_plate`, `scene_objects_035_apple` |
| `pick_bottle` | `scene_objects_001_bottle` |
| `place_milk_box` | `scene_objects_038_milk-box`, `scene_objects_074_displaystand` |
| `pour` | `scene_objects_002_bowl`, `scene_objects_019_coaster`, `scene_objects_021_cup` |
| `stack_cups` | `scene_objects_021_cup_0`, `scene_objects_021_cup_1`, `scene_objects_021_cup_3`, `scene_objects_021_cup_5` 的 episode 子集 |

读取场景信息时应解析 `scene_info_json`，不应解析 `scene_objects_*` 的 Python repr 字符串。

### 5.4 `extra_attrs` 扩展边界

`EnvRecorder.save_episode(extra_attrs=...)` 会在固定属性之后执行：

```python
for k, v in extra_attrs.items():
    f.attrs[k] = v
```

因此它可以增加任意 HDF5 attribute，也可以覆盖同名基础属性。当前批量生产链使用的是上述质量与场景字段，但 writer 本身没有 whitelist 或冲突检查。第三方文件可能拥有更多属性或被覆盖的语义，不能仅凭扩展属性名认定为标准格式。

## 6. success、done、failed、truncated 的关系

这些字段不是同义词：

| 字段 | 粒度 | 来源 | 是否可能保持/过滤 |
|---|---|---|---|
| `success_condition` | 未直接保存 | task 每步计算的瞬时条件 | 可在下一步消失 |
| `done[t]` | 每 transition | 稳定成功或显式失败 | 成功在 env 内 latch 后通常持续为 True |
| `success_once` | episode | recorder 是否曾见过 `info.success` | latch |
| `success_at_end` | episode | 保存时瞬时 `env.success_condition` | 非 latch |
| `success` | episode | 最终 `task_done` | 可因 truncation、显式失败或低质量策略变为 0 |
| `failed` | episode | finalize 时 task 显式失败条件 | 与普通“未成功”不同 |
| `truncated` | episode | normal interaction phase 的 action limit | 与 success/failure 独立 |

目录与 `success` 在现存 1113 个文件中完全一致：`succeeded_episode` 都是 1，`failed_episode` 都是 0。但不要用 `failed=0` 推断成功；许多 episode 只是未达成功条件或被质量规则拒绝。

`hold_phase()` 不更新 `task_truncated`，也不因 `env.step()` 返回 truncated 而中止。因此成功 episode 的 `episode_steps` 可以略大于 task 的 nominal action limit，而 `truncated` 仍为 0。

## 7. Zarr 格式

### 7.1 目录结构与版本

现存 store 均为 Zarr format 2：

```text
<task>.zarr/
├── .zgroup                  # {"zarr_format": 2}
├── data/
│   ├── action/
│   ├── action_ee/
│   ├── camera_extrinsic/
│   ├── camera_intrinsic/
│   ├── contact_force/
│   ├── depth/
│   ├── done/
│   ├── fingertip_points/
│   ├── imagine_point_cloud/
│   ├── joint_state/
│   ├── point_cloud/
│   ├── rgb/
│   └── segmentation/
└── meta/
    └── episode_ends/
```

根 group、`data` group、`meta` group 和全部 array 的 Zarr attrs 当前均为空。

### 7.2 Array schema

设 E 为 episode 数、`N_i` 为第 i 个 episode 的长度、`T=sum(N_i)`。`data/*` 与 HDF5 dataset 数值和 dtype 相同，仅将首维从 episode 内的 N 改为全 task 的 T：

| Zarr array | Shape | dtype | Chunks |
|---|---:|---|---:|
| `data/rgb` | `(T,240,320,3)` | `uint8` | `(100,240,320,3)` |
| `data/depth` | `(T,240,320)` | `uint16` | `(100,240,320)` |
| `data/segmentation` | `(T,240,320)` | `uint8` | `(100,240,320)` |
| `data/point_cloud` | `(T,1024,6)` | `float32` | `(100,1024,6)` |
| `data/camera_intrinsic` | `(T,9)` | `float32` | `(100,9)` |
| `data/camera_extrinsic` | `(T,12)` | `float32` | `(100,12)` |
| `data/joint_state` | `(T,19)` | `float32` | `(100,19)` |
| `data/contact_force` | `(T,15)` | `float32` | `(100,15)` |
| `data/fingertip_points` | `(T,15)` | `float32` | `(100,15)` |
| `data/imagine_point_cloud` | `(T,512,6)` | `float32` | `(100,512,6)` |
| `data/action` | `(T,19)` | `float32` | `(100,19)` |
| `data/action_ee` | `(T,21)` | `float32` | `(100,21)` |
| `data/done` | `(T,)` | `bool` | `(100,)` |
| `meta/episode_ends` | `(E,)` | `int64` | 现存为 `(125,)` |

所有 array 使用：

- numcodecs Zstd codec：`id="zstd"`, `level=3`, `checksum=False`。
- C order。
- `filters=None`。
- 数值 fill value 为 0，bool fill value 为 False。

`chunk_size=100` 是第 0 维 chunk 长度；converter 把完整 sample shape 放入同一个 chunk。因此一个 RGB chunk 的未压缩大小约为 `100×240×320×3 ≈ 23.0 MB`，读取单帧时仍可能解压整个 100 帧 chunk。

### 7.3 Episode 边界定义

`episode_ends` 保存的是 **exclusive cumulative end index**：

```text
episode_ends[i] = N_0 + N_1 + ... + N_i
start(i) = 0                       if i == 0
           episode_ends[i - 1]     otherwise
end(i) = episode_ends[i]

episode_i = data[key][start(i):end(i)]
```

最后一个值必须等于所有 `data/*` 的首维 T。当前 8 个 store 均严格递增且满足该不变量。

### 7.4 转换规则

标准 `main(task_name)`：

1. 只读取 `mimic_data/episodes/<task>/succeeded_episode`。
2. 接受 `.h5` 和 `.hdf5` 后缀，按完整路径字符串字典序排序。
3. 从第一个 HDF5 的**顶层 dataset**推导 key 集合、单帧 shape 和 dtype；key 再按字典序排序。
4. 以 `mode="w"` 新建/覆盖目标 Zarr。
5. 预检查其余文件中这些 key 的 dtype 是否与第一个文件一致。
6. 对每个文件检查所选 dataset 的首维长度是否一致。
7. 不做数值 cast，直接 append 全量数组。
8. 每个文件 append 后累计长度，最终写 `meta/episode_ends`。

转换器不写入 HDF5 root attributes，也不保存源文件名、seed、task name、success、action space、dt、场景 JSON或质量指标。Zarr 中的 episode i 只能对应“转换时排序后的第 i 个源文件”；该映射没有在 store 内持久化。

### 7.5 现存 Zarr 实测规模

| Task store | E | T | 第一个 episode end | 最终 end | 与当前成功 HDF5 边界一致 |
|---|---:|---:|---:|---:|---|
| `multi_grasp.zarr` | 125 | 23932 | 196 | 23932 | 是 |
| `open_box.zarr` | 125 | 27353 | 228 | 27353 | 是 |
| `peg_insertion.zarr` | 125 | 29326 | 250 | 29326 | 是 |
| `pick_apple_messy.zarr` | 125 | 21527 | 189 | 21527 | 是 |
| `pick_bottle.zarr` | 125 | 13542 | 114 | 13542 | 是 |
| `place_milk_box.zarr` | 125 | 29332 | 207 | 29332 | 是 |
| `pour.zarr` | 125 | 33365 | 246 | 33365 | 是 |
| `stack_cups.zarr` | 125 | 31706 | 324 | 31706 | 是 |

## 8. 现存 HDF5 数据规模

| Task | Split | Episodes | Frames | N min | N max | N mean |
|---|---|---:|---:|---:|---:|---:|
| `multi_grasp` | failed | 2 | 452 | 205 | 247 | 226.00 |
| `multi_grasp` | succeeded | 125 | 23932 | 177 | 242 | 191.46 |
| `open_box` | failed | 2 | 480 | 240 | 240 | 240.00 |
| `open_box` | succeeded | 125 | 27353 | 178 | 255 | 218.82 |
| `peg_insertion` | failed | 44 | 12471 | 225 | 328 | 283.43 |
| `peg_insertion` | succeeded | 125 | 29326 | 207 | 275 | 234.61 |
| `pick_apple_messy` | failed | 18 | 3170 | 158 | 215 | 176.11 |
| `pick_apple_messy` | succeeded | 125 | 21527 | 152 | 209 | 172.22 |
| `pick_bottle` | failed | 3 | 322 | 100 | 122 | 107.33 |
| `pick_bottle` | succeeded | 125 | 13542 | 93 | 132 | 108.34 |
| `place_milk_box` | failed | 17 | 3916 | 172 | 306 | 230.35 |
| `place_milk_box` | succeeded | 125 | 29332 | 191 | 304 | 234.66 |
| `pour` | failed | 1 | 287 | 287 | 287 | 287.00 |
| `pour` | succeeded | 125 | 33365 | 240 | 296 | 266.92 |
| `stack_cups` | failed | 26 | 6596 | 214 | 320 | 253.69 |
| `stack_cups` | succeeded | 125 | 31706 | 205 | 324 | 253.65 |
| **合计** | — | **1113** | **237777** | — | — | — |

属性普查补充结果：

- `success=1`：1000；`success=0`：113。
- `success_once=1`：1008；其中 8 个最终 `success=0`。
- `traj_quality_is_low_quality=True`：138；其中 111 个仍是成功 episode。
- `failed=1`：12，但 1113 个 `failed_reason` 全为空。
- `truncated=1`：5。
- `fallback_count>0`：77；这 77 个均具有 `fallback_reason="AtomicMotion planning failed."`。

## 9. 非标准派生 HDF5

`HDF5Pipeline` 不是标准 episode writer，而是通用转换工具。它可以：

- `KeepOnly`：只保留指定顶层 key。
- `Delete`：删除 key。
- `Rename`：重命名 key。
- `Merge`：拼接多个数组得到新 dataset。
- `Derive`：逐样本或整数组计算派生 dataset，并可改变 dtype/压缩。

仓库示例会生成仅含 `rgb`、`point_cloud`、重命名后的 `state`、带 normal 的点云和 DBSCAN labels 的 HDF5。这类文件不满足前述 13-key 标准 schema。

更重要的是，当前 `_process_one_file()` 没有复制 HDF5 文件根 attributes；dataset copy helper 也没有复制 dataset attributes。因此经该 pipeline 处理后，`task_name`、seed、时序、success、action space、场景 provenance 等 episode 元数据会全部丢失。

Zarr converter 是 data-driven 的：如果输入派生目录，第一个文件的顶层 dataset 会成为该 Zarr 的 schema。因此“Zarr 固定有 13 个 data array”只对当前标准输入和现存 8 个 store 成立，不是 converter 的硬编码保证。

## 10. 已定位问题与解释边界

> **严重度统计：** 高 6 项、中 8 项、低 3 项，共 17 项。下表给出完整索引；后续小节
> 展开最容易导致错误读取、数据丢失或训练泄漏的 9 项。

| ID | 级别 | 类型 | 定位 | Fact-check 结论 |
|---|---|---|---|---|
| SIM-H5-01 | 高 | 文档错误 | README Observation Space、CLAUDE.md Obs keys | 文档写 480×640；`Camera` 构造为 320×240，1113 个 HDF5 全为 240×320 |
| SIM-H5-02 | 高 | schema 治理 | HDF5 writer、Zarr converter | 两种容器都没有 schema name/version；兼容性只能由 key/shape/dtype 猜测 |
| SIM-H5-03 | 高 | provenance 丢失 | `hdf5_to_zarr.py` | Zarr 不保存任何 HDF5 attrs、文件名、seed 或 episode id，只有边界 |
| SIM-H5-04 | 高 | 转换覆盖风险 | `zarr.open(..., mode="w")` | 转换开始即覆盖目标 store；没有临时目录、完成标记或原子 publish |
| SIM-H5-05 | 中 | 弱验证 | `HDF5ToZarrConverter` | key/shape 取第一个文件；仅预检 dtype，未显式比较 trailing shape、属性或语义；额外 key 被忽略 |
| SIM-H5-06 | 高 | 元数据丢失 | `HDF5Pipeline._process_one_file` | 自定义 HDF5 修改不复制 root attrs，派生文件失去 episode provenance |
| SIM-H5-07 | 中 | 标签不可自描述 | segmentation producer | 文件不保存 class-id→object 的映射；0–3 可解释，其余 id 无法仅靠 HDF5 稳健还原对象身份 |
| SIM-H5-08 | 中 | 失败诊断无效 | `EnvRecorder._failed_reason` | 字段只初始化/清空/写入，从未赋具体原因；现存 1113 个值全空 |
| SIM-H5-09 | 中 | 时序易误读 | recorder trimming | `obs[t]` pre-action、`done[t]` post-action，且 terminal next obs 丢弃；仅 `obs_alignment` 部分记录该关系 |
| SIM-H5-10 | 中 | 动作真实性边界 | `robot.apply_action` 与 recorder | recorder 保存 clip 前 target，文件没有 applied/clipped action 或 clip flag |
| SIM-H5-11 | 中 | EE 元数据歧义 | `action_dim` writer | `action_space=ee` 时实际控制向量 21D，但 `action_dim` 仍写 19；现存数据未覆盖该路径 |
| SIM-H5-12 | 低 | 存储效率 | 相机矩阵与 Zarr chunks | 每帧重复 K/extrinsic；RGB Zarr chunk 未压缩约 23 MB，随机单帧访问代价较大 |
| SIM-H5-13 | 中 | 完整性 | HDF5/Zarr writer | 没有 schema validator、checksum、源清单、转换 manifest 或 end-to-end 完成标志 |
| SIM-H5-14 | 低 | 代码/注释偏差 | `BaseEnv.preprocess_obs_data` | 注释称点云 voxel 为 5 mm，实际参数为 2.5 mm |
| SIM-H5-15 | 低 | legacy 边界 | replay `_try_settle` | reader 支持可选 `done_action` attribute，但当前 writer 从不写；1113 个文件均不存在 |
| SIM-H5-16 | 高 | 发布/覆盖风险 | `EnvRecorder.save_episode` | 直接以 `"w"` 打开最终 `.h5`；同 seed 会覆盖，异常可留下对外可见的半文件，没有临时文件和原子 publish |
| SIM-H5-17 | 中 | 未强制的不变量 | `record_initial_obs`、`obs_alignment` | pre-action 对齐依赖调用者先记录初始观测；writer 不校验，却始终写固定 alignment 文本 |

### 10.1 SIM-H5-01：分辨率说明已过时

`Camera.__init__()` 明确使用 `width=320, height=240`。实测所有 `rgb` 为 `(N,240,320,3)`，`depth/segmentation` 为 `(N,240,320)`。任何根据 README 预分配 480×640 tensor 的 reader 都会失败。

### 10.2 SIM-H5-03：Zarr 无法独立恢复 episode provenance

`meta/episode_ends` 只给边界，不给 episode 名称。即使当前可通过重新扫描源 HDF5 并重复相同排序来恢复 seed，这仍依赖外部目录保持不变。复制、过滤、增删或重命名 HDF5 后，Zarr 本身无法证明每段对应哪个 episode。

### 10.3 SIM-H5-05：converter 的 schema 来自第一个文件

转换器只遍历第一个文件的顶层 dataset key：

- 后续文件缺 key 时会在访问阶段报错。
- 后续文件多出的 key 静默不进入 Zarr。
- dtype 会显式比较。
- trailing shape 不显式比较，通常在 Zarr append 时才因 shape 不兼容失败。
- 所有 HDF5 attrs 都不参与一致性验证。

因此转换成功只能说明被选择的数组可拼接，不能说明 episode 的 task、单位、action space 或时序语义一致。

### 10.4 SIM-H5-07：分割标签缺少对象字典

预处理把背景/桌面/臂/手固定映射为 0/1/2/3，其他 actor id 通过 scene 内 id 偏移到 4 以上。但文件没有保存每个最终 label 对应的 entity 名称，也没有保证 scene entity 创建顺序跨代码版本不变。`scene_info_json.objects` 记录模型信息，但没有记录 segmentation label，二者无法可靠 join。

### 10.5 SIM-H5-08：`failed_reason` 当前没有生产者

全仓库检索显示 `_failed_reason` 只在 recorder 初始化、reset 和保存时出现，没有赋值路径。属性存在并不代表诊断链已经实现。`failed=1` 的 12 个现存 episode 也全部是空原因。

### 10.6 SIM-H5-09：训练窗口的正确切分

不得跨越 `episode_ends` 建立 sequence window。对于 observation-action pair，可使用同一行 `(obs[t], action[t])`；若需要监督 next state，应使用同 episode 的 `obs[t+1]`，但最后一个 action 的 next observation 不存在，应丢弃该 transition 或改变采集格式。

### 10.7 SIM-H5-11：EE 路径仅完成代码级核实

代码允许 `action_space="ee"`，回放也会读取 `action_ee`，但当前 1113 个文件全部是 joint。文档对 EE 的 shape 和转换逻辑来自静态生产/消费链，不代表已经通过实际 episode round-trip 验证。尤其是 hold phase 的 `action` 在 step 后重新做 IK，不能当作该 transition 的精确 applied joint command。

### 10.8 SIM-H5-16：HDF5 直接写最终路径

writer 使用 `h5py.File(episode_path, "w")`。这同时带来两个事实：已有同名 seed 文件会被截断覆盖；进程在 dataset/attribute 尚未写完时退出，目录中可能留下看似正式命名的半成品。当前 reader/converter 没有完成标记来排除它。

### 10.9 SIM-H5-17：对齐语义由调用顺序保证

标准 `BaseGenerator.reset()` 正确调用 `record_initial_obs()`，因此现有生产链满足 pre-action 对齐。但 `EnvRecorder` 本身没有检查这个前置条件；若其他调用者直接 `interact_with_env()`，保存的观测将是 post-action，而 `obs_alignment` 仍会硬编码为 `obs[t]_before_action[t]`。

## 11. 推荐的读取不变量

读取标准 HDF5 时至少检查：

1. 13 个必需 dataset 是否齐全。
2. 所有 dataset 首维是否相同并等于 `episode_steps`。
3. 单帧 shape 和 dtype 是否符合第 4.2 节。
4. `action_space` 是否为受支持值；`joint` 用 `action`，`ee` 用 `action_ee`。
5. `frame_skip * physics_dt` 是否与 `dt` 在容差内一致。
6. `success_frame_idx` 是否为 -1 或位于 `[0,N)`。
7. `success_once=0` 时 `success_frame_idx` 应为 -1。
8. 解析 `scene_info_json` 时处理字段缺省，不依赖 `scene_*` 字符串副本。
9. 将 `depth==0` 当作 invalid，不当作相机原点表面。
10. 不把 `action` 当实际反馈，不把 `done[t]` 当 `obs[t]` 的状态标签。

读取 Zarr 时额外检查：

1. `episode_ends` 为 `int64` 一维、非空、严格递增。
2. 所有 `data/*` 首维相同，且等于 `episode_ends[-1]`。
3. sequence sampler 不能跨 episode boundary。
4. provenance、dt、task、action space 和成功状态必须由外部 manifest 补充；当前 store 内无法恢复。
5. 不假设任意 Zarr 都有固定 13 key；先检查实际 array 列表和 schema。

## 12. 本次验证范围

执行了以下只读验证：

- 静态追踪 HDF5 writer、全部数据生产者、Zarr converter、replay、visualizer 和 HDF5 派生 pipeline。
- 用 `/home/zhanghaoyang/miniconda3/envs/sim/bin/python`，在 `h5py 3.16.0`、`zarr 2.18.3`、`numpy 1.26.4` 下读取元数据。
- 遍历全部 1113 个 HDF5 的 key、shape、dtype、compression、首维一致性、attributes 类型与存在性。
- 核对 seed/文件名、success/目录、`dt=frame_skip*physics_dt`、success index 范围及对应 `done`，未发现不一致；1113 个文件的 dataset attributes 也全部为空。
- 遍历全部 8 个 Zarr 的 group/array、shape、dtype、chunk、codec、attrs 和 episode boundaries。
- 对每个 Zarr 核对当前成功 HDF5 的累计边界，并逐 task 比较首、末样本的全部 13 个 dataset，结果精确相等。

未执行：

- 未启动 SAPIEN 或创建环境。
- 未运行 MimicGen、episode generation、replay 或 Rerun GUI。
- 未对所有大体量 RGB/point-cloud 数值做全量 checksum；本次验证覆盖完整 metadata、完整边界和每个 task 的首末样本。
- 未生成或覆盖任何 `dexmani_sim` 数据文件。
