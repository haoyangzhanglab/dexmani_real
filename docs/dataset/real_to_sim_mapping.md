# DexMani Real v16 → DexMani Sim 标签映射表

> **用途：** 面向熟悉 DexMani Sim / DexMani Policy 命名的使用者，说明一个
> DexMani Real v16 episode 中哪些标签对应 Sim 的 13 个字段，以及哪些同名候选实际上
> 不等价。
>
> **硬约束：** 本文只描述标签和来源关系，不修改任何已录制数值。
>
> **事实核查基线：** 2026-08-16；DexMani Sim `fac5e770`、DexMani Policy
> `8ce8be59`、DexMani Real schema v16 当前工作树。

相关完整字典见 [Real v16 与 Sim 统一数据字典](hdf5_episode.md) 和
[Sim HDF5/Zarr 独立说明](sim_hdf5_zarr.md)。

## 1. 范围与结论

这是一份 **label crosswalk**，不是 Real→Sim 数值转换规范，也不生成可在 Sim 中回放的
标准 episode。

本文采用两个层次：

- `alias_only`：只登记 target label 与 source label/path，不生成新数组；
- `value_preserving_structural_view`：可选地表达 `concat`、C-order `reshape` 或 metadata
  `repeat`，但元素值、dtype、顺序和 episode 行集合必须保持不变。

无论使用哪一层，结果都必须声明：

```text
domain = real
frame_profile = real_native
numeric_transforms = []
row_selection = none
```

不能把它命名为 `sim`、`sim_world`、`sim-compatible` 或“标准 Sim episode”。字段名相同只说明
consumer-facing label 相同，不证明 shape、dtype、frame、单位、时序或物理语义相同。

### 1.1 允许的关系

| 关系 | 含义 | 限制 |
|---|---|---|
| `alias` | 记录一个 Sim/Policy label 对应哪个 Real label/path | 不读写数组 |
| `copy` | 原样复制一个已经解码的数组 | shape、dtype、元素和行序全部不变 |
| `concat_view` | 仅把多个 source label 表达为一个有序结构视图 | source dtype 必须相同；禁止隐式 promotion；必须能无损拆回 source |
| `reshape_view` | 只改变逻辑 shape | 仅 C-order；禁止 transpose、轴 permutation 或矩阵方向变化 |
| `repeat_metadata` | 把同一个 episode metadata 值关联到每一行 | 每个副本必须逐元素等于 source |
| `omit` | 明确说明没有等价来源 | 禁止用零、NaN、末帧常量或其他占位伪造 |

MP4 解码属于读取 Real 容器，不属于图像数值处理；映射得到的 RGB 必须与
`EpisodeReader` 的解码结果一致。

### 1.2 明确禁止

以下操作均不属于本文的标签映射：

- dtype cast，包括 `float64 → float32`；
- 单位缩放、归一化、clip、round 或量化；
- base/world 坐标变换、姿态旋转、矩阵求逆或光学轴变换；
- FK、IK、mesh 渲染或任何模型派生；
- RGB/depth resize、crop、rectification 或颜色变换；
- 点云 crop、voxel、FPS、随机采样、padding 或补点；
- 插值、重采样、逐行过滤、切段或压紧 episode；
- 根据 recorder status 猜造 task `done/success`。

若后续数据清洗确实需要这些操作，应建立另一份带版本和 provenance 的数值转换规范，不能
把结果归入本文的 `label_only` 映射。

仓库现已提供这层独立数值转换规范与实现，见
[Real 清洗与 Sim-label HDF5 格式](processed_hdf5.md)。其输出明确标记为
`domain=real`，不会改变本文对 label-only 映射边界的定义。

## 2. 两端容器

一个 Real v16 episode 是四件套目录：

```text
episode_.../
├── data.h5
├── depth.h5          # /depth
├── pointcloud.h5     # /pointcloud
└── rgb.mp4
```

合法 v16 有 96 个基础 `data.h5` dataset。标准 RecorderIO 还设置
`/meta.arm_sent_stream=True` 并写第 97 个 `action_arm_joint_sent`。

标准 Sim HDF5 有 13 个逐帧 dataset；Policy Zarr 使用 `data/<key>` 和
`meta/episode_ends`。本文逐一以这 13 个 Sim label 为索引，但不会为了凑齐 schema 生成
不存在或语义不成立的字段。

## 3. 13 个 Sim label 的 Real 来源

状态含义：

- **值保持结构关系**：可以描述为同值 copy/concat；仍保留 Real dtype/domain；
- **Real-native 候选**：模态相近，但直接使用 Sim 名称会掩盖结构或语义差异；
- **不等价候选**：名称或 shape 看似接近，但物理语义不同，禁止 rename；
- **无来源**：Real episode 没有对应标签。

| Sim / Policy label | Sim 标准单帧合同 | Real 来源 | 标签映射结论 |
|---|---|---|---|
| `rgb` | `(240,320,3)` `uint8` | `EpisodeReader.read_camera_all("rgb")` | **Real-native 候选。** RGB 通道和解码值可原样关联；Real H/W 由 episode 决定，默认通常是 `480×640`。不 resize 时不能声称满足标准 Sim shape。建议保留 `rgb` label，同时声明 `domain=real` 和实际 H/W。 |
| `depth` | `(240,320)` `uint16`，数值为 mm | `/depth`，并携带 `/meta.depth_scale` | **不等价候选。** Real `uint16` 是设备 raw unit，数值单位由逐 episode `depth_scale` 决定；aligned Z16 也仍携带 source-depth optical Z。原值只能标为 `depth_raw_real`，不能直接改标为标准 Sim `depth`。 |
| `segmentation` | `(240,320)` `uint8` | 无 | **无来源。** 禁止全零伪造；全零在 Sim 中表示 background，不表示 unknown。 |
| `point_cloud` | `(1024,6)` `float32`，SAPIEN world XYZRGB | `/pointcloud`，默认 `(2048,6)` `float32` | **Real-native 候选。** 数值可原样关联为 `point_cloud_real_native`；Real 使用 `xarm_base`/当前 producer world、真实滤波和不同点数，不能仅改名后宣称是 Sim-world 点云。 |
| `camera_intrinsic` | `(9,)` `float32`，逐帧 pinhole K | `/meta.camera_K`，`(9,)` `float64` | **Real-native 候选。** 可登记静态 metadata 来源；若使用 structural view，可 unchanged repeat。它只适用于未修改的 Real pixel viewport，不能声称满足 Sim dtype、分辨率或理想 pinhole 合同。 |
| `camera_extrinsic` | `(12,)` `float32`，OpenCV world→camera | `/meta.camera_T_world_camera` 或 `/meta.camera_T_eef_camera`，均为 camera→world/EEF `4×4` | **不等价候选。** 方向、基准 frame 和存储 shape 均不同；获得 Sim 语义必须求逆或组合逐帧 EEF pose，已超出标签映射。保留原字段名和方向，禁止直接 rename。 |
| `joint_state` | `(19,)` `float32`，arm 7 + hand 12，rad | `arm_qpos` `(7,)` + `hand_qpos` `(12,)`，均为 `float64` rad | **值保持结构关系。** `joint_state ↔ [arm_qpos, hand_qpos]`，顺序和单位一致。只允许登记有序 components，或形成不 cast 的 `concat_view`；因此仍不是标准 Sim `float32` array。 |
| `contact_force` | `(15,)` `float32`，world-frame link 净接触力 N | `hand_contact` 只是危险的 shape 候选 | **无等价来源。** Real 是 SDK-scaled tactile 汇总，SI 单位未验证；Sim 是 5 个 `link2` 的 world-frame 净接触力并跨 physics substeps 平均。禁止 rename/flatten。 |
| `fingertip_points` | `(15,)` `float32`，SAPIEN world | `hand_fingertip` `(5,3)` `float64`，xArm base | **不等价候选。** 即使 C-order reshape 保持数值，frame 和模型语义仍不同；Real 还包含物理 flange correction。只能保留为 `fingertip_points_real_native`/QA。 |
| `imagine_point_cloud` | `(512,6)` `float32`，完整手 mesh 的世界点云 | 无 | **无来源。** 生成它需要 Sim mesh/FK，属于数值派生。 |
| `action` | `(19,)` `float32`，arm 7 + hand 12 joint target | `action_arm_joint_sent` `(7,)` + `action_hand_joint` `(12,)`，均为 `float64` rad | **有限等价的值保持结构关系。** 顺序和单位一致，可登记为有序 components 或不 cast 的 `concat_view`。arm 是 forwarded-to-worker；hand 是 queued target、没有 ACK，不能称为硬件 applied action。 |
| `action_ee` | `(21,)` `float32`，Sim joint action 的 world-frame FK pose 9 + hand 12 | `action_arm_ee` `(9,)` + `action_hand_joint` `(12,)` 是 shape 候选 | **不等价候选。** Real arm 部分是 IK tracking target，可能不同于 sent joint action；它还位于 Real base/world，而 Sim 字段由 joint action 做 Sim world FK。禁止 concat 后 rename 为 Sim `action_ee`。 |
| `done` | `bool`，action 后的 task transition outcome | 无；`/meta.success`、`truncated` 不是候选 | **无来源。** Real `success` 表示录制发布成功，`truncated` 表示录制长度边界；禁止构造末帧 `True`、全 `False` 或复制 recorder status。 |

因此，最小且诚实的结构关系只有：

```text
joint_state ↔ [arm_qpos, hand_qpos]
action      ↔ [action_arm_joint_sent, action_hand_joint]
```

第二项还必须满足 `/meta.arm_sent_stream=True` 且 dataset 确实存在。缺失 sent stream 时，不得
静默改用 `action_arm_joint` 并继续称为同一 action source。

## 4. Real 与 Sim 的 base→world 不一致

这是标签是否等价的边界，不是待执行的变换步骤。

- Real 空间值使用 `xarm_base`；当前受支持 Real runtime 明确维持
  `Real base→world = identity`。
- 标准 Sim 空间值使用 `sapien_world`；Sim robot root 的
  `Sim base→world = Rz(+π/6)`，平移为零。

所以两端的 XYZ/pose 数值不在同一个 world frame。本文：

- 不对 Real XYZ/pose 应用 `Rz(+π/6)`；
- 不把未经旋转的 Real 数值重新标成 `sapien_world`；
- 对原样保留的空间字段持续声明 `frame=xarm_base`；
- 声明 `frame_compatibility_with_sim_world=false`。

这一差异不影响关节角标签本身，但直接阻止以下字段成为标准 Sim 同名字段：

| Real 字段 | 原生 frame / 方向 | 不能直接对应的 Sim label |
|---|---|---|
| `/pointcloud[...,0:3]` | 当前 producer world，数值等于 `xarm_base` | `point_cloud` 的 SAPIEN-world XYZ |
| `hand_fingertip` | `xarm_base` | `fingertip_points` 的 SAPIEN-world XYZ |
| `arm_ee`、`action_arm_ee` | `xarm_base` / 当前 runtime world | Sim world-frame EE pose / `action_ee` |
| `camera_T_world_camera` | camera→Real world | `camera_extrinsic` 的 Sim world→camera |
| `camera_T_eef_camera` | camera→Real EEF，且为静态 metadata | `camera_extrinsic` 的逐帧 Sim world→camera |

Real 的 frame attributes 是 producer 声明，不应单独作为跨版本几何等价证明。历史 v16 缺失
这些 additive attributes 时，应记录为 `frame_evidence=legacy_unknown`，而不是补写一个推测值。

## 5. 19-DoF 顺序和 action 语义

Real 与 Sim 的 canonical 19-DoF 标签顺序一致：

```text
[0:7]   joint1 ... joint7
[7:19]  thumb_bend, thumb_rota_joint1, thumb_rota_joint2,
        index_bend, index_joint1, index_joint2,
        mid_joint1, mid_joint2,
        ring_joint1, ring_joint2,
        pinky_joint1, pinky_joint2
```

`joint_state` 的两个 source 都是实测 qpos。`action` 的 source 边界不同：

- `action_arm_joint_sent`：已转发到 arm worker 的安全 joint target，不是硬件执行 ACK；
- `action_hand_joint`：最终 queued target，没有独立 sent/ACK stream；
- `action_arm_joint`：安全门后的 final candidate；缺 sent stream 时不能假装成 sent；
- `action_arm_ee`：IK tracking intent，不是 `action_arm_joint_sent` 的另一种无损表示。

Real 与 Sim URDF 的 `thumb_rota_joint1/2` 上限还不同：Real 约 `1.745 rad`，Sim 为
`1.57 rad`。标签映射不 clip 数值，只记录 target-limit mismatch。

## 6. 供后续清洗使用的 Real-only 标签

Sim 标准 episode 没有以下逐行字段。它们应保留为 Real provenance/quality labels，不能在映射
阶段据此删行：

| 用途 | Real labels |
|---|---|
| 行与网格身份 | `timestamp`、`source_sample_index` |
| observation 总体质量 | `flag_sample_valid`、`flag_observation_valid`、`flag_frame_status` |
| 动作边界 | `flag_action_queued`、`flag_held`、`flag_ik_ok`、`flag_retarget_ok` |
| arm/hand 状态 | `arm_connected`、`arm_state_valid`、`hand_connected`、`hand_error_state` |
| 相机/点云 | `flag_camera_fresh`、`flag_pointcloud_valid`、camera source/receive/publish timestamps |
| 触觉 | `tactile_fresh`、`tactile_calibrated`、`hand_tactile_unit_code` |
| raw action 解释 | reader 的 arm/hand raw-valid masks 及其 metadata expression |

如果某个 downstream consumer 不接受 validity labels，而用户又不允许 row selection，则只能在
consumer admission 层整 episode 接受或拒绝；不能在本映射中静默过滤坏行或重写
`episode_ends`。

## 7. Episode metadata 的标签关系

| Sim/Policy 概念 | Real 来源 | 结论 |
|---|---|---|
| task name | `/meta.task_label` | 仅可记录 label 来源；两端 task vocabulary 是否等价需要显式 alias 表，不能模糊匹配或改写 payload。 |
| control step duration | `/meta.grid_dt_s` | 都以秒表达，但一个是真机录制网格、一个是 Sim physics/control step；只登记来源，不宣称动力学等价。 |
| action space | Real 标准路径的 joint action labels | 可声明 `source_action_representation=joint`；不能据此生成 Sim scene/replay 状态。 |
| success / failed / done | 无 | recorder `success/stop_reason/truncated` 不是 task outcome。 |
| scene/object/seed | 通常无 | 不生成、不猜测。 |
| robot/frame provenance | Real `/meta` frame attrs、URDF/SRDF hashes | 原样保存；缺失写 `unknown`。 |
| camera provenance | K、depth scale、alignment/profile、calibration hashes、外参原方向 | 原样保存；不能把一个 frame 名称当作完整 RGB-D 几何证明。 |

## 8. DexMani Policy 使用边界

Policy `ReplayBuffer` 只加载 config 请求的 keys，并不要求凑齐 Sim 的 13 个字段。因此 label-only
视图可以服务于专门接受 Real-native 数据的 loader，但必须冻结 dataset class 和 config：

| 目标输入 | label-only 可提供的关系 | 边界 |
|---|---|---|
| joint | `joint_state`、`action` | 最可信；存储仍为 Real `float64`，不是标准 Sim dtype |
| RGB | 上述字段 + 原生 `rgb` | 不 resize；consumer 必须接受实际 H/W |
| point cloud | 上述字段 + 原生 `/pointcloud` | 保留点数和 `xarm_base`；consumer 必须明确接受 Real-native frame |
| RGBPC | 不支持标准 Sim 合同 | depth 单位/几何和 camera extrinsic 方向不能只靠改标签解决 |
| EE action / auxiliary EE | 不支持 | 没有语义正确、数值不变的 Sim `action_ee` 来源 |

还要注意：当前 Policy loader/agent 可能在读取后自行执行 `float32` cast、图像 resize 或点云 FPS。
这些是 consumer-side numerical transforms。本文只能保证映射存储边界没有修改值，不能保证最终
model input 仍逐元素等于 Real source。若要求端到端完全不改值，应禁用这些 consumer 路径或
fail closed。

## 9. 建议的映射 manifest

标签关系不能只存在于代码变量名中。建议保存独立 manifest，例如：

```json
{
  "schema": "dexmani-real-label-map/v1",
  "domain": "real",
  "source_schema_version": 16,
  "mapping_profile": "value_preserving_structural_view",
  "frame_profile": "real_native",
  "frame_compatibility_with_sim_world": false,
  "numeric_transforms": [],
  "row_selection": "none",
  "field_map": [
    {
      "target_label": "joint_state",
      "sources": ["arm_qpos", "hand_qpos"],
      "operation": "concat_view",
      "slices": ["0:7", "7:19"],
      "dtype": "float64",
      "unit": "rad",
      "numeric_transform": "none",
      "equivalence": "value_preserving"
    },
    {
      "target_label": "action",
      "sources": ["action_arm_joint_sent", "action_hand_joint"],
      "operation": "concat_view",
      "slices": ["0:7", "7:19"],
      "dtype": "float64",
      "unit": "rad",
      "numeric_transform": "none",
      "equivalence": "qualified_command"
    }
  ],
  "omitted_labels": [
    "rgb",
    "depth",
    "segmentation",
    "point_cloud",
    "camera_intrinsic",
    "camera_extrinsic",
    "contact_force",
    "fingertip_points",
    "imagine_point_cloud",
    "action_ee",
    "done"
  ],
  "non_equivalent_candidates": [
    "depth",
    "camera_extrinsic",
    "contact_force",
    "fingertip_points",
    "action_ee"
  ],
  "fill_policy": "omit"
}
```

实际 manifest 还应原样记录：

- source episode ID 和四个成员文件 hash；
- source `/meta` snapshot/hash、recorder commit（未知写 `unknown`）；
- 每个 label 的 source path、operation、shape、dtype、unit、frame 和等价等级；
- `arm_sent_stream` marker，以及 arm/hand command boundary；
- Real H/W、point count、depth scale、camera alignment/profile 和外参方向；
- Real/Sim URDF hash、19-DoF 顺序和 joint-limit mismatch；
- target Policy commit、dataset class、config hash 和 consumer-side transforms；
- `forbidden_candidates`，特别是 `hand_contact→contact_force`、
  `action_arm_ee→action_ee` 和 recorder status→`done`。

## 10. 验收规则

| 类别 | 必须验证 |
|---|---|
| source | 先通过 `EpisodeReader.require_valid()`；四个成员、N 和 sent-stream marker 一致。 |
| 操作白名单 | 每个关系只能是 `alias/copy/concat_view/reshape_view/repeat_metadata/omit`。 |
| 数值守恒 | copy 与 source shape/dtype/元素一致；concat 可按 slices 无损拆回 source；reshape 反向恢复后逐元素一致。 |
| 行守恒 | 所有逐帧关系保持原 N、原行序和原 episode 边界；不删行、不切段、不压紧。 |
| frame | 所有未修改空间值继续标 `xarm_base`/Real native；不得标 `sapien_world`。 |
| 缺失字段 | unavailable/non-equivalent 字段保持缺失，禁止填零、NaN 或猜造。 |
| dtype | 不执行 `astype`；Real `float64` 不能被文档称为标准 Sim `float32`。 |
| provenance | 每个 label 能反查 source path、原 shape/dtype/unit/frame 和 episode hash。 |
| consumer | 单独披露 Policy loader 的 cast/resize/FPS；loader 成功不代表数据成为 Sim-equivalent。 |

发现任何 dtype cast、单位/坐标运算、矩阵求逆、FK、resize、点云采样或 row selection 时，应
立即拒绝将该产物标记为 `label_only`。

## 11. 事实来源

| 合同 | 代码 / 文档来源 |
|---|---|
| Real v16 writer/reader/schema | `dexmani_real/recording/episode_recorder.py`、`episode_reader.py`、`episode_schema.py` |
| Real sample producer | `dexmani_real/teleop/episode_samples.py`、`teleop/loop.py` |
| Real arm base-frame FK | `dexmani_real/planning/kinematics.py` |
| Real camera/pointcloud | `dexmani_real/sensor/realsense.py`、`sensor/camera_process.py` |
| Sim 13 fields | sibling `dexmani_sim/mimic_gen/utils/env_recorder.py`、`envs/base_env.py` |
| Sim root/world FK | sibling `dexmani_sim/envs/base_env.py`、`robots/xarm7_xhand.py` |
| Sim camera/contact/hand cloud | sibling `dexmani_sim/sensors/camera.py`、`contact.py`、`imagine_point_cloud.py` |
| Policy requested keys | sibling `dexmani_policy/dataset/`、`common/replay_buffer.py` |

本文只记录标签、来源和不兼容边界；它不会通过修改 Real 数值来制造 Sim 等价性。
