# CordVIP 离线几何入口

本目录提供物体资产构建、任务模板冻结和 Raw 离线处理脚本，算法实现复用 [cordvip_geometry](../../third_party/cordvip_geometry/)。任务级 HOI 流程生成训练与部署共用的固定物体点云；独立跟踪入口用于逐行几何分析。

以下命令均在仓库根目录、已配置感知依赖的 `real_robot` 环境中执行。安装与权重准备见[几何核心说明](../../third_party/cordvip_geometry/README.md#安装与命令)。已有合法资产的复用与模板分配无需加载模型；新建资产、标注和跟踪需要相应的本地模型权重。

## 入口选择

| 入口 | 用途 | 产物 |
| --- | --- | --- |
| [build_task.py](build_task.py) | 按任务构建或复用单物体米制资产，每物体 1024 点。 | 任务各物体目录中的 `asset.json`、Mesh、`points.npy` 和 `normals.npy`。 |
| [allocate_points.py](allocate_points.py) | 从已有单物体点云选择固定索引，按表面积分配总计 1024 点，每物体至少 64 点。 | 任务目录中的 `hoi/template.npz` 与 `hoi/manifest.json`。 |
| [annotate_episodes.py](annotate_episodes.py) | 批量 Episode 共用模型，每段重新初始化，按相机序列去重。 | 写入原 Episode 的 `data.h5:/hoi`。 |
| [reconstruct_object.py](reconstruct_object.py) | 单独完成图像分割、网格生成与显式尺度校准。 | 新目录中的分割、网格及提供尺度后的 `asset/`。 |
| [track_object.py](track_object.py) | 按 Raw 行顺序分析单物体或多物体场景，可附加手部几何。 | 新目录中的 `scene.json`、`frames/*.npz` 及可用的 `model_templates/`，Raw 保持只读。 |

## 任务级 HOI 流程

**物体图像与实测尺寸 → 单物体米制资产 → 任务级固定模板 → Raw 内 `/hoi` → Zarr 训练数据**。

以 `pnp_apple_banana` 为例，运行前准备：

- 任务资产放在 `assets/pointcloud_template/object/<task_name>/<object_name>/`。已有资产需带齐 `asset.json` 指向的 Mesh、点云和法向；新物体需提供 `reference.jpg`（或配置 `image`）及实测尺度，配置格式见[任务配置](../../docs/object_reconstruction.md#hoi-五阶段流程)。
- 可选 `task.json` 的 `objects` 列表决定对象顺序；省略时按物体目录名排序，默认 Prompt 为对象名。各 Prompt 必须互不冲突，且在场景首帧中分别匹配唯一候选。
- 待标注的已发布 Raw 位于 `episodes/<task_name>/episode_*`，需包含 RGB-D、记录的相机外参，且 `task_label` 与任务名一致。

```bash
# 1. 构建或复用单物体资产
python scripts/cordvip/build_task.py --task pnp_apple_banana

# 2. 冻结任务级模板
python scripts/cordvip/allocate_points.py assets/pointcloud_template/object/pnp_apple_banana

# 3. 标注该任务下的全部 Episode，写入原 data.h5
python scripts/cordvip/annotate_episodes.py --task pnp_apple_banana
```

`build_task.py` 和 `annotate_episodes.py` 的 `--task` 接收任务名，`allocate_points.py` 的位置参数接收任务目录路径。只处理单段时，在标注命令后追加 `--episode episodes/pnp_apple_banana/episode_20261101_165909`。

重复运行会复用合法资产、校验并复用已有模板，以及跳过模板一致的完整 HOI 标注；不覆盖已有正式结果。修改照片或构建参数不会重建已发布资产，需要重建时使用新的资产目录。模板冻结后应保持物体顺序与几何一致。

使用主项目入口查看和导出：

```bash
python scripts/visualize_episode.py episodes/pnp_apple_banana/episode_20261101_165909
python scripts/export_zarr.py episodes/pnp_apple_banana \
  --output datasets/pnp_apple_banana.cordvip.zarr \
  --fields joint_state action hand_pointcloud object_pointcloud
```

`object_pointcloud` 需要显式选择；默认导出不包含它。HOI 物体点云以 xArm base 为坐标系、米为单位。存在无效物体帧的 Episode 会被拒绝导出，可用 `--exclude` 显式排除。

## 单物体重建

独立重建可直接指定图片和尺度。将图片路径、网格轴及尺寸替换为实际值，以下 `0.08` 仅为米制尺寸示例：

```bash
python scripts/cordvip/reconstruct_object.py \
  --image path/to/object.jpg --prompt apple \
  --size-axis x --size-meters 0.08 \
  --headless --output logs/cordvip/apple_build
```

提供尺度时，米制资产写入 `<output>/asset/`；省略尺度时保留未定尺度网格，后续需完成尺度校准。`--headless` 下多候选需要查看候选结果并明确指定零起始的 `--instance`；任务构建时对应 `task.json` 中的 `instance`。

## 独立跟踪分析

以下示例使用已有资产和首帧 mask。将 mask 路径替换为实际文件，mask 应与所选 Episode 首帧的原始 RGB 同尺寸、同像素网格：

```bash
python scripts/cordvip/track_object.py \
  --episode episodes/pnp_apple_banana/episode_20261101_165909 \
  --asset assets/pointcloud_template/object/pnp_apple_banana/apple \
  --initial-selection path/to/first_frame_mask.png \
  --headless --output logs/cordvip/apple_track
```

省略 `--asset` 时只输出分割与实测点云；添加 `--with-hand` 可附加同一 Raw 行实测关节对应的手部几何。多物体使用 `--scene` JSON 指定对象；无界面运行时每个对象都需提供 `initial_selection`。交互选择首帧时，使用 `--prompt apple` 并省略 `--initial-selection`、`--headless`。独立跟踪的场景点云默认位于彩色相机坐标系，单位为米。

重建和独立跟踪的 `--output` 均须是尚不存在的新目录。各脚本的完整参数可通过 `--help` 查看。

## 进一步阅读

- [物体重建与 HOI 五阶段流程](../../docs/object_reconstruction.md#hoi-五阶段流程)：任务配置、模板格式、训练与在线部署，以及已验证范围。
- [几何核心架构与 Python API](../../third_party/cordvip_geometry/docs/usage.md)：模块职责、坐标约定和结果含义。
